use std::{
    cmp::Reverse,
    collections::{HashMap, HashSet},
    hash::{Hash, Hasher},
    panic::{AssertUnwindSafe, catch_unwind},
    path::PathBuf,
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use futures::channel::oneshot;
use priority_queue::PriorityQueue;

use crate::{
    catalog::DevelopRecipe,
    preview::PreviewState,
    rendering::{RenderIdentity, RenderVariant},
};

#[derive(Clone, Debug)]
pub struct JobKey {
    pub path: PathBuf,
    pub variant: RenderVariant,
    pub render_identity: RenderIdentity,
    /// Payload is excluded from key semantics; its canonical digest is already
    /// part of `render_identity` and is verified before rendering.
    pub recipe: Option<Arc<DevelopRecipe>>,
}

impl PartialEq for JobKey {
    fn eq(&self, other: &Self) -> bool {
        self.path == other.path
            && self.variant == other.variant
            && self.render_identity == other.render_identity
    }
}

impl Eq for JobKey {}

impl Hash for JobKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.path.hash(state);
        self.variant.hash(state);
        self.render_identity.hash(state);
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct DemandScopeId(pub u64);

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum PriorityClass {
    Maintenance = 0,
    Speculative = 1,
    NearViewport = 2,
    Visible = 3,
    Interactive = 4,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct JobPriority {
    pub class: PriorityClass,
    /// Larger values execute first inside a priority class.
    pub rank: i32,
}

impl JobPriority {
    pub const INTERACTIVE: Self = Self {
        class: PriorityClass::Interactive,
        rank: 0,
    };

    pub fn visible(rank: i32) -> Self {
        Self {
            class: PriorityClass::Visible,
            rank,
        }
    }

    pub fn near_viewport(rank: i32) -> Self {
        Self {
            class: PriorityClass::NearViewport,
            rank,
        }
    }

    pub fn speculative(rank: i32) -> Self {
        Self {
            class: PriorityClass::Speculative,
            rank,
        }
    }
}

pub struct JobDemand {
    pub key: JobKey,
    pub priority: JobPriority,
    /// A newly-created UI cache request needs a completion receiver. Existing
    /// loading requests already have one and only refresh their demand.
    pub subscribe: bool,
}

pub enum JobCompletion {
    Finished(PreviewState),
    Cancelled,
}

pub struct Subscription {
    pub key: JobKey,
    pub receiver: oneshot::Receiver<JobCompletion>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SchedulerStats {
    pub queued: usize,
    pub running: usize,
    pub completed: u64,
    pub cancelled: u64,
    pub total_queue_latency_ms: u64,
    pub total_execution_ms: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum JobSnapshotStatus {
    Running,
    Queued,
    PendingSuccessor,
}

#[derive(Clone, Debug)]
pub struct JobSnapshot {
    pub key: JobKey,
    pub priority: JobPriority,
    pub status: JobSnapshotStatus,
    pub scope_count: usize,
    pub sequence: u64,
    pub cancelled: bool,
}

#[derive(Clone, Debug, Default)]
pub struct SchedulerSnapshot {
    pub jobs: Vec<JobSnapshot>,
    pub queued: usize,
    pub running: usize,
}

type Executor =
    dyn Fn(&JobKey, JobPriority, &Arc<AtomicBool>) -> PreviewState + Send + Sync + 'static;

pub struct RenderScheduler {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
}

struct Shared {
    state: Mutex<State>,
    wake: Condvar,
    executor: Arc<Executor>,
    aging_interval: Duration,
}

struct State {
    queue: PriorityQueue<JobKey, QueuePriority>,
    jobs: HashMap<JobKey, Job>,
    pending: HashMap<JobKey, Job>,
    next_sequence: u64,
    running: usize,
    completed: u64,
    cancelled: u64,
    total_queue_latency: Duration,
    total_execution: Duration,
    shutdown: bool,
}

struct Job {
    demands: HashMap<DemandScopeId, JobPriority>,
    subscribers: Vec<Subscriber>,
    status: JobStatus,
    sequence: u64,
    queued_at: Instant,
    cancelled: Arc<AtomicBool>,
}

struct Subscriber {
    scope: DemandScopeId,
    sender: oneshot::Sender<JobCompletion>,
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum JobStatus {
    Queued,
    Running,
}

#[derive(Clone, Copy, Eq, Ord, PartialEq, PartialOrd)]
struct QueuePriority {
    effective: JobPriority,
    // Earlier submissions win ties in this max-priority queue.
    sequence: Reverse<u64>,
}

const DEFAULT_AGING_INTERVAL: Duration = Duration::from_secs(2);

impl RenderScheduler {
    pub fn new(parallelism: usize, executor: Arc<Executor>) -> Self {
        Self::new_with_aging(parallelism, DEFAULT_AGING_INTERVAL, executor)
    }

    fn new_with_aging(
        parallelism: usize,
        aging_interval: Duration,
        executor: Arc<Executor>,
    ) -> Self {
        assert!(parallelism > 0, "render parallelism must be positive");
        let shared = Arc::new(Shared {
            state: Mutex::new(State {
                queue: PriorityQueue::new(),
                jobs: HashMap::new(),
                pending: HashMap::new(),
                next_sequence: 0,
                running: 0,
                completed: 0,
                cancelled: 0,
                total_queue_latency: Duration::ZERO,
                total_execution: Duration::ZERO,
                shutdown: false,
            }),
            wake: Condvar::new(),
            executor,
            aging_interval,
        });
        let workers = (0..parallelism)
            .map(|index| {
                let shared = shared.clone();
                // With more than one slot, keep one worker free of near and
                // speculative work. It may serve visible thumbnails while the
                // user is looking at the grid; opening an image clears that
                // scope and cooperatively cancels the thumbnail if necessary.
                let foreground_only = parallelism > 1 && index == 0;
                thread::Builder::new()
                    .name(format!("pics-render-{index}"))
                    .spawn(move || worker_loop(shared, foreground_only))
                    .expect("failed to start render worker")
            })
            .collect();
        Self { shared, workers }
    }

    /// Atomically replaces all demand belonging to `scope` and returns
    /// receivers requested for newly-created UI cache entries.
    pub fn replace_scope(
        &self,
        scope: DemandScopeId,
        demands: Vec<JobDemand>,
    ) -> Vec<Subscription> {
        let desired: HashSet<JobKey> = demands.iter().map(|demand| demand.key.clone()).collect();
        let mut cancelled_senders = Vec::new();
        let mut subscriptions = Vec::new();
        let mut foreground_promotion = None;
        let mut state = self.shared.state.lock().unwrap();

        let existing_keys: Vec<JobKey> = state
            .jobs
            .iter()
            .filter(|(_, job)| job.demands.contains_key(&scope))
            .map(|(key, _)| key.clone())
            .chain(
                state
                    .pending
                    .iter()
                    .filter(|(_, job)| job.demands.contains_key(&scope))
                    .map(|(key, _)| key.clone()),
            )
            .collect();
        for key in existing_keys {
            if desired.contains(&key) {
                continue;
            }
            remove_scope_interest(&mut state, &key, scope, &mut cancelled_senders);
        }

        for demand in demands {
            let previous_priority = state
                .pending
                .get(&demand.key)
                .and_then(effective_priority)
                .or_else(|| state.jobs.get(&demand.key).and_then(effective_priority));
            if demand.priority.class >= PriorityClass::Visible
                && previous_priority.is_none_or(|priority| priority < demand.priority)
            {
                foreground_promotion = foreground_promotion.max(Some(demand.priority));
            }
            let needs_successor = state.jobs.get(&demand.key).is_some_and(|job| {
                job.status == JobStatus::Running && job.cancelled.load(Ordering::Acquire)
            });
            if needs_successor || state.pending.contains_key(&demand.key) {
                if !state.pending.contains_key(&demand.key) {
                    let sequence = state.next_sequence;
                    state.next_sequence += 1;
                    state
                        .pending
                        .insert(demand.key.clone(), new_queued_job(sequence));
                }
                add_demand(
                    state.pending.get_mut(&demand.key).unwrap(),
                    scope,
                    &demand,
                    &mut subscriptions,
                );
                continue;
            }
            if !state.jobs.contains_key(&demand.key) {
                let sequence = state.next_sequence;
                state.next_sequence += 1;
                state
                    .jobs
                    .insert(demand.key.clone(), new_queued_job(sequence));
            }
            let job = state.jobs.get_mut(&demand.key).unwrap();
            add_demand(job, scope, &demand, &mut subscriptions);
            refresh_queue_priority(&mut state, &demand.key);
        }
        if let Some(priority) = foreground_promotion {
            preempt_one_lower_priority(&mut state, priority);
        }
        drop(state);
        for sender in cancelled_senders {
            let _ = sender.send(JobCompletion::Cancelled);
        }
        self.shared.wake.notify_all();
        subscriptions
    }

    pub fn clear_scope(&self, scope: DemandScopeId) {
        self.replace_scope(scope, Vec::new());
    }

    pub fn stats(&self) -> SchedulerStats {
        let state = self.shared.state.lock().unwrap();
        SchedulerStats {
            queued: state.queue.len() + state.pending.len(),
            running: state.running,
            completed: state.completed,
            cancelled: state.cancelled,
            total_queue_latency_ms: state.total_queue_latency.as_millis() as u64,
            total_execution_ms: state.total_execution.as_millis() as u64,
        }
    }

    pub fn snapshot(&self) -> SchedulerSnapshot {
        let state = self.shared.state.lock().unwrap();
        let mut running = Vec::new();
        let mut queued = Vec::new();
        for (key, job) in &state.jobs {
            let Some(priority) = effective_priority(job) else {
                continue;
            };
            let mut snapshot = JobSnapshot {
                key: key.clone(),
                priority,
                status: if job.status == JobStatus::Running {
                    JobSnapshotStatus::Running
                } else {
                    JobSnapshotStatus::Queued
                },
                scope_count: job.demands.len(),
                sequence: job.sequence,
                cancelled: job.cancelled.load(Ordering::Acquire),
            };
            if job.status == JobStatus::Running {
                running.push(snapshot);
            } else {
                let dispatch_priority =
                    state
                        .queue
                        .get_priority(key)
                        .copied()
                        .unwrap_or(QueuePriority {
                            effective: priority,
                            sequence: Reverse(job.sequence),
                        });
                snapshot.priority = dispatch_priority.effective;
                queued.push((dispatch_priority, snapshot));
            }
        }
        running.sort_by_key(|job| job.sequence);
        queued.sort_by(|(left, _), (right, _)| right.cmp(left));

        let mut pending = state
            .pending
            .iter()
            .filter_map(|(key, job)| {
                Some(JobSnapshot {
                    key: key.clone(),
                    priority: effective_priority(job)?,
                    status: JobSnapshotStatus::PendingSuccessor,
                    scope_count: job.demands.len(),
                    sequence: job.sequence,
                    cancelled: job.cancelled.load(Ordering::Acquire),
                })
            })
            .collect::<Vec<_>>();
        pending.sort_by(|left, right| {
            right
                .priority
                .cmp(&left.priority)
                .then_with(|| left.sequence.cmp(&right.sequence))
        });

        let running_count = running.len();
        let queued_count = queued.len() + pending.len();
        let mut jobs = running;
        jobs.extend(queued.into_iter().map(|(_, job)| job));
        jobs.extend(pending);
        SchedulerSnapshot {
            jobs,
            queued: queued_count,
            running: running_count,
        }
    }

    pub fn has_foreground_work(&self) -> bool {
        let state = self.shared.state.lock().unwrap();
        state
            .jobs
            .values()
            .chain(state.pending.values())
            .any(|job| {
                effective_priority(job)
                    .is_some_and(|priority| priority.class >= PriorityClass::Visible)
            })
    }
}

fn preempt_one_lower_priority(state: &mut State, incoming: JobPriority) {
    let candidate = state
        .jobs
        .iter()
        .filter(|(_, job)| {
            job.status == JobStatus::Running
                && !job.cancelled.load(Ordering::Acquire)
                && effective_priority(job).is_some_and(|priority| priority < incoming)
        })
        .min_by_key(|(_, job)| (effective_priority(job), Reverse(job.sequence)))
        .map(|(key, _)| key.clone());
    let Some(key) = candidate else {
        return;
    };

    let (demands, subscribers) = {
        let job = state.jobs.get_mut(&key).unwrap();
        job.cancelled.store(true, Ordering::Release);
        (job.demands.clone(), std::mem::take(&mut job.subscribers))
    };
    if demands.is_empty() {
        return;
    }

    if let Some(successor) = state.pending.get_mut(&key) {
        successor.demands.extend(demands);
        successor.subscribers.extend(subscribers);
        return;
    }
    let sequence = state.next_sequence;
    state.next_sequence += 1;
    let mut successor = new_queued_job(sequence);
    successor.demands = demands;
    successor.subscribers = subscribers;
    state.pending.insert(key, successor);
}

impl Drop for RenderScheduler {
    fn drop(&mut self) {
        let diagnostics = std::env::var_os("PICS_DIAGNOSTICS").is_some();
        {
            let mut state = self.shared.state.lock().unwrap();
            state.cancelled += state.queue.len() as u64 + state.pending.len() as u64;
            state.shutdown = true;
            for job in state.jobs.values() {
                job.cancelled.store(true, Ordering::Release);
            }
            for job in state.pending.values() {
                job.cancelled.store(true, Ordering::Release);
            }
        }
        self.shared.wake.notify_all();
        let workers = std::mem::take(&mut self.workers);
        if !workers.is_empty() {
            // PictureBrowser is dropped on the UI thread. Reap workers without
            // blocking window shutdown on a currently non-preemptible RAW call.
            let shared = self.shared.clone();
            let _ = thread::Builder::new()
                .name("pics-render-reaper".into())
                .spawn(move || {
                    for worker in workers {
                        let _ = worker.join();
                    }
                    if diagnostics {
                        let state = shared.state.lock().unwrap();
                        eprintln!(
                            "pics scheduler: completed={}, cancelled={}, queue-ms={}, execution-ms={}",
                            state.completed,
                            state.cancelled,
                            state.total_queue_latency.as_millis(),
                            state.total_execution.as_millis(),
                        );
                    }
                });
        }
    }
}

fn effective_priority(job: &Job) -> Option<JobPriority> {
    job.demands.values().copied().max()
}

fn new_queued_job(sequence: u64) -> Job {
    Job {
        demands: HashMap::new(),
        subscribers: Vec::new(),
        status: JobStatus::Queued,
        sequence,
        queued_at: Instant::now(),
        cancelled: Arc::new(AtomicBool::new(false)),
    }
}

fn add_demand(
    job: &mut Job,
    scope: DemandScopeId,
    demand: &JobDemand,
    subscriptions: &mut Vec<Subscription>,
) {
    job.demands.insert(scope, demand.priority);
    if demand.subscribe {
        let (sender, receiver) = oneshot::channel();
        job.subscribers.push(Subscriber { scope, sender });
        subscriptions.push(Subscription {
            key: demand.key.clone(),
            receiver,
        });
    }
}

fn refresh_queue_priority(state: &mut State, key: &JobKey) {
    let Some(job) = state.jobs.get(key) else {
        return;
    };
    if job.status != JobStatus::Queued {
        return;
    }
    if let Some(effective) = effective_priority(job) {
        state.queue.push(
            key.clone(),
            QueuePriority {
                effective,
                sequence: Reverse(job.sequence),
            },
        );
    }
}

fn age_queued_jobs(state: &mut State, interval: Duration) {
    if interval.is_zero() {
        return;
    }
    let now = Instant::now();
    let keys: Vec<_> = state.queue.iter().map(|(key, _)| key.clone()).collect();
    for key in keys {
        let Some(job) = state.jobs.get(&key) else {
            continue;
        };
        let Some(mut priority) = effective_priority(job) else {
            continue;
        };
        if priority.class < PriorityClass::NearViewport {
            let steps = (now.duration_since(job.queued_at).as_nanos() / interval.as_nanos().max(1))
                .min(3) as u8;
            let class = (priority.class as u8 + steps).min(PriorityClass::NearViewport as u8);
            priority.class = match class {
                0 => PriorityClass::Maintenance,
                1 => PriorityClass::Speculative,
                2 => PriorityClass::NearViewport,
                _ => PriorityClass::NearViewport,
            };
        }
        state.queue.change_priority(
            &key,
            QueuePriority {
                effective: priority,
                sequence: Reverse(job.sequence),
            },
        );
    }
}

fn remove_scope_interest(
    state: &mut State,
    key: &JobKey,
    scope: DemandScopeId,
    cancelled_senders: &mut Vec<oneshot::Sender<JobCompletion>>,
) {
    let mut remove_pending = false;
    if let Some(job) = state.pending.get_mut(key) {
        job.demands.remove(&scope);
        if let Some((&replacement_scope, _)) =
            job.demands.iter().max_by_key(|(_, priority)| *priority)
        {
            for subscriber in &mut job.subscribers {
                if subscriber.scope == scope {
                    subscriber.scope = replacement_scope;
                }
            }
        } else {
            cancelled_senders.extend(
                job.subscribers
                    .drain(..)
                    .map(|subscriber| subscriber.sender),
            );
        }
        remove_pending = job.demands.is_empty();
    }
    if remove_pending {
        state.cancelled += 1;
        state.pending.remove(key);
    }

    let mut remove_queued = false;
    if let Some(job) = state.jobs.get_mut(key) {
        job.demands.remove(&scope);
        if let Some((&replacement_scope, _)) =
            job.demands.iter().max_by_key(|(_, priority)| *priority)
        {
            for subscriber in &mut job.subscribers {
                if subscriber.scope == scope {
                    // The subscription represents delivery into the shared
                    // memory cache, not ownership by one viewport. Preserve it
                    // across near/speculative -> visible scope handoffs.
                    subscriber.scope = replacement_scope;
                }
            }
        } else {
            cancelled_senders.extend(
                job.subscribers
                    .drain(..)
                    .map(|subscriber| subscriber.sender),
            );
        }
        if job.demands.is_empty() {
            job.cancelled.store(true, Ordering::Release);
            remove_queued = job.status == JobStatus::Queued;
        }
    }
    if remove_queued {
        state.cancelled += 1;
        state.queue.remove(key);
        state.jobs.remove(key);
    } else {
        refresh_queue_priority(state, key);
    }
}

fn pop_next_job(state: &mut State, foreground_only: bool) -> Option<(JobKey, JobPriority)> {
    if !foreground_only {
        return state
            .queue
            .pop()
            .map(|(key, priority)| (key, priority.effective));
    }
    let key = state
        .queue
        .iter()
        .filter(|(_, priority)| priority.effective.class >= PriorityClass::Visible)
        .max_by_key(|(_, priority)| **priority)
        .map(|(key, _)| key.clone())?;
    let priority = state.queue.remove(&key)?.1.effective;
    Some((key, priority))
}

fn worker_loop(shared: Arc<Shared>, foreground_only: bool) {
    loop {
        let (key, priority, cancellation, started_at) = {
            let mut state = shared.state.lock().unwrap();
            loop {
                if state.shutdown {
                    return;
                }
                age_queued_jobs(&mut state, shared.aging_interval);
                if let Some((key, priority)) = pop_next_job(&mut state, foreground_only) {
                    let (cancellation, queue_latency) = {
                        let Some(job) = state.jobs.get_mut(&key) else {
                            continue;
                        };
                        job.status = JobStatus::Running;
                        (job.cancelled.clone(), job.queued_at.elapsed())
                    };
                    state.total_queue_latency += queue_latency;
                    state.running += 1;
                    break (key, priority, cancellation, Instant::now());
                }
                state = shared.wake.wait(state).unwrap();
            }
        };

        let result = catch_unwind(AssertUnwindSafe(|| {
            (shared.executor)(&key, priority, &cancellation)
        }))
        .unwrap_or_else(|_| PreviewState::Error("render worker panicked".into()));

        let (subscribers, completion) = {
            let mut state = shared.state.lock().unwrap();
            state.running = state.running.saturating_sub(1);
            state.total_execution += started_at.elapsed();
            let Some(job) = state.jobs.remove(&key) else {
                continue;
            };
            let cancelled = cancellation.load(Ordering::Acquire) || job.demands.is_empty();
            if cancelled {
                state.cancelled += 1;
            } else {
                state.completed += 1;
            }
            if let Some(successor) = state.pending.remove(&key)
                && !successor.demands.is_empty()
            {
                state.jobs.insert(key.clone(), successor);
                refresh_queue_priority(&mut state, &key);
            }
            (job.subscribers, if cancelled { None } else { Some(result) })
        };
        for subscriber in subscribers {
            let message = completion
                .as_ref()
                .cloned()
                .map(JobCompletion::Finished)
                .unwrap_or(JobCompletion::Cancelled);
            let _ = subscriber.sender.send(message);
        }
        shared.wake.notify_all();
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
