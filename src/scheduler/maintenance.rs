use std::{
    cmp::Reverse,
    collections::{HashMap, VecDeque},
    hash::Hash,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

/// Point-in-time view of the exclusive maintenance permit, for diagnostic
/// surfaces: whether foreground work is gating maintenance, whether the
/// permit is held (and at what rank), and how much is waiting.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AdmissionSnapshot {
    pub foreground_blocked: bool,
    pub foreground_activities: usize,
    pub active_rank: Option<i32>,
    pub waiters: usize,
    pub advertised: usize,
}

#[derive(Clone)]
pub struct MaintenanceAdmission {
    shared: Arc<AdmissionShared>,
}

struct AdmissionShared {
    state: Mutex<AdmissionState>,
    wake: Condvar,
}

#[derive(Default)]
struct AdmissionState {
    foreground_blocked: bool,
    foreground_activities: usize,
    active: Option<(thread::ThreadId, u64, i32)>,
    next_ticket: u64,
    waiters: HashMap<u64, AdmissionWaiter>,
    next_scheduler_id: u64,
    advertised: HashMap<u64, AdmissionWaiter>,
}

struct AdmissionWaiter {
    rank: i32,
    queued_at: Instant,
}

struct MaintenancePermit {
    admission: MaintenanceAdmission,
    owner: thread::ThreadId,
    scheduler_id: u64,
}

pub struct ForegroundActivity {
    admission: MaintenanceAdmission,
}

impl MaintenanceAdmission {
    pub fn new() -> Self {
        Self {
            shared: Arc::new(AdmissionShared {
                state: Mutex::new(AdmissionState::default()),
                wake: Condvar::new(),
            }),
        }
    }

    pub fn set_foreground_blocked(&self, blocked: bool) {
        self.shared.state.lock().unwrap().foreground_blocked = blocked;
        if !blocked {
            self.shared.wake.notify_all();
        }
    }

    pub fn begin_foreground_activity(&self) -> ForegroundActivity {
        let mut state = self.shared.state.lock().unwrap();
        state.foreground_activities += 1;
        let activities = state.foreground_activities;
        drop(state);
        self.shared.wake.notify_all();
        crate::trace::slow_event(&format!(
            "foreground activity started · background maintenance gated ({activities} active)"
        ));
        ForegroundActivity {
            admission: self.clone(),
        }
    }

    pub fn run_maintenance<R>(
        &self,
        rank: i32,
        cancelled: &AtomicBool,
        run: impl FnOnce() -> R,
    ) -> Option<R> {
        let scheduler_id = self.register_scheduler();
        let _permit = self.acquire(scheduler_id, rank, Instant::now(), cancelled, &|| false)?;
        Some(run())
    }

    fn foreground_blocked(state: &AdmissionState) -> bool {
        state.foreground_blocked || state.foreground_activities > 0
    }

    pub(crate) fn is_foreground_blocked(&self) -> bool {
        Self::foreground_blocked(&self.shared.state.lock().unwrap())
    }

    /// Point-in-time admission state for diagnostics surfaces.
    pub fn snapshot(&self) -> AdmissionSnapshot {
        let state = self.shared.state.lock().unwrap();
        AdmissionSnapshot {
            foreground_blocked: Self::foreground_blocked(&state),
            foreground_activities: state.foreground_activities,
            active_rank: state.active.map(|(_, _, rank)| rank),
            waiters: state.waiters.len(),
            advertised: state.advertised.len(),
        }
    }

    fn acquire(
        &self,
        scheduler_id: u64,
        rank: i32,
        queued_at: Instant,
        cancelled: &AtomicBool,
        stale: &dyn Fn() -> bool,
    ) -> Option<MaintenancePermit> {
        let mut state = self.shared.state.lock().unwrap();
        let ticket = state.next_ticket;
        state.next_ticket = state.next_ticket.wrapping_add(1);
        state
            .waiters
            .insert(ticket, AdmissionWaiter { rank, queued_at });
        self.shared.wake.notify_all();
        loop {
            if cancelled.load(Ordering::Acquire) || stale() {
                state.waiters.remove(&ticket);
                self.shared.wake.notify_all();
                return None;
            }
            let waiter_rank = admission_effective_rank(state.waiters.get(&ticket).unwrap());
            let beats_advertised = state.advertised.iter().all(|(other_id, other)| {
                *other_id == scheduler_id || admission_effective_rank(other) <= waiter_rank
            });
            let is_best = state.waiters.iter().all(|(other_ticket, other)| {
                (admission_effective_rank(other), Reverse(*other_ticket))
                    <= (waiter_rank, Reverse(ticket))
            });
            if !Self::foreground_blocked(&state)
                && state.active.is_none()
                && beats_advertised
                && is_best
            {
                state.waiters.remove(&ticket);
                let owner = thread::current().id();
                state.active = Some((owner, scheduler_id, rank));
                let waited = queued_at.elapsed();
                if waited >= Duration::from_secs(5) {
                    crate::trace::slow_event(&format!(
                        "admission: rank {rank} acquired after {} (waiters {} · advertised {} · foreground activities {})",
                        crate::trace::secs(waited),
                        state.waiters.len(),
                        state.advertised.len(),
                        state.foreground_activities,
                    ));
                } else {
                    crate::trace::event(&format!("admission: rank {rank} acquired"));
                }
                return Some(MaintenancePermit {
                    admission: self.clone(),
                    owner,
                    scheduler_id,
                });
            }
            let (next, _) = self
                .shared
                .wake
                .wait_timeout(state, Duration::from_millis(20))
                .unwrap();
            state = next;
            let waited = queued_at.elapsed();
            if waited >= Duration::from_secs(10)
                && waited - Duration::from_secs(waited.as_secs() / 10 * 10)
                    < Duration::from_millis(20)
            {
                crate::trace::slow_event(&format!(
                    "admission: rank {rank} STILL waiting after {} (waiters {} · advertised {} · foreground activities {})",
                    crate::trace::secs(waited),
                    state.waiters.len(),
                    state.advertised.len(),
                    state.foreground_activities,
                ));
            }
        }
    }

    fn register_scheduler(&self) -> u64 {
        let mut state = self.shared.state.lock().unwrap();
        let id = state.next_scheduler_id;
        state.next_scheduler_id = state.next_scheduler_id.wrapping_add(1);
        id
    }

    fn advertise(&self, scheduler_id: u64, candidate: Option<(i32, Instant)>) {
        let mut state = self.shared.state.lock().unwrap();
        if let Some((rank, queued_at)) = candidate {
            state
                .advertised
                .insert(scheduler_id, AdmissionWaiter { rank, queued_at });
        } else {
            state.advertised.remove(&scheduler_id);
        }
        drop(state);
        self.shared.wake.notify_all();
    }

    fn wait(&self, cancelled: &AtomicBool) -> bool {
        let mut state = self.shared.state.lock().unwrap();
        while Self::foreground_blocked(&state) && !cancelled.load(Ordering::Acquire) {
            let (next, _) = self
                .shared
                .wake
                .wait_timeout(state, Duration::from_millis(20))
                .unwrap();
            state = next;
        }
        !cancelled.load(Ordering::Acquire)
    }

    /// Cooperatively yields a maintenance stage while foreground work owns
    /// admission. Returns false when the job was cancelled while waiting.
    pub(crate) fn checkpoint(&self, cancelled: &AtomicBool) -> bool {
        let owner = thread::current().id();
        let mut state = self.shared.state.lock().unwrap();
        let Some((_, scheduler_id, rank)) = state.active.filter(|(active, _, _)| *active == owner)
        else {
            drop(state);
            let wait_started = Instant::now();
            let resumed = self.wait(cancelled);
            let waited = wait_started.elapsed();
            if resumed && waited >= Duration::from_secs(5) {
                crate::trace::slow_event(&format!(
                    "admission: checkpoint blocked by foreground gating for {}",
                    crate::trace::secs(waited),
                ));
            }
            return resumed;
        };
        let higher_waiting = state
            .waiters
            .values()
            .any(|waiter| admission_effective_rank(waiter) > rank)
            || state.advertised.iter().any(|(other_scheduler, waiter)| {
                *other_scheduler != scheduler_id && admission_effective_rank(waiter) > rank
            });
        if !Self::foreground_blocked(&state) && !higher_waiting {
            return !cancelled.load(Ordering::Acquire);
        }

        // A cooperative stage boundary temporarily returns the global resource
        // permit. Foreground work always wins; otherwise the highest-ranked
        // maintenance waiter runs before this job resumes.
        let yield_started = Instant::now();
        crate::trace::event(&format!("admission: rank {rank} yielding at a checkpoint"));
        state.active = None;
        let ticket = state.next_ticket;
        state.next_ticket = state.next_ticket.wrapping_add(1);
        state.waiters.insert(
            ticket,
            AdmissionWaiter {
                rank,
                queued_at: Instant::now(),
            },
        );
        self.shared.wake.notify_all();
        loop {
            if cancelled.load(Ordering::Acquire) {
                state.waiters.remove(&ticket);
                self.shared.wake.notify_all();
                return false;
            }
            let waiter_rank = admission_effective_rank(state.waiters.get(&ticket).unwrap());
            // A queued job from this same scheduler cannot execute until this
            // worker exits (notably for the production one-worker services),
            // so it must not self-deadlock the active job's reacquisition.
            // Other schedulers' advertisements remain authoritative even
            // before their workers have registered admission waiters.
            let beats_advertised = state.advertised.iter().all(|(other_scheduler, other)| {
                *other_scheduler == scheduler_id || admission_effective_rank(other) <= waiter_rank
            });
            let is_best = state.waiters.iter().all(|(other_ticket, other)| {
                (admission_effective_rank(other), Reverse(*other_ticket))
                    <= (waiter_rank, Reverse(ticket))
            });
            if !Self::foreground_blocked(&state)
                && state.active.is_none()
                && beats_advertised
                && is_best
            {
                state.waiters.remove(&ticket);
                state.active = Some((owner, scheduler_id, rank));
                let waited = yield_started.elapsed();
                if waited >= Duration::from_secs(5) {
                    crate::trace::slow_event(&format!(
                        "admission: yielded rank {rank} resumed after {} (waiters {} · advertised {})",
                        crate::trace::secs(waited),
                        state.waiters.len(),
                        state.advertised.len(),
                    ));
                }
                return true;
            }
            let (next, _) = self
                .shared
                .wake
                .wait_timeout(state, Duration::from_millis(20))
                .unwrap();
            state = next;
        }
    }

    #[cfg(test)]
    fn wait_until_waiters(&self, count: usize, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        let mut state = self.shared.state.lock().unwrap();
        while state.waiters.len() < count {
            let Some(remaining) = deadline.checked_duration_since(Instant::now()) else {
                return false;
            };
            let (next, result) = self.shared.wake.wait_timeout(state, remaining).unwrap();
            state = next;
            if result.timed_out() && state.waiters.len() < count {
                return false;
            }
        }
        true
    }
}

impl Drop for ForegroundActivity {
    fn drop(&mut self) {
        let mut state = self.admission.shared.state.lock().unwrap();
        state.foreground_activities = state.foreground_activities.saturating_sub(1);
        let activities = state.foreground_activities;
        drop(state);
        self.admission.shared.wake.notify_all();
        crate::trace::slow_event(&format!(
            "foreground activity finished · background maintenance resumes ({activities} active)"
        ));
    }
}

impl Drop for MaintenancePermit {
    fn drop(&mut self) {
        let mut state = self.admission.shared.state.lock().unwrap();
        if state.active.is_some_and(|(owner, scheduler_id, _)| {
            owner == self.owner && scheduler_id == self.scheduler_id
        }) {
            state.active = None;
        }
        drop(state);
        self.admission.shared.wake.notify_all();
    }
}

impl Default for MaintenanceAdmission {
    fn default() -> Self {
        Self::new()
    }
}

/// Snapshot status for a typed maintenance job.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaintenanceJobStatus {
    Queued,
    Running,
}

/// A maintenance executor stopped cooperatively so higher-priority work can
/// proceed. Its durable queue remains authoritative; this is neither success
/// nor failure in scheduler diagnostics.
#[derive(Debug)]
pub struct MaintenanceYield(pub &'static str);

impl std::fmt::Display for MaintenanceYield {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.0)
    }
}

impl std::error::Error for MaintenanceYield {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaintenanceJobSnapshot<K> {
    pub key: K,
    pub rank: i32,
    pub status: MaintenanceJobStatus,
    pub sequence: u64,
    pub cancelled: bool,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MaintenanceStats {
    pub queued: usize,
    pub running: usize,
    pub completed: u64,
    pub cancelled: u64,
    pub failed: u64,
    pub panicked: u64,
    pub total_queue_latency_ms: u64,
    pub total_execution_ms: u64,
    pub completed_last_minute: usize,
    pub last_error: Option<String>,
}

type Executor<K> = dyn Fn(&K, &Arc<AtomicBool>) -> anyhow::Result<()> + Send + Sync + 'static;
type WorkerExecutor<K> = dyn FnMut(&K, &Arc<AtomicBool>) -> anyhow::Result<()> + Send + 'static;
type ExecutorFactory<K> = dyn Fn(usize) -> Box<WorkerExecutor<K>> + Send + Sync + 'static;

/// A small typed queue for durable, restartable maintenance work.
///
/// Keys coalesce duplicate submissions. Cancellation is cooperative: executors
/// must check the supplied token between expensive stages and leave durable
/// state retryable. This scheduler deliberately has no result channel because
/// maintenance completion is represented by the catalog, not transient UI state.
pub struct MaintenanceScheduler<K>
where
    K: Clone + Eq + Hash + Send + 'static,
{
    shared: Arc<Shared<K>>,
    workers: Vec<JoinHandle<()>>,
}

struct Shared<K> {
    state: Mutex<State<K>>,
    wake: Condvar,
    admission: MaintenanceAdmission,
    queue_revision: AtomicU64,
    scheduler_id: u64,
}

struct State<K> {
    jobs: HashMap<K, Job>,
    queue: VecDeque<K>,
    next_sequence: u64,
    running: usize,
    completed: u64,
    cancelled: u64,
    failed: u64,
    panicked: u64,
    total_queue_latency: Duration,
    total_execution: Duration,
    recent_completions: VecDeque<Instant>,
    last_error: Option<String>,
    shutdown: bool,
    advertised_key: Option<K>,
}

struct Job {
    rank: i32,
    sequence: u64,
    queued_at: Instant,
    status: MaintenanceJobStatus,
    cancelled: Arc<AtomicBool>,
    /// A submission observed while this key was running. The worker converts
    /// it into a trailing-edge execution atomically with completing this one.
    rerun_requested: bool,
    rerun_blocked: bool,
}

#[cfg(not(test))]
// With the largest production rank gap (selected subtree 200 versus distant
// work 0), starvation is bounded to roughly 200 seconds.
const MAINTENANCE_AGING_INTERVAL: Duration = Duration::from_secs(1);
#[cfg(test)]
const MAINTENANCE_AGING_INTERVAL: Duration = Duration::from_millis(10);
// Aging must stay bounded: queued work ages from its original enqueue time,
// so an unbounded backlog's advertisement (rank 200 + hours) would
// eventually outrank — and permanently wedge — the higher-ranked inventory
// pass (rank 1000). The cap keeps every analysis rank below the inventory
// while still reordering work within the analysis backlog.
#[cfg(not(test))]
const MAINTENANCE_AGING_CAP: Duration = Duration::from_secs(500);
#[cfg(test)]
const MAINTENANCE_AGING_CAP: Duration = Duration::from_millis(500);

fn aged_rank(rank: i32, queued_at: Instant) -> i32 {
    let age = queued_at.elapsed().min(MAINTENANCE_AGING_CAP);
    rank.saturating_add(
        (age.as_millis() / MAINTENANCE_AGING_INTERVAL.as_millis().max(1)).min(i32::MAX as u128)
            as i32,
    )
}

fn admission_effective_rank(waiter: &AdmissionWaiter) -> i32 {
    aged_rank(waiter.rank, waiter.queued_at)
}

fn effective_rank(job: &Job) -> i32 {
    aged_rank(job.rank, job.queued_at)
}

fn advertise_best<K>(shared: &Shared<K>, state: &mut State<K>)
where
    K: Clone + Eq + Hash + Send + 'static,
{
    let candidate = state
        .queue
        .iter()
        .filter_map(|key| state.jobs.get(key))
        .max_by_key(|job| (effective_rank(job), Reverse(job.sequence)))
        .map(|job| (job.rank, job.queued_at));
    state.advertised_key = state
        .queue
        .iter()
        .filter_map(|key| state.jobs.get(key).map(|job| (key, job)))
        .max_by_key(|(_, job)| (effective_rank(job), Reverse(job.sequence)))
        .map(|(key, _)| key.clone());
    shared.admission.advertise(shared.scheduler_id, candidate);
}

fn consider_advertisement<K>(shared: &Shared<K>, state: &mut State<K>, key: &K)
where
    K: Clone + Eq + Hash + Send + 'static,
{
    let Some(candidate) = state.jobs.get(key) else {
        return;
    };
    let replace = state.advertised_key.as_ref() == Some(key)
        || state
            .advertised_key
            .as_ref()
            .and_then(|advertised| state.jobs.get(advertised))
            .is_none_or(|advertised| {
                (effective_rank(candidate), Reverse(candidate.sequence))
                    > (effective_rank(advertised), Reverse(advertised.sequence))
            });
    if replace {
        state.advertised_key = Some(key.clone());
        shared.admission.advertise(
            shared.scheduler_id,
            Some((candidate.rank, candidate.queued_at)),
        );
    }
}

impl<K> MaintenanceScheduler<K>
where
    K: Clone + Eq + Hash + Send + 'static,
{
    #[cfg(test)]
    pub fn new(parallelism: usize, executor: Arc<Executor<K>>) -> Self {
        Self::new_with_admission(parallelism, MaintenanceAdmission::new(), executor)
    }

    pub fn new_with_admission(
        parallelism: usize,
        admission: MaintenanceAdmission,
        executor: Arc<Executor<K>>,
    ) -> Self {
        Self::new_with_admission_factory(
            parallelism,
            admission,
            Arc::new(move |_| {
                let executor = executor.clone();
                Box::new(move |key, cancelled| executor(key, cancelled))
            }),
        )
    }

    /// Builds one stateful executor per worker. This is useful for expensive
    /// thread-affine state such as inference sessions that must be reused
    /// without being shared concurrently.
    pub fn new_with_admission_factory(
        parallelism: usize,
        admission: MaintenanceAdmission,
        executor_factory: Arc<ExecutorFactory<K>>,
    ) -> Self {
        let scheduler_id = admission.register_scheduler();
        let shared = Arc::new(Shared {
            state: Mutex::new(State {
                jobs: HashMap::new(),
                queue: VecDeque::new(),
                next_sequence: 0,
                running: 0,
                completed: 0,
                cancelled: 0,
                failed: 0,
                panicked: 0,
                total_queue_latency: Duration::ZERO,
                total_execution: Duration::ZERO,
                recent_completions: VecDeque::new(),
                last_error: None,
                shutdown: false,
                advertised_key: None,
            }),
            wake: Condvar::new(),
            admission,
            queue_revision: AtomicU64::new(0),
            scheduler_id,
        });
        let workers = (0..parallelism)
            .map(|index| {
                let shared = shared.clone();
                let mut executor = executor_factory(index);
                thread::Builder::new()
                    .name(format!("rekesz-maintenance-{index}"))
                    .spawn(move || worker_loop(shared, &mut executor))
                    .expect("failed to start maintenance worker")
            })
            .collect();
        Self { shared, workers }
    }

    /// Enqueue or promote a job. Returns false when an identical key was
    /// already queued/running; its rank is still promoted when appropriate.
    pub fn submit(&self, key: K, rank: i32) -> bool {
        let mut state = self.shared.state.lock().unwrap();
        if let Some(job) = state.jobs.get_mut(&key) {
            let previous_rank = job.rank;
            job.rank = job.rank.max(rank);
            if job.status == MaintenanceJobStatus::Running && !job.rerun_blocked {
                job.rerun_requested = true;
            }
            if job.rank != previous_rank {
                self.shared.queue_revision.fetch_add(1, Ordering::AcqRel);
                consider_advertisement(&self.shared, &mut state, &key);
                self.shared.admission.shared.wake.notify_all();
            }
            return false;
        }
        let sequence = state.next_sequence;
        state.next_sequence += 1;
        state.jobs.insert(
            key.clone(),
            Job {
                rank,
                sequence,
                queued_at: Instant::now(),
                status: MaintenanceJobStatus::Queued,
                cancelled: Arc::new(AtomicBool::new(false)),
                rerun_requested: false,
                rerun_blocked: false,
            },
        );
        consider_advertisement(&self.shared, &mut state, &key);
        state.queue.push_back(key);
        self.shared.queue_revision.fetch_add(1, Ordering::AcqRel);
        drop(state);
        self.shared.wake.notify_one();
        true
    }

    pub fn cancel(&self, key: &K) -> bool {
        self.cancel_and_status(key).is_some()
    }

    pub(crate) fn cancel_and_status(&self, key: &K) -> Option<MaintenanceJobStatus> {
        let mut state = self.shared.state.lock().unwrap();
        let job = state.jobs.get_mut(key)?;
        let status = job.status;
        job.cancelled.store(true, Ordering::Release);
        job.rerun_requested = false;
        if job.status == MaintenanceJobStatus::Queued {
            state.queue.retain(|queued| queued != key);
            state.jobs.remove(key);
            state.cancelled += 1;
            self.shared.queue_revision.fetch_add(1, Ordering::AcqRel);
            advertise_best(&self.shared, &mut state);
        }
        drop(state);
        self.shared.wake.notify_all();
        self.shared.admission.shared.wake.notify_all();
        Some(status)
    }

    pub fn cancel_all(&self) {
        let mut state = self.shared.state.lock().unwrap();
        for job in state.jobs.values_mut() {
            job.cancelled.store(true, Ordering::Release);
            job.rerun_requested = false;
            job.rerun_blocked = true;
        }
        let queued = state.queue.len() as u64;
        state.queue.clear();
        self.shared.queue_revision.fetch_add(1, Ordering::AcqRel);
        advertise_best(&self.shared, &mut state);
        state
            .jobs
            .retain(|_, job| job.status == MaintenanceJobStatus::Running);
        state.cancelled += queued;
        drop(state);
        self.shared.wake.notify_all();
        self.shared.admission.shared.wake.notify_all();
    }

    pub fn snapshot(&self) -> Vec<MaintenanceJobSnapshot<K>> {
        let state = self.shared.state.lock().unwrap();
        let mut jobs = state
            .jobs
            .iter()
            .map(|(key, job)| MaintenanceJobSnapshot {
                key: key.clone(),
                rank: job.rank,
                status: job.status,
                sequence: job.sequence,
                cancelled: job.cancelled.load(Ordering::Acquire),
            })
            .collect::<Vec<_>>();
        jobs.sort_by_key(|job| {
            (
                job.status != MaintenanceJobStatus::Running,
                Reverse(job.rank),
                job.sequence,
            )
        });
        jobs
    }

    pub fn stats(&self) -> MaintenanceStats {
        let mut state = self.shared.state.lock().unwrap();
        let cutoff = Instant::now() - Duration::from_secs(60);
        while state
            .recent_completions
            .front()
            .is_some_and(|completed| *completed < cutoff)
        {
            state.recent_completions.pop_front();
        }
        MaintenanceStats {
            queued: state.queue.len(),
            running: state.running,
            completed: state.completed,
            cancelled: state.cancelled,
            failed: state.failed,
            panicked: state.panicked,
            total_queue_latency_ms: state.total_queue_latency.as_millis() as u64,
            total_execution_ms: state.total_execution.as_millis() as u64,
            completed_last_minute: state.recent_completions.len(),
            last_error: state.last_error.clone(),
        }
    }

    /// Waits until every currently submitted job reaches a terminal state.
    ///
    /// This uses the scheduler's state transition condition rather than polling,
    /// so tests and orderly shutdown paths do not steal CPU from inference workers
    /// while the machine is under load.
    #[cfg(test)]
    pub fn wait_until_settled(&self, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        let mut state = self.shared.state.lock().unwrap();
        while !state.jobs.is_empty() {
            let Some(remaining) = deadline.checked_duration_since(Instant::now()) else {
                return false;
            };
            let (next, result) = self.shared.wake.wait_timeout(state, remaining).unwrap();
            state = next;
            if result.timed_out() && !state.jobs.is_empty() {
                return false;
            }
        }
        true
    }
}

impl<K> Drop for MaintenanceScheduler<K>
where
    K: Clone + Eq + Hash + Send + 'static,
{
    fn drop(&mut self) {
        {
            let mut state = self.shared.state.lock().unwrap();
            state.shutdown = true;
            state.cancelled += state.queue.len() as u64;
            for job in state.jobs.values() {
                job.cancelled.store(true, Ordering::Release);
            }
            self.shared
                .admission
                .advertise(self.shared.scheduler_id, None);
        }
        self.shared.wake.notify_all();
        let workers = std::mem::take(&mut self.workers);
        if !workers.is_empty() {
            let _ = thread::Builder::new()
                .name("rekesz-maintenance-reaper".into())
                .spawn(move || {
                    for worker in workers {
                        let _ = worker.join();
                    }
                });
        }
    }
}

fn worker_loop<K>(shared: Arc<Shared<K>>, executor: &mut WorkerExecutor<K>)
where
    K: Clone + Eq + Hash + Send + 'static,
{
    'worker: loop {
        let (key, cancellation, permit, started_at) = loop {
            let (key, cancellation, rank, queued_at, revision) = {
                let mut state = shared.state.lock().unwrap();
                loop {
                    if state.shutdown {
                        return;
                    }
                    let best = state
                        .queue
                        .iter()
                        .filter_map(|key| {
                            let job = state.jobs.get(key)?;
                            Some((key.clone(), effective_rank(job), Reverse(job.sequence)))
                        })
                        .max_by_key(|(_, rank, sequence)| (*rank, *sequence));
                    if let Some((key, _, _)) = best {
                        let job = state.jobs.get(&key).expect("queued job exists");
                        break (
                            key,
                            job.cancelled.clone(),
                            job.rank,
                            job.queued_at,
                            shared.queue_revision.load(Ordering::Acquire),
                        );
                    }
                    state = shared.wake.wait(state).unwrap();
                }
            };
            let stale = || shared.queue_revision.load(Ordering::Acquire) != revision;
            let Some(permit) = shared.admission.acquire(
                shared.scheduler_id,
                rank,
                queued_at,
                &cancellation,
                &stale,
            ) else {
                continue 'worker;
            };
            let mut state = shared.state.lock().unwrap();
            if state.shutdown {
                return;
            }
            if shared.queue_revision.load(Ordering::Acquire) != revision
                || !state
                    .jobs
                    .get(&key)
                    .is_some_and(|job| job.status == MaintenanceJobStatus::Queued)
            {
                drop(state);
                drop(permit);
                continue;
            }
            let Some(index) = state.queue.iter().position(|queued| queued == &key) else {
                drop(state);
                drop(permit);
                continue;
            };
            state.queue.remove(index);
            let latency = {
                let job = state.jobs.get_mut(&key).expect("queued job exists");
                job.status = MaintenanceJobStatus::Running;
                job.queued_at.elapsed()
            };
            state.running += 1;
            state.total_queue_latency += latency;
            shared.queue_revision.fetch_add(1, Ordering::AcqRel);
            advertise_best(&shared, &mut state);
            shared.admission.shared.wake.notify_all();
            break (key, cancellation, permit, Instant::now());
        };
        let result = catch_unwind(AssertUnwindSafe(|| executor(&key, &cancellation)));
        drop(permit);
        let mut state = shared.state.lock().unwrap();
        state.running = state.running.saturating_sub(1);
        state.total_execution += started_at.elapsed();
        let rerun = state.jobs.get(&key).is_some_and(|job| job.rerun_requested)
            && !state.jobs.get(&key).is_some_and(|job| job.rerun_blocked)
            && !state.shutdown;
        if cancellation.load(Ordering::Acquire) {
            state.cancelled += 1;
        } else {
            match result {
                Ok(Ok(())) => {
                    state.completed += 1;
                    state.recent_completions.push_back(Instant::now());
                    state.last_error = None;
                }
                Ok(Err(error)) => {
                    if error.downcast_ref::<MaintenanceYield>().is_none() {
                        state.failed += 1;
                        state.last_error = Some(format!("{error:#}"));
                    }
                }
                Err(_) => {
                    state.panicked += 1;
                    state.last_error = Some("maintenance worker panicked".to_owned());
                }
            }
        }
        if rerun {
            let sequence = state.next_sequence;
            state.next_sequence += 1;
            let job = state.jobs.get_mut(&key).expect("running job exists");
            job.sequence = sequence;
            job.queued_at = Instant::now();
            job.status = MaintenanceJobStatus::Queued;
            job.cancelled = Arc::new(AtomicBool::new(false));
            job.rerun_requested = false;
            job.rerun_blocked = false;
            state.queue.push_back(key);
            shared.queue_revision.fetch_add(1, Ordering::AcqRel);
        } else {
            state.jobs.remove(&key);
        }
        advertise_best(&shared, &mut state);
        drop(state);
        shared.wake.notify_all();
    }
}

#[cfg(test)]
#[path = "maintenance_tests.rs"]
mod tests;
