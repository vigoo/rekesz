use std::{
    sync::{
        Condvar, Mutex,
        atomic::{AtomicUsize, Ordering},
        mpsc,
    },
    time::{Duration, Instant},
};
use test_r::test;

use futures::executor::block_on;

use super::*;

fn key(name: &str) -> JobKey {
    JobKey {
        path: PathBuf::from(name),
        variant: RenderVariant::Thumbnail { width: 300 },
        render_identity: RenderIdentity::default(),
        recipe: None,
    }
}

fn demand(key: JobKey, priority: JobPriority) -> JobDemand {
    JobDemand {
        key,
        priority,
        subscribe: true,
    }
}

fn error_result(name: &str) -> PreviewState {
    PreviewState::Error(name.to_string().into())
}

#[test]
fn interactive_work_overtakes_queued_visible_work() {
    let gate = Arc::new((Mutex::new(false), Condvar::new()));
    let order = Arc::new(Mutex::new(Vec::new()));
    let (started_tx, started_rx) = mpsc::channel();
    let scheduler = RenderScheduler::new(1, {
        let gate = gate.clone();
        let order = order.clone();
        Arc::new(move |job, _, _| {
            order.lock().unwrap().push(job.path.clone());
            if job.path == std::path::Path::new("blocker") {
                started_tx.send(()).unwrap();
                let (lock, wake) = &*gate;
                let mut released = lock.lock().unwrap();
                while !*released {
                    released = wake.wait(released).unwrap();
                }
            }
            error_result("done")
        })
    });
    scheduler.replace_scope(
        DemandScopeId(1),
        vec![demand(key("blocker"), JobPriority::INTERACTIVE)],
    );
    started_rx.recv_timeout(Duration::from_secs(2)).unwrap();
    scheduler.replace_scope(
        DemandScopeId(2),
        vec![demand(key("visible"), JobPriority::visible(0))],
    );
    scheduler.replace_scope(
        DemandScopeId(3),
        vec![demand(key("interactive"), JobPriority::INTERACTIVE)],
    );
    {
        let (lock, wake) = &*gate;
        *lock.lock().unwrap() = true;
        wake.notify_all();
    }
    let deadline = Instant::now() + Duration::from_secs(2);
    while scheduler.stats().completed < 3 && Instant::now() < deadline {
        thread::yield_now();
    }
    assert_eq!(
        *order.lock().unwrap(),
        vec![
            key("blocker").path,
            key("interactive").path,
            key("visible").path
        ]
    );
}

#[test]
fn duplicate_demands_execute_once_and_notify_both_consumers() {
    let gate = Arc::new((Mutex::new(false), Condvar::new()));
    let executions = Arc::new(AtomicUsize::new(0));
    let (started_tx, started_rx) = mpsc::channel();
    let scheduler = RenderScheduler::new(1, {
        let gate = gate.clone();
        let executions = executions.clone();
        Arc::new(move |_, _, _| {
            executions.fetch_add(1, Ordering::SeqCst);
            started_tx.send(()).unwrap();
            let (lock, wake) = &*gate;
            let mut released = lock.lock().unwrap();
            while !*released {
                released = wake.wait(released).unwrap();
            }
            error_result("done")
        })
    });
    let first = scheduler
        .replace_scope(
            DemandScopeId(1),
            vec![demand(key("same"), JobPriority::visible(0))],
        )
        .pop()
        .unwrap();
    started_rx.recv_timeout(Duration::from_secs(2)).unwrap();
    let second = scheduler
        .replace_scope(
            DemandScopeId(2),
            vec![demand(key("same"), JobPriority::INTERACTIVE)],
        )
        .pop()
        .unwrap();
    {
        let (lock, wake) = &*gate;
        *lock.lock().unwrap() = true;
        wake.notify_all();
    }
    assert!(matches!(
        block_on(first.receiver).unwrap(),
        JobCompletion::Finished(_)
    ));
    assert!(matches!(
        block_on(second.receiver).unwrap(),
        JobCompletion::Finished(_)
    ));
    assert_eq!(executions.load(Ordering::SeqCst), 1);
}

#[test]
fn clearing_a_scope_removes_queued_work() {
    let gate = Arc::new((Mutex::new(false), Condvar::new()));
    let executed = Arc::new(Mutex::new(Vec::new()));
    let (started_tx, started_rx) = mpsc::channel();
    let scheduler = RenderScheduler::new(1, {
        let gate = gate.clone();
        let executed = executed.clone();
        Arc::new(move |job, _, _| {
            executed.lock().unwrap().push(job.path.clone());
            if job.path == std::path::Path::new("blocker") {
                started_tx.send(()).unwrap();
                let (lock, wake) = &*gate;
                let mut released = lock.lock().unwrap();
                while !*released {
                    released = wake.wait(released).unwrap();
                }
            }
            error_result("done")
        })
    });
    scheduler.replace_scope(
        DemandScopeId(1),
        vec![demand(key("blocker"), JobPriority::INTERACTIVE)],
    );
    started_rx.recv_timeout(Duration::from_secs(2)).unwrap();
    let obsolete = scheduler
        .replace_scope(
            DemandScopeId(2),
            vec![demand(key("obsolete"), JobPriority::visible(0))],
        )
        .pop()
        .unwrap();
    scheduler.clear_scope(DemandScopeId(2));
    assert!(matches!(
        block_on(obsolete.receiver).unwrap(),
        JobCompletion::Cancelled
    ));
    {
        let (lock, wake) = &*gate;
        *lock.lock().unwrap() = true;
        wake.notify_all();
    }
    let deadline = Instant::now() + Duration::from_secs(2);
    while scheduler.stats().running > 0 && Instant::now() < deadline {
        thread::yield_now();
    }
    assert!(!executed.lock().unwrap().contains(&key("obsolete").path));
}

#[test]
fn promoted_thumbnail_keeps_its_cache_subscription_during_scope_handoff() {
    let gate = Arc::new((Mutex::new(false), Condvar::new()));
    let (started_tx, started_rx) = mpsc::channel();
    let scheduler = RenderScheduler::new(1, {
        let gate = gate.clone();
        Arc::new(move |job, _, _| {
            if job.path == std::path::Path::new("blocker") {
                started_tx.send(()).unwrap();
                let (lock, wake) = &*gate;
                let mut released = lock.lock().unwrap();
                while !*released {
                    released = wake.wait(released).unwrap();
                }
            }
            error_result("done")
        })
    });
    scheduler.replace_scope(
        DemandScopeId(1),
        vec![demand(key("blocker"), JobPriority::INTERACTIVE)],
    );
    started_rx.recv_timeout(Duration::from_secs(2)).unwrap();

    let subscription = scheduler
        .replace_scope(
            DemandScopeId(2),
            vec![demand(key("thumbnail"), JobPriority::near_viewport(0))],
        )
        .pop()
        .unwrap();
    scheduler.replace_scope(
        DemandScopeId(3),
        vec![JobDemand {
            key: key("thumbnail"),
            priority: JobPriority::visible(0),
            subscribe: false,
        }],
    );
    scheduler.clear_scope(DemandScopeId(2));

    {
        let (lock, wake) = &*gate;
        *lock.lock().unwrap() = true;
        wake.notify_all();
    }
    assert!(matches!(
        block_on(subscription.receiver).unwrap(),
        JobCompletion::Finished(_)
    ));
}

#[test]
fn snapshot_exposes_running_and_dispatch_ordered_jobs() {
    let gate = Arc::new((Mutex::new(false), Condvar::new()));
    let (started_tx, started_rx) = mpsc::channel();
    let scheduler = RenderScheduler::new(1, {
        let gate = gate.clone();
        Arc::new(move |job, _, _| {
            if job.path == std::path::Path::new("running") {
                started_tx.send(()).unwrap();
                let (lock, wake) = &*gate;
                let mut released = lock.lock().unwrap();
                while !*released {
                    released = wake.wait(released).unwrap();
                }
            }
            error_result("done")
        })
    });
    scheduler.replace_scope(
        DemandScopeId(1),
        vec![demand(key("running"), JobPriority::INTERACTIVE)],
    );
    started_rx.recv_timeout(Duration::from_secs(2)).unwrap();
    scheduler.replace_scope(
        DemandScopeId(2),
        vec![
            demand(key("near"), JobPriority::near_viewport(0)),
            demand(key("visible-far"), JobPriority::visible(1)),
            demand(key("visible-near"), JobPriority::visible(10)),
        ],
    );

    let snapshot = scheduler.snapshot();
    assert_eq!(snapshot.running, 1);
    assert_eq!(snapshot.queued, 3);
    assert_eq!(snapshot.jobs[0].status, JobSnapshotStatus::Running);
    assert_eq!(snapshot.jobs[1].key.path, PathBuf::from("visible-near"));
    assert_eq!(snapshot.jobs[2].key.path, PathBuf::from("visible-far"));
    assert_eq!(snapshot.jobs[3].key.path, PathBuf::from("near"));
    assert_eq!(
        snapshot
            .jobs
            .iter()
            .filter(|job| job.priority.class >= PriorityClass::Visible && !job.cancelled)
            .count(),
        3
    );

    let (lock, wake) = &*gate;
    *lock.lock().unwrap() = true;
    wake.notify_all();
}

#[test]
fn foreground_demand_preempts_and_requeues_lower_running_work() {
    let executions = Arc::new(Mutex::new(Vec::new()));
    let background_runs = Arc::new(AtomicUsize::new(0));
    let (first_background_tx, first_background_rx) = mpsc::channel();
    let scheduler = RenderScheduler::new(1, {
        let executions = executions.clone();
        let background_runs = background_runs.clone();
        Arc::new(move |job, _, cancellation| {
            executions.lock().unwrap().push(job.path.clone());
            if job.path == std::path::Path::new("background")
                && background_runs.fetch_add(1, Ordering::SeqCst) == 0
            {
                first_background_tx.send(()).unwrap();
                while !cancellation.load(Ordering::Acquire) {
                    thread::yield_now();
                }
            }
            error_result("done")
        })
    });
    let background = scheduler
        .replace_scope(
            DemandScopeId(1),
            vec![demand(key("background"), JobPriority::near_viewport(0))],
        )
        .pop()
        .unwrap();
    first_background_rx
        .recv_timeout(Duration::from_secs(2))
        .unwrap();
    let interactive = scheduler
        .replace_scope(
            DemandScopeId(2),
            vec![demand(key("interactive"), JobPriority::INTERACTIVE)],
        )
        .pop()
        .unwrap();

    assert!(matches!(
        block_on(interactive.receiver).unwrap(),
        JobCompletion::Finished(_)
    ));
    assert!(matches!(
        block_on(background.receiver).unwrap(),
        JobCompletion::Finished(_)
    ));
    assert_eq!(
        *executions.lock().unwrap(),
        vec![
            PathBuf::from("background"),
            PathBuf::from("interactive"),
            PathBuf::from("background")
        ]
    );
}

#[test]
fn configured_parallelism_is_a_hard_limit() {
    let gate = Arc::new((Mutex::new(false), Condvar::new()));
    let active = Arc::new(AtomicUsize::new(0));
    let peak = Arc::new(AtomicUsize::new(0));
    let scheduler = RenderScheduler::new(2, {
        let gate = gate.clone();
        let active = active.clone();
        let peak = peak.clone();
        Arc::new(move |_, _, _| {
            let now = active.fetch_add(1, Ordering::SeqCst) + 1;
            peak.fetch_max(now, Ordering::SeqCst);
            let (lock, wake) = &*gate;
            let mut released = lock.lock().unwrap();
            while !*released {
                released = wake.wait(released).unwrap();
            }
            active.fetch_sub(1, Ordering::SeqCst);
            error_result("done")
        })
    });
    scheduler.replace_scope(
        DemandScopeId(1),
        (0..6)
            .map(|index| demand(key(&format!("job-{index}")), JobPriority::visible(0)))
            .collect(),
    );
    let deadline = Instant::now() + Duration::from_secs(2);
    while active.load(Ordering::SeqCst) < 2 && Instant::now() < deadline {
        thread::yield_now();
    }
    assert_eq!(peak.load(Ordering::SeqCst), 2);
    {
        let (lock, wake) = &*gate;
        *lock.lock().unwrap() = true;
        wake.notify_all();
    }
}

#[test]
fn background_work_cannot_consume_the_reserved_interactive_slot() {
    let gate = Arc::new((Mutex::new(false), Condvar::new()));
    let (started_tx, started_rx) = mpsc::channel();
    let scheduler = RenderScheduler::new(2, {
        let gate = gate.clone();
        Arc::new(move |job, _, _| {
            started_tx.send(job.path.clone()).unwrap();
            if job.path != std::path::Path::new("interactive") {
                let (lock, wake) = &*gate;
                let mut released = lock.lock().unwrap();
                while !*released {
                    released = wake.wait(released).unwrap();
                }
            }
            error_result("done")
        })
    });
    scheduler.replace_scope(
        DemandScopeId(1),
        vec![
            demand(key("background-1"), JobPriority::speculative(0)),
            demand(key("background-2"), JobPriority::speculative(0)),
        ],
    );
    assert_eq!(
        started_rx.recv_timeout(Duration::from_secs(2)).unwrap(),
        PathBuf::from("background-1")
    );
    scheduler.replace_scope(
        DemandScopeId(2),
        vec![demand(key("interactive"), JobPriority::INTERACTIVE)],
    );
    assert_eq!(
        started_rx.recv_timeout(Duration::from_secs(2)).unwrap(),
        PathBuf::from("interactive")
    );
    let (lock, wake) = &*gate;
    *lock.lock().unwrap() = true;
    wake.notify_all();
}

#[test]
fn all_planned_priority_classes_have_strict_ordering() {
    assert!(PriorityClass::Interactive > PriorityClass::Visible);
    assert!(PriorityClass::Visible > PriorityClass::NearViewport);
    assert!(PriorityClass::NearViewport > PriorityClass::Speculative);
    assert!(PriorityClass::Speculative > PriorityClass::Maintenance);
}

#[test]
fn aging_eventually_promotes_old_speculative_work() {
    let gate = Arc::new((Mutex::new(false), Condvar::new()));
    let order = Arc::new(Mutex::new(Vec::new()));
    let (started_tx, started_rx) = mpsc::channel();
    let scheduler = RenderScheduler::new_with_aging(1, Duration::from_millis(5), {
        let gate = gate.clone();
        let order = order.clone();
        Arc::new(move |job, _, _| {
            order.lock().unwrap().push(job.path.clone());
            if job.path == std::path::Path::new("blocker") {
                started_tx.send(()).unwrap();
                let (lock, wake) = &*gate;
                let mut released = lock.lock().unwrap();
                while !*released {
                    released = wake.wait(released).unwrap();
                }
            }
            error_result("done")
        })
    });
    scheduler.replace_scope(
        DemandScopeId(1),
        vec![demand(key("blocker"), JobPriority::INTERACTIVE)],
    );
    started_rx.recv_timeout(Duration::from_secs(2)).unwrap();
    scheduler.replace_scope(
        DemandScopeId(2),
        vec![demand(key("old"), JobPriority::speculative(0))],
    );
    thread::sleep(Duration::from_millis(12));
    scheduler.replace_scope(
        DemandScopeId(3),
        vec![demand(key("new-background"), JobPriority::speculative(0))],
    );
    scheduler.replace_scope(
        DemandScopeId(4),
        vec![demand(key("visible"), JobPriority::visible(0))],
    );
    {
        let (lock, wake) = &*gate;
        *lock.lock().unwrap() = true;
        wake.notify_all();
    }
    let deadline = Instant::now() + Duration::from_secs(2);
    while scheduler.stats().completed < 4 && Instant::now() < deadline {
        thread::yield_now();
    }
    assert_eq!(
        *order.lock().unwrap(),
        vec![
            key("blocker").path,
            key("visible").path,
            key("old").path,
            key("new-background").path,
        ]
    );
}

#[test]
fn cancelled_running_generation_cannot_be_resurrected() {
    let gate = Arc::new((Mutex::new(false), Condvar::new()));
    let executions = Arc::new(AtomicUsize::new(0));
    let (started_tx, started_rx) = mpsc::channel();
    let scheduler = RenderScheduler::new(1, {
        let gate = gate.clone();
        let executions = executions.clone();
        Arc::new(move |_, _, cancellation| {
            executions.fetch_add(1, Ordering::SeqCst);
            started_tx.send(()).unwrap();
            let (lock, wake) = &*gate;
            let mut released = lock.lock().unwrap();
            while !*released {
                released = wake.wait(released).unwrap();
            }
            if cancellation.load(Ordering::Acquire) {
                error_result("cancelled")
            } else {
                error_result("done")
            }
        })
    });
    scheduler.replace_scope(
        DemandScopeId(1),
        vec![demand(key("same"), JobPriority::visible(0))],
    );
    started_rx.recv_timeout(Duration::from_secs(2)).unwrap();
    scheduler.clear_scope(DemandScopeId(1));
    let replacement = scheduler
        .replace_scope(
            DemandScopeId(2),
            vec![demand(key("same"), JobPriority::INTERACTIVE)],
        )
        .pop()
        .unwrap();
    {
        let (lock, wake) = &*gate;
        *lock.lock().unwrap() = true;
        wake.notify_all();
    }
    assert!(matches!(
        block_on(replacement.receiver).unwrap(),
        JobCompletion::Finished(_)
    ));
    let deadline = Instant::now() + Duration::from_secs(2);
    while scheduler.stats().running > 0 && Instant::now() < deadline {
        thread::yield_now();
    }
    assert_eq!(executions.load(Ordering::SeqCst), 2);
}

#[test]
fn speculative_demand_has_no_foreground_subscription() {
    let scheduler = RenderScheduler::new(1, Arc::new(|_, _, _| error_result("done")));
    let subscriptions = scheduler.replace_scope(
        DemandScopeId(1),
        vec![JobDemand {
            key: key("background"),
            priority: JobPriority::speculative(0),
            subscribe: false,
        }],
    );
    assert!(subscriptions.is_empty());
    assert!(!scheduler.has_foreground_work());
}
