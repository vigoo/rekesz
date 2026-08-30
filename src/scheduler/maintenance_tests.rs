use super::*;
use std::sync::atomic::AtomicUsize;
use test_r::test;

#[test]
fn settled_wait_tracks_terminal_transition_without_polling() {
    let entered = Arc::new((Mutex::new(false), Condvar::new()));
    let release = Arc::new((Mutex::new(false), Condvar::new()));
    let scheduler = MaintenanceScheduler::new(1, {
        let entered = entered.clone();
        let release = release.clone();
        Arc::new(move |_: &u8, _| {
            let (entered_lock, entered_wake) = &*entered;
            *entered_lock.lock().unwrap() = true;
            entered_wake.notify_all();
            let (release_lock, release_wake) = &*release;
            let mut open = release_lock.lock().unwrap();
            while !*open {
                open = release_wake.wait(open).unwrap();
            }
            Ok(())
        })
    });
    assert!(scheduler.submit(1, 0));
    let (entered_lock, entered_wake) = &*entered;
    let mut is_entered = entered_lock.lock().unwrap();
    while !*is_entered {
        is_entered = entered_wake.wait(is_entered).unwrap();
    }
    drop(is_entered);
    assert!(!scheduler.wait_until_settled(Duration::ZERO));
    let (release_lock, release_wake) = &*release;
    *release_lock.lock().unwrap() = true;
    release_wake.notify_all();
    assert!(scheduler.wait_until_settled(Duration::from_secs(1)));
    assert_eq!(scheduler.stats().completed, 1);
}

#[test]
fn failed_maintenance_retains_the_actionable_error_until_success() {
    let attempts = Arc::new(AtomicUsize::new(0));
    let scheduler = MaintenanceScheduler::new(1, {
        let attempts = attempts.clone();
        Arc::new(move |_: &u8, _| {
            if attempts.fetch_add(1, Ordering::SeqCst) == 0 {
                anyhow::bail!("fixture failure detail")
            }
            Ok(())
        })
    });
    assert!(scheduler.submit(1, 0));
    assert!(scheduler.wait_until_settled(Duration::from_secs(1)));
    assert_eq!(
        scheduler.stats().last_error.as_deref(),
        Some("fixture failure detail")
    );
    assert!(scheduler.submit(1, 0));
    assert!(scheduler.wait_until_settled(Duration::from_secs(1)));
    assert_eq!(scheduler.stats().last_error, None);
}

#[test]
fn cooperative_yield_is_not_reported_as_success_or_failure() {
    let scheduler = MaintenanceScheduler::new(
        1,
        Arc::new(|_: &u8, _| Err(MaintenanceYield("fixture preemption").into())),
    );
    assert!(scheduler.submit(1, 0));
    assert!(scheduler.wait_until_settled(Duration::from_secs(1)));
    let stats = scheduler.stats();
    assert_eq!(stats.completed, 0);
    assert_eq!(stats.failed, 0);
    assert_eq!(stats.last_error, None);
}

#[test]
fn coalesces_duplicates_and_dispatches_higher_rank_first() {
    let gate = Arc::new((Mutex::new(false), Condvar::new()));
    let order = Arc::new(Mutex::new(Vec::new()));
    let scheduler = MaintenanceScheduler::new(1, {
        let gate = gate.clone();
        let order = order.clone();
        Arc::new(move |key: &u8, _| {
            if *key == 0 {
                let (lock, wake) = &*gate;
                let mut open = lock.lock().unwrap();
                while !*open {
                    open = wake.wait(open).unwrap();
                }
            }
            order.lock().unwrap().push(*key);
            Ok(())
        })
    });
    scheduler.submit(0, 0);
    while scheduler.stats().running == 0 {
        thread::yield_now();
    }
    assert!(scheduler.submit(1, 1));
    assert!(!scheduler.submit(1, 100));
    assert!(scheduler.submit(2, 50));
    let (lock, wake) = &*gate;
    *lock.lock().unwrap() = true;
    wake.notify_all();
    while scheduler.stats().completed < 3 {
        thread::yield_now();
    }
    assert_eq!(*order.lock().unwrap(), vec![0, 1, 2]);
}

#[test]
fn cancellation_and_global_concurrency_are_consistent() {
    let active = Arc::new(AtomicUsize::new(0));
    let peak = Arc::new(AtomicUsize::new(0));
    let scheduler = MaintenanceScheduler::new(2, {
        let active = active.clone();
        let peak = peak.clone();
        Arc::new(move |_: &u8, cancelled| {
            let now = active.fetch_add(1, Ordering::SeqCst) + 1;
            peak.fetch_max(now, Ordering::SeqCst);
            while !cancelled.load(Ordering::Acquire) {
                thread::yield_now();
            }
            active.fetch_sub(1, Ordering::SeqCst);
            Ok(())
        })
    });
    for key in 0..4 {
        scheduler.submit(key, 0);
    }
    while scheduler.stats().running < 1 {
        thread::yield_now();
    }
    for key in 0..4 {
        assert!(scheduler.cancel(&key));
    }
    while scheduler.stats().running > 0 {
        thread::yield_now();
    }
    assert_eq!(peak.load(Ordering::SeqCst), 1);
    assert_eq!(scheduler.stats().cancelled, 4);
    assert!(scheduler.snapshot().is_empty());
}

#[test]
fn foreground_admission_blocks_dispatch_until_released() {
    let admission = MaintenanceAdmission::new();
    admission.set_foreground_blocked(true);
    let runs = Arc::new(AtomicUsize::new(0));
    let scheduler = MaintenanceScheduler::new_with_admission(1, admission.clone(), {
        let runs = runs.clone();
        Arc::new(move |_: &u8, _| {
            runs.fetch_add(1, Ordering::SeqCst);
            Ok(())
        })
    });
    scheduler.submit(1, 0);
    thread::sleep(Duration::from_millis(30));
    assert_eq!(runs.load(Ordering::SeqCst), 0);
    admission.set_foreground_blocked(false);
    while scheduler.stats().completed == 0 {
        thread::yield_now();
    }
    assert_eq!(runs.load(Ordering::SeqCst), 1);
}

#[test]
fn scoped_foreground_activity_blocks_maintenance_until_drop() {
    let admission = MaintenanceAdmission::new();
    let activity = admission.begin_foreground_activity();
    let runs = Arc::new(AtomicUsize::new(0));
    let scheduler = MaintenanceScheduler::new_with_admission(1, admission, {
        let runs = runs.clone();
        Arc::new(move |_: &u8, _| {
            runs.fetch_add(1, Ordering::SeqCst);
            Ok(())
        })
    });
    scheduler.submit(1, 0);
    thread::sleep(Duration::from_millis(30));
    assert_eq!(runs.load(Ordering::SeqCst), 0);
    drop(activity);
    while scheduler.stats().completed == 0 {
        thread::yield_now();
    }
    assert_eq!(runs.load(Ordering::SeqCst), 1);
}

#[test]
fn worker_local_executor_state_is_created_once_and_reused() {
    let factories = Arc::new(AtomicUsize::new(0));
    let calls = Arc::new(Mutex::new(Vec::new()));
    let scheduler =
        MaintenanceScheduler::new_with_admission_factory(1, MaintenanceAdmission::new(), {
            let factories = factories.clone();
            let calls = calls.clone();
            Arc::new(move |_| {
                factories.fetch_add(1, Ordering::SeqCst);
                let mut local_sequence = 0;
                let calls = calls.clone();
                Box::new(move |_: &u8, _cancelled: &Arc<AtomicBool>| {
                    local_sequence += 1;
                    calls.lock().unwrap().push(local_sequence);
                    Ok(())
                })
            })
        });
    for key in 0..3 {
        scheduler.submit(key, 0);
    }
    while scheduler.stats().completed < 3 {
        thread::yield_now();
    }
    assert_eq!(factories.load(Ordering::SeqCst), 1);
    assert_eq!(*calls.lock().unwrap(), vec![1, 2, 3]);
}

#[test]
fn aging_prevents_low_rank_maintenance_starvation() {
    let admission = MaintenanceAdmission::new();
    admission.set_foreground_blocked(true);
    let order = Arc::new(Mutex::new(Vec::new()));
    let scheduler = MaintenanceScheduler::new_with_admission(1, admission.clone(), {
        let order = order.clone();
        Arc::new(move |key: &u8, _| {
            order.lock().unwrap().push(*key);
            Ok(())
        })
    });
    scheduler.submit(1, 0);
    thread::sleep(Duration::from_millis(30));
    scheduler.submit(2, 1);
    admission.set_foreground_blocked(false);
    while scheduler.stats().completed < 2 {
        thread::yield_now();
    }
    assert_eq!(*order.lock().unwrap(), vec![1, 2]);
}

#[test]
fn active_maintenance_checkpoint_yields_to_new_foreground_work() {
    let admission = MaintenanceAdmission::new();
    let reached_checkpoint = Arc::new((Mutex::new(false), Condvar::new()));
    let proceed = Arc::new((Mutex::new(false), Condvar::new()));
    let passed_checkpoint = Arc::new(AtomicBool::new(false));
    let scheduler = MaintenanceScheduler::new_with_admission(1, admission.clone(), {
        let admission = admission.clone();
        let reached_checkpoint = reached_checkpoint.clone();
        let proceed = proceed.clone();
        let passed_checkpoint = passed_checkpoint.clone();
        Arc::new(move |_: &u8, cancelled| {
            let (lock, wake) = &*reached_checkpoint;
            *lock.lock().unwrap() = true;
            wake.notify_all();
            let (lock, wake) = &*proceed;
            let mut ready = lock.lock().unwrap();
            while !*ready {
                ready = wake.wait(ready).unwrap();
            }
            assert!(admission.checkpoint(cancelled));
            passed_checkpoint.store(true, Ordering::SeqCst);
            Ok(())
        })
    });
    scheduler.submit(1, 0);
    let (lock, wake) = &*reached_checkpoint;
    let mut reached = lock.lock().unwrap();
    while !*reached {
        reached = wake.wait(reached).unwrap();
    }
    admission.set_foreground_blocked(true);
    let (lock, wake) = &*proceed;
    *lock.lock().unwrap() = true;
    wake.notify_all();
    thread::sleep(Duration::from_millis(30));
    assert!(!passed_checkpoint.load(Ordering::SeqCst));
    admission.set_foreground_blocked(false);
    while scheduler.stats().completed < 1 {
        thread::yield_now();
    }
}

#[test]
fn cancel_all_clears_more_than_ten_thousand_queued_jobs_without_snapshot_cleanup() {
    let admission = MaintenanceAdmission::new();
    admission.set_foreground_blocked(true);
    let scheduler =
        MaintenanceScheduler::new_with_admission(1, admission, Arc::new(|_: &usize, _| Ok(())));
    for key in 0..10_001 {
        assert!(scheduler.submit(key, 0));
    }
    scheduler.cancel_all();
    while scheduler.stats().running > 0 {
        thread::yield_now();
    }
    assert_eq!(scheduler.stats().queued, 0);
    assert_eq!(scheduler.stats().cancelled, 10_001);
}

#[test]
fn shared_admission_serializes_services_and_orders_waiters_by_rank() {
    let admission = MaintenanceAdmission::new();
    admission.set_foreground_blocked(true);
    let active = Arc::new(AtomicUsize::new(0));
    let peak = Arc::new(AtomicUsize::new(0));
    let order = Arc::new(Mutex::new(Vec::new()));

    let make_scheduler = |label: &'static str| {
        let active = active.clone();
        let peak = peak.clone();
        let order = order.clone();
        MaintenanceScheduler::new_with_admission(
            1,
            admission.clone(),
            Arc::new(move |_: &u8, _| {
                let now = active.fetch_add(1, Ordering::SeqCst) + 1;
                peak.fetch_max(now, Ordering::SeqCst);
                order.lock().unwrap().push(label);
                thread::sleep(Duration::from_millis(10));
                active.fetch_sub(1, Ordering::SeqCst);
                Ok(())
            }),
        )
    };
    let inference = make_scheduler("inference");
    let summary = make_scheduler("summary");
    let recognition = make_scheduler("recognition");
    inference.submit(1, 100);
    summary.submit(1, 10);
    recognition.submit(1, 50);
    assert!(admission.wait_until_waiters(3, Duration::from_secs(2)));
    assert_eq!(inference.stats().running, 0);
    assert_eq!(summary.stats().running, 0);
    assert_eq!(recognition.stats().running, 0);
    admission.set_foreground_blocked(false);
    assert!(inference.wait_until_settled(Duration::from_secs(2)));
    assert!(summary.wait_until_settled(Duration::from_secs(2)));
    assert!(recognition.wait_until_settled(Duration::from_secs(2)));
    assert_eq!(peak.load(Ordering::SeqCst), 1);
    assert_eq!(
        order.lock().unwrap().as_slice(),
        &["inference", "recognition", "summary"]
    );
}

#[test]
fn cancelling_ranked_waiter_releases_global_admission_without_running_it() {
    let admission = MaintenanceAdmission::new();
    admission.set_foreground_blocked(true);
    let ran = Arc::new(Mutex::new(Vec::new()));
    let high = MaintenanceScheduler::new_with_admission(1, admission.clone(), {
        let ran = ran.clone();
        Arc::new(move |key: &u8, _| {
            ran.lock().unwrap().push(*key);
            Ok(())
        })
    });
    let low = MaintenanceScheduler::new_with_admission(1, admission.clone(), {
        let ran = ran.clone();
        Arc::new(move |key: &u8, _| {
            ran.lock().unwrap().push(*key);
            Ok(())
        })
    });
    high.submit(1, 100);
    low.submit(2, 0);
    assert!(admission.wait_until_waiters(2, Duration::from_secs(2)));
    assert!(high.cancel(&1));
    admission.set_foreground_blocked(false);
    assert!(high.wait_until_settled(Duration::from_secs(2)));
    assert!(low.wait_until_settled(Duration::from_secs(2)));
    assert_eq!(*ran.lock().unwrap(), vec![2]);
}

#[test]
fn higher_ranked_service_preempts_at_cooperative_checkpoint() {
    let admission = MaintenanceAdmission::new();
    let low_entered = Arc::new((Mutex::new(false), Condvar::new()));
    let let_low_checkpoint = Arc::new((Mutex::new(false), Condvar::new()));
    let order = Arc::new(Mutex::new(Vec::new()));
    let low = MaintenanceScheduler::new_with_admission(1, admission.clone(), {
        let admission = admission.clone();
        let low_entered = low_entered.clone();
        let let_low_checkpoint = let_low_checkpoint.clone();
        let order = order.clone();
        Arc::new(move |_: &u8, cancelled| {
            order.lock().unwrap().push("low-start");
            *low_entered.0.lock().unwrap() = true;
            low_entered.1.notify_all();
            let mut proceed = let_low_checkpoint.0.lock().unwrap();
            while !*proceed {
                proceed = let_low_checkpoint.1.wait(proceed).unwrap();
            }
            drop(proceed);
            assert!(admission.checkpoint(cancelled));
            order.lock().unwrap().push("low-finish");
            Ok(())
        })
    });
    let high = MaintenanceScheduler::new_with_admission(1, admission.clone(), {
        let order = order.clone();
        Arc::new(move |_: &u8, _| {
            order.lock().unwrap().push("high");
            Ok(())
        })
    });
    low.submit(1, 0);
    let mut entered = low_entered.0.lock().unwrap();
    while !*entered {
        entered = low_entered.1.wait(entered).unwrap();
    }
    drop(entered);
    high.submit(2, 100);
    assert!(admission.wait_until_waiters(1, Duration::from_secs(2)));
    *let_low_checkpoint.0.lock().unwrap() = true;
    let_low_checkpoint.1.notify_all();
    assert!(low.wait_until_settled(Duration::from_secs(2)));
    assert!(high.wait_until_settled(Duration::from_secs(2)));
    assert_eq!(
        order.lock().unwrap().as_slice(),
        &["low-start", "high", "low-finish"]
    );
}

#[test]
fn multiworker_prefetch_cannot_hide_new_high_rank_job() {
    let admission = MaintenanceAdmission::new();
    admission.set_foreground_blocked(true);
    let order = Arc::new(Mutex::new(Vec::new()));
    let scheduler = MaintenanceScheduler::new_with_admission(3, admission.clone(), {
        let order = order.clone();
        Arc::new(move |key: &u8, _| {
            order.lock().unwrap().push(*key);
            Ok(())
        })
    });
    scheduler.submit(1, 0);
    scheduler.submit(2, 1);
    assert!(admission.wait_until_waiters(3, Duration::from_secs(2)));
    scheduler.submit(9, 100);
    admission.set_foreground_blocked(false);
    assert!(scheduler.wait_until_settled(Duration::from_secs(2)));
    assert_eq!(order.lock().unwrap().first(), Some(&9));
    assert_eq!(scheduler.stats().running, 0);
}

#[test]
fn promotion_updates_a_job_already_competing_for_global_admission() {
    let admission = MaintenanceAdmission::new();
    admission.set_foreground_blocked(true);
    let order = Arc::new(Mutex::new(Vec::new()));
    let promoted = MaintenanceScheduler::new_with_admission(1, admission.clone(), {
        let order = order.clone();
        Arc::new(move |_: &u8, _| {
            order.lock().unwrap().push("promoted");
            Ok(())
        })
    });
    let other = MaintenanceScheduler::new_with_admission(1, admission.clone(), {
        let order = order.clone();
        Arc::new(move |_: &u8, _| {
            order.lock().unwrap().push("other");
            Ok(())
        })
    });
    promoted.submit(1, 0);
    other.submit(1, 50);
    assert!(admission.wait_until_waiters(2, Duration::from_secs(2)));
    assert!(!promoted.submit(1, 100));
    admission.set_foreground_blocked(false);
    assert!(promoted.wait_until_settled(Duration::from_secs(2)));
    assert!(other.wait_until_settled(Duration::from_secs(2)));
    assert_eq!(order.lock().unwrap().as_slice(), &["promoted", "other"]);
}

#[test]
fn advertised_higher_rank_wins_checkpoint_race_before_registering_waiter() {
    let admission = MaintenanceAdmission::new();
    let low_started = Arc::new((Mutex::new(false), Condvar::new()));
    let allow_checkpoint = Arc::new((Mutex::new(false), Condvar::new()));
    let order = Arc::new(Mutex::new(Vec::new()));
    let low = MaintenanceScheduler::new_with_admission(1, admission.clone(), {
        let admission = admission.clone();
        let low_started = low_started.clone();
        let allow_checkpoint = allow_checkpoint.clone();
        let order = order.clone();
        Arc::new(move |_: &u8, cancelled| {
            order.lock().unwrap().push("low-start");
            *low_started.0.lock().unwrap() = true;
            low_started.1.notify_all();
            let mut allowed = allow_checkpoint.0.lock().unwrap();
            while !*allowed {
                allowed = allow_checkpoint.1.wait(allowed).unwrap();
            }
            drop(allowed);
            assert!(admission.checkpoint(cancelled));
            order.lock().unwrap().push("low-finish");
            Ok(())
        })
    });
    low.submit(1, 0);
    let mut started = low_started.0.lock().unwrap();
    while !*started {
        started = low_started.1.wait(started).unwrap();
    }
    drop(started);

    let high_scheduler_id = admission.register_scheduler();
    let advertised_at = Instant::now();
    admission.advertise(high_scheduler_id, Some((100, advertised_at)));
    *allow_checkpoint.0.lock().unwrap() = true;
    allow_checkpoint.1.notify_all();
    assert!(admission.wait_until_waiters(1, Duration::from_secs(2)));
    assert_eq!(order.lock().unwrap().as_slice(), &["low-start"]);

    let high_admission = admission.clone();
    let high_order = order.clone();
    let high = thread::spawn(move || {
        let cancelled = AtomicBool::new(false);
        let permit = high_admission
            .acquire(high_scheduler_id, 100, advertised_at, &cancelled, &|| false)
            .expect("advertised high-priority job must acquire");
        high_order.lock().unwrap().push("high");
        high_admission.advertise(high_scheduler_id, None);
        drop(permit);
    });
    high.join().unwrap();
    assert!(low.wait_until_settled(Duration::from_secs(2)));
    assert_eq!(
        order.lock().unwrap().as_slice(),
        &["low-start", "high", "low-finish"]
    );
}

#[test]
fn one_worker_checkpoint_does_not_yield_to_its_own_queued_job() {
    let admission = MaintenanceAdmission::new();
    let low_started = Arc::new((Mutex::new(false), Condvar::new()));
    let allow_checkpoint = Arc::new((Mutex::new(false), Condvar::new()));
    let order = Arc::new(Mutex::new(Vec::new()));
    let scheduler = MaintenanceScheduler::new_with_admission(1, admission.clone(), {
        let admission = admission.clone();
        let low_started = low_started.clone();
        let allow_checkpoint = allow_checkpoint.clone();
        let order = order.clone();
        Arc::new(move |key: &u8, cancelled| {
            if *key == 1 {
                order.lock().unwrap().push("low-start");
                *low_started.0.lock().unwrap() = true;
                low_started.1.notify_all();
                let mut allowed = allow_checkpoint.0.lock().unwrap();
                while !*allowed {
                    allowed = allow_checkpoint.1.wait(allowed).unwrap();
                }
                drop(allowed);
                assert!(admission.checkpoint(cancelled));
                order.lock().unwrap().push("low-finish");
            } else {
                order.lock().unwrap().push("high");
            }
            Ok(())
        })
    });
    scheduler.submit(1, 0);
    let mut started = low_started.0.lock().unwrap();
    while !*started {
        started = low_started.1.wait(started).unwrap();
    }
    drop(started);
    scheduler.submit(2, 100);
    *allow_checkpoint.0.lock().unwrap() = true;
    allow_checkpoint.1.notify_all();
    assert!(scheduler.wait_until_settled(Duration::from_secs(2)));
    assert_eq!(
        order.lock().unwrap().as_slice(),
        &["low-start", "low-finish", "high"],
        "a sole worker must finish its active job before it can run its own queued priority job"
    );
}
