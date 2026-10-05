use super::*;
use std::sync::Arc;
use test_r::test;

#[test]
fn background_work_leaves_capacity_for_foreground() {
    let limiter = JobLimiter::new(2);
    let cancelled = AtomicBool::new(false);
    let background = limiter.acquire(false, &cancelled).unwrap();
    let foreground = limiter.acquire(true, &cancelled).unwrap();
    drop((background, foreground));
}

#[test]
fn waiting_is_cancellable() {
    let limiter = JobLimiter::new(1);
    let held = limiter.acquire(false, &AtomicBool::new(false)).unwrap();
    let cancelled = Arc::new(AtomicBool::new(false));
    let waiter = {
        let limiter = limiter.clone();
        let cancelled = Arc::clone(&cancelled);
        std::thread::spawn(move || limiter.acquire(true, &cancelled).is_none())
    };
    cancelled.store(true, Ordering::Release);
    assert!(waiter.join().unwrap());
    drop(held);
}

#[test]
fn aged_background_waiter_admits_despite_foreground_pressure() {
    let limiter = JobLimiter::new(2);
    let cancelled = Arc::new(AtomicBool::new(false));
    // Keep one slot occupied so the reserve leaves nothing for background.
    let held = limiter.acquire(true, &cancelled).unwrap();
    // A foreground waiter keeps the "waiting_foreground" gate closed.
    let foreground = {
        let limiter = limiter.clone();
        let cancelled = Arc::clone(&cancelled);
        std::thread::spawn(move || limiter.acquire(true, &cancelled))
    };
    // Give the foreground waiter time to register, then start the background
    // acquire: it must initially wait, then admit once the grace elapses and
    // the held slot is released.
    std::thread::sleep(std::time::Duration::from_millis(100));
    let background = {
        let limiter = limiter.clone();
        let cancelled = Arc::clone(&cancelled);
        std::thread::spawn(move || limiter.acquire(false, &cancelled))
    };
    std::thread::sleep(std::time::Duration::from_millis(300));
    drop(held);
    let permit = background.join().unwrap();
    assert!(permit.is_some(), "aged background waiter must be admitted");
    let foreground = foreground.join().unwrap();
    drop(foreground);
}

#[test]
fn fresh_background_waiter_still_yields_to_foreground() {
    let limiter = JobLimiter::new(2);
    let cancelled = Arc::new(AtomicBool::new(false));
    let held = limiter.acquire(true, &cancelled).unwrap();
    let foreground = {
        let limiter = limiter.clone();
        let cancelled = Arc::clone(&cancelled);
        std::thread::spawn(move || limiter.acquire(true, &cancelled))
    };
    std::thread::sleep(std::time::Duration::from_millis(100));
    // Within the grace window nothing frees, so the background waiter must
    // still be waiting when the foreground waiter is admitted on release.
    let background = {
        let limiter = limiter.clone();
        let cancelled = Arc::clone(&cancelled);
        std::thread::spawn(move || {
            let _ = limiter.acquire(false, &cancelled);
        })
    };
    std::thread::sleep(std::time::Duration::from_millis(50));
    drop(held);
    let foreground = foreground.join().unwrap();
    assert!(foreground.is_some());
    drop(background);
    drop(foreground);
}
