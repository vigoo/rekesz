use super::*;
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
        let cancelled = cancelled.clone();
        std::thread::spawn(move || limiter.acquire(true, &cancelled).is_none())
    };
    cancelled.store(true, Ordering::Release);
    assert!(waiter.join().unwrap());
    drop(held);
}
