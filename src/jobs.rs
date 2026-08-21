use std::sync::{
    Arc, Condvar, Mutex,
    atomic::{AtomicBool, Ordering},
};
use std::time::Duration;

#[derive(Clone)]
pub struct JobLimiter {
    shared: Arc<(Mutex<LimiterState>, Condvar)>,
    capacity: usize,
}

struct LimiterState {
    available: usize,
    waiting_foreground: usize,
}

pub struct JobPermit {
    limiter: JobLimiter,
}

impl JobLimiter {
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        Self {
            shared: Arc::new((
                Mutex::new(LimiterState {
                    available: capacity,
                    waiting_foreground: 0,
                }),
                Condvar::new(),
            )),
            capacity,
        }
    }

    pub fn acquire(&self, foreground: bool, cancelled: &AtomicBool) -> Option<JobPermit> {
        let (lock, wake) = &*self.shared;
        let mut state = lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        if foreground {
            state.waiting_foreground += 1;
        }
        loop {
            if cancelled.load(Ordering::Acquire) {
                if foreground {
                    state.waiting_foreground -= 1;
                }
                wake.notify_all();
                return None;
            }
            let reserve = usize::from(self.capacity > 1);
            let admissible = if foreground {
                state.available > 0
            } else {
                state.waiting_foreground == 0 && state.available > reserve
            };
            if admissible {
                if foreground {
                    state.waiting_foreground -= 1;
                }
                state.available -= 1;
                return Some(JobPermit {
                    limiter: self.clone(),
                });
            }
            let (next, _) = wake
                .wait_timeout(state, Duration::from_millis(20))
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            state = next;
        }
    }
}

impl Drop for JobPermit {
    fn drop(&mut self) {
        let (lock, wake) = &*self.limiter.shared;
        let mut state = lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        state.available = (state.available + 1).min(self.limiter.capacity);
        wake.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
