use std::sync::{
    Arc, Condvar, Mutex,
    atomic::{AtomicBool, Ordering},
};
use std::time::Duration;

/// How long a background waiter waits before it is admitted on freed
/// capacity even with foreground waiters present. Foreground demand still
/// wins every fresh admission; the grace only prevents indefinite
/// starvation of long-running background work under sustained foreground
/// traffic.
const BACKGROUND_GRACE: Duration = Duration::from_secs(2);

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
        let wait_started = std::time::Instant::now();
        loop {
            if cancelled.load(Ordering::Acquire) {
                if foreground {
                    state.waiting_foreground -= 1;
                }
                wake.notify_all();
                return None;
            }
            let reserve = usize::from(self.capacity > 1);
            let grace_elapsed = !foreground && wait_started.elapsed() >= BACKGROUND_GRACE;
            let admissible = if foreground {
                state.available > 0
            } else {
                (state.waiting_foreground == 0 && state.available > reserve)
                    || (grace_elapsed && state.available > 0)
            };
            if admissible {
                if foreground {
                    state.waiting_foreground -= 1;
                }
                state.available -= 1;
                let waited = wait_started.elapsed();
                if !foreground && waited >= Duration::from_secs(5) {
                    crate::trace::slow_event(&format!(
                        "job limiter: background admitted after {} (available {} · foreground waiters {})",
                        crate::trace::secs(waited),
                        state.available,
                        state.waiting_foreground,
                    ));
                }
                return Some(JobPermit {
                    limiter: self.clone(),
                });
            }
            let poll = if !foreground && grace_elapsed {
                Duration::from_millis(1)
            } else {
                Duration::from_millis(20)
            };
            let (next, _) = wake
                .wait_timeout(state, poll)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            state = next;
            let waited = wait_started.elapsed();
            if !foreground
                && waited >= Duration::from_secs(10)
                && waited - heartbeat_swap(waited) < poll
            {
                crate::trace::slow_event(&format!(
                    "job limiter: background STILL waiting after {} (available {} · foreground waiters {})",
                    crate::trace::secs(waited),
                    state.available,
                    state.waiting_foreground,
                ));
            }
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
#[path = "tests.rs"]
mod tests;

/// True exactly when `waited` crossed into a new 10-second bucket since the
/// previous check, so a waiting loop can emit one heartbeat per bucket.
fn heartbeat_swap(waited: Duration) -> Duration {
    Duration::from_secs(waited.as_secs() / 10 * 10)
}
