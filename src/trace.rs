//! Lightweight tracing for background scheduling, for post-hoc analysis of
//! "why does nothing move" reports.
//!
//! Two levels:
//! - **Slow events** are always logged (stderr + `trace.log` in the cache
//!   directory): admission waits longer than a few seconds, foreground
//!   gating, inventory passes, face jobs that take or fail abnormally.
//!   Healthy operation produces almost none of these, so the log stays
//!   near-empty until something is actually wrong.
//! - **Verbose events** (stage transitions, acquisitions) are logged only
//!   when `REKESZ_TRACE=1` is set; they are per-file chatty.
//!
//! Lines are `HH:MM:SS.mmm [thread] facility: message`.

use std::fs::OpenOptions;
use std::path::PathBuf;
use std::sync::OnceLock;
use std::time::Duration;

static OUTPUT_PATH: OnceLock<PathBuf> = OnceLock::new();
static VERBOSE: OnceLock<bool> = OnceLock::new();

const MAX_TRACE_FILE_BYTES: u64 = 32 * 1024 * 1024;

/// Point tracing at the cache directory and read the verbosity flag. Safe to
/// call once at startup; without it, tracing goes to stderr only.
pub fn init(cache_root: &std::path::Path) {
    let _ = OUTPUT_PATH.set(cache_root.join("trace.log"));
    let _ = VERBOSE.set(
        std::env::var("REKESZ_TRACE")
            .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
            .unwrap_or(false),
    );
    if let Some(path) = OUTPUT_PATH.get()
        && let Ok(metadata) = std::fs::metadata(path)
        && metadata.len() > MAX_TRACE_FILE_BYTES
    {
        // Bounded: a fresh file per ~32 MiB of anomalies.
        let _ = std::fs::write(path, b"");
    }
}

fn verbose_enabled() -> bool {
    VERBOSE.get().copied().unwrap_or(false)
}

/// Verbose-only event; empty unless `REKESZ_TRACE=1`.
pub fn event(message: &str) {
    if verbose_enabled() {
        emit(message);
    }
}

/// Always-logged event: reserved for slow or anomalous behavior.
pub fn slow_event(message: &str) {
    emit(message);
}

fn emit(message: &str) {
    let thread = std::thread::current()
        .name()
        .unwrap_or("<unnamed>")
        .to_owned();
    let line = format!(
        "{} [{thread}] {message}",
        chrono::Local::now().format("%H:%M:%S%.3f")
    );
    eprintln!("{line}");
    if let Some(path) = OUTPUT_PATH.get()
        && let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path)
    {
        use std::io::Write as _;
        let _ = writeln!(file, "{line}");
    }
}

/// Human duration for trace lines ("0.4s", "12.3s", "4m2.0s").
pub fn secs(duration: Duration) -> String {
    let seconds = duration.as_secs_f32();
    if seconds >= 60.0 {
        format!("{}m{:.1}s", seconds as u64 / 60, seconds % 60.0)
    } else {
        format!("{seconds:.1}s")
    }
}

#[cfg(test)]
#[path = "trace_tests.rs"]
mod tests;
