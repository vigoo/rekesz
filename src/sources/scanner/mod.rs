//! Network scanner import source: mDNS discovery of driverless eSCL
//! scanners and the per-session scan output directory.
//!
//! A scanner source is path-backed — every scan lands as a JPEG file in
//! `scan-staging/<device>/<session>/` inside the cache — so the import
//! engine reads it like any folder. The device endpoint itself is resolved
//! from the discovery snapshot by the scan panel (Stage 3); listing,
//! duplicates, destinations, and the copy job never learn scanners exist.

pub mod discover;
// The client is a complete unit ahead of its in-app consumer (the scan
// panel lands in Stage 3); until then only the tests and the
// `scanner-probe` bin drive it, which the main build sees as dead code.
#[allow(dead_code)]
pub mod escl;

#[cfg(test)]
pub(crate) mod mock;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use crate::sources::{SourceBacking, SourceEntry, SourceId, SourceKind};

/// Token of this app run's scan directory: a restart starts an empty batch
/// and never resurrects a stale pile as selected photos. Initialized on the
/// first scanner refresh, which is process start for practical purposes.
static SESSION_TOKEN: LazyLock<String> = LazyLock::new(|| {
    format!(
        "{}-{}",
        chrono::Local::now().format("%Y%m%d-%H%M%S"),
        std::process::id()
    )
});

/// Whether [`ScannerProvider::collect`] starts the discovery daemon. Tests
/// turn this off so the poll-wiring test can drive a source refresh without
/// putting mDNS multicast on the network.
static DAEMON_ENABLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

#[cfg(test)]
pub(crate) fn set_daemon_enabled(enabled: bool) {
    DAEMON_ENABLED.store(enabled, std::sync::atomic::Ordering::Release);
}

/// File name for the next scan: `scan-<local timestamp with millis>.jpg`.
pub fn scan_file_name(now: chrono::DateTime<chrono::Local>) -> String {
    format!(
        "scan-{}{:03}.jpg",
        now.format("%Y%m%d-%H%M%S"),
        now.timestamp_subsec_millis()
    )
}

/// Scan output directory of one scanner for this app run.
pub fn session_scan_dir(cache_root: &Path, device_key: &str) -> PathBuf {
    cache_root
        .join("scan-staging")
        .join(sanitize_key(device_key))
        .join(SESSION_TOKEN.as_str())
}

/// Sessions of other runs older than this are pruned on selection. Younger
/// ones may belong to a concurrently running instance sharing the cache —
/// its un-imported scans exist nowhere else, so they stay.
const PRUNE_SESSIONS_AFTER: chrono::Duration = chrono::Duration::hours(1);

/// Create the session scan directory and prune this device's stale session
/// directories: they are unreachable once their run ends, and the flatbed
/// workflow accumulates full-resolution scans fast. A session token embeds
/// its start time (`YYYYMMDD-HHMMSS-pid`); siblings older than
/// [`PRUNE_SESSIONS_AFTER`] (and stray unparsable names) are deleted, while
/// young sessions survive for a concurrently running instance. Sibling
/// *device* directories are left alone — each device prunes its own when it
/// is selected.
pub fn prepare_session_dir(session_dir: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(session_dir)?;
    let current = session_dir.file_name().map(|name| name.to_owned());
    let Some(device_dir) = session_dir.parent() else {
        return Ok(());
    };
    for entry in std::fs::read_dir(device_dir)?.flatten() {
        let same_session = current
            .as_ref()
            .is_some_and(|current| entry.file_name() == *current);
        if same_session {
            continue;
        }
        let token = entry.file_name().to_string_lossy().into_owned();
        let stale =
            chrono::NaiveDateTime::parse_from_str(token.get(..15).unwrap_or(""), "%Y%m%d-%H%M%S")
                .map(|started| chrono::Local::now().naive_local() - started > PRUNE_SESSIONS_AFTER)
                .unwrap_or(true);
        if !stale {
            continue;
        }
        let is_dir = entry.file_type().map(|t| t.is_dir()).unwrap_or(false);
        let result = if is_dir {
            std::fs::remove_dir_all(entry.path())
        } else {
            std::fs::remove_file(entry.path())
        };
        if let Err(error) = result {
            eprintln!(
                "cannot prune stale scan session {}: {error}",
                entry.path().display()
            );
        }
    }
    Ok(())
}

pub(crate) fn sanitize_key(value: &str) -> String {
    let sanitized: String = value
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if sanitized.is_empty() || sanitized == "." || sanitized == ".." {
        "_".to_owned()
    } else {
        sanitized
    }
}

/// Split a device name ("EPSON L3160 Series") into vendor and model.
fn split_name(name: &str) -> (String, String) {
    match name.split_once(' ') {
        Some((vendor, model)) => (vendor.to_string(), model.to_string()),
        None => (name.to_string(), String::new()),
    }
}

/// Builds source entries from the discovery snapshot. Discovery runs on its
/// own daemon thread (started here on first use); a freshly started app sees
/// an empty list until the first sweep finishes (~2.5 s).
pub struct ScannerProvider;

impl ScannerProvider {
    pub fn collect(&self, cache_root: &Path) -> Vec<SourceEntry> {
        // Tests must not touch the network: the daemon's mDNS probes are
        // suppressible (import-view helpers disable them for their whole
        // process), and every other test uses [`Self::collect_from`].
        if DAEMON_ENABLED.load(std::sync::atomic::Ordering::Acquire) {
            discover::ensure_daemon();
        }
        let map = discover::SCANNERS.lock().expect("scanner registry lock");
        Self::collect_from(&map, cache_root)
    }

    /// Entries from a snapshot; split out from [`Self::collect`] so tests
    /// never start the discovery daemon (tests must not touch the network).
    /// Entries carry the session directory as their backing without creating
    /// it — the directory comes into being when the source is selected (see
    /// [`prepare_session_dir`]).
    pub(crate) fn collect_from(
        map: &HashMap<std::sync::Arc<str>, discover::ScannerDevice>,
        cache_root: &Path,
    ) -> Vec<SourceEntry> {
        let mut entries = Vec::new();
        for device in map.values() {
            // Devices whose every endpoint failed to resolve are not
            // addressable yet.
            if device.preferred_endpoint().is_none() {
                continue;
            }
            let dir = session_scan_dir(cache_root, &device.key);
            let (vendor, model) = split_name(&device.name);
            entries.push(SourceEntry {
                id: SourceId::new(format!("scanner:{}", device.key)),
                label: device.name.clone(),
                kind: SourceKind::Scanner { vendor, model },
                backing: SourceBacking::Path(dir),
                capacity: None,
            });
        }
        entries.sort_by_key(|entry| entry.label.to_lowercase());
        entries
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
