//! Staging of device files into the cache directory. Device objects (MTP
//! objects, PTP camera files) have no filesystem path, so they are
//! downloaded once into the cache to give the render pipeline (thumbnails)
//! and the EXIF capture-time reader a real file to work with.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use crate::sources::{SourceAccess, SourceBacking};

use super::ScannedFile;

fn sanitize_component(value: &str) -> String {
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

/// Where a device file is staged inside the cache directory, if its backing
/// is a device object.
pub fn staged_path(cache_root: &Path, file: &ScannedFile) -> Option<PathBuf> {
    match &file.backing {
        SourceBacking::Mtp(reference) => Some(
            cache_root
                .join("import-staging")
                .join(sanitize_component(&reference.device_id))
                .join(sanitize_component(&reference.storage_id))
                .join(format!(
                    "{}-{}",
                    sanitize_component(&reference.object),
                    sanitize_component(&file.name)
                )),
        ),
        SourceBacking::Ptp(reference) => Some(
            cache_root
                .join("import-staging")
                .join(sanitize_component(&reference.device_id))
                .join(format!(
                    "{}-{}",
                    sanitize_component(reference.folder.trim_start_matches('/')),
                    sanitize_component(&file.name)
                )),
        ),
        SourceBacking::Path(_) => None,
    }
}

/// Download a device file into the cache directory, unless a complete copy is
/// already there. Returns the staged path, or `None` for path-backed files.
pub fn stage_file(
    access: &dyn SourceAccess,
    file: &ScannedFile,
    cache_root: &Path,
    cancel: &AtomicBool,
) -> std::io::Result<Option<PathBuf>> {
    let Some(destination) = staged_path(cache_root, file) else {
        return Ok(None);
    };
    if let Ok(metadata) = std::fs::metadata(&destination) {
        // Object ids can be reused after a device replug; the recorded
        // modification time (kept from the device) guards against pairing a
        // staged file with a different photo of the same size.
        let time_matches = match (metadata.modified().ok(), file.modified) {
            (Some(staged), Some(original)) => staged == original,
            _ => true,
        };
        if metadata.len() == file.size && time_matches {
            return Ok(Some(destination));
        }
        // A stale or partial artifact from an earlier run.
        let _ = std::fs::remove_file(&destination);
    }
    if let Some(parent) = destination.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let partial = destination.with_extension("rekesz-part");
    // A leftover part file from an earlier run must go: device transfers
    // with a native to-path download refuse existing destinations.
    let _ = std::fs::remove_file(&partial);
    let written = if let Some(result) =
        access.read_file_to_path(&file.backing, &partial, cancel, &mut |_| {})
    {
        result?
    } else {
        let mut out = std::fs::File::create(&partial)?;
        let written = access.read_file(&file.backing, &mut out, cancel, &mut |_| {})?;
        drop(out);
        written
    };
    if cancel.load(Ordering::Acquire) {
        let _ = std::fs::remove_file(&partial);
        return Err(std::io::Error::new(
            std::io::ErrorKind::Interrupted,
            "staging cancelled",
        ));
    }
    if written != file.size {
        let _ = std::fs::remove_file(&partial);
        return Err(std::io::Error::other(format!(
            "staged {} bytes but the device reported {}",
            written, file.size
        )));
    }
    if let Some(modified) = file.modified {
        // Kept so the reuse check above can detect object-id reuse, and so
        // tools browsing the cache see the capture order. Set on the
        // part file: the destination only comes into being with the rename
        // below, and `rename` preserves the timestamp.
        let _ = std::fs::OpenOptions::new()
            .write(true)
            .open(&partial)
            .and_then(|handle| handle.set_modified(modified));
    }
    std::fs::rename(&partial, &destination)?;
    Ok(Some(destination))
}

#[cfg(test)]
#[path = "staging_tests.rs"]
mod tests;
