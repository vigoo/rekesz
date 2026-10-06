//! Import engine: scanning a source for importable photos, detecting
//! duplicates against the library, planning destinations, and copying files
//! into the library root.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::SystemTime;

use crate::discovery::RawJpegPreference;
use crate::sources::{SourceAccess, SourceBacking};

pub mod destination;
pub mod duplicates;
pub mod staging;

/// Safety budget for the source walk; real camera trees are much shallower.
/// Large folder sources can legitimately exceed it, which the outcome
/// reports as `truncated` rather than hiding.
const MAX_SCAN_DIRECTORIES: usize = 24 * 1024;

/// One importable photo: the primary file shown in the grid plus an optional
/// RAW/JPEG companion discovered beside it. Selecting the photo imports both.
#[derive(Clone, Debug)]
pub struct ScannedPhoto {
    /// Slash-separated directory path from the source root ("" at the root).
    /// A logical identity, never to be joined as a filesystem path; the real
    /// locations live in the file backings.
    pub directory: String,
    pub primary: ScannedFile,
    pub companion: Option<ScannedFile>,
}

impl ScannedPhoto {
    /// Stable identity within one scan, for grid selection state.
    pub fn key(&self) -> String {
        if self.directory.is_empty() {
            self.primary.name.clone()
        } else {
            format!("{}/{}", self.directory, self.primary.name)
        }
    }
}

#[derive(Clone, Debug)]
pub struct ScannedFile {
    pub backing: SourceBacking,
    pub name: String,
    pub size: u64,
    pub modified: Option<SystemTime>,
}

#[derive(Clone, Default)]
pub struct ScanProgress {
    pub directories: usize,
    pub photos: usize,
    /// Slash-separated path of the directory just scanned ("" at the root).
    pub current: String,
}

#[derive(Debug)]
pub struct ScanOutcome {
    pub photos: Vec<ScannedPhoto>,
    /// Directories that could not be listed (permissions, device errors).
    pub skipped_directories: usize,
    /// True when the scan hit the directory budget and stopped early; the
    /// photo list is then incomplete.
    pub truncated: bool,
}

#[derive(Debug)]
pub enum ScanError {
    Cancelled,
    Failed(Arc<str>),
}

impl From<std::io::Error> for ScanError {
    fn from(error: std::io::Error) -> Self {
        Self::Failed(Arc::from(format!("{error}")))
    }
}

/// Walk a source root and collect importable photos, pairing same-stem
/// CR3/JPEG files so both halves travel together.
pub fn scan_source(
    access: &dyn SourceAccess,
    root: &SourceBacking,
    preference: RawJpegPreference,
    cancel: &AtomicBool,
    progress: &mut dyn FnMut(ScanProgress),
) -> Result<ScanOutcome, ScanError> {
    let mut photos = Vec::new();
    let mut skipped_directories = 0_usize;
    let mut truncated = false;
    let mut visited = HashSet::new();
    let mut pending = vec![(root.clone(), String::new())];
    while let Some((directory, relative)) = pending.pop() {
        if cancel.load(Ordering::Acquire) {
            return Err(ScanError::Cancelled);
        }
        if !visited.insert(directory.clone()) {
            continue;
        }
        if visited.len() > MAX_SCAN_DIRECTORIES {
            truncated = true;
            continue;
        }
        let entries = match access.list_dir(&directory) {
            Ok(entries) => entries,
            Err(_) => {
                skipped_directories += 1;
                continue;
            }
        };
        let mut files = Vec::new();
        for entry in entries {
            if entry.is_dir {
                let child_relative = if relative.is_empty() {
                    entry.name.clone()
                } else {
                    format!("{relative}/{}", entry.name)
                };
                pending.push((entry.backing.clone(), child_relative));
            } else if is_supported_name(&entry.name) {
                files.push(ScannedFile {
                    backing: entry.backing.clone(),
                    name: entry.name.clone(),
                    size: entry.size,
                    modified: entry.modified,
                });
            }
        }
        photos.extend(pair_photos(&relative, files, preference));
        progress(ScanProgress {
            directories: visited.len(),
            photos: photos.len(),
            current: relative.clone(),
        });
    }
    photos.sort_by(|a, b| {
        a.directory.cmp(&b.directory).then_with(|| {
            a.primary
                .name
                .to_lowercase()
                .cmp(&b.primary.name.to_lowercase())
        })
    });
    Ok(ScanOutcome {
        photos,
        skipped_directories,
        truncated,
    })
}

fn is_supported_name(name: &str) -> bool {
    crate::discovery::is_supported_image(Path::new(name))
}

/// Group same-directory files by stem and attach CR3/JPEG companions. Unlike
/// the browser's dedup (which keeps only the preferred half of a pair), both
/// halves travel together here; any additional same-stem supported files
/// remain standalone photos.
pub(crate) fn pair_photos(
    relative: &str,
    mut files: Vec<ScannedFile>,
    preference: RawJpegPreference,
) -> Vec<ScannedPhoto> {
    let mut by_stem: BTreeMap<String, Vec<ScannedFile>> = BTreeMap::new();
    files.sort_by_key(|file| file.name.to_lowercase());
    for file in files {
        let stem = Path::new(&file.name)
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_lowercase();
        by_stem.entry(stem).or_default().push(file);
    }
    by_stem
        .into_values()
        .flat_map(|group| match cr3_jpeg_pair(&group) {
            Some((raw_index, jpeg_index)) => {
                let mut group = group;
                // Remove the higher index first so the other stays valid.
                let (low, high) = if raw_index < jpeg_index {
                    (raw_index, jpeg_index)
                } else {
                    (jpeg_index, raw_index)
                };
                let second = group.remove(high);
                let first = group.remove(low);
                let (raw, jpeg) = if raw_index < jpeg_index {
                    (first, second)
                } else {
                    (second, first)
                };
                let (primary, companion) = match preference {
                    RawJpegPreference::Raw => (raw, jpeg),
                    RawJpegPreference::Jpeg => (jpeg, raw),
                };
                let mut photos = vec![ScannedPhoto {
                    directory: relative.to_owned(),
                    primary,
                    companion: Some(companion),
                }];
                photos.extend(group.into_iter().map(|primary| ScannedPhoto {
                    directory: relative.to_owned(),
                    primary,
                    companion: None,
                }));
                photos
            }
            None => group
                .into_iter()
                .map(|primary| ScannedPhoto {
                    directory: relative.to_owned(),
                    primary,
                    companion: None,
                })
                .collect(),
        })
        .collect()
}

/// Indices of the first CR3 and the first JPEG/JPG in a stem group.
fn cr3_jpeg_pair(group: &[ScannedFile]) -> Option<(usize, usize)> {
    let has_extension = |file: &ScannedFile, expected: &[&str]| {
        Path::new(&file.name).extension().is_some_and(|actual| {
            expected
                .iter()
                .any(|expected| actual.eq_ignore_ascii_case(expected))
        })
    };
    let raw = group
        .iter()
        .position(|file| has_extension(file, &["cr3"]))?;
    let jpeg = group
        .iter()
        .position(|file| has_extension(file, &["jpg", "jpeg"]))?;
    Some((raw, jpeg))
}

/// Where imported photos land inside the library root; see
/// [`destination::DestinationPreset`].
pub use destination::DestinationPreset;

/// Everything the copy job needs. The photos are the already-selected set;
/// duplicate filtering happens before the plan is built. For device sources
/// the view replaces `Mtp` backings with staged file paths, so capture-date
/// lookup, copying, and mtime preservation all work off disk.
pub struct ImportPlan {
    pub library_root: PathBuf,
    /// Absolute folder inside the library root the layout applies within.
    pub destination_base: PathBuf,
    pub photos: Vec<ScannedPhoto>,
    pub preset: DestinationPreset,
}

#[derive(Clone, Default, Debug)]
pub struct ImportProgress {
    pub total_files: usize,
    pub total_bytes: u64,
    pub copied_files: usize,
    /// Bytes written so far, including partial bytes of failed files.
    pub copied_bytes: u64,
    pub failed_files: usize,
    /// Name of the file currently being copied.
    pub current: String,
}

#[derive(Clone, Debug)]
pub struct ImportFailure {
    pub name: String,
    pub message: Arc<str>,
}

#[derive(Clone, Debug)]
pub struct ImportReport {
    pub copied_files: usize,
    /// Bytes of successfully copied files only.
    pub copied_bytes: u64,
    /// Files that got a " (2)"-style suffix because the destination existed.
    pub renamed_files: usize,
    /// Destination directories that received at least one file; the view
    /// refreshes these subtrees after the import.
    pub directories: Vec<PathBuf>,
    pub failed: Vec<ImportFailure>,
}

#[derive(Debug)]
pub enum ImportStatus {
    Completed(ImportReport),
    Cancelled(ImportReport),
}

/// Copy the planned photos into the library root, organized by the preset's
/// date-based layout derived from each photo's capture time. Cancellation
/// stops at the next chunk and removes the partial file. Any error observed
/// while the cancel flag is set is treated as cancellation, since the run is
/// ending regardless.
pub fn run_import(
    plan: &ImportPlan,
    access: &dyn SourceAccess,
    progress: Arc<std::sync::Mutex<ImportProgress>>,
    cancel: &AtomicBool,
) -> ImportStatus {
    let total_files = plan
        .photos
        .iter()
        .map(|photo| 1 + usize::from(photo.companion.is_some()))
        .sum();
    let total_bytes = plan
        .photos
        .iter()
        .map(|photo| photo.primary.size + photo.companion.as_ref().map_or(0, |f| f.size))
        .sum();
    {
        let mut progress = progress.lock().expect("import progress lock");
        progress.total_files = total_files;
        progress.total_bytes = total_bytes;
    }

    let mut report = ImportReport {
        copied_files: 0,
        copied_bytes: 0,
        renamed_files: 0,
        directories: Vec::new(),
        failed: Vec::new(),
    };
    // The layout applies inside the library root; a plan with a base
    // outside it falls back to the root rather than escaping the library.
    let destination_base = if plan.destination_base.starts_with(&plan.library_root) {
        plan.destination_base.clone()
    } else {
        crate::trace::slow_event(&format!(
            "import: requested destination {} is outside the library root; using the library root",
            plan.destination_base.display()
        ));
        plan.library_root.clone()
    };
    crate::trace::slow_event(&format!(
        "import: plan: {} photos / {} files · {} · layout {:?} · destination {}",
        plan.photos.len(),
        total_files,
        bytes_label(total_bytes),
        plan.preset,
        destination_base.display()
    ));
    let mut directories = BTreeSet::new();
    let mut shared_copied_bytes = 0_u64;
    let mut successful_bytes = 0_u64;
    let mut counter = 1_u32;
    for photo in &plan.photos {
        if cancel.load(Ordering::Acquire) {
            break;
        }
        let date = destination::capture_date(&photo.primary);
        let directory = destination_base.join(plan.preset.directory_for(date));
        if let Err(error) = std::fs::create_dir_all(&directory) {
            record_failure(
                &mut report,
                &photo.primary,
                &progress,
                &format!("cannot create {}: {error}", directory.display()),
            );
            continue;
        }
        // The counter advances once per photo so RAW+JPEG pairs keep a
        // shared stem and still pair up in the library.
        let forced_stem = if plan.preset.uses_counter() {
            let files = std::iter::once(&photo.primary).chain(photo.companion.as_ref());
            Some(loop {
                let stem = format!("{:04}", counter);
                counter = counter.wrapping_add(1);
                let taken = files.clone().any(|file| {
                    let extension = Path::new(&file.name)
                        .extension()
                        .map(|extension| format!(".{}", extension.to_string_lossy()))
                        .unwrap_or_default();
                    directory.join(format!("{stem}{extension}")).exists()
                });
                if !taken {
                    break stem;
                }
            })
        } else {
            None
        };
        let files = std::iter::once(&photo.primary).chain(photo.companion.as_ref());
        for file in files {
            if cancel.load(Ordering::Acquire) {
                break;
            }
            copy_one(
                file,
                &directory,
                forced_stem.as_deref(),
                access,
                &progress,
                cancel,
                &mut shared_copied_bytes,
                &mut successful_bytes,
                &mut report,
                &mut directories,
            );
        }
    }
    report.directories = directories.into_iter().collect();
    {
        let mut progress = progress.lock().expect("import progress lock");
        progress.current = String::new();
        progress.copied_bytes = shared_copied_bytes;
    }
    if cancel.load(Ordering::Acquire) {
        crate::trace::slow_event(&format!(
            "import: cancelled: {} copied, {} renamed, {} failed · {} · into {}",
            report.copied_files,
            report.renamed_files,
            report.failed.len(),
            bytes_label(successful_bytes),
            destination_base.display()
        ));
        ImportStatus::Cancelled(report)
    } else {
        crate::trace::slow_event(&format!(
            "import: completed: {} copied, {} renamed, {} failed · {} · into {}",
            report.copied_files,
            report.renamed_files,
            report.failed.len(),
            bytes_label(successful_bytes),
            destination_base.display()
        ));
        ImportStatus::Completed(report)
    }
}

#[allow(clippy::too_many_arguments)]
fn copy_one(
    file: &ScannedFile,
    directory: &Path,
    forced_stem: Option<&str>,
    access: &dyn SourceAccess,
    progress: &Arc<std::sync::Mutex<ImportProgress>>,
    cancel: &AtomicBool,
    shared_copied_bytes: &mut u64,
    successful_bytes: &mut u64,
    report: &mut ImportReport,
    directories: &mut BTreeSet<PathBuf>,
) {
    {
        let mut shared = progress.lock().expect("import progress lock");
        shared.current = file.name.clone();
    }
    let requested_name = match forced_stem {
        Some(stem) => {
            let extension = Path::new(&file.name)
                .extension()
                .map(|extension| format!(".{}", extension.to_string_lossy()))
                .unwrap_or_default();
            format!("{stem}{extension}")
        }
        None => file.name.clone(),
    };
    let (mut output, destination, renamed) = match open_destination(directory, &requested_name) {
        Ok(opened) => opened,
        Err((path, error)) => {
            record_failure(
                report,
                file,
                progress,
                &format!("cannot create {}: {error}", path.display()),
            );
            return;
        }
    };
    let base_bytes = *shared_copied_bytes;
    let mut copied_in_file = 0_u64;
    let progress_target = Arc::clone(progress);
    let result = access.read_file(&file.backing, &mut output, cancel, &mut |copied| {
        progress_target
            .lock()
            .expect("import progress lock")
            .copied_bytes = base_bytes + copied;
        copied_in_file = copied;
    });
    if let Err(error) = result {
        let cancelled = cancel.load(Ordering::Acquire);
        // Never leave partial files behind.
        drop(output);
        let _ = std::fs::remove_file(&destination);
        *shared_copied_bytes = base_bytes + copied_in_file;
        if cancelled {
            return;
        }
        record_failure(
            report,
            file,
            progress,
            &format!("writing {}: {error}", destination.display()),
        );
        return;
    }
    if copied_in_file != file.size {
        // libmtp-rs returns Ok for interrupted device transfers; a short read
        // must not enter the library as a complete file.
        drop(output);
        let _ = std::fs::remove_file(&destination);
        if cancel.load(Ordering::Acquire) {
            return;
        }
        record_failure(
            report,
            file,
            progress,
            &format!(
                "source delivered {} of {} bytes to {}",
                copied_in_file,
                file.size,
                destination.display()
            ),
        );
        return;
    }
    if let Some(modified) = file.modified {
        // Best effort: keeping the original mtime preserves capture order for
        // tools other than this app.
        let _ = output.set_modified(modified);
    }
    drop(output);
    *shared_copied_bytes = base_bytes + copied_in_file;
    *successful_bytes += copied_in_file;
    report.copied_files += 1;
    if renamed {
        report.renamed_files += 1;
    }
    report.copied_bytes = *successful_bytes;
    directories.insert(directory.to_path_buf());
    crate::trace::event(&format!(
        "import: copied {} -> {}{} ({})",
        backing_label(&file.backing),
        destination.display(),
        if renamed {
            format!(" (renamed from \"{requested_name}\")")
        } else {
            String::new()
        },
        bytes_label(copied_in_file)
    ));
    let mut shared = progress.lock().expect("import progress lock");
    shared.copied_files += 1;
    shared.copied_bytes = *shared_copied_bytes;
}

fn record_failure(
    report: &mut ImportReport,
    file: &ScannedFile,
    progress: &Arc<std::sync::Mutex<ImportProgress>>,
    message: &str,
) {
    crate::trace::slow_event(&format!(
        "import: failed {}: {message}",
        backing_label(&file.backing)
    ));
    report.failed.push(ImportFailure {
        name: file.name.clone(),
        message: Arc::from(message),
    });
    let mut shared = progress.lock().expect("import progress lock");
    shared.failed_files += 1;
    shared.current = file.name.clone();
}

/// Open the destination for a new file, never overwriting: a suffix
/// " (2)", " (3)", … is inserted before the extension when the name is
/// taken. `create_new` keeps the check atomic (a dangling symlink at the
/// destination name fails the open instead of being written through).
/// Returns the opened file, its path, and whether a suffix was needed.
fn open_destination(
    directory: &Path,
    name: &str,
) -> Result<(std::fs::File, PathBuf, bool), (PathBuf, std::io::Error)> {
    let open = |candidate: PathBuf| {
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&candidate)
            .map(|file| (file, candidate.clone(), false))
            .map_err(|error| (candidate, error))
    };
    let name = sanitize_name(name);
    match open(directory.join(&name)) {
        Ok(opened) => Ok(opened),
        Err((_, first_error)) if first_error.kind() == ErrorKind::AlreadyExists => {
            let path = Path::new(&name);
            let stem = path
                .file_stem()
                .map(|stem| stem.to_string_lossy().into_owned())
                .unwrap_or_else(|| name.clone());
            let extension = path
                .extension()
                .map(|extension| format!(".{}", extension.to_string_lossy()))
                .unwrap_or_default();
            for index in 2.. {
                match open(directory.join(format!("{stem} ({index}){extension}"))) {
                    Ok((file, path, _)) => return Ok((file, path, true)),
                    Err((_, error)) if error.kind() == ErrorKind::AlreadyExists => continue,
                    Err((path, error)) => return Err((path, error)),
                }
            }
            unreachable!("unique destination loop always returns")
        }
        Err((path, error)) => Err((path, error)),
    }
}

/// Where a file's bytes come from, for trace lines: the filesystem path for
/// path-backed files (including staged device copies), the device object
/// address otherwise.
fn backing_label(backing: &SourceBacking) -> String {
    match backing {
        SourceBacking::Path(path) => path.display().to_string(),
        SourceBacking::Mtp(reference) => format!(
            "mtp://{}/{}/{}",
            reference.device_id, reference.storage_id, reference.object
        ),
        SourceBacking::Ptp(reference) => format!(
            "ptp://{}{}/{}",
            reference.device_id, reference.folder, reference.name
        ),
    }
}

/// Human-readable byte count for trace lines ("12.0 KB", "1.5 GB").
fn bytes_label(bytes: u64) -> String {
    const UNITS: [(u64, &str); 5] = [
        (1 << 40, "TB"),
        (1 << 30, "GB"),
        (1 << 20, "MB"),
        (1 << 10, "KB"),
        (1, "B"),
    ];
    let Some((factor, unit)) = UNITS.iter().find(|(factor, _)| bytes >= *factor) else {
        return "0 B".to_owned();
    };
    let value = bytes as f64 / *factor as f64;
    if value >= 100.0 {
        format!("{value:.0} {unit}")
    } else {
        format!("{value:.1} {unit}")
    }
}

/// Device-provided names are not trusted to be valid path components: path
/// separators and Windows-reserved characters become underscores, and
/// trailing dots/spaces (silently stripped by Win32) are removed.
fn sanitize_name(name: &str) -> String {
    let sanitized = name.replace(['/', '\\', ':', '<', '>', '"', '|', '?', '*'], "_");
    let trimmed = sanitized.trim_end_matches(['.', ' ']);
    let trimmed = if trimmed.is_empty() { "_" } else { trimmed };
    #[cfg(target_os = "windows")]
    {
        if is_windows_reserved_device_name(trimmed) {
            return format!("_{trimmed}");
        }
    }
    trimmed.to_owned()
}

#[cfg(target_os = "windows")]
fn is_windows_reserved_device_name(name: &str) -> bool {
    let stem = name.split('.').next().unwrap_or(name);
    let upper = stem.to_ascii_uppercase();
    const DEVICES: [&str; 22] = [
        "CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8",
        "COM9", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
    ];
    DEVICES.contains(&upper.as_str())
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
