//! External import sources: mounted volumes, MTP devices, and arbitrary
//! folders chosen by the user.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

pub mod mtp;
pub mod volumes;

/// Stable identifier for a source across refreshes.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct SourceId(Arc<str>);

impl SourceId {
    pub(crate) fn new(id: impl Into<Arc<str>>) -> Self {
        Self(id.into())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SourceKind {
    /// A mounted volume or drive; `removable` marks USB sticks, cards, discs.
    Volume { removable: bool },
    /// A connected media device (phone, camera) accessed over MTP.
    Device { vendor: String, model: String },
    /// An arbitrary directory chosen by the user.
    Folder,
}

/// How the bytes of a source item are reached.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum SourceBacking {
    /// A filesystem location. The import root for path-backed sources; a
    /// concrete file or directory when carried by a directory listing.
    Path(PathBuf),
    /// An MTP device object. `object == 0` denotes the root directory of the
    /// storage.
    Mtp(mtp::MtpObjectRef),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SourceCapacity {
    pub total: u64,
    pub available: u64,
}

impl SourceBacking {
    /// The filesystem location when this backing is path-based; device
    /// backings have none.
    pub fn root_path(&self) -> Option<PathBuf> {
        match self {
            Self::Path(path) => Some(path.clone()),
            Self::Mtp(_) => None,
        }
    }
}

/// One entry of the source list shown by the import view.
#[derive(Clone, Debug)]
pub struct SourceEntry {
    pub id: SourceId,
    pub label: String,
    pub kind: SourceKind,
    pub backing: SourceBacking,
    pub capacity: Option<SourceCapacity>,
}

/// One entry of a source directory listing. Every entry — directory or file —
/// carries the backing reference used to address it further.
#[derive(Clone, Debug)]
pub struct SourceDirEntry {
    pub name: String,
    pub is_dir: bool,
    pub backing: SourceBacking,
    /// File size in bytes; zero for directories.
    pub size: u64,
    pub modified: Option<std::time::SystemTime>,
}

/// Uniform read access to a source, whether it is a plain directory tree or a
/// device accessed over MTP.
///
/// Implementations list directories directories-first, then files, each group
/// ordered by case-insensitive name. Entries that cannot be stat'ed are
/// skipped; unreadable directories surface as an `io::Error`.
pub trait SourceAccess: Send + Sync {
    /// List a directory of this source. `dir` is the source root or a
    /// directory backing obtained from a previous listing.
    fn list_dir(&self, dir: &SourceBacking) -> std::io::Result<Vec<SourceDirEntry>>;

    /// Stream a file's bytes to `out`, invoking `progress` with each copied
    /// byte count, and return the total number of bytes written. Aborts with
    /// `ErrorKind::Interrupted` once `cancel` is set; partial output is the
    /// caller's problem to clean up.
    fn read_file(
        &self,
        file: &SourceBacking,
        out: &mut dyn std::io::Write,
        cancel: &AtomicBool,
        progress: &mut dyn FnMut(u64),
    ) -> std::io::Result<u64>;
}

/// Read access to path-backed sources (volumes, folders, network mounts).
pub struct PathSource;

const COPY_CHUNK: usize = 1024 * 1024;

impl SourceAccess for PathSource {
    fn list_dir(&self, dir: &SourceBacking) -> std::io::Result<Vec<SourceDirEntry>> {
        let SourceBacking::Path(path) = dir else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "path source cannot list MTP objects",
            ));
        };
        let mut entries = Vec::new();
        for entry in std::fs::read_dir(path)? {
            let entry = match entry {
                Ok(entry) => entry,
                Err(_) => continue,
            };
            let file_type = match entry.file_type() {
                Ok(file_type) => file_type,
                Err(_) => continue,
            };
            // `DirEntry::file_type` does not follow symlinks; resolve them
            // with the free `fs::metadata` (DirEntry::metadata is lstat) so
            // symlinked directories stay navigable and broken symlinks are
            // skipped.
            let metadata = if file_type.is_symlink() {
                match std::fs::metadata(entry.path()) {
                    Ok(metadata) if metadata.is_dir() => metadata,
                    _ => continue,
                }
            } else {
                match entry.metadata() {
                    Ok(metadata) => metadata,
                    Err(_) => continue,
                }
            };
            let name = entry.file_name().to_string_lossy().into_owned();
            entries.push(SourceDirEntry {
                is_dir: metadata.is_dir(),
                backing: SourceBacking::Path(entry.path()),
                size: if metadata.is_dir() { 0 } else { metadata.len() },
                modified: metadata.modified().ok(),
                name,
            });
        }
        entries.sort_by(|a, b| match (a.is_dir, b.is_dir) {
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            _ => a.name.to_lowercase().cmp(&b.name.to_lowercase()),
        });
        Ok(entries)
    }

    fn read_file(
        &self,
        file: &SourceBacking,
        out: &mut dyn std::io::Write,
        cancel: &AtomicBool,
        progress: &mut dyn FnMut(u64),
    ) -> std::io::Result<u64> {
        let SourceBacking::Path(path) = file else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "path source cannot read MTP objects",
            ));
        };
        let mut reader = std::io::BufReader::with_capacity(COPY_CHUNK, std::fs::File::open(path)?);
        let mut copied = 0_u64;
        loop {
            if cancel.load(std::sync::atomic::Ordering::Acquire) {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::Interrupted,
                    "import cancelled",
                ));
            }
            let chunk = std::io::BufRead::fill_buf(&mut reader)?;
            if chunk.is_empty() {
                return Ok(copied);
            }
            let len = chunk.len();
            out.write_all(chunk)?;
            std::io::BufRead::consume(&mut reader, len);
            copied += len as u64;
            progress(copied);
        }
    }
}

/// Re-collect the source list: volumes, connected devices, then user folders.
#[derive(Default)]
pub struct SourceRegistry;

impl SourceRegistry {
    pub fn new() -> Self {
        Self
    }

    /// Volumes and devices come first (removable volumes before fixed ones),
    /// then user folders. This performs filesystem and device enumeration and
    /// must not be called on the UI thread.
    pub fn refresh(&self, folders: &[PathBuf]) -> Vec<SourceEntry> {
        let raw_volumes = volumes::VolumeProvider::raw_volumes();
        let mut entries = volumes::VolumeProvider::entries_from(&raw_volumes);
        entries.extend(mtp::DeviceProvider.collect());
        merge_folder_entries(entries, folders, &raw_volumes)
    }

    /// The reader for a source's contents; device sources delegate to the
    /// platform MTP backend.
    pub fn access_for(&self, backing: &SourceBacking) -> Option<Arc<dyn SourceAccess>> {
        match backing {
            SourceBacking::Path(_) => Some(Arc::new(PathSource)),
            SourceBacking::Mtp(reference) => mtp::DeviceProvider.access_for(&reference.device_id),
        }
    }
}

/// Append user folder entries that are not already represented by a volume:
/// a folder equal to a mount point is redundant for any volume, and folders
/// below a *removable* volume's mount are covered by that volume's entry.
/// Folders that merely contain a mount (or sit on the system volume) stay.
pub(crate) fn merge_folder_entries(
    mut entries: Vec<SourceEntry>,
    folders: &[PathBuf],
    volumes: &[volumes::RawVolume],
) -> Vec<SourceEntry> {
    let mounts = entries
        .iter()
        .filter_map(|entry| match &entry.backing {
            SourceBacking::Path(path) => Some(path.clone()),
            SourceBacking::Mtp(_) => None,
        })
        .collect::<Vec<_>>();
    for folder in folders {
        let folder = canonical_folder(folder);
        if mounts
            .iter()
            .any(|mount| paths_equal(folder.as_path(), mount))
        {
            continue;
        }
        if volumes
            .iter()
            .filter(|volume| volume.removable)
            .any(|volume| folder.starts_with(&volume.mount_point))
        {
            continue;
        }
        if entries
            .iter()
            .any(|entry| matches!(&entry.backing, SourceBacking::Path(path) if paths_equal(path, folder.as_path())))
        {
            continue;
        }
        entries.push(SourceEntry {
            id: SourceId::new(format!("folder:{}", folder.display())),
            label: folder_label(&folder),
            kind: SourceKind::Folder,
            backing: SourceBacking::Path(folder.clone()),
            capacity: folder_capacity(&folder, volumes),
        });
    }
    entries
}

/// Resolve symlinks so a source keeps working regardless of how the user
/// picked the folder. Falls back to the raw path when resolution fails.
pub(crate) fn canonical_folder(folder: &Path) -> PathBuf {
    match std::fs::canonicalize(folder) {
        Ok(canonical) => strip_windows_verbatim(&canonical),
        Err(_) => folder.to_path_buf(),
    }
}

#[cfg(target_os = "windows")]
fn strip_windows_verbatim(path: &Path) -> PathBuf {
    let lossy = path.to_string_lossy();
    match lossy.strip_prefix(r"\\?\") {
        Some(stripped) => PathBuf::from(stripped),
        None => path.to_path_buf(),
    }
}

#[cfg(not(target_os = "windows"))]
fn strip_windows_verbatim(path: &Path) -> PathBuf {
    path.to_path_buf()
}

/// Path equality that ignores drive-letter casing on Windows.
fn paths_equal(a: &Path, b: &Path) -> bool {
    #[cfg(target_os = "windows")]
    {
        a.to_string_lossy()
            .eq_ignore_ascii_case(&b.to_string_lossy())
    }
    #[cfg(not(target_os = "windows"))]
    {
        a == b
    }
}

fn folder_label(folder: &Path) -> String {
    folder
        .file_name()
        .unwrap_or(folder.as_os_str())
        .to_string_lossy()
        .into_owned()
}

/// Capacity of the deepest volume mount containing `folder`.
fn folder_capacity(folder: &Path, volumes: &[volumes::RawVolume]) -> Option<SourceCapacity> {
    volumes
        .iter()
        .filter(|volume| folder.starts_with(&volume.mount_point))
        .max_by_key(|volume| volume.mount_point.as_os_str().len())
        .map(|volume| SourceCapacity {
            total: volume.total,
            available: volume.available,
        })
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
