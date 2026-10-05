//! Duplicate detection against the library: an index of lowercase file names
//! plus sizes, built from a library walk. Content hashes are future work.
//!
//! Deliberately divergent from the discovery inventory: this index covers
//! every file (including user-hidden subtrees, so hiding a folder does not
//! make re-adding its photos look new) and does not follow symlinked
//! directories, matching the best-effort nature of the check.

use std::collections::HashSet;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

/// Safety budget for the walk, mirroring the source scan; mount cycles and
/// pathological trees stop here instead of hanging the build.
const MAX_INDEX_DIRECTORIES: usize = 24 * 1024;

/// Key of a library file: lowercase name and byte size.
pub type DuplicateKey = (String, u64);

pub fn key_for(name: &str, size: u64) -> DuplicateKey {
    (name.to_lowercase(), size)
}

/// Best-effort index of everything currently in the library root. Unreadable
/// directories are skipped; the catalog directory (`.rekesz`) is excluded.
pub struct LibraryIndex {
    keys: HashSet<DuplicateKey>,
    pub files: u64,
}

impl LibraryIndex {
    pub fn build(
        root: &Path,
        cancel: &AtomicBool,
        progress: &mut dyn FnMut(u64),
    ) -> Result<Self, BuildError> {
        let mut keys = HashSet::new();
        let mut files = 0_u64;
        let mut visited = HashSet::new();
        let mut pending = vec![root.to_path_buf()];
        while let Some(directory) = pending.pop() {
            if cancel.load(Ordering::Acquire) {
                return Err(BuildError::Cancelled);
            }
            if !visited.insert(directory.clone()) || visited.len() > MAX_INDEX_DIRECTORIES {
                continue;
            }
            let entries = match std::fs::read_dir(&directory) {
                Ok(entries) => entries,
                Err(_) => continue,
            };
            for entry in entries.filter_map(|entry| entry.ok()) {
                let Ok(file_type) = entry.file_type() else {
                    continue;
                };
                if file_type.is_dir() {
                    if entry.file_name() != crate::catalog::CATALOG_DIR {
                        pending.push(entry.path());
                    }
                    continue;
                }
                let size = entry.metadata().map(|metadata| metadata.len()).unwrap_or(0);
                if keys.insert(key_for(&entry.file_name().to_string_lossy(), size)) {
                    files += 1;
                    progress(files);
                }
            }
        }
        Ok(Self { keys, files })
    }

    pub fn contains(&self, name: &str, size: u64) -> bool {
        self.keys.contains(&key_for(name, size))
    }
}

#[derive(Debug)]
pub enum BuildError {
    Cancelled,
}
