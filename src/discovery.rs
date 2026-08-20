use std::{
    collections::HashMap,
    fs, io,
    path::{Path, PathBuf},
};

/// An immutable snapshot of the directory and image layout below a root.
///
/// Building the catalog calls `read_dir` exactly once for every directory it
/// can enter. Rendering and selection never touch the filesystem tree again.
pub struct Catalog {
    root: PathBuf,
    directories: HashMap<PathBuf, Directory>,
}

pub struct Directory {
    pub children: Vec<PathBuf>,
    pub images: Vec<PathBuf>,
    pub scan_error: Option<io::Error>,
}

impl Catalog {
    pub fn scan(root: PathBuf) -> Self {
        let mut catalog = Self {
            root: root.clone(),
            directories: HashMap::new(),
        };
        catalog.scan_directory(root);
        catalog
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn directory(&self, path: &Path) -> Option<&Directory> {
        self.directories.get(path)
    }

    fn scan_directory(&mut self, path: PathBuf) {
        let entries = match fs::read_dir(&path) {
            Ok(entries) => entries,
            Err(error) => {
                self.directories.insert(
                    path,
                    Directory {
                        children: Vec::new(),
                        images: Vec::new(),
                        scan_error: Some(error),
                    },
                );
                return;
            }
        };

        let mut children = Vec::new();
        let mut images = Vec::new();
        for entry in entries.flatten() {
            let Ok(file_type) = entry.file_type() else {
                continue;
            };
            let entry_path = entry.path();
            if file_type.is_dir() {
                children.push(entry_path);
            } else if file_type.is_file() && image::ImageFormat::from_path(&entry_path).is_ok() {
                images.push(entry_path);
            }
        }

        children.sort_by_cached_key(sort_key);
        images.sort_by_cached_key(sort_key);
        let to_scan = children.clone();
        self.directories.insert(
            path,
            Directory {
                children,
                images,
                scan_error: None,
            },
        );
        for child in to_scan {
            self.scan_directory(child);
        }
    }
}

fn sort_key(path: &PathBuf) -> String {
    path.file_name()
        .unwrap_or(path.as_os_str())
        .to_string_lossy()
        .to_lowercase()
}
