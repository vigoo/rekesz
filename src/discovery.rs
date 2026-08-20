use std::{
    collections::HashMap,
    fs, io,
    path::{Path, PathBuf},
    sync::Arc,
};

pub type RequestId = u64;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct NodeId(pub(crate) usize);

pub struct DirectoryModel {
    nodes: Vec<DirectoryNode>,
    by_path: HashMap<Arc<Path>, NodeId>,
    root: NodeId,
    next_request: RequestId,
}

pub struct DirectoryNode {
    pub path: Arc<Path>,
    pub name: Arc<str>,
    pub contents: DirectoryContents,
}

pub enum DirectoryContents {
    Unloaded,
    Loading(RequestId),
    Loaded(LoadedDirectory),
    Error(Arc<str>),
}

pub struct LoadedDirectory {
    pub children: Vec<NodeId>,
    pub images: Vec<PathBuf>,
}

/// The output of the only filesystem operation used to discover a directory.
/// Per-entry metadata can be added later without changing view code.
pub struct DirectoryScan {
    pub directories: Vec<PathBuf>,
    pub images: Vec<PathBuf>,
}

impl DirectoryModel {
    pub fn new(root: PathBuf) -> Self {
        let root_path: Arc<Path> = Arc::from(root);
        let root_node = DirectoryNode {
            name: root_path.to_string_lossy().into_owned().into(),
            path: root_path.clone(),
            contents: DirectoryContents::Unloaded,
        };
        Self {
            nodes: vec![root_node],
            by_path: HashMap::from([(root_path, NodeId(0))]),
            root: NodeId(0),
            next_request: 0,
        }
    }

    pub fn root(&self) -> NodeId {
        self.root
    }

    pub fn node(&self, id: NodeId) -> &DirectoryNode {
        &self.nodes[id.0]
    }

    pub fn loaded(&self, id: NodeId) -> Option<&LoadedDirectory> {
        match &self.node(id).contents {
            DirectoryContents::Loaded(contents) => Some(contents),
            _ => None,
        }
    }

    pub fn begin_load(&mut self, id: NodeId) -> Option<(RequestId, Arc<Path>)> {
        if !matches!(self.node(id).contents, DirectoryContents::Unloaded) {
            return None;
        }
        self.next_request += 1;
        let request = self.next_request;
        let path = self.node(id).path.clone();
        self.nodes[id.0].contents = DirectoryContents::Loading(request);
        Some((request, path))
    }

    pub fn finish_load(
        &mut self,
        id: NodeId,
        request: RequestId,
        result: Result<DirectoryScan, io::Error>,
    ) -> bool {
        if !matches!(self.node(id).contents, DirectoryContents::Loading(active) if active == request)
        {
            return false;
        }
        match result {
            Ok(scan) => {
                let children = scan
                    .directories
                    .into_iter()
                    .map(|path| self.insert_node(path))
                    .collect();
                self.nodes[id.0].contents = DirectoryContents::Loaded(LoadedDirectory {
                    children,
                    images: scan.images,
                });
            }
            Err(error) => {
                self.nodes[id.0].contents = DirectoryContents::Error(error.to_string().into())
            }
        }
        true
    }

    fn insert_node(&mut self, path: PathBuf) -> NodeId {
        if let Some(id) = self.by_path.get(path.as_path()) {
            return *id;
        }
        let path: Arc<Path> = Arc::from(path);
        let name = path
            .file_name()
            .unwrap_or(path.as_os_str())
            .to_string_lossy()
            .into_owned()
            .into();
        let id = NodeId(self.nodes.len());
        self.nodes.push(DirectoryNode {
            path: path.clone(),
            name,
            contents: DirectoryContents::Unloaded,
        });
        self.by_path.insert(path, id);
        id
    }
}

/// Blocking filesystem I/O. Call only from a background executor.
pub fn scan_directory(path: &Path) -> Result<DirectoryScan, io::Error> {
    let mut directories = Vec::new();
    let mut images = Vec::new();
    for entry in fs::read_dir(path)? {
        let Ok(entry) = entry else { continue };
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        let path = entry.path();
        if file_type.is_dir() {
            directories.push(path);
        } else if file_type.is_file() && image::ImageFormat::from_path(&path).is_ok() {
            images.push(path);
        }
    }
    directories.sort_by_cached_key(|path| sort_key(path));
    images.sort_by_cached_key(|path| sort_key(path));
    Ok(DirectoryScan {
        directories,
        images,
    })
}

fn sort_key(path: &Path) -> String {
    path.file_name()
        .unwrap_or(path.as_os_str())
        .to_string_lossy()
        .to_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_node_can_only_start_one_load() {
        let mut model = DirectoryModel::new(PathBuf::from("/pictures"));
        let root = model.root();
        assert!(model.begin_load(root).is_some());
        assert!(model.begin_load(root).is_none());
    }

    #[test]
    fn loaded_and_failed_nodes_are_cached() {
        let mut model = DirectoryModel::new(PathBuf::from("/pictures"));
        let root = model.root();
        let (request, _) = model.begin_load(root).unwrap();
        assert!(model.finish_load(
            root,
            request,
            Ok(DirectoryScan {
                directories: vec![PathBuf::from("/pictures/a")],
                images: vec![PathBuf::from("/pictures/one.jpg")],
            }),
        ));
        assert!(model.begin_load(root).is_none());
        let loaded = model.loaded(root).unwrap();
        assert_eq!(loaded.children.len(), 1);
        assert_eq!(loaded.images.len(), 1);
    }

    #[test]
    fn stale_results_are_ignored() {
        let mut model = DirectoryModel::new(PathBuf::from("/pictures"));
        let root = model.root();
        let (request, _) = model.begin_load(root).unwrap();
        assert!(!model.finish_load(
            root,
            request + 1,
            Ok(DirectoryScan {
                directories: Vec::new(),
                images: Vec::new(),
            }),
        ));
        assert!(matches!(
            model.node(root).contents,
            DirectoryContents::Loading(active) if active == request
        ));
    }
}
