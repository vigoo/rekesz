use std::{
    cmp::Reverse,
    collections::HashMap,
    fs, io,
    path::{Path, PathBuf},
    sync::{Arc, Condvar, Mutex, atomic::AtomicBool},
};

use crate::{
    jobs::JobLimiter,
    metadata::{GeoCoordinate, ImageOrderingMetadata},
    rendering::{RenderMetadata, RenderVariant},
};

pub type RequestId = u64;

/// A small, independent permit pool for directory and metadata filesystem I/O.
/// It prevents rapid tree expansion from consuming render-worker capacity or
/// issuing an unbounded number of concurrent filesystem walks.
#[derive(Clone)]
pub struct DirectoryScanner {
    permits: Arc<(Mutex<ScannerState>, Condvar)>,
    capacity: usize,
    limiter: JobLimiter,
}

struct ScannerState {
    available: usize,
    next_sequence: u64,
    waiters: HashMap<PathBuf, ScanWaiter>,
}

#[derive(Clone, Copy)]
struct ScanWaiter {
    priority: DirectoryScanPriority,
    sequence: u64,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum DirectoryScanPriority {
    Background,
    Interactive,
}

struct DirectoryPermit<'a> {
    scanner: &'a DirectoryScanner,
}

impl Drop for DirectoryPermit<'_> {
    fn drop(&mut self) {
        let (lock, wake) = &*self.scanner.permits;
        let mut state = lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        state.available = (state.available + 1).min(self.scanner.capacity);
        wake.notify_all();
    }
}

impl DirectoryScanner {
    pub fn new(capacity: usize, limiter: JobLimiter) -> Self {
        Self {
            permits: Arc::new((
                Mutex::new(ScannerState {
                    available: capacity.max(1),
                    next_sequence: 0,
                    waiters: HashMap::new(),
                }),
                Condvar::new(),
            )),
            capacity: capacity.max(1),
            limiter,
        }
    }

    pub fn promote(&self, path: &Path, priority: DirectoryScanPriority) {
        let (lock, wake) = &*self.permits;
        let mut state = lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(waiter) = state.waiters.get_mut(path) {
            waiter.priority = waiter.priority.max(priority);
            wake.notify_all();
        }
    }

    pub fn scan(
        &self,
        path: &Path,
        pair_preference: RawJpegPreference,
        priority: DirectoryScanPriority,
    ) -> Result<DirectoryScan, io::Error> {
        let (lock, wake) = &*self.permits;
        let mut state = lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let sequence = state.next_sequence;
        state.next_sequence += 1;
        state
            .waiters
            .insert(path.to_path_buf(), ScanWaiter { priority, sequence });
        loop {
            let next = state
                .waiters
                .iter()
                .max_by_key(|(_, waiter)| (waiter.priority, Reverse(waiter.sequence)))
                .map(|(path, _)| path.as_path());
            let effective_priority = state
                .waiters
                .get(path)
                .map(|waiter| waiter.priority)
                .unwrap_or(priority);
            // Keep one slot available for a newly selected directory whenever
            // the configured capacity allows it. Already-running filesystem
            // calls remain non-preemptible, but background expansion cannot
            // occupy every scanner slot.
            let has_capacity = match effective_priority {
                DirectoryScanPriority::Interactive => state.available > 0,
                DirectoryScanPriority::Background => {
                    state.available > usize::from(self.capacity > 1)
                }
            };
            if has_capacity && next == Some(path) {
                state.waiters.remove(path);
                state.available -= 1;
                break;
            }
            state = wake
                .wait(state)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
        drop(state);
        let _permit = DirectoryPermit { scanner: self };
        let cancellation = AtomicBool::new(false);
        let _job_permit = self
            .limiter
            .acquire(
                priority == DirectoryScanPriority::Interactive,
                &cancellation,
            )
            .expect("directory scans use a live cancellation token");
        scan_directory(path, pair_preference)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct NodeId(pub(crate) usize);

pub struct DirectoryModel {
    nodes: Vec<DirectoryNode>,
    by_path: HashMap<Arc<Path>, NodeId>,
    root: NodeId,
    next_request: RequestId,
    image_metadata: HashMap<PathBuf, ImageMetadata>,
}

#[derive(Default)]
pub struct ImageMetadata {
    /// Applied recipes are retained independently from decoded-pixel cache
    /// entries, including after the corresponding image is evicted.
    pub renders: HashMap<RenderVariant, RenderMetadata>,
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
    pub metadata_status: DirectoryMetadataStatus,
    metadata_pending: Vec<PathBuf>,
    ordering_metadata: HashMap<PathBuf, OrderingMetadata>,
    pending_capture_order: Option<Vec<PathBuf>>,
}

#[derive(Clone, Copy, Default)]
struct OrderingMetadata {
    capture_time: Option<std::time::SystemTime>,
    fallback_time: Option<std::time::SystemTime>,
    coordinate: Option<GeoCoordinate>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DirectoryMetadataStatus {
    Pending,
    Loading(RequestId),
    Complete,
}

/// The output of the only filesystem operation used to discover a directory.
/// Per-entry metadata can be added later without changing view code.
pub struct DirectoryScan {
    pub directories: Vec<PathBuf>,
    pub images: Vec<PathBuf>,
    /// CR3 → JPEG companion for every pair collapsed by the listing's
    /// RAW/JPEG preference. Empty when the preference retains JPEGs (the
    /// listing already shows them) or when the directory has no pairs.
    /// Learned by the same scan that produced the listing, so pairs are
    /// re-learned automatically whenever a directory is displayed.
    pub companions: HashMap<PathBuf, PathBuf>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum RawJpegPreference {
    #[default]
    Raw,
    Jpeg,
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
            image_metadata: HashMap::new(),
        }
    }

    pub fn root(&self) -> NodeId {
        self.root
    }

    pub fn node(&self, id: NodeId) -> &DirectoryNode {
        &self.nodes[id.0]
    }

    pub fn node_id(&self, path: &Path) -> Option<NodeId> {
        self.by_path.get(path).copied()
    }

    pub fn ensure_node(&mut self, path: PathBuf) -> NodeId {
        self.insert_node(path)
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

    pub fn record_render_metadata(&mut self, path: PathBuf, metadata: RenderMetadata) {
        self.image_metadata
            .entry(path)
            .or_default()
            .renders
            .insert(metadata.variant, metadata);
    }

    pub fn image_metadata(&self, path: &Path) -> Option<&ImageMetadata> {
        self.image_metadata.get(path)
    }

    /// Best known capture instant for an image: the EXIF capture time, or the
    /// file fallback time when no embedded time exists.
    pub fn capture_time(&self, id: NodeId, path: &Path) -> Option<std::time::SystemTime> {
        self.loaded(id)?
            .ordering_metadata
            .get(path)
            .and_then(|metadata| metadata.capture_time.or(metadata.fallback_time))
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
                    metadata_pending: scan.images.clone(),
                    images: scan.images,
                    metadata_status: DirectoryMetadataStatus::Pending,
                    ordering_metadata: HashMap::new(),
                    pending_capture_order: None,
                });
            }
            Err(error) => {
                self.nodes[id.0].contents = DirectoryContents::Error(error.to_string().into())
            }
        }
        true
    }

    pub fn begin_metadata_load(&mut self, id: NodeId) -> Option<(RequestId, Vec<PathBuf>)> {
        let DirectoryContents::Loaded(contents) = &self.nodes[id.0].contents else {
            return None;
        };
        if contents.metadata_status != DirectoryMetadataStatus::Pending {
            return None;
        }
        self.next_request += 1;
        let request = self.next_request;
        let DirectoryContents::Loaded(contents) = &mut self.nodes[id.0].contents else {
            unreachable!();
        };
        contents.metadata_status = DirectoryMetadataStatus::Loading(request);
        Some((request, contents.metadata_pending.clone()))
    }

    pub fn apply_cached_metadata(
        &mut self,
        id: NodeId,
        mut metadata: Vec<ImageOrderingMetadata>,
        missing: Vec<PathBuf>,
    ) -> bool {
        let DirectoryContents::Loaded(contents) = &mut self.nodes[id.0].contents else {
            return false;
        };
        for entry in &metadata {
            contents.ordering_metadata.insert(
                entry.path.clone(),
                OrderingMetadata {
                    capture_time: entry.capture_time,
                    fallback_time: entry.fallback_time,
                    coordinate: entry.coordinate,
                },
            );
        }
        sort_metadata(&mut metadata);
        contents.images = metadata.into_iter().map(|entry| entry.path).collect();
        contents.metadata_pending = missing;
        contents.metadata_status = if contents.metadata_pending.is_empty() {
            DirectoryMetadataStatus::Complete
        } else {
            DirectoryMetadataStatus::Pending
        };
        true
    }

    pub fn cancel_metadata_load(&mut self, id: NodeId, request: RequestId) {
        if let DirectoryContents::Loaded(contents) = &mut self.nodes[id.0].contents
            && contents.metadata_status == DirectoryMetadataStatus::Loading(request)
        {
            contents.metadata_status = DirectoryMetadataStatus::Pending;
        }
    }

    pub fn finish_metadata_load(
        &mut self,
        id: NodeId,
        request: RequestId,
        metadata: Vec<ImageOrderingMetadata>,
    ) -> bool {
        let DirectoryContents::Loaded(contents) = &mut self.nodes[id.0].contents else {
            return false;
        };
        if contents.metadata_status != DirectoryMetadataStatus::Loading(request) {
            return false;
        }
        for entry in metadata {
            contents.ordering_metadata.insert(
                entry.path,
                OrderingMetadata {
                    capture_time: entry.capture_time,
                    fallback_time: entry.fallback_time,
                    coordinate: entry.coordinate,
                },
            );
        }
        let mut proposed = contents
            .images
            .iter()
            .map(|path| {
                let metadata = contents
                    .ordering_metadata
                    .get(path)
                    .copied()
                    .unwrap_or_default();
                ImageOrderingMetadata {
                    path: path.clone(),
                    capture_time: metadata.capture_time,
                    fallback_time: metadata.fallback_time,
                    coordinate: metadata.coordinate,
                }
            })
            .collect::<Vec<_>>();
        sort_metadata(&mut proposed);
        let proposed = proposed
            .into_iter()
            .map(|entry| entry.path)
            .collect::<Vec<_>>();
        contents.pending_capture_order = (proposed != contents.images).then_some(proposed);
        contents.metadata_pending.clear();
        contents.metadata_status = DirectoryMetadataStatus::Complete;
        true
    }

    pub fn has_pending_capture_order(&self, id: NodeId) -> bool {
        self.loaded(id)
            .is_some_and(|contents| contents.pending_capture_order.is_some())
    }

    pub fn coordinate(&self, id: NodeId, path: &Path) -> Option<GeoCoordinate> {
        self.loaded(id)?
            .ordering_metadata
            .get(path)
            .and_then(|metadata| metadata.coordinate)
    }

    pub fn apply_pending_capture_order(&mut self, id: NodeId) -> bool {
        let Some(contents) = self.loaded_mut(id) else {
            return false;
        };
        let Some(order) = contents.pending_capture_order.take() else {
            return false;
        };
        contents.images = order;
        true
    }

    fn loaded_mut(&mut self, id: NodeId) -> Option<&mut LoadedDirectory> {
        match &mut self.nodes[id.0].contents {
            DirectoryContents::Loaded(contents) => Some(contents),
            _ => None,
        }
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
pub fn scan_directory(
    path: &Path,
    pair_preference: RawJpegPreference,
) -> Result<DirectoryScan, io::Error> {
    let mut directories = Vec::new();
    let mut image_paths = Vec::new();
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let path = entry.path();
        if file_type.is_dir() {
            if entry.file_name() != crate::catalog::CATALOG_DIR {
                directories.push(path);
            }
        } else if file_type.is_file() && is_supported_image(&path) {
            let modified = entry
                .metadata()
                .ok()
                .and_then(|metadata| metadata.modified().ok());
            image_paths.push((modified, path));
        }
    }
    let modified_by_path = image_paths
        .iter()
        .map(|(modified, path)| (path.clone(), *modified))
        .collect::<HashMap<_, _>>();
    let (retained, companions) = deduplicate_cr3_jpeg_pairs(
        image_paths.into_iter().map(|(_, path)| path).collect(),
        pair_preference,
    );
    let mut images = retained
        .into_iter()
        .map(|path| (modified_by_path.get(&path).copied().flatten(), path))
        .collect::<Vec<_>>();
    directories.sort_by_cached_key(|path| sort_key(path));
    images.sort_by_cached_key(|(time, path)| (time.is_none(), *time, sort_key(path)));
    Ok(DirectoryScan {
        directories,
        images: images.into_iter().map(|(_, path)| path).collect(),
        companions,
    })
}

/// Builds the complete face-analysis inventory using the same eligibility and
/// RAW/JPEG pairing rules as the browser. Intended for a background job.
#[cfg(test)]
pub fn scan_library_inventory(
    root: &Path,
    hidden_subtrees: &std::collections::HashSet<PathBuf>,
    pair_preference: RawJpegPreference,
    cancelled: &AtomicBool,
) -> Result<Vec<PathBuf>, io::Error> {
    scan_library_inventory_with_progress(
        root,
        hidden_subtrees,
        pair_preference,
        cancelled,
        |_, _, _, _| {},
    )
}

pub fn scan_library_inventory_with_progress(
    root: &Path,
    hidden_subtrees: &std::collections::HashSet<PathBuf>,
    pair_preference: RawJpegPreference,
    cancelled: &AtomicBool,
    mut progress: impl FnMut(usize, usize, usize, &Path),
) -> Result<Vec<PathBuf>, io::Error> {
    let mut pending = vec![root.to_path_buf()];
    let mut images = Vec::new();
    let mut directories_scanned = 0;
    while let Some(directory) = pending.pop() {
        if cancelled.load(std::sync::atomic::Ordering::Acquire) {
            return Err(io::Error::new(
                io::ErrorKind::Interrupted,
                "library inventory scan cancelled",
            ));
        }
        if directory != root
            && hidden_subtrees
                .iter()
                .any(|hidden| directory.starts_with(hidden))
        {
            continue;
        }
        let scan = scan_directory(&directory, pair_preference)?;
        images.extend(scan.images);
        pending.extend(scan.directories.into_iter().filter(|child| {
            !hidden_subtrees
                .iter()
                .any(|hidden| child.starts_with(hidden))
        }));
        directories_scanned += 1;
        progress(directories_scanned, pending.len(), images.len(), &directory);
    }
    images.sort();
    Ok(images)
}

fn deduplicate_cr3_jpeg_pairs(
    paths: Vec<PathBuf>,
    preference: RawJpegPreference,
) -> (Vec<PathBuf>, HashMap<PathBuf, PathBuf>) {
    let mut by_stem: HashMap<String, Vec<PathBuf>> = HashMap::new();
    for path in paths {
        let stem = path
            .file_stem()
            .unwrap_or(path.as_os_str())
            .to_string_lossy()
            .to_lowercase();
        by_stem.entry(stem).or_default().push(path);
    }

    let mut companions = HashMap::new();
    let retained = by_stem
        .into_values()
        .flat_map(|group| {
            let cr3 = group
                .iter()
                .find(|path| has_extension(path, &["cr3"]))
                .cloned();
            let jpeg = group
                .iter()
                .find(|path| has_extension(path, &["jpg", "jpeg"]))
                .cloned();
            match (cr3, jpeg) {
                (Some(raw), Some(jpeg)) => {
                    // The non-preferred half leaves the listing but is
                    // remembered as the thumbnail companion (raw listings
                    // only; JPEG-preferred listings already show it).
                    if preference == RawJpegPreference::Raw {
                        companions.insert(raw.clone(), jpeg.clone());
                    }
                    let mut retained = group
                        .into_iter()
                        .filter(|path| !has_extension(path, &["cr3", "jpg", "jpeg"]))
                        .collect::<Vec<_>>();
                    retained.push(match preference {
                        RawJpegPreference::Raw => raw,
                        RawJpegPreference::Jpeg => jpeg,
                    });
                    retained
                }
                _ => group,
            }
        })
        .collect::<Vec<_>>();
    (retained, companions)
}

fn has_extension(path: &Path, expected: &[&str]) -> bool {
    path.extension().is_some_and(|actual| {
        expected
            .iter()
            .any(|expected| actual.eq_ignore_ascii_case(expected))
    })
}

pub(crate) fn is_supported_image(path: &Path) -> bool {
    image::ImageFormat::from_path(path).is_ok()
        || crate::formats::is_raw(path)
        || crate::formats::is_heif(path)
}

fn sort_key(path: &Path) -> String {
    path.file_name()
        .unwrap_or(path.as_os_str())
        .to_string_lossy()
        .to_lowercase()
}

pub(crate) fn sort_metadata(metadata: &mut [ImageOrderingMetadata]) {
    metadata.sort_by_cached_key(|entry| {
        (
            entry.capture_time.or(entry.fallback_time).is_none(),
            entry.capture_time.or(entry.fallback_time),
            sort_key(&entry.path),
        )
    });
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
