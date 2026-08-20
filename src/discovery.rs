use std::{
    collections::HashMap,
    fs, io,
    path::{Path, PathBuf},
    sync::Arc,
    time::SystemTime,
};

use chrono::{Local, NaiveDateTime, TimeZone as _};

use crate::rendering::{RenderMetadata, RenderVariant};

pub type RequestId = u64;

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
}

/// The output of the only filesystem operation used to discover a directory.
/// Per-entry metadata can be added later without changing view code.
pub struct DirectoryScan {
    pub directories: Vec<PathBuf>,
    pub images: Vec<PathBuf>,
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

    #[allow(dead_code)]
    pub fn image_metadata(&self, path: &Path) -> Option<&ImageMetadata> {
        self.image_metadata.get(path)
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
pub fn scan_directory(
    path: &Path,
    pair_preference: RawJpegPreference,
) -> Result<DirectoryScan, io::Error> {
    let mut directories = Vec::new();
    let mut image_paths = Vec::new();
    for entry in fs::read_dir(path)? {
        let Ok(entry) = entry else { continue };
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        let path = entry.path();
        if file_type.is_dir() {
            directories.push(path);
        } else if file_type.is_file() && is_supported_image(&path) {
            image_paths.push(path);
        }
    }
    let mut images = deduplicate_cr3_jpeg_pairs(image_paths, pair_preference)
        .into_iter()
        .map(|path| {
            let created = fs::metadata(&path)
                .ok()
                .and_then(|metadata| metadata.created().ok());
            (capture_time(&path).or(created), path)
        })
        .collect::<Vec<_>>();
    directories.sort_by_cached_key(|path| sort_key(path));
    images.sort_by_cached_key(|(time, path)| (time.is_none(), *time, sort_key(path)));
    Ok(DirectoryScan {
        directories,
        images: images.into_iter().map(|(_, path)| path).collect(),
    })
}

fn deduplicate_cr3_jpeg_pairs(paths: Vec<PathBuf>, preference: RawJpegPreference) -> Vec<PathBuf> {
    let mut by_stem: HashMap<String, Vec<PathBuf>> = HashMap::new();
    for path in paths {
        let stem = path
            .file_stem()
            .unwrap_or(path.as_os_str())
            .to_string_lossy()
            .to_lowercase();
        by_stem.entry(stem).or_default().push(path);
    }

    by_stem
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
        .collect()
}

fn has_extension(path: &Path, expected: &[&str]) -> bool {
    path.extension().is_some_and(|actual| {
        expected
            .iter()
            .any(|expected| actual.eq_ignore_ascii_case(expected))
    })
}

fn capture_time(path: &Path) -> Option<SystemTime> {
    if crate::formats::is_raw(path) {
        raw_capture_time(path).or_else(|| exif_capture_time(path))
    } else {
        exif_capture_time(path)
    }
}

fn raw_capture_time(path: &Path) -> Option<SystemTime> {
    use rawler::{decoders::RawDecodeParams, rawsource::RawSource};

    let source = RawSource::new(path).ok()?;
    let decoder = rawler::global_loader().get_decoder(&source).ok()?;
    let metadata = decoder
        .raw_metadata(&source, &RawDecodeParams::default())
        .ok()?;
    metadata
        .exif
        .date_time_original
        .as_deref()
        .and_then(parse_exif_time)
}

fn exif_capture_time(path: &Path) -> Option<SystemTime> {
    use std::{fs::File, io::BufReader};

    let file = File::open(path).ok()?;
    let exif = exif::Reader::new()
        .read_from_container(&mut BufReader::new(file))
        .ok()?;
    let field = exif.get_field(exif::Tag::DateTimeOriginal, exif::In::PRIMARY)?;
    let exif::Value::Ascii(values) = &field.value else {
        return None;
    };
    std::str::from_utf8(values.first()?)
        .ok()
        .map(|value| value.trim_matches(['\0', ' ']))
        .and_then(parse_exif_time)
}

fn parse_exif_time(value: &str) -> Option<SystemTime> {
    let naive = NaiveDateTime::parse_from_str(value, "%Y:%m:%d %H:%M:%S").ok()?;
    let local = Local.from_local_datetime(&naive).earliest()?;
    Some(local.into())
}

fn is_supported_image(path: &Path) -> bool {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recognizes_raw_and_heif_extensions_case_insensitively() {
        for name in [
            "photo.nef",
            "photo.NEF",
            "photo.cr3",
            "photo.HEIC",
            "photo.heif",
        ] {
            assert!(is_supported_image(Path::new(name)), "{name}");
        }
        assert!(!is_supported_image(Path::new("notes.txt")));
    }

    #[test]
    fn parses_exif_capture_time() {
        assert!(parse_exif_time("2026:07:24 14:17:45").is_some());
        assert!(parse_exif_time("not a timestamp").is_none());
    }

    #[test]
    fn deduplicates_cr3_jpeg_pairs_according_to_preference() {
        let paths = vec![
            PathBuf::from("IMG_1.JPG"),
            PathBuf::from("IMG_1.CR3"),
            PathBuf::from("IMG_1.PNG"),
            PathBuf::from("IMG_2.JPG"),
        ];
        let raw = deduplicate_cr3_jpeg_pairs(paths.clone(), RawJpegPreference::Raw);
        assert!(raw.contains(&PathBuf::from("IMG_1.CR3")));
        assert!(!raw.contains(&PathBuf::from("IMG_1.JPG")));
        assert!(raw.contains(&PathBuf::from("IMG_2.JPG")));
        assert!(raw.contains(&PathBuf::from("IMG_1.PNG")));

        let jpeg = deduplicate_cr3_jpeg_pairs(paths, RawJpegPreference::Jpeg);
        assert!(jpeg.contains(&PathBuf::from("IMG_1.JPG")));
        assert!(!jpeg.contains(&PathBuf::from("IMG_1.CR3")));
    }

    #[test]
    fn reads_capture_time_from_the_cr3_regression_sample() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test/IMG_0052.CR3");
        if path.exists() {
            assert!(capture_time(&path).is_some());
        }
    }

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
