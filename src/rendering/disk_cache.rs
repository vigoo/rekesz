use std::{
    fs,
    io::{BufWriter, Cursor, Write},
    path::PathBuf,
    sync::{
        Arc, RwLock,
        atomic::{AtomicU64, Ordering},
    },
    time::UNIX_EPOCH,
};

#[cfg(test)]
use image::DynamicImage;
use image::ImageFormat;

use super::{
    PhotographicSettings, RenderMetadata, RenderRequest, RenderVariant, RenderedImage,
    SourceIdentity,
};

const CACHE_VERSION: &str = "render-v2";
const DEFAULT_CACHE_BYTES: u64 = 10 * 1024 * 1024 * 1024;
const CLEANUP_INTERVAL: u64 = 32;

#[derive(Clone, Debug)]
pub struct DiskRenderCache {
    root: PathBuf,
    max_bytes: u64,
    lock: Arc<RwLock<()>>,
    stores_since_cleanup: Arc<AtomicU64>,
}

impl DiskRenderCache {
    pub fn new(root: PathBuf) -> Self {
        Self {
            root,
            max_bytes: DEFAULT_CACHE_BYTES,
            lock: Arc::new(RwLock::new(())),
            stores_since_cleanup: Arc::new(AtomicU64::new(0)),
        }
    }

    pub fn load(
        &self,
        request: &RenderRequest,
        identity: &SourceIdentity,
        settings: &PhotographicSettings,
    ) -> Result<Option<RenderedImage>, String> {
        let key = cache_key(request, identity, settings)?;
        let extension = cache_extension(request.variant);
        let image_path = self.root.join(format!("{key}.{extension}"));
        let metadata_path = self.root.join(format!("{key}.json"));
        let (image_bytes, metadata_bytes) = {
            let _guard = self
                .lock
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if !image_path.is_file() || !metadata_path.is_file() {
                return Ok(None);
            }
            (
                fs::read(image_path).map_err(|error| error.to_string())?,
                fs::read(metadata_path).map_err(|error| error.to_string())?,
            )
        };
        let metadata: RenderMetadata =
            serde_json::from_slice(&metadata_bytes).map_err(|error| error.to_string())?;
        let image = image::ImageReader::new(Cursor::new(image_bytes))
            .with_guessed_format()
            .map_err(|error| error.to_string())?
            .decode()
            .map_err(|error| error.to_string())?;
        let _guard = self
            .lock
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if self.root.join(format!("{key}.{extension}")).is_file()
            && self.root.join(format!("{key}.json")).is_file()
        {
            let _ = fs::write(
                self.root.join(format!("{key}.access")),
                unix_timestamp_nanos().to_string(),
            );
        }
        Ok(Some(RenderedImage { image, metadata }))
    }

    pub fn store(
        &self,
        request: &RenderRequest,
        identity: &SourceIdentity,
        settings: &PhotographicSettings,
        rendered: &RenderedImage,
    ) -> Result<(), String> {
        let key = cache_key(request, identity, settings)?;
        fs::create_dir_all(&self.root).map_err(|error| error.to_string())?;
        let extension = cache_extension(request.variant);
        let image_path = self.root.join(format!("{key}.{extension}"));
        let metadata_path = self.root.join(format!("{key}.json"));

        let mut image_temp =
            tempfile::NamedTempFile::new_in(&self.root).map_err(|error| error.to_string())?;
        {
            let mut writer = BufWriter::new(image_temp.as_file_mut());
            rendered
                .image
                .write_to(&mut writer, cache_format(request.variant))
                .map_err(|error| error.to_string())?;
            writer.flush().map_err(|error| error.to_string())?;
        }
        image_temp
            .as_file_mut()
            .flush()
            .map_err(|error| error.to_string())?;
        let mut metadata_temp =
            tempfile::NamedTempFile::new_in(&self.root).map_err(|error| error.to_string())?;
        {
            let mut writer = BufWriter::new(metadata_temp.as_file_mut());
            serde_json::to_writer(&mut writer, &rendered.metadata)
                .map_err(|error| error.to_string())?;
            writer.flush().map_err(|error| error.to_string())?;
        }
        metadata_temp
            .as_file_mut()
            .flush()
            .map_err(|error| error.to_string())?;
        let _guard = self
            .lock
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        image_temp
            .persist(&image_path)
            .map_err(|error| error.error.to_string())?;
        metadata_temp
            .persist(metadata_path)
            .map_err(|error| error.error.to_string())?;
        let _ = fs::write(
            self.root.join(format!("{key}.access")),
            unix_timestamp_nanos().to_string(),
        );
        drop(_guard);
        if self.stores_since_cleanup.fetch_add(1, Ordering::Relaxed) + 1 >= CLEANUP_INTERVAL {
            self.stores_since_cleanup.store(0, Ordering::Relaxed);
            let _ = self.cleanup_lru();
        }
        Ok(())
    }

    fn cleanup_lru(&self) -> Result<(), String> {
        let _guard = self
            .lock
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut entries = Vec::new();
        let mut total = 0_u64;
        for entry in fs::read_dir(&self.root).map_err(|error| error.to_string())? {
            let entry = entry.map_err(|error| error.to_string())?;
            let path = entry.path();
            if path
                .extension()
                .is_none_or(|extension| extension != "png" && extension != "qoi")
            {
                continue;
            }
            let image_size = entry.metadata().map_err(|error| error.to_string())?.len();
            let key = path.file_stem().unwrap_or_default().to_string_lossy();
            let metadata_size = fs::metadata(self.root.join(format!("{key}.json")))
                .map(|metadata| metadata.len())
                .unwrap_or(0);
            let access_size = fs::metadata(self.root.join(format!("{key}.access")))
                .map(|metadata| metadata.len())
                .unwrap_or(0);
            let size = image_size
                .saturating_add(metadata_size)
                .saturating_add(access_size);
            let access = fs::read_to_string(self.root.join(format!("{key}.access")))
                .ok()
                .and_then(|value| value.parse::<u128>().ok())
                .unwrap_or(0);
            total = total.saturating_add(size);
            entries.push((access, size, key.into_owned()));
        }
        drop(_guard);
        entries.sort_by_key(|(access, _, _)| *access);
        let _guard = self
            .lock
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        for (access, size, key) in entries {
            if total <= self.max_bytes {
                break;
            }
            let current_access = fs::read_to_string(self.root.join(format!("{key}.access")))
                .ok()
                .and_then(|value| value.parse::<u128>().ok())
                .unwrap_or(0);
            if current_access != access {
                continue;
            }
            let _ = fs::remove_file(self.root.join(format!("{key}.png")));
            let _ = fs::remove_file(self.root.join(format!("{key}.qoi")));
            let _ = fs::remove_file(self.root.join(format!("{key}.json")));
            let _ = fs::remove_file(self.root.join(format!("{key}.access")));
            total = total.saturating_sub(size);
        }
        Ok(())
    }
}

fn cache_extension(variant: RenderVariant) -> &'static str {
    match variant {
        RenderVariant::Preview { .. } => "qoi",
        RenderVariant::Thumbnail { .. } | RenderVariant::Full => "png",
    }
}

fn cache_format(variant: RenderVariant) -> ImageFormat {
    match variant {
        RenderVariant::Preview { .. } => ImageFormat::Qoi,
        RenderVariant::Thumbnail { .. } | RenderVariant::Full => ImageFormat::Png,
    }
}

fn unix_timestamp_nanos() -> u128 {
    std::time::SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
}

pub(super) fn source_identity(request: &RenderRequest) -> Result<Option<SourceIdentity>, String> {
    let source = match fs::metadata(&request.path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.to_string()),
    };
    let modified = source
        .modified()
        .ok()
        .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
        .unwrap_or_default();
    Ok(Some(SourceIdentity {
        length: source.len(),
        modified_seconds: modified.as_secs(),
        modified_nanos: modified.subsec_nanos(),
    }))
}

fn cache_key(
    request: &RenderRequest,
    identity: &SourceIdentity,
    settings: &PhotographicSettings,
) -> Result<String, String> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(CACHE_VERSION.as_bytes());
    hasher.update(request.path.as_os_str().as_encoded_bytes());
    hasher.update(&identity.length.to_le_bytes());
    hasher.update(&identity.modified_seconds.to_le_bytes());
    hasher.update(&identity.modified_nanos.to_le_bytes());
    hasher.update(
        &serde_json::to_vec(&(request.variant, settings)).map_err(|error| error.to_string())?,
    );
    Ok(hasher.finalize().to_hex().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgb};

    #[test]
    fn stores_and_loads_an_image_with_its_render_metadata() {
        let directory = tempfile::tempdir().unwrap();
        let source = directory.path().join("source.jpg");
        fs::write(&source, b"source identity").unwrap();
        let cache = DiskRenderCache::new(directory.path().join("cache"));
        let request = RenderRequest::new(
            source,
            super::super::RenderVariant::Thumbnail { width: 300 },
        );
        let rendered = RenderedImage {
            image: DynamicImage::ImageRgb8(ImageBuffer::from_pixel(2, 1, Rgb([1, 2, 3]))),
            metadata: RenderMetadata {
                source_pipeline: super::super::SourcePipeline::Standard,
                photographic: None,
                source_dimensions: (2, 1),
                output_dimensions: (2, 1),
                variant: request.variant,
                orientation_applied: None,
            },
        };
        let settings = PhotographicSettings::default();
        let identity = source_identity(&request).unwrap().unwrap();
        cache
            .store(&request, &identity, &settings, &rendered)
            .unwrap();
        let loaded = cache.load(&request, &identity, &settings).unwrap().unwrap();
        assert_eq!(loaded.image.width(), 2);
        assert_eq!(loaded.metadata, rendered.metadata);
    }

    #[test]
    fn cleanup_removes_the_least_recently_accessed_entry() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().join("cache");
        let cache = DiskRenderCache::new(root.clone());
        let settings = PhotographicSettings::default();
        let rendered = RenderedImage {
            image: DynamicImage::ImageRgb8(ImageBuffer::from_pixel(8, 8, Rgb([1, 2, 3]))),
            metadata: RenderMetadata {
                source_pipeline: super::super::SourcePipeline::Standard,
                photographic: None,
                source_dimensions: (8, 8),
                output_dimensions: (8, 8),
                variant: super::super::RenderVariant::Thumbnail { width: 300 },
                orientation_applied: None,
            },
        };
        let first = directory.path().join("first.jpg");
        let second = directory.path().join("second.jpg");
        fs::write(&first, b"first").unwrap();
        fs::write(&second, b"second").unwrap();
        let first_request = RenderRequest::new(first, rendered.metadata.variant);
        let second_request = RenderRequest::new(second, rendered.metadata.variant);
        let first_identity = source_identity(&first_request).unwrap().unwrap();
        let second_identity = source_identity(&second_request).unwrap().unwrap();
        cache
            .store(&first_request, &first_identity, &settings, &rendered)
            .unwrap();
        std::thread::sleep(std::time::Duration::from_millis(2));
        cache
            .store(&second_request, &second_identity, &settings, &rendered)
            .unwrap();
        std::thread::sleep(std::time::Duration::from_millis(2));
        cache
            .load(&first_request, &first_identity, &settings)
            .unwrap()
            .unwrap();
        let first_key = cache_key(&first_request, &first_identity, &settings).unwrap();
        let second_key = cache_key(&second_request, &second_identity, &settings).unwrap();
        let entry_size = |key: &str| {
            ["png", "json", "access"]
                .into_iter()
                .map(|extension| {
                    fs::metadata(root.join(format!("{key}.{extension}")))
                        .unwrap()
                        .len()
                })
                .sum::<u64>()
        };
        let first_size = entry_size(&first_key);
        let second_size = entry_size(&second_key);
        let constrained = DiskRenderCache {
            root: root.clone(),
            max_bytes: first_size + second_size - 1,
            lock: cache.lock.clone(),
            stores_since_cleanup: Arc::new(AtomicU64::new(0)),
        };
        constrained.cleanup_lru().unwrap();

        assert!(root.join(format!("{first_key}.png")).exists());
        assert!(!root.join(format!("{second_key}.png")).exists());
    }
}
