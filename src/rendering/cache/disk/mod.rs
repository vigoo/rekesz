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

use crate::rendering::{
    PhotographicSettings, RenderMetadata, RenderRequest, RenderVariant, RenderedImage,
    SourceIdentity,
};

const CACHE_VERSION: &str = "render-v2";
const DEFAULT_CACHE_BYTES: u64 = 10 * 1024 * 1024 * 1024;
const CLEANUP_INTERVAL: u64 = 32;

#[derive(Clone, Debug)]
pub(in crate::rendering) struct DiskRenderCache {
    root: PathBuf,
    max_bytes: u64,
    lock: Arc<RwLock<()>>,
    stores_since_cleanup: Arc<AtomicU64>,
}

impl DiskRenderCache {
    pub(in crate::rendering) fn new(root: PathBuf) -> Self {
        Self {
            root,
            max_bytes: DEFAULT_CACHE_BYTES,
            lock: Arc::new(RwLock::new(())),
            stores_since_cleanup: Arc::new(AtomicU64::new(0)),
        }
    }

    pub(in crate::rendering) fn load(
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

    pub(in crate::rendering) fn store(
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

pub(in crate::rendering) fn source_identity(
    request: &RenderRequest,
) -> Result<Option<SourceIdentity>, String> {
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
        &serde_json::to_vec(&(request.variant, request.identity, settings))
            .map_err(|error| error.to_string())?,
    );
    Ok(hasher.finalize().to_hex().to_string())
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
