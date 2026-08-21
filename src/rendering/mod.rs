//! Blocking image rendering backend.
//!
//! The backend has no UI or cache knowledge. Requests and settings are value
//! types so they can later become stable memory/disk-cache keys, while callers
//! remain free to schedule work on any executor.

mod disk_cache;
mod photographic;
pub(crate) mod source;

use std::{
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use image::DynamicImage;
use serde::{Deserialize, Serialize};

#[allow(unused_imports)]
pub use photographic::{AutoExposure, PhotographicSettings, ToneMapper};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RenderRequest {
    pub path: PathBuf,
    pub variant: RenderVariant,
}

impl RenderRequest {
    pub fn new(path: PathBuf, variant: RenderVariant) -> Self {
        Self { path, variant }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RenderVariant {
    Full,
    Thumbnail { width: u32 },
    Preview { max_width: u32, max_height: u32 },
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum SourcePipeline {
    Standard,
    Heif,
    Raw,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum OutputEncoding {
    Srgb,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct AppliedPhotographicTransform {
    pub settings: PhotographicSettings,
    pub automatic_exposure_stops: f32,
    pub total_exposure_stops: f32,
    pub output_encoding: OutputEncoding,
    pub orientation_applied: Option<u8>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RenderMetadata {
    pub source_pipeline: SourcePipeline,
    pub photographic: Option<AppliedPhotographicTransform>,
    pub source_dimensions: (u32, u32),
    pub output_dimensions: (u32, u32),
    pub variant: RenderVariant,
    /// EXIF orientation value applied before resizing, when known.
    pub orientation_applied: Option<u8>,
}

#[derive(Clone)]
pub struct RenderedImage {
    pub image: DynamicImage,
    pub metadata: RenderMetadata,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct SourceIdentity {
    length: u64,
    modified_seconds: u64,
    modified_nanos: u32,
}

#[derive(Clone, Debug)]
pub struct RenderBackend {
    pub raw: PhotographicSettings,
    disk_cache: disk_cache::DiskRenderCache,
}

impl RenderBackend {
    pub fn new(cache_root: PathBuf) -> Self {
        Self {
            raw: PhotographicSettings::default(),
            disk_cache: disk_cache::DiskRenderCache::new(cache_root),
        }
    }

    /// Filesystem cache lookup stage. This performs blocking metadata and file I/O.
    pub fn source_identity(
        &self,
        request: &RenderRequest,
    ) -> Result<Option<SourceIdentity>, String> {
        disk_cache::source_identity(request)
    }

    pub fn lookup_cache(
        &self,
        request: &RenderRequest,
        identity: Option<&SourceIdentity>,
    ) -> Result<Option<RenderedImage>, String> {
        if request.variant == RenderVariant::Full {
            return Ok(None);
        }
        let Some(identity) = identity else {
            return Ok(None);
        };
        self.disk_cache.load(request, identity, &self.raw)
    }

    /// Source decode stage. This may be CPU and memory intensive.
    pub(crate) fn decode_source(
        &self,
        request: &RenderRequest,
        cancelled: &Arc<AtomicBool>,
    ) -> image::ImageResult<Arc<source::DecodedSource>> {
        source::decode(&request.path, &self.raw, request.variant, cancelled).map(Arc::new)
    }

    pub(crate) fn render_source(
        &self,
        variant: RenderVariant,
        decoded: Arc<source::DecodedSource>,
        cancelled: &Arc<AtomicBool>,
    ) -> image::ImageResult<RenderedImage> {
        if cancelled.load(Ordering::Acquire) {
            return Err(image::ImageError::Decoding(
                image::error::DecodingError::new(
                    image::error::ImageFormatHint::Unknown,
                    "render cancelled",
                ),
            ));
        }
        let source_dimensions = decoded.source_dimensions;
        let image = match variant {
            RenderVariant::Full => decoded.image.clone(),
            RenderVariant::Thumbnail { width } => decoded.image.thumbnail(width, width),
            RenderVariant::Preview {
                max_width,
                max_height,
            } => decoded.image.thumbnail(max_width, max_height),
        };
        if cancelled.load(Ordering::Acquire) {
            return Err(image::ImageError::Decoding(
                image::error::DecodingError::new(
                    image::error::ImageFormatHint::Unknown,
                    "render cancelled",
                ),
            ));
        }
        let output_dimensions = (image.width(), image.height());
        Ok(RenderedImage {
            image,
            metadata: RenderMetadata {
                source_pipeline: decoded.source_pipeline,
                photographic: decoded.photographic,
                source_dimensions,
                output_dimensions,
                variant,
                orientation_applied: decoded.orientation_applied,
            },
        })
    }

    /// Persistent cache-write stage. Callers may defer this after publishing pixels.
    pub fn store_cache(
        &self,
        request: &RenderRequest,
        identity: &SourceIdentity,
        rendered: &RenderedImage,
    ) -> Result<bool, String> {
        if request.variant == RenderVariant::Full {
            return Ok(false);
        }
        if self.source_identity(request)?.as_ref() != Some(identity) {
            return Ok(false);
        }
        self.disk_cache
            .store(request, identity, &self.raw, rendered)
            .map(|()| true)
    }
}

#[cfg(test)]
mod tests {
    use std::{fs, thread, time::Duration};

    use image::{ImageBuffer, Rgb};

    use super::*;

    #[test]
    fn stale_render_is_not_stored_under_a_changed_source_identity() {
        let directory = tempfile::tempdir().unwrap();
        let source = directory.path().join("source.jpg");
        fs::write(&source, b"old source").unwrap();
        let backend = RenderBackend::new(directory.path().join("cache"));
        let request = RenderRequest::new(source.clone(), RenderVariant::Thumbnail { width: 300 });
        let identity = backend.source_identity(&request).unwrap().unwrap();
        let rendered = RenderedImage {
            image: DynamicImage::ImageRgb8(ImageBuffer::from_pixel(1, 1, Rgb([1, 2, 3]))),
            metadata: RenderMetadata {
                source_pipeline: SourcePipeline::Standard,
                photographic: None,
                source_dimensions: (1, 1),
                output_dimensions: (1, 1),
                variant: request.variant,
                orientation_applied: None,
            },
        };
        thread::sleep(Duration::from_millis(2));
        fs::write(&source, b"new source with a different length").unwrap();

        backend.store_cache(&request, &identity, &rendered).unwrap();
        let current = backend.source_identity(&request).unwrap().unwrap();
        assert_ne!(identity, current);
        assert!(
            backend
                .lookup_cache(&request, Some(&current))
                .unwrap()
                .is_none()
        );
    }
}
