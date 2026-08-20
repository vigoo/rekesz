//! Blocking image rendering backend.
//!
//! The backend has no UI or cache knowledge. Requests and settings are value
//! types so they can later become stable memory/disk-cache keys, while callers
//! remain free to schedule work on any executor.

mod disk_cache;
mod photographic;
mod source;

use std::path::PathBuf;

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

pub struct RenderedImage {
    pub image: DynamicImage,
    pub metadata: RenderMetadata,
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

    /// Performs file I/O and CPU-heavy decoding. Run this on a background worker.
    pub fn render(&self, request: &RenderRequest) -> image::ImageResult<RenderedImage> {
        if request.variant != RenderVariant::Full
            && let Ok(Some(cached)) = self.disk_cache.load(request, &self.raw)
        {
            return Ok(cached);
        }
        let decoded = source::decode(&request.path, &self.raw)?;
        let source_dimensions = (decoded.image.width(), decoded.image.height());
        let image = match request.variant {
            RenderVariant::Full => decoded.image,
            RenderVariant::Thumbnail { width } => decoded.image.thumbnail(width, width),
            RenderVariant::Preview {
                max_width,
                max_height,
            } => decoded.image.thumbnail(max_width, max_height),
        };
        let output_dimensions = (image.width(), image.height());
        let rendered = RenderedImage {
            image,
            metadata: RenderMetadata {
                source_pipeline: decoded.source_pipeline,
                photographic: decoded.photographic,
                source_dimensions,
                output_dimensions,
                variant: request.variant,
                orientation_applied: decoded.orientation_applied,
            },
        };
        if request.variant != RenderVariant::Full {
            // A cache write failure must not make an otherwise valid image
            // unavailable to the UI.
            let _ = self.disk_cache.store(request, &self.raw, &rendered);
        }
        Ok(rendered)
    }
}
