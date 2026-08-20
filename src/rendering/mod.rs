//! Blocking image rendering backend.
//!
//! The backend has no UI or cache knowledge. Requests and settings are value
//! types so they can later become stable memory/disk-cache keys, while callers
//! remain free to schedule work on any executor.

mod photographic;
mod source;

use std::path::PathBuf;

use image::DynamicImage;

#[allow(unused_imports)]
pub use photographic::{AutoExposure, PhotographicSettings, ToneMapper};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RenderRequest {
    pub path: PathBuf,
    pub thumbnail_width: Option<u32>,
}

impl RenderRequest {
    pub fn new(path: PathBuf, thumbnail_width: Option<u32>) -> Self {
        Self {
            path,
            thumbnail_width,
        }
    }

    pub fn variant(&self) -> RenderVariant {
        match self.thumbnail_width {
            Some(width) => RenderVariant::Thumbnail { width },
            None => RenderVariant::Full,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RenderVariant {
    Full,
    Thumbnail { width: u32 },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SourcePipeline {
    Standard,
    Heif,
    Raw,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OutputEncoding {
    Srgb,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AppliedPhotographicTransform {
    pub settings: PhotographicSettings,
    pub automatic_exposure_stops: f32,
    pub total_exposure_stops: f32,
    pub output_encoding: OutputEncoding,
    pub orientation_applied: Option<u8>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
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
}

impl Default for RenderBackend {
    fn default() -> Self {
        Self {
            raw: PhotographicSettings::default(),
        }
    }
}

impl RenderBackend {
    /// Performs file I/O and CPU-heavy decoding. Run this on a background worker.
    pub fn render(&self, request: &RenderRequest) -> image::ImageResult<RenderedImage> {
        let decoded = source::decode(&request.path, &self.raw)?;
        let source_dimensions = (decoded.image.width(), decoded.image.height());
        let image = match request.thumbnail_width {
            Some(width) => decoded.image.thumbnail(width, width),
            None => decoded.image,
        };
        let output_dimensions = (image.width(), image.height());
        Ok(RenderedImage {
            image,
            metadata: RenderMetadata {
                source_pipeline: decoded.source_pipeline,
                photographic: decoded.photographic,
                source_dimensions,
                output_dimensions,
                variant: request.variant(),
                orientation_applied: decoded.orientation_applied,
            },
        })
    }
}
