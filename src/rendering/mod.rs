//! Rendering core and blocking pipeline.
//!
//! UI code enters through [`pipeline::RenderPipeline`] with a
//! [`pipeline::RenderInvocation`]. The pipeline compiles one validated plan,
//! then drives cache lookup, decode, photographic RAW processing, develop
//! actions, resizing, and optional persistence in that order. Lower layers do
//! not depend on the pipeline or on preview/UI state; decoded data lives at a
//! neutral boundary shared by decode, cache, and photographic processing.

mod cache;
mod decode;
mod decoded;
mod develop;
mod photographic;
/// Scheduling and resource-management entry point for blocking renders.
pub(crate) mod pipeline;
mod plan;
mod resize;

use std::{
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use crate::catalog::DevelopRecipe;
use image::DynamicImage;
use serde::{Deserialize, Serialize};

#[allow(unused_imports)]
pub use photographic::{AutoExposure, PhotographicSettings, ToneMapper};

pub fn assert_hevc_decoder_available() -> anyhow::Result<()> {
    use libheif_rs::{CompressionFormat, LibHeif};

    let libheif = LibHeif::new_checked()
        .map_err(|error| anyhow::anyhow!("failed to initialize libheif: {error}"))?;
    let decoders = libheif.decoder_descriptors(1, Some(CompressionFormat::Hevc));
    anyhow::ensure!(
        !decoders.is_empty(),
        "this Pics build has no HEVC decoder registered with libheif; rebuild it with an embedded libde265 decoder"
    );
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct RenderRequest {
    pub path: PathBuf,
    pub variant: RenderVariant,
    pub identity: RenderIdentity,
}

impl RenderRequest {
    #[cfg(test)]
    pub fn new(path: PathBuf, variant: RenderVariant) -> Self {
        Self {
            path,
            variant,
            identity: RenderIdentity::default(),
        }
    }

    pub fn with_identity(path: PathBuf, variant: RenderVariant, identity: RenderIdentity) -> Self {
        Self {
            path,
            variant,
            identity,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RenderMode {
    #[default]
    Baseline,
    Developed,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RenderIdentity {
    pub mode: RenderMode,
    pub recipe_digest: [u8; 32],
    pub renderer_model: u32,
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

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RenderMetadata {
    pub source_pipeline: SourcePipeline,
    pub photographic: Option<AppliedPhotographicTransform>,
    pub source_dimensions: (u32, u32),
    pub output_dimensions: (u32, u32),
    pub variant: RenderVariant,
    /// EXIF orientation value applied before resizing, when known.
    pub orientation_applied: Option<u8>,
    pub develop: Option<DevelopRenderMetadata>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DevelopRenderMetadata {
    pub identity: RenderIdentity,
    /// Exact normalized controls used to produce this cached derivative.
    pub recipe: DevelopRecipe,
    /// Stable bitset of normalized Pics actions applied by this renderer model.
    pub applied_actions: u64,
    /// Stable bitset of approximated, clamped, or skipped recipe behavior.
    pub fidelity_warnings: u64,
}

#[derive(Clone)]
pub(crate) struct RenderedImage {
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
struct RenderBackend {
    pub raw: PhotographicSettings,
    disk_cache: cache::DiskRenderCache,
}

impl RenderBackend {
    fn new(cache_root: PathBuf) -> Self {
        Self {
            raw: PhotographicSettings::default(),
            disk_cache: cache::DiskRenderCache::new(cache_root),
        }
    }

    /// Filesystem cache lookup stage. This performs blocking metadata and file I/O.
    fn source_identity(&self, request: &RenderRequest) -> Result<Option<SourceIdentity>, String> {
        cache::source_identity(request)
    }

    fn lookup_cache(
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
        plan: &plan::RenderPlan,
        cancelled: &Arc<AtomicBool>,
    ) -> image::ImageResult<Arc<decoded::DecodedSource>> {
        decode::decode(plan, cancelled).map(Arc::new)
    }

    pub(crate) fn render_source(
        &self,
        plan: &plan::RenderPlan,
        decoded: Arc<decoded::DecodedSource>,
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
        let request_identity = plan.request().identity;
        let recipe = plan.recipe();
        let source_dimensions = decoded.source_dimensions;
        let (source_image, photographic, raw_preapplied, raw_fidelity) = match &decoded.raw_linear {
            Some(raw) => {
                let (image, metadata) =
                    photographic::render_raw(raw, &self.raw, plan.raw_recipe(), cancelled)
                        .map_err(|error| {
                            image::ImageError::Decoding(image::error::DecodingError::new(
                                image::error::ImageFormatHint::Unknown,
                                error,
                            ))
                        })?;
                let (actions, warnings) = plan.raw_metadata();
                (image, Some(metadata), actions, warnings)
            }
            None => (decoded.image.clone(), decoded.photographic, 0, 0),
        };
        let source_image = resize::before_develop(source_image, plan.resize());
        let (developed, mut applied_actions, mut fidelity_warnings) = match plan.post_recipe() {
            Some(recipe) if request_identity.mode == RenderMode::Developed => {
                develop::apply(&source_image, recipe, cancelled)?
            }
            _ => (source_image, 0, 0),
        };
        applied_actions |= raw_preapplied;
        fidelity_warnings |= raw_fidelity;
        let image = resize::output(developed, plan.resize());
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
                photographic,
                source_dimensions,
                output_dimensions,
                variant: plan.request().variant,
                orientation_applied: decoded.orientation_applied,
                develop: (request_identity.mode == RenderMode::Developed).then_some(
                    DevelopRenderMetadata {
                        identity: request_identity,
                        recipe: recipe.cloned().unwrap_or_default(),
                        applied_actions,
                        fidelity_warnings,
                    },
                ),
            },
        })
    }

    /// Persistent cache-write stage. Callers may defer this after publishing pixels.
    fn store_cache(
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
#[path = "tests.rs"]
mod tests;
