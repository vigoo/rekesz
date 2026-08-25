//! Blocking image rendering backend.
//!
//! The backend has no UI or cache knowledge. Requests and settings are value
//! types so they can later become stable memory/disk-cache keys, while callers
//! remain free to schedule work on any executor.

pub(crate) mod develop;
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
pub struct RenderRequest {
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
        source::decode(
            &request.path,
            &self.raw,
            request.variant,
            request.identity.mode == RenderMode::Developed,
            cancelled,
        )
        .map(Arc::new)
    }

    pub(crate) fn render_source(
        &self,
        variant: RenderVariant,
        request_identity: RenderIdentity,
        recipe: Option<&DevelopRecipe>,
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
        let (source_image, photographic, raw_preapplied, raw_fidelity) = match &decoded.raw_linear {
            Some(raw) => {
                let (image, metadata) = photographic::render_raw(
                    raw,
                    &self.raw,
                    (request_identity.mode == RenderMode::Developed)
                        .then_some(recipe)
                        .flatten(),
                    cancelled,
                )
                .map_err(|error| {
                    image::ImageError::Decoding(image::error::DecodingError::new(
                        image::error::ImageFormatHint::Unknown,
                        error,
                    ))
                })?;
                let actions = recipe
                    .filter(|_| request_identity.mode == RenderMode::Developed)
                    .map(|recipe| {
                        let mut actions = 0;
                        if recipe.exposure_ev.is_some() {
                            actions |= develop::actions::EXPOSURE;
                        }
                        if recipe.white_balance.is_some()
                            || recipe.white_balance_increment.is_some()
                        {
                            actions |= develop::actions::WHITE_BALANCE;
                        }
                        if recipe.contrast.is_some()
                            || recipe.highlights.is_some()
                            || recipe.shadows.is_some()
                            || recipe.whites.is_some()
                            || recipe.blacks.is_some()
                        {
                            actions |= develop::actions::BASIC_TONE;
                        }
                        if recipe.saturation.is_some() || recipe.vibrance.is_some() {
                            actions |= develop::actions::COLOR;
                        }
                        if recipe.tone_curve.is_some()
                            || recipe.tone_curve_red.is_some()
                            || recipe.tone_curve_green.is_some()
                            || recipe.tone_curve_blue.is_some()
                        {
                            actions |= develop::actions::TONE_CURVE;
                        }
                        if recipe.detail.is_some() {
                            actions |= develop::actions::DETAIL;
                        }
                        if recipe.texture.is_some() || recipe.dehaze.is_some() {
                            actions |= develop::actions::TEXTURE_DEHAZE;
                        }
                        if recipe.monochrome.is_some() {
                            actions |= develop::actions::MONOCHROME;
                        }
                        if recipe.parametric_curve.is_some() {
                            actions |= develop::actions::PARAMETRIC_CURVE;
                        }
                        actions
                    })
                    .unwrap_or(0);
                let approximate = recipe
                    .filter(|_| request_identity.mode == RenderMode::Developed)
                    .is_some_and(|recipe| {
                        recipe.white_balance.is_some()
                            || recipe.white_balance_increment.is_some()
                            || recipe.contrast.is_some()
                            || recipe.highlights.is_some()
                            || recipe.shadows.is_some()
                            || recipe.whites.is_some()
                            || recipe.blacks.is_some()
                            || recipe.saturation.is_some()
                            || recipe.vibrance.is_some()
                            || recipe.tone_curve.is_some()
                            || recipe.tone_curve_red.is_some()
                            || recipe.tone_curve_green.is_some()
                            || recipe.tone_curve_blue.is_some()
                            || recipe.detail.is_some()
                            || recipe.texture.is_some()
                            || recipe.dehaze.is_some()
                            || recipe.monochrome.is_some()
                            || recipe.parametric_curve.is_some()
                    });
                (
                    image,
                    Some(metadata),
                    actions,
                    if approximate {
                        develop::warnings::APPROXIMATED
                    } else {
                        0
                    },
                )
            }
            None => (decoded.image.clone(), decoded.photographic, 0, 0),
        };
        let source_image = match (variant, recipe) {
            (RenderVariant::Thumbnail { width }, None) => source_image.thumbnail(width, width),
            (
                RenderVariant::Preview {
                    max_width,
                    max_height,
                },
                None,
            ) => source_image.thumbnail(max_width, max_height),
            _ => source_image,
        };
        let post_recipe = recipe.map(|recipe| {
            let mut recipe = recipe.clone();
            if decoded.raw_linear.is_some() {
                recipe.exposure_ev = None;
                recipe.white_balance = None;
                recipe.white_balance_increment = None;
                recipe.contrast = None;
                recipe.highlights = None;
                recipe.shadows = None;
                recipe.whites = None;
                recipe.blacks = None;
                recipe.saturation = None;
                recipe.vibrance = None;
                recipe.tone_curve = None;
                recipe.tone_curve_red = None;
                recipe.tone_curve_green = None;
                recipe.tone_curve_blue = None;
                recipe.detail = None;
                recipe.texture = None;
                recipe.dehaze = None;
                recipe.monochrome = None;
                recipe.parametric_curve = None;
            }
            recipe
        });
        let (developed, mut applied_actions, mut fidelity_warnings) = match post_recipe.as_ref() {
            Some(recipe) if request_identity.mode == RenderMode::Developed => {
                develop::apply(&source_image, recipe, cancelled)?
            }
            _ => (source_image, 0, 0),
        };
        applied_actions |= raw_preapplied;
        fidelity_warnings |= raw_fidelity;
        let image = match variant {
            RenderVariant::Full => developed,
            RenderVariant::Thumbnail { width } => developed.thumbnail(width, width),
            RenderVariant::Preview {
                max_width,
                max_height,
            } => developed.thumbnail(max_width, max_height),
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
                photographic,
                source_dimensions,
                output_dimensions,
                variant,
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
#[path = "tests.rs"]
mod tests;
