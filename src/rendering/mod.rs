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
mod filters;
mod photographic;

/// Test access to the packaged lens profile (the module is private).
#[cfg(test)]
pub(crate) fn embedded_lens_profile_for_tests() -> Option<&'static photographic::lens::LensProfile>
{
    photographic::embedded_lens_profile()
}
/// Scheduling and resource-management entry point for blocking renders.
pub(crate) mod pipeline;
mod plan;
mod presence;
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
        "this Rekesz build has no HEVC decoder registered with libheif; rebuild it with an embedded libde265 decoder"
    );
    Ok(())
}

/// Identifies the decode + face-analysis pipeline for analysis staleness.
/// Baseline pixels are unaffected by develop-operator changes, so this only
/// moves when decoding itself changes; render-cache identity is governed by
/// `RENDERER_MODEL_VERSION` in the catalog instead.
pub(crate) const DECODER_MODEL_ID: &str = "renderer-6";
static ANALYSIS_MEMORY_BUDGET: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(768 * 1024 * 1024);

pub(crate) fn configure_analysis_memory_budget(bytes: u64) {
    ANALYSIS_MEMORY_BUDGET.store(bytes.max(64 * 1024 * 1024), Ordering::Release);
}

pub(crate) fn analysis_memory_budget() -> usize {
    ANALYSIS_MEMORY_BUDGET
        .load(Ordering::Acquire)
        .min(usize::MAX as u64) as usize
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RenderIdentity {
    pub mode: RenderMode,
    pub recipe_digest: [u8; 32],
    pub renderer_model: u32,
}

impl Default for RenderIdentity {
    fn default() -> Self {
        // The default identity (baseline renders: grid thumbnails, previews)
        // must carry the current renderer model version — a bump has to
        // invalidate baseline caches too, not just developed ones.
        Self {
            mode: RenderMode::Baseline,
            recipe_digest: [0; 32],
            renderer_model: crate::catalog::RENDERER_MODEL_VERSION,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RenderVariant {
    Full,
    Thumbnail {
        width: u32,
    },
    Preview {
        max_width: u32,
        max_height: u32,
    },
    FaceCrop {
        face_id: i64,
        face_revision: u64,
        left_millionths: u32,
        top_millionths: u32,
        right_millionths: u32,
        bottom_millionths: u32,
        width: u32,
        padding_percent: u8,
    },
    FaceContext {
        width: u32,
    },
}

impl RenderVariant {
    pub(crate) fn face_crop(face_id: i64, bounds: [f32; 4], width: u32) -> Self {
        let quantize = |value: f32| (value.clamp(0.0, 1.0) * 1_000_000.0).round() as u32;
        Self::FaceCrop {
            face_id,
            face_revision: face_revision(face_id, bounds),
            left_millionths: quantize(bounds[0]),
            top_millionths: quantize(bounds[1]),
            right_millionths: quantize(bounds[2]),
            bottom_millionths: quantize(bounds[3]),
            width,
            padding_percent: 35,
        }
    }

    pub(crate) fn face_context(width: u32) -> Self {
        Self::FaceContext { width }
    }
}

fn face_revision(face_id: i64, bounds: [f32; 4]) -> u64 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(&face_id.to_le_bytes());
    for value in bounds {
        hasher.update(&value.to_bits().to_le_bytes());
    }
    u64::from_le_bytes(hasher.finalize().as_bytes()[..8].try_into().unwrap())
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
    /// Display-space (orientation-applied) dimensions of the decoded source.
    pub source_dimensions: (u32, u32),
    pub output_dimensions: (u32, u32),
    pub variant: RenderVariant,
    /// EXIF orientation value applied before resizing, when known.
    pub orientation_applied: Option<u8>,
    /// Canonical source-space to decoded display-space mapping. Older cache
    /// entries omit this and fall back to `orientation_applied`.
    #[serde(default)]
    pub face_transform: Option<FaceTransform>,
    pub develop: Option<DevelopRenderMetadata>,
}

impl RenderMetadata {
    /// Source dimensions with the orientation applied, for display. Decoders
    /// record these directly. Disk-cache entries written before the scaled-JPEG
    /// decoder was corrected still carry pre-rotation sensor dimensions; their
    /// output aspect ratio — which undeveloped resize variants preserve —
    /// exposes the swap. Developed renders never went through the scaled
    /// decoder and may legitimately change the aspect ratio (recipes crop), so
    /// they are never reconciled.
    pub fn display_source_dimensions(&self) -> (u32, u32) {
        let (width, height) = self.source_dimensions;
        let transposed = matches!(self.orientation_applied, Some(5..=8));
        let resize_variant = matches!(
            self.variant,
            RenderVariant::Thumbnail { .. } | RenderVariant::Preview { .. }
        );
        let (output_width, output_height) = self.output_dimensions;
        let source_aspect = f64::from(width) / f64::from(height.max(1));
        let output_aspect = f64::from(output_width) / f64::from(output_height.max(1));
        let swapped = transposed
            && resize_variant
            && self.develop.is_none()
            && (source_aspect - output_aspect).abs() / source_aspect.max(output_aspect) > 0.05;
        if swapped {
            (height, width)
        } else {
            (width, height)
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DevelopRenderMetadata {
    pub identity: RenderIdentity,
    /// Exact normalized controls used to produce this cached derivative.
    pub recipe: DevelopRecipe,
    /// Stable bitset of normalized Rekesz actions applied by this renderer model.
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
        if matches!(request.variant, RenderVariant::Full) {
            return Ok(None);
        }
        let Some(identity) = identity else {
            return Ok(None);
        };
        self.disk_cache.load(request, identity, &self.raw)
    }

    fn disk_cache_usage(&self) -> Result<cache::DiskCacheUsage, String> {
        self.disk_cache.usage()
    }

    fn clear_disk_cache(&self) -> Result<(), String> {
        self.disk_cache.clear()
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
                let (image, metadata) = photographic::render_raw(
                    raw,
                    &self.raw,
                    plan.raw_recipe(),
                    plan.lens_correction(),
                    cancelled,
                )
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
        let developed =
            apply_face_variant(developed, plan.request().variant, decoded.face_transform);
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
                face_transform: Some(decoded.face_transform),
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
        if matches!(request.variant, RenderVariant::Full) {
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

fn apply_face_variant(
    image: DynamicImage,
    variant: RenderVariant,
    face_transform: FaceTransform,
) -> DynamicImage {
    let RenderVariant::FaceCrop {
        left_millionths,
        top_millionths,
        right_millionths,
        bottom_millionths,
        width,
        padding_percent,
        ..
    } = variant
    else {
        return image;
    };
    let scale = 1_000_000.0_f32;
    let bounds = face_transform.bounds([
        left_millionths as f32 / scale,
        top_millionths as f32 / scale,
        right_millionths as f32 / scale,
        bottom_millionths as f32 / scale,
    ]);
    let image_width = image.width() as f32;
    let image_height = image.height() as f32;
    let face_width = (bounds[2] - bounds[0]) * image_width;
    let face_height = (bounds[3] - bounds[1]) * image_height;
    let padding = f32::from(padding_percent) / 100.0;
    let crop_width = face_width * (1.0 + padding * 2.0);
    let crop_height = face_height * (1.0 + padding * 2.0);
    let side = crop_width
        .max(crop_height)
        .clamp(1.0, image_width.min(image_height));
    let center_x = (bounds[0] + bounds[2]) * 0.5 * image_width;
    let center_y = (bounds[1] + bounds[3]) * 0.5 * image_height;
    let pixel_left = (center_x - side * 0.5)
        .clamp(0.0, image_width - side)
        .floor() as u32;
    let pixel_top = (center_y - side * 0.5)
        .clamp(0.0, image_height - side)
        .floor() as u32;
    let pixel_width = side.ceil() as u32;
    let pixel_height = side.ceil() as u32;
    image
        .crop_imm(
            pixel_left.min(image.width().saturating_sub(1)),
            pixel_top.min(image.height().saturating_sub(1)),
            pixel_width
                .min(image.width().saturating_sub(pixel_left))
                .max(1),
            pixel_height
                .min(image.height().saturating_sub(pixel_top))
                .max(1),
        )
        .thumbnail(width, width)
}

/// Affine mapping from canonical, unoriented source coordinates to the pixels
/// emitted by the decoder. Coordinates are normalized at both ends, allowing
/// the mapping to include orientation and HEIF clean-aperture crops.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct FaceTransform([f32; 6]);

impl FaceTransform {
    pub(crate) const IDENTITY: Self = Self([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);

    pub(crate) fn orientation(orientation: u8) -> Self {
        let (x0, y0) = oriented_point(0.0, 0.0, orientation);
        let (x1, y1) = oriented_point(1.0, 0.0, orientation);
        let (x2, y2) = oriented_point(0.0, 1.0, orientation);
        Self([x1 - x0, x2 - x0, x0, y1 - y0, y2 - y0, y0])
    }

    pub(crate) fn crop(left: f32, top: f32, right: f32, bottom: f32) -> Self {
        let width = (1.0 - left - right).max(f32::EPSILON);
        let height = (1.0 - top - bottom).max(f32::EPSILON);
        Self([
            1.0 / width,
            0.0,
            -left / width,
            0.0,
            1.0 / height,
            -top / height,
        ])
    }

    /// Returns `next(self(point))`.
    pub(crate) fn then(self, next: Self) -> Self {
        let [a, b, c, d, e, f] = self.0;
        let [g, h, i, j, k, l] = next.0;
        Self([
            g * a + h * d,
            g * b + h * e,
            g * c + h * f + i,
            j * a + k * d,
            j * b + k * e,
            j * c + k * f + l,
        ])
    }

    fn point(self, x: f32, y: f32) -> (f32, f32) {
        let [a, b, c, d, e, f] = self.0;
        (a * x + b * y + c, d * x + e * y + f)
    }

    pub(crate) fn inverse(self) -> Option<Self> {
        let [a, b, c, d, e, f] = self.0;
        let determinant = a * e - b * d;
        if !determinant.is_finite() || determinant.abs() <= f32::EPSILON {
            return None;
        }
        Some(Self([
            e / determinant,
            -b / determinant,
            (b * f - e * c) / determinant,
            -d / determinant,
            a / determinant,
            (d * c - a * f) / determinant,
        ]))
    }

    pub(crate) fn mapped_point(self, point: [f32; 2]) -> [f32; 2] {
        let (x, y) = self.point(point[0], point[1]);
        [x, y]
    }

    pub fn bounds(self, bounds: [f32; 4]) -> [f32; 4] {
        let points = [
            self.point(bounds[0], bounds[1]),
            self.point(bounds[2], bounds[1]),
            self.point(bounds[0], bounds[3]),
            self.point(bounds[2], bounds[3]),
        ];
        let mut output = [
            f32::INFINITY,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
        ];
        for (x, y) in points {
            output[0] = output[0].min(x);
            output[1] = output[1].min(y);
            output[2] = output[2].max(x);
            output[3] = output[3].max(y);
        }
        output
    }
}

pub(crate) fn oriented_face_bounds(bounds: [f32; 4], orientation: u8) -> [f32; 4] {
    let transform = |x: f32, y: f32| oriented_point(x, y, orientation);
    let corners = [
        transform(bounds[0], bounds[1]),
        transform(bounds[2], bounds[1]),
        transform(bounds[2], bounds[3]),
        transform(bounds[0], bounds[3]),
    ];
    corners.iter().fold(
        [1.0_f32, 1.0_f32, 0.0_f32, 0.0_f32],
        |mut result, (x, y)| {
            result[0] = result[0].min(*x);
            result[1] = result[1].min(*y);
            result[2] = result[2].max(*x);
            result[3] = result[3].max(*y);
            result
        },
    )
}

fn oriented_point(x: f32, y: f32, orientation: u8) -> (f32, f32) {
    match orientation {
        2 => (1.0 - x, y),
        3 => (1.0 - x, 1.0 - y),
        4 => (x, 1.0 - y),
        5 => (y, x),
        6 => (1.0 - y, x),
        7 => (1.0 - y, 1.0 - x),
        8 => (y, 1.0 - x),
        _ => (x, y),
    }
}

pub(crate) fn compose_orientations(first: u8, second: u8) -> u8 {
    let samples = [(0.2_f32, 0.3_f32), (0.7, 0.4)];
    (1..=8)
        .find(|candidate| {
            samples.iter().all(|(x, y)| {
                let after_first = oriented_point(*x, *y, first);
                let expected = oriented_point(after_first.0, after_first.1, second);
                let actual = oriented_point(*x, *y, *candidate);
                (actual.0 - expected.0).abs() < 1e-6 && (actual.1 - expected.1).abs() < 1e-6
            })
        })
        .unwrap_or(1)
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
