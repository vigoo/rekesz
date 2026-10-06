//! Camera-independent photographic rendering of scene-linear RGB.

use std::sync::atomic::{AtomicBool, Ordering};

use serde::{Deserialize, Serialize};

use super::{AppliedPhotographicTransform, OutputEncoding};

mod colour;
mod exposure;
mod raw_render;
mod rawler_stage;
mod scene_linear;
mod tone_mapping;
mod white_balance;

pub(crate) mod lens;
pub(super) use raw_render::render_raw;

/// The packaged lens profile, parsed once. Render caches are invalidated
/// via `RENDERER_MODEL_VERSION` when this content changes.
pub(crate) fn embedded_lens_profile() -> Option<&'static lens::LensProfile> {
    static PROFILE: std::sync::OnceLock<Option<lens::LensProfile>> = std::sync::OnceLock::new();
    PROFILE
        .get_or_init(|| {
            serde_json::from_str(include_str!("../../../assets/lens/powershot_v1.json")).ok()
        })
        .as_ref()
}

/// The interpolated correction entry for a camera shot, or `None` when no
/// packaged profile matches the model/focal length.
pub(crate) fn embedded_lens_entry(model: &str, focal_mm: Option<f32>) -> Option<lens::FocalEntry> {
    embedded_lens_profile()?.select(model, focal_mm)
}
pub use {exposure::AutoExposure, tone_mapping::ToneMapper};

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PhotographicSettings {
    pub auto_exposure: AutoExposure,
    pub exposure_compensation_stops: f32,
    pub tone_mapper: ToneMapper,
}

fn check_cancelled(cancelled: &AtomicBool) -> Result<(), String> {
    if cancelled.load(Ordering::Acquire) {
        Err("render cancelled".into())
    } else {
        Ok(())
    }
}
