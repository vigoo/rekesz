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

pub(super) use raw_render::render_raw;
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
