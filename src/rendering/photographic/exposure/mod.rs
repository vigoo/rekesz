use std::sync::atomic::AtomicBool;

use serde::{Deserialize, Serialize};

use super::check_cancelled;

const HISTOGRAM_BINS: usize = 4096;
const MIN_LOG2_LUMINANCE: f32 = -16.0;
const MAX_LOG2_LUMINANCE: f32 = 8.0;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct AutoExposure {
    /// Scene-luminance percentile used as the exposure anchor.
    pub percentile: f32,
    /// Linear value to which the anchor is exposed before tone mapping.
    pub target: f32,
    /// Safety bound for automatic exposure compensation.
    pub max_stops: f32,
}

impl Default for AutoExposure {
    fn default() -> Self {
        Self {
            percentile: 0.50,
            target: 0.18,
            max_stops: 4.0,
        }
    }
}

pub(super) fn exposure_multiplier(
    pixels: &[[f32; 3]],
    settings: AutoExposure,
    cancelled: &AtomicBool,
) -> Result<f32, String> {
    let mut histogram = [0_u32; HISTOGRAM_BINS];
    let mut samples = 0_u64;
    // At most roughly one million samples; the full-resolution transform still
    // visits every pixel, while statistics stay cheap for very large sensors.
    let stride = pixels.len().div_ceil(1_000_000).max(1);
    for (index, pixel) in pixels.iter().step_by(stride).enumerate() {
        if index % 4096 == 0 {
            check_cancelled(cancelled)?;
        }
        let luminance = 0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2];
        if luminance.is_finite() && luminance > 0.0 {
            let normalized = ((luminance.log2() - MIN_LOG2_LUMINANCE)
                / (MAX_LOG2_LUMINANCE - MIN_LOG2_LUMINANCE))
                .clamp(0.0, 1.0);
            let bin = (normalized * (HISTOGRAM_BINS - 1) as f32) as usize;
            histogram[bin] += 1;
            samples += 1;
        }
    }
    if samples == 0 {
        return Ok(1.0);
    }
    let wanted = (samples as f32 * settings.percentile.clamp(0.0, 1.0))
        .ceil()
        .max(1.0) as u64;
    let mut accumulated = 0_u64;
    let bin = histogram
        .iter()
        .position(|count| {
            accumulated += u64::from(*count);
            accumulated >= wanted
        })
        .unwrap_or(HISTOGRAM_BINS - 1);
    let log_luminance = MIN_LOG2_LUMINANCE
        + (bin as f32 + 0.5) / HISTOGRAM_BINS as f32 * (MAX_LOG2_LUMINANCE - MIN_LOG2_LUMINANCE);
    let stops = (settings.target.max(f32::MIN_POSITIVE) / log_luminance.exp2())
        .log2()
        .clamp(-settings.max_stops.abs(), settings.max_stops.abs());
    Ok(stops.exp2())
}

#[cfg(test)]
mod tests;
