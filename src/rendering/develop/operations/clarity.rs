use std::sync::atomic::{AtomicBool, Ordering};

use image::RgbaImage;

use super::DevelopAction;
use super::log_detail::{apply_log_gain, decompose, smoothstep};
use crate::rendering::develop::{actions, check, decode_error};

/// Guided-filter radius for the base layer, referenced to a 1080-pixel short
/// edge. Lightroom's Clarity is a large-radius midtone local contrast; an
/// 8px band at reference size keeps it far above sharpening scales.
const BASE_RADIUS: f32 = 8.0;
/// Guided-filter regularization in the log2-luma domain.
const BASE_EPSILON: f32 = 0.25;
/// Positive amounts are amplified relative to negative ones, matching the
/// asymmetric response of the Lightroom control.
const POSITIVE_GAIN: f32 = 1.8;
/// Soft-knee width for shaped detail: positive detail is compressed more
/// tightly to control halos, negative detail stays softer.
const SHAPE_SIGMA_POSITIVE: f32 = 0.8;
const SHAPE_SIGMA_NEGATIVE: f32 = 1.5;
/// Positive clarity fades out near black (noise floor) and near white
/// (highlight protection); luminance values are display-linear.
const NOISE_GATE_HIGH: f32 = 0.03;
const HIGHLIGHT_PROTECT_LOW: f32 = 0.85;
const HIGHLIGHT_PROTECT_HIGH: f32 = 0.98;

pub(in crate::rendering::develop) struct Clarity {
    pub amount: f32,
}

impl DevelopAction for Clarity {
    fn action_mask(&self) -> u64 {
        actions::CLARITY
    }

    fn apply(&self, image: &mut RgbaImage, cancelled: &AtomicBool) -> image::ImageResult<()> {
        check(cancelled)?;
        let strength = self.amount.clamp(-100.0, 100.0) / 100.0;
        if strength == 0.0 {
            return Ok(());
        }
        self.apply_inner(image, cancelled)
            .map_err(|error| decode_error(&error))
    }
}

impl Clarity {
    fn apply_inner(&self, image: &mut RgbaImage, cancelled: &AtomicBool) -> Result<(), String> {
        let detail = decompose(image, BASE_RADIUS, BASE_EPSILON, cancelled)?;
        let strength = self.amount.clamp(-100.0, 100.0) / 100.0;
        let positive = strength > 0.0;
        let gain = if positive { POSITIVE_GAIN } else { 1.0 };
        let mut gains = Vec::with_capacity(detail.log_detail.len());
        for (index, &detail_value) in detail.log_detail.iter().enumerate() {
            if index % 16_384 == 0 && cancelled.load(Ordering::Acquire) {
                return Err("presence operator cancelled".to_owned());
            }
            let luma = detail.linear_luma[index];
            let mask = if positive {
                smoothstep(0.0, NOISE_GATE_HIGH, luma)
                    * (1.0 - smoothstep(HIGHLIGHT_PROTECT_LOW, HIGHLIGHT_PROTECT_HIGH, luma))
            } else {
                1.0
            };
            gains.push(shape_detail(detail_value) * gain * mask * strength);
        }
        apply_log_gain(image, &gains, cancelled)
    }
}

/// Soft-knee limiter: compresses large detail excursions quadratically so
/// strong edges gain less than the raw band would give them.
fn shape_detail(detail: f32) -> f32 {
    let sigma = if detail > 0.0 {
        SHAPE_SIGMA_POSITIVE
    } else {
        SHAPE_SIGMA_NEGATIVE
    };
    let ratio = detail / sigma;
    detail / (1.0 + ratio * ratio)
}
