//! Shared log-luminance decomposition for the presence operators.
//!
//! Clarity and Texture both extract an edge-aware detail band from
//! log2(linear luminance) via the guided filter and re-apply it as a
//! per-pixel multiplicative gain, which keeps the operators hue-neutral and
//! free of the clipping that byte-domain unsharp masks suffer.

use std::sync::atomic::{AtomicBool, Ordering};

use image::RgbaImage;

use crate::rendering::develop::{linear_to_srgb, srgb_to_linear};
use crate::rendering::filters;

/// Luminance samples below this floor are clamped before the logarithm so
/// black pixels do not produce unbounded detail.
const LOG_LUMA_FLOOR: f32 = 1.0e-4;

pub(super) struct LumaDetail {
    /// `log2(linear luma) − guided base layer`, in stops.
    pub log_detail: Vec<f32>,
    /// Linear luminance in the 0..1 display range, for tone masking.
    pub linear_luma: Vec<f32>,
}

pub(super) fn decompose(
    image: &RgbaImage,
    radius_base: f32,
    eps: f32,
    cancelled: &AtomicBool,
) -> Result<LumaDetail, String> {
    let width = image.width() as usize;
    let height = image.height() as usize;
    let mut log_luma = vec![0.0_f32; width * height];
    let mut linear_luma = vec![0.0_f32; width * height];
    let table = srgb_to_linear_table();
    for (index, pixel) in image.pixels().enumerate() {
        if index % 16_384 == 0 && cancelled.load(Ordering::Acquire) {
            return Err("presence operator cancelled".to_owned());
        }
        let red = table[pixel[0] as usize];
        let green = table[pixel[1] as usize];
        let blue = table[pixel[2] as usize];
        let luma = 0.2126 * red + 0.7152 * green + 0.0722 * blue;
        linear_luma[index] = luma;
        log_luma[index] = luma.max(LOG_LUMA_FLOOR).log2();
    }
    let base = filters::guided_smooth(&log_luma, width, height, radius_base, eps, cancelled)?;
    // Subtract the base layer in place so only one log-luma plane stays live.
    for (index, detail) in log_luma.iter_mut().enumerate() {
        if index % 16_384 == 0 && cancelled.load(Ordering::Acquire) {
            return Err("presence operator cancelled".to_owned());
        }
        *detail -= base[index];
    }
    Ok(LumaDetail {
        log_detail: log_luma,
        linear_luma,
    })
}

/// Multiplies each pixel's linear RGB by `2^gain` and re-encodes sRGB.
pub(super) fn apply_log_gain(
    image: &mut RgbaImage,
    gains: &[f32],
    cancelled: &AtomicBool,
) -> Result<(), String> {
    let table = srgb_to_linear_table();
    for (index, pixel) in image.pixels_mut().enumerate() {
        if index % 16_384 == 0 && cancelled.load(Ordering::Acquire) {
            return Err("presence operator cancelled".to_owned());
        }
        let factor = 2.0_f32.powf(gains[index]);
        for channel in 0..3 {
            let linear = table[pixel[channel] as usize] * factor;
            pixel[channel] = linear_to_srgb(linear);
        }
    }
    Ok(())
}

pub(super) fn smoothstep(low: f32, high: f32, value: f32) -> f32 {
    let scaled = ((value - low) / (high - low).max(f32::EPSILON)).clamp(0.0, 1.0);
    scaled * scaled * (3.0 - 2.0 * scaled)
}

pub(super) fn srgb_to_linear_table() -> [f32; 256] {
    let mut table = [0.0_f32; 256];
    for (value, entry) in table.iter_mut().enumerate() {
        *entry = srgb_to_linear(value as u8);
    }
    table
}

pub(super) fn to_image_error(error: String) -> image::ImageError {
    image::ImageError::Decoding(image::error::DecodingError::new(
        image::error::ImageFormatHint::Unknown,
        error,
    ))
}
