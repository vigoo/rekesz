use std::sync::atomic::AtomicBool;

use image::{Rgb, RgbImage};

use super::{PhotographicSettings, check_cancelled, tone_mapping};

#[cfg(test)]
use super::exposure::exposure_multiplier;

#[cfg(test)]
pub(super) fn render_linear_rgb(
    width: u32,
    height: u32,
    pixels: Vec<[f32; 3]>,
    settings: &PhotographicSettings,
    cancelled: &AtomicBool,
) -> Result<(RgbImage, f32), String> {
    let automatic = exposure_multiplier(&pixels, settings.auto_exposure, cancelled)?;
    render_linear_rgb_with_exposure(width, height, pixels, settings, automatic, cancelled)
}

pub(super) fn render_linear_rgb_with_exposure(
    width: u32,
    height: u32,
    pixels: Vec<[f32; 3]>,
    settings: &PhotographicSettings,
    automatic: f32,
    cancelled: &AtomicBool,
) -> Result<(RgbImage, f32), String> {
    let exposure = automatic * 2.0_f32.powf(settings.exposure_compensation_stops);
    let mut bytes = Vec::with_capacity(pixels.len().saturating_mul(3));
    for (index, pixel) in pixels.into_iter().enumerate() {
        if index % 4096 == 0 {
            check_cancelled(cancelled)?;
        }
        bytes.extend(pixel.map(|channel| {
            let linear = (channel.max(0.0) * exposure).max(0.0);
            let mapped = tone_mapping::map(settings.tone_mapper, linear);
            (linear_to_srgb(mapped) * 255.0).round().clamp(0.0, 255.0) as u8
        }));
    }
    Ok((
        RgbImage::from_raw(width, height, bytes)
            .unwrap_or_else(|| RgbImage::from_pixel(1, 1, Rgb([0; 3]))),
        automatic.log2(),
    ))
}

fn linear_to_srgb(value: f32) -> f32 {
    if value <= 0.003_130_8 {
        value * 12.92
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    }
}

#[cfg(test)]
mod tests;
