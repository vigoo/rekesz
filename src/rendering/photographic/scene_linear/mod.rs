use std::sync::atomic::AtomicBool;

use crate::catalog::DevelopRecipe;

use super::{PhotographicSettings, check_cancelled, exposure::exposure_multiplier};

pub(super) fn apply_recipe_scene_linear(
    pixels: &mut [[f32; 3]],
    width: u32,
    height: u32,
    settings: &PhotographicSettings,
    recipe: Option<&DevelopRecipe>,
    developed_baseline_stops: f32,
    cancelled: &AtomicBool,
) -> Result<f32, String> {
    // A catalog recipe represents an intentional developed rendering. Keep the
    // mandatory RAW development stages, but do not stack Rekesz's aesthetic auto
    // exposure on top of those authored controls.
    let automatic = if recipe.is_some() {
        1.0
    } else {
        exposure_multiplier(pixels, settings.auto_exposure, cancelled)?
    };
    let recipe_exposure = recipe.and_then(|recipe| recipe.exposure_ev).unwrap_or(0.0);
    let exposure = 2.0_f32.powf(recipe_exposure + developed_baseline_stops);
    for (index, pixel) in pixels.iter_mut().enumerate() {
        if index % 4096 == 0 {
            check_cancelled(cancelled)?;
        }
        for channel in pixel.iter_mut() {
            *channel *= exposure;
        }
    }
    if let Some(recipe) = recipe {
        apply_tonal_controls(pixels, recipe, cancelled)?;
        apply_detail_controls(pixels, width, height, recipe, cancelled)?;
    }
    Ok(automatic)
}

fn apply_detail_controls(
    pixels: &mut [[f32; 3]],
    width: u32,
    height: u32,
    recipe: &DevelopRecipe,
    cancelled: &AtomicBool,
) -> Result<(), String> {
    if let Some(detail) = recipe.detail {
        let radius = detail
            .sharpen_radius
            .round()
            .max(1.0 + detail.color_smoothness / 100.0)
            .clamp(1.0, 3.0) as u32;
        let low = blur(pixels, width, height, radius, cancelled)?;
        let luma_mix =
            detail.luminance_noise_reduction / 100.0 * (1.0 - detail.luminance_detail / 200.0);
        let color_mix =
            detail.color_noise_reduction / 100.0 * (0.5 + detail.color_smoothness / 200.0);
        let sharpen = detail.sharpening / 100.0 * (0.5 + detail.sharpen_detail / 100.0);
        let threshold = detail.edge_masking / 100.0 * 0.1;
        for (index, (pixel, low)) in pixels.iter_mut().zip(low).enumerate() {
            if index % 4096 == 0 {
                check_cancelled(cancelled)?;
            }
            let luma = 0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2];
            let low_luma = 0.2126 * low[0] + 0.7152 * low[1] + 0.0722 * low[2];
            let output_luma = luma * (1.0 - luma_mix)
                + low_luma * luma_mix
                + (luma - low_luma) * detail.luminance_contrast / 100.0;
            let edge = (luma - low_luma).abs();
            let edge_mask = ((edge - threshold) / 0.1).clamp(0.0, 1.0);
            for channel in 0..3 {
                let chroma = pixel[channel] - luma;
                let low_chroma = low[channel] - low_luma;
                let output_chroma = chroma * (1.0 - color_mix)
                    + low_chroma * color_mix
                    + (chroma - low_chroma) * detail.color_detail / 400.0;
                pixel[channel] = output_luma
                    + output_chroma
                    + (pixel[channel] - low[channel]) * sharpen * edge_mask;
            }
        }
    }
    if recipe.texture.is_some() || recipe.dehaze.is_some() {
        let scene_tone = |luma: f32| luma / (1.0 + luma);
        if let Some(dehaze) = recipe.dehaze {
            crate::rendering::presence::apply_dehaze(
                pixels,
                width as usize,
                height as usize,
                dehaze,
                scene_tone,
                cancelled,
            )?;
        }
        if let Some(texture) = recipe.texture {
            crate::rendering::presence::apply_texture(
                pixels,
                width as usize,
                height as usize,
                texture,
                scene_tone,
                cancelled,
            )?;
        }
    }
    if let Some(monochrome) = recipe.monochrome {
        for (index, pixel) in pixels.iter_mut().enumerate() {
            if index % 4096 == 0 {
                check_cancelled(cancelled)?;
            }
            let max = pixel.iter().copied().fold(0.0_f32, f32::max);
            let min = pixel.iter().copied().fold(f32::MAX, f32::min);
            let delta = max - min;
            let hue = if delta <= 1e-6 {
                0.0
            } else if max == pixel[0] {
                60.0 * ((pixel[1] - pixel[2]) / delta).rem_euclid(6.0)
            } else if max == pixel[1] {
                60.0 * ((pixel[2] - pixel[0]) / delta + 2.0)
            } else {
                60.0 * ((pixel[0] - pixel[1]) / delta + 4.0)
            };
            let sector = hue / 45.0;
            let left = sector.floor() as usize % 8;
            let blend = sector.fract();
            let mixer =
                monochrome.mixer[left] * (1.0 - blend) + monochrome.mixer[(left + 1) % 8] * blend;
            let gray =
                (0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2]) * (1.0 + mixer / 100.0);
            *pixel = [gray; 3];
        }
    }
    if let Some(controls) = recipe.parametric_curve {
        for (index, pixel) in pixels.iter_mut().enumerate() {
            if index % 4096 == 0 {
                check_cancelled(cancelled)?;
            }
            let luma = 0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2];
            let normalized = luma / (1.0 + luma.max(0.0));
            let centers = [0.875_f32, 0.625, 0.375, 0.125];
            let weights =
                centers.map(|center| (-0.5 * ((normalized - center) / 0.25).powi(2)).exp());
            let total = weights.iter().sum::<f32>().max(1e-6);
            let stops = controls
                .iter()
                .zip(weights)
                .map(|(value, weight)| value / 100.0 * weight)
                .sum::<f32>()
                / total;
            let gain = 2.0_f32.powf(stops);
            for channel in pixel.iter_mut() {
                *channel *= gain;
            }
        }
    }
    Ok(())
}

fn blur(
    pixels: &[[f32; 3]],
    width: u32,
    height: u32,
    radius: u32,
    cancelled: &AtomicBool,
) -> Result<Vec<[f32; 3]>, String> {
    let mut horizontal = vec![[0.0; 3]; pixels.len()];
    for y in 0..height {
        if y % 8 == 0 {
            check_cancelled(cancelled)?;
        }
        for x in 0..width {
            let mut sum = [0.0; 3];
            let mut count = 0.0;
            for sx in x.saturating_sub(radius)..=(x + radius).min(width - 1) {
                let pixel = pixels[(y * width + sx) as usize];
                for channel in 0..3 {
                    sum[channel] += pixel[channel];
                }
                count += 1.0;
            }
            horizontal[(y * width + x) as usize] = sum.map(|value| value / count);
        }
    }
    let mut output = vec![[0.0; 3]; pixels.len()];
    for y in 0..height {
        if y % 8 == 0 {
            check_cancelled(cancelled)?;
        }
        for x in 0..width {
            let mut sum = [0.0; 3];
            let mut count = 0.0;
            for sy in y.saturating_sub(radius)..=(y + radius).min(height - 1) {
                let pixel = horizontal[(sy * width + x) as usize];
                for channel in 0..3 {
                    sum[channel] += pixel[channel];
                }
                count += 1.0;
            }
            output[(y * width + x) as usize] = sum.map(|value| value / count);
        }
    }
    Ok(output)
}

fn apply_tonal_controls(
    pixels: &mut [[f32; 3]],
    recipe: &DevelopRecipe,
    cancelled: &AtomicBool,
) -> Result<(), String> {
    let contrast = recipe.contrast.unwrap_or(0.0) / 100.0;
    let highlights = recipe.highlights.unwrap_or(0.0) / 100.0;
    let shadows = recipe.shadows.unwrap_or(0.0) / 100.0;
    let whites = recipe.whites.unwrap_or(0.0) / 100.0;
    let blacks = recipe.blacks.unwrap_or(0.0) / 100.0;
    let saturation = recipe.saturation.unwrap_or(0.0) / 100.0;
    let vibrance = recipe.vibrance.unwrap_or(0.0) / 100.0;
    for (index, pixel) in pixels.iter_mut().enumerate() {
        if index % 4096 == 0 {
            check_cancelled(cancelled)?;
        }
        let original_luma = (0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2]).max(1e-6);
        let normalized = original_luma / (1.0 + original_luma);
        let shadow_mask = (1.0 - normalized).powi(2);
        let highlight_mask = normalized.powi(2);
        let mut luma = original_luma;
        luma *= 1.0 + shadows * shadow_mask * 0.5 + highlights * highlight_mask * 0.5;
        luma *= 1.0 + blacks * shadow_mask * 0.25 + whites * highlight_mask * 0.25;
        luma = 0.18 * (luma / 0.18).max(1e-6).powf(1.0 + contrast);
        let ratio = luma.max(0.0) / original_luma;
        for channel in pixel.iter_mut() {
            *channel *= ratio;
        }
        let chroma = pixel
            .iter()
            .map(|channel| (channel - luma).abs())
            .fold(0.0_f32, f32::max);
        let adaptive = (1.0 + saturation + vibrance * (1.0 - chroma.min(1.0))).max(0.0);
        for channel in pixel.iter_mut() {
            *channel = luma + (*channel - luma) * adaptive;
        }
        if let Some(curve) = &recipe.tone_curve {
            let before = (0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2]).max(1e-6);
            let after = crate::rendering::develop::sample_curve(curve, before.clamp(0.0, 1.0));
            let ratio = after.max(0.0) / before;
            for channel in pixel.iter_mut() {
                *channel *= ratio;
            }
        }
        for (channel, curve) in [
            recipe.tone_curve_red.as_ref(),
            recipe.tone_curve_green.as_ref(),
            recipe.tone_curve_blue.as_ref(),
        ]
        .into_iter()
        .enumerate()
        {
            if let Some(curve) = curve {
                pixel[channel] =
                    crate::rendering::develop::sample_curve(curve, pixel[channel].clamp(0.0, 1.0));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
