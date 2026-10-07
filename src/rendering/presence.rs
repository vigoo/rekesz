//! Presence operators: Lightroom-style Texture and Dehaze over linear RGB.
//!
//! The same cores serve the display-referred develop path (sRGB pixels
//! converted per render) and the RAW scene-linear path (demosaiced linear
//! radiance); tone masking accepts the caller's luminance normalizer because
//! the two paths use different working scales. All radii are referenced to a
//! 1080-pixel short edge.

use std::sync::atomic::{AtomicBool, Ordering};

use crate::rendering::filters::{box_min, guided_smooth, scaled_radius};

/// Texture extracts a fine mid-frequency band; finer than Clarity's band,
/// coarser than sharpening.
/// Clarity band radius (matches the u8-domain operator's BASE_RADIUS).
const CLARITY_RADIUS: f32 = 8.0;
const CLARITY_EPSILON: f32 = 0.25;
const CLARITY_POSITIVE_GAIN: f32 = 1.8;
const CLARITY_SHAPE_SIGMA_POSITIVE: f32 = 0.8;
const CLARITY_SHAPE_SIGMA_NEGATIVE: f32 = 1.5;
const CLARITY_NOISE_GATE_HIGH: f32 = 0.03;
const CLARITY_HIGHLIGHT_PROTECT_LOW: f32 = 0.85;
const CLARITY_HIGHLIGHT_PROTECT_HIGH: f32 = 0.98;

const TEXTURE_RADIUS: f32 = 2.5;
const TEXTURE_EPSILON: f32 = 0.25;
const TEXTURE_POSITIVE_GAIN: f32 = 1.5;
const TEXTURE_SHAPE_SIGMA_POSITIVE: f32 = 1.0;
const TEXTURE_SHAPE_SIGMA_NEGATIVE: f32 = 1.5;
const TEXTURE_NOISE_GATE_HIGH: f32 = 0.05;
const TEXTURE_HIGHLIGHT_PROTECT_LOW: f32 = 0.85;
const TEXTURE_HIGHLIGHT_PROTECT_HIGH: f32 = 0.98;

/// Log-luma samples below this floor are clamped before the logarithm.
const LOG_LUMA_FLOOR: f32 = 1.0e-4;

/// Dehaze transmission is refined with an edge-aware fit of the min-channel
/// ratio; epsilon regularizes in that bounded ratio domain.
const DEHAZE_RATIO_RADIUS: f32 = 8.0;
const DEHAZE_RATIO_EPSILON: f32 = 0.05;
/// Strength multiplier mapping the −100..100 slider onto transmission
/// removal, and the floor that keeps recovery from exploding in dense haze.
const DEHAZE_STRENGTH: f32 = 1.5;
const DEHAZE_MIN_TRANSMISSION: f32 = 0.08;
/// Recovery results this dark are partially rolled back toward the input so
/// positive dehaze does not crush shadows it cannot see through; the floor
/// keeps part of the effect even there.
const DEHAZE_SHADOW_PROTECT_HIGH: f32 = 0.08;
const DEHAZE_SHADOW_PROTECT_FLOOR: f32 = 0.35;
/// Saturation lift per unit of positive strength; haze removal genuinely
/// desaturates the veiling light, so the recovered scene regains chroma.
const DEHAZE_SATURATION_GAIN: f32 = 0.3;
/// Adding haze weights the mix toward the atmospheric light by at least this
/// much in already-hazy regions and at most this much plus the mapped depth.
const DEHAZE_ADD_MIN_DEPTH: f32 = 0.4;
const DEHAZE_ADD_MAX_STRENGTH: f32 = 0.7;
const ATMOSPHERIC_HAZY_FRACTION: f32 = 0.95;
const ATMOSPHERIC_BRIGHT_FRACTION: f32 = 0.95;
const HISTOGRAM_BINS: usize = 1024;

fn check_cancelled(cancelled: &AtomicBool) -> Result<(), String> {
    if cancelled.load(Ordering::Acquire) {
        Err("presence operator cancelled".to_owned())
    } else {
        Ok(())
    }
}

fn luminance(pixel: [f32; 3]) -> f32 {
    0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2]
}

fn smoothstep(low: f32, high: f32, value: f32) -> f32 {
    let scaled = ((value - low) / (high - low).max(f32::EPSILON)).clamp(0.0, 1.0);
    scaled * scaled * (3.0 - 2.0 * scaled)
}

/// Boosts or smooths the fine mid-frequency band of the image.
///
/// Positive amounts raise band contrast (crisp surfaces stay crisp, noise
/// and extremes are masked); negative amounts subtract the band, smoothing
/// surfaces while the guided base layer keeps edges intact.
/// Midtone-weighted local contrast on scene-linear pixels: an edge-aware
/// detail band from log2(luma) re-applied as a multiplicative gain. The
/// same algorithm as the u8-domain clarity operator, expressed for the
/// scene-linear stage (before tone mapping).
pub fn apply_clarity(
    pixels: &mut [[f32; 3]],
    width: usize,
    height: usize,
    amount: f32,
    normalize_tone: fn(f32) -> f32,
    cancelled: &AtomicBool,
) -> Result<(), String> {
    let strength = amount.clamp(-100.0, 100.0) / 100.0;
    if strength == 0.0 {
        return Ok(());
    }
    let mut log_luma = vec![0.0_f32; pixels.len()];
    let mut tones = vec![0.0_f32; pixels.len()];
    for (index, pixel) in pixels.iter().enumerate() {
        let luma = luminance(*pixel);
        tones[index] = normalize_tone(luma);
        log_luma[index] = luma.max(LOG_LUMA_FLOOR).log2();
    }
    let base = guided_smooth(
        &log_luma,
        width,
        height,
        CLARITY_RADIUS,
        CLARITY_EPSILON,
        cancelled,
    )?;
    let positive = strength > 0.0;
    let band_gain = if positive { CLARITY_POSITIVE_GAIN } else { 1.0 };
    for (index, pixel) in pixels.iter_mut().enumerate() {
        if index % 16_384 == 0 {
            check_cancelled(cancelled)?;
        }
        let detail = log_luma[index] - base[index];
        let shaped = shape_detail(
            detail,
            CLARITY_SHAPE_SIGMA_POSITIVE,
            CLARITY_SHAPE_SIGMA_NEGATIVE,
        );
        let mask = if positive {
            smoothstep(0.0, CLARITY_NOISE_GATE_HIGH, tones[index])
                * (1.0
                    - smoothstep(
                        CLARITY_HIGHLIGHT_PROTECT_LOW,
                        CLARITY_HIGHLIGHT_PROTECT_HIGH,
                        tones[index],
                    ))
        } else {
            1.0
        };
        let factor = 2.0_f32.powf(shaped * band_gain * mask * strength);
        for channel in pixel {
            *channel *= factor;
        }
    }
    Ok(())
}

pub fn apply_texture(
    pixels: &mut [[f32; 3]],
    width: usize,
    height: usize,
    amount: f32,
    normalize_tone: fn(f32) -> f32,
    cancelled: &AtomicBool,
) -> Result<(), String> {
    let strength = amount.clamp(-100.0, 100.0) / 100.0;
    if strength == 0.0 {
        return Ok(());
    }
    let mut log_luma = vec![0.0_f32; pixels.len()];
    let mut tones = vec![0.0_f32; pixels.len()];
    for (index, pixel) in pixels.iter().enumerate() {
        let luma = luminance(*pixel);
        tones[index] = normalize_tone(luma);
        log_luma[index] = luma.max(LOG_LUMA_FLOOR).log2();
    }
    let base = guided_smooth(
        &log_luma,
        width,
        height,
        TEXTURE_RADIUS,
        TEXTURE_EPSILON,
        cancelled,
    )?;
    let positive = strength > 0.0;
    let band_gain = if positive { TEXTURE_POSITIVE_GAIN } else { 1.0 };
    for (index, pixel) in pixels.iter_mut().enumerate() {
        if index % 16_384 == 0 {
            check_cancelled(cancelled)?;
        }
        let detail = log_luma[index] - base[index];
        let shaped = shape_detail(
            detail,
            TEXTURE_SHAPE_SIGMA_POSITIVE,
            TEXTURE_SHAPE_SIGMA_NEGATIVE,
        );
        let mask = if positive {
            smoothstep(0.0, TEXTURE_NOISE_GATE_HIGH, tones[index])
                * (1.0
                    - smoothstep(
                        TEXTURE_HIGHLIGHT_PROTECT_LOW,
                        TEXTURE_HIGHLIGHT_PROTECT_HIGH,
                        tones[index],
                    ))
        } else {
            1.0
        };
        let factor = 2.0_f32.powf(shaped * band_gain * mask * strength);
        for channel in pixel {
            *channel *= factor;
        }
    }
    Ok(())
}

fn shape_detail(detail: f32, sigma_positive: f32, sigma_negative: f32) -> f32 {
    let sigma = if detail > 0.0 {
        sigma_positive
    } else {
        sigma_negative
    };
    let ratio = detail / sigma;
    detail / (1.0 + ratio * ratio)
}

/// Removes haze for positive amounts by inverting the atmospheric scattering
/// model with a dark-channel-prior transmission estimate, or adds haze for
/// negative amounts by mixing toward the estimated atmospheric light.
pub fn apply_dehaze(
    pixels: &mut [[f32; 3]],
    width: usize,
    height: usize,
    amount: f32,
    normalize_tone: fn(f32) -> f32,
    cancelled: &AtomicBool,
) -> Result<(), String> {
    let strength = amount.clamp(-100.0, 100.0) / 100.0;
    if strength == 0.0 {
        return Ok(());
    }
    let mut dark_channel: Vec<f32> = Vec::with_capacity(pixels.len());
    for (index, pixel) in pixels.iter().enumerate() {
        if index % 16_384 == 0 {
            check_cancelled(cancelled)?;
        }
        dark_channel.push(pixel[0].min(pixel[1]).min(pixel[2]));
    }
    let atmospheric_light = estimate_atmospheric_light(pixels, &dark_channel, cancelled)?;
    drop(dark_channel);
    // True per-channel minimum of I_c / A_c; with a tinted atmospheric light
    // this differs from the dark channel over its brightest channel.
    let light = [
        atmospheric_light[0].max(f32::EPSILON),
        atmospheric_light[1].max(f32::EPSILON),
        atmospheric_light[2].max(f32::EPSILON),
    ];
    let mut ratio: Vec<f32> = Vec::with_capacity(pixels.len());
    for (index, pixel) in pixels.iter().enumerate() {
        if index % 16_384 == 0 {
            check_cancelled(cancelled)?;
        }
        ratio.push(
            (pixel[0] / light[0])
                .min(pixel[1] / light[1])
                .min(pixel[2] / light[2])
                .max(0.0),
        );
    }
    let speckle_radius = scaled_radius(2.0, width, height);
    let filtered = box_min(&ratio, width, height, speckle_radius, cancelled)?;
    drop(ratio);
    let haziness = guided_smooth(
        &filtered,
        width,
        height,
        DEHAZE_RATIO_RADIUS,
        DEHAZE_RATIO_EPSILON,
        cancelled,
    )?;
    if strength > 0.0 {
        remove_haze(
            pixels,
            strength,
            &haziness,
            &atmospheric_light,
            normalize_tone,
            cancelled,
        )
    } else {
        add_haze(pixels, -strength, &haziness, &atmospheric_light, cancelled)
    }
}

fn remove_haze(
    pixels: &mut [[f32; 3]],
    strength: f32,
    haziness: &[f32],
    atmospheric_light: &[f32; 3],
    normalize_tone: fn(f32) -> f32,
    cancelled: &AtomicBool,
) -> Result<(), String> {
    for (index, pixel) in pixels.iter_mut().enumerate() {
        if index % 16_384 == 0 {
            check_cancelled(cancelled)?;
        }
        let mapped = haziness[index] / (haziness[index] + 0.2);
        let transmission = (1.0 - strength * DEHAZE_STRENGTH * mapped).max(DEHAZE_MIN_TRANSMISSION);
        let mut recovered = [0.0_f32; 3];
        for channel in 0..3 {
            let light = atmospheric_light[channel].max(f32::EPSILON);
            let ratio = pixel[channel] / light;
            // The inversion is physically bounded: channels at or above the
            // veiling light cannot be amplified, and darker ones cannot
            // recover past black.
            let physical_floor = if ratio >= 1.0 {
                1.0
            } else {
                (1.0 - ratio).max(DEHAZE_MIN_TRANSMISSION)
            };
            let bounded = transmission.max(physical_floor);
            recovered[channel] = (pixel[channel] - light) / bounded + light;
        }
        let recovered_luma = normalize_tone(luminance(recovered).max(0.0));
        let protect = DEHAZE_SHADOW_PROTECT_FLOOR
            + (1.0 - DEHAZE_SHADOW_PROTECT_FLOOR)
                * smoothstep(0.0, DEHAZE_SHADOW_PROTECT_HIGH, recovered_luma);
        let original_luma = luminance(*pixel);
        for channel in 0..3 {
            let target = recovered[channel].max(0.0);
            let blended = pixel[channel] + (target - pixel[channel]) * protect;
            // A chroma-preserving saturation lift on top of the recovered
            // color compensates for the desaturating veiling light.
            let chroma = blended - original_luma;
            pixel[channel] = original_luma + chroma * (1.0 + strength * DEHAZE_SATURATION_GAIN);
        }
    }
    Ok(())
}

fn add_haze(
    pixels: &mut [[f32; 3]],
    strength: f32,
    haziness: &[f32],
    atmospheric_light: &[f32; 3],
    cancelled: &AtomicBool,
) -> Result<(), String> {
    for (index, pixel) in pixels.iter_mut().enumerate() {
        if index % 16_384 == 0 {
            check_cancelled(cancelled)?;
        }
        let mapped = haziness[index] / (haziness[index] + 0.2);
        let depth = DEHAZE_ADD_MIN_DEPTH + (1.0 - DEHAZE_ADD_MIN_DEPTH) * mapped;
        let mix = (strength * DEHAZE_ADD_MAX_STRENGTH * depth).min(1.0);
        for channel in 0..3 {
            pixel[channel] = pixel[channel] * (1.0 - mix) + atmospheric_light[channel] * mix;
        }
    }
    Ok(())
}

/// Estimates the atmospheric light as the mean color of the brightest pixels
/// among the haziest: the top `1 − ATMOSPHERIC_HAZY_FRACTION` by raw dark
/// channel, then the top `1 − ATMOSPHERIC_BRIGHT_FRACTION` of those by
/// luminance. Histogram percentiles keep this linear in the pixel count.
fn estimate_atmospheric_light(
    pixels: &[[f32; 3]],
    dark_channel: &[f32],
    cancelled: &AtomicBool,
) -> Result<[f32; 3], String> {
    let dark_max = dark_channel.iter().copied().fold(0.0_f32, f32::max);
    if dark_max <= f32::EPSILON {
        return Ok([1.0, 1.0, 1.0]);
    }
    let mut dark_histogram = [0_u32; HISTOGRAM_BINS];
    for (index, &dark) in dark_channel.iter().enumerate() {
        if index % 16_384 == 0 {
            check_cancelled(cancelled)?;
        }
        let bin = ((dark / dark_max) * (HISTOGRAM_BINS - 1) as f32) as usize;
        dark_histogram[bin.min(HISTOGRAM_BINS - 1)] += 1;
    }
    let Some(hazy_bin) = percentile_bin(&dark_histogram, ATMOSPHERIC_HAZY_FRACTION) else {
        return Ok([1.0, 1.0, 1.0]);
    };
    let hazy_threshold = hazy_bin as f32 / (HISTOGRAM_BINS - 1) as f32 * dark_max;
    let total = pixels.len() as f32;

    let mut luma_histogram = [0_u32; HISTOGRAM_BINS];
    let mut luma_max = 0.0_f32;
    let mut subset = 0_u64;
    for (index, (pixel, &dark)) in pixels.iter().zip(dark_channel).enumerate() {
        if index % 16_384 == 0 {
            check_cancelled(cancelled)?;
        }
        if dark >= hazy_threshold {
            let luma = luminance(*pixel);
            luma_max = luma_max.max(luma);
            subset += 1;
        }
    }
    if luma_max <= f32::EPSILON || subset == 0 {
        return Ok([1.0, 1.0, 1.0]);
    }
    for (index, (pixel, &dark)) in pixels.iter().zip(dark_channel).enumerate() {
        if index % 16_384 == 0 {
            check_cancelled(cancelled)?;
        }
        if dark >= hazy_threshold {
            let bin = ((luminance(*pixel) / luma_max) * (HISTOGRAM_BINS - 1) as f32) as usize;
            luma_histogram[bin.min(HISTOGRAM_BINS - 1)] += 1;
        }
    }
    let Some(bright_bin) = percentile_bin(&luma_histogram, ATMOSPHERIC_BRIGHT_FRACTION) else {
        return Ok([1.0, 1.0, 1.0]);
    };
    let bright_threshold = bright_bin as f32 / (HISTOGRAM_BINS - 1) as f32 * luma_max;

    let mut mean = [0.0_f32; 3];
    let mut count = 0.0_f32;
    for (index, (pixel, &dark)) in pixels.iter().zip(dark_channel).enumerate() {
        if index % 16_384 == 0 {
            check_cancelled(cancelled)?;
        }
        if dark >= hazy_threshold && luminance(*pixel) >= bright_threshold {
            for channel in 0..3 {
                mean[channel] += pixel[channel];
            }
            count += 1.0;
        }
    }
    check_cancelled(cancelled)?;
    if count < total * 1.0e-4 {
        return Ok([1.0, 1.0, 1.0]);
    }
    Ok(mean.map(|value| value / count))
}

/// Returns the lowest bin whose cumulative count reaches `fraction` of the
/// total, so the bins above it hold the top `1 − fraction` of samples.
fn percentile_bin(histogram: &[u32; HISTOGRAM_BINS], fraction: f32) -> Option<usize> {
    let total: u64 = histogram.iter().map(|&count| u64::from(count)).sum();
    if total == 0 {
        return None;
    }
    let target = (total as f32 * fraction).ceil() as u64;
    let mut cumulative = 0_u64;
    for (bin, &count) in histogram.iter().enumerate() {
        cumulative += u64::from(count);
        if cumulative >= target.max(1) {
            return Some(bin);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicBool;

    use super::{apply_dehaze, apply_texture};
    use test_r::test;

    fn not_cancelled() -> AtomicBool {
        AtomicBool::new(false)
    }

    fn identity_tone(luma: f32) -> f32 {
        luma
    }

    fn gray_with_dark_square(square_level: f32) -> (usize, usize, Vec<[f32; 3]>) {
        let width = 256;
        let height = 128;
        let mut pixels = vec![[0.5_f32, 0.5, 0.5]; width * height];
        for y in 48..80 {
            for x in 112..144 {
                pixels[y * width + x] = [square_level; 3];
            }
        }
        (width, height, pixels)
    }

    fn region_luma(pixels: &[[f32; 3]], width: usize, x: usize, y: usize) -> f32 {
        let pixel = pixels[y * width + x];
        0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2]
    }

    #[test]
    fn texture_at_zero_is_identity() {
        let cancelled = not_cancelled();
        let (width, height, mut pixels) = gray_with_dark_square(0.25);
        let original = pixels.clone();
        apply_texture(&mut pixels, width, height, 0.0, identity_tone, &cancelled)
            .expect("texture succeeds");
        assert_eq!(pixels, original);
    }

    #[test]
    fn texture_positive_boosts_band_contrast_and_negative_reduces_it() {
        let cancelled = not_cancelled();
        let width = 256;
        let height = 128;
        let mut pixels: Vec<[f32; 3]> = Vec::with_capacity(width * height);
        for x in 0..width {
            let band = 0.08 * (2.0 * std::f32::consts::PI * x as f32 / 8.0).sin();
            for _ in 0..height {
                pixels.push([0.5 + band; 3]);
            }
        }
        // Row-wise layout: the band varies along each row, so the amplitude
        // closure samples the middle row.
        let band_amplitude = |pixels: &[[f32; 3]]| -> f32 {
            let row = &pixels[64 * width..65 * width];
            let max = row.iter().map(|p| p[0]).fold(f32::MIN, f32::max);
            let min = row.iter().map(|p| p[0]).fold(f32::MAX, f32::min);
            max - min
        };
        let baseline = band_amplitude(&pixels);
        let mut boosted = pixels.clone();
        apply_texture(&mut boosted, width, height, 80.0, identity_tone, &cancelled)
            .expect("texture succeeds");
        assert!(
            band_amplitude(&boosted) > baseline,
            "positive texture must widen the band: {} -> {}",
            baseline,
            band_amplitude(&boosted)
        );
        let mut smoothed = pixels.clone();
        apply_texture(
            &mut smoothed,
            width,
            height,
            -80.0,
            identity_tone,
            &cancelled,
        )
        .expect("texture succeeds");
        assert!(
            band_amplitude(&smoothed) < baseline,
            "negative texture must narrow the band: {} -> {}",
            baseline,
            band_amplitude(&smoothed)
        );
    }

    #[test]
    fn dehaze_positive_darkens_veiled_dark_regions() {
        let cancelled = not_cancelled();
        let (width, height, mut pixels) = gray_with_dark_square(0.25);
        let before = region_luma(&pixels, width, 128, 64);
        apply_dehaze(&mut pixels, width, height, 40.0, identity_tone, &cancelled)
            .expect("dehaze succeeds");
        let after = region_luma(&pixels, width, 128, 64);
        assert!(
            after < before,
            "positive dehaze must darken a veiled dark region: {before} -> {after}"
        );
    }

    #[test]
    fn dehaze_negative_lightens_toward_the_atmospheric_light() {
        let cancelled = not_cancelled();
        let (width, height, mut pixels) = gray_with_dark_square(0.25);
        let before = region_luma(&pixels, width, 128, 64);
        apply_dehaze(&mut pixels, width, height, -60.0, identity_tone, &cancelled)
            .expect("dehaze succeeds");
        let after = region_luma(&pixels, width, 128, 64);
        assert!(
            after > before,
            "negative dehaze must lift a dark region toward the veil: {before} -> {after}"
        );
    }

    #[test]
    fn dehaze_keeps_saturated_colors_in_recovered_regions() {
        let cancelled = not_cancelled();
        let width = 128;
        let height = 128;
        let mut pixels = vec![[0.35_f32, 0.3, 0.2]; width * height];
        apply_dehaze(&mut pixels, width, height, 50.0, identity_tone, &cancelled)
            .expect("dehaze succeeds");
        for pixel in &pixels {
            assert!(pixel.iter().all(|value| value.is_finite() && *value >= 0.0));
        }
    }

    #[test]
    fn presence_cores_report_cancellation() {
        let cancelled = AtomicBool::new(true);
        let (width, height, mut pixels) = gray_with_dark_square(0.25);
        assert!(
            apply_texture(&mut pixels, width, height, 50.0, identity_tone, &cancelled).is_err()
        );
        let (width, height, mut pixels) = gray_with_dark_square(0.25);
        assert!(apply_dehaze(&mut pixels, width, height, 50.0, identity_tone, &cancelled).is_err());
    }
}

#[cfg(test)]
mod recovery_bounds_tests {
    use std::sync::atomic::AtomicBool;

    use super::apply_dehaze;
    use test_r::test;

    fn not_cancelled() -> AtomicBool {
        AtomicBool::new(false)
    }

    fn identity_tone(luma: f32) -> f32 {
        luma
    }

    #[test]
    fn dehaze_never_amplifies_pixels_beyond_the_atmospheric_light() {
        let cancelled = not_cancelled();
        let width = 128;
        let height = 128;
        // A bright sky that the atmospheric light estimate will lock onto, a
        // sun glint above it, and a dark region under it.
        let mut pixels = vec![[0.6_f32, 0.6, 0.6]; width * height];
        for y in 0..16 {
            for x in 0..width {
                pixels[y * width + x] = [0.9, 0.9, 0.9];
            }
        }
        for y in 40..48 {
            for x in 40..72 {
                pixels[y * width + x] = [1.4, 1.3, 1.2];
            }
        }
        for y in 80..112 {
            for x in 48..80 {
                pixels[y * width + x] = [0.2, 0.2, 0.2];
            }
        }
        let glint_before = pixels[44 * width + 56];
        apply_dehaze(&mut pixels, width, height, 100.0, identity_tone, &cancelled)
            .expect("dehaze succeeds");
        let glint_after = pixels[44 * width + 56];
        for channel in 0..3 {
            assert!(
                glint_after[channel] <= glint_before[channel] * 1.05 + 0.01,
                "channel {channel} amplified above the veil: {} -> {}",
                glint_before[channel],
                glint_after[channel]
            );
        }
        for pixel in &pixels {
            for value in pixel {
                assert!(value.is_finite() && *value >= 0.0, "{value}");
            }
        }
    }
}
