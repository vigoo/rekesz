//! Camera-independent photographic rendering of scene-linear RGB.

use std::{
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use image::{DynamicImage, Rgb, RgbImage};
use rawler::imgop::develop::{Intermediate, ProcessingStep, RawDevelop};
use serde::{Deserialize, Serialize};

use super::{AppliedPhotographicTransform, OutputEncoding};

#[derive(Clone)]
pub(crate) struct DecodedRaw {
    pub raw: rawler::RawImage,
    pub orientation: Option<image::metadata::Orientation>,
}

impl DecodedRaw {
    pub fn oriented_dimensions(&self) -> (u32, u32) {
        let width = self.raw.crop_area.map_or(self.raw.width, |area| area.d.w) as u32;
        let height = self.raw.crop_area.map_or(self.raw.height, |area| area.d.h) as u32;
        match self.orientation {
            Some(
                image::metadata::Orientation::Rotate90
                | image::metadata::Orientation::Rotate270
                | image::metadata::Orientation::Rotate90FlipH
                | image::metadata::Orientation::Rotate270FlipH,
            ) => (height, width),
            _ => (width, height),
        }
    }
}

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

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub enum ToneMapper {
    #[default]
    AcesFitted,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PhotographicSettings {
    pub auto_exposure: AutoExposure,
    pub exposure_compensation_stops: f32,
    pub tone_mapper: ToneMapper,
}

pub(super) fn decode_raw(path: &Path, cancelled: &Arc<AtomicBool>) -> Result<DecodedRaw, String> {
    use rawler::{decoders::RawDecodeParams, rawsource::RawSource};

    let source = RawSource::new(path).map_err(|error| error.to_string())?;
    check_cancelled(cancelled)?;
    let decoder = rawler::global_loader()
        .get_decoder(&source)
        .map_err(|error| error.to_string())?;
    let params = RawDecodeParams::default();
    let rawler_cancellation = rawler::cancel::CancellationToken::from_atomic(cancelled.clone());
    let metadata = decoder
        .raw_metadata(&source, &params)
        .map_err(|error| error.to_string())?;
    check_cancelled(cancelled)?;
    let raw = decoder
        .raw_image_cancellable(&source, &params, false, &rawler_cancellation)
        .map_err(|error| error.to_string())?;
    check_cancelled(cancelled)?;
    let orientation = metadata
        .exif
        .orientation
        .and_then(|value| u8::try_from(value).ok())
        .or_else(|| u8::try_from(raw.orientation.to_u16()).ok())
        .and_then(image::metadata::Orientation::from_exif);
    Ok(DecodedRaw { raw, orientation })
}

pub(super) fn render_raw(
    decoded: &DecodedRaw,
    settings: &PhotographicSettings,
    recipe: Option<&crate::catalog::DevelopRecipe>,
    cancelled: &Arc<AtomicBool>,
) -> Result<(DynamicImage, AppliedPhotographicTransform), String> {
    let mut overridden;
    let raw = if let Some(recipe) = recipe
        && (recipe.white_balance.is_some() || recipe.white_balance_increment.is_some())
    {
        overridden = decoded.raw.clone();
        overridden.wb_coeffs = normalized_wb_coefficients(&decoded.raw, recipe);
        &overridden
    } else {
        &decoded.raw
    };
    let develop = RawDevelop::new_with(&[
        ProcessingStep::Rescale,
        ProcessingStep::Demosaic,
        ProcessingStep::FujiRotate,
        ProcessingStep::CropActiveArea,
        ProcessingStep::WhiteBalance,
        ProcessingStep::Calibrate,
        ProcessingStep::CropDefault,
    ]);
    let token = rawler::cancel::CancellationToken::from_atomic(cancelled.clone());
    let intermediate = develop
        .develop_intermediate_cancellable(raw, &token)
        .map_err(|error| error.to_string())?;
    let (width, height, mut pixels) = linear_rgb(intermediate, cancelled)?;
    let recipe_exposure = recipe.and_then(|recipe| recipe.exposure_ev).unwrap_or(0.0);
    let developed_baseline = if recipe.is_some() { 0.75 } else { 0.0 };
    let automatic = apply_recipe_scene_linear(
        &mut pixels,
        width,
        height,
        settings,
        recipe,
        developed_baseline,
        cancelled,
    )?;
    let mut adjusted = *settings;
    let (image, automatic_exposure_stops) =
        render_linear_rgb_with_exposure(width, height, pixels, &adjusted, automatic, cancelled)?;
    let mut image = DynamicImage::ImageRgb8(image);
    if let Some(orientation) = decoded.orientation {
        image.apply_orientation(orientation);
    }
    adjusted.exposure_compensation_stops += recipe_exposure;
    adjusted.exposure_compensation_stops += developed_baseline;
    Ok((
        image,
        AppliedPhotographicTransform {
            settings: adjusted,
            automatic_exposure_stops,
            total_exposure_stops: automatic_exposure_stops
                + settings.exposure_compensation_stops
                + recipe_exposure
                + developed_baseline,
            output_encoding: OutputEncoding::Srgb,
            orientation_applied: decoded.orientation.map(|value| value.to_exif()),
        },
    ))
}

fn apply_recipe_scene_linear(
    pixels: &mut [[f32; 3]],
    width: u32,
    height: u32,
    settings: &PhotographicSettings,
    recipe: Option<&crate::catalog::DevelopRecipe>,
    developed_baseline_stops: f32,
    cancelled: &AtomicBool,
) -> Result<f32, String> {
    // A catalog recipe represents an intentional developed rendering. Keep the
    // mandatory RAW development stages, but do not stack Pics' aesthetic auto
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
        apply_scene_linear_controls(pixels, recipe, cancelled)?;
        apply_phase_c_scene_linear(pixels, width, height, recipe, cancelled)?;
    }
    Ok(automatic)
}

fn apply_phase_c_scene_linear(
    pixels: &mut [[f32; 3]],
    width: u32,
    height: u32,
    recipe: &crate::catalog::DevelopRecipe,
    cancelled: &AtomicBool,
) -> Result<(), String> {
    if let Some(detail) = recipe.detail {
        let radius = detail
            .sharpen_radius
            .round()
            .max(1.0 + detail.color_smoothness / 100.0)
            .clamp(1.0, 3.0) as u32;
        let low = blur_linear(pixels, width, height, radius, cancelled)?;
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
        let low = blur_linear(pixels, width, height, 2, cancelled)?;
        let texture = recipe.texture.unwrap_or(0.0) / 100.0;
        let dehaze = recipe.dehaze.unwrap_or(0.0) / 100.0;
        for (index, (pixel, low)) in pixels.iter_mut().zip(low).enumerate() {
            if index % 4096 == 0 {
                check_cancelled(cancelled)?;
            }
            for channel in 0..3 {
                let detail = pixel[channel] - low[channel];
                pixel[channel] = (pixel[channel] - low[channel] * dehaze * 0.2)
                    / (1.0 - dehaze * 0.2).max(0.2)
                    + detail * texture;
            }
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

fn blur_linear(
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

fn normalized_wb_coefficients(
    raw: &rawler::RawImage,
    recipe: &crate::catalog::DevelopRecipe,
) -> [f32; 4] {
    if let Some(white_balance) = recipe.white_balance {
        absolute_white_balance_coefficients(raw, white_balance)
    } else {
        white_balance_coefficients(raw.wb_coeffs, raw.neutralwb(), recipe)
    }
}

fn white_balance_coefficients(
    as_shot: [f32; 4],
    neutral: [f32; 4],
    recipe: &crate::catalog::DevelopRecipe,
) -> [f32; 4] {
    let mut coefficients = if recipe.white_balance.is_some() {
        // Absolute Lightroom WB values describe a chromaticity, not generic
        // output-RGB gains. The caller replaces this result on the RawImage,
        // so the camera matrix is required to translate that chromaticity into
        // sensor-space multipliers. `neutral` is retained as a safe fallback
        // for invalid or unavailable camera calibration data.
        neutral
    } else {
        as_shot
    };
    if let Some(increment) = recipe.white_balance_increment {
        let temperature = (increment.temperature / 100.0).clamp(-1.0, 1.0);
        let tint = (increment.tint / 100.0).clamp(-1.0, 1.0);
        coefficients[0] *= 1.0 + temperature * 0.15 + tint * 0.05;
        coefficients[1] *= 1.0 - tint * 0.08;
        coefficients[2] *= 1.0 - temperature * 0.15 + tint * 0.05;
    }
    coefficients
}

fn absolute_white_balance_coefficients(
    raw: &rawler::RawImage,
    white_balance: crate::catalog::WhiteBalance,
) -> [f32; 4] {
    let (x, y) = tinted_white_point(white_balance.temperature, white_balance.tint);
    // Lightroom's tint scale is proprietary. Treat it as a small displacement
    // perpendicular to the daylight/Planckian locus: positive values move the
    // requested white point away from green and towards magenta.
    let xyz = [x / y, 1.0, (1.0 - x - y) / y];
    let matrix = nearest_color_matrix(raw, white_balance.temperature);
    let mut coefficients = [f32::NAN; 4];
    let components = matrix.as_ref().map_or(0, |matrix| matrix.len() / 3);
    for channel in 0..components.min(4) {
        let row = matrix
            .as_ref()
            .map(|matrix| &matrix[channel * 3..channel * 3 + 3])
            .expect("a selected matrix has this row");
        let response = row[0] * xyz[0] + row[1] * xyz[1] + row[2] * xyz[2];
        if !response.is_finite() || response <= 0.0 {
            return raw.wb_coeffs;
        }
        coefficients[channel] = response.recip();
    }
    let green = coefficients[1];
    if !green.is_finite() || green <= 0.0 {
        return raw.wb_coeffs;
    }
    for coefficient in &mut coefficients[..components.min(4)] {
        *coefficient /= green;
    }
    coefficients
}

fn nearest_color_matrix(raw: &rawler::RawImage, temperature: f32) -> Option<Vec<f32>> {
    use rawler::imgop::xyz::Illuminant;
    let temperature_for = |illuminant| match illuminant {
        Illuminant::A | Illuminant::Tungsten | Illuminant::IsoStudioTungsten => 2_856.0,
        Illuminant::D50 => 5_003.0,
        Illuminant::D55 => 5_503.0,
        Illuminant::D65 | Illuminant::Daylight | Illuminant::Flash => 6_504.0,
        Illuminant::D75 => 7_504.0,
        _ => 6_000.0,
    };
    raw.color_matrix
        .iter()
        .filter(|(_, matrix)| matches!(matrix.len(), 9 | 12))
        .min_by(|(left, _), (right, _)| {
            (temperature_for(**left) - temperature)
                .abs()
                .total_cmp(&(temperature_for(**right) - temperature).abs())
        })
        .map(|(_, matrix)| matrix.clone())
}

fn tinted_white_point(temperature: f32, tint: f32) -> (f32, f32) {
    let (x, y) = planckian_locus_xy_kang2002(temperature);
    let to_uv = |x: f32, y: f32| {
        let denominator = -2.0 * x + 12.0 * y + 3.0;
        (4.0 * x / denominator, 6.0 * y / denominator)
    };
    let (u, v) = to_uv(x, y);
    let delta = (temperature * 0.01).max(1.0);
    let (u0, v0) = to_uv(
        planckian_locus_xy_kang2002(temperature - delta).0,
        planckian_locus_xy_kang2002(temperature - delta).1,
    );
    let (u1, v1) = to_uv(
        planckian_locus_xy_kang2002(temperature + delta).0,
        planckian_locus_xy_kang2002(temperature + delta).1,
    );
    let length = ((u1 - u0).powi(2) + (v1 - v0).powi(2)).sqrt().max(1e-6);
    let amount = tint.clamp(-150.0, 150.0) * 0.00005;
    let u = u - (v1 - v0) / length * amount;
    let v = v + (u1 - u0) / length * amount;
    let denominator = 2.0 * u - 8.0 * v + 4.0;
    (3.0 * u / denominator, 2.0 * v / denominator)
}

/// Evaluates the piecewise polynomial approximation of the CIE 1931 xy
/// Planckian locus from:
///
/// B. Kang, O. Moon, C. Hong, H. Lee, B. Cho, and Y. Kim, "Design of
/// Advanced Color-Temperature Control System for HDTV Applications",
/// Journal of the Korean Physical Society 41(6), pp. 865-871, 2002.
/// https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART000987865
///
/// Independent reference implementation:
/// https://colour.readthedocs.io/en/master/_modules/colour/temperature/kang2002.html
///
/// The published fit is valid from 1667 K through 25000 K. This function
/// clamps inputs to that interval; it does not estimate CCT from arbitrary xy.
fn planckian_locus_xy_kang2002(temperature: f32) -> (f32, f32) {
    let temperature = temperature.clamp(1_667.0, 25_000.0);
    let t2 = temperature * temperature;
    let t3 = t2 * temperature;
    let x = if temperature <= 4_000.0 {
        -0.266_123_9e9 / t3 - 2.343_58e5 / t2 + 0.877_695_6e3 / temperature + 0.179_910
    } else {
        -3.025_846_9e9 / t3 + 2.107_038e6 / t2 + 0.222_634_7e3 / temperature + 0.240_390
    };
    let y = if temperature <= 2_222.0 {
        -1.106_381_4 * x.powi(3) - 1.348_110_2 * x.powi(2) + 2.185_558_3 * x - 0.202_196_83
    } else if temperature <= 4_000.0 {
        -0.954_947_6 * x.powi(3) - 1.374_185_9 * x.powi(2) + 2.091_37 * x - 0.167_488_67
    } else {
        3.081_758 * x.powi(3) - 5.873_387 * x.powi(2) + 3.751_129_9 * x - 0.370_014_83
    };
    (x, y)
}

fn apply_scene_linear_controls(
    pixels: &mut [[f32; 3]],
    recipe: &crate::catalog::DevelopRecipe,
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
            let after = super::develop::sample_curve(curve, before.clamp(0.0, 1.0));
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
                    super::develop::sample_curve(curve, pixel[channel].clamp(0.0, 1.0));
            }
        }
    }
    Ok(())
}

fn linear_rgb(
    intermediate: Intermediate,
    cancelled: &AtomicBool,
) -> Result<(u32, u32, Vec<[f32; 3]>), String> {
    check_cancelled(cancelled)?;
    Ok(match intermediate {
        Intermediate::Monochrome(pixels) => {
            let mut converted = Vec::with_capacity(pixels.data.len());
            for (index, value) in pixels.data.into_iter().enumerate() {
                if index % 4096 == 0 {
                    check_cancelled(cancelled)?;
                }
                converted.push([value, value, value]);
            }
            (pixels.width as u32, pixels.height as u32, converted)
        }
        Intermediate::ThreeColor(pixels) => {
            (pixels.width as u32, pixels.height as u32, pixels.data)
        }
        // Calibration normally converts four sensor colors to RGB. Retaining a
        // graceful fallback makes this boundary robust to unusual decoders.
        Intermediate::FourColor(pixels) => {
            let mut converted = Vec::with_capacity(pixels.data.len());
            for (index, value) in pixels.data.into_iter().enumerate() {
                if index % 4096 == 0 {
                    check_cancelled(cancelled)?;
                }
                converted.push([value[0], value[1], value[2]]);
            }
            (pixels.width as u32, pixels.height as u32, converted)
        }
    })
}

#[cfg(test)]
fn render_linear_rgb(
    width: u32,
    height: u32,
    pixels: Vec<[f32; 3]>,
    settings: &PhotographicSettings,
    cancelled: &AtomicBool,
) -> Result<(RgbImage, f32), String> {
    let automatic = exposure_multiplier(&pixels, settings.auto_exposure, cancelled)?;
    render_linear_rgb_with_exposure(width, height, pixels, settings, automatic, cancelled)
}

fn render_linear_rgb_with_exposure(
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
            let mapped = match settings.tone_mapper {
                ToneMapper::AcesFitted => aces_fitted(linear),
            };
            (linear_to_srgb(mapped) * 255.0).round().clamp(0.0, 255.0) as u8
        }));
    }
    Ok((
        RgbImage::from_raw(width, height, bytes)
            .unwrap_or_else(|| RgbImage::from_pixel(1, 1, Rgb([0; 3]))),
        automatic.log2(),
    ))
}

fn exposure_multiplier(
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

fn check_cancelled(cancelled: &AtomicBool) -> Result<(), String> {
    if cancelled.load(Ordering::Acquire) {
        Err("render cancelled".into())
    } else {
        Ok(())
    }
}

fn aces_fitted(value: f32) -> f32 {
    if !value.is_finite() {
        return if value.is_sign_positive() { 1.0 } else { 0.0 };
    }
    ((value * (2.51 * value + 0.03)) / (value * (2.43 * value + 0.59) + 0.14)).clamp(0.0, 1.0)
}

fn linear_to_srgb(value: f32) -> f32 {
    if value <= 0.003_130_8 {
        value * 12.92
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
