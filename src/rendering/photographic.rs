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

pub(super) fn decode_raw(
    path: &Path,
    settings: &PhotographicSettings,
    cancelled: &Arc<AtomicBool>,
) -> Result<(DynamicImage, AppliedPhotographicTransform), String> {
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
    // Keep the result scene-linear. Display encoding belongs after exposure and
    // tone mapping, not inside camera-specific RAW development.
    let develop = RawDevelop::new_with(&[
        ProcessingStep::Rescale,
        ProcessingStep::Demosaic,
        ProcessingStep::FujiRotate,
        ProcessingStep::CropActiveArea,
        ProcessingStep::WhiteBalance,
        ProcessingStep::Calibrate,
        ProcessingStep::CropDefault,
    ]);
    let intermediate = develop
        .develop_intermediate_cancellable(&raw, &rawler_cancellation)
        .map_err(|error| error.to_string())?;
    check_cancelled(cancelled)?;
    let (width, height, pixels) = linear_rgb(intermediate, cancelled)?;
    let (image, automatic_exposure_stops) =
        render_linear_rgb(width, height, pixels, settings, cancelled)?;
    let mut image = DynamicImage::ImageRgb8(image);
    check_cancelled(cancelled)?;
    if let Some(orientation) = orientation {
        image.apply_orientation(orientation);
    }
    check_cancelled(cancelled)?;
    Ok((
        image,
        AppliedPhotographicTransform {
            settings: *settings,
            automatic_exposure_stops,
            total_exposure_stops: automatic_exposure_stops + settings.exposure_compensation_stops,
            output_encoding: OutputEncoding::Srgb,
            orientation_applied: orientation.map(|value| value.to_exif()),
        },
    ))
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

fn render_linear_rgb(
    width: u32,
    height: u32,
    pixels: Vec<[f32; 3]>,
    settings: &PhotographicSettings,
    cancelled: &AtomicBool,
) -> Result<(RgbImage, f32), String> {
    let automatic = exposure_multiplier(&pixels, settings.auto_exposure, cancelled)?;
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
mod tests {
    use super::*;

    #[test]
    fn auto_exposure_places_middle_grey_at_the_target() {
        let pixels = vec![[0.045; 3]; 100];
        let multiplier =
            exposure_multiplier(&pixels, AutoExposure::default(), &AtomicBool::new(false)).unwrap();
        assert!((multiplier - 4.0).abs() < 0.1);
    }

    #[test]
    fn auto_exposure_is_bounded_for_nearly_black_scenes() {
        let pixels = vec![[0.000_001; 3]];
        assert_eq!(
            exposure_multiplier(&pixels, AutoExposure::default(), &AtomicBool::new(false)).unwrap(),
            16.0
        );
    }

    #[test]
    fn tone_mapping_is_finite_and_bounded() {
        for value in [0.0, 0.18, 1.0, 10.0, f32::INFINITY] {
            let mapped = aces_fitted(value);
            assert!(mapped.is_finite() && (0.0..=1.0).contains(&mapped));
        }
    }

    #[test]
    fn photographic_transform_observes_cancellation() {
        let pixels = vec![[0.18; 3]; 10_000];
        assert!(
            render_linear_rgb(
                100,
                100,
                pixels,
                &PhotographicSettings::default(),
                &AtomicBool::new(true),
            )
            .is_err()
        );
    }
}
