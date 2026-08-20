//! Camera-independent photographic rendering of scene-linear RGB.

use std::path::Path;

use image::{DynamicImage, Rgb, RgbImage};
use rawler::imgop::develop::{Intermediate, ProcessingStep, RawDevelop};

use super::{AppliedPhotographicTransform, OutputEncoding};

const HISTOGRAM_BINS: usize = 4096;
const MIN_LOG2_LUMINANCE: f32 = -16.0;
const MAX_LOG2_LUMINANCE: f32 = 8.0;

#[derive(Clone, Copy, Debug, PartialEq)]
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

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum ToneMapper {
    #[default]
    AcesFitted,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PhotographicSettings {
    pub auto_exposure: AutoExposure,
    pub exposure_compensation_stops: f32,
    pub tone_mapper: ToneMapper,
}

pub(super) fn decode_raw(
    path: &Path,
    settings: &PhotographicSettings,
) -> Result<(DynamicImage, AppliedPhotographicTransform), String> {
    use rawler::{decoders::RawDecodeParams, rawsource::RawSource};

    let source = RawSource::new(path).map_err(|error| error.to_string())?;
    let decoder = rawler::global_loader()
        .get_decoder(&source)
        .map_err(|error| error.to_string())?;
    let params = RawDecodeParams::default();
    let metadata = decoder
        .raw_metadata(&source, &params)
        .map_err(|error| error.to_string())?;
    let raw = decoder
        .raw_image(&source, &params, false)
        .map_err(|error| error.to_string())?;
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
        .develop_intermediate(&raw)
        .map_err(|error| error.to_string())?;
    let (width, height, pixels) = linear_rgb(intermediate);
    let (image, automatic_exposure_stops) = render_linear_rgb(width, height, pixels, settings);
    let mut image = DynamicImage::ImageRgb8(image);
    if let Some(orientation) = orientation {
        image.apply_orientation(orientation);
    }
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

fn linear_rgb(intermediate: Intermediate) -> (u32, u32, Vec<[f32; 3]>) {
    match intermediate {
        Intermediate::Monochrome(pixels) => (
            pixels.width as u32,
            pixels.height as u32,
            pixels.data.into_iter().map(|v| [v, v, v]).collect(),
        ),
        Intermediate::ThreeColor(pixels) => {
            (pixels.width as u32, pixels.height as u32, pixels.data)
        }
        // Calibration normally converts four sensor colors to RGB. Retaining a
        // graceful fallback makes this boundary robust to unusual decoders.
        Intermediate::FourColor(pixels) => (
            pixels.width as u32,
            pixels.height as u32,
            pixels
                .data
                .into_iter()
                .map(|v| [v[0], v[1], v[2]])
                .collect(),
        ),
    }
}

fn render_linear_rgb(
    width: u32,
    height: u32,
    pixels: Vec<[f32; 3]>,
    settings: &PhotographicSettings,
) -> (RgbImage, f32) {
    let automatic = exposure_multiplier(&pixels, settings.auto_exposure);
    let exposure = automatic * 2.0_f32.powf(settings.exposure_compensation_stops);
    let bytes = pixels
        .into_iter()
        .flat_map(|pixel| {
            pixel.map(|channel| {
                let linear = (channel.max(0.0) * exposure).max(0.0);
                let mapped = match settings.tone_mapper {
                    ToneMapper::AcesFitted => aces_fitted(linear),
                };
                (linear_to_srgb(mapped) * 255.0).round().clamp(0.0, 255.0) as u8
            })
        })
        .collect();
    (
        RgbImage::from_raw(width, height, bytes)
            .unwrap_or_else(|| RgbImage::from_pixel(1, 1, Rgb([0; 3]))),
        automatic.log2(),
    )
}

fn exposure_multiplier(pixels: &[[f32; 3]], settings: AutoExposure) -> f32 {
    let mut histogram = [0_u32; HISTOGRAM_BINS];
    let mut samples = 0_u64;
    // At most roughly one million samples; the full-resolution transform still
    // visits every pixel, while statistics stay cheap for very large sensors.
    let stride = pixels.len().div_ceil(1_000_000).max(1);
    for pixel in pixels.iter().step_by(stride) {
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
        return 1.0;
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
    stops.exp2()
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
        let multiplier = exposure_multiplier(&pixels, AutoExposure::default());
        assert!((multiplier - 4.0).abs() < 0.1);
    }

    #[test]
    fn auto_exposure_is_bounded_for_nearly_black_scenes() {
        let pixels = vec![[0.000_001; 3]];
        assert_eq!(exposure_multiplier(&pixels, AutoExposure::default()), 16.0);
    }

    #[test]
    fn tone_mapping_is_finite_and_bounded() {
        for value in [0.0, 0.18, 1.0, 10.0, f32::INFINITY] {
            let mapped = aces_fitted(value);
            assert!(mapped.is_finite() && (0.0..=1.0).contains(&mapped));
        }
    }
}
