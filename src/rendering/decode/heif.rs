use std::{
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use image::{DynamicImage, ImageBuffer, Rgb, Rgba};
use libheif_rs::{ColorSpace, HeifContext, LibHeif, RgbChroma};

use super::decoding_error;
use crate::rendering::SourcePipeline;
use crate::rendering::decoded::DecodedSource;

pub(super) fn decode(
    path: &Path,
    cancelled: &Arc<AtomicBool>,
) -> image::ImageResult<DecodedSource> {
    decode_image(path, cancelled)
        .map(|image| DecodedSource {
            source_dimensions: (image.width(), image.height()),
            image,
            raw_linear: None,
            source_pipeline: SourcePipeline::Heif,
            photographic: None,
            orientation_applied: None,
        })
        .map_err(decoding_error)
}

fn decode_image(path: &Path, cancelled: &Arc<AtomicBool>) -> Result<DynamicImage, String> {
    let bytes = std::fs::read(path).map_err(|error| error.to_string())?;
    if cancelled.load(Ordering::Acquire) {
        return Err("render cancelled".into());
    }
    let context = HeifContext::read_from_bytes(&bytes).map_err(|error| error.to_string())?;
    let handle = context
        .primary_image_handle()
        .map_err(|error| error.to_string())?;
    let profile = handle
        .color_profile_raw()
        .filter(|profile| profile.data.len() <= 16 * 1024 * 1024)
        .map(|profile| HeifColorProfile::Icc(profile.data))
        .or_else(|| handle.color_profile_nclx().map(HeifColorProfile::Nclx));
    let has_alpha = handle.has_alpha_channel();
    let bit_depth = handle.luma_bits_per_pixel();
    let color_space = match (bit_depth > 8, has_alpha, cfg!(target_endian = "little")) {
        (false, false, _) => ColorSpace::Rgb(RgbChroma::Rgb),
        (false, true, _) => ColorSpace::Rgb(RgbChroma::Rgba),
        (true, false, true) => ColorSpace::Rgb(RgbChroma::HdrRgbLe),
        (true, false, false) => ColorSpace::Rgb(RgbChroma::HdrRgbBe),
        (true, true, true) => ColorSpace::Rgb(RgbChroma::HdrRgbaLe),
        (true, true, false) => ColorSpace::Rgb(RgbChroma::HdrRgbaBe),
    };
    let decoded = LibHeif::new()
        .decode(&handle, color_space, None)
        .map_err(|error| error.to_string())?;
    let plane = decoded
        .planes()
        .interleaved
        .ok_or_else(|| "decoded HEIF has no interleaved RGB plane".to_owned())?;
    let channels = if has_alpha { 4 } else { 3 };
    let sample_bytes = if bit_depth > 8 { 2 } else { 1 };
    let row_bytes = (plane.width as usize)
        .checked_mul(channels)
        .and_then(|value| value.checked_mul(sample_bytes))
        .ok_or_else(|| "decoded HEIF dimensions overflow".to_owned())?;
    if row_bytes > plane.stride {
        return Err("decoded HEIF RGB row exceeds its plane stride".into());
    }
    let capacity = row_bytes
        .checked_mul(plane.height as usize)
        .ok_or_else(|| "decoded HEIF dimensions overflow".to_owned())?;
    if plane.data.len() < plane.stride.saturating_mul(plane.height as usize) {
        return Err("decoded HEIF plane is truncated".into());
    }
    let mut pixels = Vec::with_capacity(capacity);
    for row in plane
        .data
        .chunks_exact(plane.stride)
        .take(plane.height as usize)
    {
        if cancelled.load(Ordering::Acquire) {
            return Err("render cancelled".into());
        }
        pixels.extend_from_slice(&row[..row_bytes]);
    }
    let image = if sample_bytes == 1 {
        convert_profile_to_srgb_8(&mut pixels, channels, profile.as_ref());
        if has_alpha {
            DynamicImage::ImageRgba8(
                ImageBuffer::<Rgba<u8>, _>::from_raw(plane.width, plane.height, pixels)
                    .ok_or_else(|| "invalid decoded HEIF RGBA buffer".to_owned())?,
            )
        } else {
            DynamicImage::ImageRgb8(
                ImageBuffer::<Rgb<u8>, _>::from_raw(plane.width, plane.height, pixels)
                    .ok_or_else(|| "invalid decoded HEIF RGB buffer".to_owned())?,
            )
        }
    } else {
        let shift = 16_u8.saturating_sub(bit_depth);
        let mut samples: Vec<u16> = pixels
            .as_chunks::<2>()
            .0
            .iter()
            .map(|bytes| u16::from_ne_bytes([bytes[0], bytes[1]]) << shift)
            .collect();
        convert_profile_to_srgb_16(&mut samples, channels, profile.as_ref());
        if has_alpha {
            DynamicImage::ImageRgba16(
                ImageBuffer::<Rgba<u16>, _>::from_raw(plane.width, plane.height, samples)
                    .ok_or_else(|| "invalid decoded HEIF 16-bit RGBA buffer".to_owned())?,
            )
        } else {
            DynamicImage::ImageRgb16(
                ImageBuffer::<Rgb<u16>, _>::from_raw(plane.width, plane.height, samples)
                    .ok_or_else(|| "invalid decoded HEIF 16-bit RGB buffer".to_owned())?,
            )
        }
    };
    if cancelled.load(Ordering::Acquire) {
        Err("render cancelled".into())
    } else {
        Ok(image)
    }
}

enum HeifColorProfile {
    Icc(Vec<u8>),
    Nclx(libheif_rs::ColorProfileNCLX),
}

fn mox_profile(profile: &HeifColorProfile) -> Result<moxcms::ColorProfile, String> {
    match profile {
        HeifColorProfile::Icc(data) => {
            moxcms::ColorProfile::new_from_slice(data).map_err(|error| error.to_string())
        }
        HeifColorProfile::Nclx(nclx) => {
            let primaries = moxcms::CicpColorPrimaries::try_from(nclx.color_primaries() as u8)
                .map_err(|error| error.to_string())?;
            let transfer =
                moxcms::TransferCharacteristics::try_from(nclx.transfer_characteristics() as u8)
                    .map_err(|error| error.to_string())?;
            let matrix = moxcms::MatrixCoefficients::try_from(nclx.matrix_coefficients() as u8)
                .map_err(|error| error.to_string())?;
            Ok(moxcms::ColorProfile::new_from_cicp(moxcms::CicpProfile {
                color_primaries: primaries,
                transfer_characteristics: transfer,
                matrix_coefficients: matrix,
                full_range: nclx.full_range_flag() != 0,
            }))
        }
    }
}

fn convert_profile_to_srgb_8(
    pixels: &mut [u8],
    channels: usize,
    profile: Option<&HeifColorProfile>,
) {
    let Some(source) = profile.and_then(|profile| mox_profile(profile).ok()) else {
        return;
    };
    let destination = moxcms::ColorProfile::new_srgb();
    let layout = if channels == 4 {
        moxcms::Layout::Rgba
    } else {
        moxcms::Layout::Rgb
    };
    let Ok(transform) =
        source.create_transform_8bit(layout, &destination, layout, Default::default())
    else {
        return;
    };
    let source_pixels = pixels.to_vec();
    let _ = transform.transform(&source_pixels, pixels);
}

fn convert_profile_to_srgb_16(
    pixels: &mut [u16],
    channels: usize,
    profile: Option<&HeifColorProfile>,
) {
    let Some(source) = profile.and_then(|profile| mox_profile(profile).ok()) else {
        return;
    };
    let destination = moxcms::ColorProfile::new_srgb();
    let layout = if channels == 4 {
        moxcms::Layout::Rgba
    } else {
        moxcms::Layout::Rgb
    };
    let Ok(transform) =
        source.create_transform_16bit(layout, &destination, layout, Default::default())
    else {
        return;
    };
    let source_pixels = pixels.to_vec();
    let _ = transform.transform(&source_pixels, pixels);
}
