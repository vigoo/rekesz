use std::{
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use image::{DynamicImage, ImageBuffer, ImageDecoder, Luma, Rgb, Rgba};
use libheif_rs::{ColorSpace, HeifContext, LibHeif, RgbChroma};

use super::{AppliedPhotographicTransform, PhotographicSettings, RenderVariant, SourcePipeline};

pub(crate) struct DecodedSource {
    pub image: DynamicImage,
    pub raw_linear: Option<super::photographic::DecodedRaw>,
    pub source_dimensions: (u32, u32),
    pub source_pipeline: SourcePipeline,
    pub photographic: Option<AppliedPhotographicTransform>,
    pub orientation_applied: Option<u8>,
}

pub(super) fn decode(
    path: &Path,
    _raw_settings: &PhotographicSettings,
    variant: RenderVariant,
    developed: bool,
    cancelled: &Arc<AtomicBool>,
) -> image::ImageResult<DecodedSource> {
    check_cancelled(cancelled)?;
    if crate::formats::is_heif(path) {
        return decode_heif(path, cancelled)
            .map(|image| DecodedSource {
                source_dimensions: (image.width(), image.height()),
                image,
                raw_linear: None,
                source_pipeline: SourcePipeline::Heif,
                photographic: None,
                orientation_applied: None,
            })
            .map_err(decoding_error);
    }
    if crate::formats::is_raw(path) {
        return super::photographic::decode_raw(path, cancelled)
            .map(|raw| DecodedSource {
                source_dimensions: raw.oriented_dimensions(),
                image: DynamicImage::new_rgb8(1, 1),
                source_pipeline: SourcePipeline::Raw,
                orientation_applied: raw.orientation.map(|value| value.to_exif()),
                raw_linear: Some(raw),
                photographic: None,
            })
            .map_err(decoding_error);
    }

    if is_jpeg(path) && variant != RenderVariant::Full && !developed {
        return decode_scaled_jpeg(path, variant, cancelled);
    }

    let reader = image::ImageReader::open(path)?.with_guessed_format()?;
    check_cancelled(cancelled)?;
    let mut decoder = reader.into_decoder()?;
    let orientation = decoder.orientation()?;
    let mut image = DynamicImage::from_decoder(decoder)?;
    check_cancelled(cancelled)?;
    image.apply_orientation(orientation);
    Ok(DecodedSource {
        source_dimensions: (image.width(), image.height()),
        image,
        raw_linear: None,
        source_pipeline: SourcePipeline::Standard,
        photographic: None,
        orientation_applied: Some(orientation.to_exif()),
    })
}

fn is_jpeg(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| {
            extension.eq_ignore_ascii_case("jpg") || extension.eq_ignore_ascii_case("jpeg")
        })
}

fn decode_scaled_jpeg(
    path: &Path,
    variant: RenderVariant,
    cancelled: &Arc<AtomicBool>,
) -> image::ImageResult<DecodedSource> {
    let file = std::fs::File::open(path)?;
    let mut decoder = jpeg_decoder::Decoder::new(std::io::BufReader::new(file));
    decoder
        .read_info()
        .map_err(|error| decoding_error(error.to_string()))?;
    let original = decoder
        .info()
        .ok_or_else(|| decoding_error("JPEG dimensions are unavailable".into()))?;
    let (requested_width, requested_height) = match variant {
        RenderVariant::Thumbnail { width } => (width, width),
        RenderVariant::Preview {
            max_width,
            max_height,
        } => (max_width, max_height),
        RenderVariant::Full => unreachable!("full JPEGs use the standard decoder"),
    };
    decoder
        .scale(
            requested_width.min(u32::from(u16::MAX)) as u16,
            requested_height.min(u32::from(u16::MAX)) as u16,
        )
        .map_err(|error| decoding_error(error.to_string()))?;
    check_cancelled(cancelled)?;
    let pixels = decoder
        .decode()
        .map_err(|error| decoding_error(error.to_string()))?;
    let info = decoder
        .info()
        .ok_or_else(|| decoding_error("decoded JPEG dimensions are unavailable".into()))?;
    let mut image = match info.pixel_format {
        jpeg_decoder::PixelFormat::L8 => DynamicImage::ImageLuma8(
            ImageBuffer::<Luma<u8>, _>::from_raw(info.width.into(), info.height.into(), pixels)
                .ok_or_else(|| decoding_error("invalid grayscale JPEG buffer".into()))?,
        ),
        jpeg_decoder::PixelFormat::RGB24 => DynamicImage::ImageRgb8(
            ImageBuffer::<Rgb<u8>, _>::from_raw(info.width.into(), info.height.into(), pixels)
                .ok_or_else(|| decoding_error("invalid RGB JPEG buffer".into()))?,
        ),
        format => {
            return Err(decoding_error(format!(
                "unsupported scaled JPEG pixel format: {format:?}"
            )));
        }
    };
    check_cancelled(cancelled)?;
    let orientation = decoder
        .exif_data()
        .and_then(|bytes| exif::Reader::new().read_raw(bytes.to_vec()).ok())
        .and_then(|exif| {
            exif.get_field(exif::Tag::Orientation, exif::In::PRIMARY)
                .and_then(|field| field.value.get_uint(0))
        })
        .and_then(exif_orientation);
    if let Some(orientation) = orientation {
        image.apply_orientation(orientation);
    }
    Ok(DecodedSource {
        image,
        raw_linear: None,
        source_dimensions: (original.width.into(), original.height.into()),
        source_pipeline: SourcePipeline::Standard,
        photographic: None,
        orientation_applied: orientation.map(image::metadata::Orientation::to_exif),
    })
}

fn exif_orientation(value: u32) -> Option<image::metadata::Orientation> {
    use image::metadata::Orientation;
    Some(match value {
        1 => Orientation::NoTransforms,
        2 => Orientation::FlipHorizontal,
        3 => Orientation::Rotate180,
        4 => Orientation::FlipVertical,
        5 => Orientation::Rotate90FlipH,
        6 => Orientation::Rotate90,
        7 => Orientation::Rotate270FlipH,
        8 => Orientation::Rotate270,
        _ => return None,
    })
}

fn check_cancelled(cancelled: &AtomicBool) -> image::ImageResult<()> {
    if cancelled.load(Ordering::Acquire) {
        Err(decoding_error("render cancelled".into()))
    } else {
        Ok(())
    }
}

fn decode_heif(path: &Path, cancelled: &Arc<AtomicBool>) -> Result<DynamicImage, String> {
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

fn decoding_error(error: String) -> image::ImageError {
    image::ImageError::Decoding(image::error::DecodingError::new(
        image::error::ImageFormatHint::Unknown,
        error,
    ))
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
