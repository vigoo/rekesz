use std::{
    path::Path,
    sync::{Arc, atomic::AtomicBool},
};

use image::{DynamicImage, ImageBuffer, ImageDecoder, Luma, Rgb};

use super::{check_cancelled, decoding_error};
use crate::rendering::decoded::DecodedSource;
use crate::rendering::{RenderVariant, SourcePipeline};

pub(super) fn decode(
    path: &Path,
    variant: RenderVariant,
    scaled_jpeg: bool,
    cancelled: &Arc<AtomicBool>,
) -> image::ImageResult<DecodedSource> {
    if scaled_jpeg {
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
        face_transform: crate::rendering::FaceTransform::orientation(orientation.to_exif()),
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
        RenderVariant::Full
        | RenderVariant::FaceCrop { .. }
        | RenderVariant::FaceContext { .. } => {
            unreachable!("full-resolution JPEGs use the standard decoder")
        }
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
        face_transform: orientation.map_or(crate::rendering::FaceTransform::IDENTITY, |value| {
            crate::rendering::FaceTransform::orientation(value.to_exif())
        }),
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
