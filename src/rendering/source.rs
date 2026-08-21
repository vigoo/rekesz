use std::{
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use image::{DynamicImage, ImageBuffer, ImageDecoder, Luma, Rgb};

use super::{AppliedPhotographicTransform, PhotographicSettings, RenderVariant, SourcePipeline};

pub(crate) struct DecodedSource {
    pub image: DynamicImage,
    pub source_dimensions: (u32, u32),
    pub source_pipeline: SourcePipeline,
    pub photographic: Option<AppliedPhotographicTransform>,
    pub orientation_applied: Option<u8>,
}

pub(super) fn decode(
    path: &Path,
    raw_settings: &PhotographicSettings,
    variant: RenderVariant,
    cancelled: &Arc<AtomicBool>,
) -> image::ImageResult<DecodedSource> {
    check_cancelled(cancelled)?;
    if crate::formats::is_heif(path) {
        return decode_heif(path, cancelled)
            .map(|image| DecodedSource {
                source_dimensions: (image.width(), image.height()),
                image,
                source_pipeline: SourcePipeline::Heif,
                photographic: None,
                orientation_applied: None,
            })
            .map_err(decoding_error);
    }
    if crate::formats::is_raw(path) {
        return super::photographic::decode_raw(path, raw_settings, cancelled)
            .map(|(image, photographic)| DecodedSource {
                source_dimensions: (image.width(), image.height()),
                image,
                source_pipeline: SourcePipeline::Raw,
                photographic: Some(photographic),
                orientation_applied: photographic.orientation_applied,
            })
            .map_err(decoding_error);
    }

    if is_jpeg(path) && variant != RenderVariant::Full {
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
    let image = heif::decode(&bytes).map_err(|error| error.to_string())?;
    if cancelled.load(Ordering::Acquire) {
        Err("render cancelled".into())
    } else {
        Ok(image)
    }
}

fn decoding_error(error: String) -> image::ImageError {
    image::ImageError::Decoding(image::error::DecodingError::new(
        image::error::ImageFormatHint::Unknown,
        error,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgb};

    #[test]
    fn jpeg_thumbnail_decode_uses_reduced_resolution() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("large.jpg");
        DynamicImage::ImageRgb8(ImageBuffer::from_pixel(800, 600, Rgb([1, 2, 3])))
            .save(&path)
            .unwrap();
        let decoded = decode(
            &path,
            &PhotographicSettings::default(),
            RenderVariant::Thumbnail { width: 100 },
            &Arc::new(AtomicBool::new(false)),
        )
        .unwrap();

        assert_eq!(decoded.source_dimensions, (800, 600));
        assert!(decoded.image.width() < 800);
        assert!(decoded.image.height() < 600);
    }
}
