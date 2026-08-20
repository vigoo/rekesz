use std::path::Path;

use image::{DynamicImage, ImageDecoder};

use super::{AppliedPhotographicTransform, PhotographicSettings, SourcePipeline};

pub(super) struct DecodedSource {
    pub image: DynamicImage,
    pub source_pipeline: SourcePipeline,
    pub photographic: Option<AppliedPhotographicTransform>,
    pub orientation_applied: Option<u8>,
}

pub(super) fn decode(
    path: &Path,
    raw_settings: &PhotographicSettings,
) -> image::ImageResult<DecodedSource> {
    if crate::formats::is_heif(path) {
        return decode_heif(path)
            .map(|image| DecodedSource {
                image,
                source_pipeline: SourcePipeline::Heif,
                photographic: None,
                orientation_applied: None,
            })
            .map_err(decoding_error);
    }
    if crate::formats::is_raw(path) {
        return super::photographic::decode_raw(path, raw_settings)
            .map(|(image, photographic)| DecodedSource {
                image,
                source_pipeline: SourcePipeline::Raw,
                photographic: Some(photographic),
                orientation_applied: photographic.orientation_applied,
            })
            .map_err(decoding_error);
    }

    let reader = image::ImageReader::open(path)?.with_guessed_format()?;
    let mut decoder = reader.into_decoder()?;
    let orientation = decoder.orientation()?;
    let mut image = DynamicImage::from_decoder(decoder)?;
    image.apply_orientation(orientation);
    Ok(DecodedSource {
        image,
        source_pipeline: SourcePipeline::Standard,
        photographic: None,
        orientation_applied: Some(orientation.to_exif()),
    })
}

fn decode_heif(path: &Path) -> Result<DynamicImage, String> {
    let bytes = std::fs::read(path).map_err(|error| error.to_string())?;
    heif::decode(&bytes).map_err(|error| error.to_string())
}

fn decoding_error(error: String) -> image::ImageError {
    image::ImageError::Decoding(image::error::DecodingError::new(
        image::error::ImageFormatHint::Unknown,
        error,
    ))
}
