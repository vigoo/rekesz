use std::{
    path::Path,
    sync::{Arc, atomic::AtomicBool},
};

use image::DynamicImage;

use super::decoding_error;
use crate::rendering::{
    SourcePipeline,
    decoded::{DecodedRaw, DecodedSource},
};

pub(super) fn decode(
    path: &Path,
    cancelled: &Arc<AtomicBool>,
) -> image::ImageResult<DecodedSource> {
    decode_raw(path, cancelled)
        .map(|raw| DecodedSource {
            source_dimensions: raw.oriented_dimensions(),
            image: DynamicImage::new_rgb8(1, 1),
            source_pipeline: SourcePipeline::Raw,
            orientation_applied: raw.orientation.map(|value| value.to_exif()),
            raw_linear: Some(raw),
            photographic: None,
        })
        .map_err(decoding_error)
}

fn decode_raw(path: &Path, cancelled: &Arc<AtomicBool>) -> Result<DecodedRaw, String> {
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

fn check_cancelled(cancelled: &AtomicBool) -> Result<(), String> {
    use std::sync::atomic::Ordering;
    if cancelled.load(Ordering::Acquire) {
        Err("render cancelled".into())
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests;
