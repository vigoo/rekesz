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
    analysis_memory_budget: Option<usize>,
) -> image::ImageResult<DecodedSource> {
    decode_raw(path, cancelled, analysis_memory_budget)
        .map(|raw| DecodedSource {
            source_dimensions: raw.oriented_dimensions(),
            image: DynamicImage::new_rgb8(1, 1),
            source_pipeline: SourcePipeline::Raw,
            orientation_applied: raw.orientation.map(|value| value.to_exif()),
            face_transform: raw
                .orientation
                .map_or(crate::rendering::FaceTransform::IDENTITY, |value| {
                    crate::rendering::FaceTransform::orientation(value.to_exif())
                }),
            raw_linear: Some(raw),
            photographic: None,
        })
        .map_err(decoding_error)
}

fn decode_raw(
    path: &Path,
    cancelled: &Arc<AtomicBool>,
    analysis_memory_budget: Option<usize>,
) -> Result<DecodedRaw, String> {
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
    // Rawler's dummy decode parses the container and reports sensor geometry
    // without allocating/decompressing the pixel plane. Admit the real decode
    // only after proving its worst-case u16 RGB footprint is bounded.
    let probe = decoder
        .raw_image(&source, &params, true)
        .map_err(|error| error.to_string())?;
    if !raw_decode_admitted(probe.width, probe.height, probe.cpp, analysis_memory_budget) {
        return Err("RAW dimensions exceed the decoder allocation budget".into());
    }
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

fn raw_decode_admitted(
    width: usize,
    height: usize,
    components: usize,
    budget: Option<usize>,
) -> bool {
    budget.is_none_or(|budget| raw_allocation_within_budget(width, height, components, budget))
}

fn raw_allocation_within_budget(
    width: usize,
    height: usize,
    components: usize,
    budget: usize,
) -> bool {
    let Some(pixels) = width.checked_mul(height) else {
        return false;
    };
    let Some(mosaic) = pixels
        .checked_mul(components.max(1))
        .and_then(|value| value.checked_mul(std::mem::size_of::<u16>()))
    else {
        return false;
    };
    // Raw mosaic, demosaiced scene-linear RGB and concurrent photographic
    // intermediates, final RGBA derivative, plus bounded analysis output.
    let Some(peak) = pixels
        .checked_mul(64)
        .and_then(|working| working.checked_add(mosaic))
    else {
        return false;
    };
    let analysis_output = 2048_usize * 2048 * 4;
    peak.checked_add(analysis_output)
        .is_some_and(|peak| peak <= budget)
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
