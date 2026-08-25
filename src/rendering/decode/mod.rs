use std::sync::{Arc, atomic::AtomicBool};

use super::decoded::DecodedSource;

mod heif;
mod raw;
mod standard;

pub(super) fn decode(
    plan: &super::plan::RenderPlan,
    cancelled: &Arc<AtomicBool>,
) -> image::ImageResult<DecodedSource> {
    check_cancelled(cancelled)?;
    match plan.decode() {
        super::plan::DecodePlan::Heif => heif::decode(plan.request().path.as_path(), cancelled),
        super::plan::DecodePlan::Raw => raw::decode(plan.request().path.as_path(), cancelled),
        super::plan::DecodePlan::Standard { scaled_jpeg } => standard::decode(
            plan.request().path.as_path(),
            plan.request().variant,
            scaled_jpeg,
            cancelled,
        ),
    }
}

fn check_cancelled(cancelled: &AtomicBool) -> image::ImageResult<()> {
    use std::sync::atomic::Ordering;
    if cancelled.load(Ordering::Acquire) {
        Err(decoding_error("render cancelled".into()))
    } else {
        Ok(())
    }
}

fn decoding_error(error: String) -> image::ImageError {
    image::ImageError::Decoding(image::error::DecodingError::new(
        image::error::ImageFormatHint::Unknown,
        error,
    ))
}

#[cfg(test)]
mod tests;
