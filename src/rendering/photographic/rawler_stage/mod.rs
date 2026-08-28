use std::sync::{Arc, atomic::AtomicBool};

use rawler::imgop::develop::{Intermediate, ProcessingStep, RawDevelop};

use super::check_cancelled;

pub(super) struct LinearRgb {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<[f32; 3]>,
}

/// Runs the indivisible Rawler-owned photographic stage and crosses from
/// Rawler's intermediate representation into Rekesz's scene-linear RGB boundary.
pub(super) fn develop_linear(
    raw: &rawler::RawImage,
    cancelled: &Arc<AtomicBool>,
) -> Result<LinearRgb, String> {
    let develop = RawDevelop::new_with(&[
        ProcessingStep::Rescale,
        ProcessingStep::Demosaic,
        ProcessingStep::FujiRotate,
        ProcessingStep::CropActiveArea,
        ProcessingStep::WhiteBalance,
        ProcessingStep::Calibrate,
        ProcessingStep::CropDefault,
    ]);
    let token = rawler::cancel::CancellationToken::from_atomic(cancelled.clone());
    let intermediate = develop
        .develop_intermediate_cancellable(raw, &token)
        .map_err(|error| error.to_string())?;
    linear_rgb(intermediate, cancelled)
}

fn linear_rgb(intermediate: Intermediate, cancelled: &AtomicBool) -> Result<LinearRgb, String> {
    check_cancelled(cancelled)?;
    let (width, height, pixels) = match intermediate {
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
    };
    Ok(LinearRgb {
        width,
        height,
        pixels,
    })
}
