use image::DynamicImage;

use super::plan::ResizePlan;

/// Applies the baseline-render optimization before the develop boundary.
/// Developed renders must retain full source resolution until all controls run.
pub(super) fn before_develop(image: DynamicImage, plan: ResizePlan) -> DynamicImage {
    match plan {
        ResizePlan::Derivative {
            max_width,
            max_height,
            before_develop: true,
        } => image.thumbnail(max_width, max_height),
        ResizePlan::None | ResizePlan::Derivative { .. } => image,
    }
}

/// Produces the requested final derivative dimensions.
pub(super) fn output(image: DynamicImage, plan: ResizePlan) -> DynamicImage {
    match plan {
        ResizePlan::None => image,
        ResizePlan::Derivative {
            max_width,
            max_height,
            ..
        } => image.thumbnail(max_width, max_height),
    }
}

#[cfg(test)]
mod tests;
