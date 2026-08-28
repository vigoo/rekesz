use std::sync::{Arc, atomic::AtomicBool};

use image::DynamicImage;

use super::{
    AppliedPhotographicTransform, OutputEncoding, PhotographicSettings,
    colour::render_linear_rgb_with_exposure, rawler_stage, scene_linear::apply_recipe_scene_linear,
    white_balance,
};
use crate::{catalog::DevelopRecipe, rendering::decoded::DecodedRaw};

/// Coordinates the decoded RAW source, Rawler's compound stage, and Rekesz-owned
/// scene-linear, exposure, colour, tone, and orientation stages.
pub(in crate::rendering) fn render_raw(
    decoded: &DecodedRaw,
    settings: &PhotographicSettings,
    recipe: Option<&DevelopRecipe>,
    cancelled: &Arc<AtomicBool>,
) -> Result<(DynamicImage, AppliedPhotographicTransform), String> {
    let mut overridden;
    let raw = if let Some(recipe) = recipe
        && (recipe.white_balance.is_some() || recipe.white_balance_increment.is_some())
    {
        overridden = decoded.raw.clone();
        overridden.wb_coeffs = white_balance::normalized_coefficients(&decoded.raw, recipe);
        &overridden
    } else {
        &decoded.raw
    };
    let linear = rawler_stage::develop_linear(raw, cancelled)?;
    render_linear(linear, decoded.orientation, settings, recipe, cancelled)
}

fn render_linear(
    mut linear: rawler_stage::LinearRgb,
    orientation: Option<image::metadata::Orientation>,
    settings: &PhotographicSettings,
    recipe: Option<&DevelopRecipe>,
    cancelled: &Arc<AtomicBool>,
) -> Result<(DynamicImage, AppliedPhotographicTransform), String> {
    let recipe_exposure = recipe.and_then(|recipe| recipe.exposure_ev).unwrap_or(0.0);
    let developed_baseline = if recipe.is_some() { 0.75 } else { 0.0 };
    let automatic = apply_recipe_scene_linear(
        &mut linear.pixels,
        linear.width,
        linear.height,
        settings,
        recipe,
        developed_baseline,
        cancelled,
    )?;
    let mut adjusted = *settings;
    let (image, automatic_exposure_stops) = render_linear_rgb_with_exposure(
        linear.width,
        linear.height,
        linear.pixels,
        &adjusted,
        automatic,
        cancelled,
    )?;
    let mut image = DynamicImage::ImageRgb8(image);
    if let Some(orientation) = orientation {
        image.apply_orientation(orientation);
    }
    adjusted.exposure_compensation_stops += recipe_exposure;
    adjusted.exposure_compensation_stops += developed_baseline;
    Ok((
        image,
        AppliedPhotographicTransform {
            settings: adjusted,
            automatic_exposure_stops,
            total_exposure_stops: automatic_exposure_stops
                + settings.exposure_compensation_stops
                + recipe_exposure
                + developed_baseline,
            output_encoding: OutputEncoding::Srgb,
            orientation_applied: orientation.map(|value| value.to_exif()),
        },
    ))
}

#[cfg(test)]
mod tests;
