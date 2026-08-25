use std::sync::atomic::{AtomicBool, Ordering};

use image::{DynamicImage, Rgba, RgbaImage};

use crate::catalog::DevelopRecipe;

use operations::{
    Clarity, CropAction, Detail, DevelopAction, Global, Grain, Heal, Homography, LensDistortion,
    Monochrome, ParametricCurve, Presence, Vignette,
};

mod operations;

pub mod actions {
    pub const EXPOSURE: u64 = 1 << 0;
    pub const BASIC_TONE: u64 = 1 << 1;
    pub const COLOR: u64 = 1 << 2;
    pub const WHITE_BALANCE: u64 = 1 << 3;
    pub const TONE_CURVE: u64 = 1 << 4;
    pub const CLARITY: u64 = 1 << 5;
    pub const CROP: u64 = 1 << 6;
    pub const DETAIL: u64 = 1 << 7;
    pub const TEXTURE_DEHAZE: u64 = 1 << 8;
    pub const MONOCHROME: u64 = 1 << 9;
    pub const VIGNETTE: u64 = 1 << 10;
    pub const PARAMETRIC_CURVE: u64 = 1 << 11;
    pub const UPRIGHT: u64 = 1 << 12;
    pub const LENS: u64 = 1 << 13;
    pub const HEAL: u64 = 1 << 14;
    pub const GRAIN: u64 = 1 << 15;
}

pub mod warnings {
    /// Pics implements a visually similar operator rather than Adobe's private one.
    pub const APPROXIMATED: u64 = 1 << 0;
}

pub fn apply(
    source: &DynamicImage,
    recipe: &DevelopRecipe,
    cancelled: &AtomicBool,
) -> image::ImageResult<(DynamicImage, u64, u64)> {
    check(cancelled)?;
    let mut image = source.to_rgba8();
    let mut applied = 0;
    let mut fidelity = 0;

    if let Some(action) = Global::from_recipe(recipe) {
        execute(&action, &mut image, cancelled, &mut applied, &mut fidelity)?;
    }
    if let Some(amount) = recipe.clarity {
        execute(
            &Clarity { amount },
            &mut image,
            cancelled,
            &mut applied,
            &mut fidelity,
        )?;
    }
    if let Some(parameters) = recipe.detail {
        execute(
            &Detail { parameters },
            &mut image,
            cancelled,
            &mut applied,
            &mut fidelity,
        )?;
    }
    if recipe.texture.is_some() || recipe.dehaze.is_some() {
        execute(
            &Presence {
                texture: recipe.texture,
                dehaze: recipe.dehaze,
            },
            &mut image,
            cancelled,
            &mut applied,
            &mut fidelity,
        )?;
    }
    if let Some(parameters) = recipe.monochrome {
        execute(
            &Monochrome { parameters },
            &mut image,
            cancelled,
            &mut applied,
            &mut fidelity,
        )?;
    }
    if let Some(controls) = recipe.parametric_curve {
        execute(
            &ParametricCurve { controls },
            &mut image,
            cancelled,
            &mut applied,
            &mut fidelity,
        )?;
    }
    if !recipe.heal_spots.is_empty() {
        execute(
            &Heal {
                spots: &recipe.heal_spots,
            },
            &mut image,
            cancelled,
            &mut applied,
            &mut fidelity,
        )?;
    }
    if let Some(amount) = recipe.lens_distortion {
        execute(
            &LensDistortion { amount },
            &mut image,
            cancelled,
            &mut applied,
            &mut fidelity,
        )?;
    }
    if let Some(matrix) = recipe.upright_homography {
        execute(
            &Homography { matrix },
            &mut image,
            cancelled,
            &mut applied,
            &mut fidelity,
        )?;
    }
    if let Some(parameters) = recipe.crop {
        execute(
            &CropAction { parameters },
            &mut image,
            cancelled,
            &mut applied,
            &mut fidelity,
        )?;
    }
    if let Some(parameters) = recipe.vignette {
        execute(
            &Vignette { parameters },
            &mut image,
            cancelled,
            &mut applied,
            &mut fidelity,
        )?;
    }
    if let Some(parameters) = recipe.grain {
        execute(
            &Grain { parameters },
            &mut image,
            cancelled,
            &mut applied,
            &mut fidelity,
        )?;
    }
    check(cancelled)?;
    Ok((DynamicImage::ImageRgba8(image), applied, fidelity))
}

fn execute(
    action: &impl DevelopAction,
    image: &mut RgbaImage,
    cancelled: &AtomicBool,
    applied: &mut u64,
    fidelity: &mut u64,
) -> image::ImageResult<()> {
    action.apply(image, cancelled)?;
    *applied |= action.action_mask();
    *fidelity |= action.fidelity_mask();
    Ok(())
}

fn cancellable_box_blur(
    image: &RgbaImage,
    radius: u32,
    cancelled: &AtomicBool,
) -> image::ImageResult<RgbaImage> {
    let mut horizontal = RgbaImage::new(image.width(), image.height());
    for y in 0..image.height() {
        if y % 8 == 0 {
            check(cancelled)?;
        }
        for x in 0..image.width() {
            let mut sum = [0_u32; 4];
            let mut count = 0_u32;
            for sample_x in x.saturating_sub(radius)..=(x + radius).min(image.width() - 1) {
                let pixel = image.get_pixel(sample_x, y);
                for channel in 0..4 {
                    sum[channel] += u32::from(pixel[channel]);
                }
                count += 1;
            }
            horizontal.put_pixel(x, y, Rgba(sum.map(|value| (value / count) as u8)));
        }
    }
    let mut output = RgbaImage::new(image.width(), image.height());
    for y in 0..image.height() {
        if y % 8 == 0 {
            check(cancelled)?;
        }
        for x in 0..image.width() {
            let mut sum = [0_u32; 4];
            let mut count = 0_u32;
            for sample_y in y.saturating_sub(radius)..=(y + radius).min(image.height() - 1) {
                let pixel = horizontal.get_pixel(x, sample_y);
                for channel in 0..4 {
                    sum[channel] += u32::from(pixel[channel]);
                }
                count += 1;
            }
            output.put_pixel(x, y, Rgba(sum.map(|value| (value / count) as u8)));
        }
    }
    Ok(output)
}

fn rotate_same_bounds(
    image: &RgbaImage,
    angle_degrees: f32,
    cancelled: &AtomicBool,
) -> image::ImageResult<RgbaImage> {
    let mut output = RgbaImage::new(image.width(), image.height());
    let angle = angle_degrees.to_radians();
    let (sin, cos) = angle.sin_cos();
    let cx = (image.width() as f32 - 1.0) * 0.5;
    let cy = (image.height() as f32 - 1.0) * 0.5;
    for y in 0..image.height() {
        if y % 16 == 0 {
            check(cancelled)?;
        }
        for x in 0..image.width() {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            let sx = cos * dx + sin * dy + cx;
            let sy = -sin * dx + cos * dy + cy;
            if sx >= 0.0 && sy >= 0.0 && sx < image.width() as f32 && sy < image.height() as f32 {
                output.put_pixel(x, y, bilinear_sample(image, sx, sy));
            }
        }
    }
    Ok(output)
}

fn bilinear_sample(image: &RgbaImage, x: f32, y: f32) -> Rgba<u8> {
    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let x1 = (x0 + 1).min(image.width() - 1);
    let y1 = (y0 + 1).min(image.height() - 1);
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    let samples = [
        (image.get_pixel(x0, y0), (1.0 - fx) * (1.0 - fy)),
        (image.get_pixel(x1, y0), fx * (1.0 - fy)),
        (image.get_pixel(x0, y1), (1.0 - fx) * fy),
        (image.get_pixel(x1, y1), fx * fy),
    ];
    let mut output = [0_u8; 4];
    for channel in 0..4 {
        output[channel] = samples
            .iter()
            .map(|(pixel, weight)| f32::from(pixel[channel]) * weight)
            .sum::<f32>()
            .round()
            .clamp(0.0, 255.0) as u8;
    }
    Rgba(output)
}

pub(super) fn sample_curve(points: &[[f32; 2]], value: f32) -> f32 {
    let Some(first) = points.first() else {
        return value;
    };
    if value <= first[0] {
        return first[1];
    }
    for pair in points.windows(2) {
        if value <= pair[1][0] {
            let width = (pair[1][0] - pair[0][0]).max(f32::EPSILON);
            let t = (value - pair[0][0]) / width;
            return pair[0][1] + (pair[1][1] - pair[0][1]) * t;
        }
    }
    points.last().map_or(value, |point| point[1])
}

fn luminance(rgb: [f32; 3]) -> f32 {
    0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
}

fn srgb_to_linear(value: u8) -> f32 {
    let value = f32::from(value) / 255.0;
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb(value: f32) -> u8 {
    let value = value.clamp(0.0, 1.0);
    let encoded = if value <= 0.003_130_8 {
        value * 12.92
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    };
    (encoded * 255.0).round() as u8
}

fn check(cancelled: &AtomicBool) -> image::ImageResult<()> {
    if cancelled.load(Ordering::Acquire) {
        Err(decode_error("render cancelled"))
    } else {
        Ok(())
    }
}

fn decode_error(message: &str) -> image::ImageError {
    image::ImageError::Decoding(image::error::DecodingError::new(
        image::error::ImageFormatHint::Unknown,
        message.to_owned(),
    ))
}

#[cfg(test)]
mod tests;
