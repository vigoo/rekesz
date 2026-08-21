use std::sync::atomic::{AtomicBool, Ordering};

use image::{DynamicImage, Rgba, RgbaImage, imageops};

use crate::catalog::{Crop, DevelopRecipe};

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

    if recipe.exposure_ev.is_some() {
        applied |= actions::EXPOSURE;
    }
    if recipe.contrast.is_some()
        || recipe.highlights.is_some()
        || recipe.shadows.is_some()
        || recipe.whites.is_some()
        || recipe.blacks.is_some()
    {
        applied |= actions::BASIC_TONE;
        fidelity |= warnings::APPROXIMATED;
    }
    if recipe.saturation.is_some() || recipe.vibrance.is_some() {
        applied |= actions::COLOR;
        fidelity |= warnings::APPROXIMATED;
    }
    if recipe.white_balance.is_some() || recipe.white_balance_increment.is_some() {
        applied |= actions::WHITE_BALANCE;
        fidelity |= warnings::APPROXIMATED;
    }
    if recipe.tone_curve.is_some()
        || recipe.tone_curve_red.is_some()
        || recipe.tone_curve_green.is_some()
        || recipe.tone_curve_blue.is_some()
    {
        applied |= actions::TONE_CURVE;
    }

    if applied != 0 {
        apply_global(&mut image, recipe, cancelled)?;
    }
    if let Some(amount) = recipe.clarity {
        image = apply_clarity(image, amount, cancelled)?;
        applied |= actions::CLARITY;
        fidelity |= warnings::APPROXIMATED;
    }
    if let Some(detail) = recipe.detail {
        image = apply_detail(image, detail, cancelled)?;
        applied |= actions::DETAIL;
        fidelity |= warnings::APPROXIMATED;
    }
    if recipe.texture.is_some() || recipe.dehaze.is_some() {
        apply_presence(&mut image, recipe.texture, recipe.dehaze, cancelled)?;
        applied |= actions::TEXTURE_DEHAZE;
        fidelity |= warnings::APPROXIMATED;
    }
    if let Some(monochrome) = recipe.monochrome {
        apply_monochrome(&mut image, monochrome, cancelled)?;
        applied |= actions::MONOCHROME;
        fidelity |= warnings::APPROXIMATED;
    }
    if let Some(curve) = recipe.parametric_curve {
        apply_parametric_curve(&mut image, curve, cancelled)?;
        applied |= actions::PARAMETRIC_CURVE;
        fidelity |= warnings::APPROXIMATED;
    }
    if !recipe.heal_spots.is_empty() {
        apply_heal_spots(&mut image, &recipe.heal_spots, cancelled)?;
        applied |= actions::HEAL;
        fidelity |= warnings::APPROXIMATED;
    }
    if let Some(amount) = recipe.lens_distortion {
        image = apply_lens_distortion(&image, amount, cancelled)?;
        applied |= actions::LENS;
        fidelity |= warnings::APPROXIMATED;
    }
    if let Some(matrix) = recipe.upright_homography {
        image = apply_homography(&image, matrix, cancelled)?;
        applied |= actions::UPRIGHT;
        fidelity |= warnings::APPROXIMATED;
    }
    if let Some(crop) = recipe.crop {
        image = apply_crop(image, crop, cancelled)?;
        applied |= actions::CROP;
        fidelity |= warnings::APPROXIMATED;
    }
    if let Some(vignette) = recipe.vignette {
        apply_vignette(&mut image, vignette, cancelled)?;
        applied |= actions::VIGNETTE;
        fidelity |= warnings::APPROXIMATED;
    }
    if let Some(grain) = recipe.grain {
        apply_grain(&mut image, grain, cancelled)?;
        applied |= actions::GRAIN;
        fidelity |= warnings::APPROXIMATED;
    }
    check(cancelled)?;
    Ok((DynamicImage::ImageRgba8(image), applied, fidelity))
}

fn apply_detail(
    mut image: RgbaImage,
    detail: crate::catalog::DetailRecipe,
    cancelled: &AtomicBool,
) -> image::ImageResult<RgbaImage> {
    check(cancelled)?;
    if detail.luminance_noise_reduction > 0.0 || detail.color_noise_reduction > 0.0 {
        let radius = (1.0 + detail.color_smoothness / 50.0).round() as u32;
        let smooth = cancellable_box_blur(&image, radius.clamp(1, 3), cancelled)?;
        let luma_mix =
            (detail.luminance_noise_reduction / 100.0) * (1.0 - detail.luminance_detail / 200.0);
        let luma_contrast = detail.luminance_contrast / 100.0;
        let color_mix =
            (detail.color_noise_reduction / 100.0) * (0.5 + detail.color_smoothness / 200.0);
        let color_detail = detail.color_detail / 100.0;
        for (index, (pixel, low)) in image.pixels_mut().zip(smooth.pixels()).enumerate() {
            if index % 16_384 == 0 {
                check(cancelled)?;
            }
            let source_luma = 0.2126 * f32::from(pixel[0])
                + 0.7152 * f32::from(pixel[1])
                + 0.0722 * f32::from(pixel[2]);
            let low_luma = 0.2126 * f32::from(low[0])
                + 0.7152 * f32::from(low[1])
                + 0.0722 * f32::from(low[2]);
            let denoised_luma = source_luma * (1.0 - luma_mix)
                + low_luma * luma_mix
                + (source_luma - low_luma) * luma_contrast;
            for channel in 0..3 {
                let chroma = f32::from(pixel[channel]) - source_luma;
                let low_chroma = f32::from(low[channel]) - low_luma;
                let denoised_chroma = chroma * (1.0 - color_mix)
                    + low_chroma * color_mix
                    + (chroma - low_chroma) * color_detail * 0.25;
                pixel[channel] = (denoised_luma + denoised_chroma).round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    if detail.sharpening > 0.0 {
        let radius = detail.sharpen_radius.round().clamp(1.0, 3.0) as u32;
        let blurred = cancellable_box_blur(&image, radius, cancelled)?;
        let strength =
            (detail.sharpening / 100.0).clamp(0.0, 2.0) * (0.5 + detail.sharpen_detail / 100.0);
        let threshold = detail.edge_masking / 100.0 * 32.0;
        for (index, (pixel, low)) in image.pixels_mut().zip(blurred.pixels()).enumerate() {
            if index % 16_384 == 0 {
                check(cancelled)?;
            }
            let edge = (0..3)
                .map(|channel| (f32::from(pixel[channel]) - f32::from(low[channel])).abs())
                .sum::<f32>()
                / 3.0;
            let mask = ((edge - threshold) / 16.0).clamp(0.0, 1.0);
            for channel in 0..3 {
                pixel[channel] = (f32::from(pixel[channel])
                    + (f32::from(pixel[channel]) - f32::from(low[channel])) * strength * mask)
                    .round()
                    .clamp(0.0, 255.0) as u8;
            }
        }
    }
    Ok(image)
}

fn apply_presence(
    image: &mut RgbaImage,
    texture: Option<f32>,
    dehaze: Option<f32>,
    cancelled: &AtomicBool,
) -> image::ImageResult<()> {
    let texture = texture.unwrap_or(0.0) / 100.0;
    let dehaze = dehaze.unwrap_or(0.0) / 100.0;
    let local = cancellable_box_blur(image, 2, cancelled)?;
    for (index, (pixel, low)) in image.pixels_mut().zip(local.pixels()).enumerate() {
        if index % 16_384 == 0 {
            check(cancelled)?;
        }
        for channel in 0..3 {
            let value = f32::from(pixel[channel]) / 255.0;
            let low = f32::from(low[channel]) / 255.0;
            let band_detail = value - low;
            let atmospheric = low.min(0.95);
            let restored = if dehaze >= 0.0 {
                (value - atmospheric * dehaze * 0.35) / (1.0 - dehaze * 0.35).max(0.2)
            } else {
                value * (1.0 + dehaze * 0.25) - dehaze * 0.25
            };
            pixel[channel] =
                ((restored + band_detail * texture * 1.5).clamp(0.0, 1.0) * 255.0).round() as u8;
        }
    }
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

fn apply_monochrome(
    image: &mut RgbaImage,
    recipe: crate::catalog::MonochromeRecipe,
    cancelled: &AtomicBool,
) -> image::ImageResult<()> {
    for (index, pixel) in image.pixels_mut().enumerate() {
        if index % 16_384 == 0 {
            check(cancelled)?;
        }
        let rgb = [pixel[0], pixel[1], pixel[2]].map(|value| f32::from(value) / 255.0);
        let max = rgb.iter().copied().fold(0.0_f32, f32::max);
        let min = rgb.iter().copied().fold(1.0_f32, f32::min);
        let delta = max - min;
        let hue = if delta <= 1e-6 {
            0.0
        } else if max == rgb[0] {
            60.0 * ((rgb[1] - rgb[2]) / delta).rem_euclid(6.0)
        } else if max == rgb[1] {
            60.0 * ((rgb[2] - rgb[0]) / delta + 2.0)
        } else {
            60.0 * ((rgb[0] - rgb[1]) / delta + 4.0)
        };
        let sector = hue / 45.0;
        let left = sector.floor() as usize % 8;
        let right = (left + 1) % 8;
        let blend = sector.fract();
        let mixer = recipe.mixer[left] * (1.0 - blend) + recipe.mixer[right] * blend;
        let luma = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
        let gray = (luma * (1.0 + mixer / 100.0) * 255.0)
            .round()
            .clamp(0.0, 255.0) as u8;
        pixel[0] = gray;
        pixel[1] = gray;
        pixel[2] = gray;
    }
    Ok(())
}

fn apply_parametric_curve(
    image: &mut RgbaImage,
    controls: [f32; 4],
    cancelled: &AtomicBool,
) -> image::ImageResult<()> {
    for (index, pixel) in image.pixels_mut().enumerate() {
        if index % 16_384 == 0 {
            check(cancelled)?;
        }
        let luma = luminance([
            srgb_to_linear(pixel[0]),
            srgb_to_linear(pixel[1]),
            srgb_to_linear(pixel[2]),
        ]);
        let centers = [0.875_f32, 0.625, 0.375, 0.125];
        let weights = centers.map(|center| {
            let distance = (luma.clamp(0.0, 1.0) - center) / 0.25;
            (-0.5 * distance * distance).exp()
        });
        let total = weights.iter().sum::<f32>().max(1e-6);
        let stops = controls
            .iter()
            .zip(weights)
            .map(|(control, weight)| control / 100.0 * weight)
            .sum::<f32>()
            / total;
        let gain = 2.0_f32.powf(stops);
        for channel in 0..3 {
            pixel[channel] = linear_to_srgb(srgb_to_linear(pixel[channel]) * gain);
        }
    }
    Ok(())
}

fn apply_vignette(
    image: &mut RgbaImage,
    vignette: crate::catalog::VignetteRecipe,
    cancelled: &AtomicBool,
) -> image::ImageResult<()> {
    let cx = (image.width() as f32 - 1.0) * 0.5;
    let cy = (image.height() as f32 - 1.0) * 0.5;
    let midpoint = (vignette.midpoint / 100.0).clamp(0.05, 0.95);
    let feather = (vignette.feather / 100.0).clamp(0.01, 1.0);
    let exponent = (2.0 + vignette.roundness / 50.0).clamp(0.5, 4.0);
    for y in 0..image.height() {
        if y % 16 == 0 {
            check(cancelled)?;
        }
        for x in 0..image.width() {
            let nx = ((x as f32 - cx) / cx.max(1.0)).abs();
            let ny = ((y as f32 - cy) / cy.max(1.0)).abs();
            let radius = (nx.powf(exponent) + ny.powf(exponent))
                .powf(1.0 / exponent)
                .min(1.0);
            let mask = ((radius - midpoint) / feather).clamp(0.0, 1.0);
            let gain = 2.0_f32.powf(vignette.amount / 100.0 * mask);
            let pixel = image.get_pixel_mut(x, y);
            for channel in 0..3 {
                pixel[channel] = (f32::from(pixel[channel]) * gain).clamp(0.0, 255.0) as u8;
            }
        }
    }
    Ok(())
}

fn apply_global(
    image: &mut RgbaImage,
    recipe: &DevelopRecipe,
    cancelled: &AtomicBool,
) -> image::ImageResult<()> {
    let exposure = 2.0_f32.powf(recipe.exposure_ev.unwrap_or(0.0).clamp(-5.0, 5.0));
    let contrast = recipe.contrast.unwrap_or(0.0).clamp(-100.0, 100.0) / 100.0;
    let highlights = recipe.highlights.unwrap_or(0.0).clamp(-100.0, 100.0) / 100.0;
    let shadows = recipe.shadows.unwrap_or(0.0).clamp(-100.0, 100.0) / 100.0;
    let whites = recipe.whites.unwrap_or(0.0).clamp(-100.0, 100.0) / 100.0;
    let blacks = recipe.blacks.unwrap_or(0.0).clamp(-100.0, 100.0) / 100.0;
    let saturation = recipe.saturation.unwrap_or(0.0).clamp(-100.0, 100.0) / 100.0;
    let vibrance = recipe.vibrance.unwrap_or(0.0).clamp(-100.0, 100.0) / 100.0;
    let wb = recipe.white_balance.map(|wb| {
        let temperature =
            ((wb.temperature.clamp(2_000.0, 50_000.0) - 6_500.0) / 43_500.0).clamp(-1.0, 1.0);
        let tint = (wb.tint.clamp(-150.0, 150.0) / 150.0).clamp(-1.0, 1.0);
        [
            1.0 + temperature * 0.45 + tint * 0.08,
            1.0 - tint * 0.12,
            1.0 - temperature * 0.45 + tint * 0.08,
        ]
    });
    let wb_increment = recipe.white_balance_increment.map(|wb| {
        let temperature = (wb.temperature / 100.0).clamp(-1.0, 1.0);
        let tint = (wb.tint / 100.0).clamp(-1.0, 1.0);
        [
            1.0 + temperature * 0.15 + tint * 0.05,
            1.0 - tint * 0.08,
            1.0 - temperature * 0.15 + tint * 0.05,
        ]
    });

    for (row, pixels) in image.rows_mut().enumerate() {
        if row % 16 == 0 {
            check(cancelled)?;
        }
        for pixel in pixels {
            let alpha = pixel[3];
            let mut rgb = [
                srgb_to_linear(pixel[0]),
                srgb_to_linear(pixel[1]),
                srgb_to_linear(pixel[2]),
            ];
            if let Some(wb) = wb {
                for channel in 0..3 {
                    rgb[channel] *= wb[channel];
                }
            }
            if let Some(wb) = wb_increment {
                for channel in 0..3 {
                    rgb[channel] *= wb[channel];
                }
            }
            for channel in &mut rgb {
                *channel *= exposure;
            }
            let mut luma = luminance(rgb).max(0.0);
            let shadow_mask = (1.0 - luma.clamp(0.0, 1.0)).powi(2);
            let highlight_mask = luma.clamp(0.0, 1.0).powi(2);
            let original_luma = luma.max(1e-6);
            luma += shadows * shadow_mask * 0.35;
            luma += highlights * highlight_mask * 0.35;
            luma += blacks * shadow_mask * 0.20;
            luma += whites * highlight_mask * 0.20;
            luma = 0.5 + (luma - 0.5) * (1.0 + contrast);
            let ratio = luma.max(0.0) / original_luma;
            for channel in &mut rgb {
                *channel *= ratio;
            }
            let luma = luminance(rgb);
            let chroma = rgb
                .iter()
                .map(|channel| (channel - luma).abs())
                .fold(0.0_f32, f32::max);
            let adaptive = 1.0 + saturation + vibrance * (1.0 - chroma.clamp(0.0, 1.0));
            for channel in &mut rgb {
                *channel = luma + (*channel - luma) * adaptive.max(0.0);
            }
            if let Some(curve) = &recipe.tone_curve {
                let before = luminance(rgb).max(1e-6);
                let after = sample_curve(curve, before.clamp(0.0, 1.0));
                let ratio = after.max(0.0) / before;
                for channel in &mut rgb {
                    *channel *= ratio;
                }
            }
            for (channel, curve) in [
                recipe.tone_curve_red.as_ref(),
                recipe.tone_curve_green.as_ref(),
                recipe.tone_curve_blue.as_ref(),
            ]
            .into_iter()
            .enumerate()
            {
                if let Some(curve) = curve {
                    rgb[channel] = sample_curve(curve, rgb[channel].clamp(0.0, 1.0));
                }
            }
            *pixel = Rgba([
                linear_to_srgb(rgb[0]),
                linear_to_srgb(rgb[1]),
                linear_to_srgb(rgb[2]),
                alpha,
            ]);
        }
    }
    Ok(())
}

fn apply_clarity(
    mut image: RgbaImage,
    amount: f32,
    cancelled: &AtomicBool,
) -> image::ImageResult<RgbaImage> {
    check(cancelled)?;
    let blurred = cancellable_box_blur(&image, 3, cancelled)?;
    let strength = amount.clamp(-100.0, 100.0) / 100.0;
    for (index, (pixel, low)) in image.pixels_mut().zip(blurred.pixels()).enumerate() {
        if index % 16_384 == 0 {
            check(cancelled)?;
        }
        for channel in 0..3 {
            let detail = f32::from(pixel[channel]) - f32::from(low[channel]);
            pixel[channel] = (f32::from(pixel[channel]) + detail * strength)
                .round()
                .clamp(0.0, 255.0) as u8;
        }
    }
    Ok(image)
}

fn apply_crop(
    image: RgbaImage,
    crop: Crop,
    cancelled: &AtomicBool,
) -> image::ImageResult<RgbaImage> {
    let rotated = if crop.angle_degrees.abs() > 0.001 {
        rotate_same_bounds(&image, crop.angle_degrees, cancelled)?
    } else {
        image
    };
    let width = rotated.width();
    let height = rotated.height();
    let left = (crop.left.clamp(0.0, 1.0) * width as f32).floor() as u32;
    let top = (crop.top.clamp(0.0, 1.0) * height as f32).floor() as u32;
    let right = (crop.right.clamp(0.0, 1.0) * width as f32).ceil() as u32;
    let bottom = (crop.bottom.clamp(0.0, 1.0) * height as f32).ceil() as u32;
    if right <= left || bottom <= top {
        return Err(decode_error("invalid crop bounds"));
    }
    Ok(imageops::crop_imm(&rotated, left, top, right - left, bottom - top).to_image())
}

fn apply_heal_spots(
    image: &mut RgbaImage,
    spots: &[crate::catalog::HealSpot],
    cancelled: &AtomicBool,
) -> image::ImageResult<()> {
    let source_image = image.clone();
    let scale = image.width().min(image.height()) as f32;
    let max_x = image.width().saturating_sub(1) as f32;
    let max_y = image.height().saturating_sub(1) as f32;
    for spot in spots {
        let radius = (spot.radius * scale).max(1.0);
        let destination = [spot.destination[0] * max_x, spot.destination[1] * max_y];
        let source = [spot.source[0] * max_x, spot.source[1] * max_y];
        let destination_reference = bilinear_sample(&source_image, destination[0], destination[1]);
        let source_reference = bilinear_sample(&source_image, source[0], source[1]);
        let left = (destination[0] - radius).floor().max(0.0) as u32;
        let top = (destination[1] - radius).floor().max(0.0) as u32;
        let right = (destination[0] + radius).ceil().min(image.width() as f32) as u32;
        let bottom = (destination[1] + radius).ceil().min(image.height() as f32) as u32;
        for y in top..bottom {
            if y % 8 == 0 {
                check(cancelled)?;
            }
            for x in left..right {
                let dx = x as f32 - destination[0];
                let dy = y as f32 - destination[1];
                let distance = (dx * dx + dy * dy).sqrt();
                if distance > radius {
                    continue;
                }
                let sx = source[0] + dx;
                let sy = source[1] + dy;
                if sx < 0.0 || sy < 0.0 || sx >= image.width() as f32 || sy >= image.height() as f32
                {
                    continue;
                }
                let mut sampled = bilinear_sample(&source_image, sx, sy);
                if spot.kind == crate::catalog::HealKind::Heal {
                    for channel in 0..3 {
                        sampled[channel] = (i16::from(sampled[channel])
                            + i16::from(destination_reference[channel])
                            - i16::from(source_reference[channel]))
                        .clamp(0, 255) as u8;
                    }
                }
                let edge_start = radius * (1.0 - spot.feather);
                let feather = if spot.feather <= 1e-6 || distance <= edge_start {
                    1.0
                } else {
                    (1.0 - (distance - edge_start) / (radius - edge_start)).clamp(0.0, 1.0)
                } * spot.opacity;
                let pixel = image.get_pixel_mut(x, y);
                for channel in 0..3 {
                    pixel[channel] = (f32::from(pixel[channel]) * (1.0 - feather)
                        + f32::from(sampled[channel]) * feather)
                        .round() as u8;
                }
            }
        }
    }
    Ok(())
}

fn apply_lens_distortion(
    image: &RgbaImage,
    amount: f32,
    cancelled: &AtomicBool,
) -> image::ImageResult<RgbaImage> {
    let mut output = RgbaImage::new(image.width(), image.height());
    let cx = (image.width() as f32 - 1.0) * 0.5;
    let cy = (image.height() as f32 - 1.0) * 0.5;
    let k = amount.clamp(-100.0, 100.0) / 500.0;
    for y in 0..image.height() {
        if y % 8 == 0 {
            check(cancelled)?;
        }
        for x in 0..image.width() {
            let nx = (x as f32 - cx) / cx.max(1.0);
            let ny = (y as f32 - cy) / cy.max(1.0);
            let factor = 1.0 + k * (nx * nx + ny * ny);
            let sx = cx + nx * factor * cx;
            let sy = cy + ny * factor * cy;
            if sx >= 0.0 && sy >= 0.0 && sx < image.width() as f32 && sy < image.height() as f32 {
                output.put_pixel(x, y, bilinear_sample(image, sx, sy));
            }
        }
    }
    Ok(output)
}

fn apply_homography(
    image: &RgbaImage,
    matrix: [f32; 9],
    cancelled: &AtomicBool,
) -> image::ImageResult<RgbaImage> {
    let inverse = invert_homography(matrix).ok_or_else(|| decode_error("singular homography"))?;
    let mut output = RgbaImage::new(image.width(), image.height());
    for y in 0..image.height() {
        if y % 8 == 0 {
            check(cancelled)?;
        }
        for x in 0..image.width() {
            let nx = x as f32 / image.width().saturating_sub(1).max(1) as f32;
            let ny = y as f32 / image.height().saturating_sub(1).max(1) as f32;
            let w = inverse[6] * nx + inverse[7] * ny + inverse[8];
            if w.abs() < 1e-6 {
                continue;
            }
            let sx = (inverse[0] * nx + inverse[1] * ny + inverse[2]) / w
                * image.width().saturating_sub(1) as f32;
            let sy = (inverse[3] * nx + inverse[4] * ny + inverse[5]) / w
                * image.height().saturating_sub(1) as f32;
            if sx >= 0.0 && sy >= 0.0 && sx < image.width() as f32 && sy < image.height() as f32 {
                output.put_pixel(x, y, bilinear_sample(image, sx, sy));
            }
        }
    }
    Ok(output)
}

fn invert_homography(m: [f32; 9]) -> Option<[f32; 9]> {
    let determinant = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
        + m[2] * (m[3] * m[7] - m[4] * m[6]);
    if determinant.abs() < 1e-8 {
        return None;
    }
    Some([
        (m[4] * m[8] - m[5] * m[7]) / determinant,
        (m[2] * m[7] - m[1] * m[8]) / determinant,
        (m[1] * m[5] - m[2] * m[4]) / determinant,
        (m[5] * m[6] - m[3] * m[8]) / determinant,
        (m[0] * m[8] - m[2] * m[6]) / determinant,
        (m[2] * m[3] - m[0] * m[5]) / determinant,
        (m[3] * m[7] - m[4] * m[6]) / determinant,
        (m[1] * m[6] - m[0] * m[7]) / determinant,
        (m[0] * m[4] - m[1] * m[3]) / determinant,
    ])
}

fn apply_grain(
    image: &mut RgbaImage,
    grain: crate::catalog::GrainRecipe,
    cancelled: &AtomicBool,
) -> image::ImageResult<()> {
    let cell = (1.0 + grain.size / 20.0).round().max(1.0) as u32;
    for y in 0..image.height() {
        if y % 8 == 0 {
            check(cancelled)?;
        }
        for x in 0..image.width() {
            let gx = x / cell;
            let gy = y / cell;
            let mut hash = grain.seed
                ^ u64::from(gx).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                ^ u64::from(gy).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            hash = (hash ^ (hash >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            hash = (hash ^ (hash >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            hash ^= hash >> 31;
            let noise = (hash as f64 / u64::MAX as f64 - 0.5) as f32
                * grain.amount
                * (0.5 + grain.roughness / 100.0);
            let pixel = image.get_pixel_mut(x, y);
            for channel in 0..3 {
                pixel[channel] = (f32::from(pixel[channel]) + noise)
                    .round()
                    .clamp(0.0, 255.0) as u8;
            }
        }
    }
    Ok(())
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
mod tests {
    use super::*;
    use image::ImageBuffer;

    #[test]
    fn identity_recipe_is_pixel_identical() {
        let image =
            DynamicImage::ImageRgba8(ImageBuffer::from_pixel(4, 4, Rgba([30, 60, 90, 255])));
        let (result, actions, warnings) =
            apply(&image, &DevelopRecipe::default(), &AtomicBool::new(false)).unwrap();
        assert_eq!(result.to_rgba8(), image.to_rgba8());
        assert_eq!((actions, warnings), (0, 0));
    }

    #[test]
    fn exposure_brightens_and_curve_is_monotonic() {
        let image =
            DynamicImage::ImageRgba8(ImageBuffer::from_pixel(4, 4, Rgba([64, 64, 64, 255])));
        let recipe = DevelopRecipe {
            exposure_ev: Some(1.0),
            tone_curve: Some(vec![[0.0, 0.0], [0.5, 0.4], [1.0, 1.0]]),
            ..DevelopRecipe::default()
        };
        let (result, actions, _) = apply(&image, &recipe, &AtomicBool::new(false)).unwrap();
        assert!(result.to_rgba8().get_pixel(0, 0)[0] > 64);
        assert_ne!(actions & actions::EXPOSURE, 0);
        assert!(
            sample_curve(recipe.tone_curve.as_ref().unwrap(), 0.25)
                < sample_curve(recipe.tone_curve.as_ref().unwrap(), 0.75)
        );
    }

    #[test]
    fn crop_changes_dimensions_and_cancellation_is_observed() {
        let image =
            DynamicImage::ImageRgba8(ImageBuffer::from_pixel(100, 80, Rgba([1, 2, 3, 255])));
        let recipe = DevelopRecipe {
            crop: Some(Crop {
                left: 0.25,
                top: 0.25,
                right: 0.75,
                bottom: 0.75,
                angle_degrees: 0.0,
                constrain_aspect: false,
            }),
            ..DevelopRecipe::default()
        };
        let (result, _, _) = apply(&image, &recipe, &AtomicBool::new(false)).unwrap();
        assert_eq!((result.width(), result.height()), (50, 40));
        assert!(apply(&image, &recipe, &AtomicBool::new(true)).is_err());
    }

    #[test]
    fn phase_c_detail_monochrome_and_finishing_are_applied_and_reported() {
        let source =
            DynamicImage::ImageRgba8(RgbaImage::from_pixel(16, 16, Rgba([120, 80, 40, 255])));
        let recipe = DevelopRecipe {
            detail: Some(crate::catalog::DetailRecipe {
                sharpening: 20.0,
                ..crate::catalog::DetailRecipe::default()
            }),
            texture: Some(10.0),
            monochrome: Some(crate::catalog::MonochromeRecipe { mixer: [0.0; 8] }),
            vignette: Some(crate::catalog::VignetteRecipe {
                amount: -20.0,
                midpoint: 50.0,
                feather: 50.0,
                roundness: 0.0,
            }),
            parametric_curve: Some([5.0, 0.0, 0.0, 0.0]),
            ..DevelopRecipe::default()
        };
        let (output, actions, warnings) = apply(&source, &recipe, &AtomicBool::new(false)).unwrap();
        let pixel = output.to_rgba8().get_pixel(8, 8).to_owned();

        assert_eq!(pixel[0], pixel[1]);
        assert_eq!(pixel[1], pixel[2]);
        assert_ne!(actions & actions::DETAIL, 0);
        assert_ne!(actions & actions::MONOCHROME, 0);
        assert_ne!(actions & actions::VIGNETTE, 0);
        assert_ne!(warnings & warnings::APPROXIMATED, 0);
    }

    #[test]
    fn every_monochrome_hue_sector_changes_its_matching_colour() {
        let colours = [
            [255, 0, 0, 255],
            [255, 191, 0, 255],
            [128, 255, 0, 255],
            [0, 255, 64, 255],
            [0, 255, 255, 255],
            [0, 64, 255, 255],
            [128, 0, 255, 255],
            [255, 0, 191, 255],
        ];
        for (sector, colour) in colours.into_iter().enumerate() {
            let mut baseline = RgbaImage::from_pixel(1, 1, Rgba(colour));
            let mut adjusted = baseline.clone();
            apply_monochrome(
                &mut baseline,
                crate::catalog::MonochromeRecipe { mixer: [0.0; 8] },
                &AtomicBool::new(false),
            )
            .unwrap();
            let mut mixer = [0.0; 8];
            mixer[sector] = 50.0;
            apply_monochrome(
                &mut adjusted,
                crate::catalog::MonochromeRecipe { mixer },
                &AtomicBool::new(false),
            )
            .unwrap();
            assert_ne!(baseline, adjusted, "sector {sector} was ignored");
        }
    }

    #[test]
    fn parametric_curve_is_continuous_across_old_region_boundaries() {
        let mut image = RgbaImage::new(2, 1);
        image.put_pixel(0, 0, Rgba([127, 127, 127, 255]));
        image.put_pixel(1, 0, Rgba([128, 128, 128, 255]));
        apply_parametric_curve(
            &mut image,
            [100.0, -100.0, 100.0, -100.0],
            &AtomicBool::new(false),
        )
        .unwrap();
        assert!(
            (i16::from(image.get_pixel(0, 0)[0]) - i16::from(image.get_pixel(1, 0)[0])).abs() < 8
        );
    }

    #[test]
    fn cancellable_filter_observes_active_cancellation() {
        let image = RgbaImage::from_pixel(800, 800, Rgba([100, 120, 140, 255]));
        let cancelled = std::sync::Arc::new(AtomicBool::new(false));
        let worker_cancelled = cancelled.clone();
        let worker = std::thread::spawn(move || cancellable_box_blur(&image, 3, &worker_cancelled));
        std::thread::sleep(std::time::Duration::from_millis(1));
        cancelled.store(true, Ordering::Release);
        assert!(worker.join().unwrap().is_err());
    }

    #[test]
    fn phase_d_geometry_heal_and_grain_are_deterministic_and_reported() {
        let mut source = RgbaImage::from_pixel(12, 12, Rgba([80, 80, 80, 255]));
        source.put_pixel(2, 2, Rgba([220, 30, 30, 255]));
        let recipe = DevelopRecipe {
            upright_homography: Some([1.0, 0.0, 0.05, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
            lens_distortion: Some(10.0),
            heal_spots: vec![crate::catalog::HealSpot {
                kind: crate::catalog::HealKind::Heal,
                destination: [0.5, 0.5],
                source: [2.0 / 12.0, 2.0 / 12.0],
                radius: 0.15,
                opacity: 1.0,
                feather: 0.0,
            }],
            grain: Some(crate::catalog::GrainRecipe {
                amount: 15.0,
                size: 30.0,
                roughness: 50.0,
                seed: 42,
            }),
            ..DevelopRecipe::default()
        };
        let source = DynamicImage::ImageRgba8(source);
        let first = apply(&source, &recipe, &AtomicBool::new(false)).unwrap();
        let second = apply(&source, &recipe, &AtomicBool::new(false)).unwrap();

        assert_eq!(first.0.to_rgba8(), second.0.to_rgba8());
        assert_ne!(first.1 & actions::UPRIGHT, 0);
        assert_ne!(first.1 & actions::LENS, 0);
        assert_ne!(first.1 & actions::HEAL, 0);
        assert_ne!(first.1 & actions::GRAIN, 0);
        assert_ne!(first.2 & warnings::APPROXIMATED, 0);
    }

    #[test]
    fn singular_homography_is_rejected() {
        let image = RgbaImage::from_pixel(2, 2, Rgba([0, 0, 0, 255]));
        assert!(
            apply_homography(
                &image,
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                &AtomicBool::new(false),
            )
            .is_err()
        );
    }

    #[test]
    fn clone_and_heal_have_distinct_semantics() {
        let mut source = RgbaImage::from_pixel(20, 20, Rgba([20, 30, 40, 255]));
        source.put_pixel(2, 2, Rgba([200, 50, 40, 255]));
        source.put_pixel(10, 10, Rgba([70, 90, 110, 255]));
        let base = crate::catalog::HealSpot {
            kind: crate::catalog::HealKind::Clone,
            destination: [10.0 / 19.0, 10.0 / 19.0],
            source: [2.0 / 19.0, 2.0 / 19.0],
            radius: 0.08,
            opacity: 1.0,
            feather: 0.0,
        };
        let mut cloned = source.clone();
        apply_heal_spots(&mut cloned, &[base], &AtomicBool::new(false)).unwrap();
        let mut healed = source;
        apply_heal_spots(
            &mut healed,
            &[crate::catalog::HealSpot {
                kind: crate::catalog::HealKind::Heal,
                ..base
            }],
            &AtomicBool::new(false),
        )
        .unwrap();

        assert_ne!(cloned.get_pixel(10, 10), healed.get_pixel(10, 10));
        assert!(cloned.get_pixel(10, 10)[0] > healed.get_pixel(10, 10)[0]);
    }

    #[test]
    fn heal_accepts_normalized_image_boundaries() {
        let mut image = RgbaImage::from_pixel(8, 8, Rgba([20, 30, 40, 255]));
        image.put_pixel(0, 0, Rgba([200, 100, 50, 255]));
        apply_heal_spots(
            &mut image,
            &[
                crate::catalog::HealSpot {
                    kind: crate::catalog::HealKind::Clone,
                    destination: [1.0, 1.0],
                    source: [0.0, 0.0],
                    radius: 0.01,
                    opacity: 1.0,
                    feather: 0.0,
                },
                crate::catalog::HealSpot {
                    kind: crate::catalog::HealKind::Clone,
                    destination: [0.0, 0.0],
                    source: [1.0, 1.0],
                    radius: 0.01,
                    opacity: 1.0,
                    feather: 0.0,
                },
            ],
            &AtomicBool::new(false),
        )
        .unwrap();
    }

    #[test]
    fn phase_d_pixel_passes_observe_active_cancellation() {
        fn cancelled_during(
            operation: impl FnOnce(&AtomicBool) -> image::ImageResult<()> + Send + 'static,
        ) {
            let cancelled = std::sync::Arc::new(AtomicBool::new(false));
            let worker_cancelled = cancelled.clone();
            let worker = std::thread::spawn(move || operation(&worker_cancelled));
            std::thread::sleep(std::time::Duration::from_millis(1));
            cancelled.store(true, Ordering::Release);
            assert!(worker.join().unwrap().is_err());
        }

        cancelled_during(|cancelled| {
            let mut image = RgbaImage::from_pixel(2_000, 2_000, Rgba([80, 90, 100, 255]));
            apply_heal_spots(
                &mut image,
                &[crate::catalog::HealSpot {
                    kind: crate::catalog::HealKind::Clone,
                    destination: [0.5, 0.5],
                    source: [0.25, 0.25],
                    radius: 0.5,
                    opacity: 1.0,
                    feather: 0.5,
                }],
                cancelled,
            )
        });
        cancelled_during(|cancelled| {
            apply_lens_distortion(&RgbaImage::new(2_000, 2_000), 20.0, cancelled).map(drop)
        });
        cancelled_during(|cancelled| {
            apply_homography(
                &RgbaImage::new(2_000, 2_000),
                [1.0, 0.0, 0.05, 0.0, 1.0, 0.05, 0.0, 0.0, 1.0],
                cancelled,
            )
            .map(drop)
        });
        cancelled_during(|cancelled| {
            let mut image = RgbaImage::new(2_000, 2_000);
            apply_grain(
                &mut image,
                crate::catalog::GrainRecipe {
                    amount: 20.0,
                    size: 30.0,
                    roughness: 50.0,
                    seed: 1,
                },
                cancelled,
            )
        });
    }

    #[test]
    fn grain_is_deterministic_per_asset_seed() {
        let source = RgbaImage::from_pixel(32, 32, Rgba([128, 128, 128, 255]));
        let grain = crate::catalog::GrainRecipe {
            amount: 20.0,
            size: 30.0,
            roughness: 50.0,
            seed: 1,
        };
        let mut first = source.clone();
        let mut repeat = source.clone();
        let mut other = source;
        apply_grain(&mut first, grain, &AtomicBool::new(false)).unwrap();
        apply_grain(&mut repeat, grain, &AtomicBool::new(false)).unwrap();
        apply_grain(
            &mut other,
            crate::catalog::GrainRecipe { seed: 2, ..grain },
            &AtomicBool::new(false),
        )
        .unwrap();
        assert_eq!(first, repeat);
        assert_ne!(first, other);
    }
}
