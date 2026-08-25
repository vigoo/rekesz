use std::sync::atomic::AtomicBool;

use image::RgbaImage;

use super::DevelopAction;
use crate::{
    catalog::{HealKind, HealSpot},
    rendering::develop::{actions, bilinear_sample, check},
};

pub(in crate::rendering::develop) struct Heal<'a> {
    pub spots: &'a [HealSpot],
}

impl DevelopAction for Heal<'_> {
    fn action_mask(&self) -> u64 {
        actions::HEAL
    }

    fn apply(&self, image: &mut RgbaImage, cancelled: &AtomicBool) -> image::ImageResult<()> {
        let source_image = image.clone();
        let scale = image.width().min(image.height()) as f32;
        let max_x = image.width().saturating_sub(1) as f32;
        let max_y = image.height().saturating_sub(1) as f32;
        for spot in self.spots {
            let radius = (spot.radius * scale).max(1.0);
            let destination = [spot.destination[0] * max_x, spot.destination[1] * max_y];
            let source = [spot.source[0] * max_x, spot.source[1] * max_y];
            let destination_reference =
                bilinear_sample(&source_image, destination[0], destination[1]);
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
                    if sx < 0.0
                        || sy < 0.0
                        || sx >= image.width() as f32
                        || sy >= image.height() as f32
                    {
                        continue;
                    }
                    let mut sampled = bilinear_sample(&source_image, sx, sy);
                    if spot.kind == HealKind::Heal {
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
}
