use std::sync::atomic::AtomicBool;

use image::RgbaImage;

use super::DevelopAction;
use crate::{
    catalog::MonochromeRecipe,
    rendering::develop::{actions, check},
};

pub(crate) struct Monochrome {
    pub parameters: MonochromeRecipe,
}

impl DevelopAction for Monochrome {
    fn action_mask(&self) -> u64 {
        actions::MONOCHROME
    }

    fn apply(&self, image: &mut RgbaImage, cancelled: &AtomicBool) -> image::ImageResult<()> {
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
            let mixer =
                self.parameters.mixer[left] * (1.0 - blend) + self.parameters.mixer[right] * blend;
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
}
