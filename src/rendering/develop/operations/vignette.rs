use std::sync::atomic::AtomicBool;

use image::RgbaImage;

use super::DevelopAction;
use crate::{
    catalog::VignetteRecipe,
    rendering::develop::{actions, check},
};

pub(crate) struct Vignette {
    pub parameters: VignetteRecipe,
}

impl DevelopAction for Vignette {
    fn action_mask(&self) -> u64 {
        actions::VIGNETTE
    }

    fn apply(&self, image: &mut RgbaImage, cancelled: &AtomicBool) -> image::ImageResult<()> {
        let cx = (image.width() as f32 - 1.0) * 0.5;
        let cy = (image.height() as f32 - 1.0) * 0.5;
        let midpoint = (self.parameters.midpoint / 100.0).clamp(0.05, 0.95);
        let feather = (self.parameters.feather / 100.0).clamp(0.01, 1.0);
        let exponent = (2.0 + self.parameters.roundness / 50.0).clamp(0.5, 4.0);
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
                let gain = 2.0_f32.powf(self.parameters.amount / 100.0 * mask);
                let pixel = image.get_pixel_mut(x, y);
                for channel in 0..3 {
                    pixel[channel] = (f32::from(pixel[channel]) * gain).clamp(0.0, 255.0) as u8;
                }
            }
        }
        Ok(())
    }
}
