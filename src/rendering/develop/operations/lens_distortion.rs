use std::sync::atomic::AtomicBool;

use image::RgbaImage;

use super::DevelopAction;
use crate::rendering::develop::{actions, bilinear_sample, check};

pub(crate) struct LensDistortion {
    pub amount: f32,
}

impl DevelopAction for LensDistortion {
    fn action_mask(&self) -> u64 {
        actions::LENS
    }

    fn apply(&self, image: &mut RgbaImage, cancelled: &AtomicBool) -> image::ImageResult<()> {
        let mut output = RgbaImage::new(image.width(), image.height());
        let cx = (image.width() as f32 - 1.0) * 0.5;
        let cy = (image.height() as f32 - 1.0) * 0.5;
        let k = self.amount.clamp(-100.0, 100.0) / 500.0;
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
                if sx >= 0.0 && sy >= 0.0 && sx < image.width() as f32 && sy < image.height() as f32
                {
                    output.put_pixel(x, y, bilinear_sample(image, sx, sy));
                }
            }
        }
        *image = output;
        Ok(())
    }
}
