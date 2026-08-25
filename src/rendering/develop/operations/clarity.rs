use std::sync::atomic::AtomicBool;

use image::RgbaImage;

use super::DevelopAction;
use crate::rendering::develop::{actions, cancellable_box_blur, check};

pub(crate) struct Clarity {
    pub amount: f32,
}

impl DevelopAction for Clarity {
    fn action_mask(&self) -> u64 {
        actions::CLARITY
    }

    fn apply(&self, image: &mut RgbaImage, cancelled: &AtomicBool) -> image::ImageResult<()> {
        check(cancelled)?;
        let blurred = cancellable_box_blur(image, 3, cancelled)?;
        let strength = self.amount.clamp(-100.0, 100.0) / 100.0;
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
        Ok(())
    }
}
