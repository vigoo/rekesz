use std::sync::atomic::AtomicBool;

use image::RgbaImage;

use super::DevelopAction;
use crate::rendering::develop::{actions, cancellable_box_blur, check};

pub(crate) struct Presence {
    pub texture: Option<f32>,
    pub dehaze: Option<f32>,
}

impl DevelopAction for Presence {
    fn action_mask(&self) -> u64 {
        actions::TEXTURE_DEHAZE
    }

    fn apply(&self, image: &mut RgbaImage, cancelled: &AtomicBool) -> image::ImageResult<()> {
        let texture = self.texture.unwrap_or(0.0) / 100.0;
        let dehaze = self.dehaze.unwrap_or(0.0) / 100.0;
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
                pixel[channel] = ((restored + band_detail * texture * 1.5).clamp(0.0, 1.0) * 255.0)
                    .round() as u8;
            }
        }
        Ok(())
    }
}
