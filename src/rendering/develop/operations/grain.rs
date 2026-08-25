use std::sync::atomic::AtomicBool;

use image::RgbaImage;

use super::DevelopAction;
use crate::{
    catalog::GrainRecipe,
    rendering::develop::{actions, check},
};

pub(crate) struct Grain {
    pub parameters: GrainRecipe,
}

impl DevelopAction for Grain {
    fn action_mask(&self) -> u64 {
        actions::GRAIN
    }

    fn apply(&self, image: &mut RgbaImage, cancelled: &AtomicBool) -> image::ImageResult<()> {
        let grain = self.parameters;
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
}
