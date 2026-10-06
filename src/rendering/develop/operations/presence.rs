use std::sync::atomic::{AtomicBool, Ordering};

use image::RgbaImage;

use super::DevelopAction;
use super::log_detail::{srgb_to_linear_table, to_image_error};
use crate::rendering::develop::{actions, check, linear_to_srgb};
use crate::rendering::presence as core;

pub(in crate::rendering::develop) struct Presence {
    pub texture: Option<f32>,
    pub dehaze: Option<f32>,
}

impl DevelopAction for Presence {
    fn action_mask(&self) -> u64 {
        actions::TEXTURE_DEHAZE
    }

    fn apply(&self, image: &mut RgbaImage, cancelled: &AtomicBool) -> image::ImageResult<()> {
        check(cancelled)?;
        if self.texture.is_none() && self.dehaze.is_none() {
            return Ok(());
        }
        self.apply_inner(image, cancelled).map_err(to_image_error)
    }
}

impl Presence {
    fn apply_inner(&self, image: &mut RgbaImage, cancelled: &AtomicBool) -> Result<(), String> {
        let width = image.width() as usize;
        let height = image.height() as usize;
        let table = srgb_to_linear_table();
        let mut pixels: Vec<[f32; 3]> = Vec::with_capacity(width * height);
        for (index, pixel) in image.pixels().enumerate() {
            if index % 16_384 == 0 && cancelled.load(Ordering::Acquire) {
                return Err("presence operator cancelled".to_owned());
            }
            pixels.push([
                table[pixel[0] as usize],
                table[pixel[1] as usize],
                table[pixel[2] as usize],
            ]);
        }
        if let Some(dehaze) = self.dehaze {
            core::apply_dehaze(&mut pixels, width, height, dehaze, display_tone, cancelled)?;
        }
        if let Some(texture) = self.texture {
            core::apply_texture(&mut pixels, width, height, texture, display_tone, cancelled)?;
        }
        for (index, pixel) in image.pixels_mut().enumerate() {
            if index % 16_384 == 0 && cancelled.load(Ordering::Acquire) {
                return Err("presence operator cancelled".to_owned());
            }
            let value = pixels[index];
            for channel in 0..3 {
                pixel[channel] = linear_to_srgb(value[channel]);
            }
        }
        Ok(())
    }
}

/// Display-referred luminance is already normalized to 0..1.
fn display_tone(luma: f32) -> f32 {
    luma
}
