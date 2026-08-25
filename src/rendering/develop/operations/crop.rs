use std::sync::atomic::AtomicBool;

use image::{RgbaImage, imageops};

use super::DevelopAction;
use crate::{
    catalog::Crop,
    rendering::develop::{actions, decode_error, rotate_same_bounds},
};

pub(crate) struct CropAction {
    pub parameters: Crop,
}

impl DevelopAction for CropAction {
    fn action_mask(&self) -> u64 {
        actions::CROP
    }

    fn apply(&self, image: &mut RgbaImage, cancelled: &AtomicBool) -> image::ImageResult<()> {
        let crop = self.parameters;
        let rotated = if crop.angle_degrees.abs() > 0.001 {
            rotate_same_bounds(image, crop.angle_degrees, cancelled)?
        } else {
            image.clone()
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
        *image = imageops::crop_imm(&rotated, left, top, right - left, bottom - top).to_image();
        Ok(())
    }
}
