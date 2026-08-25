use std::sync::atomic::AtomicBool;

use image::RgbaImage;

use super::DevelopAction;
use crate::rendering::develop::{actions, check, linear_to_srgb, luminance, srgb_to_linear};

pub(in crate::rendering::develop) struct ParametricCurve {
    pub controls: [f32; 4],
}

impl DevelopAction for ParametricCurve {
    fn action_mask(&self) -> u64 {
        actions::PARAMETRIC_CURVE
    }

    fn apply(&self, image: &mut RgbaImage, cancelled: &AtomicBool) -> image::ImageResult<()> {
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
            let stops = self
                .controls
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
}
