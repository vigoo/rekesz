use std::sync::atomic::AtomicBool;

use image::RgbaImage;

use super::DevelopAction;
use crate::{
    catalog::DetailRecipe,
    rendering::develop::{actions, cancellable_box_blur, check},
};

pub(in crate::rendering::develop) struct Detail {
    pub parameters: DetailRecipe,
}

impl DevelopAction for Detail {
    fn action_mask(&self) -> u64 {
        actions::DETAIL
    }

    fn apply(&self, image: &mut RgbaImage, cancelled: &AtomicBool) -> image::ImageResult<()> {
        let detail = self.parameters;
        check(cancelled)?;
        if detail.luminance_noise_reduction > 0.0 || detail.color_noise_reduction > 0.0 {
            let radius = (1.0 + detail.color_smoothness / 50.0).round() as u32;
            let smooth = cancellable_box_blur(image, radius.clamp(1, 3), cancelled)?;
            let luma_mix = (detail.luminance_noise_reduction / 100.0)
                * (1.0 - detail.luminance_detail / 200.0);
            let luma_contrast = detail.luminance_contrast / 100.0;
            let color_mix =
                (detail.color_noise_reduction / 100.0) * (0.5 + detail.color_smoothness / 200.0);
            let color_detail = detail.color_detail / 100.0;
            for (index, (pixel, low)) in image.pixels_mut().zip(smooth.pixels()).enumerate() {
                if index % 16_384 == 0 {
                    check(cancelled)?;
                }
                let source_luma = 0.2126 * f32::from(pixel[0])
                    + 0.7152 * f32::from(pixel[1])
                    + 0.0722 * f32::from(pixel[2]);
                let low_luma = 0.2126 * f32::from(low[0])
                    + 0.7152 * f32::from(low[1])
                    + 0.0722 * f32::from(low[2]);
                let denoised_luma = source_luma * (1.0 - luma_mix)
                    + low_luma * luma_mix
                    + (source_luma - low_luma) * luma_contrast;
                for channel in 0..3 {
                    let chroma = f32::from(pixel[channel]) - source_luma;
                    let low_chroma = f32::from(low[channel]) - low_luma;
                    let denoised_chroma = chroma * (1.0 - color_mix)
                        + low_chroma * color_mix
                        + (chroma - low_chroma) * color_detail * 0.25;
                    pixel[channel] =
                        (denoised_luma + denoised_chroma).round().clamp(0.0, 255.0) as u8;
                }
            }
        }
        if detail.sharpening > 0.0 {
            let radius = detail.sharpen_radius.round().clamp(1.0, 3.0) as u32;
            let blurred = cancellable_box_blur(image, radius, cancelled)?;
            let strength =
                (detail.sharpening / 100.0).clamp(0.0, 2.0) * (0.5 + detail.sharpen_detail / 100.0);
            let threshold = detail.edge_masking / 100.0 * 32.0;
            for (index, (pixel, low)) in image.pixels_mut().zip(blurred.pixels()).enumerate() {
                if index % 16_384 == 0 {
                    check(cancelled)?;
                }
                let edge = (0..3)
                    .map(|channel| (f32::from(pixel[channel]) - f32::from(low[channel])).abs())
                    .sum::<f32>()
                    / 3.0;
                let mask = ((edge - threshold) / 16.0).clamp(0.0, 1.0);
                for channel in 0..3 {
                    pixel[channel] = (f32::from(pixel[channel])
                        + (f32::from(pixel[channel]) - f32::from(low[channel])) * strength * mask)
                        .round()
                        .clamp(0.0, 255.0) as u8;
                }
            }
        }
        Ok(())
    }
}
