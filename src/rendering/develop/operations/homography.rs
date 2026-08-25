use std::sync::atomic::AtomicBool;

use image::RgbaImage;

use super::DevelopAction;
use crate::rendering::develop::{actions, bilinear_sample, check, decode_error};

pub(crate) struct Homography {
    pub matrix: [f32; 9],
}

impl DevelopAction for Homography {
    fn action_mask(&self) -> u64 {
        actions::UPRIGHT
    }

    fn apply(&self, image: &mut RgbaImage, cancelled: &AtomicBool) -> image::ImageResult<()> {
        let inverse = invert(self.matrix).ok_or_else(|| decode_error("singular homography"))?;
        let mut output = RgbaImage::new(image.width(), image.height());
        for y in 0..image.height() {
            if y % 8 == 0 {
                check(cancelled)?;
            }
            for x in 0..image.width() {
                let nx = x as f32 / image.width().saturating_sub(1).max(1) as f32;
                let ny = y as f32 / image.height().saturating_sub(1).max(1) as f32;
                let w = inverse[6] * nx + inverse[7] * ny + inverse[8];
                if w.abs() < 1e-6 {
                    continue;
                }
                let sx = (inverse[0] * nx + inverse[1] * ny + inverse[2]) / w
                    * image.width().saturating_sub(1) as f32;
                let sy = (inverse[3] * nx + inverse[4] * ny + inverse[5]) / w
                    * image.height().saturating_sub(1) as f32;
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

fn invert(m: [f32; 9]) -> Option<[f32; 9]> {
    let determinant = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
        + m[2] * (m[3] * m[7] - m[4] * m[6]);
    if determinant.abs() < 1e-8 {
        return None;
    }
    Some([
        (m[4] * m[8] - m[5] * m[7]) / determinant,
        (m[2] * m[7] - m[1] * m[8]) / determinant,
        (m[1] * m[5] - m[2] * m[4]) / determinant,
        (m[5] * m[6] - m[3] * m[8]) / determinant,
        (m[0] * m[8] - m[2] * m[6]) / determinant,
        (m[2] * m[3] - m[0] * m[5]) / determinant,
        (m[3] * m[7] - m[4] * m[6]) / determinant,
        (m[1] * m[6] - m[0] * m[7]) / determinant,
        (m[0] * m[4] - m[1] * m[3]) / determinant,
    ])
}
