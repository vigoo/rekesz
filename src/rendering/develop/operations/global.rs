use std::sync::atomic::AtomicBool;

use image::{Rgba, RgbaImage};

use super::DevelopAction;
use crate::{
    catalog::{DevelopRecipe, WhiteBalance, WhiteBalanceIncrement},
    rendering::develop::{actions, check, linear_to_srgb, luminance, sample_curve, srgb_to_linear},
};

pub(in crate::rendering::develop) struct Global<'a> {
    pub exposure_ev: Option<f32>,
    pub contrast: Option<f32>,
    pub highlights: Option<f32>,
    pub shadows: Option<f32>,
    pub whites: Option<f32>,
    pub blacks: Option<f32>,
    pub saturation: Option<f32>,
    pub vibrance: Option<f32>,
    pub white_balance: Option<WhiteBalance>,
    pub white_balance_increment: Option<WhiteBalanceIncrement>,
    pub tone_curve: Option<&'a [[f32; 2]]>,
    pub tone_curve_red: Option<&'a [[f32; 2]]>,
    pub tone_curve_green: Option<&'a [[f32; 2]]>,
    pub tone_curve_blue: Option<&'a [[f32; 2]]>,
}

impl<'a> Global<'a> {
    pub fn from_recipe(recipe: &'a DevelopRecipe) -> Option<Self> {
        let action = Self {
            exposure_ev: recipe.exposure_ev,
            contrast: recipe.contrast,
            highlights: recipe.highlights,
            shadows: recipe.shadows,
            whites: recipe.whites,
            blacks: recipe.blacks,
            saturation: recipe.saturation,
            vibrance: recipe.vibrance,
            white_balance: recipe.white_balance,
            white_balance_increment: recipe.white_balance_increment,
            tone_curve: recipe.tone_curve.as_deref(),
            tone_curve_red: recipe.tone_curve_red.as_deref(),
            tone_curve_green: recipe.tone_curve_green.as_deref(),
            tone_curve_blue: recipe.tone_curve_blue.as_deref(),
        };
        (action.action_mask() != 0).then_some(action)
    }
}

impl DevelopAction for Global<'_> {
    fn action_mask(&self) -> u64 {
        let mut mask = 0;
        if self.exposure_ev.is_some() {
            mask |= actions::EXPOSURE;
        }
        if self.contrast.is_some()
            || self.highlights.is_some()
            || self.shadows.is_some()
            || self.whites.is_some()
            || self.blacks.is_some()
        {
            mask |= actions::BASIC_TONE;
        }
        if self.saturation.is_some() || self.vibrance.is_some() {
            mask |= actions::COLOR;
        }
        if self.white_balance.is_some() || self.white_balance_increment.is_some() {
            mask |= actions::WHITE_BALANCE;
        }
        if self.tone_curve.is_some()
            || self.tone_curve_red.is_some()
            || self.tone_curve_green.is_some()
            || self.tone_curve_blue.is_some()
        {
            mask |= actions::TONE_CURVE;
        }
        mask
    }

    fn fidelity_mask(&self) -> u64 {
        if self.action_mask() & (actions::BASIC_TONE | actions::COLOR | actions::WHITE_BALANCE) != 0
        {
            super::warnings::APPROXIMATED
        } else {
            0
        }
    }

    fn apply(&self, image: &mut RgbaImage, cancelled: &AtomicBool) -> image::ImageResult<()> {
        let exposure = 2.0_f32.powf(self.exposure_ev.unwrap_or(0.0).clamp(-5.0, 5.0));
        let contrast = self.contrast.unwrap_or(0.0).clamp(-100.0, 100.0) / 100.0;
        let highlights = self.highlights.unwrap_or(0.0).clamp(-100.0, 100.0) / 100.0;
        let shadows = self.shadows.unwrap_or(0.0).clamp(-100.0, 100.0) / 100.0;
        let whites = self.whites.unwrap_or(0.0).clamp(-100.0, 100.0) / 100.0;
        let blacks = self.blacks.unwrap_or(0.0).clamp(-100.0, 100.0) / 100.0;
        let saturation = self.saturation.unwrap_or(0.0).clamp(-100.0, 100.0) / 100.0;
        let vibrance = self.vibrance.unwrap_or(0.0).clamp(-100.0, 100.0) / 100.0;
        let wb = self.white_balance.map(absolute_white_balance);
        let wb_increment = self.white_balance_increment.map(incremental_white_balance);

        for (row, pixels) in image.rows_mut().enumerate() {
            if row % 16 == 0 {
                check(cancelled)?;
            }
            for pixel in pixels {
                let alpha = pixel[3];
                let mut rgb = [
                    srgb_to_linear(pixel[0]),
                    srgb_to_linear(pixel[1]),
                    srgb_to_linear(pixel[2]),
                ];
                for balance in [wb, wb_increment].into_iter().flatten() {
                    for channel in 0..3 {
                        rgb[channel] *= balance[channel];
                    }
                }
                for channel in &mut rgb {
                    *channel *= exposure;
                }
                let mut luma = luminance(rgb).max(0.0);
                let shadow_mask = (1.0 - luma.clamp(0.0, 1.0)).powi(2);
                let highlight_mask = luma.clamp(0.0, 1.0).powi(2);
                let original_luma = luma.max(1e-6);
                luma *= 2.0_f32.powf(
                    shadows * shadow_mask * 0.75
                        + highlights * highlight_mask * 0.75
                        + blacks * shadow_mask * 0.5
                        + whites * highlight_mask * 0.5,
                );
                luma = 0.18 * (luma / 0.18).max(1e-6).powf(1.0 + contrast);
                let ratio = luma.max(0.0) / original_luma;
                for channel in &mut rgb {
                    *channel *= ratio;
                }
                let luma = luminance(rgb);
                let chroma = rgb
                    .iter()
                    .map(|channel| (channel - luma).abs())
                    .fold(0.0_f32, f32::max);
                let adaptive = 1.0 + saturation + vibrance * (1.0 - chroma.clamp(0.0, 1.0));
                for channel in &mut rgb {
                    *channel = luma + (*channel - luma) * adaptive.max(0.0);
                }
                if let Some(curve) = self.tone_curve {
                    let before = luminance(rgb).max(1e-6);
                    let after = sample_curve(curve, before.clamp(0.0, 1.0));
                    let ratio = after.max(0.0) / before;
                    for channel in &mut rgb {
                        *channel *= ratio;
                    }
                }
                for (channel, curve) in [
                    self.tone_curve_red,
                    self.tone_curve_green,
                    self.tone_curve_blue,
                ]
                .into_iter()
                .enumerate()
                {
                    if let Some(curve) = curve {
                        rgb[channel] = sample_curve(curve, rgb[channel].clamp(0.0, 1.0));
                    }
                }
                *pixel = Rgba([
                    linear_to_srgb(rgb[0]),
                    linear_to_srgb(rgb[1]),
                    linear_to_srgb(rgb[2]),
                    alpha,
                ]);
            }
        }
        Ok(())
    }
}

fn absolute_white_balance(wb: WhiteBalance) -> [f32; 3] {
    let temperature =
        ((wb.temperature.clamp(2_000.0, 50_000.0) - 6_500.0) / 43_500.0).clamp(-1.0, 1.0);
    let tint = (wb.tint.clamp(-150.0, 150.0) / 150.0).clamp(-1.0, 1.0);
    [
        1.0 + temperature * 0.45 + tint * 0.08,
        1.0 - tint * 0.12,
        1.0 - temperature * 0.45 + tint * 0.08,
    ]
}

fn incremental_white_balance(wb: WhiteBalanceIncrement) -> [f32; 3] {
    let temperature = (wb.temperature / 100.0).clamp(-1.0, 1.0);
    let tint = (wb.tint / 100.0).clamp(-1.0, 1.0);
    [
        1.0 + temperature * 0.15 + tint * 0.05,
        1.0 - tint * 0.08,
        1.0 - temperature * 0.15 + tint * 0.05,
    ]
}
