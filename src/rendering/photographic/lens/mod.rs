//! Lens correction for raw renders: profile types, runtime selection, and
//! the scene-linear application operators (peripheral illumination gain,
//! backward-mapped distortion/lateral-CA resample).
//!
//! Self-contained by design (std + serde only): the calibration binary
//! includes this module via `#[path]` and shares the profile format, the
//! estimators, and the application math with the app.
//!
//! Coordinates: radii are normalized to the half-diagonal of the cropped
//! sensor frame; the optical center is the frame center. Orientation is
//! applied later in the pipeline, so operators and calibration both work
//! in sensor space.

// Included verbatim by the calibration bin, which does not use every
// app-side helper; the module stays shared so formats cannot drift.
#![allow(dead_code)]

pub mod calibrate;

use serde::{Deserialize, Serialize};

/// Radii of the vignette gain table: entry i sits at radius i / (LEN - 1).
pub const VIGNETTE_LUT_LEN: usize = 33;

/// Calibrated lens correction for one camera model. Produced by the
/// `lens-profile-calibrate` bin from raw+camera-JPEG pairs.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LensProfile {
    /// Matches rawler's `RawImage::clean_model` (case-insensitive).
    pub model: String,
    /// Calibration points along the zoom range, sorted by `focal_mm`.
    pub entries: Vec<FocalEntry>,
}

/// One focal-length calibration entry. All radii normalize to the frame
/// half-diagonal; gains are multiplicative on scene-linear values with
/// `vignette[0] == 1.0` at the center.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FocalEntry {
    pub focal_mm: f32,
    /// Exactly [`VIGNETTE_LUT_LEN`] gains (serde stops at 32 for arrays).
    pub vignette: Vec<f32>,
    /// Backward radial distortion: `r_src = r_out * (1 + k1·r_out² + k2·r_out⁴)`.
    pub k1: f32,
    pub k2: f32,
    /// Lateral chromatic aberration as per-channel radial scales relative
    /// to green (1.0).
    pub scale_r: f32,
    pub scale_b: f32,
}

/// Convenience wrapper used by the render pipeline: select the entry for
/// a camera model + focal length from an embedded profile.
impl LensProfile {
    /// The interpolated correction entry for a shot, or `None` when the
    /// profile does not match the camera. Focal lengths between
    /// calibration points interpolate; out-of-range shots clamp to the
    /// nearest entry.
    pub fn select(&self, model: &str, focal_mm: Option<f32>) -> Option<FocalEntry> {
        if !self.model.eq_ignore_ascii_case(model) || self.entries.is_empty() {
            return None;
        }
        let focal = focal_mm?;
        let lower = self
            .entries
            .iter()
            .rev()
            .find(|entry| entry.focal_mm <= focal)
            .or_else(|| self.entries.first())?;
        let upper = self
            .entries
            .iter()
            .find(|entry| entry.focal_mm >= focal)
            .or_else(|| self.entries.last())?;
        if std::ptr::eq(lower, upper) || (upper.focal_mm - lower.focal_mm) <= f32::EPSILON {
            return Some(lower.clone());
        }
        let t = (focal - lower.focal_mm) / (upper.focal_mm - lower.focal_mm);
        let lerp = |a: f32, b: f32| a + (b - a) * t;
        let mut vignette = vec![0.0; VIGNETTE_LUT_LEN];
        for (index, value) in vignette.iter_mut().enumerate() {
            *value = lerp(lower.vignette[index], upper.vignette[index]);
        }
        Some(FocalEntry {
            focal_mm: focal,
            vignette,
            k1: lerp(lower.k1, upper.k1),
            k2: lerp(lower.k2, upper.k2),
            scale_r: lerp(lower.scale_r, upper.scale_r),
            scale_b: lerp(lower.scale_b, upper.scale_b),
        })
    }
}

/// Sample the vignette gain table at a normalized radius (bilinear over
/// the LUT, radii clamp to 0..1).
pub fn vignette_at(lut: &[f32], radius: f32) -> f32 {
    let scaled = radius.clamp(0.0, 1.0) * (VIGNETTE_LUT_LEN - 1) as f32;
    let index = scaled.floor() as usize;
    let upper = (index + 1).min(VIGNETTE_LUT_LEN - 1);
    let t = scaled - index as f32;
    lut[index] * (1.0 - t) + lut[upper] * t
}

/// Multiply scene-linear pixels by the peripheral illumination gain. The
/// radius runs over the frame half-diagonal from the frame center.
pub fn apply_vignette(pixels: &mut [[f32; 3]], width: u32, height: u32, lut: &[f32]) {
    let (cx, cy) = (width as f32 / 2.0, height as f32 / 2.0);
    let r_max = (cx * cx + cy * cy).sqrt();
    for (index, pixel) in pixels.iter_mut().enumerate() {
        let x = index as u32 % width;
        let y = index as u32 / width;
        let dx = x as f32 + 0.5 - cx;
        let dy = y as f32 + 0.5 - cy;
        let radius = (dx * dx + dy * dy).sqrt() / r_max;
        let gain = vignette_at(lut, radius);
        pixel[0] *= gain;
        pixel[1] *= gain;
        pixel[2] *= gain;
    }
}

/// Geometric correction. The profile's `k1`/`k2` describe the measured
/// clean→raw magnification `a(r) = 1 + k1·r² + k2·r⁴` (a clean feature at
/// normalized radius r sits at radius r·a(r) in the raw). The correction
/// gathers, for every output pixel, the raw radius that maps to it — the
/// fixed point of `t·a(t) = r_out` — per channel scaled by the lateral CA
/// factors (green = 1). Same-size canvas; edges clamp.
pub fn apply_geometry(pixels: &mut Vec<[f32; 3]>, width: u32, height: u32, entry: &FocalEntry) {
    let magnification = |r: f32| 1.0 + entry.k1 * r * r + entry.k2 * r * r * r * r;
    let (cx, cy) = (width as f32 / 2.0, height as f32 / 2.0);
    let r_max = (cx * cx + cy * cy).sqrt();
    let mut warped = vec![[0.0_f32; 3]; pixels.len()];
    for (index, out) in warped.iter_mut().enumerate() {
        let x = index as u32 % width;
        let y = index as u32 / width;
        let dx = x as f32 + 0.5 - cx;
        let dy = y as f32 + 0.5 - cy;
        let r_out = (dx * dx + dy * dy).sqrt() / r_max;
        // Solve t·a(t) = r_out. The naive t = r_out / a(r_out) is only
        // first-order accurate; iterate to the fixed point (converges in
        // a few steps for smooth magnifications).
        let mut t = r_out;
        for _ in 0..4 {
            t = r_out / magnification(t);
        }
        let factor = t / r_out;
        for (channel, scale) in [(0usize, entry.scale_r), (1, 1.0), (2, entry.scale_b)] {
            let offset = factor * scale;
            out[channel] = sample_bilinear(
                pixels,
                width,
                height,
                cx + dx * offset - 0.5,
                cy + dy * offset - 0.5,
                channel,
            );
        }
    }
    *pixels = warped;
}

/// Forward corruption (clean→raw) used by tests and the synthetic
/// calibration round trips: gathers the clean scene at `t·a(t)`, placing
/// clean content at the radius it would occupy in the raw.
#[cfg(test)]
pub(crate) fn corrupt_forward(
    pixels: &mut Vec<[f32; 3]>,
    width: u32,
    height: u32,
    entry: &FocalEntry,
) {
    let magnification = |r: f32| 1.0 + entry.k1 * r * r + entry.k2 * r * r * r * r;
    let (cx, cy) = (width as f32 / 2.0, height as f32 / 2.0);
    let r_max = (cx * cx + cy * cy).sqrt();
    let mut warped = vec![[0.0_f32; 3]; pixels.len()];
    for (index, out) in warped.iter_mut().enumerate() {
        let x = index as u32 % width;
        let y = index as u32 / width;
        let dx = x as f32 + 0.5 - cx;
        let dy = y as f32 + 0.5 - cy;
        let r_out = (dx * dx + dy * dy).sqrt() / r_max;
        let factor = magnification(r_out);
        for (channel, scale) in [(0usize, entry.scale_r), (1, 1.0), (2, entry.scale_b)] {
            let offset = factor * scale;
            out[channel] = sample_bilinear(
                pixels,
                width,
                height,
                cx + dx * offset - 0.5,
                cy + dy * offset - 0.5,
                channel,
            );
        }
    }
    *pixels = warped;
}

fn sample_bilinear(
    pixels: &[[f32; 3]],
    width: u32,
    height: u32,
    x: f32,
    y: f32,
    channel: usize,
) -> f32 {
    let x0 = (x.floor().max(0.0) as u32).min(width - 1);
    let y0 = (y.floor().max(0.0) as u32).min(height - 1);
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);
    let tx = (x - x0 as f32).clamp(0.0, 1.0);
    let ty = (y - y0 as f32).clamp(0.0, 1.0);
    let at = |x: u32, y: u32| pixels[(y * width + x) as usize][channel];
    let top = at(x0, y0) * (1.0 - tx) + at(x1, y0) * tx;
    let bottom = at(x0, y1) * (1.0 - tx) + at(x1, y1) * tx;
    top * (1.0 - ty) + bottom * ty
}

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(unused_imports)] // test builds only; resolves the #[test] macro
    use test_r::test;

    fn profile() -> LensProfile {
        let mut low = vec![1.0; VIGNETTE_LUT_LEN];
        for (index, value) in low.iter_mut().enumerate() {
            *value = 1.0 + index as f32 / (VIGNETTE_LUT_LEN - 1) as f32;
        }
        let mut high = vec![1.0; VIGNETTE_LUT_LEN];
        for (index, value) in high.iter_mut().enumerate() {
            *value = 1.0 + 2.0 * index as f32 / (VIGNETTE_LUT_LEN - 1) as f32;
        }
        LensProfile {
            model: "PowerShot V1".to_owned(),
            entries: vec![
                FocalEntry {
                    focal_mm: 8.2,
                    vignette: low,
                    k1: 0.1,
                    k2: 0.02,
                    scale_r: 1.001,
                    scale_b: 0.999,
                },
                FocalEntry {
                    focal_mm: 25.6,
                    vignette: high,
                    k1: 0.02,
                    k2: 0.0,
                    scale_r: 1.0,
                    scale_b: 1.0,
                },
            ],
        }
    }

    #[test]
    fn selection_interpolates_between_focal_lengths() {
        let profile = profile();
        let entry = profile.select("powershot v1", Some(16.9)).unwrap();
        // 16.9 mm sits exactly halfway between the 8.2 and 25.6 entries.
        assert!((entry.k1 - 0.06).abs() < 1e-6, "k1 {}", entry.k1);
        assert!((entry.vignette[32] - 2.5).abs() < 1e-6);
        assert!(profile.select("Other Camera", Some(8.2)).is_none());
        assert!(profile.select("PowerShot V1", None).is_none());
        // Out-of-range focal lengths clamp to the nearest entry.
        assert_eq!(
            profile.select("PowerShot V1", Some(100.0)).unwrap().k1,
            0.02
        );
    }

    #[test]
    fn vignette_gain_rises_to_the_corner_value() {
        let mut lut = vec![1.0; VIGNETTE_LUT_LEN];
        for (index, value) in lut.iter_mut().enumerate() {
            *value = 1.0 + index as f32;
        }
        assert_eq!(vignette_at(&lut, 0.0), 1.0);
        assert_eq!(vignette_at(&lut, 1.0), 33.0);
        assert!((vignette_at(&lut, 0.5) - 17.0).abs() < 1e-5);
        // Radii clamp instead of extrapolating.
        assert_eq!(vignette_at(&lut, 2.0), 33.0);
    }

    #[test]
    fn identity_geometry_leaves_the_frame_unchanged() {
        let mut pixels = vec![[0.25, 0.5, 0.75]; 64 * 48];
        let snapshot = pixels.clone();
        apply_geometry(
            &mut pixels,
            64,
            48,
            &FocalEntry {
                focal_mm: 8.2,
                vignette: vec![1.0; VIGNETTE_LUT_LEN],
                k1: 0.0,
                k2: 0.0,
                scale_r: 1.0,
                scale_b: 1.0,
            },
        );
        assert!(
            pixels
                .iter()
                .zip(snapshot)
                .all(|(a, b)| a[0] == b[0] && a[1] == b[1] && a[2] == b[2])
        );
    }

    /// THE direction guard: applying the operator to a corrupted frame
    /// must UNDO the corruption. The calibration fits the clean→raw
    /// magnification; a test that only checks the fit (not this round
    /// trip) cannot see an inverted application.
    #[test]
    fn geometry_correction_undoes_the_measured_distortion() {
        let entry = FocalEntry {
            focal_mm: 8.2,
            vignette: vec![1.0; VIGNETTE_LUT_LEN],
            k1: 0.18,
            k2: 0.04,
            scale_r: 1.0,
            scale_b: 1.0,
        };
        let (width, height) = (256_u32, 192_u32);
        let clean = vec![[0.5; 3]; (width * height) as usize];
        // Smooth gradient: bilinear resampling error stays negligible, so
        // any remaining difference is a direction/parameter error.
        let mut pattern = clean.clone();
        for (index, pixel) in pattern.iter_mut().enumerate() {
            let x = index as u32 % width;
            let y = index as u32 / width;
            let value =
                0.15 + 0.7 * (x as f32 / width as f32 * 0.6 + y as f32 / height as f32 * 0.4);
            *pixel = [value * 0.9, value, value * 1.1];
        }
        // raw = the forward corruption applied to clean.
        let mut raw = pattern.clone();
        corrupt_forward(&mut raw, width, height, &entry);
        // The raw must differ from clean (the corruption did something).
        let differing = raw
            .iter()
            .zip(&pattern)
            .filter(|(a, b)| (a[0] - b[0]).abs() > 0.01)
            .count();
        assert!(
            differing > (width * height) as usize / 4,
            "corruption no-op"
        );

        // Correction with the same profile must recover the clean frame.
        let mut corrected = raw.clone();
        apply_geometry(&mut corrected, width, height, &entry);
        let error: f32 = corrected
            .iter()
            .zip(&pattern)
            .map(|(a, b)| (a[0] - b[0]).abs().max((a[1] - b[1]).abs()))
            .sum::<f32>()
            / corrected.len() as f32;
        assert!(error < 0.01, "mean correction error {error}");
    }

    #[test]
    fn corner_gains_follow_the_profile() {
        // Flat frame, corner-lifting vignette: the center must stay put and
        // the corner must land on the LUT's corner gain.
        let mut lut = [1.0; VIGNETTE_LUT_LEN];
        *lut.last_mut().unwrap() = 4.0;
        let mut pixels = vec![[1.0; 3]; 100 * 100];
        apply_vignette(&mut pixels, 100, 100, &lut);
        let at = |x: usize, y: usize| pixels[y * 100 + x][0];
        assert!((at(50, 50) - 1.0).abs() < 1e-5);
        // The corner pixel's center sits at r = √2·49.5 / half-diagonal.
        let corner_radius =
            ((49.5_f32 * 49.5 * 2.0).sqrt()) / ((50.0_f32 * 50.0 + 50.0 * 50.0).sqrt());
        assert!(
            (at(0, 0) - vignette_at(&lut, corner_radius)).abs() < 1e-4,
            "corner {} vs r {} → {}",
            at(0, 0),
            corner_radius,
            vignette_at(&lut, corner_radius)
        );
    }
}
