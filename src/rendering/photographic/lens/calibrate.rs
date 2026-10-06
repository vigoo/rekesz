//! Calibration estimators: recover a [`super::FocalEntry`] from pairs of
//! (uncorrected raw render, camera-corrected JPEG), both provided as
//! scene-linear RGB buffers of the same dimensions.
//!
//! The camera JPEG is ground truth by construction; all global monotone
//! differences between the two renderings (tone curve, white balance,
//! exposure) are absorbed by a center-anchored monotone fit, so the
//! remaining radial structure is the lens itself.

use super::{FocalEntry, VIGNETTE_LUT_LEN};

/// One raw↔JPEG pair as scene-linear RGB (JPEG already linearized).
pub struct Pair<'a> {
    pub raw: &'a [[f32; 3]],
    pub jpeg: &'a [[f32; 3]],
    pub width: u32,
    pub height: u32,
}

fn luma(pixel: &[f32; 3]) -> f32 {
    0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2]
}

fn radius_map(width: u32, height: u32) -> (f32, f32, f32) {
    let (cx, cy) = (width as f32 / 2.0, height as f32 / 2.0);
    let r_max = (cx * cx + cy * cy).sqrt();
    (cx, cy, r_max)
}

/// Median of a slice (copies and sorts).
fn median(values: &[f32]) -> Option<f32> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Some(sorted[sorted.len() / 2])
}

/// Pool-adjacent-violators isotonic regression (non-decreasing), weighted
/// by block size.
fn isotonic(values: &[f32]) -> Vec<f32> {
    let mut blocks: Vec<(f32, usize)> = Vec::with_capacity(values.len());
    for &value in values {
        let mut block = (value, 1usize);
        loop {
            match blocks.last() {
                Some(&(last, count)) if last > block.0 => {
                    let pooled =
                        (last * count as f32 + block.0 * block.1 as f32) / (count + block.1) as f32;
                    blocks.pop();
                    block = (pooled, count + block.1);
                }
                _ => break,
            }
        }
        blocks.push(block);
    }
    blocks
        .into_iter()
        .flat_map(|(value, count)| std::iter::repeat_n(value, count))
        .collect()
}

/// Fit the camera's monotone tone curve `J = T(L)` from the center region
/// (r < 0.25, where lens falloff is negligible), and return a lookup that
/// inverts it: `T⁻¹(J) → L`.
struct ToneFit {
    /// L bin centers (ascending).
    l_bins: Vec<f32>,
    /// Isotonic-smoothed J means per L bin.
    j_values: Vec<f32>,
}

impl ToneFit {
    const BINS: usize = 96;

    fn fit(raw: &[[f32; 3]], jpeg: &[[f32; 3]], width: u32, height: u32) -> Self {
        let (cx, cy, r_max) = radius_map(width, height);
        let r_center = 0.25 * r_max;
        let mut luma_samples: Vec<(f32, f32)> = Vec::new();
        for (index, (raw_pixel, jpeg_pixel)) in raw.iter().zip(jpeg).enumerate() {
            let x = index as u32 % width;
            let y = index as u32 / width;
            let dx = x as f32 + 0.5 - cx;
            let dy = y as f32 + 0.5 - cy;
            if dx * dx + dy * dy > r_center * r_center {
                continue;
            }
            let l = luma(raw_pixel);
            let j = luma(jpeg_pixel);
            if l > 1e-6 && j > 1e-4 && j < 0.98 {
                luma_samples.push((l, j));
            }
        }
        luma_samples.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut l_bins = vec![0.0_f32; Self::BINS];
        let mut j_means = vec![0.0_f32; Self::BINS];
        if !luma_samples.is_empty() {
            let chunk = luma_samples.len().div_ceil(Self::BINS);
            for (bin, group) in luma_samples.chunks(chunk).enumerate() {
                let l_mean = group.iter().map(|(l, _)| l).sum::<f32>() / group.len() as f32;
                let j_mean = group.iter().map(|(_, j)| j).sum::<f32>() / group.len() as f32;
                l_bins[bin] = l_mean;
                j_means[bin] = j_mean;
            }
            j_means = isotonic(&j_means);
        }
        Self {
            l_bins,
            j_values: j_means,
        }
    }

    /// `T(L)`: the jpeg-domain luminance the curve maps `l` to.
    fn forward(&self, l: f32) -> Option<f32> {
        if self.l_bins.is_empty() {
            return None;
        }
        let upper = match self.l_bins.partition_point(|&value| value < l) {
            0 => return Some(self.j_values[0]),
            index if index >= self.l_bins.len() => {
                return Some(*self.j_values.last().unwrap())
            }
            index => index,
        };
        let lower = upper.saturating_sub(1);
        let (l0, l1) = (self.l_bins[lower], self.l_bins[upper]);
        let (j0, j1) = (self.j_values[lower], self.j_values[upper]);
        if l1 <= l0 {
            return Some(j0);
        }
        Some(j0 + (j1 - j0) * (l - l0) / (l1 - l0))
    }

    /// `T⁻¹(J)`: the raw-domain luminance the curve maps to `j`.
    fn invert(&self, j: f32) -> Option<f32> {
        if self.l_bins.is_empty() {
            return None;
        }
        // Locate the bracketing isotonic bins.
        let upper = match self.j_values.partition_point(|&value| value < j) {
            0 => return Some(self.l_bins[0]),
            index if index >= self.j_values.len() => return Some(*self.l_bins.last().unwrap()),
            index => index,
        };
        let lower = upper.saturating_sub(1);
        let (j0, j1) = (self.j_values[lower], self.j_values[upper]);
        let (l0, l1) = (self.l_bins[lower], self.l_bins[upper]);
        if j1 <= j0 {
            return Some(l0);
        }
        Some(l0 + (l1 - l0) * (j - j0) / (j1 - j0))
    }
}

/// Peripheral illumination gain over `pairs`: per pair, the
/// tone-neutralized luminance ratio binned by radius (annulus medians),
/// then median-combined across pairs and normalized to `G(0) = 1`.
/// Radii with insufficient data stay at 1.0.
pub fn vignette_gain(pairs: &[Pair<'_>]) -> Vec<f32> {
    const RADII: usize = VIGNETTE_LUT_LEN;
    let mut combined: Vec<Vec<f32>> = vec![Vec::new(); RADII];
    for pair in pairs {
        let tone = ToneFit::fit(pair.raw, pair.jpeg, pair.width, pair.height);
        let (cx, cy, r_max) = radius_map(pair.width, pair.height);
        let mut per_bin: Vec<Vec<f32>> = vec![Vec::new(); RADII];
        for (index, (raw_pixel, jpeg_pixel)) in pair.raw.iter().zip(pair.jpeg).enumerate() {
            let x = index as u32 % pair.width;
            let y = index as u32 / pair.width;
            let dx = x as f32 + 0.5 - cx;
            let dy = y as f32 + 0.5 - cy;
            let r = (dx * dx + dy * dy).sqrt() / r_max;
            if r >= 1.0 {
                continue;
            }
            let l = luma(raw_pixel);
            let j = luma(jpeg_pixel);
            if !(0.02..0.95).contains(&j) || l <= 1e-5 {
                continue;
            }
            let Some(l_inverted) = tone.invert(j) else {
                continue;
            };
            let ratio = l_inverted / l;
            if !(0.05..20.0).contains(&ratio) {
                continue;
            }
            let bin = ((r * (RADII - 1) as f32).round() as usize).min(RADII - 1);
            per_bin[bin].push(ratio);
        }
        for (bin, samples) in per_bin.into_iter().enumerate() {
            if let Some(value) = median(&samples) {
                combined[bin].push(value);
            }
        }
    }
    let mut curve: Vec<f32> = combined
        .iter()
        .map(|values| median(values).unwrap_or(1.0))
        .collect();
    // Normalize: the innermost populated bin is the unit point.
    let unity = curve
        .iter()
        .copied()
        .find(|&value| value > 0.0)
        .unwrap_or(1.0);
    if unity > 0.0 {
        for value in &mut curve {
            *value /= unity;
        }
    }
    curve
}

/// Patch-based normalized cross-correlation displacement of `src` against
/// `dst` at one pyramid level. Returns `(dst_x, dst_y, dx, dy)` in that
/// level's coordinates for confident matches.
fn ncc_displacements(
    dst: &GrayBuffer,
    src: &GrayBuffer,
    patch: usize,
    search: i32,
    min_ncc: f32,
) -> Vec<(f32, f32, f32, f32)> {
    let mut matches = Vec::new();
    let stride = patch;
    let mut by = patch;
    while by + patch <= dst.height as usize {
        let mut bx = patch;
        while bx + patch <= dst.width as usize {
            let mut best = (f32::MIN, 0_i32, 0_i32);
            for oy in -search..=search {
                for ox in -search..=search {
                    let sx = bx as i32 + ox;
                    let sy = by as i32 + oy;
                    if sx < 0
                        || sy < 0
                        || sx + patch as i32 > src.width as i32
                        || sy + patch as i32 > src.height as i32
                    {
                        continue;
                    }
                    let score = ncc(dst, src, bx, by, sx as usize, sy as usize, patch);
                    if score > best.0 {
                        best = (score, ox, oy);
                    }
                }
            }
            if best.0 >= min_ncc {
                matches.push((bx as f32, by as f32, best.1 as f32, best.2 as f32));
            }
            bx += stride;
        }
        by += stride;
    }
    matches
}

struct GrayBuffer {
    width: u32,
    height: u32,
    data: Vec<f32>,
}

fn gray(rgb: &[[f32; 3]], width: u32, height: u32) -> GrayBuffer {
    GrayBuffer {
        width,
        height,
        data: rgb.iter().map(luma).collect(),
    }
}

fn downsample(buffer: &GrayBuffer) -> GrayBuffer {
    let width = (buffer.width / 2).max(1);
    let height = (buffer.height / 2).max(1);
    let mut data = vec![0.0; (width * height) as usize];
    for y in 0..height {
        for x in 0..width {
            let (sx, sy) = (x as usize * 2, y as usize * 2);
            let at = |dx: usize, dy: usize| {
                buffer.data[(sy + dy).min(buffer.height as usize - 1) * buffer.width as usize
                    + (sx + dx).min(buffer.width as usize - 1)]
            };
            data[(y * width + x) as usize] = (at(0, 0) + at(1, 0) + at(0, 1) + at(1, 1)) / 4.0;
        }
    }
    GrayBuffer {
        width,
        height,
        data,
    }
}

fn ncc(
    dst: &GrayBuffer,
    src: &GrayBuffer,
    dx: usize,
    dy: usize,
    sx: usize,
    sy: usize,
    patch: usize,
) -> f32 {
    let mut sum_d = 0.0;
    let mut sum_s = 0.0;
    for y in 0..patch {
        for x in 0..patch {
            let a = dst.data[(dy + y) * dst.width as usize + dx + x];
            let b = src.data[(sy + y) * src.width as usize + sx + x];
            sum_d += a;
            sum_s += b;
        }
    }
    let count = (patch * patch) as f32;
    let (mean_d, mean_s) = (sum_d / count, sum_s / count);
    let mut cov = 0.0;
    let mut var_d = 0.0;
    let mut var_s = 0.0;
    for y in 0..patch {
        for x in 0..patch {
            let a = dst.data[(dy + y) * dst.width as usize + dx + x] - mean_d;
            let b = src.data[(sy + y) * src.width as usize + sx + x] - mean_s;
            cov += a * b;
            var_d += a * a;
            var_s += b * b;
        }
    }
    let denom = (var_d * var_s).sqrt();
    if denom < 1e-9 { 0.0 } else { cov / denom }
}

/// Estimated geometry for one focal length, plus fit diagnostics.
#[derive(Debug, Clone, Copy)]
pub struct GeometryFit {
    pub k1: f32,
    pub k2: f32,
    pub scale_r: f32,
    pub scale_b: f32,
    /// RMS radial residual of the polynomial fit (relative units).
    pub residual: f32,
    /// Number of displacement samples that survived the fit's trimming.
    pub samples: usize,
    /// Largest normalized radius the displacement field reached.
    pub max_radius: f32,
    /// Median tangential displacement (should be ~0 for a pure radial
    /// distortion; large values mean the fit is not trustworthy).
    pub tangential: f32,
}

/// Distortion + lateral CA from one pair: a coarse-to-fine patch NCC
/// displacement field between the corrected JPEG (destination) and the
/// raw render (source), fitted with a radial polynomial; per-channel
/// scales from the raw's own R↔G / B↔G displacements in the corners.
pub fn geometry_fit(pair: &Pair<'_>) -> GeometryFit {
    // Pyramid: coarsest level small enough that the largest displacement
    // (~0.25·half-diagonal at the wide end) sits inside the search window;
    // each finer level re-searches ±2 around the doubled coarse match.
    let mut levels: Vec<(GrayBuffer, GrayBuffer)> = vec![(
        gray(pair.jpeg, pair.width, pair.height),
        gray(pair.raw, pair.width, pair.height),
    )];
    while levels.last().unwrap().0.width / 2 >= 96
        && levels.last().unwrap().0.height / 2 >= 64
        && levels.len() < 8
    {
        let (dst, src) = levels.last().unwrap();
        levels.push((downsample(dst), downsample(src)));
    }
    let (coarse_dst, coarse_src) = levels.last().unwrap();
    let mut matches = ncc_displacements(coarse_dst, coarse_src, 12, 14, 0.3);
    let mut working = levels.len() - 1;

    for level in (0..levels.len() - 1).rev() {
        if matches.len() < 30 {
            break;
        }
        let (dst, src) = &levels[level];
        let mut refined = Vec::with_capacity(matches.len());
        for &(bx, by, dx, dy) in &matches {
            let (bx2, by2) = (bx * 2.0, by * 2.0);
            let (seed_x, seed_y) = (dx * 2.0, dy * 2.0);
            if bx2 + 12.0 > dst.width as f32 || by2 + 12.0 > dst.height as f32 {
                continue;
            }
            let mut best = (f32::MIN, seed_x as i32, seed_y as i32);
            for oy in -2_i32..=2 {
                for ox in -2_i32..=2 {
                    let sx = bx2 as i32 + seed_x as i32 / 2 + ox;
                    let sy = by2 as i32 + seed_y as i32 / 2 + oy;
                    if sx < 0 || sy < 0 || sx + 12 > dst.width as i32 || sy + 12 > dst.height as i32
                    {
                        continue;
                    }
                    let score = ncc(
                        dst,
                        src,
                        bx2 as usize,
                        by2 as usize,
                        sx as usize,
                        sy as usize,
                        12,
                    );
                    if score > best.0 {
                        best = (score, sx, sy);
                    }
                }
            }
            if best.0 >= 0.3 {
                // Subpixel offset from the NCC surface's neighborhood.
                let (bx_i, by_i) = (bx2 as usize, by2 as usize);
                let at = |ox: i32, oy: i32| {
                    let sx = best.1 + ox;
                    let sy = best.2 + oy;
                    if sx < 0 || sy < 0 || sx + 12 > dst.width as i32 || sy + 12 > dst.height as i32
                    {
                        return best.0;
                    }
                    ncc(dst, src, bx_i, by_i, sx as usize, sy as usize, 12)
                };
                let sub_x = subpixel_shift(at(-1, 0), best.0, at(1, 0));
                let sub_y = subpixel_shift(at(0, -1), best.0, at(0, 1));
                refined.push((
                    bx2,
                    by2,
                    best.1 as f32 - bx2 + sub_x,
                    best.2 as f32 - by2 + sub_y,
                ));
            }
        }
        if refined.len() * 2 >= matches.len() {
            matches = refined;
            working = level;
        }

        // Enough precision already; skip refining the largest levels.
        if levels[level].0.width <= 1400 {
            break;
        }
    }

    // Fit in the working level's coordinates, normalized radii.
    let (dst_level, _) = &levels[working];
    let (cx, cy, r_max) = radius_map(dst_level.width, dst_level.height);
    let mut samples: Vec<(f32, f32)> = Vec::new();
    let mut tangential_samples: Vec<f32> = Vec::new();
    for &(bx, by, dx, dy) in &matches {
        let pdx = bx + 6.0 - cx;
        let pdy = by + 6.0 - cy;
        let r_dst = (pdx * pdx + pdy * pdy).sqrt();
        if r_dst < 0.15 * r_max {
            continue;
        }
        let sdx = pdx + dx;
        let sdy = pdy + dy;
        let r_src = (sdx * sdx + sdy * sdy).sqrt();
        if r_dst <= 0.0 || r_src <= 0.0 {
            continue;
        }
        // Runtime samples the raw at r_src = r_out · s(r_out); the
        // measured field is the inverse ratio, so invert it here.
        samples.push((r_dst / r_max, r_dst / r_src - 1.0));
        let (ux, uy) = (pdx / r_dst, pdy / r_dst);
        tangential_samples.push((-dx * uy + dy * ux).abs() / r_dst);
    }
    let (k1, k2, residual) = fit_radial(&samples);
    let tangential = median(&tangential_samples).unwrap_or(0.0);
    let max_radius = samples
        .iter()
        .map(|(r, _)| *r)
        .fold(0.0_f32, |acc, r| acc.max(r));

    // Lateral CA from the raw's own channel displacements (camera JPEG
    // channels are corrected by construction).
    let (scale_r, scale_b) = channel_scales(pair, 20, 5);

    GeometryFit {
        k1,
        k2,
        scale_r,
        scale_b,
        residual,
        tangential,
        samples: samples.len(),
        max_radius,
    }
}

/// Parabola-vertex subpixel offset from three symmetric samples.
fn subpixel_shift(minus: f32, center: f32, plus: f32) -> f32 {
    let denom = minus - 2.0 * center + plus;
    if denom <= 1e-9 {
        return 0.0;
    }
    (0.5 * (minus - plus) / denom).clamp(-0.5, 0.5)
}

fn fit_radial(samples: &[(f32, f32)]) -> (f32, f32, f32) {
    let mut active = samples.to_vec();
    let mut k1 = 0.0_f32;
    let mut k2 = 0.0_f32;
    let mut residual = f32::INFINITY;
    for _ in 0..3 {
        if active.len() < 16 {
            break;
        }
        // Radius-weighted normal equations for y = a·x1 + b·x2 with
        // x1 = r², x2 = r⁴: larger radii carry the larger signal, so they
        // dominate the fit.
        let (mut s11, mut s12, mut s22, mut sy1, mut sy2) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for &(r, y) in &active {
            let weight = r * r;
            let (x1, x2) = (r * r * weight, r * r * r * r * weight);
            s11 += r * r * x1;
            s12 += r * r * x2;
            s22 += x2 * x2;
            sy1 += x1 * y;
            sy2 += x2 * y;
        }
        let det = s11 * s22 - s12 * s12;
        if det.abs() < 1e-12 {
            break;
        }
        k1 = (sy1 * s22 - sy2 * s12) / det;
        k2 = (s11 * sy2 - s12 * sy1) / det;
        let errors: Vec<f32> = active
            .iter()
            .map(|&(r, y)| {
                let predicted = k1 * r * r + k2 * r * r * r * r;
                (y - predicted).abs()
            })
            .collect();
        residual = median(&errors).unwrap_or(f32::INFINITY);
        // Trim outliers and refit.
        let limit = 2.5 * residual.max(1e-4);
        let next: Vec<(f32, f32)> = active
            .iter()
            .copied()
            .filter(|&(r, y)| {
                let predicted = k1 * r * r + k2 * r * r * r * r;
                (y - predicted).abs() <= limit
            })
            .collect();
        if next.len() < active.len() / 2 {
            break;
        }
        active = next;
    }
    (k1, k2, residual)
}

/// Per-channel radial scale of `raw`'s R and B channels relative to its G
/// channel (corner patches, patch NCC): `1 + median(radial_shift / r)`.
fn channel_scales(pair: &Pair<'_>, patch: usize, search: i32) -> (f32, f32) {
    let mut src_r = Vec::with_capacity(pair.raw.len());
    let mut src_g = Vec::with_capacity(pair.raw.len());
    let mut src_b = Vec::with_capacity(pair.raw.len());
    for pixel in pair.raw {
        src_r.push(gray_channel(pixel, 0));
        src_g.push(gray_channel(pixel, 1));
        src_b.push(gray_channel(pixel, 2));
    }
    let (width, height) = (pair.width, pair.height);
    let r_g = GrayBuffer {
        width,
        height,
        data: src_g,
    };
    let r_r = GrayBuffer {
        width,
        height,
        data: src_r,
    };
    let r_b = GrayBuffer {
        width,
        height,
        data: src_b,
    };
    let scales_for = |other: &GrayBuffer| -> f32 {
        let matches = ncc_displacements_within(&r_g, other, patch, search, 0.4, 96);
        let (cx, cy, r_max) = radius_map(width, height);
        let mut per_patch: Vec<f32> = Vec::new();
        for &(bx, by, dx, dy) in &matches {
            let px = bx + patch as f32 / 2.0;
            let py = by + patch as f32 / 2.0;
            let pdx = px - cx;
            let pdy = py - cy;
            let r = (pdx * pdx + pdy * pdy).sqrt();
            if r < 0.45 * r_max || r <= 0.0 {
                continue;
            }
            let (ux, uy) = (pdx / r, pdy / r);
            per_patch.push((dx * ux + dy * uy) / r);
        }
        1.0 + median(&per_patch).unwrap_or(0.0)
    };
    (scales_for(&r_r), scales_for(&r_b))
}

fn gray_channel(pixel: &[f32; 3], channel: usize) -> f32 {
    pixel[channel]
}

/// [`ncc_displacements`] restricted to a maximum number of patches
/// (samples evenly from the grid) — keeps the channel pass cheap.
fn ncc_displacements_within(
    dst: &GrayBuffer,
    src: &GrayBuffer,
    patch: usize,
    search: i32,
    min_ncc: f32,
    max_patches: usize,
) -> Vec<(f32, f32, f32, f32)> {
    let all = ncc_displacements(dst, src, patch, search, min_ncc);
    if all.len() <= max_patches {
        return all;
    }
    let step = all.len().div_ceil(max_patches);
    all.into_iter().step_by(step).collect()
}

/// Full calibration of one focal-length entry from a pair set.
pub fn calibrate_entry(pairs: &[Pair<'_>], focal_mm: f32) -> FocalEntry {
    let vignette = vignette_gain(pairs);
    let mut vignette_lut = vec![1.0; VIGNETTE_LUT_LEN];
    for (index, value) in vignette.into_iter().take(VIGNETTE_LUT_LEN).enumerate() {
        vignette_lut[index] = value;
    }
    let mut k1 = 0.0;
    let mut k2 = 0.0;
    let mut scale_r = 1.0;
    let mut scale_b = 1.0;
    if let Some(first) = pairs.first() {
        // Geometry uses one representative pair per focal length (median
        // behavior comes from the vignette stage); the tool reports fit
        // diagnostics so bad frames are visible.
        let fit = geometry_fit(first);
        k1 = fit.k1;
        k2 = fit.k2;
        scale_r = fit.scale_r;
        scale_b = fit.scale_b;
    }
    FocalEntry {
        focal_mm,
        vignette: vignette_lut,
        k1,
        k2,
        scale_r,
        scale_b,
    }
}

/// Residuals of a corrected raw against its camera JPEG, both scene-linear.
/// The center-anchored tone fit absorbs global rendering differences, so
/// what remains is spatial: vignette-shape error, geometric misalignment,
/// and channel-dependent falloff (lateral CA / per-channel vignette).
#[derive(Debug, Clone, Copy)]
pub struct VerifyStats {
    /// Median |T(luma_corrected) − luma_jpeg| per radius band.
    pub center_residual: f32,
    pub mid_residual: f32,
    pub periphery_residual: f32,
    /// Median signed residual at the periphery: > 0 means the corrected
    /// raw is still darker than the JPEG there (under-corrected).
    pub periphery_signed: f32,
    /// Per-channel spatial ratio (periphery / center of the channel's own
    /// corrected/jpeg ratio), 1.0 = neutral. R↔B divergence indicates
    /// uncorrected lateral CA or a per-channel vignette component.
    pub channel_r: f32,
    pub channel_b: f32,
    /// Median global displacement between JPEG and corrected raw
    /// (full-resolution pixels) — a center/offset convention mismatch.
    pub shift_px: f32,
}

fn radius_of(index: usize, width: u32, height: u32) -> f32 {
    let x = index as u32 % width;
    let y = index as u32 / width;
    let (cx, cy) = (width as f32 / 2.0, height as f32 / 2.0);
    let r_max = (cx * cx + cy * cy).sqrt();
    let dx = x as f32 + 0.5 - cx;
    let dy = y as f32 + 0.5 - cy;
    (dx * dx + dy * dy).sqrt() / r_max
}

pub fn verify_pair(
    corrected: &[[f32; 3]],
    jpeg: &[[f32; 3]],
    width: u32,
    height: u32,
) -> VerifyStats {
    let tone = ToneFit::fit(corrected, jpeg, width, height);
    let mut residuals: [(Vec<f32>, Vec<f32>); 3] = Default::default();
    let mut center_ratios: [Vec<f32>; 3] = Default::default();
    let mut periphery_ratios: [Vec<f32>; 3] = Default::default();
    for (index, (corrected_pixel, jpeg_pixel)) in corrected.iter().zip(jpeg).enumerate() {
        let r = radius_of(index, width, height);
        let l = luma(corrected_pixel);
        let j = luma(jpeg_pixel);
        if let Some(predicted) = tone.forward(l)
            && (0.02..0.95).contains(&j)
        {
            let residual = predicted - j;
            let band = if r < 0.25 {
                0
            } else if r < 0.6 {
                1
            } else {
                2
            };
            residuals[band].0.push(residual.abs());
            residuals[band].1.push(residual);
        }
        let periphery = r >= 0.6;
        for channel in 0..3 {
            let numerator = corrected_pixel[channel];
            let denominator = jpeg_pixel[channel];
            if numerator > 1e-4 && denominator > 1e-4 && numerator < 8.0 {
                let ratio = numerator / denominator;
                if (0.05..20.0).contains(&ratio) {
                    if periphery {
                        periphery_ratios[channel].push(ratio);
                    } else if r < 0.25 {
                        center_ratios[channel].push(ratio);
                    }
                }
            }
        }
    }
    let channel_spatial = |channel: usize| {
        let center = median(&center_ratios[channel]).unwrap_or(1.0);
        let periphery = median(&periphery_ratios[channel]).unwrap_or(1.0);
        if center > 0.0 {
            periphery / center
        } else {
            1.0
        }
    };
    // Global alignment: coarse NCC displacement of JPEG vs corrected.
    let mut dst = gray(jpeg, width, height);
    let mut src = gray(corrected, width, height);
    let mut scale = 1.0_f32;
    while dst.width > 640 && dst.width >= 8 && dst.height >= 8 {
        dst = downsample(&dst);
        src = downsample(&src);
        scale *= 2.0;
    }
    let shifts = ncc_displacements(&dst, &src, 16, 5, 0.3);
    let shift_px = if shifts.is_empty() {
        0.0
    } else {
        let magnitudes: Vec<f32> = shifts
            .iter()
            .map(|&(_, _, dx, dy)| (dx * dx + dy * dy).sqrt())
            .collect();
        median(&magnitudes).unwrap_or(0.0) * scale
    };
    VerifyStats {
        center_residual: median(&residuals[0].0).unwrap_or(0.0),
        mid_residual: median(&residuals[1].0).unwrap_or(0.0),
        periphery_residual: median(&residuals[2].0).unwrap_or(0.0),
        periphery_signed: median(&residuals[2].1).unwrap_or(0.0),
        channel_r: channel_spatial(0),
        channel_b: channel_spatial(2),
        shift_px,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(unused_imports)] // test builds only; resolves the #[test] macro
    use test_r::test;

    /// Deterministic aperiodic textured scene: per-block hashed value
    /// noise. Sinusoids would alias under NCC (periodic ambiguity).
    fn scene(width: u32, height: u32) -> Vec<[f32; 3]> {
        let hash = |bx: u32, by: u32| {
            let mut h = bx.wrapping_mul(73856093) ^ by.wrapping_mul(19349663);
            h = h.wrapping_mul(0x9E3779B1);
            ((h >> 8) & 0xFFFF) as f32 / 65535.0
        };
        let mut pixels = Vec::with_capacity((width * height) as usize);
        for y in 0..height {
            for x in 0..width {
                let value = 0.15 + 0.7 * hash(x / 3, y / 3);
                pixels.push([value * 1.05, value, value * 0.95]);
            }
        }
        pixels
    }

    #[test]
    fn vignette_calibration_recovers_a_known_gain() {
        let (width, height) = (512_u32, 384_u32);
        let clean = scene(width, height);
        // Apply a known gain curve to produce the "raw" side.
        let mut raw = clean.clone();
        let mut lut = [1.0; VIGNETTE_LUT_LEN];
        for (index, value) in lut.iter_mut().enumerate() {
            *value = 1.0 / (1.0 + 0.9 * (index as f32 / (VIGNETTE_LUT_LEN - 1) as f32).powi(2));
        }
        super::super::apply_vignette(&mut raw, width, height, &lut);
        let pairs = vec![Pair {
            raw: &raw,
            jpeg: &clean,
            width,
            height,
        }];
        let recovered = vignette_gain(&pairs);
        // The recovered transmittance must be the inverse gain: ~1 at the
        // center, ≈ lut⁻¹ at the corner, within a few percent.
        assert!((recovered[0] - 1.0).abs() < 0.02, "{:?}", recovered[0]);
        let corner_error = (recovered[VIGNETTE_LUT_LEN - 1] - 1.0 / lut[32]).abs();
        assert!(
            corner_error < 0.06,
            "corner {} vs {}",
            recovered[32],
            1.0 / lut[32]
        );
    }

    #[test]
    fn geometry_calibration_recovers_known_distortion() {
        let (width, height) = (512_u32, 384_u32);
        let clean = scene(width, height);
        // Warp the clean scene with a known backward map to produce the raw.
        let entry = FocalEntry {
            focal_mm: 8.2,
            vignette: vec![1.0; VIGNETTE_LUT_LEN],
            k1: 0.18,
            k2: 0.04,
            scale_r: 1.0,
            scale_b: 1.0,
        };
        let mut raw = clean.clone();
        super::super::corrupt_forward(&mut raw, width, height, &entry);
        let pairs = Pair {
            raw: &raw,
            jpeg: &clean,
            width,
            height,
        };
        let fit = geometry_fit(&pairs);
        // The raw content was pulled outward by the backward map; fitting
        // the JPEG→raw displacement must recover the same coefficients.
        assert!(
            (fit.k1 - entry.k1).abs() < 0.05 && (fit.k2 - entry.k2).abs() < 0.06,
            "fit {fit:?} vs {:?}",
            (entry.k1, entry.k2)
        );
        assert!(fit.tangential < 0.02, "tangential {}", fit.tangential);
    }
}
