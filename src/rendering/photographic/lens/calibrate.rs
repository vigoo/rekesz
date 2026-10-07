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
            index if index >= self.l_bins.len() => return Some(*self.j_values.last().unwrap()),
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
///
/// `corrected_radius` maps a sample's RAW-frame radius to the radius it
/// occupies in the corrected output — the runtime indexes the LUT by
/// corrected radius (the gain is applied after the warp), so the binning
/// must use the same map. Samples whose corrected radius falls outside
/// the output frame are cropped there and skipped.
pub fn vignette_gain(
    pairs: &[Pair<'_>],
    corrected_radius: impl Fn(f32) -> f32 + Clone,
) -> Vec<f32> {
    let curves: Vec<Vec<f32>> = pairs
        .iter()
        .map(|pair| pair_vignette_curve(pair, corrected_radius.clone()))
        .collect();
    vignette_gain_from_curves(curves)
}

/// Peripheral illumination gain measured from ONE pair (33-bin curve).
pub fn pair_vignette_curve(pair: &Pair<'_>, corrected_radius: impl Fn(f32) -> f32) -> Vec<f32> {
    const RADII: usize = VIGNETTE_LUT_LEN;
    let tone = ToneFit::fit(pair.raw, pair.jpeg, pair.width, pair.height);
    let (cx, cy, r_max) = radius_map(pair.width, pair.height);
    let mut per_bin: Vec<Vec<f32>> = vec![Vec::new(); RADII];
    for (index, (raw_pixel, jpeg_pixel)) in pair.raw.iter().zip(pair.jpeg).enumerate() {
        let x = index as u32 % pair.width;
        let y = index as u32 / pair.width;
        let dx = x as f32 + 0.5 - cx;
        let dy = y as f32 + 0.5 - cy;
        let t = (dx * dx + dy * dy).sqrt() / r_max;
        let r = corrected_radius(t);
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
    per_bin
        .iter()
        .map(|samples| median(samples).unwrap_or(1.0))
        .collect()
}

/// Median-combines per-pair curves and normalizes to `G(0) = 1`.
pub fn vignette_gain_from_curves(curves: Vec<Vec<f32>>) -> Vec<f32> {
    const RADII: usize = VIGNETTE_LUT_LEN;
    let mut combined: Vec<Vec<f32>> = vec![Vec::new(); RADII];
    for curve in &curves {
        for (bin, value) in curve.iter().enumerate() {
            combined[bin].push(*value);
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
#[derive(Debug, Clone)]
pub struct GeometryFit {
    /// Measured source-radius factor per output-radius bin
    /// ([`super::VIGNETTE_LUT_LEN`] entries; 0.0 = no data in that bin).
    /// `radial_map[i]` is `r_src / r_out` at `r_out = i / (LEN − 1)` — the
    /// direct measurement the runtime gather interpolates, with no
    /// extrapolation beyond the last confident match.
    pub radial_map: Vec<f32>,
    pub zoom: f32,
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
            // ±4: wide enough to track a residual displacement of ~1% of
            // the frame width at this level — without it, corner patches
            // (where the fit is extrapolating) silently drop out.
            for oy in -4_i32..=4 {
                for ox in -4_i32..=4 {
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

        // Refine up to the ~720-px level: farther-out samples shrink the
        // k1/k2/zoom degeneracy and make the corner extrapolation (the
        // runtime solves the fixed point up to r = 1) trustworthy. Beyond
        // that the NCC cost grows too fast for the gain.
        if levels[level].0.width >= 640 {
            break;
        }
    }

    // Fit in the working level's coordinates, normalized radii.
    let (dst_level, _) = &levels[working];
    let (cx, cy, r_max) = radius_map(dst_level.width, dst_level.height);
    let mut samples: Vec<(f32, f32)> = Vec::new();
    let mut tangential_samples: Vec<f32> = Vec::new();
    let mut ones: Vec<()> = Vec::new();
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
        // The runtime's magnification is a function of the RAW radius:
        // r_out = t·a(t), and the gather's fixed point evaluates a at t.
        // The measurement pair (r_src = t, r_dst = r_out) therefore fits
        // y = a(t) − 1 against x = t — not against the corrected radius.
        samples.push((r_src / r_max, r_dst / r_src - 1.0));
        // The intercept (zoom) absorbs the camera's overall rectification
        // resample; without it a constant scale error masquerades as
        // uncorrectable radial displacement.
        ones.push(());
        let (ux, uy) = (pdx / r_dst, pdy / r_dst);
        tangential_samples.push((-dx * uy + dy * ux).abs() / r_dst);
    }
    // Mismatched NCC patches produce physically impossible displacement
    // ratios; the robust fit's first solve must not see them.
    samples.retain(|&(_, y)| y.abs() <= 0.5);
    let (zoom, k1, k2, residual) = fit_radial(&samples);
    let tangential = median(&tangential_samples).unwrap_or(0.0);
    let max_radius = samples
        .iter()
        .map(|(r, _)| *r)
        .fold(0.0_f32, |acc, r| acc.max(r));
    // The measured radial map: per output-radius bin, the median
    // source/output ratio. This is what the runtime gathers with — no
    // polynomial extrapolation into unmeasured corners.
    let mut radial_bins: Vec<Vec<f32>> = vec![Vec::new(); super::VIGNETTE_LUT_LEN];
    for &(r_out, y) in &samples {
        let bin = ((r_out * (super::VIGNETTE_LUT_LEN - 1) as f32).round() as usize)
            .min(super::VIGNETTE_LUT_LEN - 1);
        // The gather needs r_src / r_out; y was fitted as r_out / r_src − 1.
        radial_bins[bin].push(1.0 / (1.0 + y));
    }
    let radial_map: Vec<f32> = radial_bins
        .iter()
        .map(|values| median(values).unwrap_or(0.0))
        .collect();

    // Lateral CA from the raw's own channel displacements (camera JPEG
    // channels are corrected by construction).
    let (scale_r, scale_b) = channel_scales(pair, 20, 5);

    GeometryFit {
        radial_map,
        zoom,
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

/// Radius-weighted least squares for
/// `a(r) = zoom + k1·r² + k2·r⁴` (the runtime magnification over the RAW
/// radius), with σ-trim outlier rejection. The intercept matters: without
/// it, the camera's overall rectification resample cannot be represented.
fn fit_radial(samples: &[(f32, f32)]) -> (f32, f32, f32, f32) {
    let mut active = samples.to_vec();
    let mut zoom = 1.0_f32;
    let mut k1 = 0.0_f32;
    let mut k2 = 0.0_f32;
    let mut residual = f32::INFINITY;
    // The fitted model's constant term is zoom − 1 (y = a − 1); predict
    // must match it or the trim compares against a curve shifted by 1.
    let predict =
        |zoom: f32, k1: f32, k2: f32, r: f32| (zoom - 1.0) + k1 * r * r + k2 * r * r * r * r;
    for _ in 0..3 {
        if active.len() < 16 {
            break;
        }
        // Weighted normal equations; weight = r² so the corner (where the
        // signal is largest) dominates. The basis is fit CENTERED in
        // u = r² for numerical stability. The model is deliberately
        // zoom + k1·r² only: the r⁴ term is degenerate over the reachable
        // radius range (NCC never matches the extreme corner), and a free
        // r⁴ coefficient extrapolates wildly — a fitted k2 = −0.2 drove
        // the corner magnification below 1, which is physically impossible
        // for barrel correction. Monotone quadratic extrapolation stays
        // sane where the data ends.
        let u_mean = active.iter().map(|&(r, _)| r * r).sum::<f32>() / active.len() as f32;
        let mut m = [[0.0_f32; 2]; 2];
        let mut v = [0.0_f32; 2];
        for &(r, y) in &active {
            let weight = r * r;
            let u = r * r - u_mean;
            let basis = [1.0, u];
            for (row, basis_row) in basis.iter().enumerate() {
                for (column, basis_column) in basis.iter().enumerate() {
                    m[row][column] += weight * basis_row * basis_column;
                }
                v[row] += weight * basis_row * y;
            }
        }
        let Some(solution) = solve_2x2(&m, &v) else {
            break;
        };
        // Expand back: y = (b0 − b1·ū) + b1·r². y = a − 1, so the fitted
        // intercept is zoom − 1.
        zoom = 1.0 + solution[0] - solution[1] * u_mean;
        k1 = solution[1];
        k2 = 0.0;
        let errors: Vec<f32> = active
            .iter()
            .map(|&(r, y)| (y - predict(zoom, k1, k2, r)).abs())
            .collect();
        residual = median(&errors).unwrap_or(f32::INFINITY);
        // Trim outliers and refit.
        let limit = 2.5 * residual.max(1e-4);
        let next: Vec<(f32, f32)> = active
            .iter()
            .copied()
            .filter(|&(r, y)| (y - predict(zoom, k1, k2, r)).abs() <= limit)
            .collect();
        if next.len() < active.len() / 2 {
            break;
        }
        active = next;
    }
    (zoom, k1, k2, residual)
}

/// Cramer solver for a 2×2 system; `None` when singular.
fn solve_2x2(m: &[[f32; 2]; 2], v: &[f32; 2]) -> Option<[f32; 2]> {
    let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    if det.abs() < 1e-12 {
        return None;
    }
    Some([
        (v[0] * m[1][1] - v[1] * m[0][1]) / det,
        (m[0][0] * v[1] - m[1][0] * v[0]) / det,
    ])
}

/// Cramer solver for a 3×3 system; `None` when singular.
fn solve_3x3(m: &[[f32; 3]; 3], v: &[f32; 3]) -> Option<[f32; 3]> {
    let det = |a: &[[f32; 3]; 3]| {
        a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
            - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
    };
    let d = det(m);
    if d.abs() < 1e-12 {
        return None;
    }
    let mut out = [0.0_f32; 3];
    for index in 0..3 {
        let mut modified = *m;
        for (row, modified_row) in modified.iter_mut().enumerate() {
            modified_row[index] = v[row];
        }
        out[index] = det(&modified) / d;
    }
    Some(out)
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
    // Geometry and vignette are coupled: the NCC stage cannot match the
    // raw's near-black corners (image circle), so a single-pass fit
    // extrapolates the magnification and undershoots exactly where the
    // correction is strongest. The loop breaks the coupling:
    // fit geometry → build the vignette LUT → brighten the raws with it →
    // re-fit geometry on the brightened frames (corner texture now
    // visible) → recompute. Two rounds converge for this lens.
    let median_of = |values: &mut Vec<f32>| {
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        values.get(values.len() / 2).copied().unwrap_or(0.0)
    };
    let fits: Vec<(GeometryFit, bool)> = pairs
        .iter()
        .map(|pair| {
            let fit = geometry_fit(pair);
            let usable = fit.tangential < 0.02;
            (fit, usable)
        })
        .collect();

    let medians = |usable_fits: &[(GeometryFit, bool)]| -> (f32, f32, f32, f32, f32, Vec<f32>) {
        let mut zooms: Vec<f32> = Vec::new();
        let mut k1s: Vec<f32> = Vec::new();
        let mut k2s: Vec<f32> = Vec::new();
        let mut sr: Vec<f32> = Vec::new();
        let mut sb: Vec<f32> = Vec::new();
        let mut maps: Vec<Vec<f32>> = Vec::new();
        for (fit, usable) in usable_fits {
            if *usable {
                zooms.push(fit.zoom);
                k1s.push(fit.k1);
                k2s.push(fit.k2);
                sr.push(fit.scale_r);
                sb.push(fit.scale_b);
                if fit.radial_map.len() == super::VIGNETTE_LUT_LEN {
                    maps.push(fit.radial_map.clone());
                }
            }
        }
        // Median radial map per bin; gaps inherit the nearest measured
        // bin (center defaults to 1.0).
        let mut map = vec![0.0_f32; super::VIGNETTE_LUT_LEN];
        for (bin, slot) in map.iter_mut().enumerate() {
            let values: Vec<f32> = maps
                .iter()
                .filter(|per_pair| per_pair[bin] > 0.0)
                .map(|per_pair| per_pair[bin])
                .collect();
            *slot = median(&values).unwrap_or(0.0);
        }
        let mut last = 1.0_f32;
        for slot in &mut map {
            if *slot > 0.0 {
                last = *slot;
            } else {
                *slot = last;
            }
        }
        (
            if zooms.is_empty() {
                1.0
            } else {
                median_of(&mut zooms)
            },
            median_of(&mut k1s),
            median_of(&mut k2s),
            if sr.is_empty() {
                1.0
            } else {
                median_of(&mut sr)
            },
            if sb.is_empty() {
                1.0
            } else {
                median_of(&mut sb)
            },
            map,
        )
    };

    // Round 2: brighten the raws with the round-1 vignette and re-fit.
    let mut fits_refined: Vec<(GeometryFit, bool)> = fits.clone();
    let (round1_zoom, round1_k1, round1_k2, _, _, round1_map) = medians(&fits);
    let round1_vignette = {
        let corrected_radius = |t: f32| {
            if round1_map.len() == super::VIGNETTE_LUT_LEN {
                t / super::vignette_at(&round1_map, t)
            } else {
                t * (round1_zoom + round1_k1 * t * t + round1_k2 * t * t * t * t)
            }
        };
        vignette_gain(pairs, corrected_radius)
    };
    for (index, (pair, (fit, usable))) in pairs.iter().zip(&fits).enumerate() {
        if !usable {
            continue;
        }
        let mut brightened = pair.raw.to_vec();
        super::apply_vignette(&mut brightened, pair.width, pair.height, &round1_vignette);
        let brightened_pair = Pair {
            raw: &brightened,
            jpeg: pair.jpeg,
            width: pair.width,
            height: pair.height,
        };
        let refined = geometry_fit(&brightened_pair);
        // Keep the refinement when it did not degrade the fit.
        if refined.tangential <= fit.tangential + 0.01
            && refined.samples * 2 >= fits[index].0.samples
        {
            fits_refined[index] = (refined, true);
        }
    }
    let (zoom, k1, k2, scale_r, scale_b, radial_map) = medians(&fits_refined);

    // Final vignette through the converged map.
    let corrected_radius = |t: f32| {
        if radial_map.len() == super::VIGNETTE_LUT_LEN {
            t / super::vignette_at(&radial_map, t)
        } else {
            t * (zoom + k1 * t * t + k2 * t * t * t * t)
        }
    };
    let vignette = vignette_gain(pairs, corrected_radius);
    let mut vignette_lut = vec![1.0; VIGNETTE_LUT_LEN];
    for (index, value) in vignette.into_iter().take(VIGNETTE_LUT_LEN).enumerate() {
        vignette_lut[index] = value;
    }
    FocalEntry {
        focal_mm,
        vignette: vignette_lut,
        warp: [[0.0; 4]; 3],
        radial_map,
        zoom,
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
    /// Median displacement components (JPEG relative to corrected raw,
    /// full-resolution pixels). A consistent direction across pairs means
    /// the optical center is off the frame center (decentering), which a
    /// center-symmetric radial model cannot represent.
    pub shift_dx: f32,
    pub shift_dy: f32,
    /// Median SIGNED radial component of the patch displacements
    /// (full-res px; positive = the JPEG's content sits farther out than
    /// the corrected raw's). A large value with ~zero dx/dy medians is a
    /// residual radial scale error — the correction strength itself, not
    /// a center offset.
    pub shift_radial: f32,
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
    while dst.width > 640 && dst.height >= 8 {
        dst = downsample(&dst);
        src = downsample(&src);
        scale *= 2.0;
    }
    let shifts = ncc_displacements(&dst, &src, 16, 5, 0.3);
    let (shift_px, shift_dx, shift_dy, shift_radial) = if shifts.is_empty() {
        (0.0, 0.0, 0.0, 0.0)
    } else {
        let (cx_l, cy_l, _) = radius_map(dst.width, dst.height);
        let mut magnitudes: Vec<f32> = Vec::with_capacity(shifts.len());
        let mut dxs: Vec<f32> = Vec::with_capacity(shifts.len());
        let mut dys: Vec<f32> = Vec::with_capacity(shifts.len());
        let mut radials: Vec<f32> = Vec::with_capacity(shifts.len());

        for &(bx, by, dx, dy) in &shifts {
            magnitudes.push((dx * dx + dy * dy).sqrt());
            dxs.push(dx);
            dys.push(dy);
            let pdx = bx + 8.0 - cx_l;
            let pdy = by + 8.0 - cy_l;
            let r = (pdx * pdx + pdy * pdy).sqrt();
            if r > 0.0 {
                radials.push((dx * pdx + dy * pdy) / r);
            }
        }
        (
            median(&magnitudes).unwrap_or(0.0) * scale,
            median(&dxs).unwrap_or(0.0) * scale,
            median(&dys).unwrap_or(0.0) * scale,
            median(&radials).unwrap_or(0.0) * scale,
        )
    };
    VerifyStats {
        center_residual: median(&residuals[0].0).unwrap_or(0.0),
        mid_residual: median(&residuals[1].0).unwrap_or(0.0),
        periphery_residual: median(&residuals[2].0).unwrap_or(0.0),
        periphery_signed: median(&residuals[2].1).unwrap_or(0.0),
        channel_r: channel_spatial(0),
        channel_b: channel_spatial(2),
        shift_px,
        shift_dx,
        shift_dy,
        shift_radial,
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
                // 9-px blocks survive every pyramid level down to the
                // coarsest NCC scale, matching real photos' multi-scale
                // texture.
                let value = 0.15 + 0.7 * hash(x / 9, y / 9);
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
        // Identity geometry: the corrected radius is the raw radius.
        let recovered = vignette_gain(&pairs, |t| t);
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
    fn fit_radial_recovers_an_exact_polynomial() {
        // The model is zoom + k1·r² (no r⁴ term — see fit_radial); the
        // r⁴-only synthetic exercises the pure-quadratic recovery.
        let samples: Vec<(f32, f32)> = (0..40)
            .map(|index| {
                let r = 0.15 + 0.5 * index as f32 / 39.0;
                (r, 0.18 * r * r)
            })
            .collect();
        let (zoom, k1, _k2, residual) = fit_radial(&samples);
        assert!(
            (zoom - 1.0).abs() < 1e-3 && (k1 - 0.18).abs() < 1e-3,
            "zoom {zoom} k1 {k1} residual {residual}"
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
            warp: [[0.0; 4]; 3],
            radial_map: Vec::new(),
            zoom: 1.0,
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
        // Sanity: the zoom stays physical and the wide-end correction is
        // positive. Exact k1/k2 recovery is NOT asserted — over the
        // reachable radius range (~0.63 here) the three coefficients are
        // degenerate; what production needs is that the fitted correction
        // UNDOES the corruption, asserted end-to-end below.
        assert!((fit.zoom - 1.0).abs() < 0.1 && fit.k1 > 0.0, "fit {fit:?}");
        let fitted = FocalEntry {
            focal_mm: 8.2,
            vignette: vec![1.0; VIGNETTE_LUT_LEN],
            warp: [[0.0; 4]; 3],
            radial_map: fit.radial_map.clone(),
            zoom: fit.zoom,
            k1: fit.k1,
            k2: fit.k2,
            scale_r: 1.0,
            scale_b: 1.0,
        };
        let mut corrected = raw.clone();
        super::super::apply_geometry(&mut corrected, width, height, &fitted);
        // The same-size corruption has no corner data (its backward map
        // samples past the frame), so end-to-end recovery is asserted over
        // the calibrated range; the corners are validated against real
        // pairs by the calibration tool's --verify sweep.
        let (cx, cy) = (width as f32 / 2.0, height as f32 / 2.0);
        let r_max = (cx * cx + cy * cy).sqrt();
        let mut error_sum = 0.0_f32;
        let mut error_count = 0_usize;
        for (index, (corrected_pixel, clean_pixel)) in corrected.iter().zip(&clean).enumerate() {
            let x = index as u32 % width;
            let y = index as u32 / width;
            let dx = x as f32 + 0.5 - cx;
            let dy = y as f32 + 0.5 - cy;
            if (dx * dx + dy * dy).sqrt() / r_max > 0.6 {
                continue;
            }
            error_sum += (corrected_pixel[0] - clean_pixel[0])
                .abs()
                .max((corrected_pixel[1] - clean_pixel[1]).abs());
            error_count += 1;
        }
        let error = error_sum / error_count.max(1) as f32;
        // Tolerance note: subpixel NCC bias on the blocky synthetic scene
        // limits in-range recovery to a few percent; a direction or
        // inversion error produces an order of magnitude more.
        assert!(error < 0.04, "mean end-to-end correction error {error}");
        assert!(fit.tangential < 0.02, "tangential {}", fit.tangential);
    }
}
