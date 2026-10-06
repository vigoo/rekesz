//! Shared spatial filter primitives for develop operators.
//!
//! All effect radii are referenced to a 1080-pixel short edge so thumbnails,
//! previews, and full renders receive the same visual strength. Every pass is
//! cancellable, allocates its own working buffers, and costs O(1) per pixel
//! regardless of radius.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};

use image::imageops::FilterType;
use image::{ImageBuffer, Luma};

/// Short-edge fraction relative to the 1080-pixel reference size.
pub fn size_scale(width: usize, height: usize) -> f32 {
    width.min(height) as f32 / 1080.0
}

/// Integer half-window radius scaled from a base value referenced to a
/// 1080-pixel short edge.
pub fn scaled_radius(base: f32, width: usize, height: usize) -> usize {
    ((base * size_scale(width, height)).round() as usize).max(1)
}

fn check_cancelled(cancelled: &AtomicBool) -> Result<(), String> {
    if cancelled.load(Ordering::Acquire) {
        Err("filter pass cancelled".to_owned())
    } else {
        Ok(())
    }
}

/// Separable box blur. Windows near borders cover only in-range samples
/// (shrink-window), matching the rest of the pipeline. `radius` is a
/// half-width in pixels; the full window is `2 * radius + 1` pixels across.
pub fn box_blur(
    plane: &[f32],
    width: usize,
    height: usize,
    radius: usize,
    cancelled: &AtomicBool,
) -> Result<Vec<f32>, String> {
    let mut horizontal = vec![0.0; plane.len()];
    box_blur_horizontal(plane, &mut horizontal, width, height, radius, cancelled)?;
    let mut output = vec![0.0; plane.len()];
    box_blur_vertical(&horizontal, &mut output, width, height, radius, cancelled)?;
    Ok(output)
}

fn box_blur_horizontal(
    plane: &[f32],
    output: &mut [f32],
    width: usize,
    height: usize,
    radius: usize,
    cancelled: &AtomicBool,
) -> Result<(), String> {
    let last = width - 1;
    for y in 0..height {
        if y % 16 == 0 {
            check_cancelled(cancelled)?;
        }
        let row = &plane[y * width..(y + 1) * width];
        let out = &mut output[y * width..(y + 1) * width];
        let mut sum = row[..=radius.min(last)].iter().sum::<f32>();
        out[0] = sum / (radius.min(last) + 1) as f32;
        for x in 1..width {
            if x + radius <= last {
                sum += row[x + radius];
            }
            if x > radius {
                sum -= row[x - radius - 1];
            }
            let count = (x + radius).min(last) - x.saturating_sub(radius) + 1;
            out[x] = sum / count as f32;
        }
    }
    Ok(())
}

fn box_blur_vertical(
    plane: &[f32],
    output: &mut [f32],
    width: usize,
    height: usize,
    radius: usize,
    cancelled: &AtomicBool,
) -> Result<(), String> {
    let last = height - 1;
    let mut sums = vec![0.0_f32; width];
    for y in 0..=radius.min(last) {
        let row = &plane[y * width..(y + 1) * width];
        for (sum, value) in sums.iter_mut().zip(row) {
            *sum += value;
        }
    }
    let first_count = radius.min(last) + 1;
    for (output_value, sum) in output[..width].iter_mut().zip(&sums) {
        *output_value = sum / first_count as f32;
    }
    for y in 1..height {
        if y % 16 == 0 {
            check_cancelled(cancelled)?;
        }
        let out_row = &mut output[y * width..(y + 1) * width];
        let remove_index = y.checked_sub(radius + 1).map(|row| row * width);
        let count = (y + radius).min(last) - y.saturating_sub(radius) + 1;
        let add_row = if y + radius <= last {
            Some((y + radius) * width)
        } else {
            None
        };
        for x in 0..width {
            if let Some(add_row) = add_row {
                sums[x] += plane[add_row + x];
            }
            if let Some(remove_row) = remove_index {
                sums[x] -= plane[remove_row + x];
            }
            out_row[x] = sums[x] / count as f32;
        }
    }
    Ok(())
}

/// Separable sliding-window minimum. Windows near borders cover only
/// in-range samples, which is equivalent to edge replication for minima.
pub fn box_min(
    plane: &[f32],
    width: usize,
    height: usize,
    radius: usize,
    cancelled: &AtomicBool,
) -> Result<Vec<f32>, String> {
    box_extreme(plane, width, height, radius, false, cancelled)
}

fn dominates(candidate: f32, incumbent: f32, maximum: bool) -> bool {
    if maximum {
        candidate >= incumbent
    } else {
        candidate <= incumbent
    }
}

fn box_extreme(
    plane: &[f32],
    width: usize,
    height: usize,
    radius: usize,
    maximum: bool,
    cancelled: &AtomicBool,
) -> Result<Vec<f32>, String> {
    let mut scratch = vec![0.0; plane.len()];
    extreme_horizontal(
        plane,
        &mut scratch,
        width,
        height,
        radius,
        maximum,
        cancelled,
    )?;
    let mut output = vec![0.0; plane.len()];
    extreme_vertical(
        &scratch,
        &mut output,
        width,
        height,
        radius,
        maximum,
        cancelled,
    )?;
    Ok(output)
}

/// One-dimensional sliding-window extreme over windows `[i - radius, i +
/// radius]`. The deque is fed the sequence padded with `radius` replicated
/// tail samples and emits index `i` at feed time `i + radius`, so each output
/// sees the full centered window; windows near borders cover only in-range
/// samples, which is equivalent to edge replication for minima and maxima.
fn extreme_horizontal(
    plane: &[f32],
    output: &mut [f32],
    width: usize,
    height: usize,
    radius: usize,
    maximum: bool,
    cancelled: &AtomicBool,
) -> Result<(), String> {
    let mut window: VecDeque<(usize, f32)> = VecDeque::with_capacity(4 * radius + 2);
    for y in 0..height {
        if y % 16 == 0 {
            check_cancelled(cancelled)?;
        }
        let row = &plane[y * width..(y + 1) * width];
        let out = &mut output[y * width..(y + 1) * width];
        window.clear();
        for x in 0..width + radius {
            let value = row[x.min(width - 1)];
            while window
                .back()
                .is_some_and(|&(_, incumbent)| dominates(value, incumbent, maximum))
            {
                window.pop_back();
            }
            window.push_back((x, value));
            while window
                .front()
                .is_some_and(|&(index, _)| index + 2 * radius < x)
            {
                window.pop_front();
            }
            if x >= radius {
                out[x - radius] = window.front().expect("monotonic window never empties").1;
            }
        }
    }
    Ok(())
}

fn extreme_vertical(
    plane: &[f32],
    output: &mut [f32],
    width: usize,
    height: usize,
    radius: usize,
    maximum: bool,
    cancelled: &AtomicBool,
) -> Result<(), String> {
    let mut windows: Vec<VecDeque<(usize, f32)>> = (0..width)
        .map(|_| VecDeque::with_capacity(4 * radius + 2))
        .collect();
    for y in 0..height + radius {
        if y % 16 == 0 {
            check_cancelled(cancelled)?;
        }
        let row = &plane[y.min(height - 1) * width..][..width];
        let write = y.checked_sub(radius).map(|row_index| row_index * width);
        for (x, &value) in row.iter().enumerate() {
            let window = &mut windows[x];
            while window
                .back()
                .is_some_and(|&(_, incumbent)| dominates(value, incumbent, maximum))
            {
                window.pop_back();
            }
            window.push_back((y, value));
            while window
                .front()
                .is_some_and(|&(index, _)| index + 2 * radius < y)
            {
                window.pop_front();
            }
            if let Some(out_row) = write {
                output[out_row + x] = window.front().expect("monotonic window never empties").1;
            }
        }
    }
    Ok(())
}

/// Edge-aware self-guided smoothing of a single plane: fits `q = a*I + b` in
/// each window and returns the filtered result. Flat regions collapse toward
/// the local mean while strong edges keep `a` near one and survive
/// unattenuated. Large planes are processed at a working resolution near a
/// 1080-pixel short edge with the fit coefficients upsampled, keeping cost
/// independent of source size.
///
/// The plane is interpreted as a guide in the caller's chosen domain (the
/// develop operators use log2 luminance); `eps` regularizes in that domain.
pub fn guided_smooth(
    plane: &[f32],
    width: usize,
    height: usize,
    radius_base: f32,
    eps: f32,
    cancelled: &AtomicBool,
) -> Result<Vec<f32>, String> {
    let short = width.min(height);
    let work_short = short.min(1080);
    let scale = work_short as f32 / short as f32;
    let work_width = (((width as f32) * scale).round() as usize).clamp(1, width);
    let work_height = (((height as f32) * scale).round() as usize).clamp(1, height);
    let radius = scaled_radius(radius_base, work_width, work_height);
    let (a_full, b_full) = if (work_width, work_height) == (width, height) {
        guided_coefficients(plane, width, height, radius, eps, cancelled)?
    } else {
        let work_plane = resize_plane(plane, width, height, work_width, work_height);
        let (a, b) =
            guided_coefficients(&work_plane, work_width, work_height, radius, eps, cancelled)?;
        (
            resize_plane(&a, work_width, work_height, width, height),
            resize_plane(&b, work_width, work_height, width, height),
        )
    };
    let mut output = vec![0.0; plane.len()];
    for (index, output_value) in output.iter_mut().enumerate() {
        if index % 16_384 == 0 {
            check_cancelled(cancelled)?;
        }
        *output_value = a_full[index] * plane[index] + b_full[index];
    }
    Ok(output)
}

/// Standard self-guided filter coefficient pass: `a` and `b` of the local
/// linear fit `q = a*I + b`, smoothed over the same window.
fn guided_coefficients(
    plane: &[f32],
    width: usize,
    height: usize,
    radius: usize,
    eps: f32,
    cancelled: &AtomicBool,
) -> Result<(Vec<f32>, Vec<f32>), String> {
    let mean = box_blur(plane, width, height, radius, cancelled)?;
    let squared: Vec<f32> = plane.iter().map(|value| value * value).collect();
    let mean_squared = box_blur(&squared, width, height, radius, cancelled)?;
    drop(squared);
    let variance: Vec<f32> = mean_squared
        .iter()
        .zip(&mean)
        .map(|(&mean_of_squares, &mean_value)| (mean_of_squares - mean_value * mean_value).max(0.0))
        .collect();
    drop(mean_squared);
    let a: Vec<f32> = variance
        .iter()
        .map(|&value| value / (value + eps))
        .collect();
    drop(variance);
    let b: Vec<f32> = a
        .iter()
        .zip(&mean)
        .map(|(&a_value, &mean_value)| mean_value * (1.0 - a_value))
        .collect();
    drop(mean);
    let mean_a = box_blur(&a, width, height, radius, cancelled)?;
    let mean_b = box_blur(&b, width, height, radius, cancelled)?;
    drop(a);
    drop(b);
    Ok((mean_a, mean_b))
}

fn resize_plane(
    plane: &[f32],
    width: usize,
    height: usize,
    new_width: usize,
    new_height: usize,
) -> Vec<f32> {
    debug_assert_eq!(plane.len(), width * height);
    let Some(source) =
        ImageBuffer::<Luma<f32>, Vec<f32>>::from_raw(width as u32, height as u32, plane.to_vec())
    else {
        return plane.to_vec();
    };
    let resized = image::imageops::resize(
        &source,
        new_width as u32,
        new_height as u32,
        FilterType::Triangle,
    );
    resized.into_raw()
}

#[cfg(test)]
mod tests;
