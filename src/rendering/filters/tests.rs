use std::sync::atomic::AtomicBool;

use super::{box_blur, box_min, guided_smooth, scaled_radius, size_scale};
use test_r::test;

fn not_cancelled() -> AtomicBool {
    AtomicBool::new(false)
}

/// Deterministic xorshift pseudo-random plane for brute-force comparisons.
fn pseudo_random_plane(width: usize, height: usize) -> Vec<f32> {
    let mut seed = 0x9E37_79B9_7F4A_7C15_u64;
    (0..width * height)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % 1000) as f32 / 100.0
        })
        .collect()
}

fn naive_box_blur(plane: &[f32], width: usize, height: usize, radius: usize) -> Vec<f32> {
    plane
        .iter()
        .enumerate()
        .map(|(index, _)| {
            let x = index % width;
            let y = index / width;
            let mut sum = 0.0;
            let mut count = 0.0_f32;
            for sy in y.saturating_sub(radius)..=(y + radius).min(height - 1) {
                for sx in x.saturating_sub(radius)..=(x + radius).min(width - 1) {
                    sum += plane[sy * width + sx];
                    count += 1.0;
                }
            }
            sum / count
        })
        .collect()
}

fn naive_box_min(plane: &[f32], width: usize, height: usize, radius: usize) -> Vec<f32> {
    plane
        .iter()
        .enumerate()
        .map(|(index, _)| {
            let x = index % width;
            let y = index / width;
            let mut minimum = f32::MAX;
            for sy in y.saturating_sub(radius)..=(y + radius).min(height - 1) {
                for sx in x.saturating_sub(radius)..=(x + radius).min(width - 1) {
                    minimum = minimum.min(plane[sy * width + sx]);
                }
            }
            minimum
        })
        .collect()
}

#[test]
fn box_blur_of_constant_plane_is_constant() {
    let cancelled = not_cancelled();
    let width = 33;
    let height = 17;
    let plane = vec![3.5; width * height];
    let blurred = box_blur(&plane, width, height, 6, &cancelled).expect("blur succeeds");
    for value in &blurred {
        assert!((value - 3.5).abs() < 1e-3, "{value}");
    }
}

#[test]
fn box_blur_of_ramp_is_identity_in_the_interior() {
    let cancelled = not_cancelled();
    let width = 81;
    let height = 5;
    let plane: Vec<f32> = (0..width * height)
        .map(|index| (index % width) as f32 * 0.5)
        .collect();
    let blurred = box_blur(&plane, width, height, 7, &cancelled).expect("blur succeeds");
    for y in 0..height {
        for x in 7..width - 7 {
            let index = y * width + x;
            assert!(
                (blurred[index] - plane[index]).abs() < 1e-3,
                "x={x} blurred={} original={}",
                blurred[index],
                plane[index]
            );
        }
    }
}

#[test]
fn box_blur_of_impulse_spreads_over_the_window() {
    let cancelled = not_cancelled();
    let width = 5;
    let height = 1;
    let plane = vec![0.0, 0.0, 1.0, 0.0, 0.0];
    let blurred = box_blur(&plane, width, height, 1, &cancelled).expect("blur succeeds");
    let expected = [0.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0];
    for (value, &expectation) in blurred.iter().zip(&expected) {
        assert!(
            (value - expectation).abs() < 1e-6,
            "{value} vs {expectation}"
        );
    }
}

#[test]
fn box_blur_matches_naive_reference() {
    let cancelled = not_cancelled();
    let width = 24;
    let height = 19;
    let plane = pseudo_random_plane(width, height);
    for radius in [1, 3, 8] {
        let blurred = box_blur(&plane, width, height, radius, &cancelled).expect("blur succeeds");
        let naive = naive_box_blur(&plane, width, height, radius);
        for (value, &expectation) in blurred.iter().zip(&naive) {
            assert!((value - expectation).abs() < 1e-3, "radius {radius}");
        }
    }
}

#[test]
fn box_min_matches_naive_reference() {
    let cancelled = not_cancelled();
    let width = 21;
    let height = 13;
    let plane = pseudo_random_plane(width, height);
    for radius in [1, 2, 5] {
        let minimum = box_min(&plane, width, height, radius, &cancelled).expect("min succeeds");
        let naive_min = naive_box_min(&plane, width, height, radius);
        for (value, &expectation) in minimum.iter().zip(&naive_min) {
            assert!((value - expectation).abs() < 1e-6, "radius {radius}");
        }
    }
}

#[test]
fn box_min_exact_on_known_row() {
    let cancelled = not_cancelled();
    let width = 5;
    let height = 1;
    let plane = vec![1.0, 2.0, 3.0, 2.0, 1.0];
    let minimum = box_min(&plane, width, height, 1, &cancelled).expect("min succeeds");
    assert_eq!(minimum, vec![1.0, 1.0, 2.0, 1.0, 1.0]);
}

#[test]
fn guided_smooth_of_constant_plane_is_constant() {
    let cancelled = not_cancelled();
    let width = 40;
    let height = 30;
    let plane = vec![-2.0; width * height];
    let smoothed =
        guided_smooth(&plane, width, height, 8.0, 0.25, &cancelled).expect("guided succeeds");
    for value in &smoothed {
        assert!((value + 2.0).abs() < 1e-3, "{value}");
    }
}

#[test]
fn guided_smooth_preserves_strong_edges_much_better_than_box_blur() {
    let cancelled = not_cancelled();
    let width = 64;
    let height = 32;
    let mut plane = vec![0.0_f32; width * height];
    for y in 0..height {
        for x in 0..width {
            if x >= width / 2 {
                plane[y * width + x] = 5.0;
            }
        }
    }
    let guided =
        guided_smooth(&plane, width, height, 8.0, 0.25, &cancelled).expect("guided succeeds");
    let blurred = box_blur(&plane, width, height, 8, &cancelled).expect("blur succeeds");
    let mut guided_edge_detail = 0.0_f32;
    let mut blurred_edge_detail = 0.0_f32;
    for y in 0..height {
        for x in [width / 2 - 1, width / 2] {
            guided_edge_detail =
                guided_edge_detail.max((plane[y * width + x] - guided[y * width + x]).abs());
            blurred_edge_detail =
                blurred_edge_detail.max((plane[y * width + x] - blurred[y * width + x]).abs());
        }
    }
    assert!(
        guided_edge_detail < 0.5,
        "guided edge detail {guided_edge_detail}"
    );
    assert!(
        blurred_edge_detail > 1.0,
        "box blur edge detail {blurred_edge_detail}"
    );
}

#[test]
fn scaled_radius_tracks_short_edge_relative_to_reference() {
    assert_eq!(scaled_radius(8.0, 1080, 1080), 8);
    assert_eq!(scaled_radius(8.0, 2160, 3840), 16);
    assert_eq!(scaled_radius(8.0, 270, 480), 2);
    assert_eq!(scaled_radius(2.5, 100, 100), 1);
}

#[test]
fn size_scale_uses_short_edge() {
    assert!((size_scale(1920, 1080) - 1.0).abs() < 1e-6);
    assert!((size_scale(3840, 2160) - 2.0).abs() < 1e-6);
}

#[test]
fn every_pass_reports_cancellation() {
    let cancelled = AtomicBool::new(true);
    let plane = vec![1.0; 16 * 16];
    assert!(box_blur(&plane, 16, 16, 2, &cancelled).is_err());
    assert!(box_min(&plane, 16, 16, 2, &cancelled).is_err());
    assert!(guided_smooth(&plane, 16, 16, 8.0, 0.25, &cancelled).is_err());
}

#[test]
fn guided_smooth_working_resolution_path_stays_exact_on_constant_planes() {
    let cancelled = not_cancelled();
    // Short edge above the 1080 working reference forces the downsample /
    // upsample coefficient path.
    let width = 64;
    let height = 1200;
    let plane = vec![2.5; width * height];
    let smoothed =
        guided_smooth(&plane, width, height, 8.0, 0.25, &cancelled).expect("guided succeeds");
    for value in &smoothed {
        assert!((value - 2.5).abs() < 1e-3, "{value}");
    }
}

#[test]
fn box_blur_handles_windows_wider_than_the_plane() {
    let cancelled = not_cancelled();
    let width = 6;
    let height = 1;
    let plane = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let blurred = box_blur(&plane, width, height, 9, &cancelled).expect("blur succeeds");
    let naive = naive_box_blur(&plane, width, height, 9);
    for (value, &expectation) in blurred.iter().zip(&naive) {
        assert!(
            (value - expectation).abs() < 1e-4,
            "{value} vs {expectation}"
        );
    }
}

#[test]
fn box_min_handles_windows_wider_than_the_plane() {
    let cancelled = not_cancelled();
    let width = 6;
    let height = 1;
    let plane = vec![3.0, 1.0, 4.0, 1.0, 5.0, 2.0];
    let minimum = box_min(&plane, width, height, 9, &cancelled).expect("min succeeds");
    let naive = naive_box_min(&plane, width, height, 9);
    for (value, &expectation) in minimum.iter().zip(&naive) {
        assert!(
            (value - expectation).abs() < 1e-6,
            "{value} vs {expectation}"
        );
    }
}

#[test]
fn filters_handle_single_column_and_single_row_planes() {
    let cancelled = not_cancelled();
    let plane: Vec<f32> = (0..12).map(|index| index as f32).collect();
    let column = box_blur(&plane, 1, 12, 2, &cancelled).expect("column blur succeeds");
    let naive = naive_box_blur(&plane, 1, 12, 2);
    for (value, &expectation) in column.iter().zip(&naive) {
        assert!((value - expectation).abs() < 1e-4);
    }
    let minimum = box_min(&plane, 1, 12, 2, &cancelled).expect("column min succeeds");
    let naive_min = naive_box_min(&plane, 1, 12, 2);
    for (value, &expectation) in minimum.iter().zip(&naive_min) {
        assert!((value - expectation).abs() < 1e-6);
    }
    let single = box_blur(&[7.0], 1, 1, 3, &cancelled).expect("single pixel blur succeeds");
    assert!((single[0] - 7.0).abs() < 1e-6);
}
