use image::{DynamicImage, RgbImage};
use test_r::test;

use super::*;

fn pattern() -> DynamicImage {
    DynamicImage::ImageRgb8(RgbImage::from_fn(8, 4, |x, y| {
        image::Rgb([(x * 29) as u8, (y * 61) as u8, ((x + y) * 17) as u8])
    }))
}

#[test]
fn full_render_does_not_resize() {
    let image = DynamicImage::ImageRgb8(RgbImage::new(20, 10));
    assert_eq!(output(image, ResizePlan::None).width(), 20);
}

#[test]
fn derivative_fits_within_plan_bounds() {
    let image = DynamicImage::ImageRgb8(RgbImage::new(20, 10));
    let resized = output(
        image,
        ResizePlan::Derivative {
            max_width: 8,
            max_height: 8,
            before_develop: false,
        },
    );
    assert_eq!((resized.width(), resized.height()), (8, 4));
}

#[test]
fn developed_derivative_skips_early_resize() {
    let image = DynamicImage::ImageRgb8(RgbImage::new(20, 10));
    let resized = before_develop(
        image,
        ResizePlan::Derivative {
            max_width: 8,
            max_height: 8,
            before_develop: false,
        },
    );
    assert_eq!((resized.width(), resized.height()), (20, 10));
}

#[test]
fn baseline_plan_resizes_pattern_before_develop_boundary() {
    let plan = ResizePlan::Derivative {
        max_width: 4,
        max_height: 4,
        before_develop: true,
    };
    let expected = pattern().thumbnail(4, 4).to_rgb8();
    let early = before_develop(pattern(), plan);
    assert_eq!(early.to_rgb8(), expected);
    assert_eq!(output(early, plan).to_rgb8(), expected);
}

#[test]
fn developed_plan_keeps_pattern_until_final_resize() {
    let plan = ResizePlan::Derivative {
        max_width: 4,
        max_height: 4,
        before_develop: false,
    };
    let original = pattern();
    let before = before_develop(original.clone(), plan);
    assert_eq!(before.to_rgb8(), original.to_rgb8());
    assert_eq!(
        output(before, plan).to_rgb8(),
        original.thumbnail(4, 4).to_rgb8()
    );
}
