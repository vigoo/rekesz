use super::*;
use crate::catalog::Crop;
use image::ImageBuffer;
use test_r::test;

#[test]
fn identity_recipe_is_pixel_identical() {
    let image = DynamicImage::ImageRgba8(ImageBuffer::from_pixel(4, 4, Rgba([30, 60, 90, 255])));
    let (result, actions, warnings) =
        apply(&image, &DevelopRecipe::default(), &AtomicBool::new(false)).unwrap();
    assert_eq!(result.to_rgba8(), image.to_rgba8());
    assert_eq!((actions, warnings), (0, 0));
}

#[test]
fn exposure_brightens_and_curve_is_monotonic() {
    let image = DynamicImage::ImageRgba8(ImageBuffer::from_pixel(4, 4, Rgba([64, 64, 64, 255])));
    let recipe = DevelopRecipe {
        exposure_ev: Some(1.0),
        tone_curve: Some(vec![[0.0, 0.0], [0.5, 0.4], [1.0, 1.0]]),
        ..DevelopRecipe::default()
    };
    let (result, actions, _) = apply(&image, &recipe, &AtomicBool::new(false)).unwrap();
    assert!(result.to_rgba8().get_pixel(0, 0)[0] > 64);
    assert_ne!(actions & actions::EXPOSURE, 0);
    assert!(
        sample_curve(recipe.tone_curve.as_ref().unwrap(), 0.25)
            < sample_curve(recipe.tone_curve.as_ref().unwrap(), 0.75)
    );
}

#[test]
fn crop_changes_dimensions_and_cancellation_is_observed() {
    let image = DynamicImage::ImageRgba8(ImageBuffer::from_pixel(100, 80, Rgba([1, 2, 3, 255])));
    let recipe = DevelopRecipe {
        crop: Some(Crop {
            left: 0.25,
            top: 0.25,
            right: 0.75,
            bottom: 0.75,
            angle_degrees: 0.0,
            constrain_aspect: false,
        }),
        ..DevelopRecipe::default()
    };
    let (result, _, _) = apply(&image, &recipe, &AtomicBool::new(false)).unwrap();
    assert_eq!((result.width(), result.height()), (50, 40));
    assert!(apply(&image, &recipe, &AtomicBool::new(true)).is_err());
}

#[test]
fn phase_c_detail_monochrome_and_finishing_are_applied_and_reported() {
    let source = DynamicImage::ImageRgba8(RgbaImage::from_pixel(16, 16, Rgba([120, 80, 40, 255])));
    let recipe = DevelopRecipe {
        detail: Some(crate::catalog::DetailRecipe {
            sharpening: 20.0,
            ..crate::catalog::DetailRecipe::default()
        }),
        texture: Some(10.0),
        monochrome: Some(crate::catalog::MonochromeRecipe { mixer: [0.0; 8] }),
        vignette: Some(crate::catalog::VignetteRecipe {
            amount: -20.0,
            midpoint: 50.0,
            feather: 50.0,
            roundness: 0.0,
        }),
        parametric_curve: Some([5.0, 0.0, 0.0, 0.0]),
        ..DevelopRecipe::default()
    };
    let (output, actions, warnings) = apply(&source, &recipe, &AtomicBool::new(false)).unwrap();
    let pixel = output.to_rgba8().get_pixel(8, 8).to_owned();

    assert_eq!(pixel[0], pixel[1]);
    assert_eq!(pixel[1], pixel[2]);
    assert_ne!(actions & actions::DETAIL, 0);
    assert_ne!(actions & actions::MONOCHROME, 0);
    assert_ne!(actions & actions::VIGNETTE, 0);
    assert_ne!(warnings & warnings::APPROXIMATED, 0);
}

#[test]
fn every_monochrome_hue_sector_changes_its_matching_colour() {
    let colours = [
        [255, 0, 0, 255],
        [255, 191, 0, 255],
        [128, 255, 0, 255],
        [0, 255, 64, 255],
        [0, 255, 255, 255],
        [0, 64, 255, 255],
        [128, 0, 255, 255],
        [255, 0, 191, 255],
    ];
    for (sector, colour) in colours.into_iter().enumerate() {
        let mut baseline = RgbaImage::from_pixel(1, 1, Rgba(colour));
        let mut adjusted = baseline.clone();
        Monochrome {
            parameters: crate::catalog::MonochromeRecipe { mixer: [0.0; 8] },
        }
        .apply(&mut baseline, &AtomicBool::new(false))
        .unwrap();
        let mut mixer = [0.0; 8];
        mixer[sector] = 50.0;
        Monochrome {
            parameters: crate::catalog::MonochromeRecipe { mixer },
        }
        .apply(&mut adjusted, &AtomicBool::new(false))
        .unwrap();
        assert_ne!(baseline, adjusted, "sector {sector} was ignored");
    }
}

#[test]
fn parametric_curve_is_continuous_across_old_region_boundaries() {
    let mut image = RgbaImage::new(2, 1);
    image.put_pixel(0, 0, Rgba([127, 127, 127, 255]));
    image.put_pixel(1, 0, Rgba([128, 128, 128, 255]));
    ParametricCurve {
        controls: [100.0, -100.0, 100.0, -100.0],
    }
    .apply(&mut image, &AtomicBool::new(false))
    .unwrap();
    assert!((i16::from(image.get_pixel(0, 0)[0]) - i16::from(image.get_pixel(1, 0)[0])).abs() < 8);
}

#[test]
fn cancellable_filter_observes_active_cancellation() {
    let image = RgbaImage::from_pixel(800, 800, Rgba([100, 120, 140, 255]));
    let cancelled = std::sync::Arc::new(AtomicBool::new(false));
    let worker_cancelled = cancelled.clone();
    let worker = std::thread::spawn(move || cancellable_box_blur(&image, 3, &worker_cancelled));
    std::thread::sleep(std::time::Duration::from_millis(1));
    cancelled.store(true, Ordering::Release);
    assert!(worker.join().unwrap().is_err());
}

#[test]
fn phase_d_geometry_heal_and_grain_are_deterministic_and_reported() {
    let mut source = RgbaImage::from_pixel(12, 12, Rgba([80, 80, 80, 255]));
    source.put_pixel(2, 2, Rgba([220, 30, 30, 255]));
    let recipe = DevelopRecipe {
        upright_homography: Some([1.0, 0.0, 0.05, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
        lens_distortion: Some(10.0),
        heal_spots: vec![crate::catalog::HealSpot {
            kind: crate::catalog::HealKind::Heal,
            destination: [0.5, 0.5],
            source: [2.0 / 12.0, 2.0 / 12.0],
            radius: 0.15,
            opacity: 1.0,
            feather: 0.0,
        }],
        grain: Some(crate::catalog::GrainRecipe {
            amount: 15.0,
            size: 30.0,
            roughness: 50.0,
            seed: 42,
        }),
        ..DevelopRecipe::default()
    };
    let source = DynamicImage::ImageRgba8(source);
    let first = apply(&source, &recipe, &AtomicBool::new(false)).unwrap();
    let second = apply(&source, &recipe, &AtomicBool::new(false)).unwrap();

    assert_eq!(first.0.to_rgba8(), second.0.to_rgba8());
    assert_ne!(first.1 & actions::UPRIGHT, 0);
    assert_ne!(first.1 & actions::LENS, 0);
    assert_ne!(first.1 & actions::HEAL, 0);
    assert_ne!(first.1 & actions::GRAIN, 0);
    assert_ne!(first.2 & warnings::APPROXIMATED, 0);
}

#[test]
fn singular_homography_is_rejected() {
    let image = RgbaImage::from_pixel(2, 2, Rgba([0, 0, 0, 255]));
    assert!(
        Homography {
            matrix: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        }
        .apply(&mut image.clone(), &AtomicBool::new(false))
        .is_err()
    );
}

#[test]
fn clone_and_heal_have_distinct_semantics() {
    let mut source = RgbaImage::from_pixel(20, 20, Rgba([20, 30, 40, 255]));
    source.put_pixel(2, 2, Rgba([200, 50, 40, 255]));
    source.put_pixel(10, 10, Rgba([70, 90, 110, 255]));
    let base = crate::catalog::HealSpot {
        kind: crate::catalog::HealKind::Clone,
        destination: [10.0 / 19.0, 10.0 / 19.0],
        source: [2.0 / 19.0, 2.0 / 19.0],
        radius: 0.08,
        opacity: 1.0,
        feather: 0.0,
    };
    let mut cloned = source.clone();
    Heal { spots: &[base] }
        .apply(&mut cloned, &AtomicBool::new(false))
        .unwrap();
    let mut healed = source;
    Heal {
        spots: &[crate::catalog::HealSpot {
            kind: crate::catalog::HealKind::Heal,
            ..base
        }],
    }
    .apply(&mut healed, &AtomicBool::new(false))
    .unwrap();

    assert_ne!(cloned.get_pixel(10, 10), healed.get_pixel(10, 10));
    assert!(cloned.get_pixel(10, 10)[0] > healed.get_pixel(10, 10)[0]);
}

#[test]
fn heal_accepts_normalized_image_boundaries() {
    let mut image = RgbaImage::from_pixel(8, 8, Rgba([20, 30, 40, 255]));
    image.put_pixel(0, 0, Rgba([200, 100, 50, 255]));
    Heal {
        spots: &[
            crate::catalog::HealSpot {
                kind: crate::catalog::HealKind::Clone,
                destination: [1.0, 1.0],
                source: [0.0, 0.0],
                radius: 0.01,
                opacity: 1.0,
                feather: 0.0,
            },
            crate::catalog::HealSpot {
                kind: crate::catalog::HealKind::Clone,
                destination: [0.0, 0.0],
                source: [1.0, 1.0],
                radius: 0.01,
                opacity: 1.0,
                feather: 0.0,
            },
        ],
    }
    .apply(&mut image, &AtomicBool::new(false))
    .unwrap();
}

#[test]
fn phase_d_pixel_passes_observe_active_cancellation() {
    fn cancelled_during(
        operation: impl FnOnce(&AtomicBool) -> image::ImageResult<()> + Send + 'static,
    ) {
        let cancelled = std::sync::Arc::new(AtomicBool::new(false));
        let worker_cancelled = cancelled.clone();
        let worker = std::thread::spawn(move || operation(&worker_cancelled));
        std::thread::sleep(std::time::Duration::from_millis(1));
        cancelled.store(true, Ordering::Release);
        assert!(worker.join().unwrap().is_err());
    }

    cancelled_during(|cancelled| {
        let mut image = RgbaImage::from_pixel(2_000, 2_000, Rgba([80, 90, 100, 255]));
        Heal {
            spots: &[crate::catalog::HealSpot {
                kind: crate::catalog::HealKind::Clone,
                destination: [0.5, 0.5],
                source: [0.25, 0.25],
                radius: 0.5,
                opacity: 1.0,
                feather: 0.5,
            }],
        }
        .apply(&mut image, cancelled)
    });
    cancelled_during(|cancelled| {
        LensDistortion { amount: 20.0 }.apply(&mut RgbaImage::new(2_000, 2_000), cancelled)
    });
    cancelled_during(|cancelled| {
        Homography {
            matrix: [1.0, 0.0, 0.05, 0.0, 1.0, 0.05, 0.0, 0.0, 1.0],
        }
        .apply(&mut RgbaImage::new(2_000, 2_000), cancelled)
    });
    cancelled_during(|cancelled| {
        let mut image = RgbaImage::new(2_000, 2_000);
        Grain {
            parameters: crate::catalog::GrainRecipe {
                amount: 20.0,
                size: 30.0,
                roughness: 50.0,
                seed: 1,
            },
        }
        .apply(&mut image, cancelled)
    });
}

#[test]
fn grain_is_deterministic_per_asset_seed() {
    let source = RgbaImage::from_pixel(32, 32, Rgba([128, 128, 128, 255]));
    let grain = crate::catalog::GrainRecipe {
        amount: 20.0,
        size: 30.0,
        roughness: 50.0,
        seed: 1,
    };
    let mut first = source.clone();
    let mut repeat = source.clone();
    let mut other = source;
    Grain { parameters: grain }
        .apply(&mut first, &AtomicBool::new(false))
        .unwrap();
    Grain { parameters: grain }
        .apply(&mut repeat, &AtomicBool::new(false))
        .unwrap();
    Grain {
        parameters: crate::catalog::GrainRecipe { seed: 2, ..grain },
    }
    .apply(&mut other, &AtomicBool::new(false))
    .unwrap();
    assert_eq!(first, repeat);
    assert_ne!(first, other);
}

#[test]
fn clarity_at_zero_is_a_pixel_identical_noop() {
    let image = RgbaImage::from_pixel(16, 16, Rgba([90, 140, 200, 255]));
    let mut result = image.clone();
    Clarity { amount: 0.0 }
        .apply(&mut result, &AtomicBool::new(false))
        .unwrap();
    assert_eq!(result, image);
}

#[test]
fn clarity_keeps_neutral_gradients_neutral() {
    let mut image = RgbaImage::new(64, 64);
    for y in 0..64 {
        for x in 0..64 {
            let value = (x * 4) as u8;
            image.put_pixel(x, y, Rgba([value, value, value, 255]));
        }
    }
    let mut result = image.clone();
    Clarity { amount: 100.0 }
        .apply(&mut result, &AtomicBool::new(false))
        .unwrap();
    for (original, adjusted) in image.pixels().zip(result.pixels()) {
        assert_eq!(adjusted[0], adjusted[1], "green diverged from red");
        assert_eq!(adjusted[1], adjusted[2], "blue diverged from green");
        let _ = original;
    }
}

#[test]
fn positive_clarity_increases_edge_contrast_and_negative_reduces_it() {
    let mut image = RgbaImage::new(270, 540);
    for y in 0..540 {
        for x in 0..270 {
            let value = if x < 135 { 51_u8 } else { 191_u8 };
            image.put_pixel(x, y, Rgba([value, value, value, 255]));
        }
    }
    let edge_difference = |image: &RgbaImage| {
        let before = i32::from(image.get_pixel(134, 270)[0]);
        let after = i32::from(image.get_pixel(135, 270)[0]);
        (after - before).abs()
    };
    let baseline = edge_difference(&image);
    let mut boosted = image.clone();
    Clarity { amount: 100.0 }
        .apply(&mut boosted, &AtomicBool::new(false))
        .unwrap();
    assert!(
        edge_difference(&boosted) > baseline,
        "positive clarity must open the edge gap: {} -> {}",
        baseline,
        edge_difference(&boosted)
    );
    let mut flattened = image.clone();
    Clarity { amount: -100.0 }
        .apply(&mut flattened, &AtomicBool::new(false))
        .unwrap();
    assert!(
        edge_difference(&flattened) < baseline,
        "negative clarity must close the edge gap: {} -> {}",
        baseline,
        edge_difference(&flattened)
    );
}

#[test]
fn clarity_leaves_saturated_blacks_and_whites_untouched() {
    let mut image = RgbaImage::new(270, 540);
    for y in 0..540 {
        for x in 0..270 {
            let value = if y < 180 {
                0_u8
            } else if y < 360 {
                128_u8
            } else {
                255_u8
            };
            image.put_pixel(x, y, Rgba([value, value, value, 255]));
        }
    }
    let mut result = image.clone();
    Clarity { amount: 100.0 }
        .apply(&mut result, &AtomicBool::new(false))
        .unwrap();
    for y in [0, 179, 360, 539] {
        for x in [0, 135, 269] {
            assert_eq!(
                result.get_pixel(x, y),
                image.get_pixel(x, y),
                "extreme tone at ({x},{y}) changed"
            );
        }
    }
}

#[test]
fn presence_at_zero_is_a_pixel_identical_noop() {
    let image = RgbaImage::from_pixel(16, 16, Rgba([90, 140, 200, 255]));
    let mut round_trip = image.clone();
    Presence {
        texture: Some(0.0),
        dehaze: Some(0.0),
    }
    .apply(&mut round_trip, &AtomicBool::new(false))
    .unwrap();
    assert_eq!(
        round_trip, image,
        "u8 to linear to u8 must round-trip exactly"
    );
}
