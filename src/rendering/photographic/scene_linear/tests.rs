use std::sync::atomic::AtomicBool;

use super::{apply_detail_controls, apply_recipe_scene_linear};
use crate::rendering::photographic::PhotographicSettings;
use test_r::test;

#[test]
fn catalog_recipe_uses_fixed_baseline_and_applies_authored_exposure_before_curves() {
    let mut baseline = vec![[0.25; 3]];
    let mut developed = baseline.clone();
    let recipe = crate::catalog::DevelopRecipe {
        exposure_ev: Some(1.0),
        tone_curve: Some(vec![[0.0, 0.0], [0.5, 0.25], [1.0, 1.0]]),
        ..crate::catalog::DevelopRecipe::default()
    };
    let settings = PhotographicSettings::default();
    let baseline_anchor = apply_recipe_scene_linear(
        &mut baseline,
        1,
        1,
        &settings,
        None,
        0.0,
        &AtomicBool::new(false),
    )
    .unwrap();
    let developed_anchor = apply_recipe_scene_linear(
        &mut developed,
        1,
        1,
        &settings,
        Some(&recipe),
        1.25,
        &AtomicBool::new(false),
    )
    .unwrap();
    assert_ne!(baseline_anchor, 1.0);
    assert_eq!(developed_anchor, 1.0);
    assert!((developed[0][0] - 1.0).abs() < 0.001);
}

#[test]
fn even_a_recipe_without_exposure_disables_aesthetic_auto_exposure() {
    let settings = PhotographicSettings::default();
    let mut baseline = vec![[0.05; 3]];
    let mut developed = baseline.clone();
    let recipe = crate::catalog::DevelopRecipe {
        saturation: Some(10.0),
        ..crate::catalog::DevelopRecipe::default()
    };
    let baseline_automatic = apply_recipe_scene_linear(
        &mut baseline,
        1,
        1,
        &settings,
        None,
        0.0,
        &AtomicBool::new(false),
    )
    .unwrap();
    let developed_automatic = apply_recipe_scene_linear(
        &mut developed,
        1,
        1,
        &settings,
        Some(&recipe),
        1.25,
        &AtomicBool::new(false),
    )
    .unwrap();
    assert!(baseline_automatic > 1.0);
    assert_eq!(developed_automatic, 1.0);
}

#[test]
fn sharpen_radius_changes_the_scene_linear_result() {
    let mut small_radius = vec![[0.1; 3]; 49];
    small_radius[24] = [1.0; 3];
    let mut large_radius = small_radius.clone();
    let recipe = |radius| crate::catalog::DevelopRecipe {
        detail: Some(crate::catalog::DetailRecipe {
            sharpening: 100.0,
            sharpen_radius: radius,
            sharpen_detail: 50.0,
            ..crate::catalog::DetailRecipe::default()
        }),
        ..crate::catalog::DevelopRecipe::default()
    };
    apply_detail_controls(
        &mut small_radius,
        7,
        7,
        &recipe(1.0),
        &AtomicBool::new(false),
    )
    .unwrap();
    apply_detail_controls(
        &mut large_radius,
        7,
        7,
        &recipe(3.0),
        &AtomicBool::new(false),
    )
    .unwrap();
    assert_ne!(small_radius, large_radius);
}

#[test]
fn pixel_passes_observe_cancellation() {
    let mut pixels = vec![[0.2, 0.3, 0.4]; 16];
    let recipe = crate::catalog::DevelopRecipe {
        texture: Some(10.0),
        monochrome: Some(crate::catalog::MonochromeRecipe { mixer: [0.0; 8] }),
        parametric_curve: Some([1.0; 4]),
        ..crate::catalog::DevelopRecipe::default()
    };
    assert!(apply_detail_controls(&mut pixels, 4, 4, &recipe, &AtomicBool::new(true)).is_err());
}

#[test]
fn every_scene_linear_recipe_group_changes_pixels() {
    let curve = Some(vec![[0.0, 0.15], [1.0, 0.85]]);
    let cases = [
        crate::catalog::DevelopRecipe {
            tone_curve: curve.clone(),
            ..Default::default()
        },
        crate::catalog::DevelopRecipe {
            tone_curve_red: curve.clone(),
            tone_curve_green: curve.clone(),
            tone_curve_blue: curve,
            ..Default::default()
        },
        crate::catalog::DevelopRecipe {
            detail: Some(crate::catalog::DetailRecipe {
                sharpening: 100.0,
                sharpen_radius: 1.0,
                sharpen_detail: 50.0,
                ..Default::default()
            }),
            ..Default::default()
        },
        crate::catalog::DevelopRecipe {
            texture: Some(50.0),
            dehaze: Some(25.0),
            ..Default::default()
        },
        crate::catalog::DevelopRecipe {
            monochrome: Some(crate::catalog::MonochromeRecipe { mixer: [20.0; 8] }),
            ..Default::default()
        },
        crate::catalog::DevelopRecipe {
            parametric_curve: Some([50.0, -25.0, 30.0, -10.0]),
            ..Default::default()
        },
    ];
    let source = (0..49)
        .map(|index| {
            let value = index as f32 / 50.0 + 0.01;
            [value, value * 0.7, value * 0.4]
        })
        .collect::<Vec<_>>();
    for recipe in cases {
        let mut pixels = source.clone();
        apply_recipe_scene_linear(
            &mut pixels,
            7,
            7,
            &PhotographicSettings::default(),
            Some(&recipe),
            0.0,
            &AtomicBool::new(false),
        )
        .unwrap();
        assert_ne!(
            pixels, source,
            "recipe group did not affect scene-linear pixels"
        );
    }
}
