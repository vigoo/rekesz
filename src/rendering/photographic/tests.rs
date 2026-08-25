use super::*;
use test_r::test;

#[test]
fn auto_exposure_places_middle_grey_at_the_target() {
    let pixels = vec![[0.045; 3]; 100];
    let multiplier =
        exposure_multiplier(&pixels, AutoExposure::default(), &AtomicBool::new(false)).unwrap();
    assert!((multiplier - 4.0).abs() < 0.1);
}

#[test]
fn auto_exposure_is_bounded_for_nearly_black_scenes() {
    let pixels = vec![[0.000_001; 3]];
    assert_eq!(
        exposure_multiplier(&pixels, AutoExposure::default(), &AtomicBool::new(false)).unwrap(),
        16.0
    );
}

#[test]
fn tone_mapping_is_finite_and_bounded() {
    for value in [0.0, 0.18, 1.0, 10.0, f32::INFINITY] {
        let mapped = aces_fitted(value);
        assert!(mapped.is_finite() && (0.0..=1.0).contains(&mapped));
    }
}

#[test]
fn photographic_transform_observes_cancellation() {
    let pixels = vec![[0.18; 3]; 10_000];
    assert!(
        render_linear_rgb(
            100,
            100,
            pixels,
            &PhotographicSettings::default(),
            &AtomicBool::new(true),
        )
        .is_err()
    );
}

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
fn developed_recipe_uses_identity_exposure_instead_of_rendering_black() {
    let settings = PhotographicSettings::default();
    let recipe = crate::catalog::DevelopRecipe {
        exposure_ev: Some(0.5),
        ..crate::catalog::DevelopRecipe::default()
    };
    let mut pixels = vec![[0.18; 3]; 4];
    let automatic = apply_recipe_scene_linear(
        &mut pixels,
        2,
        2,
        &settings,
        Some(&recipe),
        1.25,
        &AtomicBool::new(false),
    )
    .unwrap();
    let (rendered, automatic_stops) = render_linear_rgb_with_exposure(
        2,
        2,
        pixels,
        &settings,
        automatic,
        &AtomicBool::new(false),
    )
    .unwrap();

    assert_eq!(automatic_stops, 0.0);
    assert!(
        rendered
            .pixels()
            .all(|pixel| pixel.0[..3].iter().all(|value| *value > 0))
    );
}

#[test]
fn wb_coefficients_use_neutral_for_absolute_and_as_shot_for_incremental() {
    let as_shot = [2.0, 1.0, 1.5, 1.0];
    let neutral = [1.2, 1.0, 0.8, 1.0];
    let absolute = crate::catalog::DevelopRecipe {
        white_balance: Some(crate::catalog::WhiteBalance {
            temperature: 6_500.0,
            tint: 0.0,
        }),
        ..crate::catalog::DevelopRecipe::default()
    };
    assert_eq!(
        white_balance_coefficients(as_shot, neutral, &absolute),
        neutral
    );
    let incremental = crate::catalog::DevelopRecipe {
        white_balance_increment: Some(crate::catalog::WhiteBalanceIncrement {
            temperature: 0.0,
            tint: 0.0,
        }),
        ..crate::catalog::DevelopRecipe::default()
    };
    assert_eq!(
        white_balance_coefficients(as_shot, neutral, &incremental),
        as_shot
    );
}

#[test]
fn kang2002_planckian_locus_approximates_d65() {
    let (x, y) = planckian_locus_xy_kang2002(6_500.0);

    assert!((x - 0.3135).abs() < 0.002);
    assert!((y - 0.3237).abs() < 0.002);
}

#[test]
fn tint_moves_white_point_without_changing_temperature_coordinate_directly() {
    let neutral = tinted_white_point(6_500.0, 0.0);
    let magenta = tinted_white_point(6_500.0, 20.0);
    assert_ne!(neutral, magenta);
    assert!(neutral.0.is_finite() && magenta.1.is_finite());
}

#[test]
#[ignore = "manual investigation using external RAW samples"]
fn report_reference_raw_exposure_and_white_balance() {
    let samples = [
        (
            "/mnt/pictures/Pictures/2020/Szlovénia/2020-07-29/DSC_0546.NEF",
            None,
        ),
        (
            "/mnt/pictures/Pictures/2026/Alpok túra/2026-08-03/IMG_0997.CR3",
            None,
        ),
        (
            "/mnt/pictures/Pictures/2026/Alpok túra/2026-08-03/IMG_1056.CR3",
            Some(crate::catalog::WhiteBalance {
                temperature: 5_942.0,
                tint: 5.0,
            }),
        ),
    ];
    let cancelled = Arc::new(AtomicBool::new(false));
    for (path, white_balance) in samples {
        let decoded = decode_raw(Path::new(path), &cancelled).unwrap();
        let absolute = white_balance
            .map(|white_balance| absolute_white_balance_coefficients(&decoded.raw, white_balance));
        let (_, metadata) =
            render_raw(&decoded, &PhotographicSettings::default(), None, &cancelled).unwrap();
        eprintln!(
            "{path}: as_shot={:?} neutral={:?} absolute={absolute:?} auto_stops={}",
            decoded.raw.wb_coeffs,
            decoded.raw.neutralwb(),
            metadata.automatic_exposure_stops
        );
    }
}

#[test]
fn raw_phase_c_sharpen_radius_changes_the_scene_linear_result() {
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
    apply_phase_c_scene_linear(
        &mut small_radius,
        7,
        7,
        &recipe(1.0),
        &AtomicBool::new(false),
    )
    .unwrap();
    apply_phase_c_scene_linear(
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
fn raw_phase_c_pixel_passes_observe_cancellation() {
    let mut pixels = vec![[0.2, 0.3, 0.4]; 16];
    let recipe = crate::catalog::DevelopRecipe {
        texture: Some(10.0),
        monochrome: Some(crate::catalog::MonochromeRecipe { mixer: [0.0; 8] }),
        parametric_curve: Some([1.0; 4]),
        ..crate::catalog::DevelopRecipe::default()
    };
    assert!(
        apply_phase_c_scene_linear(&mut pixels, 4, 4, &recipe, &AtomicBool::new(true),).is_err()
    );
}
