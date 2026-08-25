use std::sync::atomic::AtomicBool;

use super::{render_linear_rgb, render_linear_rgb_with_exposure};
use crate::rendering::photographic::PhotographicSettings;
use test_r::test;

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
fn developed_recipe_uses_identity_exposure_instead_of_rendering_black() {
    use crate::rendering::photographic::scene_linear::apply_recipe_scene_linear;

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
