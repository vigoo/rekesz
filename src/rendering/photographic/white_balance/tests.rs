use super::{coefficients, planckian_locus_xy_kang2002, tinted_white_point};
use test_r::test;

#[test]
fn coefficients_use_neutral_for_absolute_and_as_shot_for_incremental() {
    let as_shot = [2.0, 1.0, 1.5, 1.0];
    let neutral = [1.2, 1.0, 0.8, 1.0];
    let absolute = crate::catalog::DevelopRecipe {
        white_balance: Some(crate::catalog::WhiteBalance {
            temperature: 6_500.0,
            tint: 0.0,
        }),
        ..crate::catalog::DevelopRecipe::default()
    };
    assert_eq!(coefficients(as_shot, neutral, &absolute), neutral);
    let incremental = crate::catalog::DevelopRecipe {
        white_balance_increment: Some(crate::catalog::WhiteBalanceIncrement {
            temperature: 0.0,
            tint: 0.0,
        }),
        ..crate::catalog::DevelopRecipe::default()
    };
    assert_eq!(coefficients(as_shot, neutral, &incremental), as_shot);
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
