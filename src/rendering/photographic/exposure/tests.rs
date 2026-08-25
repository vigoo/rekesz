use std::sync::atomic::AtomicBool;

use super::{AutoExposure, exposure_multiplier};
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
