use super::aces_fitted;
use test_r::test;

#[test]
fn tone_mapping_is_finite_and_bounded() {
    for value in [0.0, 0.18, 1.0, 10.0, f32::INFINITY] {
        let mapped = aces_fitted(value);
        assert!(mapped.is_finite() && (0.0..=1.0).contains(&mapped));
    }
}
