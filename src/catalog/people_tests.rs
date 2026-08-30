use super::*;
use test_r::test;

#[test]
fn every_lightroom_orientation_round_trips_a_valid_rectangle() {
    for orientation in ["AB", "BA", "BC", "CB", "CD", "DC", "DA", "AD"] {
        let bounds = displayed_bounds_to_source([0.1, 0.2, 0.4, 0.7], orientation).unwrap();
        assert!(valid_bounds(bounds), "{orientation}: {bounds:?}");
        assert!((bounds[2] - bounds[0] > 0.0) && (bounds[3] - bounds[1] > 0.0));
    }
}

#[test]
fn lightroom_clockwise_display_bounds_are_mapped_back_to_source() {
    let bounds = displayed_bounds_to_source([0.4, 0.1, 0.8, 0.4], "BC").unwrap();
    for (actual, expected) in bounds.into_iter().zip([0.1, 0.2, 0.4, 0.6]) {
        assert!((actual - expected).abs() < 1e-6);
    }
}
