use std::sync::{Arc, atomic::AtomicBool};

use image::GenericImageView;

use super::render_linear;
use crate::{
    catalog::{DetailRecipe, DevelopRecipe, MonochromeRecipe},
    rendering::photographic::{PhotographicSettings, rawler_stage::LinearRgb},
};
use test_r::test;

#[test]
fn authored_linear_pipeline_has_stable_pixels_and_metadata() {
    let width = 7;
    let height = 5;
    let pixels = (0..width * height)
        .map(|index| {
            let x = (index % width) as f32 / (width - 1) as f32;
            let y = (index / width) as f32 / (height - 1) as f32;
            [0.02 + x * 0.8, 0.03 + y * 0.6, 0.04 + (x + y) * 0.25]
        })
        .collect();
    let recipe = DevelopRecipe {
        exposure_ev: Some(0.35),
        contrast: Some(12.0),
        highlights: Some(-18.0),
        shadows: Some(21.0),
        whites: Some(7.0),
        blacks: Some(-5.0),
        texture: Some(8.0),
        dehaze: Some(3.0),
        saturation: Some(9.0),
        vibrance: Some(14.0),
        detail: Some(DetailRecipe {
            sharpening: 25.0,
            sharpen_radius: 1.0,
            sharpen_detail: 20.0,
            luminance_noise_reduction: 5.0,
            ..DetailRecipe::default()
        }),
        monochrome: Some(MonochromeRecipe {
            mixer: [2.0, -1.0, 3.0, 0.0, -2.0, 1.0, 4.0, -3.0],
        }),
        parametric_curve: Some([5.0, -3.0, 4.0, -2.0]),
        ..DevelopRecipe::default()
    };
    let (image, metadata) = render_linear(
        LinearRgb {
            width,
            height,
            pixels,
        },
        Some(image::metadata::Orientation::Rotate90),
        &PhotographicSettings::default(),
        Some(&recipe),
        &Arc::new(AtomicBool::new(false)),
    )
    .unwrap();

    let hash = image
        .as_bytes()
        .iter()
        .fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
        });
    assert_eq!(image.dimensions(), (height, width));
    assert_eq!(hash, 13_824_727_338_061_481_894);
    assert_eq!(metadata.automatic_exposure_stops, 0.0);
    assert!((metadata.total_exposure_stops - 1.1).abs() < f32::EPSILON);
    assert_eq!(metadata.orientation_applied, Some(6));
}
