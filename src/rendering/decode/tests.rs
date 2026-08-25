use super::*;
use crate::rendering::RenderVariant;
use image::{DynamicImage, ImageBuffer, Rgb};
use test_r::test;

#[test]
fn jpeg_thumbnail_decode_uses_reduced_resolution() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("large.jpg");
    DynamicImage::ImageRgb8(ImageBuffer::from_pixel(800, 600, Rgb([1, 2, 3])))
        .save(&path)
        .unwrap();
    let plan = super::super::plan::RenderPlan::compile(
        path,
        RenderVariant::Thumbnail { width: 100 },
        super::super::RenderIdentity::default(),
        None,
    )
    .unwrap();
    let decoded = decode(&plan, &Arc::new(AtomicBool::new(false))).unwrap();

    assert_eq!(decoded.source_dimensions, (800, 600));
    assert!(decoded.image.width() < 800);
    assert!(decoded.image.height() < 600);
}

#[test]
fn developed_jpeg_thumbnail_keeps_full_resolution_for_recipe_processing() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("large.jpg");
    DynamicImage::ImageRgb8(ImageBuffer::from_pixel(800, 600, Rgb([1, 2, 3])))
        .save(&path)
        .unwrap();
    let recipe = crate::catalog::DevelopRecipe::default();
    let identity = super::super::RenderIdentity {
        mode: super::super::RenderMode::Developed,
        recipe_digest: recipe.digest().unwrap(),
        renderer_model: crate::catalog::RENDERER_MODEL_VERSION,
    };
    let plan = super::super::plan::RenderPlan::compile(
        path,
        RenderVariant::Thumbnail { width: 100 },
        identity,
        Some(&recipe),
    )
    .unwrap();
    let decoded = decode(&plan, &Arc::new(AtomicBool::new(false))).unwrap();

    assert_eq!(decoded.image.width(), 800);
    assert_eq!(decoded.image.height(), 600);
}

#[test]
fn cancellation_is_observed_before_decoder_dispatch() {
    let plan = super::super::plan::RenderPlan::compile(
        "/does/not/exist.jpg".into(),
        RenderVariant::Full,
        super::super::RenderIdentity::default(),
        None,
    )
    .unwrap();
    let error = match decode(&plan, &Arc::new(AtomicBool::new(true))) {
        Ok(_) => panic!("cancelled decode unexpectedly succeeded"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("render cancelled"));
}
