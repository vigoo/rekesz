use super::*;
use image::{ImageBuffer, Rgb};
use test_r::test;

#[test]
fn jpeg_thumbnail_decode_uses_reduced_resolution() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("large.jpg");
    DynamicImage::ImageRgb8(ImageBuffer::from_pixel(800, 600, Rgb([1, 2, 3])))
        .save(&path)
        .unwrap();
    let decoded = decode(
        &path,
        &PhotographicSettings::default(),
        RenderVariant::Thumbnail { width: 100 },
        false,
        &Arc::new(AtomicBool::new(false)),
    )
    .unwrap();

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
    let decoded = decode(
        &path,
        &PhotographicSettings::default(),
        RenderVariant::Thumbnail { width: 100 },
        true,
        &Arc::new(AtomicBool::new(false)),
    )
    .unwrap();

    assert_eq!(decoded.image.width(), 800);
    assert_eq!(decoded.image.height(), 600);
}
