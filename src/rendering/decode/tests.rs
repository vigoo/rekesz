use super::*;
use crate::rendering::RenderVariant;
use image::{DynamicImage, ImageBuffer, Rgb};
use test_r::test;

#[test]
fn heif_peak_estimator_covers_compressed_and_16bit_alpha_coexistence() {
    assert!(super::heif::heif_allocation_within_budget(
        20_000_000,
        4_000,
        3_000,
        true,
        true,
        768 * 1024 * 1024
    ));
    assert!(!super::heif::heif_allocation_within_budget(
        20_000_000,
        50_000,
        50_000,
        true,
        true,
        768 * 1024 * 1024
    ));
    assert!(!super::heif::heif_allocation_within_budget(
        usize::MAX,
        1,
        1,
        false,
        false,
        usize::MAX
    ));
    let peak = super::heif::heif_peak_bytes(1_000_000, 4_000, 3_000, true, true).unwrap();
    assert!(peak > super::heif::heif_peak_bytes(1_000_000, 4_000, 3_000, false, false).unwrap());
    assert_eq!(
        peak,
        1_000_000 + 4_000 * 3_000 * 4 * 2 * 4 + 32 * 1024 * 1024
    );
}

#[test]
fn heif_compressed_source_is_admitted_before_read_only_in_analysis_mode() {
    let fixed = 32_usize * 1024 * 1024;
    assert!(super::heif::heif_compressed_source_admitted(1, fixed + 1));
    assert!(!super::heif::heif_compressed_source_admitted(2, fixed + 1));
    assert!(!super::heif::heif_compressed_source_admitted(
        u64::MAX,
        usize::MAX
    ));
    assert!(super::heif::heif_pre_read_admitted(u64::MAX, None));
}

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

#[test]
fn rotated_heif_reports_the_transform_applied_by_libheif() {
    use libheif_rs::{
        Channel, ColorSpace, CompressionFormat, EncodingOptions, HeifContext, Image,
        ImageOrientation, LibHeif, RgbChroma,
    };
    let mut image = Image::new(8, 4, ColorSpace::Rgb(RgbChroma::Rgb)).unwrap();
    image.create_plane(Channel::Interleaved, 8, 4, 24).unwrap();
    {
        let plane = image.planes_mut().interleaved.unwrap();
        for y in 0..4_usize {
            for x in 0..8_usize {
                let offset = y * plane.stride + x * 3;
                let color = if x < 4 { [240, 8, 8] } else { [8, 8, 240] };
                plane.data[offset..offset + 3].copy_from_slice(&color);
            }
        }
    }
    let lib_heif = LibHeif::new();
    let mut context = HeifContext::new().unwrap();
    let mut encoder = lib_heif.encoder_for_format(CompressionFormat::Av1).unwrap();
    let mut options = EncodingOptions::new().unwrap();
    options.set_image_orientation(ImageOrientation::Rotate90Cw);
    context
        .encode_image(&image, &mut encoder, Some(options))
        .unwrap();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("rotated.heic");
    std::fs::write(&path, context.write_to_bytes().unwrap()).unwrap();
    let plan = super::super::plan::RenderPlan::compile(
        path,
        RenderVariant::face_context(64),
        super::super::RenderIdentity::default(),
        None,
    )
    .unwrap();
    let decoded = decode(&plan, &Arc::new(AtomicBool::new(false))).unwrap();
    assert_eq!(decoded.orientation_applied, Some(6));
    assert_eq!((decoded.image.width(), decoded.image.height()), (4, 8));
    let crop = super::super::apply_face_variant(
        decoded.image,
        RenderVariant::FaceCrop {
            face_id: 1,
            face_revision: 1,
            left_millionths: 0,
            top_millionths: 0,
            right_millionths: 500_000,
            bottom_millionths: 1_000_000,
            width: 32,
            padding_percent: 0,
        },
        decoded.face_transform,
    )
    .to_rgb8();
    let pixel = crop.get_pixel(crop.width() / 2, crop.height() / 2).0;
    assert!(
        pixel[0] > pixel[2],
        "canonical left region should remain red"
    );
}
