use std::{fs, thread, time::Duration};
use test_r::test;

use image::{ImageBuffer, Rgb};

use super::*;

#[test]
fn packaged_libheif_has_an_hevc_decoder() {
    assert_hevc_decoder_available().unwrap();
}

#[test]
fn developed_render_applies_recipe_and_records_identity() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("source.png");
    DynamicImage::ImageRgb8(ImageBuffer::from_pixel(8, 8, Rgb([64, 64, 64])))
        .save(&source)
        .unwrap();
    let backend = RenderBackend::new(directory.path().join("cache"));
    let recipe = DevelopRecipe {
        exposure_ev: Some(1.0),
        ..DevelopRecipe::default()
    };
    let identity = RenderIdentity {
        mode: RenderMode::Developed,
        recipe_digest: recipe.digest().unwrap(),
        renderer_model: crate::catalog::RENDERER_MODEL_VERSION,
    };
    let request = RenderRequest::with_identity(source, RenderVariant::Full, identity);
    let cancelled = Arc::new(AtomicBool::new(false));
    let plan = plan::RenderPlan::compile(
        request.path.clone(),
        request.variant,
        identity,
        Some(&recipe),
    )
    .unwrap();
    let decoded = backend.decode_source(&plan, &cancelled).unwrap();
    let baseline = decoded.image.to_rgb8().get_pixel(0, 0)[0];

    let rendered = backend.render_source(&plan, decoded, &cancelled).unwrap();

    assert!(rendered.image.to_rgb8().get_pixel(0, 0)[0] > baseline);
    let metadata = rendered.metadata.develop.unwrap();
    assert_eq!(metadata.identity, identity);
    assert_ne!(metadata.applied_actions & develop::actions::EXPOSURE, 0);
}

#[test]
fn stale_render_is_not_stored_under_a_changed_source_identity() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("source.jpg");
    fs::write(&source, b"old source").unwrap();
    let backend = RenderBackend::new(directory.path().join("cache"));
    let request = RenderRequest::new(source.clone(), RenderVariant::Thumbnail { width: 300 });
    let identity = backend.source_identity(&request).unwrap().unwrap();
    let rendered = RenderedImage {
        image: DynamicImage::ImageRgb8(ImageBuffer::from_pixel(1, 1, Rgb([1, 2, 3]))),
        metadata: RenderMetadata {
            source_pipeline: SourcePipeline::Standard,
            photographic: None,
            source_dimensions: (1, 1),
            output_dimensions: (1, 1),
            variant: request.variant,
            orientation_applied: None,
            face_transform: None,
            develop: None,
        },
    };
    thread::sleep(Duration::from_millis(2));
    fs::write(&source, b"new source with a different length").unwrap();

    backend.store_cache(&request, &identity, &rendered).unwrap();
    let current = backend.source_identity(&request).unwrap().unwrap();
    assert_ne!(identity, current);
    assert!(
        backend
            .lookup_cache(&request, Some(&current))
            .unwrap()
            .is_none()
    );
}

#[test]
fn rendering_sources_preserve_dependency_direction() {
    fn visit(directory: &std::path::Path, files: &mut Vec<std::path::PathBuf>) {
        for entry in fs::read_dir(directory).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                visit(&path, files);
            } else if path.extension().is_some_and(|extension| extension == "rs") {
                files.push(path);
            }
        }
    }

    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/rendering");
    let mut files = Vec::new();
    visit(&root, &mut files);
    for path in files {
        if path == root.join("tests.rs") {
            continue;
        }
        let source = fs::read_to_string(&path).unwrap();
        assert!(
            !source.contains("crate::preview"),
            "{} imports preview",
            path.display()
        );
        assert!(!source.contains("gpui"), "{} imports GPUI", path.display());
        if path != root.join("mod.rs")
            && !path.components().any(|part| part.as_os_str() == "pipeline")
        {
            assert!(
                !source.contains("pipeline::"),
                "{} imports upward from pipeline",
                path.display()
            );
        }
    }
}

#[test]
fn pipeline_is_the_stable_rendering_entrypoint() {
    let _constructor: fn(PathBuf, usize, usize, usize) -> pipeline::RenderPipeline =
        pipeline::RenderPipeline::with_source_cache_capacity;
    fn accepts_invocation(_: pipeline::RenderInvocation<'_>) {}
    let _ = accepts_invocation;
}

#[test]
fn face_bounds_follow_exif_orientation_once() {
    let oriented = oriented_face_bounds([0.1, 0.2, 0.4, 0.6], 6);
    for (actual, expected) in oriented.into_iter().zip([0.4, 0.1, 0.8, 0.4]) {
        assert!((actual - expected).abs() < 1e-6);
    }
}

#[test]
fn face_bounds_cover_all_exif_orientations() {
    let source = [0.1, 0.2, 0.4, 0.6];
    let expected = [
        [0.1, 0.2, 0.4, 0.6],
        [0.6, 0.2, 0.9, 0.6],
        [0.6, 0.4, 0.9, 0.8],
        [0.1, 0.4, 0.4, 0.8],
        [0.2, 0.1, 0.6, 0.4],
        [0.4, 0.1, 0.8, 0.4],
        [0.4, 0.6, 0.8, 0.9],
        [0.2, 0.6, 0.6, 0.9],
    ];
    for (orientation, expected) in (1..=8).zip(expected) {
        for (actual, expected) in oriented_face_bounds(source, orientation)
            .into_iter()
            .zip(expected)
        {
            assert!(
                (actual - expected).abs() < 1e-6,
                "orientation {orientation}"
            );
        }
    }
}

#[test]
fn face_transform_inverse_round_trips_all_exif_orientations_and_crop() {
    let canonical = [0.13, 0.21, 0.64, 0.83];
    for orientation in 1..=8 {
        let transform = FaceTransform::crop(0.07, 0.11, 0.03, 0.05)
            .then(FaceTransform::orientation(orientation));
        let decoded = transform.bounds(canonical);
        let round_trip = transform.inverse().unwrap().bounds(decoded);
        for (actual, expected) in round_trip.into_iter().zip(canonical) {
            assert!(
                (actual - expected).abs() < 1e-5,
                "orientation {orientation}"
            );
        }
        let point = [0.31, 0.47];
        let round_trip = transform
            .inverse()
            .unwrap()
            .mapped_point(transform.mapped_point(point));
        assert!((round_trip[0] - point[0]).abs() < 1e-5);
        assert!((round_trip[1] - point[1]).abs() < 1e-5);
    }
}

#[test]
fn face_crop_extracts_the_requested_source_region() {
    let image = ImageBuffer::from_fn(100, 100, |x, y| Rgb([x as u8, y as u8, 0]));
    let variant = RenderVariant::FaceCrop {
        face_id: 7,
        face_revision: 1,
        left_millionths: 200_000,
        top_millionths: 300_000,
        right_millionths: 400_000,
        bottom_millionths: 500_000,
        width: 20,
        padding_percent: 0,
    };
    let cropped = apply_face_variant(
        DynamicImage::ImageRgb8(image),
        variant,
        FaceTransform::IDENTITY,
    )
    .to_rgb8();
    assert_eq!(cropped.dimensions(), (20, 20));
    let center = cropped.get_pixel(10, 10).0;
    assert!((28..=32).contains(&center[0]));
    assert!((38..=42).contains(&center[1]));
}

#[test]
fn face_crop_is_pixel_square_on_a_wide_photo() {
    let image = DynamicImage::ImageRgb8(ImageBuffer::from_pixel(400, 100, Rgb([1, 2, 3])));
    let variant = RenderVariant::FaceCrop {
        face_id: 8,
        face_revision: 1,
        left_millionths: 400_000,
        top_millionths: 200_000,
        right_millionths: 600_000,
        bottom_millionths: 800_000,
        width: 200,
        padding_percent: 0,
    };
    let cropped = apply_face_variant(image, variant, FaceTransform::IDENTITY);
    assert_eq!(cropped.width(), cropped.height());
}

#[test]
fn face_transform_composes_clean_aperture_before_rotation() {
    let transform = FaceTransform::IDENTITY
        .then(FaceTransform::crop(0.25, 0.0, 0.25, 0.0))
        .then(FaceTransform::orientation(6));
    let actual = transform.bounds([0.25, 0.0, 0.5, 1.0]);
    let expected = [0.0, 0.0, 1.0, 0.5];
    for (actual, expected) in actual.into_iter().zip(expected) {
        assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
    }
}

/// A renderer-version bump must invalidate baseline caches too: the
/// default identity (grid thumbnails, previews) has to embed the current
/// `RENDERER_MODEL_VERSION`. Regression for stale raw thumbnails/previews
/// after the lens-correction bumps.
#[test]
fn default_render_identity_carries_the_renderer_model_version() {
    let identity = RenderIdentity::default();
    assert_eq!(
        identity.renderer_model,
        crate::catalog::RENDERER_MODEL_VERSION
    );
    assert_eq!(identity.mode, RenderMode::Baseline);
    assert_eq!(identity.recipe_digest, [0; 32]);
}

#[test]
fn display_source_dimensions_reconciles_legacy_rotated_cache_entries() {
    fn metadata(
        variant: RenderVariant,
        orientation_applied: Option<u8>,
        source: (u32, u32),
        output: (u32, u32),
    ) -> RenderMetadata {
        RenderMetadata {
            source_pipeline: SourcePipeline::Standard,
            photographic: None,
            source_dimensions: source,
            output_dimensions: output,
            variant,
            orientation_applied,
            face_transform: None,
            develop: None,
        }
    }
    let preview = RenderVariant::Preview {
        max_width: 1920,
        max_height: 1080,
    };
    // Legacy scaled-JPEG entries recorded pre-rotation sensor dimensions.
    let legacy = metadata(preview, Some(6), (4032, 3024), (1512, 2016));
    assert_eq!(legacy.display_source_dimensions(), (3024, 4032));
    // Current decoders record oriented dimensions directly.
    let current = metadata(
        RenderVariant::Thumbnail { width: 300 },
        Some(6),
        (3024, 4032),
        (300, 400),
    );
    assert_eq!(current.display_source_dimensions(), (3024, 4032));
    // Without a transposing orientation nothing is ever swapped.
    let landscape = metadata(preview, None, (4032, 3024), (2016, 1512));
    assert_eq!(landscape.display_source_dimensions(), (4032, 3024));
    // Crops legitimately change the aspect ratio and must not be reconciled.
    let cropped = metadata(
        RenderVariant::FaceCrop {
            face_id: 1,
            face_revision: 1,
            left_millionths: 0,
            top_millionths: 0,
            right_millionths: 500_000,
            bottom_millionths: 1_000_000,
            width: 512,
            padding_percent: 0,
        },
        Some(6),
        (4032, 3024),
        (512, 512),
    );
    assert_eq!(cropped.display_source_dimensions(), (4032, 3024));
    // Developed renders never went through the scaled decoder and recipes may
    // crop, so their aspect deviation must not trigger a swap.
    let mut developed = metadata(preview, Some(6), (3024, 4032), (1512, 1512));
    developed.develop = Some(DevelopRenderMetadata {
        identity: RenderIdentity::default(),
        recipe: DevelopRecipe::default(),
        applied_actions: 0,
        fidelity_warnings: 0,
    });
    assert_eq!(developed.display_source_dimensions(), (3024, 4032));
}
