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
    let decoded = backend.decode_source(&request, &cancelled).unwrap();
    let baseline = decoded.image.to_rgb8().get_pixel(0, 0)[0];

    let rendered = backend
        .render_source(
            request.variant,
            identity,
            Some(&recipe),
            decoded,
            &cancelled,
        )
        .unwrap();

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
