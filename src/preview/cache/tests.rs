use std::{path::Path, sync::Arc};

use gpui::RenderImage;
use image::{Frame, ImageBuffer, Rgba};
use test_r::test;

use super::*;

fn image_state() -> PreviewState {
    image_state_for(RenderVariant::Full)
}

fn image_state_for(variant: RenderVariant) -> PreviewState {
    let buffer = ImageBuffer::from_pixel(1, 1, Rgba([0, 0, 0, 255]));
    PreviewState::Loaded(Arc::new(DecodedPreview {
        image: Arc::new(RenderImage::new(smallvec::smallvec![Frame::new(buffer)])),
        metadata: RenderMetadata {
            source_pipeline: crate::rendering::SourcePipeline::Standard,
            photographic: None,
            source_dimensions: (1, 1),
            output_dimensions: (1, 1),
            variant,
            orientation_applied: None,
            develop: None,
        },
    }))
}

fn load(cache: &mut PreviewCache, path: &Path) {
    let request = cache.begin(path, RenderVariant::Full).unwrap();
    assert!(cache.finish(path, request, image_state()));
}

#[test]
fn larger_preview_satisfies_smaller_request() {
    let mut cache = PreviewCache::new(1);
    let path = Path::new("raw.cr3");
    let large = RenderVariant::Preview {
        max_width: 3840,
        max_height: 2160,
    };
    let request = cache.begin(path, large).unwrap();
    cache.finish(path, request, image_state_for(large));
    assert!(
        cache
            .begin(
                path,
                RenderVariant::Preview {
                    max_width: 1920,
                    max_height: 1080,
                }
            )
            .is_none()
    );
    assert!(
        cache
            .begin(
                path,
                RenderVariant::Preview {
                    max_width: 5120,
                    max_height: 2880,
                }
            )
            .is_some()
    );
}

#[test]
fn evicts_least_recently_used_loaded_image() {
    let mut cache = PreviewCache::new(2);
    let first = Path::new("first");
    let second = Path::new("second");
    let third = Path::new("third");
    load(&mut cache, first);
    load(&mut cache, second);
    cache.mark_used(first);
    load(&mut cache, third);
    assert!(matches!(cache.state(first), PreviewState::Loaded(_)));
    assert!(matches!(cache.state(second), PreviewState::Unloaded));
    assert!(matches!(cache.state(third), PreviewState::Loaded(_)));
}

#[test]
fn negative_results_do_not_consume_image_capacity() {
    let mut cache = PreviewCache::new(1);
    let missing = Path::new("missing");
    let image = Path::new("image");
    let request = cache.begin(missing, RenderVariant::Full).unwrap();
    cache.finish(missing, request, PreviewState::Error("missing".into()));
    load(&mut cache, image);
    assert!(matches!(cache.state(missing), PreviewState::Error(_)));
    assert!(matches!(cache.state(image), PreviewState::Loaded(_)));
}

#[test]
fn clear_rejects_completion_from_previous_generation() {
    let mut cache = PreviewCache::new(1);
    let path = Path::new("image");
    let old_request = cache.begin(path, RenderVariant::Full).unwrap();
    cache.clear();
    let new_request = cache.begin(path, RenderVariant::Full).unwrap();
    assert!(!cache.finish(path, old_request, image_state()));
    assert!(matches!(cache.state(path), PreviewState::Loading(id) if id == new_request));
}

#[test]
fn baseline_and_developed_entries_coexist() {
    let mut cache = PreviewCache::new(2);
    let path = Path::new("image");
    let developed = RenderIdentity {
        mode: crate::rendering::RenderMode::Developed,
        recipe_digest: [7; 32],
        renderer_model: crate::catalog::RENDERER_MODEL_VERSION,
    };
    let baseline = cache
        .begin_for(path, RenderVariant::Full, RenderIdentity::default())
        .unwrap();
    cache.finish_for(path, RenderIdentity::default(), baseline, image_state());
    let request = cache
        .begin_for(path, RenderVariant::Full, developed)
        .unwrap();
    cache.finish_for(path, developed, request, image_state());
    assert!(matches!(
        cache.state_for(path, RenderIdentity::default()),
        PreviewState::Loaded(_)
    ));
    assert!(matches!(
        cache.state_for(path, developed),
        PreviewState::Loaded(_)
    ));
}

#[test]
fn completion_order_does_not_override_usage_order() {
    let mut cache = PreviewCache::new(1);
    let first = Path::new("first");
    let second = Path::new("second");
    let first_request = cache.begin(first, RenderVariant::Full).unwrap();
    let second_request = cache.begin(second, RenderVariant::Full).unwrap();
    cache.finish(second, second_request, image_state());
    cache.finish(first, first_request, image_state());
    assert!(matches!(cache.state(first), PreviewState::Unloaded));
    assert!(matches!(cache.state(second), PreviewState::Loaded(_)));
}

#[test]
fn cancelling_matching_request_restores_unloaded_state() {
    let mut cache = PreviewCache::new(1);
    let path = Path::new("cancelled");
    let request = cache.begin(path, RenderVariant::Full).unwrap();
    assert!(cache.cancel(path, request));
    assert!(matches!(cache.state(path), PreviewState::Unloaded));
    assert!(!cache.cancel(path, request));
}
