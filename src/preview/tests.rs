use super::*;
use image::{ImageBuffer, Rgba};
use std::{path::Path, sync::mpsc, time::Duration};
use test_r::test;

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

#[test]
fn a_larger_raw_preview_satisfies_a_smaller_display_request() {
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

fn load(cache: &mut PreviewCache, path: &std::path::Path) {
    let request = cache.begin(path, RenderVariant::Full).unwrap();
    assert!(cache.finish(path, request, image_state()));
}

#[test]
fn evicts_the_least_recently_used_loaded_image() {
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
fn cheap_negative_results_do_not_consume_image_capacity() {
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
fn clearing_cache_rejects_a_completion_from_the_previous_mode() {
    let mut cache = PreviewCache::new(1);
    let path = Path::new("image");
    let old_request = cache.begin(path, RenderVariant::Full).unwrap();
    cache.clear();
    let new_request = cache.begin(path, RenderVariant::Full).unwrap();

    assert!(!cache.finish(path, old_request, image_state()));
    assert!(matches!(cache.state(path), PreviewState::Loading(id) if id == new_request));
}

#[test]
fn baseline_and_developed_entries_coexist_in_memory() {
    let mut cache = PreviewCache::new(2);
    let path = Path::new("image");
    let developed = RenderIdentity {
        mode: crate::rendering::RenderMode::Developed,
        recipe_digest: [7; 32],
        renderer_model: crate::catalog::RENDERER_MODEL_VERSION,
    };
    let baseline_request = cache
        .begin_for(path, RenderVariant::Full, RenderIdentity::default())
        .unwrap();
    cache.finish_for(
        path,
        RenderIdentity::default(),
        baseline_request,
        image_state(),
    );
    let developed_request = cache
        .begin_for(path, RenderVariant::Full, developed)
        .unwrap();
    cache.finish_for(path, developed, developed_request, image_state());

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
fn developed_pipeline_rejects_a_mismatched_recipe_digest() {
    let directory = tempfile::tempdir().unwrap();
    let pipeline =
        RenderPipeline::with_source_cache_capacity(directory.path().join("cache"), 2, 1, 2);
    let recipe = DevelopRecipe {
        exposure_ev: Some(1.0),
        ..DevelopRecipe::default()
    };
    let result = pipeline.render_with_recipe(
        RenderInvocation {
            path: directory.path().join("missing.jpg"),
            variant: RenderVariant::Full,
            render_identity: RenderIdentity {
                mode: crate::rendering::RenderMode::Developed,
                recipe_digest: [9; 32],
                renderer_model: crate::catalog::RENDERER_MODEL_VERSION,
            },
            recipe: Some(&recipe),
            interactive: true,
            durable_cache_write: false,
        },
        &Arc::new(AtomicBool::new(false)),
    );

    assert!(matches!(result, PreviewState::Error(message) if message.contains("digest mismatch")));
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
fn cancelling_a_matching_request_restores_unloaded_state() {
    let mut cache = PreviewCache::new(1);
    let path = Path::new("cancelled");
    let request = cache.begin(path, RenderVariant::Full).unwrap();
    assert!(cache.cancel(path, request));
    assert!(matches!(cache.state(path), PreviewState::Unloaded));
    assert!(!cache.cancel(path, request));
}

#[test]
fn interactive_permits_bypass_the_background_reserve() {
    let pool = Arc::new(PermitPool::new(4, 2));
    let cancellation = AtomicBool::new(false);
    let background = pool.acquire(2, false, &cancellation).unwrap();
    let (sender, receiver) = mpsc::channel();
    let contender = {
        let pool = pool.clone();
        std::thread::spawn(move || {
            let cancellation = AtomicBool::new(false);
            let _permit = pool.acquire(2, false, &cancellation).unwrap();
            sender.send("background").unwrap();
        })
    };
    assert!(receiver.recv_timeout(Duration::from_millis(50)).is_err());
    let interactive = pool.acquire(2, true, &cancellation).unwrap();
    drop(interactive);
    assert!(receiver.recv_timeout(Duration::from_millis(50)).is_err());
    drop(background);
    assert_eq!(
        receiver.recv_timeout(Duration::from_secs(1)).unwrap(),
        "background"
    );
    contender.join().unwrap();
}

#[test]
fn permit_wait_is_cooperatively_cancellable() {
    let pool = Arc::new(PermitPool::new(1, 0));
    let cancellation = Arc::new(AtomicBool::new(false));
    let held = pool.acquire(1, false, &cancellation).unwrap();
    let waiter = {
        let pool = pool.clone();
        let cancellation = cancellation.clone();
        std::thread::spawn(move || pool.acquire(1, false, &cancellation).is_none())
    };
    std::thread::sleep(Duration::from_millis(30));
    cancellation.store(true, Ordering::Release);
    assert!(waiter.join().unwrap());
    drop(held);
}

#[test]
fn durable_background_writes_wait_for_bounded_queue_capacity() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("source.png");
    std::fs::write(&source, b"identity").unwrap();
    let backend = RenderBackend::new(directory.path().join("cache"));
    let request = RenderRequest::new(source, RenderVariant::Thumbnail { width: 4 });
    let identity = backend.source_identity(&request).unwrap().unwrap();
    let rendered = Arc::new(RenderedImage {
        image: DynamicImage::ImageRgba8(ImageBuffer::from_pixel(1, 1, Rgba([1, 2, 3, 255]))),
        metadata: RenderMetadata {
            source_pipeline: crate::rendering::SourcePipeline::Standard,
            photographic: None,
            source_dimensions: (1, 1),
            output_dimensions: (1, 1),
            variant: request.variant,
            orientation_applied: None,
            develop: None,
        },
    });
    let make_write = || CacheWrite {
        request: request.clone(),
        identity: identity.clone(),
        rendered: rendered.clone(),
    };
    let (sender, receiver) = mpsc::sync_channel(1);
    assert!(sender.try_send(make_write()).is_ok());
    let writer = Arc::new(CacheWriter {
        sender: Mutex::new(Some(sender)),
        worker: Mutex::new(None),
        metrics: Arc::new(PipelineMetrics::default()),
    });
    let cancellation = Arc::new(AtomicBool::new(false));
    let waiter = {
        let writer = writer.clone();
        let cancellation = cancellation.clone();
        let write = make_write();
        std::thread::spawn(move || writer.write(write, true, &cancellation))
    };
    std::thread::sleep(Duration::from_millis(30));
    assert!(!waiter.is_finished());
    receiver.recv().unwrap();
    waiter.join().unwrap();
    receiver.recv_timeout(Duration::from_secs(1)).unwrap();
}

#[test]
fn staged_pipeline_persists_a_generated_thumbnail() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("source.png");
    DynamicImage::ImageRgba8(ImageBuffer::from_pixel(8, 8, Rgba([1, 2, 3, 255])))
        .save(&source)
        .unwrap();
    let pipeline =
        RenderPipeline::with_source_cache_capacity(directory.path().join("cache"), 2, 2, 1);
    let variant = RenderVariant::Thumbnail { width: 4 };
    assert!(matches!(
        pipeline.render(
            source.clone(),
            variant,
            true,
            false,
            &Arc::new(AtomicBool::new(false))
        ),
        PreviewState::Loaded(_)
    ));
    assert_eq!(pipeline.stats().cache_misses, 1);
    let request = RenderRequest::new(source, variant);
    let identity = pipeline.backend.source_identity(&request).unwrap().unwrap();
    let deadline = std::time::Instant::now() + Duration::from_secs(2);
    while std::time::Instant::now() < deadline {
        if pipeline
            .backend
            .lookup_cache(&request, Some(&identity))
            .unwrap()
            .is_some()
        {
            return;
        }
        std::thread::yield_now();
    }
    panic!("deferred cache write did not become visible");
}

#[test]
fn decodes_the_cr3_regression_sample() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test/IMG_0052.CR3");
    if path.exists() {
        let cache_root = tempfile::tempdir().unwrap();
        let pipeline =
            RenderPipeline::with_source_cache_capacity(cache_root.path().to_path_buf(), 2, 2, 1);
        let result = pipeline.render(
            path.clone(),
            RenderVariant::Thumbnail { width: 64 },
            true,
            false,
            &Arc::new(AtomicBool::new(false)),
        );
        if let PreviewState::Error(error) = result {
            panic!("CR3 sample did not decode: {error}");
        }
        let result = pipeline.render(
            path,
            RenderVariant::Preview {
                max_width: 320,
                max_height: 240,
            },
            true,
            false,
            &Arc::new(AtomicBool::new(false)),
        );
        assert!(matches!(result, PreviewState::Loaded(_)));
        assert_eq!(pipeline.stats().decoded_source_hits, 1);
    }
}

#[test]
fn cancels_cr3_work_through_the_rawler_token() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test/IMG_0052.CR3");
    if !path.exists() {
        return;
    }
    let cache_root = tempfile::tempdir().unwrap();
    let pipeline =
        RenderPipeline::with_source_cache_capacity(cache_root.path().to_path_buf(), 2, 2, 1);
    let cancellation = Arc::new(AtomicBool::new(false));
    let worker_cancellation = cancellation.clone();
    let worker = std::thread::spawn(move || {
        pipeline.render(
            path,
            RenderVariant::Thumbnail { width: 64 },
            true,
            false,
            &worker_cancellation,
        )
    });
    std::thread::sleep(Duration::from_millis(10));
    cancellation.store(true, Ordering::Release);
    assert!(matches!(worker.join().unwrap(), PreviewState::Error(_)));
}

#[test]
fn cr3_develop_metadata_records_actions_parameters_and_approximation() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test/IMG_0052.CR3");
    if !path.exists() {
        return;
    }
    let cache_root = tempfile::tempdir().unwrap();
    let pipeline =
        RenderPipeline::with_source_cache_capacity(cache_root.path().to_path_buf(), 2, 1, 1);
    let recipe = DevelopRecipe {
        shadows: Some(25.0),
        ..DevelopRecipe::default()
    };
    let identity = RenderIdentity {
        mode: crate::rendering::RenderMode::Developed,
        recipe_digest: recipe.digest().unwrap(),
        renderer_model: crate::catalog::RENDERER_MODEL_VERSION,
    };
    let result = pipeline.render_with_recipe(
        RenderInvocation {
            path,
            variant: RenderVariant::Thumbnail { width: 64 },
            render_identity: identity,
            recipe: Some(&recipe),
            interactive: true,
            durable_cache_write: false,
        },
        &Arc::new(AtomicBool::new(false)),
    );
    let PreviewState::Loaded(preview) = result else {
        panic!("developed CR3 did not render");
    };
    let metadata = preview.metadata.develop.as_ref().unwrap();
    assert_eq!(metadata.recipe, recipe);
    assert_ne!(
        metadata.applied_actions & crate::rendering::develop::actions::BASIC_TONE,
        0
    );
    assert_ne!(
        metadata.fidelity_warnings & crate::rendering::develop::warnings::APPROXIMATED,
        0
    );
}
