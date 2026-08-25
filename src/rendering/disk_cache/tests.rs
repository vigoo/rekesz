use super::*;
use image::{ImageBuffer, Rgb};
use test_r::test;

#[test]
fn stores_and_loads_an_image_with_its_render_metadata() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("source.jpg");
    fs::write(&source, b"source identity").unwrap();
    let cache = DiskRenderCache::new(directory.path().join("cache"));
    let request = RenderRequest::new(
        source,
        super::super::RenderVariant::Thumbnail { width: 300 },
    );
    let rendered = RenderedImage {
        image: DynamicImage::ImageRgb8(ImageBuffer::from_pixel(2, 1, Rgb([1, 2, 3]))),
        metadata: RenderMetadata {
            source_pipeline: super::super::SourcePipeline::Standard,
            photographic: None,
            source_dimensions: (2, 1),
            output_dimensions: (2, 1),
            variant: request.variant,
            orientation_applied: None,
            develop: None,
        },
    };
    let settings = PhotographicSettings::default();
    let identity = source_identity(&request).unwrap().unwrap();
    cache
        .store(&request, &identity, &settings, &rendered)
        .unwrap();
    let loaded = cache.load(&request, &identity, &settings).unwrap().unwrap();
    assert_eq!(loaded.image.width(), 2);
    assert_eq!(loaded.metadata, rendered.metadata);
}

#[test]
fn phase_d_recipe_and_action_metadata_round_trip() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("source.jpg");
    fs::write(&source, b"source identity").unwrap();
    let cache = DiskRenderCache::new(directory.path().join("cache"));
    let recipe = crate::catalog::DevelopRecipe {
        upright_homography: Some([1.0, 0.0, 0.05, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
        lens_distortion: Some(10.0),
        heal_spots: vec![crate::catalog::HealSpot {
            kind: crate::catalog::HealKind::Heal,
            destination: [1.0, 1.0],
            source: [0.0, 0.0],
            radius: 0.1,
            opacity: 0.8,
            feather: 0.25,
        }],
        grain: Some(crate::catalog::GrainRecipe {
            amount: 20.0,
            size: 30.0,
            roughness: 50.0,
            seed: 42,
        }),
        ..crate::catalog::DevelopRecipe::default()
    };
    let identity = super::super::RenderIdentity {
        mode: super::super::RenderMode::Developed,
        recipe_digest: recipe.digest().unwrap(),
        renderer_model: crate::catalog::RENDERER_MODEL_VERSION,
    };
    let request = RenderRequest::with_identity(
        source,
        super::super::RenderVariant::Thumbnail { width: 300 },
        identity,
    );
    let phase_d_actions = super::super::develop::actions::HEAL
        | super::super::develop::actions::LENS
        | super::super::develop::actions::UPRIGHT
        | super::super::develop::actions::GRAIN;
    let rendered = RenderedImage {
        image: DynamicImage::ImageRgb8(ImageBuffer::from_pixel(2, 1, Rgb([1, 2, 3]))),
        metadata: RenderMetadata {
            source_pipeline: super::super::SourcePipeline::Standard,
            photographic: None,
            source_dimensions: (2, 1),
            output_dimensions: (2, 1),
            variant: request.variant,
            orientation_applied: None,
            develop: Some(super::super::DevelopRenderMetadata {
                identity,
                recipe,
                applied_actions: phase_d_actions,
                fidelity_warnings: super::super::develop::warnings::APPROXIMATED,
            }),
        },
    };
    let settings = PhotographicSettings::default();
    let source_identity = source_identity(&request).unwrap().unwrap();
    cache
        .store(&request, &source_identity, &settings, &rendered)
        .unwrap();
    let loaded = cache
        .load(&request, &source_identity, &settings)
        .unwrap()
        .unwrap();
    assert_eq!(loaded.metadata, rendered.metadata);
    assert_eq!(
        loaded.metadata.develop.unwrap().applied_actions,
        phase_d_actions
    );
}

#[test]
fn recipe_and_mode_are_part_of_the_cache_key() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("source.jpg");
    fs::write(&source, b"source").unwrap();
    let baseline = RenderRequest::new(
        source.clone(),
        super::super::RenderVariant::Thumbnail { width: 300 },
    );
    let developed = RenderRequest::with_identity(
        source,
        baseline.variant,
        super::super::RenderIdentity {
            mode: super::super::RenderMode::Developed,
            recipe_digest: [7; 32],
            renderer_model: 1,
        },
    );
    let identity = source_identity(&baseline).unwrap().unwrap();
    let settings = PhotographicSettings::default();
    assert_ne!(
        cache_key(&baseline, &identity, &settings).unwrap(),
        cache_key(&developed, &identity, &settings).unwrap()
    );
}

#[test]
fn cleanup_removes_the_least_recently_accessed_entry() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("cache");
    let cache = DiskRenderCache::new(root.clone());
    let settings = PhotographicSettings::default();
    let rendered = RenderedImage {
        image: DynamicImage::ImageRgb8(ImageBuffer::from_pixel(8, 8, Rgb([1, 2, 3]))),
        metadata: RenderMetadata {
            source_pipeline: super::super::SourcePipeline::Standard,
            photographic: None,
            source_dimensions: (8, 8),
            output_dimensions: (8, 8),
            variant: super::super::RenderVariant::Thumbnail { width: 300 },
            orientation_applied: None,
            develop: None,
        },
    };
    let first = directory.path().join("first.jpg");
    let second = directory.path().join("second.jpg");
    fs::write(&first, b"first").unwrap();
    fs::write(&second, b"second").unwrap();
    let first_request = RenderRequest::new(first, rendered.metadata.variant);
    let second_request = RenderRequest::new(second, rendered.metadata.variant);
    let first_identity = source_identity(&first_request).unwrap().unwrap();
    let second_identity = source_identity(&second_request).unwrap().unwrap();
    cache
        .store(&first_request, &first_identity, &settings, &rendered)
        .unwrap();
    std::thread::sleep(std::time::Duration::from_millis(2));
    cache
        .store(&second_request, &second_identity, &settings, &rendered)
        .unwrap();
    std::thread::sleep(std::time::Duration::from_millis(2));
    cache
        .load(&first_request, &first_identity, &settings)
        .unwrap()
        .unwrap();
    let first_key = cache_key(&first_request, &first_identity, &settings).unwrap();
    let second_key = cache_key(&second_request, &second_identity, &settings).unwrap();
    let entry_size = |key: &str| {
        ["png", "json", "access"]
            .into_iter()
            .map(|extension| {
                fs::metadata(root.join(format!("{key}.{extension}")))
                    .unwrap()
                    .len()
            })
            .sum::<u64>()
    };
    let first_size = entry_size(&first_key);
    let second_size = entry_size(&second_key);
    let constrained = DiskRenderCache {
        root: root.clone(),
        max_bytes: first_size + second_size - 1,
        lock: cache.lock.clone(),
        stores_since_cleanup: Arc::new(AtomicU64::new(0)),
    };
    constrained.cleanup_lru().unwrap();

    assert!(root.join(format!("{first_key}.png")).exists());
    assert!(!root.join(format!("{second_key}.png")).exists());
}
