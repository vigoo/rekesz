use std::{
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use test_r::test;

use super::*;
use crate::{
    catalog::DevelopRecipe,
    rendering::{RenderMode, pipeline::RenderInvocation},
};

#[test]
fn developed_renderer_rejects_mismatched_recipe_digest() {
    let directory = tempfile::tempdir().unwrap();
    let renderer =
        PreviewRenderer::with_source_cache_capacity(directory.path().join("cache"), 2, 1, 2);
    let recipe = DevelopRecipe {
        exposure_ev: Some(1.0),
        ..DevelopRecipe::default()
    };
    let result = renderer.render_with_recipe(
        RenderInvocation {
            path: directory.path().join("missing.jpg"),
            variant: RenderVariant::Full,
            render_identity: RenderIdentity {
                mode: RenderMode::Developed,
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
fn renderer_reuses_decoded_cr3_source() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test/IMG_0052.CR3");
    if !path.exists() {
        return;
    }
    let cache_root = tempfile::tempdir().unwrap();
    let renderer =
        PreviewRenderer::with_source_cache_capacity(cache_root.path().to_path_buf(), 2, 2, 1);
    for variant in [
        RenderVariant::Thumbnail { width: 64 },
        RenderVariant::Preview {
            max_width: 320,
            max_height: 240,
        },
    ] {
        let result = renderer.render(
            path.clone(),
            variant,
            true,
            false,
            &Arc::new(AtomicBool::new(false)),
        );
        assert!(matches!(result, PreviewState::Loaded(_)));
    }
    assert_eq!(renderer.stats().decoded_source_hits, 1);
}

#[test]
fn renderer_propagates_rawler_cancellation() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test/IMG_0052.CR3");
    if !path.exists() {
        return;
    }
    let cache_root = tempfile::tempdir().unwrap();
    let renderer =
        PreviewRenderer::with_source_cache_capacity(cache_root.path().to_path_buf(), 2, 2, 1);
    let cancellation = Arc::new(AtomicBool::new(false));
    let worker_cancellation = cancellation.clone();
    let worker = std::thread::spawn(move || {
        renderer.render(
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
fn developed_cr3_metadata_reaches_preview_state() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test/IMG_0052.CR3");
    if !path.exists() {
        return;
    }
    let cache_root = tempfile::tempdir().unwrap();
    let renderer =
        PreviewRenderer::with_source_cache_capacity(cache_root.path().to_path_buf(), 2, 1, 1);
    let recipe = DevelopRecipe {
        shadows: Some(25.0),
        ..DevelopRecipe::default()
    };
    let identity = RenderIdentity {
        mode: RenderMode::Developed,
        recipe_digest: recipe.digest().unwrap(),
        renderer_model: crate::catalog::RENDERER_MODEL_VERSION,
    };
    let result = renderer.render_with_recipe(
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
    assert_ne!(metadata.applied_actions, 0);
    assert_ne!(metadata.fidelity_warnings, 0);
}
