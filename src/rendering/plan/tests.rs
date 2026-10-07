use super::*;
use crate::catalog::{
    Crop, DetailRecipe, GrainRecipe, HealKind, HealSpot, MonochromeRecipe, VignetteRecipe,
    WhiteBalance, WhiteBalanceIncrement,
};
use test_r::test;

#[test]
fn raw_plan_partitions_controls_without_double_application() {
    let recipe = DevelopRecipe {
        exposure_ev: Some(1.0),
        white_balance: Some(WhiteBalance {
            temperature: 6_500.0,
            tint: 2.0,
        }),
        clarity: Some(10.0),
        crop: Some(Crop {
            left: 0.1,
            top: 0.1,
            right: 0.9,
            bottom: 0.9,
            angle_degrees: 0.0,
            constrain_aspect: false,
        }),
        ..DevelopRecipe::default()
    };
    let identity = RenderIdentity {
        mode: RenderMode::Developed,
        recipe_digest: recipe.digest().unwrap(),
        renderer_model: crate::catalog::RENDERER_MODEL_VERSION,
    };
    let plan = RenderPlan::compile(
        "/pictures/image.cr3".into(),
        RenderVariant::Full,
        identity,
        Some(&recipe),
    )
    .unwrap();

    assert_eq!(plan.raw_recipe().unwrap().exposure_ev, Some(1.0));
    assert_eq!(plan.post_recipe().unwrap().exposure_ev, None);
    assert_eq!(plan.post_recipe().unwrap().white_balance, None);
    // Clarity is raw-owned since the scene-linear presence stage.
    assert_eq!(plan.post_recipe().unwrap().clarity, None);
    assert_eq!(plan.raw_recipe().unwrap().clarity, Some(10.0));
    assert!(plan.post_recipe().unwrap().crop.is_some());
    assert_ne!(plan.raw_metadata().0 & develop::actions::EXPOSURE, 0);
}

#[test]
fn raw_partition_classifies_every_develop_recipe_control() {
    let recipe = DevelopRecipe {
        exposure_ev: Some(1.0),
        contrast: Some(1.0),
        highlights: Some(1.0),
        shadows: Some(1.0),
        whites: Some(1.0),
        blacks: Some(1.0),
        saturation: Some(1.0),
        vibrance: Some(1.0),
        clarity: Some(1.0),
        white_balance: Some(WhiteBalance {
            temperature: 6_500.0,
            tint: 1.0,
        }),
        white_balance_increment: Some(WhiteBalanceIncrement {
            temperature: 1.0,
            tint: 1.0,
        }),
        tone_curve: Some(vec![[0.0, 0.0], [1.0, 1.0]]),
        tone_curve_red: Some(vec![[0.0, 0.0], [1.0, 1.0]]),
        tone_curve_green: Some(vec![[0.0, 0.0], [1.0, 1.0]]),
        tone_curve_blue: Some(vec![[0.0, 0.0], [1.0, 1.0]]),
        crop: Some(Crop {
            left: 0.0,
            top: 0.0,
            right: 1.0,
            bottom: 1.0,
            angle_degrees: 0.0,
            constrain_aspect: false,
        }),
        detail: Some(DetailRecipe::default()),
        texture: Some(1.0),
        dehaze: Some(1.0),
        monochrome: Some(MonochromeRecipe { mixer: [1.0; 8] }),
        vignette: Some(VignetteRecipe {
            amount: 1.0,
            midpoint: 50.0,
            feather: 50.0,
            roundness: 0.0,
        }),
        parametric_curve: Some([1.0; 4]),
        upright_homography: Some([1.0; 9]),
        lens_distortion: Some(1.0),
        heal_spots: vec![HealSpot {
            kind: HealKind::Heal,
            destination: [0.5; 2],
            source: [0.4; 2],
            radius: 0.1,
            opacity: 1.0,
            feather: 0.5,
        }],
        grain: Some(GrainRecipe {
            amount: 1.0,
            size: 1.0,
            roughness: 1.0,
            seed: 1,
        }),
    };

    let partition = partition_raw_recipe(&recipe);
    let post = partition.post_recipe;
    assert_eq!(post.exposure_ev, None);
    assert_eq!(post.contrast, None);
    assert_eq!(post.highlights, None);
    assert_eq!(post.shadows, None);
    assert_eq!(post.whites, None);
    assert_eq!(post.blacks, None);
    assert_eq!(post.saturation, None);
    assert_eq!(post.vibrance, None);
    assert_eq!(post.white_balance, None);
    assert_eq!(post.white_balance_increment, None);
    assert_eq!(post.tone_curve, None);
    assert_eq!(post.tone_curve_red, None);
    assert_eq!(post.tone_curve_green, None);
    assert_eq!(post.tone_curve_blue, None);
    assert_eq!(post.detail, None);
    assert_eq!(post.texture, None);
    assert_eq!(post.dehaze, None);
    assert_eq!(post.monochrome, None);
    assert_eq!(post.parametric_curve, None);

    assert_eq!(post.clarity, None);
    assert_eq!(post.crop, recipe.crop);
    assert_eq!(post.vignette, recipe.vignette);
    assert_eq!(post.upright_homography, recipe.upright_homography);
    assert_eq!(post.lens_distortion, recipe.lens_distortion);
    assert_eq!(post.heal_spots, recipe.heal_spots);
    assert_eq!(post.grain, recipe.grain);
    assert_eq!(
        partition.applied_actions,
        develop::actions::EXPOSURE
            | develop::actions::WHITE_BALANCE
            | develop::actions::BASIC_TONE
            | develop::actions::COLOR
            | develop::actions::TONE_CURVE
            | develop::actions::CLARITY
            | develop::actions::DETAIL
            | develop::actions::TEXTURE_DEHAZE
            | develop::actions::MONOCHROME
            | develop::actions::PARAMETRIC_CURVE
    );
    assert_eq!(partition.fidelity_warnings, develop::warnings::APPROXIMATED);
}

#[test]
fn every_raw_owned_group_reports_its_action_and_approximation() {
    let curve = Some(vec![[0.0, 0.1], [1.0, 0.9]]);
    let cases = [
        (
            DevelopRecipe {
                exposure_ev: Some(1.0),
                ..Default::default()
            },
            develop::actions::EXPOSURE,
            false,
        ),
        (
            DevelopRecipe {
                white_balance: Some(WhiteBalance {
                    temperature: 6_500.0,
                    tint: 2.0,
                }),
                ..Default::default()
            },
            develop::actions::WHITE_BALANCE,
            true,
        ),
        (
            DevelopRecipe {
                contrast: Some(10.0),
                ..Default::default()
            },
            develop::actions::BASIC_TONE,
            true,
        ),
        (
            DevelopRecipe {
                saturation: Some(10.0),
                ..Default::default()
            },
            develop::actions::COLOR,
            true,
        ),
        (
            DevelopRecipe {
                tone_curve: curve.clone(),
                ..Default::default()
            },
            develop::actions::TONE_CURVE,
            true,
        ),
        (
            DevelopRecipe {
                tone_curve_red: curve,
                ..Default::default()
            },
            develop::actions::TONE_CURVE,
            true,
        ),
        (
            DevelopRecipe {
                detail: Some(DetailRecipe {
                    sharpening: 10.0,
                    ..Default::default()
                }),
                ..Default::default()
            },
            develop::actions::DETAIL,
            true,
        ),
        (
            DevelopRecipe {
                texture: Some(10.0),
                dehaze: Some(10.0),
                ..Default::default()
            },
            develop::actions::TEXTURE_DEHAZE,
            true,
        ),
        (
            DevelopRecipe {
                monochrome: Some(MonochromeRecipe { mixer: [0.0; 8] }),
                ..Default::default()
            },
            develop::actions::MONOCHROME,
            true,
        ),
        (
            DevelopRecipe {
                parametric_curve: Some([1.0; 4]),
                ..Default::default()
            },
            develop::actions::PARAMETRIC_CURVE,
            true,
        ),
    ];
    for (recipe, action, approximated) in cases {
        let partition = partition_raw_recipe(&recipe);
        assert_eq!(partition.applied_actions, action);
        assert_eq!(
            partition.fidelity_warnings != 0,
            approximated,
            "unexpected fidelity classification for action {action}"
        );
    }
}

#[test]
fn plan_has_stable_processing_order() {
    let plan = RenderPlan::compile(
        "/pictures/image.cr3".into(),
        RenderVariant::Thumbnail { width: 300 },
        RenderIdentity::default(),
        None,
    )
    .unwrap();
    assert_eq!(
        plan.stages(),
        &[
            RenderStage::CacheLookup,
            RenderStage::Decode,
            RenderStage::RawPhotographic,
            RenderStage::Resize,
            RenderStage::CacheWriteEligible,
        ]
    );
}

#[test]
fn stage_selection_matches_source_mode_and_variant_boundaries() {
    let baseline_full = RenderPlan::compile(
        "image.png".into(),
        RenderVariant::Full,
        RenderIdentity::default(),
        None,
    )
    .unwrap();
    assert_eq!(
        baseline_full.stages(),
        &[RenderStage::CacheLookup, RenderStage::Decode]
    );

    let recipe = DevelopRecipe::default();
    let developed_identity = RenderIdentity {
        mode: RenderMode::Developed,
        recipe_digest: recipe.digest().unwrap(),
        renderer_model: crate::catalog::RENDERER_MODEL_VERSION,
    };
    let standard_developed = RenderPlan::compile(
        "image.jpg".into(),
        RenderVariant::Preview {
            max_width: 1_200,
            max_height: 800,
        },
        developed_identity,
        Some(&recipe),
    )
    .unwrap();
    assert_eq!(
        standard_developed.stages(),
        &[
            RenderStage::CacheLookup,
            RenderStage::Decode,
            RenderStage::PostDevelop,
            RenderStage::Resize,
            RenderStage::CacheWriteEligible,
        ]
    );

    let raw_developed = RenderPlan::compile(
        "image.nef".into(),
        RenderVariant::Full,
        developed_identity,
        Some(&recipe),
    )
    .unwrap();
    assert_eq!(
        raw_developed.stages(),
        &[
            RenderStage::CacheLookup,
            RenderStage::Decode,
            RenderStage::RawPhotographic,
            RenderStage::PostDevelop,
        ]
    );
}

#[test]
fn decode_routing_is_explicit_and_case_insensitive() {
    let cases = [
        ("image.heic", DecodePlan::Heif, false),
        ("image.HEIF", DecodePlan::Heif, false),
        ("image.cr3", DecodePlan::Raw, true),
        ("image.NEF", DecodePlan::Raw, true),
        (
            "image.jpg",
            DecodePlan::Standard { scaled_jpeg: true },
            false,
        ),
        (
            "image.JPEG",
            DecodePlan::Standard { scaled_jpeg: true },
            false,
        ),
        (
            "image.png",
            DecodePlan::Standard { scaled_jpeg: false },
            false,
        ),
        ("image", DecodePlan::Standard { scaled_jpeg: false }, false),
    ];
    for (path, expected, raw) in cases {
        let plan = RenderPlan::compile(
            path.into(),
            RenderVariant::Thumbnail { width: 300 },
            RenderIdentity::default(),
            None,
        )
        .unwrap();
        assert_eq!(plan.decode(), expected, "{path}");
        assert_eq!(plan.is_raw_decode(), raw, "{path}");
        assert_eq!(plan.caches_decoded_source(), raw, "{path}");
    }
}

#[test]
fn full_and_developed_jpegs_do_not_use_scaled_decode() {
    let full = RenderPlan::compile(
        "image.jpg".into(),
        RenderVariant::Full,
        RenderIdentity::default(),
        None,
    )
    .unwrap();
    assert_eq!(full.decode(), DecodePlan::Standard { scaled_jpeg: false });

    let recipe = DevelopRecipe::default();
    let identity = RenderIdentity {
        mode: RenderMode::Developed,
        recipe_digest: recipe.digest().unwrap(),
        renderer_model: crate::catalog::RENDERER_MODEL_VERSION,
    };
    let developed = RenderPlan::compile(
        "image.jpg".into(),
        RenderVariant::Thumbnail { width: 300 },
        identity,
        Some(&recipe),
    )
    .unwrap();
    assert_eq!(
        developed.decode(),
        DecodePlan::Standard { scaled_jpeg: false }
    );
}

#[test]
fn resize_decision_preserves_baseline_optimization_and_developed_resolution() {
    let baseline = RenderPlan::compile(
        "image.png".into(),
        RenderVariant::Preview {
            max_width: 1_200,
            max_height: 800,
        },
        RenderIdentity::default(),
        None,
    )
    .unwrap();
    assert_eq!(
        baseline.resize(),
        ResizePlan::Derivative {
            max_width: 1_200,
            max_height: 800,
            before_develop: true,
        }
    );

    let recipe = DevelopRecipe::default();
    let identity = RenderIdentity {
        mode: RenderMode::Developed,
        recipe_digest: recipe.digest().unwrap(),
        renderer_model: crate::catalog::RENDERER_MODEL_VERSION,
    };
    let developed = RenderPlan::compile(
        "image.png".into(),
        RenderVariant::Thumbnail { width: 300 },
        identity,
        Some(&recipe),
    )
    .unwrap();
    assert_eq!(
        developed.resize(),
        ResizePlan::Derivative {
            max_width: 300,
            max_height: 300,
            before_develop: false,
        }
    );

    let full = RenderPlan::compile(
        "image.png".into(),
        RenderVariant::Full,
        RenderIdentity::default(),
        None,
    )
    .unwrap();
    assert_eq!(full.resize(), ResizePlan::None);
}

#[test]
fn developed_plan_validates_recipe_identity() {
    let identity = RenderIdentity {
        mode: RenderMode::Developed,
        recipe_digest: [7; 32],
        renderer_model: crate::catalog::RENDERER_MODEL_VERSION,
    };
    assert!(
        RenderPlan::compile(
            "/pictures/image.jpg".into(),
            RenderVariant::Full,
            identity,
            Some(&DevelopRecipe::default()),
        )
        .is_err()
    );
}

#[test]
fn developed_plan_rejects_digest_matching_invalid_recipe() {
    let recipe = DevelopRecipe {
        contrast: Some(500.0),
        ..DevelopRecipe::default()
    };
    let identity = RenderIdentity {
        mode: RenderMode::Developed,
        recipe_digest: recipe.digest().unwrap(),
        renderer_model: crate::catalog::RENDERER_MODEL_VERSION,
    };
    let error = RenderPlan::compile(
        "/pictures/image.jpg".into(),
        RenderVariant::Full,
        identity,
        Some(&recipe),
    )
    .err()
    .unwrap();
    assert!(error.contains("invalid developed recipe"));
}
