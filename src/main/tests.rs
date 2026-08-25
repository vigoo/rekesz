use super::*;
use discovery::DirectoryScan;
use futures::executor::block_on;
use std::time::Instant;
use test_r::test;

#[test]
fn converts_gpui_bgra_pixels_to_png_rgba_order() {
    let mut pixels = [3, 2, 1, 4, 30, 20, 10, 40];

    bgra_to_rgba(&mut pixels);

    assert_eq!(pixels, [1, 2, 3, 4, 10, 20, 30, 40]);
}

#[test]
fn export_dialog_remembers_the_last_chosen_directory() {
    let source = std::path::Path::new("/pictures/trip/image.nef");

    assert_eq!(
        export_dialog_directory(None, source),
        std::path::Path::new("/pictures/trip")
    );
    assert_eq!(
        export_dialog_directory(Some(std::path::Path::new("/exports/comparison")), source),
        std::path::Path::new("/exports/comparison")
    );
}

#[test]
fn full_screen_wheel_navigation_stops_at_directory_edges() {
    assert_eq!(adjacent_image_index(1, 3, -1), Some(0));
    assert_eq!(adjacent_image_index(1, 3, 1), Some(2));
    assert_eq!(adjacent_image_index(0, 3, -1), None);
    assert_eq!(adjacent_image_index(2, 3, 1), None);
    assert_eq!(adjacent_image_index(0, 0, 1), None);
}

#[test]
fn full_screen_prefetch_follows_the_navigation_direction() {
    assert_eq!(directional_adjacent_indices(2, 8, 1, 4), vec![3, 4, 5, 6]);
    assert_eq!(directional_adjacent_indices(4, 8, -1, 4), vec![3, 2, 1, 0]);
    assert_eq!(directional_adjacent_indices(6, 8, 1, 4), vec![7]);
    assert!(directional_adjacent_indices(0, 8, -1, 4).is_empty());
}

#[test]
fn progress_describes_the_running_render() {
    let path = PathBuf::from("/pictures/trip/image.nef");
    let snapshot = SchedulerSnapshot {
        jobs: vec![scheduler::JobSnapshot {
            key: JobKey {
                path,
                variant: RenderVariant::Preview {
                    max_width: 1_920,
                    max_height: 1_080,
                },
                render_identity: RenderIdentity {
                    mode: RenderMode::Developed,
                    ..RenderIdentity::default()
                },
                recipe: None,
            },
            priority: JobPriority::INTERACTIVE,
            status: JobSnapshotStatus::Running,
            scope_count: 1,
            sequence: 0,
            cancelled: false,
        }],
        queued: 0,
        running: 1,
    };

    assert_eq!(
        scheduler_progress_label(&snapshot).as_deref(),
        Some("Rendering developed preview: image.nef…")
    );

    let mut background_snapshot = snapshot;
    background_snapshot.jobs[0].priority = JobPriority::speculative(0);
    assert_eq!(
        scheduler_progress_label(&background_snapshot).as_deref(),
        Some("Pre-rendering developed preview: image.nef…")
    );
}

#[test]
fn work_progress_keeps_specific_activity_labels() {
    let mut progress = WorkProgress::default();
    let scan = progress.start("Scanning trip…");
    let metadata = progress.start("Reading metadata for 4 images…");

    assert_eq!(
        progress.current_activity(),
        Some("Reading metadata for 4 images…")
    );
    progress.finish(metadata);
    assert_eq!(progress.current_activity(), Some("Scanning trip…"));
    progress.finish(scan);
    assert_eq!(progress.current_activity(), None);
}

#[test]
#[ignore = "manual investigation using external reference samples"]
fn export_fixed_reference_renders() {
    let root = PathBuf::from("/mnt/pictures/Pictures");
    let recipes = catalog::load_recipes(&root).unwrap();
    let samples = [
        root.join("2020/Szlovénia/2020-07-29/DSC_0546.NEF"),
        root.join("2026/Alpok túra/2026-07-28/IMG_5639.HEIC"),
        root.join("2026/Alpok túra/2026-08-03/IMG_0997.CR3"),
        root.join("2026/Alpok túra/2026-08-03/IMG_1056.CR3"),
    ];
    let cache = tempfile::tempdir().unwrap();
    let pipeline = PreviewRenderer::with_source_cache_capacity(
        cache.path().to_path_buf(),
        1,
        1,
        samples.len(),
    );
    let cancelled = Arc::new(AtomicBool::new(false));
    for path in samples {
        let entry = recipes.get(&path).unwrap();
        let rendered = pipeline.render_with_recipe(
            RenderInvocation {
                path: path.clone(),
                variant: RenderVariant::Full,
                render_identity: RenderIdentity {
                    mode: RenderMode::Developed,
                    recipe_digest: entry.digest,
                    renderer_model: catalog::RENDERER_MODEL_VERSION,
                },
                recipe: Some(&entry.recipe),
                interactive: true,
                durable_cache_write: false,
            },
            &cancelled,
        );
        let stem = path.file_stem().unwrap().to_string_lossy();
        save_preview_as_png(
            rendered,
            &PathBuf::from(format!("/tmp/pics-fixed-{stem}.png")),
        )
        .unwrap();
    }
}

#[test]
fn developed_marker_is_independent_of_the_rendering_toggle() {
    let developed_path = PathBuf::from("/pictures/developed.CR3");
    let baseline_path = PathBuf::from("/pictures/baseline.CR3");
    let recipe = catalog::DevelopRecipe {
        exposure_ev: Some(1.0),
        ..catalog::DevelopRecipe::default()
    };
    let recipes = HashMap::from([(
        developed_path.clone(),
        catalog::RecipeEntry {
            digest: recipe.digest().unwrap(),
            recipe,
        },
    )]);

    assert!(has_develop_recipe(Some(&recipes), &developed_path));
    assert!(!has_develop_recipe(Some(&recipes), &baseline_path));
    assert!(!has_develop_recipe(None, &developed_path));
}

#[test]
fn develop_inspector_formats_active_recipe_groups() {
    let recipe = catalog::DevelopRecipe {
        exposure_ev: Some(1.25),
        white_balance: Some(catalog::WhiteBalance {
            temperature: 6_500.0,
            tint: 4.0,
        }),
        crop: Some(catalog::Crop {
            left: 0.1,
            top: 0.2,
            right: 0.9,
            bottom: 0.8,
            angle_degrees: 1.5,
            constrain_aspect: true,
        }),
        heal_spots: vec![catalog::HealSpot {
            kind: catalog::HealKind::Clone,
            destination: [0.5, 0.5],
            source: [0.25, 0.25],
            radius: 0.1,
            opacity: 0.8,
            feather: 0.2,
        }],
        grain: Some(catalog::GrainRecipe {
            amount: 15.0,
            size: 30.0,
            roughness: 50.0,
            seed: 42,
        }),
        ..catalog::DevelopRecipe::default()
    };
    let edits = format_develop_edits(&recipe);
    let labels = edits
        .iter()
        .map(|(label, _)| label.as_str())
        .collect::<Vec<_>>();

    assert!(labels.contains(&"Exposure"));
    assert!(labels.contains(&"White balance"));
    assert!(labels.contains(&"Crop"));
    assert!(labels.contains(&"Clone spot 1"));
    assert!(labels.contains(&"Grain"));
    assert!(edits.iter().any(|(_, value)| value.contains("6500 K")));
}

#[test]
fn collapsing_a_branch_removes_all_descendant_rows() {
    let mut model = DirectoryModel::new(PathBuf::from("/pictures"));
    let root = model.root();
    let (request, _) = model.begin_load(root).unwrap();
    model.finish_load(
        root,
        request,
        Ok(DirectoryScan {
            directories: vec![PathBuf::from("/pictures/a")],
            images: Vec::new(),
        }),
    );
    let child = model.loaded(root).unwrap().children[0];
    let (request, _) = model.begin_load(child).unwrap();
    model.finish_load(
        child,
        request,
        Ok(DirectoryScan {
            directories: vec![PathBuf::from("/pictures/a/b")],
            images: Vec::new(),
        }),
    );

    let mut expanded = HashSet::from([root, child]);
    assert_eq!(visible_rows(&model, &expanded).len(), 3);
    expanded.remove(&child);
    assert_eq!(visible_rows(&model, &expanded).len(), 2);
    assert_eq!(model.loaded(child).unwrap().children.len(), 1);
}

#[test]
fn parses_image_cache_size() {
    let config = config_from_args([
        OsString::from("pics"),
        OsString::from("--image-cache-size"),
        OsString::from("3"),
        OsString::from("/pictures"),
    ])
    .unwrap();
    assert_eq!(config.image_cache_size, 3);
    assert_eq!(config.root, PathBuf::from("/pictures"));
    assert_eq!(config.pair_preference, RawJpegPreference::Raw);
}

#[test]
fn parses_jpeg_pair_preference() {
    let config = config_from_args([
        OsString::from("pics"),
        OsString::from("--cr3-jpeg-preference"),
        OsString::from("jpeg"),
    ])
    .unwrap();
    assert_eq!(config.pair_preference, RawJpegPreference::Jpeg);
}

#[test]
fn parses_custom_cache_root() {
    let config = config_from_args([
        OsString::from("pics"),
        OsString::from("--cache-root"),
        OsString::from("relative-cache"),
    ])
    .unwrap();
    assert!(config.cache_root.is_absolute());
    assert!(config.cache_root.ends_with("relative-cache"));
}

#[test]
fn parses_job_parallelism() {
    let config = config_from_args([
        OsString::from("pics"),
        OsString::from("--job-parallelism"),
        OsString::from("3"),
    ])
    .unwrap();
    assert_eq!(config.job_parallelism, 3);
}

#[test]
fn parses_and_bounds_raw_parallelism() {
    let config = config_from_args([
        OsString::from("pics"),
        OsString::from("--job-parallelism"),
        OsString::from("3"),
        OsString::from("--raw-parallelism"),
        OsString::from("8"),
    ])
    .unwrap();
    assert_eq!(config.raw_parallelism, 3);
}

#[test]
fn automatic_raw_parallelism_respects_cpu_memory_and_job_limits() {
    assert_eq!(recommended_raw_parallelism(16, 16 * 1024_u64.pow(3), 8), 8);
    assert_eq!(recommended_raw_parallelism(16, 2 * 1024_u64.pow(3), 8), 1);
    assert_eq!(recommended_raw_parallelism(4, 16 * 1024_u64.pow(3), 8), 2);
    assert_eq!(recommended_raw_parallelism(32, 16 * 1024_u64.pow(3), 3), 3);
}

#[test]
fn rejects_zero_job_parallelism() {
    assert!(
        config_from_args([
            OsString::from("pics"),
            OsString::from("--job-parallelism"),
            OsString::from("0"),
        ])
        .is_err()
    );
}

#[test]
fn rejects_zero_raw_parallelism() {
    assert!(
        config_from_args([
            OsString::from("pics"),
            OsString::from("--raw-parallelism"),
            OsString::from("0"),
        ])
        .is_err()
    );
}

#[test]
fn thumbnail_selection_supports_replace_toggle_and_ranges() {
    let mut selection = ImageSelection::default();
    selection.select(2, false, false);
    assert_eq!(selection.indices, BTreeSet::from([2]));

    selection.select(4, false, true);
    assert_eq!(selection.indices, BTreeSet::from([2, 4]));

    selection.select(6, true, false);
    assert_eq!(selection.indices, BTreeSet::from([4, 5, 6]));
    assert_eq!(selection.focused, Some(6));
}

#[test]
fn work_progress_tracks_overlapping_tasks_and_resets_batches() {
    let mut work = WorkProgress::default();
    let first = work.start("First task");
    let second = work.start("Second task");
    work.finish(first);
    assert_eq!(work.active, 1);
    assert_eq!(work.fraction(), 0.5);
    work.finish(second);
    work.start("New batch");
    assert_eq!(work.active, 1);
    assert_eq!(work.completed, 0);
    assert_eq!(work.total, 1);
}

#[test]
fn image_zoom_starts_fitted_and_toggle_resets_pan() {
    let mut state = ImageViewState::new(PathBuf::from("photo.jpg"));
    assert_eq!(state.zoom, ZoomLevel::FitWindow);
    state.toggle_zoom();
    assert_eq!(state.zoom, ZoomLevel::ActualSize);
    state.pan = (100.0, -50.0);
    state.toggle_zoom();
    assert_eq!(state.zoom, ZoomLevel::FitWindow);
    assert_eq!(state.pan, (0.0, 0.0));
}

#[test]
fn virtual_grid_demands_only_visible_rows_and_prioritizes_the_center() {
    let images: Vec<_> = (0..20)
        .map(|index| PathBuf::from(format!("image-{index}.jpg")))
        .collect();
    let demands = visible_thumbnail_demands(&images, 1..3, 4, 300);
    let paths: Vec<_> = demands.iter().map(|demand| demand.0.clone()).collect();
    assert_eq!(paths, images[4..12]);
    let strongest = demands.iter().max_by_key(|demand| demand.2).unwrap();
    assert_eq!(strongest.0, images[8]);
}

#[test]
fn speculative_batches_are_bounded_and_follow_scroll_direction() {
    assert_eq!(speculative_indices(20, 8..12, 1, 0, 3), vec![12, 13, 14]);
    assert_eq!(speculative_indices(20, 8..12, -1, 0, 3), vec![7, 6, 5]);
    assert_eq!(speculative_indices(20, 8..12, 1, 1, 3), vec![15, 16, 17]);
    assert!(speculative_indices(4, 0..4, 1, 0, 64).is_empty());
    assert_eq!(
        speculative_indices(1_000_000_000, 100..110, 1, 0, 64).len(),
        64
    );
}

#[test]
fn prediction_identity_changes_with_grid_geometry_and_display() {
    let base = GridPredictionState {
        visible_rows: 1..3,
        columns: 4,
        thumbnail_width: 300,
        preview_variant: RenderVariant::Preview {
            max_width: 1920,
            max_height: 1080,
        },
    };
    let mut changed = base.clone();
    changed.columns = 5;
    assert_ne!(base, changed);
    changed = base.clone();
    changed.preview_variant = RenderVariant::Preview {
        max_width: 3840,
        max_height: 2160,
    };
    assert_ne!(base, changed);
}

#[test]
fn virtual_grid_callbacks_merge_fragments_but_replace_a_disjoint_viewport() {
    let variant = RenderVariant::Preview {
        max_width: 1920,
        max_height: 1080,
    };
    let mut pending = None;
    merge_grid_demand(
        &mut pending,
        PendingGridDemand {
            node: NodeId(1),
            visible_rows: 4..7,
            columns: 3,
            thumbnail_width: 300,
            preview_variant: variant,
        },
    );
    merge_grid_demand(
        &mut pending,
        PendingGridDemand {
            node: NodeId(1),
            visible_rows: 2..5,
            columns: 3,
            thumbnail_width: 300,
            preview_variant: variant,
        },
    );
    assert_eq!(pending.as_ref().unwrap().visible_rows, 2..7);

    merge_grid_demand(
        &mut pending,
        PendingGridDemand {
            node: NodeId(1),
            visible_rows: 20..23,
            columns: 3,
            thumbnail_width: 300,
            preview_variant: variant,
        },
    );
    assert_eq!(pending.unwrap().visible_rows, 20..23);
}

#[test]
#[ignore = "manual end-to-end queue benchmark"]
fn benchmark_all_scheduled_jobs() {
    let directory = PathBuf::from("/mnt/pictures/Pictures/2018/Nyaralás/2018-08-06");
    let nef = directory.join("DSC_0389.NEF");
    let jpeg = directory.join("DSC_2587.JPG");
    let cr3 = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test/IMG_0052.CR3");
    let heic = PathBuf::from("/mnt/pictures/Pictures/2026/Alpok túra/2026-07-25/IMG_5562.HEIC");
    for path in [&directory, &nef, &jpeg, &cr3, &heic] {
        assert!(path.exists(), "missing benchmark input: {}", path.display());
    }

    let started = Instant::now();
    let scan = discovery::scan_directory(&directory, RawJpegPreference::Raw).unwrap();
    let directory_scan = started.elapsed();

    let cache = tempfile::tempdir().unwrap();
    let limiter = JobLimiter::new(2);
    let metadata_loader = MetadataLoader::new(cache.path().join("metadata"), 2, limiter.clone());
    let started = Instant::now();
    let metadata =
        metadata_loader.load_directory(scan.images.clone(), Arc::new(AtomicBool::new(false)));
    let metadata_cold = started.elapsed();
    let started = Instant::now();
    let cached_metadata =
        metadata_loader.load_directory(scan.images, Arc::new(AtomicBool::new(false)));
    let metadata_cached = started.elapsed();
    assert_eq!(metadata.len(), cached_metadata.len());

    let pipeline =
        PreviewRenderer::with_source_cache_capacity(cache.path().join("renders"), 2, 2, 8);
    let render_limiter = limiter.clone();
    let pipeline_for_scheduler = pipeline.clone();
    let scheduler = RenderScheduler::new(
        2,
        Arc::new(move |key, priority, cancelled| {
            let Some(_permit) = render_limiter.acquire(
                priority.class >= scheduler::PriorityClass::Visible,
                cancelled,
            ) else {
                return PreviewState::Error("cancelled".into());
            };
            pipeline_for_scheduler.render(
                key.path.clone(),
                key.variant,
                priority.class == scheduler::PriorityClass::Interactive,
                true,
                cancelled,
            )
        }),
    );
    let mut scope = 10_000_u64;
    let mut run = |path: &PathBuf, variant: RenderVariant| {
        scope += 1;
        let started = Instant::now();
        let subscription = scheduler
            .replace_scope(
                DemandScopeId(scope),
                vec![JobDemand {
                    key: JobKey {
                        path: path.clone(),
                        variant,
                        render_identity: RenderIdentity::default(),
                        recipe: None,
                    },
                    priority: JobPriority::INTERACTIVE,
                    subscribe: true,
                }],
            )
            .pop()
            .unwrap();
        let completion = block_on(subscription.receiver).unwrap();
        assert!(matches!(
            completion,
            JobCompletion::Finished(PreviewState::Loaded(_))
        ));
        started.elapsed()
    };
    let (
        jpeg_thumbnail,
        nef_thumbnail,
        cr3_thumbnail,
        heic_thumbnail,
        nef_preview,
        cr3_preview,
        heic_preview,
    ) = {
        let mut cached_variant = |path: &PathBuf, variant: RenderVariant| {
            let writes = pipeline.stats().cache_writes;
            let total_started = Instant::now();
            let cold = run(path, variant);
            let deadline = Instant::now() + Duration::from_secs(30);
            while pipeline.stats().cache_writes == writes && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(2));
            }
            assert!(
                pipeline.stats().cache_writes > writes,
                "cache write timed out"
            );
            let persisted = total_started.elapsed();
            let cached = run(path, variant);
            (cold, persisted, cached)
        };

        (
            cached_variant(&jpeg, RenderVariant::Thumbnail { width: 300 }),
            cached_variant(&nef, RenderVariant::Thumbnail { width: 300 }),
            cached_variant(&cr3, RenderVariant::Thumbnail { width: 300 }),
            cached_variant(&heic, RenderVariant::Thumbnail { width: 300 }),
            cached_variant(
                &nef,
                RenderVariant::Preview {
                    max_width: 1920,
                    max_height: 1080,
                },
            ),
            cached_variant(
                &cr3,
                RenderVariant::Preview {
                    max_width: 1920,
                    max_height: 1080,
                },
            ),
            cached_variant(
                &heic,
                RenderVariant::Preview {
                    max_width: 1920,
                    max_height: 1080,
                },
            ),
        )
    };
    let jpeg_full = run(&jpeg, RenderVariant::Full);
    let nef_full = run(&nef, RenderVariant::Full);
    let cr3_full = run(&cr3, RenderVariant::Full);
    let heic_full = run(&heic, RenderVariant::Full);

    let overhead_scheduler = RenderScheduler::new(
        1,
        Arc::new(|_, _, _| PreviewState::Error("benchmark".into())),
    );
    let overhead_started = Instant::now();
    for index in 0..1_000_u32 {
        let subscription = overhead_scheduler
            .replace_scope(
                DemandScopeId(20_000 + u64::from(index)),
                vec![JobDemand {
                    key: JobKey {
                        path: PathBuf::from(format!("overhead-{index}")),
                        variant: RenderVariant::Thumbnail { width: 300 },
                        render_identity: RenderIdentity::default(),
                        recipe: None,
                    },
                    priority: JobPriority::INTERACTIVE,
                    subscribe: true,
                }],
            )
            .pop()
            .unwrap();
        let _ = block_on(subscription.receiver).unwrap();
    }
    let scheduler_overhead = overhead_started.elapsed() / 1_000;

    eprintln!(
        "scheduler_overhead={scheduler_overhead:?}\ndirectory_scan={directory_scan:?}\nmetadata_cold={metadata_cold:?}\nmetadata_cached={metadata_cached:?}\njpeg_thumbnail_cold={:?}\njpeg_thumbnail_persisted={:?}\njpeg_thumbnail_cached={:?}\nnef_thumbnail_cold={:?}\nnef_thumbnail_persisted={:?}\nnef_thumbnail_cached={:?}\ncr3_thumbnail_cold={:?}\ncr3_thumbnail_persisted={:?}\ncr3_thumbnail_cached={:?}\nheic_thumbnail_cold={:?}\nheic_thumbnail_persisted={:?}\nheic_thumbnail_cached={:?}\nnef_preview_cold={:?}\nnef_preview_persisted={:?}\nnef_preview_cached={:?}\ncr3_preview_cold={:?}\ncr3_preview_persisted={:?}\ncr3_preview_cached={:?}\nheic_preview_cold={:?}\nheic_preview_persisted={:?}\nheic_preview_cached={:?}\njpeg_full={jpeg_full:?}\nnef_full={nef_full:?}\ncr3_full={cr3_full:?}\nheic_full={heic_full:?}\nmetrics={:?}",
        jpeg_thumbnail.0,
        jpeg_thumbnail.1,
        jpeg_thumbnail.2,
        nef_thumbnail.0,
        nef_thumbnail.1,
        nef_thumbnail.2,
        cr3_thumbnail.0,
        cr3_thumbnail.1,
        cr3_thumbnail.2,
        heic_thumbnail.0,
        heic_thumbnail.1,
        heic_thumbnail.2,
        nef_preview.0,
        nef_preview.1,
        nef_preview.2,
        cr3_preview.0,
        cr3_preview.1,
        cr3_preview.2,
        heic_preview.0,
        heic_preview.1,
        heic_preview.2,
        pipeline.stats(),
    );
}

#[test]
#[ignore = "manual cancellation and parallel-throughput benchmark"]
fn benchmark_scheduler_cancellation_and_parallelism() {
    let directory = PathBuf::from("/mnt/pictures/Pictures/2018/Nyaralás/2018-08-06");
    let raw_paths = [
        "DSC_0389.NEF",
        "DSC_0390.NEF",
        "DSC_0391.NEF",
        "DSC_0392.NEF",
    ]
    .map(|name| directory.join(name));
    let jpeg = directory.join("DSC_2587.JPG");
    let cr3 = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test/IMG_0052.CR3");
    for path in raw_paths.iter().chain([&jpeg, &cr3]) {
        assert!(
            path.is_file(),
            "missing benchmark input: {}",
            path.display()
        );
    }

    let throughput = |parallelism: usize| {
        let cache = tempfile::tempdir().unwrap();
        let pipeline = PreviewRenderer::with_source_cache_capacity(
            cache.path().to_path_buf(),
            parallelism,
            parallelism,
            8,
        );
        let executor_pipeline = pipeline.clone();
        let scheduler = RenderScheduler::new(
            parallelism,
            Arc::new(move |key, _, cancelled| {
                executor_pipeline.render(key.path.clone(), key.variant, true, false, cancelled)
            }),
        );
        let started = Instant::now();
        let subscriptions = scheduler.replace_scope(
            DemandScopeId(30_000 + parallelism as u64),
            raw_paths
                .iter()
                .cloned()
                .map(|path| JobDemand {
                    key: JobKey {
                        path,
                        variant: RenderVariant::Thumbnail { width: 300 },
                        render_identity: RenderIdentity::default(),
                        recipe: None,
                    },
                    priority: JobPriority::INTERACTIVE,
                    subscribe: true,
                })
                .collect(),
        );
        for subscription in subscriptions {
            assert!(matches!(
                block_on(subscription.receiver).unwrap(),
                JobCompletion::Finished(PreviewState::Loaded(_))
            ));
        }
        started.elapsed()
    };

    let cancellation_handoff = |raw: &PathBuf| {
        let cache = tempfile::tempdir().unwrap();
        let pipeline =
            PreviewRenderer::with_source_cache_capacity(cache.path().to_path_buf(), 1, 1, 1);
        let executor_pipeline = pipeline.clone();
        let scheduler = RenderScheduler::new(
            1,
            Arc::new(move |key, _, cancelled| {
                executor_pipeline.render(key.path.clone(), key.variant, true, false, cancelled)
            }),
        );
        let raw_scope = DemandScopeId(40_000);
        let _ = scheduler.replace_scope(
            raw_scope,
            vec![JobDemand {
                key: JobKey {
                    path: raw.clone(),
                    variant: RenderVariant::Thumbnail { width: 300 },
                    render_identity: RenderIdentity::default(),
                    recipe: None,
                },
                priority: JobPriority::INTERACTIVE,
                subscribe: true,
            }],
        );
        let deadline = Instant::now() + Duration::from_secs(5);
        while scheduler.snapshot().running == 0 && Instant::now() < deadline {
            std::thread::yield_now();
        }
        scheduler.clear_scope(raw_scope);
        let started = Instant::now();
        let subscription = scheduler
            .replace_scope(
                DemandScopeId(40_001),
                vec![JobDemand {
                    key: JobKey {
                        path: jpeg.clone(),
                        variant: RenderVariant::Thumbnail { width: 300 },
                        render_identity: RenderIdentity::default(),
                        recipe: None,
                    },
                    priority: JobPriority::INTERACTIVE,
                    subscribe: true,
                }],
            )
            .pop()
            .unwrap();
        assert!(matches!(
            block_on(subscription.receiver).unwrap(),
            JobCompletion::Finished(PreviewState::Loaded(_))
        ));
        started.elapsed()
    };

    eprintln!(
        "raw_batch_p1={:?}\nraw_batch_p2={:?}\nraw_batch_p4={:?}\nnef_cancel_to_jpeg={:?}\ncr3_cancel_to_jpeg={:?}",
        throughput(1),
        throughput(2),
        throughput(4),
        cancellation_handoff(&raw_paths[0]),
        cancellation_handoff(&cr3),
    );
}
