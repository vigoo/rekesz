use super::*;

struct FailingPeopleRetryInventory;

#[test]
fn stale_inventory_completion_cannot_own_replacement_progress() {
    assert!(owns_inventory_progress(7, 7));
    assert!(!owns_inventory_progress(8, 7));
}

#[test]
fn repeated_inventory_refresh_coalesces_into_active_run() {
    let mut active = false;
    assert!(claim_inventory_run(&mut active));
    assert!(!claim_inventory_run(&mut active));
    assert!(!claim_inventory_run(&mut active));
    active = false;
    assert!(claim_inventory_run(&mut active));
}

impl MaintenanceDriver for FailingPeopleRetryInventory {
    fn inventory(
        &self,
        _: &Path,
        _: &HashSet<PathBuf>,
        _: discovery::RawJpegPreference,
        _: &Path,
        _: &[PathBuf],
        _: InventoryCoordination<'_>,
    ) -> anyhow::Result<MaintenanceBatch> {
        anyhow::bail!("intentional inventory failure")
    }

    fn refill(&self, _: &Path, _: &Path, _: &[PathBuf]) -> anyhow::Result<MaintenanceRefill> {
        anyhow::bail!("intentional refill failure")
    }
}

#[gpui::test]
fn retry_failed_people_wakes_recognition_even_when_inventory_fails(cx: &mut gpui::TestAppContext) {
    let root = tempfile::tempdir().unwrap();
    let cache = tempfile::tempdir().unwrap();
    catalog::initialize_test_catalog(root.path()).unwrap();
    let connection = rusqlite::Connection::open(catalog::catalog_path(root.path())).unwrap();
    connection
        .execute(
            "INSERT INTO recognition_refresh_queue(entity_kind, entity_id, model_id, status,
                attempts, error, updated_at)
             VALUES('face', 999, 'sface-2021dec', 'error', 5, 'test', datetime('now'))",
            [],
        )
        .unwrap();
    drop(connection);
    let mut config = gpui_test_config(root.path(), cache.path());
    config.settings.automatic_face_analysis = false;
    let (view, cx) = cx.add_window_view(|_, cx| PictureBrowser::new(&config, cx));
    cx.update_entity(&view, |view, cx| {
        view.maintenance_driver = Arc::new(FailingPeopleRetryInventory);
        view.retry_failed_people_analysis(cx);
    });
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline
        && view.read_with(&*cx, |view, _| {
            view.recognition_refresh.stats().completed == 0
        })
    {
        cx.run_until_parked();
        std::thread::sleep(Duration::from_millis(2));
    }
    assert!(view.read_with(&*cx, |view, _| {
        view.recognition_refresh.stats().completed > 0
            && view
                .settings_status
                .as_deref()
                .is_some_and(|status| status.contains("intentional inventory failure"))
    }));
}
use rusqlite::OptionalExtension;

#[test]
fn face_crop_scheduler_policy_capture_changes_live_before_cache_purge() {
    let policy = Arc::new(AtomicBool::new(true));
    let captured = policy.clone();
    let durable_for =
        move |variant, background| durable_render_cache_write(&captured, variant, background);
    let crop = RenderVariant::face_crop(1, [0.1, 0.1, 0.9, 0.9], 64);
    assert!(durable_for(crop, false));

    // This is the ordering used by save_settings: publish MemoryOnly first;
    // only then may asynchronous cache purging begin.
    policy.store(false, Ordering::Release);
    assert!(!durable_for(crop, false));
    assert!(
        !durable_for(crop, true),
        "background priority must not override MemoryOnly for face crops"
    );
    assert!(durable_for(RenderVariant::Thumbnail { width: 64 }, true));
}

#[test]
fn face_worker_admission_preserves_os_and_foreground_memory() {
    let gib = 1024_u64 * 1024 * 1024;
    assert_eq!(admitted_face_workers(gib, 768 * 1024 * 1024, 4), 0);
    assert_eq!(admitted_face_workers(4 * gib, 768 * 1024 * 1024, 8), 3);
    assert_eq!(admitted_face_workers(16 * gib, 768 * 1024 * 1024, 2), 2);
}

#[test]
fn maintenance_diagnostics_bursts_coalesce_to_one_trailing_refresh() {
    let mut throttle = RefreshThrottle::default();
    assert!(throttle.request());
    for _ in 0..100 {
        assert!(!throttle.request());
    }
    assert!(throttle.complete());
    assert!(!throttle.request(), "trailing refresh is already in flight");
    assert!(
        throttle.complete(),
        "new dirtiness requests exactly one more refresh"
    );
    assert!(!throttle.complete());
}

fn deterministic_maintenance_coordinator() -> MaintenanceCoordinator {
    MaintenanceCoordinator {
        refill_active: Duration::from_millis(2),
        refill_error: Duration::from_millis(30),
        inventory_watch: Duration::from_millis(60),
        inventory_error: Duration::from_millis(120),
        throughput_window: Duration::from_millis(61),
    }
}

#[test]
fn drained_refill_dispatches_at_the_durable_retry_deadline() {
    let coordinator = deterministic_maintenance_coordinator();
    let deadline = Duration::from_millis(47);
    assert_eq!(
        coordinator.refill_after(false, false, Some(deadline), false),
        Some(deadline)
    );
    assert_eq!(coordinator.refill_after(false, false, None, false), None);
    assert_eq!(
        coordinator.refill_after(false, true, None, false),
        Some(Duration::from_millis(2))
    );
    assert_eq!(
        coordinator.refill_after(false, false, None, true),
        Some(Duration::from_millis(30))
    );
    assert_eq!(
        coordinator.refill_after(true, true, Some(deadline), true),
        None
    );
}

#[test]
fn drained_inventory_detects_add_delete_and_recovers_after_error() {
    let coordinator = deterministic_maintenance_coordinator();
    let mut fake_now = Duration::ZERO;

    // A successful drained scan arms the watcher. The next scan represents a
    // newly added asset, and the following successful scan a deleted asset.
    for _inventory_change in ["drained", "added", "deleted"] {
        let delay = coordinator
            .inventory_after(false, false, None, true)
            .unwrap();
        fake_now += delay;
    }
    assert_eq!(fake_now, Duration::from_millis(180));

    // A transient scan error backs off, then a successful recovery returns to
    // the normal watch cadence. Pausing suppresses either rearm.
    fake_now += coordinator
        .inventory_after(false, true, None, false)
        .unwrap();
    fake_now += coordinator
        .inventory_after(false, false, None, false)
        .unwrap();
    assert_eq!(fake_now, Duration::from_millis(360));
    assert_eq!(coordinator.inventory_after(true, false, None, false), None);
    assert_eq!(coordinator.inventory_after(true, true, None, false), None);
}

#[test]
fn repeated_throughput_callbacks_rearm_until_the_counter_reaches_zero() {
    let coordinator = deterministic_maintenance_coordinator();
    let mut fake_now = Duration::ZERO;
    for completed in [3, 2] {
        fake_now += coordinator.throughput_after(completed).unwrap();
    }
    assert_eq!(fake_now, Duration::from_millis(122));
    assert_eq!(coordinator.throughput_after(0), None);
}

#[test]
fn production_maintenance_driver_dispatches_durable_retry_and_reconciles_inventory() {
    let _ort = face_engine::ORT_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let root = tempfile::tempdir().unwrap();
    let source = root.path().join("driver.png");
    image::RgbImage::new(32, 32).save(&source).unwrap();
    catalog::initialize_test_catalog(root.path()).unwrap();
    let driver = ProductionMaintenanceDriver;
    let jobs = driver
        .inventory(
            root.path(),
            &HashSet::new(),
            discovery::RawJpegPreference::Raw,
            root.path(),
            &[],
            InventoryCoordination {
                progress: &FaceInventoryProgress::default(),
                admission: &scheduler::MaintenanceAdmission::new(),
                cancelled: &AtomicBool::new(false),
                identity_mode: catalog::InventoryIdentityMode::VerifyContents,
            },
        )
        .unwrap();
    assert_eq!(jobs.len(), 1);
    let connection = rusqlite::Connection::open(catalog::catalog_path(root.path())).unwrap();
    connection
        .execute(
            "UPDATE face_analysis_state SET status='error', failure_class='transient',
               retry_after=datetime('now', '+1 hour')",
            [],
        )
        .unwrap();
    drop(connection);
    let (jobs, wake) = driver.refill(root.path(), root.path(), &[]).unwrap();
    assert!(jobs.is_empty());
    assert!(wake.is_some());

    rusqlite::Connection::open(catalog::catalog_path(root.path()))
        .unwrap()
        .execute(
            "UPDATE face_analysis_state SET retry_after=datetime('now', '-1 second')",
            [],
        )
        .unwrap();
    let (mut jobs, _) = driver.refill(root.path(), root.path(), &[]).unwrap();
    let job = jobs.remove(0).0;
    let service = face_engine::FaceAnalysisService::new_with_pipeline(
        1,
        JobLimiter::new(2),
        scheduler::MaintenanceAdmission::new(),
        rendering::pipeline::RenderPipeline::with_source_cache_capacity(
            root.path().join("driver-cache"),
            1,
            1,
            1,
        ),
    );
    assert!(service.enqueue(job, 0));
    let deadline = Instant::now() + Duration::from_secs(20);
    while service.stats().completed == 0 && Instant::now() < deadline {
        std::thread::yield_now();
    }
    assert_eq!(service.stats().completed, 1);

    std::fs::remove_file(source).unwrap();
    let replacement = root.path().join("replacement.png");
    image::RgbImage::new(32, 32).save(&replacement).unwrap();
    driver
        .inventory(
            root.path(),
            &HashSet::new(),
            discovery::RawJpegPreference::Raw,
            root.path(),
            &[],
            InventoryCoordination {
                progress: &FaceInventoryProgress::default(),
                admission: &scheduler::MaintenanceAdmission::new(),
                cancelled: &AtomicBool::new(false),
                identity_mode: catalog::InventoryIdentityMode::VerifyContents,
            },
        )
        .unwrap();
    let paths = rusqlite::Connection::open(catalog::catalog_path(root.path()))
        .unwrap()
        .prepare("SELECT relative_path FROM assets ORDER BY id")
        .unwrap()
        .query_map([], |row| row.get::<_, Vec<u8>>(0))
        .unwrap()
        .map(|value| catalog::decode_relative_path(&value.unwrap()).unwrap())
        .collect::<Vec<_>>();
    assert!(!paths.contains(&PathBuf::from("driver.png")));
    assert!(paths.contains(&PathBuf::from("replacement.png")));
}
use discovery::DirectoryScan;
use futures::executor::block_on;
use std::path::Path;
use std::time::Instant;
use test_r::test;

fn gpui_test_config(root: &Path, cache: &Path) -> Config {
    Config {
        settings: AppSettings {
            root: root.to_path_buf(),
            image_cache_size: 2,
            pair_preference: config::PairPreference::Raw,
            cache_root: cache.to_path_buf(),
            job_parallelism: 1,
            raw_parallelism: 1,
            online_map_permission: config::OnlineMapPermission::Disabled,
            map_cache_bytes: 1024 * 1024,
            map_switcher_placement: config::MapSwitcherPlacement::Header,
            automatic_face_analysis: true,
            face_workers: 1,
            face_worker_memory_bytes: 512 * 1024 * 1024,
            face_crop_disk_cache: config::FaceCropDiskCachePolicy::default(),
            face_gpu: config::FaceGpuPolicy::default(),
            import_recent_sources: Vec::new(),
            import_destination_preset: Default::default(),
            import_include_duplicates: false,
        },
        settings_path: root.join("settings.json"),
    }
}

fn people_test_face(face_id: i64, person_id: i64, path: &Path) -> catalog::people::FaceReviewItem {
    catalog::people::FaceReviewItem {
        face_id,
        person_id,
        person_name: Some("Ada".to_owned()),
        path: path.to_path_buf(),
        bounds: [0.2, 0.2, 0.8, 0.8],
        origin: catalog::people::FaceOrigin::RekeszDetected,
        kind: catalog::people::FaceAssignmentKind::Suggested,
        provenance: "rekesz".to_owned(),
        confidence: Some(0.91),
    }
}

fn seed_people_ui_catalog(root: &Path, count: usize) -> Vec<catalog::people::FaceReviewItem> {
    catalog::initialize_test_catalog(root).unwrap();
    let photo = root.join("people-ui.png");
    image::RgbImage::new(320, 240).save(&photo).unwrap();
    let mut connection = rusqlite::Connection::open(catalog::catalog_path(root)).unwrap();
    let transaction = connection.transaction().unwrap();
    transaction
        .execute(
            "INSERT INTO assets(id, relative_path, scope_key) VALUES(1, ?1, ?2)",
            rusqlite::params![
                catalog::encode_relative_path(Path::new("people-ui.png")).unwrap(),
                catalog::scope_key(Path::new("people-ui.png")).unwrap()
            ],
        )
        .unwrap();
    transaction
        .execute(
            "INSERT INTO people(id, display_name, created_at, updated_at) \
             VALUES(10, 'Ada', datetime('now'), datetime('now'))",
            [],
        )
        .unwrap();
    let mut faces = Vec::with_capacity(count);
    for index in 0..count {
        let face_id = i64::try_from(index).unwrap() + 100;
        transaction
            .execute(
                "INSERT INTO faces(id, asset_id, provenance, source_key, left, top, right, bottom, \
                 origin, ignored, updated_at) VALUES(?1, 1, 'rekesz', ?2, .2, .2, .8, .8, \
                 'rekesz', 0, datetime('now'))",
                rusqlite::params![face_id, format!("people-ui-{face_id}")],
            )
            .unwrap();
        transaction
            .execute(
                "INSERT INTO face_assignments(face_id, person_id, kind, provenance, confidence, updated_at) \
                 VALUES(?1, 10, 'suggested', 'rekesz', .91, datetime('now'))",
                [face_id],
            )
            .unwrap();
        faces.push(people_test_face(face_id, 10, &photo));
    }
    transaction.commit().unwrap();
    faces
}

#[test]
fn representative_candidates_are_complete_ranked_and_exclude_unusable_faces() {
    use catalog::people::{
        FaceAssignment, FaceAssignmentKind, FaceOrigin, FaceRecord, PeopleCatalog, PersonRecord,
    };
    let face = |id, bounds, ignored| FaceRecord {
        id,
        path: PathBuf::from(format!("{id}.jpg")),
        bounds,
        origin: FaceOrigin::LightroomDetected,
        ignored,
        cluster: None,
    };
    let assignment = |face_id: i64, provenance: &str| FaceAssignment {
        face_id,
        person_id: 10,
        kind: FaceAssignmentKind::Confirmed,
        provenance: provenance.to_owned(),
        confidence: None,
    };
    let people = PeopleCatalog {
        people: vec![PersonRecord {
            id: 10,
            name: "Person".to_owned(),
            confirmed_faces: 6,
            suggested_faces: 0,
            representative_face_id: Some(2),
        }],
        faces: vec![
            face(1, [0.0, 0.0, 0.4, 0.4], false),
            face(2, [0.0, 0.0, 0.3, 0.3], false),
            face(3, [0.0, 0.0, 0.2, 0.2], false),
            face(4, [0.0, 0.0, 0.1, 0.1], false),
            face(5, [0.0, 0.0, 0.01, 0.01], false),
            face(6, [0.0, 0.0, 0.5, 0.5], true),
        ],
        assignments: vec![
            assignment(1, "lightroom"),
            assignment(2, "lightroom"),
            assignment(3, "rekesz"),
            assignment(4, "rekesz"),
            assignment(5, "rekesz"),
            assignment(6, "rekesz"),
        ],
    };
    let visible = (1..=6).collect();

    let candidates = people_representative_faces(&people, 10, &visible)
        .into_iter()
        .map(|face| face.id)
        .collect::<Vec<_>>();

    assert_eq!(candidates, vec![2, 3, 4, 1]);
}

#[test]
fn people_face_selection_uses_composite_identity_and_standard_click_semantics() {
    let keys = [
        PeopleFaceKey {
            face_id: 1,
            person_id: 10,
        },
        PeopleFaceKey {
            face_id: 1,
            person_id: 20,
        },
        PeopleFaceKey {
            face_id: 2,
            person_id: 10,
        },
        PeopleFaceKey {
            face_id: 3,
            person_id: 10,
        },
    ];
    let mut selection = PeopleFaceSelection::default();

    selection.click(keys[0], &keys, false, false);
    selection.click(keys[2], &keys, true, false);
    assert_eq!(selection.selected, keys[..=2].iter().copied().collect());
    assert!(selection.selected.contains(&keys[0]));
    assert!(selection.selected.contains(&keys[1]));

    selection.click(keys[1], &keys, false, true);
    assert!(!selection.selected.contains(&keys[1]));
    selection.click(keys[3], &keys, false, false);
    assert_eq!(selection.selected, HashSet::from([keys[3]]));
}

#[test]
fn people_face_selection_keyboard_moves_extends_selects_all_and_clears() {
    let keys = (0..7)
        .map(|index| PeopleFaceKey {
            face_id: index,
            person_id: 10,
        })
        .collect::<Vec<_>>();
    let mut selection = PeopleFaceSelection::default();

    selection.move_focus(&keys, 1, false);
    assert_eq!(selection.selected, HashSet::from([keys[1]]));
    selection.move_focus(&keys, 3, true);
    assert_eq!(selection.selected, keys[1..=4].iter().copied().collect());

    selection.select_all(&keys);
    assert_eq!(selection.selected.len(), keys.len());
    selection.clear();
    assert_eq!(selection, PeopleFaceSelection::default());
}

#[test]
fn people_bulk_accept_rejects_multiple_identities_for_one_face() {
    let conflicting = [
        (7, 10, PeopleDecision::Confirm),
        (7, 11, PeopleDecision::Confirm),
    ];
    assert!(validate_people_decision_batch(&conflicting).is_err());
    let key = PeopleFaceKey {
        face_id: 7,
        person_id: 10,
    };
    let mut selection = PeopleFaceSelection::default();
    selection.selected.insert(key);
    if validate_people_decision_batch(&conflicting).is_ok() {
        selection.clear();
    }
    assert!(selection.selected.contains(&key));
    assert!(
        validate_people_decision_batch(&[
            (7, 10, PeopleDecision::Reject),
            (7, 11, PeopleDecision::Reject),
        ])
        .is_ok()
    );
}

#[test]
fn people_review_save_and_undo_are_serialized() {
    assert!(can_start_people_review_save(false));
    assert!(!can_start_people_review_save(true));
    assert!(can_start_people_undo(false, 0));
    assert!(!can_start_people_undo(true, 0));
    assert!(!can_start_people_undo(false, 1));
}

#[test]
fn disabled_automatic_face_analysis_skips_startup_inventory_work() {
    let root = tempfile::tempdir().unwrap();
    let cache = tempfile::tempdir().unwrap();
    let mut settings = gpui_test_config(root.path(), cache.path()).settings;
    settings.automatic_face_analysis = false;
    assert!(!should_start_automatic_face_inventory(&settings));
    settings.automatic_face_analysis = true;
    assert!(should_start_automatic_face_inventory(&settings));
}

#[test]
fn people_decision_coordinator_chunks_large_payloads_and_preserves_order() {
    let decisions = (0..1_203).collect::<Vec<_>>();
    let chunks = coordinate_people_decision_chunks(&decisions);
    assert_eq!(
        chunks.iter().map(Vec::len).collect::<Vec<_>>(),
        [500, 500, 203]
    );
    assert_eq!(chunks.into_iter().flatten().collect::<Vec<_>>(), decisions);
}

#[test]
fn optimistic_people_transition_keeps_identity_and_face_wide_rules() {
    let accept = (7, 10, PeopleDecision::Confirm);
    assert!(people_decision_affects_row(&accept, 7, 10));
    assert!(!people_decision_affects_row(&accept, 7, 11));

    let ignore = (7, 10, PeopleDecision::NotFace);
    assert!(people_decision_affects_row(&ignore, 7, 10));
    assert!(people_decision_affects_row(&ignore, 7, 11));
    assert!(!people_decision_affects_row(&ignore, 8, 10));
}

#[test]
fn people_operation_preparation_deduplicates_face_identities() {
    let selection = PeopleFaceSelection {
        selected: HashSet::from([
            PeopleFaceKey {
                face_id: 7,
                person_id: 10,
            },
            PeopleFaceKey {
                face_id: 7,
                person_id: 11,
            },
            PeopleFaceKey {
                face_id: 8,
                person_id: 10,
            },
        ]),
        anchor: None,
        focus: None,
    };
    assert_eq!(prepare_selected_face_ids(&selection).unwrap(), vec![7, 8]);
    assert!(prepare_selected_face_ids(&PeopleFaceSelection::default()).is_err());
}

#[test]
fn people_operation_dialog_validation_rejects_unsafe_inputs() {
    assert!(
        validate_people_operation_dialog(&PeopleOperationDialog::Name {
            operation: PeopleNameOperation::Rename { person_id: 1 },
            input: "  ".to_owned(),
            cursor: 2,
            selection_anchor: None,
        })
        .is_err()
    );
    assert!(
        validate_people_operation_dialog(&PeopleOperationDialog::Name {
            operation: PeopleNameOperation::Create { face_ids: vec![4] },
            input: "Ada".to_owned(),
            cursor: 3,
            selection_anchor: None,
        })
        .is_ok()
    );
    assert!(
        validate_people_operation_dialog(&PeopleOperationDialog::Target {
            operation: PeopleTargetOperation::Merge {
                source_person_id: 3,
            },
            target_person_id: Some(3),
        })
        .is_err()
    );
    assert!(
        validate_people_operation_dialog(&PeopleOperationDialog::Target {
            operation: PeopleTargetOperation::Move {
                source_person_id: 3,
                face_ids: vec![9],
            },
            target_person_id: Some(4),
        })
        .is_ok()
    );
}

#[test]
fn people_name_editor_replaces_unicode_selection_and_enforces_limit() {
    let mut input = "Åda Lovelace".to_owned();
    let mut cursor = 3;
    let mut anchor = Some(0);
    assert_eq!(people_editor_selected_text(&input, cursor, anchor), "Åda");
    people_editor_replace(&mut input, &mut cursor, &mut anchor, "Ada");
    assert_eq!(input, "Ada Lovelace");
    assert_eq!(cursor, 3);
    assert_eq!(anchor, None);

    let mut input = "x".repeat(199);
    let mut cursor = 199;
    people_editor_replace(&mut input, &mut cursor, &mut None, "yz");
    assert_eq!(input.chars().count(), 200);
    assert!(input.ends_with('y'));
}

#[test]
fn people_merge_preview_text_reports_exact_consequences() {
    let preview = catalog::people::MergePeoplePreview {
        source_person_id: 1,
        target_person_id: 2,
        source_confirmed_faces: 7,
        target_confirmed_faces: 5,
        confirmation_deduplications: 2,
        assignment_deduplications: 3,
        source_representative_face_id: Some(11),
        target_representative_face_id: Some(12),
        resulting_representative_face_id: Some(12),
        source_representative_will_be_discarded: true,
        external_links_moved: 1,
    };
    let text = people_merge_preview_text(&preview);
    assert!(text.contains("Result: 10 confirmed faces"));
    assert!(text.contains("2 duplicate confirmations"));
    assert!(text.contains("3 duplicate assignments"));
    assert!(text.contains("1 external catalog link"));
    assert!(text.contains("source example is discarded"));
}

#[test]
fn context_overlay_uses_clean_aperture_and_orientation_transform() {
    let transform = rendering::FaceTransform::IDENTITY
        .then(rendering::FaceTransform::crop(0.25, 0.0, 0.25, 0.0))
        .then(rendering::FaceTransform::orientation(6));
    let metadata = rendering::RenderMetadata {
        source_pipeline: rendering::SourcePipeline::Heif,
        photographic: None,
        source_dimensions: (4, 8),
        output_dimensions: (4, 8),
        variant: RenderVariant::face_context(320),
        orientation_applied: Some(6),
        face_transform: Some(transform),
        develop: None,
    };

    let actual = face_overlay_bounds(&metadata, [0.25, 0.0, 0.5, 1.0]);
    for (actual, expected) in actual.into_iter().zip([0.0, 0.0, 1.0, 0.5]) {
        assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
    }
}

fn write_photo_stub(root: &Path, name: &str, latitude: f64, longitude: f64) -> PathBuf {
    let path = root.join(format!("{name}.jpg"));
    image::RgbImage::from_pixel(8, 8, image::Rgb([32, 96, 160]))
        .save(&path)
        .unwrap();
    std::fs::write(
        root.join(format!("{name}.xmp")),
        format!(
            "<rdf:Description exif:GPSLatitude=\"{latitude}\" exif:GPSLongitude=\"{longitude}\"/>"
        ),
    )
    .unwrap();
    path
}

fn double_click(cx: &mut gpui::VisualTestContext, position: gpui::Point<gpui::Pixels>) {
    let modifiers = gpui::Modifiers::none();
    cx.simulate_event(MouseDownEvent {
        position,
        modifiers,
        button: MouseButton::Left,
        click_count: 2,
        first_mouse: false,
    });
    cx.simulate_event(MouseUpEvent {
        position,
        modifiers,
        button: MouseButton::Left,
        click_count: 2,
    });
}

#[gpui::test]
fn gpui_native_test_context_can_paint_and_click(cx: &mut gpui::TestAppContext) {
    #[derive(Default)]
    struct ClickHarness {
        clicked: bool,
    }

    impl gpui::Render for ClickHarness {
        fn render(
            &mut self,
            _: &mut gpui::Window,
            cx: &mut gpui::Context<Self>,
        ) -> impl gpui::IntoElement {
            div().size_full().child(
                div()
                    .size(px(80.0))
                    .debug_selector(|| "#gpui-test-button".to_owned())
                    .on_mouse_up(
                        MouseButton::Left,
                        cx.listener(|this, _, _, cx| {
                            this.clicked = true;
                            cx.notify();
                        }),
                    ),
            )
        }
    }

    let (view, cx) = cx.add_window_view(|_, _| ClickHarness::default());
    cx.update(|window, _| window.refresh());
    cx.run_until_parked();
    let bounds = cx.debug_bounds("#gpui-test-button").unwrap();
    cx.simulate_click(bounds.center(), gpui::Modifiers::none());
    assert!(view.read_with(cx, |view, _| view.clicked));
}

#[gpui::test]
fn gpui_thumbnail_grid_selects_extends_toggles_opens_and_closes(cx: &mut gpui::TestAppContext) {
    let root = tempfile::tempdir().unwrap();
    let cache = tempfile::tempdir().unwrap();
    for index in 0..4 {
        write_photo_stub(root.path(), &format!("photo-{index}"), 47.5, 19.05);
    }
    let mut config = gpui_test_config(root.path(), cache.path());
    config.settings.automatic_face_analysis = false;
    let (view, cx) = cx.add_window_view(|_, cx| PictureBrowser::new(&config, cx));
    cx.run_until_parked();

    let first = cx.debug_bounds("#thumbnail-0").unwrap();
    cx.simulate_click(first.center(), gpui::Modifiers::none());
    let second = cx.debug_bounds("#thumbnail-1").unwrap();
    cx.simulate_click(
        second.center(),
        gpui::Modifiers {
            control: true,
            ..Default::default()
        },
    );
    assert_eq!(
        view.read_with(&*cx, |view, _| match &view.main_view {
            MainView::Dir(_, selection) => selection.indices.clone(),
            MainView::Image(_) => unreachable!(),
        }),
        BTreeSet::from([0, 1])
    );

    let fourth = cx.debug_bounds("#thumbnail-3").unwrap();
    cx.simulate_click(
        fourth.center(),
        gpui::Modifiers {
            shift: true,
            ..Default::default()
        },
    );
    assert_eq!(
        view.read_with(&*cx, |view, _| match &view.main_view {
            MainView::Dir(_, selection) => selection.indices.clone(),
            MainView::Image(_) => unreachable!(),
        }),
        BTreeSet::from([1, 2, 3])
    );

    double_click(cx, fourth.center());
    assert!(view.read_with(&*cx, |view, _| matches!(
        &view.main_view,
        MainView::Image(state) if state.path.ends_with("photo-3.jpg")
    )));
    cx.dispatch_action(CloseImage);
    assert!(view.read_with(&*cx, |view, _| matches!(
        view.main_view,
        MainView::Dir(_, _)
    )));
}

#[gpui::test]
fn gpui_grid_keyboard_actions_move_extend_and_open_selection(cx: &mut gpui::TestAppContext) {
    cx.update(bind_picture_browser_keys);
    let root = tempfile::tempdir().unwrap();
    let cache = tempfile::tempdir().unwrap();
    for index in 0..4 {
        write_photo_stub(root.path(), &format!("photo-{index}"), 47.5, 19.05);
    }
    let mut config = gpui_test_config(root.path(), cache.path());
    config.settings.automatic_face_analysis = false;
    let (view, cx) = cx.add_window_view(|_, cx| PictureBrowser::new(&config, cx));
    cx.run_until_parked();
    let first = cx.debug_bounds("#thumbnail-0").unwrap();
    cx.simulate_click(first.center(), gpui::Modifiers::none());

    cx.simulate_keystrokes("right shift-right");
    assert_eq!(
        view.read_with(&*cx, |view, _| match &view.main_view {
            MainView::Dir(_, selection) => selection.indices.clone(),
            MainView::Image(_) => unreachable!(),
        }),
        BTreeSet::from([1, 2])
    );
    cx.simulate_keystrokes("enter");
    assert!(view.read_with(&*cx, |view, _| matches!(view.main_view, MainView::Image(_))));
    cx.simulate_keystrokes("escape");
    assert!(view.read_with(&*cx, |view, _| matches!(
        view.main_view,
        MainView::Dir(_, _)
    )));
}

#[gpui::test]
fn gpui_image_view_double_clicks_zooms_and_drag_pans(cx: &mut gpui::TestAppContext) {
    let root = tempfile::tempdir().unwrap();
    let cache = tempfile::tempdir().unwrap();
    write_photo_stub(root.path(), "photo", 47.5, 19.05);
    let config = gpui_test_config(root.path(), cache.path());
    let (view, cx) = cx.add_window_view(|_, cx| PictureBrowser::new(&config, cx));
    cx.run_until_parked();
    let thumbnail = cx.debug_bounds("#thumbnail-0").unwrap();
    double_click(cx, thumbnail.center());
    let preview_deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let loaded = view.read_with(&*cx, |view, _| {
            let MainView::Image(state) = &view.main_view else {
                return false;
            };
            matches!(
                view.previews
                    .state_for(&state.path, view.render_identity_for(&state.path)),
                PreviewState::Loaded(_)
            )
        });
        if loaded {
            break;
        }
        assert!(
            Instant::now() < preview_deadline,
            "image preview did not load within five seconds"
        );
        cx.run_until_parked();
        std::thread::sleep(Duration::from_millis(2));
    }

    let viewport = cx.debug_bounds("#image-viewport").unwrap();
    double_click(cx, viewport.center());
    assert!(view.read_with(&*cx, |view, _| matches!(
        &view.main_view,
        MainView::Image(state) if state.zoom == ZoomLevel::ActualSize
    )));
    let start = viewport.center();
    let end = point(start.x + px(20.0), start.y + px(15.0));
    cx.simulate_mouse_down(start, MouseButton::Left, gpui::Modifiers::none());
    cx.simulate_mouse_move(end, MouseButton::Left, gpui::Modifiers::none());
    assert!(view.read_with(&*cx, |view, _| matches!(
        &view.main_view,
        MainView::Image(state) if state.drag_last.is_some()
    )));
    cx.simulate_mouse_up(end, MouseButton::Left, gpui::Modifiers::none());
    assert!(view.read_with(&*cx, |view, _| matches!(
        &view.main_view,
        MainView::Image(state) if state.drag_last.is_none()
    )));
    double_click(cx, viewport.center());
    assert!(view.read_with(&*cx, |view, _| matches!(
        &view.main_view,
        MainView::Image(state) if state.zoom == ZoomLevel::FitWindow && state.pan == (0.0, 0.0)
    )));
}

#[gpui::test]
fn gpui_grid_recursive_controls_enter_and_exit_recursive_mode(cx: &mut gpui::TestAppContext) {
    let root = tempfile::tempdir().unwrap();
    let cache = tempfile::tempdir().unwrap();
    std::fs::create_dir(root.path().join("trip")).unwrap();
    write_photo_stub(&root.path().join("trip"), "photo", 47.5, 19.05);
    let config = gpui_test_config(root.path(), cache.path());
    let (view, cx) = cx.add_window_view(|_, cx| PictureBrowser::new(&config, cx));
    cx.run_until_parked();

    let enter = cx.debug_bounds("#show-images-recursively").unwrap();
    cx.simulate_click(enter.center(), gpui::Modifiers::none());
    cx.run_until_parked();
    assert!(view.read_with(&*cx, |view, _| view.recursive_grid.is_some()));
    assert!(view.read_with(&*cx, |view, _| matches!(
        view.recursive_grid.as_ref().map(|state| &state.status),
        Some(RecursiveGridStatus::Loaded(images)) if images.paths.len() == 1
    )));
    assert!(cx.debug_bounds("#thumbnail-0").is_some());
    let exit = cx.debug_bounds("#exit-recursive-grid").unwrap();
    cx.simulate_click(exit.center(), gpui::Modifiers::none());
    cx.run_until_parked();
    assert!(view.read_with(&*cx, |view, _| view.recursive_grid.is_none()));
    assert!(cx.debug_bounds("#show-images-recursively").is_some());
}

#[gpui::test]
fn gpui_settings_dialog_cancels_and_saves_painted_control_changes(cx: &mut gpui::TestAppContext) {
    let root = tempfile::tempdir().unwrap();
    let cache = tempfile::tempdir().unwrap();
    let config = gpui_test_config(root.path(), cache.path());
    let (view, cx) = cx.add_window_view(|_, cx| PictureBrowser::new(&config, cx));
    cx.simulate_resize(size(px(1200.0), px(900.0)));
    cx.run_until_parked();
    let root_row = cx.debug_bounds("#directory-0").unwrap();
    cx.simulate_click(root_row.center(), gpui::Modifiers::none());
    cx.dispatch_action(OpenSettings);
    cx.run_until_parked();

    let pair = cx.debug_bounds("#settings-pair-preference").unwrap();
    cx.simulate_click(pair.center(), gpui::Modifiers::none());
    let cancel = cx.debug_bounds("#settings-cancel").unwrap();
    cx.simulate_click(cancel.center(), gpui::Modifiers::none());
    assert_eq!(
        view.read_with(&*cx, |view, _| view.settings.pair_preference),
        config::PairPreference::Raw
    );
    assert!(view.read_with(&*cx, |view, _| view.settings_draft.is_none()));

    cx.dispatch_action(OpenSettings);
    cx.run_until_parked();
    let pair = cx.debug_bounds("#settings-pair-preference").unwrap();
    cx.simulate_click(pair.center(), gpui::Modifiers::none());
    let online = cx.debug_bounds("#settings-online-map").unwrap();
    cx.simulate_click(online.center(), gpui::Modifiers::none());
    let save = cx.debug_bounds("#settings-save").unwrap();
    cx.simulate_click(save.center(), gpui::Modifiers::none());
    assert_eq!(
        view.read_with(&*cx, |view, _| view.settings.pair_preference),
        config::PairPreference::Jpeg
    );
    assert_eq!(
        view.read_with(&*cx, |view, _| view.settings.online_map_permission),
        config::OnlineMapPermission::Enabled
    );
    assert!(view.read_with(&*cx, |view, _| view.settings_draft.is_none()));
    let saved: AppSettings =
        serde_json::from_slice(&std::fs::read(&config.settings_path).unwrap()).unwrap();
    assert_eq!(saved.pair_preference, config::PairPreference::Jpeg);
    assert_eq!(
        saved.online_map_permission,
        config::OnlineMapPermission::Enabled
    );
}

#[gpui::test]
fn gpui_directory_tree_click_and_keyboard_navigation_select_and_expand(
    cx: &mut gpui::TestAppContext,
) {
    cx.update(bind_picture_browser_keys);
    let root = tempfile::tempdir().unwrap();
    let cache = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(root.path().join("alpha/nested")).unwrap();
    std::fs::create_dir(root.path().join("beta")).unwrap();
    let config = gpui_test_config(root.path(), cache.path());
    let (view, cx) = cx.add_window_view(|_, cx| PictureBrowser::new(&config, cx));
    cx.run_until_parked();

    let alpha = cx.debug_bounds("#directory-1").unwrap();
    cx.simulate_click(alpha.center(), gpui::Modifiers::none());
    assert!(view.read_with(&*cx, |view, _| {
        view.model.node(view.selected).path.ends_with("alpha")
    }));

    cx.simulate_keystrokes("right");
    cx.run_until_parked();
    assert!(cx.debug_bounds("#directory-2").is_some());
    cx.simulate_keystrokes("down");
    assert!(view.read_with(&*cx, |view, _| {
        view.model.node(view.selected).path.ends_with("nested")
    }));
    cx.simulate_keystrokes("left");
    assert!(view.read_with(&*cx, |view, _| {
        view.model.node(view.selected).path.ends_with("alpha")
    }));
    cx.simulate_keystrokes("left");
    cx.run_until_parked();
    assert_eq!(view.read_with(&*cx, |view, _| view.visible.len()), 3);
}

#[gpui::test]
fn gpui_path_context_menu_copies_image_path_and_hides_directory(cx: &mut gpui::TestAppContext) {
    let root = tempfile::tempdir().unwrap();
    let cache = tempfile::tempdir().unwrap();
    let trip = root.path().join("trip");
    std::fs::create_dir(&trip).unwrap();
    let photo = write_photo_stub(root.path(), "photo", 47.5, 19.05);
    let config = gpui_test_config(root.path(), cache.path());
    let (view, cx) = cx.add_window_view(|_, cx| PictureBrowser::new(&config, cx));
    cx.run_until_parked();

    let thumbnail = cx.debug_bounds("#thumbnail-0").unwrap();
    cx.simulate_mouse_down(
        thumbnail.center(),
        MouseButton::Right,
        gpui::Modifiers::none(),
    );
    let copy = cx.debug_bounds("#copy-absolute-path-menu-item").unwrap();
    cx.simulate_mouse_down(copy.center(), MouseButton::Left, gpui::Modifiers::none());
    assert_eq!(
        cx.read_from_clipboard().and_then(|item| item.text()),
        Some(photo.to_string_lossy().into_owned())
    );
    assert!(view.read_with(&*cx, |view, _| view.path_context_menu.is_none()));

    let directory = cx.debug_bounds("#directory-1").unwrap();
    cx.simulate_mouse_down(
        directory.center(),
        MouseButton::Right,
        gpui::Modifiers::none(),
    );
    let hide = cx.debug_bounds("#toggle-hidden-subtree-menu-item").unwrap();
    cx.simulate_mouse_down(hide.center(), MouseButton::Left, gpui::Modifiers::none());
    assert!(view.read_with(&*cx, |view, _| view.hidden_subtrees.contains(&trip)));
}

#[gpui::test]
fn gpui_linux_file_menu_opens_cache_dialog_and_close_button_closes_it(
    cx: &mut gpui::TestAppContext,
) {
    if !USE_IN_WINDOW_MENU {
        return;
    }
    let root = tempfile::tempdir().unwrap();
    let cache = tempfile::tempdir().unwrap();
    let config = gpui_test_config(root.path(), cache.path());
    let (view, cx) = cx.add_window_view(|_, cx| PictureBrowser::new(&config, cx));
    cx.run_until_parked();
    let file = cx.debug_bounds("#file-menu-button").unwrap();
    cx.simulate_click(file.center(), gpui::Modifiers::none());
    let cache_item = cx.debug_bounds("#cache-statistics-menu-item").unwrap();
    cx.simulate_click(cache_item.center(), gpui::Modifiers::none());
    assert!(view.read_with(&*cx, |view, _| view.show_cache_statistics));
    let close = cx.debug_bounds("#cache-statistics-close").unwrap();
    cx.simulate_click(close.center(), gpui::Modifiers::none());
    assert!(!view.read_with(&*cx, |view, _| view.show_cache_statistics));
}

#[gpui::test]
fn gpui_cache_dialog_clear_button_removes_downloaded_map_cache(cx: &mut gpui::TestAppContext) {
    let root = tempfile::tempdir().unwrap();
    let cache = tempfile::tempdir().unwrap();
    std::fs::create_dir(cache.path().join("map")).unwrap();
    let map_cache = cache.path().join("map/ambient.sqlite3");
    std::fs::write(&map_cache, b"cached tiles").unwrap();
    let config = gpui_test_config(root.path(), cache.path());
    let (view, cx) = cx.add_window_view(|_, cx| PictureBrowser::new(&config, cx));
    cx.update_entity(&view, |view, cx| {
        view.show_cache_statistics = true;
        cx.notify();
    });
    cx.update(|window, _| window.refresh());
    cx.run_until_parked();
    let clear = cx.debug_bounds("#clear-downloaded-map-tiles").unwrap();
    cx.simulate_click(clear.center(), gpui::Modifiers::none());
    cx.run_until_parked();
    assert!(!map_cache.exists());
    assert!(!view.read_with(&*cx, |view, _| view.map_cache_clearing));
}

#[gpui::test]
fn gpui_cluster_gallery_paginates_and_zoom_action_updates_the_view(cx: &mut gpui::TestAppContext) {
    let root = tempfile::tempdir().unwrap();
    let cache = tempfile::tempdir().unwrap();
    let config = gpui_test_config(root.path(), cache.path());
    let (view, cx) = cx.add_window_view(|_, cx| PictureBrowser::new(&config, cx));
    let coordinate = metadata::GeoCoordinate::new(47.5, 19.05).unwrap();
    cx.update_entity(&view, |view, cx| {
        view.directory_view_mode = DirectoryViewMode::Map;
        view.map_render_size = (1280, 800);
        view.map_camera.zoom = 3.0;
        view.map_card = MapCard::new(
            coordinate,
            (0..6)
                .map(|index| MapCardItem {
                    path: root.path().join(format!("{index}.jpg")),
                })
                .collect(),
            7,
            None,
            Some(8),
        );
        cx.notify();
    });
    cx.update(|window, _| window.refresh());
    cx.run_until_parked();

    let next = cx.debug_bounds("#map-card-next").unwrap();
    cx.simulate_click(next.center(), gpui::Modifiers::none());
    assert_eq!(
        view.read_with(cx, |view, _| view.map_card.as_ref().unwrap().page),
        1
    );

    let zoom = cx.debug_bounds("#map-card-zoom").unwrap();
    cx.simulate_click(zoom.center(), gpui::Modifiers::none());
    assert!(view.read_with(cx, |view, _| view.map_card.is_none()));
    assert!(view.read_with(cx, |view, _| view.map_camera.zoom > 3.0));
}

#[gpui::test]
fn gpui_map_cache_bar_and_map_job_use_the_regular_ui(cx: &mut gpui::TestAppContext) {
    let root = tempfile::tempdir().unwrap();
    let cache = tempfile::tempdir().unwrap();
    std::fs::create_dir(cache.path().join("map")).unwrap();
    std::fs::write(cache.path().join("map/ambient.sqlite3"), vec![0_u8; 512]).unwrap();
    let mut config = gpui_test_config(root.path(), cache.path());
    config.settings.map_cache_bytes = 1024;
    let (view, cx) = cx.add_window_view(|_, cx| PictureBrowser::new(&config, cx));
    cx.update_entity(&view, |view, cx| {
        view.directory_view_mode = DirectoryViewMode::Map;
        view.map_job = Some(MapJob {
            epoch: 42,
            label: "Rendering map and downloading visible tiles…".to_owned(),
            status: MapJobStatus::Running,
        });
        view.show_job_queue = true;
        cx.notify();
    });
    cx.update(|window, _| window.refresh());
    cx.run_until_parked();
    assert!(cx.debug_bounds("#map-job-queue-row").is_some());
    assert!(cx.debug_bounds("#background-progress").is_some());
    let grid = cx.debug_bounds("#show-grid-view").unwrap();
    cx.simulate_click(grid.center(), gpui::Modifiers::none());
    assert!(view.read_with(cx, |view, _| view.map_job.is_none()));

    cx.update_entity(&view, |view, cx| {
        view.show_cache_statistics = true;
        cx.notify();
    });
    cx.run_until_parked();
    let bar = cx.debug_bounds("#map-cache-capacity-bar").unwrap();
    let fill = cx.debug_bounds("#map-cache-capacity-fill").unwrap();
    assert!(f32::from(bar.size.width) > 100.0);
    assert!(f32::from(bar.size.height) > 0.0);
    assert!((f32::from(fill.size.width) / f32::from(bar.size.width) - 0.5).abs() < 0.01);
}

#[gpui::test]
fn gpui_job_queue_pauses_face_maintenance(cx: &mut gpui::TestAppContext) {
    let root = tempfile::tempdir().unwrap();
    let cache = tempfile::tempdir().unwrap();
    let mut config = gpui_test_config(root.path(), cache.path());
    config.settings.automatic_face_analysis = false;
    let (view, cx) = cx.add_window_view(|_, cx| PictureBrowser::new(&config, cx));
    cx.update_entity(&view, |view, cx| {
        view.show_job_queue = true;
        cx.notify();
    });
    cx.run_until_parked();
    let toggle = cx.debug_bounds("#job-queue-face-analysis-toggle").unwrap();
    cx.simulate_click(toggle.center(), gpui::Modifiers::none());
    assert!(view.read_with(cx, |view, _| view.face_analysis_paused));
}

#[gpui::test]
fn gpui_job_queue_shows_stage_and_durably_cancels_one_face_job(cx: &mut gpui::TestAppContext) {
    let root = tempfile::tempdir().unwrap();
    let cache = tempfile::tempdir().unwrap();
    let source = root.path().join("face.png");
    image::RgbImage::new(32, 32).save(&source).unwrap();
    catalog::initialize_test_catalog(root.path()).unwrap();
    face_engine::sync_analysis_inventory(root.path(), std::slice::from_ref(&source)).unwrap();
    let job = face_engine::pending_analysis_jobs(root.path(), 1)
        .unwrap()
        .remove(0);
    let config = gpui_test_config(root.path(), cache.path());
    let (view, cx) = cx.add_window_view(|_, cx| PictureBrowser::new(&config, cx));
    cx.update_entity(&view, |view, cx| {
        view.show_job_queue = true;
        assert!(view.face_analysis.enqueue(job.clone(), 0));
        cx.notify();
    });
    cx.run_until_parked();
    let row = cx.debug_bounds("#face-analysis-job-queue-row").unwrap();
    cx.simulate_click(row.center(), gpui::Modifiers::none());
    assert!(
        face_engine::pending_analysis_jobs(root.path(), 1)
            .unwrap()
            .is_empty()
    );
}

#[gpui::test]
fn gpui_job_queue_paints_and_cancels_people_summary_job(cx: &mut gpui::TestAppContext) {
    let root = tempfile::tempdir().unwrap();
    let cache = tempfile::tempdir().unwrap();
    catalog::initialize_test_catalog(root.path()).unwrap();
    let config = gpui_test_config(root.path(), cache.path());
    let (view, cx) = cx.add_window_view(|_, cx| PictureBrowser::new(&config, cx));
    cx.update_entity(&view, |view, cx| {
        view.scheduler
            .maintenance_admission()
            .set_foreground_blocked(true);
        view.show_job_queue = true;
        assert!(
            view.people_summary_rebuild
                .enqueue(root.path().to_path_buf(), 0)
        );
        cx.notify();
    });
    cx.update(|window, _| window.refresh());
    cx.run_until_parked();
    let row = cx.debug_bounds("#people-summary-job-queue-row").unwrap();
    cx.simulate_click(row.center(), gpui::Modifiers::none());
    let status = catalog::people::people_summary_rebuild_status(root.path()).unwrap();
    assert_eq!(status.queue_status.as_deref(), Some("pending"));
    assert_eq!(status.worker_id, None);
    assert_eq!(status.lease_expires_at, None);
}

#[gpui::test]
fn gpui_job_queue_paints_and_cancels_recognition_job(cx: &mut gpui::TestAppContext) {
    let root = tempfile::tempdir().unwrap();
    let cache = tempfile::tempdir().unwrap();
    catalog::initialize_test_catalog(root.path()).unwrap();
    let config = gpui_test_config(root.path(), cache.path());
    let (view, cx) = cx.add_window_view(|_, cx| PictureBrowser::new(&config, cx));
    cx.update_entity(&view, |view, cx| {
        view.scheduler
            .maintenance_admission()
            .set_foreground_blocked(true);
        view.show_job_queue = true;
        assert!(
            view.recognition_refresh
                .enqueue(root.path().to_path_buf(), 42)
        );
        cx.notify();
    });
    cx.update(|window, _| window.refresh());
    cx.run_until_parked();
    let row = cx.debug_bounds("#recognition-job-queue-row").unwrap();
    cx.simulate_click(row.center(), gpui::Modifiers::none());
    assert!(view.read_with(cx, |view, _| {
        view.recognition_refresh
            .snapshot()
            .iter()
            .all(|job| job.cancelled)
    }));
}

#[gpui::test]
fn gpui_mixed_maintenance_rows_keep_recognition_click_index_and_retryable_work(
    cx: &mut gpui::TestAppContext,
) {
    let root = tempfile::tempdir().unwrap();
    let cache = tempfile::tempdir().unwrap();
    let sources = [root.path().join("one.png"), root.path().join("two.png")];
    for source in &sources {
        image::RgbImage::new(32, 32).save(source).unwrap();
    }
    catalog::initialize_test_catalog(root.path()).unwrap();
    face_engine::sync_analysis_inventory(root.path(), &sources).unwrap();
    let jobs = face_engine::pending_analysis_jobs(root.path(), 2).unwrap();
    let mut config = gpui_test_config(root.path(), cache.path());
    config.settings.automatic_face_analysis = false;
    let (view, cx) = cx.add_window_view(|_, cx| PictureBrowser::new(&config, cx));
    cx.update_entity(&view, |view, cx| {
        view.scheduler
            .maintenance_admission()
            .set_foreground_blocked(true);
        rusqlite::Connection::open(catalog::catalog_path(root.path()))
            .unwrap()
            .execute(
                "INSERT INTO recognition_refresh_queue(
                     entity_kind, entity_id, model_id, status, attempts, updated_at)
                 VALUES('face', 777, 'sface-2021dec', 'pending', 0, datetime('now'))",
                [],
            )
            .unwrap();
        view.show_job_queue = true;
        for job in jobs {
            assert!(view.face_analysis.enqueue(job, 10));
        }
        assert!(
            view.recognition_refresh
                .enqueue(root.path().to_path_buf(), 20)
        );
        assert!(
            view.people_summary_rebuild
                .enqueue(root.path().to_path_buf(), 0)
        );
        cx.notify();
    });
    cx.update(|window, _| window.refresh());
    cx.run_until_parked();
    assert_eq!(
        view.read_with(cx, |view, _| view.face_job_queue_snapshot.len()),
        2
    );
    assert_eq!(
        view.read_with(cx, |view, _| view.recognition_job_queue_snapshot.len()),
        1
    );
    assert_eq!(
        view.read_with(cx, |view, _| view.summary_job_queue_snapshot.len()),
        1
    );
    assert!(cx.debug_bounds("#face-analysis-job-queue-row").is_some());
    assert!(cx.debug_bounds("#recognition-job-queue-row").is_some());
    assert!(cx.debug_bounds("#people-summary-job-queue-row").is_some());
    let recognition = cx.debug_bounds("#recognition-job-queue-row").unwrap();
    cx.simulate_click(recognition.center(), gpui::Modifiers::none());
    let cancellation_deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let cancelled = view.read_with(cx, |view, _| {
            view.face_analysis.snapshot().len() == 2
                && view.people_summary_rebuild.snapshot().len() == 1
                && view
                    .recognition_refresh
                    .snapshot()
                    .iter()
                    .all(|job| job.cancelled)
        });
        if cancelled {
            break;
        }
        assert!(
            Instant::now() < cancellation_deadline,
            "recognition row did not observe cooperative cancellation"
        );
        cx.run_until_parked();
        std::thread::sleep(Duration::from_millis(2));
    }
    let deadline = Instant::now() + Duration::from_secs(5);
    let durable_status = loop {
        let status: String = rusqlite::Connection::open(catalog::catalog_path(root.path()))
            .unwrap()
            .query_row(
                "SELECT status FROM recognition_refresh_queue WHERE entity_id=777",
                [],
                |row| row.get(0),
            )
            .unwrap();
        if status == "pending" || Instant::now() >= deadline {
            break status;
        }
        cx.run_until_parked();
        std::thread::sleep(Duration::from_millis(2));
    };
    assert_eq!(durable_status, "pending");
}

#[gpui::test]
fn gpui_people_overview_paints_representative_and_scope_cancels_stale_query(
    cx: &mut gpui::TestAppContext,
) {
    let root = tempfile::tempdir().unwrap();
    let cache = tempfile::tempdir().unwrap();
    let faces = seed_people_ui_catalog(root.path(), 1);
    let connection = rusqlite::Connection::open(catalog::catalog_path(root.path())).unwrap();
    connection
        .execute(
            "UPDATE face_assignments SET kind='confirmed' WHERE face_id=?1 AND person_id=10",
            [faces[0].face_id],
        )
        .unwrap();
    drop(connection);
    let mut overview = catalog::people::list_people_page(root.path(), None, 0, 30).unwrap();
    assert_eq!(overview.items[0].confirmed_faces, 1);
    assert_eq!(overview.items[0].suggested_faces, 0);
    assert_eq!(
        overview.items[0].representative_face_id,
        Some(faces[0].face_id)
    );
    let template = overview.items[0].clone();
    overview.items = (0..30)
        .map(|index| {
            let mut person = template.clone();
            person.id = 10 + index;
            person.name = format!("Person {index}");
            if let Some(face) = &mut person.representative {
                face.face_id = faces[0].face_id + index;
            }
            person.representative_face_id = Some(faces[0].face_id + index);
            person
        })
        .collect();
    overview.total = 30;
    let mut config = gpui_test_config(root.path(), cache.path());
    config.settings.automatic_face_analysis = false;
    let (view, cx) = cx.add_window_view(|_, cx| PictureBrowser::new(&config, cx));
    cx.update_entity(&view, |view, cx| {
        view.directory_view_mode = DirectoryViewMode::People;
        view.people_overview = PeoplePageLoad::Loaded(overview);
        view.refresh_people_render_demand(cx);
        view.people_query_cancelled = Arc::new(AtomicBool::new(false));
        cx.notify();
    });
    cx.run_until_parked();
    assert!(cx.debug_bounds("#person-card-10").is_some());
    assert!(cx.debug_bounds("#people-review-queue").is_some());
    assert!(view.read_with(&*cx, |view, _| {
        !view.people_rendered_overview_indices.is_empty()
            && view.people_rendered_overview_indices.len() < 30
            && view.scheduler.snapshot().jobs.iter().all(|job| {
                !matches!(job.key.variant, RenderVariant::FaceCrop { face_id, .. }
                    if face_id == faces[0].face_id + 29)
                    || job.cancelled
            })
    }));
    for _ in 0..100 {
        let ready = view.read_with(&*cx, |view, _| {
            let identity = view.render_identity_for(&faces[0].path);
            matches!(
                view.thumbnails.state_for_variant(
                    &faces[0].path,
                    identity,
                    RenderVariant::face_crop(faces[0].face_id, faces[0].bounds, 320),
                ),
                PreviewState::Loaded(_) | PreviewState::Error(_)
            )
        });
        if ready {
            break;
        }
        std::thread::sleep(Duration::from_millis(5));
        cx.run_until_parked();
    }
    let representative_state = view.read_with(&*cx, |view, _| {
        let identity = view.render_identity_for(&faces[0].path);
        view.thumbnails.state_for_variant(
            &faces[0].path,
            identity,
            RenderVariant::face_crop(faces[0].face_id, faces[0].bounds, 320),
        )
    });
    let state_label = match &representative_state {
        PreviewState::Unloaded => "unloaded",
        PreviewState::Loading(_) => "loading",
        PreviewState::Loaded(_) => "loaded",
        PreviewState::Error(_) => "error",
    };
    assert!(
        matches!(representative_state, PreviewState::Loaded(_)),
        "confirmed representative crop did not paint: {state_label}"
    );

    let old_cancel = view.read_with(&*cx, |view, _| view.people_query_cancelled.clone());
    let subtree = cx.debug_bounds("#people-subtree-scope").unwrap();
    cx.simulate_click(subtree.center(), gpui::Modifiers::none());
    assert!(old_cancel.load(Ordering::Acquire));
    assert!(!view.read_with(&*cx, |view, _| view.people_whole_library));
    assert!(
        view.read_with(&*cx, |view, _| {
            view.scheduler.snapshot().jobs.iter().all(|job| {
                !matches!(job.key.variant, RenderVariant::FaceCrop { face_id, .. }
                if face_id == faces[0].face_id)
                    || job.cancelled
            })
        }),
        "scope changes must clear the old representative crop demand immediately"
    );
}

#[gpui::test]
fn gpui_representative_failure_advances_then_exhausts_to_stable_placeholder(
    cx: &mut gpui::TestAppContext,
) {
    let root = tempfile::tempdir().unwrap();
    let cache = tempfile::tempdir().unwrap();
    catalog::initialize_test_catalog(root.path()).unwrap();
    let valid = root.path().join("valid-representative.png");
    image::RgbImage::new(64, 64).save(&valid).unwrap();
    let connection = rusqlite::Connection::open(catalog::catalog_path(root.path())).unwrap();
    for (asset_id, relative) in [
        (1_i64, "missing-first.png"),
        (2, "valid-representative.png"),
        (3, "missing-exhaust-a.png"),
        (4, "missing-exhaust-b.png"),
    ] {
        connection
            .execute(
                "INSERT INTO assets(id,relative_path,scope_key) VALUES(?1,?2,?3)",
                rusqlite::params![
                    asset_id,
                    catalog::encode_relative_path(Path::new(relative)).unwrap(),
                    catalog::scope_key(Path::new(relative)).unwrap()
                ],
            )
            .unwrap();
    }
    connection
        .execute_batch(
            "INSERT INTO people(id,display_name,created_at,updated_at)
               VALUES(10,'A Fallback','x','x'),(20,'Z Exhaust','x','x');
             INSERT INTO faces(id,asset_id,provenance,source_key,left,top,right,bottom,origin,
                 detection_confidence,updated_at) VALUES
               (1,1,'rekesz','bad-first',.1,.1,.8,.8,'rekesz',.99,'2030'),
               (2,2,'rekesz','good-second',.1,.1,.8,.8,'rekesz',.90,'2029'),
               (3,3,'rekesz','bad-a',.1,.1,.8,.8,'rekesz',.99,'2030'),
               (4,4,'rekesz','bad-b',.1,.1,.8,.8,'rekesz',.90,'2029');
             INSERT INTO face_assignments(face_id,person_id,kind,provenance,updated_at) VALUES
               (1,10,'confirmed','rekesz','x'),(2,10,'confirmed','rekesz','x'),
               (3,20,'confirmed','rekesz','x'),(4,20,'confirmed','rekesz','x');
             UPDATE people SET representative_face_id=1,representative_is_manual=1 WHERE id=10;
             UPDATE people SET representative_face_id=3,representative_is_manual=1 WHERE id=20;",
        )
        .unwrap();
    drop(connection);
    catalog::people::rebuild_people_summary(root.path(), Arc::new(AtomicBool::new(false))).unwrap();

    let mut config = gpui_test_config(root.path(), cache.path());
    config.settings.automatic_face_analysis = false;
    let (view, cx) = cx.add_window_view(|_, cx| PictureBrowser::new(&config, cx));
    cx.update_entity(&view, |view, cx| {
        view.directory_view_mode = DirectoryViewMode::People;
        view.refresh_people_page(cx);
    });
    for _ in 0..500 {
        cx.update(|window, _| window.refresh());
        cx.run_until_parked();
        let fallback_settled = view.read_with(&*cx, |view, _| {
            matches!(
                view.thumbnails.state_for_variant(
                    &valid,
                    view.render_identity_for(&valid),
                    RenderVariant::face_crop(2, [0.1, 0.1, 0.8, 0.8], 320),
                ),
                PreviewState::Loaded(_)
            ) && matches!(&view.people_overview, PeoplePageLoad::Loaded(page)
                if page.items.iter().any(|person| person.id == 10
                    && person.representative_face_id == Some(2)))
                && view.people_representative_offsets.get(&10) == Some(&1)
        });
        if fallback_settled {
            break;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    // Paint once per expected failure, then allow that request and its async
    // replacement query to settle without cancelling it with another paint.
    for expected_offset in [1_u64, 2] {
        cx.update(|window, _| window.refresh());
        for _ in 0..500 {
            cx.run_until_parked();
            if view.read_with(&*cx, |view, _| {
                view.people_representative_offsets.get(&20) == Some(&expected_offset)
                    && matches!(&view.people_overview, PeoplePageLoad::Loaded(page)
                        if page.items.iter().any(|person| person.id == 20
                            && (expected_offset == 1 || person.representative.is_none())))
            }) {
                break;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
    }
    cx.update(|window, _| window.refresh());
    cx.run_until_parked();
    assert!(cx.debug_bounds("#person-card-10").is_some());
    assert!(cx.debug_bounds("#person-card-20").is_some());
    let fallback_state = view.read_with(&*cx, |view, _| {
        (
            view.people_representative_offsets.clone(),
            match &view.people_overview {
                PeoplePageLoad::Loaded(page) => page
                    .items
                    .iter()
                    .map(|person| (person.id, person.representative_face_id))
                    .collect::<Vec<_>>(),
                _ => Vec::new(),
            },
        )
    });
    assert!(
        view.read_with(&*cx, |view, _| {
            matches!(&view.people_overview, PeoplePageLoad::Loaded(page)
            if page.items.iter().any(|person| person.id == 10
                && person.representative_face_id == Some(2))
            && page.items.iter().any(|person| person.id == 20
                && person.representative.is_none()))
                && view.people_representative_offsets.get(&10) == Some(&1)
                && view.people_representative_offsets.get(&20) == Some(&2)
        }),
        "representative fallback did not settle: {fallback_state:?}"
    );
    let exhausted_offset = view.read_with(&*cx, |view, _| view.people_representative_offsets[&20]);
    for _ in 0..20 {
        cx.run_until_parked();
    }
    assert_eq!(
        view.read_with(&*cx, |view, _| view.people_representative_offsets[&20]),
        exhausted_offset,
        "exhausted placeholder must not start a failure/requery loop"
    );
    assert!(view.read_with(&*cx, |view, _| {
        view.scheduler.snapshot().jobs.iter().all(|job| {
            !matches!(
                job.key.variant,
                RenderVariant::FaceCrop { face_id: 3 | 4, .. }
            ) || job.cancelled
        })
    }));
}

#[gpui::test]
fn gpui_people_detail_virtualizes_toggles_context_and_multiselects(cx: &mut gpui::TestAppContext) {
    let root = tempfile::tempdir().unwrap();
    let cache = tempfile::tempdir().unwrap();
    seed_people_ui_catalog(root.path(), ninety_people_faces());
    let mut config = gpui_test_config(root.path(), cache.path());
    config.settings.automatic_face_analysis = false;
    let (view, cx) = cx.add_window_view(|_, cx| PictureBrowser::new(&config, cx));
    cx.update_entity(&view, |view, cx| {
        view.directory_view_mode = DirectoryViewMode::People;
        view.selected_person = Some(10);
        view.selected_person_name = Some("Ada".to_owned());
        view.refresh_people_page(cx);
    });
    for _ in 0..100 {
        cx.run_until_parked();
        if view.read_with(&*cx, |view, _| {
            matches!(&view.people_faces, PeoplePageLoad::Loaded(page) if page.items.len() == 30 && page.total == 90)
        }) {
            break;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    cx.update(|window, _| window.refresh());
    cx.run_until_parked();
    assert!(
        view.read_with(&*cx, |view, _| {
            !view.people_rendered_face_indices.is_empty()
                && view.people_rendered_face_indices.len() < 30
                && !view.people_rendered_face_indices.contains(&29)
        }),
        "the uniform-list processor must paint a strict viewport subset"
    );

    let toggle = cx.debug_bounds("#people-context-toggle").unwrap();
    cx.simulate_click(toggle.center(), gpui::Modifiers::none());
    assert!(view.read_with(&*cx, |view, _| view.people_context_mode));
    assert!(
        view.read_with(&*cx, |view, _| {
            view.scheduler.snapshot().jobs.iter().all(|job| {
                !matches!(job.key.variant, RenderVariant::FaceCrop { .. }) || job.cancelled
            })
        }),
        "switching to context must remove or cooperatively cancel every stale crop demand"
    );
    for _ in 0..100 {
        let context_ready = view.read_with(&*cx, |view, _| {
            let PeoplePageLoad::Loaded(page) = &view.people_faces else {
                return false;
            };
            let face = &page.items[0];
            matches!(
                view.thumbnails.state_for_variant(
                    &face.path,
                    view.render_identity_for(&face.path),
                    RenderVariant::face_context(320),
                ),
                PreviewState::Loaded(_)
            )
        });
        if context_ready {
            break;
        }
        std::thread::sleep(Duration::from_millis(5));
        cx.run_until_parked();
    }
    assert!(view.read_with(&*cx, |view, _| {
        let PeoplePageLoad::Loaded(page) = &view.people_faces else {
            return false;
        };
        let face = &page.items[0];
        matches!(
            view.thumbnails.state_for_variant(
                &face.path,
                view.render_identity_for(&face.path),
                RenderVariant::face_context(320),
            ),
            PreviewState::Loaded(_)
        )
    }));

    // A single tile can override the grid default and retain a genuinely mixed
    // crop/context demand set.
    let face_toggle = cx.debug_bounds("#toggle-face-context-100-10").unwrap();
    cx.simulate_click(face_toggle.center(), gpui::Modifiers::none());
    cx.run_until_parked();
    assert!(view.read_with(&*cx, |view, _| {
        !view.people_face_uses_context(PeopleFaceKey {
            face_id: 100,
            person_id: 10,
        }) && view.people_face_uses_context(PeopleFaceKey {
            face_id: 101,
            person_id: 10,
        }) && view.people_face_context_overrides.len() == 1
    }));
    assert!(view.read_with(&*cx, |view, _| {
        let PeoplePageLoad::Loaded(page) = &view.people_faces else {
            return false;
        };
        let crop = &page.items[0];
        let context = &page.items[1];
        matches!(
            view.thumbnails.state_for_variant(
                &crop.path,
                view.render_identity_for(&crop.path),
                RenderVariant::face_crop(crop.face_id, crop.bounds, 320),
            ),
            PreviewState::Loading(_) | PreviewState::Loaded(_)
        ) && matches!(
            view.thumbnails.state_for_variant(
                &context.path,
                view.render_identity_for(&context.path),
                RenderVariant::face_context(320),
            ),
            PreviewState::Loading(_) | PreviewState::Loaded(_)
        )
    }));
    let first = cx.debug_bounds("#people-face-tile-100-10").unwrap();
    let second = cx.debug_bounds("#people-face-tile-101-10").unwrap();
    cx.simulate_click(first.center(), gpui::Modifiers::none());
    cx.simulate_click(
        second.center(),
        gpui::Modifiers {
            control: true,
            ..Default::default()
        },
    );
    assert_eq!(
        view.read_with(&*cx, |view, _| view.people_face_selection.selected.len()),
        2
    );
    assert!(cx.debug_bounds("#people-selection-actions").is_some());
    assert!(cx.debug_bounds("#accept-selected-faces").is_some());
    let clear_selected = cx.debug_bounds("#clear-selected-faces").unwrap();

    cx.simulate_keystrokes("shift-right");
    assert!(
        view.read_with(&*cx, |view, _| {
            view.people_face_selection.selected
                == HashSet::from([
                    PeopleFaceKey {
                        face_id: 101,
                        person_id: 10,
                    },
                    PeopleFaceKey {
                        face_id: 102,
                        person_id: 10,
                    },
                ])
        }),
        "Shift+Right must extend from the keyboard anchor in painted order"
    );
    cx.simulate_keystrokes("ctrl-a");
    assert_eq!(
        view.read_with(&*cx, |view, _| view.people_face_selection.selected.len()),
        30,
        "Ctrl/Cmd+A selects the loaded result deliberately"
    );
    cx.simulate_keystrokes("escape");
    assert!(view.read_with(&*cx, |view, _| {
        view.people_face_selection.selected.is_empty()
    }));

    cx.simulate_click(first.center(), gpui::Modifiers::none());
    let third = cx.debug_bounds("#people-face-tile-102-10").unwrap();
    cx.simulate_click(
        third.center(),
        gpui::Modifiers {
            shift: true,
            ..Default::default()
        },
    );
    assert_eq!(
        view.read_with(&*cx, |view, _| view.people_face_selection.selected.len()),
        3,
        "Shift-click must select the painted range"
    );
    cx.simulate_click(clear_selected.center(), gpui::Modifiers::none());
    assert!(view.read_with(&*cx, |view, _| {
        view.people_face_selection.selected.is_empty()
    }));

    cx.simulate_click(first.center(), gpui::Modifiers::none());
    cx.update(|window, _| window.refresh());
    cx.run_until_parked();
    let create_person = cx.debug_bounds("#create-person-from-selected").unwrap();
    cx.simulate_click(create_person.center(), gpui::Modifiers::none());
    cx.update(|window, _| window.refresh());
    cx.run_until_parked();
    assert!(cx.update(|window, cx| { view.read(cx).people_operation_focus.is_focused(window) }));
    cx.simulate_keystrokes("escape");
    cx.update(|window, _| window.refresh());
    cx.run_until_parked();
    assert!(view.read_with(&*cx, |view, _| view.people_operation_dialog.is_none()));
    assert!(cx.update(|window, cx| { view.read(cx).people_grid_focus.is_focused(window) }));

    cx.simulate_click(first.center(), gpui::Modifiers::none());
    cx.simulate_click(
        second.center(),
        gpui::Modifiers {
            control: true,
            ..Default::default()
        },
    );
    let accept_selected = cx.debug_bounds("#accept-selected-faces").unwrap();
    cx.simulate_click(accept_selected.center(), gpui::Modifiers::none());
    cx.run_until_parked();
    assert!(cx.update(|window, cx| { view.read(cx).people_grid_focus.is_focused(window) }));
    let connection = rusqlite::Connection::open(catalog::catalog_path(root.path())).unwrap();
    for face_id in [100_i64, 101] {
        assert_eq!(
            connection
                .query_row(
                    "SELECT kind FROM face_assignments WHERE face_id=?1 AND person_id=10 AND provenance='rekesz'",
                    [face_id],
                    |row| row.get::<_, String>(0),
                )
                .unwrap(),
            "confirmed"
        );
    }
    drop(connection);

    let third = cx.debug_bounds("#people-face-tile-102-10").unwrap();
    let fourth = cx.debug_bounds("#people-face-tile-103-10").unwrap();
    cx.simulate_click(third.center(), gpui::Modifiers::none());
    cx.simulate_click(
        fourth.center(),
        gpui::Modifiers {
            control: true,
            ..Default::default()
        },
    );
    cx.update(|window, _| window.refresh());
    cx.run_until_parked();
    let reject_selected = cx.debug_bounds("#reject-selected-faces").unwrap();
    cx.simulate_click(reject_selected.center(), gpui::Modifiers::none());
    for _ in 0..20 {
        cx.run_until_parked();
        if cx.update(|window, cx| view.read(cx).people_grid_focus.is_focused(window)) {
            break;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    assert!(cx.update(|window, cx| { view.read(cx).people_grid_focus.is_focused(window) }));

    let fifth = cx.debug_bounds("#people-face-tile-104-10").unwrap();
    let sixth = cx.debug_bounds("#people-face-tile-105-10").unwrap();
    cx.simulate_click(fifth.center(), gpui::Modifiers::none());
    cx.simulate_click(
        sixth.center(),
        gpui::Modifiers {
            control: true,
            ..Default::default()
        },
    );
    cx.update(|window, _| window.refresh());
    cx.run_until_parked();
    let ignore_selected = cx.debug_bounds("#ignore-selected-faces").unwrap();
    cx.simulate_click(ignore_selected.center(), gpui::Modifiers::none());
    cx.run_until_parked();
    assert!(cx.update(|window, cx| { view.read(cx).people_grid_focus.is_focused(window) }));

    let connection = rusqlite::Connection::open(catalog::catalog_path(root.path())).unwrap();
    for face_id in [102_i64, 103] {
        assert_eq!(
            connection
                .query_row(
                    "SELECT kind FROM face_assignments WHERE face_id=?1 AND person_id=10 AND provenance='rekesz'",
                    [face_id],
                    |row| row.get::<_, String>(0),
                )
                .unwrap(),
            "rejected"
        );
    }
    for face_id in [104_i64, 105] {
        assert_eq!(
            connection
                .query_row(
                    "SELECT locally_ignored FROM faces WHERE id=?1",
                    [face_id],
                    |row| { row.get::<_, i64>(0) }
                )
                .unwrap(),
            1
        );
    }
}

#[gpui::test]
fn gpui_people_bulk_failure_rolls_back_and_can_retry_without_duplicate_submission(
    cx: &mut gpui::TestAppContext,
) {
    let root = tempfile::tempdir().unwrap();
    let cache = tempfile::tempdir().unwrap();
    let faces = seed_people_ui_catalog(root.path(), 2);
    let mut config = gpui_test_config(root.path(), cache.path());
    config.settings.automatic_face_analysis = false;
    let (view, cx) = cx.add_window_view(|_, cx| PictureBrowser::new(&config, cx));
    cx.update_entity(&view, |view, cx| {
        view.directory_view_mode = DirectoryViewMode::People;
        view.selected_person = Some(10);
        view.selected_person_name = Some("Ada".to_owned());
        view.people_faces = PeoplePageLoad::Loaded(catalog::people::FaceReviewPage {
            total: 2,
            offset: 0,
            items: faces,
        });
        cx.notify();
    });
    cx.update(|window, _| window.refresh());
    cx.run_until_parked();
    let first = cx.debug_bounds("#people-face-tile-100-10").unwrap();
    let second = cx.debug_bounds("#people-face-tile-101-10").unwrap();
    cx.simulate_click(first.center(), gpui::Modifiers::none());
    cx.simulate_click(
        second.center(),
        gpui::Modifiers {
            control: true,
            ..Default::default()
        },
    );
    cx.update(|window, _| window.refresh());
    cx.run_until_parked();

    let catalog_path = catalog::catalog_path(root.path());
    let held_catalog = root.path().join("held-bulk-catalog.sqlite3");
    std::fs::rename(&catalog_path, &held_catalog).unwrap();
    let reject = cx.debug_bounds("#reject-selected-faces").unwrap();
    cx.simulate_click(reject.center(), gpui::Modifiers::none());
    cx.simulate_click(reject.center(), gpui::Modifiers::none());
    cx.run_until_parked();
    assert!(view.read_with(&*cx, |view, _| {
        view.pending_people_decisions.is_empty()
            && view.people_decision_error.is_some()
            && matches!(
                &view.people_faces,
                PeoplePageLoad::Loaded(page)
                    if page.items.iter().all(|face| face.kind
                        == catalog::people::FaceAssignmentKind::Suggested)
            )
    }));

    std::fs::rename(&held_catalog, &catalog_path).unwrap();
    cx.update(|window, _| window.refresh());
    cx.run_until_parked();
    let reject = cx.debug_bounds("#reject-selected-faces").unwrap();
    cx.simulate_click(reject.center(), gpui::Modifiers::none());
    cx.run_until_parked();
    let connection = rusqlite::Connection::open(catalog_path).unwrap();
    assert_eq!(
        connection
            .query_row(
                "SELECT COUNT(*) FROM face_assignments WHERE person_id=10 AND kind='rejected'",
                [],
                |row| row.get::<_, i64>(0),
            )
            .unwrap(),
        2
    );
    assert_eq!(
        connection
            .query_row(
                "SELECT COUNT(DISTINCT batch_id) FROM face_decision_journal WHERE face_id IN (100, 101)",
                [],
                |row| row.get::<_, i64>(0),
            )
            .unwrap(),
        1,
        "duplicate painted submission must not create duplicate durable batches"
    );
}

#[gpui::test]
fn stale_failed_people_save_cannot_roll_back_a_newer_page(cx: &mut gpui::TestAppContext) {
    let root = tempfile::tempdir().unwrap();
    let cache = tempfile::tempdir().unwrap();
    let faces = seed_people_ui_catalog(root.path(), 2);
    let newer = faces[1].clone();
    let mut config = gpui_test_config(root.path(), cache.path());
    config.settings.automatic_face_analysis = false;
    let (view, cx) = cx.add_window_view(|_, cx| PictureBrowser::new(&config, cx));
    cx.update_entity(&view, |view, cx| {
        view.directory_view_mode = DirectoryViewMode::People;
        view.selected_person = Some(10);
        view.people_faces = PeoplePageLoad::Loaded(catalog::people::FaceReviewPage {
            total: 2,
            offset: 0,
            items: faces,
        });
        cx.notify();
    });
    let catalog_path = catalog::catalog_path(root.path());
    let held_catalog = root.path().join("held-stale-save.sqlite3");
    std::fs::rename(&catalog_path, &held_catalog).unwrap();
    cx.update_entity(&view, |view, cx| {
        assert!(view.apply_people_decision_batch(vec![(100, 10, PeopleDecision::Reject)], cx,));
        view.people_query_generation += 1;
        view.people_page = 1;
        view.people_faces = PeoplePageLoad::Loaded(catalog::people::FaceReviewPage {
            total: 2,
            offset: 30,
            items: vec![newer],
        });
        cx.notify();
    });
    cx.run_until_parked();
    assert!(view.read_with(&*cx, |view, _| {
        view.people_decision_error.is_none()
            && matches!(&view.people_faces, PeoplePageLoad::Loaded(page)
                if page.offset == 30 && page.items.len() == 1 && page.items[0].face_id == 101)
    }));
    std::fs::rename(held_catalog, catalog_path).unwrap();
}

#[gpui::test]
fn delayed_older_people_repository_query_cannot_overwrite_newer_page(
    cx: &mut gpui::TestAppContext,
) {
    let root = tempfile::tempdir().unwrap();
    let cache = tempfile::tempdir().unwrap();
    seed_people_ui_catalog(root.path(), 40);
    let mut config = gpui_test_config(root.path(), cache.path());
    config.settings.automatic_face_analysis = false;
    let (view, cx) = cx.add_window_view(|_, cx| PictureBrowser::new(&config, cx));
    *PEOPLE_QUERY_TEST_DELAY
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) =
        Some((root.path().to_path_buf(), Duration::from_millis(150)));
    cx.update_entity(&view, |view, cx| {
        view.directory_view_mode = DirectoryViewMode::People;
        view.selected_person = Some(10);
        view.people_page = 0;
        view.refresh_people_page(cx);
        view.people_page = 1;
        view.refresh_people_page(cx);
    });
    for _ in 0..300 {
        cx.run_until_parked();
        if view.read_with(&*cx, |view, _| {
            matches!(&view.people_faces, PeoplePageLoad::Loaded(page) if page.offset == 30)
        }) {
            std::thread::sleep(Duration::from_millis(175));
            cx.run_until_parked();
            break;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    assert!(view.read_with(&*cx, |view, _| {
        matches!(&view.people_faces, PeoplePageLoad::Loaded(page)
            if page.offset == 30 && page.items.len() == 10)
    }));
}

const fn ninety_people_faces() -> usize {
    90
}

fn current_process_rss_kib() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    status.lines().find_map(|line| {
        line.strip_prefix("VmRSS:")?
            .split_whitespace()
            .next()?
            .parse()
            .ok()
    })
}

#[gpui::test]
#[ignore = "tagged release-only GPUI People performance contract"]
fn people_gpui_reference_performance(cx: &mut gpui::TestAppContext) {
    const PAGE_SAMPLES: [(usize, &str); 7] = [
        (0, "#people-face-tile-100-10"),
        (1, "#people-face-tile-130-10"),
        (10, "#people-face-tile-400-10"),
        (100, "#people-face-tile-3100-10"),
        (1_000, "#people-face-tile-30100-10"),
        (2_000, "#people-face-tile-60100-10"),
        (3_332, "#people-face-tile-100060-10"),
    ];
    let root = tempfile::tempdir().unwrap();
    let cache = tempfile::tempdir().unwrap();
    seed_people_ui_catalog(root.path(), 100_000);
    let subtree = root.path().join("performance-subtree");
    std::fs::create_dir(&subtree).unwrap();
    image::RgbImage::new(320, 240)
        .save(subtree.join("overview.png"))
        .unwrap();
    {
        let mut connection =
            rusqlite::Connection::open(catalog::catalog_path(root.path())).unwrap();
        let transaction = connection.transaction().unwrap();
        transaction
            .execute(
                "INSERT INTO assets(id,relative_path,scope_key) VALUES(2,?1,?2)",
                rusqlite::params![
                    catalog::encode_relative_path(Path::new("performance-subtree/overview.png"))
                        .unwrap(),
                    catalog::scope_key(Path::new("performance-subtree/overview.png")).unwrap()
                ],
            )
            .unwrap();
        for person_id in 11_i64..=110 {
            let face_id = person_id + 100_089;
            let asset_id = if person_id <= 60 { 1_i64 } else { 2_i64 };
            transaction
                .execute(
                    "INSERT INTO people(id,display_name,created_at,updated_at)
                     VALUES(?1,?2,'2030-01-01T00:00:00Z','2030-01-01T00:00:00Z')",
                    rusqlite::params![person_id, format!("Performance person {person_id:03}")],
                )
                .unwrap();
            transaction
                .execute(
                    "INSERT INTO faces(id,asset_id,provenance,source_key,left,top,right,bottom,origin,ignored,detection_confidence,updated_at)
                     VALUES(?1,?2,'rekesz',?3,.2,.2,.8,.8,'rekesz',0,.91,datetime('now'))",
                    rusqlite::params![face_id, asset_id, format!("overview-{face_id}")],
                )
                .unwrap();
            transaction
                .execute(
                    "INSERT INTO face_assignments(face_id,person_id,kind,provenance,confidence,updated_at) VALUES(?1,?2,'confirmed','rekesz',.91,datetime('now'))",
                    rusqlite::params![face_id, person_id],
                )
                .unwrap();
        }
        transaction.commit().unwrap();
    }
    catalog::people::rebuild_people_summary(root.path(), Arc::new(AtomicBool::new(false))).unwrap();
    let mut config = gpui_test_config(root.path(), cache.path());
    config.settings.automatic_face_analysis = false;
    let (view, cx) = cx.add_window_view(|_, cx| PictureBrowser::new(&config, cx));

    // Exercise overview paging, every production sort, and both scope modes
    // before entering the 100k-face detail journey.
    for sort in [
        catalog::people::PeopleSort::Name,
        catalog::people::PeopleSort::Confirmed,
        catalog::people::PeopleSort::Review,
        catalog::people::PeopleSort::RecentlyChanged,
    ] {
        for page in [0, 3] {
            cx.update_entity(&view, |view, cx| {
                view.directory_view_mode = DirectoryViewMode::People;
                view.selected_person = None;
                view.people_special_page = None;
                view.people_sort = sort;
                view.people_page = page;
                view.people_whole_library = true;
                view.refresh_people_page(cx);
            });
            for _ in 0..300 {
                cx.run_until_parked();
                if view.read_with(&*cx, |view, _| {
                    matches!(&view.people_overview, PeoplePageLoad::Loaded(result)
                        if result.offset == (page * 30) as u64)
                }) {
                    break;
                }
                std::thread::sleep(Duration::from_millis(2));
            }
            view.read_with(&*cx, |view, _| match &view.people_overview {
                PeoplePageLoad::Loaded(result) => {
                    assert_eq!(result.offset, (page * 30) as u64);
                    assert_eq!(result.items.len(), if page == 3 { 11 } else { 30 });
                }
                PeoplePageLoad::Loading => panic!("overview {sort:?}/{page} remained loading"),
                PeoplePageLoad::Error(error) => panic!("overview {sort:?}/{page} failed: {error}"),
            });
            cx.update(|window, _| window.refresh());
            cx.run_until_parked();
            assert!(view.read_with(&*cx, |view, _| {
                !view.people_rendered_overview_indices.is_empty()
                    && view.people_rendered_overview_indices.len() < 30
            }));
        }
    }

    let subtree_node = cx.update_entity(&view, |view, _| view.model.ensure_node(subtree.clone()));
    for sort in [
        catalog::people::PeopleSort::Name,
        catalog::people::PeopleSort::Confirmed,
        catalog::people::PeopleSort::Review,
        catalog::people::PeopleSort::RecentlyChanged,
    ] {
        for page in [0, 1] {
            cx.update_entity(&view, |view, cx| {
                view.selected = subtree_node;
                view.selected_person = None;
                view.people_special_page = None;
                view.people_sort = sort;
                view.people_page = page;
                view.people_whole_library = false;
                view.refresh_people_page(cx);
            });
            for _ in 0..300 {
                cx.run_until_parked();
                if view.read_with(&*cx, |view, _| {
                    matches!(&view.people_overview, PeoplePageLoad::Loaded(result)
                        if result.offset == (page * 30) as u64)
                }) {
                    break;
                }
                std::thread::sleep(Duration::from_millis(2));
            }
            view.read_with(&*cx, |view, _| match &view.people_overview {
                PeoplePageLoad::Loaded(result) => {
                    assert_eq!(result.total, 50, "wrong {sort:?} subtree total");
                    assert_eq!(result.offset, (page * 30) as u64);
                    assert_eq!(result.items.len(), if page == 0 { 30 } else { 20 });
                    let expected_start = if page == 0 { 61_i64 } else { 91_i64 };
                    assert_eq!(
                        result
                            .items
                            .iter()
                            .map(|person| person.id)
                            .collect::<Vec<_>>(),
                        (expected_start..expected_start + result.items.len() as i64)
                            .collect::<Vec<_>>(),
                        "wrong {sort:?} subtree order/page"
                    );
                    assert!(result.items.iter().all(|person| {
                        person.confirmed_faces == 1 && person.suggested_faces == 0
                    }));
                }
                PeoplePageLoad::Loading => {
                    panic!("subtree overview {sort:?} page {page} remained loading")
                }
                PeoplePageLoad::Error(error) => {
                    panic!("subtree overview {sort:?} page {page} failed: {error}")
                }
            });
        }
    }

    cx.update_entity(&view, |view, cx| {
        view.people_whole_library = true;
        view.people_sort = catalog::people::PeopleSort::Name;
        view.people_page = 0;
        view.refresh_people_page(cx);
    });
    for _ in 0..300 {
        cx.run_until_parked();
        if view.read_with(&*cx, |view, _| {
            matches!(&view.people_overview, PeoplePageLoad::Loaded(result)
                if result.offset == 0 && result.total == 101)
        }) {
            break;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    assert!(view.read_with(&*cx, |view, _| {
        matches!(&view.people_overview, PeoplePageLoad::Loaded(result)
            if result.total == 101 && result.items.len() == 30)
    }));

    let skeleton_started = std::time::Instant::now();
    cx.update_entity(&view, |view, cx| {
        view.directory_view_mode = DirectoryViewMode::People;
        view.people_whole_library = true;
        view.selected_person = Some(10);
        view.selected_person_name = Some("Performance person".to_owned());
        view.people_faces = PeoplePageLoad::Loading;
        cx.notify();
    });
    cx.update(|window, _| window.refresh());
    cx.run_until_parked();
    assert!(cx.debug_bounds("#people-back").is_some());
    let skeleton_ms = skeleton_started.elapsed().as_secs_f64() * 1000.0;
    assert!(
        skeleton_ms < 100.0,
        "first painted skeleton exceeded 100 ms"
    );

    cx.update_entity(&view, |view, cx| view.refresh_people_page(cx));

    for _ in 0..200 {
        cx.run_until_parked();
        if view.read_with(&*cx, |view, _| {
            matches!(&view.people_faces, PeoplePageLoad::Loaded(page) if page.items.len() == 30)
        }) {
            break;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    let mut max_frame_ms = 0.0_f64;
    let mut max_rendered = 0_usize;
    let mut max_demanded = 0_usize;
    let mut max_loaded_cache = 0_usize;
    let rss_before_scroll_kib = current_process_rss_kib();
    let mut max_scroll_rss_kib = rss_before_scroll_kib.unwrap_or_default();
    for (page, first_tile_selector) in PAGE_SAMPLES {
        cx.update_entity(&view, |view, cx| {
            view.people_page = page;
            view.refresh_people_page(cx);
        });
        for _ in 0..200 {
            cx.run_until_parked();
            if view.read_with(&*cx, |view, _| {
                matches!(&view.people_faces, PeoplePageLoad::Loaded(result) if result.offset == (page * 30) as u64)
            }) {
                break;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        view.read_with(&*cx, |view, _| match &view.people_faces {
            PeoplePageLoad::Loaded(result) => {
                assert_eq!(
                    result.offset,
                    (page * 30) as u64,
                    "page {page} loaded wrong offset"
                );
                assert_eq!(
                    result.items.len(),
                    30,
                    "page {page} loaded a partial result"
                );
            }
            PeoplePageLoad::Loading => panic!("page {page} remained loading after the deadline"),
            PeoplePageLoad::Error(error) => panic!("page {page} failed to load: {error}"),
        });
        let frame_started = std::time::Instant::now();
        cx.update(|window, _| window.refresh());
        cx.run_until_parked();
        max_frame_ms = max_frame_ms.max(frame_started.elapsed().as_secs_f64() * 1000.0);
        let (rendered, demanded, loaded_cache) = view.read_with(&*cx, |view, _| {
            (
                view.people_rendered_face_indices.len(),
                view.scheduler
                    .snapshot()
                    .jobs
                    .iter()
                    .filter(|job| {
                        matches!(
                            job.key.variant,
                            RenderVariant::FaceCrop { .. } | RenderVariant::FaceContext { .. }
                        ) && !job.cancelled
                    })
                    .count(),
                view.thumbnails.stats().loaded,
            )
        });
        max_rendered = max_rendered.max(rendered);
        max_demanded = max_demanded.max(demanded);
        max_loaded_cache = max_loaded_cache.max(loaded_cache);
        max_scroll_rss_kib = max_scroll_rss_kib.max(current_process_rss_kib().unwrap_or_default());
        assert!(
            rendered > 0,
            "page {page} painted no virtualized face indices"
        );
        assert!(rendered < 30);
        assert!(demanded <= rendered);
        assert!(
            demanded > 0 || loaded_cache > 0,
            "page {page} neither demanded nor loaded a visible crop"
        );
        assert!(
            cx.debug_bounds(first_tile_selector).is_some(),
            "page {page} painted no first visible tile"
        );
        let off_page_face_id = if page == 0 { 130 } else { 100 };
        assert!(
            view.read_with(&*cx, |view, _| view.scheduler.snapshot().jobs.iter().all(|job| {
                !matches!(job.key.variant, RenderVariant::FaceCrop { face_id, .. } if face_id == off_page_face_id)
                    || job.cancelled
            })),
            "page {page} retained active off-page crop demand"
        );
    }
    // Cross enough distinct virtual pages to exceed the bounded preview cache,
    // then revisit old pages. This catches caches that look bounded in a short
    // smoke test but grow with scroll distance.
    let mut long_scroll_loaded = Vec::new();
    for page in (0..3_300).step_by(200) {
        cx.update_entity(&view, |view, cx| {
            view.people_page = page;
            view.refresh_people_page(cx);
        });
        for _ in 0..200 {
            cx.run_until_parked();
            if view.read_with(&*cx, |view, _| {
                matches!(&view.people_faces, PeoplePageLoad::Loaded(result)
                    if result.offset == (page * 30) as u64)
            }) {
                break;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        cx.update(|window, _| window.refresh());
        cx.run_until_parked();
        let loaded = view.read_with(&*cx, |view, _| view.thumbnails.stats().loaded);
        long_scroll_loaded.push(loaded);
        max_loaded_cache = max_loaded_cache.max(loaded);
        max_scroll_rss_kib = max_scroll_rss_kib.max(current_process_rss_kib().unwrap_or_default());
    }
    let cache_capacity = view.read_with(&*cx, |view, _| view.thumbnails.stats().capacity);
    let midpoint = long_scroll_loaded.len() / 2;
    let early_peak = long_scroll_loaded[..midpoint]
        .iter()
        .copied()
        .max()
        .unwrap_or_default();
    let late_peak = long_scroll_loaded[midpoint..]
        .iter()
        .copied()
        .max()
        .unwrap_or_default();
    assert!(early_peak > 0);
    assert!(
        late_peak <= early_peak.saturating_add(max_rendered),
        "loaded preview count grew with long-distance scroll: {early_peak} -> {late_peak}"
    );
    assert!(max_loaded_cache <= cache_capacity);
    for page in [0, 1, 2] {
        cx.update_entity(&view, |view, cx| {
            view.people_page = page;
            view.refresh_people_page(cx);
        });
        for _ in 0..200 {
            cx.run_until_parked();
            if view.read_with(&*cx, |view, _| {
                matches!(&view.people_faces, PeoplePageLoad::Loaded(result)
                    if result.offset == (page * 30) as u64)
            }) {
                break;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        cx.update(|window, _| window.refresh());
        cx.run_until_parked();
        assert!(view.read_with(&*cx, |view, _| {
            view.thumbnails.stats().loaded <= view.thumbnails.stats().capacity
        }));
    }
    assert!(
        max_frame_ms < 8.0,
        "painted steady frame exceeded 8 ms: {max_frame_ms:.3} ms"
    );
    assert!(max_loaded_cache <= view.read_with(&*cx, |view, _| view.thumbnails.stats().capacity));
    let scroll_rss_growth_kib = rss_before_scroll_kib
        .map(|before| max_scroll_rss_kib.saturating_sub(before))
        .unwrap_or_default();
    assert!(
        scroll_rss_growth_kib <= 65_536,
        "widely spaced page/paint cycles grew RSS by more than 64 MiB"
    );

    let first = cx.debug_bounds("#people-face-tile-160-10").unwrap();
    let second = cx.debug_bounds("#people-face-tile-161-10").unwrap();
    cx.simulate_click(first.center(), gpui::Modifiers::none());
    cx.simulate_click(
        second.center(),
        gpui::Modifiers {
            control: true,
            ..Default::default()
        },
    );
    assert_eq!(
        view.read_with(&*cx, |view, _| view.people_face_selection.selected.len()),
        2,
        "painted multiselect failed at 100k scale"
    );

    cx.update_entity(&view, |view, cx| {
        if let PeoplePageLoad::Loaded(page) = &view.people_faces {
            view.people_face_selection.selected = page
                .items
                .iter()
                .map(|face| PeopleFaceKey {
                    face_id: face.face_id,
                    person_id: face.person_id,
                })
                .collect();
        }
        view.apply_selected_people_decision(PeopleDecision::Confirm, cx);
    });
    let bulk_started = std::time::Instant::now();
    for _ in 0..500 {
        cx.run_until_parked();
        if view.read_with(&*cx, |view, _| view.pending_people_decisions.is_empty()) {
            break;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    let bulk_ms = bulk_started.elapsed().as_secs_f64() * 1000.0;
    assert!(bulk_ms < 1_000.0, "30-row durable bulk exceeded one second");
    assert!(view.read_with(&*cx, |view, _| {
        view.pending_people_decisions.is_empty() && view.people_decision_error.is_none()
    }));
    let connection = rusqlite::Connection::open(catalog::catalog_path(root.path())).unwrap();
    assert_eq!(
        connection
            .query_row(
                "SELECT COUNT(*) FROM face_assignments WHERE person_id=10 AND kind='confirmed'",
                [],
                |row| row.get::<_, i64>(0),
            )
            .unwrap(),
        30
    );
    assert_eq!(
        connection
            .query_row("SELECT COUNT(*) FROM face_decision_journal", [], |row| row
                .get::<_, i64>(
                0
            ),)
            .unwrap(),
        30
    );
    assert_eq!(
        connection
            .query_row(
                "SELECT COUNT(DISTINCT batch_id) FROM face_decision_journal",
                [],
                |row| row.get::<_, i64>(0),
            )
            .unwrap(),
        1
    );
    println!(
        "{{\"face_count\":100100,\"overview_people\":101,\"skeleton_ms\":{skeleton_ms:.3},\"max_frame_ms\":{max_frame_ms:.3},\"max_rendered\":{max_rendered},\"max_demanded\":{max_demanded},\"max_loaded_cache\":{max_loaded_cache},\"scroll_rss_growth_kib\":{scroll_rss_growth_kib},\"bulk_30_ms\":{bulk_ms:.3}}}"
    );
}

#[gpui::test]
fn gpui_people_accept_reject_and_not_face_persist_across_restart_and_rollback(
    cx: &mut gpui::TestAppContext,
) {
    let root = tempfile::tempdir().unwrap();
    let cache = tempfile::tempdir().unwrap();
    let faces = seed_people_ui_catalog(root.path(), 3);
    let mut config = gpui_test_config(root.path(), cache.path());
    config.settings.automatic_face_analysis = false;
    let (view, cx) = cx.add_window_view(|_, cx| PictureBrowser::new(&config, cx));
    cx.update_entity(&view, |view, cx| {
        view.directory_view_mode = DirectoryViewMode::People;
        view.selected_person = Some(10);
        view.selected_person_name = Some("Ada".to_owned());
        view.people_faces = PeoplePageLoad::Loaded(catalog::people::FaceReviewPage {
            total: 3,
            offset: 0,
            items: faces,
        });
        cx.notify();
    });
    cx.run_until_parked();
    for selector in [
        "#accept-face-100-10",
        "#reject-face-101-10",
        "#ignore-face-102-10",
    ] {
        let bounds = cx.debug_bounds(selector).unwrap();
        cx.simulate_click(bounds.center(), gpui::Modifiers::none());
        cx.run_until_parked();
    }
    let connection = rusqlite::Connection::open(catalog::catalog_path(root.path())).unwrap();
    let kind = |face_id| {
        connection
            .query_row(
                "SELECT kind FROM face_assignments WHERE face_id=?1 AND provenance='rekesz'",
                [face_id],
                |row| row.get::<_, String>(0),
            )
            .optional()
            .unwrap()
    };
    assert_eq!(kind(100).as_deref(), Some("confirmed"));
    assert_eq!(kind(101).as_deref(), Some("rejected"));
    assert!(
        matches!(kind(102).as_deref(), None | Some("suggested")),
        "recognition may asynchronously remove its now-ineligible cache row"
    );
    assert_eq!(
        connection
            .query_row(
                "SELECT locally_ignored FROM faces WHERE id=102",
                [],
                |row| { row.get::<_, i64>(0) }
            )
            .unwrap(),
        1
    );
    drop(connection);

    let persisted = catalog::people::list_person_faces_page(root.path(), 10, None, 0, 30).unwrap();
    assert!(persisted.items.iter().any(|face| {
        face.face_id == 100 && face.kind == catalog::people::FaceAssignmentKind::Confirmed
    }));
    assert!(
        persisted.items.iter().all(|face| face.face_id != 101),
        "a rejected identity must remain absent after reopening the catalog"
    );
    assert!(
        persisted.items.iter().all(|face| face.face_id != 102),
        "a locally ignored face must remain absent after reopening the catalog"
    );

    let grid = cx.debug_bounds("#show-grid-view").unwrap();
    cx.simulate_click(grid.center(), gpui::Modifiers::none());
    let people = cx.debug_bounds("#show-people-view").unwrap();
    cx.simulate_click(people.center(), gpui::Modifiers::none());
    cx.run_until_parked();
    assert!(view.read_with(&*cx, |view, _| {
        matches!(
            &view.people_faces,
            PeoplePageLoad::Loaded(page)
                if page.items.iter().any(|face| face.face_id == 100
                    && face.kind == catalog::people::FaceAssignmentKind::Confirmed)
                    && page.items.iter().all(|face| face.face_id != 101 && face.face_id != 102)
        )
    }));

    // Recreate the actual GPUI state and query the catalog again, rather than
    // treating a repository read as a restart test.
    let (restarted, restarted_cx) = cx.add_window_view(|_, cx| PictureBrowser::new(&config, cx));
    restarted_cx.update_entity(&restarted, |view, cx| {
        view.directory_view_mode = DirectoryViewMode::People;
        view.selected_person = Some(10);
        view.selected_person_name = Some("Ada".to_owned());
        view.refresh_people_page(cx);
    });
    restarted_cx.run_until_parked();
    assert!(restarted.read_with(&*restarted_cx, |view, _| {
        matches!(
            &view.people_faces,
            PeoplePageLoad::Loaded(page)
                if page.items.iter().any(|face| face.face_id == 100
                    && face.kind == catalog::people::FaceAssignmentKind::Confirmed)
                    && page.items.iter().all(|face| face.face_id != 101 && face.face_id != 102)
        )
    }));

    // A failed save must restore the optimistic row state. The path is moved only
    // inside this test's temporary directory, never a user catalog.
    let catalog_path = catalog::catalog_path(root.path());
    let held_catalog = root.path().join("held-catalog.sqlite3");
    std::fs::rename(&catalog_path, &held_catalog).unwrap();
    cx.update_entity(&view, |view, cx| {
        if let PeoplePageLoad::Loaded(page) = &mut view.people_faces {
            page.items[0].kind = catalog::people::FaceAssignmentKind::Suggested;
        }
        view.apply_people_decision(100, 10, PeopleDecision::Reject, cx);
    });
    cx.run_until_parked();
    assert!(view.read_with(&*cx, |view, _| {
        matches!(
            &view.people_faces,
            PeoplePageLoad::Loaded(page)
                if page.items[0].kind == catalog::people::FaceAssignmentKind::Suggested
                    && view.people_decision_error.is_some()
        )
    }));
    std::fs::rename(held_catalog, catalog_path).unwrap();
}

#[gpui::test]
fn gpui_people_keyboard_only_view_person_and_back_journey(cx: &mut gpui::TestAppContext) {
    let root = tempfile::tempdir().unwrap();
    let cache = tempfile::tempdir().unwrap();
    seed_people_ui_catalog(root.path(), 3);
    let mut config = gpui_test_config(root.path(), cache.path());
    config.settings.automatic_face_analysis = false;
    let (view, cx) = cx.add_window_view(|_, cx| PictureBrowser::new(&config, cx));
    cx.update(|window, cx| view.read(cx).people_view_switch_focus.focus(window));
    cx.simulate_keystrokes("shift-tab tab enter");
    for _ in 0..200 {
        cx.run_until_parked();
        if view.read_with(&*cx, |view, _| {
            matches!(view.people_overview, PeoplePageLoad::Loaded(_))
        }) {
            break;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    assert_eq!(
        view.read_with(&*cx, |view, _| view.directory_view_mode),
        DirectoryViewMode::People
    );
    cx.update(|window, cx| view.read(cx).people_primary_card_focus.focus(window));
    cx.simulate_keystrokes("shift-tab tab space");
    for _ in 0..200 {
        cx.run_until_parked();
        if view.read_with(&*cx, |view, _| {
            matches!(view.people_faces, PeoplePageLoad::Loaded(_))
        }) {
            break;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    assert_eq!(
        view.read_with(&*cx, |view, _| view.selected_person),
        Some(10)
    );
    assert!(cx.update(|window, cx| view.read(cx).people_grid_focus.is_focused(window)));
    cx.update(|window, cx| view.read(cx).people_back_focus.focus(window));
    cx.simulate_keystrokes("shift-tab tab enter");
    for _ in 0..200 {
        cx.run_until_parked();
        if view.read_with(&*cx, |view, _| view.selected_person.is_none()) {
            break;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    assert!(view.read_with(&*cx, |view, _| view.selected_person.is_none()));
}

#[gpui::test]
fn gpui_people_accessibility_actions_match_pointer_and_keyboard_state(
    cx: &mut gpui::TestAppContext,
) {
    let root = tempfile::tempdir().unwrap();
    let cache = tempfile::tempdir().unwrap();
    seed_people_ui_catalog(root.path(), 3);
    let mut config = gpui_test_config(root.path(), cache.path());
    config.settings.automatic_face_analysis = false;
    let (view, cx) = cx.add_window_view(|_, cx| PictureBrowser::new(&config, cx));
    cx.update_entity(&view, |view, cx| {
        view.directory_view_mode = DirectoryViewMode::People;
        view.selected_person = Some(10);
        view.selected_person_name = Some("Ada".to_owned());
        view.refresh_people_page(cx);
    });
    for _ in 0..100 {
        cx.run_until_parked();
        if view.read_with(&*cx, |view, _| {
            matches!(&view.people_faces, PeoplePageLoad::Loaded(page) if !page.items.is_empty())
        }) {
            break;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    cx.update(|window, _| window.refresh());
    cx.run_until_parked();
    assert!(view.read_with(&*cx, |view, _| {
        let snapshot = view.people_accessibility.snapshot();
        let labels = snapshot
            .update
            .nodes
            .iter()
            .filter_map(|(_, node)| node.label())
            .collect::<HashSet<_>>();
        labels.contains("Back to People")
            && labels.contains("Previous People page")
            && labels.contains("Next People page")
            && labels.contains("Rename person")
            && labels.contains("Merge person")
            && labels.iter().any(|label| label.starts_with("Open "))
            && labels.iter().any(|label| label.starts_with("Mark "))
    }));
    view.read_with(&*cx, |view, _| {
        view.people_accessibility.request_click(
            accessibility::PeopleAccessibilityAction::ToggleFaceContext {
                face_id: 100,
                person_id: 10,
            },
        );
    });
    cx.update(|window, _| window.refresh());
    cx.run_until_parked();
    assert!(view.read_with(&*cx, |view, _| {
        view.people_face_uses_context(PeopleFaceKey {
            face_id: 100,
            person_id: 10,
        })
    }));

    view.read_with(&*cx, |view, _| {
        view.people_accessibility.request_click(
            accessibility::PeopleAccessibilityAction::SelectFace {
                face_id: 100,
                person_id: 10,
            },
        );
    });
    cx.update(|window, _| window.refresh());
    cx.run_until_parked();
    assert!(view.read_with(&*cx, |view, _| {
        view.people_face_selection
            .selected
            .contains(&PeopleFaceKey {
                face_id: 100,
                person_id: 10,
            })
    }));

    // A mixed pending/non-pending selection advertises an enabled-but-busy
    // bulk action bound only to the eligible subset, just like the painted
    // toolbar's save coordinator.
    cx.update_entity(&view, |view, cx| {
        view.people_face_selection.selected.insert(PeopleFaceKey {
            face_id: 101,
            person_id: 10,
        });
        view.pending_people_decisions.insert((100, 10));
        cx.notify();
    });
    cx.update(|window, _| window.refresh());
    cx.run_until_parked();
    assert!(view.read_with(&*cx, |view, _| {
        let snapshot = view.people_accessibility.snapshot();
        let accept = snapshot
            .update
            .nodes
            .iter()
            .map(|(_, node)| node)
            .find(|node| node.label() == Some("Accept selected faces"))
            .unwrap();
        accept.is_busy()
            && !accept.is_disabled()
            && accept.supports_action(accesskit::Action::Click)
    }));
    view.read_with(&*cx, |view, _| {
        view.people_accessibility.request_click(
            accessibility::PeopleAccessibilityAction::BulkDecision {
                decision: PeopleDecision::Confirm,
                faces: vec![(101, 10)],
            },
        );
    });
    cx.update(|window, _| window.refresh());
    for _ in 0..100 {
        cx.run_until_parked();
        if view.read_with(&*cx, |view, _| {
            matches!(&view.people_faces, PeoplePageLoad::Loaded(page) if page.items.iter().any(|face| face.face_id == 101 && face.kind == catalog::people::FaceAssignmentKind::Confirmed))
        }) {
            break;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    assert!(view.read_with(&*cx, |view, _| {
        matches!(&view.people_faces, PeoplePageLoad::Loaded(page) if page.items.iter().any(|face| face.face_id == 101 && face.kind == catalog::people::FaceAssignmentKind::Confirmed))
    }));
    cx.update_entity(&view, |view, _| {
        view.pending_people_decisions.remove(&(100, 10));
    });

    view.read_with(&*cx, |view, _| {
        view.people_accessibility
            .request_click(accessibility::PeopleAccessibilityAction::RenamePerson(10));
    });
    cx.update(|window, _| window.refresh());
    cx.run_until_parked();
    assert!(view.read_with(&*cx, |view, _| matches!(
        view.people_operation_dialog,
        Some(PeopleOperationDialog::Name { .. })
    )));
    view.read_with(&*cx, |view, _| {
        view.people_accessibility
            .request_click(accessibility::PeopleAccessibilityAction::CancelDialog);
    });
    cx.update(|window, _| window.refresh());
    cx.run_until_parked();
    assert!(view.read_with(&*cx, |view, _| view.people_operation_dialog.is_none()));

    // Disabled paging requests and stale/not-current actions are ignored.
    view.read_with(&*cx, |view, _| {
        view.people_accessibility
            .request_click(accessibility::PeopleAccessibilityAction::NextPage);
    });
    cx.update(|window, _| window.refresh());
    cx.run_until_parked();
    assert_eq!(view.read_with(&*cx, |view, _| view.people_page), 0);

    // A request from an older semantic tree cannot mutate the replacement
    // page, even when its stable face key exists there too.
    view.read_with(&*cx, |view, _| {
        view.people_accessibility.request_click(
            accessibility::PeopleAccessibilityAction::AcceptFace {
                face_id: 100,
                person_id: 10,
            },
        );
    });
    cx.update_entity(&view, |view, _| view.people_query_generation += 1);
    cx.update(|window, _| window.refresh());
    cx.run_until_parked();
    assert!(view.read_with(&*cx, |view, _| {
        !view.pending_people_decisions.contains(&(100, 10))
    }));

    view.read_with(&*cx, |view, _| {
        view.people_accessibility.request_click(
            accessibility::PeopleAccessibilityAction::MarkNotFace {
                face_id: 100,
                person_id: 10,
            },
        );
    });
    cx.update(|window, _| window.refresh());
    for _ in 0..100 {
        cx.run_until_parked();
        if view.read_with(&*cx, |view, _| {
            matches!(&view.people_faces, PeoplePageLoad::Loaded(page) if page.items.iter().all(|face| face.face_id != 100))
        }) {
            break;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    let not_face_state = view.read_with(&*cx, |view, _| {
        (
            match &view.people_faces {
                PeoplePageLoad::Loaded(page) => page
                    .items
                    .iter()
                    .map(|face| (face.face_id, face.kind))
                    .collect::<Vec<_>>(),
                _ => Vec::new(),
            },
            view.pending_people_decisions.clone(),
            view.people_decision_error.clone(),
        )
    });
    assert!(
        not_face_state.0.iter().any(|(face_id, kind)| {
            *face_id == 100 && *kind == catalog::people::FaceAssignmentKind::Ignored
        }),
        "not-a-face accessibility dispatch did not settle: {not_face_state:?}"
    );

    cx.update_entity(&view, |view, cx| {
        view.people_faces = PeoplePageLoad::Error(Arc::from("synthetic retry state"));
        cx.notify();
    });
    cx.update(|window, _| window.refresh());
    cx.run_until_parked();
    view.read_with(&*cx, |view, _| {
        view.people_accessibility
            .request_click(accessibility::PeopleAccessibilityAction::RetryPage);
    });
    cx.update(|window, _| window.refresh());
    for _ in 0..100 {
        cx.run_until_parked();
        if view.read_with(&*cx, |view, _| {
            matches!(view.people_faces, PeoplePageLoad::Loaded(_))
        }) {
            break;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    assert!(view.read_with(&*cx, |view, _| matches!(
        view.people_faces,
        PeoplePageLoad::Loaded(_)
    )));

    // Opening a photo through AT leaves People, unregisters AT-SPI, and
    // discards another request queued in the same frame.
    let (photo, photo_face, photo_person) =
        view.read_with(&*cx, |view, _| match &view.people_faces {
            PeoplePageLoad::Loaded(page) => (
                page.items[0].path.clone(),
                page.items[0].face_id,
                page.items[0].person_id,
            ),
            _ => unreachable!(),
        });
    let context_before_navigation = view.read_with(&*cx, |view, _| {
        view.people_face_uses_context(PeopleFaceKey {
            face_id: photo_face,
            person_id: photo_person,
        })
    });
    view.read_with(&*cx, |view, _| {
        view.people_accessibility.request_click(
            accessibility::PeopleAccessibilityAction::OpenPhoto(photo.clone()),
        );
        view.people_accessibility.request_click(
            accessibility::PeopleAccessibilityAction::ToggleFaceContext {
                face_id: photo_face,
                person_id: photo_person,
            },
        );
    });
    cx.update(|window, _| window.refresh());
    cx.run_until_parked();
    cx.update(|window, _| window.refresh());
    cx.run_until_parked();
    assert!(view.read_with(&*cx, |view, _| matches!(view.main_view, MainView::Image(_))));
    assert_eq!(
        view.read_with(&*cx, |view, _| {
            view.people_face_uses_context(PeopleFaceKey {
                face_id: photo_face,
                person_id: photo_person,
            })
        }),
        context_before_navigation,
        "a trailing same-frame People action must not mutate state after Open photo"
    );
    assert!(view.read_with(&*cx, |view, _| {
        view.people_accessibility.take_actions().is_empty()
    }));
    #[cfg(target_os = "linux")]
    assert!(view.read_with(&*cx, |view, _| !view.people_accessibility.adapter_active()));
}

#[gpui::test]
fn gpui_empty_map_offers_and_starts_recursive_mode(cx: &mut gpui::TestAppContext) {
    let root = tempfile::tempdir().unwrap();
    std::fs::create_dir(root.path().join("trip")).unwrap();
    let cache = tempfile::tempdir().unwrap();
    let config = gpui_test_config(root.path(), cache.path());
    let (view, cx) = cx.add_window_view(|_, cx| PictureBrowser::new(&config, cx));
    cx.run_until_parked();
    cx.update_entity(&view, |view, cx| {
        view.directory_view_mode = DirectoryViewMode::Map;
        cx.notify();
    });
    cx.run_until_parked();
    let button = cx.debug_bounds("#map-show-images-recursively").unwrap();
    cx.simulate_click(button.center(), gpui::Modifiers::none());
    assert!(view.read_with(cx, |view, _| view.recursive_grid.is_some()));
}

#[gpui::test]
fn gpui_outer_zoom_cluster_click_opens_all_cluster_photos(cx: &mut gpui::TestAppContext) {
    let root = tempfile::tempdir().unwrap();
    for (name, latitude, longitude) in
        [("one", "47.500N", "19.050E"), ("two", "47.501N", "19.051E")]
    {
        std::fs::write(root.path().join(format!("{name}.jpg")), b"").unwrap();
        std::fs::write(
            root.path().join(format!("{name}.xmp")),
            format!(
                "<rdf:Description exif:GPSLatitude=\"{latitude}\" exif:GPSLongitude=\"{longitude}\"/>"
            ),
        )
        .unwrap();
    }
    let cache = tempfile::tempdir().unwrap();
    let config = gpui_test_config(root.path(), cache.path());
    let (view, cx) = cx.add_window_view(|_, cx| PictureBrowser::new(&config, cx));
    cx.run_until_parked();
    cx.update_entity(&view, |view, cx| {
        view.directory_view_mode = DirectoryViewMode::Map;
        view.map_camera = map::Camera {
            center: metadata::GeoCoordinate::new(47.5, 19.05).unwrap(),
            zoom: 2.0,
        };
        view.map_pending_fit = None;
        view.map_fit_node = None;
        cx.notify();
    });
    cx.run_until_parked();

    let marker = cx.debug_bounds("#map-marker-0").unwrap();
    cx.simulate_click(marker.center(), gpui::Modifiers::none());
    assert_eq!(
        view.read_with(cx, |view, _| view.map_card.as_ref().unwrap().items.len()),
        2
    );
    assert!(cx.debug_bounds("#map-card-zoom").is_some());
    let thumbnail = cx.debug_bounds("#map-gallery-item-0").unwrap();
    cx.simulate_click(thumbnail.center(), gpui::Modifiers::none());
    assert!(view.read_with(cx, |view, _| matches!(view.main_view, MainView::Image(_))));
}

#[gpui::test]
fn gpui_map_surface_drag_and_wheel_update_camera_during_real_events(cx: &mut gpui::TestAppContext) {
    let root = tempfile::tempdir().unwrap();
    let cache = tempfile::tempdir().unwrap();
    write_photo_stub(root.path(), "photo", 47.5, 19.05);
    let config = gpui_test_config(root.path(), cache.path());
    let (view, cx) = cx.add_window_view(|_, cx| PictureBrowser::new(&config, cx));
    cx.run_until_parked();
    cx.update_entity(&view, |view, cx| {
        view.directory_view_mode = DirectoryViewMode::Map;
        view.map_camera = map::Camera {
            center: metadata::GeoCoordinate::new(47.5, 19.05).unwrap(),
            zoom: 5.0,
        };
        view.map_pending_fit = None;
        view.map_fit_node = None;
        cx.notify();
    });
    cx.run_until_parked();
    let surface = cx.debug_bounds("#map-view").unwrap();
    let start = point(surface.origin.x + px(40.0), surface.origin.y + px(40.0));
    let end = point(start.x + px(80.0), start.y + px(40.0));
    let before = view.read_with(&*cx, |view, _| view.map_camera);
    cx.simulate_mouse_down(start, MouseButton::Left, gpui::Modifiers::none());
    cx.simulate_mouse_move(end, MouseButton::Left, gpui::Modifiers::none());
    let during = view.read_with(&*cx, |view, _| view.map_camera);
    assert_ne!(during.center, before.center);
    cx.simulate_mouse_up(end, MouseButton::Left, gpui::Modifiers::none());
    assert!(view.read_with(&*cx, |view, _| view.map_drag_last.is_none()));

    cx.simulate_event(ScrollWheelEvent {
        position: end,
        delta: gpui::ScrollDelta::Lines(point(0.0, -3.0)),
        modifiers: gpui::Modifiers::none(),
        touch_phase: gpui::TouchPhase::Moved,
    });
    assert!(view.read_with(&*cx, |view, _| view.map_camera.zoom > during.zoom));
}

#[gpui::test]
fn gpui_map_switcher_consent_and_zoom_controls_drive_the_painted_ui(cx: &mut gpui::TestAppContext) {
    let root = tempfile::tempdir().unwrap();
    let cache = tempfile::tempdir().unwrap();
    write_photo_stub(root.path(), "photo", 47.5, 19.05);
    let mut config = gpui_test_config(root.path(), cache.path());
    config.settings.online_map_permission = config::OnlineMapPermission::Unknown;
    let (view, cx) = cx.add_window_view(|_, cx| PictureBrowser::new(&config, cx));
    cx.run_until_parked();

    let show_map = cx.debug_bounds("#show-map-view").unwrap();
    cx.simulate_click(show_map.center(), gpui::Modifiers::none());
    cx.run_until_parked();
    assert_eq!(
        view.read_with(&*cx, |view, _| view.directory_view_mode),
        DirectoryViewMode::Map
    );
    let offline = cx.debug_bounds("#keep-map-offline").unwrap();
    cx.simulate_click(offline.center(), gpui::Modifiers::none());
    assert_eq!(
        view.read_with(&*cx, |view, _| view.settings.online_map_permission),
        config::OnlineMapPermission::Disabled
    );

    let before = view.read_with(&*cx, |view, _| view.map_camera.zoom);
    let zoom_in = cx.debug_bounds("#map-zoom-in").unwrap();
    cx.simulate_mouse_down(zoom_in.center(), MouseButton::Left, gpui::Modifiers::none());
    assert!(view.read_with(&*cx, |view, _| view.map_camera.zoom > before));
    let zoomed = view.read_with(&*cx, |view, _| view.map_camera.zoom);
    let fit = cx.debug_bounds("#map-fit").unwrap();
    cx.simulate_mouse_down(fit.center(), MouseButton::Left, gpui::Modifiers::none());
    assert_ne!(view.read_with(&*cx, |view, _| view.map_camera.zoom), zoomed);

    let grid = cx.debug_bounds("#show-grid-view").unwrap();
    cx.simulate_click(grid.center(), gpui::Modifiers::none());
    assert_eq!(
        view.read_with(&*cx, |view, _| view.directory_view_mode),
        DirectoryViewMode::Grid
    );
}

#[gpui::test]
fn gpui_real_large_cluster_fetches_and_paints_the_second_gallery_page(
    cx: &mut gpui::TestAppContext,
) {
    let root = tempfile::tempdir().unwrap();
    let cache = tempfile::tempdir().unwrap();
    for index in 0..7 {
        write_photo_stub(root.path(), &format!("photo-{index}"), 47.5, 19.05);
    }
    let config = gpui_test_config(root.path(), cache.path());
    let (view, cx) = cx.add_window_view(|_, cx| PictureBrowser::new(&config, cx));
    cx.run_until_parked();
    cx.update_entity(&view, |view, cx| {
        view.directory_view_mode = DirectoryViewMode::Map;
        view.map_camera.zoom = 2.0;
        view.map_pending_fit = None;
        view.map_fit_node = None;
        cx.notify();
    });
    cx.run_until_parked();
    let marker = cx.debug_bounds("#map-marker-0").unwrap();
    cx.simulate_click(marker.center(), gpui::Modifiers::none());
    let first_page = view.read_with(&*cx, |view, _| {
        view.map_card
            .as_ref()
            .unwrap()
            .page_items()
            .iter()
            .map(|item| item.path.clone())
            .collect::<Vec<_>>()
    });
    assert_eq!(first_page.len(), 6);
    let next = cx.debug_bounds("#map-card-next").unwrap();
    cx.simulate_click(next.center(), gpui::Modifiers::none());
    cx.run_until_parked();
    let second_page = view.read_with(&*cx, |view, _| {
        view.map_card
            .as_ref()
            .unwrap()
            .page_items()
            .iter()
            .map(|item| item.path.clone())
            .collect::<Vec<_>>()
    });
    assert_eq!(second_page.len(), 1);
    assert!(!first_page.contains(&second_page[0]));
    assert!(cx.debug_bounds("#map-gallery-item-0").is_some());
}

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
fn map_marker_card_requires_photos_and_wraps_cluster_navigation() {
    let coordinate = metadata::GeoCoordinate::new(47.5, 19.05).unwrap();
    assert!(MapCard::new(coordinate, Vec::new(), 0, None, None).is_none());
    let items = (0..6)
        .map(|index| MapCardItem {
            path: PathBuf::from(format!("{index}.jpg")),
        })
        .collect::<Vec<_>>();
    let mut card = MapCard::new(coordinate, items, 7, None, Some(8)).unwrap();
    assert_eq!(card.page_count(), 2);
    assert_eq!(card.page_items().len(), 6);
    assert_eq!(card.page, 0);
    card.navigate(-1);
    assert_eq!(card.page, 1);
    card.navigate(1);
    assert_eq!(card.page, 0);
}

#[test]
fn map_surface_geometry_is_identity_for_the_rendered_camera() {
    let camera = map::Camera {
        center: metadata::GeoCoordinate::new(47.5, 19.0).unwrap(),
        zoom: 8.25,
    };
    assert_eq!(
        map_surface_geometry(camera, camera, (1200, 800), (1200, 800)),
        MapSurfaceGeometry {
            left: 0.0,
            top: 0.0,
            width: 1200.0,
            height: 800.0,
        }
    );
}

#[test]
fn transformed_raster_and_live_markers_share_the_same_camera() {
    let rendered = map::Camera {
        center: metadata::GeoCoordinate::new(47.4979, 19.0402).unwrap(),
        zoom: 7.0,
    };
    let current = map::Camera {
        center: metadata::GeoCoordinate::new(47.55, 19.12).unwrap(),
        zoom: 8.35,
    };
    let viewport = (1280, 720);
    let geometry = map_surface_geometry(rendered, current, viewport, viewport);
    for coordinate in [
        metadata::GeoCoordinate::new(47.4979, 19.0402).unwrap(),
        metadata::GeoCoordinate::new(47.62, 18.91).unwrap(),
    ] {
        let rendered_position = map_screen_position(coordinate, rendered, viewport).unwrap();
        let transformed_x = geometry.left + rendered_position.0 * geometry.width;
        let transformed_y = geometry.top + rendered_position.1 * geometry.height;
        let current_position = map_screen_position(coordinate, current, viewport).unwrap();
        assert!((transformed_x - current_position.0 * viewport.0 as f32).abs() < 0.01);
        assert!((transformed_y - current_position.1 * viewport.1 as f32).abs() < 0.01);
    }
}

#[test]
fn oversized_map_raster_covers_short_pans_without_blank_edges() {
    let rendered = map::Camera::default();
    let mut current = rendered;
    let viewport = (1000, 700);
    let raster = (
        viewport.0 + MAP_RENDER_OVERSCAN * 2,
        viewport.1 + MAP_RENDER_OVERSCAN * 2,
    );
    let (center_x, center_y) = map::mercator_project(current.center);
    let world = 256.0 * 2_f64.powf(current.zoom);
    current.center =
        map::mercator_unproject(center_x + 60.0 / world, center_y - 40.0 / world).unwrap();
    let geometry = map_surface_geometry(rendered, current, viewport, raster);
    assert!(geometry.left <= 0.0);
    assert!(geometry.top <= 0.0);
    assert!(geometry.left + geometry.width >= viewport.0 as f32);
    assert!(geometry.top + geometry.height >= viewport.1 as f32);
}

#[test]
fn map_surface_transform_remains_aligned_across_the_antimeridian() {
    let rendered = map::Camera {
        center: metadata::GeoCoordinate::new(70.0, 179.9).unwrap(),
        zoom: 6.5,
    };
    let current = map::Camera {
        center: metadata::GeoCoordinate::new(70.1, -179.9).unwrap(),
        zoom: 7.25,
    };
    let viewport = (900, 600);
    let geometry = map_surface_geometry(rendered, current, viewport, viewport);
    let coordinate = metadata::GeoCoordinate::new(70.05, 179.98).unwrap();
    let old = map_screen_position(coordinate, rendered, viewport).unwrap();
    let live = map_screen_position(coordinate, current, viewport).unwrap();
    assert!((geometry.left + old.0 * geometry.width - live.0 * viewport.0 as f32).abs() < 0.01);
    assert!((geometry.top + old.1 * geometry.height - live.1 * viewport.1 as f32).abs() < 0.01);
}

#[test]
fn cursor_anchored_map_zoom_preserves_the_coordinate_under_the_pointer() {
    let viewport = (1200, 800);
    let anchor = (930.0, 210.0);
    let camera = map::Camera {
        center: metadata::GeoCoordinate::new(47.5, 19.05).unwrap(),
        zoom: 7.0,
    };
    let (center_x, center_y) = map::mercator_project(camera.center);
    let world = 256.0 * 2_f64.powf(camera.zoom);
    let anchored_coordinate = map::mercator_unproject(
        center_x + (anchor.0 - f64::from(viewport.0) / 2.0) / world,
        center_y + (anchor.1 - f64::from(viewport.1) / 2.0) / world,
    )
    .unwrap();
    let zoomed = zoom_map_camera(camera, 1.0, anchor, viewport);
    let position = map_screen_position(anchored_coordinate, zoomed, viewport).unwrap();
    assert!((position.0 * viewport.0 as f32 - anchor.0 as f32).abs() < 0.01);
    assert!((position.1 * viewport.1 as f32 - anchor.1 as f32).abs() < 0.01);
}

#[test]
fn repeated_map_zoom_steps_have_exact_bounded_final_zoom() {
    let mut camera = map::Camera::default();
    for _ in 0..50 {
        camera = zoom_map_camera(camera, 1.0, (400.0, 300.0), (800, 600));
    }
    assert_eq!(camera.zoom, 20.0);
    for _ in 0..50 {
        camera = zoom_map_camera(camera, -1.0, (400.0, 300.0), (800, 600));
    }
    assert_eq!(camera.zoom, 0.0);
}

#[test]
fn map_detail_status_distinguishes_loading_from_completed_empty_detail() {
    assert_eq!(
        map_detail_status(8.0, config::OnlineMapPermission::Enabled, true, None),
        "Bundled map · loading online detail"
    );
    assert_eq!(
        map_detail_status(
            8.0,
            config::OnlineMapPermission::Enabled,
            false,
            Some((0, None)),
        ),
        "Bundled map · no supported detail in visible tiles"
    );
    assert_eq!(
        map_detail_status(
            8.0,
            config::OnlineMapPermission::Enabled,
            false,
            Some((3, None)),
        ),
        "Detailed map · 3 vector tiles"
    );
}

#[test]
fn map_cache_capacity_progress_is_bounded_and_handles_zero_capacity() {
    assert_eq!(cache_capacity_fraction(0, 100), 0.0);
    assert_eq!(cache_capacity_fraction(25, 100), 0.25);
    assert_eq!(cache_capacity_fraction(100, 100), 1.0);
    assert_eq!(cache_capacity_fraction(150, 100), 1.0);
    assert_eq!(cache_capacity_fraction(1, 0), 0.0);
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
fn people_grid_adds_fixed_width_cards_instead_of_stretching_three() {
    assert_eq!(people_grid_column_count(700.0, 220.0, 12.0), 3);
    assert_eq!(people_grid_column_count(1_640.0, 220.0, 12.0), 7);
}

#[test]
fn face_inventory_progress_reports_fraction_stage_and_current_file() {
    let progress = FaceInventoryProgress::default();
    progress.stage.store(1, Ordering::Release);
    assert_eq!(
        progress.label().as_deref(),
        Some("Scanning library · 0 folders scanned · 0 queued · 0 images found · preparing…")
    );
    progress.scan_directories.store(3, Ordering::Release);
    progress.scan_pending.store(1, Ordering::Release);
    progress.scan_images.store(12, Ordering::Release);
    assert_eq!(progress.fraction(), Some(0.75));
    progress.discovered.store(42, Ordering::Release);
    progress.processed.store(17, Ordering::Release);
    *progress.current_file.lock().unwrap() = Some(PathBuf::from("trip/portrait.jpg"));
    progress.stage.store(2, Ordering::Release);
    assert_eq!(
        progress.label().as_deref(),
        Some("Checking image identity 17/42 · portrait.jpg")
    );
    assert_eq!(progress.fraction(), Some(17.0 / 42.0));
    progress.stage.store(3, Ordering::Release);
    assert_eq!(
        progress.label().as_deref(),
        Some("Updating face-analysis inventory 17/42 · portrait.jpg")
    );
    progress.stage.store(4, Ordering::Release);
    assert_eq!(
        progress.label().as_deref(),
        Some("Prioritizing pending face analysis across 42 images…")
    );
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
            &PathBuf::from(format!("/tmp/rekesz-fixed-{stem}.png")),
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
            companions: HashMap::new(),
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
            companions: HashMap::new(),
        }),
    );

    let mut expanded = HashSet::from([root, child]);
    assert_eq!(
        visible_rows(&model, &expanded, &HashSet::new(), true).len(),
        3
    );
    expanded.remove(&child);
    assert_eq!(
        visible_rows(&model, &expanded, &HashSet::new(), true).len(),
        2
    );
    assert_eq!(model.loaded(child).unwrap().children.len(), 1);
}

#[test]
fn hidden_subtree_moves_to_the_virtual_bottom_section() {
    let mut model = DirectoryModel::new(PathBuf::from("/pictures"));
    let root = model.root();
    let (request, _) = model.begin_load(root).unwrap();
    model.finish_load(
        root,
        request,
        Ok(DirectoryScan {
            directories: vec![PathBuf::from("/pictures/a"), PathBuf::from("/pictures/z")],
            images: Vec::new(),
            companions: HashMap::new(),
        }),
    );
    let hidden = HashSet::from([PathBuf::from("/pictures/a")]);
    let rows = visible_rows(&model, &HashSet::from([root]), &hidden, true);

    assert_eq!(rows.len(), 4);
    assert_eq!(
        model.node(rows[1].node).path.as_ref(),
        Path::new("/pictures/z")
    );
    assert_eq!(rows[2].node, HIDDEN_NODE);
    assert_eq!(
        model.node(rows[3].node).path.as_ref(),
        Path::new("/pictures/a")
    );
    assert_eq!(rows[3].depth, 1);
}

#[test]
fn recursive_grid_collects_globally_sorted_images_and_excludes_hidden_subtrees() {
    let root = tempfile::tempdir().unwrap();
    let visible = root.path().join("visible");
    let hidden = root.path().join("hidden");
    std::fs::create_dir_all(&visible).unwrap();
    std::fs::create_dir_all(&hidden).unwrap();
    std::fs::write(visible.join("z.jpg"), b"not a real jpeg").unwrap();
    std::fs::write(visible.join("a.jpg"), b"not a real jpeg").unwrap();
    std::fs::write(hidden.join("hidden.jpg"), b"not a real jpeg").unwrap();
    for name in ["a.jpg", "z.jpg"] {
        std::fs::File::open(visible.join(name))
            .unwrap()
            .set_times(std::fs::FileTimes::new().set_modified(std::time::SystemTime::UNIX_EPOCH))
            .unwrap();
    }
    let cache = tempfile::tempdir().unwrap();
    let limiter = JobLimiter::new(2);
    let scanner = DirectoryScanner::new(2, limiter.clone());
    let metadata_loader = MetadataLoader::new(cache.path().to_path_buf(), 2, limiter);

    let images = collect_recursive_images(
        &scanner,
        &metadata_loader,
        root.path().to_path_buf(),
        RawJpegPreference::Raw,
        &HashSet::from([hidden]),
        Arc::new(AtomicBool::new(false)),
    )
    .unwrap();

    assert_eq!(images.paths, [visible.join("a.jpg"), visible.join("z.jpg")]);
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

// ---- Import view UI tests ----

/// Builds a source folder card with `folders` subfolders and `photos` photos.
fn write_import_source(root: &Path, name: &str, folders: usize, photos: usize) -> PathBuf {
    let card = root.join(name);
    std::fs::create_dir_all(&card).unwrap();
    for index in 0..folders {
        std::fs::create_dir(card.join(format!("folder-{index:02}"))).unwrap();
    }
    for index in 0..photos {
        image::RgbImage::from_pixel(8, 8, image::Rgb([32, 96, 160]))
            .save(card.join(format!("photo-{index}.jpg")))
            .unwrap();
    }
    card
}

/// Opens the app directly in the import view with `card` injected as a folder
/// source, selects it, and waits for the listing to load. `library` is the
/// (separate) library root the import would copy into.
fn import_view_with_card<'a>(
    cx: &'a mut gpui::TestAppContext,
    library: &Path,
    card: &Path,
) -> (
    gpui::Entity<PictureBrowser>,
    &'a mut gpui::VisualTestContext,
) {
    let cache = library.parent().unwrap().join("cache");
    let mut config = gpui_test_config(library, &cache);
    config.settings.automatic_face_analysis = false;
    let (view, cx) = cx.add_window_view(|_, cx| PictureBrowser::new(&config, cx));
    view.update(&mut *cx, |view, cx| {
        view.set_directory_view_mode(DirectoryViewMode::Import, cx);
        let id = sources::SourceId::new(format!("folder:{}", card.display()));
        view.import_view.sources.push(sources::SourceEntry {
            id: id.clone(),
            label: "Card".to_owned(),
            kind: sources::SourceKind::Folder,
            backing: sources::SourceBacking::Path(card.to_path_buf()),
            capacity: None,
        });
        view.select_import_source(id, cx);
    });
    cx.run_until_parked();
    assert!(view.read_with(&*cx, |view, _| view.import_view.listing_loaded()));
    (view, cx)
}

#[gpui::test]
fn import_destination_picker_rows_keep_natural_height(cx: &mut gpui::TestAppContext) {
    let root = tempfile::tempdir().unwrap();
    let library = root.path().join("library");
    std::fs::create_dir(&library).unwrap();
    for index in 0..40 {
        std::fs::create_dir(library.join(format!("folder-{index:02}"))).unwrap();
    }
    let card = write_import_source(root.path(), "card", 0, 1);
    let (_view, cx) = import_view_with_card(cx, &library, &card);

    let toggle = cx
        .debug_bounds("#import-destination-picker")
        .expect("picker toggle renders");
    cx.simulate_click(toggle.center(), gpui::Modifiers::none());
    cx.run_until_parked();

    let panel = cx
        .debug_bounds("#import-destination-panel")
        .expect("picker panel renders");
    let first = cx
        .debug_bounds("#picker-entry-0")
        .expect("first folder row renders");
    let second = cx
        .debug_bounds("#picker-entry-1")
        .expect("second row renders");
    let last = cx
        .debug_bounds("#picker-entry-39")
        .expect("last row renders");

    // The panel must not clamp its content: a max-height flex column lets
    // taffy shrink the rows to fit, and truncated row text then clips itself
    // down to nothing, rendering the picker as an empty bordered box.
    assert!(
        panel.size.height >= first.size.height * 40.0,
        "panel {:?} must grow with its {} rows instead of clamping",
        panel.size,
        40
    );
    for row in [first, second, last] {
        assert!(
            row.size.height >= px(20.0),
            "row height {:?} collapsed; flex children were shrunk to fit a max-height",
            row.size.height
        );
    }
    // Rows keep a uniform, natural cadence. Line heights carry sub-pixel
    // fractions, so compare the average over the whole run, not exact sums.
    let spacing = second.origin.y - first.origin.y;
    let average = (last.origin.y - first.origin.y) / 39.0;
    assert!(
        spacing >= px(25.0) && (average - spacing).abs() < px(0.5),
        "rows compressed or uneven: first gap {spacing}, average {average}"
    );
}

#[gpui::test]
fn import_destination_picker_navigates_and_picks_destination(cx: &mut gpui::TestAppContext) {
    let root = tempfile::tempdir().unwrap();
    let library = root.path().join("library");
    std::fs::create_dir_all(library.join("2024").join("06")).unwrap();
    std::fs::create_dir(library.join("2025")).unwrap();
    let card = write_import_source(root.path(), "card", 0, 0);
    let (_view, cx) = import_view_with_card(cx, &library, &card);

    let toggle = cx.debug_bounds("#import-destination-picker").unwrap();
    cx.simulate_click(toggle.center(), gpui::Modifiers::none());
    cx.run_until_parked();

    assert_eq!(
        _view.read_with(&*cx, |view, _| view
            .import_view
            .picker_entries
            .clone()
            .expect("entries loaded")),
        vec!["2024".to_owned(), "2025".to_owned()]
    );

    // Into 2024: the listing shows the nested month folder and the ".." row.
    let entry = cx.debug_bounds("#picker-entry-0").unwrap();
    cx.simulate_click(entry.center(), gpui::Modifiers::none());
    cx.run_until_parked();
    assert_eq!(
        _view.read_with(&*cx, |view, _| (
            view.import_view.picker_dir.clone(),
            view.import_view.picker_entries.clone().unwrap()
        )),
        (library.join("2024"), vec!["06".to_owned()])
    );
    assert!(cx.debug_bounds("#picker-up-0").is_some());

    // Back to the library root via the dedicated row.
    let root_row = cx.debug_bounds("#picker-root-0").unwrap();
    cx.simulate_click(root_row.center(), gpui::Modifiers::none());
    cx.run_until_parked();
    assert_eq!(
        _view.read_with(&*cx, |view, _| view.import_view.picker_dir.clone()),
        library.clone()
    );

    // "Use this folder" closes the picker and retargets the import base.
    let use_folder = cx.debug_bounds("#picker-use-folder").unwrap();
    cx.simulate_click(use_folder.center(), gpui::Modifiers::none());
    cx.run_until_parked();
    assert_eq!(
        _view.read_with(&*cx, |view, _| (
            view.import_view.destination_picker_open,
            view.import_view.destination_base.clone()
        )),
        (false, Some(library.clone()))
    );
}

#[gpui::test]
fn import_destination_picker_creates_folder(cx: &mut gpui::TestAppContext) {
    let root = tempfile::tempdir().unwrap();
    let library = root.path().join("library");
    std::fs::create_dir(&library).unwrap();
    let card = write_import_source(root.path(), "card", 0, 0);
    let (_view, cx) = import_view_with_card(cx, &library, &card);

    let toggle = cx.debug_bounds("#import-destination-picker").unwrap();
    cx.simulate_click(toggle.center(), gpui::Modifiers::none());
    cx.run_until_parked();

    let new_folder = cx.debug_bounds("#picker-new-folder-0").unwrap();
    cx.simulate_click(new_folder.center(), gpui::Modifiers::none());
    cx.run_until_parked();
    assert!(cx.debug_bounds("#import-new-folder-create").is_some());

    // The app installs its keymap in main(); tests must do the same for
    // keystroke-driven actions like the input's Enter binding.
    cx.update(|_, cx| bind_picture_browser_keys(cx));

    // "d" is regression cover: the Developed toggle used to be a keymap
    // binding, which beats key listeners and silently swallowed the letter.
    cx.simulate_keystrokes("d 2 0 2 4");
    cx.run_until_parked();
    assert_eq!(
        _view.read_with(&*cx, |view, _| view.import_view.new_folder_name.clone()),
        "d2024"
    );
    cx.simulate_keystrokes("enter");
    cx.run_until_parked();

    assert!(
        library.join("d2024").is_dir(),
        "typed folder was not created"
    );
    assert_eq!(
        _view.read_with(&*cx, |view, _| (
            view.import_view.creating_folder,
            view.import_view.picker_entries.clone().unwrap()
        )),
        (false, vec!["d2024".to_owned()])
    );
}

#[gpui::test]
fn import_source_listing_navigates_by_tiles_and_breadcrumbs(cx: &mut gpui::TestAppContext) {
    let root = tempfile::tempdir().unwrap();
    let library = root.path().join("library");
    std::fs::create_dir(&library).unwrap();
    let card = write_import_source(root.path(), "card", 0, 2);
    std::fs::create_dir_all(card.join("2024").join("06")).unwrap();
    image::RgbImage::from_pixel(8, 8, image::Rgb([32, 96, 160]))
        .save(card.join("2024").join("nested.jpg"))
        .unwrap();
    let (_view, cx) = import_view_with_card(cx, &library, &card);

    assert_eq!(
        _view.read_with(&*cx, |view, _| (
            view.import_view.photos.len(),
            view.import_view.selection.len(),
        )),
        (2, 2)
    );

    // The subdirectory tile navigates one level down.
    let tile = cx
        .debug_bounds("#import-folder-0")
        .expect("subdirectory tile renders");
    cx.simulate_click(tile.center(), gpui::Modifiers::none());
    cx.run_until_parked();
    assert_eq!(
        _view.read_with(&*cx, |view, _| (
            view.import_view.current_relative(),
            view.import_view.photos.len()
        )),
        (Some("2024".to_owned()), 1)
    );

    // The breadcrumb root crumb returns to the source root.
    let crumb = cx.debug_bounds("#import-crumb-0").unwrap();
    cx.simulate_click(crumb.center(), gpui::Modifiers::none());
    cx.run_until_parked();
    assert_eq!(
        _view.read_with(&*cx, |view, _| view.import_view.current_relative()),
        Some(String::new())
    );

    // Select-all / select-none drive the selection from the header toggles.
    let select_none = cx.debug_bounds("#import-select-none").unwrap();
    cx.simulate_click(select_none.center(), gpui::Modifiers::none());
    cx.run_until_parked();
    assert_eq!(
        _view.read_with(&*cx, |view, _| view.import_view.selection.len()),
        0
    );
    let select_all = cx.debug_bounds("#import-select-all").unwrap();
    cx.simulate_click(select_all.center(), gpui::Modifiers::none());
    cx.run_until_parked();
    assert_eq!(
        _view.read_with(&*cx, |view, _| view.import_view.selection.len()),
        2
    );
}

#[gpui::test]
fn import_layout_combobox_and_duplicates_toggle_persist(cx: &mut gpui::TestAppContext) {
    let root = tempfile::tempdir().unwrap();
    let cache = tempfile::tempdir().unwrap();
    let mut config = gpui_test_config(root.path(), cache.path());
    config.settings.automatic_face_analysis = false;
    let (view, cx) = cx.add_window_view(|_, cx| PictureBrowser::new(&config, cx));
    let settings_path = view.read_with(&*cx, |view, _| view.settings_path.clone());

    let file_button = cx.debug_bounds("#file-menu-button").unwrap();
    cx.simulate_click(file_button.center(), gpui::Modifiers::none());
    cx.run_until_parked();
    let item = cx.debug_bounds("#import-photos-menu-item").unwrap();
    cx.simulate_click(item.center(), gpui::Modifiers::none());
    cx.run_until_parked();

    let initial_preset = view.read_with(&*cx, |view, _| view.import_view.preset);
    let button = cx.debug_bounds("#import-layout-button").unwrap();
    cx.simulate_click(button.center(), gpui::Modifiers::none());
    cx.run_until_parked();
    let option = cx.debug_bounds("#import-layout-option-2").unwrap();
    cx.simulate_click(option.center(), gpui::Modifiers::none());
    cx.run_until_parked();

    let duplicates = cx.debug_bounds("#import-duplicates").unwrap();
    cx.simulate_click(duplicates.center(), gpui::Modifiers::none());
    cx.run_until_parked();

    let (preset, include_duplicates) = view.read_with(&*cx, |view, _| {
        (view.import_view.preset, view.import_view.include_duplicates)
    });
    assert_ne!(preset, initial_preset, "layout preset did not change");
    assert!(include_duplicates);

    let saved = std::fs::read_to_string(&settings_path).unwrap();
    assert!(
        saved.contains("\"import_include_duplicates\": true"),
        "duplicates policy not persisted: {saved}"
    );
}

#[gpui::test]
fn import_footer_progress_counts_each_file_once(cx: &mut gpui::TestAppContext) {
    let root = tempfile::tempdir().unwrap();
    let library = root.path().join("library");
    std::fs::create_dir(&library).unwrap();
    let card = write_import_source(root.path(), "card", 0, 3);
    let (_view, cx) = import_view_with_card(cx, &library, &card);

    // Let the whole-directory thumbnail pass (batched on timers) and the
    // decodes settle. Progress reads sit behind a 250 ms throttle, so every
    // read below expires it first.
    for _ in 0..6 {
        cx.executor().advance_clock(Duration::from_millis(300));
        cx.run_until_parked();
    }
    let (mut rendered, total) = _view.read_with(&*cx, |view, _| view.import_thumbnail_progress());
    assert_eq!(total, 3, "target is the listing's file count");

    // Repeated scans must neither compound the done counter nor grow the
    // target — the regression for the runaway "6249 / 6293" footer, where
    // every 250 ms rescan re-added all terminal files to the high-water
    // mark and both numbers raced past the listing size. Rendered may grow
    // (decodes finish between scans) but stays bounded by the target.
    for _ in 0..4 {
        cx.executor().advance_clock(Duration::from_millis(300));
        let (again_rendered, again_total) =
            _view.read_with(&*cx, |view, _| view.import_thumbnail_progress());
        assert_eq!(again_total, 3, "target must not grow across rescans");
        assert!(
            again_rendered >= rendered && again_rendered <= again_total,
            "rendered must be monotonic and bounded: {rendered} → {again_rendered}"
        );
        rendered = again_rendered;
    }
}

fn scanned_device_file(device_id: &str, name: &str) -> import::ScannedFile {
    import::ScannedFile {
        backing: crate::sources::SourceBacking::Ptp(crate::sources::ptp::PtpObjectRef {
            device_id: Arc::from(device_id),
            folder: Arc::from("/DCIM"),
            name: Arc::from(name),
        }),
        name: name.to_owned(),
        size: 10,
        modified: None,
    }
}

#[test]
fn paired_photos_thumbnail_from_the_jpeg_companion() {
    use import_view::import_thumbnail_path;
    use std::path::PathBuf;

    let staged = |path: &str| {
        let mut map = std::collections::HashMap::new();
        map.insert(
            crate::sources::SourceBacking::Ptp(crate::sources::ptp::PtpObjectRef {
                device_id: Arc::from("gphoto:usb:001,002"),
                folder: Arc::from("/DCIM"),
                name: Arc::from("IMG_0001"),
            }),
            PathBuf::from(path),
        );
        map
    };

    let raw_primary = import::ScannedPhoto {
        directory: String::new(),
        primary: import::ScannedFile {
            backing: crate::sources::SourceBacking::Path(PathBuf::from("/card/IMG_0001.CR3")),
            name: "IMG_0001.CR3".to_owned(),
            size: 10,
            modified: None,
        },
        companion: Some(import::ScannedFile {
            backing: crate::sources::SourceBacking::Path(PathBuf::from("/card/IMG_0001.JPG")),
            name: "IMG_0001.JPG".to_owned(),
            size: 4,
            modified: None,
        }),
    };
    let device_pair = import::ScannedPhoto {
        directory: String::new(),
        primary: scanned_device_file("gphoto:usb:001,002", "IMG_0001.CR3"),
        companion: Some(scanned_device_file("gphoto:usb:001,002", "IMG_0001.JPG")),
    };
    let mut device_staged = staged("/cache/import-staging/IMG_0001.CR3");
    device_staged.insert(
        crate::sources::SourceBacking::Ptp(crate::sources::ptp::PtpObjectRef {
            device_id: Arc::from("gphoto:usb:001,002"),
            folder: Arc::from("/DCIM"),
            name: Arc::from("IMG_0001.JPG"),
        }),
        PathBuf::from("/cache/import-staging/IMG_0001.JPG"),
    );

    // Raw-primary pair: the JPEG companion displays.
    assert_eq!(
        import_thumbnail_path(&raw_primary, &std::collections::HashMap::new()),
        Some(PathBuf::from("/card/IMG_0001.JPG"))
    );
    // Device pair: the staged companion displays.
    assert_eq!(
        import_thumbnail_path(&device_pair, &device_staged),
        Some(PathBuf::from("/cache/import-staging/IMG_0001.JPG"))
    );
    // Companion not staged yet: nothing renders (no raw stand-in).
    assert_eq!(
        import_thumbnail_path(&device_pair, &staged("/cache/other")),
        None
    );
    // JPEG-primary pairing (raw companion) keeps displaying the primary.
    let jpeg_primary = import::ScannedPhoto {
        directory: String::new(),
        primary: import::ScannedFile {
            backing: crate::sources::SourceBacking::Path(PathBuf::from("/card/IMG_0002.JPG")),
            name: "IMG_0002.JPG".to_owned(),
            size: 4,
            modified: None,
        },
        companion: Some(import::ScannedFile {
            backing: crate::sources::SourceBacking::Path(PathBuf::from("/card/IMG_0002.CR3")),
            name: "IMG_0002.CR3".to_owned(),
            size: 10,
            modified: None,
        }),
    };
    assert_eq!(
        import_thumbnail_path(&jpeg_primary, &std::collections::HashMap::new()),
        Some(PathBuf::from("/card/IMG_0002.JPG"))
    );
    // Unpaired files display themselves.
    let unpaired = import::ScannedPhoto {
        directory: String::new(),
        primary: import::ScannedFile {
            backing: crate::sources::SourceBacking::Path(PathBuf::from("/card/IMG_0003.CR3")),
            name: "IMG_0003.CR3".to_owned(),
            size: 10,
            modified: None,
        },
        companion: None,
    };
    assert_eq!(
        import_thumbnail_path(&unpaired, &std::collections::HashMap::new()),
        Some(PathBuf::from("/card/IMG_0003.CR3"))
    );
}

#[gpui::test]
fn import_footer_completes_for_paired_photos(cx: &mut gpui::TestAppContext) {
    let root = tempfile::tempdir().unwrap();
    let library = root.path().join("library");
    std::fs::create_dir(&library).unwrap();
    let card = write_import_source(root.path(), "card", 0, 1);
    // A CR3+JPG pair whose raw fails to decode: the pair's thumbnail must
    // come from the JPEG (the raw is never requested, so its decode error
    // must not keep the footer from completing).
    std::fs::write(card.join("pair-1.CR3"), b"not a raw").unwrap();
    image::RgbImage::from_pixel(8, 8, image::Rgb([32, 96, 160]))
        .save(card.join("pair-1.jpg"))
        .unwrap();
    let (_view, cx) = import_view_with_card(cx, &library, &card);

    // Poll to completion: under a fully loaded test runner the decodes and
    // the batched pass need an unbounded number of pump cycles.
    let mut progress = (0, 0);
    for _ in 0..40 {
        cx.executor().advance_clock(Duration::from_millis(300));
        let _ = cx.refresh();
        cx.run_until_parked();
        progress = _view.read_with(&*cx, |view, _| view.import_thumbnail_progress_uncached());
        if progress.0 >= progress.1 {
            break;
        }
    }
    let (rendered, total) = progress;
    assert_eq!(total, 2, "target counts photos, not files");
    assert_eq!(rendered, 2, "footer must complete via the JPEG companions");
}

#[test]
fn packaged_lens_profile_loads_and_selects() {
    let profile = rendering::embedded_lens_profile_for_tests().expect("packaged profile");
    assert_eq!(profile.model, "PowerShot V1");
    // The wide end's black-circle recovery is the headline number from the
    // calibration against the real card.
    let wide = profile.select("PowerShot V1", Some(8.2)).unwrap();
    assert!(*wide.vignette.last().unwrap() > 10.0, "corner gain");
    assert!(wide.k1 > 0.05, "wide-end distortion correction");
    // Tele end: nearly flat.
    let tele = profile.select("PowerShot V1", Some(25.6)).unwrap();
    assert!(*tele.vignette.last().unwrap() < 3.0);
}

#[test]
fn thumbnail_display_path_prefers_jpeg_companions_and_keeps_developed_raws() {
    let raw = PathBuf::from("/lib/IMG_1.CR3");
    let jpeg = PathBuf::from("/lib/IMG_1.JPG");
    let mut companions = std::collections::HashMap::new();
    companions.insert(raw.clone(), jpeg.clone());
    let recipes = Arc::new(std::collections::HashMap::new());
    let none: Option<&std::collections::HashMap<PathBuf, catalog::RecipeEntry>> = Some(&recipes);

    // Baseline: paired raw → JPEG companion.
    assert_eq!(
        thumbnail_display_path_for(&raw, false, none, &companions),
        jpeg
    );
    // Unpaired files and JPEGs display themselves.
    assert_eq!(
        thumbnail_display_path_for(&jpeg, false, none, &companions),
        jpeg
    );
    let unpaired = PathBuf::from("/lib/IMG_9.CR3");
    assert_eq!(
        thumbnail_display_path_for(&unpaired, false, none, &companions),
        unpaired
    );
    // A raw with a develop recipe (Developed mode on) keeps the raw: the
    // recipe's edits exist only there.
    let mut edited = std::collections::HashMap::new();
    edited.insert(
        raw.clone(),
        catalog::RecipeEntry {
            recipe: catalog::DevelopRecipe::default(),
            digest: [0; 32],
        },
    );
    assert_eq!(
        thumbnail_display_path_for(&raw, true, Some(&edited), &companions),
        raw
    );
}

/// Library-grid regression for pair re-learning: a pre-existing CR3+JPG
/// pair (never imported through the app) must be paired by the directory
/// scan that produces the listing, and the tile must render from the JPEG
/// even though the raw fails to decode.
#[gpui::test]
fn library_grid_learns_cr3_jpeg_pairs_from_the_listing_scan(cx: &mut gpui::TestAppContext) {
    let root = tempfile::tempdir().unwrap();
    let cache = tempfile::tempdir().unwrap();
    // The raw half is garbage: if the tile rendered from it, the grid
    // would show "Cannot decode" instead of the JPEG's pixels.
    std::fs::write(root.path().join("pair-1.CR3"), b"not a raw").unwrap();
    image::RgbImage::from_pixel(8, 8, image::Rgb([200, 100, 50]))
        .save(root.path().join("pair-1.jpg"))
        .unwrap();
    write_photo_stub(root.path(), "photo-0", 47.5, 19.05);
    let mut config = gpui_test_config(root.path(), cache.path());
    config.settings.automatic_face_analysis = false;
    let (view, cx) = cx.add_window_view(|_, cx| PictureBrowser::new(&config, cx));
    cx.run_until_parked();

    // The scan that produced the listing must have taught the browser the
    // pair — this is the automatic re-learning of existing libraries.
    let (pair, total) = view.read_with(&*cx, |view, _| {
        (
            view.thumbnail_companions
                .get(&root.path().join("pair-1.CR3"))
                .cloned(),
            view.thumbnail_companions.len(),
        )
    });
    assert_eq!(pair, Some(root.path().join("pair-1.jpg")));
    assert_eq!(total, 1, "only the pair is mapped");

    // The tile resolves to the JPEG and reaches a terminal (loaded) state.
    for _ in 0..10 {
        cx.executor().advance_clock(Duration::from_millis(300));
        let _ = cx.refresh();
        cx.run_until_parked();
    }
    let resolved = view.read_with(&*cx, |view, _| {
        let raw = root.path().join("pair-1.CR3");
        let display = view.thumbnail_display_path(&raw);
        let identity = view.render_identity_for(&display);
        match view.thumbnails.state_for(&display, identity) {
            PreviewState::Loaded(_) => "loaded",
            PreviewState::Error(_) => "error",
            _ => "pending",
        }
    });
    assert_eq!(resolved, "loaded", "paired tile must render from the JPEG");
}
