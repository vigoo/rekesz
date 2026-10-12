use std::fs;

use rusqlite::params;
use test_r::test;

use super::*;
use crate::catalog::initialize_test_catalog;

const MODEL: &str = "test-recognizer-v1";

// CI runners are slower and more contended than dev machines; REKESZ_TEST_TIMING_SCALE
// stretches the timing guards without weakening them locally.
fn scaled_budget(budget: std::time::Duration) -> std::time::Duration {
    let scale = std::env::var("REKESZ_TEST_TIMING_SCALE")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(1.0)
        .max(1.0);
    budget.mul_f64(scale)
}

fn vector(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn process(root: &Path, worker: &str) -> Result<usize> {
    process_pending_refresh_batch(
        root,
        worker,
        &MaintenanceAdmission::new(),
        &AtomicBool::new(false),
    )
}

fn install_test_calibration(connection: &Connection) {
    ensure_recognition_schema(connection).unwrap();
    connection
        .execute(
            "INSERT INTO recognition_policy_artifacts(model_id, policy_version,
             prototype_algorithm_version, absolute_threshold, nearest_exemplar_threshold,
             second_best_margin, calibration_digest, enabled, installed_at)
             VALUES(?1, ?2, ?3, .5, .45, .08, zeroblob(32), 1, datetime('now'))",
            params![
                MODEL,
                RECOGNITION_POLICY_VERSION,
                PROTOTYPE_ALGORITHM_VERSION
            ],
        )
        .unwrap();
}

fn fixture() -> tempfile::TempDir {
    let root = tempfile::tempdir().unwrap();
    fs::write(root.path().join("photo.jpg"), b"photo").unwrap();
    initialize_test_catalog(root.path()).unwrap();
    let connection = writable_connection(root.path()).unwrap();
    connection
        .execute_batch(
            "INSERT INTO assets(id, relative_path, scope_key)
             VALUES(1, X'70686f746f2e6a7067', X'01');
             INSERT INTO people(id, display_name, created_at, updated_at) VALUES
                 (1, 'one', datetime('now'), datetime('now')),
                 (2, 'two', datetime('now'), datetime('now'));
             INSERT INTO faces(id, asset_id, provenance, source_key, left, top, right, bottom,
                               origin, updated_at) VALUES
                 (1, 1, 'rekesz', '1', .1, .1, .2, .2, 'rekesz-detected', datetime('now')),
                 (2, 1, 'rekesz', '2', .2, .2, .3, .3, 'rekesz-detected', datetime('now')),
                 (3, 1, 'rekesz', '3', .3, .3, .4, .4, 'rekesz-detected', datetime('now')),
                 (4, 1, 'rekesz', '4', .4, .4, .5, .5, 'rekesz-detected', datetime('now')),
                 (5, 1, 'rekesz', '5', .5, .5, .6, .6, 'rekesz-detected', datetime('now'));
             INSERT INTO face_assignments(face_id, person_id, kind, provenance, updated_at) VALUES
                 (1, 1, 'confirmed', 'rekesz', datetime('now')),
                 (2, 1, 'confirmed', 'rekesz', datetime('now')),
                 (3, 2, 'confirmed', 'rekesz', datetime('now'));",
        )
        .unwrap();
    for (face_id, values) in [
        (1, [1.0, 0.0, 0.0]),
        (2, [0.9, 0.1, 0.0]),
        (3, [0.0, 1.0, 0.0]),
        (4, [0.98, 0.02, 0.0]),
        (5, [0.7, 0.7, 0.0]),
    ] {
        connection
            .execute(
                "INSERT INTO face_embeddings(face_id, model_id, dimensions, vector, created_at)
                 VALUES(?1, ?2, 3, ?3, datetime('now'))",
                params![face_id, MODEL, vector(&values)],
            )
            .unwrap();
    }
    root
}

#[test]
fn robust_prototype_scores_with_absolute_nearest_and_margin_guards() {
    let root = fixture();
    assert!(rebuild_person_prototype(root.path(), 1, MODEL).unwrap());
    assert!(rebuild_person_prototype(root.path(), 2, MODEL).unwrap());
    let suggestions = refresh_incremental(
        root.path(),
        &[],
        &[4, 5],
        MODEL,
        RecognitionPolicy::default(),
    )
    .unwrap();
    assert_eq!(suggestions.len(), 1);
    assert_eq!(suggestions[0].face_id, 4);
    assert_eq!(suggestions[0].person_id, 1);
    assert!(suggestions[0].nearest_exemplar_score > 0.9);
    assert!(suggestions[0].second_best_margin > 0.08);
    let connection = writable_connection(root.path()).unwrap();
    let metadata: (String, i64, i64) = connection
        .query_row(
            "SELECT recognizer_model_id, prototype_generation, policy_version
             FROM recognition_suggestions WHERE face_id=4",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .unwrap();
    assert_eq!(metadata, (MODEL.into(), 1, RECOGNITION_POLICY_VERSION));
}

#[test]
fn rejection_ignore_confirmation_and_lightroom_precedence_are_hard_constraints() {
    let root = fixture();
    rebuild_person_prototype(root.path(), 1, MODEL).unwrap();
    rebuild_person_prototype(root.path(), 2, MODEL).unwrap();
    let connection = writable_connection(root.path()).unwrap();
    connection
        .execute(
            "INSERT INTO face_assignments(face_id, person_id, kind, provenance, updated_at)
             VALUES(4, 1, 'rejected', 'rekesz', datetime('now'))",
            [],
        )
        .unwrap();
    assert!(
        suggest_face_in_connection(&connection, 4, MODEL, RecognitionPolicy::default())
            .unwrap()
            .is_none()
    );
    connection
        .execute("DELETE FROM face_assignments WHERE face_id=4", [])
        .unwrap();
    connection
        .execute("UPDATE faces SET locally_ignored=1 WHERE id=4", [])
        .unwrap();
    assert!(
        suggest_face_in_connection(&connection, 4, MODEL, RecognitionPolicy::default())
            .unwrap()
            .is_none()
    );
    connection
        .execute("UPDATE faces SET locally_ignored=0 WHERE id=4", [])
        .unwrap();
    connection
        .execute(
            "INSERT INTO face_assignments(face_id, person_id, kind, provenance, updated_at)
             VALUES(4, 1, 'suggested', 'lightroom', datetime('now'))",
            [],
        )
        .unwrap();
    assert!(
        suggest_face_in_connection(&connection, 4, MODEL, RecognitionPolicy::default())
            .unwrap()
            .is_none()
    );
    assert_eq!(
        connection
            .query_row(
                "SELECT count(*) FROM face_assignments WHERE face_id=4 AND provenance='lightroom'",
                [],
                |row| row.get::<_, i64>(0)
            )
            .unwrap(),
        1
    );
}

#[test]
fn confirmation_change_rebuilds_generation_and_rescores_only_stale_faces() {
    let root = fixture();
    rebuild_person_prototype(root.path(), 1, MODEL).unwrap();
    rebuild_person_prototype(root.path(), 2, MODEL).unwrap();
    refresh_incremental(root.path(), &[], &[4], MODEL, RecognitionPolicy::default()).unwrap();
    let connection = writable_connection(root.path()).unwrap();
    connection
        .execute(
            "INSERT INTO face_assignments(face_id, person_id, kind, provenance, updated_at)
             VALUES(5, 1, 'confirmed', 'rekesz', datetime('now'))",
            [],
        )
        .unwrap();
    drop(connection);
    refresh_incremental(root.path(), &[1], &[5], MODEL, RecognitionPolicy::default()).unwrap();
    let connection = writable_connection(root.path()).unwrap();
    assert_eq!(
        connection
            .query_row(
                "SELECT generation FROM person_prototypes WHERE person_id=1 AND model_id=?1",
                [MODEL],
                |row| row.get::<_, i64>(0)
            )
            .unwrap(),
        2
    );
    let stale: i64 = connection.query_row(
            "SELECT count(*) FROM recognition_suggestions WHERE person_id=1 AND prototype_generation<2",
            [], |row| row.get(0)).unwrap();
    assert_eq!(stale, 0);
}

#[test]
fn exact_search_is_deterministic_at_representative_scale() {
    let root = fixture();
    let mut connection = writable_connection(root.path()).unwrap();
    // Exercise thousands of people without pretending this exact index is
    // ANN. This is a scale regression for deterministic bounded behavior.
    let transaction = connection.transaction().unwrap();
    for person_id in 10..2_010_i64 {
        transaction
            .execute(
                "INSERT INTO people(id, display_name, created_at, updated_at)
                 VALUES(?1, ?2, datetime('now'), datetime('now'))",
                params![person_id, format!("person-{person_id}")],
            )
            .unwrap();
        transaction
            .execute(
                "INSERT INTO faces(id, asset_id, provenance, source_key, left, top, right, bottom,
                 origin, updated_at) VALUES(?1, 1, 'rekesz', ?2, .1, .1, .2, .2,
                 'rekesz-detected', datetime('now'))",
                params![person_id, format!("scale-{person_id}")],
            )
            .unwrap();
        transaction
            .execute(
                "INSERT INTO face_assignments(face_id, person_id, kind, provenance, updated_at)
                 VALUES(?1, ?1, 'confirmed', 'rekesz', datetime('now'))",
                [person_id],
            )
            .unwrap();
        transaction
            .execute(
                "INSERT INTO face_embeddings(face_id, model_id, dimensions, vector, created_at)
                 VALUES(?1, ?2, 3, ?3, datetime('now'))",
                params![person_id, MODEL, vector(&[0.0, 0.0, 1.0])],
            )
            .unwrap();
        transaction
            .execute(
                "INSERT INTO person_prototypes(person_id, model_id, dimensions, vector,
                 exemplar_count, algorithm_version, generation, updated_at)
                 VALUES(?1, ?2, 3, ?3, 1, ?4, 1, datetime('now'))",
                params![
                    person_id,
                    MODEL,
                    vector(&[0.0, 0.0, 1.0]),
                    PROTOTYPE_ALGORITHM_VERSION
                ],
            )
            .unwrap();
    }
    transaction.commit().unwrap();
    drop(connection);
    rebuild_person_prototype(root.path(), 1, MODEL).unwrap();
    let first =
        refresh_incremental(root.path(), &[], &[4], MODEL, RecognitionPolicy::default()).unwrap();
    let second =
        refresh_incremental(root.path(), &[], &[4], MODEL, RecognitionPolicy::default()).unwrap();
    assert_eq!(first[0].person_id, second[0].person_id);
    assert_eq!(first[0].score, second[0].score);
}

#[test]
fn ann_index_retrieves_exact_identity_at_one_hundred_thousand_people() {
    let dimensions = 8;
    let expected = 54_321;
    let prototypes = (0..100_000_usize)
        .map(|index| {
            let mut centroid = if index == expected {
                vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            } else {
                (0..dimensions)
                    .map(|dimension| {
                        let value = ((index * 31 + dimension * 17 + 11) % 10_007) as f32 + 1.0;
                        if dimension == 0 { value * 0.05 } else { value }
                    })
                    .collect::<Vec<_>>()
            };
            normalize(&mut centroid).unwrap();
            Prototype {
                person_id: index as i64 + 1,
                generation: 1,
                exemplars: vec![centroid.clone()],
                centroid,
            }
        })
        .collect::<Vec<_>>();
    let query = prototypes[expected].centroid.clone();
    let vector_payload_bytes = prototypes
        .iter()
        .map(|prototype| {
            (prototype.centroid.len() + prototype.exemplars.iter().map(Vec::len).sum::<usize>())
                * size_of::<f32>()
        })
        .sum::<usize>();
    // The owned prototype/exemplar payload stays bounded independently of
    // catalog face count; HNSW owns one additional centroid copy.
    assert!(vector_payload_bytes < 7 * 1024 * 1024);

    let build_started = std::time::Instant::now();
    let index = build_prototype_index(prototypes, 1, None).unwrap();
    assert!(build_started.elapsed() < scaled_budget(std::time::Duration::from_secs(120)));
    assert_eq!(index.version, ANN_INDEX_VERSION);
    assert_eq!(index.prototypes.len(), 100_000);
    let candidates = prototype_candidate_ids(&index, &query, ANN_CANDIDATES);
    assert!(!candidates.is_empty());
    let best = index
        .prototypes
        .iter()
        .enumerate()
        .max_by(|left, right| {
            dot(&query, &left.1.centroid)
                .total_cmp(&dot(&query, &right.1.centroid))
                .then_with(|| right.0.cmp(&left.0))
        })
        .map(|(position, _)| position)
        .unwrap();
    assert_eq!(best, expected);
}

#[test]
fn ann_rebuild_observes_cancellation_before_graph_mutation() {
    let prototypes = (0..257)
        .map(|person_id| Prototype {
            person_id,
            generation: 1,
            centroid: vec![1.0, 0.0, 0.0],
            exemplars: vec![vec![1.0, 0.0, 0.0]],
        })
        .collect();
    let cancelled = AtomicBool::new(true);
    let error = build_prototype_index(
        prototypes,
        1,
        Some((&MaintenanceAdmission::new(), &cancelled)),
    )
    .err()
    .expect("cancelled ANN build must fail");
    assert!(error.to_string().contains("cancelled"));
}

#[test]
fn ann_construction_yields_to_foreground_and_cancels_after_start() {
    let prototypes = (0..20_000)
        .map(|person_id| {
            let mut centroid = (0..32)
                .map(|dimension| ((person_id * 13 + dimension * 29 + 1) % 997) as f32 + 1.0)
                .collect::<Vec<_>>();
            normalize(&mut centroid).unwrap();
            Prototype {
                person_id,
                generation: 1,
                exemplars: vec![centroid.clone()],
                centroid,
            }
        })
        .collect::<Vec<_>>();
    let admission = MaintenanceAdmission::new();
    admission.set_foreground_blocked(true);
    let cancelled = Arc::new(AtomicBool::new(false));
    let worker_admission = admission.clone();
    let worker_cancelled = cancelled.clone();
    let (finished_tx, finished_rx) = std::sync::mpsc::channel();
    let worker = std::thread::spawn(move || {
        let result =
            build_prototype_index(prototypes, 1, Some((&worker_admission, &worker_cancelled)));
        finished_tx.send(result.map(|_| ())).unwrap();
    });
    assert!(
        finished_rx
            .recv_timeout(std::time::Duration::from_millis(75))
            .is_err()
    );
    admission.set_foreground_blocked(false);
    std::thread::sleep(std::time::Duration::from_millis(25));
    let cancel_started = std::time::Instant::now();
    cancelled.store(true, Ordering::Release);
    let result = finished_rx
        .recv_timeout(std::time::Duration::from_millis(100))
        .expect("ANN construction did not release within 100 ms");
    let Err(error) = result else {
        panic!("cancelled index waiter unexpectedly received an index");
    };
    assert!(error.to_string().contains("cancelled"));
    assert!(cancel_started.elapsed() <= std::time::Duration::from_millis(100));
    worker.join().unwrap();
}

#[test]
fn ann_randomized_rebuilds_match_exact_second_best_ground_truth() {
    let root = fixture();
    let connection = writable_connection(root.path()).unwrap();
    let policy = RecognitionPolicy::default();
    let admission = MaintenanceAdmission::new();
    let cancelled = AtomicBool::new(false);
    for clustered in [true, false] {
        for generation in 1..=4 {
            let prototypes = (0..300)
                .map(|index| {
                    let centroid = if index == 0 {
                        vec![0.999_8, 0.020_4, 0.0]
                    } else if clustered && index == 1 {
                        vec![0.999_7, 0.021_4, 0.0]
                    } else {
                        let angle = 1.0 + (index as f32 * 0.017).sin() * 0.2;
                        vec![angle.cos(), angle.sin(), 0.0]
                    };
                    Prototype {
                        person_id: index + 100,
                        generation,
                        exemplars: vec![centroid.clone()],
                        centroid,
                    }
                })
                .collect();
            let index = build_prototype_index(prototypes, generation, None).unwrap();
            let suggestion = calculate_face_suggestion(
                &connection,
                4,
                MODEL,
                policy,
                &index,
                &admission,
                &cancelled,
            )
            .unwrap();
            if clustered {
                assert!(
                    suggestion.is_none(),
                    "close exact runner-up must prevent acceptance"
                );
            } else {
                assert_eq!(suggestion.unwrap().person_id, 100);
            }
        }
    }
}

#[test]
fn mixed_prototype_dimensions_fail_durably_and_recover_without_a_panic() {
    let root = fixture();
    let connection = writable_connection(root.path()).unwrap();
    ensure_recognition_schema(&connection).unwrap();
    install_test_calibration(&connection);
    rebuild_person_prototype_in(&connection, 1, MODEL).unwrap();
    rebuild_person_prototype_in(&connection, 2, MODEL).unwrap();
    connection
        .execute(
            "UPDATE face_embeddings SET dimensions=2, vector=?1 WHERE face_id=3",
            [vector(&[0.0, 1.0])],
        )
        .unwrap();
    connection
        .execute(
            "UPDATE person_prototypes SET dimensions=2, vector=?1 WHERE person_id=2",
            [vector(&[0.0, 1.0])],
        )
        .unwrap();
    enqueue_refresh_in_connection(&connection, &[], &[4]).unwrap();
    drop(connection);
    assert_eq!(
        process_pending_refresh_batch(
            root.path(),
            "dimension-worker",
            &MaintenanceAdmission::new(),
            &AtomicBool::new(false),
        )
        .unwrap(),
        0
    );
    let connection = writable_connection(root.path()).unwrap();
    let (status, error): (String, String) = connection
        .query_row(
            "SELECT status, error FROM recognition_refresh_queue
                 WHERE entity_kind='face' AND entity_id=4",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .unwrap();
    assert_eq!(status, "error");
    assert!(error.contains("nonzero dimension"));
    connection
        .execute(
            "UPDATE face_embeddings SET dimensions=3, vector=?1 WHERE face_id=3",
            [vector(&[0.0, 1.0, 0.0])],
        )
        .unwrap();
    rebuild_person_prototype_in(&connection, 2, MODEL).unwrap();
    enqueue_refresh_in_connection(&connection, &[], &[4]).unwrap();
    drop(connection);
    assert_eq!(
        process_pending_refresh_batch(
            root.path(),
            "dimension-worker",
            &MaintenanceAdmission::new(),
            &AtomicBool::new(false),
        )
        .unwrap(),
        1
    );
}

#[test]
fn prototype_generation_invalidates_the_process_wide_ann_cache() {
    let root = fixture();
    let connection = writable_connection(root.path()).unwrap();
    ensure_recognition_schema(&connection).unwrap();
    rebuild_person_prototype_in(&connection, 1, MODEL).unwrap();
    rebuild_person_prototype_in(&connection, 2, MODEL).unwrap();
    let first = cached_prototype_index(root.path(), &connection, MODEL, None).unwrap();
    rebuild_person_prototype_in(&connection, 1, MODEL).unwrap();
    let second = cached_prototype_index(root.path(), &connection, MODEL, None).unwrap();
    assert!(!Arc::ptr_eq(&first, &second));
    assert_ne!(
        first
            .prototypes
            .iter()
            .find(|prototype| prototype.person_id == 1)
            .unwrap()
            .generation,
        second
            .prototypes
            .iter()
            .find(|prototype| prototype.person_id == 1)
            .unwrap()
            .generation
    );
}

#[test]
fn index_waiter_cancels_within_one_hundred_ms_and_other_catalog_progresses() {
    let blocked_root = fixture();
    let connection = writable_connection(blocked_root.path()).unwrap();
    ensure_recognition_schema(&connection).unwrap();
    rebuild_person_prototype_in(&connection, 1, MODEL).unwrap();
    let generation: i64 = connection
        .query_row(
            "SELECT generation FROM recognition_index_state WHERE model_id=?1",
            [MODEL],
            |row| row.get(0),
        )
        .unwrap();
    let key = (
        blocked_root.path().to_path_buf(),
        MODEL.to_owned(),
        generation,
    );
    let build = Arc::new(IndexBuild {
        result: Mutex::new(None),
        wake: Condvar::new(),
    });
    PROTOTYPE_INDEX_CACHE
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap()
        .insert(key.clone(), build);
    drop(connection);

    let cancelled = Arc::new(AtomicBool::new(false));
    let waiter_cancelled = cancelled.clone();
    let waiter_root = blocked_root.path().to_path_buf();
    let (tx, rx) = std::sync::mpsc::channel();
    let waiter = std::thread::spawn(move || {
        let connection = writable_connection(&waiter_root).unwrap();
        tx.send(cached_prototype_index(
            &waiter_root,
            &connection,
            MODEL,
            Some((&MaintenanceAdmission::new(), &waiter_cancelled)),
        ))
        .unwrap();
    });
    let other_root = fixture();
    let other_connection = writable_connection(other_root.path()).unwrap();
    ensure_recognition_schema(&other_connection).unwrap();
    rebuild_person_prototype_in(&other_connection, 1, MODEL).unwrap();
    assert!(cached_prototype_index(other_root.path(), &other_connection, MODEL, None).is_ok());

    let started = std::time::Instant::now();
    cancelled.store(true, Ordering::Release);
    let result = rx
        .recv_timeout(std::time::Duration::from_millis(100))
        .expect("index waiter did not release within 100 ms");
    let error = result.err().expect("cancelled waiter returned an index");
    assert!(error.to_string().contains("cancelled"));
    assert!(started.elapsed() <= std::time::Duration::from_millis(100));
    waiter.join().unwrap();
    PROTOTYPE_INDEX_CACHE
        .get()
        .unwrap()
        .lock()
        .unwrap()
        .remove(&key);
}

#[test]
fn model_build_contention_cancels_within_one_hundred_ms_without_blocking_other_keys() {
    let root = tempfile::tempdir().unwrap();
    let held = acquire_model_build_permit(root.path(), MODEL, None).unwrap();
    let cancelled = Arc::new(AtomicBool::new(false));
    let waiter_cancelled = cancelled.clone();
    let waiter_root = root.path().to_path_buf();
    let (tx, rx) = std::sync::mpsc::channel();
    let waiter = std::thread::spawn(move || {
        tx.send(acquire_model_build_permit(
            &waiter_root,
            MODEL,
            Some((&MaintenanceAdmission::new(), &waiter_cancelled)),
        ))
        .unwrap();
    });

    let unrelated = acquire_model_build_permit(root.path(), "unrelated-model", None)
        .expect("an unrelated model must not wait behind the active build");
    drop(unrelated);

    let started = std::time::Instant::now();
    cancelled.store(true, Ordering::Release);
    let result = rx
        .recv_timeout(std::time::Duration::from_millis(100))
        .expect("contended build did not release within 100 ms");
    let error = result.err().expect("cancelled build acquired the permit");
    assert!(error.to_string().contains("cancelled"));
    assert!(started.elapsed() <= std::time::Duration::from_millis(100));
    waiter.join().unwrap();
    drop(held);
}

#[test]
fn older_generation_completion_preserves_newer_inflight_single_flight_entry() {
    let root = PathBuf::from("/catalog-overlap-regression");
    let old_key = (root.clone(), MODEL.to_owned(), 1);
    let new_key = (root.clone(), MODEL.to_owned(), 2);
    let old = Arc::new(IndexBuild {
        result: Mutex::new(Some(Err(Arc::from("completed marker")))),
        wake: Condvar::new(),
    });
    let newer = Arc::new(IndexBuild {
        result: Mutex::new(None),
        wake: Condvar::new(),
    });
    let cache = Mutex::new(HashMap::from([
        (old_key.clone(), old.clone()),
        (new_key.clone(), newer.clone()),
    ]));

    finalize_index_build_cache(&cache, &root, MODEL, 1, &old_key, &old, true);

    let cache = cache.lock().unwrap();
    assert!(
        cache
            .get(&new_key)
            .is_some_and(|entry| Arc::ptr_eq(entry, &newer))
    );
}

#[test]
fn cancelled_cached_builder_is_removed_and_retry_succeeds() {
    let root = fixture();
    let connection = writable_connection(root.path()).unwrap();
    ensure_recognition_schema(&connection).unwrap();
    rebuild_person_prototype_in(&connection, 1, MODEL).unwrap();
    let held = acquire_model_build_permit(root.path(), MODEL, None).unwrap();
    let cancelled = AtomicBool::new(true);
    let first = cached_prototype_index(
        root.path(),
        &connection,
        MODEL,
        Some((&MaintenanceAdmission::new(), &cancelled)),
    );
    assert!(
        first
            .err()
            .expect("cancelled builder unexpectedly returned an index")
            .to_string()
            .contains("cancelled")
    );
    drop(held);
    cancelled.store(false, Ordering::Release);
    assert!(cached_prototype_index(root.path(), &connection, MODEL, None).is_ok());
}

#[test]
fn poisoned_model_gate_recovers_without_stranding_waiters() {
    let root = tempfile::tempdir().unwrap();
    let gates = MODEL_BUILD_GATES.get_or_init(|| Mutex::new(HashMap::new()));
    let gate = Arc::new(BuildGate {
        active: Mutex::new(false),
        wake: Condvar::new(),
    });
    gates
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .insert((root.path().to_path_buf(), MODEL.to_owned()), gate.clone());
    let poisoned = gate.clone();
    assert!(
        std::thread::spawn(move || {
            let _guard = poisoned.active.lock().unwrap();
            panic!("intentional gate poison");
        })
        .join()
        .is_err()
    );
    let started = std::time::Instant::now();
    let permit = acquire_model_build_permit(root.path(), MODEL, None).unwrap();
    assert!(started.elapsed() < std::time::Duration::from_millis(100));
    drop(permit);
}

#[test]
fn five_hundred_twelve_dimension_generation_churn_releases_old_indexes() {
    let prototypes = (0..300)
        .map(|person_id| {
            let mut centroid = (0..512)
                .map(|dimension| ((person_id * 17 + dimension * 31 + 1) % 997) as f32 + 1.0)
                .collect::<Vec<_>>();
            normalize(&mut centroid).unwrap();
            Prototype {
                person_id,
                generation: 1,
                exemplars: vec![centroid.clone()],
                centroid,
            }
        })
        .collect::<Vec<_>>();
    for generation in 1..=3 {
        let index = Arc::new(build_prototype_index(prototypes.clone(), generation, None).unwrap());
        let old = Arc::downgrade(&index);
        assert_eq!(index.prototypes.len(), 300);
        drop(index);
        assert!(
            old.upgrade().is_none(),
            "old ANN generation remained retained"
        );
    }
    let payload = prototypes
        .iter()
        .map(|prototype| {
            prototype.centroid.len() + prototype.exemplars.iter().map(Vec::len).sum::<usize>()
        })
        .sum::<usize>()
        * size_of::<f32>();
    assert!(payload < 2 * 1024 * 1024);
}

#[test]
fn real_dimension_multi_query_is_exact_and_metric_prunes_ann_candidates() {
    let prototypes = (0..5_000_usize)
        .map(|person_id| {
            let cluster = person_id % 32;
            let mut centroid = vec![0.0_f32; 512];
            centroid[cluster] = 1.0;
            for (dimension, value) in centroid.iter_mut().enumerate().skip(32) {
                *value = (((person_id * 17 + dimension * 13) % 101) as f32 - 50.0) * 0.000_02;
            }
            normalize(&mut centroid).unwrap();
            Prototype {
                person_id: person_id as i64,
                generation: 1,
                exemplars: vec![centroid.clone()],
                centroid,
            }
        })
        .collect::<Vec<_>>();
    let index = build_prototype_index(prototypes, 1, None).unwrap();
    let admission = MaintenanceAdmission::new();
    let cancelled = AtomicBool::new(false);
    let started = std::time::Instant::now();
    let mut visited = 0_usize;
    let mut unseeded_visited = 0_usize;
    for query_id in (0..5_000).step_by(251).take(20) {
        let query = &index.prototypes[query_id].centroid;
        let seeds = prototype_candidate_ids(&index, query, ANN_CANDIDATES);
        let (best, second, query_visited) = certified_best_two(
            &index,
            query,
            &HashSet::new(),
            &seeds,
            &admission,
            &cancelled,
        )
        .unwrap();
        visited += query_visited;
        let (unseeded_best, unseeded_second, query_unseeded) =
            certified_best_two(&index, query, &HashSet::new(), &[], &admission, &cancelled)
                .unwrap();
        unseeded_visited += query_unseeded;
        let mut exact = index
            .prototypes
            .iter()
            .enumerate()
            .map(|(id, prototype)| (id, metric_distance(query, &prototype.centroid)))
            .collect::<Vec<_>>();
        exact.sort_by(|left, right| {
            left.1
                .total_cmp(&right.1)
                .then_with(|| left.0.cmp(&right.0))
        });
        assert_eq!(best.unwrap().0, exact[0].0);
        assert_eq!(second.unwrap().0, exact[1].0);
        assert_eq!(unseeded_best.unwrap().0, exact[0].0);
        assert_eq!(unseeded_second.unwrap().0, exact[1].0);
    }
    assert!(
        visited * 10 < unseeded_visited * 9,
        "ANN seeds did not reduce dimension work by 10%: {visited} vs {unseeded_visited}"
    );
    assert!(started.elapsed() < std::time::Duration::from_secs(10));
}

#[cfg(target_os = "linux")]
#[test]
fn ann_resource_512d_cached_generation_overlap_stays_within_budget() {
    fn resident_bytes() -> usize {
        let statm = fs::read_to_string("/proc/self/statm").unwrap();
        let resident_pages = statm
            .split_whitespace()
            .nth(1)
            .unwrap()
            .parse::<usize>()
            .unwrap();
        resident_pages * 4_096
    }
    let root = fixture();
    let mut connection = writable_connection(root.path()).unwrap();
    ensure_recognition_schema(&connection).unwrap();
    let transaction = connection.transaction().unwrap();
    for person_id in 20_000..21_000_i64 {
        let mut embedding = (0..512)
            .map(|dimension| ((person_id as usize * 17 + dimension * 31) % 997) as f32 + 1.0)
            .collect::<Vec<_>>();
        normalize(&mut embedding).unwrap();
        let bytes = encode_embedding(&embedding);
        transaction
            .execute(
                "INSERT INTO people(id, display_name, created_at, updated_at)
                 VALUES(?1, ?2, datetime('now'), datetime('now'))",
                params![person_id, format!("resource-{person_id}")],
            )
            .unwrap();
        transaction
            .execute(
                "INSERT INTO faces(id, asset_id, provenance, source_key, left, top, right, bottom,
                 origin, updated_at) VALUES(?1, 1, 'rekesz', ?2, .1, .1, .2, .2,
                 'rekesz-detected', datetime('now'))",
                params![person_id, format!("resource-{person_id}")],
            )
            .unwrap();
        transaction
            .execute(
                "INSERT INTO face_assignments(face_id, person_id, kind, provenance, updated_at)
                 VALUES(?1, ?1, 'confirmed', 'rekesz', datetime('now'))",
                [person_id],
            )
            .unwrap();
        transaction
            .execute(
                "INSERT INTO face_embeddings(face_id, model_id, dimensions, vector, created_at)
                 VALUES(?1, ?2, 512, ?3, datetime('now'))",
                params![person_id, MODEL, &bytes],
            )
            .unwrap();
        transaction
            .execute(
                "INSERT INTO person_prototypes(person_id, model_id, dimensions, vector,
                 exemplar_count, algorithm_version, generation, updated_at)
                 VALUES(?1, ?2, 512, ?3, 1, ?4, 1, datetime('now'))",
                params![person_id, MODEL, &bytes, PROTOTYPE_ALGORITHM_VERSION],
            )
            .unwrap();
    }
    bump_index_generation(&transaction, MODEL).unwrap();
    transaction.commit().unwrap();
    let baseline = resident_bytes();
    let old = cached_prototype_index(root.path(), &connection, MODEL, None).unwrap();
    bump_index_generation(&connection, MODEL).unwrap();
    let replacement = cached_prototype_index(root.path(), &connection, MODEL, None).unwrap();
    let peak_delta = resident_bytes().saturating_sub(baseline);
    assert!(!Arc::ptr_eq(&old, &replacement));
    assert_eq!(old.prototypes.len(), replacement.prototypes.len());
    let cached_generations = PROTOTYPE_INDEX_CACHE
        .get()
        .unwrap()
        .lock()
        .unwrap()
        .keys()
        .filter(|(path, model, _)| path == root.path() && model == MODEL)
        .count();
    assert_eq!(cached_generations, 1);
    assert!(
        peak_delta < 256 * 1024 * 1024,
        "ANN_RESOURCE peak_rss_delta={peak_delta}"
    );
    eprintln!(
        "ANN_RESOURCE dimensions=512 people=1000 active_generations=2 peak_rss_delta={peak_delta}"
    );
    drop(old);
    drop(replacement);
}

/// Release-only evidence for the catalog ceiling. This deliberately creates a
/// large on-disk fixture and retains the prior index while its replacement is
/// built, so it must not run as part of the ordinary developer test suite.
#[cfg(target_os = "linux")]
#[test]
#[ignore = "release resource gate: builds two 100k-person 512-D indexes"]
fn ann_release_100k_512d_cached_overlap_throughput_and_resources() {
    fn resident_bytes() -> usize {
        let statm = fs::read_to_string("/proc/self/statm").unwrap();
        statm
            .split_whitespace()
            .nth(1)
            .unwrap()
            .parse::<usize>()
            .unwrap()
            * 4_096
    }

    let root = fixture();
    let mut connection = writable_connection(root.path()).unwrap();
    ensure_recognition_schema(&connection).unwrap();
    connection
        .execute_batch(
            "DELETE FROM face_assignments;
                 DELETE FROM face_embeddings;
                 DELETE FROM person_prototypes;",
        )
        .unwrap();
    let transaction = connection.transaction().unwrap();
    let mut person_insert = transaction
        .prepare_cached(
            "INSERT INTO people(id, display_name, created_at, updated_at)
                 VALUES(?1, ?2, datetime('now'), datetime('now'))",
        )
        .unwrap();
    let mut face_insert = transaction
        .prepare_cached(
            "INSERT INTO faces(id, asset_id, provenance, source_key, left, top, right, bottom,
                 origin, updated_at) VALUES(?1, 1, 'rekesz', ?2, .1, .1, .2, .2,
                 'rekesz-detected', datetime('now'))",
        )
        .unwrap();
    let mut assignment_insert = transaction
        .prepare_cached(
            "INSERT INTO face_assignments(face_id, person_id, kind, provenance, updated_at)
                 VALUES(?1, ?1, 'confirmed', 'rekesz', datetime('now'))",
        )
        .unwrap();
    let mut embedding_insert = transaction
        .prepare_cached(
            "INSERT INTO face_embeddings(face_id, model_id, dimensions, vector, created_at)
                 VALUES(?1, ?2, 512, ?3, datetime('now'))",
        )
        .unwrap();
    let mut prototype_insert = transaction
        .prepare_cached(
            "INSERT INTO person_prototypes(person_id, model_id, dimensions, vector,
                 exemplar_count, algorithm_version, generation, updated_at)
                 VALUES(?1, ?2, 512, ?3, 1, ?4, 1, datetime('now'))",
        )
        .unwrap();
    for offset in 0..100_000_usize {
        let person_id = 100_000 + offset as i64;
        // Catalog identity ordering is independent from embedding neighborhoods;
        // grouping here prevents the exact counterfactual from receiving an
        // unrealistic round-robin identity hint from insertion order.
        let cluster = (offset / 782).min(127);
        let mut vector = vec![0.0_f32; 512];
        vector[cluster] = 1.0;
        for (dimension, value) in vector.iter_mut().enumerate().skip(128) {
            *value = (((offset * 17 + dimension * 13) % 1_009) as f32 - 504.0) * 0.000_02;
        }
        normalize(&mut vector).unwrap();
        let bytes = encode_embedding(&vector);
        person_insert
            .execute(params![person_id, format!("scale-{person_id}")])
            .unwrap();
        face_insert
            .execute(params![person_id, format!("scale-{person_id}")])
            .unwrap();
        assignment_insert.execute([person_id]).unwrap();
        embedding_insert
            .execute(params![person_id, MODEL, &bytes])
            .unwrap();
        prototype_insert
            .execute(params![
                person_id,
                MODEL,
                &bytes,
                PROTOTYPE_ALGORITHM_VERSION
            ])
            .unwrap();
    }
    drop(prototype_insert);
    drop(embedding_insert);
    drop(assignment_insert);
    drop(face_insert);
    drop(person_insert);
    bump_index_generation(&transaction, MODEL).unwrap();
    transaction.commit().unwrap();

    let baseline = resident_bytes();
    let peak = Arc::new(std::sync::atomic::AtomicUsize::new(baseline));
    let monitor_stop = Arc::new(AtomicBool::new(false));
    let monitor_peak = peak.clone();
    let monitor_cancel = monitor_stop.clone();
    let monitor = std::thread::spawn(move || {
        while !monitor_cancel.load(Ordering::Acquire) {
            monitor_peak.fetch_max(resident_bytes(), Ordering::AcqRel);
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        monitor_peak.fetch_max(resident_bytes(), Ordering::AcqRel);
    });
    let build_started = std::time::Instant::now();
    let old = cached_prototype_index(root.path(), &connection, MODEL, None).unwrap();
    bump_index_generation(&connection, MODEL).unwrap();
    let replacement = cached_prototype_index(root.path(), &connection, MODEL, None).unwrap();
    let build_elapsed = build_started.elapsed();
    assert!(!Arc::ptr_eq(&old, &replacement));
    drop(old);
    let cancel_at = |phase| {
        bump_index_generation(&connection, MODEL).unwrap();
        let hook = Arc::new(TestBuildHook {
            requested: phase,
            reached: Mutex::new(false),
            released: Mutex::new(false),
            wake: Condvar::new(),
        });
        TEST_BUILD_HOOKS
            .get_or_init(|| Mutex::new(HashMap::new()))
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(root.path().to_path_buf(), hook.clone());
        let cancellation = Arc::new(AtomicBool::new(false));
        let worker_cancellation = cancellation.clone();
        let worker_root = root.path().to_path_buf();
        let (cancel_tx, cancel_rx) = std::sync::mpsc::channel();
        let cancel_worker = std::thread::spawn(move || {
            let connection = writable_connection(&worker_root).unwrap();
            cancel_tx
                .send(cached_prototype_index(
                    &worker_root,
                    &connection,
                    MODEL,
                    Some((&MaintenanceAdmission::new(), &worker_cancellation)),
                ))
                .unwrap();
        });
        let reached = hook.reached.lock().unwrap();
        let (reached, timeout) = hook
            .wake
            .wait_timeout_while(reached, std::time::Duration::from_secs(60), |ready| !*ready)
            .unwrap();
        assert!(*reached && !timeout.timed_out());
        let started = std::time::Instant::now();
        cancellation.store(true, Ordering::Release);
        *hook.released.lock().unwrap() = true;
        hook.wake.notify_all();
        let result = cancel_rx
            .recv_timeout(std::time::Duration::from_millis(100))
            .expect("100k 512-D build did not cancel within 100 ms");
        assert!(result.err().unwrap().to_string().contains("cancelled"));
        let elapsed = started.elapsed();
        cancel_worker.join().unwrap();
        TEST_BUILD_HOOKS
            .get()
            .unwrap()
            .lock()
            .unwrap()
            .remove(root.path());
        elapsed
    };
    let row_cancellation = cancel_at(TestBuildPhase::SqlRows);
    let metric_cancellation = cancel_at(TestBuildPhase::MetricNodes);
    let hnsw_cancellation = cancel_at(TestBuildPhase::HnswInsertions);
    assert!(row_cancellation <= std::time::Duration::from_millis(100));
    assert!(metric_cancellation <= std::time::Duration::from_millis(100));
    assert!(hnsw_cancellation <= std::time::Duration::from_millis(100));
    let retry_started = std::time::Instant::now();
    let retried = cached_prototype_index(root.path(), &connection, MODEL, None).unwrap();
    let retry_elapsed = retry_started.elapsed();
    monitor_stop.store(true, Ordering::Release);
    monitor.join().unwrap();
    let peak_delta = peak.load(Ordering::Acquire).saturating_sub(baseline);

    let admission = MaintenanceAdmission::new();
    let cancelled = AtomicBool::new(false);
    let query_started = std::time::Instant::now();
    let mut production_query_time = std::time::Duration::ZERO;
    let mut visited = 0_usize;
    let mut unseeded_visited = 0_usize;
    for query_id in (0..100_000).step_by(4_999).take(20) {
        let centroid = &replacement.prototypes[query_id].centroid;
        let mut noise = (0..512)
            .map(|dimension| {
                (((query_id * 37 + dimension * 101 + 17) % 1_009) as f32 - 504.0) / 504.0
            })
            .collect::<Vec<_>>();
        let projection = dot(&noise, centroid);
        for (noise, centroid) in noise.iter_mut().zip(centroid) {
            *noise -= projection * centroid;
        }
        normalize(&mut noise).unwrap();
        let mut query = centroid
            .iter()
            .zip(noise)
            .map(|(centroid, noise)| centroid * 0.92 + noise * 0.392)
            .collect::<Vec<_>>();
        normalize(&mut query).unwrap();
        let held_out_similarity = dot(centroid, &query);
        assert!((0.90..=0.94).contains(&held_out_similarity));
        let seeds = prototype_candidate_ids(&replacement, &query, ANN_CANDIDATES);
        let production_started = std::time::Instant::now();
        let (best, second, query_visited) = certified_best_two(
            &replacement,
            &query,
            &HashSet::new(),
            &seeds,
            &admission,
            &cancelled,
        )
        .unwrap();
        production_query_time += production_started.elapsed();
        let (unseeded_best, unseeded_second, empty_visited) = certified_best_two(
            &replacement,
            &query,
            &HashSet::new(),
            &[],
            &admission,
            &cancelled,
        )
        .unwrap();
        let mut exact = replacement
            .prototypes
            .iter()
            .enumerate()
            .map(|(id, prototype)| (id, metric_distance(&query, &prototype.centroid)))
            .collect::<Vec<_>>();
        exact.sort_by(|left, right| {
            left.1
                .total_cmp(&right.1)
                .then_with(|| left.0.cmp(&right.0))
        });
        assert_eq!(best.unwrap().0, exact[0].0);
        assert_eq!(second.unwrap().0, exact[1].0);
        assert_eq!(unseeded_best.unwrap().0, exact[0].0);
        assert_eq!(unseeded_second.unwrap().0, exact[1].0);
        visited += query_visited;
        unseeded_visited += empty_visited;
    }
    let query_elapsed = query_started.elapsed();
    eprintln!("ANN_QUERY_WORK seeded={visited} unseeded={unseeded_visited}");
    assert!(
        visited * 10 < unseeded_visited * 9,
        "ANN seeds did not reduce certified work by at least 10%: seeded={visited} unseeded={unseeded_visited}"
    );
    assert!(production_query_time < std::time::Duration::from_secs(10));
    assert!(build_elapsed < std::time::Duration::from_secs(120));
    let rss_budget = std::env::var("PEOPLE_MAX_RSS_KIB")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(2_097_152)
        * 1_024;
    assert!(
        peak_delta <= rss_budget,
        "peak RSS delta {peak_delta} exceeded configured budget {rss_budget}"
    );
    eprintln!(
        "ANN_RELEASE_RESOURCE dimensions=512 people=100000 active_generations=2 \
             peak_rss_delta={peak_delta} build_ms={} production_query_ms={} validation_ms={} certified_nodes={visited} \
             unseeded_nodes={unseeded_visited} row_cancel_ms={} metric_cancel_ms={} hnsw_cancel_ms={} retry_ms={}",
        build_elapsed.as_millis(),
        production_query_time.as_millis(),
        query_elapsed.as_millis(),
        row_cancellation.as_millis(),
        metric_cancellation.as_millis(),
        hnsw_cancellation.as_millis(),
        retry_elapsed.as_millis()
    );
    drop(replacement);
    drop(retried);
}

#[test]
fn durable_refresh_queue_rebuilds_and_retries_without_partial_results() {
    let root = fixture();
    let connection = writable_connection(root.path()).unwrap();
    install_test_calibration(&connection);
    enqueue_refresh_in_connection(&connection, &[1, 2], &[4]).unwrap();
    assert_eq!(
        connection
            .query_row(
                "SELECT count(*) FROM recognition_refresh_queue",
                [],
                |row| { row.get::<_, i64>(0) }
            )
            .unwrap(),
        4
    );
    drop(connection);
    assert_eq!(process(root.path(), "worker-a").unwrap(), 3);
    let connection = writable_connection(root.path()).unwrap();
    assert_eq!(
        connection
            .query_row(
                "SELECT count(*) FROM face_assignments
                     WHERE face_id=4 AND person_id=1 AND kind='suggested' AND provenance='rekesz'",
                [],
                |row| row.get::<_, i64>(0),
            )
            .unwrap(),
        1
    );
    let generations: (i64, i64) = connection
        .query_row(
            "SELECT prototype.generation, suggestion.prototype_generation
             FROM person_prototypes prototype JOIN recognition_suggestions suggestion
               ON suggestion.person_id=prototype.person_id
             WHERE prototype.person_id=1 AND prototype.model_id=?1 AND suggestion.face_id=4",
            [MODEL],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .unwrap();
    assert_eq!(generations.0, generations.1);
    connection
        .execute(
            "UPDATE face_embeddings SET dimensions=4 WHERE face_id=1 AND model_id=?1",
            [MODEL],
        )
        .unwrap();
    enqueue_refresh_in_connection(&connection, &[1], &[]).unwrap();
    drop(connection);
    assert_eq!(process(root.path(), "worker-b").unwrap(), 1);
    let connection = writable_connection(root.path()).unwrap();
    let failed: (String, i64, String) = connection
        .query_row(
            "SELECT status, attempts, error FROM recognition_refresh_queue
                 WHERE entity_kind='person' AND entity_id=1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .unwrap();
    assert_eq!(failed.0, "pending");
    assert_eq!(failed.1, 1);
    assert!(failed.2.contains("dimensions"));
    assert_eq!(
        connection
            .query_row(
                "SELECT generation FROM person_prototypes WHERE person_id=1 AND model_id=?1",
                [MODEL],
                |row| row.get::<_, i64>(0),
            )
            .unwrap(),
        1,
        "failed rebuild must roll back its partial generation change"
    );
}

#[test]
fn automatic_recognition_is_inert_without_a_matching_calibration_artifact() {
    let root = fixture();
    let connection = writable_connection(root.path()).unwrap();
    enqueue_refresh_in_connection(&connection, &[], &[4]).unwrap();
    drop(connection);
    assert_eq!(process(root.path(), "uncalibrated").unwrap(), 1);
    let connection = writable_connection(root.path()).unwrap();
    assert_eq!(
        connection
            .query_row(
                "SELECT count(*) FROM face_assignments
                     WHERE face_id=4 AND provenance='rekesz' AND kind='suggested'",
                [],
                |row| row.get::<_, i64>(0),
            )
            .unwrap(),
        0
    );
}

#[test]
fn calibration_install_queues_global_prototype_and_candidate_rebuild() {
    let root = fixture();
    install_calibration_artifact(root.path(), MODEL, RecognitionPolicy::default(), &[7; 32])
        .unwrap();
    let connection = writable_connection(root.path()).unwrap();
    assert_eq!(
        connection
            .query_row(
                "SELECT count(*) FROM recognition_policy_artifacts WHERE model_id=?1 AND enabled=1",
                [MODEL],
                |row| row.get::<_, i64>(0)
            )
            .unwrap(),
        1
    );
    assert_eq!(
        connection
            .query_row(
                "SELECT count(*) FROM recognition_refresh_queue WHERE model_id=?1",
                [MODEL],
                |row| row.get::<_, i64>(0)
            )
            .unwrap(),
        3
    );
    assert_eq!(
        connection
            .query_row(
                "SELECT after_face_id FROM recognition_rescore_state WHERE model_id=?1",
                [MODEL],
                |row| row.get::<_, i64>(0)
            )
            .unwrap(),
        0
    );
}

#[test]
fn shipped_policy_activates_fresh_catalog_and_confirmation_improves_later_suggestion() {
    let root = fixture();
    let connection = writable_connection(root.path()).unwrap();
    connection
        .execute(
            "UPDATE face_embeddings SET model_id=?1",
            [PRODUCTION_SFACE_MODEL_ID],
        )
        .unwrap();
    connection
        .execute("DELETE FROM face_assignments", [])
        .unwrap();
    drop(connection);

    assert!(ensure_production_calibration(root.path()).unwrap());
    assert!(!ensure_production_calibration(root.path()).unwrap());
    let status = production_calibration_status(&writable_connection(root.path()).unwrap()).unwrap();
    assert_eq!(
        status,
        CalibrationStatus {
            active: true,
            policy_version: RECOGNITION_POLICY_VERSION,
            artifact_digest: PRODUCTION_SFACE_POLICY_DIGEST.to_owned(),
        }
    );
    let diagnostics = crate::catalog::people::diagnostics(root.path()).unwrap();
    assert!(diagnostics.recognition_policy_active);
    assert_eq!(
        diagnostics.recognition_policy_version,
        RECOGNITION_POLICY_VERSION
    );
    assert_eq!(
        diagnostics.recognition_policy_digest,
        PRODUCTION_SFACE_POLICY_DIGEST
    );

    crate::catalog::people::apply_decisions(
        root.path(),
        &crate::catalog::people::DecisionBatch {
            id: "production-accept".to_owned(),
            commands: vec![crate::catalog::people::DecisionCommand {
                face_id: 1,
                person_id: Some(1),
                action: crate::catalog::people::DecisionAction::Confirm,
            }],
        },
    )
    .unwrap();
    let admission = MaintenanceAdmission::new();
    let cancelled = AtomicBool::new(false);
    while process_pending_refresh_batch(root.path(), "production-worker", &admission, &cancelled)
        .unwrap()
        > 0
    {}
    let connection = writable_connection(root.path()).unwrap();
    assert_eq!(
        connection
            .query_row(
                "SELECT count(*) FROM person_prototypes
                     WHERE person_id=1 AND model_id=?1",
                [PRODUCTION_SFACE_MODEL_ID],
                |row| row.get::<_, i64>(0),
            )
            .unwrap(),
        1
    );
    assert_eq!(
        connection
            .query_row(
                "SELECT count(*) FROM face_assignments
                     WHERE face_id=4 AND person_id=1 AND kind='suggested'
                       AND provenance='rekesz'",
                [],
                |row| row.get::<_, i64>(0),
            )
            .unwrap(),
        1,
        "a real decision must feed the shipped policy and affect a later face"
    );
}

#[test]
fn algorithm_version_mismatch_invalidates_derived_prototypes_and_suggestions() {
    let root = fixture();
    rebuild_person_prototype(root.path(), 1, MODEL).unwrap();
    let connection = writable_connection(root.path()).unwrap();
    ensure_recognition_schema(&connection).unwrap();
    connection
        .execute(
            "UPDATE person_prototypes SET algorithm_version=0 WHERE person_id=1",
            [],
        )
        .unwrap();
    connection
        .execute(
            "INSERT INTO recognition_suggestions(face_id, person_id, recognizer_model_id,
                 prototype_generation, prototype_algorithm_version, policy_version, score,
                 nearest_exemplar_score, second_best_margin, created_at)
                 VALUES(4, 1, ?1, 1, 0, 0, .9, .9, .2, datetime('now'))",
            [MODEL],
        )
        .unwrap();
    connection
        .execute(
            "INSERT INTO face_assignments(face_id, person_id, kind, provenance, confidence,
             updated_at) VALUES(4, 1, 'suggested', 'rekesz', .9, datetime('now'))",
            [],
        )
        .unwrap();
    ensure_recognition_schema(&connection).unwrap();
    assert_eq!(
        connection
            .query_row(
                "SELECT (SELECT count(*) FROM person_prototypes),
                            (SELECT count(*) FROM recognition_suggestions),
                            (SELECT count(*) FROM face_assignments WHERE provenance='rekesz'
                             AND kind='suggested')",
                [],
                |row| Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, i64>(2)?
                )),
            )
            .unwrap(),
        (0, 0, 0)
    );
}

#[test]
fn cancelled_refresh_releases_claim_without_consuming_a_retry_and_reenqueue_resets_errors() {
    let root = fixture();
    let connection = writable_connection(root.path()).unwrap();
    install_test_calibration(&connection);
    enqueue_refresh_in_connection(&connection, &[], &[4]).unwrap();
    connection
        .execute(
            "UPDATE recognition_refresh_queue SET status='error', attempts=5, error='old'",
            [],
        )
        .unwrap();
    enqueue_refresh_in_connection(&connection, &[], &[4]).unwrap();
    assert_eq!(
        connection
            .query_row(
                "SELECT attempts FROM recognition_refresh_queue WHERE entity_kind='face'",
                [],
                |row| row.get::<_, i64>(0),
            )
            .unwrap(),
        0
    );
    drop(connection);
    let cancelled = AtomicBool::new(true);
    assert_eq!(
        process_pending_refresh_batch(
            root.path(),
            "cancelled",
            &MaintenanceAdmission::new(),
            &cancelled,
        )
        .unwrap(),
        0
    );
    let connection = writable_connection(root.path()).unwrap();
    assert_eq!(
        connection
            .query_row(
                "SELECT status, attempts FROM recognition_refresh_queue",
                [],
                |row| Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?)),
            )
            .unwrap(),
        ("pending".into(), 0)
    );
}

#[test]
fn malformed_queue_row_is_isolated_from_healthy_work() {
    let root = fixture();
    let connection = writable_connection(root.path()).unwrap();
    install_test_calibration(&connection);
    connection
        .execute_batch("PRAGMA ignore_check_constraints=ON")
        .unwrap();
    connection
        .execute(
            "INSERT INTO recognition_refresh_queue(entity_kind, entity_id, model_id,
                 status, updated_at) VALUES('poison', 999, ?1, 'pending', datetime('now'))",
            [MODEL],
        )
        .unwrap();
    connection
        .execute_batch("PRAGMA ignore_check_constraints=OFF")
        .unwrap();
    enqueue_refresh_in_connection(&connection, &[], &[4]).unwrap();
    drop(connection);
    assert_eq!(process(root.path(), "isolation").unwrap(), 1);
    let connection = writable_connection(root.path()).unwrap();
    assert_eq!(
        connection
            .query_row(
                "SELECT status FROM recognition_refresh_queue WHERE entity_kind='poison'",
                [],
                |row| row.get::<_, String>(0),
            )
            .unwrap(),
        "error"
    );
    assert_eq!(
        connection
            .query_row(
                "SELECT count(*) FROM recognition_refresh_queue WHERE entity_kind='face'",
                [],
                |row| row.get::<_, i64>(0),
            )
            .unwrap(),
        0
    );
}

#[test]
fn prototype_training_uses_deterministic_bounded_sampling_above_1024_confirmations() {
    let root = fixture();
    let mut connection = writable_connection(root.path()).unwrap();
    let transaction = connection.transaction().unwrap();
    for face_id in 10_000..11_025_i64 {
        transaction
            .execute(
                "INSERT INTO faces(id, asset_id, provenance, source_key, left, top, right, bottom,
                 origin, updated_at) VALUES(?1, 1, 'rekesz', ?2, .1, .1, .2, .2,
                 'rekesz-detected', datetime('now'))",
                params![face_id, format!("sample-{face_id}")],
            )
            .unwrap();
        transaction
            .execute(
                "INSERT INTO face_assignments(face_id, person_id, kind, provenance, updated_at)
                 VALUES(?1, 1, 'confirmed', 'rekesz', datetime('now'))",
                [face_id],
            )
            .unwrap();
        transaction
            .execute(
                "INSERT INTO face_embeddings(face_id, model_id, dimensions, vector, created_at)
                 VALUES(?1, ?2, 3, ?3, datetime('now'))",
                params![face_id, MODEL, vector(&[1.0, 0.0, 0.0])],
            )
            .unwrap();
    }
    transaction.commit().unwrap();
    drop(connection);
    rebuild_person_prototype(root.path(), 1, MODEL).unwrap();
    let connection = writable_connection(root.path()).unwrap();
    assert_eq!(
        connection
            .query_row(
                "SELECT exemplar_count FROM person_prototypes WHERE person_id=1 AND model_id=?1",
                [MODEL],
                |row| row.get::<_, i64>(0)
            )
            .unwrap(),
        1_024
    );
}

fn claimed_face_calculation(root: &Path) -> (Connection, i64, FaceCalculation) {
    rebuild_person_prototype(root, 1, MODEL).unwrap();
    rebuild_person_prototype(root, 2, MODEL).unwrap();
    let connection = writable_connection(root).unwrap();
    install_test_calibration(&connection);
    enqueue_refresh_in_connection(&connection, &[], &[4]).unwrap();
    let queue_id: i64 = connection
        .query_row(
            "SELECT id FROM recognition_refresh_queue WHERE entity_kind='face' AND entity_id=4",
            [],
            |row| row.get(0),
        )
        .unwrap();
    connection
        .execute(
            "UPDATE recognition_refresh_queue SET status='running', worker_id='race-worker',
             lease_expires_at=datetime('now', '+5 minutes') WHERE id=?1",
            [queue_id],
        )
        .unwrap();
    let prototypes = load_prototypes(&connection, MODEL, None).unwrap();
    let policy = RecognitionPolicy::default();
    let index_generation = connection
        .query_row(
            "SELECT generation FROM recognition_index_state WHERE model_id=?1",
            [MODEL],
            |row| row.get(0),
        )
        .unwrap();
    let prototype_index = build_prototype_index(prototypes, index_generation, None).unwrap();
    let suggestion = calculate_face_suggestion(
        &connection,
        4,
        MODEL,
        policy,
        &prototype_index,
        &MaintenanceAdmission::new(),
        &AtomicBool::new(false),
    )
    .unwrap();
    let generations = prototype_index
        .prototypes
        .iter()
        .map(|value| (value.person_id, value.generation))
        .collect();
    (
        connection,
        queue_id,
        FaceCalculation {
            suggestion,
            policy,
            ann_index_version: prototype_index.version,
            index_generation: prototype_index.generation,
            prototype_generations: generations,
        },
    )
}

#[test]
fn compare_and_store_discards_changed_face_or_prototype_snapshot() {
    for mutate_face in [true, false] {
        let root = fixture();
        let (connection, queue_id, calculation) = claimed_face_calculation(root.path());
        if mutate_face {
            connection
                .execute("UPDATE faces SET locally_ignored=1 WHERE id=4", [])
                .unwrap();
        } else {
            connection
                .execute(
                    "UPDATE person_prototypes SET generation=generation+1
                     WHERE person_id=1 AND model_id=?1",
                    [MODEL],
                )
                .unwrap();
        }
        assert!(
            store_calculated_suggestion(
                &connection,
                4,
                MODEL,
                &calculation,
                queue_id,
                "race-worker"
            )
            .unwrap()
            .is_none()
        );
        assert_eq!(
            connection
                .query_row(
                    "SELECT count(*) FROM face_assignments WHERE face_id=4 AND provenance='rekesz'
                 AND kind='suggested'",
                    [],
                    |row| row.get::<_, i64>(0)
                )
                .unwrap(),
            0
        );
        assert_eq!(
            connection
                .query_row(
                    "SELECT status FROM recognition_refresh_queue WHERE id=?1",
                    [queue_id],
                    |row| row.get::<_, String>(0)
                )
                .unwrap(),
            "pending"
        );
    }
}

#[test]
fn compare_and_store_discards_snapshot_when_a_new_prototype_appears() {
    let root = fixture();
    let (connection, queue_id, calculation) = claimed_face_calculation(root.path());
    connection
        .execute(
            "INSERT INTO people(id, display_name, created_at, updated_at)
                 VALUES(3, 'three', datetime('now'), datetime('now'))",
            [],
        )
        .unwrap();
    connection
        .execute(
            "INSERT INTO person_prototypes(person_id, model_id, dimensions, vector,
                 exemplar_count, algorithm_version, generation, updated_at)
                 VALUES(3, ?1, 3, ?2, 1, ?3, 1, datetime('now'))",
            params![
                MODEL,
                vector(&[0.98, 0.02, 0.0]),
                PROTOTYPE_ALGORITHM_VERSION
            ],
        )
        .unwrap();

    assert!(
        store_calculated_suggestion(&connection, 4, MODEL, &calculation, queue_id, "race-worker")
            .unwrap()
            .is_none()
    );
    assert_eq!(
        connection
            .query_row(
                "SELECT status FROM recognition_refresh_queue WHERE id=?1",
                [queue_id],
                |row| row.get::<_, String>(0)
            )
            .unwrap(),
        "pending"
    );
    assert_eq!(
        connection
            .query_row(
                "SELECT count(*) FROM recognition_suggestions WHERE face_id=4",
                [],
                |row| row.get::<_, i64>(0)
            )
            .unwrap(),
        0
    );
}

#[test]
fn generation_bump_hides_old_rekesz_suggestion_before_rescore() {
    let root = fixture();
    let (connection, queue_id, calculation) = claimed_face_calculation(root.path());
    assert!(
        store_calculated_suggestion(&connection, 4, MODEL, &calculation, queue_id, "race-worker",)
            .unwrap()
            .is_some()
    );
    assert_eq!(
        connection
            .query_row(
                "SELECT count(*) FROM effective_face_assignments
                     WHERE face_id=4 AND provenance='rekesz' AND kind='suggested'",
                [],
                |row| row.get::<_, i64>(0),
            )
            .unwrap(),
        1
    );
    bump_index_generation(&connection, MODEL).unwrap();
    assert_eq!(
        connection
            .query_row(
                "SELECT count(*) FROM effective_face_assignments
                     WHERE face_id=4 AND provenance='rekesz' AND kind='suggested'",
                [],
                |row| row.get::<_, i64>(0),
            )
            .unwrap(),
        0
    );
}

#[test]
fn expired_reclaimed_lease_cannot_commit_an_old_calculation() {
    let root = fixture();
    let (connection, queue_id, calculation) = claimed_face_calculation(root.path());
    connection
        .execute(
            "UPDATE recognition_refresh_queue SET worker_id='new-worker',
             lease_expires_at=datetime('now', '+5 minutes') WHERE id=?1",
            [queue_id],
        )
        .unwrap();
    assert!(
        store_calculated_suggestion(&connection, 4, MODEL, &calculation, queue_id, "race-worker")
            .unwrap()
            .is_none()
    );
    assert_eq!(
        connection
            .query_row(
                "SELECT count(*) FROM recognition_suggestions WHERE face_id=4",
                [],
                |row| row.get::<_, i64>(0)
            )
            .unwrap(),
        0
    );
}

#[test]
fn hundred_thousand_face_rescore_expands_a_bounded_rolling_window() {
    let root = fixture();
    let connection = writable_connection(root.path()).unwrap();
    connection.execute_batch(
            "WITH RECURSIVE n(value) AS (VALUES(100000) UNION ALL SELECT value+1 FROM n WHERE value<199999)
             INSERT INTO faces(id, asset_id, provenance, source_key, left, top, right, bottom,
                               origin, updated_at)
             SELECT value, 1, 'rekesz', printf('large-%d', value), .1, .1, .2, .2,
                    'rekesz-detected', datetime('now') FROM n;
             INSERT INTO face_embeddings(face_id, model_id, dimensions, vector, created_at)
             SELECT id, 'test-recognizer-v1', 3, X'0000803f0000000000000000', datetime('now')
             FROM faces WHERE id>=100000;",
        ).unwrap();
    enqueue_global_rescore(&connection, MODEL).unwrap();
    for _ in 0..10 {
        expand_global_rescore(&connection, MODEL).unwrap();
    }
    let (queued, cursor): (i64, i64) = connection
        .query_row(
            "SELECT (SELECT count(*) FROM recognition_refresh_queue WHERE entity_kind='face'),
                    after_face_id FROM recognition_rescore_state WHERE model_id=?1",
            [MODEL],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .unwrap();
    assert_eq!(queued, MAX_QUEUED_FACES as i64);
    assert!((100_000..199_999).contains(&cursor));
}

#[test]
fn rolling_rescore_expansion_keeps_its_sole_marker_pending() {
    let root = fixture();
    let connection = writable_connection(root.path()).unwrap();
    install_test_calibration(&connection);
    enqueue_global_rescore(&connection, MODEL).unwrap();
    drop(connection);

    assert_eq!(
        process_pending_refresh_batch(
            root.path(),
            "marker-worker",
            &MaintenanceAdmission::new(),
            &AtomicBool::new(false)
        )
        .unwrap(),
        1
    );
    let connection = writable_connection(root.path()).unwrap();
    assert_eq!(
        connection
            .query_row(
                "SELECT status FROM recognition_refresh_queue
                     WHERE entity_kind='person' AND entity_id=0 AND model_id=?1",
                [MODEL],
                |row| row.get::<_, String>(0)
            )
            .unwrap(),
        "pending"
    );
    assert!(
        connection
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM recognition_refresh_queue
                 WHERE entity_kind='face' AND model_id=?1)",
                [MODEL],
                |row| row.get::<_, bool>(0)
            )
            .unwrap()
    );
}
