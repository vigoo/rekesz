use super::*;
use crate::rendering::FaceTransform;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

fn lock_ort() -> std::sync::MutexGuard<'static, ()> {
    ORT_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

struct CancellingStageHook {
    target: FaceAnalysisStage,
    observed: Arc<AtomicUsize>,
}

impl ProductionAnalysisStages for CancellingStageHook {
    fn reached(&self, stage: FaceAnalysisStage, cancelled: &Arc<AtomicBool>) {
        self.observed
            .store(stage as usize + 1, AtomicOrdering::Release);
        if stage == self.target {
            cancelled.store(true, Ordering::Release);
        }
    }
}

fn durable_state(root: &Path, asset_id: i64) -> String {
    Connection::open(catalog::catalog_path(root))
        .unwrap()
        .query_row(
            "SELECT status FROM face_analysis_state WHERE asset_id=?1",
            [asset_id],
            |row| row.get(0),
        )
        .unwrap()
}

#[test]
fn packaged_face_models_licenses_and_policy_match_release_manifest() {
    validate_packaged_face_assets().unwrap();
}

#[test]
fn persisted_analysis_error_categories_never_include_source_paths() {
    for (error, expected) in [
        (
            anyhow::anyhow!("cannot decode /private/alice/photo.jpg"),
            "source-io-failed",
        ),
        (
            anyhow::anyhow!("unsupported /secret/file.xyz"),
            "unsupported-format",
        ),
        (
            anyhow::anyhow!("ONNX model failed at /home/alice"),
            "inference-model-failed",
        ),
        (
            anyhow::anyhow!("failure for C:\\Users\\Alice\\face.jpg"),
            "analysis-failed",
        ),
    ] {
        let (category, _) = analysis_error_category(&error);
        assert_eq!(category, expected);
        assert!(!category.contains('/') && !category.contains('\\'));
    }
}

#[cfg(target_os = "linux")]
fn deny_network_syscalls_for_current_thread() {
    let mut filter = [
        libc::sock_filter {
            code: 0x20,
            jt: 0,
            jf: 0,
            k: 0,
        },
        libc::sock_filter {
            code: 0x15,
            jt: 0,
            jf: 1,
            k: libc::SYS_socket as u32,
        },
        libc::sock_filter {
            code: 0x06,
            jt: 0,
            jf: 0,
            k: 0x0005_0000 | libc::EPERM as u32,
        },
        libc::sock_filter {
            code: 0x15,
            jt: 0,
            jf: 1,
            k: libc::SYS_connect as u32,
        },
        libc::sock_filter {
            code: 0x06,
            jt: 0,
            jf: 0,
            k: 0x0005_0000 | libc::EPERM as u32,
        },
        libc::sock_filter {
            code: 0x06,
            jt: 0,
            jf: 0,
            k: 0x7fff_0000,
        },
    ];
    let program = libc::sock_fprog {
        len: filter.len() as u16,
        filter: filter.as_mut_ptr(),
    };
    // SAFETY: the filter array remains live for both prctl calls; no-new-privs
    // is set first and the filter only changes this test thread.
    unsafe {
        assert_eq!(libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0), 0);
        assert_eq!(
            libc::prctl(libc::PR_SET_SECCOMP, libc::SECCOMP_MODE_FILTER, &program),
            0
        );
    }
}

#[cfg(target_os = "linux")]
#[test]
fn real_detect_and_embed_run_with_socket_syscalls_denied() {
    let _ort = lock_ort();
    deny_network_syscalls_for_current_thread();
    let socket_error = std::net::TcpStream::connect(("127.0.0.1", 9)).unwrap_err();
    assert_eq!(socket_error.kind(), std::io::ErrorKind::PermissionDenied);
    let image = DynamicImage::new_rgb8(112, 112);
    let mut engine = FaceEngine::new().unwrap();
    engine.detect(&image).unwrap();
    let face = DetectedFace {
        bounds: [0.0, 0.0, 1.0, 1.0],
        landmarks: [
            [0.3, 0.35],
            [0.7, 0.35],
            [0.5, 0.55],
            [0.35, 0.75],
            [0.65, 0.75],
        ],
        confidence: 1.0,
    };
    assert_eq!(engine.embedding(&image, &face).unwrap().len(), 128);
}

fn synthetic_fixture(bytes: &[u8]) -> DynamicImage {
    image::load_from_memory(bytes).expect("synthetic face fixture must decode")
}

#[test]
fn packaged_models_detect_synthetic_frontal_and_group_faces() {
    let _ort = lock_ort();
    let mut engine = FaceEngine::new().unwrap();
    let frontal = synthetic_fixture(include_bytes!(
        "../../assets/face/fixtures/synthetic-person-a-frontal.png"
    ));
    let detected = engine.detect(&frontal).unwrap();
    assert_eq!(detected.len(), 1, "frontal fixture should contain one face");
    assert!(detected[0].confidence >= 0.5);
    assert!(
        detected[0]
            .landmarks
            .iter()
            .flatten()
            .all(|value| value.is_finite() && (0.0..=1.0).contains(value))
    );

    let group = synthetic_fixture(include_bytes!(
        "../../assets/face/fixtures/synthetic-three-person-group.png"
    ));
    let group_faces = engine.detect(&group).unwrap();
    assert_eq!(
        group_faces.len(),
        3,
        "group fixture should exercise three-face detection and NMS"
    );
}

#[test]
fn packaged_sface_scores_same_synthetic_identity_above_different_identity() {
    let _ort = lock_ort();
    let mut engine = FaceEngine::new().unwrap();
    let images = [
        synthetic_fixture(include_bytes!(
            "../../assets/face/fixtures/synthetic-person-a-frontal.png"
        )),
        synthetic_fixture(include_bytes!(
            "../../assets/face/fixtures/synthetic-person-a-turned.png"
        )),
        synthetic_fixture(include_bytes!(
            "../../assets/face/fixtures/synthetic-person-b-frontal.png"
        )),
    ];
    let mut embeddings = Vec::new();
    for image in &images {
        let faces = engine.detect(image).unwrap();
        assert_eq!(faces.len(), 1);
        embeddings.push(engine.embedding(image, &faces[0]).unwrap());
    }
    let same = cosine_similarity(&embeddings[0], &embeddings[1]).unwrap();
    let different = cosine_similarity(&embeddings[0], &embeddings[2]).unwrap();
    assert!(
        same > different + 0.10,
        "same identity score {same} must safely exceed different identity score {different}"
    );
}

fn jpeg_with_orientation(image: &DynamicImage, orientation: u16) -> Vec<u8> {
    let mut encoded = std::io::Cursor::new(Vec::new());
    image
        .write_to(&mut encoded, image::ImageFormat::Jpeg)
        .unwrap();
    let encoded = encoded.into_inner();
    assert_eq!(&encoded[..2], &[0xff, 0xd8]);
    let mut tiff = Vec::new();
    tiff.extend_from_slice(b"II");
    tiff.extend_from_slice(&42_u16.to_le_bytes());
    tiff.extend_from_slice(&8_u32.to_le_bytes());
    tiff.extend_from_slice(&1_u16.to_le_bytes());
    tiff.extend_from_slice(&0x0112_u16.to_le_bytes());
    tiff.extend_from_slice(&3_u16.to_le_bytes());
    tiff.extend_from_slice(&1_u32.to_le_bytes());
    tiff.extend_from_slice(&orientation.to_le_bytes());
    tiff.extend_from_slice(&0_u16.to_le_bytes());
    tiff.extend_from_slice(&0_u32.to_le_bytes());
    let length = u16::try_from(2 + 6 + tiff.len()).unwrap();
    let mut result = encoded[..2].to_vec();
    result.extend_from_slice(&[0xff, 0xe1]);
    result.extend_from_slice(&length.to_be_bytes());
    result.extend_from_slice(b"Exif\0\0");
    result.extend_from_slice(&tiff);
    result.extend_from_slice(&encoded[2..]);
    result
}

fn write_oriented_synthetic_dng(path: &Path, image: &DynamicImage, orientation: u16) {
    use rawler::{
        dng::{DNG_VERSION_V1_4, DngCompression, writer::DngWriter},
        tags::TiffCommonTag,
    };
    let image = image.to_rgb8();
    let mut dng = DngWriter::new(
        std::io::BufWriter::new(std::fs::File::create(path).unwrap()),
        DNG_VERSION_V1_4,
    )
    .unwrap();
    dng.root_ifd_mut()
        .add_tag(TiffCommonTag::Orientation, orientation);
    let mut raw = dng.subframe(0);
    raw.rgb_image_u8(
        image.as_raw(),
        image.width() as usize,
        image.height() as usize,
        DngCompression::Uncompressed,
        1,
    )
    .unwrap();
    raw.finalize().unwrap();
    dng.close().unwrap();
}

#[test]
fn oriented_jpeg_and_raw_face_crops_preserve_the_synthetic_face_region() {
    let _ort = lock_ort();
    let source = synthetic_fixture(include_bytes!(
        "../../assets/face/fixtures/synthetic-person-a-frontal.png"
    ))
    .resize_exact(512, 384, image::imageops::FilterType::Triangle);
    let mut engine = FaceEngine::new().unwrap();
    let faces = engine.detect(&source).unwrap();
    assert_eq!(faces.len(), 1);
    let canonical_bounds = FaceTransform::orientation(6)
        .inverse()
        .unwrap()
        .bounds(faces[0].bounds);
    let stored = source.rotate270();
    let directory = tempfile::tempdir().unwrap();
    let jpeg = directory.path().join("synthetic-oriented.jpg");
    std::fs::write(&jpeg, jpeg_with_orientation(&stored, 6)).unwrap();
    let dng = directory.path().join("synthetic-oriented.dng");
    write_oriented_synthetic_dng(&dng, &stored, 6);
    let pipeline =
        RenderPipeline::with_source_cache_capacity(directory.path().join("cache"), 1, 1, 1);
    let variant = RenderVariant::FaceCrop {
        face_id: 1,
        face_revision: 1,
        left_millionths: (canonical_bounds[0] * 1_000_000.0).round() as u32,
        top_millionths: (canonical_bounds[1] * 1_000_000.0).round() as u32,
        right_millionths: (canonical_bounds[2] * 1_000_000.0).round() as u32,
        bottom_millionths: (canonical_bounds[3] * 1_000_000.0).round() as u32,
        width: 512,
        padding_percent: 50,
    };
    for path in [jpeg, dng] {
        let crop = pipeline
            .render(
                path.clone(),
                variant,
                false,
                false,
                &Arc::new(AtomicBool::new(false)),
            )
            .unwrap();
        let detected = engine.detect(&crop.image).unwrap();
        assert_eq!(
            detected.len(),
            1,
            "oriented {} face crop must contain the requested face",
            path.extension().unwrap().to_string_lossy()
        );
    }
}
use test_r::test;

#[test]
fn geometry_and_similarity_are_stable() {
    assert_eq!(
        intersection_over_union([0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 3.0, 3.0]),
        1.0 / 7.0
    );
    assert_eq!(cosine_similarity(&[1.0, 0.0], &[0.5, 0.5]), Some(0.5));
    assert_eq!(cosine_similarity(&[1.0], &[1.0, 0.0]), None);
    let source = [[0.0, 0.0], [2.0, 0.0], [1.0, 1.0], [0.0, 2.0], [2.0, 2.0]];
    let target = source.map(|point| [3.0 - 2.0 * point[1], 4.0 + 2.0 * point[0]]);
    let transform = similarity_transform(&source, &target).unwrap();
    assert!((transform[0]).abs() < 1e-5);
    assert!((transform[1] - 2.0).abs() < 1e-5);
    assert!((transform[2] - 3.0).abs() < 1e-5);
    assert!((transform[3] - 4.0).abs() < 1e-5);
}

#[test]
fn reconciliation_rejects_ambiguous_overlap_in_both_directions() {
    let detected = |bounds| DetectedFace {
        bounds,
        landmarks: realistic_landmarks(bounds),
        confidence: 1.0,
    };
    let existing_face = |id, bounds| ReconciliationFace {
        id,
        bounds,
        provenance: FaceProvenance::Lightroom,
        landmarks: StoredLandmarks::Missing,
    };
    let existing = vec![
        existing_face(10, [0.10, 0.10, 0.40, 0.40]),
        existing_face(11, [0.12, 0.10, 0.42, 0.40]),
    ];
    let detection = detected([0.11, 0.10, 0.41, 0.40]);
    assert_eq!(
        conservative_reconciliation(&[&detection], &existing),
        vec![ReconciliationMatch::New]
    );

    let existing = vec![existing_face(10, [0.10, 0.10, 0.40, 0.40])];
    let first = detected([0.10, 0.10, 0.40, 0.40]);
    let second = detected([0.11, 0.10, 0.41, 0.40]);
    assert_eq!(
        conservative_reconciliation(&[&first, &second], &existing),
        vec![ReconciliationMatch::New, ReconciliationMatch::New]
    );
    assert_eq!(
        conservative_reconciliation(&[&first], &existing),
        vec![ReconciliationMatch::Reuse(10)]
    );

    let existing = vec![
        existing_face(20, [0.05, 0.05, 0.25, 0.25]),
        existing_face(21, [0.70, 0.70, 0.90, 0.90]),
    ];
    let last = detected([0.70, 0.70, 0.90, 0.90]);
    let first = detected([0.05, 0.05, 0.25, 0.25]);
    assert_eq!(
        conservative_reconciliation(&[&last, &first], &existing),
        vec![
            ReconciliationMatch::Reuse(21),
            ReconciliationMatch::Reuse(20)
        ],
        "detector result order must not determine durable face identity"
    );
}

#[test]
fn reconciliation_uses_landmarks_when_the_existing_face_has_them() {
    let bounds = [0.10, 0.10, 0.50, 0.50];
    let landmarks = realistic_landmarks(bounds);
    let compatible = DetectedFace {
        bounds,
        landmarks,
        confidence: 1.0,
    };
    let incompatible = DetectedFace {
        bounds,
        landmarks: [
            landmarks[1],
            landmarks[0],
            landmarks[2],
            landmarks[4],
            landmarks[3],
        ],
        confidence: 1.0,
    };
    let existing = [ReconciliationFace {
        id: 42,
        bounds,
        provenance: FaceProvenance::Rekesz,
        landmarks: StoredLandmarks::Valid(landmarks),
    }];
    assert_eq!(
        conservative_reconciliation(&[&compatible], &existing),
        vec![ReconciliationMatch::Reuse(42)]
    );
    assert_eq!(
        conservative_reconciliation(&[&incompatible], &existing),
        vec![ReconciliationMatch::New],
        "a geometrically overlapping face with incompatible landmarks must not inherit identity"
    );
}

fn realistic_landmarks(bounds: [f32; 4]) -> [[f32; 2]; 5] {
    let width = bounds[2] - bounds[0];
    let height = bounds[3] - bounds[1];
    [
        [bounds[0] + width * 0.30, bounds[1] + height * 0.35],
        [bounds[0] + width * 0.70, bounds[1] + height * 0.35],
        [bounds[0] + width * 0.50, bounds[1] + height * 0.55],
        [bounds[0] + width * 0.35, bounds[1] + height * 0.75],
        [bounds[0] + width * 0.65, bounds[1] + height * 0.75],
    ]
}

#[test]
fn geometry_fallback_has_conservative_boundaries_and_canonical_orientation() {
    let existing = [ReconciliationFace {
        id: 7,
        bounds: [0.10, 0.10, 0.50, 0.50],
        provenance: FaceProvenance::Lightroom,
        landmarks: StoredLandmarks::Missing,
    }];
    let detected = |bounds| DetectedFace {
        bounds,
        landmarks: realistic_landmarks(bounds),
        confidence: 1.0,
    };
    let shifted = detected([0.14, 0.10, 0.54, 0.50]);
    let adjacent = detected([0.36, 0.10, 0.76, 0.50]);
    let below_overlap = detected([0.24, 0.10, 0.64, 0.50]);
    assert_eq!(
        conservative_reconciliation(&[&shifted], &existing),
        [ReconciliationMatch::Reuse(7)]
    );
    assert_eq!(
        conservative_reconciliation(&[&adjacent], &existing),
        [ReconciliationMatch::New]
    );
    assert_eq!(
        conservative_reconciliation(&[&below_overlap], &existing),
        [ReconciliationMatch::New]
    );

    // These are the same source-space box after a 90-degree displayed-space
    // round trip; reconciliation intentionally consumes canonical bounds.
    let canonical_after_orientation = detected([0.10, 0.10, 0.50, 0.50]);
    assert_eq!(
        conservative_reconciliation(&[&canonical_after_orientation], &existing),
        [ReconciliationMatch::Reuse(7)]
    );
}

#[test]
fn reconciliation_geometry_thresholds_are_inclusive_and_scale_stable() {
    let unit = [0.0, 0.0, 1.0, 1.0];
    let shift_at_iou = (1.0 - RECONCILIATION_MIN_IOU) / (1.0 + RECONCILIATION_MIN_IOU);
    let shifted = |shift| [shift, 0.0, 1.0 + shift, 1.0];
    let epsilon = 1.0e-5;
    assert!(
        intersection_over_union(unit, shifted(shift_at_iou - epsilon)) > RECONCILIATION_MIN_IOU
    );
    assert!(
        (intersection_over_union(unit, shifted(shift_at_iou)) - RECONCILIATION_MIN_IOU).abs()
            < epsilon
    );
    assert!(
        intersection_over_union(unit, shifted(shift_at_iou + epsilon)) < RECONCILIATION_MIN_IOU
    );

    let diagonal = 2.0_f32.hypot(1.0);
    let center_at_limit = RECONCILIATION_MAX_CENTER_DISTANCE * diagonal;
    let wide = [0.0, 0.0, 2.0, 1.0];
    let wide_shifted = |shift| [shift, 0.0, 2.0 + shift, 1.0];
    assert!(
        normalized_center_distance(wide_shifted(center_at_limit - epsilon), wide)
            < RECONCILIATION_MAX_CENTER_DISTANCE
    );
    assert!(
        (normalized_center_distance(wide_shifted(center_at_limit), wide)
            - RECONCILIATION_MAX_CENTER_DISTANCE)
            .abs()
            < epsilon
    );
    assert!(
        normalized_center_distance(wide_shifted(center_at_limit + epsilon), wide)
            > RECONCILIATION_MAX_CENTER_DISTANCE
    );

    for scale in [0.1, 0.5, 10.0] {
        let bounds = [0.0, 0.0, 2.0 * scale, scale];
        let shifted = [
            center_at_limit * scale,
            0.0,
            (2.0 + center_at_limit) * scale,
            scale,
        ];
        assert!(
            (normalized_center_distance(shifted, bounds) - RECONCILIATION_MAX_CENTER_DISTANCE)
                .abs()
                < epsilon
        );
    }
}

#[test]
fn malformed_landmarks_never_fall_back_to_geometry() {
    let bounds = [0.10, 0.10, 0.50, 0.50];
    let detection = DetectedFace {
        bounds,
        landmarks: realistic_landmarks(bounds),
        confidence: 1.0,
    };
    let malformed = [
        vec![0_u8; 39],
        (0..10)
            .flat_map(|index| if index == 0 { f32::NAN } else { 0.2 }.to_le_bytes())
            .collect(),
        (0..10)
            .flat_map(|index| if index == 0 { f32::INFINITY } else { 0.2 }.to_le_bytes())
            .collect(),
    ];
    for bytes in malformed {
        assert!(decode_stored_landmarks(&bytes).is_none());
        let existing = [ReconciliationFace {
            id: 8,
            bounds,
            provenance: FaceProvenance::Rekesz,
            landmarks: StoredLandmarks::Invalid,
        }];
        assert_eq!(
            conservative_reconciliation(&[&detection], &existing),
            [ReconciliationMatch::New]
        );
    }
}

#[test]
fn cross_provenance_conflict_does_not_create_a_third_identity() {
    let bounds = [0.10, 0.10, 0.50, 0.50];
    let landmarks = realistic_landmarks(bounds);
    let detection = DetectedFace {
        bounds,
        landmarks,
        confidence: 1.0,
    };
    let existing = [
        ReconciliationFace {
            id: 1,
            bounds,
            provenance: FaceProvenance::Lightroom,
            landmarks: StoredLandmarks::Missing,
        },
        ReconciliationFace {
            id: 2,
            bounds,
            provenance: FaceProvenance::Rekesz,
            landmarks: StoredLandmarks::Valid(landmarks),
        },
    ];
    assert_eq!(
        conservative_reconciliation(&[&detection], &existing),
        [ReconciliationMatch::Conflict(vec![1, 2])]
    );
}

#[test]
fn persisted_malformed_landmarks_cannot_capture_new_detections() {
    let root = tempfile::tempdir().unwrap();
    let source = root.path().join("face.png");
    RgbImage::new(64, 64).save(&source).unwrap();
    catalog::initialize_test_catalog(root.path()).unwrap();
    sync_analysis_inventory(root.path(), std::slice::from_ref(&source)).unwrap();
    let connection = Connection::open(catalog::catalog_path(root.path())).unwrap();
    let identity: (i64, i64, Vec<u8>) = connection
            .query_row(
                "SELECT source_size, source_modified_ns, source_fingerprint FROM analysis_asset_inventory WHERE asset_id=1",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
    let bounds = [
        [0.05, 0.05, 0.25, 0.25],
        [0.40, 0.05, 0.60, 0.25],
        [0.70, 0.05, 0.90, 0.25],
    ];
    let malformed = [
        vec![0_u8; 39],
        (0..10)
            .flat_map(|index| (if index == 0 { f32::NAN } else { 0.2_f32 }).to_le_bytes())
            .collect(),
        (0..10)
            .flat_map(|index| (if index == 0 { f32::INFINITY } else { 0.2_f32 }).to_le_bytes())
            .collect(),
    ];
    for (index, (bounds, landmarks)) in bounds.into_iter().zip(malformed).enumerate() {
        let face_id = 100 + index as i64;
        connection
            .execute(
                "INSERT INTO faces(id, asset_id, provenance, source_key, left, top, right,
                                       bottom, origin, updated_at)
                     VALUES(?1, 1, 'rekesz', ?2, ?3, ?4, ?5, ?6,
                            'rekesz-detected', datetime('now'))",
                params![
                    face_id,
                    format!("malformed-{index}"),
                    bounds[0],
                    bounds[1],
                    bounds[2],
                    bounds[3]
                ],
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO face_embeddings(face_id, model_id, dimensions, vector, landmarks,
                                                  source_size, source_modified_ns, created_at)
                     VALUES(?1, 'sface-2021dec', 2, zeroblob(8), ?2, ?3, ?4, datetime('now'))",
                params![face_id, landmarks, identity.0, identity.1],
            )
            .unwrap();
    }
    connection
        .execute(
            "UPDATE face_analysis_state SET status='running', worker_id='malformed-worker',
                 lease_expires_at=datetime('now', '+30 minutes') WHERE asset_id=1",
            [],
        )
        .unwrap();
    let results = bounds.map(|bounds| {
        (
            DetectedFace {
                bounds,
                landmarks: realistic_landmarks(bounds),
                confidence: 0.9,
            },
            vec![1.0, 0.0],
        )
    });
    store_analysis(
        &connection,
        1,
        AnalyzedSourceIdentity {
            size: identity.0,
            modified_ns: identity.1,
            fingerprint: &identity.2,
            path: None,
        },
        "malformed-worker",
        &results,
        &AtomicBool::new(false),
    )
    .unwrap();
    assert_eq!(
        connection
            .query_row(
                "SELECT count(*) FROM faces WHERE asset_id=1 AND retired=0",
                [],
                |row| row.get::<_, i64>(0),
            )
            .unwrap(),
        3
    );
    assert_eq!(
        connection
            .query_row(
                "SELECT count(*) FROM faces WHERE id BETWEEN 100 AND 102 AND retired=1",
                [],
                |row| row.get::<_, i64>(0),
            )
            .unwrap(),
        3
    );
}

#[test]
fn analysis_persists_embedding_source_identity_and_opaque_face_key() {
    let root = tempfile::tempdir().unwrap();
    let source = root.path().join("face.png");
    RgbImage::new(32, 32).save(&source).unwrap();
    catalog::initialize_test_catalog(root.path()).unwrap();
    sync_analysis_inventory(root.path(), std::slice::from_ref(&source)).unwrap();
    let connection = Connection::open(catalog::catalog_path(root.path())).unwrap();
    let identity: (i64, i64, Vec<u8>) = connection
            .query_row(
                "SELECT source_size, source_modified_ns, source_fingerprint FROM analysis_asset_inventory
                 WHERE asset_id=1",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
    connection
        .execute(
            "UPDATE face_analysis_state SET status='running', worker_id='test-worker',
                 lease_expires_at=datetime('now', '+30 minutes') WHERE asset_id=1",
            [],
        )
        .unwrap();
    let face = DetectedFace {
        bounds: [0.1, 0.1, 0.4, 0.4],
        landmarks: [[0.2, 0.2]; 5],
        confidence: 0.9,
    };
    store_analysis(
        &connection,
        1,
        AnalyzedSourceIdentity {
            size: identity.0,
            modified_ns: identity.1,
            fingerprint: &identity.2,
            path: None,
        },
        "test-worker",
        &[(face, vec![1.0, 0.0])],
        &AtomicBool::new(false),
    )
    .unwrap();

    let persisted: (i64, i64, Vec<u8>, String) = connection
        .query_row(
            "SELECT embedding.source_size, embedding.source_modified_ns,
                        embedding.source_fingerprint, face.source_key
                 FROM face_embeddings embedding
                 JOIN faces face ON face.id=embedding.face_id",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        )
        .unwrap();
    assert_eq!((persisted.0, persisted.1, persisted.2), identity);
    assert!(persisted.3.starts_with("detected:"));
    assert_eq!(persisted.3.len(), "detected:".len() + 32);
}

#[test]
fn initial_lightroom_geometry_reconciliation_survives_reimport_and_landmark_rerun() {
    let root = tempfile::tempdir().unwrap();
    let source = root.path().join("face.png");
    RgbImage::new(32, 32).save(&source).unwrap();
    catalog::initialize_test_catalog(root.path()).unwrap();
    sync_analysis_inventory(root.path(), std::slice::from_ref(&source)).unwrap();
    let connection = Connection::open(catalog::catalog_path(root.path())).unwrap();
    let identity: (i64, i64, Vec<u8>) = connection
            .query_row(
                "SELECT source_size, source_modified_ns, source_fingerprint FROM analysis_asset_inventory WHERE asset_id=1",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
    connection
        .execute_batch(
            "INSERT INTO people(id, display_name, created_at, updated_at)
                    VALUES(9, 'Imported', datetime('now'), datetime('now'));
                 INSERT INTO people(id, display_name, created_at, updated_at)
                    VALUES(10, 'Locally rejected', datetime('now'), datetime('now'));
                 INSERT INTO faces(id, asset_id, provenance, source_key, left, top, right, bottom,
                                   origin, ignored, retired, updated_at)
                    VALUES(50, 1, 'lightroom', 'lr-face-50', .1, .1, .5, .5,
                           'lightroom-detected', 0, 0, datetime('now'));
                 INSERT INTO face_assignments(face_id, person_id, kind, provenance, updated_at)
                    VALUES(50, 9, 'confirmed', 'lightroom', datetime('now'));
                 INSERT INTO face_assignments(face_id, person_id, kind, provenance, updated_at)
                    VALUES(50, 10, 'rejected', 'rekesz', datetime('now'));",
        )
        .unwrap();
    let landmarks = realistic_landmarks([0.1, 0.1, 0.5, 0.5]);
    catalog::recognition::ensure_production_calibration(root.path()).unwrap();
    connection
        .execute(
            "UPDATE face_analysis_state SET status='running', worker_id='test-worker',
                 lease_expires_at=datetime('now', '+30 minutes') WHERE asset_id=1",
            [],
        )
        .unwrap();
    store_analysis(
        &connection,
        1,
        AnalyzedSourceIdentity {
            size: identity.0,
            modified_ns: identity.1,
            fingerprint: &identity.2,
            path: None,
        },
        "test-worker",
        &[(
            DetectedFace {
                bounds: [0.1, 0.1, 0.5, 0.5],
                landmarks,
                confidence: 0.95,
            },
            vec![1.0, 0.0],
        )],
        &AtomicBool::new(false),
    )
    .unwrap();

    assert_eq!(
        connection
            .query_row(
                "SELECT count(*) FROM recognition_refresh_queue
                     WHERE entity_kind='person' AND entity_id=9
                       AND model_id='sface-2021dec' AND status='pending'",
                [],
                |row| row.get::<_, i64>(0),
            )
            .unwrap(),
        1,
        "the first embedding written onto an imported confirmation must queue its owner"
    );
    catalog::recognition::process_pending_refresh_batch(
        root.path(),
        "first-imported-prototype",
        &MaintenanceAdmission::new(),
        &AtomicBool::new(false),
    )
    .unwrap();
    assert_eq!(
        connection
            .query_row(
                "SELECT exemplar_count FROM person_prototypes
                     WHERE person_id=9 AND model_id='sface-2021dec'",
                [],
                |row| row.get::<_, i64>(0),
            )
            .unwrap(),
        1
    );

    assert_eq!(
        connection
            .query_row("SELECT count(*) FROM faces WHERE asset_id=1", [], |row| {
                row.get::<_, i64>(0)
            })
            .unwrap(),
        1,
        "a compatible detection must reuse, not duplicate, the imported face"
    );
    assert_eq!(
        connection
            .query_row(
                "SELECT face_id FROM face_embeddings WHERE model_id='sface-2021dec'",
                [],
                |row| row.get::<_, i64>(0),
            )
            .unwrap(),
        50
    );
    // Lightroom reimport updates its row without owning or deleting the
    // Rekesz embedding. The next detector pass now uses landmark matching.
    connection
        .execute_batch(
            "UPDATE faces SET left=.1, top=.1, right=.5, bottom=.5,
                                  origin='lightroom-detected', ignored=0,
                                  updated_at=datetime('now') WHERE id=50;
                 UPDATE face_analysis_state SET status='running', worker_id='test-worker-2',
                    lease_expires_at=datetime('now', '+30 minutes') WHERE asset_id=1;",
        )
        .unwrap();
    store_analysis(
        &connection,
        1,
        AnalyzedSourceIdentity {
            size: identity.0,
            modified_ns: identity.1,
            fingerprint: &identity.2,
            path: None,
        },
        "test-worker-2",
        &[(
            DetectedFace {
                bounds: [0.1, 0.1, 0.5, 0.5],
                landmarks,
                confidence: 0.96,
            },
            vec![1.0, 0.0],
        )],
        &AtomicBool::new(false),
    )
    .unwrap();
    assert_eq!(
        connection
            .query_row("SELECT count(*) FROM faces WHERE asset_id=1", [], |row| {
                row.get::<_, i64>(0)
            })
            .unwrap(),
        1
    );
    let landmark_bytes = landmarks
        .iter()
        .flatten()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    connection
        .execute(
            "INSERT INTO faces(id, asset_id, provenance, source_key, left, top, right, bottom,
                                   origin, updated_at)
                 VALUES(51, 1, 'rekesz', 'preexisting-duplicate', .1, .1, .5, .5,
                        'rekesz-detected', datetime('now'))",
            [],
        )
        .unwrap();
    connection
        .execute(
            "INSERT INTO face_embeddings(face_id, model_id, dimensions, vector, landmarks,
                                              source_size, source_modified_ns, created_at)
                 VALUES(51, 'sface-2021dec', 2, zeroblob(8), ?1, ?2, ?3, datetime('now'))",
            params![landmark_bytes, identity.0, identity.1],
        )
        .unwrap();
    connection
        .execute(
            "UPDATE face_analysis_state SET status='running', worker_id='test-worker-3',
                 lease_expires_at=datetime('now', '+30 minutes') WHERE asset_id=1",
            [],
        )
        .unwrap();
    store_analysis(
        &connection,
        1,
        AnalyzedSourceIdentity {
            size: identity.0,
            modified_ns: identity.1,
            fingerprint: &identity.2,
            path: None,
        },
        "test-worker-3",
        &[(
            DetectedFace {
                bounds: [0.1, 0.1, 0.5, 0.5],
                landmarks,
                confidence: 0.97,
            },
            vec![1.0, 0.0],
        )],
        &AtomicBool::new(false),
    )
    .unwrap();
    assert_eq!(
        connection
            .query_row(
                "SELECT count(*) FROM faces WHERE asset_id=1 AND retired=0",
                [],
                |row| row.get::<_, i64>(0),
            )
            .unwrap(),
        2,
        "a cross-provenance conflict must not create a third face or retire either candidate"
    );
    assert_eq!(
        connection
            .query_row(
                "SELECT conflict_count FROM face_reconciliation_conflicts WHERE asset_id=1",
                [],
                |row| row.get::<_, i64>(0),
            )
            .unwrap(),
        1
    );
    assert_eq!(
        catalog::people::diagnostics(root.path())
            .unwrap()
            .reconciliation_conflicts,
        1,
        "operators must see a durable non-biometric conflict count"
    );
    let decisions = connection
            .prepare(
                "SELECT person_id, kind, provenance FROM face_assignments WHERE face_id=50 ORDER BY person_id",
            )
            .unwrap()
            .query_map([], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                ))
            })
            .unwrap()
            .collect::<rusqlite::Result<Vec<_>>>()
            .unwrap();
    assert_eq!(
        decisions,
        vec![
            (9, "confirmed".to_owned(), "lightroom".to_owned()),
            (10, "rejected".to_owned(), "rekesz".to_owned()),
        ]
    );
}

#[test]
fn face_engine_has_no_network_or_map_dependency() {
    let source = include_str!("../face_engine.rs");
    assert!(
        !source.contains(&["image", "::open"].concat()),
        "face analysis must use Rekesz's orientation-aware multi-format decoder"
    );
    let forbidden = [
        ["req", "west"].concat(),
        ["ur", "eq"].concat(),
        ["hy", "per::"].concat(),
        ["ma", "p::"].concat(),
        ["Map", "Renderer"].concat(),
    ];
    for forbidden in forbidden {
        assert!(
            !source.contains(&forbidden),
            "face engine must remain offline and independent of map clients: {forbidden}"
        );
    }
}

#[test]
fn bundled_models_load_and_black_frame_has_no_faces() {
    let _ort = lock_ort();
    let mut engine = FaceEngine::new().unwrap();
    assert!(
        engine
            .detect(&DynamicImage::ImageRgb8(RgbImage::new(32, 32)))
            .unwrap()
            .is_empty()
    );
}

#[test]
fn analysis_write_path_matches_the_owned_catalog_schema() {
    let _ort = lock_ort();
    let root = tempfile::tempdir().unwrap();
    let source = root.path().join("black.png");
    RgbImage::new(32, 32).save(&source).unwrap();
    catalog::initialize_test_catalog(root.path()).unwrap();
    sync_analysis_inventory(root.path(), std::slice::from_ref(&source)).unwrap();
    let catalog_path = catalog::catalog_path(root.path());

    assert_eq!(index_batch(root.path(), 1).unwrap(), 1);
    let connection = Connection::open(&catalog_path).unwrap();
    let state: (String, i64, Option<String>) = connection
        .query_row(
            "SELECT status, attempts, error FROM face_analysis_state WHERE asset_id=1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .unwrap();
    assert_eq!(state, ("complete".to_owned(), 0, None));
    catalog::recognition::ensure_production_calibration(root.path()).unwrap();

    connection.execute_batch(
            "INSERT INTO people(id, display_name, created_at, updated_at) VALUES
                 (9, 'Local owner', datetime('now'), datetime('now')),
                 (10, 'Lightroom owner', datetime('now'), datetime('now'));
             INSERT INTO faces(id, asset_id, provenance, source_key, left, top, right, bottom,
                 origin, updated_at) VALUES
                 (90, 1, 'rekesz', 'stale-derived', .1, .1, .2, .2, 'rekesz-detected', datetime('now')),
                 (91, 1, 'lightroom', 'imported', .2, .2, .3, .3, 'lightroom', datetime('now')),
                 (92, 1, 'rekesz', 'user-confirmed', .4, .4, .6, .6, 'rekesz-detected', datetime('now'));
             INSERT INTO face_assignments(face_id, person_id, kind, provenance, confidence, updated_at)
                 VALUES(90, 10, 'suggested', 'rekesz', .8, datetime('now')),
                       (90, 10, 'rejected', 'rekesz', NULL, datetime('now')),
                       (91, 10, 'confirmed', 'lightroom', NULL, datetime('now')),
                       (92, 9, 'confirmed', 'rekesz', NULL, datetime('now'));
             INSERT INTO face_embeddings(face_id, model_id, dimensions, vector, created_at)
                 VALUES(90, 'sface-2021dec', 1, zeroblob(4), datetime('now')),
                       (91, 'sface-2021dec', 1, zeroblob(4), datetime('now')),
                       (92, 'sface-2021dec', 1, zeroblob(4), datetime('now'));
             INSERT INTO person_prototypes(person_id, model_id, dimensions, vector,
                 exemplar_count, generation, updated_at)
                 VALUES(9, 'sface-2021dec', 1, zeroblob(4), 1, 4, datetime('now')),
                       (10, 'sface-2021dec', 1, zeroblob(4), 1, 5, datetime('now'));
             INSERT INTO recognition_suggestions(face_id, person_id, recognizer_model_id,
                 prototype_generation, prototype_algorithm_version, ann_index_version,
                 scored_index_generation, policy_version, score, nearest_exemplar_score,
                 second_best_margin, created_at)
                 VALUES(90, 10, 'sface-2021dec', 5, 1, 1, 5, 1, .9, .9, .1, datetime('now'));
             DELETE FROM recognition_refresh_queue;
             DELETE FROM recognition_rescore_state;
             UPDATE face_analysis_state SET status='pending' WHERE asset_id=1;",
        ).unwrap();
    connection
        .execute_batch(
            "CREATE TRIGGER abort_stale_embedding_delete
                 BEFORE DELETE ON face_embeddings WHEN old.face_id=92
                 BEGIN SELECT raise(ABORT, 'injected stale rollback'); END;",
        )
        .unwrap();
    drop(connection);
    assert!(
        format!("{:#}", index_batch(root.path(), 1).unwrap_err())
            .contains("injected stale rollback")
    );
    let connection = Connection::open(&catalog_path).unwrap();
    assert_eq!(
            connection
                .query_row(
                    "SELECT
                       (SELECT retired FROM faces WHERE id=90),
                       (SELECT count(*) FROM face_embeddings WHERE face_id IN (90, 91, 92)),
                       (SELECT count(*) FROM face_assignments WHERE face_id=90 AND kind='suggested'),
                       (SELECT count(*) FROM recognition_refresh_queue),
                       (SELECT count(*) FROM person_prototypes WHERE person_id IN (9, 10))",
                    [],
                    |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?, row.get::<_, i64>(2)?, row.get::<_, i64>(3)?, row.get::<_, i64>(4)?)),
                )
                .unwrap(),
            (0, 3, 1, 0, 2),
            "a late reconciliation failure must roll back retirement, artifacts, assignments, and prototype invalidation together"
        );
    connection
            .execute_batch(
                "DROP TRIGGER abort_stale_embedding_delete;
                 UPDATE face_analysis_state SET status='pending', retry_after=NULL WHERE asset_id=1;",
            )
            .unwrap();
    drop(connection);
    assert_eq!(index_batch(root.path(), 1).unwrap(), 1);
    let connection = Connection::open(&catalog_path).unwrap();
    let stale: (i64, i64, i64) = connection
        .query_row(
            "SELECT retired,
                (SELECT count(*) FROM face_embeddings WHERE face_id=90),
                (SELECT count(*) FROM face_assignments WHERE face_id=90 AND kind='suggested')
             FROM faces WHERE id=90",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .unwrap();
    assert_eq!(stale, (1, 0, 0));
    let preserved: (i64, i64, i64) = connection
        .query_row(
            "SELECT (SELECT count(*) FROM face_assignments WHERE face_id=90 AND kind='rejected'),
                    (SELECT count(*) FROM faces WHERE id=91 AND retired=0),
                    (SELECT count(*) FROM faces WHERE id=92 AND retired=0)",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .unwrap();
    assert_eq!(preserved, (1, 1, 1));
    assert_eq!(
        connection
            .query_row(
                "SELECT
                       (SELECT count(*) FROM face_embeddings WHERE face_id=92),
                       (SELECT count(*) FROM person_prototypes WHERE person_id=9
                          AND model_id='sface-2021dec'),
                       (SELECT count(*) FROM recognition_refresh_queue
                          WHERE entity_kind='person' AND entity_id=9
                            AND model_id='sface-2021dec' AND status='pending'),
                       (SELECT count(*) FROM face_embeddings WHERE face_id=91),
                       (SELECT count(*) FROM person_prototypes WHERE person_id=10
                          AND model_id='sface-2021dec'),
                       (SELECT count(*) FROM recognition_refresh_queue
                          WHERE entity_kind='person' AND entity_id=10
                            AND model_id='sface-2021dec' AND status='pending')",
                [],
                |row| Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, i64>(3)?,
                    row.get::<_, i64>(4)?,
                    row.get::<_, i64>(5)?
                )),
            )
            .unwrap(),
        (0, 1, 1, 0, 1, 1),
        "Lightroom and local confirmations must independently queue their own model-specific prototype rebuilds"
    );
    catalog::recognition::process_pending_refresh_batch(
        root.path(),
        "stale-lightroom-prototype",
        &MaintenanceAdmission::new(),
        &AtomicBool::new(false),
    )
    .unwrap();
    assert_eq!(
        connection
            .query_row(
                "SELECT
                       (SELECT count(*) FROM person_prototypes WHERE person_id=10
                          AND model_id='sface-2021dec'),
                       (SELECT count(*) FROM recognition_suggestions WHERE person_id=10
                          AND recognizer_model_id='sface-2021dec'),
                       (SELECT count(*) FROM person_prototypes WHERE person_id=9
                          AND model_id='sface-2021dec')",
                [],
                |row| Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, i64>(2)?
                )),
            )
            .unwrap(),
        (0, 0, 0),
        "the worker must independently remove the Lightroom prototype and its stale suggestions"
    );
}

#[test]
fn inventory_wait_is_upserted_and_suppressed_until_inventory_refresh() {
    let root = tempfile::tempdir().unwrap();
    let source = root.path().join("wait.jpg");
    std::fs::write(&source, b"identity").unwrap();
    catalog::initialize_test_catalog(root.path()).unwrap();
    sync_analysis_inventory(root.path(), std::slice::from_ref(&source)).unwrap();
    let connection = Connection::open(catalog::catalog_path(root.path())).unwrap();
    let asset_id: i64 = connection
        .query_row("SELECT id FROM assets", [], |row| row.get(0))
        .unwrap();
    connection
        .execute("DELETE FROM face_analysis_state", [])
        .unwrap();
    defer_identity_mismatch(
        &connection,
        asset_id,
        99,
        101,
        b"test-fingerprint",
        "filesystem identity differs from inventory; waiting for inventory refresh",
    )
    .unwrap();
    drop(connection);
    assert!(pending_analysis_jobs(root.path(), 10).unwrap().is_empty());
    assert!(pending_analysis_jobs(root.path(), 10).unwrap().is_empty());
    sync_analysis_inventory(root.path(), std::slice::from_ref(&source)).unwrap();
    assert_eq!(pending_analysis_jobs(root.path(), 10).unwrap().len(), 1);
}

#[test]
fn pending_jobs_prioritize_selected_subtree_and_stop_when_drained() {
    let root = tempfile::tempdir().unwrap();
    let selected = root.path().join("selected");
    let distant = root.path().join("distant");
    std::fs::create_dir_all(&selected).unwrap();
    std::fs::create_dir_all(&distant).unwrap();
    let selected_file = selected.join("a.jpg");
    let distant_file = distant.join("b.jpg");
    let recent_file = distant.join("recent.jpg");
    let older_recent_file = distant.join("older.jpg");
    std::fs::write(&selected_file, b"selected").unwrap();
    std::fs::write(&distant_file, b"distant").unwrap();
    std::fs::write(&recent_file, b"recent").unwrap();
    std::fs::write(&older_recent_file, b"older").unwrap();
    sync_analysis_inventory(
        root.path(),
        &[
            distant_file,
            selected_file.clone(),
            recent_file.clone(),
            older_recent_file.clone(),
        ],
    )
    .unwrap();

    let jobs = pending_analysis_jobs_prioritized(
        root.path(),
        10,
        Some(&selected),
        &[recent_file, older_recent_file, selected_file],
    )
    .unwrap();
    assert_eq!(jobs.len(), 4);
    assert_eq!(jobs[0].1, 200);
    assert_eq!(jobs[1].1, 100);
    assert_eq!(jobs[2].1, 99);
    assert_eq!(jobs[3].1, 0);
    assert_eq!(next_analysis_wake(root.path()).unwrap(), None);

    let connection = Connection::open(catalog::catalog_path(root.path())).unwrap();
    connection
        .execute(
            "UPDATE face_analysis_state SET status='error', failure_class='transient',
                   retry_after=datetime('now', '+2 seconds')",
            [],
        )
        .unwrap();
    assert!(next_analysis_wake(root.path()).unwrap().is_some());
}

#[test]
fn paused_service_leaves_durable_pending_work_and_resume_completes_it() {
    let _ort = lock_ort();
    let root = tempfile::tempdir().unwrap();
    let source = root.path().join("pause.png");
    RgbImage::new(32, 32).save(&source).unwrap();
    catalog::initialize_test_catalog(root.path()).unwrap();
    sync_analysis_inventory(root.path(), std::slice::from_ref(&source)).unwrap();
    let job = pending_analysis_jobs(root.path(), 1).unwrap().remove(0);
    let admission = MaintenanceAdmission::new();
    admission.set_foreground_blocked(true);
    let pipeline = RenderPipeline::with_source_cache_capacity(root.path().join("cache"), 1, 1, 1);
    let service =
        FaceAnalysisService::new_with_pipeline(1, JobLimiter::new(2), admission.clone(), pipeline);
    assert!(service.enqueue(job.clone(), 0));
    assert_eq!(service.snapshot()[0].stage, FaceAnalysisStage::Queued);
    service.set_paused(true);
    let cancel_deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    while !service.snapshot().is_empty() && std::time::Instant::now() < cancel_deadline {
        std::thread::yield_now();
    }
    assert!(service.snapshot().is_empty());
    assert!(service.stages.lock().unwrap().is_empty());
    assert_eq!(
        Connection::open(catalog::catalog_path(root.path()))
            .unwrap()
            .query_row(
                "SELECT status FROM face_analysis_state WHERE asset_id=?1",
                [job.asset_id],
                |row| row.get::<_, String>(0)
            )
            .unwrap(),
        "pending"
    );
    service.set_paused(false);
    assert!(service.enqueue(job, 0));
    admission.set_foreground_blocked(false);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
    while service.stats().completed == 0 && std::time::Instant::now() < deadline {
        std::thread::yield_now();
    }
    assert_eq!(service.stats().completed, 1);
    assert_eq!(
        service.instrumentation(),
        FaceAnalysisInstrumentationSnapshot {
            engines_constructed: 1,
            active_inference: 0,
            max_active_inference: 1,
            inference_started: 1,
            inference_completed: 1,
        }
    );
}

#[test]
fn production_service_bounds_and_reuses_inference_sessions_under_concurrency() {
    let _ort = lock_ort();
    let root = tempfile::tempdir().unwrap();
    let sources = (0..4)
        .map(|index| {
            let source = root.path().join(format!("concurrent-{index}.png"));
            RgbImage::new(32, 32).save(&source).unwrap();
            source
        })
        .collect::<Vec<_>>();
    catalog::initialize_test_catalog(root.path()).unwrap();
    sync_analysis_inventory(root.path(), &sources).unwrap();
    let jobs = pending_analysis_jobs(root.path(), sources.len()).unwrap();
    let admission = MaintenanceAdmission::new();
    admission.set_foreground_blocked(true);
    let service = FaceAnalysisService::new_with_pipeline(
        2,
        JobLimiter::new(2),
        admission.clone(),
        RenderPipeline::with_source_cache_capacity(root.path().join("cache"), 2, 1, 1),
    );
    for job in jobs {
        assert!(service.enqueue(job, 0));
    }
    admission.set_foreground_blocked(false);
    assert!(
        service
            .scheduler
            .wait_until_settled(std::time::Duration::from_secs(240)),
        "all production workers must reach a terminal state under concurrent suite load"
    );
    let stats = service.stats();
    assert_eq!(stats.completed, sources.len() as u64);
    assert_eq!(stats.failed, 0);
    assert_eq!(stats.panicked, 0);
    assert_eq!(stats.cancelled, 0);
    let observed = service.instrumentation();
    assert_eq!(observed.active_inference, 0);
    assert_eq!(observed.inference_started, sources.len());
    assert_eq!(observed.inference_completed, sources.len());
    assert_eq!(observed.engines_constructed, 1);
    assert_eq!(observed.max_active_inference, 1);
    assert!(
        observed.engines_constructed < sources.len(),
        "sessions must be reused instead of constructed per asset"
    );
}

#[test]
fn concurrent_service_initialization_is_controlled_and_independently_bounded() {
    let _ort = lock_ort();
    let roots = [tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap()];
    let admission = MaintenanceAdmission::new();
    admission.set_foreground_blocked(true);
    let mut services = Vec::new();
    for root in &roots {
        let source = root.path().join("service.png");
        RgbImage::new(32, 32).save(&source).unwrap();
        catalog::initialize_test_catalog(root.path()).unwrap();
        sync_analysis_inventory(root.path(), std::slice::from_ref(&source)).unwrap();
        let job = pending_analysis_jobs(root.path(), 1).unwrap().remove(0);
        let service = FaceAnalysisService::new_with_pipeline(
            1,
            JobLimiter::new(2),
            admission.clone(),
            RenderPipeline::with_source_cache_capacity(root.path().join("cache"), 1, 1, 1),
        );
        assert!(service.enqueue(job, 0));
        services.push(service);
    }
    admission.set_foreground_blocked(false);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    while services
        .iter()
        .any(|service| service.stats().completed == 0)
        && std::time::Instant::now() < deadline
    {
        std::thread::yield_now();
    }
    for service in services {
        assert_eq!(service.stats().completed, 1);
        assert_eq!(service.instrumentation().engines_constructed, 1);
        assert_eq!(service.instrumentation().max_active_inference, 1);
    }
}

#[test]
fn pending_batches_advance_beyond_ten_thousand_assets() {
    let root = tempfile::tempdir().unwrap();
    catalog::initialize_test_catalog(root.path()).unwrap();
    let connection = Connection::open(catalog::catalog_path(root.path())).unwrap();
    connection
            .execute_batch(
                "WITH RECURSIVE ids(id) AS (VALUES(1) UNION ALL SELECT id+1 FROM ids WHERE id<10001)
                 INSERT INTO assets(id, relative_path, scope_key)
                   SELECT id, CAST(printf('p/%05d.jpg', id) AS BLOB), CAST('p' AS BLOB) FROM ids;
                 INSERT INTO analysis_asset_inventory(asset_id, source_size, source_modified_ns,
                                                       decoder_model_id, discovered_at)
                   SELECT id, 1, 1, 'renderer-6', datetime('now') FROM assets;
                 INSERT INTO face_analysis_state(asset_id, detector_model_id, recognizer_model_id,
                     decoder_model_id, source_size, source_modified_ns, status, attempts, updated_at)
                   SELECT id, 'yunet-2023mar+sface-2021dec', 'sface-2021dec',
                     'renderer-6', 1, 1, 'pending', 0, datetime('now') FROM assets;",
            )
            .unwrap();
    let first = pending_analysis_jobs(root.path(), 10_000).unwrap();
    assert_eq!(first.len(), 10_000);
    connection
        .execute(
            "UPDATE face_analysis_state SET status='complete' WHERE asset_id<=10000",
            [],
        )
        .unwrap();
    let second = pending_analysis_jobs(root.path(), 10_000).unwrap();
    assert_eq!(second.len(), 1);
    assert_eq!(second[0].asset_id, 10_001);
}

#[test]
fn individual_cancel_is_durable_until_explicit_retry() {
    let root = tempfile::tempdir().unwrap();
    let source = root.path().join("cancel.jpg");
    std::fs::write(&source, b"source").unwrap();
    catalog::initialize_test_catalog(root.path()).unwrap();
    sync_analysis_inventory(root.path(), std::slice::from_ref(&source)).unwrap();
    let job = pending_analysis_jobs(root.path(), 1).unwrap().remove(0);
    assert!(snooze_analysis_job(root.path(), job.asset_id).unwrap());
    assert!(pending_analysis_jobs(root.path(), 1).unwrap().is_empty());
    assert_eq!(next_analysis_wake(root.path()).unwrap(), None);
    catalog::people::retry_failed_analysis(root.path()).unwrap();
    assert_eq!(pending_analysis_jobs(root.path(), 1).unwrap(), vec![job]);
}

#[test]
fn duplicate_enqueue_never_regresses_an_active_stage_to_queued() {
    let root = tempfile::tempdir().unwrap();
    let admission = MaintenanceAdmission::new();
    admission.set_foreground_blocked(true);
    let service = FaceAnalysisService::new_with_pipeline(
        1,
        JobLimiter::new(2),
        admission,
        RenderPipeline::with_source_cache_capacity(root.path().join("cache"), 1, 1, 1),
    );
    let job = FaceAnalyzeJob {
        root: root.path().to_path_buf(),
        asset_id: 7,
    };
    assert!(service.enqueue(job.clone(), 0));
    set_analysis_stage(&service.stages, &job, FaceAnalysisStage::Detecting);
    assert!(!service.enqueue(job.clone(), 100));
    assert_eq!(
        service.stages.lock().unwrap().get(&job),
        Some(&FaceAnalysisStage::Detecting)
    );
    service.set_paused(true);
}

#[test]
fn cancelling_running_face_preserves_stage_until_execution_guard_exits() {
    let root = tempfile::tempdir().unwrap();
    let job = FaceAnalyzeJob {
        root: root.path().to_path_buf(),
        asset_id: 88,
    };
    let stages = Arc::new(Mutex::new(HashMap::new()));
    let entered = Arc::new((Mutex::new(false), std::sync::Condvar::new()));
    let release = Arc::new((Mutex::new(false), std::sync::Condvar::new()));
    let scheduler = MaintenanceScheduler::new_with_admission(1, MaintenanceAdmission::new(), {
        let stages = stages.clone();
        let entered = entered.clone();
        let release = release.clone();
        Arc::new(move |job: &FaceAnalyzeJob, cancelled| {
            let _guard = AnalysisStageGuard {
                stages: stages.clone(),
                job: job.clone(),
            };
            set_analysis_stage(&stages, job, FaceAnalysisStage::Detecting);
            *entered.0.lock().unwrap() = true;
            entered.1.notify_all();
            while !cancelled.load(Ordering::Acquire) {
                std::thread::yield_now();
            }
            let mut allowed = release.0.lock().unwrap();
            while !*allowed {
                allowed = release.1.wait(allowed).unwrap();
            }
            Ok(())
        })
    });
    let service = FaceAnalysisService {
        scheduler,
        paused: Arc::new(AtomicBool::new(false)),
        enabled: true,
        stages,
        instrumentation: Arc::new(FaceAnalysisInstrumentation::default()),
    };
    assert!(service.enqueue(job.clone(), 0));
    let mut is_entered = entered.0.lock().unwrap();
    while !*is_entered {
        is_entered = entered.1.wait(is_entered).unwrap();
    }
    drop(is_entered);
    assert!(service.cancel(&job));
    let snapshot = service.snapshot();
    assert_eq!(snapshot.len(), 1);
    assert_eq!(snapshot[0].status, MaintenanceJobStatus::Running);
    assert_eq!(snapshot[0].stage, FaceAnalysisStage::Detecting);
    assert!(snapshot[0].cancelled);
    *release.0.lock().unwrap() = true;
    release.1.notify_all();
    assert!(
        service
            .scheduler
            .wait_until_settled(std::time::Duration::from_secs(2))
    );
    assert!(service.stages.lock().unwrap().is_empty());
}

#[test]
fn cancellation_at_every_analysis_stage_restores_durable_pending_state() {
    let _ort = lock_ort();
    for stage in [
        FaceAnalysisStage::Decoding,
        FaceAnalysisStage::Detecting,
        FaceAnalysisStage::Embedding,
        FaceAnalysisStage::Reconciling,
        FaceAnalysisStage::Committing,
    ]
    .into_iter()
    {
        let root = tempfile::tempdir().unwrap();
        let source = root.path().join("stage.png");
        RgbImage::new(32, 32).save(&source).unwrap();
        catalog::initialize_test_catalog(root.path()).unwrap();
        sync_analysis_inventory(root.path(), std::slice::from_ref(&source)).unwrap();
        let job = pending_analysis_jobs(root.path(), 1).unwrap().remove(0);
        let observed = Arc::new(AtomicUsize::new(0));
        let service = FaceAnalysisService::new_with_pipeline_and_stages(
            1,
            JobLimiter::new(2),
            MaintenanceAdmission::new(),
            RenderPipeline::with_source_cache_capacity(root.path().join("cache"), 1, 1, 1),
            Arc::new(CancellingStageHook {
                target: stage,
                observed: observed.clone(),
            }),
            None,
        );
        assert!(service.enqueue(job.clone(), 0));
        assert!(
            service
                .scheduler
                .wait_until_settled(std::time::Duration::from_secs(120)),
            "stage {stage:?} must reach a terminal scheduler transition"
        );
        if observed.load(AtomicOrdering::Acquire) == 0 {
            let durable: (String, Option<String>) =
                Connection::open(catalog::catalog_path(root.path()))
                    .unwrap()
                    .query_row(
                        "SELECT status, error FROM face_analysis_state WHERE asset_id=?1",
                        [job.asset_id],
                        |row| Ok((row.get(0)?, row.get(1)?)),
                    )
                    .unwrap();
            panic!("stage {stage:?} was not reached; durable state: {durable:?}");
        }
        assert_eq!(service.stats().failed, 0, "{stage:?}");
        assert_eq!(
            durable_state(root.path(), job.asset_id),
            "pending",
            "{stage:?}"
        );
        let connection = Connection::open(catalog::catalog_path(root.path())).unwrap();
        let artifacts: i64 = connection
            .query_row(
                "SELECT count(*) FROM faces WHERE asset_id=?1 AND provenance='rekesz'",
                [job.asset_id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(artifacts, 0, "{stage:?}");
    }
}

#[test]
fn abandoned_running_job_is_reclaimed_by_a_recreated_service() {
    let _ort = lock_ort();
    let root = tempfile::tempdir().unwrap();
    let source = root.path().join("restart.png");
    RgbImage::new(32, 32).save(&source).unwrap();
    catalog::initialize_test_catalog(root.path()).unwrap();
    sync_analysis_inventory(root.path(), std::slice::from_ref(&source)).unwrap();
    let connection = Connection::open(catalog::catalog_path(root.path())).unwrap();
    connection
        .execute(
            "UPDATE face_analysis_state SET status='running', worker_id='dead-process',
                   lease_expires_at=datetime('now', '-1 second')",
            [],
        )
        .unwrap();
    drop(connection);
    let job = pending_analysis_jobs(root.path(), 1).unwrap().remove(0);
    let service = FaceAnalysisService::new_with_pipeline(
        1,
        JobLimiter::new(2),
        MaintenanceAdmission::new(),
        RenderPipeline::with_source_cache_capacity(root.path().join("restart-cache"), 1, 1, 1),
    );
    assert!(service.enqueue(job.clone(), 0));
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
    while service.stats().completed == 0 && std::time::Instant::now() < deadline {
        std::thread::yield_now();
    }
    assert_eq!(durable_state(root.path(), job.asset_id), "complete");
}

#[test]
fn exact_model_and_decoder_version_changes_requeue_only_stale_assets() {
    let root = tempfile::tempdir().unwrap();
    let paths = (1..=4)
        .map(|id| {
            let path = root.path().join(format!("{id}.jpg"));
            std::fs::write(&path, [id as u8]).unwrap();
            path
        })
        .collect::<Vec<_>>();
    catalog::initialize_test_catalog(root.path()).unwrap();
    sync_analysis_inventory(root.path(), &paths).unwrap();
    let connection = Connection::open(catalog::catalog_path(root.path())).unwrap();
    connection
        .execute("UPDATE face_analysis_state SET status='complete'", [])
        .unwrap();
    connection
        .execute(
            "UPDATE face_analysis_state SET detector_model_id='old' WHERE asset_id=2",
            [],
        )
        .unwrap();
    connection
        .execute(
            "UPDATE face_analysis_state SET recognizer_model_id='old' WHERE asset_id=3",
            [],
        )
        .unwrap();
    connection
        .execute(
            "UPDATE face_analysis_state SET decoder_model_id='old' WHERE asset_id=4",
            [],
        )
        .unwrap();
    drop(connection);
    assert_eq!(
        pending_analysis_jobs(root.path(), 10)
            .unwrap()
            .into_iter()
            .map(|job| job.asset_id)
            .collect::<Vec<_>>(),
        vec![2, 3, 4]
    );
}

#[test]
fn ten_thousand_asset_refill_coordinator_drains_multiple_real_scheduler_batches() {
    let root = tempfile::tempdir().unwrap();
    catalog::initialize_test_catalog(root.path()).unwrap();
    let connection = Connection::open(catalog::catalog_path(root.path())).unwrap();
    connection.execute_batch(
            "WITH RECURSIVE ids(id) AS (VALUES(1) UNION ALL SELECT id+1 FROM ids WHERE id<10001)
             INSERT INTO assets(id, relative_path, scope_key)
               SELECT id, CAST(printf('scale/%05d.jpg', id) AS BLOB), CAST('scale' AS BLOB) FROM ids;
             INSERT INTO analysis_asset_inventory(asset_id, source_size, source_modified_ns,
                                                   decoder_model_id, discovered_at)
               SELECT id, 1, 1, 'renderer-6', datetime('now') FROM assets;
             INSERT INTO face_analysis_state(asset_id, detector_model_id, recognizer_model_id,
                 decoder_model_id, source_size, source_modified_ns, status, attempts, updated_at)
               SELECT id, 'yunet-2023mar+sface-2021dec', 'sface-2021dec', 'renderer-6',
                 1, 1, 'pending', 0, datetime('now') FROM assets;",
        ).unwrap();
    drop(connection);
    let scheduler = MaintenanceScheduler::new_with_admission(
        1,
        MaintenanceAdmission::new(),
        Arc::new(|job: &FaceAnalyzeJob, _| {
            Connection::open(catalog::catalog_path(&job.root))?.execute(
                "UPDATE face_analysis_state SET status='complete' WHERE asset_id=?1",
                [job.asset_id],
            )?;
            Ok(())
        }),
    );
    let mut submitted = 0;
    loop {
        let batch = pending_analysis_jobs(root.path(), 777).unwrap();
        if batch.is_empty() {
            break;
        }
        let before = scheduler.stats().completed;
        for job in batch {
            submitted += usize::from(scheduler.submit(job, 0));
        }
        while scheduler.stats().completed < before + 777_u64.min(10_001 - before) {
            std::thread::yield_now();
        }
    }
    assert_eq!(submitted, 10_001);
    assert!(pending_analysis_jobs(root.path(), 1).unwrap().is_empty());
}

#[test]
fn recognition_service_drains_more_than_one_batch_and_recovers_expired_restart_work() {
    let root = tempfile::tempdir().unwrap();
    catalog::initialize_test_catalog(root.path()).unwrap();
    let connection = Connection::open(catalog::catalog_path(root.path())).unwrap();
    connection
        .execute_batch(
            "WITH RECURSIVE ids(id) AS
                    (VALUES(1) UNION ALL SELECT id+1 FROM ids WHERE id<41)
                 INSERT INTO recognition_refresh_queue(
                    entity_kind, entity_id, model_id, status, attempts, updated_at)
                 SELECT 'face', id, 'sface-2021dec', 'pending', 0, datetime('now') FROM ids;",
        )
        .unwrap();
    drop(connection);

    let admission = MaintenanceAdmission::new();
    let service = RecognitionRefreshService::new(1, admission.clone());
    assert!(service.enqueue(root.path().to_path_buf(), 0));
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while service.stats().completed == 0 && std::time::Instant::now() < deadline {
        std::thread::yield_now();
    }
    assert_eq!(service.stats().completed, 1);
    let connection = Connection::open(catalog::catalog_path(root.path())).unwrap();
    assert_eq!(
        connection
            .query_row(
                "SELECT count(*) FROM recognition_refresh_queue",
                [],
                |row| row.get::<_, i64>(0),
            )
            .unwrap(),
        0,
        "one scheduler wake must cooperatively drain all 41 rows, not only 16"
    );
    connection
        .execute(
            "INSERT INTO recognition_refresh_queue(entity_kind, entity_id, model_id,
                    status, attempts, worker_id, lease_expires_at, updated_at)
                 VALUES('face', 99, 'sface-2021dec', 'running', 1, 'dead-worker',
                        datetime('now', '-1 second'), datetime('now'))",
            [],
        )
        .unwrap();
    drop(connection);
    drop(service);

    let restarted = RecognitionRefreshService::new(1, admission);
    assert!(restarted.enqueue(root.path().to_path_buf(), 0));
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while restarted.stats().completed == 0 && std::time::Instant::now() < deadline {
        std::thread::yield_now();
    }
    assert_eq!(restarted.stats().completed, 1);
    assert_eq!(
        Connection::open(catalog::catalog_path(root.path()))
            .unwrap()
            .query_row(
                "SELECT count(*) FROM recognition_refresh_queue",
                [],
                |row| row.get::<_, i64>(0),
            )
            .unwrap(),
        0,
        "a new process must reclaim an expired durable lease"
    );

    let connection = Connection::open(catalog::catalog_path(root.path())).unwrap();
    connection
        .execute(
            "INSERT INTO recognition_refresh_queue(entity_kind, entity_id, model_id,
                    status, attempts, retry_after, updated_at)
                 VALUES('face', 100, 'sface-2021dec', 'pending', 2,
                        datetime('now', '+1 hour'), datetime('now'))",
            [],
        )
        .unwrap();
    drop(connection);
    let retry_aware = RecognitionRefreshService::new(1, MaintenanceAdmission::new());
    assert!(retry_aware.enqueue(root.path().to_path_buf(), 0));
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    while retry_aware.stats().completed == 0 && std::time::Instant::now() < deadline {
        std::thread::yield_now();
    }
    assert_eq!(retry_aware.stats().completed, 1);
    assert_eq!(
        Connection::open(catalog::catalog_path(root.path()))
            .unwrap()
            .query_row(
                "SELECT attempts FROM recognition_refresh_queue WHERE entity_id=100",
                [],
                |row| row.get::<_, i64>(0),
            )
            .unwrap(),
        2,
        "a retry-delayed row must remain dormant instead of causing a drain busy-loop"
    );
}

#[test]
fn recognition_service_exposes_and_cancels_individual_catalog_job() {
    let root = tempfile::tempdir().unwrap();
    catalog::initialize_test_catalog(root.path()).unwrap();
    let admission = MaintenanceAdmission::new();
    admission.set_foreground_blocked(true);
    let service = RecognitionRefreshService::new(1, admission.clone());
    assert!(service.enqueue(root.path().to_path_buf(), 73));
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    while service.snapshot().is_empty() && std::time::Instant::now() < deadline {
        std::thread::yield_now();
    }
    let snapshot = service.snapshot();
    assert_eq!(snapshot.len(), 1);
    assert_eq!(snapshot[0].job.root, root.path());
    assert_eq!(snapshot[0].rank, 73);
    assert!(service.cancel(root.path()));
    admission.set_foreground_blocked(false);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    while !service.snapshot().is_empty() && std::time::Instant::now() < deadline {
        std::thread::yield_now();
    }
    assert!(service.snapshot().is_empty());
    assert_eq!(service.stats().cancelled, 1);
    assert_eq!(service.stats().completed, 0);
}
