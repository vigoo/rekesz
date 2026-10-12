use super::*;
use std::{
    collections::HashSet,
    sync::mpsc,
    thread,
    time::{Duration, Instant, SystemTime},
};
use test_r::test;

// CI runners are slower and more contended than dev machines; REKESZ_TEST_TIMING_SCALE
// stretches the timing guards without weakening them locally.
fn scaled_budget(budget: Duration) -> Duration {
    let scale = std::env::var("REKESZ_TEST_TIMING_SCALE")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(1.0)
        .max(1.0);
    budget.mul_f64(scale)
}

#[test]
fn reserved_catalog_directory_is_not_discovered() {
    let root = tempfile::tempdir().unwrap();
    fs::create_dir(root.path().join(crate::catalog::CATALOG_DIR)).unwrap();
    fs::create_dir(root.path().join("visible")).unwrap();
    let scan = scan_directory(root.path(), RawJpegPreference::Raw).unwrap();
    assert_eq!(scan.directories, [root.path().join("visible")]);
}

#[test]
fn recognizes_raw_and_heif_extensions_case_insensitively() {
    for name in [
        "photo.nef",
        "photo.NEF",
        "photo.cr3",
        "photo.HEIC",
        "photo.heif",
    ] {
        assert!(is_supported_image(Path::new(name)), "{name}");
    }
    assert!(!is_supported_image(Path::new("notes.txt")));
}

#[test]
fn promoting_a_queued_directory_overtakes_background_scans() {
    let root = tempfile::tempdir().unwrap();
    let background_path = root.path().join("background");
    let selected_path = root.path().join("selected");
    fs::create_dir(&background_path).unwrap();
    fs::create_dir(&selected_path).unwrap();

    let scanner = DirectoryScanner::new(2, JobLimiter::new(2));
    {
        let (lock, _) = &*scanner.permits;
        lock.lock().unwrap().available = 0;
    }
    let (finished_tx, finished_rx) = mpsc::channel();
    for path in [background_path.clone(), selected_path.clone()] {
        let scanner = scanner.clone();
        let finished_tx = finished_tx.clone();
        thread::spawn(move || {
            scanner
                .scan(
                    &path,
                    RawJpegPreference::Raw,
                    DirectoryScanPriority::Background,
                )
                .unwrap();
            finished_tx.send(path).unwrap();
        });
    }

    let deadline = Instant::now() + scaled_budget(Duration::from_secs(2));
    loop {
        let (lock, _) = &*scanner.permits;
        if lock.lock().unwrap().waiters.len() == 2 {
            break;
        }
        assert!(Instant::now() < deadline, "scans did not enter the queue");
        thread::yield_now();
    }

    scanner.promote(&selected_path, DirectoryScanPriority::Interactive);
    {
        let (lock, wake) = &*scanner.permits;
        lock.lock().unwrap().available = 1;
        wake.notify_all();
    }
    assert_eq!(
        finished_rx
            .recv_timeout(scaled_budget(Duration::from_secs(2)))
            .unwrap(),
        selected_path
    );

    // Release the background-reserved capacity so the helper thread can
    // finish instead of leaking into later tests.
    {
        let (lock, wake) = &*scanner.permits;
        lock.lock().unwrap().available = 2;
        wake.notify_all();
    }
    assert_eq!(
        finished_rx
            .recv_timeout(scaled_budget(Duration::from_secs(2)))
            .unwrap(),
        background_path
    );
}

#[test]
fn deduplicates_cr3_jpeg_pairs_according_to_preference() {
    let paths = vec![
        PathBuf::from("IMG_1.JPG"),
        PathBuf::from("IMG_1.CR3"),
        PathBuf::from("IMG_1.PNG"),
        PathBuf::from("IMG_2.JPG"),
    ];
    let (raw, companions) = deduplicate_cr3_jpeg_pairs(paths.clone(), RawJpegPreference::Raw);
    assert!(raw.contains(&PathBuf::from("IMG_1.CR3")));
    assert!(!raw.contains(&PathBuf::from("IMG_1.JPG")));
    assert!(raw.contains(&PathBuf::from("IMG_2.JPG")));
    assert!(raw.contains(&PathBuf::from("IMG_1.PNG")));
    // The collapsed JPEG is remembered as the raw's thumbnail companion.
    assert_eq!(
        companions.get(&PathBuf::from("IMG_1.CR3")),
        Some(&PathBuf::from("IMG_1.JPG"))
    );
    assert_eq!(companions.len(), 1);

    let (jpeg, jpeg_companions) = deduplicate_cr3_jpeg_pairs(paths, RawJpegPreference::Jpeg);
    assert!(jpeg.contains(&PathBuf::from("IMG_1.JPG")));
    assert!(!jpeg.contains(&PathBuf::from("IMG_1.CR3")));
    // JPEG-preferred listings already display the JPEG; nothing to map.
    assert!(jpeg_companions.is_empty());
}

#[test]
fn a_node_can_only_start_one_load() {
    let mut model = DirectoryModel::new(PathBuf::from("/pictures"));
    let root = model.root();
    assert!(model.begin_load(root).is_some());
    assert!(model.begin_load(root).is_none());
}

#[test]
fn loaded_and_failed_nodes_are_cached() {
    let mut model = DirectoryModel::new(PathBuf::from("/pictures"));
    let root = model.root();
    let (request, _) = model.begin_load(root).unwrap();
    assert!(model.finish_load(
        root,
        request,
        Ok(DirectoryScan {
            directories: vec![PathBuf::from("/pictures/a")],
            images: vec![PathBuf::from("/pictures/one.jpg")],
            companions: HashMap::new(),
        }),
    ));
    assert!(model.begin_load(root).is_none());
    let loaded = model.loaded(root).unwrap();
    assert_eq!(loaded.children.len(), 1);
    assert_eq!(loaded.images.len(), 1);
}

#[test]
fn newly_loaded_metadata_proposes_order_without_applying_it() {
    let mut model = DirectoryModel::new(PathBuf::from("/pictures"));
    let root = model.root();
    let (request, _) = model.begin_load(root).unwrap();
    model.finish_load(
        root,
        request,
        Ok(DirectoryScan {
            directories: Vec::new(),
            images: vec![
                PathBuf::from("/pictures/later.jpg"),
                PathBuf::from("/pictures/earlier.jpg"),
            ],
            companions: HashMap::new(),
        }),
    );
    let (request, _) = model.begin_metadata_load(root).unwrap();
    assert!(model.finish_metadata_load(
        root,
        request,
        vec![
            ImageOrderingMetadata {
                path: PathBuf::from("/pictures/later.jpg"),
                capture_time: Some(SystemTime::UNIX_EPOCH + Duration::from_secs(20)),
                fallback_time: None,
                coordinate: GeoCoordinate::new(47.5, 19.05),
            },
            ImageOrderingMetadata {
                path: PathBuf::from("/pictures/earlier.jpg"),
                capture_time: Some(SystemTime::UNIX_EPOCH + Duration::from_secs(10)),
                fallback_time: None,
                coordinate: None,
            },
        ],
    ));
    assert_eq!(
        model.coordinate(root, Path::new("/pictures/later.jpg")),
        GeoCoordinate::new(47.5, 19.05)
    );
    assert_eq!(
        model.loaded(root).unwrap().images,
        vec![
            PathBuf::from("/pictures/later.jpg"),
            PathBuf::from("/pictures/earlier.jpg")
        ]
    );
    assert!(model.has_pending_capture_order(root));
    assert!(model.apply_pending_capture_order(root));
    assert_eq!(
        model.loaded(root).unwrap().images,
        vec![
            PathBuf::from("/pictures/earlier.jpg"),
            PathBuf::from("/pictures/later.jpg")
        ]
    );
}

#[test]
fn cached_metadata_is_applied_to_the_initial_order() {
    let mut model = DirectoryModel::new(PathBuf::from("/pictures"));
    let root = model.root();
    let (request, _) = model.begin_load(root).unwrap();
    model.finish_load(
        root,
        request,
        Ok(DirectoryScan {
            directories: Vec::new(),
            images: vec![
                PathBuf::from("/pictures/later.jpg"),
                PathBuf::from("/pictures/earlier.jpg"),
            ],
            companions: HashMap::new(),
        }),
    );
    assert!(model.apply_cached_metadata(
        root,
        vec![
            ImageOrderingMetadata {
                path: PathBuf::from("/pictures/later.jpg"),
                capture_time: Some(SystemTime::UNIX_EPOCH + Duration::from_secs(20)),
                fallback_time: None,
                coordinate: None,
            },
            ImageOrderingMetadata {
                path: PathBuf::from("/pictures/earlier.jpg"),
                capture_time: Some(SystemTime::UNIX_EPOCH + Duration::from_secs(10)),
                fallback_time: None,
                coordinate: None,
            },
        ],
        Vec::new(),
    ));
    assert_eq!(
        model.loaded(root).unwrap().images,
        vec![
            PathBuf::from("/pictures/earlier.jpg"),
            PathBuf::from("/pictures/later.jpg")
        ]
    );
    assert!(!model.has_pending_capture_order(root));
}

#[test]
fn metadata_that_keeps_the_current_order_does_not_offer_apply() {
    let mut model = DirectoryModel::new(PathBuf::from("/pictures"));
    let root = model.root();
    let later = PathBuf::from("/pictures/later.jpg");
    let earlier = PathBuf::from("/pictures/earlier.jpg");
    let (request, _) = model.begin_load(root).unwrap();
    model.finish_load(
        root,
        request,
        Ok(DirectoryScan {
            directories: Vec::new(),
            images: vec![earlier.clone(), later.clone()],
            companions: HashMap::new(),
        }),
    );
    let (request, _) = model.begin_metadata_load(root).unwrap();
    assert!(model.finish_metadata_load(
        root,
        request,
        vec![
            ImageOrderingMetadata {
                path: earlier,
                capture_time: Some(SystemTime::UNIX_EPOCH + Duration::from_secs(10)),
                fallback_time: None,
                coordinate: None,
            },
            ImageOrderingMetadata {
                path: later,
                capture_time: Some(SystemTime::UNIX_EPOCH + Duration::from_secs(20)),
                fallback_time: None,
                coordinate: None,
            },
        ],
    ));
    assert!(!model.has_pending_capture_order(root));
}

#[test]
fn initial_directory_order_uses_modification_time() {
    let root = tempfile::tempdir().unwrap();
    let later = root.path().join("later.jpg");
    let earlier = root.path().join("earlier.jpg");
    fs::write(&later, b"not decoded during enumeration").unwrap();
    fs::write(&earlier, b"not decoded during enumeration").unwrap();
    // Windows needs a write handle to set file times; File::open is read-only.
    fs::OpenOptions::new()
        .write(true)
        .open(&later)
        .unwrap()
        .set_times(
            fs::FileTimes::new().set_modified(SystemTime::UNIX_EPOCH + Duration::from_secs(20)),
        )
        .unwrap();
    fs::OpenOptions::new()
        .write(true)
        .open(&earlier)
        .unwrap()
        .set_times(
            fs::FileTimes::new().set_modified(SystemTime::UNIX_EPOCH + Duration::from_secs(10)),
        )
        .unwrap();

    let scan = scan_directory(root.path(), RawJpegPreference::Raw).unwrap();
    assert_eq!(scan.images, vec![earlier, later]);
}

#[test]
fn stale_results_are_ignored() {
    let mut model = DirectoryModel::new(PathBuf::from("/pictures"));
    let root = model.root();
    let (request, _) = model.begin_load(root).unwrap();
    assert!(!model.finish_load(
        root,
        request + 1,
        Ok(DirectoryScan {
            directories: Vec::new(),
            images: Vec::new(),
            companions: HashMap::new(),
        }),
    ));
    assert!(matches!(
        model.node(root).contents,
        DirectoryContents::Loading(active) if active == request
    ));
}

#[test]
fn library_inventory_applies_hidden_supported_and_pair_rules() {
    let root = tempfile::tempdir().unwrap();
    let visible = root.path().join("visible");
    let hidden = root.path().join("hidden");
    fs::create_dir_all(&visible).unwrap();
    fs::create_dir_all(&hidden).unwrap();
    fs::write(visible.join("pair.cr3"), b"raw").unwrap();
    fs::write(visible.join("pair.jpg"), b"jpeg").unwrap();
    fs::write(visible.join("notes.txt"), b"no").unwrap();
    fs::write(hidden.join("secret.jpg"), b"hidden").unwrap();

    let inventory = scan_library_inventory(
        root.path(),
        &HashSet::from([hidden]),
        RawJpegPreference::Jpeg,
        &AtomicBool::new(false),
    )
    .unwrap();
    assert_eq!(inventory, vec![visible.join("pair.jpg")]);
}

#[test]
fn library_inventory_progress_reports_traversal_frontier_and_current_directory() {
    let root = tempfile::tempdir().unwrap();
    let nested = root.path().join("nested");
    fs::create_dir(&nested).unwrap();
    fs::write(nested.join("portrait.jpg"), b"image").unwrap();
    let mut updates = Vec::new();
    let images = scan_library_inventory_with_progress(
        root.path(),
        &HashSet::new(),
        RawJpegPreference::Raw,
        &AtomicBool::new(false),
        |scanned, pending, images, current| {
            updates.push((scanned, pending, images, current.to_path_buf()));
        },
    )
    .unwrap();
    assert_eq!(images, vec![nested.clone().join("portrait.jpg")]);
    assert_eq!(updates.last(), Some(&(2, 0, 1, nested)));
}

#[test]
fn cancelled_library_inventory_is_an_error_not_a_destructive_partial_snapshot() {
    let root = tempfile::tempdir().unwrap();
    let cancelled = AtomicBool::new(true);
    let error = scan_library_inventory(
        root.path(),
        &HashSet::new(),
        RawJpegPreference::Raw,
        &cancelled,
    )
    .unwrap_err();
    assert_eq!(error.kind(), std::io::ErrorKind::Interrupted);
}
