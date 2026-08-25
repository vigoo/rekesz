use super::*;
use std::{
    sync::mpsc,
    thread,
    time::{Duration, Instant, SystemTime},
};
use test_r::test;

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

    let deadline = Instant::now() + Duration::from_secs(2);
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
        finished_rx.recv_timeout(Duration::from_secs(2)).unwrap(),
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
        finished_rx.recv_timeout(Duration::from_secs(2)).unwrap(),
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
    let raw = deduplicate_cr3_jpeg_pairs(paths.clone(), RawJpegPreference::Raw);
    assert!(raw.contains(&PathBuf::from("IMG_1.CR3")));
    assert!(!raw.contains(&PathBuf::from("IMG_1.JPG")));
    assert!(raw.contains(&PathBuf::from("IMG_2.JPG")));
    assert!(raw.contains(&PathBuf::from("IMG_1.PNG")));

    let jpeg = deduplicate_cr3_jpeg_pairs(paths, RawJpegPreference::Jpeg);
    assert!(jpeg.contains(&PathBuf::from("IMG_1.JPG")));
    assert!(!jpeg.contains(&PathBuf::from("IMG_1.CR3")));
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
            },
            ImageOrderingMetadata {
                path: PathBuf::from("/pictures/earlier.jpg"),
                capture_time: Some(SystemTime::UNIX_EPOCH + Duration::from_secs(10)),
                fallback_time: None,
            },
        ],
    ));
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
        }),
    );
    assert!(model.apply_cached_metadata(
        root,
        vec![
            ImageOrderingMetadata {
                path: PathBuf::from("/pictures/later.jpg"),
                capture_time: Some(SystemTime::UNIX_EPOCH + Duration::from_secs(20)),
                fallback_time: None,
            },
            ImageOrderingMetadata {
                path: PathBuf::from("/pictures/earlier.jpg"),
                capture_time: Some(SystemTime::UNIX_EPOCH + Duration::from_secs(10)),
                fallback_time: None,
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
            },
            ImageOrderingMetadata {
                path: later,
                capture_time: Some(SystemTime::UNIX_EPOCH + Duration::from_secs(20)),
                fallback_time: None,
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
    fs::File::open(&later)
        .unwrap()
        .set_times(
            fs::FileTimes::new().set_modified(SystemTime::UNIX_EPOCH + Duration::from_secs(20)),
        )
        .unwrap();
    fs::File::open(&earlier)
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
        }),
    ));
    assert!(matches!(
        model.node(root).contents,
        DirectoryContents::Loading(active) if active == request
    ));
}
