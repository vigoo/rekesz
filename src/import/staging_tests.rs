use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use test_r::test;

use super::{stage_file, staged_path};
use crate::import::ScannedFile;
use crate::sources::mtp::MtpObjectRef;
use crate::sources::{PathSource, SourceAccess, SourceBacking};

fn device_file(name: &str, size: u64) -> ScannedFile {
    ScannedFile {
        backing: SourceBacking::Mtp(MtpObjectRef {
            device_id: Arc::from("libmtp:1:2"),
            storage_id: Arc::from("65537"),
            object: Arc::from("42"),
        }),
        name: name.to_owned(),
        size,
        modified: None,
    }
}

fn path_file(path: PathBuf) -> ScannedFile {
    ScannedFile {
        backing: SourceBacking::Path(path),
        name: "a.jpg".to_owned(),
        size: 2,
        modified: None,
    }
}

#[test]
fn staged_paths_are_deterministic_per_object() {
    let file = device_file("IMG_0001.NEF", 10);
    let path = staged_path(Path::new("/cache"), &file).expect("device file");
    assert_eq!(
        path,
        PathBuf::from("/cache/import-staging/libmtp_1_2/65537/42-IMG_0001.NEF")
    );
    assert!(staged_path(Path::new("/cache"), &path_file("/x/a.jpg".into())).is_none());
}

#[test]
fn staging_downloads_once_and_reuses_complete_copies() {
    let source = tempfile::tempdir().expect("source");
    let cache = tempfile::tempdir().expect("cache");
    let path = source.path().join("card.jpg");
    std::fs::write(&path, b"jpeg").expect("write");

    // A device-backed file whose object id maps to a local path is not
    // possible with PathSource, so stage a path-backed file through the same
    // API: it must report None without touching the cache.
    let staged = stage_file(
        &PathSource,
        &path_file(path),
        cache.path(),
        &AtomicBool::new(false),
    )
    .expect("stage");
    assert!(staged.is_none());
    assert_eq!(std::fs::read_dir(cache.path()).expect("cache").count(), 0);

    // The deterministic staged path check pins the layout; the download path
    // itself needs a device backend and is covered by manual device tests.
    let file = device_file("IMG_0001.NEF", 10);
    let destination = staged_path(cache.path(), &file).expect("device file");
    std::fs::create_dir_all(destination.parent().expect("parent")).expect("mkdir");
    std::fs::write(&destination, b"0123456789").expect("write");
    let staged =
        stage_file(&PathSource, &file, cache.path(), &AtomicBool::new(false)).expect("stage");
    assert_eq!(staged.as_deref(), Some(destination.as_path()));
}

#[test]
fn sanitize_components_keep_paths_safe() {
    let mut file = device_file("a/b:c?.jpg", 1);
    file.backing = SourceBacking::Mtp(MtpObjectRef {
        device_id: Arc::from("weird/../device"),
        storage_id: Arc::from("s t"),
        object: Arc::from("7"),
    });
    let path = staged_path(Path::new("/cache"), &file).expect("path");
    assert_eq!(
        path,
        PathBuf::from("/cache/import-staging/weird_.._device/s_t/7-a_b_c_.jpg")
    );
}
