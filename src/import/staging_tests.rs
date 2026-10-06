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

fn ptp_file(name: &str, size: u64) -> ScannedFile {
    ScannedFile {
        backing: SourceBacking::Ptp(crate::sources::ptp::PtpObjectRef {
            device_id: Arc::from("gphoto:usb:005,012"),
            folder: Arc::from("/DCIM/100CANON"),
            name: Arc::from(""),
        }),
        name: name.to_owned(),
        size,
        modified: None,
    }
}

#[test]
fn ptp_staged_paths_key_folder_and_name() {
    let file = ptp_file("IMG_0001.JPG", 10);
    let path = staged_path(Path::new("/cache"), &file).expect("device file");
    assert_eq!(
        path,
        PathBuf::from("/cache/import-staging/gphoto_usb_005_012/DCIM_100CANON-IMG_0001.JPG")
    );
}

/// A source with a native to-path transfer, like the gphoto2 backend:
/// the bytes never pass through `read_file`.
struct DirectSource {
    payload: Vec<u8>,
    modified: Option<std::time::SystemTime>,
}

impl SourceAccess for DirectSource {
    fn list_dir(
        &self,
        _dir: &SourceBacking,
    ) -> std::io::Result<Vec<crate::sources::SourceDirEntry>> {
        unreachable!("staging tests never list")
    }

    fn read_file(
        &self,
        _file: &SourceBacking,
        _out: &mut dyn std::io::Write,
        _cancel: &AtomicBool,
        _progress: &mut dyn FnMut(u64),
    ) -> std::io::Result<u64> {
        unreachable!("direct source transfers via read_file_to_path")
    }

    fn read_file_to_path(
        &self,
        _file: &SourceBacking,
        destination: &Path,
        cancel: &AtomicBool,
        progress: &mut dyn FnMut(u64),
    ) -> Option<std::io::Result<u64>> {
        if cancel.load(std::sync::atomic::Ordering::Acquire) {
            return Some(Err(std::io::Error::new(
                std::io::ErrorKind::Interrupted,
                "cancelled",
            )));
        }
        if let Err(error) = std::fs::write(destination, &self.payload) {
            return Some(Err(error));
        }
        progress(self.payload.len() as u64);
        Some(Ok(self.payload.len() as u64))
    }
}

#[test]
fn direct_staging_carries_the_device_timestamp() {
    let cache = tempfile::tempdir().expect("cache");
    let modified =
        std::time::SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(1_700_000_000);
    let mut file = ptp_file("IMG_0001.JPG", 4);
    file.modified = Some(modified);

    let source = DirectSource {
        payload: b"jpeg".to_vec(),
        modified: None,
    };
    let staged = stage_file(&source, &file, cache.path(), &AtomicBool::new(false))
        .expect("stage")
        .expect("device file");
    assert_eq!(std::fs::read(&staged).expect("read"), b"jpeg");
    // The device mtime must survive staging (the reuse check compares it
    // against the device timestamp on every browse).
    assert_eq!(
        std::fs::metadata(&staged).expect("meta").modified().ok(),
        Some(modified)
    );

    // A second stage of the same file reuses the copy.
    let again = stage_file(&source, &file, cache.path(), &AtomicBool::new(false))
        .expect("stage")
        .expect("device file");
    assert_eq!(again, staged);
}
