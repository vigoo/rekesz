use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, SystemTime};

use chrono::{Local, NaiveDate, TimeZone};
use test_r::test;

use super::ScannedFile;
use super::ScannedPhoto;
use super::destination::{DestinationPreset, capture_date};
use super::duplicates::{LibraryIndex, key_for};
use super::{ImportPlan, ImportProgress, ImportStatus, ScanError, run_import, scan_source};
use crate::discovery::RawJpegPreference;
use crate::sources::{PathSource, SourceAccess, SourceBacking, SourceDirEntry, SourceRegistry};

const FIXED_MTIME: u64 = 1_700_000_000;

/// SourceAccess wrapper that flips a cancel flag after the first reported
/// chunk, making mid-copy cancellation deterministic in tests.
struct CancellingAfterFirstChunk {
    cancel: Arc<AtomicBool>,
}

impl SourceAccess for CancellingAfterFirstChunk {
    fn list_dir(&self, dir: &SourceBacking) -> std::io::Result<Vec<SourceDirEntry>> {
        PathSource.list_dir(dir)
    }

    fn read_file(
        &self,
        file: &SourceBacking,
        out: &mut dyn std::io::Write,
        cancel: &AtomicBool,
        progress: &mut dyn FnMut(u64),
    ) -> std::io::Result<u64> {
        let cancel_flag = Arc::clone(&self.cancel);
        PathSource.read_file(file, out, cancel, &mut |copied| {
            cancel_flag.store(true, Ordering::Release);
            progress(copied);
        })
    }
}

fn fixed_time() -> SystemTime {
    SystemTime::UNIX_EPOCH + Duration::from_secs(FIXED_MTIME)
}

fn expected_fixed_date() -> NaiveDate {
    Local
        .timestamp_opt(FIXED_MTIME as i64, 0)
        .single()
        .expect("fixed timestamp")
        .date_naive()
}

fn file_photo(directory: &str, path: PathBuf, name: &str, size: u64) -> ScannedPhoto {
    ScannedPhoto {
        directory: directory.to_owned(),
        primary: ScannedFile {
            backing: SourceBacking::Path(path),
            name: name.to_owned(),
            size,
            modified: Some(fixed_time()),
        },
        companion: None,
    }
}

#[test]
fn destination_presets_format_dates() {
    let date = NaiveDate::from_ymd_opt(2026, 10, 4).expect("date");
    assert_eq!(
        DestinationPreset::YearDay.directory_for(date),
        PathBuf::from("2026").join("2026-10-04")
    );
    assert_eq!(
        DestinationPreset::YearMonth.directory_for(date),
        PathBuf::from("2026").join("10")
    );
    // The flat layouts land directly in the chosen destination folder.
    assert_eq!(
        DestinationPreset::KeepFileNames.directory_for(date),
        PathBuf::new()
    );
    assert_eq!(
        DestinationPreset::Counter.directory_for(date),
        PathBuf::new()
    );
    assert!(DestinationPreset::Counter.uses_counter());
    assert!(!DestinationPreset::KeepFileNames.uses_counter());
}

#[test]
fn capture_date_prefers_mtime_fallback() {
    let root = tempfile::tempdir().expect("tempdir");
    let path = root.path().join("x.jpg");
    std::fs::write(&path, b"jpeg").expect("write");
    // Windows needs a write handle to set file times; File::open is read-only.
    std::fs::OpenOptions::new()
        .write(true)
        .open(&path)
        .and_then(|file| file.set_modified(fixed_time()))
        .expect("set mtime");
    let file = ScannedFile {
        backing: SourceBacking::Path(path),
        name: "x.jpg".to_owned(),
        size: 4,
        modified: Some(fixed_time()),
    };
    assert_eq!(capture_date(&file), expected_fixed_date());
}

#[test]
fn capture_date_uses_device_mtime_without_filesystem() {
    let file = ScannedFile {
        backing: SourceBacking::Mtp(crate::sources::mtp::MtpObjectRef {
            device_id: Arc::from("dev"),
            storage_id: Arc::from("stor"),
            object: Arc::from("12"),
        }),
        name: "IMG_0001.NEF".to_owned(),
        size: 10,
        modified: Some(fixed_time()),
    };
    assert_eq!(capture_date(&file), expected_fixed_date());
}

#[test]
fn capture_date_falls_back_to_today() {
    let before = Local::now().date_naive();
    let file = ScannedFile {
        backing: SourceBacking::Path(PathBuf::from("/nonexistent/nope.jpg")),
        name: "nope.jpg".to_owned(),
        size: 0,
        modified: None,
    };
    let date = capture_date(&file);
    let after = Local::now().date_naive();
    assert!(date >= before && date <= after);
}

#[test]
fn scan_source_collects_and_pairs_photos() {
    let root = tempfile::tempdir().expect("tempdir");
    std::fs::write(root.path().join("top.jpg"), b"top").expect("write");
    let dcim = root.path().join("DCIM");
    std::fs::create_dir(&dcim).expect("mkdir");
    std::fs::write(dcim.join("IMG_001.CR3"), b"raw-data").expect("write");
    std::fs::write(dcim.join("IMG_001.JPG"), b"jpeg-data").expect("write");
    std::fs::write(dcim.join("IMG_002.NEF"), b"nef").expect("write");
    std::fs::write(dcim.join("notes.txt"), b"skip").expect("write");
    let sub = dcim.join("sub");
    std::fs::create_dir(&sub).expect("mkdir");
    std::fs::write(sub.join("IMG_003.JPG"), b"sub").expect("write");

    let outcome = scan_source(
        &PathSource,
        &SourceBacking::Path(root.path().to_path_buf()),
        RawJpegPreference::Raw,
        &AtomicBool::new(false),
        &mut |_| {},
    )
    .expect("scan");
    assert_eq!(outcome.skipped_directories, 0);
    let listed = outcome
        .photos
        .iter()
        .map(|photo| {
            (
                photo.directory.as_str(),
                photo.primary.name.as_str(),
                photo.companion.as_ref().map(|file| file.name.as_str()),
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(
        listed,
        vec![
            ("", "top.jpg", None),
            ("DCIM", "IMG_001.CR3", Some("IMG_001.JPG")),
            ("DCIM", "IMG_002.NEF", None),
            ("DCIM/sub", "IMG_003.JPG", None),
        ]
    );
    assert_eq!(outcome.photos[0].primary.size, 3);
    assert_eq!(outcome.photos[1].primary.size, 8);
    assert_eq!(
        outcome.photos[1]
            .companion
            .as_ref()
            .expect("companion")
            .size,
        9
    );
}

#[test]
fn scan_source_respects_jpeg_preference() {
    let root = tempfile::tempdir().expect("tempdir");
    std::fs::write(root.path().join("P.CR3"), b"raw").expect("write");
    std::fs::write(root.path().join("P.JPG"), b"jpg").expect("write");
    let outcome = scan_source(
        &PathSource,
        &SourceBacking::Path(root.path().to_path_buf()),
        RawJpegPreference::Jpeg,
        &AtomicBool::new(false),
        &mut |_| {},
    )
    .expect("scan");
    assert_eq!(outcome.photos[0].primary.name, "P.JPG");
    assert_eq!(
        outcome.photos[0].companion.as_ref().expect("c").name,
        "P.CR3"
    );
}

#[test]
fn scan_source_honours_cancellation() {
    let root = tempfile::tempdir().expect("tempdir");
    std::fs::write(root.path().join("a.jpg"), b"a").expect("write");
    let error = scan_source(
        &PathSource,
        &SourceBacking::Path(root.path().to_path_buf()),
        RawJpegPreference::Raw,
        &AtomicBool::new(true),
        &mut |_| {},
    )
    .expect_err("cancelled");
    assert!(matches!(error, ScanError::Cancelled));
}

#[test]
fn library_index_maps_files_and_skips_catalog() {
    let root = tempfile::tempdir().expect("tempdir");
    std::fs::write(root.path().join("a.jpg"), b"12345").expect("write");
    let sub = root.path().join("sub");
    std::fs::create_dir(&sub).expect("mkdir");
    std::fs::write(sub.join("b.nef"), b"nef!").expect("write");
    let catalog = root.path().join(crate::catalog::CATALOG_DIR);
    std::fs::create_dir(&catalog).expect("mkdir");
    std::fs::write(catalog.join("catalog.sqlite3"), b"catalog").expect("write");

    let index =
        LibraryIndex::build(root.path(), &AtomicBool::new(false), &mut |_| {}).expect("index");
    assert_eq!(index.files, 2);
    assert!(index.contains("A.JPG", 5));
    assert!(index.contains("b.nef", 4));
    assert!(!index.contains("catalog.sqlite3", 7));
}

#[test]
fn run_import_copies_with_preset_layout_and_mtime() {
    let source = tempfile::tempdir().expect("source");
    let library = tempfile::tempdir().expect("library");
    std::fs::write(source.path().join("a.jpg"), b"ABCDE").expect("write");
    std::fs::write(source.path().join("b.cr3"), b"RAW").expect("write");
    std::fs::write(source.path().join("b.jpg"), b"JPEG").expect("write");
    for name in ["a.jpg", "b.cr3", "b.jpg"] {
        std::fs::File::options()
            .write(true)
            .open(source.path().join(name))
            .and_then(|file| file.set_modified(fixed_time()))
            .expect("set mtime");
    }
    let outcome = scan_source(
        &PathSource,
        &SourceBacking::Path(source.path().to_path_buf()),
        RawJpegPreference::Raw,
        &AtomicBool::new(false),
        &mut |_| {},
    )
    .expect("scan");
    let base = library.path().join("2026");
    let plan = ImportPlan {
        library_root: library.path().to_path_buf(),
        destination_base: base.clone(),
        photos: outcome.photos,
        preset: DestinationPreset::YearDay,
    };
    let progress = Arc::new(std::sync::Mutex::new(ImportProgress::default()));
    let status = run_import(
        &plan,
        &PathSource,
        Arc::clone(&progress),
        &AtomicBool::new(false),
    );
    let ImportStatus::Completed(report) = status else {
        panic!("expected completion, got {status:?}");
    };
    assert_eq!(report.copied_files, 3);
    assert_eq!(report.copied_bytes, 12);
    assert!(report.failed.is_empty());

    let day = expected_fixed_date();
    let directory = base.join(DestinationPreset::YearDay.directory_for(day));
    let contents = std::fs::read(directory.join("a.jpg")).expect("read");
    assert_eq!(contents, b"ABCDE");
    let raw_modified = std::fs::metadata(directory.join("b.cr3"))
        .expect("metadata")
        .modified()
        .expect("mtime");
    assert_eq!(raw_modified, fixed_time());
    assert_eq!(report.directories, vec![directory.clone()]);
    let shared = progress.lock().expect("progress");
    assert_eq!(shared.total_files, 3);
    assert_eq!(shared.total_bytes, 12);
    assert_eq!(shared.copied_files, 3);
    assert!(shared.current.is_empty());
}

#[test]
fn run_import_renames_on_collision() {
    let source = tempfile::tempdir().expect("source");
    let library = tempfile::tempdir().expect("library");
    std::fs::write(source.path().join("a.jpg"), b"new").expect("write");
    let directory = library.path().to_path_buf();
    std::fs::create_dir_all(&directory).expect("mkdir");
    std::fs::write(directory.join("a.jpg"), b"old").expect("write");

    let plan = ImportPlan {
        library_root: library.path().to_path_buf(),
        destination_base: library.path().to_path_buf(),
        photos: vec![file_photo("", source.path().join("a.jpg"), "a.jpg", 3)],
        preset: DestinationPreset::KeepFileNames,
    };
    let status = run_import(
        &plan,
        &PathSource,
        Arc::new(std::sync::Mutex::new(ImportProgress::default())),
        &AtomicBool::new(false),
    );
    let ImportStatus::Completed(report) = status else {
        panic!("expected completion, got {status:?}");
    };
    assert_eq!(report.renamed_files, 1);
    assert_eq!(
        std::fs::read(directory.join("a.jpg")).expect("read"),
        b"old"
    );
    assert_eq!(
        std::fs::read(directory.join("a (2).jpg")).expect("read"),
        b"new"
    );
}

#[test]
fn run_import_cancels_without_partial_files() {
    let source = tempfile::tempdir().expect("source");
    let library = tempfile::tempdir().expect("library");
    let payload = vec![9_u8; 3 * 1024 * 1024];
    std::fs::write(source.path().join("big.jpg"), &payload).expect("write");

    let plan = ImportPlan {
        library_root: library.path().to_path_buf(),
        destination_base: library.path().to_path_buf(),
        photos: vec![file_photo(
            "",
            source.path().join("big.jpg"),
            "big.jpg",
            payload.len() as u64,
        )],
        preset: DestinationPreset::KeepFileNames,
    };
    let cancel = Arc::new(AtomicBool::new(false));
    // Deterministic mid-copy cancellation: the access wrapper flips the flag
    // inside the first progress callback, so the second chunk aborts.
    let access = CancellingAfterFirstChunk {
        cancel: Arc::clone(&cancel),
    };
    let progress = Arc::new(std::sync::Mutex::new(ImportProgress::default()));
    let status = run_import(&plan, &access, progress, &cancel);
    let ImportStatus::Cancelled(report) = status else {
        panic!("expected cancellation, got {status:?}");
    };
    assert_eq!(report.copied_files, 0);
    let directory = library.path().to_path_buf();
    assert!(!directory.join("big.jpg").exists());
    assert!(cancel.load(Ordering::Acquire));
}

#[test]
fn run_import_records_per_file_failures() {
    let library = tempfile::tempdir().expect("library");
    // The source directory must outlive the import, so bind it outside the
    // photo list (a TempDir deletes its tree on drop).
    let source = tempfile::tempdir().expect("source");
    let ok_path = source.path().join("ok.jpg");
    std::fs::write(&ok_path, b"ok").expect("write");
    std::fs::File::options()
        .write(true)
        .open(&ok_path)
        .and_then(|file| file.set_modified(fixed_time()))
        .expect("set mtime");
    let plan = ImportPlan {
        library_root: library.path().to_path_buf(),
        destination_base: library.path().to_path_buf(),
        photos: vec![
            file_photo(
                "",
                PathBuf::from("/nonexistent/missing.jpg"),
                "missing.jpg",
                5,
            ),
            file_photo("", ok_path, "ok.jpg", 2),
        ],
        preset: DestinationPreset::KeepFileNames,
    };
    let progress = Arc::new(std::sync::Mutex::new(ImportProgress::default()));
    let status = run_import(&plan, &PathSource, progress, &AtomicBool::new(false));
    let ImportStatus::Completed(report) = status else {
        panic!("expected completion, got {status:?}");
    };
    assert_eq!(report.copied_files, 1);
    assert_eq!(report.failed.len(), 1);
    assert_eq!(report.failed[0].name, "missing.jpg");
    let directory = library.path().to_path_buf();
    assert_eq!(
        std::fs::read(directory.join("ok.jpg")).expect("read"),
        b"ok"
    );
}

#[test]
fn run_import_counter_naming_skips_existing_numbers() {
    let source = tempfile::tempdir().expect("source");
    let library = tempfile::tempdir().expect("library");
    std::fs::write(source.path().join("a.jpg"), b"one").expect("write");
    std::fs::write(source.path().join("b.jpg"), b"two").expect("write");
    std::fs::write(source.path().join("c.nef"), b"three").expect("write");
    // A pre-existing 0001.jpg must be skipped, not overwritten.
    std::fs::write(library.path().join("0001.jpg"), b"old").expect("write");

    let plan = ImportPlan {
        library_root: library.path().to_path_buf(),
        destination_base: library.path().to_path_buf(),
        photos: vec![
            file_photo("", source.path().join("a.jpg"), "a.jpg", 3),
            file_photo("", source.path().join("b.jpg"), "b.jpg", 3),
            file_photo("", source.path().join("c.nef"), "c.nef", 5),
        ],
        preset: DestinationPreset::Counter,
    };
    let status = run_import(
        &plan,
        &PathSource,
        Arc::new(std::sync::Mutex::new(ImportProgress::default())),
        &AtomicBool::new(false),
    );
    let ImportStatus::Completed(report) = status else {
        panic!("expected completion, got {status:?}");
    };
    assert_eq!(report.copied_files, 3);
    assert_eq!(report.renamed_files, 0);
    assert_eq!(
        std::fs::read(library.path().join("0001.jpg")).expect("read"),
        b"old"
    );
    assert_eq!(
        std::fs::read(library.path().join("0002.jpg")).expect("read"),
        b"one"
    );
    assert_eq!(
        std::fs::read(library.path().join("0003.jpg")).expect("read"),
        b"two"
    );
    assert_eq!(
        std::fs::read(library.path().join("0004.nef")).expect("read"),
        b"three"
    );
}

#[test]
fn run_import_keep_names_lands_in_destination_base() {
    let source = tempfile::tempdir().expect("source");
    let library = tempfile::tempdir().expect("library");
    let base = library.path().join("2026").join("Xyz vacation");
    std::fs::create_dir_all(&base).expect("mkdir");
    std::fs::write(source.path().join("IMG_1.jpg"), b"x").expect("write");

    let plan = ImportPlan {
        library_root: library.path().to_path_buf(),
        destination_base: base.clone(),
        photos: vec![file_photo(
            "",
            source.path().join("IMG_1.jpg"),
            "IMG_1.jpg",
            1,
        )],
        preset: DestinationPreset::KeepFileNames,
    };
    let status = run_import(
        &plan,
        &PathSource,
        Arc::new(std::sync::Mutex::new(ImportProgress::default())),
        &AtomicBool::new(false),
    );
    let ImportStatus::Completed(report) = status else {
        panic!("expected completion, got {status:?}");
    };
    assert_eq!(report.copied_files, 1);
    assert_eq!(report.directories, vec![base.clone()]);
    assert_eq!(std::fs::read(base.join("IMG_1.jpg")).expect("read"), b"x");
}

#[test]
fn registry_access_for_path_sources() {
    let registry = SourceRegistry::new();
    assert!(
        registry
            .access_for(&SourceBacking::Path(PathBuf::from("/tmp")))
            .is_some()
    );
    // Device backings resolve through the platform MTP backend, which only
    // exists on unix; the Windows stub has no registry entry.
    #[cfg(unix)]
    assert!(
        registry
            .access_for(&SourceBacking::Mtp(crate::sources::mtp::MtpObjectRef {
                device_id: Arc::from("d"),
                storage_id: Arc::from("s"),
                object: Arc::from("0"),
            }))
            .is_some()
    );
}

#[test]
fn pairing_keeps_extra_same_stem_files() {
    let root = tempfile::tempdir().expect("tempdir");
    std::fs::write(root.path().join("P.CR3"), b"raw").expect("write");
    std::fs::write(root.path().join("P.JPG"), b"jpg").expect("write");
    std::fs::write(root.path().join("P.HEIC"), b"heic").expect("write");
    let outcome = scan_source(
        &PathSource,
        &SourceBacking::Path(root.path().to_path_buf()),
        RawJpegPreference::Raw,
        &AtomicBool::new(false),
        &mut |_| {},
    )
    .expect("scan");
    assert_eq!(outcome.photos.len(), 2);
    assert_eq!(outcome.photos[0].primary.name, "P.CR3");
    assert_eq!(
        outcome.photos[0]
            .companion
            .as_ref()
            .expect("companion")
            .name,
        "P.JPG"
    );
    assert_eq!(outcome.photos[1].primary.name, "P.HEIC");
    assert!(outcome.photos[1].companion.is_none());
    assert!(!outcome.truncated);
    assert_eq!(outcome.photos[0].key(), "P.CR3");
}

#[test]
fn scan_outcome_reports_truncation_flag_shape() {
    // The flag is false for a normal tree; the budget itself is a safety
    // valve for pathological sources and is not exercised here.
    let root = tempfile::tempdir().expect("tempdir");
    std::fs::write(root.path().join("a.jpg"), b"a").expect("write");
    let outcome = scan_source(
        &PathSource,
        &SourceBacking::Path(root.path().to_path_buf()),
        RawJpegPreference::Raw,
        &AtomicBool::new(false),
        &mut |_| {},
    )
    .expect("scan");
    assert!(!outcome.truncated);
    assert_eq!(outcome.skipped_directories, 0);
}

#[test]
fn sanitize_names_are_safe_path_components() {
    use super::sanitize_name;
    assert_eq!(sanitize_name("DSC_0001.NEF"), "DSC_0001.NEF");
    assert_eq!(
        sanitize_name("a/b\\c:d<e>f\"g|h?i*j"),
        "a_b_c_d_e_f_g_h_i_j"
    );
    assert_eq!(sanitize_name("name."), "name");
    assert_eq!(sanitize_name("name "), "name");
    assert_eq!(sanitize_name(""), "_");
}

/// A device-style source: rejects path backings exactly like the MTP/PTP
/// backends do. Regression cover for staged imports — staged device files
/// carry path backings and must be read through the path source even
/// though the import runs with the device's source access.
struct DeviceOnlySource;

impl SourceAccess for DeviceOnlySource {
    fn list_dir(&self, _dir: &SourceBacking) -> std::io::Result<Vec<SourceDirEntry>> {
        Err(std::io::Error::other("device source cannot list in tests"))
    }

    fn read_file(
        &self,
        file: &SourceBacking,
        _out: &mut dyn std::io::Write,
        _cancel: &AtomicBool,
        _progress: &mut dyn FnMut(u64),
    ) -> std::io::Result<u64> {
        match file {
            SourceBacking::Path(_) => {
                Err(std::io::Error::other("camera source cannot address paths"))
            }
            _ => Err(std::io::Error::other("no device in tests")),
        }
    }
}

#[test]
fn staged_copies_import_through_the_path_source() {
    use crate::sources::staged_import_access;
    let source = tempfile::tempdir().expect("source");
    let library = tempfile::tempdir().expect("library");
    std::fs::write(source.path().join("IMG_0001.JPG"), b"jpeg").expect("write");

    let plan = ImportPlan {
        library_root: library.path().to_path_buf(),
        destination_base: library.path().to_path_buf(),
        photos: vec![file_photo(
            "",
            source.path().join("IMG_0001.JPG"),
            "IMG_0001.JPG",
            4,
        )],
        preset: DestinationPreset::KeepFileNames,
    };
    let progress = Arc::new(std::sync::Mutex::new(ImportProgress::default()));
    // The import view wraps the source access exactly like this once
    // preview staging has swapped device backings for cache paths.
    let access = staged_import_access(Arc::new(DeviceOnlySource));
    let status = run_import(
        &plan,
        access.as_ref(),
        Arc::clone(&progress),
        &AtomicBool::new(false),
    );
    let ImportStatus::Completed(report) = status else {
        panic!("import must complete");
    };
    assert!(
        report.failed.is_empty(),
        "staged files must import via the path source: {:?}",
        report.failed
    );
    assert_eq!(report.copied_files, 1);
}
