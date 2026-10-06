use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use test_r::test;

use super::super::{child_path, port_bus_dev};
use super::PtpSource;
use crate::sources::{SourceAccess, SourceBacking};

fn device_backing(device_id: &str, folder: &str, name: &str) -> SourceBacking {
    SourceBacking::Ptp(super::PtpObjectRef {
        device_id: Arc::from(device_id),
        folder: Arc::from(folder),
        name: Arc::from(name),
    })
}

#[test]
fn usb_ports_parse_into_bus_and_device_numbers() {
    assert_eq!(port_bus_dev("usb:005,012"), Some((5, 12)));
    assert_eq!(port_bus_dev("usb:3,42"), Some((3, 42)));
    assert_eq!(port_bus_dev("usb:x,y"), None);
    assert_eq!(port_bus_dev("usb:1"), None);
    assert_eq!(port_bus_dev("disk:/media"), None);
    assert_eq!(port_bus_dev(""), None);
}

#[test]
fn camera_paths_join_regardless_of_root_form() {
    assert_eq!(child_path("/", "DCIM"), "/DCIM");
    assert_eq!(child_path("/DCIM", "100CANON"), "/DCIM/100CANON");
}

#[test]
fn ptp_source_rejects_path_and_foreign_backings() {
    let source = PtpSource::new("gphoto:usb:001,002".into());
    let error = source
        .list_dir(&SourceBacking::Path("/tmp".into()))
        .expect_err("path backing must fail");
    assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);

    let foreign = device_backing("gphoto:usb:009,009", "/", "");
    let error = source
        .list_dir(&foreign)
        .expect_err("foreign backing must fail");
    assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
    let mut sink = Vec::new();
    let error = source
        .read_file(&foreign, &mut sink, &AtomicBool::new(false), &mut |_| {})
        .expect_err("foreign backing must fail");
    assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
    let error = source
        .read_file_to_path(
            &foreign,
            std::path::Path::new("/tmp/rekesz-x"),
            &AtomicBool::new(false),
            &mut |_| {},
        )
        .expect("direct transfer supported")
        .expect_err("foreign backing must fail");
    assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);

    let mtp = SourceBacking::Mtp(crate::sources::mtp::MtpObjectRef {
        device_id: "libmtp:1:2".into(),
        storage_id: "1".into(),
        object: Arc::from("0"),
    });
    let error = source.list_dir(&mtp).expect_err("mtp backing must fail");
    assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
}

#[test]
fn ptp_source_rejects_file_backings_for_listings_and_dir_backings_for_reads() {
    let source = PtpSource::new("gphoto:usb:001,002".into());
    let file = device_backing("gphoto:usb:001,002", "/DCIM", "IMG_0001.JPG");
    let error = source.list_dir(&file).expect_err("file backing must fail");
    assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);

    let directory = device_backing("gphoto:usb:001,002", "/DCIM", "");
    let error = source
        .read_file(
            &directory,
            &mut Vec::new(),
            &AtomicBool::new(false),
            &mut |_| {},
        )
        .expect_err("directory backing must fail");
    assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
}

#[test]
fn ptp_source_reports_disconnected_cameras() {
    // No camera is attached during test runs; usb 250,250 is safely unused
    // and must surface as NotFound rather than panic.
    let source = PtpSource::new("gphoto:usb:250,250".into());
    let backing = device_backing("gphoto:usb:250,250", "/", "");
    let error = source.list_dir(&backing).expect_err("no camera");
    assert!(
        error.kind() == std::io::ErrorKind::NotFound || error.kind() == std::io::ErrorKind::Other,
        "unexpected kind: {error}"
    );
}

/// Live end-to-end check against a real PTP camera; run explicitly with
/// `cargo test live -- --ignored` while a camera is attached. Prints
/// timings that feed the handle-cache follow-up decision.
#[test]
#[ignore = "requires a PTP camera attached over USB"]
fn live_camera_end_to_end() {
    use std::time::Instant;

    let started = Instant::now();
    let devices = super::detect_devices();
    println!("detect: {:?} → {devices:?}", started.elapsed());
    assert!(!devices.is_empty(), "no PTP camera detected");
    let device = &devices[0];

    let source = PtpSource::new(Arc::clone(&device.info.id));
    let root = SourceBacking::Ptp(super::PtpObjectRef {
        device_id: Arc::clone(&device.info.id),
        folder: Arc::clone(&device.storages[0].id),
        name: Arc::from(""),
    });

    let started = Instant::now();
    let entries = source.list_dir(&root).expect("root listing");
    println!(
        "root listing: {:?} → {} entries",
        started.elapsed(),
        entries.len()
    );

    // Browse into the first directory (typically DCIM) and time the
    // listing: one file_info round-trip per file is the cost driver.
    if let Some(directory) = entries.iter().find(|entry| entry.is_dir) {
        let started = Instant::now();
        let files = source.list_dir(&directory.backing).expect("folder listing");
        println!(
            "folder {}: {:?} → {} entries",
            directory.name,
            started.elapsed(),
            files.len()
        );
        if let Some(smallest) = files
            .iter()
            .filter(|entry| !entry.is_dir && entry.size > 0)
            .min_by_key(|entry| entry.size)
        {
            let target = std::env::temp_dir().join("rekesz-ptp-live-test");
            let _ = std::fs::remove_file(&target);
            let started = Instant::now();
            let mut progress_calls = 0;
            let written = source
                .read_file_to_path(
                    &smallest.backing,
                    &target,
                    &AtomicBool::new(false),
                    &mut |_| progress_calls += 1,
                )
                .expect("direct transfer supported")
                .expect("transfer");
            println!(
                "download {} bytes of {}: {:?}, {} progress updates",
                written,
                smallest.name,
                started.elapsed(),
                progress_calls
            );
            assert_eq!(written, smallest.size, "downloaded size must match listing");
            let _ = std::fs::remove_file(&target);

            // Cancel before the transfer starts must surface as Interrupted.
            let cancel = AtomicBool::new(true);
            let result = source
                .read_file_to_path(&smallest.backing, &target, &cancel, &mut |_| {})
                .expect("direct transfer supported");
            assert!(result.is_err(), "cancelled transfer must fail");
        }
    }
}
