use std::sync::Arc;

use test_r::test;

use super::{MtpSource, parse_device_id};
use crate::sources::{PathSource, SourceAccess, SourceBacking};

#[test]
fn device_ids_round_trip_through_the_parser() {
    assert_eq!(parse_device_id("libmtp:3:42"), Some((3, 42)));
    assert_eq!(parse_device_id("libmtp:0:0"), Some((0, 0)));
    assert_eq!(parse_device_id("vol:/media/x"), None);
    assert_eq!(parse_device_id("libmtp:x:y"), None);
    assert_eq!(parse_device_id("libmtp:3"), None);
}

#[test]
fn mtp_source_rejects_path_backings() {
    let source = MtpSource::new("libmtp:1:2".into());
    let error = source
        .list_dir(&SourceBacking::Path("/tmp".into()))
        .expect_err("path backing must fail");
    assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
}

#[test]
fn mtp_source_reports_disconnected_devices() {
    // No device is attached during test runs; bus 250 dev 250 is safely
    // unused and must surface as NotFound rather than panic.
    let source = MtpSource::new("libmtp:250:250".into());
    let backing = SourceBacking::Mtp(crate::sources::mtp::MtpObjectRef {
        device_id: "libmtp:250:250".into(),
        storage_id: "1".into(),
        object: Arc::from("0"),
    });
    let error = source.list_dir(&backing).expect_err("no device");
    // Environments without USB access fail detection differently; only the
    // input-validation kinds are wrong here.
    assert!(
        error.kind() == std::io::ErrorKind::NotFound
            || error.kind() == std::io::ErrorKind::Other
            || error.kind() == std::io::ErrorKind::Unsupported,
        "unexpected kind: {error}"
    );
}

#[test]
fn mtp_source_rejects_foreign_device_backings() {
    let source = MtpSource::new("libmtp:250:250".into());
    let foreign = SourceBacking::Mtp(crate::sources::mtp::MtpObjectRef {
        device_id: "libmtp:1:2".into(),
        storage_id: "1".into(),
        object: Arc::from("0"),
    });
    let error = source.list_dir(&foreign).expect_err("foreign backing");
    assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
    let mut sink = Vec::new();
    let error = source
        .read_file(
            &foreign,
            &mut sink,
            &std::sync::atomic::AtomicBool::new(false),
            &mut |_| {},
        )
        .expect_err("foreign backing");
    assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
    // Path backings are rejected before any device lookup.
    let error = PathSource
        .list_dir(&SourceBacking::Mtp(crate::sources::mtp::MtpObjectRef {
            device_id: "libmtp:1:2".into(),
            storage_id: "1".into(),
            object: Arc::from("0"),
        }))
        .expect_err("path source");
    assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
}
