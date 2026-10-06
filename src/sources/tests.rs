use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use test_r::test;

use super::PathSource;
use super::SourceAccess;
use super::SourceBacking;
use super::SourceEntry;
use super::SourceId;
use super::SourceKind;
use super::mtp::MtpObjectRef;
use super::volumes::{RawVolume, VolumeClass, classify, is_system_mount, volume_label};

fn volume(mount_point: &str, file_system: &str, removable: bool, name: &str) -> RawVolume {
    RawVolume {
        mount_point: PathBuf::from(mount_point),
        name: name.to_owned(),
        file_system: file_system.to_owned(),
        total: 64 * 1024 * 1024 * 1024,
        available: 16 * 1024 * 1024 * 1024,
        removable,
    }
}

#[test]
fn pseudo_file_systems_are_ignored() {
    assert_eq!(
        classify(&volume("/proc", "proc", false, "")),
        VolumeClass::Ignored
    );
    assert_eq!(
        classify(&volume("/run", "tmpfs", false, "")),
        VolumeClass::Ignored
    );
    assert_eq!(
        classify(&volume("/snap/core22", "squashfs", false, "")),
        VolumeClass::Ignored
    );
}

#[test]
fn gvfs_mounts_are_ignored_but_udisks_media_is_removable() {
    assert_eq!(
        classify(&volume(
            "/run/user/1000/gvfs/mtp:host=%5Busb%3A001%2C006%5D",
            "fuse.gvfsd-fuse",
            false,
            ""
        )),
        VolumeClass::Ignored
    );
    assert_eq!(
        classify(&volume("/run/media/daniel/STICK", "vfat", false, "STICK")),
        VolumeClass::Removable
    );
    assert_eq!(
        classify(&volume("/media/daniel/DISK", "ntfs", false, "")),
        VolumeClass::Removable
    );
}

#[test]
fn prefix_lookalikes_do_not_change_classification() {
    // "/runmedia" must not trip the "/run" ignore rule, and "/media2" must
    // not count as a removable-media mount point.
    assert_eq!(
        classify(&volume("/runmedia", "ext4", false, "")),
        VolumeClass::Fixed
    );
    assert_eq!(
        classify(&volume("/media2/disk", "ext4", false, "")),
        VolumeClass::Fixed
    );
}

#[test]
fn sysinfo_removable_flag_is_honored() {
    assert_eq!(
        classify(&volume("/mnt/stick", "exfat", true, "")),
        VolumeClass::Removable
    );
}

#[test]
fn optical_file_systems_are_removable() {
    assert_eq!(
        classify(&volume("/mnt/cdrom", "iso9660", false, "")),
        VolumeClass::Removable
    );
    assert_eq!(
        classify(&volume("/mnt/bluray", "udf", false, "")),
        VolumeClass::Removable
    );
}

#[test]
fn root_mount_is_the_system_volume() {
    assert_eq!(
        classify(&volume("/", "ext4", false, "")),
        VolumeClass::System
    );
}

#[test]
fn windows_system_drive_is_the_system_volume() {
    assert!(is_system_mount(Path::new("C:\\"), Some("C:")));
    assert!(is_system_mount(Path::new("c:\\"), Some("C:")));
    assert!(!is_system_mount(Path::new("D:\\"), Some("C:")));
    assert!(!is_system_mount(Path::new("D:\\"), None));
    assert!(is_system_mount(Path::new("/"), None));
}

#[test]
fn network_file_systems_are_network_sources() {
    assert_eq!(
        classify(&volume("/mnt/nas", "nfs4", false, "")),
        VolumeClass::Network
    );
    assert_eq!(
        classify(&volume("/mnt/share", "cifs", false, "")),
        VolumeClass::Network
    );
}

#[test]
fn ordinary_mounts_are_fixed_volumes() {
    assert_eq!(
        classify(&volume("/mnt/data", "ext4", false, "")),
        VolumeClass::Fixed
    );
    assert_eq!(
        classify(&volume("/home", "btrfs", false, "")),
        VolumeClass::Fixed
    );
}

#[test]
fn macos_volumes_are_removable_and_system_mounts_ignored() {
    assert_eq!(
        classify(&volume("/Volumes/SD CARD", "exfat", false, "SD CARD")),
        VolumeClass::Removable
    );
    assert_eq!(
        classify(&volume("/System/Volumes/Data", "apfs", false, "")),
        VolumeClass::Ignored
    );
}

#[test]
fn labels_fall_back_to_mount_names() {
    let mut volumes = vec![
        volume("/", "ext4", false, "/dev/nvme0n1p2"),
        volume("/media/daniel/KINGSTON", "vfat", true, "/dev/sdb1"),
    ];
    // Duplicated mount points collapse to a single entry.
    volumes.push(volume("/media/daniel/KINGSTON", "vfat", true, "/dev/sdb1"));
    let entries = super::volumes::entries_from_volumes(&volumes);
    assert_eq!(entries.len(), 2);
    // Device paths are never shown as labels.
    assert_eq!(entries[0].label, "KINGSTON");
    assert_eq!(entries[0].id, SourceId::new("vol:/media/daniel/KINGSTON"));
    assert_eq!(entries[1].label, "System");
    let kinds = entries
        .iter()
        .map(|entry| entry.kind.clone())
        .collect::<Vec<_>>();
    assert!(matches!(&kinds[0], SourceKind::Volume { removable: true }));
    assert!(matches!(&kinds[1], SourceKind::Volume { removable: false }));
}

#[test]
fn windows_drive_roots_get_drive_letter_labels() {
    let labeled = volume("C:\\", "NTFS", false, "Windows");
    assert_eq!(volume_label(&labeled), "Windows");
    let unlabeled = volume("E:\\", "exfat", true, "");
    assert_eq!(volume_label(&unlabeled), "E:");
}

#[test]
fn entries_sort_removable_first() {
    let volumes = vec![
        volume("/mnt/data", "ext4", false, "zeta"),
        volume("/media/stick", "vfat", true, "alpha"),
        volume("/mnt/nas", "nfs4", false, "network"),
    ];
    let entries = super::volumes::entries_from_volumes(&volumes);
    let labels = entries
        .iter()
        .map(|entry| entry.label.as_str())
        .collect::<Vec<_>>();
    // Removable volumes first, then fixed internal volumes, then network mounts.
    assert_eq!(labels, vec!["alpha", "zeta", "network"]);
}

#[test]
fn folders_are_appended_and_deduplicated_against_volumes() {
    // The system root volume must not swallow user folders; a removable
    // volume entry covers its own subfolders.
    let volumes = vec![
        volume("/", "ext4", false, "System"),
        volume("/media/daniel/STICK", "vfat", true, "STICK"),
    ];
    let volume_entries = super::volumes::entries_from_volumes(&volumes);
    assert_eq!(volume_entries.len(), 2);
    let merged = super::merge_folder_entries(
        volume_entries,
        &[
            PathBuf::from("/media/daniel/STICK/DCIM"),
            PathBuf::from("/home/daniel/pictures"),
            PathBuf::from("/home/daniel/pictures"),
        ],
        &volumes,
    );
    assert_eq!(merged.len(), 3);
    assert_eq!(merged[0].label, "STICK");
    assert_eq!(merged[1].label, "System");
    assert_eq!(merged[2].kind, SourceKind::Folder);
    assert_eq!(merged[2].label, "pictures");
    assert_eq!(
        merged[2].backing,
        SourceBacking::Path(PathBuf::from("/home/daniel/pictures"))
    );
    assert_eq!(
        merged[2].capacity.map(|capacity| capacity.total),
        Some(64 * 1024 * 1024 * 1024)
    );
}

#[test]
fn folders_containing_mounts_are_kept() {
    // A fixed mount below a folder does not make the folder redundant: the
    // rest of the folder's contents must stay reachable.
    let volumes = vec![volume("/home/user/usb", "vfat", true, "USB")];
    let entries = super::merge_folder_entries(
        super::volumes::entries_from_volumes(&volumes),
        &[PathBuf::from("/home/user")],
        &volumes,
    );
    assert_eq!(entries.len(), 2);
    assert_eq!(entries[1].label, "user");
}

#[test]
fn path_source_lists_directories_and_files_sorted() {
    let root = tempfile::tempdir().expect("tempdir");
    std::fs::create_dir(root.path().join("DCIM")).expect("mkdir");
    std::fs::create_dir(root.path().join("backup")).expect("mkdir");
    std::fs::write(root.path().join("b.raw"), b"raw").expect("write");
    std::fs::write(root.path().join("a.jpg"), b"jpeg-data").expect("write");
    std::fs::write(root.path().join("notes.txt"), b"skip").expect("write");

    let entries = PathSource
        .list_dir(&SourceBacking::Path(root.path().to_path_buf()))
        .expect("list");
    let names = entries
        .iter()
        .map(|entry| (entry.name.as_str(), entry.is_dir))
        .collect::<Vec<_>>();
    // Dirs first, each group case-insensitively ordered (see SourceAccess).
    assert_eq!(
        names,
        vec![
            ("backup", true),
            ("DCIM", true),
            ("a.jpg", false),
            ("b.raw", false),
            ("notes.txt", false),
        ]
    );
    let file = &entries[2];
    assert_eq!(file.name, "a.jpg");
    assert_eq!(file.size, 9);
    assert!(file.modified.is_some());
    assert_eq!(file.backing, SourceBacking::Path(root.path().join("a.jpg")));
}

#[test]
fn path_source_resolves_symlinked_directories() {
    let root = tempfile::tempdir().expect("tempdir");
    let target = tempfile::tempdir().expect("target");
    std::fs::create_dir(target.path().join("inner")).expect("mkdir");
    std::fs::write(target.path().join("inner").join("x.jpg"), b"x").expect("write");
    #[cfg(unix)]
    std::os::unix::fs::symlink(target.path(), root.path().join("link")).expect("symlink");
    #[cfg(windows)]
    std::os::windows::fs::symlink_dir(target.path(), root.path().join("link")).expect("symlink");

    let entries = PathSource
        .list_dir(&SourceBacking::Path(root.path().to_path_buf()))
        .expect("list");
    assert_eq!(entries.len(), 1);
    assert!(entries[0].is_dir);
    assert_eq!(entries[0].name, "link");
}

#[test]
fn path_source_reads_files_with_progress_and_cancellation() {
    let root = tempfile::tempdir().expect("tempdir");
    let payload = vec![7_u8; 3 * 1024 * 1024 + 17];
    std::fs::write(root.path().join("big.bin"), &payload).expect("write");

    let mut received = Vec::new();
    let mut last_progress = 0;
    let total = PathSource
        .read_file(
            &SourceBacking::Path(root.path().join("big.bin")),
            &mut received,
            &AtomicBool::new(false),
            &mut |copied| last_progress = copied as usize,
        )
        .expect("read");
    assert_eq!(total as usize, payload.len());
    assert_eq!(received.len(), payload.len());
    assert!(received.iter().all(|byte| *byte == 7));
    assert_eq!(last_progress, payload.len());

    let cancel = AtomicBool::new(true);
    let mut sink = Vec::new();
    let error = PathSource
        .read_file(
            &SourceBacking::Path(root.path().join("big.bin")),
            &mut sink,
            &cancel,
            &mut |_| {},
        )
        .expect_err("cancelled read must fail");
    assert_eq!(error.kind(), std::io::ErrorKind::Interrupted);
}

#[test]
fn path_source_rejects_mtp_backings() {
    let mtp = SourceBacking::Mtp(MtpObjectRef {
        device_id: Arc::from("d"),
        storage_id: Arc::from("s"),
        object: Arc::from("0"),
    });
    assert!(PathSource.list_dir(&mtp).is_err());
    let mut sink = Vec::new();
    assert!(
        PathSource
            .read_file(&mtp, &mut sink, &AtomicBool::new(false), &mut |_| {})
            .is_err()
    );
    // Path sources resolve through PathSource; device backings resolve
    // through the platform MTP backend.
    assert!(
        super::SourceRegistry::new()
            .access_for(&SourceBacking::Path(PathBuf::from("/")))
            .is_some()
    );
}

#[test]
fn ptp_entries_aliasing_mtp_devices_are_deduped() {
    use crate::sources::{SourceEntry, SourceId, SourceKind, dedup_device_entries};
    use std::path::PathBuf;

    let entry = |id: &str, backing: crate::sources::SourceBacking| SourceEntry {
        id: SourceId::new(id),
        label: id.to_owned(),
        kind: SourceKind::Device {
            vendor: String::new(),
            model: "Camera".to_owned(),
        },
        backing,
        capacity: None,
    };

    let mtp = crate::sources::SourceBacking::Mtp(MtpObjectRef {
        device_id: Arc::from("libmtp:5:12"),
        storage_id: Arc::from("1"),
        object: Arc::from("0"),
    });
    // usb:005,012 is the same physical device as libmtp:5:12.
    let ptp_alias = crate::sources::ptp::PtpObjectRef {
        device_id: Arc::from("gphoto:usb:005,012"),
        folder: Arc::from("/"),
        name: Arc::from(""),
    };
    let ptp_other = crate::sources::ptp::PtpObjectRef {
        device_id: Arc::from("gphoto:usb:007,001"),
        folder: Arc::from("/"),
        name: Arc::from(""),
    };

    let deduped = dedup_device_entries(vec![
        entry("mtp", mtp),
        entry("ptp-alias", crate::sources::SourceBacking::Ptp(ptp_alias)),
        entry(
            "path",
            crate::sources::SourceBacking::Path(PathBuf::from("/media/card")),
        ),
        entry("ptp-other", crate::sources::SourceBacking::Ptp(ptp_other)),
    ]);
    assert_eq!(deduped.len(), 3);
    assert_eq!(deduped[0].id, SourceId::new("mtp"));
    assert_eq!(deduped[1].id, SourceId::new("path"));
    assert_eq!(deduped[2].id, SourceId::new("ptp-other"));
}
