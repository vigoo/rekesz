//! Mounted volume discovery backed by sysinfo, with platform-agnostic
//! classification of which volumes are user-visible import sources.
//!
//! sysinfo does not report optical or network drives on Windows, so those
//! classifications are best-effort and exercised by the other platforms.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use super::{SourceBacking, SourceCapacity, SourceEntry, SourceId, SourceKind};

/// A volume as reported by the platform, normalized for classification.
pub(crate) struct RawVolume {
    pub mount_point: PathBuf,
    pub name: String,
    pub file_system: String,
    pub total: u64,
    pub available: u64,
    pub removable: bool,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) enum VolumeClass {
    Removable,
    System,
    Fixed,
    Network,
    Ignored,
}

fn is_pseudo_file_system(file_system: &str) -> bool {
    matches!(
        file_system,
        "proc"
            | "procfs"
            | "sysfs"
            | "devtmpfs"
            | "devfs"
            | "devpts"
            | "tmpfs"
            | "mfs"
            | "squashfs"
            | "overlay"
            | "ramfs"
            | "cgroup"
            | "cgroup2"
            | "cgroupfs"
            | "securityfs"
            | "pstore"
            | "bpf"
            | "debugfs"
            | "tracefs"
            | "fusectl"
            | "configfs"
            | "mqueue"
            | "hugetlbfs"
            | "binfmt_misc"
            | "efivarfs"
            | "nsfs"
            | "rpc_pipefs"
            | "autofs"
            | "fdesc"
    )
}

fn is_network_file_system(file_system: &str) -> bool {
    matches!(
        file_system,
        "nfs"
            | "nfs4"
            | "cifs"
            | "smbfs"
            | "smb2"
            | "sshfs"
            | "9p"
            | "ncpfs"
            | "afp"
            | "afpfs"
            | "davfs"
            | "webdav"
    ) || file_system.starts_with("fuse.sshfs")
        || file_system.starts_with("fuse.rclone")
}

/// Mount prefixes that never hold user import sources. The rules are applied
/// on every platform: the prefixes only exist where they mean something, and
/// keeping one list makes classification testable everywhere. `/run/media` is
/// excluded because udisks2 mounts removable volumes there.
fn is_ignored_mount(mount: &Path) -> bool {
    if mount.starts_with("/run/media") {
        return false;
    }
    [
        "/proc",
        "/sys",
        "/dev",
        "/run",
        "/boot",
        "/efi",
        "/snap",
        "/var/lib",
        "/var/snap",
        "/System",
        "/private/var",
        "/private/tmp",
        "/private/etc",
        "/net",
    ]
    .iter()
    .any(|prefix| mount.starts_with(prefix))
}

fn is_optical_file_system(file_system: &str) -> bool {
    matches!(file_system, "iso9660" | "udf" | "cdda" | "cd9660" | "cdfs")
}

/// System mounts: the POSIX root, and the Windows drive the OS boots from.
pub(crate) fn is_system_mount(mount: &Path, windows_system_drive: Option<&str>) -> bool {
    if mount == Path::new("/") {
        return true;
    }
    let Some(system_drive) = windows_system_drive else {
        return false;
    };
    let lossy = mount.to_string_lossy();
    lossy.eq_ignore_ascii_case(system_drive)
        || lossy
            .strip_suffix('\\')
            .is_some_and(|drive| drive.eq_ignore_ascii_case(system_drive))
}

#[cfg(target_os = "windows")]
fn windows_system_drive() -> Option<String> {
    std::env::var_os("SystemDrive").map(|drive| drive.to_string_lossy().into_owned())
}

#[cfg(not(target_os = "windows"))]
fn windows_system_drive() -> Option<String> {
    None
}

pub(crate) fn classify(volume: &RawVolume) -> VolumeClass {
    let file_system = volume.file_system.to_ascii_lowercase();
    if is_pseudo_file_system(&file_system) || is_ignored_mount(&volume.mount_point) {
        return VolumeClass::Ignored;
    }
    if is_network_file_system(&file_system) {
        return VolumeClass::Network;
    }
    let mounts_at_removable_media = volume.mount_point.starts_with("/media")
        || volume.mount_point.starts_with("/run/media")
        || volume.mount_point.starts_with("/Volumes");
    if volume.removable || mounts_at_removable_media || is_optical_file_system(&file_system) {
        return VolumeClass::Removable;
    }
    if is_system_mount(&volume.mount_point, windows_system_drive().as_deref()) {
        return VolumeClass::System;
    }
    VolumeClass::Fixed
}

pub(crate) fn volume_label(volume: &RawVolume) -> String {
    // Linux reports device paths ("/dev/sdb1") as the disk name; the mount
    // point carries the useful information there.
    let name = volume.name.trim();
    if !name.is_empty() && !name.starts_with("/dev/") {
        return name.to_owned();
    }
    let mount = &volume.mount_point;
    if mount == Path::new("/") {
        return "System".to_owned();
    }
    let lossy = mount.to_string_lossy();
    // Windows drive roots ("C:\") have no file name component.
    if lossy.len() == 3 && lossy.as_bytes()[1] == b':' {
        return format!("{}:", &lossy[..1]);
    }
    mount
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| lossy.into_owned())
}

/// Build the volume source entries, dropping pseudo mounts and deduplicating
/// repeated mount points, sorted removable-first then by label.
pub(crate) fn entries_from_volumes(volumes: &[RawVolume]) -> Vec<SourceEntry> {
    let mut seen = HashSet::new();
    let mut entries = Vec::new();
    for volume in volumes {
        let class = classify(volume);
        if class == VolumeClass::Ignored {
            continue;
        }
        if !seen.insert(volume.mount_point.clone()) {
            continue;
        }
        entries.push((
            class,
            SourceEntry {
                id: SourceId::new(format!("vol:{}", volume.mount_point.display())),
                label: volume_label(volume),
                kind: SourceKind::Volume {
                    removable: class == VolumeClass::Removable,
                },
                backing: SourceBacking::Path(volume.mount_point.clone()),
                capacity: Some(SourceCapacity {
                    total: volume.total,
                    available: volume.available,
                }),
            },
        ));
    }
    entries.sort_by(|(class_a, entry_a), (class_b, entry_b)| {
        class_a.cmp(class_b).then_with(|| {
            entry_a
                .label
                .to_lowercase()
                .cmp(&entry_b.label.to_lowercase())
        })
    });
    entries.into_iter().map(|(_, entry)| entry).collect()
}

pub(crate) struct VolumeProvider;

impl VolumeProvider {
    pub(crate) fn entries_from(volumes: &[RawVolume]) -> Vec<SourceEntry> {
        entries_from_volumes(volumes)
    }

    pub(crate) fn raw_volumes() -> Vec<RawVolume> {
        let disks = sysinfo::Disks::new_with_refreshed_list();
        disks
            .list()
            .iter()
            .map(|disk| RawVolume {
                mount_point: disk.mount_point().to_path_buf(),
                name: disk.name().to_string_lossy().into_owned(),
                file_system: disk.file_system().to_string_lossy().into_owned(),
                total: disk.total_space(),
                available: disk.available_space(),
                removable: disk.is_removable(),
            })
            .collect()
    }
}
