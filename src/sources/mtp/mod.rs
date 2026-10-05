//! MTP device support (phones, media players, many cameras). Detection and
//! file access come from platform backends: libmtp on Linux/macOS and the
//! Windows Portable Devices API on Windows; both are hidden behind
//! [`backend`].

use std::sync::Arc;

use super::{SourceBacking, SourceCapacity, SourceEntry, SourceId, SourceKind};

/// Identifies one object (file or folder) on an MTP device storage.
/// `object` is the backend's native object id as a string; the empty string
/// denotes the root directory of the storage.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct MtpObjectRef {
    pub device_id: Arc<str>,
    pub storage_id: Arc<str>,
    pub object: Arc<str>,
}

/// Static description of a connected device.
///
/// `id` must be stable for as long as the device stays attached so that
/// source ids — and with them the UI's selection — survive a source-list
/// refresh. libmtp uses the USB bus/device pair; WPD uses its own device id.
#[derive(Clone, Debug)]
pub(crate) struct DeviceInfo {
    pub id: Arc<str>,
    pub vendor: String,
    pub model: String,
}

/// One storage area of a device (internal memory, SD card, …).
#[derive(Clone, Debug)]
pub(crate) struct StorageInfo {
    pub id: Arc<str>,
    pub label: String,
    pub capacity: u64,
    pub available: u64,
}

#[derive(Clone)]
pub(crate) struct DetectedDevice {
    pub info: DeviceInfo,
    pub storages: Vec<StorageInfo>,
}

#[cfg(all(unix, feature = "mtp"))]
pub(crate) mod libmtp_backend;
#[cfg(windows)]
pub(crate) mod wpd_backend;

/// Platform backend contract: enumerate connected devices with their
/// storages. Session-based access (listing and reading objects) is defined by
/// the backend types themselves and reached through `super::SourceAccess`
/// implementations.
#[cfg(all(unix, feature = "mtp"))]
pub(crate) use libmtp_backend as backend;
#[cfg(windows)]
pub(crate) use wpd_backend as backend;

#[cfg(not(any(all(unix, feature = "mtp"), windows)))]
pub(crate) mod backend {
    use super::DetectedDevice;
    use crate::sources::{SourceAccess, SourceBacking};
    use std::sync::{Arc, atomic::AtomicBool};

    pub(crate) fn detect_devices() -> Vec<DetectedDevice> {
        Vec::new()
    }

    pub(crate) fn source_for(_device_id: &Arc<str>) -> Option<Arc<dyn SourceAccess>> {
        None
    }
}

pub struct DeviceProvider;

impl DeviceProvider {
    /// One entry per device storage, labeled "<model> (<storage>)".
    pub fn collect(&self) -> Vec<SourceEntry> {
        backend::detect_devices()
            .into_iter()
            .flat_map(device_entries)
            .collect()
    }

    /// The reader for a detected device's contents.
    pub fn access_for(
        &self,
        device_id: &Arc<str>,
    ) -> Option<Arc<dyn crate::sources::SourceAccess>> {
        backend::source_for(device_id)
    }
}

fn device_entries(device: DetectedDevice) -> Vec<SourceEntry> {
    let model = device.info.model.clone();
    device
        .storages
        .into_iter()
        .map(|storage| {
            let label = if storage.label.trim().is_empty() {
                model.clone()
            } else {
                format!("{model} ({})", storage.label)
            };
            SourceEntry {
                id: SourceId::new(format!("device:{}:{}", device.info.id, storage.id)),
                label,
                kind: SourceKind::Device {
                    vendor: device.info.vendor.clone(),
                    model: model.clone(),
                },
                backing: SourceBacking::Mtp(super::mtp::MtpObjectRef {
                    device_id: device.info.id.clone(),
                    storage_id: storage.id.clone(),
                    object: Arc::from(""),
                }),
                capacity: Some(SourceCapacity {
                    total: storage.capacity,
                    available: storage.available,
                }),
            }
        })
        .collect()
}
