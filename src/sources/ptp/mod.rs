//! PTP camera support (most digital cameras over USB). Detection and file
//! access come from platform backends: libgphoto2 on Linux/macOS; Windows
//! reaches cameras through the WPD backend and needs none of this — both
//! are hidden behind [`backend`].

use std::sync::Arc;

use super::{SourceBacking, SourceCapacity, SourceEntry, SourceId, SourceKind};

/// Identifies one object (file or folder) on a PTP camera. PTP addresses
/// objects by path, not by numeric id: `folder` is the camera-side absolute
/// directory of the entry and `name` the file name — empty for a directory,
/// in which case the entry *is* the directory at `folder`.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct PtpObjectRef {
    pub device_id: Arc<str>,
    pub folder: Arc<str>,
    pub name: Arc<str>,
}

/// Static description of a connected camera. `id` (`gphoto:<port path>`)
/// stays stable for as long as the camera stays attached so that source
/// ids — and with them the UI's selection — survive a source-list refresh.
#[derive(Clone, Debug)]
pub(crate) struct DeviceInfo {
    pub id: Arc<str>,
    pub vendor: String,
    pub model: String,
}

/// One storage area of a camera (SD card, internal memory, …). `id` is the
/// camera-side base directory; cameras that do not report storage details
/// surface a single synthetic root instead.
#[derive(Clone, Debug)]
pub(crate) struct StorageInfo {
    pub id: Arc<str>,
    pub label: String,
    pub total: Option<u64>,
    pub available: Option<u64>,
}

#[derive(Clone, Debug)]
pub(crate) struct DetectedDevice {
    pub info: DeviceInfo,
    pub storages: Vec<StorageInfo>,
}

#[cfg(all(unix, feature = "ptp"))]
pub(crate) mod gphoto_backend;
#[cfg(all(unix, feature = "ptp"))]
pub(crate) use gphoto_backend as backend;

#[cfg(not(all(unix, feature = "ptp")))]
pub(crate) mod backend {
    use super::DetectedDevice;
    use crate::sources::SourceAccess;
    use std::sync::Arc;

    pub(crate) fn detect_devices() -> Vec<DetectedDevice> {
        Vec::new()
    }

    pub(crate) fn source_for(_device_id: &Arc<str>) -> Option<Arc<dyn SourceAccess>> {
        None
    }
}

/// The camera-side path of a listing entry: the folder itself for
/// directories, the parent folder + file name for files.
pub(crate) fn child_path(folder: &str, name: &str) -> String {
    if folder.ends_with('/') {
        format!("{folder}{name}")
    } else {
        format!("{folder}/{name}")
    }
}

/// `usb:BBB,DDD` → `(bus, device)`, used to recognize PTP entries that
/// alias a device the libmtp backend already lists. Other port styles
/// (network, disk) have no USB identity to compare.
pub(crate) fn port_bus_dev(port: &str) -> Option<(u32, u32)> {
    let rest = port.strip_prefix("usb:")?;
    let (bus, dev) = rest.split_once(',')?;
    Some((bus.parse().ok()?, dev.parse().ok()?))
}

/// The port path carried by a `gphoto:<port>` device id.
pub(crate) fn port_of_device_id(id: &str) -> Option<&str> {
    id.strip_prefix("gphoto:")
}

pub struct DeviceProvider;

impl DeviceProvider {
    /// One entry per camera storage, labeled "<model> (<storage>)".
    pub fn collect(&self) -> Vec<SourceEntry> {
        backend::detect_devices()
            .into_iter()
            .flat_map(device_entries)
            .collect()
    }

    /// The reader for a detected camera's contents.
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
                backing: SourceBacking::Ptp(PtpObjectRef {
                    device_id: device.info.id.clone(),
                    folder: storage.id.clone(),
                    name: Arc::from(""),
                }),
                capacity: match (storage.total, storage.available) {
                    (Some(total), Some(available)) if total > 0 => {
                        Some(SourceCapacity { total, available })
                    }
                    _ => None,
                },
            }
        })
        .collect()
}
