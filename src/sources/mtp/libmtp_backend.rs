//! libmtp-backed MTP device access (Linux/macOS).
//!
//! libmtp device handles own raw pointers and are neither `Send` nor
//! `Sync`, so every operation opens the device afresh, does its work, and
//! drops the handle. Detection results are cached briefly to keep the
//! source-list polling cheap.

use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use libmtp_rs::device::raw::detect_raw_devices;
use libmtp_rs::device::{MtpDevice, StorageSort};
use libmtp_rs::object::{Object, filetypes::Filetype};
use libmtp_rs::storage::{Parent, Storage};
use libmtp_rs::util::{CallbackReturn, HandlerReturn};

use crate::sources::{SourceAccess, SourceBacking, SourceDirEntry};

use super::{DetectedDevice, DeviceInfo, MtpObjectRef, StorageInfo};

/// How long detection results are reused across source-list polls.
const DETECTION_CACHE: Duration = Duration::from_secs(10);

/// Serializes every device open and transfer. libmtp claims the USB
/// interface per device; a second concurrent open fails with EBUSY, which
/// would otherwise make the source rail flicker and staging/import reads
/// race each other.
static DEVICE_LOCK: Mutex<()> = Mutex::new(());

fn device_lock() -> std::sync::MutexGuard<'static, ()> {
    DEVICE_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

pub(crate) fn detect_devices() -> Vec<DetectedDevice> {
    static CACHE: Mutex<Option<(Instant, Vec<DetectedDevice>)>> = Mutex::new(None);
    {
        let cache = CACHE
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some((at, devices)) = cache.as_ref()
            && at.elapsed() < DETECTION_CACHE
        {
            return devices.clone();
        }
    }
    let devices = detect_devices_uncached();
    // A failed sweep must not blank the rail: keep the previous list.
    if !devices.is_empty()
        && let Ok(mut cache) = CACHE.lock()
    {
        *cache = Some((Instant::now(), devices.clone()));
    }
    devices
}

fn detect_devices_uncached() -> Vec<DetectedDevice> {
    let _lock = device_lock();
    let raw_devices = match detect_raw_devices() {
        Ok(raw_devices) => raw_devices,
        Err(error) => {
            eprintln!("mtp: cannot detect devices: {error}");
            return Vec::new();
        }
    };
    raw_devices
        .into_iter()
        .filter_map(|raw| {
            let id = format!("libmtp:{}:{}", raw.bus_number(), raw.dev_number());
            let entry = raw.device_entry();
            let mut device = match raw.open_uncached() {
                Some(device) => device,
                None => {
                    eprintln!("mtp: cannot open device {id} (busy or unauthorized)");
                    return None;
                }
            };
            // Storage metadata needs one refresh; failure yields no storages
            // and the device is skipped until the next poll.
            if let Err(error) = device.update_storage(StorageSort::ByFreeSpace) {
                eprintln!("mtp: cannot list storages of {id}: {error}");
                return None;
            }
            let device_id: Arc<str> = Arc::from(id);
            let storages = device
                .storage_pool()
                .iter()
                .map(|(_, storage)| StorageInfo {
                    id: Arc::from(storage.id().to_string()),
                    label: storage.description().unwrap_or("Storage").to_owned(),
                    capacity: storage.maximum_capacity(),
                    available: storage.free_space_in_bytes(),
                })
                .collect();
            Some(DetectedDevice {
                info: DeviceInfo {
                    id: device_id,
                    vendor: entry.vendor.to_owned(),
                    model: entry.product.to_owned(),
                },
                storages,
            })
        })
        .collect()
}

/// Directories first, then files, each case-insensitively ordered — the
/// ordering contract documented on `SourceAccess`.
fn sort_source_entries(mut entries: Vec<SourceDirEntry>) -> Vec<SourceDirEntry> {
    entries.sort_by(|a, b| match (a.is_dir, b.is_dir) {
        (true, false) => std::cmp::Ordering::Less,
        (false, true) => std::cmp::Ordering::Greater,
        _ => a.name.to_lowercase().cmp(&b.name.to_lowercase()),
    });
    entries
}

/// The reader for a detected device's contents.
pub(crate) fn source_for(device_id: &Arc<str>) -> Option<Arc<dyn SourceAccess>> {
    Some(Arc::new(MtpSource::new(Arc::clone(device_id))))
}

/// Read access to one connected MTP device, addressed by its detection id
/// (`libmtp:<bus>:<dev>`).
#[derive(Clone)]
pub(crate) struct MtpSource {
    device_id: Arc<str>,
}

impl MtpSource {
    pub(crate) fn new(device_id: Arc<str>) -> Self {
        Self { device_id }
    }

    fn with_device<T>(
        &self,
        operation: impl FnOnce(&mut MtpDevice) -> io::Result<T>,
    ) -> io::Result<T> {
        let target =
            parse_device_id(&self.device_id).ok_or_else(|| invalid("malformed device id"))?;
        let raw_devices =
            detect_raw_devices().map_err(|error| io::Error::other(error.to_string()))?;
        for raw in raw_devices {
            if (raw.bus_number(), raw.dev_number()) != target {
                continue;
            }
            let mut device = raw.open_uncached().ok_or_else(|| {
                io::Error::new(io::ErrorKind::NotFound, "device could not be opened")
            })?;
            return operation(&mut device);
        }
        Err(io::Error::new(
            io::ErrorKind::NotFound,
            "device is not connected",
        ))
    }
}

impl SourceAccess for MtpSource {
    fn list_dir(&self, dir: &SourceBacking) -> io::Result<Vec<SourceDirEntry>> {
        let reference = reference_for(&self.device_id, dir)?;
        let storage_id = storage_id_of(reference)?;
        self.with_device(|device| {
            device
                .update_storage(StorageSort::ByFreeSpace)
                .map_err(mtp_io_error)?;
            let object: u32 = reference
                .object
                .parse()
                .map_err(|_| invalid("malformed object id"))?;
            let pool = device.storage_pool();
            let storage = pool
                .by_id(storage_id)
                .ok_or_else(|| not_found("storage is not available"))?;
            Ok(list_storage_dir(reference, storage, object))
        })
    }

    fn read_file(
        &self,
        file: &SourceBacking,
        out: &mut dyn Write,
        cancel: &AtomicBool,
        progress: &mut dyn FnMut(u64),
    ) -> io::Result<u64> {
        let reference = reference_for(&self.device_id, file)?;
        let storage_id = storage_id_of(reference)?;
        self.with_device(|device| {
            device
                .update_storage(StorageSort::ByFreeSpace)
                .map_err(mtp_io_error)?;
            let pool = device.storage_pool();
            let storage = pool
                .by_id(storage_id)
                .ok_or_else(|| not_found("storage is not available"))?;
            let object: u32 = reference
                .object
                .parse()
                .map_err(|_| invalid("malformed object id"))?;
            let written = Arc::new(AtomicU64::new(0));
            // libmtp-rs only surfaces handler errors; a cancelled or
            // interrupted transfer still returns Ok, so the total from the
            // progress callback is the only way to detect truncation.
            let total = Arc::new(AtomicU64::new(0));
            let result = storage.get_file_to_handler_with_callback(
                object,
                |data| match out.write_all(data) {
                    Ok(()) => {
                        written.fetch_add(data.len() as u64, Ordering::Release);
                        HandlerReturn::Ok(data.len() as u32)
                    }
                    Err(_) => HandlerReturn::Error,
                },
                |_sent, total_bytes| {
                    total.store(total_bytes, Ordering::Release);
                    progress(written.load(Ordering::Acquire));
                    if cancel.load(Ordering::Acquire) {
                        CallbackReturn::Cancel
                    } else {
                        CallbackReturn::Continue
                    }
                },
            );
            result.map_err(|error| {
                if cancel.load(Ordering::Acquire) {
                    io::Error::new(io::ErrorKind::Interrupted, "import cancelled")
                } else {
                    mtp_io_error(error)
                }
            })?;
            let written = written.load(Ordering::Acquire);
            if cancel.load(Ordering::Acquire) {
                return Err(io::Error::new(
                    io::ErrorKind::Interrupted,
                    "import cancelled",
                ));
            }
            let total_bytes = total.load(Ordering::Acquire);
            if total_bytes != 0 && written != total_bytes {
                return Err(io::Error::other(format!(
                    "device delivered {written} of {total_bytes} bytes"
                )));
            }
            Ok(written)
        })
    }
}

fn list_storage_dir(
    reference: &MtpObjectRef,
    storage: &Storage<'_>,
    object: u32,
) -> Vec<SourceDirEntry> {
    let parent = if object == 0 {
        Parent::Root
    } else {
        Parent::Folder(object)
    };
    let entries = storage
        .files_and_folders(parent)
        .into_iter()
        .map(|file| {
            let is_dir = matches!(file.ftype(), Filetype::Folder);
            SourceDirEntry {
                name: file.name().to_owned(),
                is_dir,
                backing: SourceBacking::Mtp(MtpObjectRef {
                    device_id: Arc::clone(&reference.device_id),
                    storage_id: Arc::clone(&reference.storage_id),
                    object: Arc::from(file.id().to_string()),
                }),
                size: if is_dir { 0 } else { file.size() },
                modified: Some(file.modification_date().into()),
            }
        })
        .collect();
    sort_source_entries(entries)
}

fn storage_id_of(reference: &MtpObjectRef) -> io::Result<u32> {
    reference
        .storage_id
        .parse()
        .map_err(|_| invalid("malformed storage id"))
}

fn not_found(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::NotFound, message)
}

fn reference_for<'a>(
    device_id: &Arc<str>,
    backing: &'a SourceBacking,
) -> io::Result<&'a MtpObjectRef> {
    match backing {
        SourceBacking::Mtp(reference) if &reference.device_id == device_id => Ok(reference),
        SourceBacking::Mtp(_) => Err(invalid("backing belongs to another device")),
        SourceBacking::Path(_) => Err(invalid("device source cannot address paths")),
    }
}

/// Device ids are `libmtp:<bus>:<dev>`; the pair stays valid for as long as
/// the device stays plugged into the same port.
fn parse_device_id(id: &str) -> Option<(u32, u8)> {
    let rest = id.strip_prefix("libmtp:")?;
    let (bus, dev) = rest.split_once(':')?;
    Some((bus.parse().ok()?, dev.parse().ok()?))
}

fn mtp_io_error(error: libmtp_rs::error::Error) -> io::Error {
    io::Error::other(error.to_string())
}

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}

#[cfg(test)]
#[path = "libmtp_backend_tests.rs"]
mod tests;
