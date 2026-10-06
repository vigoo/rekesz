//! libgphoto2-backed PTP camera access (Linux/macOS).
//!
//! Camera handles are opened per operation and dropped afterwards — the
//! same shape as the libmtp backend; a persistent handle would need
//! replug and invalidation handling this backend avoids for now.
//! libgphoto2 executes every call on a single background thread, so the
//! C-level work is serialized process-wide. Detection results are cached
//! briefly to keep the source-list polling cheap, and the poller never
//! blocks on a running transfer: detection uses `try_lock` and keeps the
//! previous result while a download holds the device.

use std::collections::HashMap;
use std::io::{self, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock, Mutex, OnceLock};
use std::task::{Context as TaskContext, Poll, Waker};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use gphoto2::context::ProgressHandler;
use gphoto2::task::Task;
use gphoto2::{Camera, Context};

use crate::sources::{SourceAccess, SourceBacking, SourceDirEntry};

use super::{DetectedDevice, DeviceInfo, PtpObjectRef, StorageInfo, child_path, port_of_device_id};

/// How long detection results are reused across source-list polls.
const DETECTION_CACHE: Duration = Duration::from_secs(10);

/// Poll cadence while a task is being pin-and-polled.
const TASK_POLL: Duration = Duration::from_millis(25);

/// Serializes camera file operations (open → work → drop). Detection takes
/// the same lock only via `try_lock`: a running download must neither stall
/// the source-list poller nor race a fresh detection into the same device.
static DEVICE_LOCK: Mutex<()> = Mutex::new(());

/// Shared non-cancellable flag for operations without a cancel channel
/// (detection, listing, device lookup).
static NO_CANCEL: AtomicBool = AtomicBool::new(false);

/// A process-lifetime context. Upstream `Drop for Context` queues a task it
/// never starts (leaking the GPContext), so the shared context is never
/// dropped; per-task progress and cancel handlers install and uninstall
/// safely because tasks run serialized on the crate's worker thread.
fn shared_context() -> &'static Context {
    static CONTEXT: OnceLock<Context> = OnceLock::new();
    CONTEXT.get_or_init(|| Context::new().expect("libgphoto2 context creation"))
}

/// Byte progress of the running transfer, written by the crate's worker
/// thread through [`ProgressRelay`] and read by the polling thread.
type ProgressState = Arc<Mutex<u64>>;

struct ProgressRelay {
    state: ProgressState,
}

impl ProgressHandler for ProgressRelay {
    fn start(&mut self, _target: f32, _message: String) -> u32 {
        0
    }

    fn update(&mut self, _id: u32, progress: f32) {
        if let Ok(mut current) = self.state.lock() {
            *current = progress.max(0.0) as u64;
        }
    }

    fn stop(&mut self, _id: u32) {}
}

/// Poll a crate task to completion on this thread. The crate's `wait`
/// consumes the task, which would make an in-flight cancellation
/// unreachable; pinning the task instead keeps `Task::cancel` reachable
/// until the loop observes the app-level cancel flag (latched — the flag
/// keeps being set until libgphoto2 aborts the transfer). `Waker::noop` is
/// enough: the loop polls on a fixed cadence and needs no wake-ups.
fn poll_task<T>(task: Task<T>, cancel: &AtomicBool, mut on_progress: impl FnMut(u64)) -> T
where
    T: 'static + Send,
{
    let state: ProgressState = Arc::new(Mutex::new(0));
    let mut task = Box::pin(task.with_progress_handler(ProgressRelay {
        state: Arc::clone(&state),
    }));
    let mut context = TaskContext::from_waker(Waker::noop());
    let mut cancel_sent = false;
    loop {
        if let Poll::Ready(value) = task.as_mut().poll(&mut context) {
            return value;
        }
        if !cancel_sent && cancel.load(Ordering::Acquire) {
            task.cancel();
            cancel_sent = true;
        }
        if let Ok(current) = state.lock() {
            on_progress(*current);
        }
        std::thread::sleep(TASK_POLL);
    }
}

/// Cancellation is reported by libgphoto2 as a plain error; the app-level
/// flag decides how it surfaces, exactly as in the libmtp backend.
fn transfer_result<T>(result: Result<T, gphoto2::Error>, cancel: &AtomicBool) -> io::Result<T> {
    match result {
        Ok(value) => Ok(value),
        Err(error) => {
            if cancel.load(Ordering::Acquire) {
                Err(io::Error::new(
                    io::ErrorKind::Interrupted,
                    "import cancelled",
                ))
            } else {
                Err(io::Error::other(error.to_string()))
            }
        }
    }
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
    let previous = {
        let cache = CACHE
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        cache
            .as_ref()
            .map(|(_, devices)| devices.clone())
            .unwrap_or_default()
    };
    // A download holds the device for its whole duration; sweeping would
    // only log EBUSY noise, so keep showing the previous list instead.
    let devices = if DEVICE_LOCK.try_lock().is_ok() {
        detect_devices_uncached()
    } else {
        previous
    };
    // A failed sweep must not blank the rail: keep the previous list.
    if !devices.is_empty()
        && let Ok(mut cache) = CACHE.lock()
    {
        *cache = Some((Instant::now(), devices.clone()));
    }
    devices
}

/// Cameras with an open PTP session, keyed by device id. Canon-class
/// cameras physically react to session open/close — the lens extends and
/// retracts on every `gp_camera_init` — so handles are opened once and
/// kept for the app's lifetime; the periodic detection refresh doubles as
/// the keepalive that also stops the camera powering off mid-import. The
/// crate's single worker thread serializes the C-level calls and
/// `DEVICE_LOCK` serializes Rust-level use; `Camera: Send` is asserted by
/// the crate's own compile tests.
struct CachedCamera {
    camera: Camera,
    model: String,
    storages: Vec<StorageInfo>,
}

static CAMERAS: LazyLock<Mutex<HashMap<Arc<str>, CachedCamera>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

fn detect_devices_uncached() -> Vec<DetectedDevice> {
    let context = shared_context();
    // A healthy cached camera holds its USB interface, so the sweep below
    // skips it instead of opening a second session (which is what made the
    // lens cycle); its liveness is checked through the open session.
    let swept = match poll_task(context.list_cameras(), &NO_CANCEL, |_| {}) {
        Ok(descriptors) => descriptors.into_iter().collect::<Vec<_>>(),
        Err(error) => {
            eprintln!("ptp: cannot list cameras: {error}");
            Vec::new()
        }
    };
    let mut cache = CAMERAS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    // Keepalive + eviction: a storage query over the open session fails
    // once the camera is unplugged; fresh capacity lands in the listing.
    let mut gone = Vec::new();
    for (id, entry) in cache.iter_mut() {
        match poll_task(entry.camera.storages(), &NO_CANCEL, |_| {}) {
            Ok(storages) if !storages.is_empty() => entry.storages = storage_infos(&storages),
            Ok(_) => {}
            Err(_) => {
                eprintln!("ptp: camera {id} is gone");
                gone.push(Arc::clone(id));
            }
        }
    }
    for id in gone {
        cache.remove(&id);
    }
    // Open cameras that appeared since the last sweep.
    for descriptor in swept {
        if !descriptor.port.starts_with("usb:") {
            continue;
        }
        let id: Arc<str> = Arc::from(format!("gphoto:{}", descriptor.port));
        if cache.contains_key(&id) {
            continue;
        }
        match poll_task(context.get_camera(&descriptor), &NO_CANCEL, |_| {}) {
            Ok(camera) => {
                let storages = match poll_task(camera.storages(), &NO_CANCEL, |_| {}) {
                    Ok(storages) if !storages.is_empty() => storage_infos(&storages),
                    // Cameras without storage reporting stay browsable from
                    // a synthetic root.
                    _ => vec![StorageInfo {
                        id: Arc::from("/"),
                        label: String::new(),
                        total: None,
                        available: None,
                    }],
                };
                cache.insert(
                    id,
                    CachedCamera {
                        camera,
                        model: descriptor.model.clone(),
                        storages,
                    },
                );
            }
            // Somebody else (another app, another stack in this process)
            // holds the camera: it stays invisible until they let go.
            Err(error) => eprintln!("ptp: cannot open camera {}: {error}", id),
        }
    }
    cache
        .iter()
        .map(|(id, entry)| DetectedDevice {
            info: DeviceInfo {
                id: Arc::clone(id),
                vendor: String::new(),
                model: entry.model.clone(),
            },
            storages: entry.storages.clone(),
        })
        .collect()
}

fn storage_infos(storages: &[gphoto2::filesys::StorageInfo]) -> Vec<StorageInfo> {
    storages
        .iter()
        .map(|storage| StorageInfo {
            id: Arc::from(
                storage
                    .base_directory()
                    .unwrap_or_default()
                    .trim_end()
                    .to_owned(),
            ),
            label: storage.label().unwrap_or_default().into_owned(),
            total: storage.capacity_kb(),
            available: storage.free_kb(),
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

pub(crate) fn source_for(device_id: &Arc<str>) -> Option<Arc<dyn SourceAccess>> {
    Some(Arc::new(PtpSource::new(Arc::clone(device_id))))
}

/// Read access to one detected camera, addressed by its detection id
/// (`gphoto:<port path>`).
#[derive(Clone)]
pub(crate) struct PtpSource {
    device_id: Arc<str>,
}

impl PtpSource {
    pub(crate) fn new(device_id: Arc<str>) -> Self {
        Self { device_id }
    }

    fn with_camera<T>(
        &self,
        operation: impl FnOnce(&Camera, &Context) -> io::Result<T>,
    ) -> io::Result<T> {
        port_of_device_id(&self.device_id).ok_or_else(|| invalid("malformed device id"))?;
        let context = shared_context();
        let _lock = DEVICE_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut cache = CAMERAS
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if !cache.contains_key(&self.device_id) {
            // First touch of this camera (detection may not have swept it
            // yet): resolve the port and open the session once, for good.
            let port = port_of_device_id(&self.device_id).expect("validated above");
            let descriptor = poll_task(context.list_cameras(), &NO_CANCEL, |_| {})
                .map_err(|error| io::Error::other(error.to_string()))?
                .into_iter()
                .find(|descriptor| descriptor.port == port)
                .ok_or_else(|| not_found("camera is not connected"))?;
            let camera = poll_task(context.get_camera(&descriptor), &NO_CANCEL, |_| {})
                .map_err(|error| io::Error::other(error.to_string()))?;
            cache.insert(
                Arc::clone(&self.device_id),
                CachedCamera {
                    camera,
                    model: descriptor.model.clone(),
                    storages: Vec::new(),
                },
            );
        }
        let entry = cache.get(&self.device_id).expect("inserted above");
        operation(&entry.camera, context)
    }
}

impl SourceAccess for PtpSource {
    fn list_dir(&self, dir: &SourceBacking) -> io::Result<Vec<SourceDirEntry>> {
        let reference = directory_reference(&self.device_id, dir)?;
        self.with_camera(|camera, _context| {
            let filesystem = camera.fs();
            let folder = reference.folder.to_string();
            let mut entries = Vec::new();
            for name in poll_task(filesystem.list_folders(&folder), &NO_CANCEL, |_| {})
                .map_err(|error| io::Error::other(error.to_string()))?
            {
                entries.push(SourceDirEntry {
                    name: name.clone(),
                    is_dir: true,
                    backing: SourceBacking::Ptp(PtpObjectRef {
                        device_id: Arc::clone(&reference.device_id),
                        folder: Arc::from(child_path(&folder, &name)),
                        name: Arc::from(""),
                    }),
                    size: 0,
                    modified: None,
                });
            }
            for name in poll_task(filesystem.list_files(&folder), &NO_CANCEL, |_| {})
                .map_err(|error| io::Error::other(error.to_string()))?
            {
                let info = poll_task(filesystem.file_info(&folder, &name), &NO_CANCEL, |_| {}).ok();
                let size = info
                    .as_ref()
                    .and_then(|info| info.file().size())
                    .unwrap_or(0);
                let modified = info
                    .as_ref()
                    .and_then(|info| info.file().mtime())
                    .and_then(modification_time);
                entries.push(SourceDirEntry {
                    name: name.clone(),
                    is_dir: false,
                    backing: SourceBacking::Ptp(PtpObjectRef {
                        device_id: Arc::clone(&reference.device_id),
                        folder: Arc::from(folder.as_str()),
                        name: Arc::from(name),
                    }),
                    size,
                    modified,
                });
            }
            Ok(sort_source_entries(entries))
        })
    }

    fn read_file(
        &self,
        file: &SourceBacking,
        out: &mut dyn Write,
        cancel: &AtomicBool,
        progress: &mut dyn FnMut(u64),
    ) -> io::Result<u64> {
        let reference = file_reference(&self.device_id, file)?;
        self.with_camera(|camera, context| {
            let filesystem = camera.fs();
            let camera_file = transfer_result(
                poll_task(
                    filesystem.download(&reference.folder, &reference.name),
                    cancel,
                    |_| {},
                ),
                cancel,
            )?;
            let data = transfer_result(
                poll_task(camera_file.get_data(context), cancel, |_| {}),
                cancel,
            )?;
            if cancel.load(Ordering::Acquire) {
                return Err(io::Error::new(
                    io::ErrorKind::Interrupted,
                    "import cancelled",
                ));
            }
            out.write_all(&data)?;
            progress(data.len() as u64);
            Ok(data.len() as u64)
        })
    }

    fn read_file_to_path(
        &self,
        file: &SourceBacking,
        destination: &Path,
        cancel: &AtomicBool,
        progress: &mut dyn FnMut(u64),
    ) -> Option<io::Result<u64>> {
        let reference = match file_reference(&self.device_id, file) {
            Ok(reference) => reference,
            Err(error) => return Some(Err(error)),
        };
        Some(self.with_camera(|camera, _context| {
            let state: ProgressState = Arc::new(Mutex::new(0));
            let task = camera
                .fs()
                .download_to(&reference.folder, &reference.name, destination)
                .with_progress_handler(ProgressRelay {
                    state: Arc::clone(&state),
                });
            let result = poll_task(task, cancel, progress);
            match result {
                // The handle owns the destination fd; drop it before the
                // written-byte check so the size is settled.
                Ok(_camera_file) => Ok(std::fs::metadata(destination)
                    .map(|metadata| metadata.len())
                    .unwrap_or(0)),
                Err(error) => transfer_result(Err(error), cancel),
            }
        }))
    }
}

fn modification_time(seconds: i64) -> Option<SystemTime> {
    if seconds < 0 {
        return None;
    }
    UNIX_EPOCH.checked_add(Duration::from_secs(seconds as u64))
}

fn not_found(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::NotFound, message)
}

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}

fn directory_reference<'a>(
    device_id: &Arc<str>,
    backing: &'a SourceBacking,
) -> io::Result<&'a PtpObjectRef> {
    match backing {
        SourceBacking::Ptp(reference) if &reference.device_id == device_id => {
            if reference.name.is_empty() {
                Ok(reference)
            } else {
                Err(invalid("backing addresses a file"))
            }
        }
        SourceBacking::Ptp(_) => Err(invalid("backing belongs to another camera")),
        SourceBacking::Mtp(_) => Err(invalid("camera source cannot address MTP objects")),
        SourceBacking::Path(_) => Err(invalid("camera source cannot address paths")),
    }
}

fn file_reference<'a>(
    device_id: &Arc<str>,
    backing: &'a SourceBacking,
) -> io::Result<&'a PtpObjectRef> {
    match backing {
        SourceBacking::Ptp(reference) if &reference.device_id == device_id => {
            if reference.name.is_empty() {
                Err(invalid("backing addresses a directory"))
            } else {
                Ok(reference)
            }
        }
        SourceBacking::Ptp(_) => Err(invalid("backing belongs to another camera")),
        SourceBacking::Mtp(_) => Err(invalid("camera source cannot address MTP objects")),
        SourceBacking::Path(_) => Err(invalid("camera source cannot address paths")),
    }
}

#[cfg(test)]
#[path = "gphoto_backend_tests.rs"]
mod tests;
