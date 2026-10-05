//! Windows Portable Devices (WPD) backed MTP device access (Windows).
//!
//! The full backend (device enumeration via `IPortableDeviceManager`,
//! object enumeration via `IPortableDeviceContent`, streaming reads via
//! `IPortableDeviceResources`) is implemented against the Windows CI
//! runner in the import plan's stage 6; this module keeps the dispatch
//! surface ready and compiles on Windows without offering devices yet.

use std::sync::Arc;

use crate::sources::SourceAccess;

use super::DetectedDevice;

pub(crate) fn detect_devices() -> Vec<DetectedDevice> {
    Vec::new()
}

pub(crate) fn source_for(_device_id: &Arc<str>) -> Option<Arc<dyn SourceAccess>> {
    None
}
