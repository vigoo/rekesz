//! Destination layout for imported photos: the user picks a base folder
//! inside the library root, and a naming layout applied within it.

use std::path::PathBuf;
use std::time::SystemTime;

use chrono::{Datelike, Local, NaiveDate, TimeZone};
use serde::{Deserialize, Serialize};

use crate::sources::SourceBacking;

use super::ScannedFile;

/// Naming layout applied inside the chosen destination folder.
#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, Eq, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub enum DestinationPreset {
    /// `2026/2026-10-04` subfolders by capture date.
    #[default]
    YearDay,
    /// `2026/10` subfolders by capture month.
    YearMonth,
    /// Photos land directly in the destination folder under their own names.
    KeepFileNames,
    /// Photos land directly in the destination folder, renamed to a
    /// four-digit counter (`0001`, `0002`, …) with the original extension.
    Counter,
}

/// All layouts with their labels, in menu order.
pub const ALL_PRESETS: [(DestinationPreset, &str); 4] = [
    (DestinationPreset::YearDay, "Date folders: YYYY/YYYY-MM-DD"),
    (DestinationPreset::YearMonth, "Date folders: YYYY/MM"),
    (DestinationPreset::KeepFileNames, "Original file names"),
    (DestinationPreset::Counter, "Numbered: 0001, 0002, …"),
];

impl DestinationPreset {
    /// Subfolder path (relative to the destination folder) a photo captured
    /// on `date` lands in; empty for the flat layouts.
    pub fn directory_for(self, date: NaiveDate) -> PathBuf {
        let year = format!("{:04}", date.year());
        let day = format!("{:04}-{:02}-{:02}", date.year(), date.month(), date.day());
        match self {
            Self::YearDay => PathBuf::from(year).join(day),
            Self::YearMonth => PathBuf::from(year).join(format!("{:02}", date.month())),
            Self::KeepFileNames | Self::Counter => PathBuf::new(),
        }
    }

    /// Whether files are renamed to a running counter instead of keeping
    /// their names.
    pub fn uses_counter(self) -> bool {
        matches!(self, Self::Counter)
    }
}

/// Capture date of a source file: EXIF/XMP capture time when the backing is a
/// readable path, else the file's modification time, else today. Device files
/// fall back to their modification time; staged copies carry the device
/// timestamp, and their EXIF is read like any library file.
///
/// Times are interpreted in the machine's local zone, consistent with the
/// metadata loader's EXIF handling; capture dates shift accordingly for
/// photos taken in other timezones.
pub fn capture_date(file: &ScannedFile) -> NaiveDate {
    let time = match &file.backing {
        SourceBacking::Path(path) => crate::metadata::file_capture_time(path),
        SourceBacking::Mtp(_) | SourceBacking::Ptp(_) => None,
    }
    .or(file.modified);
    date_from(time).unwrap_or_else(|| Local::now().date_naive())
}

fn date_from(time: Option<SystemTime>) -> Option<NaiveDate> {
    let time = time?;
    // Pre-epoch timestamps (scans of old media) keep their own date instead
    // of collapsing into "today"; out-of-range values yield None.
    let seconds = match time.duration_since(std::time::UNIX_EPOCH) {
        Ok(ahead) => ahead.as_secs() as i64,
        Err(error) => -(error.duration().as_secs() as i64),
    };
    Some(Local.timestamp_opt(seconds, 0).single()?.date_naive())
}
