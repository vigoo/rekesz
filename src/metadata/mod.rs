use std::{
    fs,
    io::BufReader,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::{SystemTime, UNIX_EPOCH},
};

use crate::jobs::JobLimiter;
use chrono::{Local, NaiveDateTime, TimeZone as _};
use rayon::prelude::*;

const CACHE_VERSION: &str = "metadata-v3";

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GeoCoordinate {
    latitude: f64,
    longitude: f64,
}

impl GeoCoordinate {
    pub fn new(latitude: f64, longitude: f64) -> Option<Self> {
        (latitude.is_finite()
            && longitude.is_finite()
            && (-90.0..=90.0).contains(&latitude)
            && (-180.0..=180.0).contains(&longitude))
        .then_some(Self {
            latitude,
            longitude,
        })
    }

    pub fn latitude(self) -> f64 {
        self.latitude
    }

    pub fn longitude(self) -> f64 {
        self.longitude
    }
}

#[derive(Clone, Debug)]
pub struct ImageOrderingMetadata {
    pub path: PathBuf,
    pub capture_time: Option<SystemTime>,
    pub fallback_time: Option<SystemTime>,
    pub coordinate: Option<GeoCoordinate>,
}

#[derive(Clone)]
pub struct MetadataLoader {
    cache: Option<Arc<Mutex<rusqlite::Connection>>>,
    pool: Arc<rayon::ThreadPool>,
    limiter: JobLimiter,
}

struct CachedMetadata {
    capture_unix_seconds: Option<u64>,
    coordinate: Option<GeoCoordinate>,
}

impl MetadataLoader {
    pub fn new(cache_root: PathBuf, parallelism: usize, limiter: JobLimiter) -> Self {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(parallelism.max(1))
            .thread_name(|index| format!("rekesz-metadata-{index}"))
            .build()
            .expect("failed to create metadata worker pool");
        let cache = open_cache(&cache_root.join(format!("{CACHE_VERSION}.sqlite3")))
            .map(|connection| Arc::new(Mutex::new(connection)));
        Self {
            cache,
            pool: Arc::new(pool),
            limiter,
        }
    }

    pub fn load_directory(
        &self,
        paths: Vec<PathBuf>,
        cancelled: Arc<AtomicBool>,
    ) -> Vec<ImageOrderingMetadata> {
        let (metadata, pending): (Vec<_>, Vec<_>) = self.pool.install(|| {
            paths
                .into_par_iter()
                .filter_map(|path| {
                    if cancelled.load(Ordering::Acquire) {
                        return None;
                    }
                    let _permit = self.limiter.acquire(false, &cancelled)?;
                    Some(self.resolve(path, &cancelled))
                })
                .unzip()
        });
        self.store_batch(pending.into_iter().flatten());
        metadata
    }

    pub fn load_cached_directory(
        &self,
        paths: Vec<PathBuf>,
        cancelled: Arc<AtomicBool>,
    ) -> (Vec<ImageOrderingMetadata>, Vec<PathBuf>) {
        let results: Vec<_> = self.pool.install(|| {
            paths
                .into_par_iter()
                .filter_map(|path| {
                    if cancelled.load(Ordering::Acquire) {
                        return None;
                    }
                    let _permit = self.limiter.acquire(true, &cancelled)?;
                    Some(self.resolve_cached(path))
                })
                .collect()
        });
        let mut entries = Vec::with_capacity(results.len());
        let mut missing = Vec::new();
        for (entry, cached) in results {
            if !cached {
                missing.push(entry.path.clone());
            }
            entries.push(entry);
        }
        (entries, missing)
    }

    fn resolve(
        &self,
        path: PathBuf,
        cancelled: &AtomicBool,
    ) -> (ImageOrderingMetadata, Option<(String, CachedMetadata)>) {
        let (cached_entry, cached) = self.resolve_cached(path);
        if cached {
            return (cached_entry, None);
        }
        let ImageOrderingMetadata {
            path,
            fallback_time,
            ..
        } = cached_entry;
        let xmp_path = adjacent_xmp(&path);
        let key = metadata_cache_key(&path, xmp_path.as_deref()).ok();
        let capture_time = xmp_path.as_deref().and_then(xmp_capture_time).or_else(|| {
            if cancelled.load(Ordering::Acquire) {
                None
            } else {
                embedded_capture_time(&path)
            }
        });
        let coordinate = xmp_path.as_deref().and_then(xmp_coordinate).or_else(|| {
            if cancelled.load(Ordering::Acquire) {
                None
            } else {
                embedded_coordinate(&path)
            }
        });
        let pending = (!cancelled.load(Ordering::Acquire))
            .then(|| {
                key.map(|key| {
                    (
                        key,
                        CachedMetadata {
                            capture_unix_seconds: capture_time.and_then(to_unix_seconds),
                            coordinate,
                        },
                    )
                })
            })
            .flatten();
        (
            ImageOrderingMetadata {
                path,
                capture_time,
                fallback_time,
                coordinate,
            },
            pending,
        )
    }

    fn resolve_cached(&self, path: PathBuf) -> (ImageOrderingMetadata, bool) {
        let source_metadata = fs::metadata(&path).ok();
        let fallback_time = source_metadata
            .as_ref()
            .and_then(|metadata| metadata.modified().ok());
        let xmp_path = adjacent_xmp(&path);
        let key = metadata_cache_key(&path, xmp_path.as_deref()).ok();
        if let Some(cached) = key.as_deref().and_then(|key| self.load_cache(key)) {
            return (
                ImageOrderingMetadata {
                    path,
                    capture_time: cached.capture_unix_seconds.map(unix_time),
                    fallback_time,
                    coordinate: cached.coordinate,
                },
                true,
            );
        }
        (
            ImageOrderingMetadata {
                path,
                capture_time: None,
                fallback_time,
                coordinate: None,
            },
            false,
        )
    }

    fn load_cache(&self, key: &str) -> Option<CachedMetadata> {
        let cache = self.cache.as_ref()?.lock().ok()?;
        cache
            .query_row(
                "SELECT capture_unix_seconds, latitude, longitude FROM image_metadata WHERE identity = ?1",
                [key],
                |row| {
                    Ok(CachedMetadata {
                        capture_unix_seconds: row
                            .get::<_, Option<i64>>(0)?
                            .and_then(|value| value.try_into().ok()),
                        coordinate: match (row.get(1)?, row.get(2)?) {
                            (Some(latitude), Some(longitude)) => {
                                GeoCoordinate::new(latitude, longitude)
                            }
                            _ => None,
                        },
                    })
                },
            )
            .ok()
    }

    fn store_batch(&self, metadata: impl Iterator<Item = (String, CachedMetadata)>) {
        let Some(cache) = &self.cache else {
            return;
        };
        let Ok(mut cache) = cache.lock() else {
            return;
        };
        let Ok(transaction) = cache.transaction() else {
            return;
        };
        {
            let Ok(mut statement) = transaction.prepare_cached(
                "INSERT INTO image_metadata(identity, capture_unix_seconds, latitude, longitude) VALUES (?1, ?2, ?3, ?4) \
                 ON CONFLICT(identity) DO UPDATE SET capture_unix_seconds = excluded.capture_unix_seconds, \
                 latitude = excluded.latitude, longitude = excluded.longitude",
            ) else {
                return;
            };
            for (key, metadata) in metadata {
                let _ = statement.execute(rusqlite::params![
                    key,
                    metadata
                        .capture_unix_seconds
                        .and_then(|value| i64::try_from(value).ok()),
                    metadata.coordinate.map(GeoCoordinate::latitude),
                    metadata.coordinate.map(GeoCoordinate::longitude),
                ]);
            }
        }
        let _ = transaction.commit();
    }
}

fn open_cache(path: &Path) -> Option<rusqlite::Connection> {
    fs::create_dir_all(path.parent()?).ok()?;
    let connection = rusqlite::Connection::open(path).ok()?;
    connection
        .execute_batch(
            "PRAGMA journal_mode=WAL;
             PRAGMA synchronous=NORMAL;
             CREATE TABLE IF NOT EXISTS image_metadata (
                 identity TEXT PRIMARY KEY NOT NULL,
                 capture_unix_seconds INTEGER,
                 latitude REAL,
                 longitude REAL
             ) WITHOUT ROWID;",
        )
        .ok()?;
    Some(connection)
}

fn adjacent_xmp(path: &Path) -> Option<PathBuf> {
    [
        PathBuf::from(format!("{}.xmp", path.to_string_lossy())),
        path.with_extension("xmp"),
        PathBuf::from(format!("{}.XMP", path.to_string_lossy())),
        path.with_extension("XMP"),
    ]
    .into_iter()
    .find(|candidate| candidate.is_file())
}

fn metadata_cache_key(path: &Path, xmp: Option<&Path>) -> Result<String, std::io::Error> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(CACHE_VERSION.as_bytes());
    hash_file_identity(&mut hasher, path)?;
    if let Some(xmp) = xmp {
        hash_file_identity(&mut hasher, xmp)?;
    }
    Ok(hasher.finalize().to_hex().to_string())
}

fn hash_file_identity(hasher: &mut blake3::Hasher, path: &Path) -> Result<(), std::io::Error> {
    let metadata = fs::metadata(path)?;
    hasher.update(path.to_string_lossy().as_bytes());
    hasher.update(&metadata.len().to_le_bytes());
    let modified = metadata
        .modified()
        .ok()
        .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
        .unwrap_or_default();
    hasher.update(&modified.as_secs().to_le_bytes());
    hasher.update(&modified.subsec_nanos().to_le_bytes());
    Ok(())
}

fn xmp_capture_time(path: &Path) -> Option<SystemTime> {
    let text = fs::read_to_string(path).ok()?;
    let value = text
        .split_once("exif:DateTimeOriginal=\"")?
        .1
        .split_once('"')?
        .0;
    parse_exif_time(value)
}

fn xmp_coordinate(path: &Path) -> Option<GeoCoordinate> {
    let text = fs::read_to_string(path).ok()?;
    let latitude = xmp_attribute(&text, "exif:GPSLatitude")?;
    let longitude = xmp_attribute(&text, "exif:GPSLongitude")?;
    GeoCoordinate::new(
        parse_xmp_angle(latitude, ['N', 'S'])?,
        parse_xmp_angle(longitude, ['E', 'W'])?,
    )
}

fn xmp_attribute<'a>(text: &'a str, name: &str) -> Option<&'a str> {
    text.split_once(&format!("{name}=\""))?
        .1
        .split_once('"')
        .map(|v| v.0)
}

fn parse_xmp_angle(value: &str, allowed_directions: [char; 2]) -> Option<f64> {
    let value = value.trim();
    let direction = value.chars().last().filter(|c| "NSEWnsew".contains(*c));
    let unsigned = direction
        .map_or(value, |_| &value[..value.len() - 1])
        .trim();
    let parts = unsigned
        .split([',', ' '])
        .filter(|part| !part.is_empty())
        .map(str::parse::<f64>)
        .collect::<Result<Vec<_>, _>>()
        .ok()?;
    let first = *parts.first()?;
    if parts.iter().any(|value| !value.is_finite()) {
        return None;
    }
    if direction.is_some_and(|value| {
        !allowed_directions.contains(&value.to_ascii_uppercase()) || first.is_sign_negative()
    }) {
        return None;
    }
    let magnitude = match parts.as_slice() {
        [decimal] => decimal.abs(),
        [degrees, minutes] if (0.0..60.0).contains(minutes) => degrees.abs() + minutes / 60.0,
        [degrees, minutes, seconds]
            if (0.0..60.0).contains(minutes) && (0.0..60.0).contains(seconds) =>
        {
            degrees.abs() + minutes / 60.0 + seconds / 3600.0
        }
        _ => return None,
    };
    let sign = match direction.map(|c| c.to_ascii_uppercase()) {
        Some('S' | 'W') => -1.0,
        Some('N' | 'E') => 1.0,
        None if first.is_sign_negative() => -1.0,
        None => 1.0,
        _ => return None,
    };
    Some(sign * magnitude)
}

fn embedded_capture_time(path: &Path) -> Option<SystemTime> {
    if crate::formats::is_raw(path) {
        raw_capture_time(path).or_else(|| exif_capture_time(path))
    } else {
        exif_capture_time(path)
    }
}

/// Best-effort capture time for an arbitrary file: an adjacent XMP sidecar's
/// `exif:DateTimeOriginal`, else the embedded EXIF/RAW capture time. Used by
/// the import engine for files that are not in the library yet.
pub fn file_capture_time(path: &Path) -> Option<SystemTime> {
    adjacent_xmp(path)
        .as_deref()
        .and_then(xmp_capture_time)
        .or_else(|| embedded_capture_time(path))
}

fn embedded_coordinate(path: &Path) -> Option<GeoCoordinate> {
    if crate::formats::is_raw(path) {
        raw_coordinate(path).or_else(|| exif_coordinate(path))
    } else if path
        .extension()
        .and_then(|value| value.to_str())
        .is_some_and(|value| matches!(value.to_ascii_lowercase().as_str(), "heic" | "heif"))
    {
        heif_coordinate(path).or_else(|| exif_coordinate(path))
    } else {
        exif_coordinate(path)
    }
}

fn heif_coordinate(path: &Path) -> Option<GeoCoordinate> {
    let bytes = fs::read(path).ok()?;
    let context = libheif_rs::HeifContext::read_from_bytes(&bytes).ok()?;
    let handle = context.primary_image_handle().ok()?;
    let mut ids = vec![0; handle.number_of_metadata_blocks(b"Exif").max(0) as usize];
    let count = handle.metadata_block_ids(&mut ids, b"Exif");
    ids.truncate(count);
    ids.into_iter()
        .find_map(|id| heif_exif_coordinate(&handle.metadata(id).ok()?))
}

fn heif_exif_coordinate(data: &[u8]) -> Option<GeoCoordinate> {
    let offset = u32::from_be_bytes(data.get(..4)?.try_into().ok()?) as usize;
    let tiff = data.get(4_usize.checked_add(offset)?..)?;
    let exif = exif::Reader::new().read_raw(tiff.to_vec()).ok()?;
    coordinate_from_exif(&exif)
}

fn raw_coordinate(path: &Path) -> Option<GeoCoordinate> {
    use rawler::{decoders::RawDecodeParams, rawsource::RawSource};

    let source = RawSource::new(path).ok()?;
    let decoder = rawler::global_loader().get_decoder(&source).ok()?;
    let metadata = decoder
        .raw_metadata(&source, &RawDecodeParams::default())
        .ok()?;
    let gps = metadata.exif.gps?;
    let latitude = raw_dms(
        gps.gps_latitude?,
        gps.gps_latitude_ref.as_deref()?,
        ['N', 'S'],
    )?;
    let longitude = raw_dms(
        gps.gps_longitude?,
        gps.gps_longitude_ref.as_deref()?,
        ['E', 'W'],
    )?;
    GeoCoordinate::new(latitude, longitude)
}

fn raw_dms(
    values: [rawler::formats::tiff::Rational; 3],
    direction: &str,
    allowed_directions: [char; 2],
) -> Option<f64> {
    let values = values
        .into_iter()
        .map(|value| f32::try_from(value).ok().map(f64::from))
        .collect::<Option<Vec<_>>>()?
        .try_into()
        .ok()?;
    dms_coordinate(values, direction, allowed_directions)
}

fn exif_coordinate(path: &Path) -> Option<GeoCoordinate> {
    let file = fs::File::open(path).ok()?;
    let exif = exif::Reader::new()
        .read_from_container(&mut BufReader::new(file))
        .ok()?;
    coordinate_from_exif(&exif)
}

fn coordinate_from_exif(exif: &exif::Exif) -> Option<GeoCoordinate> {
    let latitude = exif_dms(
        exif,
        exif::Tag::GPSLatitude,
        exif::Tag::GPSLatitudeRef,
        ['N', 'S'],
    )?;
    let longitude = exif_dms(
        exif,
        exif::Tag::GPSLongitude,
        exif::Tag::GPSLongitudeRef,
        ['E', 'W'],
    )?;
    GeoCoordinate::new(latitude, longitude)
}

fn exif_dms(
    exif: &exif::Exif,
    tag: exif::Tag,
    reference: exif::Tag,
    allowed_directions: [char; 2],
) -> Option<f64> {
    let exif::Value::Rational(values) = &exif.get_field(tag, exif::In::PRIMARY)?.value else {
        return None;
    };
    let values: [f64; 3] = values
        .iter()
        .take(3)
        .map(|value| value.to_f64())
        .collect::<Vec<_>>()
        .try_into()
        .ok()?;
    let direction = exif
        .get_field(reference, exif::In::PRIMARY)?
        .display_value()
        .to_string();
    dms_coordinate(
        values,
        direction.trim_matches(['\"', '\0', ' ']),
        allowed_directions,
    )
}

fn dms_coordinate(values: [f64; 3], direction: &str, allowed_directions: [char; 2]) -> Option<f64> {
    if values
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
        || values[1] >= 60.0
        || values[2] >= 60.0
    {
        return None;
    }
    let direction = direction.trim().to_ascii_uppercase();
    if !allowed_directions
        .iter()
        .any(|allowed| direction == allowed.to_string())
    {
        return None;
    }
    let sign = match direction.as_str() {
        "N" | "E" => 1.0,
        "S" | "W" => -1.0,
        _ => return None,
    };
    Some(sign * (values[0] + values[1] / 60.0 + values[2] / 3600.0))
}

fn raw_capture_time(path: &Path) -> Option<SystemTime> {
    use rawler::{decoders::RawDecodeParams, rawsource::RawSource};

    let source = RawSource::new(path).ok()?;
    let decoder = rawler::global_loader().get_decoder(&source).ok()?;
    let metadata = decoder
        .raw_metadata(&source, &RawDecodeParams::default())
        .ok()?;
    metadata
        .exif
        .date_time_original
        .as_deref()
        .and_then(parse_exif_time)
}

fn exif_capture_time(path: &Path) -> Option<SystemTime> {
    let file = fs::File::open(path).ok()?;
    let exif = exif::Reader::new()
        .read_from_container(&mut BufReader::new(file))
        .ok()?;
    let field = exif.get_field(exif::Tag::DateTimeOriginal, exif::In::PRIMARY)?;
    let exif::Value::Ascii(values) = &field.value else {
        return None;
    };
    std::str::from_utf8(values.first()?)
        .ok()
        .map(|value| value.trim_matches(['\0', ' ']))
        .and_then(parse_exif_time)
}

fn parse_exif_time(value: &str) -> Option<SystemTime> {
    let value = value.get(..19).unwrap_or(value);
    let naive = NaiveDateTime::parse_from_str(value, "%Y:%m:%d %H:%M:%S").ok()?;
    let local = Local.from_local_datetime(&naive).earliest()?;
    Some(local.into())
}

fn to_unix_seconds(time: SystemTime) -> Option<u64> {
    time.duration_since(UNIX_EPOCH)
        .ok()
        .map(|value| value.as_secs())
}

fn unix_time(seconds: u64) -> SystemTime {
    UNIX_EPOCH + std::time::Duration::from_secs(seconds)
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
