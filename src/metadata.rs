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

const CACHE_VERSION: &str = "metadata-v1";

#[derive(Clone, Debug)]
pub struct ImageOrderingMetadata {
    pub path: PathBuf,
    pub capture_time: Option<SystemTime>,
    pub fallback_time: Option<SystemTime>,
}

#[derive(Clone)]
pub struct MetadataLoader {
    cache: Option<Arc<Mutex<rusqlite::Connection>>>,
    pool: Arc<rayon::ThreadPool>,
    limiter: JobLimiter,
}

struct CachedMetadata {
    capture_unix_seconds: Option<u64>,
}

impl MetadataLoader {
    pub fn new(cache_root: PathBuf, parallelism: usize, limiter: JobLimiter) -> Self {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(parallelism.max(1))
            .thread_name(|index| format!("pics-metadata-{index}"))
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
        let pending = (!cancelled.load(Ordering::Acquire))
            .then(|| {
                key.map(|key| {
                    (
                        key,
                        CachedMetadata {
                            capture_unix_seconds: capture_time.and_then(to_unix_seconds),
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
                },
                true,
            );
        }
        (
            ImageOrderingMetadata {
                path,
                capture_time: None,
                fallback_time,
            },
            false,
        )
    }

    fn load_cache(&self, key: &str) -> Option<CachedMetadata> {
        let cache = self.cache.as_ref()?.lock().ok()?;
        cache
            .query_row(
                "SELECT capture_unix_seconds FROM image_metadata WHERE identity = ?1",
                [key],
                |row| {
                    Ok(CachedMetadata {
                        capture_unix_seconds: row
                            .get::<_, Option<i64>>(0)?
                            .and_then(|value| value.try_into().ok()),
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
                "INSERT INTO image_metadata(identity, capture_unix_seconds) VALUES (?1, ?2) \
                 ON CONFLICT(identity) DO UPDATE SET capture_unix_seconds = excluded.capture_unix_seconds",
            ) else {
                return;
            };
            for (key, metadata) in metadata {
                let _ = statement.execute(rusqlite::params![
                    key,
                    metadata
                        .capture_unix_seconds
                        .and_then(|value| i64::try_from(value).ok())
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
                 capture_unix_seconds INTEGER
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

fn embedded_capture_time(path: &Path) -> Option<SystemTime> {
    if crate::formats::is_raw(path) {
        raw_capture_time(path).or_else(|| exif_capture_time(path))
    } else {
        exif_capture_time(path)
    }
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
mod tests {
    use super::*;

    #[test]
    fn parses_fractional_xmp_capture_time() {
        assert!(parse_exif_time("2018:08:06 11:28:37.000").is_some());
        assert!(parse_exif_time("not a timestamp").is_none());
    }

    #[test]
    fn reads_capture_time_from_the_cr3_regression_sample() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test/IMG_0052.CR3");
        if path.exists() {
            assert!(embedded_capture_time(&path).is_some());
        }
    }

    #[test]
    fn xmp_is_authoritative_and_persistently_cached() {
        let root = tempfile::tempdir().unwrap();
        let cache = tempfile::tempdir().unwrap();
        let image = root.path().join("photo.nef");
        fs::write(&image, b"not actually a raw file").unwrap();
        fs::write(
            root.path().join("photo.nef.xmp"),
            br#"<rdf:Description exif:DateTimeOriginal="2018:08:06 11:28:37.000"/>"#,
        )
        .unwrap();
        let loader = MetadataLoader::new(cache.path().to_path_buf(), 1, JobLimiter::new(1));
        let result = loader.load_directory(vec![image.clone()], Arc::new(AtomicBool::new(false)));
        assert!(result[0].capture_time.is_some());
        assert!(
            cache
                .path()
                .join(format!("{CACHE_VERSION}.sqlite3"))
                .is_file()
        );

        // A cache hit must not attempt to decode the invalid source file.
        let result = loader.load_directory(vec![image.clone()], Arc::new(AtomicBool::new(false)));
        assert!(result[0].capture_time.is_some());
        let (cached, missing) =
            loader.load_cached_directory(vec![image], Arc::new(AtomicBool::new(false)));
        assert!(missing.is_empty());
        assert!(cached[0].capture_time.is_some());
    }
}
