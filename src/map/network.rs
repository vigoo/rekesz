use std::{
    fs,
    io::Read as _,
    path::Path,
    sync::OnceLock,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use reqwest::{StatusCode, blocking::Client, redirect::Policy};

use super::{OnlineMapAccess, PROVIDER_MAX_ZOOM, immutable_tile_template};

const MAX_TILE_BYTES: u64 = 8 * 1024 * 1024;
static SECURE_CLIENT: OnceLock<Result<Client, String>> = OnceLock::new();

#[derive(Clone, Copy, Debug, Default, serde::Deserialize, serde::Serialize)]
pub(crate) struct NetworkStatsSnapshot {
    pub hits: u64,
    pub misses: u64,
    pub downloads: u64,
    pub errors: u64,
    pub not_modified: u64,
}

#[derive(Clone, Debug, serde::Deserialize)]
pub(crate) struct TileJson {
    pub tiles: Vec<String>,
    pub maxzoom: u8,
    pub attribution: String,
}

impl TileJson {
    fn validate(self) -> Result<Self, String> {
        if self.tiles.len() != 1 || !immutable_tile_template(&self.tiles[0]) {
            return Err("OpenFreeMap returned an untrusted tile template".to_owned());
        }
        if self.maxzoom == 0 || f64::from(self.maxzoom) > PROVIDER_MAX_ZOOM {
            return Err("OpenFreeMap returned an unsupported maximum zoom".to_owned());
        }
        if self.attribution.trim().is_empty() {
            return Err("OpenFreeMap returned empty attribution".to_owned());
        }
        Ok(self)
    }

    pub fn tile_url(&self, zoom: u8, x: u32, y: u32) -> Result<String, String> {
        let url = self.tiles[0]
            .replace("{z}", &zoom.to_string())
            .replace("{x}", &x.to_string())
            .replace("{y}", &y.to_string());
        super::is_allowed_provider_resource(&url)
            .then_some(url)
            .ok_or_else(|| "tile URL escaped the OpenFreeMap allowlist".to_owned())
    }
}

pub(crate) fn fetch_tilejson(
    cache_path: &Path,
    capacity: u64,
    access: Option<OnlineMapAccess>,
) -> Result<Option<(TileJson, NetworkStatsSnapshot)>, String> {
    const MAX_TILEJSON_BYTES: u64 = 1024 * 1024;
    let endpoint = access
        .map(OnlineMapAccess::tilejson_endpoint)
        .unwrap_or(super::TILEJSON_ENDPOINT);
    let Some(resource) =
        load_resource_with_limit(cache_path, capacity, endpoint, access, MAX_TILEJSON_BYTES)?
    else {
        return Ok(None);
    };
    if resource.body.len() as u64 > MAX_TILEJSON_BYTES {
        return Err("OpenFreeMap TileJSON response is too large".to_owned());
    }
    let tilejson = serde_json::from_slice::<TileJson>(&resource.body)
        .map_err(|error| error.to_string())?
        .validate()?;
    Ok(Some((tilejson, resource.stats)))
}

#[derive(Debug)]
pub(crate) struct CachedResource {
    pub body: Vec<u8>,
    pub stats: NetworkStatsSnapshot,
}

struct NetworkResponse {
    status: StatusCode,
    final_url: String,
    etag: Option<String>,
    content_length: Option<u64>,
    body: Vec<u8>,
}

pub(crate) fn load_resource(
    cache_path: &Path,
    capacity: u64,
    url: &str,
    access: Option<OnlineMapAccess>,
) -> Result<Option<CachedResource>, String> {
    load_resource_with_limit(cache_path, capacity, url, access, MAX_TILE_BYTES)
}

fn load_resource_with_limit(
    cache_path: &Path,
    capacity: u64,
    url: &str,
    access: Option<OnlineMapAccess>,
    maximum_bytes: u64,
) -> Result<Option<CachedResource>, String> {
    load_resource_with_request(
        cache_path,
        capacity,
        url,
        access,
        maximum_bytes,
        request_resource,
    )
}

fn load_resource_with_request(
    cache_path: &Path,
    capacity: u64,
    url: &str,
    access: Option<OnlineMapAccess>,
    maximum_bytes: u64,
    request: impl Fn(OnlineMapAccess, &str, Option<&str>, u64) -> Result<NetworkResponse, String>,
) -> Result<Option<CachedResource>, String> {
    if !super::is_allowed_provider_resource(url) {
        return Err("map tile URL rejected by provider allowlist".to_owned());
    }
    if let Some(parent) = cache_path.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let mut connection = open_cache(cache_path)?;
    let elapsed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| error.to_string())?;
    let now = elapsed.as_secs() as i64;
    let accessed = elapsed.as_nanos().min(i64::MAX as u128) as i64;
    let cached = connection
        .query_row(
            "SELECT body, etag, expires FROM resources WHERE url = ?1",
            [url],
            |row| {
                Ok((
                    row.get::<_, Vec<u8>>(0)?,
                    row.get::<_, Option<String>>(1)?,
                    row.get::<_, i64>(2)?,
                ))
            },
        )
        .ok();
    if let Some((body, _, expires)) = &cached
        && (*expires > now || access.is_none())
    {
        connection
            .execute(
                "UPDATE resources SET accessed = ?1 WHERE url = ?2",
                (accessed, url),
            )
            .map_err(|error| error.to_string())?;
        return Ok(Some(CachedResource {
            body: body.clone(),
            stats: NetworkStatsSnapshot {
                hits: 1,
                ..Default::default()
            },
        }));
    }
    let Some(access) = access else {
        return Ok(None);
    };
    let cached_etag = cached.as_ref().and_then(|(_, etag, _)| etag.as_deref());
    let response = match request(access, url, cached_etag, maximum_bytes) {
        Ok(response) => response,
        Err(error) => {
            return cached.map_or_else(
                || Err(error.to_string()),
                |(body, _, _)| {
                    Ok(Some(CachedResource {
                        body,
                        stats: NetworkStatsSnapshot {
                            hits: 1,
                            errors: 1,
                            ..Default::default()
                        },
                    }))
                },
            );
        }
    };
    if (response.status.is_redirection() && response.status != StatusCode::NOT_MODIFIED)
        || response.final_url != url
        || !access.accepts_resource(&response.final_url)
    {
        return Err("map tile response rejected".to_owned());
    }
    if response.status == StatusCode::NOT_MODIFIED {
        let Some((body, _, _)) = cached else {
            return Err("received 304 without a cached map tile".to_owned());
        };
        connection
            .execute(
                "UPDATE resources SET expires = ?1, accessed = ?2 WHERE url = ?3",
                (now + 3600, accessed, url),
            )
            .map_err(|error| error.to_string())?;
        return Ok(Some(CachedResource {
            body,
            stats: NetworkStatsSnapshot {
                hits: 1,
                not_modified: 1,
                ..Default::default()
            },
        }));
    }
    if response.status != StatusCode::OK {
        return cached.map_or_else(
            || Err(format!("map tile request failed ({})", response.status)),
            |(body, _, _)| {
                Ok(Some(CachedResource {
                    body,
                    stats: NetworkStatsSnapshot {
                        hits: 1,
                        errors: 1,
                        ..Default::default()
                    },
                }))
            },
        );
    }
    if response
        .content_length
        .is_some_and(|size| size > maximum_bytes)
    {
        return Err("map tile response is too large".to_owned());
    }
    let NetworkResponse { etag, body, .. } = response;
    if body.len() as u64 > maximum_bytes {
        return Err("map tile response is too large".to_owned());
    }
    store_resource(&mut connection, url, &body, etag, now, accessed, capacity)?;
    Ok(Some(CachedResource {
        body,
        stats: NetworkStatsSnapshot {
            misses: 1,
            downloads: 1,
            ..Default::default()
        },
    }))
}

fn open_cache(cache_path: &Path) -> Result<rusqlite::Connection, String> {
    fn initialize(connection: &rusqlite::Connection) -> rusqlite::Result<()> {
        connection.execute_batch(
            "PRAGMA journal_mode=WAL;
             PRAGMA busy_timeout=5000;
             CREATE TABLE IF NOT EXISTS resources (
               url TEXT PRIMARY KEY,
               body BLOB NOT NULL,
               etag TEXT,
               expires INTEGER NOT NULL,
               accessed INTEGER NOT NULL,
               size INTEGER NOT NULL
             );",
        )?;
        connection.query_row("PRAGMA quick_check", [], |_| Ok(()))?;
        connection
            .prepare("SELECT url, body, etag, expires, accessed, size FROM resources LIMIT 0")?;
        Ok(())
    }

    let connection = rusqlite::Connection::open(cache_path).map_err(|error| error.to_string())?;
    if initialize(&connection).is_ok() {
        return Ok(connection);
    }
    drop(connection);
    for candidate in [
        cache_path.to_path_buf(),
        cache_path.with_extension("sqlite3-wal"),
        cache_path.with_extension("sqlite3-shm"),
    ] {
        match fs::remove_file(candidate) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.to_string()),
        }
    }
    let connection = rusqlite::Connection::open(cache_path).map_err(|error| error.to_string())?;
    initialize(&connection).map_err(|error| error.to_string())?;
    Ok(connection)
}

fn store_resource(
    connection: &mut rusqlite::Connection,
    url: &str,
    body: &[u8],
    etag: Option<String>,
    now: i64,
    accessed: i64,
    capacity: u64,
) -> Result<(), String> {
    let transaction = connection
        .transaction()
        .map_err(|error| error.to_string())?;
    transaction
        .execute(
            "INSERT OR REPLACE INTO resources(url, body, etag, expires, accessed, size)
             VALUES(?1, ?2, ?3, ?4, ?5, ?6)",
            (url, body, etag, now + 3600, accessed, body.len() as i64),
        )
        .map_err(|error| error.to_string())?;
    let mut total: i64 = transaction
        .query_row("SELECT COALESCE(SUM(size), 0) FROM resources", [], |row| {
            row.get(0)
        })
        .map_err(|error| error.to_string())?;
    while total > capacity.min(i64::MAX as u64) as i64 {
        let removed: i64 = transaction
            .query_row(
                "SELECT size FROM resources ORDER BY accessed ASC, url ASC LIMIT 1",
                [],
                |row| row.get(0),
            )
            .map_err(|error| error.to_string())?;
        transaction
            .execute(
                "DELETE FROM resources WHERE url IN
                 (SELECT url FROM resources ORDER BY accessed ASC, url ASC LIMIT 1)",
                [],
            )
            .map_err(|error| error.to_string())?;
        total -= removed;
    }
    transaction.commit().map_err(|error| error.to_string())?;
    Ok(())
}

fn request_resource(
    access: OnlineMapAccess,
    url: &str,
    etag: Option<&str>,
    maximum_bytes: u64,
) -> Result<NetworkResponse, String> {
    let client = secure_client(access)?;
    let mut request = client.get(url);
    if let Some(etag) = etag {
        request = request.header(reqwest::header::IF_NONE_MATCH, etag);
    }
    let response = request.send().map_err(|error| error.to_string())?;
    let status = response.status();
    let final_url = response.url().as_str().to_owned();
    let content_length = response.content_length();
    let etag = response
        .headers()
        .get(reqwest::header::ETAG)
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned);
    let mut body = Vec::new();
    response
        .take(maximum_bytes.saturating_add(1))
        .read_to_end(&mut body)
        .map_err(|error| error.to_string())?;
    Ok(NetworkResponse {
        status,
        final_url,
        etag,
        content_length,
        body,
    })
}

fn secure_client(access: OnlineMapAccess) -> Result<&'static Client, String> {
    SECURE_CLIENT
        .get_or_init(|| {
            Client::builder()
                .https_only(true)
                .redirect(Policy::none())
                .user_agent(access.user_agent())
                .connect_timeout(Duration::from_secs(3))
                .timeout(Duration::from_secs(5))
                .build()
                .map_err(|error| error.to_string())
        })
        .as_ref()
        .map_err(Clone::clone)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use test_r::test;

    fn enabled_access() -> OnlineMapAccess {
        OnlineMapAccess::from_permission(crate::config::OnlineMapPermission::Enabled).unwrap()
    }

    fn fake_response(status: StatusCode, url: &str, body: &[u8]) -> NetworkResponse {
        NetworkResponse {
            status,
            final_url: url.to_owned(),
            etag: Some("fixture-etag".to_owned()),
            content_length: Some(body.len() as u64),
            body: body.to_vec(),
        }
    }

    #[test]
    fn tilejson_rejects_mutable_or_untrusted_templates() {
        for template in [
            "https://tiles.openfreemap.org/planet/{z}/{x}/{y}.pbf",
            "https://evil.example/planet/20260823_080002_pt/{z}/{x}/{y}.pbf",
        ] {
            assert!(
                TileJson {
                    tiles: vec![template.to_owned()],
                    maxzoom: 14,
                    attribution: "required".to_owned(),
                }
                .validate()
                .is_err()
            );
        }
    }

    #[test]
    fn validated_tilejson_expands_only_allowlisted_urls() {
        let tilejson = TileJson {
            tiles: vec![
                "https://tiles.openfreemap.org/planet/20260823_080002_pt/{z}/{x}/{y}.pbf"
                    .to_owned(),
            ],
            maxzoom: 14,
            attribution: "required".to_owned(),
        }
        .validate()
        .unwrap();
        assert_eq!(
            tilejson.tile_url(8, 140, 91).unwrap(),
            "https://tiles.openfreemap.org/planet/20260823_080002_pt/8/140/91.pbf"
        );
    }

    #[test]
    fn offline_access_uses_expired_cache_without_network_permission() {
        let directory = tempfile::tempdir().unwrap();
        let cache = directory.path().join("map.sqlite3");
        assert!(
            load_resource(&cache, 1024, super::super::TILEJSON_ENDPOINT, None)
                .unwrap()
                .is_none()
        );
        let mut connection = rusqlite::Connection::open(&cache).unwrap();
        store_resource(
            &mut connection,
            super::super::TILEJSON_ENDPOINT,
            b"cached tilejson",
            Some("fixture-etag".to_owned()),
            1,
            1,
            1024,
        )
        .unwrap();
        connection
            .execute(
                "UPDATE resources SET expires = 0 WHERE url = ?1",
                [super::super::TILEJSON_ENDPOINT],
            )
            .unwrap();
        drop(connection);
        let cached = load_resource(&cache, 1024, super::super::TILEJSON_ENDPOINT, None)
            .unwrap()
            .unwrap();
        assert_eq!(cached.body, b"cached tilejson");
        assert_eq!(cached.stats.hits, 1);
        assert_eq!(cached.stats.downloads, 0);
    }

    #[test]
    fn cache_capacity_evicts_oldest_accessed_resource() {
        let directory = tempfile::tempdir().unwrap();
        let cache = directory.path().join("map.sqlite3");
        load_resource(&cache, 1024, super::super::TILEJSON_ENDPOINT, None).unwrap();
        let mut connection = rusqlite::Connection::open(&cache).unwrap();
        let urls = [
            "https://tiles.openfreemap.org/planet/20260823_080002_pt/1/0/0.pbf",
            "https://tiles.openfreemap.org/planet/20260823_080002_pt/1/0/1.pbf",
            "https://tiles.openfreemap.org/planet/20260823_080002_pt/1/1/0.pbf",
        ];
        for (accessed, url) in urls.iter().enumerate() {
            store_resource(
                &mut connection,
                url,
                &[accessed as u8; 3],
                None,
                accessed as i64,
                accessed as i64,
                6,
            )
            .unwrap();
        }
        let stored = connection
            .prepare("SELECT url FROM resources ORDER BY accessed")
            .unwrap()
            .query_map([], |row| row.get::<_, String>(0))
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(stored, urls[1..]);
    }

    #[test]
    fn cache_hit_refreshes_lru_with_subsecond_access_stamp() {
        let directory = tempfile::tempdir().unwrap();
        let cache = directory.path().join("map.sqlite3");
        load_resource(&cache, 1024, super::super::TILEJSON_ENDPOINT, None).unwrap();
        let urls = [
            super::super::TILEJSON_ENDPOINT,
            "https://tiles.openfreemap.org/planet/20260823_080002_pt/1/0/0.pbf",
            "https://tiles.openfreemap.org/planet/20260823_080002_pt/1/0/1.pbf",
        ];
        let mut connection = open_cache(&cache).unwrap();
        store_resource(&mut connection, urls[0], &[0], None, 1, 1, 2).unwrap();
        store_resource(&mut connection, urls[1], &[1], None, 1, 2, 2).unwrap();
        drop(connection);
        load_resource(&cache, 2, urls[0], None).unwrap().unwrap();
        let mut connection = open_cache(&cache).unwrap();
        store_resource(&mut connection, urls[2], &[2], None, 1, 3, 2).unwrap();
        let stored = connection
            .prepare("SELECT url FROM resources ORDER BY url")
            .unwrap()
            .query_map([], |row| row.get::<_, String>(0))
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert!(stored.contains(&urls[0].to_owned()));
        assert!(stored.contains(&urls[2].to_owned()));
        assert!(!stored.contains(&urls[1].to_owned()));
    }

    #[test]
    fn resource_larger_than_capacity_is_not_retained() {
        let directory = tempfile::tempdir().unwrap();
        let cache = directory.path().join("map.sqlite3");
        load_resource(&cache, 1024, super::super::TILEJSON_ENDPOINT, None).unwrap();
        let mut connection = rusqlite::Connection::open(&cache).unwrap();
        store_resource(
            &mut connection,
            super::super::TILEJSON_ENDPOINT,
            &[0; 8],
            None,
            1,
            1,
            7,
        )
        .unwrap();
        assert_eq!(
            connection
                .query_row("SELECT COUNT(*) FROM resources", [], |row| row
                    .get::<_, i64>(0))
                .unwrap(),
            0
        );
    }

    #[test]
    fn corrupt_cache_is_recreated_as_disposable_data() {
        let directory = tempfile::tempdir().unwrap();
        let cache = directory.path().join("map.sqlite3");
        fs::write(&cache, b"not a sqlite database").unwrap();
        assert!(
            load_resource(&cache, 1024, super::super::TILEJSON_ENDPOINT, None)
                .unwrap()
                .is_none()
        );
        let connection = rusqlite::Connection::open(cache).unwrap();
        assert_eq!(
            connection
                .query_row("SELECT COUNT(*) FROM resources", [], |row| row
                    .get::<_, i64>(0))
                .unwrap(),
            0
        );
    }

    #[test]
    fn concurrent_cache_writers_are_serialized_by_busy_timeout() {
        let directory = tempfile::tempdir().unwrap();
        let cache = directory.path().join("map.sqlite3");
        load_resource(&cache, 1024, super::super::TILEJSON_ENDPOINT, None).unwrap();
        std::thread::scope(|scope| {
            let handles = (0..8)
                .map(|index| {
                    let cache = cache.clone();
                    scope.spawn(move || {
                        let mut connection = open_cache(&cache).unwrap();
                        let url = format!(
                            "https://tiles.openfreemap.org/planet/20260823_080002_pt/4/{index}/0.pbf"
                        );
                        store_resource(
                            &mut connection,
                            &url,
                            &[index as u8],
                            None,
                            index,
                            index,
                            1024,
                        )
                    })
                })
                .collect::<Vec<_>>();
            for handle in handles {
                handle.join().unwrap().unwrap();
            }
        });
        let connection = open_cache(&cache).unwrap();
        assert_eq!(
            connection
                .query_row("SELECT COUNT(*) FROM resources", [], |row| row
                    .get::<_, i64>(0))
                .unwrap(),
            8
        );
    }

    #[test]
    fn deterministic_http_download_and_304_revalidation() {
        let directory = tempfile::tempdir().unwrap();
        let cache = directory.path().join("map.sqlite3");
        let url = super::super::TILEJSON_ENDPOINT;
        let downloaded = load_resource_with_request(
            &cache,
            1024,
            url,
            Some(enabled_access()),
            1024,
            |_, requested, etag, _| {
                assert_eq!(requested, url);
                assert_eq!(etag, None);
                Ok(fake_response(StatusCode::OK, url, b"fixture"))
            },
        )
        .unwrap()
        .unwrap();
        assert_eq!(downloaded.stats.downloads, 1);
        let connection = open_cache(&cache).unwrap();
        connection
            .execute("UPDATE resources SET expires = 0", [])
            .unwrap();
        drop(connection);
        let revalidated = load_resource_with_request(
            &cache,
            1024,
            url,
            Some(enabled_access()),
            1024,
            |_, _, etag, _| {
                assert_eq!(etag, Some("fixture-etag"));
                Ok(fake_response(StatusCode::NOT_MODIFIED, url, &[]))
            },
        )
        .unwrap()
        .unwrap();
        assert_eq!(revalidated.body, b"fixture");
        assert_eq!(revalidated.stats.hits, 1);
        assert_eq!(revalidated.stats.not_modified, 1);
    }

    #[test]
    fn deterministic_http_failures_fall_back_only_when_cached() {
        let directory = tempfile::tempdir().unwrap();
        let cache = directory.path().join("map.sqlite3");
        let url = super::super::TILEJSON_ENDPOINT;
        let error = load_resource_with_request(
            &cache,
            1024,
            url,
            Some(enabled_access()),
            1024,
            |_, _, _, _| Err("offline".to_owned()),
        )
        .unwrap_err();
        assert_eq!(error, "offline");
        let mut connection = open_cache(&cache).unwrap();
        store_resource(&mut connection, url, b"stale", None, 1, 1, 1024).unwrap();
        connection
            .execute("UPDATE resources SET expires = 0", [])
            .unwrap();
        drop(connection);
        for status in [StatusCode::INTERNAL_SERVER_ERROR, StatusCode::NOT_FOUND] {
            let fallback = load_resource_with_request(
                &cache,
                1024,
                url,
                Some(enabled_access()),
                1024,
                |_, _, _, _| Ok(fake_response(status, url, b"error")),
            )
            .unwrap()
            .unwrap();
            assert_eq!(fallback.body, b"stale");
            assert_eq!(fallback.stats.hits, 1);
            assert_eq!(fallback.stats.errors, 1);
        }
    }

    #[test]
    fn deterministic_http_rejects_redirects_and_oversized_bodies() {
        let directory = tempfile::tempdir().unwrap();
        let cache = directory.path().join("map.sqlite3");
        let url = super::super::TILEJSON_ENDPOINT;
        for response in [
            fake_response(StatusCode::FOUND, url, &[]),
            fake_response(
                StatusCode::OK,
                "https://tiles.openfreemap.org/planet/other",
                &[],
            ),
            fake_response(StatusCode::OK, url, &[0; 9]),
        ] {
            assert!(
                load_resource_with_request(
                    &cache,
                    1024,
                    url,
                    Some(enabled_access()),
                    8,
                    |_, _, _, _| Ok(NetworkResponse {
                        status: response.status,
                        final_url: response.final_url.clone(),
                        etag: response.etag.clone(),
                        content_length: response.content_length,
                        body: response.body.clone(),
                    }),
                )
                .is_err()
            );
        }
        let declared = NetworkResponse {
            status: StatusCode::OK,
            final_url: url.to_owned(),
            etag: None,
            content_length: Some(9),
            body: vec![0; 1],
        };
        assert!(
            load_resource_with_request(
                &cache,
                1024,
                url,
                Some(enabled_access()),
                8,
                |_, _, _, _| Ok(NetworkResponse {
                    status: declared.status,
                    final_url: declared.final_url.clone(),
                    etag: None,
                    content_length: declared.content_length,
                    body: declared.body.clone(),
                }),
            )
            .is_err()
        );
    }

    #[test]
    fn cache_only_mode_never_invokes_the_http_transport() {
        let directory = tempfile::tempdir().unwrap();
        let calls = Cell::new(0);
        assert!(
            load_resource_with_request(
                &directory.path().join("map.sqlite3"),
                1024,
                super::super::TILEJSON_ENDPOINT,
                None,
                1024,
                |_, _, _, _| {
                    calls.set(calls.get() + 1);
                    Err("must not run".to_owned())
                },
            )
            .unwrap()
            .is_none()
        );
        assert_eq!(calls.get(), 0);
    }
}
