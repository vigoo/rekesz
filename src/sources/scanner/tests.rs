//! Scanner source tests: discovery normalization, capabilities parsing
//! against the captured device fixtures, the settings writer, region math,
//! and the job state machine against a mock eSCL server. No test touches
//! the network or a real device.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use super::discover::{
    DiscoveredService, SERVICE_PLAIN, SERVICE_TLS, device_key, endpoint_base_url, expire_devices,
    instance_name, note_service,
};
use super::escl::{
    ColorMode, EsclClient, InputCaps, NextDocument, ScanJobError, ScanJobTimings, ScanRegion,
    ScanSettings, UNITS_PER_INCH, job_path_from_location, mm_to_units, parse_capabilities,
    parse_job_state, region_from_fractions, run_scan_job, units_to_mm,
};
use super::mock::{MockReply, spawn_mock};
use super::{ScannerProvider, prepare_session_dir, session_scan_dir, split_name};

// The binary's test harness is test-r; plain `#[test]` would never register.
use test_r::test;

fn platen_caps() -> InputCaps {
    InputCaps {
        min_width: 96,
        max_width: 2550,
        min_height: 96,
        max_height: 3510,
    }
}

fn service(
    service_type: &str,
    uuid: &str,
    port: u16,
    address: std::net::IpAddr,
) -> DiscoveredService {
    DiscoveredService {
        service_type: service_type.to_string(),
        instance_name: "EPSON L3160 Series".to_string(),
        port,
        addresses: vec![address],
        txt: vec![
            ("rs".to_string(), "eSCL".to_string()),
            ("ty".to_string(), "EPSON L3160 Series".to_string()),
            ("uuid".to_string(), format!("uuid:{uuid}")),
        ],
    }
}

#[test]
fn device_key_strips_uuid_prefix() {
    let txt = vec![("uuid".to_string(), "uuid:cfe9-1104".to_string())];
    assert_eq!(&*device_key(&txt, "EPSON L3160 Series"), "cfe9-1104");
}

#[test]
fn device_key_falls_back_to_instance_name() {
    let txt: Vec<(String, String)> = Vec::new();
    assert_eq!(
        &*device_key(&txt, "EPSON L3160 Series"),
        "name:EPSON L3160 Series"
    );
}

#[test]
fn resource_prefix_is_trimmed_and_defaults() {
    let prefixed = vec![("rs".to_string(), "/eSCL/".to_string())];
    assert_eq!(super::discover::resource_prefix(&prefixed), "eSCL");
    assert_eq!(super::discover::resource_prefix(&[]), "eSCL");
}

#[test]
fn endpoint_base_url_prefers_ipv4_and_carries_the_port() {
    let addresses = ["fd00::1".parse().unwrap(), "192.0.2.10".parse().unwrap()];
    let url = endpoint_base_url(true, &addresses, 443, "eSCL").unwrap();
    assert_eq!(url, "https://192.0.2.10:443/eSCL");
    let v6_only = ["fd00::1".parse().unwrap()];
    let url = endpoint_base_url(false, &v6_only, 80, "eSCL").unwrap();
    assert_eq!(url, "http://[fd00::1]:80/eSCL");
    assert_eq!(endpoint_base_url(true, &[], 443, "eSCL"), None);
}

#[test]
fn note_service_merges_transports_and_tls_sorts_first() {
    let mut map = HashMap::new();
    let now = Instant::now();
    note_service(
        &mut map,
        &service(SERVICE_PLAIN, "u1", 80, "192.0.2.10".parse().unwrap()),
        now,
    );
    note_service(
        &mut map,
        &service(SERVICE_TLS, "u1", 443, "192.0.2.10".parse().unwrap()),
        now,
    );
    assert_eq!(map.len(), 1);
    let device = map.values().next().unwrap();
    assert_eq!(device.name, "EPSON L3160 Series");
    assert_eq!(device.endpoints.len(), 2);
    assert!(device.endpoints[0].tls, "TLS endpoint must sort first");
    assert_eq!(device.endpoints[0].base_url, "https://192.0.2.10:443/eSCL");
    assert!(device.preferred_endpoint().unwrap().tls);
}

#[test]
fn note_service_re_seeing_refreshes_and_moves_addresses() {
    let mut map = HashMap::new();
    let now = Instant::now();
    note_service(
        &mut map,
        &service(SERVICE_TLS, "u1", 443, "192.0.2.10".parse().unwrap()),
        now,
    );
    std::thread::sleep(Duration::from_millis(2));
    note_service(
        &mut map,
        &service(SERVICE_TLS, "u1", 443, "192.0.2.20".parse().unwrap()),
        Instant::now(),
    );
    let device = map.values().next().unwrap();
    assert_eq!(device.endpoints.len(), 1, "one endpoint per transport");
    assert!(device.endpoints[0].base_url.contains("192.0.2.20"));
    assert!(device.last_seen > now);
}

#[test]
fn expire_devices_drops_only_stale_entries() {
    let mut map = HashMap::new();
    let now = Instant::now();
    note_service(
        &mut map,
        &service(SERVICE_TLS, "fresh", 443, "192.0.2.10".parse().unwrap()),
        now,
    );
    let stale_past = now - Duration::from_secs(60);
    note_service(
        &mut map,
        &service(SERVICE_TLS, "stale", 443, "192.0.2.20".parse().unwrap()),
        stale_past,
    );
    expire_devices(&mut map, Instant::now());
    assert_eq!(map.len(), 1);
    assert!(map.contains_key("fresh"));
}

#[test]
fn session_scan_dir_is_per_device_and_per_run() {
    let cache = tempfile::tempdir().unwrap();
    let dir = session_scan_dir(cache.path(), "cfe9-1104");
    assert!(dir.starts_with(cache.path().join("scan-staging")));
    assert!(dir.to_string_lossy().contains("cfe9-1104"));
    assert_eq!(&*dir, &*session_scan_dir(cache.path(), "cfe9-1104"));
    let other = session_scan_dir(cache.path(), "other-device");
    assert_ne!(dir, other);
}

#[test]
fn split_name_separates_vendor_and_model() {
    assert_eq!(
        split_name("EPSON L3160 Series"),
        ("EPSON".to_string(), "L3160 Series".to_string())
    );
    assert_eq!(split_name("Weird"), ("Weird".to_string(), String::new()));
}

const CAPABILITIES_XML: &str = include_str!("testdata/capabilities.xml");

#[test]
fn capabilities_parse_the_captured_device_document() {
    let caps = parse_capabilities(CAPABILITIES_XML).unwrap();
    assert_eq!(caps.platen, platen_caps());
    assert_eq!(caps.max_scan_regions, 1);
    assert_eq!(
        caps.color_modes,
        vec![
            "BlackAndWhite1".to_string(),
            "Grayscale8".to_string(),
            "RGB24".to_string()
        ]
    );
    assert!(caps.document_formats.contains(&"image/jpeg".to_string()));
    assert_eq!(caps.resolutions, vec![100, 200, 300, 600, 1200]);
}

/// The same document without namespace prefixes: devices vary between
/// prefixed and default-namespaced forms and the parser must accept both.
#[test]
fn capabilities_parse_without_namespace_prefixes() {
    let stripped = CAPABILITIES_XML
        .replace("<scan:", "<")
        .replace("</scan:", "</")
        .replace("<pwg:", "<")
        .replace("</pwg:", "</")
        .replace("xmlns:scan=", "xmlns=")
        .replace("xmlns:pwg=", "xmlns=");
    let caps = parse_capabilities(&stripped).unwrap();
    assert_eq!(caps.platen, platen_caps());
    assert_eq!(caps.resolutions, vec![100, 200, 300, 600, 1200]);
}

#[test]
fn capabilities_reject_devices_without_jpeg() {
    let xml = r#"<scan:ScannerCapabilities xmlns:scan="x" xmlns:pwg="y">
        <scan:Platen><scan:PlatenInputCaps>
            <scan:MinWidth>10</scan:MinWidth><scan:MaxWidth>100</scan:MaxWidth>
            <scan:MinHeight>10</scan:MinHeight><scan:MaxHeight>100</scan:MaxHeight>
            <scan:SettingProfiles><scan:SettingProfile>
                <scan:DocumentFormats><pwg:DocumentFormat>application/pdf</pwg:DocumentFormat></scan:DocumentFormats>
            </scan:SettingProfile></scan:SettingProfiles>
        </scan:PlatenInputCaps></scan:Platen>
    </scan:ScannerCapabilities>"#;
    let error = parse_capabilities(xml).unwrap_err();
    assert!(matches!(
        error,
        super::escl::EsclError::UnsupportedDevice(_)
    ));
}

#[test]
fn capabilities_expand_resolution_ranges() {
    let xml = r#"<ScannerCapabilities>
        <Platen><PlatenInputCaps>
            <MinWidth>10</MinWidth><MaxWidth>100</MaxWidth>
            <MinHeight>10</MinHeight><MaxHeight>100</MaxHeight>
            <SettingProfiles><SettingProfile>
                <DocumentFormats><DocumentFormat>image/jpeg</DocumentFormat></DocumentFormats>
                <SupportedResolutions><ResolutionRange>
                    <Min>50</Min><Max>250</Max><Step>50</Step>
                </ResolutionRange></SupportedResolutions>
            </SettingProfile></SettingProfiles>
        </PlatenInputCaps></Platen>
    </ScannerCapabilities>"#;
    let caps = parse_capabilities(xml).unwrap();
    assert_eq!(caps.resolutions, vec![50, 100, 150, 200, 250]);
}

#[test]
fn settings_xml_matches_the_live_validated_payload() {
    let settings = ScanSettings {
        region: ScanRegion {
            x_offset: 0,
            y_offset: 0,
            width: 2550,
            height: 3510,
        },
        color_mode: ColorMode::Rgb24,
        document_format: "image/jpeg".to_string(),
        resolution: 300,
    };
    assert_eq!(
        settings.to_xml(),
        include_str!("testdata/settings-golden.xml")
    );
}

#[test]
fn settings_xml_writes_grayscale8_spelling() {
    let settings = ScanSettings {
        region: ScanRegion::full_platen(&platen_caps()),
        color_mode: ColorMode::Grayscale8,
        document_format: "image/jpeg".to_string(),
        resolution: 300,
    };
    assert!(
        settings
            .to_xml()
            .contains("<scan:ColorMode>Grayscale8</scan:ColorMode>")
    );
    assert!(!settings.to_xml().contains("Gray8<"));
    let binary = ScanSettings {
        color_mode: ColorMode::BlackAndWhite1,
        ..settings
    };
    assert!(binary.to_xml().contains(">BlackAndWhite1<"));
}

#[test]
fn region_math_converts_units_and_millimetres() {
    // 1/300 inch: 300 units = 25.4 mm exactly.
    assert!((units_to_mm(300.0) - 25.4).abs() < 1e-9);
    assert!((mm_to_units(25.4) - UNITS_PER_INCH).abs() < 1e-9);
    assert!((units_to_mm(mm_to_units(210.0)) - 210.0).abs() < 1e-9);
}

#[test]
fn region_from_fractions_clamps_and_is_corner_order_independent() {
    let caps = platen_caps();
    let full = region_from_fractions(&caps, 0.0, 0.0, 1.0, 1.0);
    assert_eq!(
        full,
        ScanRegion {
            x_offset: 0,
            y_offset: 0,
            width: 2550,
            height: 3510
        }
    );
    let a = region_from_fractions(&caps, 0.1, 0.2, 0.5, 0.6);
    let b = region_from_fractions(&caps, 0.5, 0.6, 0.1, 0.2);
    assert_eq!(a, b, "opposite drag corners must give the same region");
    assert_eq!(a.width, (0.4_f64 * 2550.0).round() as u32);
    assert_eq!(a.x_offset, (0.1_f64 * 2550.0).round() as u32);
    // A too-small selection clamps up to the advertised minimum; a
    // bottom-right drag cannot push the region past the platen.
    let tiny = region_from_fractions(&caps, 0.999, 0.999, 1.0, 1.0);
    assert_eq!(tiny.width, caps.min_width);
    assert_eq!(tiny.x_offset, caps.max_width - caps.min_width);
    let inverted = region_from_fractions(&caps, 1.2, -0.5, -0.2, 0.8);
    assert_eq!(inverted.x_offset, 0);
    assert_eq!(inverted.width, caps.max_width);
}

#[test]
fn job_path_from_location_keeps_only_the_path() {
    assert_eq!(
        job_path_from_location("https://192.0.2.10:443/eSCL/ScanJobs/abc"),
        "/eSCL/ScanJobs/abc"
    );
    // A hostile authority is discarded; only the path survives.
    assert_eq!(
        job_path_from_location("http://evil.example.com:9/eSCL/ScanJobs/hij"),
        "/eSCL/ScanJobs/hij"
    );
    assert_eq!(
        job_path_from_location("/eSCL/ScanJobs/abc"),
        "/eSCL/ScanJobs/abc"
    );
    assert_eq!(
        job_path_from_location("eSCL/ScanJobs/abc"),
        "/eSCL/ScanJobs/abc"
    );
}

#[test]
fn job_state_parses_from_prefixed_and_plain_bodies() {
    assert_eq!(
        parse_job_state(
            "<scan:JobStatus xmlns:scan=\"x\"><pwg:JobState>Processing</pwg:JobState></scan:JobStatus>"
        ),
        Some("Processing".to_string())
    );
    assert_eq!(
        parse_job_state("<JobStatus><JobState>Completed</JobState></JobStatus>"),
        Some("Completed".to_string())
    );
    assert_eq!(parse_job_state("<nothing/>"), None);
    assert_eq!(parse_job_state("not xml at all"), None);
}

fn test_settings() -> ScanSettings {
    ScanSettings {
        region: ScanRegion::full_platen(&platen_caps()),
        color_mode: ColorMode::Rgb24,
        document_format: "image/jpeg".to_string(),
        resolution: 300,
    }
}

fn test_timings() -> ScanJobTimings {
    ScanJobTimings {
        guard: Duration::from_secs(10),
        total: Duration::from_secs(30),
        post_retry_pause: Duration::from_millis(1),
        poll_pause: Duration::from_millis(1),
    }
}

fn fake_jpeg() -> Vec<u8> {
    (0..64 * 1024_u32).map(|index| index as u8).collect()
}

#[test]
fn scan_job_happy_path_posts_fetches_and_deletes() {
    let deleted = Arc::new(AtomicBool::new(false));
    let deleted_clone = Arc::clone(&deleted);
    let (base, recorded) = spawn_mock(Box::new(move |method, path, _host, _body| {
        match (method, path) {
            ("POST", "/eSCL/ScanJobs") => MockReply::created("/eSCL/ScanJobs/abc"),
            ("GET", "/eSCL/ScanJobs/abc/NextDocument") => {
                MockReply::image("image/jpeg", fake_jpeg())
            }
            ("DELETE", "/eSCL/ScanJobs/abc") => {
                deleted_clone.store(true, Ordering::Release);
                MockReply::status(200)
            }
            _ => MockReply::status(404),
        }
    }));
    let client = EsclClient::new(&base).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let destination = dir.path().join("scan.jpg");
    let cancel = AtomicBool::new(false);
    let size = run_scan_job(
        &client,
        &test_settings(),
        &destination,
        &cancel,
        &test_timings(),
        &mut |_| {},
    )
    .unwrap();
    assert_eq!(size, fake_jpeg().len() as u64);
    assert_eq!(std::fs::metadata(&destination).unwrap().len(), size as u64);
    assert!(
        !destination.with_extension("rekesz-part").exists(),
        "part file must be renamed"
    );
    assert!(
        deleted.load(Ordering::Acquire),
        "the job must be DELETEd on success"
    );
    let recorded = recorded.lock().unwrap();
    assert!(
        recorded
            .iter()
            .any(|(method, path, _)| method == "POST" && path == "/eSCL/ScanJobs")
    );
    let (_, _, host) = recorded.first().unwrap();
    // The Host header carries the explicit port: eSCL devices (Epson
    // especially) are picky about it.
    assert!(
        host.contains(':'),
        "Host header must carry the port, got {host}"
    );
}

#[test]
fn scan_job_retries_busy_post_then_succeeds() {
    let posts = Arc::new(AtomicUsize::new(0));
    let posts_clone = Arc::clone(&posts);
    let (base, _recorded) = spawn_mock(Box::new(move |method, path, _host, _body| {
        match (method, path) {
            ("POST", "/eSCL/ScanJobs") => {
                let attempt = posts_clone.fetch_add(1, Ordering::AcqRel);
                if attempt < 2 {
                    MockReply::status(503)
                } else {
                    MockReply::created("/eSCL/ScanJobs/busy-ok")
                }
            }
            ("GET", "/eSCL/ScanJobs/busy-ok/NextDocument") => {
                MockReply::image("image/jpeg", vec![1, 2, 3])
            }
            _ => MockReply::status(404),
        }
    }));
    let client = EsclClient::new(&base).unwrap();
    let cancel = AtomicBool::new(false);
    let size = run_scan_job(
        &client,
        &test_settings(),
        &tempfile::tempdir().unwrap().path().join("scan.jpg"),
        &cancel,
        &test_timings(),
        &mut |_| {},
    )
    .unwrap();
    assert_eq!(size, 3);
    assert_eq!(posts.load(Ordering::Acquire), 3);
}

#[test]
fn scan_job_surfaces_busy_after_exhausting_post_retries() {
    let (base, _recorded) = spawn_mock(Box::new(|method, path, _host, _body| {
        match (method, path) {
            ("POST", "/eSCL/ScanJobs") => MockReply::status(503),
            _ => MockReply::status(404),
        }
    }));
    let client = EsclClient::new(&base).unwrap();
    let cancel = AtomicBool::new(false);
    let error = run_scan_job(
        &client,
        &test_settings(),
        &tempfile::tempdir().unwrap().path().join("scan.jpg"),
        &cancel,
        &test_timings(),
        &mut |_| {},
    )
    .unwrap_err();
    assert!(matches!(error, ScanJobError::Busy));
}

#[test]
fn scan_job_maps_409_to_rejected_settings() {
    let (base, _recorded) = spawn_mock(Box::new(|method, path, _host, _body| {
        match (method, path) {
            ("POST", "/eSCL/ScanJobs") => MockReply::status(409),
            _ => MockReply::status(404),
        }
    }));
    let client = EsclClient::new(&base).unwrap();
    let cancel = AtomicBool::new(false);
    let error = run_scan_job(
        &client,
        &test_settings(),
        &tempfile::tempdir().unwrap().path().join("scan.jpg"),
        &cancel,
        &test_timings(),
        &mut |_| {},
    )
    .unwrap_err();
    let ScanJobError::Failed(message) = error else {
        panic!("expected Failed, got {error:?}");
    };
    assert!(message.contains("rejected"), "got: {message}");
}

#[test]
fn scan_job_polls_through_busy_document_responses() {
    let polls = Arc::new(AtomicUsize::new(0));
    let polls_clone = Arc::clone(&polls);
    let (base, _recorded) = spawn_mock(Box::new(move |method, path, _host, _body| {
        match (method, path) {
            ("POST", "/eSCL/ScanJobs") => MockReply::created("/eSCL/ScanJobs/j"),
            ("GET", "/eSCL/ScanJobs/j/NextDocument") => {
                let attempt = polls_clone.fetch_add(1, Ordering::AcqRel);
                if attempt < 2 {
                    MockReply::status(503)
                } else {
                    MockReply::image("image/jpeg", vec![9, 9])
                }
            }
            _ => MockReply::status(404),
        }
    }));
    let client = EsclClient::new(&base).unwrap();
    let cancel = AtomicBool::new(false);
    let size = run_scan_job(
        &client,
        &test_settings(),
        &tempfile::tempdir().unwrap().path().join("scan.jpg"),
        &cancel,
        &test_timings(),
        &mut |_| {},
    )
    .unwrap();
    assert_eq!(size, 2);
    assert_eq!(polls.load(Ordering::Acquire), 3);
}

#[test]
fn scan_job_fails_when_the_job_ends_without_a_document() {
    let (base, _recorded) = spawn_mock(Box::new(|method, path, _host, _body| {
        match (method, path) {
            ("POST", "/eSCL/ScanJobs") => MockReply::created("/eSCL/ScanJobs/gone"),
            ("GET", "/eSCL/ScanJobs/gone/NextDocument") => MockReply::status(404),
            _ => MockReply::status(404),
        }
    }));
    let client = EsclClient::new(&base).unwrap();
    let cancel = AtomicBool::new(false);
    let error = run_scan_job(
        &client,
        &test_settings(),
        &tempfile::tempdir().unwrap().path().join("scan.jpg"),
        &cancel,
        &test_timings(),
        &mut |_| {},
    )
    .unwrap_err();
    let ScanJobError::Failed(message) = error else {
        panic!("expected Failed, got {error:?}");
    };
    assert!(message.contains("ended the job"), "got: {message}");
}

#[test]
fn scan_job_treats_xml_document_bodies_as_status_not_image() {
    let polls = Arc::new(AtomicUsize::new(0));
    let polls_clone = Arc::clone(&polls);
    let (base, _recorded) = spawn_mock(Box::new(move |method, path, _host, _body| {
        match (method, path) {
            ("POST", "/eSCL/ScanJobs") => MockReply::created("/eSCL/ScanJobs/xml"),
            ("GET", "/eSCL/ScanJobs/xml/NextDocument") => {
                let attempt = polls_clone.fetch_add(1, Ordering::AcqRel);
                if attempt < 1 {
                    MockReply::xml(
                        "<pwg:JobStatus><pwg:JobState>Processing</pwg:JobState></pwg:JobStatus>",
                    )
                } else {
                    MockReply::image("image/jpeg", vec![7])
                }
            }
            _ => MockReply::status(404),
        }
    }));
    let client = EsclClient::new(&base).unwrap();
    let cancel = AtomicBool::new(false);
    let size = run_scan_job(
        &client,
        &test_settings(),
        &tempfile::tempdir().unwrap().path().join("scan.jpg"),
        &cancel,
        &test_timings(),
        &mut |_| {},
    )
    .unwrap();
    assert_eq!(size, 1, "the XML status body must not become the image");
}

#[test]
fn scan_job_accepts_chunked_document_transfers() {
    let (base, _recorded) = spawn_mock(Box::new(|method, path, _host, _body| {
        match (method, path) {
            ("POST", "/eSCL/ScanJobs") => MockReply::created("/eSCL/ScanJobs/chunked"),
            ("GET", "/eSCL/ScanJobs/chunked/NextDocument") => {
                MockReply::image("image/jpeg", fake_jpeg()).chunked()
            }
            _ => MockReply::status(404),
        }
    }));
    let client = EsclClient::new(&base).unwrap();
    let cancel = AtomicBool::new(false);
    let dir = tempfile::tempdir().unwrap();
    let destination = dir.path().join("scan.jpg");
    let size = run_scan_job(
        &client,
        &test_settings(),
        &destination,
        &cancel,
        &test_timings(),
        &mut |_| {},
    )
    .unwrap();
    assert_eq!(
        size,
        fake_jpeg().len() as u64,
        "size comes from the file, not a declared length"
    );
}

#[test]
fn scan_job_resolves_job_paths_to_its_own_origin() {
    let (base, recorded) = spawn_mock(Box::new(|method, path, _host, _body| {
        match (method, path) {
            ("POST", "/eSCL/ScanJobs") => {
                MockReply::created("http://evil.example.com:9/eSCL/ScanJobs/hij")
            }
            ("GET", "/eSCL/ScanJobs/hij/NextDocument") => MockReply::image("image/jpeg", vec![1]),
            _ => MockReply::status(404),
        }
    }));
    let client = EsclClient::new(&base).unwrap();
    let cancel = AtomicBool::new(false);
    run_scan_job(
        &client,
        &test_settings(),
        &tempfile::tempdir().unwrap().path().join("scan.jpg"),
        &cancel,
        &test_timings(),
        &mut |_| {},
    )
    .unwrap();
    let recorded = recorded.lock().unwrap();
    assert!(
        recorded
            .iter()
            .any(|(method, path, _)| method == "GET" && path == "/eSCL/ScanJobs/hij/NextDocument"),
        "the document request must land on our origin, path-only: {recorded:?}"
    );
}

#[test]
fn scan_job_cancels_with_a_delete_and_no_leftover_file() {
    let deleted = Arc::new(AtomicBool::new(false));
    let deleted_clone = Arc::clone(&deleted);
    let (base, _recorded) = spawn_mock(Box::new(move |method, path, _host, _body| {
        match (method, path) {
            ("POST", "/eSCL/ScanJobs") => MockReply::created("/eSCL/ScanJobs/c"),
            ("GET", "/eSCL/ScanJobs/c/NextDocument") => MockReply::status(503),
            ("DELETE", "/eSCL/ScanJobs/c") => {
                deleted_clone.store(true, Ordering::Release);
                MockReply::status(200)
            }
            _ => MockReply::status(404),
        }
    }));
    let client = EsclClient::new(&base).unwrap();
    let cancel = Arc::new(AtomicBool::new(false));
    let dir = tempfile::tempdir().unwrap();
    let destination = dir.path().join("scan.jpg");
    let handle = {
        let client = client.clone();
        let settings = test_settings();
        let cancel = Arc::clone(&cancel);
        let destination = destination.clone();
        let timings = ScanJobTimings {
            guard: Duration::from_secs(30),
            total: Duration::from_secs(60),
            post_retry_pause: Duration::from_millis(1),
            poll_pause: Duration::from_millis(2),
        };
        std::thread::spawn(move || {
            run_scan_job(
                &client,
                &settings,
                &destination,
                &cancel,
                &timings,
                &mut |_| {},
            )
        })
    };
    std::thread::sleep(Duration::from_millis(60));
    cancel.store(true, Ordering::Release);
    let error = handle.join().unwrap().unwrap_err();
    assert!(matches!(error, ScanJobError::Canceled));
    assert!(
        deleted.load(Ordering::Acquire),
        "cancel must DELETE the job"
    );
    assert!(!destination.exists());
    assert!(
        !destination.with_extension("rekesz-part").exists(),
        "cancel must remove the part file"
    );
}

#[test]
fn scan_job_times_out_when_the_device_never_delivers() {
    let deleted = Arc::new(AtomicBool::new(false));
    let deleted_clone = Arc::clone(&deleted);
    let (base, _recorded) = spawn_mock(Box::new(move |method, path, _host, _body| {
        match (method, path) {
            ("POST", "/eSCL/ScanJobs") => MockReply::created("/eSCL/ScanJobs/slow"),
            ("GET", "/eSCL/ScanJobs/slow/NextDocument") => MockReply::status(503),
            ("DELETE", "/eSCL/ScanJobs/slow") => {
                deleted_clone.store(true, Ordering::Release);
                MockReply::status(200)
            }
            _ => MockReply::status(404),
        }
    }));
    let client = EsclClient::new(&base).unwrap();
    let cancel = AtomicBool::new(false);
    let error = run_scan_job(
        &client,
        &test_settings(),
        &tempfile::tempdir().unwrap().path().join("scan.jpg"),
        &cancel,
        &ScanJobTimings {
            // Every poll answer counts as activity, so the idle guard never
            // fires; the overall budget is what ends this job.
            guard: Duration::from_secs(30),
            total: Duration::from_millis(150),
            post_retry_pause: Duration::from_millis(1),
            poll_pause: Duration::from_millis(5),
        },
        &mut |_| {},
    )
    .unwrap_err();
    assert!(matches!(error, ScanJobError::TimedOut));
    assert!(
        deleted.load(Ordering::Acquire),
        "timeout must DELETE the job"
    );
}

#[test]
fn next_document_reports_job_status_from_prefixed_xml() {
    let (base, _recorded) = spawn_mock(Box::new(|method, path, _host, _body| {
        match (method, path) {
            ("POST", "/eSCL/ScanJobs") => MockReply::created("/eSCL/ScanJobs/st"),
            ("GET", "/eSCL/ScanJobs/st/NextDocument") => MockReply::xml(
                "<scan:JobStatus xmlns:scan=\"x\"><pwg:JobState>Processing</pwg:JobState></scan:JobStatus>",
            ),
            _ => MockReply::status(404),
        }
    }));
    let client = EsclClient::new(&base).unwrap();
    let cancel = AtomicBool::new(false);
    let mut sink = Vec::new();
    let result = client
        .next_document("/eSCL/ScanJobs/st", &mut sink, &cancel, &mut |_| {})
        .unwrap();
    assert_eq!(
        result,
        NextDocument::JobStatus(Some("Processing".to_string())),
        "an XML body is status, never image bytes"
    );
    assert!(sink.is_empty());
}

#[test]
fn capabilities_request_parses_the_mock_document() {
    let body = CAPABILITIES_XML.to_string();
    let (base, _recorded) = spawn_mock(Box::new(move |method, path, _host, _body| {
        match (method, path) {
            ("GET", "/eSCL/ScannerCapabilities") => {
                MockReply::image("text/xml", body.clone().into_bytes())
            }
            _ => MockReply::status(404),
        }
    }));
    let client = EsclClient::new(&base).unwrap();
    let caps = client.capabilities().unwrap();
    assert_eq!(caps.platen, platen_caps());
}

#[test]
fn provider_collects_snapshot_entries_without_touching_the_filesystem() {
    let cache = tempfile::tempdir().unwrap();
    // The pure collect path is tested against a local snapshot; the daemon
    // and its network probes are never started from tests.
    let mut map = HashMap::new();
    note_service(
        &mut map,
        &service(
            SERVICE_TLS,
            "provider-test",
            443,
            "192.0.2.10".parse().unwrap(),
        ),
        Instant::now(),
    );
    let entries = ScannerProvider::collect_from(&map, cache.path());
    let expected_id = crate::sources::SourceId::new("scanner:provider-test");
    let entry = entries
        .iter()
        .find(|entry| entry.id == expected_id)
        .expect("seeded scanner must be collected");
    assert_eq!(entry.label, "EPSON L3160 Series");
    let crate::sources::SourceKind::Scanner { vendor, model } = &entry.kind else {
        panic!("expected Scanner kind");
    };
    assert_eq!((vendor.as_str(), model.as_str()), ("EPSON", "L3160 Series"));
    let backing_path = match &entry.backing {
        crate::sources::SourceBacking::Path(path) => path.clone(),
        _ => panic!("scanner sources are path-backed"),
    };
    assert_eq!(
        backing_path,
        session_scan_dir(cache.path(), "provider-test")
    );
    assert!(
        !backing_path.exists(),
        "the session dir comes into being on selection, not from the provider"
    );
}

#[test]
fn prepare_session_dir_creates_and_prunes_stale_sessions() {
    let cache = tempfile::tempdir().unwrap();
    let device_dir = cache.path().join("scan-staging").join("dev");
    let current = device_dir.join("session-current");
    let stale = device_dir.join("session-old");
    std::fs::create_dir_all(&stale).unwrap();
    std::fs::write(stale.join("scan.jpg"), b"x").unwrap();
    std::fs::create_dir_all(&current).unwrap();
    let stray = device_dir.join("stray.tmp");
    std::fs::write(&stray, b"x").unwrap();

    prepare_session_dir(&current).unwrap();

    assert!(current.is_dir());
    assert!(
        !stale.exists(),
        "earlier runs' sessions are unreachable and pruned"
    );
    assert!(
        !stray.exists(),
        "stray files in the device directory are pruned too"
    );

    // The current session's files survive a re-selection.
    std::fs::write(current.join("scan-001.jpg"), b"x").unwrap();
    prepare_session_dir(&current).unwrap();
    assert!(current.join("scan-001.jpg").exists());
}

#[test]
fn scan_output_names_are_supported_and_part_files_excluded() {
    let dir = tempfile::tempdir().unwrap();
    let scan = dir.path().join("scan-20261020-101500-123.jpg");
    std::fs::write(&scan, b"jpeg-bytes").unwrap();
    let part = dir.path().join("scan-20261020-101501-456.jpg.rekesz-part");
    std::fs::write(&part, b"partial").unwrap();

    assert!(crate::discovery::is_supported_image(&scan));
    assert!(
        !crate::discovery::is_supported_image(&part),
        "part files must not enter the listing"
    );

    let cancel = AtomicBool::new(false);
    let outcome = crate::import::scan_source(
        &crate::sources::PathSource,
        &crate::sources::SourceBacking::Path(dir.path().to_path_buf()),
        crate::discovery::RawJpegPreference::Jpeg,
        &cancel,
        &mut |_| {},
    )
    .unwrap();
    assert_eq!(
        outcome.photos.len(),
        1,
        "only the finished scan is importable"
    );
    assert_eq!(
        outcome.photos[0].primary.name,
        "scan-20261020-101500-123.jpg"
    );
    assert!(
        outcome.photos[0].companion.is_none(),
        "a JPEG-only scan directory pairs nothing"
    );
}

#[test]
fn capabilities_sanitize_inverted_or_empty_device_bounds() {
    let xml = |min_w: &str, max_w: &str| {
        format!(
            r#"<ScannerCapabilities>
                <Platen><PlatenInputCaps>
                    <MinWidth>{min_w}</MinWidth><MaxWidth>{max_w}</MaxWidth>
                    <MinHeight>96</MinHeight><MaxHeight>3510</MaxHeight>
                    <SettingProfiles><SettingProfile>
                        <DocumentFormats><DocumentFormat>image/jpeg</DocumentFormat></DocumentFormats>
                        <SupportedResolutions><DiscreteResolutions>
                            <DiscreteResolution><XResolution>300</XResolution></DiscreteResolution>
                        </DiscreteResolutions></SupportedResolutions>
                    </SettingProfile></SettingProfiles>
                </PlatenInputCaps></Platen>
            </ScannerCapabilities>"#
        )
    };
    // Inverted bounds from a buggy or hostile device: clamp(min > max)
    // panics, so the parser must sanitize instead of passing them through.
    let caps = parse_capabilities(&xml("2000", "1000")).unwrap();
    assert_eq!(caps.platen.min_width, 2000);
    assert_eq!(caps.platen.max_width, 2000);
    let caps = parse_capabilities(&xml("0", "0")).unwrap();
    assert_eq!((caps.platen.min_width, caps.platen.max_width), (1, 1));
    // The sanitized ranges can never panic region clamping.
    let _ = region_from_fractions(&caps.platen, 0.0, 0.0, 1.0, 1.0);
}

#[test]
fn capabilities_ignore_adf_and_feeder_sections() {
    let xml = r#"<ScannerCapabilities>
        <Platen><PlatenInputCaps>
            <MinWidth>96</MinWidth><MaxWidth>2550</MaxWidth>
            <MinHeight>96</MinHeight><MaxHeight>3510</MaxHeight>
            <SettingProfiles><SettingProfile>
                <ColorModes><ColorMode>RGB24</ColorMode></ColorModes>
                <DocumentFormats><pwg:DocumentFormat xmlns:pwg="p">image/jpeg</pwg:DocumentFormat></DocumentFormats>
                <SupportedResolutions><DiscreteResolutions>
                    <DiscreteResolution><XResolution>300</XResolution></DiscreteResolution>
                </DiscreteResolutions></SupportedResolutions>
            </SettingProfile></SettingProfiles>
        </PlatenInputCaps></Platen>
        <Adf><AdfInputCaps>
            <MinWidth>96</MinWidth><MaxWidth>2550</MaxWidth>
            <MinHeight>96</MinHeight><MaxHeight>3510</MaxHeight>
            <SettingProfiles><SettingProfile>
                <ColorModes><ColorMode>NIR</ColorMode></ColorModes>
                <DocumentFormats><DocumentFormat>application/pdf</DocumentFormat></DocumentFormats>
                <SupportedResolutions><DiscreteResolutions>
                    <DiscreteResolution><XResolution>33</XResolution></DiscreteResolution>
                </DiscreteResolutions></SupportedResolutions>
            </SettingProfile></SettingProfiles>
        </AdfInputCaps></Adf>
        <Feeder><FeederInputCaps>
            <SettingProfiles><SettingProfile>
                <ColorModes><ColorMode>Magenta</ColorMode></ColorModes>
            </SettingProfile></SettingProfiles>
        </FeederInputCaps></Feeder>
    </ScannerCapabilities>"#;
    let caps = parse_capabilities(xml).unwrap();
    assert_eq!(caps.color_modes, vec!["RGB24".to_string()]);
    assert_eq!(caps.document_formats, vec!["image/jpeg".to_string()]);
    assert_eq!(caps.resolutions, vec![300]);
}

#[test]
fn scan_job_rejects_redirects_instead_of_following_them() {
    let (base, recorded) = spawn_mock(Box::new(|method, path, _host, _body| {
        match (method, path) {
            ("POST", "/eSCL/ScanJobs") => {
                let mut reply = MockReply::status(302);
                reply.location = Some("http://evil.example.com:9/eSCL/ScanJobs/hij".to_string());
                reply
            }
            _ => MockReply::status(404),
        }
    }));
    let client = EsclClient::new(&base).unwrap();
    let cancel = AtomicBool::new(false);
    let error = run_scan_job(
        &client,
        &test_settings(),
        &tempfile::tempdir().unwrap().path().join("scan.jpg"),
        &cancel,
        &test_timings(),
        &mut |_| {},
    )
    .unwrap_err();
    let ScanJobError::Failed(message) = error else {
        panic!("expected Failed, got {error:?}");
    };
    assert!(
        message.contains("302"),
        "the redirect surfaces as a status error: {message}"
    );
    assert!(
        !recorded
            .lock()
            .unwrap()
            .iter()
            .any(|(method, ..)| method == "DELETE"),
        "no job was created, so nothing is deleted"
    );
}

#[test]
fn scan_job_maps_410_to_a_ended_job() {
    let (base, _recorded) = spawn_mock(Box::new(|method, path, _host, _body| {
        match (method, path) {
            ("POST", "/eSCL/ScanJobs") => MockReply::created("/eSCL/ScanJobs/gone410"),
            ("GET", "/eSCL/ScanJobs/gone410/NextDocument") => MockReply::status(410),
            _ => MockReply::status(404),
        }
    }));
    let client = EsclClient::new(&base).unwrap();
    let cancel = AtomicBool::new(false);
    let error = run_scan_job(
        &client,
        &test_settings(),
        &tempfile::tempdir().unwrap().path().join("scan.jpg"),
        &cancel,
        &test_timings(),
        &mut |_| {},
    )
    .unwrap_err();
    assert!(matches!(error, ScanJobError::Failed(ref m) if m.contains("ended the job")));
}

#[test]
fn scan_job_rejects_empty_document_bodies() {
    let (base, _recorded) = spawn_mock(Box::new(|method, path, _host, _body| {
        match (method, path) {
            ("POST", "/eSCL/ScanJobs") => MockReply::created("/eSCL/ScanJobs/empty"),
            ("GET", "/eSCL/ScanJobs/empty/NextDocument") => {
                MockReply::image("image/jpeg", Vec::new())
            }
            _ => MockReply::status(404),
        }
    }));
    let client = EsclClient::new(&base).unwrap();
    let cancel = AtomicBool::new(false);
    let error = run_scan_job(
        &client,
        &test_settings(),
        &tempfile::tempdir().unwrap().path().join("scan.jpg"),
        &cancel,
        &test_timings(),
        &mut |_| {},
    )
    .unwrap_err();
    assert!(
        matches!(error, ScanJobError::Failed(ref m) if m.contains("empty")),
        "an empty body must not be renamed in as a scan: {error:?}"
    );
}

#[test]
fn scan_job_reads_status_xml_even_without_an_xml_content_type() {
    let polls = Arc::new(AtomicUsize::new(0));
    let polls_clone = Arc::clone(&polls);
    let (base, _recorded) = spawn_mock(Box::new(move |method, path, _host, _body| {
        match (method, path) {
            ("POST", "/eSCL/ScanJobs") => MockReply::created("/eSCL/ScanJobs/plain"),
            ("GET", "/eSCL/ScanJobs/plain/NextDocument") => {
                let attempt = polls_clone.fetch_add(1, Ordering::AcqRel);
                if attempt < 1 {
                    let mut reply = MockReply::xml(
                        "<pwg:JobStatus><pwg:JobState>Processing</pwg:JobState></pwg:JobStatus>",
                    );
                    reply.content_type = Some("text/plain".to_string());
                    reply
                } else {
                    MockReply::image("image/jpeg", vec![5, 6, 7, 8])
                }
            }
            _ => MockReply::status(404),
        }
    }));
    let client = EsclClient::new(&base).unwrap();
    let cancel = AtomicBool::new(false);
    let size = run_scan_job(
        &client,
        &test_settings(),
        &tempfile::tempdir().unwrap().path().join("scan.jpg"),
        &cancel,
        &test_timings(),
        &mut |_| {},
    )
    .unwrap();
    assert_eq!(
        size, 4,
        "a leading '<' sniffs status XML regardless of content type"
    );
}

#[test]
fn scan_job_fails_after_repeated_truncated_transfers() {
    let (base, _recorded) = spawn_mock(Box::new(|method, path, _host, _body| {
        match (method, path) {
            ("POST", "/eSCL/ScanJobs") => MockReply::created("/eSCL/ScanJobs/cut"),
            ("GET", "/eSCL/ScanJobs/cut/NextDocument") => {
                MockReply::image("image/jpeg", vec![1; 10]).truncated()
            }
            _ => MockReply::status(404),
        }
    }));
    let client = EsclClient::new(&base).unwrap();
    let cancel = AtomicBool::new(false);
    let error = run_scan_job(
        &client,
        &test_settings(),
        &tempfile::tempdir().unwrap().path().join("scan.jpg"),
        &cancel,
        &test_timings(),
        &mut |_| {},
    )
    .unwrap_err();
    assert!(
        matches!(error, ScanJobError::Failed(_)),
        "ten consecutive truncated transfers must end the job: {error:?}"
    );
}

#[test]
fn scan_job_recreates_a_lost_job_when_the_scanner_goes_idle() {
    // The device accepted the job and scanned, but never delivers: the
    // polls stall with 503 while ScannerStatus reports Idle. The runner
    // must DELETE the lost job and create a fresh one.
    let posts = Arc::new(AtomicUsize::new(0));
    let posts_clone = Arc::clone(&posts);
    let (base, _recorded) = spawn_mock(Box::new(move |method, path, _host, _body| {
        match (method, path) {
            ("POST", "/eSCL/ScanJobs") => {
                let n = posts_clone.fetch_add(1, Ordering::AcqRel);
                if n == 0 {
                    MockReply::created("/eSCL/ScanJobs/lost")
                } else {
                    MockReply::created("/eSCL/ScanJobs/fresh")
                }
            }
            ("GET", "/eSCL/ScanJobs/lost/NextDocument") => MockReply::status(503),
            ("GET", "/eSCL/ScanJobs/fresh/NextDocument") => {
                MockReply::image("image/jpeg", vec![1, 2, 3])
            }
            ("GET", "/eSCL/ScannerStatus") => MockReply::xml(
                "<pwg:ScannerStatus xmlns:pwg=\"p\"><pwg:State>Idle</pwg:State></pwg:ScannerStatus>",
            ),
            _ => MockReply::status(404),
        }
    }));
    let client = EsclClient::new(&base).unwrap();
    let cancel = AtomicBool::new(false);
    let size = run_scan_job(
        &client,
        &test_settings(),
        &tempfile::tempdir().unwrap().path().join("scan.jpg"),
        &cancel,
        &test_timings(),
        &mut |_| {},
    )
    .unwrap();
    assert_eq!(size, 3);
    assert_eq!(
        posts.load(Ordering::Acquire),
        2,
        "the job was recreated once"
    );
}

#[test]
fn scan_job_keeps_waiting_while_the_scanner_is_processing() {
    // A long legitimate scan answers 503 while ScannerStatus reports
    // Processing: the job must NOT be recreated.
    let posts = Arc::new(AtomicUsize::new(0));
    let polls = Arc::new(AtomicUsize::new(0));
    let posts_clone = Arc::clone(&posts);
    let polls_clone = Arc::clone(&polls);
    let (base, _recorded) = spawn_mock(Box::new(move |method, path, _host, _body| {
        match (method, path) {
            ("POST", "/eSCL/ScanJobs") => {
                posts_clone.fetch_add(1, Ordering::AcqRel);
                MockReply::created("/eSCL/ScanJobs/slow")
            }
            ("GET", "/eSCL/ScanJobs/slow/NextDocument") => {
                let attempt = polls_clone.fetch_add(1, Ordering::AcqRel);
                if attempt < 12 {
                    MockReply::status(503)
                } else {
                    MockReply::image("image/jpeg", vec![8])
                }
            }
            ("GET", "/eSCL/ScannerStatus") => MockReply::xml(
                "<pwg:ScannerStatus xmlns:pwg=\"p\"><pwg:State>Processing</pwg:State></pwg:ScannerStatus>",
            ),
            _ => MockReply::status(404),
        }
    }));
    let client = EsclClient::new(&base).unwrap();
    let cancel = AtomicBool::new(false);
    let size = run_scan_job(
        &client,
        &test_settings(),
        &tempfile::tempdir().unwrap().path().join("scan.jpg"),
        &cancel,
        &test_timings(),
        &mut |_| {},
    )
    .unwrap();
    assert_eq!(size, 1);
    assert_eq!(
        posts.load(Ordering::Acquire),
        1,
        "no recreation while Processing"
    );
}

#[test]
fn scan_job_honours_cancel_during_job_creation() {
    let posts = Arc::new(AtomicUsize::new(0));
    let posts_clone = Arc::clone(&posts);
    let (base, _recorded) = spawn_mock(Box::new(move |method, path, _host, _body| {
        match (method, path) {
            ("POST", "/eSCL/ScanJobs") => {
                posts_clone.fetch_add(1, Ordering::AcqRel);
                // A wedged device: the POST answer takes its time.
                std::thread::sleep(Duration::from_millis(80));
                MockReply::status(503)
            }
            _ => MockReply::status(404),
        }
    }));
    let client = EsclClient::new(&base).unwrap();
    let cancel = Arc::new(AtomicBool::new(false));
    let runner = {
        let cancel = Arc::clone(&cancel);
        let settings = test_settings();
        std::thread::spawn(move || {
            run_scan_job(
                &client,
                &settings,
                &tempfile::tempdir().unwrap().path().join("scan.jpg"),
                &cancel,
                &ScanJobTimings {
                    guard: Duration::from_secs(30),
                    total: Duration::from_secs(60),
                    post_retry_pause: Duration::from_millis(20),
                    poll_pause: Duration::from_millis(1),
                },
                &mut |_| {},
            )
        })
    };
    std::thread::sleep(Duration::from_millis(40));
    cancel.store(true, Ordering::Release);
    let outcome = runner.join().unwrap();
    assert!(matches!(outcome, Err(ScanJobError::Canceled)));
    assert!(
        posts.load(Ordering::Acquire) < 3,
        "cancel must cut the busy-retry window short"
    );
}

#[test]
fn device_key_ignores_empty_uuid_values() {
    let txt = vec![("uuid".to_string(), String::new())];
    assert_eq!(
        &*device_key(&txt, "Broken Device"),
        "name:Broken Device",
        "an empty uuid must not merge distinct devices"
    );
}

#[test]
fn instance_names_strip_the_whole_service_suffix() {
    assert_eq!(
        instance_name("Foo. Bar._uscan._tcp.local."),
        "Foo. Bar",
        "instance names may contain literal dots"
    );
    assert_eq!(instance_name("X._uscans._tcp.local."), "X");
    assert_eq!(instance_name("plain"), "plain");
}

#[test]
fn endpoint_base_url_skips_unconnectable_link_local_ipv6() {
    let link_local_only = ["fe80::1".parse().unwrap()];
    assert_eq!(
        endpoint_base_url(true, &link_local_only, 443, "eSCL"),
        None,
        "link-local v6 without a scope can never connect"
    );
    let mixed = ["fe80::1".parse().unwrap(), "192.0.2.10".parse().unwrap()];
    assert_eq!(
        endpoint_base_url(true, &mixed, 443, "eSCL").as_deref(),
        Some("https://192.0.2.10:443/eSCL")
    );
    let global_v6 = ["fd00::1".parse().unwrap()];
    assert_eq!(
        endpoint_base_url(false, &global_v6, 80, "eSCL").as_deref(),
        Some("http://[fd00::1]:80/eSCL")
    );
}
