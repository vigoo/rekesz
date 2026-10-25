//! eSCL protocol client: capabilities parsing, scan-settings writing, and
//! the scan-job lifecycle.
//!
//! The lifecycle is driven by `GET {job}/NextDocument`, the pull model every
//! proven client (sane-airscan, NAPS2) implements: the job URI is never
//! polled for the document. POST requires exactly `201 Created`; `503` is
//! retried with a bounded pause (the device is busy or waking); `409` means
//! the settings were rejected. On the document request, `503` keeps the job
//! alive (devices return it mid-scan while the engine runs), `404`/`410`
//! mean the job ended, and a `200` with an XML body is a status payload, not
//! image bytes. The job URI is the `Location` header of the POST resolved
//! against the client's own origin using only its path — device Locations
//! can be relative or carry a hostile authority.

use std::fmt;
use std::io::{Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use reqwest::Url;
use reqwest::blocking::Client;
use reqwest::header::{CONTENT_TYPE, LOCATION};

/// eSCL scan-region unit: `pwg:ContentRegionUnits`
/// `escl:ThreeHundredthsOfInches` — 1/300 inch, so a 300 dpi render maps
/// one unit to one pixel.
pub const UNITS_PER_INCH: f64 = 300.0;

/// Millimetres to scan-region units; the inverse of [`units_to_mm`], which
/// the scan panel (Stage 3) uses to show the picked region in mm.
#[allow(dead_code)]
pub fn mm_to_units(mm: f64) -> f64 {
    mm * UNITS_PER_INCH / 25.4
}

pub fn units_to_mm(units: f64) -> f64 {
    units * 25.4 / UNITS_PER_INCH
}

#[derive(Debug)]
pub enum EsclError {
    InvalidBaseUrl(String),
    Transport(String),
    Status {
        code: u16,
        message: String,
    },
    /// `409 Conflict` on ScanJobs POST: the settings were rejected (e.g. a
    /// resolution or color mode the device did not advertise).
    SettingsRejected,
    /// `503` on ScanJobs POST past the bounded retries: the device is in use
    /// by another client or still waking.
    Busy,
    /// POST succeeded but carried no usable `Location` header.
    MissingLocation,
    /// The device does not advertise what we need (no platen, no JPEG).
    UnsupportedDevice(&'static str),
    Xml(String),
    Io(std::io::Error),
    Canceled,
}

impl fmt::Display for EsclError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidBaseUrl(url) => write!(f, "invalid scanner URL {url}"),
            Self::Transport(message) => write!(f, "scanner connection failed: {message}"),
            Self::Status { code, message } => write!(f, "scanner returned {code}: {message}"),
            Self::SettingsRejected => {
                write!(f, "the scanner rejected the scan settings (409)")
            }
            Self::Busy => write!(f, "scanner busy"),
            Self::MissingLocation => write!(f, "scanner did not return a scan job location"),
            Self::UnsupportedDevice(why) => write!(f, "unsupported scanner: {why}"),
            Self::Xml(message) => write!(f, "malformed scanner response: {message}"),
            Self::Io(error) => write!(f, "scan transfer failed: {error}"),
            Self::Canceled => write!(f, "scan cancelled"),
        }
    }
}

// ---------------------------------------------------------------------------
// Capabilities
// ---------------------------------------------------------------------------

/// Platen geometry in 1/300-inch units.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InputCaps {
    pub min_width: u32,
    pub max_width: u32,
    pub min_height: u32,
    pub max_height: u32,
}

#[derive(Clone, Debug)]
pub struct ScannerCapabilities {
    pub platen: InputCaps,
    pub max_scan_regions: u32,
    /// Raw `ColorMode` spellings in advertisement order.
    pub color_modes: Vec<String>,
    /// `DocumentFormat`/`DocumentFormatExt` values in advertisement order.
    pub document_formats: Vec<String>,
    /// Discrete resolutions sorted ascending; a `ResolutionRange` is
    /// expanded when small, else reduced to its bounds.
    pub resolutions: Vec<u32>,
}

/// Element-local-name matching that ignores namespace prefixes: device XML
/// varies between `scan:Platen`, default-namespaced, and differently
/// prefixed forms, which serde-style derive cannot handle.
fn local_name(name: &str) -> &str {
    name.rsplit(':').next().unwrap_or(name)
}

/// Sanitize a device-supplied [min, max] range into one that always holds
/// `1 <= min <= max`, whatever the wire said.
fn sane_range(min: u32, max: u32) -> (u32, u32) {
    let max = max.max(min).max(1);
    let min = min.min(max).max(1);
    (min, max)
}

/// Parse a ScannerCapabilities document. Unknown elements are skipped for
/// forward compatibility.
pub fn parse_capabilities(xml: &str) -> Result<ScannerCapabilities, EsclError> {
    let mut reader = quick_xml::Reader::from_str(xml);
    reader.config_mut().trim_text(true);

    let mut stack: Vec<String> = Vec::new();
    let mut text = String::new();

    // The event borrow does not outlive the match arm, so element names are
    // stacked as owned strings.
    fn stack_has(stack: &[String], name: &str) -> bool {
        stack.iter().any(|entry| entry == name)
    }

    let mut platen: Option<InputCaps> = None;
    let mut max_scan_regions = 1_u32;
    let mut color_modes: Vec<String> = Vec::new();
    let mut document_formats: Vec<String> = Vec::new();
    let mut discrete_resolutions: Vec<u32> = Vec::new();
    let mut range: [Option<u32>; 3] = [None; 3]; // min, max, step

    // trim_text drops whitespace-only text events, so every Text seen here
    // is the content of the leaf the stack currently closes.
    fn push_unique(list: &mut Vec<String>, value: String) {
        if !list.contains(&value) {
            list.push(value);
        }
    }

    loop {
        match reader.read_event() {
            Ok(quick_xml::events::Event::Start(start)) => {
                let name = local_name(start.name().as_ref()).to_string();
                if name == "PlatenInputCaps" {
                    platen = Some(InputCaps {
                        min_width: 0,
                        max_width: 0,
                        min_height: 0,
                        max_height: 0,
                    });
                }
                stack.push(name);
                text.clear();
            }
            Ok(quick_xml::events::Event::End(_)) => {
                if let Some(name) = stack.pop() {
                    let in_platen_caps = stack_has(&stack, "PlatenInputCaps");
                    // ADF/Feeder sections repeat the same descendant names
                    // with different values; only the platen's lists may
                    // feed the UI.
                    let in_platen_section = stack_has(&stack, "Platen")
                        && !stack_has(&stack, "Adf")
                        && !stack_has(&stack, "Feeder");
                    let value = text.trim().to_string();
                    match name.as_str() {
                        "MinWidth" if in_platen_caps => {
                            if let Some(caps) = platen.as_mut() {
                                caps.min_width = value.parse().unwrap_or(0);
                            }
                        }
                        "MaxWidth" if in_platen_caps => {
                            if let Some(caps) = platen.as_mut() {
                                caps.max_width = value.parse().unwrap_or(0);
                            }
                        }
                        "MinHeight" if in_platen_caps => {
                            if let Some(caps) = platen.as_mut() {
                                caps.min_height = value.parse().unwrap_or(0);
                            }
                        }
                        "MaxHeight" if in_platen_caps => {
                            if let Some(caps) = platen.as_mut() {
                                caps.max_height = value.parse().unwrap_or(0);
                            }
                        }
                        "MaxScanRegions" if in_platen_caps => {
                            max_scan_regions = value.parse().unwrap_or(1);
                        }
                        "ColorMode" if in_platen_section && stack_has(&stack, "ColorModes") => {
                            push_unique(&mut color_modes, value);
                        }
                        "DocumentFormat" | "DocumentFormatExt"
                            if in_platen_section && stack_has(&stack, "DocumentFormats") =>
                        {
                            push_unique(&mut document_formats, value);
                        }
                        "XResolution"
                            if in_platen_section && stack_has(&stack, "DiscreteResolution") =>
                        {
                            if let Ok(resolution) = value.parse()
                                && !discrete_resolutions.contains(&resolution)
                            {
                                discrete_resolutions.push(resolution);
                            }
                        }
                        "Min" if in_platen_section && stack_has(&stack, "ResolutionRange") => {
                            range[0] = value.parse().ok();
                        }
                        "Max" if in_platen_section && stack_has(&stack, "ResolutionRange") => {
                            range[1] = value.parse().ok();
                        }
                        "Step" if in_platen_section && stack_has(&stack, "ResolutionRange") => {
                            range[2] = value.parse().ok();
                        }
                        _ => {}
                    }
                }
                text.clear();
            }
            Ok(quick_xml::events::Event::Text(t)) => {
                match quick_xml::escape::unescape(t.as_ref()) {
                    Ok(chunk) => text.push_str(&chunk),
                    Err(_) => text.push_str(t.as_ref()),
                }
            }
            Ok(quick_xml::events::Event::CData(t)) => {
                text.push_str(t.as_ref());
            }
            Ok(quick_xml::events::Event::Eof) => break,
            Ok(_) => {}
            Err(error) => return Err(EsclError::Xml(error.to_string())),
        }
    }

    let Some(mut platen) = platen else {
        return Err(EsclError::UnsupportedDevice("no platen"));
    };
    // Device-supplied geometry is untrusted network input: guarantee
    // 1 <= min <= max so region clamping can never see an inverted or empty
    // range (clamp with min > max panics, and a malformed capabilities
    // document must not abort the app).
    (platen.min_width, platen.max_width) = sane_range(platen.min_width, platen.max_width);
    (platen.min_height, platen.max_height) = sane_range(platen.min_height, platen.max_height);
    if !document_formats.iter().any(|format| format == "image/jpeg") {
        return Err(EsclError::UnsupportedDevice("no image/jpeg support"));
    }
    let mut resolutions = discrete_resolutions;
    if resolutions.is_empty()
        && let [Some(min), Some(max), Some(step)] = range
    {
        let step = step.max(1);
        if max >= min && (max - min) / step <= 32 {
            resolutions = (min..=max).step_by(step as usize).collect();
        } else if max >= min {
            resolutions = vec![min, max];
        }
    }
    resolutions.sort_unstable();
    if resolutions.is_empty() {
        return Err(EsclError::UnsupportedDevice("no advertised resolutions"));
    }
    Ok(ScannerCapabilities {
        platen,
        max_scan_regions,
        color_modes,
        document_formats,
        resolutions,
    })
}

/// Extract the scanner state (Idle/Processing/…) from a ScannerStatus
/// body, if present.
pub fn parse_scanner_state(xml: &str) -> Option<String> {
    let mut reader = quick_xml::Reader::from_str(xml);
    reader.config_mut().trim_text(true);
    let mut capture = false;
    let mut text = String::new();
    loop {
        match reader.read_event() {
            Ok(quick_xml::events::Event::Start(start)) => {
                capture = local_name(start.name().as_ref()) == "State";
                text.clear();
            }
            Ok(quick_xml::events::Event::Text(t)) if capture => {
                text.push_str(t.as_ref());
            }
            Ok(quick_xml::events::Event::End(_)) if capture => {
                return Some(text.trim().to_string());
            }
            Ok(quick_xml::events::Event::Eof) => return None,
            Ok(_) => {}
            Err(_) => return None,
        }
    }
}

/// Extract the `JobState` text (Pending/Processing/Completed/Canceled/
/// Aborted) from a JobStatus or ScannerStatus body, if present.
pub fn parse_job_state(xml: &str) -> Option<String> {
    let mut reader = quick_xml::Reader::from_str(xml);
    reader.config_mut().trim_text(true);
    let mut capture = false;
    let mut text = String::new();
    loop {
        match reader.read_event() {
            Ok(quick_xml::events::Event::Start(start)) => {
                capture = local_name(start.name().as_ref()) == "JobState";
                text.clear();
            }
            Ok(quick_xml::events::Event::Text(t)) if capture => {
                text.push_str(t.as_ref());
            }
            Ok(quick_xml::events::Event::End(_)) if capture => {
                return Some(text.trim().to_string());
            }
            Ok(quick_xml::events::Event::Eof) => return None,
            Ok(_) => {}
            Err(_) => return None,
        }
    }
}

// ---------------------------------------------------------------------------
// Scan settings
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorMode {
    BlackAndWhite1,
    Grayscale8,
    Rgb24,
}

impl ColorMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::BlackAndWhite1 => "BlackAndWhite1",
            Self::Grayscale8 => "Grayscale8",
            Self::Rgb24 => "RGB24",
        }
    }
}

/// One scan region in 1/300-inch units.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScanRegion {
    pub x_offset: u32,
    pub y_offset: u32,
    pub width: u32,
    pub height: u32,
}

impl ScanRegion {
    /// Consumed by the main-crate tests and the Stage 3 scan panel; the
    /// `scanner-probe` bin builds regions from fractions instead.
    #[allow(dead_code)]
    pub fn full_platen(caps: &InputCaps) -> Self {
        Self {
            x_offset: 0,
            y_offset: 0,
            width: caps.max_width,
            height: caps.max_height,
        }
    }
}

/// A scan region picked as fractions of the full platen, corner-order
/// independent, clamped to the platen's advertised minimum and maximum.
pub fn region_from_fractions(caps: &InputCaps, x0: f64, y0: f64, x1: f64, y1: f64) -> ScanRegion {
    let clamp01 = |value: f64| value.clamp(0.0, 1.0);
    let (lo_x, hi_x) = (clamp01(x0.min(x1)), clamp01(x0.max(x1)));
    let (lo_y, hi_y) = (clamp01(y0.min(y1)), clamp01(y0.max(y1)));
    let max_width = caps.max_width as f64;
    let max_height = caps.max_height as f64;
    let width = (((hi_x - lo_x) * max_width).round() as i64)
        .clamp(caps.min_width as i64, caps.max_width as i64);
    let height = (((hi_y - lo_y) * max_height).round() as i64)
        .clamp(caps.min_height as i64, caps.max_height as i64);
    let x_offset =
        ((lo_x * max_width).round() as i64).clamp(0, (caps.max_width as i64 - width).max(0));
    let y_offset =
        ((lo_y * max_height).round() as i64).clamp(0, (caps.max_height as i64 - height).max(0));
    ScanRegion {
        x_offset: x_offset as u32,
        y_offset: y_offset as u32,
        width: width as u32,
        height: height as u32,
    }
}

#[derive(Clone, Debug)]
pub struct ScanSettings {
    pub region: ScanRegion,
    pub color_mode: ColorMode,
    /// A MIME type the device advertised (always `image/jpeg` for us).
    pub document_format: String,
    /// A resolution from the device's advertised list.
    pub resolution: u32,
}

impl ScanSettings {
    /// The exact element tree and namespace bindings sane-airscan writes:
    /// `pwg:` for the PWG common elements, `scan:` for the eSCL extensions,
    /// a single `pwg:ScanRegion` wrapped in `pwg:ScanRegions`, and
    /// `Grayscale8` (not `Gray8`) for grayscale. Byte-for-byte identical to
    /// what the L3160 accepted in the Stage 1 investigation.
    pub fn to_xml(&self) -> String {
        let escape = |value: &str| {
            value
                .replace('&', "&amp;")
                .replace('<', "&lt;")
                .replace('>', "&gt;")
        };
        format!(
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
             <pwg:ScanSettings xmlns:pwg=\"http://www.pwg.org/schemas/2010/12/sm\" \
             xmlns:scan=\"http://schemas.hp.com/imaging/escl/2011/05/03\">\n\
             \x20 <pwg:Version>2.0</pwg:Version>\n\
             \x20 <pwg:InputSource>Platen</pwg:InputSource>\n\
             \x20 <pwg:ScanRegions>\n\
             \x20   <pwg:ScanRegion>\n\
             \x20     <pwg:Height>{height}</pwg:Height>\n\
             \x20     <pwg:Width>{width}</pwg:Width>\n\
             \x20     <pwg:XOffset>{x}</pwg:XOffset>\n\
             \x20     <pwg:YOffset>{y}</pwg:YOffset>\n\
             \x20     <pwg:ContentRegionUnits>escl:ThreeHundredthsOfInches</pwg:ContentRegionUnits>\n\
             \x20   </pwg:ScanRegion>\n\
             \x20 </pwg:ScanRegions>\n\
             \x20 <scan:ColorMode>{color}</scan:ColorMode>\n\
             \x20 <scan:DocumentFormat>{format}</scan:DocumentFormat>\n\
             \x20 <scan:XResolution>{resolution}</scan:XResolution>\n\
             \x20 <scan:YResolution>{resolution}</scan:YResolution>\n\
             </pwg:ScanSettings>\n",
            height = self.region.height,
            width = self.region.width,
            x = self.region.x_offset,
            y = self.region.y_offset,
            color = self.color_mode.as_str(),
            format = escape(&self.document_format),
            resolution = self.resolution,
        )
    }
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct EsclClient {
    http: Client,
    /// `DELETE` is best-effort cleanup and must never wedge a cancel path.
    delete_http: Client,
    /// `scheme://host:port` of the discovered device. Discovery always
    /// writes the port into the URL; note the url crate normalizes default
    /// ports away (443/80), so no explicit-port Host header survives for
    /// them — the live A/B found this firmware does not care (and if a
    /// device ever does, it needs an explicit `Host:` header override).
    origin: String,
    /// The `rs` resource prefix path from discovery ("" when none).
    base_path: String,
}

impl EsclClient {
    /// Build a client for a discovered base URL
    /// (`https://192.0.2.10:443/eSCL`). Printer certificates are self-signed
    /// and name hosts, not the bare IPs mDNS resolves, so both certificate
    /// and hostname validation are off — the address comes from our own
    /// discovery on the local network, which is the trust anchor.
    pub fn new(base_url: &str) -> Result<Self, EsclError> {
        let url =
            Url::parse(base_url).map_err(|error| EsclError::InvalidBaseUrl(error.to_string()))?;
        let origin = format!("{}://{}", url.scheme(), url.authority());
        let base_path = url.path().trim_end_matches('/').to_string();
        Ok(Self {
            http: Self::http_client(Duration::from_secs(120))?,
            delete_http: Self::http_client(Duration::from_secs(5))?,
            origin,
            base_path,
        })
    }

    fn http_client(total_timeout: Duration) -> Result<Client, EsclError> {
        Client::builder()
            .danger_accept_invalid_certs(true)
            .danger_accept_invalid_hostnames(true)
            // Redirects are never followed: origin pinning is the containment
            // that remains once certificate validation is off, and a 307 on
            // the ScanJobs POST would otherwise replay the settings body to
            // whatever host the device names (sane-airscan does not follow
            // redirects either).
            .redirect(reqwest::redirect::Policy::none())
            .connect_timeout(Duration::from_secs(10))
            .timeout(total_timeout)
            .build()
            .map_err(|error| EsclError::Transport(error.to_string()))
    }

    /// Resolve a path (absolute, or relative to the base path) against the
    /// client's own origin. Job paths from `Location` arrive here already
    /// reduced to their path component, so a hostile authority in the header
    /// is discarded by construction.
    fn url(&self, path: &str) -> Url {
        let full = if path.starts_with('/') {
            format!("{}{}", self.origin, path)
        } else {
            format!("{}/{path}", self.origin_element_base())
        };
        Url::parse(&full).unwrap_or_else(|_| Url::parse(&self.origin).expect("origin URL"))
    }

    fn origin_element_base(&self) -> String {
        if self.base_path.is_empty() {
            self.origin.clone()
        } else {
            format!("{}{}", self.origin, self.base_path)
        }
    }

    fn url_for_endpoint(&self, suffix: &str) -> Url {
        self.url(&format!(
            "{}/{}",
            self.base_path,
            suffix.trim_start_matches('/')
        ))
    }

    /// Read the device state from ScannerStatus (Idle, Processing, …).
    /// `None` when the read or parse fails — callers treat that as unknown
    /// and keep waiting.
    pub fn scanner_state(&self) -> Option<String> {
        let response = self
            .http
            .get(self.url_for_endpoint("ScannerStatus"))
            .send()
            .ok()?;
        if response.status().as_u16() != 200 {
            return None;
        }
        parse_scanner_state(&response.text().ok()?)
    }

    /// Fetch and parse the ScannerCapabilities document.
    pub fn capabilities(&self) -> Result<ScannerCapabilities, EsclError> {
        let response = self
            .http
            .get(self.url_for_endpoint("ScannerCapabilities"))
            .send()
            .map_err(|error| EsclError::Transport(error.to_string()))?;
        let code = response.status().as_u16();
        if code != 200 {
            return Err(EsclError::Status {
                code,
                message: "GET ScannerCapabilities".to_string(),
            });
        }
        let body = response
            .text()
            .map_err(|error| EsclError::Transport(error.to_string()))?;
        parse_capabilities(&body)
    }

    /// Create a scan job and return its job path. Retries `503` up to three
    /// attempts before giving up with [`EsclError::Busy`]; honours `cancel`
    /// between attempts. Transport errors are never retried: the POST may
    /// already have created a job whose `Location` we never saw, and a
    /// leaked job surfaces later as mystery "scanner busy" (sane-airscan
    /// fails immediately for the same reason).
    pub fn start_job(
        &self,
        settings: &ScanSettings,
        cancel: &AtomicBool,
        retry_pause: Duration,
    ) -> Result<String, EsclError> {
        let body = settings.to_xml();
        for attempt in 0..3 {
            if cancel.load(Ordering::Acquire) {
                return Err(EsclError::Canceled);
            }
            if attempt > 0 {
                std::thread::sleep(retry_pause.max(Duration::from_millis(1)));
            }
            if cancel.load(Ordering::Acquire) {
                return Err(EsclError::Canceled);
            }
            let response = self
                .http
                .post(self.url_for_endpoint("ScanJobs"))
                .header(CONTENT_TYPE, "text/xml")
                .body(body.clone())
                .send();
            match response {
                Ok(response) if response.status().as_u16() == 201 => {
                    let location = response
                        .headers()
                        .get(LOCATION)
                        .and_then(|value| value.to_str().ok());
                    return match location {
                        Some(location) => Ok(job_path_from_location(location)),
                        // Without a Location there is nothing to DELETE; the
                        // job lingers until the device's own timeout reaps it.
                        None => Err(EsclError::MissingLocation),
                    };
                }
                Ok(response) if response.status().as_u16() == 409 => {
                    return Err(EsclError::SettingsRejected);
                }
                Ok(response) if response.status().as_u16() == 503 => continue,
                Ok(response) => {
                    return Err(EsclError::Status {
                        code: response.status().as_u16(),
                        message: "POST ScanJobs".to_string(),
                    });
                }
                Err(error) => {
                    return Err(EsclError::Transport(error.to_string()));
                }
            }
        }
        Err(EsclError::Busy)
    }

    /// One poll of the job's document. See the module docs for the state
    /// mapping. Image bytes stream into `out` with progress callbacks; XML
    /// bodies are parsed as job status and never treated as image data.
    pub fn next_document(
        &self,
        job_path: &str,
        out: &mut dyn Write,
        cancel: &AtomicBool,
        progress: &mut dyn FnMut(u64),
    ) -> Result<NextDocument, EsclError> {
        let response = self
            .http
            .get(self.url(&format!("{job_path}/NextDocument")))
            .send()
            .map_err(|error| EsclError::Transport(error.to_string()))?;
        match response.status().as_u16() {
            503 => Ok(NextDocument::Busy),
            404 | 410 => Ok(NextDocument::Gone),
            200 => {
                let content_type = response
                    .headers()
                    .get(CONTENT_TYPE)
                    .and_then(|value| value.to_str().ok())
                    .unwrap_or("")
                    .to_ascii_lowercase();
                let mut response = response;
                let mut total = 0_u64;
                let mut buffer = [0_u8; 64 * 1024];
                // The content type decides, but a device that labels its
                // status payload as text/plain (or omits the header) is
                // caught by the leading '<': status XML must never become
                // the "image".
                let first_read = response.read(&mut buffer).map_err(EsclError::Io)?;
                let looks_like_xml = content_type.contains("xml")
                    || (first_read > 0 && !content_type.starts_with("image/") && buffer[0] == b'<');
                if looks_like_xml {
                    let mut body = buffer[..first_read].to_vec();
                    response.read_to_end(&mut body).map_err(EsclError::Io)?;
                    return Ok(NextDocument::JobStatus(parse_job_state(
                        &String::from_utf8_lossy(&body),
                    )));
                }
                if first_read > 0 {
                    out.write_all(&buffer[..first_read])
                        .map_err(EsclError::Io)?;
                    total += first_read as u64;
                    progress(total);
                }
                loop {
                    if cancel.load(Ordering::Acquire) {
                        return Err(EsclError::Canceled);
                    }
                    let read = response.read(&mut buffer).map_err(EsclError::Io)?;
                    if read == 0 {
                        break;
                    }
                    out.write_all(&buffer[..read]).map_err(EsclError::Io)?;
                    total += read as u64;
                    progress(total);
                }
                if total == 0 {
                    // An empty body renamed in as a scan would be a silent
                    // data loss (NAPS2 errors the same way).
                    return Err(EsclError::Xml(
                        "the scanner delivered an empty image".to_string(),
                    ));
                }
                Ok(NextDocument::Document(total))
            }
            code => Err(EsclError::Status {
                code,
                message: format!("GET {job_path}/NextDocument"),
            }),
        }
    }

    /// Best-effort job cleanup on every exit path (success, error, cancel,
    /// timeout). A `404` here just means the device already reaped the job.
    pub fn delete_job(&self, job_path: &str) {
        let _ = self.delete_http.delete(self.url(job_path)).send();
    }
}

/// Human-readable byte count for status lines ("12.0 KB", "1.5 MB"). Used
/// by the scan panel; the `scanner-probe` bin (which shares this file)
/// never calls it.
#[allow(dead_code)]
pub fn format_bytes(bytes: u64) -> String {
    const UNITS: [(u64, &str); 5] = [
        (1 << 40, "TB"),
        (1 << 30, "GB"),
        (1 << 20, "MB"),
        (1 << 10, "KB"),
        (1, "B"),
    ];
    let Some((factor, unit)) = UNITS.iter().find(|(factor, _)| bytes >= *factor) else {
        return "0 B".to_owned();
    };
    let value = bytes as f64 / *factor as f64;
    if value >= 100.0 {
        format!("{value:.0} {unit}")
    } else {
        format!("{value:.1} {unit}")
    }
}

/// The result of one [`EsclClient::next_document`] poll.
#[derive(Debug, PartialEq, Eq)]
pub enum NextDocument {
    /// The image streamed completely; the value is the byte count.
    Document(u64),
    /// The device returned a status payload instead of the document.
    JobStatus(Option<String>),
    /// The engine is not ready to serve the document yet; keep polling.
    Busy,
    /// The job is gone (reaped or aborted); no document will come.
    Gone,
}

/// The `Location` header of a job POST reduced to its path component:
/// absolute URLs keep only the path (scheme, host, and port of the header
/// are never trusted — sane-airscan behavior), relative references get a
/// leading slash.
pub fn job_path_from_location(location: &str) -> String {
    if let Ok(url) = Url::parse(location)
        && url.host().is_some()
    {
        return url.path().to_string();
    }
    if location.starts_with('/') {
        location.to_string()
    } else {
        format!("/{location}")
    }
}

// ---------------------------------------------------------------------------
// Job runner
// ---------------------------------------------------------------------------

/// Tunables of [`run_scan_job`]; the defaults are the real-device values and
/// tests inject small ones instead of sleeping through scaled retries.
#[derive(Clone, Copy, Debug)]
pub struct ScanJobTimings {
    /// Idle watchdog: the job fails if the device stays silent (no poll
    /// answer, no bytes) for this long. A device that keeps answering is
    /// never timed out — high-dpi flatbed scans legitimately run for
    /// minutes, and the device reports 503 while its engine works.
    pub guard: Duration,
    /// Overall wall-clock budget of one job, including job creation; a
    /// safety net even against a device that keeps responding forever.
    pub total: Duration,
    /// Pause between ScanJobs POST retries.
    pub post_retry_pause: Duration,
    /// Pause between NextDocument polls.
    pub poll_pause: Duration,
}

impl Default for ScanJobTimings {
    fn default() -> Self {
        Self {
            guard: Duration::from_secs(120),
            total: Duration::from_secs(600),
            post_retry_pause: Duration::from_millis(1500),
            poll_pause: Duration::from_millis(1000),
        }
    }
}

/// Coarse phase of a running job, for the status line.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScanJobPhase {
    /// Creating the job (includes busy retries).
    Starting,
    /// The job exists; the engine is working up to delivering the document.
    Scanning,
    /// Image bytes are streaming.
    Transferring,
}

#[derive(Clone, Debug)]
pub enum ScanJobError {
    Canceled,
    Busy,
    TimedOut,
    Failed(String),
}

impl fmt::Display for ScanJobError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Canceled => write!(f, "scan cancelled"),
            Self::Busy => write!(f, "scanner busy — in use by another application"),
            Self::TimedOut => write!(f, "the scanner did not deliver the scan in time"),
            Self::Failed(message) => write!(f, "{message}"),
        }
    }
}

/// How many consecutive busy polls may pass before ScannerStatus is
/// consulted, and how often a genuinely lost job is recreated (max 2
/// recreations = 3 job POSTs total).
const STALL_CHECK_POLLS: u32 = 8;
const MAX_JOB_RECREATIONS: u32 = 2;

enum PollTurn {
    Done(Result<u64, ScanJobError>),
    /// The device reports Idle while our job is still pending: it lost the
    /// job (known Epson behavior after an aborted scan) and a fresh job
    /// must be created.
    Stalled,
}

/// Run one scan job end to end: create the job, poll for the document,
/// stream it into a `<destination>.rekesz-part` file, rename on success, and
/// `DELETE` the job on every exit path. Returns the delivered byte count as
/// recorded by the filesystem (devices may stream chunked responses without
/// a declared length).
///
/// If the device keeps answering "still processing" while its ScannerStatus
/// reports Idle, it has lost the job (observed on Epson devices after an
/// aborted scan): the job is deleted and recreated, up to
/// [`MAX_JOB_RECREATIONS`] times.
pub fn run_scan_job(
    client: &EsclClient,
    settings: &ScanSettings,
    destination: &Path,
    cancel: &AtomicBool,
    timings: &ScanJobTimings,
    progress: &mut dyn FnMut(ScanJobPhase),
) -> Result<u64, ScanJobError> {
    progress(ScanJobPhase::Starting);
    // The clock starts before the POST: a wedged device during job creation
    // is bounded by the same budgets as the polling phase.
    let started = Instant::now();
    let mut last_activity = Instant::now();
    let partial = destination.with_extension("rekesz-part");
    let mut transport_failures = 0_u32;
    let mut recreations = 0_u32;
    let result = loop {
        progress(ScanJobPhase::Starting);
        let created = match client.start_job(settings, cancel, timings.post_retry_pause) {
            Ok(job_path) => job_path,
            Err(EsclError::Busy) => break Err(ScanJobError::Busy),
            Err(EsclError::Canceled) => break Err(ScanJobError::Canceled),
            Err(error) => break Err(ScanJobError::Failed(error.to_string())),
        };
        let mut busy_polls = 0_u32;
        let turn: PollTurn = loop {
            if cancel.load(Ordering::Acquire) {
                break PollTurn::Done(Err(ScanJobError::Canceled));
            }
            if last_activity.elapsed() > timings.guard || started.elapsed() > timings.total {
                break PollTurn::Done(Err(ScanJobError::TimedOut));
            }
            progress(ScanJobPhase::Scanning);
            let _ = std::fs::remove_file(&partial);
            let mut file = match std::fs::File::create(&partial) {
                Ok(file) => file,
                Err(error) => break PollTurn::Done(Err(ScanJobError::Failed(error.to_string()))),
            };
            let mut transferring = false;
            let poll = client.next_document(&created, &mut file, cancel, &mut |bytes| {
                if !transferring && bytes > 0 {
                    transferring = true;
                    progress(ScanJobPhase::Transferring);
                }
            });
            drop(file);
            match &poll {
                Ok(
                    NextDocument::Busy
                    | NextDocument::JobStatus(_)
                    | NextDocument::Gone
                    | NextDocument::Document(_),
                ) => {
                    // Any answer from the device — even a busy one — is a
                    // sign of life, and ends any failure streak.
                    last_activity = Instant::now();
                    transport_failures = 0;
                }
                Err(_) => {
                    // No answer is no liveness: the idle guard measures from
                    // the job start again. Consecutive failures are counted
                    // so a device answering with garbage cannot poll
                    // forever, while isolated hiccups stay cheap.
                    last_activity = started;
                    transport_failures += 1;
                }
            }
            match poll {
                Ok(NextDocument::Document(_)) => {
                    if let Err(error) = std::fs::rename(&partial, destination) {
                        break PollTurn::Done(Err(ScanJobError::Failed(error.to_string())));
                    }
                    let size = std::fs::metadata(destination)
                        .map(|metadata| metadata.len())
                        .unwrap_or(0);
                    break PollTurn::Done(Ok(size));
                }
                Ok(NextDocument::JobStatus(state)) => {
                    // Aborted/Canceled are terminal: no document will come.
                    // Completed keeps polling — the document is typically
                    // available on the next NextDocument read.
                    if matches!(state.as_deref(), Some("Aborted") | Some("Canceled")) {
                        break PollTurn::Done(Err(ScanJobError::Failed(
                            "the scanner aborted the job".to_string(),
                        )));
                    }
                }
                Ok(NextDocument::Busy) => {
                    // A long legitimate scan answers 503 for minutes; but if
                    // this drags on while the device reports Idle, it has
                    // lost our job and a fresh one is needed.
                    busy_polls += 1;
                    if busy_polls.is_multiple_of(STALL_CHECK_POLLS)
                        && client.scanner_state().as_deref() == Some("Idle")
                    {
                        break PollTurn::Stalled;
                    }
                }
                Ok(NextDocument::Gone) => {
                    break PollTurn::Done(Err(ScanJobError::Failed(
                        "the scanner ended the job before delivering the image".to_string(),
                    )));
                }
                Err(EsclError::Canceled) => {
                    break PollTurn::Done(Err(ScanJobError::Canceled));
                }
                Err(EsclError::Io(_)) if cancel.load(Ordering::Acquire) => {
                    break PollTurn::Done(Err(ScanJobError::Canceled));
                }
                Err(error) => {
                    if transport_failures >= 10 {
                        break PollTurn::Done(Err(ScanJobError::Failed(error.to_string())));
                    }
                }
            }
            std::thread::sleep(timings.poll_pause.max(Duration::from_millis(1)));
        };
        // The job of this turn is over either way: clean up its artifacts.
        let _ = std::fs::remove_file(&partial);
        client.delete_job(&created);
        match turn {
            PollTurn::Done(result) => break result,
            PollTurn::Stalled => {
                recreations += 1;
                if recreations > MAX_JOB_RECREATIONS {
                    break Err(ScanJobError::Failed(
                        "the scanner accepted the scan but never delivered the image".to_string(),
                    ));
                }
                // A recreated job is new liveness.
                last_activity = Instant::now();
            }
        }
    };
    if result.is_err() {
        let _ = std::fs::remove_file(&partial);
    }
    result
}
