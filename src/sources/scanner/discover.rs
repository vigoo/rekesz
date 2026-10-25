//! mDNS discovery of driverless eSCL scanners (Apple AirScan / Mopria Scan).
//!
//! A single daemon thread sweeps `_uscans._tcp` (TLS) and `_uscan._tcp`
//! advertisements and keeps a shared snapshot in [`SCANNERS`], keyed by the
//! device UUID so entries survive address changes. The import view's source
//! refresh only reads the snapshot; sweeps never run on the poll path.
//!
//! Normalizations follow sane-airscan: the TXT `uuid` value carries a
//! `uuid:` prefix which is stripped (devices without one fall back to a key
//! derived from the instance name), and the `rs` resource prefix is trimmed
//! and defaults to `eSCL`.

use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::{Arc, LazyLock, Mutex, OnceLock};
use std::time::{Duration, Instant};

use mdns_sd::{ResolvedService, ServiceDaemon, ServiceEvent};

/// Shared snapshot of discovered scanners, keyed by the stable device key
/// (see [`device_key`]). Guarded by a plain mutex: readers hold it for the
/// microseconds it takes to clone entries.
pub static SCANNERS: LazyLock<Mutex<HashMap<Arc<str>, ScannerDevice>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

pub const SERVICE_TLS: &str = "_uscans._tcp.local.";
pub const SERVICE_PLAIN: &str = "_uscan._tcp.local.";

const SCAN_WINDOW: Duration = Duration::from_millis(2500);
const SWEEP_INTERVAL: Duration = Duration::from_secs(5);
const EXPIRY: Duration = Duration::from_secs(30);

/// One reachable transport of a scanner. The base URL carries the SRV
/// port; non-default ports then also appear in the Host header.
#[derive(Clone, Debug)]
pub struct ScannerEndpoint {
    pub base_url: String,
    pub tls: bool,
    pub last_seen: Instant,
}

/// A discovered scanner. `name` is the human-readable model name from the
/// TXT `ty` key ("EPSON L3160 Series").
#[derive(Clone, Debug)]
pub struct ScannerDevice {
    pub key: Arc<str>,
    pub name: String,
    /// TLS transports first; at most one endpoint per transport is kept.
    pub endpoints: Vec<ScannerEndpoint>,
    pub last_seen: Instant,
}

impl ScannerDevice {
    /// The endpoint to talk to: TLS preferred, then the most recently seen.
    pub fn preferred_endpoint(&self) -> Option<&ScannerEndpoint> {
        self.endpoints
            .iter()
            .max_by_key(|endpoint| (endpoint.tls, endpoint.last_seen))
    }
}

/// One resolved mDNS advertisement, decoupled from the mdns-sd types so the
/// normalization logic below is testable without a network.
pub struct DiscoveredService {
    /// e.g. `_uscans._tcp.local.`; the transport is detected from this name.
    pub service_type: String,
    /// DNS-SD instance name ("EPSON L3160 Series").
    pub instance_name: String,
    pub port: u16,
    /// Resolved addresses; IPv4 preferred when several exist.
    pub addresses: Vec<IpAddr>,
    /// The TXT keys we consume (`rs`, `ty`, `uuid`).
    pub txt: Vec<(String, String)>,
}

fn txt_value<'a>(txt: &'a [(String, String)], key: &str) -> Option<&'a str> {
    txt.iter()
        .find(|(name, _)| name == key)
        .map(|(_, value)| value.as_str())
}

/// Stable identity of a scanner: the TXT `uuid` with its `uuid:` prefix
/// stripped, or the instance name when the device advertises no usable one.
/// An empty or whitespace uuid counts as absent — treating it as a key
/// would merge every such device into one.
pub fn device_key(txt: &[(String, String)], instance_name: &str) -> Arc<str> {
    match txt_value(txt, "uuid")
        .map(str::trim)
        .filter(|uuid| !uuid.is_empty())
    {
        Some(uuid) => Arc::from(uuid.strip_prefix("uuid:").unwrap_or(uuid)),
        None => Arc::from(format!("name:{instance_name}")),
    }
}

/// eSCL URI prefix from the TXT `rs` key, trimmed of slashes; "eSCL" when
/// absent (the convention sane-airscan applies).
pub fn resource_prefix(txt: &[(String, String)]) -> String {
    let prefix = txt_value(txt, "rs").unwrap_or("eSCL");
    prefix.trim_matches('/').to_string()
}

/// Base URL of an endpoint, or None when the service resolved no usable
/// address. IPv6 link-local addresses are skipped: their interface scope is
/// lost in DNS-SD resolution, and `http://[fe80::…]/` without a zone can
/// never connect — such a device would look discovered forever and never
/// answer.
pub fn endpoint_base_url(
    tls: bool,
    addresses: &[IpAddr],
    port: u16,
    prefix: &str,
) -> Option<String> {
    let address = addresses
        .iter()
        .find(|address| matches!(address, IpAddr::V4(_)))
        .or_else(|| addresses.iter().find(|address| !is_link_local_v6(address)))?;
    let host = match address {
        IpAddr::V4(v4) => v4.to_string(),
        IpAddr::V6(v6) => format!("[{v6}]"),
    };
    let scheme = if tls { "https" } else { "http" };
    Some(format!("{scheme}://{host}:{port}/{prefix}"))
}

/// `fe80::/10` — link-local IPv6 without its interface scope is
/// unconnectable.
fn is_link_local_v6(address: &IpAddr) -> bool {
    match address {
        IpAddr::V6(v6) => (v6.segments()[0] & 0xffc0) == 0xfe80,
        IpAddr::V4(_) => false,
    }
}

/// Insert or refresh an advertisement in the snapshot. A second advertisement
/// of the same device on the other transport adds an endpoint; re-seeing one
/// refreshes its `last_seen`. TLS endpoints sort first.
pub fn note_service(
    map: &mut HashMap<Arc<str>, ScannerDevice>,
    service: &DiscoveredService,
    now: Instant,
) {
    let tls = service.service_type.starts_with("_uscans");
    let prefix = resource_prefix(&service.txt);
    let Some(base_url) = endpoint_base_url(tls, &service.addresses, service.port, &prefix) else {
        return;
    };
    let key = device_key(&service.txt, &service.instance_name);
    let name = txt_value(&service.txt, "ty")
        .unwrap_or(&service.instance_name)
        .to_string();
    let device = map.entry(key.clone()).or_insert_with(|| ScannerDevice {
        key,
        name: name.clone(),
        endpoints: Vec::new(),
        last_seen: now,
    });
    device.name = name;
    device.last_seen = now;
    match device
        .endpoints
        .iter_mut()
        .find(|endpoint| endpoint.tls == tls)
    {
        Some(endpoint) => {
            endpoint.base_url = base_url;
            endpoint.last_seen = now;
        }
        None => device.endpoints.push(ScannerEndpoint {
            base_url,
            tls,
            last_seen: now,
        }),
    }
    device
        .endpoints
        .sort_by_key(|endpoint| std::cmp::Reverse(endpoint.tls));
}

/// Drop devices not re-seen within [`EXPIRY`] of `now`.
pub fn expire_devices(map: &mut HashMap<Arc<str>, ScannerDevice>, now: Instant) {
    let Some(cutoff) = now.checked_sub(EXPIRY) else {
        return;
    };
    map.retain(|_, device| device.last_seen >= cutoff);
}

/// `EPSON L3160 Series._uscans._tcp.local.` -> `EPSON L3160 Series`. The
/// known service suffix is stripped whole: instance names may contain
/// literal dots, and splitting at the first dot would truncate them.
pub(crate) fn instance_name(fullname: &str) -> &str {
    for suffix in ["._uscans._tcp.local.", "._uscan._tcp.local."] {
        if let Some(name) = fullname.strip_suffix(suffix) {
            return name;
        }
    }
    fullname.split('.').next().unwrap_or(fullname)
}

fn service_from_resolved(service_type: &str, resolved: &ResolvedService) -> DiscoveredService {
    let txt = ["rs", "ty", "uuid"]
        .iter()
        .filter_map(|key| {
            resolved
                .get_property_val_str(key)
                .map(|value| (key.to_string(), value.to_string()))
        })
        .collect();
    let mut addresses: Vec<IpAddr> = resolved
        .get_addresses_v4()
        .into_iter()
        .map(IpAddr::V4)
        .collect();
    if addresses.is_empty() {
        addresses = resolved
            .get_addresses()
            .iter()
            .map(|scoped| scoped.to_ip_addr())
            .collect();
    }
    DiscoveredService {
        service_type: service_type.to_string(),
        instance_name: instance_name(resolved.get_fullname()).to_string(),
        port: resolved.get_port(),
        addresses,
        txt,
    }
}

/// One sweep: browse both transports for [`SCAN_WINDOW`], folding every
/// resolved advertisement into the snapshot, then stop the subscriptions.
/// Each `browse` call is a fresh subscription carrying one initial query —
/// mdns-sd does not re-query an existing browse on human timescales — and a
/// browse whose bounded channel fills silently unsubscribes, so events are
/// drained continuously and the receivers are dropped at sweep end.
fn sweep(daemon: &ServiceDaemon) {
    let mut receivers = Vec::new();
    for service_type in [SERVICE_TLS, SERVICE_PLAIN] {
        match daemon.browse(service_type) {
            Ok(receiver) => receivers.push((service_type, receiver)),
            Err(_) => continue,
        }
    }
    if receivers.is_empty() {
        return;
    }
    let deadline = Instant::now() + SCAN_WINDOW;
    while Instant::now() < deadline {
        for (service_type, receiver) in &receivers {
            while let Ok(event) = receiver.try_recv() {
                if let ServiceEvent::ServiceResolved(resolved) = event {
                    let service = service_from_resolved(service_type, &resolved);
                    let mut map = SCANNERS.lock().expect("scanner registry lock");
                    note_service(&mut map, &service, Instant::now());
                }
            }
        }
        // Floor sleep: keeps the drain thread idle-paced even under constant
        // mDNS chatter instead of busy-spinning.
        std::thread::sleep(Duration::from_millis(25));
    }
    for (service_type, _) in &receivers {
        let _ = daemon.stop_browse(service_type);
    }
}

fn daemon_loop() {
    let Ok(daemon) = ServiceDaemon::new() else {
        return;
    };
    loop {
        sweep(&daemon);
        {
            let mut map = SCANNERS.lock().expect("scanner registry lock");
            expire_devices(&mut map, Instant::now());
        }
        std::thread::sleep(SWEEP_INTERVAL);
    }
}

/// Start the discovery thread once per process. The first snapshot entries
/// appear after the first sweep (~2.5 s).
pub fn ensure_daemon() {
    static STARTED: OnceLock<()> = OnceLock::new();
    STARTED.get_or_init(|| {
        let _ = std::thread::Builder::new()
            .name("scanner-discovery".to_string())
            .spawn(daemon_loop);
    });
}
