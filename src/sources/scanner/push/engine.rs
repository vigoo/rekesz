//! Push-scan engine: the shared state between the UI and the two socket
//! loops (announcement listener, trigger listener), plus the admission
//! policy for panel pushes. See scanner-push-scan-plan.md.
//!
//! Threading contract: the engine threads never touch gpui. The scan-panel
//! poll chain publishes [`EngineInput`] every 250 ms and consumes accepted
//! triggers from [`EngineOutput`]; the socket loops read the input at
//! decision time and write the output. The engine runs unconditionally
//! (process-wide sockets), but stays silent unless the UI has armed it —
//! unanswered announcements leave the scanner panel unchanged, exactly as
//! if the engine were not running.

use std::collections::HashMap;
use std::net::{Ipv4Addr, SocketAddr, UdpSocket};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use super::announce;
use crate::trace;

/// Multicast group + port the printer announces push-scan availability on.
pub const ANNOUNCE_GROUP: Ipv4Addr = Ipv4Addr::new(239, 255, 255, 253);
pub const ANNOUNCE_PORT: u16 = 2968;
/// TCP event port advertised in the keepalive; the printer POSTs the
/// push-scan trigger here.
pub const TRIGGER_PORT: u16 = 2968;
/// Keepalive burst: the reference sends three, 500 ms apart.
pub const KEEPALIVE_BURST: usize = 3;
pub const KEEPALIVE_INTERVAL: Duration = Duration::from_millis(500);
/// Repeat announcements (each burst arrives 3×) dedupe on (source, seq).
pub const ANNOUNCE_DEDUPE_WINDOW: Duration = Duration::from_secs(30);
/// An armed flag older than this is treated as disarmed: a stopped poll
/// chain (scanner deselected, view left) must disarm the engine even
/// without an explicit teardown call.
pub const TICK_STALENESS: Duration = Duration::from_secs(2);
/// Destination ID in keepalive replies. The real Event Manager capture the
/// reference pins its tests to carries 0x02 here (as does the L3160's own
/// announcement at byte 13); a keepalive with a different ID is silently
/// ignored by the printer.
pub const DESTINATION_ID: u8 = 0x02;

/// What the UI tells the engine. Published by the scan-panel poll chain.
#[derive(Clone, Debug)]
pub struct EngineInput {
    /// Scanner source selected with capabilities loaded and the import view
    /// active; keepalives are only answered, triggers only accepted, while
    /// this holds.
    pub armed: bool,
    /// A scan or copy job is in flight — panel triggers are refused with
    /// the protocol ERROR response rather than left hanging.
    pub busy: bool,
    /// Endpoint addresses of the selected scanner; announcements from any
    /// other source are ignored (other Epsons on the LAN must not trigger).
    pub scanner_ips: Vec<std::net::IpAddr>,
    /// Destination name shown on the scanner panel ("Rekesz").
    pub destination_name: String,
    /// Last poll tick; arms go stale after [`TICK_STALENESS`].
    pub ticked_at: Option<Instant>,
    /// The scan panel's push-session settings (None = caps missing).
    pub scan: Option<PushScanSettings>,
    /// The device's session scan directory; push output lands here.
    pub scan_dir: Option<std::path::PathBuf>,
}

/// The scan settings a push session runs with, published by the UI each tick.
#[derive(Clone, Copy, Debug)]
pub struct PushScanSettings {
    /// Wire DPI (the panel's clamped pick).
    pub dpi: u32,
    /// The crop region as fractions of the platen (x0, y0, x1, y1).
    pub region: (f64, f64, f64, f64),
    /// The panel's colour mode; black-and-white maps to grayscale (device
    /// JPEG output is 8-bit per channel).
    pub color: super::dialect::ParaColorMode,
}

impl Default for EngineInput {
    fn default() -> Self {
        EngineInput {
            armed: false,
            busy: false,
            scanner_ips: Vec::new(),
            destination_name: "Rekesz".to_string(),
            ticked_at: None,
            scan: None,
            scan_dir: None,
        }
    }
}

/// What the engine tells the UI. Consumed by the scan-panel poll chain.
#[derive(Clone, Debug, Default)]
pub struct EngineOutput {
    /// A native ESC/I-2 session is running: admission refuses new pushes,
    /// and the UI shows the scanning status.
    pub scan_active: bool,
    /// A finished push scan: the staged JPEG's path, or the failure reason.
    pub scan_done: Option<Result<std::path::PathBuf, String>>,
}

/// The panel-trigger admission decision for a PushScan POST.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Admission {
    /// OK response, and the UI starts the bridged scan.
    Accept,
    /// OK response, no scan — the panel's email/preview actions are
    /// unsupported here (the panel then reports a communication problem,
    /// same as the reference).
    OkNoSession,
    /// ERROR response: unarmed, busy, or a trigger is already pending.
    Refuse,
}

pub fn admit_push_scan(
    input: &EngineInput,
    action: u8,
    session_active: bool,
    now: Instant,
) -> Admission {
    let armed = input
        .ticked_at
        .is_some_and(|at| now.duration_since(at) < TICK_STALENESS)
        && input.armed;
    // A session in flight or a push while another waits to be consumed both
    // refuse; the panel needs scan settings + a scan directory to be usable
    // at all.
    if !armed || input.busy || session_active || input.scan.is_none() || input.scan_dir.is_none() {
        return Admission::Refuse;
    }
    match action {
        1 | 2 => Admission::Accept,
        _ => Admission::OkNoSession,
    }
}

/// The engine singleton: shared state plus the socket loops (started once,
/// process-wide; tests run with the loops disabled but the state usable).
pub struct PushEngine {
    input: Arc<Mutex<EngineInput>>,
    output: Arc<Mutex<EngineOutput>>,
}

impl PushEngine {
    fn new() -> PushEngine {
        PushEngine {
            input: Arc::new(Mutex::new(EngineInput::default())),
            output: Arc::new(Mutex::new(EngineOutput::default())),
        }
    }

    /// Publish a UI-side update to the engine inputs.
    pub fn publish_input(&self, update: impl FnOnce(&mut EngineInput)) {
        if let Ok(mut input) = self.input.lock() {
            update(&mut input);
        }
    }

    /// Consume a finished push scan's result, if one waits.
    pub fn take_scan_done(&self) -> Option<Result<std::path::PathBuf, String>> {
        self.output
            .lock()
            .ok()
            .and_then(|mut output| output.scan_done.take())
    }

    /// Snapshot of the engine-output flags the UI reads each tick.
    pub fn output_snapshot(&self) -> EngineOutput {
        self.output
            .lock()
            .map(|output| output.clone())
            .unwrap_or_default()
    }

    fn input(&self) -> Arc<Mutex<EngineInput>> {
        Arc::clone(&self.input)
    }

    fn output(&self) -> Arc<Mutex<EngineOutput>> {
        Arc::clone(&self.output)
    }

    fn spawn_threads(&self) {
        let input = self.input();
        std::thread::Builder::new()
            .name("push-announce".to_string())
            .spawn(move || announce::run_announce_loop(input))
            .expect("spawn push-announce thread");
        let input = self.input();
        let output = self.output();
        std::thread::Builder::new()
            .name("push-trigger".to_string())
            .spawn(move || super::trigger::run_trigger_loop(input, output))
            .expect("spawn push-trigger thread");
    }
}

static ENGINE: std::sync::LazyLock<PushEngine> = std::sync::LazyLock::new(|| {
    let engine = PushEngine::new();
    // House invariant: tests never bind real ports. The socket loops exist
    // only in the real app build; under `cargo test` the shared state stays
    // usable (admission policy, UI plumbing) without any listener.
    #[cfg(not(test))]
    {
        if crate::sources::scanner::daemon_enabled() {
            engine.spawn_threads();
        } else {
            trace::event("push: engine loops disabled (daemon gating)");
        }
    }
    engine
});

/// The process-wide push-scan engine.
pub fn push_engine() -> &'static PushEngine {
    &ENGINE
}

/// Is the armed flag fresh enough to answer the printer with?
pub fn armed_now(input: &EngineInput, now: Instant) -> bool {
    input.armed
        && input
            .ticked_at
            .is_some_and(|at| now.duration_since(at) < TICK_STALENESS)
}

/// Announcement dedupe: (source, seq) → first seen. Repeat announcements of
/// the same burst (the printer sends each 3×) answer once.
pub struct AnnounceDedupe {
    seen: HashMap<(std::net::IpAddr, u8), Instant>,
}

impl AnnounceDedupe {
    pub fn new() -> AnnounceDedupe {
        AnnounceDedupe {
            seen: HashMap::new(),
        }
    }

    /// True when this (source, seq) burst is new within the dedupe window.
    pub fn first_of_burst(&mut self, source: std::net::IpAddr, seq: u8, now: Instant) -> bool {
        let window = ANNOUNCE_DEDUPE_WINDOW;
        self.seen
            .retain(|_, seen| now.duration_since(*seen) < window);
        self.seen
            .insert((source, seq), now)
            .is_none_or(|seen| now.duration_since(seen) >= window)
    }

    /// Drop a burst entry so the next announcement can retry it (used when
    /// a keepalive burst could not start, e.g. no route to the printer).
    pub fn forget(&mut self, source: std::net::IpAddr, seq: u8) {
        self.seen.remove(&(source, seq));
    }
}

/// The local address facing the printer, resolved per announcement (no new
/// dependencies: a connected UDP socket reports its local endpoint).
pub fn local_ip_facing(printer: std::net::IpAddr) -> Option<std::net::IpAddr> {
    let socket = UdpSocket::bind(SocketAddr::from(([0, 0, 0, 0], 0))).ok()?;
    socket
        .connect(SocketAddr::new(printer, ANNOUNCE_PORT))
        .ok()?;
    socket.local_addr().ok().map(|address| address.ip())
}
