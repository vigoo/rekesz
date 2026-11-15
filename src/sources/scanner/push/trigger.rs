//! Push-scan trigger listener (TCP 2968): the printer POSTs a SOAP-ish
//! request here after the user picks our destination and presses OK on the
//! panel. Epson's headers are `"Name : value"` (space before the colon —
//! invalid HTTP), so requests are parsed on the raw socket, not with an
//! HTTP library. The response is a byte-exact port of the reference's
//! `buildPushScanResponse`. See scanner-push-scan-plan.md.

use std::io::{Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use super::engine::{self, Admission, EngineInput, EngineOutput};
use crate::trace;

/// Epson's push-scan SOAP namespace (schema constant in the reference).
const XML_NAMESPACE: &str = "http://schema.epson.net/EpsonNet/Scan/2004/pushscan";
/// A silent client must not hold the listener; the reference idles out at
/// 10 s and caps requests at 64 KiB.
const IDLE_TIMEOUT: Duration = Duration::from_secs(10);
const MAX_REQUEST_BYTES: usize = 64 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TriggerKind {
    JobList,
    PushScan,
    ScanEnd,
    Unknown,
}

/// The parsed pieces of an incoming trigger request.
#[derive(Clone, Debug)]
pub struct TriggerRequest {
    pub kind: TriggerKind,
    /// Verbatim `x-uid` echo value ("1" when the request has none).
    pub xuid: String,
    /// Mirrored `x-protocol-version` ("2.00" default).
    pub protocol_version: String,
    /// First char of `PushScanIDIn`: sides ('0' simplex, '1' duplex).
    pub sides: Option<char>,
    /// Second char of `PushScanIDIn`: action ('1' JPG, '2' PDF, '3' email,
    /// '4' preview, anything else unknown).
    pub action: Option<char>,
    /// `CapabilityIn` values from the request body (jobList requests).
    pub capabilities: Vec<String>,
}

/// Read and parse one trigger request. Returns None when the client is
/// silent, oversized, or unparseable — the connection is simply dropped,
/// like the reference's over-cap path.
pub fn read_request(stream: &mut TcpStream) -> Option<TriggerRequest> {
    let _ = stream.set_read_timeout(Some(IDLE_TIMEOUT));
    let mut buffer = Vec::new();
    let mut chunk = [0_u8; 4096];
    let header_end;
    loop {
        if let Some(position) = find_subslice(&buffer, b"\r\n\r\n") {
            header_end = position + 4;
            break;
        }
        if buffer.len() > MAX_REQUEST_BYTES {
            return None;
        }
        let read = stream.read(&mut chunk).ok()?;
        if read == 0 {
            return None;
        }
        buffer.extend_from_slice(&chunk[..read]);
    }
    let headers = String::from_utf8_lossy(&buffer[..header_end]).into_owned();
    // Clamped before any arithmetic: a hostile length must not overflow or
    // slice-panic the connection thread.
    let content_length = header_value(&headers, "content-length")
        .and_then(|value| value.trim().parse::<usize>().ok())
        .map(|length| length.min(MAX_REQUEST_BYTES))
        .unwrap_or(0);
    if header_end + content_length > MAX_REQUEST_BYTES {
        return None;
    }
    while buffer.len() < header_end + content_length {
        let read = stream.read(&mut chunk).ok()?;
        if read == 0 {
            return None;
        }
        buffer.extend_from_slice(&chunk[..read]);
    }
    let body =
        String::from_utf8_lossy(&buffer[header_end..header_end + content_length]).into_owned();
    let xuid = header_value(&headers, "x-uid").unwrap_or("1").to_string();
    let protocol_version = header_value(&headers, "x-protocol-version")
        .unwrap_or("2.00")
        .to_string();
    let sides_action = extract_element(&body, "PushScanIDIn");
    let (sides, action) = match sides_action.as_deref().map(str::chars) {
        Some(mut chars) => (chars.next(), chars.next()),
        None => (None, None),
    };
    Some(TriggerRequest {
        kind: classify_kind(&body),
        xuid,
        protocol_version,
        sides,
        action,
        capabilities: extract_capabilities(&body),
    })
}

/// First occurrence of `Name : value` in the header block, tolerating the
/// Epson `" : "` spacing; names compare case-insensitively.
fn header_value<'a>(headers: &'a str, name: &str) -> Option<&'a str> {
    for line in headers.lines() {
        let Some((field, value)) = line.split_once(':') else {
            continue;
        };
        if field.trim().eq_ignore_ascii_case(name) {
            return Some(value.trim());
        }
    }
    None
}

/// Classify the request by its SOAP root element, JobList first (the
/// reference's order); element names may carry a namespace prefix.
fn classify_kind(body: &str) -> TriggerKind {
    for (tag, kind) in [
        ("JobList", TriggerKind::JobList),
        ("PushScan", TriggerKind::PushScan),
        ("ScanEnd", TriggerKind::ScanEnd),
    ] {
        if element_present(body, tag) {
            return kind;
        }
    }
    TriggerKind::Unknown
}

/// Does `<Tag` (optionally namespace-prefixed `<x:Tag`) occur in `body`?
fn element_present(body: &str, tag: &str) -> bool {
    let bytes = body.as_bytes();
    for index in 0..bytes.len() {
        if bytes[index] != b'<' {
            continue;
        }
        let rest = &body[index + 1..];
        let name = rest
            .find(|c: char| c == '>' || c == '<' || c.is_ascii_whitespace())
            .map(|end| &rest[..end])
            .unwrap_or(rest);
        let name = name.strip_suffix('/').unwrap_or(name);
        let name = name.rsplit(':').next().unwrap_or(name);
        if name == tag {
            return true;
        }
    }
    false
}

/// Text of the first `<Tag>value</Tag>` and the byte offset just past its
/// closing tag (namespace prefixes tolerated). Closing tags (`</p:Tag>`)
/// must not match: walk back over the prefix and require a literal `<`
/// before the name, like the reference's regex.
pub fn extract_element_from(body: &str, tag: &str) -> Option<(String, usize)> {
    let open_len = tag.len() + 2; // '<' + tag + '>'
    let bytes = body.as_bytes();
    let mut search_from = 0;
    loop {
        let open = body[search_from..].find(tag)?;
        let absolute = search_from + open;
        // Walk back over `word chars + ':'`; only a literal '<' may precede.
        let mut prefix_start = absolute;
        while prefix_start > 0
            && (bytes[prefix_start - 1].is_ascii_alphanumeric() || bytes[prefix_start - 1] == b':')
        {
            prefix_start -= 1;
        }
        let prefix_ok = prefix_start > 0 && bytes[prefix_start - 1] == b'<';
        let after = absolute + tag.len();
        if prefix_ok && body[after..].starts_with('>') {
            let value_start = after + 1;
            let close = body[value_start..].find("</")?;
            let value = body[value_start..value_start + close].to_string();
            let close_start = value_start + close;
            let close_end = close_start + body[close_start..].find('>')? + 1;
            return Some((value, close_end));
        }
        search_from = absolute + open_len.max(1);
        if search_from >= body.len() {
            return None;
        }
    }
}

/// Text of the first `<Tag>value</Tag>` (namespace prefixes tolerated).
pub fn extract_element(body: &str, tag: &str) -> Option<String> {
    extract_element_from(body, tag).map(|(value, _)| value)
}

/// Every `<CapabilityIn>value</CapabilityIn>` in the body (jobList echoes).
fn extract_capabilities(body: &str) -> Vec<String> {
    let mut capabilities = Vec::new();
    let mut search_from = 0;
    while let Some((value, end)) = extract_element_from(&body[search_from..], "CapabilityIn") {
        capabilities.push(value);
        search_from += end;
    }
    capabilities
}

/// The full HTTP/1.0 response: byte-exact port of the reference's
/// `buildPushScanResponse` (headers in `" : "` Epson style, `x-uid` echoed
/// verbatim, `x-status` hardcoded 0001, capabilities only for OK).
pub fn build_response(
    status: &str,
    protocol_version: &str,
    xuid: &str,
    capabilities: &[String],
) -> String {
    let mut body = format!(
        "<?xml version=\"1.0\" ?>\r\n\
         <s:Envelope xmlns:s=\"http://www.w3.org/2003/05/soap-envelope\">\r\n\
         \x20 <s:Body>\r\n\
         \x20   <p:PushScanResponse xmlns:p=\"{XML_NAMESPACE}\">\r\n\
         \x20     <StatusOut>{status}</StatusOut>\r\n"
    );
    if status == "OK" {
        for capability in capabilities {
            body.push_str(&format!(
                "      <CapabilityOut>{capability}</CapabilityOut>\r\n"
            ));
        }
    }
    body.push_str("    </p:PushScanResponse>\r\n");
    body.push_str("  </s:Body>\r\n</s:Envelope>\r\n");
    format!(
        "HTTP/1.0 200 OK\r\n\
         Server : Epson Net Scan Monitor/2.0\r\n\
         Content-Type : application/octet-stream\r\n\
         Content-Length : {}\r\n\
         x-protocol-name : Epson Network Service Protocol\r\n\
         x-protocol-version : {}\r\n\
         x-uid : {}\r\n\
         x-status : 0001\r\n\
         \r\n\
         {body}",
        body.len(),
        protocol_version,
        xuid
    )
}

/// The trigger listener: accept on TCP 2968 and answer each connection on
/// its own short-lived thread. SO_REUSEADDR is set pre-bind (socket2) so a
/// restart within the TIME_WAIT window of a finished push does not strand
/// the feature — on Windows the option is skipped: its TCP semantics allow
/// hijacking a live listener, which is worse than a delayed restart.
pub fn run_trigger_loop(input: Arc<Mutex<EngineInput>>, output: Arc<Mutex<EngineOutput>>) {
    let listener = match bind_trigger_listener() {
        Ok(listener) => listener,
        Err(error) => {
            trace::slow_event(&format!("push: trigger listener failed to start: {error}"));
            return;
        }
    };
    trace::event("push: trigger listener up (tcp 2968)");
    for stream in listener.incoming() {
        let Ok(stream) = stream else {
            continue;
        };
        let peer = stream
            .peer_addr()
            .map(|address| address.to_string())
            .unwrap_or_else(|_| "<unknown>".to_string());
        trace::event(&format!("push: trigger connection accepted from {peer}"));
        let input = Arc::clone(&input);
        let output = Arc::clone(&output);
        let _ = std::thread::Builder::new()
            .name("push-trigger-conn".to_string())
            .spawn(move || {
                handle_connection(stream, &input, &output);
            });
    }
}

/// Bind the trigger listener; the reuse option goes on before bind via
/// socket2 (std has no way to set it, and post-bind is a no-op).
fn bind_trigger_listener() -> std::io::Result<TcpListener> {
    use socket2::{Domain, Protocol, Socket, Type};
    let socket = Socket::new(Domain::IPV4, Type::STREAM, Some(Protocol::TCP))?;
    #[cfg(unix)]
    socket.set_reuse_address(true)?;
    socket.bind(
        &std::net::SocketAddr::from((std::net::Ipv4Addr::UNSPECIFIED, engine::TRIGGER_PORT)).into(),
    )?;
    socket.listen(4)?;
    Ok(TcpListener::from(socket))
}

/// Answer one trigger connection: validate the peer, parse, admit, respond,
/// half-close. The admission reservation is made inside the decision lock so
/// two concurrent POSTs cannot both pass the pending check (the reference
/// reserves before responding and abandons the reservation on failure).
pub fn handle_connection(
    mut stream: TcpStream,
    input: &Arc<Mutex<EngineInput>>,
    output: &Arc<Mutex<EngineOutput>>,
) {
    // Peer validation before a byte is read: only the selected scanner may
    // talk to the trigger port (the reference pauses until validated; a
    // rejected peer gets no response, just a close).
    let peer = match stream.peer_addr() {
        Ok(peer) => peer.ip(),
        Err(_) => return,
    };
    let from_selected_scanner = input
        .lock()
        .map(|input| input.scanner_ips.contains(&peer))
        .unwrap_or(false);
    if !from_selected_scanner {
        trace::event(&format!(
            "push: trigger from unselected peer {peer} — connection dropped"
        ));
        return;
    }
    let Some(request) = read_request(&mut stream) else {
        trace::event("push: trigger connection dropped (unparseable or oversized)");
        return;
    };
    trace::event(&format!(
        "push: trigger {:?} xuid={} version={} sides={:?} action={:?} caps={}",
        request.kind,
        request.xuid,
        request.protocol_version,
        request.sides,
        request.action,
        request.capabilities.len()
    ));
    let (status, capabilities, reserved) = match request.kind {
        TriggerKind::PushScan => {
            let now = std::time::Instant::now();
            let decision = match (input.lock(), output.lock()) {
                (Ok(input), Ok(mut output)) => {
                    let decision = engine::admit_push_scan(
                        &input,
                        action_byte(request.action),
                        output.scan_active,
                        now,
                    );
                    if decision == Admission::Accept {
                        // Reserve the session slot now; abandoned below if
                        // the response never goes out.
                        output.scan_active = true;
                    }
                    decision
                }
                _ => Admission::Refuse,
            };
            match decision {
                Admission::Accept => ("OK", Vec::new(), true),
                Admission::OkNoSession => ("OK", Vec::new(), false),
                Admission::Refuse => {
                    trace::event(
                        "push: push scan refused (disarmed, busy, or a session is active)",
                    );
                    ("ERROR", Vec::new(), false)
                }
            }
        }
        TriggerKind::JobList => ("OK", request.capabilities.clone(), false),
        TriggerKind::ScanEnd | TriggerKind::Unknown => ("OK", Vec::new(), false),
    };
    let response = build_response(
        status,
        &request.protocol_version,
        &request.xuid,
        &capabilities,
    );
    // Write + FIN half-close in the reference's socket.end pattern; a
    // missing or broken response makes the printer show a scanning error.
    let sent = stream.write_all(response.as_bytes()).is_ok();
    let _ = stream.flush();
    let _ = stream.shutdown(Shutdown::Write);
    // A close with unread receive-buffer bytes would RST; drain briefly.
    let _ = stream.set_read_timeout(Some(Duration::from_millis(200)));
    let mut sink = [0_u8; 4096];
    for _ in 0..4 {
        match stream.read(&mut sink) {
            Ok(0) | Err(_) => break,
            Ok(_) => continue,
        }
    }
    if !sent && reserved {
        // The reservation must not block real triggers after a dead write.
        if let Ok(mut output) = output.lock() {
            output.scan_active = false;
        }
    }
    if sent && reserved {
        // The native session replaces the stage-1 eSCL bridge: the printer
        // now expects the ESC/I-2 scan on port 1865 right after the answered
        // trigger. It runs on this connection thread; the result lands in
        // the output slot for the UI's poll chain to ingest.
        let (scan, scan_dir) = match input.lock() {
            Ok(input) => (input.scan, input.scan_dir.clone()),
            Err(_) => (None, None),
        };
        let dpi_note = scan.as_ref().map(|scan| scan.dpi.to_string());
        trace::event(&format!(
            "push: push scan accepted — running the ESC/I-2 session at {dpi_note:?} DPI"
        ));
        let result = match (scan, scan_dir) {
            (Some(scan), Some(scan_dir)) => {
                let cancel = std::sync::atomic::AtomicBool::new(false);
                // A panic mid-session must not skip the slot cleanup below:
                // catch it and report it as a failed scan instead.
                let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    super::session::run_flatbed(peer, scan, &cancel)
                }));
                let scanned = match outcome {
                    Ok(result) => result.map_err(|error| error.to_string()),
                    Err(panic) => Err(format!(
                        "the scan session panicked: {}",
                        panic
                            .downcast_ref::<&str>()
                            .map(|message| (*message).to_string())
                            .or_else(|| panic.downcast_ref::<String>().cloned())
                            .unwrap_or_else(|| "unknown panic".to_string())
                    )),
                };
                scanned.and_then(|jpeg| {
                    write_scan_file(&scan_dir, &jpeg)
                        .map_err(|error| format!("cannot write the scan file: {error}"))
                })
            }
            _ => Err("no scan settings or scan folder published".to_string()),
        };
        if let Ok(mut output) = output.lock() {
            output.scan_done = Some(result);
            output.scan_active = false;
        }
    }
}

/// Deliver the session's JPEG with the house staged-write convention:
/// `<destination>.rekesz-part` written whole, then renamed into place so a
/// torn write is never listable.
fn write_scan_file(scan_dir: &std::path::Path, jpeg: &[u8]) -> std::io::Result<std::path::PathBuf> {
    std::fs::create_dir_all(scan_dir)?;
    let destination = crate::sources::scanner::unique_scan_path(scan_dir);
    let partial = destination.with_extension("rekesz-part");
    std::fs::write(&partial, jpeg)?;
    std::fs::rename(&partial, &destination)?;
    Ok(destination)
}

/// Panel action byte: '1' JPG, '2' PDF; email ('3') and preview ('4') and
/// anything unknown are answered without a session.
fn action_byte(action: Option<char>) -> u8 {
    match action {
        Some('1') => 1,
        Some('2') => 2,
        Some('3') => 3,
        Some('4') => 4,
        _ => 0,
    }
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}
