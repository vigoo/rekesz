//! Announcement listener (UDP 2968, multicast group 239.255.255.253) and
//! keepalive answering: the printer shouts when the panel's Scan → Computer
//! menu is opened, and a computer becomes a panel destination by replying
//! with three unicast keepalives carrying its name, address, and event
//! port. Byte formats are ports of epson2paperless's `keepalive.ts`.

use std::net::{IpAddr, Ipv4Addr, SocketAddr, UdpSocket};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use super::engine::{self, AnnounceDedupe, EngineInput, KEEPALIVE_BURST, KEEPALIVE_INTERVAL};
use crate::trace;

/// A parsed `02 06` announcement from the printer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Announcement {
    /// The address the announcement arrived from.
    pub source: IpAddr,
    /// Sequence counter (byte 11); incremented per burst, echoed in the
    /// keepalive.
    pub seq: u8,
    /// The printer's product ID record ("1157" for the L3160), if present.
    pub pid: Option<String>,
    /// The keepalive tag list the printer asked for, in its order. Falls
    /// back to the NetScanMonitor v2 triple when the tag walk fails
    /// (framing inconsistencies abort the walk in the reference, and the
    /// PID-based default for our device class is v2).
    pub requested: Vec<String>,
}

/// Known keepalive tags; anything else in a requested record is skipped at
/// build time.
const KNOWN_TAGS: [&str; 5] = ["Ver", "ClientName", "IPAddress", "EventPort", "Group"];
/// The v2 keepalive tag set — what the L3160 asks for, and the fallback.
const V2_TAGS: [&str; 3] = ["ClientName", "IPAddress", "EventPort"];

pub fn parse_announcement(data: &[u8], source: IpAddr) -> Option<Announcement> {
    if data.len() < 12 || data[0..4] != [0x02, 0x06, 0x00, 0x00] {
        return None;
    }
    let seq = data[11];
    // The tag walk aborts whole on any framing inconsistency (the reference
    // returns null and falls back to the PID-based choice); the PID is
    // scanned from the whole packet regardless.
    let requested = walk_records(data)
        .and_then(|records| {
            records
                .iter()
                .find(|record| names_known_tags(record))
                .map(|record| {
                    record
                        .split(',')
                        .map(str::to_string)
                        .collect::<Vec<String>>()
                })
        })
        .unwrap_or_else(|| V2_TAGS.iter().map(|tag| tag.to_string()).collect());
    Some(Announcement {
        source,
        seq,
        pid: extract_pid(data),
        requested,
    })
}

/// PID record scan over the whole packet (the reference regexes the whole
/// buffer, tolerating walk aborts): `PID 1157` → "1157", hex-validated.
fn extract_pid(data: &[u8]) -> Option<String> {
    let text = String::from_utf8_lossy(data);
    let mut search = 0;
    while let Some(found) = text[search..].find("PID ") {
        let start = search + found + "PID ".len();
        if start + 4 <= text.len() {
            let id = &text[start..start + 4];
            if id.chars().all(|c| c.is_ascii_hexdigit()) {
                return Some(id.to_string());
            }
        }
        search = start.max(search + 1);
        if search >= text.len() {
            break;
        }
    }
    None
}

/// The announcement's record walk: skip the service string (length field at
/// bytes 16–19, string at 20+), then length-prefixed NUL-terminated records.
/// Any framing inconsistency aborts the whole walk — `None`, and the caller
/// falls back to the v2 tag choice (the reference's null-return semantics).
fn walk_records(data: &[u8]) -> Option<Vec<String>> {
    if data.len() < 20 {
        return None;
    }
    let service_len = u32::from_be_bytes([data[16], data[17], data[18], data[19]]) as usize;
    let mut offset = 20 + service_len;
    // The service string is NUL-terminated; the records follow it.
    offset = match data.get(offset) {
        Some(0) => offset + 1,
        _ => return None,
    };
    let mut records = Vec::new();
    while let Some(&length) = data.get(offset) {
        if length == 0 {
            break;
        }
        let start = offset + 1;
        let end = start + length as usize;
        // The record must carry its NUL terminator inside the packet.
        if data.get(end) != Some(&0) {
            return None;
        }
        let text: String = data[start..end].iter().map(|&b| b as char).collect();
        records.push(text);
        offset = end + 1;
    }
    Some(records)
}

fn names_known_tags(record: &str) -> bool {
    let shaped = record.split(',').all(|tag| {
        let mut chars = tag.chars();
        matches!(chars.next(), Some(first) if first.is_ascii_alphabetic())
            && chars.all(|rest| rest.is_ascii_alphanumeric())
            && !tag.is_empty()
    });
    shaped
        && KNOWN_TAGS
            .iter()
            .any(|known| record.split(',').any(|tag| tag == *known))
}

/// Values the keepalive carries per requested tag.
pub struct KeepaliveValues<'a> {
    pub client_name: &'a str,
    pub ip: &'a str,
    pub event_port: u16,
}

/// Build the `02 07` keepalive answering an announcement. The payload
/// mirrors the printer's requested tag list in its order (for the L3160
/// that is exactly the v2 triple, but the code must not hardcode it);
/// unknown requested tags are skipped. Wire layout: bytes 0–1 `02 07`,
/// byte 4 total length, byte 11 seq echo, byte 13 destination ID, bytes
/// 14–15 language, bytes 16–19 BE u32 string length (excluding the NUL),
/// bytes 20+ `(Tag=value),(Tag=value)` + NUL.
pub fn build_keepalive(
    seq: u8,
    requested: &[String],
    values: &KeepaliveValues<'_>,
) -> Option<Vec<u8>> {
    let payload: Vec<String> = requested
        .iter()
        .filter_map(|tag| {
            let value = match tag.as_str() {
                "ClientName" => values.client_name.to_string(),
                "IPAddress" => values.ip.to_string(),
                "EventPort" => values.event_port.to_string(),
                "Ver" => "3.0".to_string(),
                "Group" => "0".to_string(),
                _ => return None,
            };
            Some(format!("({tag}={value})"))
        })
        .collect();
    if payload.is_empty() {
        return None;
    }
    let mut string_bytes: Vec<u8> = payload.join(",").into_bytes();
    string_bytes.push(0);
    let mut packet = vec![0_u8; 20];
    packet[0] = 0x02;
    packet[1] = 0x07;
    packet[4] = (packet.len() + string_bytes.len()) as u8;
    packet[11] = seq;
    packet[13] = engine::DESTINATION_ID;
    packet[14] = b'e';
    packet[15] = b'n';
    let string_len = (string_bytes.len() - 1) as u32;
    packet[16..20].copy_from_slice(&string_len.to_be_bytes());
    packet.extend_from_slice(&string_bytes);
    Some(packet)
}

/// Does the announcement's tag list ask for NetScanMonitor v3? v3 keepalives
/// leave from an ephemeral socket; v2 leaves from the 2968-bound socket.
fn requests_v3(requested: &[String]) -> bool {
    requested.iter().any(|tag| tag == "Ver")
}

/// Bind the announcement socket: UDP 2968 with SO_REUSEADDR set **before**
/// bind (the option only influences the bind itself), then joined to the
/// announcement multicast group on the wildcard interface. std::net exposes
/// neither option, so the socket comes up through socket2 — one portable
/// implementation for all three platforms.
fn bind_announce_socket() -> Result<UdpSocket, String> {
    use socket2::{Domain, Protocol, Socket, Type};
    let socket = Socket::new(Domain::IPV4, Type::DGRAM, Some(Protocol::UDP))
        .map_err(|error| format!("socket(AF_INET, SOCK_DGRAM): {error}"))?;
    socket
        .set_reuse_address(true)
        .map_err(|error| format!("setsockopt SO_REUSEADDR: {error}"))?;
    socket
        .bind(&SocketAddr::from((Ipv4Addr::UNSPECIFIED, engine::ANNOUNCE_PORT)).into())
        .map_err(|error| format!("bind udp {}: {error}", engine::ANNOUNCE_PORT))?;
    socket
        .join_multicast_v4(&engine::ANNOUNCE_GROUP, &Ipv4Addr::UNSPECIFIED)
        .map_err(|error| format!("join multicast {}: {error}", engine::ANNOUNCE_GROUP))?;
    Ok(UdpSocket::from(socket))
}

/// The announce gate, factored out of the loop so tests can exercise it:
/// answer only while the input is freshly armed and the announcement comes
/// from the selected scanner.
pub fn should_answer(input: &EngineInput, source: IpAddr, now: Instant) -> bool {
    engine::armed_now(input, now) && input.scanner_ips.contains(&source)
}

/// The announcement listener: bind UDP 2968 (shared), join the multicast
/// group, parse and dedupe announcements, and answer keepalives while the
/// UI has armed a scanner this engine's announcement source belongs to.
pub fn run_announce_loop(input: Arc<Mutex<EngineInput>>) {
    let socket = match bind_announce_socket() {
        Ok(socket) => socket,
        Err(error) => {
            trace::slow_event(&format!(
                "push: announcement listener failed to start: {error}"
            ));
            return;
        }
    };
    trace::event("push: announcement listener up (udp 2968, group 239.255.255.253)");
    let mut dedupe = AnnounceDedupe::new();
    let mut buffer = [0_u8; 4096];
    loop {
        // A transient receive error is logged and the loop continues — a
        // dead listener needs an app restart to recover, so it is not
        // entered lightly.
        let (size, origin) = match socket.recv_from(&mut buffer) {
            Ok(received) => received,
            Err(error) => {
                trace::slow_event(&format!("push: announcement recv failed: {error}"));
                continue;
            }
        };
        let source = origin.ip();
        let Some(announcement) = parse_announcement(&buffer[..size], source) else {
            trace::event(&format!(
                "push: unparseable datagram from {source} ({size}B)"
            ));
            continue;
        };
        let now = Instant::now();
        if !dedupe.first_of_burst(source, announcement.seq, now) {
            trace::event(&format!(
                "push: announcement repeat from {source} seq {} (deduped)",
                announcement.seq
            ));
            continue;
        }
        let snapshot = input.lock().map(|input| input.clone());
        let Ok(input) = snapshot else {
            return;
        };
        if !should_answer(&input, source, now) {
            trace::event(&format!(
                "push: announcement from {source} seq {} pid {:?} (silent: disarmed or unknown device)",
                announcement.seq, announcement.pid
            ));
            continue;
        }
        if !answer_announcement(&socket, &announcement, &input) {
            // The burst could not even start (no route-facing address):
            // forget the burst so the next announcement retries.
            dedupe.forget(source, announcement.seq);
        }
    }
}

/// Send the keepalive burst for one accepted announcement: three unicast
/// `02 07` packets 500 ms apart (v2 leaves from the bound socket, v3 from
/// an ephemeral one — the L3160 is v2). Returns false when the burst was
/// skipped (no local address facing the printer).
fn answer_announcement(
    socket: &UdpSocket,
    announcement: &Announcement,
    input: &EngineInput,
) -> bool {
    let source = announcement.source;
    let Some(facing) = engine::local_ip_facing(source) else {
        trace::slow_event(&format!(
            "push: no local address facing {source} — keepalive skipped"
        ));
        return false;
    };
    let values = KeepaliveValues {
        client_name: &input.destination_name,
        ip: &facing.to_string(),
        event_port: engine::TRIGGER_PORT,
    };
    let Some(packet) = build_keepalive(announcement.seq, &announcement.requested, &values) else {
        return true;
    };
    let v3 = requests_v3(&announcement.requested);
    let target = SocketAddr::new(source, engine::ANNOUNCE_PORT);
    trace::event(&format!(
        "push: answering announcement from {source} seq {} pid {:?} ({})",
        announcement.seq,
        announcement.pid,
        if v3 { "v3" } else { "v2" }
    ));
    trace::event(&format!(
        "push: keepalive {} B → {target} name={:?} ip={facing} port={} destId={:02x} hex={}",
        packet.len(),
        input.destination_name,
        engine::TRIGGER_PORT,
        packet[13],
        packet
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    ));
    // v2 leaves from the 2968-bound socket (the port Event Manager's peers
    // expect); v3 leaves from an ephemeral one.
    let ephemeral = if v3 {
        UdpSocket::bind(SocketAddr::from(([0, 0, 0, 0], 0))).ok()
    } else {
        None
    };
    let sender: &UdpSocket = ephemeral.as_ref().unwrap_or(socket);
    for index in 0..KEEPALIVE_BURST {
        if index > 0 {
            std::thread::sleep(KEEPALIVE_INTERVAL);
        }
        if let Err(error) = sender.send_to(&packet, target) {
            trace::slow_event(&format!("push: keepalive send to {target} failed: {error}"));
        }
    }
    true
}
