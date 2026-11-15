//! Push-scan tests: wire codec goldens against the reference protocol
//! facts, announcement parsing against the live-captured packet, keepalive
//! and trigger-response goldens, admission gating, and a scripted
//! trigger-connection exchange. No test touches the network or a real
//! device.

use std::net::{IpAddr, Ipv4Addr, SocketAddr, TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use super::announce::{self, KeepaliveValues};
use super::dialect::{self, ParaColorMode};
use super::engine::{
    self, ANNOUNCE_DEDUPE_WINDOW, Admission, EngineInput, EngineOutput, PushScanSettings,
    TICK_STALENESS,
};
use super::session::{self, Link as SessionLink};
use super::trigger::{self, TriggerKind};
use super::wire::{self, ParsedPacket, parse_is_packet};

// The binary's test harness is test-r; plain `#[test]` would never register.
use test_r::test;

// CI runners are slower and more contended than dev machines; REKESZ_TEST_TIMING_SCALE
// stretches the timing guards without weakening them locally.
fn scaled_budget(budget: Duration) -> Duration {
    let scale = std::env::var("REKESZ_TEST_TIMING_SCALE")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(1.0)
        .max(1.0);
    budget.mul_f64(scale)
}

#[test]
fn lock_packet_matches_reference_bytes() {
    // "IS" + 0x2100 + offset 0x000c + payload 7 + padding + 01 a0 04 + u32BE(300).
    let expected: Vec<u8> = vec![
        b'I', b'S', 0x21, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x07, 0x00, 0x00, 0x01, 0xa0, 0x04,
        0x00, 0x00, 0x01, 0x2c,
    ];
    assert_eq!(wire::build_lock_packet(), expected);
}

#[test]
fn unlock_packet_is_header_only() {
    let expected: Vec<u8> = vec![
        b'I', b'S', 0x21, 0x01, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    ];
    assert_eq!(wire::build_unlock_packet(), expected);
}

#[test]
fn passthru_packet_carries_size_preamble_and_command() {
    // FS Y with a 1-byte expected reply: preamble is cmd size (2) and reply size (1).
    let expected: Vec<u8> = vec![
        b'I', b'S', 0x20, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x0a, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x02, 0x00, 0x00, 0x00, 0x01, 0x1c, 0x59,
    ];
    assert_eq!(
        wire::build_passthru_packet(&wire::FS_Y, wire::LEGACY_REPLY_SIZE),
        expected
    );
}

#[test]
fn pureread_packet_has_zero_command_size() {
    let packet = wire::build_pureread_packet(wire::ESCI2_REPLY_SIZE);
    let payload = &packet[wire::IS_HEADER_SIZE..];
    assert_eq!(payload.len(), 8);
    assert_eq!(&payload[0..4], &[0, 0, 0, 0], "cmd size must be 0");
    assert_eq!(&payload[4..8], &wire::ESCI2_REPLY_SIZE.to_be_bytes());
}

#[test]
fn esci2_command_pads_name_and_zeroes_length() {
    assert_eq!(wire::build_esci2_command("INFO"), b"INFOx0000000".to_vec());
    // Three-char names pad to four before the literal 'x'.
    assert_eq!(wire::build_esci2_command("FIN"), b"FIN x0000000".to_vec());
}

#[test]
fn para_header_uses_seven_uppercase_hex_digits() {
    assert_eq!(wire::build_para_header(0x3a8), b"PARAx00003A8".to_vec());
    assert_eq!(wire::build_para_header(0), b"PARAx0000000".to_vec());
}

#[test]
fn parse_is_packet_streams_fragments_then_completes() {
    let packet = wire::build_passthru_packet(&wire::FS_Y, wire::LEGACY_REPLY_SIZE);
    // This protocol family fragments IS frames; a 1-byte head of a genuine
    // frame is NeedMore, not Malformed.
    assert!(matches!(
        parse_is_packet(&packet[..1]),
        ParsedPacket::NeedMore
    ));
    let (head, _rest) = packet.split_at(5);
    assert!(matches!(parse_is_packet(head), ParsedPacket::NeedMore));
    assert!(matches!(
        parse_is_packet(&packet[..packet.len() - 1]),
        ParsedPacket::NeedMore
    ));
    match parse_is_packet(&packet) {
        ParsedPacket::Packet {
            consumed,
            kind,
            payload,
        } => {
            assert_eq!(consumed, packet.len());
            assert_eq!(kind, wire::TYPE_PASSTHRU);
            assert_eq!(payload, vec![0, 0, 0, 2, 0, 0, 0, 1, 0x1c, 0x59]);
        }
        other => panic!("expected a packet, got {other:?}"),
    }
    // Two packets in one buffer: the parser reads exactly the first.
    let mut doubled = packet.clone();
    doubled.extend_from_slice(&packet);
    match parse_is_packet(&doubled) {
        ParsedPacket::Packet { consumed, .. } => assert_eq!(consumed, packet.len()),
        other => panic!("expected a packet, got {other:?}"),
    }
}

#[test]
fn parse_is_packet_rejects_non_is_data() {
    assert!(matches!(
        parse_is_packet(b"HTTP/1.0 200 OK"),
        ParsedPacket::Malformed
    ));
}

#[test]
fn parse_is_packet_ignores_printer_data_offset_bytes() {
    // Printers send 0x300C (some 0x380C) in the data-offset bytes; parsing
    // reads only the type and the payload size.
    let mut printer_sized = wire::build_lock_packet();
    printer_sized[4] = 0x30;
    printer_sized[5] = 0x0c;
    match parse_is_packet(&printer_sized) {
        ParsedPacket::Packet { consumed, kind, .. } => {
            assert_eq!(consumed, printer_sized.len());
            assert_eq!(kind, wire::TYPE_LOCK);
        }
        other => panic!("expected a packet, got {other:?}"),
    }
}

#[test]
fn parse_is_packet_rejects_absurd_declared_payload() {
    let mut hostile = wire::build_lock_packet();
    // 4 GiB declared payload: a garbled stream must not buffer toward it.
    hostile[6..10].copy_from_slice(&0xffff_ffff_u32.to_be_bytes());
    assert!(matches!(parse_is_packet(&hostile), ParsedPacket::Malformed));
}

#[test]
fn esci2_reply_parses_cmd_length_and_tokens() {
    // IMG metadata frame: declared chunk 0x1234, tokens then #--- filler
    // padding out the 52-byte token block.
    let mut frame = b"IMG x0001234#typIMGR#pen#---".to_vec();
    frame.resize(64, b'-');
    let reply = wire::parse_esci2_reply(&frame).expect("parses");
    assert_eq!(reply.cmd, "IMG");
    assert_eq!(reply.length, 0x1234);
    assert_eq!(
        reply.tokens,
        vec![
            ("typ".to_string(), "IMGR".to_string()),
            ("pen".to_string(), String::new())
        ]
    );
}

#[test]
fn esci2_reply_reads_adf_last_page_marker() {
    // ADF page-end marker: #lftd000 means last page; d001 means more pages.
    let mut frame = b"IMG x0000001#lftd000#---".to_vec();
    frame.resize(64, b'-');
    let reply = wire::parse_esci2_reply(&frame).expect("parses");
    assert_eq!(
        reply
            .tokens
            .first()
            .map(|(key, value)| (key.as_str(), value.as_str())),
        Some(("lft", "d000"))
    );
}

#[test]
fn esci2_reply_rejects_frame_without_x_separator() {
    let mut frame = b"IMG?0001234".to_vec();
    frame.resize(64, b'-');
    assert!(wire::parse_esci2_reply(&frame).is_none());
    assert!(wire::parse_esci2_reply(&frame[..8]).is_none());
}

// The announcement captured live from the L3160 (192.168.50.67) on
// 2026-11-14, when the panel's Scan → Computer menu was opened — see
// testdata/announcement-capture-2026-11-14.log. Seq byte 11 = 0x01.
const CAPTURED_ANNOUNCEMENT: [u8; 92] = [
    0x02, 0x06, 0x00, 0x00, 0x5c, 0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x02, 0x65, 0x6e,
    0x00, 0x00, 0x00, 0x1c, 0x73, 0x65, 0x72, 0x76, 0x69, 0x63, 0x65, 0x3a, 0x4e, 0x65, 0x74, 0x53,
    0x63, 0x61, 0x6e, 0x4d, 0x6f, 0x6e, 0x69, 0x74, 0x6f, 0x72, 0x2d, 0x61, 0x67, 0x65, 0x6e, 0x74,
    0x00, 0x08, 0x50, 0x49, 0x44, 0x20, 0x31, 0x31, 0x35, 0x37, 0x00, 0x1e, 0x43, 0x6c, 0x69, 0x65,
    0x6e, 0x74, 0x4e, 0x61, 0x6d, 0x65, 0x2c, 0x49, 0x50, 0x41, 0x64, 0x64, 0x72, 0x65, 0x73, 0x73,
    0x2c, 0x45, 0x76, 0x65, 0x6e, 0x74, 0x50, 0x6f, 0x72, 0x74, 0x00, 0x00,
];

#[test]
fn announcement_parses_the_captured_l3160_packet() {
    let source = IpAddr::V4(Ipv4Addr::new(192, 168, 50, 67));
    let announcement = announce::parse_announcement(&CAPTURED_ANNOUNCEMENT, source)
        .expect("the captured packet parses");
    assert_eq!(announcement.seq, 1);
    assert_eq!(announcement.pid.as_deref(), Some("1157"));
    assert_eq!(
        announcement.requested,
        vec![
            "ClientName".to_string(),
            "IPAddress".to_string(),
            "EventPort".to_string()
        ]
    );
}

#[test]
fn announcement_rejects_garbage() {
    let source = IpAddr::V4(Ipv4Addr::LOCALHOST);
    assert!(announce::parse_announcement(&[], source).is_none());
    assert!(announce::parse_announcement(b"HTTP/1.0 200 OK\r\n\r\n", source).is_none());
    let mut wrong_magic = CAPTURED_ANNOUNCEMENT;
    wrong_magic[1] = 0x07;
    assert!(announce::parse_announcement(&wrong_magic, source).is_none());
}

#[test]
fn announcement_falls_back_to_v2_tags_when_records_break() {
    // A truncated record walk (service length points past the packet) must
    // not break the parse: the v2 tag triple is the reference's fallback.
    let mut truncated = Vec::from(&CAPTURED_ANNOUNCEMENT[..60]);
    truncated[11] = 9;
    let source = IpAddr::V4(Ipv4Addr::LOCALHOST);
    let announcement = announce::parse_announcement(&truncated, source).expect("parses");
    assert_eq!(announcement.seq, 9);
    assert_eq!(
        announcement.requested,
        vec![
            "ClientName".to_string(),
            "IPAddress".to_string(),
            "EventPort".to_string()
        ]
    );
}

#[test]
fn keepalive_matches_the_reference_layout() {
    let values = KeepaliveValues {
        client_name: "REKESZ",
        ip: "192.168.50.159",
        event_port: 2968,
    };
    let packet = announce::build_keepalive(
        1,
        &[
            "ClientName".to_string(),
            "IPAddress".to_string(),
            "EventPort".to_string(),
        ],
        &values,
    )
    .expect("v2 keepalive builds");
    // Payload: "(ClientName=REKESZ),(IPAddress=192.168.50.159),(EventPort=2968)"
    // = 63 bytes + NUL; total = 20 + 64 = 84 (byte 4). Byte 13 = destination
    // ID 0x02, as in the real Event Manager capture the reference pins.
    let mut expected: Vec<u8> = vec![
        0x02, 0x07, 0x00, 0x00, 0x54, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x02, b'e',
        b'n', 0x00, 0x00, 0x00, 0x3f,
    ];
    expected
        .extend_from_slice(b"(ClientName=REKESZ),(IPAddress=192.168.50.159),(EventPort=2968)\0");
    assert_eq!(packet, expected);
    assert_eq!(packet[4] as usize, packet.len());
}

#[test]
fn keepalive_mirrors_the_requested_tag_order() {
    let values = KeepaliveValues {
        client_name: "N",
        ip: "10.0.0.1",
        event_port: 2968,
    };
    let packet = announce::build_keepalive(
        7,
        &[
            "IPAddress".to_string(),
            "ClientName".to_string(),
            "EventPort".to_string(),
        ],
        &values,
    )
    .expect("builds");
    let payload = &packet[20..];
    let text = String::from_utf8(payload[..payload.len() - 1].to_vec()).expect("ascii");
    assert_eq!(text, "(IPAddress=10.0.0.1),(ClientName=N),(EventPort=2968)");
}

#[test]
fn keepalive_skips_unknown_requested_tags() {
    let values = KeepaliveValues {
        client_name: "N",
        ip: "10.0.0.1",
        event_port: 2968,
    };
    let packet =
        announce::build_keepalive(0, &["ClientName".to_string(), "Bogus".to_string()], &values)
            .expect("builds");
    let payload = &packet[20..];
    let text = String::from_utf8(payload[..payload.len() - 1].to_vec()).expect("ascii");
    assert_eq!(text, "(ClientName=N)");
}

fn push_settings() -> PushScanSettings {
    PushScanSettings {
        dpi: 300,
        region: (0.0, 0.0, 1.0, 1.0),
        color: ParaColorMode::Color,
    }
}

fn armed_input(now: Instant) -> EngineInput {
    EngineInput {
        armed: true,
        busy: false,
        scanner_ips: vec![IpAddr::V4(Ipv4Addr::new(192, 168, 50, 67))],
        destination_name: "Rekesz".to_string(),
        ticked_at: Some(now),
        scan: Some(push_settings()),
        scan_dir: Some(std::path::PathBuf::from("/tmp/rekesz-test-scans")),
    }
}

#[test]
fn admission_accepts_armed_idle_push() {
    let now = Instant::now();
    assert_eq!(
        engine::admit_push_scan(&armed_input(now), 1, false, now),
        Admission::Accept
    );
    assert_eq!(
        engine::admit_push_scan(&armed_input(now), 2, false, now),
        Admission::Accept
    );
}

#[test]
fn admission_refuses_unarmed_stale_busy_or_pending() {
    let now = Instant::now();
    let mut unarmed = armed_input(now);
    unarmed.armed = false;
    assert_eq!(
        engine::admit_push_scan(&unarmed, 1, false, now),
        Admission::Refuse
    );
    // The poll chain stopped: the armed flag is stale beyond TICK_STALENESS.
    let stale = armed_input(now - TICK_STALENESS - Duration::from_millis(1));
    assert_eq!(
        engine::admit_push_scan(&stale, 1, false, now),
        Admission::Refuse
    );
    let mut busy = armed_input(now);
    busy.busy = true;
    assert_eq!(
        engine::admit_push_scan(&busy, 1, false, now),
        Admission::Refuse
    );
    assert_eq!(
        engine::admit_push_scan(&armed_input(now), 1, true, now),
        Admission::Refuse
    );
}

#[test]
fn admission_answers_unsupported_actions_without_a_session() {
    let now = Instant::now();
    assert_eq!(
        engine::admit_push_scan(&armed_input(now), 4, false, now),
        Admission::OkNoSession
    );
    assert_eq!(
        engine::admit_push_scan(&armed_input(now), 0, false, now),
        Admission::OkNoSession
    );
}

#[test]
fn announce_dedupe_passes_one_packet_per_burst() {
    let mut dedupe = engine::AnnounceDedupe::new();
    let now = Instant::now();
    let source = IpAddr::V4(Ipv4Addr::LOCALHOST);
    assert!(dedupe.first_of_burst(source, 5, now));
    assert!(!dedupe.first_of_burst(source, 5, now + Duration::from_millis(50)));
    // A new seq in the same window is a new burst.
    assert!(dedupe.first_of_burst(source, 6, now + Duration::from_millis(60)));
    // The same seq after the window is a new burst again (seq wrap safety).
    assert!(dedupe.first_of_burst(
        source,
        5,
        now + ANNOUNCE_DEDUPE_WINDOW + Duration::from_secs(1)
    ));
}

/// A reference-shaped PushScan POST with the Epson `" : "` header spacing.
fn push_scan_request(xuid: &str, id_in: &str) -> String {
    let body = format!(
        "<?xml version=\"1.0\" ?>\r\n\
         <s:Envelope xmlns:s=\"http://www.w3.org/2003/05/soap-envelope\">\r\n\
         \x20 <s:Body>\r\n\
         \x20   <p:PushScan xmlns:p=\"http://schema.epson.net/EpsonNet/Scan/2004/pushscan\">\r\n\
         \x20     <PushScanIDIn>{id_in}</PushScanIDIn>\r\n\
         \x20   </p:PushScan>\r\n\
         \x20 </s:Body>\r\n\
         </s:Envelope>\r\n"
    );
    format!(
        "POST /HTTP/1.0\r\n\
         Content-Type : text/xml\r\n\
         Content-Length : {}\r\n\
         x-uid : {}\r\n\
         x-protocol-version : 2.00\r\n\
         \r\n\
         {body}",
        body.len(),
        xuid
    )
}

#[test]
fn trigger_kind_classification_prefers_joblist() {
    let mut request =
        trigger_request_of("<JobList></JobList><p:PushScan xmlns:p=\"x\"></p:PushScan>");
    assert_eq!(request.kind, TriggerKind::JobList);
    assert!(request.capabilities.is_empty());
    request = trigger_request_of(
        "<p:PushScan xmlns:p=\"x\"><PushScanIDIn>01</PushScanIDIn></p:PushScan>",
    );
    assert_eq!(request.kind, TriggerKind::PushScan);
    assert_eq!(request.sides, Some('0'));
    assert_eq!(request.action, Some('1'));
    request = trigger_request_of("<p:ScanEnd xmlns:p=\"x\"></p:ScanEnd>");
    assert_eq!(request.kind, TriggerKind::ScanEnd);
    request = trigger_request_of("<Nothing/>");
    assert_eq!(request.kind, TriggerKind::Unknown);
}

fn trigger_request_of(body: &str) -> trigger::TriggerRequest {
    // Build the raw request text, then run it through the parser via a
    // loopback socket pair.
    let listener = TcpListener::bind("127.0.0.1:0").expect("listener");
    let address = listener.local_addr().expect("address");
    let client = TcpStream::connect(address).expect("client");
    let (server, _) = listener.accept().expect("accept");
    let raw = format!(
        "POST / HTTP/1.0\r\nContent-Type : text/xml\r\nContent-Length : {}\r\nx-uid : 777\r\nx-protocol-version : 3.01\r\n\r\n{body}",
        body.len()
    );
    std::thread::spawn(move || {
        let mut client = client;
        std::io::Write::write_all(&mut client, raw.as_bytes()).ok();
        std::io::Write::flush(&mut client).ok();
        let _ = client.shutdown(std::net::Shutdown::Write);
    });
    let mut server = server;
    trigger::read_request(&mut server).expect("parses")
}

#[test]
fn response_golden_matches_the_reference_bytes() {
    // Hand-written from the reference's buildResponseBody/buildPushScanResponse:
    // every line and its indentation is pinned, and the Content-Length must
    // equal this literal's byte length.
    let expected_body = "<?xml version=\"1.0\" ?>\r\n\
         <s:Envelope xmlns:s=\"http://www.w3.org/2003/05/soap-envelope\">\r\n\
         \x20 <s:Body>\r\n\
         \x20   <p:PushScanResponse xmlns:p=\"http://schema.epson.net/EpsonNet/Scan/2004/pushscan\">\r\n\
         \x20     <StatusOut>OK</StatusOut>\r\n\
         \x20   </p:PushScanResponse>\r\n\
         \x20 </s:Body>\r\n\
         </s:Envelope>\r\n";
    let response = trigger::build_response("OK", "2.00", "abc", &[]);
    let (headers, body) = response
        .split_once("\r\n\r\n")
        .expect("headers and body separated by a blank line");
    assert!(headers.starts_with("HTTP/1.0 200 OK\r\n"));
    assert!(headers.contains("Server : Epson Net Scan Monitor/2.0\r\n"));
    assert!(headers.contains("Content-Type : application/octet-stream\r\n"));
    assert!(headers.contains(&format!("Content-Length : {}\r\n", expected_body.len())));
    assert!(headers.contains("x-protocol-name : Epson Network Service Protocol\r\n"));
    assert!(headers.contains("x-protocol-version : 2.00\r\n"));
    assert!(headers.contains("x-uid : abc\r\n"));
    // The last header line: the split consumed its trailing CRLF.
    assert!(headers.ends_with("x-status : 0001"));
    assert_eq!(body, expected_body);
}

#[test]
fn response_error_carries_error_status_without_capabilities() {
    let response =
        trigger::build_response("ERROR", "2.00", "9", &["WouldHaveBeenEchoed".to_string()]);
    assert!(response.contains("<StatusOut>ERROR</StatusOut>"));
    assert!(!response.contains("CapabilityOut"));
    assert!(response.contains("x-uid : 9\r\n"));
}

#[test]
fn response_joblist_echoes_requested_capabilities() {
    let response = trigger::build_response("OK", "2.00", "1", &["ScanJpg".to_string()]);
    assert!(response.contains("<CapabilityOut>ScanJpg</CapabilityOut>"));
}

/// Armed engine input whose scanner address is the loopback — the scripted
/// trigger tests connect from 127.0.0.1, so the peer gate must accept it.
fn loopback_input(now: Instant) -> EngineInput {
    let mut input = armed_input(now);
    input.scanner_ips = vec![IpAddr::V4(Ipv4Addr::LOCALHOST)];
    input
}

#[test]
fn scripted_connection_answers_ok_and_attempts_the_session() {
    // Premise check: the test needs nothing to listen on the session port,
    // or the session's connect would succeed and stall the state machine
    // (60 s read budget per state) instead of refusing fast. A runner
    // service squatting on 1865 fails here with the evidence instead of an
    // opaque "None" after the wait. A probe refused for other reasons (the
    // transient CI bind flake) says nothing about the port being occupied,
    // so it only notes and continues.
    match TcpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 1865))) {
        Ok(listener) => drop(listener),
        Err(error) if error.kind() == std::io::ErrorKind::AddrInUse => {
            panic!("the ESC/I-2 session port 1865 is already in use on this machine: {error}");
        }
        Err(_) => {}
    }
    let listener = TcpListener::bind("127.0.0.1:0").expect("listener");
    let address = listener.local_addr().expect("address");
    let input = Arc::new(Mutex::new(loopback_input(Instant::now())));
    let output = Arc::new(Mutex::new(EngineOutput::default()));
    let server_input = Arc::clone(&input);
    let server_output = Arc::clone(&output);
    std::thread::spawn(move || {
        let (server, _) = listener.accept().expect("accept");
        trigger::handle_connection(server, &server_input, &server_output);
    });
    let mut client = TcpStream::connect(address).expect("client");
    std::io::Write::write_all(&mut client, push_scan_request("42", "01").as_bytes())
        .expect("request written");
    let _ = client.shutdown(std::net::Shutdown::Write);
    let mut response = Vec::new();
    // Reads to EOF: the handler half-closes after the response (FIN).
    std::io::Read::read_to_end(&mut client, &mut response).expect("response read");
    let response = String::from_utf8_lossy(&response).into_owned();
    assert!(response.starts_with("HTTP/1.0 200 OK\r\n"), "{response}");
    assert!(response.contains("x-uid : 42\r\n"));
    assert!(response.contains("<StatusOut>OK</StatusOut>"));
    // The accepted push reserves the session slot and then attempts the
    // session; with no printer listening on the loopback ESC/I-2 port the
    // connect fails and reports through the result slot. The session
    // attempt happens after the response is flushed; wait for it past the
    // session's own 10 s connect budget — on a contended runner even the
    // loopback refusal can take that long, so the wait scales like the
    // other timing budgets.
    let mut done = None;
    let deadline = Instant::now() + scaled_budget(Duration::from_secs(15));
    while Instant::now() < deadline {
        done = output.lock().expect("output").scan_done.take();
        if done.is_some() {
            break;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    assert!(
        matches!(&done, Some(Err(message)) if message.contains("connect")),
        "the accept path must attempt the session and report why it stopped: {done:?}"
    );
}

#[test]
fn scripted_connection_refuses_busy_with_error_status() {
    let listener = TcpListener::bind("127.0.0.1:0").expect("listener");
    let address = listener.local_addr().expect("address");
    let mut state = loopback_input(Instant::now());
    state.busy = true;
    let input = Arc::new(Mutex::new(state));
    let output = Arc::new(Mutex::new(EngineOutput::default()));
    let server_input = Arc::clone(&input);
    let server_output = Arc::clone(&output);
    std::thread::spawn(move || {
        let (server, _) = listener.accept().expect("accept");
        trigger::handle_connection(server, &server_input, &server_output);
    });
    let mut client = TcpStream::connect(address).expect("client");
    std::io::Write::write_all(&mut client, push_scan_request("7", "01").as_bytes())
        .expect("request written");
    let _ = client.shutdown(std::net::Shutdown::Write);
    let mut response = Vec::new();
    std::io::Read::read_to_end(&mut client, &mut response).expect("response read");
    assert!(String::from_utf8_lossy(&response).contains("<StatusOut>ERROR</StatusOut>"));
    assert!(!output.lock().expect("output").scan_active);
}

#[test]
fn scripted_connection_answers_scan_end_without_a_trigger() {
    let listener = TcpListener::bind("127.0.0.1:0").expect("listener");
    let address = listener.local_addr().expect("address");
    let input = Arc::new(Mutex::new(loopback_input(Instant::now())));
    let output = Arc::new(Mutex::new(EngineOutput::default()));
    let server_input = Arc::clone(&input);
    let server_output = Arc::clone(&output);
    std::thread::spawn(move || {
        let (server, _) = listener.accept().expect("accept");
        trigger::handle_connection(server, &server_input, &server_output);
    });
    let mut client = TcpStream::connect(address).expect("client");
    let body =
        "<p:ScanEnd xmlns:p=\"http://schema.epson.net/EpsonNet/Scan/2004/pushscan\"></p:ScanEnd>";
    let request = format!(
        "POST / HTTP/1.0\r\nContent-Type : text/xml\r\nContent-Length : {}\r\nx-uid : 5\r\nx-protocol-version : 2.00\r\n\r\n{body}",
        body.len()
    );
    std::io::Write::write_all(&mut client, request.as_bytes()).expect("request written");
    let _ = client.shutdown(std::net::Shutdown::Write);
    let mut response = Vec::new();
    std::io::Read::read_to_end(&mut client, &mut response).expect("response read");
    let response = String::from_utf8_lossy(&response).into_owned();
    assert!(response.contains("<StatusOut>OK</StatusOut>"));
    assert!(response.contains("<p:PushScanResponse"));
    assert!(!output.lock().expect("output").scan_active);
}

#[test]
fn socket_options_constants_are_stable() {
    // The keepalive event port comes from the engine constants; pin them so
    // a refactor cannot silently move the wire contract.
    assert_eq!(engine::ANNOUNCE_PORT, 2968);
    assert_eq!(engine::TRIGGER_PORT, 2968);
    assert_eq!(SocketAddr::from(([192, 168, 50, 67], 2968)).port(), 2968);
}

#[test]
fn announce_gate_answers_only_armed_selected_device() {
    let now = Instant::now();
    let printer = IpAddr::V4(Ipv4Addr::new(192, 168, 50, 67));
    let other = IpAddr::V4(Ipv4Addr::new(192, 168, 50, 99));
    let input = armed_input(now);
    assert!(announce::should_answer(&input, printer, now));
    // Another Epson on the LAN must not trigger.
    assert!(!announce::should_answer(&input, other, now));
    // Disarmed: silence even for the selected device.
    let mut disarmed = input.clone();
    disarmed.armed = false;
    assert!(!announce::should_answer(&disarmed, printer, now));
}

#[test]
fn extract_element_ignores_closing_tags() {
    // Two capabilities: the closing tag of the first must not match as an
    // opening tag of the second (the reference regex cannot match closings).
    let body = "<p:CapabilityIn>A</p:CapabilityIn><p:CapabilityIn>B</p:CapabilityIn>";
    let listener = TcpListener::bind("127.0.0.1:0").expect("listener");
    let address = listener.local_addr().expect("address");
    let client = TcpStream::connect(address).expect("client");
    let (server, _) = listener.accept().expect("accept");
    let raw = format!(
        "POST / HTTP/1.0\r\nContent-Length : {}\r\n\r\n{body}",
        body.len()
    );
    std::thread::spawn(move || {
        let mut client = client;
        std::io::Write::write_all(&mut client, raw.as_bytes()).ok();
        std::io::Write::flush(&mut client).ok();
        let _ = client.shutdown(std::net::Shutdown::Write);
    });
    let mut server = server;
    let request = trigger::read_request(&mut server).expect("parses");
    assert_eq!(request.capabilities, vec!["A".to_string(), "B".to_string()]);
}

#[test]
fn read_request_survives_hostile_content_length() {
    // A garbage Content-Length must drop the connection, not panic.
    let listener = TcpListener::bind("127.0.0.1:0").expect("listener");
    let address = listener.local_addr().expect("address");
    let client = TcpStream::connect(address).expect("client");
    let (server, _) = listener.accept().expect("accept");
    let raw = "POST / HTTP/1.0\r\nContent-Length : 18446744073709551611\r\n\r\n";
    std::thread::spawn(move || {
        let mut client = client;
        std::io::Write::write_all(&mut client, raw.as_bytes()).ok();
        std::io::Write::flush(&mut client).ok();
        let _ = client.shutdown(std::net::Shutdown::Write);
    });
    let mut server = server;
    // The clamp makes the parser read to EOF (client already closed) → None.
    assert!(trigger::read_request(&mut server).is_none());
}

// ---- Stage 2: capability parsing, PARA composition, session replay ----

/// The L3160's CAPA reply from the stage-0 live probe.
const L3160_CAPA: &[u8] = include_bytes!("testdata/l3160-capa-init1.bin");
const L3160_INFO: &[u8] = include_bytes!("testdata/l3160-info-init1.bin");
const GOLDEN_PARA_300: &[u8] = include_bytes!("testdata/golden-para-flatbed-300dpi.bin");
const GOLDEN_PARA_600: &[u8] = include_bytes!("testdata/golden-para-flatbed-600dpi.bin");
/// A real ET-4800 flatbed JPEG session, both directions (the ET-4800's CAPA
/// shape matches the L3160's): printer→host and host→printer byte streams.
const ET4800_PRINTER_STREAM: &[u8] = include_bytes!("testdata/et4800-flatbed-jpg-printer.bin");
const ET4800_HOST_STREAM: &[u8] = include_bytes!("testdata/et4800-flatbed-jpg-host.bin");

#[test]
fn l3160_capa_tokens_parse_from_the_live_probe_fixture() {
    let capa = dialect::parse_capa_tokens(L3160_CAPA);
    assert_eq!(capa.gmm_list.as_deref(), Some("UG10UG18"));
    assert_eq!(capa.cmx_list.as_deref(), Some("UNITUM08"));
    assert_eq!(capa.fmt_list.as_deref(), Some("RAW JPG"));
    assert_eq!(
        capa.rsm_list.as_deref(),
        Some(&[75, 150, 300, 600, 1200][..])
    );
    assert_eq!(
        capa.rss_list.as_deref(),
        Some(&[75, 150, 300, 400, 600, 1200, 2400][..])
    );
    assert_eq!(
        capa.fb_rsms_list.as_deref(),
        Some(&[75, 150, 300, 600, 1200][..])
    );
    assert_eq!(capa.jpg_range, Some((1, 100)));
    // Resolution negotiation: RSMLIST ∩ RSSLIST = 75/150/300/600.
    // 1200 is in BOTH RSMLIST and RSSLIST, so it survives the intersection.
    assert_eq!(
        dialect::advertised_dpi_set(&capa),
        vec![75, 150, 300, 600, 1200]
    );
    assert_eq!(dialect::select_wire_dpi(300, &[75, 150, 300, 600]), 300);
    assert_eq!(dialect::select_wire_dpi(200, &[75, 150, 300, 600]), 300);
    assert_eq!(dialect::select_wire_dpi(1200, &[75, 150, 300, 600]), 600);
}

#[test]
fn l3160_info_tokens_parse_from_the_live_probe_fixture() {
    let info = dialect::parse_info_tokens(L3160_INFO);
    assert_eq!(info.pid.as_deref(), Some("1157"));
    assert_eq!(info.fb_area.as_deref(), Some("d850i0001170"));
    assert_eq!(info.fb_align.as_deref(), Some("LEFT"));
}

#[test]
fn region_window_maps_the_crop_over_the_info_area() {
    let info = dialect::parse_info_tokens(L3160_INFO);
    let area = dialect::parse_area(info.fb_area.as_deref().expect("area")).expect("two values");
    // Full platen: the whole advertised area at the wire DPI — the same
    // rectangle an app-initiated eSCL scan captures.
    let full = dialect::region_window(area, (0.0, 0.0, 1.0, 1.0), 300).expect("full");
    assert_eq!(
        full,
        dialect::Extents {
            x0: 0,
            y0: 0,
            w: 2550,
            h: 3510
        }
    );
    let full600 = dialect::region_window(area, (0.0, 0.0, 1.0, 1.0), 600).expect("full 600");
    assert_eq!(full600.w, 5100);
    assert_eq!(full600.h, 7020);
    // A half-size crop anchored in the top-left quadrant.
    let crop = dialect::region_window(area, (0.0, 0.0, 0.5, 0.5), 300).expect("crop");
    assert_eq!(
        crop,
        dialect::Extents {
            x0: 0,
            y0: 0,
            w: 1275,
            h: 1755
        }
    );
    // Degenerate: a zero-area crop is refused.
    assert!(dialect::region_window(area, (0.5, 0.5, 0.5, 0.5), 300).is_err());
}

#[test]
fn composed_para_matches_the_reference_golden() {
    // The goldens were produced by the reference's own composer for the
    // ET-4800 dialect inputs (flatbed, jpg, A4 window, quality 90) and are
    // byte-identical to what a real Windows driver sent the ET-4800 — the
    // same CAPA shape as the L3160.
    let dialect = dialect::FlatbedDialect::et4800();
    let window = dialect::Extents {
        x0: 0,
        y0: 0,
        w: 2481,
        h: 3506,
    };
    assert_eq!(
        dialect::compose_para(dialect, 300, window, ParaColorMode::Color).as_slice(),
        GOLDEN_PARA_300
    );
    assert_eq!(
        dialect::compose_para(
            dialect,
            600,
            dialect::Extents {
                x0: 0,
                y0: 0,
                w: 4962,
                h: 7012
            },
            ParaColorMode::Color
        )
        .as_slice(),
        GOLDEN_PARA_600
    );
}

/// A scripted printer: replays a recorded printer→host stream and captures
/// everything the session writes.
struct ReplayLink {
    incoming: &'static [u8],
    position: usize,
    sent: Vec<u8>,
}

impl SessionLink for ReplayLink {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let remaining = self.incoming.len() - self.position;
        if remaining == 0 {
            return Ok(0);
        }
        let take = remaining.min(buf.len());
        buf[..take].copy_from_slice(&self.incoming[self.position..self.position + take]);
        self.position += take;
        Ok(take)
    }

    fn write_all(&mut self, data: &[u8]) -> std::io::Result<()> {
        self.sent.extend_from_slice(data);
        Ok(())
    }
}

#[test]
fn session_replays_the_real_et4800_capture_byte_for_byte() {
    let mut link = ReplayLink {
        incoming: ET4800_PRINTER_STREAM,
        position: 0,
        sent: Vec::new(),
    };
    // The recording's driver sent the A4 window (2481×3506) over the
    // 2550×3510 area — express it as the crop fractions that reproduce it.
    let settings = PushScanSettings {
        dpi: 300,
        region: (0.0, 0.0, 2481.0 / 2550.0, 3506.0 / 3510.0),
        color: ParaColorMode::Color,
    };
    let image = session::drive_for_test(&mut link, dialect::FlatbedDialect::et4800(), settings)
        .expect("the session completes");
    // The recorded host traffic must be reproduced exactly.
    assert_eq!(
        link.sent.as_slice(),
        ET4800_HOST_STREAM,
        "host stream diverges from the real capture"
    );
    // The image the session assembled is the device-encoded JPEG.
    assert!(image.len() > 100_000, "{} bytes of scan", image.len());
    assert_eq!(&image[..2], &[0xff, 0xd8], "JPEG SOI");
}

// ---- Stage 2: scripted-session tests over synthetic printer streams ----

/// A 64-byte ESC/I-2 reply frame: echoed command, declared body length,
/// token block padded with `#---` filler.
fn meta(cmd: &str, body_len: u32, tokens: &str) -> Vec<u8> {
    let mut frame = vec![0_u8; 64];
    let padded = format!("{cmd:<4}");
    frame[0..4].copy_from_slice(padded.as_bytes());
    frame[4] = b'x';
    // The declared length is 7 HEX digits on the wire.
    frame[5..12].copy_from_slice(format!("{body_len:07X}").as_bytes());
    let mut block = format!("#{tokens}");
    while block.len() < 52 {
        block.push_str("#---");
    }
    frame[12..12 + block.len().min(52)].copy_from_slice(&block.as_bytes()[..52.min(block.len())]);
    frame
}

/// A synthetic printer stream for a complete minimal flatbed session: the
/// L3160 fixture's INFO/CAPA bodies, one 4-byte #pen image chunk, clean
/// teardown. `inject` runs mid-stream (async events, error frames…).
fn synthetic_session_stream(inject_after_lock: Option<Vec<u8>>, img_meta_tokens: &str) -> Vec<u8> {
    let mut stream = Vec::new();
    let push = |stream: &mut Vec<u8>, frame: Vec<u8>| stream.extend_from_slice(&frame);
    push(
        &mut stream,
        wire::build_is_packet(wire::TYPE_WELCOME, &[0x01, 0x04, 0x00, 0x00, 0x00]),
    );
    push(
        &mut stream,
        wire::build_is_packet(wire::TYPE_LOCK_ACK, &[wire::ACK_BYTE]),
    );
    if let Some(injected) = inject_after_lock {
        push(&mut stream, injected);
    }
    push(
        &mut stream,
        wire::build_is_packet(wire::TYPE_PASSTHRU_REPLY, &[wire::ACK_BYTE]),
    );
    // INIT1: INFO + CAPA from the live-probe fixtures.
    push(
        &mut stream,
        wire::build_is_packet(
            wire::TYPE_PASSTHRU_REPLY,
            &meta("INFO", L3160_INFO.len() as u32, ""),
        ),
    );
    push(
        &mut stream,
        wire::build_is_packet(wire::TYPE_PASSTHRU_REPLY, L3160_INFO),
    );
    push(
        &mut stream,
        wire::build_is_packet(
            wire::TYPE_PASSTHRU_REPLY,
            &meta("CAPA", L3160_CAPA.len() as u32, ""),
        ),
    );
    push(
        &mut stream,
        wire::build_is_packet(wire::TYPE_PASSTHRU_REPLY, L3160_CAPA),
    );
    push(
        &mut stream,
        wire::build_is_packet(wire::TYPE_PASSTHRU_REPLY, &meta("FIN", 0, "")),
    );
    // INIT2: small second-mode replies.
    push(
        &mut stream,
        wire::build_is_packet(wire::TYPE_PASSTHRU_REPLY, &[wire::ACK_BYTE]),
    );
    push(
        &mut stream,
        wire::build_is_packet(wire::TYPE_PASSTHRU_REPLY, &meta("INFO", 2, "")),
    );
    push(
        &mut stream,
        wire::build_is_packet(wire::TYPE_PASSTHRU_REPLY, b"OK"),
    );
    push(
        &mut stream,
        wire::build_is_packet(wire::TYPE_PASSTHRU_REPLY, &meta("CAPA", 2, "")),
    );
    push(
        &mut stream,
        wire::build_is_packet(wire::TYPE_PASSTHRU_REPLY, b"OK"),
    );
    push(
        &mut stream,
        wire::build_is_packet(wire::TYPE_PASSTHRU_REPLY, &meta("RESA", 2, "")),
    );
    push(
        &mut stream,
        wire::build_is_packet(wire::TYPE_PASSTHRU_REPLY, b"OK"),
    );
    push(
        &mut stream,
        wire::build_is_packet(wire::TYPE_PASSTHRU_REPLY, &meta("FIN", 0, "")),
    );
    // Two init polls (the L3160 dialect's count).
    for _ in 0..2 {
        push(
            &mut stream,
            wire::build_is_packet(wire::TYPE_PASSTHRU_REPLY, &[wire::ACK_BYTE]),
        );
        push(
            &mut stream,
            wire::build_is_packet(wire::TYPE_PASSTHRU_REPLY, &meta("STAT", 2, "")),
        );
        push(
            &mut stream,
            wire::build_is_packet(wire::TYPE_PASSTHRU_REPLY, b"OK"),
        );
        push(
            &mut stream,
            wire::build_is_packet(wire::TYPE_PASSTHRU_REPLY, &meta("FIN", 0, "")),
        );
    }
    push(
        &mut stream,
        wire::build_is_packet(wire::TYPE_PASSTHRU_REPLY, &[wire::ACK_BYTE]),
    );
    push(
        &mut stream,
        wire::build_is_packet(wire::TYPE_PASSTHRU_REPLY, &meta("STAT", 2, "")),
    );
    push(
        &mut stream,
        wire::build_is_packet(wire::TYPE_PASSTHRU_REPLY, b"OK"),
    );
    // PARA: the empty header ack (skipped by the filter), then #parOK.
    push(
        &mut stream,
        wire::build_is_packet(wire::TYPE_PASSTHRU_REPLY, &[]),
    );
    push(
        &mut stream,
        wire::build_is_packet(wire::TYPE_PASSTHRU_REPLY, &meta("PARA", 0, "parOK")),
    );
    push(
        &mut stream,
        wire::build_is_packet(wire::TYPE_PASSTHRU_REPLY, &meta("TRDT", 0, "")),
    );
    // One image chunk, page end marker in the metadata.
    push(
        &mut stream,
        wire::build_is_packet(wire::TYPE_PASSTHRU_REPLY, &meta("IMG", 4, img_meta_tokens)),
    );
    push(
        &mut stream,
        wire::build_is_packet(wire::TYPE_PASSTHRU_REPLY, b"JPEG"),
    );
    push(
        &mut stream,
        wire::build_is_packet(wire::TYPE_PASSTHRU_REPLY, &meta("FIN", 0, "")),
    );
    push(
        &mut stream,
        wire::build_is_packet(wire::TYPE_UNLOCK_ACK, &[]),
    );
    stream
}

#[test]
fn synthetic_session_delivers_the_device_jpeg() {
    let mut link = ReplayLink {
        incoming: Box::leak(synthetic_session_stream(None, "pen").into_boxed_slice()),
        position: 0,
        sent: Vec::new(),
    };
    let image =
        session::drive_for_test(&mut link, dialect::FlatbedDialect::l3160(), push_settings())
            .expect("the session completes");
    assert_eq!(image, b"JPEG");
    // The host traffic ends with the unlock.
    assert!(link.sent.ends_with(&wire::build_unlock_packet()));
}

#[test]
fn synthetic_session_refuses_rejected_parameters() {
    let link = ReplayLink {
        incoming: Box::leak(synthetic_session_stream(None, "pen").into_boxed_slice()),
        position: 0,
        sent: Vec::new(),
    };
    // Rewrite the PARA reply's #par token to NG: find the PARA META frame
    // (64 B, cmd PARA) and patch its token block.
    let mut stream = link.incoming.to_vec();
    let mut offset = 0;
    while offset + 12 <= stream.len() {
        if &stream[offset..offset + 2] != b"IS" {
            break;
        }
        let kind = u16::from_be_bytes([stream[offset + 2], stream[offset + 3]]);
        let size = u32::from_be_bytes([
            stream[offset + 6],
            stream[offset + 7],
            stream[offset + 8],
            stream[offset + 9],
        ]) as usize;
        if kind == wire::TYPE_PASSTHRU_REPLY
            && size == 64
            && &stream[offset + 12..offset + 16] == b"PARA"
        {
            let token_start = offset + 12 + 12;
            stream[token_start..token_start + 8].copy_from_slice(b"#parNG  ");
            break;
        }
        offset += 12 + size;
    }
    let mut link = ReplayLink {
        incoming: Box::leak(stream.into_boxed_slice()),
        position: 0,
        sent: Vec::new(),
    };
    let error =
        session::drive_for_test(&mut link, dialect::FlatbedDialect::l3160(), push_settings())
            .expect_err("a #parNG reply must stop the session");
    assert!(
        error.to_string().contains("refused the scan parameters"),
        "{error}"
    );
}

#[test]
fn synthetic_session_fails_on_fatal_async_events() {
    let injected = wire::build_is_packet(wire::TYPE_ASYNC_EVENT, &[0x02]);
    let mut link = ReplayLink {
        incoming: Box::leak(synthetic_session_stream(Some(injected), "pen").into_boxed_slice()),
        position: 0,
        sent: Vec::new(),
    };
    let error =
        session::drive_for_test(&mut link, dialect::FlatbedDialect::l3160(), push_settings())
            .expect_err("a disconnect event must stop the session");
    assert!(error.to_string().contains("async event 0x02"), "{error}");
}

#[test]
fn synthetic_session_reports_image_errors() {
    let mut link = ReplayLink {
        incoming: Box::leak(synthetic_session_stream(None, "pen#errFB  PJ  ").into_boxed_slice()),
        position: 0,
        sent: Vec::new(),
    };
    let error =
        session::drive_for_test(&mut link, dialect::FlatbedDialect::l3160(), push_settings())
            .expect_err("an in-image #err must stop the session");
    assert!(
        error.to_string().contains("the printer reported an error"),
        "{error}"
    );
}

// ---- Stage 3: teardown guard + fail-fast dispatch ----

/// The happy-path synthetic stream, truncated right after the pre-PARA
/// status: the session fails mid-way with the printer still locked.
fn truncated_session_stream() -> Vec<u8> {
    let full = synthetic_session_stream(None, "pen");
    // Cut before the PARA frames: everything through the pre-PARA STAT body
    // is present, so the session fails after LOCK with the printer locked.
    let mut stream = Vec::new();
    let mut offset = 0;
    while offset + 12 <= full.len() {
        if &full[offset..offset + 2] != b"IS" {
            break;
        }
        let kind = u16::from_be_bytes([full[offset + 2], full[offset + 3]]);
        let size = u32::from_be_bytes([
            full[offset + 6],
            full[offset + 7],
            full[offset + 8],
            full[offset + 9],
        ]) as usize;
        // Cut after the first small two-phase body (the INIT2 INFO reply):
        // the session gets through INIT1 (LOCK sent) and fails mid-INIT2.
        if kind == wire::TYPE_PASSTHRU_REPLY && size == 2 {
            let end = offset + 12 + size;
            let next_size =
                u32::from_be_bytes([full[end + 6], full[end + 7], full[end + 8], full[end + 9]])
                    as usize;
            let stop = (end + 12 + next_size).min(full.len());
            // Keep the session from its start, through the next frame.
            stream.extend_from_slice(&full[..stop]);
            return stream;
        }
        offset += 12 + size;
    }
    full
}

#[test]
fn session_sends_unlock_on_teardown_after_lock() {
    let mut link = ReplayLink {
        incoming: Box::leak(truncated_session_stream().into_boxed_slice()),
        position: 0,
        sent: Vec::new(),
    };
    // The stream ends mid-session: the failure must not leave the printer
    // locked — the guard's teardown UNLOCK is the last thing written.
    let result =
        session::drive_for_test(&mut link, dialect::FlatbedDialect::l3160(), push_settings());
    assert!(result.is_err(), "the truncated stream must fail");
    assert!(
        link.sent.ends_with(&wire::build_unlock_packet()),
        "teardown after LOCK must send UNLOCK"
    );
}

#[test]
fn session_does_not_unlock_before_lock() {
    // A legacy-ESC/I welcome fails the session before LOCK is sent: the
    // guard must stay silent (no unlock goes out for a lock never held).
    let mut link = ReplayLink {
        incoming: Box::leak(
            wire::build_is_packet(wire::TYPE_WELCOME, &[0x01, 0x02, 0x00, 0x00, 0x00])
                .into_boxed_slice(),
        ),
        position: 0,
        sent: Vec::new(),
    };
    let error =
        session::drive_for_test(&mut link, dialect::FlatbedDialect::l3160(), push_settings())
            .expect_err("a legacy welcome must stop the session");
    assert!(error.to_string().contains("legacy ESC/I"), "{error}");
    assert!(
        !link.sent.ends_with(&wire::build_unlock_packet()),
        "no UNLOCK for a lock never held"
    );
}

#[test]
fn session_fails_fast_on_unexpected_frames() {
    // A job-reply frame (0xa300) injected right after the lock ack: the
    // state machine must stop, not skip, per the reference's dispatch.
    let injected = wire::build_is_packet(wire::TYPE_JOB_REPLY, &[0xde, 0xad]);
    let mut link = ReplayLink {
        incoming: Box::leak(synthetic_session_stream(Some(injected), "pen").into_boxed_slice()),
        position: 0,
        sent: Vec::new(),
    };
    let error =
        session::drive_for_test(&mut link, dialect::FlatbedDialect::l3160(), push_settings())
            .expect_err("an unexpected frame must stop the session");
    assert!(error.to_string().contains("unexpected 0xa300"), "{error}");
}
