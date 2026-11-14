//! Epson NetScanMonitor push-scan wire codec: IS packet framing, lock/unlock
//! and passthru packet builders, and ESC/I-2 command/reply-frame parsing.
//!
//! Byte-exact port of the protocol facts documented by the MIT-licensed
//! epson2paperless project (`docs/PROTOCOL-REFERENCE.md`, `src/protocol.ts`,
//! `src/esci2/commands.ts`, `src/esci2/graph.ts`). Deliberately free of
//! `crate::` references so the `scanner-probe` binary can include this file
//! via `#[path]`, like escl.rs and discover.rs.

/// IS packet header: "IS" magic + type + data offset + payload size + padding.
pub const IS_HEADER_SIZE: usize = 12;
/// ESC/I-2 command replies are 64-byte frames (52-byte token block).
pub const ESCI2_REPLY_SIZE: u32 = 64;
/// Legacy FS-command replies (the FS Y/Z/X ACK) are one byte.
pub const LEGACY_REPLY_SIZE: u32 = 1;
/// ESC/I-2 session lock timeout; the printer refuses other locks meanwhile,
/// so a crashed session must still send UNLOCK before teardown.
pub const LOCK_TIMEOUT_SECONDS: u32 = 300;

pub const TYPE_WELCOME: u16 = 0x8000;
pub const TYPE_ASYNC_EVENT: u16 = 0x9000;
pub const TYPE_PASSTHRU_REPLY: u16 = 0xa000;
pub const TYPE_LOCK_ACK: u16 = 0xa100;
pub const TYPE_UNLOCK_ACK: u16 = 0xa101;
pub const TYPE_LEGACY_IMAGE_CHUNK: u16 = 0xa200;
pub const TYPE_JOB_REPLY: u16 = 0xa300;
pub const TYPE_PASSTHRU: u16 = 0x2000;
pub const TYPE_LOCK: u16 = 0x2100;
pub const TYPE_UNLOCK: u16 = 0x2101;
pub const TYPE_LEGACY_STREAM_CONFIG: u16 = 0x2200;
pub const TYPE_JOB_COMMAND: u16 = 0x2300;

/// Raw legacy ESC/I-2 mode-switch bytes: FS Y (init poll / entering),
/// FS Z (second init), FS X (mode switch before scanning).
pub const FS_Y: [u8; 2] = [0x1c, 0x59];
pub const FS_Z: [u8; 2] = [0x1c, 0x5a];
pub const FS_X: [u8; 2] = [0x1c, 0x58];

/// The single-byte ACK the printer sends for FS commands and lock/unlock.
pub const ACK_BYTE: u8 = 0x06;

/// Outcome of trying to parse one IS packet from the front of a stream
/// buffer. TCP may split or coalesce packets arbitrarily, so parsing is
/// incremental: feed more bytes on [`ParsedPacket::NeedMore`].
#[derive(Debug)]
pub enum ParsedPacket {
    NeedMore,
    /// A complete packet; `consumed` bytes of the buffer belong to it. Only
    /// the type (header bytes 2–3) and payload size (bytes 6–9) are read —
    /// the data-offset bytes 4–5 vary per printer (`0x300C`, `0x380C`) and
    /// are ignored on parse.
    Packet {
        consumed: usize,
        kind: u16,
        payload: Vec<u8>,
    },
    Malformed,
}

/// Refusal threshold for a declared payload size; a garbled or hostile
/// stream must not make the caller buffer toward 4 GB.
pub const MAX_PAYLOAD_SIZE: usize = 64 * 1024 * 1024;

pub fn parse_is_packet(buffer: &[u8]) -> ParsedPacket {
    // Fragments shorter than the magic can still be the head of a genuine
    // frame — this protocol family demonstrably fragments IS frames.
    if buffer.len() < 2 {
        return ParsedPacket::NeedMore;
    }
    if &buffer[0..2] != b"IS" {
        return ParsedPacket::Malformed;
    }
    if buffer.len() < IS_HEADER_SIZE {
        return ParsedPacket::NeedMore;
    }
    let kind = u16::from_be_bytes([buffer[2], buffer[3]]);
    let payload_size = u32::from_be_bytes([buffer[6], buffer[7], buffer[8], buffer[9]]) as usize;
    if payload_size > MAX_PAYLOAD_SIZE {
        return ParsedPacket::Malformed;
    }
    let total = IS_HEADER_SIZE + payload_size;
    if buffer.len() < total {
        return ParsedPacket::NeedMore;
    }
    ParsedPacket::Packet {
        consumed: total,
        kind,
        payload: buffer[IS_HEADER_SIZE..total].to_vec(),
    }
}

/// Build an IS packet. The host writes data offset `0x000C` — the value
/// Windows drivers put on the wire, confirmed by the reference's Frida
/// captures; printers send their own offsets, which parsing ignores.
pub fn build_is_packet(kind: u16, payload: &[u8]) -> Vec<u8> {
    let mut packet = Vec::with_capacity(IS_HEADER_SIZE + payload.len());
    packet.extend_from_slice(b"IS");
    packet.extend_from_slice(&kind.to_be_bytes());
    packet.extend_from_slice(&0x000c_u16.to_be_bytes());
    packet.extend_from_slice(&(payload.len() as u32).to_be_bytes());
    packet.extend_from_slice(&[0, 0]);
    packet.extend_from_slice(payload);
    packet
}

/// Session lock (300 s): 7-byte payload `01 a0 04` + BE u32 timeout.
pub fn build_lock_packet() -> Vec<u8> {
    let mut payload = [0_u8; 7];
    payload[0] = 0x01;
    payload[1] = 0xa0;
    payload[2] = 0x04;
    payload[3..7].copy_from_slice(&LOCK_TIMEOUT_SECONDS.to_be_bytes());
    build_is_packet(TYPE_LOCK, &payload)
}

/// Session unlock: header only, empty payload.
pub fn build_unlock_packet() -> Vec<u8> {
    build_is_packet(TYPE_UNLOCK, &[])
}

/// Passthru command: 8-byte preamble (command size, expected reply size)
/// followed by the command bytes. The ESC/I-2 command set travels inside
/// these; reply_size tells the printer how many bytes to answer with.
pub fn build_passthru_packet(command: &[u8], reply_size: u32) -> Vec<u8> {
    let mut payload = Vec::with_capacity(8 + command.len());
    payload.extend_from_slice(&(command.len() as u32).to_be_bytes());
    payload.extend_from_slice(&reply_size.to_be_bytes());
    payload.extend_from_slice(command);
    build_is_packet(TYPE_PASSTHRU, &payload)
}

/// Pure-read: pulls queued reply bytes without sending a command
/// (cmd_size 0). Used for the second phase of two-phase reads and for
/// image chunks.
pub fn build_pureread_packet(reply_size: u32) -> Vec<u8> {
    let mut payload = [0_u8; 8];
    payload[4..8].copy_from_slice(&reply_size.to_be_bytes());
    build_is_packet(TYPE_PASSTHRU, &payload)
}

/// Zero-parameter ESC/I-2 command: 4-char space-padded name + "x" + 7 zero
/// hex digits, e.g. `INFOx0000000`.
pub fn build_esci2_command(name: &str) -> Vec<u8> {
    format!("{:<4}x0000000", name).into_bytes()
}

/// PARA command header: `PARAx` + 7-digit uppercase hex of the body length
/// (the body itself travels in a second passthru, never the same one).
pub fn build_para_header(body_len: usize) -> Vec<u8> {
    format!("PARAx{:07X}", body_len).into_bytes()
}

/// A parsed ESC/I-2 reply frame: echoed command name, declared body length
/// (drives the following pure-read), and the `#`-token block.
pub struct Esci2Reply {
    pub cmd: String,
    pub length: u32,
    pub tokens: Vec<(String, String)>,
}

/// Parse a 64-byte ESC/I-2 reply frame: bytes 0–3 command name
/// (right-trimmed), byte 4 the literal `x`, bytes 5–11 seven hex digits of
/// declared body length, bytes 12+ the token block (`#`-prefixed 3-char
/// keys, `#---` filler terminates).
pub fn parse_esci2_reply(payload: &[u8]) -> Option<Esci2Reply> {
    if payload.len() < 12 || payload[4] != b'x' {
        return None;
    }
    let cmd: String = payload[0..4].iter().map(|&b| b as char).collect();
    let cmd = cmd.trim_end().to_string();
    let hex: String = payload[5..12].iter().map(|&b| b as char).collect();
    let length = u32::from_str_radix(&hex, 16).ok()?;
    Some(Esci2Reply {
        cmd,
        length,
        tokens: parse_tokens(&payload[12..]),
    })
}

/// Token-block parse: `#`-prefixed 3-char keys with values running to the
/// next `#`; valueless markers (`#pen`) carry an empty value; `#---` filler
/// terminates. Values are preserved untrimmed per the reference — callers
/// trim if they care.
fn parse_tokens(block: &[u8]) -> Vec<(String, String)> {
    let mut tokens = Vec::new();
    let text = String::from_utf8_lossy(block).into_owned();
    for segment in text.split('#') {
        let bytes = segment.as_bytes();
        if bytes.is_empty() {
            continue;
        }
        if bytes.len() < 3 {
            break;
        }
        let key: String = bytes[0..3].iter().map(|&b| b as char).collect();
        // `#---` filler pads the 52-byte token block; it ends the parse.
        if key == "---" {
            break;
        }
        let value: String = bytes[3..].iter().map(|&b| b as char).collect();
        tokens.push((key, value));
    }
    tokens
}
