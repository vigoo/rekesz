//! The ESC/I-2 flatbed scan session: the state machine the printer's push
//! scan expects on TCP 1865 after the trigger is answered. Sequence ports of
//! epson2paperless's `graph.ts` flatbed path (MIT), validated byte-for-byte
//! against a committed real ET-4800 session capture (testdata/
//! et4800-flatbed-jpg-*.bin) via the ET-4800 dialect.
//!
//! Wire order (flatbed): welcome → LOCK → FS Y → INFO/CAPA two-phase → FIN →
//! FS Z → INFO/CAPA/RESA → FIN → N× (FS Y → STAT → drain → FIN) → FS X →
//! STAT (BUSY/WUP wait) → PARA (header + body passthrus, #parOK) → TRDT →
//! IMG loop → FIN → UNLOCK (empty-payload ack). No POSTSCAN — that is
//! ADF-only. Any failure after LOCK still sends UNLOCK before teardown, or
//! the printer stays locked for the 300 s session timeout.

use std::io::{Read, Write};
use std::net::{Shutdown, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use super::dialect::{self, FlatbedDialect};
use super::wire::{
    self, Esci2Reply, ParsedPacket, build_esci2_command, build_lock_packet, build_para_header,
    build_passthru_packet, build_pureread_packet, build_unlock_packet, parse_esci2_reply,
    parse_is_packet,
};
use crate::trace;

/// Rolling per-state timeout (the reference's Wi-Fi warm-up tolerance); also
/// the socket read timeout, so a silent printer cannot wedge the session.
const STATE_BUDGET: Duration = Duration::from_secs(60);
/// Session-level ceiling: a trickling peer cannot extend a session forever.
const SESSION_BUDGET: Duration = Duration::from_secs(600);
/// Re-STAT cadence while the printer reports BUSY/WUP before PARA.
const NOT_READY_RETRY: Duration = Duration::from_millis(100);
/// Zero-length IMG replies without #pen mean "not ready yet"; one model sent
/// thousands before data arrived. Paced so the ceiling is ~100 s.
const MAX_ZERO_IMG_RETRIES: u32 = 5000;
const ZERO_IMG_RETRY_PAUSE: Duration = Duration::from_millis(20);
/// Assembled-image ceiling: a hostile or buggy peer cannot grow the buffer
/// forever (a 1200-DPI A4 JPEG is well under 100 MB).
const MAX_IMAGE_BYTES: usize = 256 * 1024 * 1024;
/// FS command replies are one byte carrying the ACK.
const LEGACY_REPLY_SIZE: u32 = wire::LEGACY_REPLY_SIZE;
/// ESC/I-2 command replies are 64-byte frames.
const ESCI2_REPLY_SIZE: u32 = wire::ESCI2_REPLY_SIZE;

/// The transport the session runs over, so tests can replay recorded
/// sessions instead of opening sockets.
pub trait Link {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize>;
    fn write_all(&mut self, data: &[u8]) -> std::io::Result<()>;
}

impl Link for TcpStream {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        Read::read(self, buf)
    }
    fn write_all(&mut self, data: &[u8]) -> std::io::Result<()> {
        Write::write_all(self, data)
    }
}

/// The session link with lock tracking: if the printer is still locked when
/// the session tears down (an error, the printer's own cancel event, a
/// panic unwind), the guard sends UNLOCK — otherwise the printer refuses
/// sessions for its 300 s lock timeout. The happy path's own UNLOCK
/// disarms it.
struct GuardedLink<'a> {
    inner: &'a mut dyn Link,
    locked: bool,
}

impl Link for GuardedLink<'_> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.inner.read(buf)
    }

    fn write_all(&mut self, data: &[u8]) -> std::io::Result<()> {
        // The session writes whole IS packets; track the lock frames.
        if data.len() >= 4 {
            match u16::from_be_bytes([data[2], data[3]]) {
                wire::TYPE_LOCK => self.locked = true,
                wire::TYPE_UNLOCK => self.locked = false,
                _ => {}
            }
        }
        self.inner.write_all(data)
    }
}

impl Drop for GuardedLink<'_> {
    fn drop(&mut self) {
        if self.locked {
            // Best effort: the printer may already be gone.
            let _ = self.inner.write_all(&build_unlock_packet());
            trace::event("push: session teardown sent the missing UNLOCK");
        }
    }
}

#[derive(Debug)]
pub enum SessionError {
    Failed(String),
    Cancelled,
}

impl std::fmt::Display for SessionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SessionError::Failed(message) => write!(f, "{message}"),
            SessionError::Cancelled => write!(f, "cancelled"),
        }
    }
}

/// One session run: connect to the printer's ESC/I-2 port, drive the flatbed
/// state machine with `dialect`'s parameters at `dpi`, and return the
/// device-encoded JPEG. `cancel` is checked between frames.
pub struct Session<'a> {
    pub printer: std::net::IpAddr,
    pub dialect: FlatbedDialect,
    pub settings: super::engine::PushScanSettings,
    pub cancel: &'a AtomicBool,
}

/// A two-phase read's pieces: the 64-byte META frame (with its tokens) and,
/// when the META declared one, the drained body.
struct MetaAndBody {
    reply: Esci2Reply,
    body: Option<Vec<u8>>,
}

impl Session<'_> {
    pub fn run(&self) -> Result<Vec<u8>, SessionError> {
        let address = std::net::SocketAddr::new(self.printer, 1865);
        let mut stream = TcpStream::connect_timeout(&address, Duration::from_secs(10))
            .map_err(|error| SessionError::Failed(format!("connect {address}: {error}")))?;
        let _ = stream.set_nodelay(true);
        // A silent printer (sleep, Wi-Fi black hole, crashed firmware) must
        // fail the read within the state budget — a blocking read with no
        // timeout would wedge the engine thread forever.
        let _ = stream.set_read_timeout(Some(STATE_BUDGET));
        match self.drive(&mut stream) {
            Ok(image) => {
                let _ = stream.shutdown(Shutdown::Write);
                Ok(image)
            }
            Err(error) => {
                // GuardedLink's drop already sent the UNLOCK (the printer
                // otherwise refuses sessions for its 300 s lock timeout);
                // drain briefly so the close is a FIN, not an RST. (std's
                // inherent read — not the Link impl — drives the raw stream.)
                let _ = stream.set_read_timeout(Some(Duration::from_millis(200)));
                let mut sink = [0_u8; 512];
                for _ in 0..2 {
                    match std::io::Read::read(&mut stream, &mut sink) {
                        Ok(0) | Err(_) => break,
                        Ok(_) => continue,
                    }
                }
                let _ = stream.shutdown(Shutdown::Write);
                Err(error)
            }
        }
    }

    fn drive(&self, link: &mut dyn Link) -> Result<Vec<u8>, SessionError> {
        let mut guarded = GuardedLink {
            inner: link,
            locked: false,
        };
        let mut machine = Machine {
            link: &mut guarded,
            buffer: Vec::new(),
            cancel: self.cancel,
            started: Instant::now(),
        };
        // Welcome: the printer speaks first on plain TCP. Payload byte 1 is
        // the family discriminator (0x02 = legacy ESC/I, anything else ESC/I-2).
        let (kind, payload) = machine.read_frame("welcome")?;
        if kind != wire::TYPE_WELCOME {
            return Err(SessionError::Failed(format!(
                "expected the 0x8000 welcome, got 0x{kind:04x}"
            )));
        }
        if payload.get(1) == Some(&0x02) {
            return Err(SessionError::Failed(
                "printer announced legacy ESC/I — this session speaks ESC/I-2 only".to_string(),
            ));
        }
        machine.send(&build_lock_packet(), "LOCK")?;
        machine.expect_ack(wire::TYPE_LOCK_ACK, "LOCK")?;

        machine.send(
            &build_passthru_packet(&wire::FS_Y, LEGACY_REPLY_SIZE),
            "FS Y",
        )?;
        machine.expect_fs_ack("FS Y")?;
        let info_body = machine
            .two_phase("INFO")?
            .body
            .ok_or_else(|| SessionError::Failed("the INFO reply declared no body".to_string()))?;
        let capa_body = machine
            .two_phase("CAPA")?
            .body
            .ok_or_else(|| SessionError::Failed("the CAPA reply declared no body".to_string()))?;
        machine.single_envelope("FIN")?;

        // The pinned dialect is validated against what the device just said:
        // a CAPA that lacks our gamma mode or the JPG format would be
        // refused by the firmware at PARA time anyway.
        let info = dialect::parse_info_tokens(&info_body);
        let capa = dialect::parse_capa_tokens(&capa_body);
        dialect::log_device(&info, &capa);
        dialect::check_settings(self.dialect, &capa, self.settings.color)
            .map_err(SessionError::Failed)?;

        machine.send(
            &build_passthru_packet(&wire::FS_Z, LEGACY_REPLY_SIZE),
            "FS Z",
        )?;
        machine.expect_fs_ack("FS Z")?;
        machine.two_phase("INFO")?;
        machine.two_phase("CAPA")?;
        machine.two_phase("RESA")?;
        machine.single_envelope("FIN")?;

        for _poll in 0..self.dialect.init_polls {
            machine.send(
                &build_passthru_packet(&wire::FS_Y, LEGACY_REPLY_SIZE),
                "FS Y",
            )?;
            machine.expect_fs_ack("FS Y")?;
            machine.two_phase("STAT")?;
            machine.single_envelope("FIN")?;
        }

        machine.send(
            &build_passthru_packet(&wire::FS_X, LEGACY_REPLY_SIZE),
            "FS X",
        )?;
        machine.expect_fs_ack("FS X")?;

        // POST_MODE status: wait out BUSY/WUP (a printer waking from sleep
        // re-STATs every 100 ms), stop on real errors, drain the body.
        let not_ready_start = Instant::now();
        let stat = loop {
            let stat = machine.two_phase("STAT")?;
            let not_ready = stat
                .reply
                .tokens
                .iter()
                .find(|(key, _)| key == "nrd")
                .map(|(_, value)| value.trim().to_string());
            match not_ready.as_deref() {
                Some("BUSY") | Some("WUP") => {
                    if not_ready_start.elapsed() > STATE_BUDGET {
                        return Err(SessionError::Failed(
                            "the printer stayed busy before PARA for 60s".to_string(),
                        ));
                    }
                    std::thread::sleep(NOT_READY_RETRY);
                    continue;
                }
                Some("RSVD") | Some("AFEN") => {
                    return Err(SessionError::Failed(format!(
                        "the printer reported #nrd {} before PARA",
                        not_ready.unwrap()
                    )));
                }
                _ => {}
            }
            if stat.reply.tokens.iter().any(|(key, _)| key == "err") {
                return Err(SessionError::Failed(
                    "the printer reported #err before PARA".to_string(),
                ));
            }
            break stat;
        };
        if let Some(body) = stat.body {
            Self::judge_stat(&body, "pre-PARA status")?;
        }

        // Resolution negotiation: the panel's pick, resolved against the
        // device's advertised set (exact, else the smallest value above —
        // the image then comes back at the wire DPI — else the maximum).
        let advertised = dialect::advertised_dpi_set(&capa);
        if advertised.is_empty() {
            return Err(SessionError::Failed(
                "the device advertises no usable resolutions".to_string(),
            ));
        }
        let wire_dpi = dialect::select_wire_dpi(self.settings.dpi, &advertised);
        if wire_dpi != self.settings.dpi {
            trace::event(&format!(
                "push: {} DPI not advertised — scanning at {wire_dpi} DPI",
                self.settings.dpi
            ));
        }
        let window = dialect::scan_window(&info, self.settings.region, wire_dpi)
            .map_err(SessionError::Failed)?;
        let para = dialect::compose_para(self.dialect, wire_dpi, window, self.settings.color);

        // PARA: the header passthru (reply_size 0) is acknowledged with an
        // empty envelope, which the empty-filter skips; the body passthru
        // answers with a 64-byte frame whose #par token must read OK.
        machine.send(
            &build_passthru_packet(&build_para_header(para.len()), 0),
            "PARA header",
        )?;
        machine.send(&build_passthru_packet(&para, ESCI2_REPLY_SIZE), "PARA body")?;
        let reply = machine.read_passthru_reply("PARA body")?;
        let parsed = parse_esci2_reply(&reply)
            .ok_or_else(|| SessionError::Failed("unparseable PARA reply".to_string()))?;
        if parsed.cmd != "PARA" {
            return Err(SessionError::Failed(format!(
                "PARA reply echo says {:?}",
                parsed.cmd
            )));
        }
        let par_ok = parsed
            .tokens
            .iter()
            .any(|(key, value)| key == "par" && value.trim() == "OK");
        if !par_ok {
            return Err(SessionError::Failed(format!(
                "the printer refused the scan parameters ({:?})",
                parsed.tokens
            )));
        }

        machine.send(
            &build_passthru_packet(&build_esci2_command("TRDT"), ESCI2_REPLY_SIZE),
            "TRDT",
        )?;
        machine.read_passthru_reply("TRDT")?;

        // The image is complete once #pen arrives; everything after it is
        // teardown, and a failure there must not discard the scan.
        let image = machine.img_loop()?;
        // Everything after the last image chunk is cleanup: the reference
        // treats UNLOCKING as a cleanup state, so a failure there must not
        // discard the completed scan. (Deliberate divergence: the reference
        // still rejects the scan when the FIN ack fails — the FIN ack is the
        // printer's transfer-completed confirmation; we forgive it.)
        if machine.single_envelope("FIN").is_err() {
            trace::slow_event("push: session FIN after the image failed — keeping the scan");
        }
        if machine.send(&build_unlock_packet(), "UNLOCK").is_err() {
            trace::slow_event(
                "push: session unlock send failed — keeping the scan (the printer self-heals)",
            );
        } else {
            match machine.expect_kind(wire::TYPE_UNLOCK_ACK, "UNLOCK") {
                Ok(ack) => {
                    trace::event(&format!("push: session ← unlock ack ({} B)", ack.len()));
                }
                Err(error) => {
                    trace::slow_event(&format!(
                        "push: session unlock ack not received ({error}) — keeping the scan"
                    ));
                }
            }
        }
        Ok(image)
    }

    /// `#ERR` segments name a 4-char unit + a 4-char code; an error on the
    /// flatbed (the unit being scanned) stops the scan. Byte-safe slicing:
    /// reply bodies come off the wire and may be anything.
    fn judge_stat(body: &[u8], phase: &str) -> Result<(), SessionError> {
        for segment in dialect::split_hash_segments(body) {
            if let Some(rest) = segment.strip_prefix("#ERR") {
                let unit = rest
                    .as_bytes()
                    .get(..4)
                    .map(|bytes| String::from_utf8_lossy(bytes).trim().to_string());
                if unit.as_deref() == Some("FB") {
                    return Err(SessionError::Failed(format!(
                        "{phase}: the flatbed reported an error ({rest})"
                    )));
                }
            }
        }
        Ok(())
    }
}

/// The state machine's I/O: IS framing over any [`Link`], with empty-
/// envelope filtering, cancel checks, and async-event handling.
struct Machine<'a> {
    link: &'a mut dyn Link,
    buffer: Vec<u8>,
    cancel: &'a AtomicBool,
    started: Instant,
}

impl Machine<'_> {
    fn send(&mut self, packet: &[u8], label: &str) -> Result<(), SessionError> {
        self.check_cancel(label)?;
        self.link
            .write_all(packet)
            .map_err(|error| SessionError::Failed(format!("{label}: write: {error}")))?;
        trace::event(&format!(
            "push: session → {label} ({} B payload)",
            packet.len() - wire::IS_HEADER_SIZE
        ));
        Ok(())
    }

    fn check_cancel(&self, label: &str) -> Result<(), SessionError> {
        if self.cancel.load(Ordering::Relaxed) {
            trace::event(&format!("push: session cancelled during {label}"));
            return Err(SessionError::Cancelled);
        }
        if self.started.elapsed() > SESSION_BUDGET {
            return Err(SessionError::Failed(format!(
                "the session exceeded its {}s ceiling during {label}",
                SESSION_BUDGET.as_secs()
            )));
        }
        Ok(())
    }

    /// Read one IS frame, dispatching async events (0x02 disconnect / 0x80
    /// timeout / 0xa0 server error are fatal; 0x03 is the printer
    /// cancelling; anything else is informational, like the reference).
    fn read_frame(&mut self, label: &str) -> Result<(u16, Vec<u8>), SessionError> {
        let deadline = Instant::now() + STATE_BUDGET;
        loop {
            self.check_cancel(label)?;
            match parse_is_packet(&self.buffer) {
                ParsedPacket::Packet {
                    consumed,
                    kind,
                    payload,
                } => {
                    self.buffer.drain(..consumed);
                    if kind == wire::TYPE_ASYNC_EVENT {
                        let code = payload.first().copied().unwrap_or(0);
                        return match code {
                            0x03 => Err(SessionError::Cancelled),
                            0x02 | 0x80 | 0xa0 => Err(SessionError::Failed(format!(
                                "printer async event 0x{code:02x} during {label}"
                            ))),
                            _ => continue, // ScanStart, Stop, unknown — info-only
                        };
                    }
                    return Ok((kind, payload));
                }
                ParsedPacket::Malformed => {
                    let head = self.buffer.len().min(16);
                    return Err(SessionError::Failed(format!(
                        "session stream is not IS-framed (head: {:02x?}) at {label}",
                        &self.buffer[..head]
                    )));
                }
                ParsedPacket::NeedMore => {}
            }
            if Instant::now() > deadline {
                return Err(SessionError::Failed(format!(
                    "timed out waiting for a frame during {label}"
                )));
            }
            let mut chunk = [0_u8; 8192];
            let read = self
                .link
                .read(&mut chunk)
                .map_err(|error| SessionError::Failed(format!("{label}: read: {error}")))?;
            if read == 0 {
                return Err(SessionError::Failed(format!(
                    "printer closed the connection during {label}"
                )));
            }
            self.buffer.extend_from_slice(&chunk[..read]);
        }
    }

    /// Wait for a frame of `kind`. Empty 0xa000 envelopes are the wire's
    /// "wait for body" signals and stay filtered (they arrive as the PARA
    /// header ack, among others); any other unexpected frame fails the
    /// session fast, like the reference's dispatch.
    fn expect_kind(&mut self, kind: u16, label: &str) -> Result<Vec<u8>, SessionError> {
        loop {
            let (frame_kind, payload) = self.read_frame(label)?;
            if frame_kind == wire::TYPE_PASSTHRU_REPLY && payload.is_empty() {
                continue;
            }
            if frame_kind == kind {
                return Ok(payload);
            }
            return Err(SessionError::Failed(format!(
                "unexpected 0x{frame_kind:04x} frame ({} B) during {label}, wanted 0x{kind:04x}",
                payload.len()
            )));
        }
    }

    /// LOCK ACK: a payload whose first byte is 0x06.
    fn expect_ack(&mut self, kind: u16, label: &str) -> Result<(), SessionError> {
        let payload = self.expect_kind(kind, label)?;
        if payload.first() != Some(&wire::ACK_BYTE) {
            return Err(SessionError::Failed(format!(
                "{label}: expected ACK 0x06, got {payload:02x?}"
            )));
        }
        Ok(())
    }

    /// FS Y/Z/X: a 1-byte 0xa000 reply carrying the ACK.
    fn expect_fs_ack(&mut self, label: &str) -> Result<(), SessionError> {
        let payload = self.read_passthru_reply(label)?;
        if payload.first() != Some(&wire::ACK_BYTE) {
            return Err(SessionError::Failed(format!(
                "{label}: expected ACK 0x06, got {payload:02x?}"
            )));
        }
        Ok(())
    }

    fn read_passthru_reply(&mut self, label: &str) -> Result<Vec<u8>, SessionError> {
        self.expect_kind(wire::TYPE_PASSTHRU_REPLY, label)
    }

    /// Two-phase read: 64-byte META frame declaring the body length, then a
    /// pure-read of exactly that length.
    fn two_phase(&mut self, name: &str) -> Result<MetaAndBody, SessionError> {
        self.send(
            &build_passthru_packet(&build_esci2_command(name), ESCI2_REPLY_SIZE),
            name,
        )?;
        let meta = self.read_passthru_reply(name)?;
        if meta.len() != ESCI2_REPLY_SIZE as usize {
            return Err(SessionError::Failed(format!(
                "{name}: META frame is {} bytes, want {ESCI2_REPLY_SIZE}",
                meta.len()
            )));
        }
        let reply = parse_esci2_reply(&meta)
            .ok_or_else(|| SessionError::Failed(format!("{name}: unparseable META frame")))?;
        if reply.cmd != name {
            return Err(SessionError::Failed(format!(
                "{name}: META echo says {:?}",
                reply.cmd
            )));
        }
        let body = if reply.length > 0 {
            self.send(
                &build_pureread_packet(reply.length),
                &format!("{name}:pureread"),
            )?;
            let data = self.read_passthru_reply(&format!("{name}:pureread"))?;
            if data.len() != reply.length as usize {
                return Err(SessionError::Failed(format!(
                    "{name}: body is {} bytes, META declared {}",
                    data.len(),
                    reply.length
                )));
            }
            Some(data)
        } else {
            None
        };
        Ok(MetaAndBody { reply, body })
    }

    /// FIN: a single 64-byte envelope reply — no two-phase body, no ACK byte.
    fn single_envelope(&mut self, name: &str) -> Result<(), SessionError> {
        self.send(
            &build_passthru_packet(&build_esci2_command(name), ESCI2_REPLY_SIZE),
            name,
        )?;
        self.read_passthru_reply(name)?;
        Ok(())
    }

    /// The image read loop: IMG → META (chunk length + #pen tokens) →
    /// pure-read of the chunk → … until #pen (flatbed pages always end).
    /// Metadata replies are judged like the reference's IMG_META state: #nrd
    /// RSVD/AFEN and #err end the session.
    fn img_loop(&mut self) -> Result<Vec<u8>, SessionError> {
        let mut image = Vec::new();
        let mut zero_retries = 0_u32;
        loop {
            self.send(
                &build_passthru_packet(&build_esci2_command("IMG"), ESCI2_REPLY_SIZE),
                "IMG",
            )?;
            let meta = self.read_passthru_reply("IMG")?;
            if meta.len() != ESCI2_REPLY_SIZE as usize {
                return Err(SessionError::Failed(format!(
                    "IMG: metadata frame is {} bytes, want {ESCI2_REPLY_SIZE}",
                    meta.len()
                )));
            }
            let reply = parse_esci2_reply(&meta)
                .ok_or_else(|| SessionError::Failed("IMG: unparseable metadata".to_string()))?;
            if reply.cmd != "IMG" {
                return Err(SessionError::Failed(format!(
                    "IMG: metadata echo says {:?}",
                    reply.cmd
                )));
            }
            for (key, value) in &reply.tokens {
                match key.as_str() {
                    "err" | "ERR" => {
                        return Err(SessionError::Failed(format!(
                            "IMG: the printer reported an error ({value})"
                        )));
                    }
                    "nrd" if value.trim() == "RSVD" || value.trim() == "AFEN" => {
                        return Err(SessionError::Failed(format!(
                            "IMG: the printer reported #nrd {}",
                            value.trim()
                        )));
                    }
                    _ => {}
                }
            }
            let pen = reply.tokens.iter().any(|(key, _)| key == "pen");
            if reply.length == 0 {
                if pen {
                    break;
                }
                zero_retries += 1;
                if zero_retries > MAX_ZERO_IMG_RETRIES {
                    return Err(SessionError::Failed(
                        "the printer sent zero-length IMG replies too many times".to_string(),
                    ));
                }
                std::thread::sleep(ZERO_IMG_RETRY_PAUSE);
                continue;
            }
            zero_retries = 0;
            if image.len() + reply.length as usize > MAX_IMAGE_BYTES {
                return Err(SessionError::Failed(
                    "the scan image exceeds the assembled-size ceiling".to_string(),
                ));
            }
            self.send(&build_pureread_packet(reply.length), "IMG:pureread")?;
            let mut chunk = self.read_passthru_reply("IMG:pureread")?;
            // Some printers re-echo the IMG metadata header in response to
            // the pure-read; detect a same-length IMG META and read again.
            if chunk.len() == ESCI2_REPLY_SIZE as usize
                && let Some(echo) = parse_esci2_reply(&chunk)
                && echo.cmd == "IMG"
                && echo.length == reply.length
            {
                chunk = self.read_passthru_reply("IMG:pureread")?;
            }
            if chunk.len() != reply.length as usize {
                return Err(SessionError::Failed(format!(
                    "IMG: chunk is {} bytes, metadata declared {}",
                    chunk.len(),
                    reply.length
                )));
            }
            image.extend_from_slice(&chunk);
            if pen {
                break;
            }
        }
        Ok(image)
    }
}

/// Convenience wrapper the trigger path calls: run the session over a fresh
/// TCP connection and return the device-encoded JPEG bytes.
pub fn run_flatbed(
    printer: std::net::IpAddr,
    settings: super::engine::PushScanSettings,
    cancel: &AtomicBool,
) -> Result<Vec<u8>, SessionError> {
    let session = Session {
        printer,
        dialect: FlatbedDialect::l3160(),
        settings,
        cancel,
    };
    session.run()
}

/// Replay entry for tests: drive the state machine over a scripted link.
#[cfg(test)]
pub(crate) fn drive_for_test(
    link: &mut dyn Link,
    dialect: FlatbedDialect,
    settings: super::engine::PushScanSettings,
) -> Result<Vec<u8>, SessionError> {
    let cancel = AtomicBool::new(false);
    let session = Session {
        printer: std::net::IpAddr::V4(std::net::Ipv4Addr::LOCALHOST),
        dialect,
        settings,
        cancel: &cancel,
    };
    session.drive(link)
}
