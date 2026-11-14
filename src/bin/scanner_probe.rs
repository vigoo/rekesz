//! Manual eSCL scanner probe: discovery, capabilities, and one-off scans
//! against real hardware — the live-investigation counterpart of
//! `lens-profile-calibrate`, and the pre-UI exercise of the Stage 1 client.
//! Also carries the push-scan dialect probe (`push-probe`, stage 0 of
//! scanner-push-scan-plan.md).

#[path = "../sources/scanner/discover.rs"]
mod discover;
#[path = "../sources/scanner/escl.rs"]
mod escl;
// The wire codec carries the full protocol constant set; the probe uses the
// subset its stage-0 exchange needs, the rest lands with later stages.
#[allow(dead_code)]
#[path = "../sources/scanner/push/wire.rs"]
mod wire;

use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

use clap::{Parser, Subcommand};

use discover::SCANNERS;
use escl::{
    ColorMode, EsclClient, ScanJobTimings, ScanSettings, region_from_fractions, units_to_mm,
};

#[derive(Parser)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Sweep mDNS for eSCL scanners and print what is found.
    Discover {
        /// Seconds to keep sweeping.
        #[arg(default_value_t = 8)]
        seconds: u64,
    },
    /// Print a device's parsed capabilities.
    Capabilities {
        /// eSCL base URL, e.g. https://192.0.2.10:443/eSCL
        url: String,
    },
    /// Run one scan to a JPEG file.
    Scan {
        /// eSCL base URL.
        url: String,
        /// Output JPEG path.
        #[arg(short, long)]
        out: PathBuf,
        /// Resolution; must be one of the device's advertised list.
        #[arg(long, default_value_t = 300)]
        dpi: u32,
        /// Color mode: rgb, gray, or bw.
        #[arg(long, default_value = "rgb")]
        color: String,
        /// Region as fractions of the platen (opposite corners).
        #[arg(long, default_value_t = 0.0)]
        x0: f64,
        #[arg(long, default_value_t = 0.0)]
        y0: f64,
        #[arg(long, default_value_t = 1.0)]
        x1: f64,
        #[arg(long, default_value_t = 1.0)]
        y1: f64,
    },
    /// Probe the Epson push-scan channel: classify the TCP 1865 transport
    /// (plain vs TLS via the unsolicited welcome) and dump the INIT replies
    /// (INFO/CAPA). Read-only — no scan, no PARA. See
    /// scanner-push-scan-plan.md stage 0.
    PushProbe {
        /// Printer IP address (the device announcing on UDP 2968).
        ip: String,
        /// Directory to write raw body dumps into.
        #[arg(short, long)]
        out: Option<PathBuf>,
    },
}

fn color_mode(name: &str) -> Option<ColorMode> {
    match name {
        "rgb" => Some(ColorMode::Rgb24),
        "gray" => Some(ColorMode::Grayscale8),
        "bw" => Some(ColorMode::BlackAndWhite1),
        _ => None,
    }
}

fn main() {
    match Args::parse().command {
        Command::Discover { seconds } => {
            discover::ensure_daemon();
            let deadline = Instant::now() + Duration::from_secs(seconds);
            let mut seen = 0_usize;
            while Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(500));
                let map = SCANNERS.lock().expect("scanner registry lock");
                for device in map.values() {
                    println!("{}  {}", device.key, device.name);
                    for endpoint in &device.endpoints {
                        let preferred = device
                            .preferred_endpoint()
                            .is_some_and(|best| std::ptr::eq(best, endpoint));
                        println!(
                            "    {}{} {}",
                            if endpoint.tls { "tls " } else { "plain" },
                            if preferred { "*" } else { " " },
                            endpoint.base_url
                        );
                    }
                    seen += 1;
                }
            }
            println!("{seen} advertisements observed");
        }
        Command::Capabilities { url } => {
            let client = EsclClient::new(&url).unwrap_or_else(|error| panic!("{error}"));
            let caps = client
                .capabilities()
                .unwrap_or_else(|error| panic!("{error}"));
            println!(
                "platen: {}x{} (min {}x{}) in 1/300-inch units = {:.0}x{:.0} mm",
                caps.platen.max_width,
                caps.platen.max_height,
                caps.platen.min_width,
                caps.platen.min_height,
                units_to_mm(caps.platen.max_width as f64),
                units_to_mm(caps.platen.max_height as f64),
            );
            println!("max scan regions: {}", caps.max_scan_regions);
            println!("color modes: {}", caps.color_modes.join(", "));
            println!("document formats: {}", caps.document_formats.join(", "));
            println!(
                "resolutions: {}",
                caps.resolutions
                    .iter()
                    .map(|r| r.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
        Command::Scan {
            url,
            out,
            dpi,
            color,
            x0,
            y0,
            x1,
            y1,
        } => {
            let color_mode =
                color_mode(&color).unwrap_or_else(|| panic!("unknown color mode {color}"));
            let client = EsclClient::new(&url).unwrap_or_else(|error| panic!("{error}"));
            let caps = client
                .capabilities()
                .unwrap_or_else(|error| panic!("{error}"));
            if !caps.resolutions.contains(&dpi) {
                panic!(
                    "{} dpi not advertised; use one of {:?}",
                    dpi, caps.resolutions
                );
            }
            let region = region_from_fractions(&caps.platen, x0, y0, x1, y1);
            println!(
                "region: {}x{} at ({},{}) = {:.0}x{:.0} mm",
                region.width,
                region.height,
                region.x_offset,
                region.y_offset,
                units_to_mm(region.width as f64),
                units_to_mm(region.height as f64),
            );
            let settings = ScanSettings {
                region,
                color_mode,
                document_format: "image/jpeg".to_string(),
                resolution: dpi,
            };
            let started = Instant::now();
            let cancel = AtomicBool::new(false);
            let size = escl::run_scan_job(
                &client,
                &settings,
                &out,
                &cancel,
                &ScanJobTimings::default(),
                &mut |phase| {
                    eprintln!("[{:>5.1}s] {phase:?}", started.elapsed().as_secs_f32());
                },
            )
            .unwrap_or_else(|error| panic!("{error}"));
            println!("{} bytes -> {}", size, out.display());
        }
        Command::PushProbe { ip, out } => {
            push_probe::run(&ip, out);
        }
    }
}

/// Push-scan dialect probe: plain-TCP first (the ET-2810 class in the
/// epson2paperless reference speaks plain ESC/I-2 on port 1865), classify by
/// the unsolicited welcome, then run LOCK → INIT1 → INIT2 → UNLOCK with full
/// traffic logging. Read-only on the device: no PARA, no IMG, no scan.
mod push_probe {
    use std::io::{Read, Write};
    use std::net::{Shutdown, TcpStream};
    use std::path::PathBuf;
    use std::time::{Duration, Instant};

    use super::wire::{self, ParsedPacket};

    const CONNECT_BUDGET: Duration = Duration::from_secs(10);
    const WELCOME_BUDGET: Duration = Duration::from_secs(5);
    const STEP_BUDGET: Duration = Duration::from_secs(60);
    const TEARDOWN_BUDGET: Duration = Duration::from_secs(5);

    type Step<T> = Result<T, String>;

    struct Session {
        stream: TcpStream,
        buffer: Vec<u8>,
        dump_dir: Option<PathBuf>,
    }

    impl Session {
        fn connect(ip: &str, dump_dir: Option<PathBuf>) -> Step<Session> {
            let address = std::net::SocketAddr::new(
                ip.parse()
                    .map_err(|_| format!("{ip} is not an IP address"))?,
                1865,
            );
            println!("connecting to {address} (plain TCP)…");
            let stream = TcpStream::connect_timeout(&address, CONNECT_BUDGET)
                .map_err(|error| format!("connect {address}: {error}"))?;
            let _ = stream.set_nodelay(true);
            Ok(Session {
                stream,
                buffer: Vec::new(),
                dump_dir,
            })
        }

        fn send(&mut self, packet: &[u8], label: &str) -> Step<()> {
            self.stream
                .write_all(packet)
                .map_err(|error| format!("{label}: write: {error}"))?;
            let kind = u16::from_be_bytes([packet[2], packet[3]]);
            eprintln!(
                "-> {label} type=0x{kind:04x} payload={}",
                packet.len() - wire::IS_HEADER_SIZE
            );
            Ok(())
        }

        fn read_packet(&mut self, budget: Duration) -> Step<(u16, Vec<u8>)> {
            let deadline = Instant::now() + budget;
            loop {
                match wire::parse_is_packet(&self.buffer) {
                    ParsedPacket::Packet {
                        consumed,
                        kind,
                        payload,
                    } => {
                        self.buffer.drain(..consumed);
                        eprintln!(
                            "<- type=0x{kind:04x} payload={} {}",
                            payload.len(),
                            ascii_preview(&payload)
                        );
                        return Ok((kind, payload));
                    }
                    ParsedPacket::Malformed => {
                        let head = self.buffer.len().min(16);
                        return Err(format!(
                            "stream is not IS-framed (head: {:02x?})",
                            &self.buffer[..head]
                        ));
                    }
                    ParsedPacket::NeedMore => {}
                }
                let now = Instant::now();
                if now >= deadline {
                    return Err("timed out waiting for a complete IS packet".to_string());
                }
                self.stream
                    .set_read_timeout(Some(deadline - now))
                    .map_err(|error| format!("set_read_timeout: {error}"))?;
                let mut chunk = [0_u8; 4096];
                let read = self
                    .stream
                    .read(&mut chunk)
                    .map_err(|error| format!("read: {error}"))?;
                if read == 0 {
                    return Err("printer closed the connection".to_string());
                }
                self.buffer.extend_from_slice(&chunk[..read]);
            }
        }

        /// Read the next packet matching `want`, logging and skipping
        /// anything else. Empty envelopes are "wait for body" pre-data
        /// signals on the ESC/I-2 wire and are skipped unless `allow_empty`
        /// (a zero-length declared body completes on the first envelope).
        /// Probe-grade leniency: any unmatched type is skipped — the stage-2
        /// session engine must fail fast on unexpected types instead.
        fn read_matching(
            &mut self,
            budget: Duration,
            allow_empty: bool,
            want: impl Fn(u16, &[u8]) -> bool,
        ) -> Step<(u16, Vec<u8>)> {
            loop {
                let (kind, payload) = self.read_packet(budget)?;
                if payload.is_empty() && !allow_empty {
                    eprintln!("   (skipping empty 0x{kind:04x} envelope)");
                    continue;
                }
                if want(kind, &payload) {
                    return Ok((kind, payload));
                }
                eprintln!("   (skipping 0x{kind:04x} len={})", payload.len());
            }
        }

        fn read_passthru_reply(&mut self, allow_empty: bool) -> Step<Vec<u8>> {
            let (_, payload) = self.read_matching(STEP_BUDGET, allow_empty, |kind, _| {
                kind == wire::TYPE_PASSTHRU_REPLY
            })?;
            Ok(payload)
        }

        /// FS Y/Z/X: passthru with a 1-byte expected reply carrying 0x06.
        fn fs_command(&mut self, bytes: &[u8], label: &str) -> Step<()> {
            self.send(
                &wire::build_passthru_packet(bytes, wire::LEGACY_REPLY_SIZE),
                label,
            )?;
            let payload = self.read_passthru_reply(false)?;
            if payload.first() != Some(&wire::ACK_BYTE) {
                return Err(format!("{label}: expected ACK 0x06, got {payload:02x?}"));
            }
            Ok(())
        }

        /// ESC/I-2 command + two-phase read: a 64-byte META frame declares
        /// the body length, then a pure-read pulls exactly that many bytes.
        fn esci2_command(&mut self, name: &str) -> Step<(wire::Esci2Reply, Option<Vec<u8>>)> {
            self.send(
                &wire::build_passthru_packet(
                    &wire::build_esci2_command(name),
                    wire::ESCI2_REPLY_SIZE,
                ),
                name,
            )?;
            let meta = self.read_passthru_reply(false)?;
            if meta.len() != wire::ESCI2_REPLY_SIZE as usize {
                return Err(format!(
                    "{name}: META frame is {} bytes, want {}",
                    meta.len(),
                    wire::ESCI2_REPLY_SIZE
                ));
            }
            let reply = wire::parse_esci2_reply(&meta)
                .ok_or_else(|| format!("{name}: unparseable META frame {meta:02x?}"))?;
            if reply.cmd != name {
                return Err(format!("{name}: META echo says {:?}", reply.cmd));
            }
            eprintln!(
                "   {name}: declared body {} bytes, tokens {:?}",
                reply.length, reply.tokens
            );
            if let Some(dir) = &self.dump_dir {
                let _ = std::fs::write(dir.join(format!("{name}.meta.bin")), &meta);
            }
            let body = if reply.length > 0 {
                self.send(
                    &wire::build_pureread_packet(reply.length),
                    &format!("{name}:pureread"),
                )?;
                let data = self.read_passthru_reply(false)?;
                if data.len() != reply.length as usize {
                    return Err(format!(
                        "{name}: body is {} bytes, META declared {}",
                        data.len(),
                        reply.length
                    ));
                }
                Some(data)
            } else {
                self.send(&wire::build_pureread_packet(0), &format!("{name}:pureread"))?;
                let data = self.read_passthru_reply(true)?;
                if data.is_empty() { None } else { Some(data) }
            };
            Ok((reply, body))
        }

        /// FIN is a single-envelope exchange in the reference (a 64-byte
        /// reply, no two-phase body, no ACK byte) — the next command follows
        /// immediately, no pure-read.
        fn fin_command(&mut self) -> Step<()> {
            self.send(
                &wire::build_passthru_packet(
                    &wire::build_esci2_command("FIN"),
                    wire::ESCI2_REPLY_SIZE,
                ),
                "FIN",
            )?;
            let reply = self.read_passthru_reply(false)?;
            if let Some(parsed) = wire::parse_esci2_reply(&reply) {
                eprintln!(
                    "   FIN: declared body {} bytes, tokens {:?}",
                    parsed.length, parsed.tokens
                );
            }
            Ok(())
        }

        /// Close politely: unread receive-buffer bytes make the kernel send
        /// RST instead of FIN on close, so drain briefly and half-close
        /// write-only. Never a full shutdown mid-conversation.
        fn polite_close(&mut self) {
            let _ = self
                .stream
                .set_read_timeout(Some(Duration::from_millis(200)));
            let mut sink = [0_u8; 4096];
            for _ in 0..4 {
                match self.stream.read(&mut sink) {
                    Ok(0) | Err(_) => break,
                    Ok(_) => continue,
                }
            }
            let _ = self.stream.shutdown(Shutdown::Write);
        }
    }

    fn ascii_preview(payload: &[u8]) -> String {
        let text: String = payload
            .iter()
            .map(|&b| {
                if (32..127).contains(&b) {
                    b as char
                } else {
                    '.'
                }
            })
            .collect();
        format!("asc:{text}")
    }

    fn report_body(label: &str, body: Option<&[u8]>, out: Option<&PathBuf>) {
        match body {
            Some(bytes) => {
                println!("{label}: {} bytes", bytes.len());
                println!("  {}", String::from_utf8_lossy(bytes).trim_end());
                if let Some(dir) = out {
                    let path = dir.join(format!("{label}.bin"));
                    match std::fs::write(&path, bytes) {
                        Ok(()) => println!("  saved {}", path.display()),
                        Err(error) => eprintln!("  save {}: {error}", path.display()),
                    }
                }
            }
            None => println!("{label}: (no body)"),
        }
    }

    fn run_inner(ip: &str, out: Option<&PathBuf>) -> Step<String> {
        if let Some(dir) = out {
            std::fs::create_dir_all(dir)
                .map_err(|error| format!("create out dir {}: {error}", dir.display()))?;
        }
        let mut session = Session::connect(ip, out.cloned())?;

        // Transport classification: a plain-TCP ESC/I-2 device sends an
        // unsolicited welcome on connect. Nothing within 5 s means the
        // device wants TLS first (esci2 over TLS).
        let transport = match session.read_packet(WELCOME_BUDGET) {
            Ok((wire::TYPE_WELCOME, payload)) => {
                // Discriminator is payload BYTE 1 (frame offset 13): both
                // real welcomes start with 0x01 — `01 04 00 00 00` is
                // esci2-plain, `01 02 00 00 00` is legacy ESC/I. Caveat: the
                // ET-7700 (PID 112B) speaks ESC/I-2 with a legacy-shaped
                // welcome, so a legacy classification alone is not proof.
                if payload.len() < 2 {
                    return Err(format!("welcome too short to classify: {payload:02x?}"));
                }
                let legacy = payload[1] == 0x02;
                let class = if legacy {
                    "legacy ESC/I"
                } else {
                    "esci2-plain"
                };
                let head = payload.len().min(8);
                println!(
                    "welcome: {} bytes {:02x?} → classified {class}",
                    payload.len(),
                    &payload[..head]
                );
                if let Some(dir) = out {
                    let _ = std::fs::write(dir.join("welcome.bin"), &payload);
                }
                if legacy {
                    return Err(
                        "legacy ESC/I welcome — this probe drives the ESC/I-2 sequence \
                         only and stops before LOCK (the reference routes legacy \
                         devices to a separate command graph)"
                            .to_string(),
                    );
                }
                class.to_string()
            }
            Ok((kind, _)) => {
                return Err(format!("expected welcome 0x8000, got 0x{kind:04x}"));
            }
            Err(error) => {
                return Err(format!(
                    "no unsolicited welcome on plain TCP 1865 ({error}) — the device \
                     likely speaks ESC/I-2 over TLS; this probe does not do TLS yet"
                ));
            }
        };

        let result = probe_steps(&mut session);
        // Polite teardown in every case: never leave the printer locked for
        // its 300 s session timeout. (Interrupting the probe with Ctrl-C
        // after LOCK skips this — expect a ~5-minute device lockout then.)
        if session.send(&wire::build_unlock_packet(), "UNLOCK").is_ok() {
            match session.read_matching(TEARDOWN_BUDGET, true, |kind, _| {
                kind == wire::TYPE_UNLOCK_ACK
            }) {
                Ok((_, payload)) => {
                    if payload.first() != Some(&wire::ACK_BYTE) {
                        eprintln!("   unlock ack unexpected payload: {payload:02x?}");
                    } else {
                        println!("unlock acknowledged");
                    }
                }
                Err(error) => eprintln!("   unlock ack not received: {error}"),
            }
        }
        session.polite_close();
        let summary = result?;
        Ok(format!("transport: {transport}; probe complete: {summary}"))
    }

    fn probe_steps(session: &mut Session) -> Step<String> {
        session.send(&wire::build_lock_packet(), "LOCK")?;
        let (_, payload) =
            session.read_matching(STEP_BUDGET, false, |kind, _| kind == wire::TYPE_LOCK_ACK)?;
        if payload.first() != Some(&wire::ACK_BYTE) {
            return Err(format!("lock ack: payload {payload:02x?}"));
        }

        // INIT1: enter FS Y mode, read identity and capabilities.
        session.fs_command(&wire::FS_Y, "FS Y")?;
        let (_, info1) = session.esci2_command("INFO")?;
        let (_, capa1) = session.esci2_command("CAPA")?;
        session.fin_command()?;

        // INIT2: second mode with resolution/area capabilities.
        session.fs_command(&wire::FS_Z, "FS Z")?;
        let (_, info2) = session.esci2_command("INFO")?;
        let (_, capa2) = session.esci2_command("CAPA")?;
        let (_, resa) = session.esci2_command("RESA")?;
        session.fin_command()?;

        report_body("info-init1", info1.as_deref(), session.dump_dir.as_ref());
        report_body("capa-init1", capa1.as_deref(), session.dump_dir.as_ref());
        report_body("info-init2", info2.as_deref(), session.dump_dir.as_ref());
        report_body("capa-init2", capa2.as_deref(), session.dump_dir.as_ref());
        report_body("resa-init2", resa.as_deref(), session.dump_dir.as_ref());
        Ok("LOCK, INIT1 (INFO/CAPA/FIN), INIT2 (INFO/CAPA/RESA/FIN)".to_string())
    }

    pub(super) fn run(ip: &str, out: Option<PathBuf>) {
        match run_inner(ip, out.as_ref()) {
            Ok(summary) => println!("{summary}"),
            Err(error) => {
                eprintln!("probe failed: {error}");
                std::process::exit(1);
            }
        }
    }
}
