//! Manual eSCL scanner probe: discovery, capabilities, and one-off scans
//! against real hardware — the live-investigation counterpart of
//! `lens-profile-calibrate`, and the pre-UI exercise of the Stage 1 client.

#[path = "../sources/scanner/discover.rs"]
mod discover;
#[path = "../sources/scanner/escl.rs"]
mod escl;

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
    }
}
