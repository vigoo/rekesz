//! Lens-profile calibration: derive a `lens::LensProfile` (peripheral
//! illumination, distortion, lateral CA) from pairs of uncorrected raw
//! renders and the camera's own corrected JPEGs.
//!
//! Usage: `lens-profile-calibrate --dir <folder with CR3+JPEG pairs>
//! --out assets/lens/<camera>.json [--model <override>] [--pairs-per-focal N]`
//!
//! The lens module is shared verbatim with the app via `#[path]`, so the
//! profile format and the applied math cannot drift.

#[path = "../rendering/photographic/lens/mod.rs"]
mod lens;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, bail};
use clap::Parser;
use rawler::imgop::develop::{ProcessingStep, RawDevelop};
use rawler::{decoders::RawDecodeParams, rawsource::RawSource};

/// Same step list as the app's photographic pipeline
/// (`rendering/photographic/rawler_stage`).
const DEVELOP_STEPS: [ProcessingStep; 7] = [
    ProcessingStep::Rescale,
    ProcessingStep::Demosaic,
    ProcessingStep::FujiRotate,
    ProcessingStep::CropActiveArea,
    ProcessingStep::WhiteBalance,
    ProcessingStep::Calibrate,
    ProcessingStep::CropDefault,
];

#[derive(Parser)]
struct Args {
    /// Folder containing CR3 files with same-stem JPEG companions.
    #[arg(long)]
    dir: PathBuf,
    /// Output profile JSON path.
    #[arg(long)]
    out: PathBuf,
    /// Override the profile's camera model key.
    #[arg(long)]
    model: Option<String>,
    /// Maximum pairs per focal length (vignette medians benefit from more).
    #[arg(long, default_value_t = 3)]
    pairs_per_focal: usize,
    /// Verification mode: instead of calibrating, apply the correction from
    /// this profile to every pair and report residuals against the camera
    /// JPEGs (one-time diagnostic; processes all pairs, streaming).
    #[arg(long)]
    verify: Option<PathBuf>,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    if let Some(profile_path) = &args.verify {
        return verify(&args, profile_path);
    }
    let pairs = discover_pairs(&args.dir)?;
    if pairs.is_empty() {
        bail!("no CR3+JPEG pairs found in {}", args.dir.display());
    }
    println!("found {} pairs", pairs.len());

    // Decode and group by focal length.
    let mut groups: BTreeMap<u32, Vec<CalibratedPair>> = BTreeMap::new();
    let mut model: Option<String> = None;
    let mut decoded = 0_usize;
    for (raw_path, jpeg_path) in &pairs {
        match decode_pair(raw_path, jpeg_path) {
            Ok(pair) => {
                decoded += 1;
                if model.is_none() {
                    model = Some(pair.model.clone());
                }
                let bucket = (pair.focal_mm * 10.0).round() as u32;
                let group = groups.entry(bucket).or_default();
                if group.len() < args.pairs_per_focal {
                    group.push(pair);
                } else {
                    println!(
                        "skipping {} (group {} mm full)",
                        raw_path.display(),
                        bucket as f32 / 10.0
                    );
                }
            }
            Err(error) => eprintln!("skipping {}: {error}", raw_path.display()),
        }
    }
    if decoded == 0 {
        bail!("no pair could be decoded");
    }
    let model = match (args.model, model) {
        (Some(overridden), _) | (None, Some(overridden)) => overridden,
        (None, None) => bail!("camera model unknown; pass --model"),
    };

    // Calibrate one entry per focal-length group.
    let mut entries = Vec::new();
    for (bucket, group) in groups {
        let focal_mm = bucket as f32 / 10.0;
        let pair_refs: Vec<lens::calibrate::Pair<'_>> = group
            .iter()
            .map(|pair| lens::calibrate::Pair {
                raw: &pair.linear,
                jpeg: &pair.jpeg_linear,
                width: pair.width,
                height: pair.height,
            })
            .collect();
        let mut entry = lens::calibrate::calibrate_entry(&pair_refs, focal_mm);
        // Geometry from the first pair of the group, with diagnostics.
        let first = &group[0];
        let fit = lens::calibrate::geometry_fit(&lens::calibrate::Pair {
            raw: &first.linear,
            jpeg: &first.jpeg_linear,
            width: first.width,
            height: first.height,
        });
        entry.k1 = fit.k1;
        entry.k2 = fit.k2;
        entry.scale_r = fit.scale_r;
        entry.scale_b = fit.scale_b;
        println!(
            "{focal_mm:>5.1} mm · {} pairs · corner gain {:.3} · k1 {:+.4} k2 {:+.4} \
             · CA r {:+.5} b {:+.5} · residual {:.4} · tangential {:.4} · {} samples to r {:.2}",
            group.len(),
            entry.vignette.last().unwrap_or(&1.0),
            entry.k1,
            entry.k2,
            entry.scale_r,
            entry.scale_b,
            fit.residual,
            fit.tangential,
            fit.samples,
            fit.max_radius,
        );
        entries.push(entry);
    }

    let profile = lens::LensProfile { model, entries };
    std::fs::create_dir_all(args.out.parent().unwrap_or(Path::new(".")))?;
    std::fs::write(&args.out, serde_json::to_string_pretty(&profile)?)?;
    println!("wrote {}", args.out.display());
    Ok(())
}

struct CalibratedPair {
    model: String,
    focal_mm: f32,
    width: u32,
    height: u32,
    linear: Vec<[f32; 3]>,
    jpeg_linear: Vec<[f32; 3]>,
}

fn discover_pairs(dir: &Path) -> anyhow::Result<Vec<(PathBuf, PathBuf)>> {
    let mut pairs = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let Some(extension) = path.extension().and_then(|e| e.to_str()) else {
            continue;
        };
        if !extension.eq_ignore_ascii_case("cr3") {
            continue;
        }
        let stem = path
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned();
        // Companion lookup is case-insensitive (cards use .JPG).
        let jpeg = std::fs::read_dir(dir)?.find_map(|sibling| {
            let sibling = sibling.ok()?.path();
            let sibling_extension = sibling.extension().and_then(|e| e.to_str())?;
            if !sibling_extension.eq_ignore_ascii_case("jpg")
                && !sibling_extension.eq_ignore_ascii_case("jpeg")
            {
                return None;
            }
            let sibling_stem = sibling
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned();
            (sibling_stem == stem).then_some(sibling)
        });
        if let Some(jpeg) = jpeg {
            pairs.push((path, jpeg));
        }
    }
    pairs.sort();
    Ok(pairs)
}

fn decode_pair(raw_path: &Path, jpeg_path: &Path) -> anyhow::Result<CalibratedPair> {
    use rayon::prelude::*;

    let raw_path = raw_path.to_path_buf();
    let jpeg_path = jpeg_path.to_path_buf();
    let raw_future = std::thread::spawn(move || -> anyhow::Result<_> {
        let source = RawSource::new(&raw_path)?;
        let decoder = rawler::global_loader().get_decoder(&source)?;
        let params = RawDecodeParams::default();
        let metadata = decoder
            .raw_metadata(&source, &params)
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        let focal_mm = metadata
            .exif
            .focal_length
            .map(|rational| rational.n as f32 / rational.d.max(1) as f32)
            .filter(|value| *value > 0.0)
            .context("no focal length")?;
        let raw = decoder
            .raw_image(&source, &params, false)
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        let model = raw.clean_model.clone();
        let develop = RawDevelop::new_with(&DEVELOP_STEPS);
        let intermediate = develop
            .develop_intermediate_cancellable(&raw, &rawler::cancel::CancellationToken::new())
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        let (width, height, linear) = match intermediate {
            rawler::imgop::develop::Intermediate::ThreeColor(pixels) => {
                (pixels.width as u32, pixels.height as u32, pixels.data)
            }
            rawler::imgop::develop::Intermediate::Monochrome(pixels) => {
                let data = pixels
                    .data
                    .into_iter()
                    .map(|value| [value, value, value])
                    .collect();
                (pixels.width as u32, pixels.height as u32, data)
            }
            rawler::imgop::develop::Intermediate::FourColor(pixels) => {
                let data = pixels
                    .data
                    .into_iter()
                    .map(|value| [value[0], value[1], value[2]])
                    .collect();
                (pixels.width as u32, pixels.height as u32, data)
            }
        };
        Ok((model, focal_mm, width, height, linear))
    });

    let jpeg_future = std::thread::spawn(move || -> anyhow::Result<_> {
        let image = image::open(&jpeg_path)?;
        let rgb = image.to_rgb8();
        let (width, height) = (rgb.width(), rgb.height());
        let linear: Vec<[f32; 3]> = rgb
            .par_chunks_exact(3)
            .map(|channel| {
                [
                    srgb_to_linear(channel[0]),
                    srgb_to_linear(channel[1]),
                    srgb_to_linear(channel[2]),
                ]
            })
            .collect();
        Ok((width, height, linear))
    });

    let (model, focal_mm, width, height, linear) = raw_future
        .join()
        .map_err(|_| anyhow::anyhow!("raw thread panicked"))??;
    let (jw, jh, jpeg_linear) = jpeg_future
        .join()
        .map_err(|_| anyhow::anyhow!("jpeg thread panicked"))??;
    if (jw, jh) != (width, height) {
        bail!("dimension mismatch: raw {width}x{height} vs jpeg {jw}x{jh}");
    }
    Ok(CalibratedPair {
        model,
        focal_mm,
        width,
        height,
        linear,
        jpeg_linear,
    })
}

fn srgb_to_linear(value: u8) -> f32 {
    let value = value as f32 / 255.0;
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}\n
fn verify(args: &Args, profile_path: &Path) -> anyhow::Result<()> {
    let profile: lens::LensProfile =
        serde_json::from_str(&std::fs::read_to_string(profile_path)?)
            .context("parsing the lens profile")?;
    let pairs = discover_pairs(&args.dir)?;
    if pairs.is_empty() {
        bail!("no CR3+JPEG pairs found in {}", args.dir.display());
    }
    println!(
        "verifying {} pairs against {}",
        pairs.len(),
        profile_path.display()
    );

    struct Row {
        name: String,
        focal_mm: f32,
        stats: lens::calibrate::VerifyStats,
    }
    let mut rows: Vec<Row> = Vec::new();
    let mut skipped = 0_usize;
    for (raw_path, jpeg_path) in &pairs {
        let name = raw_path
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned();
        let pair = match decode_pair(raw_path, jpeg_path) {
            Ok(pair) => pair,
            Err(error) => {
                eprintln!("skipping {name}: {error}");
                skipped += 1;
                continue;
            }
        };
        let Some(mut entry) = profile.select(&pair.model, Some(pair.focal_mm)) else {
            eprintln!("skipping {name}: no profile entry for {}", pair.model);
            skipped += 1;
            continue;
        };
        let mut corrected = pair.linear;
        lens::apply_geometry(&mut corrected, pair.width, pair.height, &entry);
        lens::apply_vignette(&mut corrected, pair.width, pair.height, &entry.vignette);
        let stats = lens::calibrate::verify_pair(&corrected, &pair.jpeg_linear, pair.width, pair.height);
        rows.push(Row {
            name,
            focal_mm: pair.focal_mm,
            stats,
        });
    }

    // Aggregate per focal length (medians).
    rows.sort_by(|a, b| a.focal_mm.total_cmp(&b.focal_mm).then(a.name.cmp(&b.name)));
    println!(
        "\n{:>9} {:>5} {:>8} {:>8} {:>9} {:>8} {:>7} {:>7} {:>6}",
        "focal", "n", "center", "mid", "periphery", "signed", "ch_r", "ch_b", "shift"
    );
    for group in rows.chunk_by(|a, b| (a.focal_mm * 10.0) as u32 == (b.focal_mm * 10.0) as u32) {
        let pick = |values: &mut Vec<f32>| {
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            values[values.len() / 2]
        };
        let mut center: Vec<f32> = group.iter().map(|row| row.stats.center_residual).collect();
        let mut mid: Vec<f32> = group.iter().map(|row| row.stats.mid_residual).collect();
        let mut periphery: Vec<f32> = group.iter().map(|row| row.stats.periphery_residual).collect();
        let mut signed: Vec<f32> = group.iter().map(|row| row.stats.periphery_signed).collect();
        let mut channel_r: Vec<f32> = group.iter().map(|row| row.stats.channel_r).collect();
        let mut channel_b: Vec<f32> = group.iter().map(|row| row.stats.channel_b).collect();
        let mut shift: Vec<f32> = group.iter().map(|row| row.stats.shift_px).collect();
        println!(
            "{:>8.1}mm {:>5} {:>8.4} {:>8.4} {:>9.4} {:+>8.4} {:>7.3} {:>7.3} {:>5.1}px",
            group[0].focal_mm,
            group.len(),
            pick(&mut center),
            pick(&mut mid),
            pick(&mut periphery),
            pick(&mut signed),
            pick(&mut channel_r),
            pick(&mut channel_b),
            pick(&mut shift),
        );
    }

    // Worst pairs by periphery residual.
    let mut worst: Vec<&Row> = rows.iter().collect();
    worst.sort_by(|a, b| b.stats.periphery_residual.total_cmp(&a.stats.periphery_residual));
    println!("\nworst pairs (periphery residual):");
    for row in worst.iter().take(8) {
        println!(
            "  {} ({:.1}mm): periphery {:.4} (signed {:+.4}) ch_r {:.3} ch_b {:.3} shift {:.1}px",
            row.name,
            row.focal_mm,
            row.stats.periphery_residual,
            row.stats.periphery_signed,
            row.stats.channel_r,
            row.stats.channel_b,
            row.stats.shift_px,
        );
    }
    if skipped > 0 {
        println!("(skipped {skipped} pairs)");
    }
    Ok(())
}
