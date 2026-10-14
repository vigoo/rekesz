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
    /// Output profile JSON path (not needed with --verify/--dump-pair).
    #[arg(long, required_unless_present_all = ["verify", "dump_pair"])]
    out: Option<PathBuf>,
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
    /// Visual dump mode: render one pair (by file stem) corrected vs the
    /// camera JPEG into PNGs (full frame + crops) for inspection.
    #[arg(long)]
    dump_pair: Option<String>,
    /// DNG-oracle mode: build the profile's per-channel warp from
    /// OpcodeList3 of Adobe-converted DNGs in this folder (falls back to
    /// pair-measured vignette as in calibrate mode).
    #[arg(long)]
    from_dng: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    if args.from_dng {
        return from_dng(&args);
    }
    if let Some(stem) = &args.dump_pair {
        let profile_path = args
            .verify
            .as_ref()
            .or(args.out.as_ref())
            .expect("--dump-pair requires --verify <profile> or --out <profile>");
        return dump_pair(&args, profile_path, stem);
    }
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
    let out = args.out.expect("required for calibration by clap");
    std::fs::create_dir_all(out.parent().unwrap_or(Path::new(".")))?;
    std::fs::write(&out, serde_json::to_string_pretty(&profile)?)?;
    println!("wrote {}", out.display());
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
}
fn verify(args: &Args, profile_path: &Path) -> anyhow::Result<()> {
    let profile: lens::LensProfile = serde_json::from_str(&std::fs::read_to_string(profile_path)?)
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
        let Some(entry) = profile.select(&pair.model, Some(pair.focal_mm)) else {
            eprintln!("skipping {name}: no profile entry for {}", pair.model);
            skipped += 1;
            continue;
        };
        let mut corrected = pair.linear;
        lens::apply_geometry(&mut corrected, pair.width, pair.height, &entry);
        lens::apply_vignette(&mut corrected, pair.width, pair.height, &entry.vignette);
        let stats =
            lens::calibrate::verify_pair(&corrected, &pair.jpeg_linear, pair.width, pair.height);
        rows.push(Row {
            name,
            focal_mm: pair.focal_mm,
            stats,
        });
    }

    // Aggregate per focal length (medians).
    rows.sort_by(|a, b| a.focal_mm.total_cmp(&b.focal_mm).then(a.name.cmp(&b.name)));
    println!(
        "\n{:>9} {:>5} {:>8} {:>8} {:>9} {:>8} {:>7} {:>7} {:>12}",
        "focal", "n", "center", "mid", "periphery", "signed", "ch_r", "ch_b", "shift"
    );
    for group in rows.chunk_by(|a, b| (a.focal_mm * 10.0) as u32 == (b.focal_mm * 10.0) as u32) {
        let pick = |values: &mut Vec<f32>| {
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            values[values.len() / 2]
        };
        let mut center: Vec<f32> = group.iter().map(|row| row.stats.center_residual).collect();
        let mut mid: Vec<f32> = group.iter().map(|row| row.stats.mid_residual).collect();
        let mut periphery: Vec<f32> = group
            .iter()
            .map(|row| row.stats.periphery_residual)
            .collect();
        let mut signed: Vec<f32> = group.iter().map(|row| row.stats.periphery_signed).collect();
        let mut channel_r: Vec<f32> = group.iter().map(|row| row.stats.channel_r).collect();
        let mut channel_b: Vec<f32> = group.iter().map(|row| row.stats.channel_b).collect();
        let mut shift: Vec<f32> = group.iter().map(|row| row.stats.shift_px).collect();
        println!(
            "{:>8.1}mm {:>5} {:>8.4} {:>8.4} {:>9.4} {:+>8.4} {:>7.3} {:>7.3} {:>4.1}px rad {:+>5.1}",
            group[0].focal_mm,
            group.len(),
            pick(&mut center),
            pick(&mut mid),
            pick(&mut periphery),
            pick(&mut signed),
            pick(&mut channel_r),
            pick(&mut channel_b),
            pick(&mut shift),
            {
                let mut radial: Vec<f32> = group.iter().map(|row| row.stats.shift_radial).collect();
                pick(&mut radial)
            },
        );
    }

    // Worst pairs by periphery residual.
    let mut worst: Vec<&Row> = rows.iter().collect();
    worst.sort_by(|a, b| {
        b.stats
            .periphery_residual
            .total_cmp(&a.stats.periphery_residual)
    });
    println!("\nworst pairs (periphery residual):");
    for row in worst.iter().take(8) {
        println!(
            "  {} ({:.1}mm): periphery {:.4} (signed {:+.4}) ch_r {:.3} ch_b {:.3} shift {:.1}px radial {:+.1}",
            row.name,
            row.focal_mm,
            row.stats.periphery_residual,
            row.stats.periphery_signed,
            row.stats.channel_r,
            row.stats.channel_b,
            row.stats.shift_px,
            row.stats.shift_radial,
        );
    }
    if skipped > 0 {
        println!("(skipped {skipped} pairs)");
    }
    Ok(())
}

fn dump_pair(args: &Args, profile_path: &Path, stem: &str) -> anyhow::Result<()> {
    let profile: lens::LensProfile = serde_json::from_str(&std::fs::read_to_string(profile_path)?)
        .context("parsing the lens profile")?;
    let pairs = discover_pairs(&args.dir)?;
    let (raw_path, jpeg_path) = pairs
        .into_iter()
        .find(|(raw, _)| {
            raw.file_stem()
                .is_some_and(|candidate| candidate.to_string_lossy() == stem)
        })
        .with_context(|| format!("no pair for stem {stem}"))?;
    let pair = decode_pair(&raw_path, &jpeg_path)?;
    let entry = profile
        .select(&pair.model, Some(pair.focal_mm))
        .context("no profile entry")?;
    println!(
        "stem {stem}: focal {:.1}mm zoom {:.4} k1 {:+.4} k2 {:+.4} corner {:.2}",
        pair.focal_mm,
        entry.zoom,
        entry.k1,
        entry.k2,
        entry.vignette.last().unwrap_or(&1.0)
    );
    let mut corrected = pair.linear.clone();
    lens::apply_geometry(&mut corrected, pair.width, pair.height, &entry);
    lens::apply_vignette(&mut corrected, pair.width, pair.height, &entry.vignette);

    let out_dir = args.dir.join(format!("lens-dump-{stem}"));
    std::fs::create_dir_all(&out_dir)?;
    write_mono_png(
        &out_dir.join("corrected.png"),
        &corrected,
        pair.width,
        pair.height,
    )?;
    write_mono_png(
        &out_dir.join("jpeg.png"),
        &pair.jpeg_linear,
        pair.width,
        pair.height,
    )?;
    // Difference map, amplified: |corrected − jpeg| per pixel, ×8, clipped.
    let mut difference = vec![[0.0_f32; 3]; corrected.len()];
    for (index, slot) in difference.iter_mut().enumerate() {
        for channel in 0..3 {
            slot[channel] = ((corrected[index][channel] - pair.jpeg_linear[index][channel]).abs()
                * 8.0)
                .min(1.0);
        }
    }
    write_mono_png(
        &out_dir.join("difference-x8.png"),
        &difference,
        pair.width,
        pair.height,
    )?;
    // Crops: center, mid-radius (right), corner (top-right) — corrected on
    // top of jpeg for direct comparison, 480×480 each.
    for (name, cx, cy) in [
        ("center", pair.width / 2, pair.height / 2),
        ("mid-right", pair.width * 3 / 4, pair.height / 2),
        ("corner-tr", pair.width - 300, 300),
    ] {
        for (label, buffer) in [("corrected", &corrected), ("jpeg", &pair.jpeg_linear)] {
            write_crop_png(
                &out_dir.join(format!("crop-{name}-{label}.png")),
                buffer,
                pair.width,
                pair.height,
                cx,
                cy,
                300,
            )?;
        }
    }
    println!("wrote {}", out_dir.display());
    Ok(())
}

fn write_mono_png(path: &Path, pixels: &[[f32; 3]], width: u32, height: u32) -> anyhow::Result<()> {
    // Scene-linear → simple gamma for display.
    let mut image = image::RgbImage::new(width, height);
    for (target, pixel) in image.pixels_mut().zip(pixels) {
        for channel in 0..3 {
            target[channel] = (pixel[channel].max(0.0).sqrt().min(1.0) * 255.0) as u8;
        }
    }
    image.save(path)?;
    Ok(())
}

fn write_crop_png(
    path: &Path,
    pixels: &[[f32; 3]],
    width: u32,
    height: u32,
    cx: u32,
    cy: u32,
    half: u32,
) -> anyhow::Result<()> {
    let size = half * 2;
    let mut image = image::RgbImage::new(size, size);
    for (index, target) in image.pixels_mut().enumerate() {
        let x = index as u32 % size;
        let y = index as u32 / size;
        let sx = (cx + x).saturating_sub(half).min(width - 1);
        let sy = (cy + y).saturating_sub(half).min(height - 1);
        let pixel = pixels[(sy * width + sx) as usize];
        for channel in 0..3 {
            target[channel] = (pixel[channel].max(0.0).sqrt().min(1.0) * 255.0) as u8;
        }
    }
    image.save(path)?;
    Ok(())
}

/// DNG-oracle calibration: the per-channel warp comes from Adobe's
/// OpcodeList3 (Adobe decodes the camera's proprietary correction block),
/// the vignette LUT stays pair-measured (the DNG carries no vignette
/// opcodes for this camera), binned by the corrected radius implied by
/// the warp planes.
fn from_dng(args: &Args) -> anyhow::Result<()> {
    let mut dngs: Vec<PathBuf> = Vec::new();
    for entry in std::fs::read_dir(&args.dir)? {
        let path = entry?.path();
        if path.is_dir() {
            if let Ok(children) = std::fs::read_dir(&path) {
                dngs.extend(
                    children
                        .filter_map(|child| Some(child.ok()?.path()))
                        .filter(|path| {
                            path.extension()
                                .is_some_and(|extension| extension.eq_ignore_ascii_case("dng"))
                        }),
                );
            }
        } else if path
            .extension()
            .is_some_and(|extension| extension.eq_ignore_ascii_case("dng"))
        {
            dngs.push(path);
        }
    }
    if dngs.is_empty() {
        bail!("no DNG files in {}", args.dir.display());
    }

    struct Shot {
        focal_mm: f32,
        model: String,
        planes: [[f64; 4]; 3],
        curve: Vec<f32>,
    }

    let mut shots: Vec<Shot> = Vec::new();
    for dng in &dngs {
        let planes =
            parse_warp_rectilinear(dng).context(format!("parsing opcodes of {}", dng.display()))?;
        let stem = dng
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned();
        println!("processing {stem}");
        // Companions may sit next to the DNG or in the copied raw folders.
        let raw_path = ["CR3"]
            .iter()
            .find_map(|extension| {
                let candidate = dng.with_extension(extension);
                candidate.is_file().then_some(candidate)
            })
            .or_else(|| {
                ["cr3", "cr3-manual"].iter().find_map(|sub| {
                    let candidate = args.dir.join(sub).join(format!("{stem}.CR3"));
                    candidate.is_file().then_some(candidate)
                })
            });
        let Some(raw_path) = raw_path else {
            // Duplicates from repeated conversions carry a _1 suffix and
            // have no pair; they only duplicate an existing shot's data.
            eprintln!("skipping {stem}: no companion CR3");
            continue;
        };
        let jpeg_path = ["jpg", "jpeg", "JPG", "JPEG"]
            .iter()
            .find_map(|extension| {
                let sibling = dng.with_extension(extension);
                sibling.is_file().then_some(sibling)
            })
            .or_else(|| {
                ["cr3", "cr3-manual"].iter().find_map(|sub| {
                    ["jpg", "jpeg", "JPG", "JPEG"].iter().find_map(|extension| {
                        let candidate = args.dir.join(sub).join(format!("{stem}.{extension}"));
                        candidate.is_file().then_some(candidate)
                    })
                })
            })
            .context("no companion JPEG")?;
        let focal_mm = exiftool_focal(&raw_path)?;
        let model = exiftool_model(&raw_path)?;
        // Stream: decode one pair, extract the 33-bin vignette curve with
        // this shot's own warp, drop the megabyte-scale buffers.
        let pair = decode_pair(&raw_path, &jpeg_path)?;
        // The warp maps output→raw (r_src = r_out·f(r_out)); a raw pixel
        // at t lands at the output radius solving r_out·f(r_out) = t —
        // iterate to it (the camera's corner magnification makes the
        // first-order t/f(t) notably wrong at the wide end).
        let f_g = move |t: f32| {
            let [a, b, c, d] = planes[1];
            (a as f32)
                + (b as f32) * t * t
                + (c as f32) * t * t * t * t
                + (d as f32) * t * t * t * t * t * t
        };
        let corrected_radius = move |t: f32| {
            let mut r_out = t;
            for _ in 0..5 {
                r_out = t / f_g(r_out);
            }
            r_out
        };
        let curve = lens::calibrate::pair_vignette_curve(
            &lens::calibrate::Pair {
                raw: &pair.linear,
                jpeg: &pair.jpeg_linear,
                width: pair.width,
                height: pair.height,
            },
            corrected_radius,
        );
        shots.push(Shot {
            focal_mm,
            model,
            planes,
            curve,
        });
    }
    println!("decoded {} DNG shots", shots.len());

    shots.sort_by(|a, b| a.focal_mm.total_cmp(&b.focal_mm));
    let mut entries: Vec<lens::FocalEntry> = Vec::new();
    for group in shots.chunk_by(|a, b| (a.focal_mm * 10.0).round() == (b.focal_mm * 10.0).round()) {
        let focal_mm = group[0].focal_mm;
        // Median warp planes across the group.
        let mut planes: [[Vec<f32>; 4]; 3] = Default::default();
        for shot in group {
            for (plane_values, shot_planes) in planes.iter_mut().zip(&shot.planes) {
                for (slot, value) in plane_values.iter_mut().zip(shot_planes) {
                    slot.push(*value as f32);
                }
            }
        }
        let warp: [[f32; 4]; 3] =
            planes.map(|plane| plane.map(|mut values| median_of(&mut values).unwrap_or(0.0)));

        // Vignette LUT, measured on the pairs, binned by the corrected
        // radius implied by the warp (each shot's curve was binned by its
        // own iterative inverse — see above).
        let curves: Vec<Vec<f32>> = group.iter().map(|shot| shot.curve.clone()).collect();
        let vignette = lens::calibrate::vignette_gain_from_curves(curves);
        let mut vignette_lut = vec![1.0; lens::VIGNETTE_LUT_LEN];
        for (index, value) in vignette
            .into_iter()
            .take(lens::VIGNETTE_LUT_LEN)
            .enumerate()
        {
            vignette_lut[index] = value;
        }

        println!(
            "{focal_mm:>5.1} mm · {} shots · warp G [{:.4} {:.4} {:.4} {:.4}] · corner gain {:.2}",
            group.len(),
            warp[1][0],
            warp[1][1],
            warp[1][2],
            warp[1][3],
            vignette_lut.last().unwrap_or(&1.0),
        );
        entries.push(lens::FocalEntry {
            focal_mm,
            vignette: vignette_lut,
            warp,
            radial_map: Vec::new(),
            zoom: 1.0,
            k1: 0.0,
            k2: 0.0,
            scale_r: 1.0,
            scale_b: 1.0,
        });
    }

    // exiftool reports "Canon PowerShot V1"; the runtime matches against
    // rawler's clean_model, which drops the make.
    let model = shots[0].model.trim_start_matches("Canon ").to_owned();
    let profile = lens::LensProfile { model, entries };
    let out = args.out.clone().expect("required for calibration by clap");
    std::fs::create_dir_all(out.parent().unwrap_or(Path::new(".")))?;
    std::fs::write(&out, serde_json::to_string_pretty(&profile)?)?;
    println!("wrote {}", out.display());
    Ok(())
}

fn median_of(values: &mut [f32]) -> Option<f32> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Some(values[values.len() / 2])
}

fn exiftool_focal(path: &Path) -> anyhow::Result<f32> {
    let output = std::process::Command::new("exiftool")
        .args(["-s3", "-FocalLength", path.to_str().context("path")?])
        .output()?;
    let text = String::from_utf8_lossy(&output.stdout);
    let value = text.trim().trim_end_matches(" mm");
    value.parse().context("parsing focal length")
}

fn exiftool_model(path: &Path) -> anyhow::Result<String> {
    let output = std::process::Command::new("exiftool")
        .args(["-s3", "-Model", path.to_str().context("path")?])
        .output()?;
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

/// Parses `WarpRectilinear` from OpcodeList3 (via exiftool's raw bytes):
/// big-endian u32 count, then per opcode u32 id, u32 version, u32 flags,
/// u32 param size, then params: u32 planes, planes × 6×f64
/// (kr0..kr3, kt0, kt1), f64×2 center.
fn parse_warp_rectilinear(dng: &Path) -> anyhow::Result<[[f64; 4]; 3]> {
    let output = std::process::Command::new("exiftool")
        .args(["-b", "-OpcodeList3", dng.to_str().context("path")?])
        .output()?;
    let data = output.stdout;
    if data.len() < 20 {
        bail!("no OpcodeList3 in {}", dng.display());
    }
    let (count,) = struct_unpack_u32_be(&data, 0);
    let mut pos = 4_usize;
    for _ in 0..count {
        let (id, _version, _flags, psize) = struct_unpack_4(&data, pos);
        pos += 16;
        let psize = psize as usize;
        let params = &data[pos..pos + psize];
        if id == 1 {
            let (planes,) = struct_unpack_u32_be(params, 0);
            let mut out = [[0.0_f64; 4]; 3];
            for (plane, slot) in out.iter_mut().enumerate().take(planes as usize) {
                let base = 4 + plane * 48;
                for (coefficient, slot_value) in slot.iter_mut().enumerate() {
                    let (value,) = struct_unpack_f64_be(params, base + coefficient * 8);
                    *slot_value = value;
                }
            }
            return Ok(out);
        }
        pos += psize;
    }
    bail!("no WarpRectilinear opcode found")
}

fn struct_unpack_u32_be(data: &[u8], at: usize) -> (u32,) {
    (u32::from_be_bytes(data[at..at + 4].try_into().unwrap()),)
}

fn struct_unpack_4(data: &[u8], at: usize) -> (u32, u32, u32, u32) {
    (
        u32::from_be_bytes(data[at..at + 4].try_into().unwrap()),
        u32::from_be_bytes(data[at + 4..at + 8].try_into().unwrap()),
        u32::from_be_bytes(data[at + 8..at + 12].try_into().unwrap()),
        u32::from_be_bytes(data[at + 12..at + 16].try_into().unwrap()),
    )
}

fn struct_unpack_f64_be(data: &[u8], at: usize) -> (f64,) {
    (f64::from_be_bytes(data[at..at + 8].try_into().unwrap()),)
}
