//! ESC/I-2 capability layer: INFO/CAPA token parsing, scan-window math, and
//! PARA composition for the L3160 dialect. Ports of epson2paperless's
//! `capabilities.ts`, `scan-window.ts`, `resolution.ts`, and
//! `para-composer.ts` (MIT), restricted to what a flatbed-only JPEG push
//! scan needs.
//!
//! Dialect provenance: the L3160's CAPA fingerprint (de76c930…) maps to the
//! reference's ET-2750/XP-4100 entry — its STRUCTURE is pinned from there
//! (window 2477×3500 @ 300, two init polls). The COLOUR pair deliberately
//! diverges: we send the ET-4800's captured Epson-look curve + matrix
//! (wire-valid here — the CAPA lists UG18 and UM08) so the device bakes in
//! the same colour processing an eSCL scan carries; the reference's neutral
//! pair leaves colour to a host-side pipeline Rekesz does not have. Flip to
//! the neutral pair if live scans disagree with eSCL colour.

use super::colour::{CMX_ET4800_UM08, GAMMA_DS575W_MONO, GAMMA_ET4800_STOCK};
use crate::trace;

/// A scan window in device pixels: offset and size at the wire DPI.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Extents {
    pub x0: i32,
    pub y0: i32,
    pub w: i32,
    pub h: i32,
}

/// Parsed INFO reply tokens (only what a flatbed push scan consumes).
#[derive(Clone, Debug, Default)]
pub struct InfoTokens {
    pub pid: Option<String>,
    pub firmware: Option<String>,
    /// `#FB AREA` value, e.g. "d850i0001170" (width, height in 1/100 inch).
    pub fb_area: Option<String>,
    /// `#FB ALGN`: "LEFT", "RIGT" or "CNTR".
    pub fb_align: Option<String>,
    /// Raw `#`-segments in encounter order (diagnostics).
    pub segments: Vec<String>,
}

/// Parsed CAPA reply tokens (only what a flatbed push scan consumes).
#[derive(Clone, Debug, Default)]
pub struct CapaTokens {
    pub gmm_list: Option<String>,
    pub cmx_list: Option<String>,
    pub col_list: Option<String>,
    pub fmt_list: Option<String>,
    pub rsm_list: Option<Vec<u32>>,
    pub rss_list: Option<Vec<u32>>,
    pub fb_rsms_list: Option<Vec<u32>>,
    pub jpg_range: Option<(u32, u32)>,
    /// Raw `#`-segments in encounter order (diagnostics).
    pub segments: Vec<String>,
}

/// Split a reply body into its `#`-prefixed segments. `#FB ` carries a
/// literal trailing space in the prefix; segments are the text between `#`s.
pub fn split_hash_segments(body: &[u8]) -> Vec<String> {
    let text = String::from_utf8_lossy(body).into_owned();
    let mut segments = Vec::new();
    let bytes = text.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] != b'#' {
            index += 1;
            continue;
        }
        let end = bytes[index + 1..]
            .iter()
            .position(|&byte| byte == b'#')
            .map(|offset| index + 1 + offset)
            .unwrap_or(bytes.len());
        // Each segment keeps its '#' and runs to the next '#' or the end.
        segments.push(text[index..end].to_string());
        index = end;
    }
    segments
}

/// A CAPA list's items: four characters each ("UG10UG18" → UG10, UG18),
/// trimmed of space padding.
pub fn list_items(list: &str) -> Vec<&str> {
    list.as_bytes()
        .chunks(4)
        .map(|chunk| std::str::from_utf8(chunk).unwrap_or(""))
        .map(str::trim)
        .collect()
}

/// Parse a concatenated Epson numeric list ("d050d075…i0001200"): `d` values
/// are 3-digit, `i` values 7-digit — both forms occur mixed in one list.
fn parse_numeric_list(text: &str) -> Vec<u32> {
    let bytes = text.as_bytes();
    let mut out = Vec::new();
    let mut index = 0;
    while index < bytes.len() {
        match bytes[index] {
            b'd' if index + 4 <= bytes.len()
                && bytes[index + 1..index + 4].iter().all(u8::is_ascii_digit) =>
            {
                out.push(
                    std::str::from_utf8(&bytes[index + 1..index + 4])
                        .unwrap_or("0")
                        .parse()
                        .unwrap_or(0),
                );
                index += 4;
            }
            b'i' if index + 8 <= bytes.len()
                && bytes[index + 1..index + 8].iter().all(u8::is_ascii_digit) =>
            {
                out.push(
                    std::str::from_utf8(&bytes[index + 1..index + 8])
                        .unwrap_or("0")
                        .parse()
                        .unwrap_or(0),
                );
                index += 8;
            }
            _ => index += 1,
        }
    }
    out
}

/// The raw text after `prefix` in the first matching segment (the `#FB AREA`
/// / `#FB ALGN` prefixes carry a literal trailing space), trimmed of
/// trailing whitespace.
fn text_after_prefix<'a>(segments: &'a [String], prefix: &str) -> Option<&'a str> {
    segments
        .iter()
        .find(|segment| segment.starts_with(prefix))
        .map(|segment| segment[prefix.len()..].trim_end())
}

/// A `#PRDhNNN` value: NNN hex digits give the value's length; the space-
/// padded `PID NNNN` form yields "PID 1157".
fn parse_prd_value(segment: &str) -> Option<&str> {
    let rest = segment.strip_prefix("#PRDh")?;
    let len_hex = rest.get(..3)?;
    let value = rest.get(3..)?;
    let length = usize::from_str_radix(len_hex, 16).ok()?;
    let end = value.len().min(length);
    // The value may end mid-character for a multibyte name; lossy decoding
    // below tolerates that.
    Some(value[..end].trim_end_matches([' ', '\0']))
}

pub fn parse_info_tokens(body: &[u8]) -> InfoTokens {
    let segments = split_hash_segments(body);
    let mut pid = None;
    let mut firmware = None;
    for segment in &segments {
        if let Some(prd) = parse_prd_value(segment) {
            // Canonical form: "PID NNNN" space-padded to 16.
            if let Some(id) = prd.strip_prefix("PID ") {
                pid = Some(id.trim().to_string());
            }
        }
        if let Some(rest) = segment.strip_prefix("#VERh")
            && let Some(value) = rest.get(3..)
        {
            firmware = Some(value.trim_end().to_string());
        }
    }
    InfoTokens {
        pid,
        firmware,
        fb_area: text_after_prefix(&segments, "#FB AREA").map(str::to_string),
        fb_align: text_after_prefix(&segments, "#FB ALGN").map(str::to_string),
        segments,
    }
}

pub fn parse_capa_tokens(body: &[u8]) -> CapaTokens {
    let segments = split_hash_segments(body);
    let list = |prefix: &str| -> Option<Vec<u32>> {
        text_after_prefix(&segments, prefix).map(parse_numeric_list)
    };
    let jpg_range = text_after_prefix(&segments, "#JPGRANG").map(|text| {
        let values = parse_numeric_list(text);
        (
            values.first().copied().unwrap_or(1),
            *values.get(1).unwrap_or(&100),
        )
    });
    CapaTokens {
        gmm_list: text_after_prefix(&segments, "#GMMLIST").map(str::to_string),
        cmx_list: text_after_prefix(&segments, "#CMXLIST").map(str::to_string),
        col_list: text_after_prefix(&segments, "#COLLIST").map(str::to_string),
        fmt_list: text_after_prefix(&segments, "#FMTLIST").map(str::to_string),
        rsm_list: list("#RSMLIST"),
        rss_list: list("#RSSLIST"),
        fb_rsms_list: list("#FB RSMSLIST"),
        jpg_range,
        segments,
    }
}

/// An INFO area value ("d850i0001170": width, then height) in 1/100 inch.
pub fn parse_area(text: &str) -> Option<(u32, u32)> {
    let values = parse_numeric_list(text);
    if values.len() == 2 {
        Some((values[0], values[1]))
    } else {
        None
    }
}

/// A length in 1/100 inch as whole pixels at `dpi`, floored so it never
/// overshoots the area.
pub fn to_pixels(hundredths: u32, dpi: u32) -> i32 {
    ((hundredths as u64 * dpi as u64) / 100) as i32
}

/// The wire DPIs a flatbed scan may request: `#RSMLIST` ∩ `#RSSLIST`, capped
/// by the flatbed's own `#FB RSMSLIST` maximum when it advertises one.
pub fn advertised_dpi_set(capa: &CapaTokens) -> Vec<u32> {
    let mut base: Vec<u32> = match (&capa.rsm_list, &capa.rss_list) {
        (Some(rsm), Some(rss)) => {
            let rss: std::collections::HashSet<u32> = rss.iter().copied().collect();
            rsm.iter()
                .copied()
                .filter(|dpi| rss.contains(dpi))
                .collect()
        }
        (Some(rsm), None) => rsm.clone(),
        (None, Some(rss)) => rss.clone(),
        (None, None) => match &capa.fb_rsms_list {
            Some(list) => {
                let mut sorted = list.clone();
                sorted.sort_unstable();
                return sorted;
            }
            None => return Vec::new(),
        },
    };
    if let Some(source_list) = &capa.fb_rsms_list
        && let Some(cap) = source_list.iter().max()
    {
        base.retain(|dpi| dpi <= cap);
    }
    base.sort_unstable();
    base
}

/// The spec's selection rule: exact, else the smallest advertised value above
/// the target (the host downsamples), else the maximum (target capped).
pub fn select_wire_dpi(target: u32, advertised: &[u32]) -> u32 {
    if advertised.is_empty() {
        return target;
    }
    if advertised.contains(&target) {
        return target;
    }
    let above = advertised.iter().filter(|dpi| **dpi > target).min();
    match above {
        Some(dpi) => *dpi,
        None => *advertised.iter().max().unwrap_or(&target),
    }
}

/// Reference DPI the standard profile's windows are pinned at.
pub const STANDARD_BASE_DPI: u32 = 300;
/// The PARA trailer's buffer-size constant (the reference's captured value).
const TRAILING_BSZ: &str = "#BSZi1048576";
/// JPEG quality the captured drivers send.
pub const JPEG_QUALITY: u32 = 90;

fn pad7(n: u32) -> String {
    format!("{n:07}")
}

fn pad3(n: u32) -> String {
    format!("{n:03}")
}

/// The flatbed dialect a session runs with: gamma mode, colour segments,
/// and the init-poll count. The scan window is not pinned — it comes from
/// the app's crop region over the INFO-advertised scan area.
#[derive(Clone, Copy, Debug)]
pub struct FlatbedDialect {
    pub gmm: &'static str,
    pub init_polls: u32,
    pub gamma: &'static [u8],
    pub cmx: &'static [u8],
}

impl FlatbedDialect {
    /// The L3160 dialect: structure pinned from the fingerprint-matched
    /// ET-2750/XP-4100 entry (two init polls), colour pair deliberately the
    /// ET-4800's Epson-look capture (see module docs).
    pub fn l3160() -> FlatbedDialect {
        FlatbedDialect {
            gmm: "UG18",
            init_polls: 2,
            gamma: &GAMMA_ET4800_STOCK,
            cmx: &CMX_ET4800_UM08,
        }
    }

    /// The ET-4800 dialect as captured — the replay fixture's session used
    /// exactly this (three init polls).
    #[cfg(test)]
    pub fn et4800() -> FlatbedDialect {
        FlatbedDialect {
            gmm: "UG18",
            init_polls: 3,
            gamma: &GAMMA_ET4800_STOCK,
            cmx: &CMX_ET4800_UM08,
        }
    }
}

/// Validate the pinned dialect's demands against the device's advertised
/// capability lists; a violation would be refused by the firmware at PARA
/// time, so refuse it here with a readable reason.
pub fn check_settings(
    dialect: FlatbedDialect,
    capa: &CapaTokens,
    color: ParaColorMode,
) -> Result<(), String> {
    let gmm_items = list_items(capa.gmm_list.as_deref().unwrap_or(""));
    if gmm_items.is_empty() {
        return Err("the CAPA reply lists no gamma modes".to_string());
    }
    if !gmm_items.contains(&dialect.gmm) {
        return Err(format!(
            "the device does not list gamma mode {}",
            dialect.gmm
        ));
    }
    let formats = capa
        .fmt_list
        .as_deref()
        .map(|list| list.split_whitespace().collect::<Vec<_>>())
        .unwrap_or_default();
    if !formats.contains(&"JPG") {
        return Err("the device does not list the JPG output format".to_string());
    }
    // The colour code the mode sends must be in the device's #COLLIST.
    let col_items = list_items(capa.col_list.as_deref().unwrap_or(""));
    if !col_items.contains(&col_code(color)) {
        return Err(format!(
            "the device does not list the {} colour mode",
            col_code(color)
        ));
    }
    if let Some((min, max)) = capa.jpg_range
        && !(JPEG_QUALITY >= min && JPEG_QUALITY <= max)
    {
        return Err(format!(
            "JPEG quality {JPEG_QUALITY} is outside the device's {min}..{max} range"
        ));
    }
    Ok(())
}

/// The ESC/I-2 scan window for a crop region: fractions of the platen
/// mapped onto the scan area INFO advertises, at the wire DPI, clamped
/// inside the area. Mirrors the eSCL `region_from_fractions` rounding so
/// panel-initiated and app-initiated scans capture the same rectangle.
pub fn region_window(
    area: (u32, u32),
    region: (f64, f64, f64, f64),
    dpi: u32,
) -> Result<Extents, String> {
    let clamp01 = |value: f64| value.clamp(0.0, 1.0);
    let (lo_x, hi_x) = (
        clamp01(region.0.min(region.2)),
        clamp01(region.0.max(region.2)),
    );
    let (lo_y, hi_y) = (
        clamp01(region.1.min(region.3)),
        clamp01(region.1.max(region.3)),
    );
    let w_px = to_pixels(area.0, dpi);
    let h_px = to_pixels(area.1, dpi);
    let x0 = ((lo_x * f64::from(w_px)).round() as i32).clamp(0, w_px);
    let y0 = ((lo_y * f64::from(h_px)).round() as i32).clamp(0, h_px);
    let x1 = ((hi_x * f64::from(w_px)).round() as i32).clamp(x0, w_px);
    let y1 = ((hi_y * f64::from(h_px)).round() as i32).clamp(y0, h_px);
    if x1 - x0 < 1 || y1 - y0 < 1 {
        return Err("the crop region is too small to scan".to_string());
    }
    Ok(Extents {
        x0,
        y0,
        w: x1 - x0,
        h: y1 - y0,
    })
}

/// The scan window for a session: the crop region mapped over the flatbed
/// area the INFO reply advertises.
pub fn scan_window(
    info: &InfoTokens,
    region: (f64, f64, f64, f64),
    dpi: u32,
) -> Result<Extents, String> {
    let area_text = info
        .fb_area
        .as_deref()
        .ok_or("the INFO reply advertises no flatbed area")?;
    let area = parse_area(area_text).ok_or("the flatbed area token is not two numbers")?;
    region_window(area, region, dpi)
}

/// The colour mode a push scan runs with. Device JPEG output is 8-bit per
/// channel, so the panel's black-and-white pick maps to grayscale too (a
/// 1-bit scan cannot be JPEG-encoded).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParaColorMode {
    Color,
    Grayscale,
}

/// The `#COL` code a colour mode sends: 24-bit colour, or 8-bit grayscale —
/// both in the L3160's `#COLLIST`.
pub fn col_code(mode: ParaColorMode) -> &'static str {
    match mode {
        ParaColorMode::Color => "C024",
        ParaColorMode::Grayscale => "M008",
    }
}

/// Compose the standard-profile flatbed JPEG PARA body for `window`, `dpi`
/// and `color` — byte-for-byte the ET-4800-class layout: `#FB ` +
/// resolution/colour/format prefix + `#GMM` + tone curve + matrix + `#ACQ`
/// window + `#BSZ` trailer. Greyscale swaps `#COLC024` for `#COLM008` and
/// the RGB gamma triplet for the single-channel `#GMTMONO` curve, exactly as
/// the DS-575W's captured greyscale PARA does.
pub fn compose_para(
    dialect: FlatbedDialect,
    dpi: u32,
    window: Extents,
    color: ParaColorMode,
) -> Vec<u8> {
    let mut para = Vec::with_capacity(960);
    para.extend_from_slice(b"#FB ");
    para.extend_from_slice(format!("#RSMi{}#RSSi{}", pad7(dpi), pad7(dpi)).as_bytes());
    para.extend_from_slice(format!("#COL{}#FMTJPG ", col_code(color)).as_bytes());
    para.extend_from_slice(format!("#JPGd{}", pad3(JPEG_QUALITY)).as_bytes());
    para.extend_from_slice(format!("#GMM{}", dialect.gmm).as_bytes());
    match color {
        ParaColorMode::Color => para.extend_from_slice(dialect.gamma),
        ParaColorMode::Grayscale => para.extend_from_slice(&GAMMA_DS575W_MONO),
    }
    para.extend_from_slice(dialect.cmx);
    para.extend_from_slice(
        format!(
            "#ACQi{}i{}i{}i{}",
            pad7(window.x0.max(0) as u32),
            pad7(window.y0.max(0) as u32),
            pad7(window.w.max(0) as u32),
            pad7(window.h.max(0) as u32)
        )
        .as_bytes(),
    );
    para.extend_from_slice(TRAILING_BSZ.as_bytes());
    para
}

/// Log a device's identity + capability shape once per session (diagnostics
/// for new firmware).
pub fn log_device(info: &InfoTokens, capa: &CapaTokens) {
    trace::event(&format!(
        "push: device pid={:?} firmware={:?} area={:?} align={:?} gmm={:?} cmx={:?} fmt={:?}",
        info.pid,
        info.firmware,
        info.fb_area,
        info.fb_align,
        capa.gmm_list,
        capa.cmx_list,
        capa.fmt_list
    ));
}
