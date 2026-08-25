use std::{
    path::Path,
    sync::{Arc, atomic::AtomicBool},
};

use super::decode_raw;
use crate::rendering::photographic::{PhotographicSettings, render_raw};
use test_r::test;

#[test]
#[ignore = "manual investigation using external RAW samples"]
fn report_reference_raw_exposure_and_white_balance() {
    let samples = [
        "/mnt/pictures/Pictures/2020/Szlovénia/2020-07-29/DSC_0546.NEF",
        "/mnt/pictures/Pictures/2026/Alpok túra/2026-08-03/IMG_0997.CR3",
        "/mnt/pictures/Pictures/2026/Alpok túra/2026-08-03/IMG_1056.CR3",
    ];
    let cancelled = Arc::new(AtomicBool::new(false));
    for path in samples {
        let decoded = decode_raw(Path::new(path), &cancelled).unwrap();
        let (_, metadata) =
            render_raw(&decoded, &PhotographicSettings::default(), None, &cancelled).unwrap();
        eprintln!(
            "{path}: as_shot={:?} neutral={:?} auto_stops={}",
            decoded.raw.wb_coeffs,
            decoded.raw.neutralwb(),
            metadata.automatic_exposure_stops
        );
    }
}
