use std::{
    fs::File,
    io::BufWriter,
    path::Path,
    sync::{Arc, atomic::AtomicBool},
};

use super::{decode_raw, raw_allocation_within_budget, raw_decode_admitted};
use crate::rendering::photographic::{PhotographicSettings, render_raw};
use test_r::test;

#[test]
fn synthetic_oriented_dng_decodes_through_the_real_raw_path() {
    use image::GenericImageView as _;
    use rawler::{
        dng::{DNG_VERSION_V1_4, DngCompression, writer::DngWriter},
        tags::TiffCommonTag,
    };

    let source = image::load_from_memory(include_bytes!(
        "../../../../assets/face/fixtures/synthetic-person-a-frontal.png"
    ))
    .unwrap()
    .resize_exact(256, 192, image::imageops::FilterType::Triangle)
    .to_rgb8();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("synthetic-face.dng");
    let mut dng = DngWriter::new(
        BufWriter::new(File::create(&path).unwrap()),
        DNG_VERSION_V1_4,
    )
    .unwrap();
    dng.root_ifd_mut()
        .add_tag(TiffCommonTag::Orientation, 6_u16);
    let mut raw = dng.subframe(0);
    raw.rgb_image_u8(
        source.as_raw(),
        source.width() as usize,
        source.height() as usize,
        DngCompression::Uncompressed,
        1,
    )
    .unwrap();
    raw.finalize().unwrap();
    dng.close().unwrap();

    let decoded = decode_raw(&path, &Arc::new(AtomicBool::new(false)), None).unwrap();
    assert_eq!((decoded.raw.width, decoded.raw.height), (256, 192));
    assert_eq!(decoded.orientation.map(|value| value.to_exif()), Some(6));
    let (rendered, _) = render_raw(
        &decoded,
        &PhotographicSettings::default(),
        None,
        &Arc::new(AtomicBool::new(false)),
    )
    .unwrap();
    assert_eq!(rendered.dimensions(), (192, 256));
}

#[test]
fn raw_probe_rejects_oversized_and_overflowing_allocations() {
    assert!(raw_allocation_within_budget(
        3_000,
        2_000,
        1,
        768 * 1024 * 1024
    ));
    assert!(!raw_allocation_within_budget(
        50_000,
        50_000,
        1,
        768 * 1024 * 1024
    ));
    assert!(!raw_allocation_within_budget(usize::MAX, 2, 4, usize::MAX));
    assert!(!raw_decode_admitted(
        8_000,
        6_000,
        1,
        Some(64 * 1024 * 1024)
    ));
    assert!(raw_decode_admitted(8_000, 6_000, 1, None));
}

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
        let decoded = decode_raw(Path::new(path), &cancelled, None).unwrap();
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
