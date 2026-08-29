use super::*;
use test_r::test;

#[test]
fn parses_fractional_xmp_capture_time() {
    assert!(parse_exif_time("2018:08:06 11:28:37.000").is_some());
    assert!(parse_exif_time("not a timestamp").is_none());
}

#[test]
fn validates_coordinates_and_converts_hemispheres() {
    assert_eq!(GeoCoordinate::new(90.0, -180.0).unwrap().latitude(), 90.0);
    assert!(GeoCoordinate::new(90.1, 0.0).is_none());
    assert!(GeoCoordinate::new(0.0, f64::NAN).is_none());
    assert_eq!(parse_xmp_angle("47,30,0N", ['N', 'S']), Some(47.5));
    assert_eq!(parse_xmp_angle("19 3 0 E", ['E', 'W']), Some(19.05));
    assert_eq!(parse_xmp_angle("33,15S", ['N', 'S']), Some(-33.25));
    assert_eq!(parse_xmp_angle("-122.5", ['E', 'W']), Some(-122.5));
    assert!(parse_xmp_angle("47,61N", ['N', 'S']).is_none());
    assert!(parse_xmp_angle("47E", ['N', 'S']).is_none());
    assert!(parse_xmp_angle("-47S", ['N', 'S']).is_none());
    for invalid in ["", "N", "NaN", "inf", "47,NaN,N"] {
        assert!(parse_xmp_angle(invalid, ['N', 'S']).is_none());
    }
}

fn embedded_gps_tiff(latitude_ref: u8, longitude_ref: u8, denominator: u32) -> Vec<u8> {
    fn short(value: u16, output: &mut Vec<u8>) {
        output.extend_from_slice(&value.to_le_bytes());
    }
    fn long(value: u32, output: &mut Vec<u8>) {
        output.extend_from_slice(&value.to_le_bytes());
    }
    fn entry(tag: u16, kind: u16, count: u32, value: u32, output: &mut Vec<u8>) {
        short(tag, output);
        short(kind, output);
        long(count, output);
        long(value, output);
    }
    fn rationals(values: [u32; 3], denominator: u32, output: &mut Vec<u8>) {
        for value in values {
            long(value, output);
            long(denominator, output);
        }
    }

    let mut bytes = b"II".to_vec();
    short(42, &mut bytes);
    long(8, &mut bytes);
    short(1, &mut bytes);
    entry(0x8825, 4, 1, 26, &mut bytes);
    long(0, &mut bytes);
    short(4, &mut bytes);
    entry(1, 2, 2, u32::from(latitude_ref), &mut bytes);
    entry(2, 5, 3, 80, &mut bytes);
    entry(3, 2, 2, u32::from(longitude_ref), &mut bytes);
    entry(4, 5, 3, 104, &mut bytes);
    long(0, &mut bytes);
    rationals([47, 30, 0], denominator, &mut bytes);
    rationals([19, 3, 0], denominator, &mut bytes);
    bytes
}

fn gps_jpeg(latitude_ref: u8, longitude_ref: u8) -> Vec<u8> {
    let tiff = embedded_gps_tiff(latitude_ref, longitude_ref, 1);
    let app1_length = u16::try_from(2 + 6 + tiff.len()).unwrap();
    let mut jpeg = vec![0xff, 0xd8, 0xff, 0xe1];
    jpeg.extend_from_slice(&app1_length.to_be_bytes());
    jpeg.extend_from_slice(b"Exif\0\0");
    jpeg.extend_from_slice(&tiff);
    jpeg.extend_from_slice(&[0xff, 0xd9]);
    jpeg
}

#[test]
fn reads_generated_embedded_exif_gps_and_hemispheres() {
    let exif = exif::Reader::new()
        .read_raw(embedded_gps_tiff(b'S', b'W', 1))
        .unwrap();
    assert_eq!(
        coordinate_from_exif(&exif),
        GeoCoordinate::new(-47.5, -19.05)
    );
}

#[test]
fn reads_gps_from_a_generated_jpeg_exif_container() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("gps.jpg");
    fs::write(&path, gps_jpeg(b'N', b'E')).unwrap();
    assert_eq!(exif_coordinate(&path), GeoCoordinate::new(47.5, 19.05));
}

#[test]
fn xmp_gps_is_authoritative_over_embedded_exif_gps() {
    let root = tempfile::tempdir().unwrap();
    let cache = tempfile::tempdir().unwrap();
    let image = root.path().join("gps.jpg");
    fs::write(&image, gps_jpeg(b'N', b'E')).unwrap();
    fs::write(
        root.path().join("gps.jpg.xmp"),
        br#"<rdf:Description exif:GPSLatitude="33,15S" exif:GPSLongitude="151,12E"/>"#,
    )
    .unwrap();
    let loader = MetadataLoader::new(cache.path().to_path_buf(), 1, JobLimiter::new(1));
    let loaded = loader.load_directory(vec![image], Arc::new(AtomicBool::new(false)));
    assert_eq!(loaded[0].coordinate, GeoCoordinate::new(-33.25, 151.2));
}

#[test]
fn rejects_incomplete_or_non_finite_embedded_exif_gps() {
    let invalid = exif::Reader::new()
        .read_raw(embedded_gps_tiff(b'N', b'E', 0))
        .unwrap();
    assert_eq!(coordinate_from_exif(&invalid), None);
    assert_eq!(dms_coordinate([47.0, 60.0, 0.0], "N", ['N', 'S']), None);
    assert_eq!(dms_coordinate([47.0, 0.0, 0.0], "E", ['N', 'S']), None);
}

#[test]
fn reads_heif_exif_payload_offsets_and_raw_rationals() {
    let tiff = embedded_gps_tiff(b'N', b'E', 1);
    let mut payload = 0_u32.to_be_bytes().to_vec();
    payload.extend_from_slice(&tiff);
    assert_eq!(
        heif_exif_coordinate(&payload),
        GeoCoordinate::new(47.5, 19.05)
    );
    use rawler::formats::tiff::Rational;
    assert_eq!(
        raw_dms(
            [
                Rational { n: 47, d: 1 },
                Rational { n: 30, d: 1 },
                Rational { n: 0, d: 1 },
            ],
            "S",
            ['N', 'S'],
        ),
        Some(-47.5)
    );
}

#[test]
fn reads_gps_from_a_generated_heif_container() {
    use libheif_rs::{
        Channel, ColorSpace, CompressionFormat, HeifContext, Image, LibHeif, RgbChroma,
    };

    let mut image = Image::new(2, 2, ColorSpace::Rgb(RgbChroma::Rgb)).unwrap();
    image.create_plane(Channel::Interleaved, 2, 2, 24).unwrap();
    image.planes_mut().interleaved.unwrap().data.fill(128);
    let lib_heif = LibHeif::new();
    let mut context = HeifContext::new().unwrap();
    let mut encoder = lib_heif.encoder_for_format(CompressionFormat::Av1).unwrap();
    let handle = context.encode_image(&image, &mut encoder, None).unwrap();
    context
        .add_exif_metadata(&handle, &embedded_gps_tiff(b'N', b'E', 1))
        .unwrap();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("gps.heic");
    fs::write(&path, context.write_to_bytes().unwrap()).unwrap();
    assert_eq!(heif_coordinate(&path), GeoCoordinate::new(47.5, 19.05));
}

#[test]
fn reads_xmp_coordinate_and_caches_it() {
    let root = tempfile::tempdir().unwrap();
    let cache = tempfile::tempdir().unwrap();
    let image = root.path().join("photo.jpg");
    let xmp = root.path().join("photo.jpg.xmp");
    fs::write(&image, b"not actually an image").unwrap();
    fs::write(
        xmp,
        br#"<rdf:Description exif:GPSLatitude="47,30,0N" exif:GPSLongitude="19,3,0E"/>"#,
    )
    .unwrap();
    let loader = MetadataLoader::new(cache.path().to_path_buf(), 1, JobLimiter::new(1));
    let loaded = loader.load_directory(vec![image.clone()], Arc::new(AtomicBool::new(false)));
    let coordinate = loaded[0].coordinate.unwrap();
    assert_eq!(coordinate, GeoCoordinate::new(47.5, 19.05).unwrap());

    let (cached, missing) =
        loader.load_cached_directory(vec![image], Arc::new(AtomicBool::new(false)));
    assert!(missing.is_empty());
    assert_eq!(cached[0].coordinate, Some(coordinate));
}

#[test]
fn caches_explicit_coordinate_absence() {
    let root = tempfile::tempdir().unwrap();
    let cache = tempfile::tempdir().unwrap();
    let image = root.path().join("no-location.jpg");
    fs::write(&image, b"not actually an image").unwrap();
    let loader = MetadataLoader::new(cache.path().to_path_buf(), 1, JobLimiter::new(1));
    let loaded = loader.load_directory(vec![image.clone()], Arc::new(AtomicBool::new(false)));
    assert_eq!(loaded[0].coordinate, None);
    let (cached, missing) =
        loader.load_cached_directory(vec![image], Arc::new(AtomicBool::new(false)));
    assert!(missing.is_empty());
    assert_eq!(cached[0].coordinate, None);
}

#[test]
fn coordinate_cache_version_does_not_reuse_pre_coordinate_rows() {
    let root = tempfile::tempdir().unwrap();
    let cache = tempfile::tempdir().unwrap();
    let legacy = cache.path().join("metadata-v1.sqlite3");
    fs::write(&legacy, b"legacy cache remains isolated").unwrap();
    let image = root.path().join("gps.jpg");
    fs::write(&image, gps_jpeg(b'N', b'E')).unwrap();
    let loader = MetadataLoader::new(cache.path().to_path_buf(), 1, JobLimiter::new(1));
    let loaded = loader.load_directory(vec![image], Arc::new(AtomicBool::new(false)));
    assert_eq!(loaded[0].coordinate, GeoCoordinate::new(47.5, 19.05));
    assert_eq!(fs::read(legacy).unwrap(), b"legacy cache remains isolated");
    assert!(cache.path().join("metadata-v3.sqlite3").is_file());
}

#[test]
fn reads_capture_time_from_the_cr3_regression_sample() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test/IMG_0052.CR3");
    if path.exists() {
        assert!(embedded_capture_time(&path).is_some());
    }
}

#[test]
#[ignore = "requires the user's external GPS-bearing CR3 sample"]
fn reads_gps_from_the_external_cr3_regression_sample() {
    let path = Path::new("/mnt/pictures/Pictures/2026/Alpok túra/2026-07-31/IMG_0736.CR3");
    let coordinate = embedded_coordinate(path).unwrap();
    assert!((coordinate.latitude() - 45.192_905_555_555_56).abs() < 0.000_01);
    assert!((coordinate.longitude() - 5.730_708_333_333_333).abs() < 0.000_01);
}

#[test]
fn xmp_is_authoritative_and_persistently_cached() {
    let root = tempfile::tempdir().unwrap();
    let cache = tempfile::tempdir().unwrap();
    let image = root.path().join("photo.nef");
    fs::write(&image, b"not actually a raw file").unwrap();
    fs::write(
        root.path().join("photo.nef.xmp"),
        br#"<rdf:Description exif:DateTimeOriginal="2018:08:06 11:28:37.000"/>"#,
    )
    .unwrap();
    let loader = MetadataLoader::new(cache.path().to_path_buf(), 1, JobLimiter::new(1));
    let result = loader.load_directory(vec![image.clone()], Arc::new(AtomicBool::new(false)));
    assert!(result[0].capture_time.is_some());
    assert!(
        cache
            .path()
            .join(format!("{CACHE_VERSION}.sqlite3"))
            .is_file()
    );

    // A cache hit must not attempt to decode the invalid source file.
    let result = loader.load_directory(vec![image.clone()], Arc::new(AtomicBool::new(false)));
    assert!(result[0].capture_time.is_some());
    let (cached, missing) =
        loader.load_cached_directory(vec![image], Arc::new(AtomicBool::new(false)));
    assert!(missing.is_empty());
    assert!(cached[0].capture_time.is_some());
}
