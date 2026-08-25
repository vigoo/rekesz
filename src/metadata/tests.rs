use super::*;
use test_r::test;

#[test]
fn parses_fractional_xmp_capture_time() {
    assert!(parse_exif_time("2018:08:06 11:28:37.000").is_some());
    assert!(parse_exif_time("not a timestamp").is_none());
}

#[test]
fn reads_capture_time_from_the_cr3_regression_sample() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test/IMG_0052.CR3");
    if path.exists() {
        assert!(embedded_capture_time(&path).is_some());
    }
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
