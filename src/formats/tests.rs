use super::*;
use test_r::test;

#[test]
fn raw_extensions_are_case_insensitive() {
    assert!(is_raw(Path::new("photo.ARW")));
    assert!(is_raw(Path::new("photo.dng")));
    assert!(!is_raw(Path::new("photo.jpg")));
}
