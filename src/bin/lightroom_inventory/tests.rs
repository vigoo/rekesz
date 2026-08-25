use super::*;
use test_r::test;

#[test]
fn parses_and_flattens_lightroom_table() {
    let value = Parser::new(
            r#"s = { Exposure2012 = 1.25, Enabled = true, Name = "Custom", Curve = { 0, 10, 255, 250, }, }"#,
        )
        .parse_assignment()
        .unwrap();
    let mut properties = BTreeMap::new();
    flatten_value("", &value, &mut properties);
    assert_eq!(properties["Exposure2012"].numeric_min, Some(1.25));
    assert_eq!(properties["Curve"].tables, 1);
    assert_eq!(properties["Curve.[2]"].numeric_min, Some(255.0));
}

#[test]
fn parses_lightroom_bracketed_string_keys() {
    let value = Parser::new(r#"s = { Group = { ["x-default"] = "Profiles" } }"#)
        .parse_assignment()
        .unwrap();
    let mut properties = BTreeMap::new();
    flatten_value("", &value, &mut properties);
    assert_eq!(properties["Group.x-default"].strings, 1);
}

#[test]
fn latest_catalog_skips_corrupt_catalogs_directory() {
    let temp = tempfile::tempdir().unwrap();
    fs::write(temp.path().join("valid.lrcat"), []).unwrap();
    let corrupt = temp.path().join("Corrupt Catalogs");
    fs::create_dir(&corrupt).unwrap();
    fs::write(corrupt.join("newer.lrcat"), []).unwrap();
    assert_eq!(
        newest_catalog(temp.path()).unwrap(),
        temp.path().join("valid.lrcat")
    );
}
