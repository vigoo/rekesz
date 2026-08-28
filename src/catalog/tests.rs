use super::*;
use test_r::test;

fn lightroom_fixture(root: &Path, recipe: &str) -> tempfile::TempDir {
    let fixture = tempfile::tempdir().unwrap();
    let path = fixture.path().join("fixture.lrcat");
    let connection = Connection::open(path).unwrap();
    connection
            .execute_batch(
                "CREATE TABLE AgLibraryRootFolder(id_local INTEGER PRIMARY KEY, absolutePath TEXT);
                 CREATE TABLE AgLibraryFolder(id_local INTEGER PRIMARY KEY, rootFolder INTEGER, pathFromRoot TEXT);
                 CREATE TABLE AgLibraryFile(id_local INTEGER PRIMARY KEY, folder INTEGER, baseName TEXT, extension TEXT);
                 CREATE TABLE Adobe_images(id_local INTEGER PRIMARY KEY, rootFile INTEGER, masterImage INTEGER);
                 CREATE TABLE Adobe_imageDevelopSettings(image INTEGER, processVersion TEXT, text TEXT);",
            )
            .unwrap();
    connection
        .execute(
            "INSERT INTO AgLibraryRootFolder VALUES(1, ?1)",
            [format!("{}/", root.display())],
        )
        .unwrap();
    connection
        .execute("INSERT INTO AgLibraryFolder VALUES(1, 1, '')", [])
        .unwrap();
    connection
        .execute("INSERT INTO AgLibraryFile VALUES(1, 1, 'photo', 'CR3')", [])
        .unwrap();
    connection
        .execute("INSERT INTO Adobe_images VALUES(1, 1, NULL)", [])
        .unwrap();
    connection
        .execute(
            "INSERT INTO Adobe_imageDevelopSettings VALUES(1, '15.4', ?1)",
            [recipe],
        )
        .unwrap();
    fixture
}

#[test]
fn canonical_digest_is_stable_and_changes_with_recipe() {
    let recipe = DevelopRecipe {
        exposure_ev: Some(1.0),
        ..DevelopRecipe::default()
    };
    assert_eq!(recipe.digest().unwrap(), recipe.clone().digest().unwrap());
    let changed = DevelopRecipe {
        exposure_ev: Some(2.0),
        ..DevelopRecipe::default()
    };
    assert_ne!(recipe.digest().unwrap(), changed.digest().unwrap());
}

#[test]
fn normalization_keeps_only_supported_non_defaults() {
    let values =
        Parser::new("s={Exposure2012=1,Contrast2012=0,Unknown=7,ToneCurvePV2012={0,0,255,255}}")
            .parse_assignment()
            .unwrap();
    let (recipe, unsupported, warnings) =
        normalize_recipe("15.4", &values, ImportCapabilities::ALL);
    assert_eq!(recipe.exposure_ev, Some(1.0));
    assert_eq!(recipe.contrast, None);
    assert_eq!(recipe.tone_curve, None);
    assert_eq!(unsupported, ["Unknown"]);
    assert!(warnings.is_empty());
}

#[test]
fn lightroom_effective_defaults_do_not_create_a_recipe() {
    let values = Parser::new(
            "s={ToneCurvePV2012={0,0,22,16,40,35,127,127,224,230,240,246,255,255},Sharpness=40,SharpenRadius=1,SharpenDetail=25,SharpenEdgeMasking=0,LuminanceSmoothing=0,LuminanceNoiseReductionDetail=50,LuminanceNoiseReductionContrast=0,ColorNoiseReduction=25,ColorNoiseReductionDetail=50,ColorNoiseReductionSmoothness=50}",
        )
        .parse_assignment()
        .unwrap();

    for version in ["6.7", "11.0", "15.4"] {
        let (recipe, unsupported, warnings) =
            normalize_recipe(version, &values, ImportCapabilities::ALL);
        assert!(recipe.is_empty(), "{version}: {recipe:?}");
        assert!(unsupported.is_empty(), "{version}: {unsupported:?}");
        assert!(warnings.is_empty(), "{version}: {warnings:?}");
    }
}

#[test]
fn deviations_from_lightroom_defaults_remain_develop_edits() {
    let changed_curve =
        Parser::new("s={ToneCurvePV2012={0,0,22,16,40,35,127,120,224,230,240,246,255,255}}")
            .parse_assignment()
            .unwrap();
    let changed_detail = Parser::new(
            "s={Sharpness=41,SharpenRadius=1,SharpenDetail=25,SharpenEdgeMasking=0,LuminanceSmoothing=0,LuminanceNoiseReductionDetail=50,LuminanceNoiseReductionContrast=0,ColorNoiseReduction=25,ColorNoiseReductionDetail=50,ColorNoiseReductionSmoothness=50}",
        )
        .parse_assignment()
        .unwrap();

    let (curve_recipe, _, _) = normalize_recipe("15.4", &changed_curve, ImportCapabilities::ALL);
    let (detail_recipe, _, _) = normalize_recipe("15.4", &changed_detail, ImportCapabilities::ALL);
    assert!(curve_recipe.tone_curve.is_some());
    assert!(detail_recipe.detail.is_some());
}

#[test]
fn phase_b_normalization_is_stable_across_catalog_process_versions() {
    let values = Parser::new(
            "s={Exposure2012=1.25,WhiteBalance='Custom',CustomTemperature=7200,CustomTint=12,IncrementalTemperature=4,IncrementalTint=-3,ToneCurve={0,0,128,160,255,255},ToneCurvePV2012Red={0,0,255,240}}",
        )
        .parse_assignment()
        .unwrap();
    for version in ["6.7", "11.0", "15.4"] {
        let (recipe, unsupported, warnings) =
            normalize_recipe(version, &values, ImportCapabilities::ALL);
        assert_eq!(recipe.exposure_ev, Some(1.25));
        assert_eq!(recipe.white_balance.unwrap().temperature, 7200.0);
        assert!(recipe.white_balance_increment.is_none());
        assert_eq!(recipe.tone_curve.as_ref().unwrap().len(), 3);
        assert!(recipe.tone_curve_red.is_some());
        assert!(unsupported.is_empty(), "{version}: {unsupported:?}");
        assert!(warnings.is_empty(), "{version}: {warnings:?}");
    }
}

#[test]
fn incremental_white_balance_is_used_only_without_an_absolute_family() {
    let incremental = Parser::new(
        "s={WhiteBalance='As Shot',CustomIncrementalTemperature=8,CustomIncrementalTint=-2}",
    )
    .parse_assignment()
    .unwrap();
    let (recipe, _, _) = normalize_recipe("15.4", &incremental, ImportCapabilities::ALL);
    assert!(recipe.white_balance.is_none());
    assert_eq!(recipe.white_balance_increment.unwrap().temperature, 8.0);

    let absolute = Parser::new(
            "s={WhiteBalance='Custom',CustomTemperature=7000,CustomTint=4,CustomIncrementalTemperature=8}",
        )
        .parse_assignment()
        .unwrap();
    let (recipe, _, _) = normalize_recipe("15.4", &absolute, ImportCapabilities::ALL);
    assert!(recipe.white_balance.is_some());
    assert!(recipe.white_balance_increment.is_none());
}

#[test]
fn phase_c_controls_are_normalized_without_adobe_profile_payloads() {
    let values = Parser::new(
            "s={CameraProfile='Adobe Standard',CameraProfileDigest='secret',Sharpness=40,SharpenRadius=1,LuminanceSmoothing=20,ColorNoiseReduction=25,Texture=10,Dehaze=5,ConvertToGrayscale=true,GrayMixerRed=20,PostCropVignetteAmount=-15,ParametricHighlights=10}",
        )
        .parse_assignment()
        .unwrap();
    let (recipe, unsupported, warnings) =
        normalize_recipe("15.4", &values, ImportCapabilities::ALL);

    assert_eq!(recipe.detail.unwrap().sharpening, 40.0);
    assert_eq!(recipe.texture, Some(10.0));
    assert!(recipe.monochrome.is_some());
    assert_eq!(recipe.vignette.unwrap().amount, -15.0);
    assert_eq!(recipe.parametric_curve.unwrap()[0], 10.0);
    assert!(unsupported.contains(&"CameraProfile".to_owned()));
    assert!(
        warnings
            .iter()
            .any(|warning| warning.contains("no compatible resolver"))
    );
}

#[test]
fn phase_d_omits_unselected_upright_and_normalizes_heal_manual_lens_and_grain() {
    let values = Parser::new(
            "s={UprightTransform_4='1,0,0.1,0,1,0,0,0,1',LensManualDistortionAmount=12,RetouchInfo={{spotType='heal',centerX=0.5,centerY=0.5,sourceX=0.25,sourceY=0.25,radius=0.1,opacity=0.8}},GrainSize=57,GrainFrequency=81}",
        )
        .parse_assignment()
        .unwrap();
    let (recipe, unsupported, warnings) =
        normalize_recipe("15.4", &values, ImportCapabilities::ALL);

    assert!(recipe.upright_homography.is_none());
    assert_eq!(recipe.lens_distortion, Some(12.0));
    assert_eq!(recipe.heal_spots.len(), 1);
    assert_eq!(recipe.heal_spots[0].kind, HealKind::Heal);
    assert_eq!(recipe.heal_spots[0].opacity, 0.8);
    assert_eq!(recipe.grain.unwrap().roughness, 81.0);
    assert!(unsupported.is_empty(), "{unsupported:?}");
    assert!(
        warnings
            .iter()
            .any(|warning| warning.contains("active-mode"))
    );
    assert!(
        warnings
            .iter()
            .any(|warning| warning.contains("grain amount"))
    );
}

#[test]
fn retouch_areas_is_canonical_and_legacy_is_only_a_fallback() {
    let values = Parser::new(
            "s={RetouchAreas={{SpotType='clone',CenterX=0.5,CenterY=0.6,SourceX=0.2,SourceY=0.3,Radius=0.1,Opacity=0.7,Feather=25}},RetouchInfo={{spotType='heal',centerX=0.1,centerY=0.1,sourceX=0.9,sourceY=0.9,radius=0.2}}}",
        )
        .parse_assignment()
        .unwrap();
    let (recipe, _, _) = normalize_recipe("15.4", &values, ImportCapabilities::ALL);

    assert_eq!(recipe.heal_spots.len(), 1);
    assert_eq!(recipe.heal_spots[0].kind, HealKind::Clone);
    assert_eq!(recipe.heal_spots[0].destination, [0.5, 0.6]);
    assert_eq!(recipe.heal_spots[0].feather, 0.25);
}

#[test]
fn active_unresolved_lens_profiles_warn_but_inactive_profiles_do_not() {
    let inactive = Parser::new("s={LensProfileEnable=0,LensProfileName='Adobe Lens Profile'}")
        .parse_assignment()
        .unwrap();
    let (_, _, warnings) = normalize_recipe("15.4", &inactive, ImportCapabilities::ALL);
    assert!(warnings.is_empty());

    let active = Parser::new("s={LensProfileEnable=1,LensProfileName='Adobe Lens Profile'}")
        .parse_assignment()
        .unwrap();
    let (recipe, _, warnings) = normalize_recipe("15.4", &active, ImportCapabilities::ALL);
    assert!(recipe.lens_distortion.is_none());
    assert!(
        warnings
            .iter()
            .any(|warning| warning.contains("lens profile"))
    );
}

#[test]
fn singular_homography_is_rejected_by_recipe_validation() {
    let recipe = DevelopRecipe {
        upright_homography: Some([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        ..DevelopRecipe::default()
    };
    assert!(recipe.validate().is_err());
}

#[test]
fn phase_d_inventory_assumptions_are_guarded() {
    let inventory: serde_json::Value =
        serde_json::from_str(include_str!("../../lightroom-inventory.json")).unwrap();
    let properties = inventory["properties"].as_object().unwrap();

    assert_eq!(properties["LensProfileEnable"]["numeric_min"], 0.0);
    assert_eq!(properties["LensProfileEnable"]["numeric_max"], 0.0);
    assert_eq!(properties["UprightCenterMode"]["numeric_max"], 0.0);
    assert_eq!(properties["RetouchAreas"]["present"], 1);
    assert_eq!(properties["GrainSize"]["numeric_min"], 25.0);
    assert_eq!(properties["GrainSize"]["numeric_max"], 57.0);
    assert_eq!(properties["GrainFrequency"]["present"], 1);
    let common = |name: &str| {
        properties[name]["most_common_values"][0]["value"]
            .as_str()
            .unwrap()
    };
    let assert_only_common = |name: &str, expected: &str| {
        assert_eq!(properties[name]["distinct_values"], 1);
        assert_eq!(common(name), expected);
        assert_eq!(
            properties[name]["most_common_values"][0]["count"], properties[name]["present"],
            "{name} gained a non-default value; reevaluate its import policy"
        );
    };
    for empty_table in ["AILook", "PointColors", "RedEyeInfo", "LensBlur"] {
        assert_only_common(empty_table, "<table:0>");
    }
    assert_only_common("Look.Parameters.PointColors", "<table:0>");
    assert_only_common("Look.Parameters.LensBlur.Active", "false");
    for default_zero in [
        "ColorGradeGlobalHue",
        "ColorGradeGlobalLum",
        "ColorGradeGlobalSat",
        "ColorGradeHighlightLum",
        "ColorGradeMidtoneHue",
        "ColorGradeMidtoneLum",
        "ColorGradeMidtoneSat",
        "ColorGradeShadowLum",
    ] {
        assert_eq!(properties[default_zero]["numeric_min"], 0.0);
        assert_eq!(properties[default_zero]["numeric_max"], 0.0);
    }
    assert_eq!(properties["ColorGradeBlending"]["numeric_min"], 50.0);
    assert_eq!(properties["ColorGradeBlending"]["numeric_max"], 50.0);
    assert_eq!(properties["RetouchAreas.[0].Masks"]["present"], 1);
    assert_eq!(common("RetouchAreas.[0].Masks"), "<table:1>");
}

#[test]
fn identity_upright_and_default_grain_are_not_stored() {
    let values = Parser::new("s={UprightTransform_4='1,0,0,0,1,0,0,0,1',GrainSize=25}")
        .parse_assignment()
        .unwrap();
    let (recipe, _, _) = normalize_recipe("15.4", &values, ImportCapabilities::ALL);
    assert!(recipe.upright_homography.is_none());
    assert!(recipe.grain.is_none());
}

#[test]
fn existing_catalog_is_never_created_implicitly() {
    let directory = tempfile::tempdir().unwrap();
    assert!(load_recipes(directory.path()).unwrap().is_empty());
    assert!(!directory.path().join(CATALOG_DIR).exists());
}

#[test]
fn import_is_read_only_minimal_and_reloadable() {
    let root = tempfile::tempdir().unwrap();
    fs::write(root.path().join("photo.CR3"), b"test").unwrap();
    let fixture = lightroom_fixture(
        root.path(),
        "s={Exposure2012=1.5,Contrast2012=0,UnknownAdobeThing=42}",
    );
    let lightroom_path = fixture.path().join("fixture.lrcat");
    let before = fs::read(&lightroom_path).unwrap();

    let report =
        import_lightroom_inner(root.path(), &lightroom_path, true, ImportCapabilities::ALL)
            .unwrap();

    assert_eq!(report.imported_recipes, 1);
    assert_eq!(
        report.unsupported_property_occurrences["UnknownAdobeThing"],
        1
    );
    assert_eq!(fs::read(&lightroom_path).unwrap(), before);
    assert!(!lightroom_path.with_extension("lrcat-wal").exists());
    assert!(!lightroom_path.with_extension("lrcat-shm").exists());
    let recipes = load_recipes(root.path()).unwrap();
    let entry = &recipes[&root.path().join("photo.CR3")];
    let expected = DevelopRecipe {
        exposure_ev: Some(1.5),
        ..DevelopRecipe::default()
    };
    assert_eq!(entry.digest, expected.digest().unwrap());

    let connection = Connection::open(root.path().join(CATALOG_DIR).join(CATALOG_FILE)).unwrap();
    let stored: String = connection
        .query_row("SELECT recipe_json FROM develop_recipes", [], |row| {
            row.get(0)
        })
        .unwrap();
    assert!(!stored.contains("UnknownAdobeThing"));
    assert!(!stored.contains("ProcessVersion"));
}

#[test]
fn failed_reimport_rolls_back_existing_recipes() {
    let root = tempfile::tempdir().unwrap();
    fs::write(root.path().join("photo.CR3"), b"test").unwrap();
    let fixture = lightroom_fixture(root.path(), "s={Exposure2012=1}");
    let lightroom_path = fixture.path().join("fixture.lrcat");
    import_lightroom_inner(root.path(), &lightroom_path, true, ImportCapabilities::ALL).unwrap();
    let before = load_recipes(root.path()).unwrap()[&root.path().join("photo.CR3")].digest;
    Connection::open(&lightroom_path)
        .unwrap()
        .execute(
            "UPDATE Adobe_imageDevelopSettings SET text='not a recipe'",
            [],
        )
        .unwrap();

    assert!(
        import_lightroom_inner(root.path(), &lightroom_path, true, ImportCapabilities::ALL,)
            .is_err()
    );
    let after = load_recipes(root.path()).unwrap()[&root.path().join("photo.CR3")].digest;
    assert_eq!(after, before);
}

#[test]
fn dry_run_and_failed_first_import_do_not_create_a_catalog() {
    let root = tempfile::tempdir().unwrap();
    fs::write(root.path().join("photo.CR3"), b"test").unwrap();
    let fixture = lightroom_fixture(root.path(), "s={Exposure2012=1}");
    let lightroom_path = fixture.path().join("fixture.lrcat");
    analyze_lightroom(root.path(), &lightroom_path).unwrap();
    assert!(!root.path().join(CATALOG_DIR).exists());

    Connection::open(&lightroom_path)
        .unwrap()
        .execute("UPDATE Adobe_imageDevelopSettings SET text='broken'", [])
        .unwrap();
    assert!(import_lightroom(root.path(), &lightroom_path).is_err());
    assert!(!root.path().join(CATALOG_DIR).exists());
}

#[test]
fn reimport_removes_recipes_that_now_normalize_to_lightroom_defaults() {
    let root = tempfile::tempdir().unwrap();
    fs::write(root.path().join("photo.CR3"), b"test").unwrap();
    let fixture = lightroom_fixture(root.path(), "s={Exposure2012=1}");
    let lightroom_path = fixture.path().join("fixture.lrcat");
    import_lightroom(root.path(), &lightroom_path).unwrap();
    assert_eq!(load_recipes(root.path()).unwrap().len(), 1);

    Connection::open(&lightroom_path)
            .unwrap()
            .execute(
                "UPDATE Adobe_imageDevelopSettings SET text=?1",
                ["s={ToneCurvePV2012={0,0,22,16,40,35,127,127,224,230,240,246,255,255},Sharpness=40,SharpenRadius=1,SharpenDetail=25,SharpenEdgeMasking=0,LuminanceSmoothing=0,LuminanceNoiseReductionDetail=50,LuminanceNoiseReductionContrast=0,ColorNoiseReduction=25,ColorNoiseReductionDetail=50,ColorNoiseReductionSmoothness=50}"],
            )
            .unwrap();

    let report = import_lightroom(root.path(), &lightroom_path).unwrap();
    assert_eq!(report.empty_supported_recipes, 1);
    assert!(load_recipes(root.path()).unwrap().is_empty());
}

#[test]
fn source_resolution_rejects_parent_traversal() {
    let parent = tempfile::tempdir().unwrap();
    let root = parent.path().join("root");
    fs::create_dir(&root).unwrap();
    fs::write(parent.path().join("outside.CR3"), b"outside").unwrap();
    assert!(resolve_relative_source(&root, root.to_str().unwrap(), "..", "outside.CR3").is_none());
}

#[cfg(unix)]
#[test]
fn relative_path_encoding_round_trips_non_unicode_and_backslashes() {
    use std::os::unix::ffi::OsStrExt as _;
    let path = Path::new(std::ffi::OsStr::from_bytes(b"dir/invalid-\xff\\name.CR3"));
    assert_eq!(
        decode_relative_path(&encode_relative_path(path).unwrap()).unwrap(),
        path
    );
}

#[test]
fn unknown_process_version_imports_no_actions() {
    let values = Parser::new("s={Exposure2012=1}")
        .parse_assignment()
        .unwrap();
    let (recipe, unsupported, warnings) =
        normalize_recipe("99.0", &values, ImportCapabilities::ALL);
    assert!(recipe.is_empty());
    assert!(unsupported.contains(&"UnsupportedProcessVersion:99.0".to_owned()));
    assert_eq!(warnings, ["unsupported process version 99.0"]);
}

#[test]
fn portable_unicode_paths_use_cross_platform_encoding() {
    let path = Path::new("2025/Kanada/photo.CR3");
    let encoded = encode_relative_path(path).unwrap();
    assert_eq!(encoded.first(), Some(&b'p'));
    assert_eq!(decode_portable_path(&encoded).unwrap(), path);
}

#[test]
fn incompatible_owned_schema_is_not_modified() {
    let root = tempfile::tempdir().unwrap();
    fs::write(root.path().join("photo.CR3"), b"test").unwrap();
    let fixture = lightroom_fixture(root.path(), "s={Exposure2012=1}");
    let directory = root.path().join(CATALOG_DIR);
    fs::create_dir(&directory).unwrap();
    let path = directory.join(CATALOG_FILE);
    let connection = Connection::open(&path).unwrap();
    connection
        .execute_batch(&format!(
            "PRAGMA application_id={SQLITE_APPLICATION_ID};
                 PRAGMA user_version=2;
                 CREATE TABLE marker(value TEXT);
                 INSERT INTO marker VALUES('untouched');"
        ))
        .unwrap();
    drop(connection);
    let before = fs::read(&path).unwrap();

    assert!(
        import_lightroom_inner(
            root.path(),
            &fixture.path().join("fixture.lrcat"),
            true,
            ImportCapabilities::ALL,
        )
        .is_err()
    );
    assert_eq!(fs::read(path).unwrap(), before);
}

#[test]
fn older_renderer_catalog_remains_readable_and_reimport_updates_revision() {
    let root = tempfile::tempdir().unwrap();
    fs::write(root.path().join("photo.CR3"), b"test").unwrap();
    let fixture = lightroom_fixture(root.path(), "s={Exposure2012=1}");
    let lightroom_path = fixture.path().join("fixture.lrcat");
    import_lightroom_inner(root.path(), &lightroom_path, true, ImportCapabilities::ALL).unwrap();
    let path = catalog_path(root.path());
    let connection = Connection::open(&path).unwrap();
    connection
        .execute("UPDATE catalog_meta SET renderer_model=?1", [3_u32])
        .unwrap();
    drop(connection);

    assert_eq!(load_recipes(root.path()).unwrap().len(), 1);
    import_lightroom_inner(root.path(), &lightroom_path, true, ImportCapabilities::ALL).unwrap();
    let connection = Connection::open(path).unwrap();
    let renderer_model: u32 = connection
        .query_row("SELECT renderer_model FROM catalog_meta", [], |row| {
            row.get(0)
        })
        .unwrap();
    assert_eq!(renderer_model, RENDERER_MODEL_VERSION);
}

#[test]
fn conflicting_master_rows_are_reported_and_not_applied() {
    let root = tempfile::tempdir().unwrap();
    fs::write(root.path().join("photo.CR3"), b"test").unwrap();
    let fixture = lightroom_fixture(root.path(), "s={Exposure2012=1}");
    let path = fixture.path().join("fixture.lrcat");
    let connection = Connection::open(&path).unwrap();
    connection
        .execute("INSERT INTO Adobe_images VALUES(2, 1, NULL)", [])
        .unwrap();
    connection
        .execute(
            "INSERT INTO Adobe_imageDevelopSettings VALUES(2, '15.4', 's={Exposure2012=2}')",
            [],
        )
        .unwrap();
    drop(connection);
    let report =
        import_lightroom_inner(root.path(), &path, false, ImportCapabilities::ALL).unwrap();
    assert_eq!(report.conflicting_sources, 1);
    assert!(import_lightroom_inner(root.path(), &path, true, ImportCapabilities::ALL).is_err());
    assert!(!root.path().join(CATALOG_DIR).exists());
}

#[test]
fn parser_rejects_trailing_garbage() {
    assert!(Parser::new("s={} garbage").parse_assignment().is_err());
}

#[test]
fn hidden_subtrees_round_trip_without_develop_recipes() {
    let root = tempfile::tempdir().unwrap();
    let hidden = HashSet::from([root.path().join("private"), root.path().join("archive/old")]);

    save_hidden_subtrees(root.path(), &hidden).unwrap();

    assert_eq!(load_hidden_subtrees(root.path()).unwrap(), hidden);
    assert!(load_recipes(root.path()).unwrap().is_empty());
}
