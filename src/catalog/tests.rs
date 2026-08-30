use super::*;
use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{Arc, atomic::AtomicBool},
    time::{Duration, Instant},
};
use test_r::test;

#[test]
fn production_recognition_policy_tampering_fails_closed_in_diagnostics_and_execution() {
    for corruption in [
        "UPDATE recognition_policy_artifacts SET absolute_threshold=.1",
        "UPDATE recognition_policy_artifacts SET calibration_digest=zeroblob(32)",
    ] {
        let root = tempfile::tempdir().unwrap();
        initialize_test_catalog(root.path()).unwrap();
        recognition::ensure_production_calibration(root.path()).unwrap();
        let connection = Connection::open(catalog_path(root.path())).unwrap();
        connection.execute(corruption, []).unwrap();
        connection
            .execute(
                "INSERT INTO recognition_refresh_queue(entity_kind, entity_id, model_id, status,
                    attempts, updated_at)
                 VALUES('face', 77, ?1, 'pending', 0, datetime('now'))",
                [recognition::PRODUCTION_SFACE_MODEL_ID],
            )
            .unwrap();
        assert!(
            !recognition::production_calibration_status(&connection)
                .unwrap()
                .active
        );
        drop(connection);
        assert_eq!(
            recognition::process_pending_refresh_batch(
                root.path(),
                "tamper-test",
                &crate::scheduler::MaintenanceAdmission::new(),
                &AtomicBool::new(false),
            )
            .unwrap(),
            2
        );
        assert_eq!(
            Connection::open(catalog_path(root.path()))
                .unwrap()
                .query_row("SELECT count(*) FROM recognition_suggestions", [], |row| {
                    row.get::<_, i64>(0)
                })
                .unwrap(),
            0
        );
    }
}

struct RunningStageBarrier {
    target: crate::face_engine::FaceAnalysisStage,
    reached: Arc<(std::sync::Mutex<bool>, std::sync::Condvar)>,
    required_query: Option<Arc<QueryWorkloadLifecycle>>,
}

impl crate::face_engine::ProductionAnalysisStages for RunningStageBarrier {
    fn reached(&self, stage: crate::face_engine::FaceAnalysisStage, cancelled: &Arc<AtomicBool>) {
        if stage != self.target {
            return;
        }
        if let Some(query) = &self.required_query {
            while query.in_progress.load(std::sync::atomic::Ordering::Acquire) == 0
                && !query.finished.load(std::sync::atomic::Ordering::Acquire)
                && !cancelled.load(std::sync::atomic::Ordering::Acquire)
            {
                std::thread::yield_now();
            }
            let overlapping = query.started.load(std::sync::atomic::Ordering::Acquire)
                && !query.finished.load(std::sync::atomic::Ordering::Acquire)
                && query.in_progress.load(std::sync::atomic::Ordering::Acquire) > 0;
            query
                .overlap_observed
                .store(overlapping, std::sync::atomic::Ordering::Release);
        }
        let (lock, wake) = &*self.reached;
        *lock.lock().unwrap() = true;
        wake.notify_all();
        while !cancelled.load(std::sync::atomic::Ordering::Acquire) {
            let guard = lock.lock().unwrap();
            drop(wake.wait_timeout(guard, Duration::from_millis(5)).unwrap());
        }
    }
}

fn wait_for_stage(reached: &Arc<(std::sync::Mutex<bool>, std::sync::Condvar)>) -> bool {
    let deadline = Instant::now() + Duration::from_secs(120);
    let (lock, wake) = &**reached;
    let mut observed = lock.lock().unwrap();
    while !*observed && Instant::now() < deadline {
        observed = wake
            .wait_timeout(observed, Duration::from_millis(10))
            .unwrap()
            .0;
    }
    *observed
}

struct ReleasableStageBarrier {
    target: crate::face_engine::FaceAnalysisStage,
    reached: Arc<(std::sync::Mutex<bool>, std::sync::Condvar)>,
    release: Arc<(std::sync::Mutex<bool>, std::sync::Condvar)>,
}

impl crate::face_engine::ProductionAnalysisStages for ReleasableStageBarrier {
    fn reached(&self, stage: crate::face_engine::FaceAnalysisStage, cancelled: &Arc<AtomicBool>) {
        if stage != self.target {
            return;
        }
        *self.reached.0.lock().unwrap() = true;
        self.reached.1.notify_all();
        let mut released = self.release.0.lock().unwrap();
        while !*released && !cancelled.load(std::sync::atomic::Ordering::Acquire) {
            released = self
                .release
                .1
                .wait_timeout(released, Duration::from_millis(10))
                .unwrap()
                .0;
        }
    }
}

#[derive(Default)]
struct QueryWorkloadLifecycle {
    started: AtomicBool,
    finished: AtomicBool,
    in_progress: std::sync::atomic::AtomicUsize,
    overlap_observed: AtomicBool,
    wake: std::sync::Condvar,
    lock: std::sync::Mutex<()>,
}

struct QueryInProgress<'a>(&'a QueryWorkloadLifecycle);

impl Drop for QueryInProgress<'_> {
    fn drop(&mut self) {
        self.0
            .in_progress
            .fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
        self.0.wake.notify_all();
    }
}

impl QueryWorkloadLifecycle {
    fn begin(&self) -> QueryInProgress<'_> {
        self.in_progress
            .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        self.started
            .store(true, std::sync::atomic::Ordering::Release);
        self.wake.notify_all();
        QueryInProgress(self)
    }

    fn finish(&self) {
        self.finished
            .store(true, std::sync::atomic::Ordering::Release);
        self.wake.notify_all();
    }

    fn wait_until_started(&self, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        let mut guard = self.lock.lock().unwrap();
        while !self.started.load(std::sync::atomic::Ordering::Acquire) && Instant::now() < deadline
        {
            guard = self
                .wake
                .wait_timeout(guard, Duration::from_millis(10))
                .unwrap()
                .0;
        }
        self.started.load(std::sync::atomic::Ordering::Acquire)
    }
}

#[test]
fn query_workload_lifecycle_has_bounded_start_wait_and_exact_in_progress_state() {
    let lifecycle = QueryWorkloadLifecycle::default();
    assert!(!lifecycle.wait_until_started(Duration::from_millis(20)));
    let active = lifecycle.begin();
    assert!(lifecycle.wait_until_started(Duration::from_millis(20)));
    assert_eq!(
        lifecycle
            .in_progress
            .load(std::sync::atomic::Ordering::Acquire),
        1
    );
    assert!(
        !lifecycle
            .finished
            .load(std::sync::atomic::Ordering::Acquire)
    );
    drop(active);
    assert_eq!(
        lifecycle
            .in_progress
            .load(std::sync::atomic::Ordering::Acquire),
        0
    );
    lifecycle.finish();
    assert!(
        lifecycle
            .finished
            .load(std::sync::atomic::Ordering::Acquire)
    );
}

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
                 CREATE TABLE Adobe_imageDevelopSettings(image INTEGER, processVersion TEXT, text TEXT);
                 CREATE TABLE Adobe_variables(id_local INTEGER PRIMARY KEY, id_global BLOB, name TEXT, value BLOB);",
            )
            .unwrap();
    connection
        .execute(
            "INSERT INTO Adobe_variables VALUES(1, X'01', 'Adobe_storeProviderID', ?1)",
            [blake3::hash(fixture.path().as_os_str().as_encoded_bytes()).as_bytes()],
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
fn lightroom_auxiliary_selection_resolves_main_catalog_and_rejects_archives() {
    let root = tempfile::tempdir().unwrap();
    let catalog = root.path().join("Travel.lrcat");
    Connection::open(&catalog)
        .unwrap()
        .execute("CREATE TABLE marker(id INTEGER)", [])
        .unwrap();
    let auxiliary = root.path().join("Travel.lrcat-data");
    fs::write(&auxiliary, b"not sqlite auxiliary data").unwrap();
    assert_eq!(lightroom_database_path(&auxiliary).unwrap(), catalog);

    let backup = root.path().join("backup.zip");
    fs::write(&backup, b"PK\x03\x04fixture").unwrap();
    assert!(
        format!("{:#}", lightroom_database_path(&backup).unwrap_err())
            .contains("compressed Lightroom backup")
    );
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
fn lightroom_people_import_uses_catalog_path_when_all_global_ids_are_missing() {
    let root = tempfile::tempdir().unwrap();
    fs::write(root.path().join("photo.CR3"), b"test").unwrap();
    let fixture = lightroom_fixture(root.path(), "s={Exposure2012=1}");
    let lightroom_path = fixture.path().join("fixture.lrcat");
    let connection = Connection::open(&lightroom_path).unwrap();
    connection
        .execute(
            "UPDATE Adobe_variables SET id_global=NULL, name='unrelated', value=NULL WHERE id_local=1",
            [],
        )
        .unwrap();
    connection
        .execute_batch(
            "CREATE TABLE AgLibraryFace(
                 id_local INTEGER PRIMARY KEY, image INTEGER, tl_x REAL, tl_y REAL,
                 br_x REAL, br_y REAL, imageOrientation TEXT, origination REAL,
                 ignored INTEGER, cluster INTEGER);
             CREATE TABLE AgLibraryKeyword(
                 id_local INTEGER PRIMARY KEY, id_global BLOB, name TEXT, keywordType TEXT);
             CREATE TABLE AgLibraryKeywordFace(
                 id_local INTEGER PRIMARY KEY, face INTEGER, tag INTEGER,
                 userPick INTEGER, userReject INTEGER);
             CREATE TABLE AgLibraryFaceCluster(id_local INTEGER PRIMARY KEY);
             INSERT INTO AgLibraryFace VALUES(10, 1, .1, .2, .4, .7, 'BC', 3, 0, NULL);",
        )
        .unwrap();
    drop(connection);

    import_lightroom_inner(root.path(), &lightroom_path, true, ImportCapabilities::ALL).unwrap();
    let imported = people::load(root.path()).unwrap();
    assert_eq!(imported.faces.len(), 1);
    assert_eq!(imported.faces[0].bounds, [0.2, 0.6, 0.7, 0.9]);
}

#[test]
fn lightroom_people_import_preserves_confirmations_and_supports_local_review() {
    let root = tempfile::tempdir().unwrap();
    fs::write(root.path().join("photo.CR3"), b"test").unwrap();
    let fixture = lightroom_fixture(root.path(), "s={Exposure2012=1}");
    let lightroom_path = fixture.path().join("fixture.lrcat");
    let connection = Connection::open(&lightroom_path).unwrap();
    connection
        .execute_batch(
            "CREATE TABLE AgLibraryFace(
                 id_local INTEGER PRIMARY KEY, image INTEGER, tl_x REAL, tl_y REAL,
                 br_x REAL, br_y REAL, imageOrientation TEXT, origination REAL,
                 ignored INTEGER, cluster INTEGER);
             CREATE TABLE AgLibraryKeyword(
                 id_local INTEGER PRIMARY KEY, id_global BLOB, name TEXT, keywordType TEXT);
             CREATE TABLE AgLibraryKeywordFace(
                 id_local INTEGER PRIMARY KEY, face INTEGER, tag INTEGER,
                 userPick INTEGER, userReject INTEGER);
             CREATE TABLE AgLibraryFaceCluster(id_local INTEGER PRIMARY KEY);
             CREATE TABLE AgLibraryFaceData(data BLOB);
             INSERT INTO AgLibraryFace VALUES(10, 1, 0.1, 0.2, 0.4, 0.7, 'BC', 1, 0, 7);
             INSERT INTO AgLibraryKeyword VALUES(20, X'010203', 'Ada', 'person');
             INSERT INTO AgLibraryKeywordFace VALUES(30, 10, 20, 1, 0);
             INSERT INTO AgLibraryFaceCluster VALUES(7);",
        )
        .unwrap();
    drop(connection);

    let report =
        import_lightroom_inner(root.path(), &lightroom_path, true, ImportCapabilities::ALL)
            .unwrap();
    assert_eq!(report.imported_faces, 1);
    assert_eq!(report.confirmed_face_assignments, 1);
    assert_eq!(report.imported_face_clusters, 1);

    let people = people::load(root.path()).unwrap();
    assert_eq!(people.people.len(), 1);
    assert_eq!(people.people[0].name, "Ada");
    let canonical_bounds = [0.2, 0.6, 0.7, 0.9];
    assert_eq!(people.faces[0].bounds, canonical_bounds);
    assert_eq!(
        people.assignments[0].kind,
        people::FaceAssignmentKind::Confirmed
    );
    let page = people::list_people_page(root.path(), None, 0, 20).unwrap();
    assert_eq!(page.total, 1);
    assert_eq!(page.unnamed_faces, 0);
    assert_eq!(page.review_faces, 0);
    assert_eq!(page.items[0].confirmed_faces, 1);
    assert_eq!(
        page.items[0].representative_face_id,
        Some(people.faces[0].id)
    );
    let representative = page.items[0].representative.as_ref().unwrap();
    assert_eq!(representative.face_id, people.faces[0].id);
    assert_eq!(representative.path, root.path().join("photo.CR3"));
    assert_eq!(representative.bounds, canonical_bounds);
    let cancelled = Arc::new(AtomicBool::new(true));
    let error = people::list_people_page_sorted_cancellable(
        root.path(),
        None,
        people::PeopleSort::Name,
        0,
        20,
        HashMap::new(),
        cancelled,
    )
    .unwrap_err();
    assert!(format!("{error:#}").contains("interrupted"));
    let faces =
        people::list_person_faces_page(root.path(), people.people[0].id, None, 0, 20).unwrap();
    assert_eq!(faces.total, 1);
    assert_eq!(faces.items[0].path, root.path().join("photo.CR3"));
    assert_eq!(faces.items[0].kind, people::FaceAssignmentKind::Confirmed);

    crate::face_engine::sync_analysis_inventory(root.path(), &[root.path().join("photo.CR3")])
        .unwrap();
    let asset_id = Connection::open(root.path().join(CATALOG_DIR).join(CATALOG_FILE))
        .unwrap()
        .query_row("SELECT id FROM assets", [], |row| row.get::<_, i64>(0))
        .unwrap();
    let imported_landmarks: Option<Vec<u8>> = Connection::open(catalog_path(root.path()))
        .unwrap()
        .query_row(
            "SELECT embedding.landmarks FROM faces face
             LEFT JOIN face_embeddings embedding ON embedding.face_id=face.id
             WHERE face.provenance='lightroom'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(
        imported_landmarks, None,
        "a freshly imported Lightroom face must exercise the NULL-landmark geometry fallback"
    );
    let landmarks = [
        [0.45, 0.205],
        [0.65, 0.205],
        [0.55, 0.265],
        [0.475, 0.325],
        [0.625, 0.325],
    ];
    crate::face_engine::store_test_analysis(root.path(), asset_id, canonical_bounds, landmarks)
        .unwrap();
    let rekesz_catalog = catalog_path(root.path());
    let connection = Connection::open(&rekesz_catalog).unwrap();
    connection
        .execute_batch(
            "DELETE FROM recognition_refresh_queue;
             DELETE FROM recognition_rescore_state;
             INSERT INTO person_prototypes(person_id, model_id, dimensions, vector,
                 exemplar_count, generation, updated_at)
             VALUES(1, 'sface-2021dec', 2, zeroblob(8), 1, 7, datetime('now'));
             INSERT INTO recognition_suggestions(face_id, person_id, recognizer_model_id,
                 prototype_generation, prototype_algorithm_version, ann_index_version,
                 scored_index_generation, policy_version, score, nearest_exemplar_score,
                 second_best_margin, created_at)
             VALUES(1, 1, 'sface-2021dec', 7, 1, 1, 7, 1, .9, .9, .1, datetime('now'));",
        )
        .unwrap();
    let epoch_before_reimport: i64 = connection
        .query_row(
            "SELECT source_epoch FROM people_summary_epoch WHERE id=1",
            [],
            |row| row.get(0),
        )
        .unwrap();
    drop(connection);
    Connection::open(&lightroom_path)
        .unwrap()
        .execute(
            "UPDATE AgLibraryKeywordFace SET userPick=0 WHERE id_local=30",
            [],
        )
        .unwrap();
    import_lightroom_inner(root.path(), &lightroom_path, true, ImportCapabilities::ALL).unwrap();
    let connection = Connection::open(&rekesz_catalog).unwrap();
    let epoch_after_reimport: i64 = connection
        .query_row(
            "SELECT source_epoch FROM people_summary_epoch WHERE id=1",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(epoch_after_reimport, epoch_before_reimport + 1);
    assert_eq!(
        connection
            .query_row(
                "SELECT count(*) FROM recognition_refresh_queue
                 WHERE model_id='sface-2021dec' AND status='pending'
                   AND ((entity_kind='person' AND entity_id=1)
                     OR (entity_kind='face' AND entity_id=1))",
                [],
                |row| row.get::<_, i64>(0),
            )
            .unwrap(),
        2,
        "changing imported confirmation membership must atomically invalidate its prototype and face score"
    );
    assert_eq!(
        connection
            .query_row(
                "SELECT after_face_id FROM recognition_rescore_state WHERE model_id='sface-2021dec'",
                [],
                |row| row.get::<_, i64>(0),
            )
            .unwrap(),
        0,
        "prototype membership changes must restart the catalog rescore window"
    );
    drop(connection);
    Connection::open(&lightroom_path)
        .unwrap()
        .execute(
            "UPDATE AgLibraryKeywordFace SET userPick=1 WHERE id_local=30",
            [],
        )
        .unwrap();
    import_lightroom_inner(root.path(), &lightroom_path, true, ImportCapabilities::ALL).unwrap();
    crate::face_engine::store_test_analysis(root.path(), asset_id, canonical_bounds, landmarks)
        .unwrap();
    let reconciled = people::load(root.path()).unwrap();
    assert_eq!(reconciled.faces.len(), 1);
    assert_eq!(reconciled.assignments.len(), 1);

    people::reject(root.path(), people.faces[0].id, people.people[0].id).unwrap();
    let rejected = people::load(root.path()).unwrap();
    assert!(rejected.assignments.is_empty());
    let overview = people::list_people_page(root.path(), None, 0, 20).unwrap();
    assert_eq!(overview.unnamed_faces, 1);
    let unnamed = people::list_unnamed_faces_page_cancellable(
        root.path(),
        None,
        0,
        20,
        Arc::new(AtomicBool::new(false)),
    )
    .unwrap();
    assert_eq!(unnamed.total, 1);
    assert_eq!(
        unnamed.items[0].kind,
        people::FaceAssignmentKind::Unassigned
    );
    assert_eq!(
        people::list_people_page(root.path(), None, 0, 20)
            .unwrap()
            .total,
        0
    );

    import_lightroom_inner(root.path(), &lightroom_path, true, ImportCapabilities::ALL).unwrap();
    let reimported = people::load(root.path()).unwrap();
    assert!(
        reimported.assignments.is_empty(),
        "local rejection must suppress a reimported Lightroom confirmation"
    );

    people::confirm(root.path(), people.faces[0].id, people.people[0].id).unwrap();
    let reviewed = people::load(root.path()).unwrap();
    assert!(reviewed.assignments.iter().any(|assignment| {
        assignment.kind == people::FaceAssignmentKind::Confirmed
            && assignment.provenance == "rekesz"
    }));
    import_lightroom_inner(root.path(), &lightroom_path, true, ImportCapabilities::ALL).unwrap();
    let confirmed_reimport = people::load(root.path()).unwrap();
    assert_eq!(confirmed_reimport.assignments.len(), 1);
    assert_eq!(confirmed_reimport.assignments[0].provenance, "rekesz");

    Connection::open(&lightroom_path)
        .unwrap()
        .execute(
            "INSERT INTO AgLibraryKeyword VALUES(21, X'040506', 'Grace', 'person')",
            [],
        )
        .unwrap();
    let moved_catalog = fixture.path().join("renamed.lrcat");
    fs::rename(&lightroom_path, &moved_catalog).unwrap();
    import_lightroom_inner(root.path(), &moved_catalog, true, ImportCapabilities::ALL).unwrap();
    let moved_reimport = people::load(root.path()).unwrap();
    assert_eq!(moved_reimport.faces.len(), 1);
    assert_eq!(moved_reimport.people.len(), 1);
    assert_eq!(moved_reimport.assignments.len(), 1);

    let connection = Connection::open(catalog_path(root.path())).unwrap();
    connection
        .execute_batch(&format!(
            "INSERT OR IGNORE INTO face_assignments(face_id, person_id, kind, provenance, updated_at)
                 VALUES({}, {}, 'confirmed', 'lightroom', datetime('now'));
             INSERT OR REPLACE INTO face_embeddings(face_id, model_id, dimensions, vector,
                 source_size, source_modified_ns, created_at)
                 VALUES({}, 'sface-2021dec', 2, zeroblob(8), 4, 4, datetime('now'));
             DELETE FROM recognition_refresh_queue;",
            people.faces[0].id, people.people[0].id, people.faces[0].id
        ))
        .unwrap();
    drop(connection);
    Connection::open(&moved_catalog)
        .unwrap()
        .execute_batch(
            "DELETE FROM AgLibraryKeywordFace WHERE face=10;
             DELETE FROM AgLibraryFace WHERE id_local=10;",
        )
        .unwrap();
    import_lightroom_inner(root.path(), &moved_catalog, true, ImportCapabilities::ALL).unwrap();
    let connection = Connection::open(catalog_path(root.path())).unwrap();
    assert_eq!(
        connection
            .query_row(
                "SELECT count(*) FROM recognition_refresh_queue
                 WHERE model_id='sface-2021dec' AND status='pending'
                   AND ((entity_kind='person' AND entity_id=?1)
                     OR (entity_kind='face' AND entity_id=?2))",
                params![people.people[0].id, people.faces[0].id],
                |row| row.get::<_, i64>(0),
            )
            .unwrap(),
        2,
        "a disappeared imported exemplar retained by a local decision must invalidate person and face recognition state"
    );
    assert_eq!(
        connection
            .query_row(
                "SELECT retired FROM faces WHERE id=?1",
                [people.faces[0].id],
                |row| row.get::<_, i64>(0),
            )
            .unwrap(),
        0
    );
}

#[test]
fn people_pages_scope_analysis_and_page_multi_person_suggestions_stably() {
    let root = tempfile::tempdir().unwrap();
    fs::write(root.path().join("photo.CR3"), b"test").unwrap();
    let fixture = lightroom_fixture(root.path(), "s={Exposure2012=1}");
    import_lightroom_inner(
        root.path(),
        &fixture.path().join("fixture.lrcat"),
        true,
        ImportCapabilities::ALL,
    )
    .unwrap();

    let connection = Connection::open(root.path().join(CATALOG_DIR).join(CATALOG_FILE)).unwrap();
    for (id, relative) in [(101, "inside/a.jpg"), (102, "outside/b.jpg")] {
        connection
            .execute(
                "INSERT INTO assets(id, relative_path, scope_key) VALUES(?1, ?2, ?3)",
                params![
                    id,
                    encode_relative_path(Path::new(relative)).unwrap(),
                    scope_key(Path::new(relative)).unwrap()
                ],
            )
            .unwrap();
    }
    connection
        .execute_batch(
            "INSERT INTO people(id, display_name, created_at, updated_at)
                 VALUES(101, 'Alpha', '2020-01-01', '2020-01-01'),
                       (102, 'beta', '2020-01-01', '2020-01-01'),
                       (103, 'Zed', '2020-01-01', '2022-01-01');
             INSERT INTO faces(id, asset_id, provenance, source_key, left, top, right, bottom,
                               origin, ignored, updated_at)
                 VALUES(101, 101, 'rekesz', 'test-101', .1, .1, .3, .3, 'rekesz', 0, '2020-01-01'),
                       (102, 101, 'rekesz', 'test-102', .1, .1, .3, .3, 'rekesz', 0, '2020-01-01'),
                       (103, 101, 'rekesz', 'test-103', .1, .1, .3, .3, 'rekesz', 0, '2020-01-01'),
                       (104, 101, 'rekesz', 'test-104', .1, .1, .3, .3, 'rekesz', 0, '2020-01-01'),
                       (105, 101, 'rekesz', 'test-105', .1, .1, .3, .3, 'rekesz', 0, '2020-01-01');
             INSERT INTO face_assignments(face_id, person_id, kind, provenance, confidence, updated_at)
                 VALUES(101, 101, 'confirmed', 'rekesz', NULL, '2020-01-01'),
                       (102, 101, 'confirmed', 'rekesz', NULL, '2020-01-01'),
                       (103, 102, 'confirmed', 'rekesz', NULL, '2020-01-01'),
                       (104, 103, 'confirmed', 'rekesz', NULL, '2020-01-01'),
                       (105, 101, 'suggested', 'rekesz', .9, '2020-01-01'),
                       (105, 102, 'suggested', 'rekesz', .8, '2020-01-01');
             INSERT INTO face_analysis_state(asset_id, detector_model_id, recognizer_model_id,
                                              source_size, source_modified_ns, status, attempts, updated_at)
                 VALUES(101, 'detector', 'recognizer', 1, 1, 'pending', 0, '2020-01-01'),
                       (102, 'detector', 'recognizer', 1, 1, 'error', 1, '2020-01-01');",
        )
        .unwrap();

    let subtree = root.path().join("inside");
    let overview = people::list_people_page(root.path(), Some(&subtree), 0, 20).unwrap();
    assert_eq!(overview.analysis_pending, 1);
    assert_eq!(overview.analysis_errors, 0);
    assert_eq!(overview.review_faces, 2);

    let first = people::list_review_queue_page_cancellable(
        root.path(),
        Some(&subtree),
        0,
        1,
        Arc::new(AtomicBool::new(false)),
    )
    .unwrap();
    let second = people::list_review_queue_page_cancellable(
        root.path(),
        Some(&subtree),
        1,
        1,
        Arc::new(AtomicBool::new(false)),
    )
    .unwrap();
    assert_eq!(first.total, 2);
    assert_eq!(first.items[0].face_id, 105);
    assert_eq!(first.items[0].person_id, 101);
    assert_eq!(first.items[0].person_name.as_deref(), Some("Alpha"));
    assert_eq!(second.items[0].face_id, 105);
    assert_eq!(second.items[0].person_id, 102);
    assert_eq!(second.items[0].person_name.as_deref(), Some("beta"));

    let recent = people::list_people_page_sorted(
        root.path(),
        Some(&subtree),
        people::PeopleSort::RecentlyChanged,
        0,
        20,
    )
    .unwrap();
    assert_eq!(
        recent
            .items
            .iter()
            .map(|person| person.id)
            .collect::<Vec<_>>(),
        [103, 101, 102]
    );
    let confirmed = people::list_people_page_sorted(
        root.path(),
        Some(&subtree),
        people::PeopleSort::Confirmed,
        0,
        20,
    )
    .unwrap();
    assert_eq!(
        confirmed
            .items
            .iter()
            .map(|person| person.id)
            .collect::<Vec<_>>(),
        [101, 102, 103]
    );

    // Differential oracle: dirty state forces the authoritative effective-view
    // path; a rebuild then serves the same requests exclusively from v5 rows.
    connection
        .execute_batch(
            "INSERT INTO people(id,display_name,created_at,updated_at)
               VALUES(104,'No faces','2020-01-01','2020-01-01'),
                     (110,'İpek','2020-01-01','2020-01-01'),
                     (111,'Åke','2020-01-01','2020-01-01'),
                     (112,'Retired confirmed only','2020-01-01','2020-01-01'),
                     (113,'Retired suggested only','2020-01-01','2020-01-01');
             INSERT INTO faces(id,asset_id,provenance,source_key,left,top,right,bottom,origin,
                               retired,ignored,locally_ignored,updated_at)
               VALUES(106,101,'rekesz','retired',.1,.1,.3,.3,'rekesz',1,0,0,'2020-01-01'),
                     (107,101,'rekesz','ignored',.1,.1,.3,.3,'rekesz',0,1,0,'2020-01-01'),
                     (108,101,'rekesz','local-ignore',.1,.1,.3,.3,'rekesz',0,0,1,'2020-01-01'),
                     (109,101,'rekesz','generation',.1,.1,.3,.3,'rekesz',0,0,0,'2020-01-01'),
                     (110,101,'rekesz','unicode-1',.1,.1,.3,.3,'rekesz',0,0,0,'2020-01-01'),
                     (111,101,'rekesz','unicode-2',.1,.1,.3,.3,'rekesz',0,0,0,'2020-01-01'),
                     (112,101,'rekesz','retired-confirmed-only',.1,.1,.3,.3,'rekesz',1,0,0,'2020-01-01'),
                     (113,101,'rekesz','retired-suggested-only',.1,.1,.3,.3,'rekesz',1,0,0,'2020-01-01');
             INSERT INTO face_assignments(face_id,person_id,kind,provenance,confidence,updated_at)
               VALUES(106,101,'confirmed','rekesz',NULL,'2020-01-01'),
                     (107,101,'suggested','rekesz',.7,'2020-01-01'),
                     (108,102,'confirmed','rekesz',NULL,'2020-01-01'),
                     (109,103,'suggested','rekesz',.95,'2020-01-01'),
                     (110,110,'confirmed','rekesz',NULL,'2020-01-01'),
                     (111,111,'confirmed','rekesz',NULL,'2020-01-01'),
                     (112,112,'confirmed','rekesz',NULL,'2020-01-01'),
                     (113,113,'suggested','rekesz',.75,'2020-01-01');
             INSERT INTO recognition_index_state(model_id,generation,index_version,updated_at)
               VALUES('differential-model',7,1,'2020-01-01');
             INSERT INTO recognition_suggestions(face_id,person_id,recognizer_model_id,
               prototype_generation,prototype_algorithm_version,ann_index_version,
               scored_index_generation,policy_version,score,nearest_exemplar_score,
               second_best_margin,created_at)
               VALUES(109,103,'differential-model',1,1,1,7,1,.95,.95,.2,'2020-01-01');",
        )
        .unwrap();
    drop(connection);
    let scopes = [None, Some(subtree.as_path())];
    let sorts = [
        people::PeopleSort::Name,
        people::PeopleSort::Confirmed,
        people::PeopleSort::Review,
        people::PeopleSort::RecentlyChanged,
    ];
    let pages = [(0, 1), (1, 1), (0, 20)];
    let mut authoritative = Vec::new();
    for scope in scopes {
        for sort in sorts {
            for (offset, limit) in pages {
                for representative_offsets in [HashMap::new(), HashMap::from([(101_i64, 1_u64)])] {
                    authoritative.push(
                        people::list_people_page_sorted_cancellable(
                            root.path(),
                            scope,
                            sort,
                            offset,
                            limit,
                            representative_offsets,
                            Arc::new(AtomicBool::new(false)),
                        )
                        .unwrap(),
                    );
                }
            }
        }
    }
    assert_eq!(
        authoritative[4]
            .items
            .iter()
            .map(|person| person.name.as_str())
            .collect::<Vec<_>>(),
        ["Alpha", "beta", "İpek", "Zed", "Åke"]
    );
    people::rebuild_people_summary(root.path(), Arc::new(AtomicBool::new(false))).unwrap();
    let mut index = 0;
    for scope in scopes {
        for sort in sorts {
            for (offset, limit) in pages {
                for representative_offsets in [HashMap::new(), HashMap::from([(101_i64, 1_u64)])] {
                    let summarized = people::list_people_page_sorted_cancellable(
                        root.path(),
                        scope,
                        sort,
                        offset,
                        limit,
                        representative_offsets,
                        Arc::new(AtomicBool::new(false)),
                    )
                    .unwrap();
                    assert_eq!(summarized, authoritative[index]);
                    index += 1;
                }
            }
        }
    }
    let retired_hidden =
        people::list_people_page_sorted(root.path(), None, people::PeopleSort::Name, 0, 50)
            .unwrap();
    assert!(!retired_hidden.items.iter().any(|person| person.id == 112));
    assert!(!retired_hidden.items.iter().any(|person| person.id == 113));
    assert!(
        !people::list_review_queue_page_cancellable(
            root.path(),
            None,
            0,
            50,
            Arc::new(AtomicBool::new(false)),
        )
        .unwrap()
        .items
        .iter()
        .any(|face| face.face_id == 113)
    );

    // Exercise both transitions. Each UPDATE dirties the summary, so the
    // first observations are authoritative fallback results; after rebuild,
    // the exact same requests must be served by the fresh materialization.
    for retired in [0_i64, 1_i64] {
        Connection::open(catalog_path(root.path()))
            .unwrap()
            .execute(
                "UPDATE faces SET retired=?1 WHERE id IN (112,113)",
                [retired],
            )
            .unwrap();
        let live_overview =
            people::list_people_page_sorted(root.path(), None, people::PeopleSort::Name, 0, 50)
                .unwrap();
        let live_review = people::list_review_queue_page_cancellable(
            root.path(),
            None,
            0,
            50,
            Arc::new(AtomicBool::new(false)),
        )
        .unwrap();
        assert_eq!(
            live_overview.items.iter().any(|person| person.id == 112),
            retired == 0
        );
        assert_eq!(
            live_overview.items.iter().any(|person| person.id == 113),
            retired == 0
        );
        assert_eq!(
            live_review.items.iter().any(|face| face.face_id == 113),
            retired == 0
        );
        people::rebuild_people_summary(root.path(), Arc::new(AtomicBool::new(false))).unwrap();
        assert_eq!(
            people::list_people_page_sorted(root.path(), None, people::PeopleSort::Name, 0, 50,)
                .unwrap(),
            live_overview
        );
        assert_eq!(
            people::list_review_queue_page_cancellable(
                root.path(),
                None,
                0,
                50,
                Arc::new(AtomicBool::new(false)),
            )
            .unwrap(),
            live_review
        );
    }
    let connection = Connection::open(catalog_path(root.path())).unwrap();
    connection
        .execute(
            "UPDATE recognition_index_state SET generation=8 WHERE model_id='differential-model'",
            [],
        )
        .unwrap();
    drop(connection);
    let stale_generation = people::list_review_queue_page_cancellable(
        root.path(),
        None,
        0,
        20,
        Arc::new(AtomicBool::new(false)),
    )
    .unwrap();
    assert!(
        !stale_generation
            .items
            .iter()
            .any(|face| face.face_id == 109)
    );
    people::rebuild_people_summary(root.path(), Arc::new(AtomicBool::new(false))).unwrap();
    assert_eq!(
        people::list_review_queue_page_cancellable(
            root.path(),
            None,
            0,
            20,
            Arc::new(AtomicBool::new(false)),
        )
        .unwrap(),
        stale_generation
    );
    for seed in 0_i64..8 {
        let connection = Connection::open(catalog_path(root.path())).unwrap();
        connection
            .execute(
                "UPDATE faces SET ignored=((id+?1)%5=0), locally_ignored=((id+?1)%7=0)
                 WHERE id BETWEEN 101 AND 111",
                [seed],
            )
            .unwrap();
        drop(connection);
        let mut oracle = Vec::new();
        for sort in sorts {
            for (offset, limit) in [(0, 2), (2, 2), (0, 30)] {
                oracle.push(
                    people::list_people_page_sorted(
                        root.path(),
                        Some(&subtree),
                        sort,
                        offset,
                        limit,
                    )
                    .unwrap(),
                );
            }
        }
        people::rebuild_people_summary(root.path(), Arc::new(AtomicBool::new(false))).unwrap();
        let mut index = 0;
        for sort in sorts {
            for (offset, limit) in [(0, 2), (2, 2), (0, 30)] {
                assert_eq!(
                    people::list_people_page_sorted(
                        root.path(),
                        Some(&subtree),
                        sort,
                        offset,
                        limit
                    )
                    .unwrap(),
                    oracle[index],
                    "seed={seed} sort={sort:?} offset={offset}"
                );
                index += 1;
            }
        }
    }
}

#[test]
fn decision_batches_are_atomic_idempotent_undoable_and_survive_reimport() {
    let root = tempfile::tempdir().unwrap();
    fs::write(root.path().join("photo.CR3"), b"test").unwrap();
    let fixture = lightroom_fixture(root.path(), "s={Exposure2012=1}");
    let lightroom = fixture.path().join("fixture.lrcat");
    import_lightroom_inner(root.path(), &lightroom, true, ImportCapabilities::ALL).unwrap();
    let connection = Connection::open(root.path().join(CATALOG_DIR).join(CATALOG_FILE)).unwrap();
    connection
        .execute_batch(
            "INSERT INTO people(id, display_name, created_at, updated_at)
                 VALUES(201, 'Ada', '2020-01-01', '2020-01-01'),
                       (202, 'Grace', '2020-01-01', '2020-01-01');
             INSERT INTO faces(id, asset_id, provenance, source_key, left, top, right, bottom,
                               origin, ignored, updated_at)
                 VALUES(201, 1, 'rekesz', 'decision-201', .1, .1, .3, .3, 'rekesz', 0, '2020-01-01'),
                       (202, 1, 'rekesz', 'decision-202', .4, .4, .6, .6, 'rekesz', 0, '2020-01-01');
             INSERT INTO face_assignments(face_id, person_id, kind, provenance, confidence, updated_at)
                 VALUES(201, 201, 'suggested', 'rekesz', .8, '2020-01-01');",
        )
        .unwrap();
    drop(connection);

    let invalid = people::DecisionBatch {
        id: "invalid-atomic".into(),
        commands: vec![
            people::DecisionCommand {
                face_id: 201,
                person_id: Some(201),
                action: people::DecisionAction::Confirm,
            },
            people::DecisionCommand {
                face_id: 999,
                person_id: Some(201),
                action: people::DecisionAction::Confirm,
            },
        ],
    };
    assert!(people::apply_decisions(root.path(), &invalid).is_err());
    let connection = Connection::open(root.path().join(CATALOG_DIR).join(CATALOG_FILE)).unwrap();
    let local_confirmations: i64 = connection
        .query_row(
            "SELECT count(*) FROM face_assignments WHERE provenance='rekesz' AND kind='confirmed'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(
        local_confirmations, 0,
        "an invalid row rolls back the batch"
    );
    drop(connection);

    let batch = people::DecisionBatch {
        id: "review-1".into(),
        commands: vec![
            people::DecisionCommand {
                face_id: 201,
                person_id: Some(201),
                action: people::DecisionAction::Confirm,
            },
            people::DecisionCommand {
                face_id: 202,
                person_id: None,
                action: people::DecisionAction::Ignore,
            },
        ],
    };
    let applied = people::apply_decisions(root.path(), &batch).unwrap();
    assert!(applied.applied);
    assert_eq!(applied.faces, [201, 202]);
    assert!(
        !people::apply_decisions(root.path(), &batch)
            .unwrap()
            .applied
    );
    let connection = Connection::open(root.path().join(CATALOG_DIR).join(CATALOG_FILE)).unwrap();
    assert_eq!(
        connection
            .query_row(
                "SELECT count(*) FROM face_decision_journal WHERE batch_id='review-1'",
                [],
                |row| row.get::<_, i64>(0)
            )
            .unwrap(),
        2
    );
    drop(connection);

    assert_eq!(
        people::undo_most_recent_decision_batch(root.path()).unwrap(),
        Some("review-1".into())
    );
    assert_eq!(
        people::undo_most_recent_decision_batch(root.path()).unwrap(),
        None
    );
    let restored = people::list_person_faces_page(root.path(), 201, None, 0, 20).unwrap();
    assert_eq!(
        restored.items[0].kind,
        people::FaceAssignmentKind::Suggested
    );
    let connection = Connection::open(root.path().join(CATALOG_DIR).join(CATALOG_FILE)).unwrap();
    assert_eq!(
        connection
            .query_row(
                "SELECT locally_ignored FROM faces WHERE id=202",
                [],
                |row| { row.get::<_, i64>(0) }
            )
            .unwrap(),
        0
    );
    drop(connection);

    for (id, command) in [
        (
            "grouped/chunk/0",
            people::DecisionCommand {
                face_id: 201,
                person_id: Some(201),
                action: people::DecisionAction::Confirm,
            },
        ),
        (
            "grouped/chunk/1",
            people::DecisionCommand {
                face_id: 202,
                person_id: None,
                action: people::DecisionAction::Ignore,
            },
        ),
    ] {
        people::apply_decisions(
            root.path(),
            &people::DecisionBatch {
                id: id.into(),
                commands: vec![command],
            },
        )
        .unwrap();
    }
    assert_eq!(
        people::undo_most_recent_decision_batch(root.path()).unwrap(),
        Some("grouped".into())
    );
    let connection = Connection::open(root.path().join(CATALOG_DIR).join(CATALOG_FILE)).unwrap();
    assert_eq!(
        connection
            .query_row(
                "SELECT count(*) FROM face_decision_journal WHERE batch_id LIKE 'grouped/chunk/%'",
                [],
                |row| row.get::<_, i64>(0),
            )
            .unwrap(),
        0
    );
    assert_eq!(
        connection
            .query_row(
                "SELECT locally_ignored FROM faces WHERE id=202",
                [],
                |row| { row.get::<_, i64>(0) }
            )
            .unwrap(),
        0
    );
    drop(connection);

    let rejection = people::DecisionBatch {
        id: "review-2".into(),
        commands: vec![people::DecisionCommand {
            face_id: 201,
            person_id: Some(201),
            action: people::DecisionAction::Reject,
        }],
    };
    people::apply_decisions(root.path(), &rejection).unwrap();
    import_lightroom_inner(root.path(), &lightroom, true, ImportCapabilities::ALL).unwrap();
    let connection = Connection::open(root.path().join(CATALOG_DIR).join(CATALOG_FILE)).unwrap();
    assert_eq!(
        connection
            .query_row(
                "SELECT count(*) FROM face_assignments
                 WHERE face_id=201 AND person_id=201 AND kind='rejected' AND provenance='rekesz'",
                [],
                |row| row.get::<_, i64>(0)
            )
            .unwrap(),
        1
    );
}

#[test]
fn decisions_invalidate_and_undo_manual_representatives_and_touch_all_people() {
    let root = tempfile::tempdir().unwrap();
    fs::write(root.path().join("photo.CR3"), b"test").unwrap();
    let fixture = lightroom_fixture(root.path(), "s={Exposure2012=1}");
    import_lightroom_inner(
        root.path(),
        &fixture.path().join("fixture.lrcat"),
        true,
        ImportCapabilities::ALL,
    )
    .unwrap();
    let catalog = root.path().join(CATALOG_DIR).join(CATALOG_FILE);
    let connection = Connection::open(&catalog).unwrap();
    connection
        .execute_batch(
            "INSERT INTO people(id, display_name, created_at, updated_at)
                 VALUES(401, 'Old', '2000-01-01', '2000-01-01'),
                       (402, 'New', '2000-01-01', '2000-01-01');
             INSERT INTO faces(id, asset_id, provenance, source_key, left, top, right, bottom,
                               origin, ignored, updated_at)
                 VALUES(401, 1, 'rekesz', 'representative-401', .1, .1, .3, .3,
                        'rekesz', 0, '2000-01-01');
             INSERT INTO face_assignments(face_id, person_id, kind, provenance, updated_at)
                 VALUES(401, 401, 'confirmed', 'rekesz', '2000-01-01');
             INSERT INTO face_embeddings(face_id, model_id, dimensions, vector, created_at)
                 VALUES(401, 'decision-test-model', 2, X'0000803f00000000', '2000-01-01');
             UPDATE people SET representative_face_id=401, representative_is_manual=1
                 WHERE id=401;",
        )
        .unwrap();
    drop(connection);

    people::apply_decisions(
        root.path(),
        &people::DecisionBatch {
            id: "representative-reject".into(),
            commands: vec![people::DecisionCommand {
                face_id: 401,
                person_id: Some(401),
                action: people::DecisionAction::Reject,
            }],
        },
    )
    .unwrap();
    let connection = Connection::open(&catalog).unwrap();
    assert_eq!(
        connection
            .query_row(
                "SELECT representative_face_id FROM people WHERE id=401",
                [],
                |row| row.get::<_, Option<i64>>(0),
            )
            .unwrap(),
        None
    );
    drop(connection);
    people::undo_most_recent_decision_batch(root.path()).unwrap();
    let connection = Connection::open(&catalog).unwrap();
    assert_eq!(
        connection
            .query_row(
                "SELECT count(*) FROM recognition_refresh_queue
                 WHERE model_id='decision-test-model' AND status='pending'",
                [],
                |row| row.get::<_, i64>(0),
            )
            .unwrap(),
        3,
        "undo must enqueue the affected person, face, and coarse catalog rescore marker"
    );
    assert_eq!(
        connection
            .query_row(
                "SELECT representative_face_id, representative_is_manual FROM people WHERE id=401",
                [],
                |row| Ok((row.get::<_, Option<i64>>(0)?, row.get::<_, i64>(1)?)),
            )
            .unwrap(),
        (Some(401), 1)
    );
    drop(connection);
    people::apply_decisions(
        root.path(),
        &people::DecisionBatch {
            id: "representative-ignore".into(),
            commands: vec![people::DecisionCommand {
                face_id: 401,
                person_id: None,
                action: people::DecisionAction::Ignore,
            }],
        },
    )
    .unwrap();
    let connection = Connection::open(&catalog).unwrap();
    assert_eq!(
        connection
            .query_row(
                "SELECT representative_face_id FROM people WHERE id=401",
                [],
                |row| row.get::<_, Option<i64>>(0),
            )
            .unwrap(),
        None
    );
    drop(connection);
    people::undo_most_recent_decision_batch(root.path()).unwrap();
    let connection = Connection::open(&catalog).unwrap();
    assert_eq!(
        connection
            .query_row(
                "SELECT representative_face_id FROM people WHERE id=401",
                [],
                |row| row.get::<_, Option<i64>>(0),
            )
            .unwrap(),
        Some(401)
    );
    connection
        .execute(
            "UPDATE people SET updated_at='2000-01-01' WHERE id IN (401,402)",
            [],
        )
        .unwrap();
    drop(connection);

    people::apply_decisions(
        root.path(),
        &people::DecisionBatch {
            id: "representative-move".into(),
            commands: vec![people::DecisionCommand {
                face_id: 401,
                person_id: Some(402),
                action: people::DecisionAction::Confirm,
            }],
        },
    )
    .unwrap();
    let connection = Connection::open(catalog).unwrap();
    let changed: (String, String, Option<i64>) = connection
        .query_row(
            "SELECT old.updated_at, new.updated_at, old.representative_face_id
             FROM people old JOIN people new ON new.id=402 WHERE old.id=401",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .unwrap();
    assert_ne!(changed.0, "2000-01-01");
    assert_ne!(changed.1, "2000-01-01");
    assert_eq!(changed.2, None);
}

#[test]
fn person_operations_validate_and_preserve_confirmed_ownership() {
    let root = tempfile::tempdir().unwrap();
    fs::write(root.path().join("photo.CR3"), b"test").unwrap();
    let fixture = lightroom_fixture(root.path(), "s={Exposure2012=1}");
    import_lightroom_inner(
        root.path(),
        &fixture.path().join("fixture.lrcat"),
        true,
        ImportCapabilities::ALL,
    )
    .unwrap();
    let connection = Connection::open(root.path().join(CATALOG_DIR).join(CATALOG_FILE)).unwrap();
    connection
        .execute_batch(
            "INSERT INTO faces(id, asset_id, provenance, source_key, left, top, right, bottom,
                               origin, ignored, updated_at)
                 VALUES(301, 1, 'rekesz', 'person-301', .1, .1, .3, .3, 'rekesz', 0, '2020-01-01'),
                       (302, 1, 'rekesz', 'person-302', .4, .4, .6, .6, 'rekesz', 0, '2020-01-01'),
                       (303, 1, 'rekesz', 'person-303', .2, .2, .5, .5, 'rekesz', 0, '2020-01-01');",
        )
        .unwrap();
    drop(connection);

    assert!(people::create_person_from_faces(root.path(), " ", &[301]).is_err());
    assert!(people::create_person_from_faces(root.path(), "Ada", &[301, 999]).is_err());
    let ada = people::create_person_from_faces(root.path(), "Ada", &[301, 302]).unwrap();
    people::rename_person(root.path(), ada, "Ada Lovelace").unwrap();
    people::choose_person_representative(root.path(), ada, 302).unwrap();
    assert!(people::choose_person_representative(root.path(), ada, 999).is_err());
    let grace = people::create_person_from_faces(root.path(), "Grace", &[301]).unwrap();
    people::assign_faces_to_person(root.path(), grace, &[303]).unwrap();
    assert!(people::assign_faces_to_person(root.path(), 999, &[303]).is_err());
    people::move_confirmed_faces(root.path(), ada, grace, &[302]).unwrap();
    assert!(people::move_confirmed_faces(root.path(), ada, grace, &[302]).is_err());
    people::merge_people(root.path(), ada, grace).unwrap();
    assert!(people::merge_people(root.path(), grace, grace).is_err());

    let connection = Connection::open(root.path().join(CATALOG_DIR).join(CATALOG_FILE)).unwrap();
    assert_eq!(
        connection
            .query_row("SELECT count(*) FROM people WHERE id=?1", [ada], |row| {
                row.get::<_, i64>(0)
            })
            .unwrap(),
        0
    );
    assert_eq!(
        connection
            .query_row(
                "SELECT count(*) FROM effective_face_assignments
                 WHERE person_id=?1 AND kind='confirmed'",
                [grace],
                |row| row.get::<_, i64>(0)
            )
            .unwrap(),
        3
    );
}

#[test]
fn generated_people_catalog_profile() {
    run_generated_people_catalog_profile(10_000, Duration::from_millis(500));
}

#[test]
#[ignore = "release-gate profile; run using the documented release command"]
fn generated_people_catalog_profile_60k() {
    run_generated_people_catalog_profile(60_000, Duration::from_millis(50));
}

#[test]
#[ignore = "release-gate profile; run using the documented release command"]
fn generated_people_catalog_profile_250k() {
    run_generated_people_catalog_profile(250_000, Duration::from_millis(100));
}

#[test]
fn representative_quality_order_is_identical_in_summary_and_live_fallback() {
    let root = tempfile::tempdir().unwrap();
    initialize_test_catalog(root.path()).unwrap();
    let encode = |points: [[f32; 2]; 5]| {
        points
            .into_iter()
            .flatten()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>()
    };
    let frontal = encode([
        [0.36, 0.38],
        [0.64, 0.38],
        [0.50, 0.53],
        [0.41, 0.67],
        [0.59, 0.67],
    ]);
    let profile = encode([
        [0.26, 0.32],
        [0.52, 0.46],
        [0.61, 0.55],
        [0.40, 0.65],
        [0.55, 0.72],
    ]);
    let catalog = catalog_path(root.path());
    let connection = Connection::open(&catalog).unwrap();
    connection
        .execute(
            "INSERT INTO assets(id,relative_path,scope_key) VALUES(1,?1,X'')",
            [encode_relative_path(Path::new("photo.jpg")).unwrap()],
        )
        .unwrap();
    connection
        .execute_batch(
            "INSERT INTO people(id,display_name,created_at,updated_at) VALUES(1,'Person','x','x');
             INSERT INTO faces(id,asset_id,provenance,source_key,left,top,right,bottom,origin,
               detection_confidence,updated_at) VALUES
               (1,1,'rekesz','profile',.1,.1,.9,.9,'rekesz',.9,'2030-01-02'),
               (2,1,'rekesz','frontal',.1,.1,.9,.9,'rekesz',.9,'2030-01-01');
             INSERT INTO face_assignments(face_id,person_id,kind,provenance,updated_at) VALUES
               (1,1,'confirmed','rekesz','x'),(2,1,'confirmed','rekesz','x');",
        )
        .unwrap();
    for (face_id, landmarks) in [(1, profile), (2, frontal)] {
        connection
            .execute(
                "INSERT INTO face_embeddings(face_id,model_id,dimensions,vector,landmarks,created_at)
                 VALUES(?1,'sface-2021dec',1,X'00000000',?2,'x')",
                params![face_id, landmarks],
            )
            .unwrap();
    }
    drop(connection);

    people::rebuild_people_summary(root.path(), Arc::new(AtomicBool::new(false))).unwrap();
    let summary = people::list_people_page(root.path(), None, 0, 30).unwrap();
    assert_eq!(summary.items[0].representative_face_id, Some(2));

    // This invalidates the summary; the exact fallback must retain the same
    // quality ordering while the rebuild is pending.
    Connection::open(&catalog)
        .unwrap()
        .execute("UPDATE people SET updated_at='2031' WHERE id=1", [])
        .unwrap();
    let fallback = people::list_people_page(root.path(), None, 0, 30).unwrap();
    assert_eq!(fallback.items[0].representative_face_id, Some(2));

    // A stored automatic/cached representative is not a user preference. Both
    // the live fallback and the next summary rebuild must continue to apply the
    // current quality policy unless the manual bit is set.
    Connection::open(&catalog)
        .unwrap()
        .execute(
            "UPDATE people SET representative_face_id=1,representative_is_manual=0 WHERE id=1",
            [],
        )
        .unwrap();
    assert_eq!(
        people::list_people_page(root.path(), None, 0, 30)
            .unwrap()
            .items[0]
            .representative_face_id,
        Some(2)
    );
    people::rebuild_people_summary(root.path(), Arc::new(AtomicBool::new(false))).unwrap();
    assert_eq!(
        people::list_people_page(root.path(), None, 0, 30)
            .unwrap()
            .items[0]
            .representative_face_id,
        Some(2)
    );

    // Confidence precedes pose quality, and an explicit manual choice precedes
    // every automatic quality dimension.
    let connection = Connection::open(&catalog).unwrap();
    connection
        .execute("UPDATE faces SET detection_confidence=.95 WHERE id=1", [])
        .unwrap();
    drop(connection);
    assert_eq!(
        people::list_people_page(root.path(), None, 0, 30)
            .unwrap()
            .items[0]
            .representative_face_id,
        Some(1)
    );
    Connection::open(&catalog)
        .unwrap()
        .execute(
            "UPDATE people SET representative_face_id=2,representative_is_manual=1 WHERE id=1",
            [],
        )
        .unwrap();
    assert_eq!(
        people::list_people_page(root.path(), None, 0, 30)
            .unwrap()
            .items[0]
            .representative_face_id,
        Some(2)
    );
}

#[test]
#[ignore = "tagged reference-host resource gate; run through scripts/people-release-validation.sh"]
fn reference_people_resource_profile() {
    use std::time::Instant;

    fn environment_limit(name: &str, default: f64) -> f64 {
        std::env::var(name)
            .ok()
            .map(|value| {
                value
                    .parse::<f64>()
                    .unwrap_or_else(|_| panic!("{name} must be a number, got {value:?}"))
            })
            .unwrap_or(default)
    }

    fn cpu_seconds() -> f64 {
        let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
        // SAFETY: getrusage initializes the provided rusage structure on success.
        let result = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
        assert_eq!(result, 0, "getrusage failed");
        // SAFETY: the successful call above initialized the structure.
        let usage = unsafe { usage.assume_init() };
        let timeval_seconds =
            |value: libc::timeval| value.tv_sec as f64 + value.tv_usec as f64 / 1_000_000.0;
        timeval_seconds(usage.ru_utime) + timeval_seconds(usage.ru_stime)
    }

    let reference_tag = std::env::var("PEOPLE_REFERENCE_TAG")
        .expect("PEOPLE_REFERENCE_TAG must identify the reference host/configuration");
    let artifact = std::env::var("PEOPLE_RESOURCE_ARTIFACT")
        .expect("PEOPLE_RESOURCE_ARTIFACT must name the JSON result artifact");
    let max_rss_kib = environment_limit("PEOPLE_MAX_RSS_KIB", 2_097_152.0);
    let max_cpu_percent = environment_limit("PEOPLE_MAX_CPU_PERCENT", 400.0);
    let max_cancel_ms = environment_limit("PEOPLE_MAX_CANCEL_LATENCY_MS", 100.0);

    let _ort = crate::face_engine::ORT_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let analysis_root = tempfile::tempdir().unwrap();
    let sources = [
        analysis_root.path().join("running.png"),
        analysis_root.path().join("queued.png"),
    ];
    for source in &sources {
        image::RgbImage::new(256, 256).save(source).unwrap();
    }
    initialize_test_catalog(analysis_root.path()).unwrap();
    crate::face_engine::sync_analysis_inventory(analysis_root.path(), &sources).unwrap();
    let jobs = crate::face_engine::pending_analysis_jobs(analysis_root.path(), 2).unwrap();
    let reached = Arc::new((std::sync::Mutex::new(false), std::sync::Condvar::new()));
    let lifecycle = Arc::new(QueryWorkloadLifecycle::default());
    let service = crate::face_engine::FaceAnalysisService::new_with_pipeline_and_stages(
        1,
        crate::jobs::JobLimiter::new(2),
        crate::scheduler::MaintenanceAdmission::new(),
        crate::rendering::pipeline::RenderPipeline::with_source_cache_capacity(
            analysis_root.path().join("resource-cache"),
            1,
            1,
            1,
        ),
        Arc::new(RunningStageBarrier {
            target: crate::face_engine::FaceAnalysisStage::Embedding,
            reached: reached.clone(),
            required_query: Some(lifecycle.clone()),
        }),
        None,
    );
    let cpu_before = cpu_seconds();
    let wall_started = Instant::now();
    let query_lifecycle = lifecycle.clone();
    let query = std::thread::spawn(move || {
        // Two consecutive production workloads keep the representative host
        // busy across model/session warm-up as well as the real detect call.
        for _ in 0..2 {
            run_generated_people_catalog_profile_with_lifecycle(
                60_000,
                Duration::from_millis(1_000),
                Some(&query_lifecycle),
            );
        }
        query_lifecycle.finish();
    });
    assert!(
        lifecycle.wait_until_started(Duration::from_secs(20)),
        "production People workload never entered its first query"
    );
    for job in jobs {
        assert!(service.enqueue(job, 0));
    }
    assert!(wait_for_stage(&reached));
    let workload_started = true;
    let workload_finished = false;
    let inference_overlapped_workload = lifecycle
        .overlap_observed
        .load(std::sync::atomic::Ordering::Acquire);
    let workload_in_progress = usize::from(inference_overlapped_workload);
    assert!(
        inference_overlapped_workload,
        "the production People workload ended before real detection completed"
    );
    let live = service.stats();
    assert_eq!(live.running, 1);
    assert_eq!(live.queued, 1);
    let inference = service.instrumentation();
    assert!(inference.inference_started > 0);
    assert!(inference.inference_completed > 0);

    let mut sampled_peak_rss_kib = 0.0_f64;
    while !query.is_finished() || wall_started.elapsed() < Duration::from_millis(250) {
        let status = fs::read_to_string("/proc/self/status").unwrap_or_default();
        let current = status
            .lines()
            .find_map(|line| line.strip_prefix("VmRSS:"))
            .and_then(|value| value.split_whitespace().next())
            .and_then(|value| value.parse::<f64>().ok())
            .unwrap_or(0.0);
        sampled_peak_rss_kib = sampled_peak_rss_kib.max(current);
        std::thread::sleep(Duration::from_millis(5));
    }
    query.join().unwrap();
    let wall_seconds = wall_started.elapsed().as_secs_f64().max(f64::EPSILON);
    let cpu_percent = (cpu_seconds() - cpu_before) / wall_seconds * 100.0;

    let cancel_started = Instant::now();
    service.set_paused(true);
    while service.stats().running + service.stats().queued > 0
        && cancel_started.elapsed() < Duration::from_secs(5)
    {
        std::thread::yield_now();
    }
    let cancellation_ms = cancel_started.elapsed().as_secs_f64() * 1_000.0;
    assert_eq!(service.stats().failed, 0);
    let connection = Connection::open(catalog_path(analysis_root.path())).unwrap();
    let durable_pending: i64 = connection
        .query_row(
            "SELECT count(*) FROM face_analysis_state WHERE status='pending'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    let partial_faces: i64 = connection
        .query_row(
            "SELECT count(*) FROM faces WHERE provenance='rekesz'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    let partial_embeddings: i64 = connection
        .query_row("SELECT count(*) FROM face_embeddings", [], |row| row.get(0))
        .unwrap();
    let partial_landmarks: i64 = connection
        .query_row(
            "SELECT count(*) FROM face_embeddings WHERE landmarks IS NOT NULL",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(durable_pending, 2);
    assert_eq!(partial_faces, 0);
    assert_eq!(partial_embeddings, 0);
    assert_eq!(partial_landmarks, 0);

    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    // SAFETY: getrusage initializes the provided rusage structure on success.
    assert_eq!(
        unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) },
        0
    );
    // SAFETY: the successful call above initialized the structure.
    let process_peak_rss_kib = unsafe { usage.assume_init() }.ru_maxrss as f64;
    let peak_rss_kib = sampled_peak_rss_kib.max(process_peak_rss_kib);
    let report = serde_json::json!({
        "schema": 1,
        "reference_tag": reference_tag,
        "face_count": 60_000,
        "wall_seconds": wall_seconds,
        "cpu_percent": cpu_percent,
        "peak_rss_kib": peak_rss_kib,
        "cancellation_latency_ms": cancellation_ms,
        "running_jobs_observed": live.running,
        "queued_jobs_observed": live.queued,
        "inference_started": inference.inference_started,
        "inference_completed": inference.inference_completed,
        "inference_overlapped_people_workload": inference_overlapped_workload,
        "people_workload_started_at_inference": workload_started,
        "people_workload_finished_at_inference": workload_finished,
        "people_workload_in_progress_at_inference": workload_in_progress,
        "durable_pending_after_cancel": durable_pending,
        "partial_faces_after_cancel": partial_faces,
        "partial_embeddings_after_cancel": partial_embeddings,
        "partial_landmarks_after_cancel": partial_landmarks,
        "thresholds": {
            "max_cpu_percent": max_cpu_percent,
            "max_rss_kib": max_rss_kib,
            "max_cancellation_latency_ms": max_cancel_ms,
        }
    });
    if let Some(parent) = std::path::Path::new(&artifact).parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(&artifact, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
    eprintln!("PEOPLE_RESOURCE_REPORT={report}");

    assert!(
        peak_rss_kib <= max_rss_kib,
        "peak RSS {peak_rss_kib:.0} KiB exceeds {max_rss_kib:.0} KiB"
    );
    assert!(
        cpu_percent <= max_cpu_percent,
        "CPU {cpu_percent:.1}% exceeds {max_cpu_percent:.1}%"
    );
    assert!(
        cancellation_ms <= max_cancel_ms,
        "cancellation {cancellation_ms:.3} ms exceeds {max_cancel_ms:.3} ms"
    );
}

#[test]
fn people_release_script_contains_all_mandatory_gates() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let script = fs::read_to_string(manifest.join("scripts/people-release-validation.sh")).unwrap();
    for command in [
        "reference_people_resource_profile",
        "generated_people_catalog_profile_60k",
        "generated_people_catalog_profile_250k",
        "long_reimport_restart_cancel_soak",
        "real_detect_and_embed_run_with_socket_syscalls_denied",
        "face_engine_has_no_network_or_map_dependency",
        "migration_rejects_disagreeing_version_markers_without_schema_changes",
        "v1_catalog_migrates_to_current_with_path_backup_and_ann_schema",
        "v2_catalog_migrates_to_v3_with_paths_and_effective_views",
        "v3_recognition_migration_rolls_back_all_columns_on_failure",
        "real_dimension_multi_query_is_exact_and_metric_prunes_ann_candidates",
        "ann_resource_512d_cached_generation_overlap_stays_within_budget",
        "ann_release_100k_512d_cached_overlap_throughput_and_resources",
        "index_waiter_cancels_within_one_hundred_ms_and_other_catalog_progresses",
        "model_build_contention_cancels_within_one_hundred_ms_without_blocking_other_keys",
        "check-people-artifact-privacy.sh",
        "--release",
        "--nocapture",
    ] {
        assert!(script.contains(command), "release script omits {command}");
    }
    assert!(script.contains("PEOPLE_REFERENCE_TAG"));
    assert!(script.contains("PEOPLE_MAX_RSS_KIB"));
    assert!(script.contains("people-resource-limit.sh"));
    assert!(script.contains("run_people_resource_monitored"));
    assert!(script.contains("PEOPLE_MAX_CPU_PERCENT"));
    assert!(script.contains("PEOPLE_MAX_CANCEL_LATENCY_MS"));
    assert!(script.contains("test_artifact_manifest"));
    assert!(script.contains("test-artifacts.before.txt"));
    assert!(script.contains("unexpectedly compiled code"));
    assert!(script.contains("compile-release-tests"));
    assert!(script.contains("--message-format=json"));
    assert!(script.contains("cargo test --quiet --no-run"));
    assert!(script.contains("cargo test --quiet --release --no-run"));
    assert!(script.lines().any(|line| {
        line.trim()
            == "run_gate quality-tests-lightroom \"$debug_lightroom\" --test-threads=1 --nocapture"
    }));
    assert!(script.lines().any(|line| {
        line.trim()
            == "run_gate quality-tests-rekesz \"$debug_rekesz\" --test-threads=1 --nocapture"
    }));
    assert!(script.contains("run_exact_gate recognition-ann-catalog-ceiling ann_release_100k"));
    assert!(script.contains("run_people_exact_test"));
    assert!(!script.contains("run_gate quality-tests cargo test"));
    let resource_helper =
        fs::read_to_string(manifest.join("scripts/people-resource-limit.sh")).unwrap();
    assert!(resource_helper.contains("--exact"));
    assert!(resource_helper.contains("--include-ignored"));
    assert!(resource_helper.contains("--test-threads=1 --nocapture"));
}

#[test]
fn people_resource_limit_parser_enforces_valid_missing_and_nonnumeric_fixtures() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let helper = manifest.join("scripts/people-resource-limit.sh");
    let fixtures = tempfile::tempdir().unwrap();
    let run = |name: &str, contents: &str, budget: &str| {
        let resource = fixtures.path().join(name);
        fs::write(&resource, contents).unwrap();
        std::process::Command::new("bash")
            .args([
                "-c",
                "source \"$1\"; enforce_people_rss_resource_file \"$2\" \"$3\"",
                "people-resource-test",
            ])
            .arg(&helper)
            .arg(resource)
            .arg(budget)
            .status()
            .unwrap()
            .success()
    };
    assert!(run("under.txt", "max_rss_kib=999\n", "1000"));
    assert!(!run("over.txt", "max_rss_kib=1001\n", "1000"));
    assert!(!run("missing.txt", "wall_seconds=1\n", "1000"));
    assert!(!run("nonnumeric.txt", "max_rss_kib=unknown\n", "1000"));

    let portable_bin = fixtures.path().join("bin");
    fs::create_dir(&portable_bin).unwrap();
    assert!(!portable_bin.join("time").exists());
    let supervisor =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/debug/people-resource-supervisor");
    assert!(supervisor.exists(), "test supervisor binary was not built");
    let monitored = |name: &str, command: &str, budget: &str, cpu_budget: &str| {
        let resource = fixtures.path().join(format!("{name}.resources"));
        let log = fixtures.path().join(format!("{name}.log"));
        let status = std::process::Command::new("/bin/bash")
            .args([
                "-c",
                "source \"$1\"; run_people_resource_monitored \"$2\" \"$3\" \"$4\" \"$5\" -c \"$6\"",
                "people-monitor-test",
            ])
            .arg(&helper)
            .arg(&resource)
            .arg(&log)
            .arg(budget)
            .arg("/bin/sh")
            .arg(command)
            .env("PATH", &portable_bin)
            .env("PEOPLE_RESOURCE_SUPERVISOR", &supervisor)
            .env("PEOPLE_MAX_CPU_PERCENT", cpu_budget)
            .status()
            .unwrap();
        (status, resource)
    };
    let (success, resource) = monitored(
        "success",
        "/usr/bin/sleep 0.30; printf fully-flushed",
        "100000",
        "400",
    );
    assert!(success.success());
    let measured = fs::read_to_string(resource).unwrap();
    assert!(measured.contains("wall_seconds="));
    assert!(measured.contains("sampled_group_peak_rss_kib="));
    assert!(measured.contains("max_individual_peak_rss_kib="));
    assert!(measured.contains("max_rss_kib="));
    assert!(measured.contains("cpu_percent="));
    let rss = measured
        .lines()
        .find_map(|line| line.strip_prefix("max_rss_kib="))
        .unwrap()
        .parse::<usize>()
        .unwrap();
    assert!(rss > 0);
    assert_eq!(
        fs::read_to_string(fixtures.path().join("success.log")).unwrap(),
        "fully-flushed"
    );
    let (failure, _) = monitored("failure", "exit 7", "100000", "400");
    assert_eq!(failure.code(), Some(7));
    let (rss_failure, _) = monitored(
        "rss-over",
        "/usr/bin/python3 -c 'x=bytearray(16000000); import time; time.sleep(.1)'",
        "1024",
        "400",
    );
    assert_eq!(rss_failure.code(), Some(1));
    let (aggregate_failure, aggregate_resource) = monitored(
        "rss-aggregate-over",
        "/usr/bin/python3 -c 'import subprocess; p=[subprocess.Popen([\"/usr/bin/python3\",\"-c\",\"x=bytearray(12000000); import time; time.sleep(.4)\"]) for _ in range(2)]; [x.wait() for x in p]'",
        "18000",
        "400",
    );
    assert_eq!(aggregate_failure.code(), Some(1));
    let aggregate = fs::read_to_string(aggregate_resource).unwrap();
    let group_peak = aggregate
        .lines()
        .find_map(|line| line.strip_prefix("sampled_group_peak_rss_kib="))
        .unwrap()
        .parse::<usize>()
        .unwrap();
    let individual_peak = aggregate
        .lines()
        .find_map(|line| line.strip_prefix("max_individual_peak_rss_kib="))
        .unwrap()
        .parse::<usize>()
        .unwrap();
    assert!(group_peak > 18_000);
    assert!(individual_peak < group_peak);
    let (cpu_failure, _) = monitored(
        "cpu-over",
        "/usr/bin/python3 -c 'sum(i*i for i in range(10000000))'",
        "100000",
        "1",
    );
    assert_eq!(cpu_failure.code(), Some(1));

    let descendant_pid = fixtures.path().join("descendant.pid");
    let signal_resources = fixtures.path().join("signal.resources");
    let signal_log = fixtures.path().join("signal.log");
    let mut supervised = std::process::Command::new(&supervisor)
        .args([
            signal_resources.as_os_str(),
            signal_log.as_os_str(),
            std::ffi::OsStr::new("100000"),
            std::ffi::OsStr::new("400"),
            std::ffi::OsStr::new("--"),
            std::ffi::OsStr::new("/bin/sh"),
            std::ffi::OsStr::new("-c"),
        ])
        .arg(format!(
            "sleep 30 & echo $! > '{}'; wait",
            descendant_pid.display()
        ))
        .spawn()
        .unwrap();
    for _ in 0..100 {
        if descendant_pid.exists() {
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    let descendant = fs::read_to_string(&descendant_pid)
        .unwrap()
        .trim()
        .parse::<i32>()
        .unwrap();
    unsafe { libc::kill(supervised.id() as i32, libc::SIGTERM) };
    assert_eq!(supervised.wait().unwrap().code(), Some(128 + libc::SIGTERM));
    assert_eq!(
        unsafe { libc::kill(descendant, 0) },
        -1,
        "descendant survived supervisor termination"
    );

    let immediate_resources = fixtures.path().join("immediate.resources");
    let immediate_log = fixtures.path().join("immediate.log");
    let immediate_marker = fixtures.path().join("immediate.marker");
    let immediate_release = fixtures.path().join("immediate.release");
    let immediate_child_pid = fixtures.path().join("immediate-child.pid");
    let mut immediate = std::process::Command::new(&supervisor)
        .args([
            immediate_resources.as_os_str(),
            immediate_log.as_os_str(),
            std::ffi::OsStr::new("100000"),
            std::ffi::OsStr::new("400"),
            std::ffi::OsStr::new("--"),
            std::ffi::OsStr::new("/bin/sh"),
            std::ffi::OsStr::new("-c"),
            std::ffi::OsStr::new("sleep 30 & wait"),
        ])
        .env("PEOPLE_SUPERVISOR_TEST_PRE_READY_MARKER", &immediate_marker)
        .env(
            "PEOPLE_SUPERVISOR_TEST_PRE_READY_RELEASE",
            &immediate_release,
        )
        .env(
            "PEOPLE_SUPERVISOR_TEST_CHILD_PID_FILE",
            &immediate_child_pid,
        )
        .spawn()
        .unwrap();
    for _ in 0..1_000 {
        if immediate_marker.exists() && immediate_child_pid.exists() {
            break;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    let actual_child_group = fs::read_to_string(&immediate_child_pid)
        .unwrap()
        .parse::<i32>()
        .unwrap();
    unsafe { libc::kill(immediate.id() as i32, libc::SIGTERM) };
    fs::write(&immediate_release, b"release").unwrap();
    assert_eq!(immediate.wait().unwrap().code(), Some(128 + libc::SIGTERM));
    assert_eq!(
        unsafe { libc::kill(-actual_child_group, 0) },
        -1,
        "pre-readiness pending signal left the actual child group alive"
    );

    let (sampler_failure, sampler_resource) = {
        let resource = fixtures.path().join("sampler-failure.resources");
        let log = fixtures.path().join("sampler-failure.log");
        let status = std::process::Command::new(&supervisor)
            .args([
                resource.as_os_str(),
                log.as_os_str(),
                std::ffi::OsStr::new("100000"),
                std::ffi::OsStr::new("400"),
                std::ffi::OsStr::new("--"),
                std::ffi::OsStr::new("/bin/sleep"),
                std::ffi::OsStr::new("30"),
            ])
            .env("PEOPLE_SUPERVISOR_FAIL_SAMPLER", "1")
            .status()
            .unwrap();
        (status, resource)
    };
    assert_eq!(sampler_failure.code(), Some(1));
    assert!(
        fs::read_to_string(sampler_resource)
            .unwrap()
            .contains("sampler_error=injected process-group sampler failure")
    );

    let manifest_repo = fixtures.path().join("manifest-repo");
    fs::create_dir(&manifest_repo).unwrap();
    let manifest_test = std::process::Command::new("/bin/bash")
        .current_dir(&manifest_repo)
        .args([
            "-c",
            "git init -q; git config user.email test@example.invalid; git config user.name test; printf one > input.rs; mkdir bins; printf executable > bins/test-bin; chmod +x bins/test-bin; git add input.rs; git commit -qm initial; source \"$1\"; write_people_test_artifact_manifest before bins; printf two >> input.rs; write_people_test_artifact_manifest after bins; ! cmp before after",
            "people-manifest-test",
        ])
        .arg(&helper)
        .status()
        .unwrap();
    assert!(manifest_test.success());

    let mapped = fixtures.path().join("mapped-tests");
    let mapped_markers = fixtures.path().join("mapped-markers");
    fs::create_dir(&mapped).unwrap();
    fs::create_dir(&mapped_markers).unwrap();
    let first = mapped.join("first");
    let second = mapped.join("second");
    let failing = mapped.join("failing");
    fs::write(&first, "#!/bin/sh\nexit 0\n").unwrap();
    fs::write(&second, "#!/bin/sh\nexit 0\n").unwrap();
    fs::write(&failing, "#!/bin/sh\nexit 7\n").unwrap();
    assert!(
        std::process::Command::new("chmod")
            .args(["+x"])
            .args([&first, &second, &failing])
            .status()
            .unwrap()
            .success()
    );
    let successful_mapping = mapped.join("successful.txt");
    fs::write(
        &successful_mapping,
        format!(
            "lightroom_inventory={}\nrekesz={}\n",
            first.display(),
            second.display()
        ),
    )
    .unwrap();
    let run_mapping = |mapping: &Path| {
        std::process::Command::new("/bin/bash")
            .args([
                "-c",
                "source \"$1\"; run_people_mapped_test_binaries \"$2\" \"$3\"",
                "people-mapped-test",
            ])
            .arg(&helper)
            .arg(mapping)
            .arg(&mapped_markers)
            .status()
            .unwrap()
    };
    assert!(run_mapping(&successful_mapping).success());
    assert!(mapped_markers.join("lightroom_inventory.executed").exists());
    assert!(mapped_markers.join("rekesz.executed").exists());
    let failing_mapping = mapped.join("failing.txt");
    fs::write(&failing_mapping, format!("failure={}\n", failing.display())).unwrap();
    assert_eq!(run_mapping(&failing_mapping).code(), Some(7));

    let exact_fixture = mapped.join("exact-fixture");
    let exact_argv = mapped.join("exact.argv");
    fs::write(
        &exact_fixture,
        "#!/bin/sh\nif [ \"$1\" = --list ]; then printf 'suite::unique\\na::shared\\nb::shared\\n'; exit 0; fi\nprintf '%s\\n' \"$@\" > \"$PEOPLE_EXACT_ARGV\"\n",
    )
    .unwrap();
    assert!(
        std::process::Command::new("chmod")
            .args(["+x"])
            .arg(&exact_fixture)
            .status()
            .unwrap()
            .success()
    );
    let exact = |suffix: &str| {
        std::process::Command::new("/bin/bash")
            .args([
                "-c",
                "source \"$1\"; run_people_exact_test \"$2\" \"$3\" false",
                "people-exact-fixture",
            ])
            .arg(&helper)
            .arg(&exact_fixture)
            .arg(suffix)
            .env("PEOPLE_EXACT_ARGV", &exact_argv)
            .status()
            .unwrap()
    };
    assert!(exact("unique").success());
    let argv = fs::read_to_string(&exact_argv).unwrap();
    assert!(argv.contains("--exact\nunique\n"));
    assert_eq!(argv.matches("unique").count(), 1);
    assert!(!exact("shared").success());
    assert!(!exact("absent").success());

    let mapper =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/debug/people-test-artifacts");
    assert!(mapper.exists());
    let mut duplicate_mapper = std::process::Command::new(mapper)
        .stdin(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    let artifact = r#"{"reason":"compiler-artifact","target":{"name":"duplicate"},"profile":{"test":true},"executable":"/tmp/one"}"#;
    {
        use std::io::Write as _;
        let stdin = duplicate_mapper.stdin.as_mut().unwrap();
        writeln!(stdin, "{artifact}").unwrap();
        writeln!(stdin, "{}", artifact.replace("/tmp/one", "/tmp/two")).unwrap();
    }
    assert!(!duplicate_mapper.wait().unwrap().success());
}

#[test]
fn people_release_privacy_scanner_accepts_aggregates_and_rejects_identifiers() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let scanner = manifest.join("scripts/check-people-artifact-privacy.sh");
    let artifacts = tempfile::tempdir().unwrap();
    let report = artifacts.path().join("reference-resources.json");
    let safe_report = br#"{
        "schema":1,"reference_tag":"safe-host","face_count":60000,"wall_seconds":1.0,
        "cpu_percent":2.0,"peak_rss_kib":1234,"cancellation_latency_ms":3.0,
        "running_jobs_observed":1,"queued_jobs_observed":1,"inference_started":1,
        "inference_completed":1,"inference_overlapped_people_workload":true,
        "people_workload_started_at_inference":true,"people_workload_finished_at_inference":false,
        "people_workload_in_progress_at_inference":1,"durable_pending_after_cancel":2,
        "partial_faces_after_cancel":0,"partial_embeddings_after_cancel":0,
        "partial_landmarks_after_cancel":0,
        "thresholds":{"max_cpu_percent":400,"max_rss_kib":2097152,"max_cancellation_latency_ms":100}
    }"#;
    fs::write(&report, safe_report).unwrap();
    let reference = concat!(
        "reference_tag=safe-host\n",
        "revision=1111111111111111111111111111111111111111\n",
        "worktree_dirty=true\n",
        "worktree_diff_sha256=2222222222222222222222222222222222222222222222222222222222222222\n",
        "worktree_status_sha256=3333333333333333333333333333333333333333333333333333333333333333\n",
        "kernel=Linux test x86_64\n",
        "cpu_count=16\n",
        "max_rss_kib=2097152\n",
        "max_cpu_percent=400\n",
        "max_cancel_latency_ms=100\n",
    );
    fs::write(artifacts.path().join("reference.txt"), reference).unwrap();
    assert!(
        std::process::Command::new(&scanner)
            .arg(artifacts.path())
            .status()
            .unwrap()
            .success()
    );
    fs::remove_file(artifacts.path().join("reference.txt")).unwrap();
    assert!(
        !std::process::Command::new(&scanner)
            .arg(artifacts.path())
            .status()
            .unwrap()
            .success(),
        "privacy scanner accepted a bundle without reference metadata"
    );
    fs::write(artifacts.path().join("reference.txt"), reference).unwrap();

    let fractional_limits = reference
        .replace("max_cpu_percent=400", "max_cpu_percent=400.5")
        .replace("max_cancel_latency_ms=100", "max_cancel_latency_ms=100.25");
    fs::write(artifacts.path().join("reference.txt"), fractional_limits).unwrap();
    fs::write(
        &report,
        String::from_utf8_lossy(safe_report)
            .replace("\"max_cpu_percent\":400", "\"max_cpu_percent\":400.5")
            .replace(
                "\"max_cancellation_latency_ms\":100",
                "\"max_cancellation_latency_ms\":100.25",
            ),
    )
    .unwrap();
    assert!(
        std::process::Command::new(&scanner)
            .arg(artifacts.path())
            .status()
            .unwrap()
            .success(),
        "reference metadata rejected finite positive fractional limits"
    );
    fs::write(artifacts.path().join("reference.txt"), reference).unwrap();
    fs::write(&report, safe_report).unwrap();

    // Compiler/test manifests retain executable paths for reproducibility and
    // are the only root-level files for which those paths are expected.
    fs::write(
        artifacts.path().join("debug-test-executables.txt"),
        b"lightroom-inventory=target/debug/lightroom-inventory\nrekesz=target/debug/rekesz\n",
    )
    .unwrap();
    let safe_manifest = concat!(
        "revision=1111111111111111111111111111111111111111\n",
        "build_input_sha256=2222222222222222222222222222222222222222222222222222222222222222\n",
        "worktree_status_sha256=3333333333333333333333333333333333333333333333333333333333333333\n",
        "4444444444444444444444444444444444444444444444444444444444444444  target/debug/deps/rekesz-deadbeef\n",
        "1234 target/debug/deps/rekesz-deadbeef\n",
    );
    for name in [
        "test-artifacts.before.txt",
        "test-artifacts.current.txt",
        "test-artifacts.after.txt",
    ] {
        fs::write(artifacts.path().join(name), safe_manifest).unwrap();
    }
    assert!(
        std::process::Command::new(&scanner)
            .arg(artifacts.path())
            .status()
            .unwrap()
            .success()
    );

    let nested = artifacts.path().join("retained/reports");
    fs::create_dir_all(&nested).unwrap();
    fs::write(nested.join("recognition.log"), b"embedding=[0.1, 0.2]\n").unwrap();
    assert!(
        !std::process::Command::new(&scanner)
            .arg(artifacts.path())
            .status()
            .unwrap()
            .success()
    );
    fs::remove_file(nested.join("recognition.log")).unwrap();

    fs::write(
        nested.join("worker.log"),
        b"opened /srv/photos/private.JPG\n",
    )
    .unwrap();
    assert!(
        !std::process::Command::new(&scanner)
            .arg(artifacts.path())
            .status()
            .unwrap()
            .success()
    );
    fs::remove_file(nested.join("worker.log")).unwrap();

    fs::write(
        nested.join("debug-test-executables.txt"),
        b"source_path=/private/library/original.raw\n",
    )
    .unwrap();
    assert!(
        !std::process::Command::new(&scanner)
            .arg(artifacts.path())
            .status()
            .unwrap()
            .success()
    );
    fs::remove_file(nested.join("debug-test-executables.txt")).unwrap();

    for (name, payload) in [
        ("camel.log", b"faceEmbedding: [0.1, 0.2]".as_slice()),
        ("prose.log", b"source path /opt/photos/a.raw".as_slice()),
        ("utf16.log", b"s\0o\0u\0r\0c\0e\0P\0a\0t\0h\0".as_slice()),
        ("archive.gz", b"\x1f\x8bpayload".as_slice()),
        ("pdf.log", b"%PDF-1.7 disguised".as_slice()),
        ("zip.log", b"PK\x03\x04disguised".as_slice()),
        ("png.log", b"\x89PNG\r\n\x1a\ndisguised".as_slice()),
        ("sqlite.log", b"SQLite format 3\0disguised".as_slice()),
        ("sevenzip.log", b"7z\xbc\xaf\x27\x1cdisguised".as_slice()),
        ("rar.log", b"Rar!\x1a\x07disguised".as_slice()),
        ("video.log", b"RIFF1234AVI disguised".as_slice()),
        ("relative.log", b"original_filename=family.jpg".as_slice()),
        ("asset.log", b"assetPath=photos/family".as_slice()),
        ("user.log", b"user_name=Private".as_slice()),
        ("identity.log", b"identity=/identity".as_slice()),
        (
            "absolute.log",
            b"opened /opt/private-library/family-secret".as_slice(),
        ),
        ("data.log", b"source is /data/family-secret".as_slice()),
        ("var.log", b"source is /var/lib/private-library".as_slice()),
        (
            "trailing.log",
            b"source is /data/private-library/".as_slice(),
        ),
        (
            "repeated.log",
            b"source is //var/lib/private-library".as_slice(),
        ),
        ("unicode.log", "source is /データ/秘密".as_bytes()),
    ] {
        fs::write(nested.join(name), payload).unwrap();
        assert!(
            !std::process::Command::new(&scanner)
                .arg(artifacts.path())
                .status()
                .unwrap()
                .success(),
            "privacy scanner accepted bypass fixture {name}"
        );
        fs::remove_file(nested.join(name)).unwrap();
    }

    fs::write(
        artifacts.path().join("debug-test-executables.txt"),
        b"lightroom-inventory=target/debug/lightroom-inventory\nrekesz=target/debug/rekesz\npersonName=Private\n",
    )
    .unwrap();
    assert!(
        !std::process::Command::new(&scanner)
            .arg(artifacts.path())
            .status()
            .unwrap()
            .success(),
        "approved manifest filename must not permit extra unsafe fields"
    );
    fs::write(
        artifacts.path().join("debug-test-executables.txt"),
        b"lightroom-inventory=target/debug/lightroom-inventory\nrekesz=target/debug/rekesz\n",
    )
    .unwrap();

    for invalid_path in [
        "target//debug/rekesz",
        "target/debug/../rekesz",
        "target/debug/rekesz test",
        "target/debug/rekesz=extra",
        "target/release/rekesz",
    ] {
        fs::write(
            artifacts.path().join("debug-test-executables.txt"),
            format!(
                "lightroom-inventory=target/debug/lightroom-inventory\nrekesz={invalid_path}\n"
            ),
        )
        .unwrap();
        assert!(
            !std::process::Command::new(&scanner)
                .arg(artifacts.path())
                .status()
                .unwrap()
                .success(),
            "build mapping accepted unsafe path {invalid_path}"
        );
    }
    fs::write(
        artifacts.path().join("debug-test-executables.txt"),
        b"lightroom-inventory=target/debug/lightroom-inventory\nrekesz=target/debug/rekesz\n",
    )
    .unwrap();

    fs::write(
        artifacts.path().join("test-artifacts.current.txt"),
        concat!(
            "revision=1111111111111111111111111111111111111111\n",
            "build_input_sha256=2222222222222222222222222222222222222222222222222222222222222222\n",
            "worktree_status_sha256=3333333333333333333333333333333333333333333333333333333333333333\n",
            "4444444444444444444444444444444444444444444444444444444444444444  target/debug/deps/rekesz-one\n",
            "1234 target/debug/deps/rekesz-two\n",
        ),
    )
    .unwrap();
    assert!(
        !std::process::Command::new(&scanner)
            .arg(artifacts.path())
            .status()
            .unwrap()
            .success(),
        "test manifest accepted mismatched digest and timestamp path sets"
    );
    fs::write(
        artifacts.path().join("test-artifacts.current.txt"),
        safe_manifest,
    )
    .unwrap();
    fs::write(
        artifacts.path().join("test-artifacts.current.txt"),
        safe_manifest.replace("1234 target/", "1235 target/"),
    )
    .unwrap();
    assert!(
        !std::process::Command::new(&scanner)
            .arg(artifacts.path())
            .status()
            .unwrap()
            .success(),
        "scanner accepted individually valid manifests with inconsistent timestamps"
    );
    fs::write(
        artifacts.path().join("test-artifacts.current.txt"),
        safe_manifest,
    )
    .unwrap();

    for (field, replacement) in [
        ("reference_tag=safe-host", "reference_tag=other-host"),
        ("max_rss_kib=2097152", "max_rss_kib=2097153"),
        ("max_cpu_percent=400", "max_cpu_percent=400.5"),
        ("max_cancel_latency_ms=100", "max_cancel_latency_ms=100.5"),
        (
            "revision=1111111111111111111111111111111111111111",
            "revision=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        ),
        (
            "worktree_status_sha256=3333333333333333333333333333333333333333333333333333333333333333",
            "worktree_status_sha256=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        ),
    ] {
        fs::write(
            artifacts.path().join("reference.txt"),
            reference.replace(field, replacement),
        )
        .unwrap();
        assert!(
            !std::process::Command::new(&scanner)
                .arg(artifacts.path())
                .status()
                .unwrap()
                .success(),
            "privacy scanner accepted cross-artifact mismatch for {field}"
        );
    }
    fs::write(artifacts.path().join("reference.txt"), reference).unwrap();

    for invalid_report in [
        b"{}".as_slice(),
        br#"{"thresholds":{"max_cpu_percent":[]}}"#,
    ] {
        fs::write(&report, invalid_report).unwrap();
        assert!(
            !std::process::Command::new(&scanner)
                .arg(artifacts.path())
                .status()
                .unwrap()
                .success(),
            "aggregate scanner accepted incomplete or wrongly typed JSON"
        );
    }
    fs::write(
        &report,
        String::from_utf8_lossy(safe_report)
            .replace("\"inference_started\":1", "\"inference_started\":1.5"),
    )
    .unwrap();
    assert!(
        !std::process::Command::new(&scanner)
            .arg(artifacts.path())
            .status()
            .unwrap()
            .success(),
        "aggregate scanner accepted a fractional inference counter"
    );
    fs::write(
        &report,
        String::from_utf8_lossy(safe_report).replace(
            "\"people_workload_in_progress_at_inference\":1",
            "\"people_workload_in_progress_at_inference\":true",
        ),
    )
    .unwrap();
    assert!(
        !std::process::Command::new(&scanner)
            .arg(artifacts.path())
            .status()
            .unwrap()
            .success(),
        "aggregate scanner accepted a boolean workload-in-progress counter"
    );
    fs::write(&report, safe_report).unwrap();

    fs::write(
        artifacts.path().join("reference.txt"),
        reference.replace("worktree_dirty=true", "worktree_dirty=maybe"),
    )
    .unwrap();
    assert!(
        !std::process::Command::new(&scanner)
            .arg(artifacts.path())
            .status()
            .unwrap()
            .success(),
        "reference metadata accepted an invalid typed value"
    );
    for invalid_limit in ["0", "-0.5", "NaN", "1."] {
        fs::write(
            artifacts.path().join("reference.txt"),
            reference.replace(
                "max_cpu_percent=400",
                &format!("max_cpu_percent={invalid_limit}"),
            ),
        )
        .unwrap();
        assert!(
            !std::process::Command::new(&scanner)
                .arg(artifacts.path())
                .status()
                .unwrap()
                .success(),
            "reference metadata accepted invalid CPU limit {invalid_limit}"
        );
    }
    fs::write(artifacts.path().join("reference.txt"), reference).unwrap();

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;

        let secret = "PRIVATE_OPERAND_DO_NOT_ECHO";
        let shim = tempfile::tempdir().unwrap();
        let path = format!(
            "{}:{}",
            shim.path().display(),
            std::env::var("PATH").unwrap()
        );
        let mktemp = shim.path().join("mktemp");
        fs::write(
            &mktemp,
            format!("#!/bin/sh\nprintf '%s\\n' '{secret}' >&2\nexit 72\n"),
        )
        .unwrap();
        fs::set_permissions(&mktemp, fs::Permissions::from_mode(0o755)).unwrap();
        let allocation = std::process::Command::new(&scanner)
            .arg(artifacts.path())
            .env("PATH", &path)
            .output()
            .unwrap();
        assert!(!allocation.status.success());
        assert!(!String::from_utf8_lossy(&allocation.stdout).contains(secret));
        assert!(!String::from_utf8_lossy(&allocation.stderr).contains(secret));
        fs::remove_file(mktemp).unwrap();

        let find = shim.path().join("find");
        fs::write(
            &find,
            format!("#!/bin/sh\nprintf '%s\\n' '{secret}' >&2\nexit 73\n"),
        )
        .unwrap();
        fs::set_permissions(&find, fs::Permissions::from_mode(0o755)).unwrap();
        let traversal = std::process::Command::new(&scanner)
            .arg(artifacts.path())
            .env("PATH", &path)
            .output()
            .unwrap();
        assert!(!traversal.status.success());
        assert!(!String::from_utf8_lossy(&traversal.stdout).contains(secret));
        assert!(!String::from_utf8_lossy(&traversal.stderr).contains(secret));

        fs::remove_file(find).unwrap();
        let rg = shim.path().join("rg");
        fs::write(
            &rg,
            format!("#!/bin/sh\nprintf '%s\\n' '{secret}' >&2\nexit 2\n"),
        )
        .unwrap();
        fs::set_permissions(&rg, fs::Permissions::from_mode(0o755)).unwrap();
        let audit = std::process::Command::new(&scanner)
            .arg(artifacts.path())
            .env("PATH", &path)
            .output()
            .unwrap();
        assert!(!audit.status.success());
        assert!(!String::from_utf8_lossy(&audit.stdout).contains(secret));
        assert!(!String::from_utf8_lossy(&audit.stderr).contains(secret));
    }

    fs::write(&report, br#"{"face_count":60000,"person_name":"Private"}"#).unwrap();
    assert!(
        !std::process::Command::new(&scanner)
            .arg(artifacts.path())
            .status()
            .unwrap()
            .success()
    );
}

#[test]
fn people_release_privacy_gate_runs_on_failure_and_before_upload() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let release =
        fs::read_to_string(manifest.join("scripts/people-release-validation.sh")).unwrap();
    let workflow =
        fs::read_to_string(manifest.join(".github/workflows/people-release-validation.yml"))
            .unwrap();
    assert!(release.contains("trap finalize_privacy EXIT"));
    assert!(release.contains("local release_status=$?"));
    assert!(release.contains("exit \"$release_status\""));
    assert!(
        !release.contains("privacy-artifacts"),
        "privacy scanning must not create a new retained log after inspecting the tree"
    );
    let after = release.find("test-artifacts.after.txt").unwrap();
    let final_scan = release[after..]
        .find("check-people-artifact-privacy.sh")
        .unwrap();
    assert!(final_scan > 0, "final scan must follow the final manifest");
    let privacy_step = workflow.find("id: privacy").unwrap();
    let upload_step = workflow.find("actions/upload-artifact").unwrap();
    assert!(privacy_step < upload_step);
    assert!(workflow.contains("steps.privacy.outcome == 'success'"));
}

#[test]
fn people_release_artifact_cleanup_rejects_aliases_without_touching_sentinels() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let prepare = manifest.join("scripts/prepare-people-release-artifacts.sh");
    let test_root = tempfile::Builder::new()
        .prefix("privacy-cleanup-test-")
        .tempdir_in(manifest.join("target"))
        .unwrap();
    let sentinel = test_root.path().join("sentinel");
    fs::write(&sentinel, b"keep").unwrap();
    for unsafe_path in [
        manifest.clone(),
        manifest.join("target"),
        manifest.join("target/../target"),
        manifest.join("target/debug"),
        manifest.join("target/release"),
        manifest.join("target/foo"),
        manifest.join("target/people-release-validation/nested"),
    ] {
        assert!(
            !std::process::Command::new(&prepare)
                .arg(unsafe_path)
                .current_dir(&manifest)
                .status()
                .unwrap()
                .success()
        );
        assert_eq!(fs::read(&sentinel).unwrap(), b"keep");
    }
    let outside = tempfile::tempdir().unwrap();
    let unique = test_root.path().file_name().unwrap().to_string_lossy();
    let symlink = manifest
        .join("target")
        .join(format!("people-release-validation-{unique}"));
    #[cfg(unix)]
    std::os::unix::fs::symlink(outside.path(), &symlink).unwrap();
    assert!(
        !std::process::Command::new(&prepare)
            .arg(&symlink)
            .current_dir(&manifest)
            .status()
            .unwrap()
            .success()
    );
    assert_eq!(fs::read(&sentinel).unwrap(), b"keep");
    fs::remove_file(symlink).unwrap();

    let intermediate = manifest
        .join("target")
        .join(format!("privacy-intermediate-{unique}"));
    #[cfg(unix)]
    std::os::unix::fs::symlink(outside.path(), &intermediate).unwrap();
    assert!(
        !std::process::Command::new(&prepare)
            .arg(intermediate.join("people-release-validation"))
            .current_dir(&manifest)
            .status()
            .unwrap()
            .success()
    );
    assert_eq!(fs::read(&sentinel).unwrap(), b"keep");
    fs::remove_file(intermediate).unwrap();

    let dedicated = manifest
        .join("target")
        .join(format!("people-release-validation-clean-{unique}"));
    fs::create_dir(&dedicated).unwrap();
    fs::write(dedicated.join("stale.log"), b"stale").unwrap();
    assert!(
        std::process::Command::new(&prepare)
            .arg(&dedicated)
            .current_dir(&manifest)
            .output()
            .unwrap()
            .status
            .success()
    );
    assert!(fs::read_dir(&dedicated).unwrap().next().is_none());

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;

        let secret = "PRIVATE_CLEANUP_OPERAND_DO_NOT_ECHO";
        let shim = tempfile::tempdir().unwrap();
        let find = shim.path().join("find");
        fs::write(
            &find,
            format!("#!/bin/sh\nprintf '%s\\n' '{secret}' >&2\nexit 74\n"),
        )
        .unwrap();
        fs::set_permissions(&find, fs::Permissions::from_mode(0o755)).unwrap();
        let path = format!(
            "{}:{}",
            shim.path().display(),
            std::env::var("PATH").unwrap()
        );
        let failure = std::process::Command::new(&prepare)
            .arg(&dedicated)
            .current_dir(&manifest)
            .env("PATH", path)
            .output()
            .unwrap();
        assert!(!failure.status.success());
        assert!(!String::from_utf8_lossy(&failure.stdout).contains(secret));
        assert!(!String::from_utf8_lossy(&failure.stderr).contains(secret));
    }

    assert_eq!(fs::read(&sentinel).unwrap(), b"keep");
    fs::remove_dir(dedicated).unwrap();
}

#[test]
fn people_release_privacy_scanner_rejects_a_symlink_root() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let scanner = manifest.join("scripts/check-people-artifact-privacy.sh");
    let artifacts = tempfile::tempdir().unwrap();
    fs::write(
        artifacts.path().join("reference-resources.json"),
        b"{\"face_count\":1}",
    )
    .unwrap();
    let parent = tempfile::tempdir().unwrap();
    let alias = parent.path().join("artifacts");
    #[cfg(unix)]
    std::os::unix::fs::symlink(artifacts.path(), &alias).unwrap();
    assert!(
        !std::process::Command::new(scanner)
            .arg(alias)
            .status()
            .unwrap()
            .success()
    );
}

#[test]
fn people_summary_triggers_dirty_and_rebuild_publishes_current_epoch() {
    let root = tempfile::tempdir().unwrap();
    initialize_test_catalog(root.path()).unwrap();
    let catalog = catalog_path(root.path());
    let connection = Connection::open(&catalog).unwrap();
    let initial: (i64, i64) = connection
        .query_row(
            "SELECT source_epoch,published_epoch FROM people_summary_epoch WHERE id=1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .unwrap();
    connection
        .execute(
            "INSERT INTO assets(id,relative_path,scope_key) VALUES(1,?1,?2)",
            params![
                encode_relative_path(Path::new("summary.jpg")).unwrap(),
                scope_key(Path::new("summary.jpg")).unwrap()
            ],
        )
        .unwrap();
    let dirty: (i64, i64, String, i64) = connection
        .query_row(
            "SELECT e.source_epoch,e.published_epoch,q.status,q.requested_epoch
             FROM people_summary_epoch e JOIN people_summary_rebuild_queue q ON q.id=e.id",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        )
        .unwrap();
    assert!(dirty.0 > initial.0);
    assert_ne!(dirty.0, dirty.1);
    assert_eq!(dirty.2, "pending");
    assert_eq!(dirty.3, dirty.0);
    drop(connection);
    let rebuilt =
        people::rebuild_people_summary(root.path(), Arc::new(AtomicBool::new(false))).unwrap();
    let connection = Connection::open(catalog).unwrap();
    let state: (i64, i64, i64) = connection
        .query_row(
            "SELECT source_epoch,published_epoch,
             EXISTS(SELECT 1 FROM people_summary_rebuild_queue) FROM people_summary_epoch WHERE id=1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .unwrap();
    assert_eq!(state.0, rebuilt as i64);
    assert_eq!(state.0, state.1);
    assert_eq!(state.2, 0);
    people::list_fresh_people_page_with_authoritative_reads_denied(root.path()).unwrap();
    people::list_fresh_special_pages_with_authoritative_reads_denied(root.path()).unwrap();
}

#[test]
fn typed_people_summary_maintenance_refills_coalesces_pauses_and_converges() {
    let root = tempfile::tempdir().unwrap();
    initialize_test_catalog(root.path()).unwrap();
    people::rebuild_people_summary(root.path(), Arc::new(AtomicBool::new(false))).unwrap();
    let admission = crate::scheduler::MaintenanceAdmission::new();
    admission.set_foreground_blocked(true);
    let service = crate::face_engine::PeopleSummaryRebuildService::new(admission.clone());

    let connection = Connection::open(catalog_path(root.path())).unwrap();
    connection
        .execute(
            "INSERT INTO assets(id,relative_path,scope_key) VALUES(1,?1,?2)",
            params![
                encode_relative_path(Path::new("post-startup.jpg")).unwrap(),
                scope_key(Path::new("post-startup.jpg")).unwrap()
            ],
        )
        .unwrap();
    drop(connection);
    let durable = people::people_summary_rebuild_status(root.path()).unwrap();
    assert!(durable.needs_work());
    assert_eq!(durable.queue_status.as_deref(), Some("pending"));

    assert!(service.enqueue(root.path().to_path_buf(), 0));
    assert!(!service.enqueue(root.path().to_path_buf(), 10));
    std::thread::sleep(Duration::from_millis(40));
    assert_eq!(service.stats().running, 0);
    assert_eq!(service.stats().queued, 1);
    assert!(
        people::people_summary_rebuild_status(root.path())
            .unwrap()
            .needs_work()
    );

    service.set_paused(true);
    admission.set_foreground_blocked(false);
    for _ in 0..100 {
        if service.stats().running == 0 {
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    assert!(
        people::people_summary_rebuild_status(root.path())
            .unwrap()
            .needs_work()
    );
    assert!(service.stats().cancelled >= 1);

    service.set_paused(false);
    assert!(service.enqueue(root.path().to_path_buf(), 0));
    for _ in 0..500 {
        if !people::people_summary_rebuild_status(root.path())
            .unwrap()
            .needs_work()
        {
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    let converged = people::people_summary_rebuild_status(root.path()).unwrap();
    assert_eq!(converged.source_epoch, converged.published_epoch);
    assert_eq!(converged.queue_status, None);
    assert!(service.stats().completed >= 1);
}

#[test]
fn typed_people_summary_maintenance_reclaims_an_expired_durable_lease() {
    let root = tempfile::tempdir().unwrap();
    initialize_test_catalog(root.path()).unwrap();
    let connection = Connection::open(catalog_path(root.path())).unwrap();
    connection
        .execute(
            "INSERT INTO assets(id,relative_path,scope_key) VALUES(1,?1,?2)",
            params![
                encode_relative_path(Path::new("expired.jpg")).unwrap(),
                scope_key(Path::new("expired.jpg")).unwrap()
            ],
        )
        .unwrap();
    connection
        .execute(
            "UPDATE people_summary_rebuild_queue SET status='pending',retry_after=datetime('now','+1 hour') WHERE id=1",
            [],
        )
        .unwrap();
    assert!(
        !people::people_summary_rebuild_status(root.path())
            .unwrap()
            .ready_to_enqueue()
    );
    connection
        .execute(
            "UPDATE people_summary_rebuild_queue SET status='running',worker_id='dead',
             lease_expires_at=datetime('now','-1 minute'),error='interrupted' WHERE id=1",
            [],
        )
        .unwrap();
    drop(connection);
    let service = crate::face_engine::PeopleSummaryRebuildService::new(
        crate::scheduler::MaintenanceAdmission::new(),
    );
    assert!(service.enqueue(root.path().to_path_buf(), 0));
    for _ in 0..500 {
        if !people::people_summary_rebuild_status(root.path())
            .unwrap()
            .needs_work()
        {
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    assert!(
        !people::people_summary_rebuild_status(root.path())
            .unwrap()
            .needs_work()
    );
}

#[test]
fn typed_people_summary_maintenance_records_error_and_periodic_retry_recovers() {
    let root = tempfile::tempdir().unwrap();
    initialize_test_catalog(root.path()).unwrap();
    let connection = Connection::open(catalog_path(root.path())).unwrap();
    connection
        .execute(
            "INSERT INTO assets(id,relative_path,scope_key) VALUES(1,?1,?2)",
            params![vec![0xff_u8], Vec::<u8>::new()],
        )
        .unwrap();
    drop(connection);
    let service = crate::face_engine::PeopleSummaryRebuildService::new(
        crate::scheduler::MaintenanceAdmission::new(),
    );
    assert!(service.enqueue(root.path().to_path_buf(), 0));
    for _ in 0..200 {
        if service.stats().failed > 0 {
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    assert_eq!(service.stats().failed, 1);
    let failed = people::people_summary_rebuild_status(root.path()).unwrap();
    assert_eq!(failed.queue_status.as_deref(), Some("pending"));
    assert!(failed.error.is_some());
    assert!(!failed.retry_ready);

    let connection = Connection::open(catalog_path(root.path())).unwrap();
    connection
        .execute(
            "UPDATE assets SET relative_path=?1,scope_key=?2 WHERE id=1",
            params![
                encode_relative_path(Path::new("recovered.jpg")).unwrap(),
                scope_key(Path::new("recovered.jpg")).unwrap()
            ],
        )
        .unwrap();
    drop(connection);
    assert!(service.enqueue(root.path().to_path_buf(), 0));
    for _ in 0..500 {
        if !people::people_summary_rebuild_status(root.path())
            .unwrap()
            .needs_work()
        {
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    assert!(
        !people::people_summary_rebuild_status(root.path())
            .unwrap()
            .needs_work()
    );
}

#[test]
fn people_summary_rebuild_reaches_repeated_cooperative_stage_and_batch_checkpoints() {
    let root = tempfile::tempdir().unwrap();
    initialize_test_catalog(root.path()).unwrap();
    let mut connection = Connection::open(catalog_path(root.path())).unwrap();
    let transaction = connection.transaction().unwrap();
    for id in 1..=1_001_i64 {
        let relative = PathBuf::from(format!("batch/{id}.jpg"));
        transaction
            .execute(
                "INSERT INTO assets(id,relative_path,scope_key) VALUES(?1,?2,?3)",
                params![
                    id,
                    encode_relative_path(&relative).unwrap(),
                    scope_key(&relative).unwrap()
                ],
            )
            .unwrap();
    }
    transaction.commit().unwrap();
    drop(connection);
    let checkpoints = std::sync::atomic::AtomicUsize::new(0);
    people::rebuild_people_summary_with_checkpoint(
        root.path(),
        Arc::new(AtomicBool::new(false)),
        &|_| {
            checkpoints.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
            true
        },
    )
    .unwrap();
    assert!(checkpoints.load(std::sync::atomic::Ordering::Acquire) >= 10);
}

#[test]
fn people_summary_long_sql_preempts_within_100ms_and_retry_converges() {
    let _progress_test = people::PEOPLE_SUMMARY_PROGRESS_TEST_LOCK.lock().unwrap();
    let root = tempfile::tempdir().unwrap();
    initialize_test_catalog(root.path()).unwrap();
    let connection = Connection::open(catalog_path(root.path())).unwrap();
    connection
        .execute(
            "INSERT INTO assets(id,relative_path,scope_key) VALUES(1,?1,?2)",
            params![
                encode_relative_path(Path::new("shared.jpg")).unwrap(),
                scope_key(Path::new("shared.jpg")).unwrap()
            ],
        )
        .unwrap();
    connection.execute_batch(
        "INSERT INTO people(id,display_name,created_at,updated_at) VALUES(1,'Person',datetime('now'),datetime('now'));
         WITH RECURSIVE n(i) AS (VALUES(1) UNION ALL SELECT i+1 FROM n WHERE i<250000)
         INSERT INTO faces(id,asset_id,provenance,source_key,left,top,right,bottom,origin,updated_at)
         SELECT i,1,'rekesz',printf('scale-%d',i),0.1,0.1,0.2,0.2,'rekesz',datetime('now') FROM n;
         WITH RECURSIVE n(i) AS (VALUES(1) UNION ALL SELECT i+1 FROM n WHERE i<250000)
         INSERT INTO face_assignments(face_id,person_id,kind,provenance,confidence,updated_at)
         SELECT i,1,'suggested','rekesz',0.9,datetime('now') FROM n;",
    )
    .unwrap();
    drop(connection);

    let progress_barrier = Arc::new((
        std::sync::Mutex::new((false, false)),
        std::sync::Condvar::new(),
    ));
    let progress_steps = Arc::new(std::sync::atomic::AtomicU64::new(0));
    *people::PEOPLE_SUMMARY_PROGRESS_HOOK.lock().unwrap() =
        Some(people::PeopleSummaryProgressHook {
            root: root.path().to_path_buf(),
            threshold_steps: 1_000_000,
            steps: progress_steps.clone(),
            barrier: progress_barrier.clone(),
        });
    let preempt = Arc::new(AtomicBool::new(false));
    let worker_preempt = preempt.clone();
    let root_path = root.path().to_path_buf();
    let worker = std::thread::spawn(move || {
        people::rebuild_people_summary_with_preemption(
            &root_path,
            Arc::new(AtomicBool::new(false)),
            &|_| true,
            Arc::new(move || worker_preempt.load(std::sync::atomic::Ordering::Acquire)),
        )
    });
    let (lock, wake) = &*progress_barrier;
    let mut state = lock.lock().unwrap();
    while !state.0 {
        state = wake.wait(state).unwrap();
    }
    assert!(progress_steps.load(std::sync::atomic::Ordering::Acquire) >= 1_000_000);
    preempt.store(true, std::sync::atomic::Ordering::Release);
    let started = Instant::now();
    state.1 = true;
    wake.notify_all();
    drop(state);
    let error = worker.join().unwrap().unwrap_err();
    *people::PEOPLE_SUMMARY_PROGRESS_HOOK.lock().unwrap() = None;
    let preempt_latency = started.elapsed();
    eprintln!(
        "PEOPLE_SUMMARY_PREEMPT rows=250000 vm_steps={} latency_ms={}",
        progress_steps.load(std::sync::atomic::Ordering::Acquire),
        preempt_latency.as_millis()
    );
    assert!(preempt_latency <= Duration::from_millis(100));
    assert!(
        format!("{error:#}").contains("interrupted")
            || format!("{error:#}").contains("cancelled")
            || format!("{error:#}").contains("preempted")
    );
    assert!(
        people::people_summary_rebuild_status(root.path())
            .unwrap()
            .needs_work()
    );
    let requeued = people::people_summary_rebuild_status(root.path()).unwrap();
    assert_eq!(requeued.queue_status.as_deref(), Some("pending"));
    assert_eq!(requeued.worker_id, None);
    assert_eq!(requeued.lease_expires_at, None);
    assert_eq!(requeued.retry_after, None);
    assert_eq!(requeued.error, None);
    let connection = Connection::open(catalog_path(root.path())).unwrap();
    let partial_rows: i64 = connection
        .query_row(
            "SELECT count(*) FROM people_scope_summary WHERE build_epoch=?1",
            [requeued.source_epoch as i64],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(partial_rows, 0, "preempted TEMP aggregate leaked live rows");
    drop(connection);
    people::rebuild_people_summary(root.path(), Arc::new(AtomicBool::new(false))).unwrap();
    assert!(
        !people::people_summary_rebuild_status(root.path())
            .unwrap()
            .needs_work()
    );
    let connection = Connection::open(catalog_path(root.path())).unwrap();
    let obsolete: i64 = connection
        .query_row(
            "SELECT count(*) FROM people_scope_summary WHERE build_epoch<>(SELECT published_epoch FROM people_summary_epoch WHERE id=1)",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(obsolete, 0);
}

#[test]
fn individual_people_summary_cancel_leaves_durable_pending_work() {
    let _progress_test = people::PEOPLE_SUMMARY_PROGRESS_TEST_LOCK.lock().unwrap();
    let root = tempfile::tempdir().unwrap();
    initialize_test_catalog(root.path()).unwrap();
    let connection = Connection::open(catalog_path(root.path())).unwrap();
    connection
        .execute(
            "INSERT INTO assets(id,relative_path,scope_key) VALUES(1,?1,?2)",
            params![
                encode_relative_path(Path::new("cancel-active.jpg")).unwrap(),
                scope_key(Path::new("cancel-active.jpg")).unwrap()
            ],
        )
        .unwrap();
    connection
        .execute_batch(
            "INSERT INTO people(id,display_name,created_at,updated_at) VALUES(1,'Person',datetime('now'),datetime('now'));
             WITH RECURSIVE n(i) AS (VALUES(1) UNION ALL SELECT i+1 FROM n WHERE i<25000)
             INSERT INTO faces(id,asset_id,provenance,source_key,left,top,right,bottom,origin,updated_at)
             SELECT i,1,'rekesz',printf('cancel-%d',i),0.1,0.1,0.2,0.2,'rekesz',datetime('now') FROM n;
             WITH RECURSIVE n(i) AS (VALUES(1) UNION ALL SELECT i+1 FROM n WHERE i<25000)
             INSERT INTO face_assignments(face_id,person_id,kind,provenance,confidence,updated_at)
             SELECT i,1,'suggested','rekesz',0.9,datetime('now') FROM n;",
        )
        .unwrap();
    drop(connection);
    let progress_barrier = Arc::new((
        std::sync::Mutex::new((false, false)),
        std::sync::Condvar::new(),
    ));
    *people::PEOPLE_SUMMARY_PROGRESS_HOOK.lock().unwrap() =
        Some(people::PeopleSummaryProgressHook {
            root: root.path().to_path_buf(),
            threshold_steps: 100_000,
            steps: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            barrier: progress_barrier.clone(),
        });
    let admission = crate::scheduler::MaintenanceAdmission::new();
    let service = crate::face_engine::PeopleSummaryRebuildService::new(admission);
    assert!(service.enqueue(root.path().to_path_buf(), 0));
    let (lock, wake) = &*progress_barrier;
    let mut state = lock.lock().unwrap();
    while !state.0 {
        state = wake.wait(state).unwrap();
    }
    let claimed = people::people_summary_rebuild_status(root.path()).unwrap();
    assert_eq!(claimed.queue_status.as_deref(), Some("running"));
    assert!(claimed.worker_id.is_some());
    assert!(claimed.lease_expires_at.is_some());
    assert!(service.cancel(root.path()));
    state.1 = true;
    wake.notify_all();
    drop(state);
    for _ in 0..100 {
        if service.stats().running == 0 {
            break;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    let status = people::people_summary_rebuild_status(root.path()).unwrap();
    assert!(status.needs_work());
    assert_eq!(status.queue_status.as_deref(), Some("pending"));
    assert_eq!(status.worker_id, None);
    assert_eq!(status.lease_expires_at, None);
    assert!(service.stats().cancelled >= 1);
    let connection = Connection::open(catalog_path(root.path())).unwrap();
    let partial_rows: i64 = connection
        .query_row(
            "SELECT count(*) FROM people_scope_summary WHERE build_epoch=?1",
            [status.source_epoch as i64],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(partial_rows, 0);
    drop(connection);
    *people::PEOPLE_SUMMARY_PROGRESS_HOOK.lock().unwrap() = None;
    people::rebuild_people_summary(root.path(), Arc::new(AtomicBool::new(false))).unwrap();
    assert!(
        !people::people_summary_rebuild_status(root.path())
            .unwrap()
            .needs_work()
    );
    let connection = Connection::open(catalog_path(root.path())).unwrap();
    let obsolete: i64 = connection
        .query_row(
            "SELECT count(*) FROM people_scope_summary WHERE build_epoch<>(SELECT published_epoch FROM people_summary_epoch WHERE id=1)",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(obsolete, 0);
}

#[test]
fn cancelled_people_summary_rebuild_never_publishes_and_remains_retryable() {
    let root = tempfile::tempdir().unwrap();
    initialize_test_catalog(root.path()).unwrap();
    let catalog = catalog_path(root.path());
    let connection = Connection::open(&catalog).unwrap();
    connection
        .execute(
            "INSERT INTO assets(id,relative_path,scope_key) VALUES(1,?1,?2)",
            params![
                encode_relative_path(Path::new("cancelled.jpg")).unwrap(),
                scope_key(Path::new("cancelled.jpg")).unwrap()
            ],
        )
        .unwrap();
    drop(connection);

    let error =
        people::rebuild_people_summary(root.path(), Arc::new(AtomicBool::new(true))).unwrap_err();
    assert!(format!("{error:#}").contains("cancelled"));
    let connection = Connection::open(&catalog).unwrap();
    let state: (i64, i64, String) = connection
        .query_row(
            "SELECT e.source_epoch,e.published_epoch,q.status
             FROM people_summary_epoch e JOIN people_summary_rebuild_queue q ON q.id=e.id",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .unwrap();
    assert_ne!(state.0, state.1);
    assert_eq!(state.2, "pending");
    drop(connection);

    let rebuilt =
        people::rebuild_people_summary(root.path(), Arc::new(AtomicBool::new(false))).unwrap();
    let connection = Connection::open(catalog).unwrap();
    let epochs: (i64, i64) = connection
        .query_row(
            "SELECT source_epoch,published_epoch FROM people_summary_epoch WHERE id=1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .unwrap();
    assert_eq!(epochs, (rebuilt as i64, rebuilt as i64));
}

#[test]
fn fresh_people_summary_returns_zero_for_empty_root_and_scope() {
    let root = tempfile::tempdir().unwrap();
    initialize_test_catalog(root.path()).unwrap();
    people::rebuild_people_summary(root.path(), Arc::new(AtomicBool::new(false))).unwrap();
    for scope in [None, Some(root.path().join("empty/deep"))] {
        let page = people::list_people_page(root.path(), scope.as_deref(), 0, 30).unwrap();
        assert!(page.items.is_empty());
        assert_eq!(page.total, 0);
        assert_eq!(page.unnamed_faces, 0);
        assert_eq!(page.review_faces, 0);
        assert_eq!(page.analysis_pending, 0);
        assert_eq!(page.analysis_errors, 0);
    }
}

#[test]
fn people_summary_cas_rejects_a_source_mutation_before_publish() {
    let root = tempfile::tempdir().unwrap();
    initialize_test_catalog(root.path()).unwrap();
    let catalog = catalog_path(root.path());
    let connection = Connection::open(&catalog).unwrap();
    connection
        .execute(
            "INSERT INTO assets(id,relative_path,scope_key) VALUES(1,?1,?2)",
            params![
                encode_relative_path(Path::new("first.jpg")).unwrap(),
                scope_key(Path::new("first.jpg")).unwrap()
            ],
        )
        .unwrap();
    drop(connection);

    let error = people::rebuild_people_summary_before_publish(root.path(), &|| {
        let connection = Connection::open(&catalog).unwrap();
        connection
            .execute(
                "INSERT INTO assets(id,relative_path,scope_key) VALUES(2,?1,?2)",
                params![
                    encode_relative_path(Path::new("late.jpg")).unwrap(),
                    scope_key(Path::new("late.jpg")).unwrap()
                ],
            )
            .unwrap();
    })
    .unwrap_err();
    assert!(
        format!("{error:#}").contains("changed during rebuild"),
        "{error:#}"
    );
    let connection = Connection::open(&catalog).unwrap();
    let state: (i64, i64, String, i64) = connection
        .query_row(
            "SELECT e.source_epoch,e.published_epoch,q.status,q.requested_epoch
             FROM people_summary_epoch e JOIN people_summary_rebuild_queue q ON q.id=e.id",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        )
        .unwrap();
    assert_ne!(state.0, state.1);
    assert_eq!(state.2, "pending");
    assert_eq!(state.3, state.0);
    drop(connection);

    people::rebuild_people_summary(root.path(), Arc::new(AtomicBool::new(false))).unwrap();
    let connection = Connection::open(catalog).unwrap();
    let epochs: (i64, i64) = connection
        .query_row(
            "SELECT source_epoch,published_epoch FROM people_summary_epoch WHERE id=1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .unwrap();
    assert_eq!(epochs.0, epochs.1);
}

#[test]
fn people_summary_temp_build_does_not_block_live_writers() {
    let root = tempfile::tempdir().unwrap();
    initialize_test_catalog(root.path()).unwrap();
    let catalog = catalog_path(root.path());
    let connection = Connection::open(&catalog).unwrap();
    connection
        .execute(
            "INSERT INTO assets(id,relative_path,scope_key) VALUES(1,?1,?2)",
            params![
                encode_relative_path(Path::new("overlap/a.jpg")).unwrap(),
                scope_key(Path::new("overlap/a.jpg")).unwrap()
            ],
        )
        .unwrap();
    drop(connection);
    let entered = Arc::new(std::sync::Barrier::new(2));
    let release = Arc::new(std::sync::Barrier::new(2));
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let rebuild_root = root.path().to_path_buf();
    let worker = {
        let entered = entered.clone();
        let release = release.clone();
        let calls = calls.clone();
        std::thread::spawn(move || {
            people::rebuild_people_summary_with_checkpoint(
                &rebuild_root,
                Arc::new(AtomicBool::new(false)),
                &move |_| {
                    if calls.fetch_add(1, std::sync::atomic::Ordering::AcqRel) == 1 {
                        entered.wait();
                        release.wait();
                    }
                    true
                },
            )
        })
    };
    entered.wait();
    assert!(
        !worker.is_finished(),
        "rebuild must remain held at the TEMP-stage checkpoint"
    );
    let mut writer = Connection::open(&catalog).unwrap();
    // A 20ms busy timeout makes this a deterministic lock assertion: if the
    // held shadow build retained a live-database writer lock, BEGIN IMMEDIATE
    // would return SQLITE_BUSY instead of reaching COMMIT.
    writer.busy_timeout(Duration::from_millis(20)).unwrap();
    let transaction = writer
        .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
        .expect("held TEMP-stage rebuild must not block a live writer");
    transaction
        .execute(
            "INSERT INTO assets(id,relative_path,scope_key) VALUES(2,?1,?2)",
            params![
                encode_relative_path(Path::new("overlap/b.jpg")).unwrap(),
                scope_key(Path::new("overlap/b.jpg")).unwrap()
            ],
        )
        .unwrap();
    transaction
        .commit()
        .expect("bounded writer transaction must commit while rebuild is held");
    assert!(!worker.is_finished());
    release.wait();
    assert!(
        worker.join().unwrap().is_err(),
        "source mutation must defeat CAS"
    );
}

#[test]
fn bounded_people_summary_gc_yields_to_live_writers() {
    let root = tempfile::tempdir().unwrap();
    initialize_test_catalog(root.path()).unwrap();
    let catalog = catalog_path(root.path());
    let connection = Connection::open(&catalog).unwrap();
    let keep: i64 = connection
        .query_row(
            "SELECT published_epoch FROM people_summary_epoch",
            [],
            |row| row.get(0),
        )
        .unwrap();
    connection
        .execute_batch(
            "WITH RECURSIVE n(x) AS (VALUES(1) UNION ALL SELECT x+1 FROM n WHERE x<1200)
         INSERT INTO people_scope_totals
         SELECT 9000+x,CAST(x AS BLOB),0,0,0,0,0 FROM n;",
        )
        .unwrap();
    drop(connection);
    let entered = Arc::new(std::sync::Barrier::new(2));
    let release = Arc::new(std::sync::Barrier::new(2));
    let gc_root = root.path().to_path_buf();
    let gc = {
        let entered = entered.clone();
        let release = release.clone();
        std::thread::spawn(move || {
            let first = AtomicBool::new(true);
            people::gc_people_summary_epochs_with_checkpoint(&gc_root, keep, &|| {
                if first.swap(false, std::sync::atomic::Ordering::AcqRel) {
                    entered.wait();
                    release.wait();
                }
            })
        })
    };
    entered.wait();
    let started = std::time::Instant::now();
    Connection::open(&catalog)
        .unwrap()
        .execute("UPDATE catalog_meta SET last_imported_at='gc-writer'", [])
        .unwrap();
    assert!(started.elapsed() < Duration::from_millis(100));
    release.wait();
    gc.join().unwrap().unwrap();
    assert_eq!(
        Connection::open(catalog)
            .unwrap()
            .query_row(
                "SELECT count(*) FROM people_scope_totals WHERE build_epoch<>?1",
                [keep],
                |row| row.get::<_, i64>(0)
            )
            .unwrap(),
        0
    );
}

#[test]
fn people_summary_gc_preserves_active_claimed_generation() {
    let root = tempfile::tempdir().unwrap();
    initialize_test_catalog(root.path()).unwrap();
    let catalog = catalog_path(root.path());
    let connection = Connection::open(&catalog).unwrap();
    let keep: i64 = connection
        .query_row(
            "SELECT published_epoch FROM people_summary_epoch",
            [],
            |row| row.get(0),
        )
        .unwrap();
    connection
        .execute(
            "INSERT INTO people_scope_totals VALUES(42,X'42',0,0,0,0,0)",
            [],
        )
        .unwrap();
    connection.execute("UPDATE people_summary_rebuild_queue SET status='running',worker_id='winner-b',claimed_epoch=42,lease_expires_at=datetime('now','+5 minutes')", []).unwrap();
    drop(connection);
    people::gc_people_summary_epochs_with_checkpoint(root.path(), keep, &|| {}).unwrap();
    let connection = Connection::open(&catalog).unwrap();
    assert_eq!(
        connection
            .query_row(
                "SELECT count(*) FROM people_scope_totals WHERE build_epoch=42",
                [],
                |row| row.get::<_, i64>(0)
            )
            .unwrap(),
        1
    );
    connection.execute("UPDATE people_summary_rebuild_queue SET status='pending',worker_id=NULL,claimed_epoch=NULL,lease_expires_at=NULL", []).unwrap();
    drop(connection);
    people::gc_people_summary_epochs_with_checkpoint(root.path(), keep, &|| {}).unwrap();
    assert_eq!(
        Connection::open(catalog)
            .unwrap()
            .query_row(
                "SELECT count(*) FROM people_scope_totals WHERE build_epoch=42",
                [],
                |row| row.get::<_, i64>(0)
            )
            .unwrap(),
        0
    );
}

#[test]
fn reclaimed_people_summary_worker_cannot_publish_or_clobber_winner() {
    let root = tempfile::tempdir().unwrap();
    initialize_test_catalog(root.path()).unwrap();
    let catalog = catalog_path(root.path());
    Connection::open(&catalog)
        .unwrap()
        .execute(
            "INSERT INTO assets(id,relative_path,scope_key) VALUES(1,?1,?2)",
            params![
                encode_relative_path(Path::new("race/a.jpg")).unwrap(),
                scope_key(Path::new("race/a.jpg")).unwrap()
            ],
        )
        .unwrap();
    let entered = Arc::new(std::sync::Barrier::new(2));
    let release = Arc::new(std::sync::Barrier::new(2));
    let old_root = root.path().to_path_buf();
    let old = {
        let entered = entered.clone();
        let release = release.clone();
        std::thread::spawn(move || {
            let blocked = AtomicBool::new(false);
            people::rebuild_people_summary_with_checkpoint(
                &old_root,
                Arc::new(AtomicBool::new(false)),
                &move |_| {
                    if !blocked.swap(true, std::sync::atomic::Ordering::AcqRel) {
                        entered.wait();
                        release.wait();
                    }
                    true
                },
            )
        })
    };
    entered.wait();
    Connection::open(&catalog)
        .unwrap()
        .execute(
            "UPDATE people_summary_rebuild_queue SET lease_expires_at=datetime('now','-1 second')",
            [],
        )
        .unwrap();
    let winner_entered = Arc::new(std::sync::Barrier::new(2));
    let winner_release = Arc::new(std::sync::Barrier::new(2));
    let winner_root = root.path().to_path_buf();
    let winner_catalog = catalog.clone();
    let winner = {
        let entered = winner_entered.clone();
        let release = winner_release.clone();
        std::thread::spawn(move || {
            let blocked = AtomicBool::new(false);
            people::rebuild_people_summary_with_checkpoint(
                &winner_root,
                Arc::new(AtomicBool::new(false)),
                &move |_| {
                    let copied: bool = Connection::open(&winner_catalog)
                        .unwrap()
                        .query_row(
                            "SELECT EXISTS(SELECT 1 FROM asset_scope_ancestors a
                              JOIN people_summary_rebuild_queue q
                                ON q.claimed_epoch=a.build_epoch
                             WHERE q.status='running')",
                            [],
                            |row| row.get(0),
                        )
                        .unwrap();
                    if copied && !blocked.swap(true, std::sync::atomic::Ordering::AcqRel) {
                        entered.wait();
                        release.wait();
                    }
                    true
                },
            )
        })
    };
    winner_entered.wait();
    release.wait();
    assert!(old.join().unwrap().is_err());
    let connection = Connection::open(&catalog).unwrap();
    let winner_owner: String = connection
        .query_row(
            "SELECT worker_id FROM people_summary_rebuild_queue WHERE status='running'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert!(winner_owner.starts_with("people-summary-"));
    drop(connection);
    winner_release.wait();
    let winner = winner.join().unwrap().unwrap();
    let connection = Connection::open(catalog).unwrap();
    assert_eq!(
        connection
            .query_row(
                "SELECT published_epoch FROM people_summary_epoch WHERE id=1",
                [],
                |row| row.get::<_, i64>(0)
            )
            .unwrap(),
        winner as i64
    );
}

#[test]
fn people_summary_dependency_trigger_matrix_is_complete_and_transactional() {
    let root = tempfile::tempdir().unwrap();
    initialize_test_catalog(root.path()).unwrap();
    let connection = Connection::open(catalog_path(root.path())).unwrap();
    for table in [
        "face_assignments",
        "faces",
        "people",
        "assets",
        "recognition_suggestions",
        "recognition_index_state",
        "face_analysis_state",
    ] {
        for event in ["INSERT", "UPDATE", "DELETE"] {
            let name = format!("people_summary_dirty_{table}_{event}");
            let sql: String = connection
                .query_row(
                    "SELECT sql FROM sqlite_master WHERE type='trigger' AND name=?1",
                    [&name],
                    |row| row.get(0),
                )
                .unwrap_or_else(|error| panic!("missing {name}: {error}"));
            assert!(sql.contains("source_epoch=source_epoch+1"), "{name}");
            assert!(sql.contains("people_summary_rebuild_queue"), "{name}");
            assert!(sql.contains("ELSE 'pending'"), "{name}");
            assert!(sql.contains("THEN 'running'"), "{name}");
            assert!(sql.contains("RAISE(ABORT"), "{name}");
        }
    }
}

#[test]
fn nested_people_summary_bulk_invalidation_bumps_once_and_rolls_back() {
    let root = tempfile::tempdir().unwrap();
    initialize_test_catalog(root.path()).unwrap();
    let mut connection = Connection::open(catalog_path(root.path())).unwrap();
    let initial: i64 = connection
        .query_row("SELECT source_epoch FROM people_summary_epoch", [], |row| {
            row.get(0)
        })
        .unwrap();
    {
        let transaction = connection.transaction().unwrap();
        begin_people_summary_bulk(&transaction).unwrap();
        transaction
            .execute(
                "INSERT INTO people(display_name,created_at,updated_at) VALUES('rollback','x','x')",
                [],
            )
            .unwrap();
        transaction.rollback().unwrap();
    }
    assert_eq!(
        connection
            .query_row("SELECT source_epoch FROM people_summary_epoch", [], |row| {
                row.get::<_, i64>(0)
            })
            .unwrap(),
        initial
    );
    let transaction = connection.transaction().unwrap();
    begin_people_summary_bulk(&transaction).unwrap();
    begin_people_summary_bulk(&transaction).unwrap();
    transaction
        .execute(
            "INSERT INTO people(display_name,created_at,updated_at) VALUES('nested','x','x')",
            [],
        )
        .unwrap();
    finish_people_summary_bulk(&transaction).unwrap();
    finish_people_summary_bulk(&transaction).unwrap();
    transaction.commit().unwrap();
    assert_eq!(
        connection
            .query_row("SELECT source_epoch FROM people_summary_epoch", [], |row| {
                row.get::<_, i64>(0)
            })
            .unwrap(),
        initial + 1
    );
}

#[test]
fn v4_people_summary_migration_failure_rolls_back_and_retries_cleanly() {
    let root = tempfile::tempdir().unwrap();
    initialize_test_catalog(root.path()).unwrap();
    let mut connection = Connection::open(catalog_path(root.path())).unwrap();
    connection
        .execute_batch(
            "PRAGMA foreign_keys=OFF;
             DROP TABLE people_summary_epoch;
             CREATE TABLE people_summary_epoch(wrong INTEGER);
             UPDATE catalog_meta SET schema_version=4 WHERE id=1;
             PRAGMA user_version=4;",
        )
        .unwrap();
    let error = initialize_schema(&mut connection).unwrap_err();
    assert!(
        format!("{error:#}").contains("people_summary_epoch"),
        "{error:#}"
    );
    assert_eq!(
        connection
            .query_row("PRAGMA user_version", [], |row| row.get::<_, i64>(0))
            .unwrap(),
        4
    );
    assert_eq!(
        connection
            .query_row("SELECT schema_version FROM catalog_meta", [], |row| row
                .get::<_, i64>(0))
            .unwrap(),
        4
    );
    connection
        .execute_batch("DROP TABLE people_summary_epoch;")
        .unwrap();
    drop(connection);
    let mut connection = Connection::open(catalog_path(root.path())).unwrap();
    initialize_schema(&mut connection).unwrap();
    assert_eq!(
        connection
            .query_row("PRAGMA user_version", [], |row| row.get::<_, i64>(0))
            .unwrap(),
        6
    );
    let state: (i64, i64, String) = connection
        .query_row(
            "SELECT e.source_epoch,e.published_epoch,q.status FROM people_summary_epoch e
             JOIN people_summary_rebuild_queue q ON q.id=e.id",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .unwrap();
    assert_ne!(state.0, state.1);
    assert_eq!(state.2, "pending");
}

fn run_generated_people_catalog_profile(face_count: usize, query_budget: Duration) {
    run_generated_people_catalog_profile_with_lifecycle(face_count, query_budget, None);
}

fn run_generated_people_catalog_profile_with_lifecycle(
    face_count: usize,
    query_budget: Duration,
    lifecycle: Option<&QueryWorkloadLifecycle>,
) {
    let asset_count = face_count.div_ceil(4);
    let person_count = face_count.div_ceil(20);
    let root = tempfile::tempdir().unwrap();
    fs::create_dir(root.path().join("profile")).unwrap();
    initialize_test_catalog(root.path()).unwrap();
    let catalog = root.path().join(CATALOG_DIR).join(CATALOG_FILE);
    let mut connection = Connection::open(&catalog).unwrap();
    let transaction = connection.transaction().unwrap();
    {
        let mut asset = transaction
            .prepare("INSERT INTO assets(id, relative_path, scope_key) VALUES(?1, ?2, ?3)")
            .unwrap();
        for id in 1..=asset_count as i64 {
            let relative = PathBuf::from(format!("profile/2024/08/30/event/photo-{id}.jpg"));
            asset
                .execute(params![
                    id,
                    encode_relative_path(&relative).unwrap(),
                    scope_key(&relative).unwrap()
                ])
                .unwrap();
        }
        let sibling = PathBuf::from("profile-other/2024/08/30/event/collision.jpg");
        asset
            .execute(params![
                asset_count as i64 + 1,
                encode_relative_path(&sibling).unwrap(),
                scope_key(&sibling).unwrap()
            ])
            .unwrap();
        let mut person = transaction
            .prepare(
                "INSERT INTO people(id, display_name, created_at, updated_at)
                 VALUES(?1, ?2, datetime('now'), datetime('now'))",
            )
            .unwrap();
        for id in 1..=person_count as i64 {
            person
                .execute(params![id, format!("Person {id:06}")])
                .unwrap();
        }
        let mut face = transaction
            .prepare(
                "INSERT INTO faces(id, asset_id, provenance, source_key, left, top, right, bottom,
                                   origin, ignored, updated_at)
                 VALUES(?1, ?2, 'rekesz', ?3, .1, .1, .2, .2, 'rekesz', 0,
                        '2030-01-01T00:00:00Z')",
            )
            .unwrap();
        let mut assignment = transaction
            .prepare(
                "INSERT INTO face_assignments(face_id, person_id, kind, provenance, updated_at)
                 VALUES(?1, ?2, ?3, 'rekesz', datetime('now'))",
            )
            .unwrap();
        for id in 1..=face_count as i64 {
            let asset_id = (id - 1) / 4 + 1;
            let person_id = (id - 1) / 20 + 1;
            face.execute(params![id, asset_id, format!("generated-{id}")])
                .unwrap();
            let kind = if id % 10 == 0 {
                "suggested"
            } else {
                "confirmed"
            };
            assignment.execute(params![id, person_id, kind]).unwrap();
        }
        for extra in 1..=60_i64 {
            let id = face_count as i64 + extra;
            face.execute(params![id, 1_i64, format!("representative-{extra}")])
                .unwrap();
            assignment.execute(params![id, 1_i64, "confirmed"]).unwrap();
        }
    }
    transaction.commit().unwrap();
    let rebuild_started = std::time::Instant::now();
    people::rebuild_people_summary(root.path(), Arc::new(AtomicBool::new(false))).unwrap();
    eprintln!(
        "generated People profile: {face_count} faces, summary rebuild {:?}",
        rebuild_started.elapsed()
    );
    if face_count >= 250_000 {
        let catalog = catalog_path(root.path());
        let baseline_bytes = fs::metadata(&catalog).unwrap().len();
        let mut cycle_bytes = Vec::new();
        for cycle in 0..2 {
            Connection::open(&catalog)
                .unwrap()
                .execute(
                    "UPDATE people SET updated_at=?1 WHERE id=1",
                    [format!("2030-01-0{}", cycle + 1)],
                )
                .unwrap();
            let cycle_started = std::time::Instant::now();
            people::rebuild_people_summary(root.path(), Arc::new(AtomicBool::new(false))).unwrap();
            assert!(cycle_started.elapsed() < Duration::from_secs(35));
            let connection = Connection::open(&catalog).unwrap();
            for table in [
                "people_scope_summary",
                "people_scope_totals",
                "people_scope_representatives",
                "people_scope_faces",
                "asset_scope_ancestors",
            ] {
                let generations: i64 = connection
                    .query_row(
                        &format!("SELECT count(DISTINCT build_epoch) FROM {table}"),
                        [],
                        |row| row.get(0),
                    )
                    .unwrap();
                assert!(generations <= 1, "obsolete {table} generation survived GC");
            }
            cycle_bytes.push(fs::metadata(&catalog).unwrap().len());
        }
        let final_bytes = fs::metadata(catalog).unwrap().len();
        let connection = Connection::open(catalog_path(root.path())).unwrap();
        let page_count: i64 = connection
            .query_row("PRAGMA page_count", [], |row| row.get(0))
            .unwrap();
        let freelist: i64 = connection
            .query_row("PRAGMA freelist_count", [], |row| row.get(0))
            .unwrap();
        eprintln!(
            "generated People profile: churn catalog bytes {baseline_bytes} -> {cycle_bytes:?}; \
             final={final_bytes}, pages={page_count}, freelist={freelist}"
        );
        assert!(final_bytes <= baseline_bytes.saturating_mul(2));
        assert!(cycle_bytes[1] <= cycle_bytes[0].saturating_add(16 * 1024 * 1024));
    }
    let started = std::time::Instant::now();
    let query_active = lifecycle.map(QueryWorkloadLifecycle::begin);
    let page = people::list_person_faces_page(root.path(), 1, None, 0, 30).unwrap();
    drop(query_active);
    let elapsed = started.elapsed();
    assert_eq!(page.total, 80);
    assert_eq!(page.items.len(), 30);
    assert!(
        elapsed <= query_budget,
        "detail query {elapsed:?} exceeds {query_budget:?}"
    );
    eprintln!("generated People profile: {face_count} faces, first detail page {elapsed:?}");
    for sort in [
        people::PeopleSort::Name,
        people::PeopleSort::Confirmed,
        people::PeopleSort::Review,
        people::PeopleSort::RecentlyChanged,
    ] {
        let started = std::time::Instant::now();
        let query_active = lifecycle.map(QueryWorkloadLifecycle::begin);
        let overview = people::list_people_page_sorted(root.path(), None, sort, 0, 30).unwrap();
        drop(query_active);
        assert_eq!(overview.total, person_count as u64);
        assert!(overview.items.len() <= 30, "paged DTO must remain bounded");
        assert!(overview.items[0].representative.is_some());
        let elapsed = started.elapsed();
        assert!(
            elapsed <= query_budget,
            "{sort:?} query {elapsed:?} exceeds {query_budget:?}"
        );
        eprintln!(
            "generated People profile: {face_count} faces, {sort:?} overview {elapsed:?}",
            elapsed = elapsed
        );
    }
    let late_started = std::time::Instant::now();
    let late_overview = people::list_people_page_sorted_cancellable(
        root.path(),
        None,
        people::PeopleSort::Name,
        person_count.saturating_sub(30) as u64,
        30,
        HashMap::from([(1_i64, 50_u64)]),
        Arc::new(AtomicBool::new(false)),
    )
    .unwrap();
    assert!(!late_overview.items.is_empty());
    assert!(late_started.elapsed() <= query_budget);
    let rank_fifty = people::list_people_page_sorted_cancellable(
        root.path(),
        None,
        people::PeopleSort::Name,
        0,
        1,
        HashMap::from([(1_i64, 50_u64)]),
        Arc::new(AtomicBool::new(false)),
    )
    .unwrap();
    // Person 1 has 18 ordinary confirmed faces before the 60 extra candidates.
    // Their quality fields and fixed timestamps tie, so the documented final
    // face-ID fallback makes zero-based rank 50 the 33rd extra face.
    assert_eq!(
        rank_fifty.items[0].representative_face_id,
        Some(face_count as i64 + 33)
    );
    let started = std::time::Instant::now();
    let query_active = lifecycle.map(QueryWorkloadLifecycle::begin);
    let review = people::list_review_queue_page_cancellable(
        root.path(),
        None,
        0,
        30,
        Arc::new(AtomicBool::new(false)),
    )
    .unwrap();
    drop(query_active);
    assert_eq!(review.total, (face_count / 10) as u64);
    assert!(review.items.len() <= 30);
    let review_elapsed = started.elapsed();
    assert!(
        review_elapsed <= query_budget,
        "review query {review_elapsed:?} exceeds {query_budget:?}"
    );
    let late_review_started = std::time::Instant::now();
    let late_review = people::list_review_queue_page_cancellable(
        root.path(),
        None,
        (face_count / 10).saturating_sub(30) as u64,
        30,
        Arc::new(AtomicBool::new(false)),
    )
    .unwrap();
    assert!(!late_review.items.is_empty());
    assert!(late_review_started.elapsed() <= query_budget);
    for relative_scope in ["profile", "profile/2024/08", "profile/2024/08/30/event"] {
        let scoped_started = std::time::Instant::now();
        let query_active = lifecycle.map(QueryWorkloadLifecycle::begin);
        let scoped = people::list_people_page_sorted(
            root.path(),
            Some(&root.path().join(relative_scope)),
            people::PeopleSort::Review,
            0,
            30,
        )
        .unwrap();
        drop(query_active);
        assert_eq!(scoped.total, person_count as u64);
        assert!(scoped.items.len() <= 30);
        let scoped_elapsed = scoped_started.elapsed();
        assert!(
            scoped_elapsed <= query_budget,
            "deep-subtree {relative_scope} query {scoped_elapsed:?} exceeds {query_budget:?}"
        );
    }
    let cancelled = Arc::new(AtomicBool::new(true));
    assert!(
        people::list_review_queue_page_cancellable(root.path(), None, 0, 30, cancelled).is_err()
    );

    for (name, sql) in [
        (
            "overview",
            "EXPLAIN QUERY PLAN SELECT person_id,display_name,confirmed_faces,suggested_faces
             FROM people_scope_summary WHERE build_epoch=1 AND scope_key=X''
             ORDER BY display_name COLLATE NOCASE,person_id LIMIT 30",
        ),
        (
            "detail",
            "EXPLAIN QUERY PLAN SELECT face.id FROM effective_face_assignments assignment
             JOIN faces face ON face.id=assignment.face_id JOIN assets asset ON asset.id=face.asset_id
             WHERE assignment.person_id=1 AND face.ignored=0 AND face.locally_ignored=0
             ORDER BY (assignment.kind='suggested') DESC, coalesce(assignment.confidence, 0) DESC,
             face.id LIMIT 30 OFFSET 0",
        ),
        (
            "review",
            "EXPLAIN QUERY PLAN SELECT face_id FROM people_scope_faces
             WHERE build_epoch=1 AND scope_key=X'' AND queue_kind='review'
             ORDER BY face_id,person_id LIMIT 30 OFFSET 0",
        ),
    ] {
        let mut statement = connection.prepare(sql).unwrap();
        let plan = statement
            .query_map([], |row| row.get::<_, String>(3))
            .unwrap()
            .collect::<rusqlite::Result<Vec<_>>>()
            .unwrap()
            .join("\n");
        assert!(plan.contains("INDEX"), "production {name} query lost indexed access:\n{plan}");
        if name != "detail" {
            assert!(
                !plan.contains("SCAN faces") && !plan.contains("effective_face_assignments"),
                "fresh summary {name} unexpectedly touches authoritative face tables:\n{plan}"
            );
        }
    }

    let in_flight_cancelled = Arc::new(AtomicBool::new(false));
    let query_cancelled = in_flight_cancelled.clone();
    let query_root = root.path().to_path_buf();
    let started = std::time::Instant::now();
    let query = std::thread::spawn(move || {
        people::list_people_page_sorted_cancellable(
            &query_root,
            None,
            people::PeopleSort::Confirmed,
            0,
            30,
            HashMap::new(),
            query_cancelled,
        )
    });
    std::thread::sleep(Duration::from_millis(1));
    in_flight_cancelled.store(true, std::sync::atomic::Ordering::Release);
    assert!(
        query.join().unwrap().is_err(),
        "large in-flight production query ignored cancellation"
    );
    assert!(
        started.elapsed() < Duration::from_millis(500),
        "query cancellation was not prompt"
    );
    eprintln!(
        "generated People profile: {face_count} faces, review query {elapsed:?}",
        elapsed = review_elapsed
    );
}

#[test]
fn people_diagnostics_are_aggregate_and_failed_jobs_are_retryable() {
    let root = tempfile::tempdir().unwrap();
    initialize_test_catalog(root.path()).unwrap();
    let catalog = root.path().join(CATALOG_DIR).join(CATALOG_FILE);
    let connection = Connection::open(&catalog).unwrap();
    connection
        .execute_batch(
            "INSERT INTO assets(id, relative_path, scope_key) VALUES(1, X'61', X'61');
             INSERT INTO analysis_asset_inventory(asset_id, source_size, source_modified_ns,
                 decoder_model_id, discovered_at)
                 VALUES(1, 1, 1, 'renderer-test', datetime('now'));
             INSERT INTO face_analysis_state(asset_id, detector_model_id, recognizer_model_id,
                 decoder_model_id, source_size, source_modified_ns, status, attempts, error, updated_at)
                 VALUES(1, 'detector-test', 'recognizer-test', 'renderer-test', 1, 1,
                        'error', 3, 'private details must not escape', datetime('now'));",
        )
        .unwrap();
    drop(connection);
    let diagnostics = people::diagnostics(root.path()).unwrap();
    assert_eq!(diagnostics.schema_version, 6);
    assert_eq!(diagnostics.detector_model_id, "detector-test");
    assert_eq!(diagnostics.recognizer_model_id, "recognizer-test");
    assert_eq!(diagnostics.analysis_errors, 1);
    let retried = people::retry_failed_analysis(root.path()).unwrap();
    assert_eq!(retried.analysis_jobs, 1);
    assert_eq!(
        people::diagnostics(root.path()).unwrap().analysis_pending,
        1
    );
}

#[test]
fn repeated_reimport_and_expired_lease_recovery_preserve_local_people_decisions() {
    run_reimport_restart_cancel_soak(3, None);
}

#[test]
#[ignore = "multi-hour release soak on the reference host"]
fn long_reimport_restart_cancel_soak() {
    let seconds = std::env::var("PEOPLE_SOAK_SECONDS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(4 * 60 * 60);
    run_reimport_restart_cancel_soak(usize::MAX, Some(Duration::from_secs(seconds)));
}

fn run_reimport_restart_cancel_soak(cycles: usize, max_duration: Option<Duration>) {
    let _ort = crate::face_engine::ORT_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let root = tempfile::tempdir().unwrap();
    fs::write(root.path().join("photo.CR3"), b"test").unwrap();
    let analysis_source = root.path().join("analysis.png");
    image::RgbImage::new(32, 32).save(&analysis_source).unwrap();
    let fixture = lightroom_fixture(root.path(), "s={Exposure2012=1}");
    let lightroom = fixture.path().join("fixture.lrcat");
    import_lightroom_inner(root.path(), &lightroom, true, ImportCapabilities::ALL).unwrap();
    let face_id = 900;
    let person_id = 900;
    let connection = Connection::open(root.path().join(CATALOG_DIR).join(CATALOG_FILE)).unwrap();
    connection
        .execute_batch(
            "INSERT INTO people(id, display_name, created_at, updated_at)
             VALUES(900, 'Soak person', datetime('now'), datetime('now'));
         INSERT INTO faces(id, asset_id, provenance, source_key, left, top, right, bottom,
                           origin, ignored, updated_at)
             VALUES(900, 1, 'rekesz', 'soak-face', .1, .1, .2, .2,
                    'rekesz', 0, datetime('now'));
         INSERT INTO face_assignments(face_id, person_id, kind, provenance, updated_at)
             VALUES(900, 900, 'confirmed', 'rekesz', datetime('now'));",
        )
        .unwrap();
    drop(connection);
    people::apply_decisions(
        root.path(),
        &people::DecisionBatch {
            id: "soak-decision".into(),
            commands: vec![people::DecisionCommand {
                face_id,
                person_id: Some(person_id),
                action: people::DecisionAction::Reject,
            }],
        },
    )
    .unwrap();
    let soak_started = std::time::Instant::now();
    for cycle in 0..cycles {
        if cycle > 0 && max_duration.is_some_and(|limit| soak_started.elapsed() >= limit) {
            break;
        }
        // Exercise a cancellable reader on its own connection while the import transaction
        // replaces Lightroom-owned rows. Opening every operation through the public APIs also
        // models process restart: no connection or in-memory queue state is retained.
        let cancelled = Arc::new(AtomicBool::new(false));
        let reader_cancelled = cancelled.clone();
        let reader_root = root.path().to_path_buf();
        let reader = std::thread::spawn(move || {
            people::list_people_page_sorted_cancellable(
                &reader_root,
                None,
                people::PeopleSort::Confirmed,
                0,
                30,
                HashMap::new(),
                reader_cancelled,
            )
        });
        crate::face_engine::sync_analysis_inventory(
            root.path(),
            std::slice::from_ref(&analysis_source),
        )
        .unwrap();
        let connection =
            Connection::open(root.path().join(CATALOG_DIR).join(CATALOG_FILE)).unwrap();
        connection
            .execute(
                "UPDATE face_analysis_state SET status='running', worker_id=?1,
                     lease_expires_at='2000-01-01'",
                [format!("expired-{cycle}")],
            )
            .unwrap();
        drop(connection);
        assert!(
            !crate::face_engine::pending_analysis_jobs(root.path(), 10)
                .unwrap()
                .is_empty()
        );
        let job = crate::face_engine::pending_analysis_jobs(root.path(), 10)
            .unwrap()
            .into_iter()
            .find(|job| {
                Connection::open(root.path().join(CATALOG_DIR).join(CATALOG_FILE))
                    .unwrap()
                    .query_row(
                        "SELECT relative_path FROM assets WHERE id=?1",
                        [job.asset_id],
                        |row| row.get::<_, Vec<u8>>(0),
                    )
                    .is_ok_and(|relative| {
                        decode_relative_path(&relative).unwrap().as_path()
                            == Path::new("analysis.png")
                    })
            })
            .unwrap();
        let admission = crate::scheduler::MaintenanceAdmission::new();
        let reached = Arc::new((std::sync::Mutex::new(false), std::sync::Condvar::new()));
        let service = crate::face_engine::FaceAnalysisService::new_with_pipeline_and_stages(
            1,
            crate::jobs::JobLimiter::new(2),
            admission,
            crate::rendering::pipeline::RenderPipeline::with_source_cache_capacity(
                root.path().join("soak-cache"),
                1,
                1,
                1,
            ),
            Arc::new(RunningStageBarrier {
                target: crate::face_engine::FaceAnalysisStage::Decoding,
                reached: reached.clone(),
                required_query: None,
            }),
            None,
        );
        assert!(service.enqueue(job.clone(), 0));
        assert!(
            wait_for_stage(&reached),
            "real service did not reach barrier; stats={:?}, durable={:?}",
            service.stats(),
            Connection::open(catalog_path(root.path()))
                .unwrap()
                .query_row(
                    "SELECT status, error FROM face_analysis_state WHERE asset_id=?1",
                    [job.asset_id],
                    |row| Ok((row.get::<_, String>(0)?, row.get::<_, Option<String>>(1)?))
                )
                .unwrap()
        );
        assert_eq!(service.stats().running, 1);

        let import_root = root.path().to_path_buf();
        let import_catalog = lightroom.clone();
        let transaction_begun = Arc::new(AtomicBool::new(false));
        let release_transaction = Arc::new(AtomicBool::new(false));
        let thread_begun = transaction_begun.clone();
        let thread_release = release_transaction.clone();
        let importer = std::thread::spawn(move || {
            install_lightroom_transaction_begun_hook(move || {
                thread_begun.store(true, std::sync::atomic::Ordering::Release);
                while !thread_release.load(std::sync::atomic::Ordering::Acquire) {
                    std::thread::yield_now();
                }
            });
            import_lightroom_inner(&import_root, &import_catalog, true, ImportCapabilities::ALL)
        });
        let overlap_deadline = Instant::now() + Duration::from_secs(5);
        while !transaction_begun.load(std::sync::atomic::Ordering::Acquire)
            && Instant::now() < overlap_deadline
        {
            std::thread::yield_now();
        }
        assert!(
            transaction_begun.load(std::sync::atomic::Ordering::Acquire),
            "Lightroom importer never entered its destination transaction"
        );
        assert_eq!(service.stats().running, 1);
        assert!(*reached.0.lock().unwrap());
        service.set_paused(true);
        drop(service);
        release_transaction.store(true, std::sync::atomic::Ordering::Release);
        importer.join().unwrap().unwrap();
        cancelled.store(true, std::sync::atomic::Ordering::Release);
        let _ = reader.join().unwrap();

        let cleanup_deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let state: String = Connection::open(catalog_path(root.path()))
                .unwrap()
                .query_row(
                    "SELECT status FROM face_analysis_state WHERE asset_id=?1",
                    [job.asset_id],
                    |row| row.get(0),
                )
                .unwrap();
            if state == "pending" || Instant::now() >= cleanup_deadline {
                break;
            }
            std::thread::yield_now();
        }
        let connection = Connection::open(catalog_path(root.path())).unwrap();
        assert_eq!(
            connection
                .query_row(
                    "SELECT status FROM face_analysis_state WHERE asset_id=?1",
                    [job.asset_id],
                    |row| row.get::<_, String>(0),
                )
                .unwrap(),
            "pending"
        );
        assert_eq!(
            connection
                .query_row(
                    "SELECT count(*) FROM faces WHERE asset_id=?1 AND provenance='rekesz'",
                    [job.asset_id],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap(),
            0
        );
        drop(connection);

        let restarted = crate::face_engine::FaceAnalysisService::new_with_pipeline(
            1,
            crate::jobs::JobLimiter::new(2),
            crate::scheduler::MaintenanceAdmission::new(),
            crate::rendering::pipeline::RenderPipeline::with_source_cache_capacity(
                root.path().join("soak-cache"),
                1,
                1,
                1,
            ),
        );
        assert!(restarted.enqueue(job.clone(), 0));
        let deadline = std::time::Instant::now() + Duration::from_secs(20);
        while restarted.stats().completed == 0 && std::time::Instant::now() < deadline {
            std::thread::yield_now();
        }
        assert_eq!(restarted.stats().completed, 1);
        drop(restarted);
        let connection =
            Connection::open(root.path().join(CATALOG_DIR).join(CATALOG_FILE)).unwrap();
        assert_eq!(
            connection
                .query_row(
                    "SELECT count(*) FROM faces WHERE asset_id=?1 AND provenance='rekesz'",
                    [job.asset_id],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap(),
            0,
            "cancel/restart must not leave or duplicate detected faces"
        );
        assert_eq!(
            connection
                .query_row(
                    "SELECT count(*) FROM face_embeddings embedding JOIN faces face
                     ON face.id=embedding.face_id WHERE face.asset_id=?1",
                    [job.asset_id],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap(),
            0
        );
        assert_eq!(
            connection
                .query_row(
                    "SELECT count(*) FROM face_assignments WHERE face_id=?1 AND person_id=?2
                     AND kind='rejected' AND provenance='rekesz'",
                    params![face_id, person_id],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap(),
            1
        );
        assert_eq!(
            connection
                .query_row(
                    "SELECT count(*) FROM effective_face_assignments
                     WHERE face_id=?1 AND person_id=?2",
                    params![face_id, person_id],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap(),
            0,
            "the preserved local rejection must suppress the effective assignment"
        );
        assert_eq!(people::diagnostics(root.path()).unwrap().analysis_errors, 0);
    }
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
fn migration_rejects_disagreeing_version_markers_without_schema_changes() {
    let root = tempfile::tempdir().unwrap();
    let directory = root.path().join(CATALOG_DIR);
    fs::create_dir(&directory).unwrap();
    let path = directory.join(CATALOG_FILE);
    let mut connection = Connection::open(&path).unwrap();
    connection
        .execute_batch(&format!(
            "PRAGMA application_id={SQLITE_APPLICATION_ID};
             PRAGMA user_version=1;
             CREATE TABLE catalog_meta(
                id INTEGER PRIMARY KEY, schema_version INTEGER NOT NULL,
                renderer_model INTEGER NOT NULL, created_at TEXT NOT NULL,
                last_imported_at TEXT);
             INSERT INTO catalog_meta VALUES(1, 2, 1, datetime('now'), NULL);"
        ))
        .unwrap();

    assert!(initialize_schema(&mut connection).is_err());
    assert!(initialize_schema(&mut connection).is_err());
    let backups = fs::read_dir(&directory)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            path.file_name().is_some_and(|name| {
                name.to_string_lossy()
                    .starts_with(&format!("{CATALOG_FILE}.pre-v1-"))
                    && name.to_string_lossy().ends_with("-backup.sqlite3")
            })
        })
        .collect::<Vec<_>>();
    assert_eq!(
        backups.len(),
        2,
        "each migration attempt gets a fresh backup"
    );
    for backup in backups {
        let restored = Connection::open_with_flags(
            backup,
            OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )
        .unwrap();
        assert_eq!(
            restored
                .query_row("PRAGMA integrity_check", [], |row| row.get::<_, String>(0))
                .unwrap(),
            "ok"
        );
        assert_eq!(
            restored
                .query_row("SELECT schema_version FROM catalog_meta", [], |row| {
                    row.get::<_, i64>(0)
                })
                .unwrap(),
            2
        );
    }
    assert_eq!(
        connection
            .query_row("PRAGMA user_version", [], |row| row.get::<_, i64>(0))
            .unwrap(),
        1
    );
    let people_table: bool = connection
        .query_row(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='people')",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert!(!people_table);
}

#[test]
fn v3_recognition_migration_rolls_back_all_columns_on_failure() {
    let root = tempfile::tempdir().unwrap();
    let directory = root.path().join(CATALOG_DIR);
    fs::create_dir(&directory).unwrap();
    let path = directory.join(CATALOG_FILE);
    let mut connection = Connection::open(&path).unwrap();
    initialize_schema(&mut connection).unwrap();
    connection
        .execute_batch(
            "INSERT INTO assets(id, relative_path, scope_key) VALUES(900, X'70686f746f2e6a7067', X'01');
             INSERT INTO people(id, display_name, created_at, updated_at)
                VALUES(900, 'migration person', datetime('now'), datetime('now'));
             INSERT INTO faces(id, asset_id, provenance, source_key, left, top, right, bottom,
                               origin, updated_at)
                VALUES(900, 900, 'rekesz', 'migration-face', .1, .1, .2, .2,
                       'rekesz-detected', datetime('now'));
             INSERT INTO face_assignments(face_id, person_id, kind, provenance, updated_at)
                VALUES(900, 900, 'confirmed', 'rekesz', datetime('now'));
             DROP VIEW effective_face_assignments;
             DROP VIEW effective_confirmations;
             DROP TABLE recognition_index_state;
             ALTER TABLE recognition_suggestions DROP COLUMN scored_index_generation;
             ALTER TABLE recognition_suggestions DROP COLUMN ann_index_version;
             UPDATE catalog_meta SET schema_version=3 WHERE id=1;
             PRAGMA user_version=3;
             CREATE TRIGGER reject_v4_migration BEFORE UPDATE ON catalog_meta
             BEGIN SELECT RAISE(ABORT, 'injected migration failure'); END;",
        )
        .unwrap();

    assert!(initialize_schema(&mut connection).is_err());
    assert_eq!(
        connection
            .query_row("PRAGMA user_version", [], |row| row.get::<_, i64>(0))
            .unwrap(),
        3
    );
    for column in ["ann_index_version", "scored_index_generation"] {
        let present: bool = connection
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM pragma_table_info('recognition_suggestions')
                               WHERE name=?1)",
                [column],
                |row| row.get(0),
            )
            .unwrap();
        assert!(!present, "failed migration leaked column {column}");
    }
    let backup = fs::read_dir(&directory)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .find(|path| {
            path.file_name().is_some_and(|name| {
                name.to_string_lossy()
                    .starts_with(&format!("{CATALOG_FILE}.pre-v3-"))
                    && name.to_string_lossy().ends_with("-backup.sqlite3")
            })
        })
        .unwrap();
    let restored_root = tempfile::tempdir().unwrap();
    let restored_directory = restored_root.path().join(CATALOG_DIR);
    fs::create_dir(&restored_directory).unwrap();
    let restored_path = restored_directory.join(CATALOG_FILE);
    fs::copy(backup, &restored_path).unwrap();
    let mut restored = Connection::open(&restored_path).unwrap();
    assert_eq!(
        restored
            .query_row("PRAGMA integrity_check", [], |row| row.get::<_, String>(0))
            .unwrap(),
        "ok"
    );
    assert_eq!(
        restored
            .query_row("SELECT schema_version FROM catalog_meta", [], |row| row
                .get::<_, i64>(0))
            .unwrap(),
        3
    );
    assert_eq!(
        restored
            .query_row(
                "SELECT count(*) FROM face_assignments WHERE face_id=900",
                [],
                |row| { row.get::<_, i64>(0) }
            )
            .unwrap(),
        1
    );
    restored
        .execute("DROP TRIGGER reject_v4_migration", [])
        .unwrap();
    initialize_schema(&mut restored).unwrap();
    assert_eq!(
        restored
            .query_row("PRAGMA user_version", [], |row| row.get::<_, i64>(0))
            .unwrap(),
        6
    );
    assert_eq!(
        restored
            .query_row(
                "SELECT count(*) FROM face_assignments WHERE face_id=900",
                [],
                |row| { row.get::<_, i64>(0) }
            )
            .unwrap(),
        1
    );
}

#[test]
fn v1_catalog_migrates_to_current_with_path_backup_and_ann_schema() {
    let root = tempfile::tempdir().unwrap();
    let directory = root.path().join(CATALOG_DIR);
    fs::create_dir(&directory).unwrap();
    let path = directory.join(CATALOG_FILE);
    let mut connection = Connection::open(&path).unwrap();
    initialize_schema(&mut connection).unwrap();
    connection
        .execute(
            "INSERT INTO assets(id, relative_path, scope_key) VALUES(901, ?1, ?2)",
            rusqlite::params![
                encode_relative_path(Path::new("legacy/photo.jpg")).unwrap(),
                scope_key(Path::new("legacy/photo.jpg")).unwrap()
            ],
        )
        .unwrap();
    connection
        .execute_batch(
            "PRAGMA foreign_keys=OFF;
             DROP VIEW effective_face_assignments; DROP VIEW effective_confirmations;
             DROP TABLE face_decision_journal; DROP TABLE recognition_policy_artifacts;
             DROP TABLE recognition_rescore_state; DROP TABLE recognition_refresh_queue;
             DROP TABLE recognition_index_state; DROP TABLE recognition_suggestions;
             DROP TABLE person_prototypes; DROP TABLE face_analysis_state;
             DROP TABLE face_embeddings; DROP TABLE face_cluster_memberships;
             DROP TABLE face_assignments; DROP TABLE faces; DROP TABLE external_people;
             DROP TABLE people;
             DROP TRIGGER IF EXISTS assets_scope_key_insert_guard;
             DROP TRIGGER IF EXISTS assets_scope_key_update_guard;
             DROP INDEX IF EXISTS assets_scope_key;
             ALTER TABLE assets DROP COLUMN scope_key;
             UPDATE catalog_meta SET schema_version=1 WHERE id=1;
             PRAGMA user_version=1;
             PRAGMA foreign_keys=ON;",
        )
        .unwrap();

    initialize_schema(&mut connection).unwrap();
    assert_eq!(
        connection
            .query_row("PRAGMA user_version", [], |row| row.get::<_, i64>(0))
            .unwrap(),
        6
    );
    assert_eq!(
        connection
            .query_row("SELECT scope_key FROM assets WHERE id=901", [], |row| {
                row.get::<_, Vec<u8>>(0)
            })
            .unwrap(),
        scope_key(Path::new("legacy/photo.jpg")).unwrap()
    );
    let ann_columns: i64 = connection
        .query_row(
            "SELECT count(*) FROM pragma_table_info('recognition_suggestions')
             WHERE name IN ('ann_index_version','scored_index_generation')",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(ann_columns, 2);
    let backup = fs::read_dir(&directory)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .find(|path| {
            path.file_name().is_some_and(|name| {
                name.to_string_lossy()
                    .starts_with(&format!("{CATALOG_FILE}.pre-v1-"))
                    && name.to_string_lossy().ends_with("-backup.sqlite3")
            })
        })
        .unwrap();
    let backup = Connection::open_with_flags(
        backup,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .unwrap();
    assert_eq!(
        backup
            .query_row("PRAGMA integrity_check", [], |row| row.get::<_, String>(0))
            .unwrap(),
        "ok"
    );
    assert_eq!(
        backup
            .query_row("PRAGMA user_version", [], |row| row.get::<_, i64>(0))
            .unwrap(),
        1
    );
    assert_eq!(
        backup
            .query_row("SELECT relative_path FROM assets WHERE id=901", [], |row| {
                row.get::<_, Vec<u8>>(0)
            })
            .unwrap(),
        encode_relative_path(Path::new("legacy/photo.jpg")).unwrap()
    );
}

#[test]
fn v2_catalog_migrates_to_v3_with_paths_and_effective_views() {
    let root = tempfile::tempdir().unwrap();
    let directory = root.path().join(CATALOG_DIR);
    fs::create_dir(&directory).unwrap();
    let path = directory.join(CATALOG_FILE);
    let connection = Connection::open(&path).unwrap();
    connection
        .execute_batch(&format!(
            "PRAGMA application_id={SQLITE_APPLICATION_ID}; PRAGMA user_version=2;
             CREATE TABLE catalog_meta(id INTEGER PRIMARY KEY, schema_version INTEGER NOT NULL,
                renderer_model INTEGER NOT NULL, created_at TEXT NOT NULL, last_imported_at TEXT);
             INSERT INTO catalog_meta VALUES(1, 2, 1, datetime('now'), NULL);
             CREATE TABLE assets(id INTEGER PRIMARY KEY, relative_path BLOB NOT NULL UNIQUE);
             CREATE TABLE develop_recipes(asset_id INTEGER PRIMARY KEY, recipe_json TEXT NOT NULL,
                recipe_digest BLOB NOT NULL, imported_at TEXT NOT NULL);
             CREATE TABLE hidden_subtrees(relative_path BLOB PRIMARY KEY, hidden_at TEXT NOT NULL);
             CREATE TABLE people(id INTEGER PRIMARY KEY, display_name TEXT NOT NULL,
                locally_renamed INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
             CREATE TABLE external_people(person_id INTEGER NOT NULL, provenance TEXT NOT NULL,
                external_key BLOB NOT NULL, PRIMARY KEY(provenance, external_key));
             CREATE TABLE faces(id INTEGER PRIMARY KEY, asset_id INTEGER NOT NULL,
                provenance TEXT NOT NULL, source_key TEXT NOT NULL, left REAL NOT NULL, top REAL NOT NULL,
                right REAL NOT NULL, bottom REAL NOT NULL, origin TEXT NOT NULL, ignored INTEGER NOT NULL,
                updated_at TEXT NOT NULL, UNIQUE(provenance, source_key));
             CREATE TABLE face_assignments(face_id INTEGER NOT NULL, person_id INTEGER NOT NULL,
                kind TEXT NOT NULL, provenance TEXT NOT NULL, confidence REAL, updated_at TEXT NOT NULL,
                PRIMARY KEY(face_id, person_id, kind, provenance));
             CREATE TABLE face_cluster_memberships(face_id INTEGER NOT NULL, provenance TEXT NOT NULL,
                cluster_key TEXT NOT NULL, PRIMARY KEY(face_id, provenance));
             CREATE TABLE face_embeddings(face_id INTEGER NOT NULL, model_id TEXT NOT NULL,
                dimensions INTEGER NOT NULL, vector BLOB NOT NULL, landmarks BLOB, created_at TEXT NOT NULL,
                PRIMARY KEY(face_id, model_id));
             CREATE TABLE face_analysis_state(asset_id INTEGER PRIMARY KEY, detector_model_id TEXT NOT NULL,
                source_size INTEGER NOT NULL, source_modified_ns INTEGER NOT NULL, status TEXT NOT NULL,
                attempts INTEGER NOT NULL, error TEXT, updated_at TEXT NOT NULL);"
        ))
        .unwrap();
    connection
        .execute(
            "INSERT INTO assets(id, relative_path) VALUES(1, ?1)",
            [encode_relative_path(Path::new("folder/photo.jpg")).unwrap()],
        )
        .unwrap();
    connection
        .execute_batch(
            "PRAGMA journal_mode=WAL;
         INSERT INTO people(id, display_name, created_at, updated_at)
             VALUES(1, 'Backup person', datetime('now'), datetime('now'));
         INSERT INTO faces(id, asset_id, provenance, source_key, left, top, right, bottom,
                           origin, ignored, updated_at)
             VALUES(1, 1, 'rekesz', 'backup-face', .1, .1, .2, .2, 'rekesz', 0, datetime('now'));
         INSERT INTO face_assignments(face_id, person_id, kind, provenance, updated_at)
             VALUES(1, 1, 'confirmed', 'rekesz', datetime('now'));",
        )
        .unwrap();

    drop(connection);
    prepare_existing_catalog(root.path()).unwrap();
    let connection = Connection::open(&path).unwrap();
    let backup_path = fs::read_dir(&directory)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .find(|path| {
            path.file_name().is_some_and(|name| {
                name.to_string_lossy()
                    .starts_with(&format!("{CATALOG_FILE}.pre-v2-"))
                    && name.to_string_lossy().ends_with("-backup.sqlite3")
            })
        })
        .unwrap();
    assert!(backup_path.is_file());
    let backup = Connection::open_with_flags(
        backup_path,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .unwrap();
    assert_eq!(
        backup
            .query_row("PRAGMA user_version", [], |row| row.get::<_, i64>(0))
            .unwrap(),
        2
    );
    assert_eq!(
        backup
            .query_row("PRAGMA integrity_check", [], |row| row.get::<_, String>(0))
            .unwrap(),
        "ok"
    );
    assert_eq!(
        backup
            .query_row("SELECT count(*) FROM face_assignments", [], |row| {
                row.get::<_, i64>(0)
            })
            .unwrap(),
        1
    );
    assert_eq!(
        backup
            .query_row("SELECT schema_version FROM catalog_meta", [], |row| {
                row.get::<_, i64>(0)
            })
            .unwrap(),
        2
    );
    assert_eq!(
        connection
            .query_row("PRAGMA user_version", [], |row| row.get::<_, i64>(0))
            .unwrap(),
        6
    );
    assert_eq!(
        connection
            .query_row("SELECT scope_key FROM assets WHERE id=1", [], |row| {
                row.get::<_, Vec<u8>>(0)
            })
            .unwrap(),
        scope_key(Path::new("folder/photo.jpg")).unwrap()
    );
    let effective_view: bool = connection
        .query_row(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='view' AND name='effective_face_assignments')",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert!(effective_view);
    drop(connection);
    assert!(load_recipes(root.path()).unwrap().is_empty());
    assert!(load_hidden_subtrees(root.path()).unwrap().is_empty());
    assert!(people::people_summary_rebuild_status(root.path()).is_ok());
}

#[test]
fn missing_catalog_has_idle_summary_status_without_implicit_creation() {
    let root = tempfile::tempdir().unwrap();
    assert_eq!(
        people::people_summary_rebuild_status(root.path()).unwrap(),
        people::PeopleSummaryRebuildStatus::default()
    );
    assert!(!catalog_path(root.path()).exists());
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

#[test]
fn asset_inventory_reconciles_identity_and_removal() {
    let root = tempfile::tempdir().unwrap();
    initialize_test_catalog(root.path()).unwrap();
    let first = root.path().join("first.jpg");
    let second = root.path().join("nested").join("second.heic");
    std::fs::create_dir_all(second.parent().unwrap()).unwrap();
    std::fs::write(&first, b"one").unwrap();
    std::fs::write(&second, b"two").unwrap();
    let epoch = || {
        Connection::open(catalog_path(root.path()))
            .unwrap()
            .query_row("SELECT source_epoch FROM people_summary_epoch", [], |row| {
                row.get::<_, i64>(0)
            })
            .unwrap()
    };
    let before = epoch();

    let initial = sync_asset_inventory(
        root.path(),
        &[first.clone(), second.clone()],
        "detector-a",
        "recognizer-a",
    )
    .unwrap();
    assert_eq!((initial.present, initial.added), (2, 2));
    assert_eq!(epoch(), before + 1);
    let stable = sync_asset_inventory(
        root.path(),
        &[first.clone(), second.clone()],
        "detector-a",
        "recognizer-a",
    )
    .unwrap();
    assert_eq!(stable.invalidated, 0);
    assert_eq!(epoch(), before + 1, "unchanged UPSERT must not invalidate");

    std::fs::write(&first, b"changed identity").unwrap();
    let changed = sync_asset_inventory(
        root.path(),
        std::slice::from_ref(&first),
        "detector-b",
        "recognizer-b",
    )
    .unwrap();
    assert_eq!((changed.invalidated, changed.removed), (1, 1));
    assert_eq!(epoch(), before + 3);

    let connection = Connection::open(catalog_path(root.path())).unwrap();
    let inventory: i64 = connection
        .query_row("SELECT count(*) FROM analysis_asset_inventory", [], |row| {
            row.get(0)
        })
        .unwrap();
    let state: (String, String, String) = connection
        .query_row(
            "SELECT detector_model_id, recognizer_model_id, status FROM face_analysis_state",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .unwrap();
    assert_eq!(inventory, 1);
    assert_eq!(
        state,
        ("detector-b".into(), "recognizer-b".into(), "pending".into())
    );
}

#[test]
fn large_inventory_sync_releases_writer_lock_between_batches() {
    let root = tempfile::tempdir().unwrap();
    initialize_test_catalog(root.path()).unwrap();
    let mut paths = Vec::new();
    for index in 0..257 {
        let path = root.path().join(format!("image-{index}.jpg"));
        std::fs::write(&path, b"image").unwrap();
        paths.push(path);
    }
    let mut boundary_visits = 0;
    sync_asset_inventory_with_progress(
        root.path(),
        &paths,
        "detector",
        "recognizer",
        InventoryIdentityMode::VerifyContents,
        |processed, _, _| {
            if processed == 256 {
                boundary_visits += 1;
                if boundary_visits == 2 {
                    let writer = Connection::open(catalog_path(root.path())).unwrap();
                    writer.busy_timeout(Duration::from_millis(50)).unwrap();
                    writer
                        .execute(
                            "UPDATE catalog_meta SET last_imported_at='batch-writer'",
                            [],
                        )
                        .expect("an interactive writer can run between inventory batches");
                }
            }
        },
        || true,
    )
    .unwrap();
    assert!(boundary_visits >= 2);
}

#[test]
fn metadata_first_inventory_reuses_fingerprint_without_reading_unchanged_contents() {
    let root = tempfile::tempdir().unwrap();
    initialize_test_catalog(root.path()).unwrap();
    let path = root.path().join("unchanged.jpg");
    std::fs::write(&path, b"unchanged image bytes").unwrap();
    sync_asset_inventory(
        root.path(),
        std::slice::from_ref(&path),
        "detector",
        "recognizer",
    )
    .unwrap();

    let chunks = std::rc::Rc::new(std::cell::Cell::new(0));
    let observed = chunks.clone();
    install_content_identity_chunk_hook(move || observed.set(observed.get() + 1));
    sync_asset_inventory_with_progress(
        root.path(),
        std::slice::from_ref(&path),
        "detector",
        "recognizer",
        InventoryIdentityMode::MetadataFirst,
        |_, _, _| {},
        || true,
    )
    .unwrap();
    install_content_identity_chunk_hook(|| {});
    assert_eq!(chunks.get(), 0, "unchanged content must not be reopened");
}

#[test]
fn v5_identity_migration_returns_analysis_to_pending_and_discards_derived_vectors() {
    let root = tempfile::tempdir().unwrap();
    initialize_test_catalog(root.path()).unwrap();
    let source = root.path().join("migration.jpg");
    std::fs::write(&source, b"migration-source").unwrap();
    sync_asset_inventory(
        root.path(),
        std::slice::from_ref(&source),
        "detector",
        "recognizer",
    )
    .unwrap();
    let catalog_path = catalog_path(root.path());
    let mut connection = Connection::open(&catalog_path).unwrap();
    connection.execute_batch(
            "INSERT INTO faces(id,asset_id,provenance,source_key,left,top,right,bottom,origin,
             ignored,updated_at) VALUES(70,1,'rekesz','migration-derived',.1,.1,.2,.2,
             'rekesz-detected',0,datetime('now'));
         INSERT INTO face_embeddings(face_id,model_id,dimensions,vector,source_size,
             source_modified_ns,source_fingerprint,created_at)
         SELECT 70,'recognizer',1,zeroblob(4),source_size,source_modified_ns,
             source_fingerprint,datetime('now') FROM analysis_asset_inventory WHERE asset_id=1;
         UPDATE face_analysis_state SET status='complete',attempts=3;
         PRAGMA foreign_keys=OFF;
         ALTER TABLE analysis_asset_inventory RENAME TO analysis_asset_inventory_v6;
         ALTER TABLE face_analysis_state RENAME TO face_analysis_state_v6;
         ALTER TABLE face_embeddings RENAME TO face_embeddings_v6;
         CREATE TABLE analysis_asset_inventory(
           asset_id INTEGER PRIMARY KEY REFERENCES assets(id) ON DELETE CASCADE,
           source_size INTEGER NOT NULL,source_modified_ns INTEGER NOT NULL,
           decoder_model_id TEXT NOT NULL DEFAULT 'renderer-6',discovered_at TEXT NOT NULL);
         CREATE TABLE face_analysis_state(
           asset_id INTEGER PRIMARY KEY REFERENCES assets(id) ON DELETE CASCADE,
           detector_model_id TEXT NOT NULL,recognizer_model_id TEXT NOT NULL DEFAULT 'sface-2021dec',
           decoder_model_id TEXT NOT NULL DEFAULT 'renderer-6',source_size INTEGER NOT NULL,
           source_modified_ns INTEGER NOT NULL,status TEXT NOT NULL,
           attempts INTEGER NOT NULL DEFAULT 0,worker_id TEXT,lease_expires_at TEXT,
           retry_after TEXT,error TEXT,updated_at TEXT NOT NULL);
         CREATE TABLE face_embeddings(
           face_id INTEGER NOT NULL REFERENCES faces(id) ON DELETE CASCADE,model_id TEXT NOT NULL,
           dimensions INTEGER NOT NULL,vector BLOB NOT NULL,landmarks BLOB,
           source_size INTEGER,source_modified_ns INTEGER,created_at TEXT NOT NULL,
           PRIMARY KEY(face_id,model_id));
         INSERT INTO analysis_asset_inventory SELECT asset_id,source_size,source_modified_ns,
           decoder_model_id,discovered_at FROM analysis_asset_inventory_v6;
         INSERT INTO face_analysis_state SELECT asset_id,detector_model_id,recognizer_model_id,
           decoder_model_id,source_size,source_modified_ns,status,attempts,worker_id,
           lease_expires_at,retry_after,error,updated_at FROM face_analysis_state_v6;
         INSERT INTO face_embeddings SELECT face_id,model_id,dimensions,vector,landmarks,
           source_size,source_modified_ns,created_at FROM face_embeddings_v6;
         DROP TABLE analysis_asset_inventory_v6;
         DROP TABLE face_analysis_state_v6;
         DROP TABLE face_embeddings_v6;
         CREATE INDEX face_analysis_eligibility
           ON face_analysis_state(status,retry_after,lease_expires_at);
         UPDATE catalog_meta SET schema_version=5 WHERE id=1;
         PRAGMA user_version=5;
         CREATE TRIGGER fail_v6_migration BEFORE UPDATE OF schema_version ON catalog_meta
           WHEN NEW.schema_version=6 BEGIN SELECT RAISE(ABORT,'injected v6 failure'); END;
         PRAGMA foreign_keys=ON;")
        .unwrap();
    let failure = initialize_schema(&mut connection).unwrap_err();
    assert!(format!("{failure:#}").contains("injected v6 failure"));
    for (table, column) in [
        ("analysis_asset_inventory", "source_fingerprint"),
        ("face_analysis_state", "source_fingerprint"),
        ("face_analysis_state", "failure_class"),
        ("face_embeddings", "source_fingerprint"),
    ] {
        let present: bool = connection
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM pragma_table_info(?1) WHERE name=?2)",
                params![table, column],
                |row| row.get(0),
            )
            .unwrap();
        assert!(!present, "failed migration leaked {table}.{column}");
    }
    connection
        .execute("DROP TRIGGER fail_v6_migration", [])
        .unwrap();
    initialize_schema(&mut connection).unwrap();
    let (version, status, attempts, fingerprint, embeddings): (
        i64,
        String,
        i64,
        Option<Vec<u8>>,
        i64,
    ) = connection
        .query_row(
            "SELECT (SELECT schema_version FROM catalog_meta),state.status,state.attempts,
                state.source_fingerprint,(SELECT count(*) FROM face_embeddings)
           FROM face_analysis_state state WHERE asset_id=1",
            [],
            |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                ))
            },
        )
        .unwrap();
    assert_eq!(version, 6);
    assert_eq!(
        (status.as_str(), attempts, fingerprint, embeddings),
        ("pending", 0, Some(Vec::new()), 0)
    );
    assert_eq!(
        connection
            .query_row("PRAGMA foreign_key_check", [], |row| row
                .get::<_, Option<String>>(0))
            .optional()
            .unwrap(),
        None
    );
    let backup = std::fs::read_dir(catalog_path.parent().unwrap())
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .find(|path| {
            path.file_name().is_some_and(|name| {
                name.to_string_lossy()
                    .starts_with("catalog.sqlite3.pre-v5-")
                    && name.to_string_lossy().ends_with("-backup.sqlite3")
            })
        })
        .unwrap();
    let backup = Connection::open(backup).unwrap();
    assert_eq!(
        backup
            .query_row("PRAGMA integrity_check", [], |row| row.get::<_, String>(0))
            .unwrap(),
        "ok"
    );
    assert_eq!(
        backup
            .query_row("PRAGMA user_version", [], |row| row.get::<_, i64>(0))
            .unwrap(),
        5
    );
    let legacy_column: bool = backup
        .query_row(
            "SELECT EXISTS(SELECT 1 FROM pragma_table_info('face_analysis_state')
          WHERE name='source_fingerprint')",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert!(!legacy_column);
}

#[test]
fn inventory_fingerprint_invalidates_metadata_preserving_byte_replacement() {
    let root = tempfile::tempdir().unwrap();
    initialize_test_catalog(root.path()).unwrap();
    let source = root.path().join("same.jpg");
    std::fs::write(&source, b"AAAA-same-length").unwrap();
    sync_asset_inventory(
        root.path(),
        std::slice::from_ref(&source),
        "detector-a",
        "recognizer-a",
    )
    .unwrap();
    let original_modified = std::fs::metadata(&source).unwrap().modified().unwrap();
    let connection = Connection::open(catalog_path(root.path())).unwrap();
    connection
        .execute_batch(
            "UPDATE face_analysis_state SET status='complete';
         INSERT INTO faces(id,asset_id,provenance,source_key,left,top,right,bottom,origin,
             ignored,updated_at) VALUES(50,1,'rekesz','derived',.1,.1,.2,.2,
             'rekesz-detected',0,datetime('now'));
         INSERT INTO face_embeddings(face_id,model_id,dimensions,vector,source_size,
             source_modified_ns,source_fingerprint,created_at)
         SELECT 50,'recognizer-a',1,zeroblob(4),source_size,source_modified_ns,
             source_fingerprint,datetime('now') FROM analysis_asset_inventory WHERE asset_id=1;",
        )
        .unwrap();
    let before: Vec<u8> = connection
        .query_row(
            "SELECT source_fingerprint FROM analysis_asset_inventory WHERE asset_id=1",
            [],
            |row| row.get(0),
        )
        .unwrap();
    drop(connection);

    std::fs::write(&source, b"BBBB-same-length").unwrap();
    std::fs::File::options()
        .write(true)
        .open(&source)
        .unwrap()
        .set_times(std::fs::FileTimes::new().set_modified(original_modified))
        .unwrap();
    let changed = sync_asset_inventory(
        root.path(),
        std::slice::from_ref(&source),
        "detector-a",
        "recognizer-a",
    )
    .unwrap();
    assert_eq!(changed.invalidated, 1);
    let connection = Connection::open(catalog_path(root.path())).unwrap();
    let (after, status, embeddings): (Vec<u8>, String, i64) = connection
        .query_row(
            "SELECT inventory.source_fingerprint,state.status,
                (SELECT count(*) FROM face_embeddings embedding JOIN faces face
                   ON face.id=embedding.face_id WHERE face.asset_id=1)
           FROM analysis_asset_inventory inventory JOIN face_analysis_state state
             ON state.asset_id=inventory.asset_id WHERE inventory.asset_id=1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .unwrap();
    assert_ne!(after, before);
    assert_eq!(status, "pending");
    assert_eq!(
        embeddings, 0,
        "stale derived embeddings must be invalidated immediately"
    );
}

#[test]
fn analysis_retry_classes_have_bounded_automatic_and_explicit_manual_recovery() {
    let root = tempfile::tempdir().unwrap();
    initialize_test_catalog(root.path()).unwrap();
    let source = root.path().join("retry.jpg");
    std::fs::write(&source, b"retry-source").unwrap();
    crate::face_engine::sync_analysis_inventory(root.path(), std::slice::from_ref(&source))
        .unwrap();
    let connection = Connection::open(catalog_path(root.path())).unwrap();
    connection
        .execute(
            "UPDATE face_analysis_state SET status='error',attempts=5,failure_class='exhausted',
           error='old-version-failed',retry_after=NULL,decoder_model_id='older-decoder'
         WHERE asset_id=1",
            [],
        )
        .unwrap();
    for attempt in 1..=5 {
        crate::face_engine::record_test_analysis_failure(root.path(), 1, false).unwrap();
        let (attempts, class, has_retry): (i64, String, bool) = connection
            .query_row(
                "SELECT attempts,failure_class,retry_after IS NOT NULL
               FROM face_analysis_state WHERE asset_id=1",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        assert_eq!(attempts, attempt);
        assert_eq!(
            class,
            if attempt == 5 {
                "exhausted"
            } else {
                "transient"
            }
        );
        assert_eq!(has_retry, attempt < 5);
    }
    assert!(
        crate::face_engine::pending_analysis_jobs(root.path(), 10)
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        crate::face_engine::next_analysis_wake(root.path()).unwrap(),
        None
    );
    assert_eq!(
        people::retry_failed_analysis(root.path())
            .unwrap()
            .analysis_jobs,
        1
    );
    assert_eq!(
        crate::face_engine::pending_analysis_jobs(root.path(), 10)
            .unwrap()
            .len(),
        1
    );

    crate::face_engine::record_test_analysis_failure(root.path(), 1, true).unwrap();
    assert!(
        crate::face_engine::pending_analysis_jobs(root.path(), 10)
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        crate::face_engine::next_analysis_wake(root.path()).unwrap(),
        None
    );
    connection
        .execute(
            "UPDATE face_analysis_state SET decoder_model_id='older-decoder' WHERE asset_id=1",
            [],
        )
        .unwrap();
    assert_eq!(
        crate::face_engine::pending_analysis_jobs(root.path(), 10)
            .unwrap()
            .len(),
        1
    );
}

#[test]
fn setup_failure_resets_exhausted_budget_for_each_changed_engine_or_source_identity() {
    let _ort = crate::face_engine::ORT_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    for (identity, mutation) in [
        (
            "detector",
            "UPDATE face_analysis_state SET detector_model_id='stale-version' WHERE asset_id=1",
        ),
        (
            "recognizer",
            "UPDATE face_analysis_state SET recognizer_model_id='stale-version' WHERE asset_id=1",
        ),
        (
            "decoder",
            "UPDATE face_analysis_state SET decoder_model_id='stale-version' WHERE asset_id=1",
        ),
        ("source-size", ""),
        ("source-mtime", ""),
        ("source-fingerprint", ""),
    ] {
        let root = tempfile::tempdir().unwrap();
        initialize_test_catalog(root.path()).unwrap();
        let source = root.path().join("setup.jpg");
        std::fs::write(&source, b"setup-failure-source").unwrap();
        crate::face_engine::sync_analysis_inventory(root.path(), std::slice::from_ref(&source))
            .unwrap();
        let connection = Connection::open(catalog_path(root.path())).unwrap();
        connection
            .execute(
                "UPDATE face_analysis_state SET status='error',attempts=5,
               failure_class='exhausted',error='old setup failure',retry_after=NULL
             WHERE asset_id=1",
                [],
            )
            .unwrap();
        if mutation.is_empty() {
            let original_modified = std::fs::metadata(&source).unwrap().modified().unwrap();
            match identity {
                "source-size" => std::fs::write(&source, b"setup-failure-source-longer").unwrap(),
                "source-mtime" => std::fs::File::options()
                    .write(true)
                    .open(&source)
                    .unwrap()
                    .set_times(
                        std::fs::FileTimes::new().set_modified(std::time::SystemTime::UNIX_EPOCH),
                    )
                    .unwrap(),
                "source-fingerprint" => {
                    std::fs::write(&source, b"SETUP-FAILURE-SOURCE").unwrap();
                    std::fs::File::options()
                        .write(true)
                        .open(&source)
                        .unwrap()
                        .set_times(std::fs::FileTimes::new().set_modified(original_modified))
                        .unwrap();
                }
                _ => unreachable!(),
            }
            let current = stable_content_identity(&source).unwrap();
            connection
                .execute(
                    "UPDATE analysis_asset_inventory SET source_size=?1,source_modified_ns=?2,
                   source_fingerprint=?3 WHERE asset_id=1",
                    params![
                        current.size,
                        current.modified_ns,
                        current.fingerprint.as_slice()
                    ],
                )
                .unwrap();
        } else {
            connection.execute(mutation, []).unwrap();
        }
        drop(connection);
        crate::face_engine::install_face_engine_setup_failure(root.path().to_path_buf(), 1);
        let service = crate::face_engine::FaceAnalysisService::new_with_pipeline(
            1,
            crate::jobs::JobLimiter::new(2),
            crate::scheduler::MaintenanceAdmission::new(),
            crate::rendering::pipeline::RenderPipeline::with_source_cache_capacity(
                root.path().join("setup-cache"),
                1,
                1,
                1,
            ),
        );
        assert!(service.enqueue(
            crate::face_engine::FaceAnalyzeJob {
                root: root.path().to_path_buf(),
                asset_id: 1,
            },
            0
        ));
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        while service.stats().failed == 0 && std::time::Instant::now() < deadline {
            std::thread::yield_now();
        }
        assert_eq!(
            service.stats().failed,
            1,
            "{identity} setup failure did not run"
        );
        drop(service);
        let connection = Connection::open(catalog_path(root.path())).unwrap();
        let state: (i64, String, bool, String, String, String, String, bool) = connection
            .query_row(
                "SELECT state.attempts,state.failure_class,state.retry_after IS NOT NULL,
                    state.detector_model_id,state.recognizer_model_id,state.decoder_model_id,state.error,
                    state.source_size=inventory.source_size
                      AND state.source_modified_ns=inventory.source_modified_ns
                      AND state.source_fingerprint=inventory.source_fingerprint
               FROM face_analysis_state state JOIN analysis_asset_inventory inventory
                 ON inventory.asset_id=state.asset_id WHERE state.asset_id=1",
                [],
                |row| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get(3)?,
                        row.get(4)?,
                        row.get(5)?,
                        row.get(6)?,
                        row.get(7)?,
                    ))
                },
            )
            .unwrap();
        assert_eq!(state.0, 1, "{identity} must receive a fresh retry budget");
        assert_eq!(state.1, "transient");
        assert!(state.2);
        assert_eq!(state.3, "yunet-2023mar+sface-2021dec");
        assert_eq!(state.4, "sface-2021dec");
        assert_eq!(state.5, crate::rendering::DECODER_MODEL_ID);
        assert_eq!(state.6, "face-engine-unavailable");
        assert!(
            state.7,
            "{identity} must persist the current inventory identity"
        );
    }
}

#[test]
fn repeated_service_setup_failures_exhaust_unchanged_identity_after_five_attempts() {
    let _ort = crate::face_engine::ORT_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let root = tempfile::tempdir().unwrap();
    initialize_test_catalog(root.path()).unwrap();
    let source = root.path().join("repeated-setup.jpg");
    std::fs::write(&source, b"repeated-setup-failure-source").unwrap();
    crate::face_engine::sync_analysis_inventory(root.path(), std::slice::from_ref(&source))
        .unwrap();

    for attempt in 1..=5_i64 {
        assert_eq!(
            crate::face_engine::pending_analysis_jobs(root.path(), 10)
                .unwrap()
                .len(),
            1,
            "attempt {attempt} must become automatically eligible when due"
        );
        crate::face_engine::install_face_engine_setup_failure(root.path().to_path_buf(), 1);
        let service = crate::face_engine::FaceAnalysisService::new_with_pipeline(
            1,
            crate::jobs::JobLimiter::new(2),
            crate::scheduler::MaintenanceAdmission::new(),
            crate::rendering::pipeline::RenderPipeline::with_source_cache_capacity(
                root.path().join(format!("setup-cache-{attempt}")),
                1,
                1,
                1,
            ),
        );
        assert!(service.enqueue(
            crate::face_engine::FaceAnalyzeJob {
                root: root.path().to_path_buf(),
                asset_id: 1,
            },
            0,
        ));
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        while service.stats().failed == 0 && std::time::Instant::now() < deadline {
            std::thread::yield_now();
        }
        assert_eq!(
            service.stats().failed,
            1,
            "setup attempt {attempt} did not run"
        );
        drop(service);

        let connection = Connection::open(catalog_path(root.path())).unwrap();
        let (stored_attempts, class, retry, backoff): (i64, String, Option<String>, Option<i64>) =
            connection
                .query_row(
                    "SELECT attempts,failure_class,retry_after,
                            unixepoch(retry_after)-unixepoch(updated_at)
                       FROM face_analysis_state WHERE asset_id=1",
                    [],
                    |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
                )
                .unwrap();
        assert_eq!(stored_attempts, attempt);
        if attempt < 5 {
            assert_eq!(class, "transient");
            assert!(retry.is_some());
            assert_eq!(backoff, Some((1_i64 << attempt).min(3600)));
            assert!(
                crate::face_engine::next_analysis_wake(root.path())
                    .unwrap()
                    .is_some()
            );
            assert!(
                crate::face_engine::pending_analysis_jobs(root.path(), 10)
                    .unwrap()
                    .is_empty()
            );
            connection
                .execute(
                    "UPDATE face_analysis_state SET retry_after=datetime('now','-1 second')
                       WHERE asset_id=1",
                    [],
                )
                .unwrap();
        } else {
            assert_eq!(class, "exhausted");
            assert_eq!(retry, None);
            assert_eq!(backoff, None);
            assert!(
                crate::face_engine::pending_analysis_jobs(root.path(), 10)
                    .unwrap()
                    .is_empty()
            );
            assert_eq!(
                crate::face_engine::next_analysis_wake(root.path()).unwrap(),
                None
            );
        }
    }
}

#[test]
fn losing_service_setup_failure_cannot_clobber_an_active_analysis_claim() {
    let _ort = crate::face_engine::ORT_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let root = tempfile::tempdir().unwrap();
    initialize_test_catalog(root.path()).unwrap();
    let source = root.path().join("setup-race.png");
    image::RgbImage::new(32, 32).save(&source).unwrap();
    crate::face_engine::sync_analysis_inventory(root.path(), std::slice::from_ref(&source))
        .unwrap();
    let reached = Arc::new((std::sync::Mutex::new(false), std::sync::Condvar::new()));
    let release = Arc::new((std::sync::Mutex::new(false), std::sync::Condvar::new()));
    let service_a = crate::face_engine::FaceAnalysisService::new_with_pipeline_and_stages(
        1,
        crate::jobs::JobLimiter::new(3),
        crate::scheduler::MaintenanceAdmission::new(),
        crate::rendering::pipeline::RenderPipeline::with_source_cache_capacity(
            root.path().join("race-cache-a"),
            1,
            1,
            1,
        ),
        Arc::new(ReleasableStageBarrier {
            target: crate::face_engine::FaceAnalysisStage::Decoding,
            reached: reached.clone(),
            release: release.clone(),
        }),
        None,
    );
    let job = crate::face_engine::FaceAnalyzeJob {
        root: root.path().to_path_buf(),
        asset_id: 1,
    };
    assert!(service_a.enqueue(job.clone(), 0));
    assert!(
        wait_for_stage(&reached),
        "service A did not establish its running claim"
    );
    let before: (String, String) = Connection::open(catalog_path(root.path()))
        .unwrap()
        .query_row(
            "SELECT status,worker_id FROM face_analysis_state WHERE asset_id=1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .unwrap();
    assert_eq!(before.0, "running");

    crate::face_engine::install_face_engine_setup_failure(root.path().to_path_buf(), 1);
    let service_b = crate::face_engine::FaceAnalysisService::new_with_pipeline(
        1,
        crate::jobs::JobLimiter::new(3),
        crate::scheduler::MaintenanceAdmission::new(),
        crate::rendering::pipeline::RenderPipeline::with_source_cache_capacity(
            root.path().join("race-cache-b"),
            1,
            1,
            1,
        ),
    );
    assert!(service_b.enqueue(job.clone(), 0));
    let deadline = Instant::now() + Duration::from_secs(10);
    while service_b.stats().failed == 0 && Instant::now() < deadline {
        std::thread::yield_now();
    }
    assert_eq!(service_b.stats().failed, 1);
    let during: (String, String, Option<String>) = Connection::open(catalog_path(root.path()))
        .unwrap()
        .query_row(
            "SELECT status,worker_id,error FROM face_analysis_state WHERE asset_id=1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .unwrap();
    assert_eq!(
        (during.0, during.1, during.2),
        ("running".into(), before.1, None),
        "losing setup failure must not clear the winning lease"
    );
    *release.0.lock().unwrap() = true;
    release.1.notify_all();
    let deadline = Instant::now() + Duration::from_secs(20);
    while service_a.stats().completed == 0 && Instant::now() < deadline {
        std::thread::yield_now();
    }
    assert_eq!(service_a.stats().completed, 1);
    #[derive(Debug, PartialEq)]
    struct CompletedState {
        status: String,
        attempts: i64,
        worker_id: Option<String>,
        lease_expires_at: Option<String>,
        error: Option<String>,
        failure_class: Option<String>,
        retry_after: Option<String>,
        detector_model_id: String,
        recognizer_model_id: String,
        decoder_model_id: String,
        source_size: i64,
        source_modified_ns: i64,
        source_fingerprint: Vec<u8>,
    }
    let read_completed_state = || {
        Connection::open(catalog_path(root.path()))
            .unwrap()
            .query_row(
                "SELECT status,attempts,worker_id,lease_expires_at,error,failure_class,retry_after,
                        detector_model_id,recognizer_model_id,decoder_model_id,source_size,
                        source_modified_ns,source_fingerprint
                   FROM face_analysis_state WHERE asset_id=1",
                [],
                |row| {
                    Ok(CompletedState {
                        status: row.get(0)?,
                        attempts: row.get(1)?,
                        worker_id: row.get(2)?,
                        lease_expires_at: row.get(3)?,
                        error: row.get(4)?,
                        failure_class: row.get(5)?,
                        retry_after: row.get(6)?,
                        detector_model_id: row.get(7)?,
                        recognizer_model_id: row.get(8)?,
                        decoder_model_id: row.get(9)?,
                        source_size: row.get(10)?,
                        source_modified_ns: row.get(11)?,
                        source_fingerprint: row.get(12)?,
                    })
                },
            )
            .unwrap()
    };
    let completed = read_completed_state();
    assert_eq!(completed.status, "complete");
    assert_eq!(completed.attempts, 0);
    assert_eq!(completed.worker_id, None);
    assert_eq!(completed.lease_expires_at, None);
    assert_eq!(completed.error, None);
    assert_eq!(completed.failure_class, None);
    assert_eq!(completed.retry_after, None);

    crate::face_engine::install_face_engine_setup_failure(root.path().to_path_buf(), 1);
    let service_c = crate::face_engine::FaceAnalysisService::new_with_pipeline(
        1,
        crate::jobs::JobLimiter::new(3),
        crate::scheduler::MaintenanceAdmission::new(),
        crate::rendering::pipeline::RenderPipeline::with_source_cache_capacity(
            root.path().join("race-cache-c"),
            1,
            1,
            1,
        ),
    );
    assert!(service_c.enqueue(job, 0));
    let deadline = Instant::now() + Duration::from_secs(10);
    while service_c.stats().failed == 0 && Instant::now() < deadline {
        std::thread::yield_now();
    }
    assert_eq!(service_c.stats().failed, 1);
    assert_eq!(
        read_completed_state(),
        completed,
        "a setup failure must not mutate an identity-current completed analysis"
    );
}

#[test]
fn analysis_commit_revalidates_content_not_only_size_and_mtime() {
    let root = tempfile::tempdir().unwrap();
    initialize_test_catalog(root.path()).unwrap();
    let source = root.path().join("race.jpg");
    std::fs::write(&source, b"original-bytes").unwrap();
    crate::face_engine::sync_analysis_inventory(root.path(), std::slice::from_ref(&source))
        .unwrap();
    let modified = std::fs::metadata(&source).unwrap().modified().unwrap();
    std::fs::write(&source, b"replaced-bytes").unwrap();
    std::fs::File::options()
        .write(true)
        .open(&source)
        .unwrap()
        .set_times(std::fs::FileTimes::new().set_modified(modified))
        .unwrap();
    let error = crate::face_engine::store_test_analysis(
        root.path(),
        1,
        [0.1, 0.1, 0.2, 0.2],
        [[0.15, 0.15]; 5],
    )
    .unwrap_err();
    assert!(format!("{error:#}").contains("identity changed"));
    let connection = Connection::open(catalog_path(root.path())).unwrap();
    assert_eq!(
        connection
            .query_row("SELECT count(*) FROM faces WHERE asset_id=1", [], |row| row
                .get::<_, i64>(0))
            .unwrap(),
        0
    );
}

#[test]
fn analysis_commit_rejects_metadata_only_source_race() {
    let root = tempfile::tempdir().unwrap();
    initialize_test_catalog(root.path()).unwrap();
    let source = root.path().join("metadata-race.jpg");
    std::fs::write(&source, b"unchanged-content").unwrap();
    crate::face_engine::sync_analysis_inventory(root.path(), std::slice::from_ref(&source))
        .unwrap();
    std::fs::File::options()
        .write(true)
        .open(&source)
        .unwrap()
        .set_times(std::fs::FileTimes::new().set_modified(std::time::SystemTime::UNIX_EPOCH))
        .unwrap();
    let error = crate::face_engine::store_test_analysis(
        root.path(),
        1,
        [0.1, 0.1, 0.2, 0.2],
        [[0.15, 0.15]; 5],
    )
    .unwrap_err();
    assert!(format!("{error:#}").contains("identity changed"));
}

#[test]
fn stable_identity_retries_a_file_mutated_during_chunked_hashing() {
    let root = tempfile::tempdir().unwrap();
    let source = root.path().join("chunked.bin");
    std::fs::write(&source, vec![b'A'; 192 * 1024]).unwrap();
    let mutations = std::rc::Rc::new(std::cell::Cell::new(0_u32));
    let hook_mutations = mutations.clone();
    let hook_source = source.clone();
    install_content_identity_chunk_hook(move || {
        let invocation = hook_mutations.get();
        hook_mutations.set(invocation + 1);
        if invocation == 0 {
            std::fs::write(&hook_source, vec![b'B'; 192 * 1024]).unwrap();
            std::fs::File::options()
                .write(true)
                .open(&hook_source)
                .unwrap()
                .set_times(
                    std::fs::FileTimes::new().set_modified(std::time::SystemTime::UNIX_EPOCH),
                )
                .unwrap();
        }
    });
    let observed = stable_content_identity(&source).unwrap();
    install_content_identity_chunk_hook(|| {});
    let stable = stable_content_identity(&source).unwrap();
    assert!(
        mutations.get() >= 2,
        "identity reader must retry after mid-hash mutation"
    );
    assert_eq!(observed, stable);
}
