use super::*;
use test_r::test;

fn dirs() -> (tempfile::TempDir, directories::ProjectDirs) {
    let temp = tempfile::tempdir().unwrap();
    let dirs = directories::ProjectDirs::from_path(temp.path().join("project")).unwrap();
    (temp, dirs)
}

#[test]
fn persisted_settings_are_loaded_and_cli_values_override_them() {
    let (_temp, dirs) = dirs();
    let path = dirs.config_dir().join("settings.json");
    let stored = AppSettings {
        root: "/stored".into(),
        image_cache_size: 12,
        pair_preference: PairPreference::Raw,
        cache_root: "/stored-cache".into(),
        job_parallelism: 6,
        raw_parallelism: 2,
        online_map_permission: OnlineMapPermission::Unknown,
        map_cache_bytes: DEFAULT_MAP_CACHE_BYTES,
        map_switcher_placement: MapSwitcherPlacement::Header,
        automatic_face_analysis: true,
        face_workers: default_face_workers(),
        face_worker_memory_bytes: default_face_worker_memory_bytes(),
        face_crop_disk_cache: FaceCropDiskCachePolicy::default(),
        face_gpu: FaceGpuPolicy::default(),
        import_recent_sources: Vec::new(),
        import_destination_preset: Default::default(),
        import_include_duplicates: false,
    };
    save(&path, &stored).unwrap();
    let config = from_args_with_dirs(
        [
            "rekesz",
            "--image-cache-size",
            "3",
            "--cr3-jpeg-preference",
            "jpeg",
        ]
        .into_iter()
        .map(OsString::from),
        dirs,
    )
    .unwrap();
    assert_eq!(config.settings.image_cache_size, 3);
    assert_eq!(config.settings.pair_preference, PairPreference::Jpeg);
    assert_eq!(config.settings.job_parallelism, 6);
    assert_eq!(config.settings.root, PathBuf::from("/stored"));
}

#[test]
fn save_creates_the_platform_configuration_directory() {
    let (_temp, dirs) = dirs();
    let path = dirs.config_dir().join("settings.json");
    let settings = defaults(&dirs).unwrap();
    save(&path, &settings).unwrap();
    assert_eq!(
        serde_json::from_slice::<AppSettings>(&fs::read(path).unwrap()).unwrap(),
        settings
    );
}

#[test]
fn legacy_settings_receive_private_map_defaults() {
    let (_temp, dirs) = dirs();
    let path = dirs.config_dir().join("settings.json");
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(
        &path,
        br#"{
            "root": "/pictures",
            "image_cache_size": 8,
            "pair_preference": "raw",
            "cache_root": "/cache",
            "job_parallelism": 4,
            "raw_parallelism": 2
        }"#,
    )
    .unwrap();
    let config = from_args_with_dirs(["rekesz"].map(OsString::from), dirs).unwrap();
    assert_eq!(
        config.settings.online_map_permission,
        OnlineMapPermission::Unknown
    );
    assert_eq!(config.settings.map_cache_bytes, DEFAULT_MAP_CACHE_BYTES);
    assert_eq!(
        config.settings.map_switcher_placement,
        MapSwitcherPlacement::Header
    );
}

#[test]
fn rejects_an_empty_map_cache() {
    let (_temp, dirs) = dirs();
    let mut settings = defaults(&dirs).unwrap();
    settings.map_cache_bytes = 0;
    assert!(save(&dirs.config_dir().join("settings.json"), &settings).is_err());
}

#[test]
fn clap_rejects_zero_and_unknown_options() {
    let (_temp, project_dirs) = dirs();
    assert!(
        from_args_with_dirs(
            ["rekesz", "--job-parallelism", "0"].map(OsString::from),
            project_dirs,
        )
        .is_err()
    );
    let (_temp, project_dirs) = dirs();
    assert!(
        from_args_with_dirs(["rekesz", "--unknown"].map(OsString::from), project_dirs,).is_err()
    );
}

#[test]
fn automatic_raw_parallelism_respects_cpu_memory_and_job_limits() {
    assert_eq!(recommended_raw_parallelism(16, 16 * 1024_u64.pow(3), 8), 8);
    assert_eq!(recommended_raw_parallelism(16, 2 * 1024_u64.pow(3), 8), 1);
    assert_eq!(recommended_raw_parallelism(4, 16 * 1024_u64.pow(3), 8), 2);
    assert_eq!(recommended_raw_parallelism(32, 16 * 1024_u64.pow(3), 3), 3);
}
