use super::*;
use discovery::DirectoryScan;
use futures::executor::block_on;
use std::path::Path;
use std::time::Instant;
use test_r::test;

fn gpui_test_config(root: &Path, cache: &Path) -> Config {
    Config {
        settings: AppSettings {
            root: root.to_path_buf(),
            image_cache_size: 2,
            pair_preference: config::PairPreference::Raw,
            cache_root: cache.to_path_buf(),
            job_parallelism: 1,
            raw_parallelism: 1,
            online_map_permission: config::OnlineMapPermission::Disabled,
            map_cache_bytes: 1024 * 1024,
            map_switcher_placement: config::MapSwitcherPlacement::Header,
        },
        settings_path: root.join("settings.json"),
    }
}

fn write_photo_stub(root: &Path, name: &str, latitude: f64, longitude: f64) -> PathBuf {
    let path = root.join(format!("{name}.jpg"));
    image::RgbImage::from_pixel(8, 8, image::Rgb([32, 96, 160]))
        .save(&path)
        .unwrap();
    std::fs::write(
        root.join(format!("{name}.xmp")),
        format!(
            "<rdf:Description exif:GPSLatitude=\"{latitude}\" exif:GPSLongitude=\"{longitude}\"/>"
        ),
    )
    .unwrap();
    path
}

fn double_click(cx: &mut gpui::VisualTestContext, position: gpui::Point<gpui::Pixels>) {
    let modifiers = gpui::Modifiers::none();
    cx.simulate_event(MouseDownEvent {
        position,
        modifiers,
        button: MouseButton::Left,
        click_count: 2,
        first_mouse: false,
    });
    cx.simulate_event(MouseUpEvent {
        position,
        modifiers,
        button: MouseButton::Left,
        click_count: 2,
    });
}

#[gpui::test]
fn gpui_native_test_context_can_paint_and_click(cx: &mut gpui::TestAppContext) {
    #[derive(Default)]
    struct ClickHarness {
        clicked: bool,
    }

    impl gpui::Render for ClickHarness {
        fn render(
            &mut self,
            _: &mut gpui::Window,
            cx: &mut gpui::Context<Self>,
        ) -> impl gpui::IntoElement {
            div().size_full().child(
                div()
                    .size(px(80.0))
                    .debug_selector(|| "#gpui-test-button".to_owned())
                    .on_mouse_up(
                        MouseButton::Left,
                        cx.listener(|this, _, _, cx| {
                            this.clicked = true;
                            cx.notify();
                        }),
                    ),
            )
        }
    }

    let (view, cx) = cx.add_window_view(|_, _| ClickHarness::default());
    cx.update(|window, _| window.refresh());
    cx.run_until_parked();
    let bounds = cx.debug_bounds("#gpui-test-button").unwrap();
    cx.simulate_click(bounds.center(), gpui::Modifiers::none());
    assert!(view.read_with(cx, |view, _| view.clicked));
}

#[gpui::test]
fn gpui_thumbnail_grid_selects_extends_toggles_opens_and_closes(cx: &mut gpui::TestAppContext) {
    let root = tempfile::tempdir().unwrap();
    let cache = tempfile::tempdir().unwrap();
    for index in 0..4 {
        write_photo_stub(root.path(), &format!("photo-{index}"), 47.5, 19.05);
    }
    let config = gpui_test_config(root.path(), cache.path());
    let (view, cx) = cx.add_window_view(|_, cx| PictureBrowser::new(&config, cx));
    cx.run_until_parked();

    let first = cx.debug_bounds("#thumbnail-0").unwrap();
    cx.simulate_click(first.center(), gpui::Modifiers::none());
    let second = cx.debug_bounds("#thumbnail-1").unwrap();
    cx.simulate_click(
        second.center(),
        gpui::Modifiers {
            control: true,
            ..Default::default()
        },
    );
    assert_eq!(
        view.read_with(&*cx, |view, _| match &view.main_view {
            MainView::Dir(_, selection) => selection.indices.clone(),
            MainView::Image(_) => unreachable!(),
        }),
        BTreeSet::from([0, 1])
    );

    let fourth = cx.debug_bounds("#thumbnail-3").unwrap();
    cx.simulate_click(
        fourth.center(),
        gpui::Modifiers {
            shift: true,
            ..Default::default()
        },
    );
    assert_eq!(
        view.read_with(&*cx, |view, _| match &view.main_view {
            MainView::Dir(_, selection) => selection.indices.clone(),
            MainView::Image(_) => unreachable!(),
        }),
        BTreeSet::from([1, 2, 3])
    );

    double_click(cx, fourth.center());
    assert!(view.read_with(&*cx, |view, _| matches!(
        &view.main_view,
        MainView::Image(state) if state.path.ends_with("photo-3.jpg")
    )));
    cx.dispatch_action(CloseImage);
    assert!(view.read_with(&*cx, |view, _| matches!(
        view.main_view,
        MainView::Dir(_, _)
    )));
}

#[gpui::test]
fn gpui_grid_keyboard_actions_move_extend_and_open_selection(cx: &mut gpui::TestAppContext) {
    cx.update(bind_picture_browser_keys);
    let root = tempfile::tempdir().unwrap();
    let cache = tempfile::tempdir().unwrap();
    for index in 0..4 {
        write_photo_stub(root.path(), &format!("photo-{index}"), 47.5, 19.05);
    }
    let config = gpui_test_config(root.path(), cache.path());
    let (view, cx) = cx.add_window_view(|_, cx| PictureBrowser::new(&config, cx));
    cx.run_until_parked();
    let first = cx.debug_bounds("#thumbnail-0").unwrap();
    cx.simulate_click(first.center(), gpui::Modifiers::none());

    cx.simulate_keystrokes("right shift-right");
    assert_eq!(
        view.read_with(&*cx, |view, _| match &view.main_view {
            MainView::Dir(_, selection) => selection.indices.clone(),
            MainView::Image(_) => unreachable!(),
        }),
        BTreeSet::from([1, 2])
    );
    cx.simulate_keystrokes("enter");
    assert!(view.read_with(&*cx, |view, _| matches!(view.main_view, MainView::Image(_))));
    cx.simulate_keystrokes("escape");
    assert!(view.read_with(&*cx, |view, _| matches!(
        view.main_view,
        MainView::Dir(_, _)
    )));
}

#[gpui::test]
fn gpui_image_view_double_clicks_zooms_and_drag_pans(cx: &mut gpui::TestAppContext) {
    let root = tempfile::tempdir().unwrap();
    let cache = tempfile::tempdir().unwrap();
    write_photo_stub(root.path(), "photo", 47.5, 19.05);
    let config = gpui_test_config(root.path(), cache.path());
    let (view, cx) = cx.add_window_view(|_, cx| PictureBrowser::new(&config, cx));
    cx.run_until_parked();
    let thumbnail = cx.debug_bounds("#thumbnail-0").unwrap();
    double_click(cx, thumbnail.center());
    cx.run_until_parked();
    assert!(view.read_with(&*cx, |view, _| {
        let MainView::Image(state) = &view.main_view else {
            return false;
        };
        matches!(
            view.previews
                .state_for(&state.path, view.render_identity_for(&state.path)),
            PreviewState::Loaded(_)
        )
    }));

    let viewport = cx.debug_bounds("#image-viewport").unwrap();
    double_click(cx, viewport.center());
    assert!(view.read_with(&*cx, |view, _| matches!(
        &view.main_view,
        MainView::Image(state) if state.zoom == ZoomLevel::ActualSize
    )));
    let start = viewport.center();
    let end = point(start.x + px(20.0), start.y + px(15.0));
    cx.simulate_mouse_down(start, MouseButton::Left, gpui::Modifiers::none());
    cx.simulate_mouse_move(end, MouseButton::Left, gpui::Modifiers::none());
    assert!(view.read_with(&*cx, |view, _| matches!(
        &view.main_view,
        MainView::Image(state) if state.drag_last.is_some()
    )));
    cx.simulate_mouse_up(end, MouseButton::Left, gpui::Modifiers::none());
    assert!(view.read_with(&*cx, |view, _| matches!(
        &view.main_view,
        MainView::Image(state) if state.drag_last.is_none()
    )));
    double_click(cx, viewport.center());
    assert!(view.read_with(&*cx, |view, _| matches!(
        &view.main_view,
        MainView::Image(state) if state.zoom == ZoomLevel::FitWindow && state.pan == (0.0, 0.0)
    )));
}

#[gpui::test]
fn gpui_grid_recursive_controls_enter_and_exit_recursive_mode(cx: &mut gpui::TestAppContext) {
    let root = tempfile::tempdir().unwrap();
    let cache = tempfile::tempdir().unwrap();
    std::fs::create_dir(root.path().join("trip")).unwrap();
    write_photo_stub(&root.path().join("trip"), "photo", 47.5, 19.05);
    let config = gpui_test_config(root.path(), cache.path());
    let (view, cx) = cx.add_window_view(|_, cx| PictureBrowser::new(&config, cx));
    cx.run_until_parked();

    let enter = cx.debug_bounds("#show-images-recursively").unwrap();
    cx.simulate_click(enter.center(), gpui::Modifiers::none());
    cx.run_until_parked();
    assert!(view.read_with(&*cx, |view, _| view.recursive_grid.is_some()));
    assert!(view.read_with(&*cx, |view, _| matches!(
        view.recursive_grid.as_ref().map(|state| &state.status),
        Some(RecursiveGridStatus::Loaded(images)) if images.paths.len() == 1
    )));
    assert!(cx.debug_bounds("#thumbnail-0").is_some());
    let exit = cx.debug_bounds("#exit-recursive-grid").unwrap();
    cx.simulate_click(exit.center(), gpui::Modifiers::none());
    cx.run_until_parked();
    assert!(view.read_with(&*cx, |view, _| view.recursive_grid.is_none()));
    assert!(cx.debug_bounds("#show-images-recursively").is_some());
}

#[gpui::test]
fn gpui_settings_dialog_cancels_and_saves_painted_control_changes(cx: &mut gpui::TestAppContext) {
    let root = tempfile::tempdir().unwrap();
    let cache = tempfile::tempdir().unwrap();
    let config = gpui_test_config(root.path(), cache.path());
    let (view, cx) = cx.add_window_view(|_, cx| PictureBrowser::new(&config, cx));
    cx.simulate_resize(size(px(1200.0), px(900.0)));
    cx.run_until_parked();
    let root_row = cx.debug_bounds("#directory-0").unwrap();
    cx.simulate_click(root_row.center(), gpui::Modifiers::none());
    cx.dispatch_action(OpenSettings);
    cx.run_until_parked();

    let pair = cx.debug_bounds("#settings-pair-preference").unwrap();
    cx.simulate_click(pair.center(), gpui::Modifiers::none());
    let cancel = cx.debug_bounds("#settings-cancel").unwrap();
    cx.simulate_click(cancel.center(), gpui::Modifiers::none());
    assert_eq!(
        view.read_with(&*cx, |view, _| view.settings.pair_preference),
        config::PairPreference::Raw
    );
    assert!(view.read_with(&*cx, |view, _| view.settings_draft.is_none()));

    cx.dispatch_action(OpenSettings);
    cx.run_until_parked();
    let pair = cx.debug_bounds("#settings-pair-preference").unwrap();
    cx.simulate_click(pair.center(), gpui::Modifiers::none());
    let online = cx.debug_bounds("#settings-online-map").unwrap();
    cx.simulate_click(online.center(), gpui::Modifiers::none());
    let save = cx.debug_bounds("#settings-save").unwrap();
    cx.simulate_click(save.center(), gpui::Modifiers::none());
    assert_eq!(
        view.read_with(&*cx, |view, _| view.settings.pair_preference),
        config::PairPreference::Jpeg
    );
    assert_eq!(
        view.read_with(&*cx, |view, _| view.settings.online_map_permission),
        config::OnlineMapPermission::Enabled
    );
    assert!(view.read_with(&*cx, |view, _| view.settings_draft.is_none()));
    let saved: AppSettings =
        serde_json::from_slice(&std::fs::read(&config.settings_path).unwrap()).unwrap();
    assert_eq!(saved.pair_preference, config::PairPreference::Jpeg);
    assert_eq!(
        saved.online_map_permission,
        config::OnlineMapPermission::Enabled
    );
}

#[gpui::test]
fn gpui_directory_tree_click_and_keyboard_navigation_select_and_expand(
    cx: &mut gpui::TestAppContext,
) {
    cx.update(bind_picture_browser_keys);
    let root = tempfile::tempdir().unwrap();
    let cache = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(root.path().join("alpha/nested")).unwrap();
    std::fs::create_dir(root.path().join("beta")).unwrap();
    let config = gpui_test_config(root.path(), cache.path());
    let (view, cx) = cx.add_window_view(|_, cx| PictureBrowser::new(&config, cx));
    cx.run_until_parked();

    let alpha = cx.debug_bounds("#directory-1").unwrap();
    cx.simulate_click(alpha.center(), gpui::Modifiers::none());
    assert!(view.read_with(&*cx, |view, _| {
        view.model.node(view.selected).path.ends_with("alpha")
    }));

    cx.simulate_keystrokes("right");
    cx.run_until_parked();
    assert!(cx.debug_bounds("#directory-2").is_some());
    cx.simulate_keystrokes("down");
    assert!(view.read_with(&*cx, |view, _| {
        view.model.node(view.selected).path.ends_with("nested")
    }));
    cx.simulate_keystrokes("left");
    assert!(view.read_with(&*cx, |view, _| {
        view.model.node(view.selected).path.ends_with("alpha")
    }));
    cx.simulate_keystrokes("left");
    cx.run_until_parked();
    assert_eq!(view.read_with(&*cx, |view, _| view.visible.len()), 3);
}

#[gpui::test]
fn gpui_path_context_menu_copies_image_path_and_hides_directory(cx: &mut gpui::TestAppContext) {
    let root = tempfile::tempdir().unwrap();
    let cache = tempfile::tempdir().unwrap();
    let trip = root.path().join("trip");
    std::fs::create_dir(&trip).unwrap();
    let photo = write_photo_stub(root.path(), "photo", 47.5, 19.05);
    let config = gpui_test_config(root.path(), cache.path());
    let (view, cx) = cx.add_window_view(|_, cx| PictureBrowser::new(&config, cx));
    cx.run_until_parked();

    let thumbnail = cx.debug_bounds("#thumbnail-0").unwrap();
    cx.simulate_mouse_down(
        thumbnail.center(),
        MouseButton::Right,
        gpui::Modifiers::none(),
    );
    let copy = cx.debug_bounds("#copy-absolute-path-menu-item").unwrap();
    cx.simulate_mouse_down(copy.center(), MouseButton::Left, gpui::Modifiers::none());
    assert_eq!(
        cx.read_from_clipboard().and_then(|item| item.text()),
        Some(photo.to_string_lossy().into_owned())
    );
    assert!(view.read_with(&*cx, |view, _| view.path_context_menu.is_none()));

    let directory = cx.debug_bounds("#directory-1").unwrap();
    cx.simulate_mouse_down(
        directory.center(),
        MouseButton::Right,
        gpui::Modifiers::none(),
    );
    let hide = cx.debug_bounds("#toggle-hidden-subtree-menu-item").unwrap();
    cx.simulate_mouse_down(hide.center(), MouseButton::Left, gpui::Modifiers::none());
    assert!(view.read_with(&*cx, |view, _| view.hidden_subtrees.contains(&trip)));
}

#[gpui::test]
fn gpui_linux_file_menu_opens_cache_dialog_and_close_button_closes_it(
    cx: &mut gpui::TestAppContext,
) {
    if !USE_IN_WINDOW_MENU {
        return;
    }
    let root = tempfile::tempdir().unwrap();
    let cache = tempfile::tempdir().unwrap();
    let config = gpui_test_config(root.path(), cache.path());
    let (view, cx) = cx.add_window_view(|_, cx| PictureBrowser::new(&config, cx));
    cx.run_until_parked();
    let file = cx.debug_bounds("#file-menu-button").unwrap();
    cx.simulate_click(file.center(), gpui::Modifiers::none());
    let cache_item = cx.debug_bounds("#cache-statistics-menu-item").unwrap();
    cx.simulate_click(cache_item.center(), gpui::Modifiers::none());
    assert!(view.read_with(&*cx, |view, _| view.show_cache_statistics));
    let close = cx.debug_bounds("#cache-statistics-close").unwrap();
    cx.simulate_click(close.center(), gpui::Modifiers::none());
    assert!(!view.read_with(&*cx, |view, _| view.show_cache_statistics));
}

#[gpui::test]
fn gpui_cache_dialog_clear_button_removes_downloaded_map_cache(cx: &mut gpui::TestAppContext) {
    let root = tempfile::tempdir().unwrap();
    let cache = tempfile::tempdir().unwrap();
    std::fs::create_dir(cache.path().join("map")).unwrap();
    let map_cache = cache.path().join("map/ambient.sqlite3");
    std::fs::write(&map_cache, b"cached tiles").unwrap();
    let config = gpui_test_config(root.path(), cache.path());
    let (view, cx) = cx.add_window_view(|_, cx| PictureBrowser::new(&config, cx));
    cx.update_entity(&view, |view, cx| {
        view.show_cache_statistics = true;
        cx.notify();
    });
    cx.run_until_parked();
    let clear = cx.debug_bounds("#clear-downloaded-map-tiles").unwrap();
    cx.simulate_click(clear.center(), gpui::Modifiers::none());
    cx.run_until_parked();
    assert!(!map_cache.exists());
    assert!(!view.read_with(&*cx, |view, _| view.map_cache_clearing));
}

#[gpui::test]
fn gpui_cluster_gallery_paginates_and_zoom_action_updates_the_view(cx: &mut gpui::TestAppContext) {
    let root = tempfile::tempdir().unwrap();
    let cache = tempfile::tempdir().unwrap();
    let config = gpui_test_config(root.path(), cache.path());
    let (view, cx) = cx.add_window_view(|_, cx| PictureBrowser::new(&config, cx));
    let coordinate = metadata::GeoCoordinate::new(47.5, 19.05).unwrap();
    cx.update_entity(&view, |view, cx| {
        view.directory_view_mode = DirectoryViewMode::Map;
        view.map_render_size = (1280, 800);
        view.map_camera.zoom = 3.0;
        view.map_card = MapCard::new(
            coordinate,
            (0..6)
                .map(|index| MapCardItem {
                    path: root.path().join(format!("{index}.jpg")),
                })
                .collect(),
            7,
            None,
            Some(8),
        );
        cx.notify();
    });
    cx.run_until_parked();

    let next = cx.debug_bounds("#map-card-next").unwrap();
    cx.simulate_click(next.center(), gpui::Modifiers::none());
    assert_eq!(
        view.read_with(cx, |view, _| view.map_card.as_ref().unwrap().page),
        1
    );

    let zoom = cx.debug_bounds("#map-card-zoom").unwrap();
    cx.simulate_click(zoom.center(), gpui::Modifiers::none());
    assert!(view.read_with(cx, |view, _| view.map_card.is_none()));
    assert!(view.read_with(cx, |view, _| view.map_camera.zoom > 3.0));
}

#[gpui::test]
fn gpui_map_cache_bar_and_map_job_use_the_regular_ui(cx: &mut gpui::TestAppContext) {
    let root = tempfile::tempdir().unwrap();
    let cache = tempfile::tempdir().unwrap();
    std::fs::create_dir(cache.path().join("map")).unwrap();
    std::fs::write(cache.path().join("map/ambient.sqlite3"), vec![0_u8; 512]).unwrap();
    let mut config = gpui_test_config(root.path(), cache.path());
    config.settings.map_cache_bytes = 1024;
    let (view, cx) = cx.add_window_view(|_, cx| PictureBrowser::new(&config, cx));
    cx.update_entity(&view, |view, cx| {
        view.directory_view_mode = DirectoryViewMode::Map;
        view.map_job = Some(MapJob {
            epoch: 42,
            label: "Rendering map and downloading visible tiles…".to_owned(),
            status: MapJobStatus::Running,
        });
        view.show_job_queue = true;
        cx.notify();
    });
    cx.run_until_parked();
    assert!(cx.debug_bounds("#map-job-queue-row").is_some());
    assert!(cx.debug_bounds("#background-progress").is_some());
    let grid = cx.debug_bounds("#show-grid-view").unwrap();
    cx.simulate_click(grid.center(), gpui::Modifiers::none());
    assert!(view.read_with(cx, |view, _| view.map_job.is_none()));

    cx.update_entity(&view, |view, cx| {
        view.show_cache_statistics = true;
        cx.notify();
    });
    cx.run_until_parked();
    let bar = cx.debug_bounds("#map-cache-capacity-bar").unwrap();
    let fill = cx.debug_bounds("#map-cache-capacity-fill").unwrap();
    assert!(f32::from(bar.size.width) > 100.0);
    assert!(f32::from(bar.size.height) > 0.0);
    assert!((f32::from(fill.size.width) / f32::from(bar.size.width) - 0.5).abs() < 0.01);
}

#[gpui::test]
fn gpui_empty_map_offers_and_starts_recursive_mode(cx: &mut gpui::TestAppContext) {
    let root = tempfile::tempdir().unwrap();
    std::fs::create_dir(root.path().join("trip")).unwrap();
    let cache = tempfile::tempdir().unwrap();
    let config = gpui_test_config(root.path(), cache.path());
    let (view, cx) = cx.add_window_view(|_, cx| PictureBrowser::new(&config, cx));
    cx.run_until_parked();
    cx.update_entity(&view, |view, cx| {
        view.directory_view_mode = DirectoryViewMode::Map;
        cx.notify();
    });
    cx.run_until_parked();
    let button = cx.debug_bounds("#map-show-images-recursively").unwrap();
    cx.simulate_click(button.center(), gpui::Modifiers::none());
    assert!(view.read_with(cx, |view, _| view.recursive_grid.is_some()));
}

#[gpui::test]
fn gpui_outer_zoom_cluster_click_opens_all_cluster_photos(cx: &mut gpui::TestAppContext) {
    let root = tempfile::tempdir().unwrap();
    for (name, latitude, longitude) in
        [("one", "47.500N", "19.050E"), ("two", "47.501N", "19.051E")]
    {
        std::fs::write(root.path().join(format!("{name}.jpg")), b"").unwrap();
        std::fs::write(
            root.path().join(format!("{name}.xmp")),
            format!(
                "<rdf:Description exif:GPSLatitude=\"{latitude}\" exif:GPSLongitude=\"{longitude}\"/>"
            ),
        )
        .unwrap();
    }
    let cache = tempfile::tempdir().unwrap();
    let config = gpui_test_config(root.path(), cache.path());
    let (view, cx) = cx.add_window_view(|_, cx| PictureBrowser::new(&config, cx));
    cx.run_until_parked();
    cx.update_entity(&view, |view, cx| {
        view.directory_view_mode = DirectoryViewMode::Map;
        view.map_camera = map::Camera {
            center: metadata::GeoCoordinate::new(47.5, 19.05).unwrap(),
            zoom: 2.0,
        };
        view.map_pending_fit = None;
        view.map_fit_node = None;
        cx.notify();
    });
    cx.run_until_parked();

    let marker = cx.debug_bounds("#map-marker-0").unwrap();
    cx.simulate_click(marker.center(), gpui::Modifiers::none());
    assert_eq!(
        view.read_with(cx, |view, _| view.map_card.as_ref().unwrap().items.len()),
        2
    );
    assert!(cx.debug_bounds("#map-card-zoom").is_some());
    let thumbnail = cx.debug_bounds("#map-gallery-item-0").unwrap();
    cx.simulate_click(thumbnail.center(), gpui::Modifiers::none());
    assert!(view.read_with(cx, |view, _| matches!(view.main_view, MainView::Image(_))));
}

#[gpui::test]
fn gpui_map_surface_drag_and_wheel_update_camera_during_real_events(cx: &mut gpui::TestAppContext) {
    let root = tempfile::tempdir().unwrap();
    let cache = tempfile::tempdir().unwrap();
    write_photo_stub(root.path(), "photo", 47.5, 19.05);
    let config = gpui_test_config(root.path(), cache.path());
    let (view, cx) = cx.add_window_view(|_, cx| PictureBrowser::new(&config, cx));
    cx.run_until_parked();
    cx.update_entity(&view, |view, cx| {
        view.directory_view_mode = DirectoryViewMode::Map;
        view.map_camera = map::Camera {
            center: metadata::GeoCoordinate::new(47.5, 19.05).unwrap(),
            zoom: 5.0,
        };
        view.map_pending_fit = None;
        view.map_fit_node = None;
        cx.notify();
    });
    cx.run_until_parked();
    let surface = cx.debug_bounds("#map-view").unwrap();
    let start = point(surface.origin.x + px(40.0), surface.origin.y + px(40.0));
    let end = point(start.x + px(80.0), start.y + px(40.0));
    let before = view.read_with(&*cx, |view, _| view.map_camera);
    cx.simulate_mouse_down(start, MouseButton::Left, gpui::Modifiers::none());
    cx.simulate_mouse_move(end, MouseButton::Left, gpui::Modifiers::none());
    let during = view.read_with(&*cx, |view, _| view.map_camera);
    assert_ne!(during.center, before.center);
    cx.simulate_mouse_up(end, MouseButton::Left, gpui::Modifiers::none());
    assert!(view.read_with(&*cx, |view, _| view.map_drag_last.is_none()));

    cx.simulate_event(ScrollWheelEvent {
        position: end,
        delta: gpui::ScrollDelta::Lines(point(0.0, -3.0)),
        modifiers: gpui::Modifiers::none(),
        touch_phase: gpui::TouchPhase::Moved,
    });
    assert!(view.read_with(&*cx, |view, _| view.map_camera.zoom > during.zoom));
}

#[gpui::test]
fn gpui_map_switcher_consent_and_zoom_controls_drive_the_painted_ui(cx: &mut gpui::TestAppContext) {
    let root = tempfile::tempdir().unwrap();
    let cache = tempfile::tempdir().unwrap();
    write_photo_stub(root.path(), "photo", 47.5, 19.05);
    let mut config = gpui_test_config(root.path(), cache.path());
    config.settings.online_map_permission = config::OnlineMapPermission::Unknown;
    let (view, cx) = cx.add_window_view(|_, cx| PictureBrowser::new(&config, cx));
    cx.run_until_parked();

    let show_map = cx.debug_bounds("#show-map-view").unwrap();
    cx.simulate_click(show_map.center(), gpui::Modifiers::none());
    cx.run_until_parked();
    assert_eq!(
        view.read_with(&*cx, |view, _| view.directory_view_mode),
        DirectoryViewMode::Map
    );
    let offline = cx.debug_bounds("#keep-map-offline").unwrap();
    cx.simulate_click(offline.center(), gpui::Modifiers::none());
    assert_eq!(
        view.read_with(&*cx, |view, _| view.settings.online_map_permission),
        config::OnlineMapPermission::Disabled
    );

    let before = view.read_with(&*cx, |view, _| view.map_camera.zoom);
    let zoom_in = cx.debug_bounds("#map-zoom-in").unwrap();
    cx.simulate_mouse_down(zoom_in.center(), MouseButton::Left, gpui::Modifiers::none());
    assert!(view.read_with(&*cx, |view, _| view.map_camera.zoom > before));
    let zoomed = view.read_with(&*cx, |view, _| view.map_camera.zoom);
    let fit = cx.debug_bounds("#map-fit").unwrap();
    cx.simulate_mouse_down(fit.center(), MouseButton::Left, gpui::Modifiers::none());
    assert_ne!(view.read_with(&*cx, |view, _| view.map_camera.zoom), zoomed);

    let grid = cx.debug_bounds("#show-grid-view").unwrap();
    cx.simulate_click(grid.center(), gpui::Modifiers::none());
    assert_eq!(
        view.read_with(&*cx, |view, _| view.directory_view_mode),
        DirectoryViewMode::Grid
    );
}

#[gpui::test]
fn gpui_real_large_cluster_fetches_and_paints_the_second_gallery_page(
    cx: &mut gpui::TestAppContext,
) {
    let root = tempfile::tempdir().unwrap();
    let cache = tempfile::tempdir().unwrap();
    for index in 0..7 {
        write_photo_stub(root.path(), &format!("photo-{index}"), 47.5, 19.05);
    }
    let config = gpui_test_config(root.path(), cache.path());
    let (view, cx) = cx.add_window_view(|_, cx| PictureBrowser::new(&config, cx));
    cx.run_until_parked();
    cx.update_entity(&view, |view, cx| {
        view.directory_view_mode = DirectoryViewMode::Map;
        view.map_camera.zoom = 2.0;
        view.map_pending_fit = None;
        view.map_fit_node = None;
        cx.notify();
    });
    cx.run_until_parked();
    let marker = cx.debug_bounds("#map-marker-0").unwrap();
    cx.simulate_click(marker.center(), gpui::Modifiers::none());
    let first_page = view.read_with(&*cx, |view, _| {
        view.map_card
            .as_ref()
            .unwrap()
            .page_items()
            .iter()
            .map(|item| item.path.clone())
            .collect::<Vec<_>>()
    });
    assert_eq!(first_page.len(), 6);
    let next = cx.debug_bounds("#map-card-next").unwrap();
    cx.simulate_click(next.center(), gpui::Modifiers::none());
    cx.run_until_parked();
    let second_page = view.read_with(&*cx, |view, _| {
        view.map_card
            .as_ref()
            .unwrap()
            .page_items()
            .iter()
            .map(|item| item.path.clone())
            .collect::<Vec<_>>()
    });
    assert_eq!(second_page.len(), 1);
    assert!(!first_page.contains(&second_page[0]));
    assert!(cx.debug_bounds("#map-gallery-item-0").is_some());
}

#[test]
fn converts_gpui_bgra_pixels_to_png_rgba_order() {
    let mut pixels = [3, 2, 1, 4, 30, 20, 10, 40];

    bgra_to_rgba(&mut pixels);

    assert_eq!(pixels, [1, 2, 3, 4, 10, 20, 30, 40]);
}

#[test]
fn export_dialog_remembers_the_last_chosen_directory() {
    let source = std::path::Path::new("/pictures/trip/image.nef");

    assert_eq!(
        export_dialog_directory(None, source),
        std::path::Path::new("/pictures/trip")
    );
    assert_eq!(
        export_dialog_directory(Some(std::path::Path::new("/exports/comparison")), source),
        std::path::Path::new("/exports/comparison")
    );
}

#[test]
fn full_screen_wheel_navigation_stops_at_directory_edges() {
    assert_eq!(adjacent_image_index(1, 3, -1), Some(0));
    assert_eq!(adjacent_image_index(1, 3, 1), Some(2));
    assert_eq!(adjacent_image_index(0, 3, -1), None);
    assert_eq!(adjacent_image_index(2, 3, 1), None);
    assert_eq!(adjacent_image_index(0, 0, 1), None);
}

#[test]
fn map_marker_card_requires_photos_and_wraps_cluster_navigation() {
    let coordinate = metadata::GeoCoordinate::new(47.5, 19.05).unwrap();
    assert!(MapCard::new(coordinate, Vec::new(), 0, None, None).is_none());
    let items = (0..6)
        .map(|index| MapCardItem {
            path: PathBuf::from(format!("{index}.jpg")),
        })
        .collect::<Vec<_>>();
    let mut card = MapCard::new(coordinate, items, 7, None, Some(8)).unwrap();
    assert_eq!(card.page_count(), 2);
    assert_eq!(card.page_items().len(), 6);
    assert_eq!(card.page, 0);
    card.navigate(-1);
    assert_eq!(card.page, 1);
    card.navigate(1);
    assert_eq!(card.page, 0);
}

#[test]
fn map_surface_geometry_is_identity_for_the_rendered_camera() {
    let camera = map::Camera {
        center: metadata::GeoCoordinate::new(47.5, 19.0).unwrap(),
        zoom: 8.25,
    };
    assert_eq!(
        map_surface_geometry(camera, camera, (1200, 800), (1200, 800)),
        MapSurfaceGeometry {
            left: 0.0,
            top: 0.0,
            width: 1200.0,
            height: 800.0,
        }
    );
}

#[test]
fn transformed_raster_and_live_markers_share_the_same_camera() {
    let rendered = map::Camera {
        center: metadata::GeoCoordinate::new(47.4979, 19.0402).unwrap(),
        zoom: 7.0,
    };
    let current = map::Camera {
        center: metadata::GeoCoordinate::new(47.55, 19.12).unwrap(),
        zoom: 8.35,
    };
    let viewport = (1280, 720);
    let geometry = map_surface_geometry(rendered, current, viewport, viewport);
    for coordinate in [
        metadata::GeoCoordinate::new(47.4979, 19.0402).unwrap(),
        metadata::GeoCoordinate::new(47.62, 18.91).unwrap(),
    ] {
        let rendered_position = map_screen_position(coordinate, rendered, viewport).unwrap();
        let transformed_x = geometry.left + rendered_position.0 * geometry.width;
        let transformed_y = geometry.top + rendered_position.1 * geometry.height;
        let current_position = map_screen_position(coordinate, current, viewport).unwrap();
        assert!((transformed_x - current_position.0 * viewport.0 as f32).abs() < 0.01);
        assert!((transformed_y - current_position.1 * viewport.1 as f32).abs() < 0.01);
    }
}

#[test]
fn oversized_map_raster_covers_short_pans_without_blank_edges() {
    let rendered = map::Camera::default();
    let mut current = rendered;
    let viewport = (1000, 700);
    let raster = (
        viewport.0 + MAP_RENDER_OVERSCAN * 2,
        viewport.1 + MAP_RENDER_OVERSCAN * 2,
    );
    let (center_x, center_y) = map::mercator_project(current.center);
    let world = 256.0 * 2_f64.powf(current.zoom);
    current.center =
        map::mercator_unproject(center_x + 60.0 / world, center_y - 40.0 / world).unwrap();
    let geometry = map_surface_geometry(rendered, current, viewport, raster);
    assert!(geometry.left <= 0.0);
    assert!(geometry.top <= 0.0);
    assert!(geometry.left + geometry.width >= viewport.0 as f32);
    assert!(geometry.top + geometry.height >= viewport.1 as f32);
}

#[test]
fn map_surface_transform_remains_aligned_across_the_antimeridian() {
    let rendered = map::Camera {
        center: metadata::GeoCoordinate::new(70.0, 179.9).unwrap(),
        zoom: 6.5,
    };
    let current = map::Camera {
        center: metadata::GeoCoordinate::new(70.1, -179.9).unwrap(),
        zoom: 7.25,
    };
    let viewport = (900, 600);
    let geometry = map_surface_geometry(rendered, current, viewport, viewport);
    let coordinate = metadata::GeoCoordinate::new(70.05, 179.98).unwrap();
    let old = map_screen_position(coordinate, rendered, viewport).unwrap();
    let live = map_screen_position(coordinate, current, viewport).unwrap();
    assert!((geometry.left + old.0 * geometry.width - live.0 * viewport.0 as f32).abs() < 0.01);
    assert!((geometry.top + old.1 * geometry.height - live.1 * viewport.1 as f32).abs() < 0.01);
}

#[test]
fn cursor_anchored_map_zoom_preserves_the_coordinate_under_the_pointer() {
    let viewport = (1200, 800);
    let anchor = (930.0, 210.0);
    let camera = map::Camera {
        center: metadata::GeoCoordinate::new(47.5, 19.05).unwrap(),
        zoom: 7.0,
    };
    let (center_x, center_y) = map::mercator_project(camera.center);
    let world = 256.0 * 2_f64.powf(camera.zoom);
    let anchored_coordinate = map::mercator_unproject(
        center_x + (anchor.0 - f64::from(viewport.0) / 2.0) / world,
        center_y + (anchor.1 - f64::from(viewport.1) / 2.0) / world,
    )
    .unwrap();
    let zoomed = zoom_map_camera(camera, 1.0, anchor, viewport);
    let position = map_screen_position(anchored_coordinate, zoomed, viewport).unwrap();
    assert!((position.0 * viewport.0 as f32 - anchor.0 as f32).abs() < 0.01);
    assert!((position.1 * viewport.1 as f32 - anchor.1 as f32).abs() < 0.01);
}

#[test]
fn repeated_map_zoom_steps_have_exact_bounded_final_zoom() {
    let mut camera = map::Camera::default();
    for _ in 0..50 {
        camera = zoom_map_camera(camera, 1.0, (400.0, 300.0), (800, 600));
    }
    assert_eq!(camera.zoom, 20.0);
    for _ in 0..50 {
        camera = zoom_map_camera(camera, -1.0, (400.0, 300.0), (800, 600));
    }
    assert_eq!(camera.zoom, 0.0);
}

#[test]
fn map_detail_status_distinguishes_loading_from_completed_empty_detail() {
    assert_eq!(
        map_detail_status(8.0, config::OnlineMapPermission::Enabled, true, None),
        "Bundled map · loading online detail"
    );
    assert_eq!(
        map_detail_status(
            8.0,
            config::OnlineMapPermission::Enabled,
            false,
            Some((0, None)),
        ),
        "Bundled map · no supported detail in visible tiles"
    );
    assert_eq!(
        map_detail_status(
            8.0,
            config::OnlineMapPermission::Enabled,
            false,
            Some((3, None)),
        ),
        "Detailed map · 3 vector tiles"
    );
}

#[test]
fn map_cache_capacity_progress_is_bounded_and_handles_zero_capacity() {
    assert_eq!(cache_capacity_fraction(0, 100), 0.0);
    assert_eq!(cache_capacity_fraction(25, 100), 0.25);
    assert_eq!(cache_capacity_fraction(100, 100), 1.0);
    assert_eq!(cache_capacity_fraction(150, 100), 1.0);
    assert_eq!(cache_capacity_fraction(1, 0), 0.0);
}

#[test]
fn full_screen_prefetch_follows_the_navigation_direction() {
    assert_eq!(directional_adjacent_indices(2, 8, 1, 4), vec![3, 4, 5, 6]);
    assert_eq!(directional_adjacent_indices(4, 8, -1, 4), vec![3, 2, 1, 0]);
    assert_eq!(directional_adjacent_indices(6, 8, 1, 4), vec![7]);
    assert!(directional_adjacent_indices(0, 8, -1, 4).is_empty());
}

#[test]
fn progress_describes_the_running_render() {
    let path = PathBuf::from("/pictures/trip/image.nef");
    let snapshot = SchedulerSnapshot {
        jobs: vec![scheduler::JobSnapshot {
            key: JobKey {
                path,
                variant: RenderVariant::Preview {
                    max_width: 1_920,
                    max_height: 1_080,
                },
                render_identity: RenderIdentity {
                    mode: RenderMode::Developed,
                    ..RenderIdentity::default()
                },
                recipe: None,
            },
            priority: JobPriority::INTERACTIVE,
            status: JobSnapshotStatus::Running,
            scope_count: 1,
            sequence: 0,
            cancelled: false,
        }],
        queued: 0,
        running: 1,
    };

    assert_eq!(
        scheduler_progress_label(&snapshot).as_deref(),
        Some("Rendering developed preview: image.nef…")
    );

    let mut background_snapshot = snapshot;
    background_snapshot.jobs[0].priority = JobPriority::speculative(0);
    assert_eq!(
        scheduler_progress_label(&background_snapshot).as_deref(),
        Some("Pre-rendering developed preview: image.nef…")
    );
}

#[test]
fn work_progress_keeps_specific_activity_labels() {
    let mut progress = WorkProgress::default();
    let scan = progress.start("Scanning trip…");
    let metadata = progress.start("Reading metadata for 4 images…");

    assert_eq!(
        progress.current_activity(),
        Some("Reading metadata for 4 images…")
    );
    progress.finish(metadata);
    assert_eq!(progress.current_activity(), Some("Scanning trip…"));
    progress.finish(scan);
    assert_eq!(progress.current_activity(), None);
}

#[test]
#[ignore = "manual investigation using external reference samples"]
fn export_fixed_reference_renders() {
    let root = PathBuf::from("/mnt/pictures/Pictures");
    let recipes = catalog::load_recipes(&root).unwrap();
    let samples = [
        root.join("2020/Szlovénia/2020-07-29/DSC_0546.NEF"),
        root.join("2026/Alpok túra/2026-07-28/IMG_5639.HEIC"),
        root.join("2026/Alpok túra/2026-08-03/IMG_0997.CR3"),
        root.join("2026/Alpok túra/2026-08-03/IMG_1056.CR3"),
    ];
    let cache = tempfile::tempdir().unwrap();
    let pipeline = PreviewRenderer::with_source_cache_capacity(
        cache.path().to_path_buf(),
        1,
        1,
        samples.len(),
    );
    let cancelled = Arc::new(AtomicBool::new(false));
    for path in samples {
        let entry = recipes.get(&path).unwrap();
        let rendered = pipeline.render_with_recipe(
            RenderInvocation {
                path: path.clone(),
                variant: RenderVariant::Full,
                render_identity: RenderIdentity {
                    mode: RenderMode::Developed,
                    recipe_digest: entry.digest,
                    renderer_model: catalog::RENDERER_MODEL_VERSION,
                },
                recipe: Some(&entry.recipe),
                interactive: true,
                durable_cache_write: false,
            },
            &cancelled,
        );
        let stem = path.file_stem().unwrap().to_string_lossy();
        save_preview_as_png(
            rendered,
            &PathBuf::from(format!("/tmp/rekesz-fixed-{stem}.png")),
        )
        .unwrap();
    }
}

#[test]
fn developed_marker_is_independent_of_the_rendering_toggle() {
    let developed_path = PathBuf::from("/pictures/developed.CR3");
    let baseline_path = PathBuf::from("/pictures/baseline.CR3");
    let recipe = catalog::DevelopRecipe {
        exposure_ev: Some(1.0),
        ..catalog::DevelopRecipe::default()
    };
    let recipes = HashMap::from([(
        developed_path.clone(),
        catalog::RecipeEntry {
            digest: recipe.digest().unwrap(),
            recipe,
        },
    )]);

    assert!(has_develop_recipe(Some(&recipes), &developed_path));
    assert!(!has_develop_recipe(Some(&recipes), &baseline_path));
    assert!(!has_develop_recipe(None, &developed_path));
}

#[test]
fn develop_inspector_formats_active_recipe_groups() {
    let recipe = catalog::DevelopRecipe {
        exposure_ev: Some(1.25),
        white_balance: Some(catalog::WhiteBalance {
            temperature: 6_500.0,
            tint: 4.0,
        }),
        crop: Some(catalog::Crop {
            left: 0.1,
            top: 0.2,
            right: 0.9,
            bottom: 0.8,
            angle_degrees: 1.5,
            constrain_aspect: true,
        }),
        heal_spots: vec![catalog::HealSpot {
            kind: catalog::HealKind::Clone,
            destination: [0.5, 0.5],
            source: [0.25, 0.25],
            radius: 0.1,
            opacity: 0.8,
            feather: 0.2,
        }],
        grain: Some(catalog::GrainRecipe {
            amount: 15.0,
            size: 30.0,
            roughness: 50.0,
            seed: 42,
        }),
        ..catalog::DevelopRecipe::default()
    };
    let edits = format_develop_edits(&recipe);
    let labels = edits
        .iter()
        .map(|(label, _)| label.as_str())
        .collect::<Vec<_>>();

    assert!(labels.contains(&"Exposure"));
    assert!(labels.contains(&"White balance"));
    assert!(labels.contains(&"Crop"));
    assert!(labels.contains(&"Clone spot 1"));
    assert!(labels.contains(&"Grain"));
    assert!(edits.iter().any(|(_, value)| value.contains("6500 K")));
}

#[test]
fn collapsing_a_branch_removes_all_descendant_rows() {
    let mut model = DirectoryModel::new(PathBuf::from("/pictures"));
    let root = model.root();
    let (request, _) = model.begin_load(root).unwrap();
    model.finish_load(
        root,
        request,
        Ok(DirectoryScan {
            directories: vec![PathBuf::from("/pictures/a")],
            images: Vec::new(),
        }),
    );
    let child = model.loaded(root).unwrap().children[0];
    let (request, _) = model.begin_load(child).unwrap();
    model.finish_load(
        child,
        request,
        Ok(DirectoryScan {
            directories: vec![PathBuf::from("/pictures/a/b")],
            images: Vec::new(),
        }),
    );

    let mut expanded = HashSet::from([root, child]);
    assert_eq!(
        visible_rows(&model, &expanded, &HashSet::new(), true).len(),
        3
    );
    expanded.remove(&child);
    assert_eq!(
        visible_rows(&model, &expanded, &HashSet::new(), true).len(),
        2
    );
    assert_eq!(model.loaded(child).unwrap().children.len(), 1);
}

#[test]
fn hidden_subtree_moves_to_the_virtual_bottom_section() {
    let mut model = DirectoryModel::new(PathBuf::from("/pictures"));
    let root = model.root();
    let (request, _) = model.begin_load(root).unwrap();
    model.finish_load(
        root,
        request,
        Ok(DirectoryScan {
            directories: vec![PathBuf::from("/pictures/a"), PathBuf::from("/pictures/z")],
            images: Vec::new(),
        }),
    );
    let hidden = HashSet::from([PathBuf::from("/pictures/a")]);
    let rows = visible_rows(&model, &HashSet::from([root]), &hidden, true);

    assert_eq!(rows.len(), 4);
    assert_eq!(
        model.node(rows[1].node).path.as_ref(),
        Path::new("/pictures/z")
    );
    assert_eq!(rows[2].node, HIDDEN_NODE);
    assert_eq!(
        model.node(rows[3].node).path.as_ref(),
        Path::new("/pictures/a")
    );
    assert_eq!(rows[3].depth, 1);
}

#[test]
fn recursive_grid_collects_globally_sorted_images_and_excludes_hidden_subtrees() {
    let root = tempfile::tempdir().unwrap();
    let visible = root.path().join("visible");
    let hidden = root.path().join("hidden");
    std::fs::create_dir_all(&visible).unwrap();
    std::fs::create_dir_all(&hidden).unwrap();
    std::fs::write(visible.join("z.jpg"), b"not a real jpeg").unwrap();
    std::fs::write(visible.join("a.jpg"), b"not a real jpeg").unwrap();
    std::fs::write(hidden.join("hidden.jpg"), b"not a real jpeg").unwrap();
    for name in ["a.jpg", "z.jpg"] {
        std::fs::File::open(visible.join(name))
            .unwrap()
            .set_times(std::fs::FileTimes::new().set_modified(std::time::SystemTime::UNIX_EPOCH))
            .unwrap();
    }
    let cache = tempfile::tempdir().unwrap();
    let limiter = JobLimiter::new(2);
    let scanner = DirectoryScanner::new(2, limiter.clone());
    let metadata_loader = MetadataLoader::new(cache.path().to_path_buf(), 2, limiter);

    let images = collect_recursive_images(
        &scanner,
        &metadata_loader,
        root.path().to_path_buf(),
        RawJpegPreference::Raw,
        &HashSet::from([hidden]),
        Arc::new(AtomicBool::new(false)),
    )
    .unwrap();

    assert_eq!(images.paths, [visible.join("a.jpg"), visible.join("z.jpg")]);
}

#[test]
fn thumbnail_selection_supports_replace_toggle_and_ranges() {
    let mut selection = ImageSelection::default();
    selection.select(2, false, false);
    assert_eq!(selection.indices, BTreeSet::from([2]));

    selection.select(4, false, true);
    assert_eq!(selection.indices, BTreeSet::from([2, 4]));

    selection.select(6, true, false);
    assert_eq!(selection.indices, BTreeSet::from([4, 5, 6]));
    assert_eq!(selection.focused, Some(6));
}

#[test]
fn work_progress_tracks_overlapping_tasks_and_resets_batches() {
    let mut work = WorkProgress::default();
    let first = work.start("First task");
    let second = work.start("Second task");
    work.finish(first);
    assert_eq!(work.active, 1);
    assert_eq!(work.fraction(), 0.5);
    work.finish(second);
    work.start("New batch");
    assert_eq!(work.active, 1);
    assert_eq!(work.completed, 0);
    assert_eq!(work.total, 1);
}

#[test]
fn image_zoom_starts_fitted_and_toggle_resets_pan() {
    let mut state = ImageViewState::new(PathBuf::from("photo.jpg"));
    assert_eq!(state.zoom, ZoomLevel::FitWindow);
    state.toggle_zoom();
    assert_eq!(state.zoom, ZoomLevel::ActualSize);
    state.pan = (100.0, -50.0);
    state.toggle_zoom();
    assert_eq!(state.zoom, ZoomLevel::FitWindow);
    assert_eq!(state.pan, (0.0, 0.0));
}

#[test]
fn virtual_grid_demands_only_visible_rows_and_prioritizes_the_center() {
    let images: Vec<_> = (0..20)
        .map(|index| PathBuf::from(format!("image-{index}.jpg")))
        .collect();
    let demands = visible_thumbnail_demands(&images, 1..3, 4, 300);
    let paths: Vec<_> = demands.iter().map(|demand| demand.0.clone()).collect();
    assert_eq!(paths, images[4..12]);
    let strongest = demands.iter().max_by_key(|demand| demand.2).unwrap();
    assert_eq!(strongest.0, images[8]);
}

#[test]
fn speculative_batches_are_bounded_and_follow_scroll_direction() {
    assert_eq!(speculative_indices(20, 8..12, 1, 0, 3), vec![12, 13, 14]);
    assert_eq!(speculative_indices(20, 8..12, -1, 0, 3), vec![7, 6, 5]);
    assert_eq!(speculative_indices(20, 8..12, 1, 1, 3), vec![15, 16, 17]);
    assert!(speculative_indices(4, 0..4, 1, 0, 64).is_empty());
    assert_eq!(
        speculative_indices(1_000_000_000, 100..110, 1, 0, 64).len(),
        64
    );
}

#[test]
fn prediction_identity_changes_with_grid_geometry_and_display() {
    let base = GridPredictionState {
        visible_rows: 1..3,
        columns: 4,
        thumbnail_width: 300,
        preview_variant: RenderVariant::Preview {
            max_width: 1920,
            max_height: 1080,
        },
    };
    let mut changed = base.clone();
    changed.columns = 5;
    assert_ne!(base, changed);
    changed = base.clone();
    changed.preview_variant = RenderVariant::Preview {
        max_width: 3840,
        max_height: 2160,
    };
    assert_ne!(base, changed);
}

#[test]
fn virtual_grid_callbacks_merge_fragments_but_replace_a_disjoint_viewport() {
    let variant = RenderVariant::Preview {
        max_width: 1920,
        max_height: 1080,
    };
    let mut pending = None;
    merge_grid_demand(
        &mut pending,
        PendingGridDemand {
            node: NodeId(1),
            visible_rows: 4..7,
            columns: 3,
            thumbnail_width: 300,
            preview_variant: variant,
        },
    );
    merge_grid_demand(
        &mut pending,
        PendingGridDemand {
            node: NodeId(1),
            visible_rows: 2..5,
            columns: 3,
            thumbnail_width: 300,
            preview_variant: variant,
        },
    );
    assert_eq!(pending.as_ref().unwrap().visible_rows, 2..7);

    merge_grid_demand(
        &mut pending,
        PendingGridDemand {
            node: NodeId(1),
            visible_rows: 20..23,
            columns: 3,
            thumbnail_width: 300,
            preview_variant: variant,
        },
    );
    assert_eq!(pending.unwrap().visible_rows, 20..23);
}

#[test]
#[ignore = "manual end-to-end queue benchmark"]
fn benchmark_all_scheduled_jobs() {
    let directory = PathBuf::from("/mnt/pictures/Pictures/2018/Nyaralás/2018-08-06");
    let nef = directory.join("DSC_0389.NEF");
    let jpeg = directory.join("DSC_2587.JPG");
    let cr3 = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test/IMG_0052.CR3");
    let heic = PathBuf::from("/mnt/pictures/Pictures/2026/Alpok túra/2026-07-25/IMG_5562.HEIC");
    for path in [&directory, &nef, &jpeg, &cr3, &heic] {
        assert!(path.exists(), "missing benchmark input: {}", path.display());
    }

    let started = Instant::now();
    let scan = discovery::scan_directory(&directory, RawJpegPreference::Raw).unwrap();
    let directory_scan = started.elapsed();

    let cache = tempfile::tempdir().unwrap();
    let limiter = JobLimiter::new(2);
    let metadata_loader = MetadataLoader::new(cache.path().join("metadata"), 2, limiter.clone());
    let started = Instant::now();
    let metadata =
        metadata_loader.load_directory(scan.images.clone(), Arc::new(AtomicBool::new(false)));
    let metadata_cold = started.elapsed();
    let started = Instant::now();
    let cached_metadata =
        metadata_loader.load_directory(scan.images, Arc::new(AtomicBool::new(false)));
    let metadata_cached = started.elapsed();
    assert_eq!(metadata.len(), cached_metadata.len());

    let pipeline =
        PreviewRenderer::with_source_cache_capacity(cache.path().join("renders"), 2, 2, 8);
    let render_limiter = limiter.clone();
    let pipeline_for_scheduler = pipeline.clone();
    let scheduler = RenderScheduler::new(
        2,
        Arc::new(move |key, priority, cancelled| {
            let Some(_permit) = render_limiter.acquire(
                priority.class >= scheduler::PriorityClass::Visible,
                cancelled,
            ) else {
                return PreviewState::Error("cancelled".into());
            };
            pipeline_for_scheduler.render(
                key.path.clone(),
                key.variant,
                priority.class == scheduler::PriorityClass::Interactive,
                true,
                cancelled,
            )
        }),
    );
    let mut scope = 10_000_u64;
    let mut run = |path: &PathBuf, variant: RenderVariant| {
        scope += 1;
        let started = Instant::now();
        let subscription = scheduler
            .replace_scope(
                DemandScopeId(scope),
                vec![JobDemand {
                    key: JobKey {
                        path: path.clone(),
                        variant,
                        render_identity: RenderIdentity::default(),
                        recipe: None,
                    },
                    priority: JobPriority::INTERACTIVE,
                    subscribe: true,
                }],
            )
            .pop()
            .unwrap();
        let completion = block_on(subscription.receiver).unwrap();
        assert!(matches!(
            completion,
            JobCompletion::Finished(PreviewState::Loaded(_))
        ));
        started.elapsed()
    };
    let (
        jpeg_thumbnail,
        nef_thumbnail,
        cr3_thumbnail,
        heic_thumbnail,
        nef_preview,
        cr3_preview,
        heic_preview,
    ) = {
        let mut cached_variant = |path: &PathBuf, variant: RenderVariant| {
            let writes = pipeline.stats().cache_writes;
            let total_started = Instant::now();
            let cold = run(path, variant);
            let deadline = Instant::now() + Duration::from_secs(30);
            while pipeline.stats().cache_writes == writes && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(2));
            }
            assert!(
                pipeline.stats().cache_writes > writes,
                "cache write timed out"
            );
            let persisted = total_started.elapsed();
            let cached = run(path, variant);
            (cold, persisted, cached)
        };

        (
            cached_variant(&jpeg, RenderVariant::Thumbnail { width: 300 }),
            cached_variant(&nef, RenderVariant::Thumbnail { width: 300 }),
            cached_variant(&cr3, RenderVariant::Thumbnail { width: 300 }),
            cached_variant(&heic, RenderVariant::Thumbnail { width: 300 }),
            cached_variant(
                &nef,
                RenderVariant::Preview {
                    max_width: 1920,
                    max_height: 1080,
                },
            ),
            cached_variant(
                &cr3,
                RenderVariant::Preview {
                    max_width: 1920,
                    max_height: 1080,
                },
            ),
            cached_variant(
                &heic,
                RenderVariant::Preview {
                    max_width: 1920,
                    max_height: 1080,
                },
            ),
        )
    };
    let jpeg_full = run(&jpeg, RenderVariant::Full);
    let nef_full = run(&nef, RenderVariant::Full);
    let cr3_full = run(&cr3, RenderVariant::Full);
    let heic_full = run(&heic, RenderVariant::Full);

    let overhead_scheduler = RenderScheduler::new(
        1,
        Arc::new(|_, _, _| PreviewState::Error("benchmark".into())),
    );
    let overhead_started = Instant::now();
    for index in 0..1_000_u32 {
        let subscription = overhead_scheduler
            .replace_scope(
                DemandScopeId(20_000 + u64::from(index)),
                vec![JobDemand {
                    key: JobKey {
                        path: PathBuf::from(format!("overhead-{index}")),
                        variant: RenderVariant::Thumbnail { width: 300 },
                        render_identity: RenderIdentity::default(),
                        recipe: None,
                    },
                    priority: JobPriority::INTERACTIVE,
                    subscribe: true,
                }],
            )
            .pop()
            .unwrap();
        let _ = block_on(subscription.receiver).unwrap();
    }
    let scheduler_overhead = overhead_started.elapsed() / 1_000;

    eprintln!(
        "scheduler_overhead={scheduler_overhead:?}\ndirectory_scan={directory_scan:?}\nmetadata_cold={metadata_cold:?}\nmetadata_cached={metadata_cached:?}\njpeg_thumbnail_cold={:?}\njpeg_thumbnail_persisted={:?}\njpeg_thumbnail_cached={:?}\nnef_thumbnail_cold={:?}\nnef_thumbnail_persisted={:?}\nnef_thumbnail_cached={:?}\ncr3_thumbnail_cold={:?}\ncr3_thumbnail_persisted={:?}\ncr3_thumbnail_cached={:?}\nheic_thumbnail_cold={:?}\nheic_thumbnail_persisted={:?}\nheic_thumbnail_cached={:?}\nnef_preview_cold={:?}\nnef_preview_persisted={:?}\nnef_preview_cached={:?}\ncr3_preview_cold={:?}\ncr3_preview_persisted={:?}\ncr3_preview_cached={:?}\nheic_preview_cold={:?}\nheic_preview_persisted={:?}\nheic_preview_cached={:?}\njpeg_full={jpeg_full:?}\nnef_full={nef_full:?}\ncr3_full={cr3_full:?}\nheic_full={heic_full:?}\nmetrics={:?}",
        jpeg_thumbnail.0,
        jpeg_thumbnail.1,
        jpeg_thumbnail.2,
        nef_thumbnail.0,
        nef_thumbnail.1,
        nef_thumbnail.2,
        cr3_thumbnail.0,
        cr3_thumbnail.1,
        cr3_thumbnail.2,
        heic_thumbnail.0,
        heic_thumbnail.1,
        heic_thumbnail.2,
        nef_preview.0,
        nef_preview.1,
        nef_preview.2,
        cr3_preview.0,
        cr3_preview.1,
        cr3_preview.2,
        heic_preview.0,
        heic_preview.1,
        heic_preview.2,
        pipeline.stats(),
    );
}

#[test]
#[ignore = "manual cancellation and parallel-throughput benchmark"]
fn benchmark_scheduler_cancellation_and_parallelism() {
    let directory = PathBuf::from("/mnt/pictures/Pictures/2018/Nyaralás/2018-08-06");
    let raw_paths = [
        "DSC_0389.NEF",
        "DSC_0390.NEF",
        "DSC_0391.NEF",
        "DSC_0392.NEF",
    ]
    .map(|name| directory.join(name));
    let jpeg = directory.join("DSC_2587.JPG");
    let cr3 = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test/IMG_0052.CR3");
    for path in raw_paths.iter().chain([&jpeg, &cr3]) {
        assert!(
            path.is_file(),
            "missing benchmark input: {}",
            path.display()
        );
    }

    let throughput = |parallelism: usize| {
        let cache = tempfile::tempdir().unwrap();
        let pipeline = PreviewRenderer::with_source_cache_capacity(
            cache.path().to_path_buf(),
            parallelism,
            parallelism,
            8,
        );
        let executor_pipeline = pipeline.clone();
        let scheduler = RenderScheduler::new(
            parallelism,
            Arc::new(move |key, _, cancelled| {
                executor_pipeline.render(key.path.clone(), key.variant, true, false, cancelled)
            }),
        );
        let started = Instant::now();
        let subscriptions = scheduler.replace_scope(
            DemandScopeId(30_000 + parallelism as u64),
            raw_paths
                .iter()
                .cloned()
                .map(|path| JobDemand {
                    key: JobKey {
                        path,
                        variant: RenderVariant::Thumbnail { width: 300 },
                        render_identity: RenderIdentity::default(),
                        recipe: None,
                    },
                    priority: JobPriority::INTERACTIVE,
                    subscribe: true,
                })
                .collect(),
        );
        for subscription in subscriptions {
            assert!(matches!(
                block_on(subscription.receiver).unwrap(),
                JobCompletion::Finished(PreviewState::Loaded(_))
            ));
        }
        started.elapsed()
    };

    let cancellation_handoff = |raw: &PathBuf| {
        let cache = tempfile::tempdir().unwrap();
        let pipeline =
            PreviewRenderer::with_source_cache_capacity(cache.path().to_path_buf(), 1, 1, 1);
        let executor_pipeline = pipeline.clone();
        let scheduler = RenderScheduler::new(
            1,
            Arc::new(move |key, _, cancelled| {
                executor_pipeline.render(key.path.clone(), key.variant, true, false, cancelled)
            }),
        );
        let raw_scope = DemandScopeId(40_000);
        let _ = scheduler.replace_scope(
            raw_scope,
            vec![JobDemand {
                key: JobKey {
                    path: raw.clone(),
                    variant: RenderVariant::Thumbnail { width: 300 },
                    render_identity: RenderIdentity::default(),
                    recipe: None,
                },
                priority: JobPriority::INTERACTIVE,
                subscribe: true,
            }],
        );
        let deadline = Instant::now() + Duration::from_secs(5);
        while scheduler.snapshot().running == 0 && Instant::now() < deadline {
            std::thread::yield_now();
        }
        scheduler.clear_scope(raw_scope);
        let started = Instant::now();
        let subscription = scheduler
            .replace_scope(
                DemandScopeId(40_001),
                vec![JobDemand {
                    key: JobKey {
                        path: jpeg.clone(),
                        variant: RenderVariant::Thumbnail { width: 300 },
                        render_identity: RenderIdentity::default(),
                        recipe: None,
                    },
                    priority: JobPriority::INTERACTIVE,
                    subscribe: true,
                }],
            )
            .pop()
            .unwrap();
        assert!(matches!(
            block_on(subscription.receiver).unwrap(),
            JobCompletion::Finished(PreviewState::Loaded(_))
        ));
        started.elapsed()
    };

    eprintln!(
        "raw_batch_p1={:?}\nraw_batch_p2={:?}\nraw_batch_p4={:?}\nnef_cancel_to_jpeg={:?}\ncr3_cancel_to_jpeg={:?}",
        throughput(1),
        throughput(2),
        throughput(4),
        cancellation_handoff(&raw_paths[0]),
        cancellation_handoff(&cr3),
    );
}
