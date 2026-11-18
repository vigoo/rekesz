//! Screenshot tour: a dev-only harness that drives the real window through the
//! app's UI surfaces so an external script can capture screenshots of each
//! settled state.
//!
//! Enabled with `REKESZ_SHOT_TOUR=1`. The tour mutates view state directly
//! (the same fields the interactive handlers set), waits for each state to
//! settle, prints `REKESZ_SHOT_READY:<step>` on stdout, and waits for a
//! `NEXT` line on stdin so the controlling script can capture a screenshot in
//! between. A step whose prerequisites are missing (no scanner source, no
//! people data, …) emits `REKESZ_SHOT_SKIP:<step>` instead and is captured
//! as absent.
//!
//! `REKESZ_SHOT_SUITE` selects the step list:
//! - `real` (default): the library under review. Every user action that
//!   would write into the library (import copy, destination folder creation)
//!   is refused; see [`real_suite_tour_active`].
//! - `scratch`: a disposable library for transient states (empty views,
//!   banners, a real import job, an active job queue). Mutations land in the
//!   scratch library's own catalog and directories.
//!
//! Directory inputs come from the environment:
//! `REKESZ_SHOT_DIR` (photo-rich demo directory), `REKESZ_SHOT_EMPTY_DIR`
//! (directory without images, with subfolders), `REKESZ_SHOT_DEST` (import
//! destination for the scratch suite).

use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures::StreamExt;
use gpui::{AsyncWindowContext, Context, WeakEntity, Window};

use super::*;

/// Whether the screenshot tour is running at all.
pub(crate) fn tour_enabled() -> bool {
    std::env::var_os("REKESZ_SHOT_TOUR")
        .map(|value| value == "1")
        .unwrap_or(false)
}

fn suite() -> &'static str {
    match std::env::var("REKESZ_SHOT_SUITE").as_deref() {
        Ok("scratch") => "scratch",
        _ => "real",
    }
}

/// While the tour runs against the real library, actions that copy files into
/// the library or create folders in it are refused. Callers gate the import
/// start and destination-folder creation paths.
pub(crate) fn real_suite_tour_active() -> bool {
    tour_enabled() && suite() == "real"
}

#[derive(Clone, Default)]
struct TourContext {
    demo_dir: Option<PathBuf>,
    empty_dir: Option<PathBuf>,
    import_destination: Option<PathBuf>,
}

impl TourContext {
    fn from_env() -> Self {
        Self {
            demo_dir: env_path("REKESZ_SHOT_DIR"),
            empty_dir: env_path("REKESZ_SHOT_EMPTY_DIR"),
            import_destination: env_path("REKESZ_SHOT_DEST"),
        }
    }
}

fn env_path(key: &str) -> Option<PathBuf> {
    std::env::var_os(key)
        .map(PathBuf::from)
        .filter(|path| path.is_dir())
}

type ApplyFn =
    fn(&mut PictureBrowser, &mut Window, &mut Context<PictureBrowser>, &TourContext) -> bool;
type ReadyFn = fn(&PictureBrowser, &TourContext) -> bool;

struct TourStep {
    id: &'static str,
    apply: ApplyFn,
    ready: Option<ReadyFn>,
    /// Milliseconds the capture script waits after the marker before shooting:
    /// thumbnail decodes, map rendering, dialog animations.
    settle_ms: u64,
    timeout: Duration,
}

impl TourStep {
    /// A step with no readiness condition; the script's settle delay covers
    /// asynchronous rendering.
    const fn settle(id: &'static str, apply: ApplyFn, settle_ms: u64) -> Self {
        Self {
            id,
            apply,
            ready: None,
            settle_ms,
            timeout: Duration::from_secs(90),
        }
    }

    /// A step that waits for `ready` (bounded by `timeout`) before emitting
    /// its marker; a timeout or inapplicable step is skipped.
    const fn waiting(
        id: &'static str,
        apply: ApplyFn,
        ready: ReadyFn,
        settle_ms: u64,
        timeout: Duration,
    ) -> Self {
        Self {
            id,
            apply,
            ready: Some(ready),
            settle_ms,
            timeout,
        }
    }
}

/// Starts the tour on the first render pass, where a `&Window` is available.
pub(crate) fn maybe_start(window: &Window, cx: &mut Context<PictureBrowser>) {
    if !tour_enabled() {
        return;
    }
    cx.spawn_in(window, async move |this, cx| {
        run_tour(this, cx).await;
    })
    .detach();
}

async fn run_tour(this: WeakEntity<PictureBrowser>, cx: &mut AsyncWindowContext) {
    let context = TourContext::from_env();
    let steps = match suite() {
        "scratch" => scratch_steps(),
        _ => real_steps(),
    };
    // Leave the app alone while startup (catalog load, first grid scan)
    // settles, so the first capture shows a quiet initial state.
    cx.background_executor().timer(Duration::from_secs(3)).await;

    let (next_tx, mut next_rx) = futures::channel::mpsc::unbounded::<String>();
    std::thread::spawn(move || forward_stdin_lines(next_tx));

    for step in steps {
        let applied = this.update_in(cx, |this, window, cx| {
            window.activate_window();
            (step.apply)(this, window, cx, &context)
        });
        let applied = match applied {
            Ok(applied) => applied,
            Err(_) => return,
        };
        let mut ready = applied;
        if applied && let Some(ready_check) = step.ready {
            ready = wait_for(&this, cx, &context, step.timeout, ready_check).await;
        }
        if ready {
            emit(&format!("REKESZ_SHOT_READY:{}", step.id));
            emit(&format!("REKESZ_SHOT_SETTLE:{}", step.settle_ms));
        } else {
            emit(&format!("REKESZ_SHOT_SKIP:{}", step.id));
        }
        // Park until the script answers, so the state stays exactly as the
        // script saw it while it captures. Every marker costs exactly one
        // NEXT — skipping parks too, otherwise stale NEXTs would desync the
        // captures. A closed channel (script exited or closed stdin) ends the
        // parking loop and lets the tour wind down.
        while let Some(line) = next_rx.next().await {
            if line.trim().eq_ignore_ascii_case("next") {
                break;
            }
        }
    }
    emit("REKESZ_SHOT_DONE");
}

async fn wait_for(
    this: &WeakEntity<PictureBrowser>,
    cx: &mut AsyncWindowContext,
    context: &TourContext,
    timeout: Duration,
    ready: impl Fn(&PictureBrowser, &TourContext) -> bool,
) -> bool {
    let deadline = Instant::now() + timeout;
    loop {
        let ready_now = this
            .update(cx, |this, _cx| ready(this, context))
            .unwrap_or(false);
        if ready_now || Instant::now() >= deadline {
            return ready_now;
        }
        cx.background_executor()
            .timer(Duration::from_millis(100))
            .await;
    }
}

fn emit(line: &str) {
    println!("{line}");
    let _ = std::io::stdout().flush();
}

fn forward_stdin_lines(sender: futures::channel::mpsc::UnboundedSender<String>) {
    let stdin = std::io::stdin();
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        if sender.unbounded_send(line).is_err() {
            break;
        }
    }
}

// ---------------------------------------------------------------------------
// Step list: real library. Cleanup for each surface happens at the top of
// the next step's apply, so every capture starts from a quiet base.
// ---------------------------------------------------------------------------

fn real_steps() -> Vec<TourStep> {
    vec![
        TourStep::settle("grid-initial", no_op, 1500),
        TourStep::waiting(
            "grid-demo",
            apply_demo_dir,
            ready_main_view_is_demo,
            // A cold thumbnail cache on a photo-rich directory renders for
            // several seconds; give the grid time to fill before shooting.
            9000,
            Duration::from_secs(20),
        ),
        TourStep::settle("grid-selection", apply_grid_selection, 2500),
        TourStep::settle("grid-tooltip", apply_grid_tooltip, 2000),
        TourStep::settle("menu-file", apply_file_menu, 500),
        TourStep::settle("menu-view", apply_view_menu, 500),
        TourStep::settle("context-menu-thumbnail", apply_thumbnail_context_menu, 500),
        TourStep::waiting(
            "viewer-fit",
            apply_open_viewer,
            ready_image_view,
            2000,
            Duration::from_secs(30),
        ),
        TourStep::settle("viewer-1to1", apply_zoom_1to1, 2500),
        TourStep::settle("develop-inspector", apply_develop_inspector, 800),
        TourStep::settle("develop-baseline", apply_develop_baseline, 500),
        TourStep::waiting(
            "import-sources",
            apply_open_import,
            ready_import_sources,
            800,
            Duration::from_secs(20),
        ),
        TourStep::waiting(
            "import-listing",
            apply_select_folder_source,
            ready_import_listing,
            1500,
            Duration::from_secs(40),
        ),
        TourStep::waiting(
            "import-destination-picker",
            apply_destination_picker,
            ready_destination_picker,
            800,
            Duration::from_secs(15),
        ),
        TourStep::settle("import-layout-combobox", apply_layout_combobox, 500),
        TourStep::waiting(
            "import-scanner-select",
            apply_select_scanner_source,
            ready_scanner_caps,
            500,
            Duration::from_secs(15),
        ),
        TourStep::settle("scanner-panel", apply_seed_scanner_panel, 800),
        TourStep::waiting(
            "map-view",
            apply_exit_import_open_map,
            ready_map_rendered,
            2000,
            Duration::from_secs(30),
        ),
        TourStep::settle("map-permission-overlay", apply_map_permission_overlay, 500),
        TourStep::settle("map-card", apply_map_card, 1500),
        TourStep::waiting(
            "people-triage-queue",
            apply_people_overview,
            ready_people_triage,
            1500,
            Duration::from_secs(90),
        ),
        TourStep::waiting(
            "people-detail",
            apply_person_detail,
            ready_people_faces,
            1500,
            Duration::from_secs(45),
        ),
        TourStep::settle("people-selection-toolbar", apply_select_faces, 800),
        TourStep::settle("people-rename-dialog", apply_people_rename_dialog, 500),
        TourStep::waiting(
            "people-merge-dialog",
            apply_people_merge_dialog,
            ready_people_merge_preview,
            800,
            Duration::from_secs(120),
        ),
        TourStep::settle("dialog-settings", apply_settings_dialog, 500),
        TourStep::waiting(
            "dialog-cache-statistics",
            apply_cache_statistics_dialog,
            ready_cache_statistics,
            800,
            Duration::from_secs(20),
        ),
        TourStep::settle("dialog-lightroom-import", apply_lightroom_dialog, 500),
        TourStep::settle(
            "dialog-selection-create",
            apply_selection_create_dialog,
            500,
        ),
        TourStep::settle("selections-overflow-menu", apply_selections_overflow, 500),
        TourStep::settle("selection-context-menu", apply_selection_context_menu, 500),
        TourStep::settle("job-queue-panel", apply_job_queue, 800),
        TourStep::settle("grid-final", apply_final_cleanup, 800),
    ]
}

fn no_op(
    _: &mut PictureBrowser,
    _: &mut Window,
    _: &mut Context<PictureBrowser>,
    _: &TourContext,
) -> bool {
    true
}

fn apply_demo_dir(
    this: &mut PictureBrowser,
    _: &mut Window,
    cx: &mut Context<PictureBrowser>,
    context: &TourContext,
) -> bool {
    let Some(demo) = context.demo_dir.clone() else {
        return false;
    };
    dismiss_overlays(this);
    this.tour_tooltip_index = None;
    let node = this.model.ensure_node(demo);
    this.rebuild_visible();
    this.select(node, cx);
    true
}

fn ready_main_view_is_demo(this: &PictureBrowser, context: &TourContext) -> bool {
    let Some(demo) = context.demo_dir.as_deref() else {
        return false;
    };
    match &this.main_view {
        MainView::Dir(path, _) => path == demo,
        MainView::Image(_) => false,
    }
}

fn apply_grid_selection(
    this: &mut PictureBrowser,
    _: &mut Window,
    cx: &mut Context<PictureBrowser>,
    _: &TourContext,
) -> bool {
    if this.directory_view_mode != DirectoryViewMode::Grid || this.selected_images().is_empty() {
        return false;
    }
    this.tour_tooltip_index = None;
    this.select_thumbnail(0, false, false, cx);
    let extra = this.selected_images().len().min(4);
    for index in 1..extra {
        this.select_thumbnail(index, true, false, cx);
    }
    true
}

fn apply_grid_tooltip(
    this: &mut PictureBrowser,
    _: &mut Window,
    cx: &mut Context<PictureBrowser>,
    _: &TourContext,
) -> bool {
    if this.directory_view_mode != DirectoryViewMode::Grid || this.selected_images().is_empty() {
        return false;
    }
    this.tour_tooltip_index = Some(0);
    cx.notify();
    true
}

fn apply_file_menu(
    this: &mut PictureBrowser,
    _: &mut Window,
    cx: &mut Context<PictureBrowser>,
    _: &TourContext,
) -> bool {
    this.tour_tooltip_index = None;
    this.show_view_menu = false;
    this.show_file_menu = true;
    cx.notify();
    true
}

fn apply_view_menu(
    this: &mut PictureBrowser,
    _: &mut Window,
    cx: &mut Context<PictureBrowser>,
    _: &TourContext,
) -> bool {
    this.show_file_menu = false;
    this.show_view_menu = true;
    cx.notify();
    true
}

fn apply_thumbnail_context_menu(
    this: &mut PictureBrowser,
    _: &mut Window,
    cx: &mut Context<PictureBrowser>,
    _: &TourContext,
) -> bool {
    this.show_view_menu = false;
    let Some(path) = this.selected_images().first().cloned() else {
        return false;
    };
    this.path_context_menu = Some(PathContextMenu {
        path,
        position: (430.0, 300.0),
        exportable: true,
        directory: None,
    });
    cx.notify();
    true
}

fn apply_open_viewer(
    this: &mut PictureBrowser,
    _: &mut Window,
    cx: &mut Context<PictureBrowser>,
    _: &TourContext,
) -> bool {
    this.path_context_menu = None;
    let paths: Vec<PathBuf> = this.selected_images().to_vec();
    let pick = paths
        .iter()
        .find(|path| has_develop_recipe(this.recipes.as_deref(), path))
        .or_else(|| paths.first());
    let Some(path) = pick.cloned() else {
        return false;
    };
    this.show_image(path, cx);
    true
}

fn ready_image_view(this: &PictureBrowser, _: &TourContext) -> bool {
    matches!(this.main_view, MainView::Image(_))
}

fn apply_zoom_1to1(
    this: &mut PictureBrowser,
    _: &mut Window,
    cx: &mut Context<PictureBrowser>,
    _: &TourContext,
) -> bool {
    let MainView::Image(state) = &mut this.main_view else {
        return false;
    };
    state.zoom = ZoomLevel::ActualSize;
    state.pan = (0.0, 0.0);
    cx.notify();
    true
}

fn apply_develop_inspector(
    this: &mut PictureBrowser,
    _: &mut Window,
    cx: &mut Context<PictureBrowser>,
    _: &TourContext,
) -> bool {
    this.show_develop_inspector = true;
    this.developed = true;
    cx.notify();
    true
}

fn apply_develop_baseline(
    this: &mut PictureBrowser,
    _: &mut Window,
    cx: &mut Context<PictureBrowser>,
    _: &TourContext,
) -> bool {
    this.developed = false;
    cx.notify();
    true
}

fn apply_open_import(
    this: &mut PictureBrowser,
    window: &mut Window,
    cx: &mut Context<PictureBrowser>,
    _: &TourContext,
) -> bool {
    this.show_develop_inspector = false;
    this.exit_grid_selection_mode(cx);
    // Back to the image viewer's origin (the grid) so the import module opens
    // from a plain library view.
    if matches!(this.main_view, MainView::Image(_)) {
        this.close_image(&CloseImage, window, cx);
    }
    this.open_import_view(window, cx);
    true
}

fn ready_import_sources(this: &PictureBrowser, _: &TourContext) -> bool {
    !this.import_view.sources.is_empty()
}

fn apply_select_folder_source(
    this: &mut PictureBrowser,
    _: &mut Window,
    cx: &mut Context<PictureBrowser>,
    _: &TourContext,
) -> bool {
    this.import_view.layout_menu_open = false;
    this.import_view.destination_picker_open = false;
    let entry = this
        .import_view
        .sources
        .iter()
        .find(|entry| matches!(entry.kind, sources::SourceKind::Folder))
        .or_else(|| {
            this.import_view
                .sources
                .iter()
                .find(|entry| matches!(entry.kind, sources::SourceKind::Volume { .. }))
        })
        .cloned();
    let Some(entry) = entry else {
        return false;
    };
    this.select_import_source(entry.id, cx);
    true
}

fn ready_import_listing(this: &PictureBrowser, _: &TourContext) -> bool {
    matches!(this.import_view.listing, import_view::ImportListing::Loaded)
        && !this.import_view.photos.is_empty()
}

fn apply_destination_picker(
    this: &mut PictureBrowser,
    window: &mut Window,
    cx: &mut Context<PictureBrowser>,
    _: &TourContext,
) -> bool {
    if !this.import_view.destination_picker_open {
        this.toggle_destination_picker(window, cx);
    }
    true
}

fn ready_destination_picker(this: &PictureBrowser, _: &TourContext) -> bool {
    this.import_view.destination_picker_open && this.import_view.picker_entries.is_some()
}

fn apply_layout_combobox(
    this: &mut PictureBrowser,
    _: &mut Window,
    cx: &mut Context<PictureBrowser>,
    _: &TourContext,
) -> bool {
    this.import_view.destination_picker_open = false;
    this.import_view.layout_menu_open = true;
    cx.notify();
    true
}

fn apply_select_scanner_source(
    this: &mut PictureBrowser,
    _: &mut Window,
    cx: &mut Context<PictureBrowser>,
    _: &TourContext,
) -> bool {
    this.import_view.layout_menu_open = false;
    let entry = this
        .import_view
        .sources
        .iter()
        .find(|entry| matches!(entry.kind, sources::SourceKind::Scanner { .. }))
        .cloned();
    let Some(entry) = entry else {
        return false;
    };
    this.select_import_source(entry.id, cx);
    true
}

fn ready_scanner_caps(this: &PictureBrowser, _: &TourContext) -> bool {
    matches!(
        this.import_view.scan_panel.caps,
        import_view::ScanCapsState::Loaded(_) | import_view::ScanCapsState::Error(_)
    )
}

fn apply_seed_scanner_panel(
    this: &mut PictureBrowser,
    _: &mut Window,
    cx: &mut Context<PictureBrowser>,
    _: &TourContext,
) -> bool {
    let panel = &mut this.import_view.scan_panel;
    // The region overlay only renders over a loaded preview; seed one so the
    // capture shows the rectangle, handles, and region label without
    // touching the real device.
    if matches!(panel.preview, import_view::ScanPreviewState::Idle) {
        panel.preview = seeded_scan_preview();
    }
    if !matches!(panel.caps, import_view::ScanCapsState::Loaded(_)) {
        panel.caps = import_view::ScanCapsState::Loaded(Arc::new(
            sources::scanner::escl::ScannerCapabilities {
                platen: sources::scanner::escl::InputCaps {
                    min_width: 64,
                    max_width: 2480,
                    min_height: 64,
                    max_height: 3508,
                },
                max_scan_regions: 1,
                color_modes: vec!["RGB24".into(), "GRAY8".into(), "Black&1".into()],
                document_formats: vec!["image/jpeg".into()],
                resolutions: vec![150, 300, 600],
            },
        ));
    }
    panel.region = import_view::ScanRegionFractions {
        x0: 0.18,
        y0: 0.15,
        x1: 0.86,
        y1: 0.82,
    };
    panel.dpi = 300;
    panel.color = config::ScanColorMode::Rgb;
    cx.notify();
    true
}

fn seeded_scan_preview() -> import_view::ScanPreviewState {
    let mut image = image::RgbaImage::from_pixel(1240, 1754, image::Rgba([0xd8, 0xd5, 0xd0, 0xff]));
    // A lighter "document" area with a few darker bars, so the region
    // rectangle has something to sit over.
    for y in 260..1500 {
        for x in 180..1060 {
            image.put_pixel(x, y, image::Rgba([0xf3, 0xf1, 0xec, 0xff]));
        }
    }
    for (y0, y1) in [(420., 470.), (560., 610.), (700., 1350.)] {
        for y in y0 as u32..y1 as u32 {
            for x in 280..960 {
                image.put_pixel(x, y, image::Rgba([0xb9, 0xb4, 0xac, 0xff]));
            }
        }
    }
    let rendered = gpui::RenderImage::new(smallvec::smallvec![image::Frame::new(image)]);
    import_view::ScanPreviewState::Loaded {
        image: Arc::new(rendered),
        width: 1240,
        height: 1754,
    }
}

fn apply_exit_import_open_map(
    this: &mut PictureBrowser,
    _: &mut Window,
    cx: &mut Context<PictureBrowser>,
    _: &TourContext,
) -> bool {
    if this.directory_view_mode == DirectoryViewMode::Import {
        this.exit_import_view(cx);
    }
    this.map_card = None;
    this.set_directory_view_mode(DirectoryViewMode::Map, cx);
    true
}

fn ready_map_rendered(this: &PictureBrowser, _: &TourContext) -> bool {
    !matches!(this.map_render, MapRenderState::Loading)
}

fn apply_map_permission_overlay(
    this: &mut PictureBrowser,
    _: &mut Window,
    cx: &mut Context<PictureBrowser>,
    _: &TourContext,
) -> bool {
    this.settings.online_map_permission = config::OnlineMapPermission::Unknown;
    cx.notify();
    true
}

fn apply_map_card(
    this: &mut PictureBrowser,
    _: &mut Window,
    cx: &mut Context<PictureBrowser>,
    _: &TourContext,
) -> bool {
    this.settings.online_map_permission = config::OnlineMapPermission::Disabled;
    let paths: Vec<PathBuf> = this.selected_images().to_vec();
    let items: Vec<MapCardItem> = paths
        .iter()
        .take(6)
        .cloned()
        .map(|path| MapCardItem { path })
        .collect();
    if items.is_empty() {
        return false;
    }
    let total = items.len();
    let coordinate = metadata::GeoCoordinate::new(47.4979, 19.0402);
    let Some(coordinate) = coordinate else {
        return false;
    };
    this.map_card = Some(MapCard {
        items,
        page: 0,
        total,
        cluster_id: None,
        direct_indices: Some((0..total).collect()),
        coordinate,
        expansion_zoom: None,
    });
    cx.notify();
    true
}

fn apply_people_overview(
    this: &mut PictureBrowser,
    _: &mut Window,
    cx: &mut Context<PictureBrowser>,
    _: &TourContext,
) -> bool {
    this.map_card = None;
    this.settings.online_map_permission = config::OnlineMapPermission::Disabled;
    this.people_whole_library = true;
    this.set_directory_view_mode(DirectoryViewMode::People, cx);
    true
}

/// People mode now opens the triage work queue; it is ready once the queue
/// scope exists (the face list may legitimately be empty).
fn ready_people_triage(this: &PictureBrowser, _: &TourContext) -> bool {
    this.people_triage.is_some()
}

fn apply_person_detail(
    this: &mut PictureBrowser,
    _: &mut Window,
    cx: &mut Context<PictureBrowser>,
    _: &TourContext,
) -> bool {
    let PeoplePageLoad::Loaded(page) = &this.people_overview else {
        return false;
    };
    let Some(person) = page
        .items
        .iter()
        .max_by_key(|person| person.confirmed_faces + person.suggested_faces)
        .cloned()
    else {
        return false;
    };
    this.selected_person_name = Some(person.name.clone());
    this.selected_person = Some(person.id);
    this.people_special_page = None;
    this.people_page = 0;
    this.set_people_context_default(false, cx);
    this.refresh_people_page(cx);
    true
}

fn ready_people_faces(this: &PictureBrowser, _: &TourContext) -> bool {
    matches!(
        &this.people_faces,
        PeoplePageLoad::Loaded(page) if !page.items.is_empty()
    )
}

fn apply_select_faces(
    this: &mut PictureBrowser,
    _: &mut Window,
    cx: &mut Context<PictureBrowser>,
    _: &TourContext,
) -> bool {
    let keys = this.visible_people_face_keys();
    let picked: Vec<_> = keys.into_iter().take(2).collect();
    if picked.is_empty() {
        return false;
    }
    for key in picked {
        this.select_people_face(key, false, true, cx);
    }
    true
}

fn apply_people_rename_dialog(
    this: &mut PictureBrowser,
    window: &mut Window,
    cx: &mut Context<PictureBrowser>,
    _: &TourContext,
) -> bool {
    let Some(person_id) = this.selected_person else {
        return false;
    };
    let input = this
        .selected_person_name
        .clone()
        .unwrap_or_else(|| "Unnamed".to_owned());
    this.open_people_name_dialog(PeopleNameOperation::Rename { person_id }, input, window, cx);
    true
}

fn apply_people_merge_dialog(
    this: &mut PictureBrowser,
    window: &mut Window,
    cx: &mut Context<PictureBrowser>,
    _: &TourContext,
) -> bool {
    let Some(person_id) = this.selected_person else {
        return false;
    };
    this.people_operation_dialog = None;
    this.open_people_target_dialog(
        PeopleTargetOperation::Merge {
            source_person_id: person_id,
        },
        window,
        cx,
    );
    true
}

fn ready_people_merge_preview(this: &PictureBrowser, _: &TourContext) -> bool {
    matches!(
        &this.people_merge_preview,
        Some(PeopleMergePreviewLoad::Loaded(_))
    )
}

fn apply_settings_dialog(
    this: &mut PictureBrowser,
    _: &mut Window,
    cx: &mut Context<PictureBrowser>,
    _: &TourContext,
) -> bool {
    this.people_operation_dialog = None;
    this.people_merge_preview = None;
    this.settings_draft = Some(this.settings.clone());
    this.show_cache_statistics = false;
    this.settings_status = None;
    cx.notify();
    true
}

fn apply_cache_statistics_dialog(
    this: &mut PictureBrowser,
    window: &mut Window,
    cx: &mut Context<PictureBrowser>,
    _: &TourContext,
) -> bool {
    this.settings_draft = None;
    this.open_cache_statistics(&OpenCacheStatistics, window, cx);
    true
}

fn ready_cache_statistics(this: &PictureBrowser, _: &TourContext) -> bool {
    this.show_cache_statistics && this.disk_cache_usage.is_some()
}

fn apply_lightroom_dialog(
    this: &mut PictureBrowser,
    _: &mut Window,
    cx: &mut Context<PictureBrowser>,
    _: &TourContext,
) -> bool {
    this.show_cache_statistics = false;
    this.lightroom_import_dialog = Some(LightroomImportDialog {
        path: PathBuf::from("/photos/lightroom/catalog.lrcat"),
        state: LightroomImportState::Succeeded {
            images: 133_516,
            recipes: 133_516,
            faces: 48_123,
        },
    });
    cx.notify();
    true
}

fn apply_selection_create_dialog(
    this: &mut PictureBrowser,
    _: &mut Window,
    cx: &mut Context<PictureBrowser>,
    _: &TourContext,
) -> bool {
    this.lightroom_import_dialog = None;
    this.selection_dialog = Some(SelectionDialog::Create {
        input: "Best of 2026".to_owned(),
    });
    cx.notify();
    true
}

fn apply_selections_overflow(
    this: &mut PictureBrowser,
    _: &mut Window,
    cx: &mut Context<PictureBrowser>,
    _: &TourContext,
) -> bool {
    this.selection_dialog = None;
    this.selections_menu_open = true;
    cx.notify();
    true
}

fn apply_selection_context_menu(
    this: &mut PictureBrowser,
    _: &mut Window,
    cx: &mut Context<PictureBrowser>,
    _: &TourContext,
) -> bool {
    this.selections_menu_open = false;
    let directory = this.selection_album();
    let Some(selection) = this
        .album_selections(&directory)
        .first()
        .map(|selection| selection.name.clone())
    else {
        return false;
    };
    this.selection_context_menu = Some(SelectionContextMenu {
        directory,
        name: selection,
        position: (700.0, 80.0),
    });
    cx.notify();
    true
}

fn apply_job_queue(
    this: &mut PictureBrowser,
    _: &mut Window,
    cx: &mut Context<PictureBrowser>,
    _: &TourContext,
) -> bool {
    this.selection_context_menu = None;
    this.show_job_queue = true;
    cx.notify();
    true
}

fn apply_final_cleanup(
    this: &mut PictureBrowser,
    _: &mut Window,
    cx: &mut Context<PictureBrowser>,
    _: &TourContext,
) -> bool {
    this.show_job_queue = false;
    if this.directory_view_mode != DirectoryViewMode::Grid {
        this.set_directory_view_mode(DirectoryViewMode::Grid, cx);
    }
    true
}

fn dismiss_overlays(this: &mut PictureBrowser) {
    this.show_file_menu = false;
    this.show_view_menu = false;
    this.path_context_menu = None;
    this.selection_context_menu = None;
    this.selections_menu_open = false;
    this.selection_dialog = None;
    this.lightroom_import_dialog = None;
    this.settings_draft = None;
    this.show_cache_statistics = false;
    this.people_operation_dialog = None;
    this.map_card = None;
}

// ---------------------------------------------------------------------------
// Step list: scratch library (transient states, a real import job, an active
// job queue). Everything writes only into the scratch library.
// ---------------------------------------------------------------------------

fn scratch_steps() -> Vec<TourStep> {
    vec![
        TourStep::settle("grid-initial", no_op, 1500),
        TourStep::waiting(
            "grid-empty-folders",
            apply_empty_dir,
            ready_main_view_is_empty,
            1000,
            Duration::from_secs(20),
        ),
        TourStep::waiting(
            "grid-recursive-banner",
            apply_recursive_grid,
            ready_recursive_loaded,
            1000,
            Duration::from_secs(40),
        ),
        TourStep::waiting(
            "import-open-scratch",
            apply_exit_recursive_open_import,
            ready_import_sources,
            800,
            Duration::from_secs(20),
        ),
        TourStep::waiting(
            "import-listing-scratch",
            apply_select_folder_source,
            ready_import_listing,
            1200,
            Duration::from_secs(40),
        ),
        TourStep::settle("import-progress", apply_seed_import_progress, 600),
        TourStep::waiting(
            "import-report",
            apply_start_real_import,
            ready_import_report,
            1000,
            Duration::from_secs(180),
        ),
        TourStep::settle("thumbnail-progress-footer", apply_close_import_report, 400),
        TourStep::settle("switcher-floating", apply_switcher_floating, 800),
        TourStep::settle("map-permission-empty", apply_map_permission_empty, 1800),
        TourStep::waiting(
            "job-queue-active",
            apply_enable_face_analysis,
            ready_job_queue_rows,
            800,
            Duration::from_secs(60),
        ),
        TourStep::waiting(
            "grid-final-scratch",
            apply_pause_face_analysis,
            ready_background_idle,
            800,
            Duration::from_secs(120),
        ),
    ]
}

fn apply_empty_dir(
    this: &mut PictureBrowser,
    _: &mut Window,
    cx: &mut Context<PictureBrowser>,
    context: &TourContext,
) -> bool {
    let Some(empty) = context.empty_dir.clone() else {
        return false;
    };
    dismiss_overlays(this);
    let node = this.model.ensure_node(empty);
    this.rebuild_visible();
    this.select(node, cx);
    true
}

fn ready_main_view_is_empty(this: &PictureBrowser, context: &TourContext) -> bool {
    let Some(empty) = context.empty_dir.as_deref() else {
        return false;
    };
    match &this.main_view {
        MainView::Dir(path, _) => path == empty,
        MainView::Image(_) => false,
    }
}

fn apply_recursive_grid(
    this: &mut PictureBrowser,
    _: &mut Window,
    cx: &mut Context<PictureBrowser>,
    _: &TourContext,
) -> bool {
    if this.recursive_grid.is_some() {
        return true;
    }
    this.start_recursive_grid(cx);
    true
}

fn ready_recursive_loaded(this: &PictureBrowser, _: &TourContext) -> bool {
    matches!(
        &this.recursive_grid,
        Some(state) if matches!(state.status, RecursiveGridStatus::Loaded(_))
    )
}

fn apply_exit_recursive_open_import(
    this: &mut PictureBrowser,
    window: &mut Window,
    cx: &mut Context<PictureBrowser>,
    _: &TourContext,
) -> bool {
    if this.recursive_grid.is_some() {
        this.stop_recursive_grid(cx);
    }
    this.open_import_view(window, cx);
    true
}

fn apply_seed_import_progress(
    this: &mut PictureBrowser,
    _: &mut Window,
    cx: &mut Context<PictureBrowser>,
    _: &TourContext,
) -> bool {
    // Seed a frozen mid-run footer instead of racing a real copy: even a
    // few hundred megabytes finish in under a second on a fast disk, so a
    // timing-based capture would usually show the finished state.
    this.import_view.job_progress = import::ImportProgress {
        total_files: 12,
        total_bytes: 628_500_000,
        copied_files: 7,
        copied_bytes: 366_600_000,
        failed_files: 0,
        current: "Scan-008.jpg".to_owned(),
    };
    this.import_view.job = Some(import_view::ImportJobHandle {
        progress: Arc::new(std::sync::Mutex::new(import::ImportProgress {
            total_files: 12,
            total_bytes: 628_500_000,
            copied_files: 7,
            copied_bytes: 366_600_000,
            failed_files: 0,
            current: "Scan-008.jpg".to_owned(),
        })),
        cancel: Arc::new(AtomicBool::new(false)),
        work_id: 0,
    });
    cx.notify();
    true
}

fn apply_start_real_import(
    this: &mut PictureBrowser,
    _: &mut Window,
    cx: &mut Context<PictureBrowser>,
    context: &TourContext,
) -> bool {
    let Some(destination) = context.import_destination.clone() else {
        return false;
    };
    // Drop the seeded progress state, then run a real (fast) import so the
    // report dialog appears from an actual job.
    this.import_view.job = None;
    this.import_select_all(cx);
    this.import_view.destination_base = Some(destination);
    this.import_view.destination_picker_open = false;
    this.start_import(cx);
    true
}

fn ready_import_report(this: &PictureBrowser, _: &TourContext) -> bool {
    this.import_view.report.is_some()
}

fn apply_close_import_report(
    this: &mut PictureBrowser,
    _: &mut Window,
    cx: &mut Context<PictureBrowser>,
    _: &TourContext,
) -> bool {
    this.import_view.report = None;
    cx.notify();
    true
}

fn apply_switcher_floating(
    this: &mut PictureBrowser,
    _: &mut Window,
    cx: &mut Context<PictureBrowser>,
    _: &TourContext,
) -> bool {
    if this.directory_view_mode == DirectoryViewMode::Import {
        this.exit_import_view(cx);
    }
    this.settings.map_switcher_placement = config::MapSwitcherPlacement::Floating;
    cx.notify();
    true
}

fn apply_map_permission_empty(
    this: &mut PictureBrowser,
    _: &mut Window,
    cx: &mut Context<PictureBrowser>,
    _: &TourContext,
) -> bool {
    this.settings.map_switcher_placement = config::MapSwitcherPlacement::Header;
    this.settings.online_map_permission = config::OnlineMapPermission::Unknown;
    this.set_directory_view_mode(DirectoryViewMode::Map, cx);
    true
}

fn apply_enable_face_analysis(
    this: &mut PictureBrowser,
    _: &mut Window,
    cx: &mut Context<PictureBrowser>,
    _: &TourContext,
) -> bool {
    this.settings.online_map_permission = config::OnlineMapPermission::Disabled;
    this.set_directory_view_mode(DirectoryViewMode::Grid, cx);
    this.settings.automatic_face_analysis = true;
    this.face_analysis_paused = false;
    this.refresh_face_inventory(cx);
    this.show_job_queue = true;
    true
}

fn ready_job_queue_rows(this: &PictureBrowser, _: &TourContext) -> bool {
    this.face_inventory_active
        || !this.job_queue_snapshot.jobs.is_empty()
        || !this.face_job_queue_snapshot.is_empty()
        || !this.summary_job_queue_snapshot.is_empty()
}

fn apply_pause_face_analysis(
    this: &mut PictureBrowser,
    _: &mut Window,
    cx: &mut Context<PictureBrowser>,
    _: &TourContext,
) -> bool {
    this.face_analysis_paused = true;
    this.settings.automatic_face_analysis = false;
    this.show_job_queue = false;
    if this.directory_view_mode != DirectoryViewMode::Grid {
        this.set_directory_view_mode(DirectoryViewMode::Grid, cx);
    }
    true
}

fn ready_background_idle(this: &PictureBrowser, _: &TourContext) -> bool {
    this.work.active == 0 && !this.face_inventory_active
}
