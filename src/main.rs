// A custom test harness does not apply libtest's implicit dead-code and
// unused-import allowances to the binary under test.
#![cfg_attr(test, allow(dead_code, unused_imports))]

use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    ops::Range,
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use anyhow::Result;
use gpui::{
    AnyElement, App, AppContext as _, Application, Bounds, ClickEvent, ClipboardItem, Context,
    CursorStyle, FocusHandle, KeyBinding, ListSizingBehavior, Menu, MenuItem, MouseButton,
    MouseDownEvent, MouseMoveEvent, MouseUpEvent, PathPromptOptions, Render, ScrollStrategy,
    ScrollWheelEvent, SharedString, UniformListScrollHandle, Window, WindowBounds, WindowOptions,
    actions, canvas, div, img, point, prelude::*, px, relative, rgb, size, uniform_list,
};

mod catalog;
mod config;
mod discovery;
mod formats;
mod jobs;
mod metadata;
mod preview;
mod rendering;
mod scheduler;

use config::{AppSettings, Config};
use discovery::{
    DirectoryContents, DirectoryMetadataStatus, DirectoryModel, DirectoryScanPriority,
    DirectoryScanner, NodeId, RawJpegPreference,
};
use jobs::JobLimiter;
use metadata::MetadataLoader;
use preview::{PreviewCache, PreviewRenderer, PreviewState};
use rendering::pipeline::RenderInvocation;
use rendering::{RenderIdentity, RenderMode, RenderVariant};
use scheduler::{
    DemandScopeId, JobCompletion, JobDemand, JobKey, JobPriority, JobSnapshotStatus, PriorityClass,
    RenderScheduler, SchedulerSnapshot,
};

actions!(
    pics,
    [
        ToggleDeveloped,
        ToggleJobQueue,
        ToggleDevelopInspector,
        OpenSettings,
        ImportLightroom
    ]
);

actions!(
    directory_tree,
    [TreeUp, TreeDown, TreeLeft, TreeRight, TreeConfirm]
);
actions!(
    main_view,
    [
        OpenImage,
        CloseImage,
        GridLeft,
        GridRight,
        GridUp,
        GridDown,
        GridExtendLeft,
        GridExtendRight,
        GridExtendUp,
        GridExtendDown
    ]
);

const DEFAULT_THUMBNAIL_WIDTH: u32 = 300;
const THUMBNAIL_GAP: f32 = 12.0;
const IMAGE_SCOPE: DemandScopeId = DemandScopeId(1);
const GRID_SCOPE: DemandScopeId = DemandScopeId(2);
const NEAR_GRID_SCOPE: DemandScopeId = DemandScopeId(3);
const IDLE_PREFETCH_SCOPE: DemandScopeId = DemandScopeId(4);
const FOCUSED_PREFETCH_SCOPE: DemandScopeId = DemandScopeId(5);
const IMAGE_PREFETCH_SCOPE: DemandScopeId = DemandScopeId(6);
const IDLE_PREFETCH_DELAY: Duration = Duration::from_millis(400);
const IDLE_PREFETCH_BATCH: usize = 64;
const IMAGE_PREFETCH_COUNT: usize = 4;
const JOB_QUEUE_PANEL_WIDTH: f32 = 430.0;
const DEVELOP_PANEL_WIDTH: f32 = 320.0;
const SCHEDULER_REFRESH_INTERVAL: Duration = Duration::from_millis(100);
const USE_IN_WINDOW_MENU: bool = cfg!(target_os = "linux");
const IMAGE_WHEEL_THRESHOLD_PX: f32 = 30.0;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct ImageSelection {
    indices: BTreeSet<usize>,
    anchor: Option<usize>,
    focused: Option<usize>,
}

impl ImageSelection {
    fn select(&mut self, index: usize, extend: bool, toggle: bool) {
        if extend {
            let anchor = self.anchor.unwrap_or(index);
            self.indices = (anchor.min(index)..=anchor.max(index)).collect();
        } else if toggle {
            if !self.indices.remove(&index) {
                self.indices.insert(index);
            }
            self.anchor = Some(index);
        } else {
            self.indices.clear();
            self.indices.insert(index);
            self.anchor = Some(index);
        }
        self.focused = Some(index);
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ZoomLevel {
    FitWindow,
    ActualSize,
}

struct ImageViewState {
    path: PathBuf,
    zoom: ZoomLevel,
    pan: (f32, f32),
    drag_last: Option<(f32, f32)>,
    wheel_delta_y: f32,
}

impl ImageViewState {
    fn new(path: PathBuf) -> Self {
        Self {
            path,
            zoom: ZoomLevel::FitWindow,
            pan: (0.0, 0.0),
            drag_last: None,
            wheel_delta_y: 0.0,
        }
    }

    fn toggle_zoom(&mut self) {
        self.zoom = match self.zoom {
            ZoomLevel::FitWindow => ZoomLevel::ActualSize,
            ZoomLevel::ActualSize => ZoomLevel::FitWindow,
        };
        self.pan = (0.0, 0.0);
        self.drag_last = None;
    }
}

enum MainView {
    Image(ImageViewState),
    Dir(PathBuf, ImageSelection),
}

#[derive(Clone, Copy)]
struct VisibleRow {
    node: NodeId,
    depth: usize,
}

#[derive(Default)]
struct WorkProgress {
    active: usize,
    completed: usize,
    total: usize,
    next_id: u64,
    activities: BTreeMap<u64, String>,
}

impl WorkProgress {
    fn start(&mut self, activity: impl Into<String>) -> u64 {
        if self.active == 0 {
            self.completed = 0;
            self.total = 0;
        }
        self.active += 1;
        self.total += 1;
        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1);
        self.activities.insert(id, activity.into());
        id
    }

    fn finish(&mut self, id: u64) {
        if self.activities.remove(&id).is_none() {
            return;
        }
        self.active = self.active.saturating_sub(1);
        self.completed = (self.completed + 1).min(self.total);
    }

    fn fraction(&self) -> f32 {
        if self.total == 0 {
            0.0
        } else {
            self.completed as f32 / self.total as f32
        }
    }

    fn current_activity(&self) -> Option<&str> {
        self.activities
            .last_key_value()
            .map(|(_, label)| label.as_str())
    }
}

struct PictureBrowser {
    model: DirectoryModel,
    visible: Vec<VisibleRow>,
    expanded: HashSet<NodeId>,
    selected: NodeId,
    previews: PreviewCache,
    raw_previews: PreviewCache,
    thumbnails: PreviewCache,
    main_view: MainView,
    saved_selection: ImageSelection,
    thumbnail_width: u32,
    grid_columns: usize,
    scroll: UniformListScrollHandle,
    thumbnail_scroll: UniformListScrollHandle,
    tree_focus: FocusHandle,
    main_focus: FocusHandle,
    sidebar_width: f32,
    resizing_sidebar: bool,
    pair_preference: RawJpegPreference,
    work: WorkProgress,
    render_pipeline: PreviewRenderer,
    scheduler: RenderScheduler,
    directory_scanner: DirectoryScanner,
    grid_demand_epoch: u64,
    grid_prediction: Option<GridPredictionState>,
    scroll_direction: i8,
    idle_prefetch_batch: usize,
    prefetch_preview_variant: RenderVariant,
    focused_prefetch_epoch: u64,
    image_prefetch_epoch: u64,
    pending_grid_demand: Option<PendingGridDemand>,
    grid_publish_scheduled: bool,
    show_job_queue: bool,
    show_develop_inspector: bool,
    show_view_menu: bool,
    show_file_menu: bool,
    settings: AppSettings,
    settings_path: PathBuf,
    settings_draft: Option<AppSettings>,
    settings_status: Option<String>,
    job_queue_snapshot: SchedulerSnapshot,
    job_queue_scroll: UniformListScrollHandle,
    scheduler_refresh_scheduled: bool,
    path_context_menu: Option<PathContextMenu>,
    last_export_directory: Option<PathBuf>,
    system_clipboard: Option<arboard::Clipboard>,
    metadata_loader: MetadataLoader,
    metadata_load: Option<(NodeId, discovery::RequestId, Arc<AtomicBool>)>,
    scrollbar_drag: Option<ScrollbarDrag>,
    recipes: Option<Arc<HashMap<PathBuf, catalog::RecipeEntry>>>,
    developed: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ScrollbarTarget {
    Tree,
    Grid,
}

#[derive(Clone, Copy)]
enum SettingsNumber {
    ImageCacheSize,
    JobParallelism,
    RawParallelism,
}

#[derive(Clone, Copy, Debug)]
struct ScrollbarDrag {
    target: ScrollbarTarget,
    pointer_offset: gpui::Pixels,
}

fn has_develop_recipe(
    recipes: Option<&HashMap<PathBuf, catalog::RecipeEntry>>,
    path: &std::path::Path,
) -> bool {
    recipes.is_some_and(|recipes| recipes.contains_key(path))
}

fn path_display_name(path: &std::path::Path) -> String {
    path.file_name()
        .unwrap_or(path.as_os_str())
        .to_string_lossy()
        .into_owned()
}

fn adjacent_image_index(current: usize, image_count: usize, delta: isize) -> Option<usize> {
    if image_count == 0 {
        return None;
    }
    let target = current
        .saturating_add_signed(delta)
        .min(image_count.saturating_sub(1));
    (target != current).then_some(target)
}

fn directional_adjacent_indices(
    current: usize,
    image_count: usize,
    direction: isize,
    count: usize,
) -> Vec<usize> {
    let mut indices = Vec::with_capacity(count);
    let mut index = current;
    for _ in 0..count {
        let Some(next) = adjacent_image_index(index, image_count, direction) else {
            break;
        };
        indices.push(next);
        index = next;
    }
    indices
}

fn render_activity_label(
    path: &std::path::Path,
    variant: RenderVariant,
    mode: RenderMode,
    status: JobSnapshotStatus,
) -> String {
    let action = match status {
        JobSnapshotStatus::Running => "Rendering",
        JobSnapshotStatus::Queued | JobSnapshotStatus::PendingSuccessor => "Preparing",
    };
    let stage = match (variant, mode) {
        (RenderVariant::Full, RenderMode::Developed) => "developed image",
        (RenderVariant::Full, RenderMode::Baseline) => "full image",
        (RenderVariant::Preview { .. }, RenderMode::Developed) => "developed preview",
        (RenderVariant::Preview { .. }, RenderMode::Baseline) => "preview",
        (RenderVariant::Thumbnail { .. }, RenderMode::Developed) => "developed thumbnail",
        (RenderVariant::Thumbnail { .. }, RenderMode::Baseline) => "thumbnail",
    };
    format!("{action} {stage}: {}…", path_display_name(path))
}

fn scheduler_progress_label(snapshot: &SchedulerSnapshot) -> Option<String> {
    let jobs = snapshot
        .jobs
        .iter()
        .filter(|job| !job.cancelled)
        .collect::<Vec<_>>();
    let current = jobs
        .iter()
        .find(|job| {
            job.status == JobSnapshotStatus::Running && job.priority.class >= PriorityClass::Visible
        })
        .or_else(|| {
            jobs.iter()
                .find(|job| job.status == JobSnapshotStatus::Running)
        })
        .or_else(|| {
            jobs.iter()
                .find(|job| job.priority.class >= PriorityClass::Visible)
        })
        .or_else(|| jobs.first())?;
    let mut label = render_activity_label(
        &current.key.path,
        current.key.variant,
        current.key.render_identity.mode,
        current.status,
    );
    if current.priority.class < PriorityClass::Visible {
        label = match current.status {
            JobSnapshotStatus::Running => label.replacen("Rendering", "Pre-rendering", 1),
            JobSnapshotStatus::Queued | JobSnapshotStatus::PendingSuccessor => {
                label.replacen("Preparing", "Waiting to pre-render", 1)
            }
        };
    }
    if jobs.len() > 1 {
        label.push_str(&format!(" ({} more)", jobs.len() - 1));
    }
    Some(label)
}

fn format_develop_edits(recipe: &catalog::DevelopRecipe) -> Vec<(String, String)> {
    let mut edits = Vec::new();
    macro_rules! push_value {
        ($name:literal, $value:expr) => {
            if let Some(value) = $value {
                edits.push(($name.to_owned(), format!("{value:.2}")));
            }
        };
    }
    push_value!("Exposure", recipe.exposure_ev);
    push_value!("Contrast", recipe.contrast);
    push_value!("Highlights", recipe.highlights);
    push_value!("Shadows", recipe.shadows);
    push_value!("Whites", recipe.whites);
    push_value!("Blacks", recipe.blacks);
    push_value!("Saturation", recipe.saturation);
    push_value!("Vibrance", recipe.vibrance);
    push_value!("Clarity", recipe.clarity);
    push_value!("Texture", recipe.texture);
    push_value!("Dehaze", recipe.dehaze);
    if let Some(wb) = recipe.white_balance {
        edits.push((
            "White balance".into(),
            format!("{:.0} K, tint {:.2}", wb.temperature, wb.tint),
        ));
    }
    if let Some(wb) = recipe.white_balance_increment {
        edits.push((
            "White balance offset".into(),
            format!("temperature {:+.2}, tint {:+.2}", wb.temperature, wb.tint),
        ));
    }
    for (name, curve) in [
        ("Tone curve", recipe.tone_curve.as_ref()),
        ("Red tone curve", recipe.tone_curve_red.as_ref()),
        ("Green tone curve", recipe.tone_curve_green.as_ref()),
        ("Blue tone curve", recipe.tone_curve_blue.as_ref()),
    ] {
        if let Some(curve) = curve {
            edits.push((
                name.into(),
                curve
                    .iter()
                    .map(|point| format!("({:.3}, {:.3})", point[0], point[1]))
                    .collect::<Vec<_>>()
                    .join(", "),
            ));
        }
    }
    if let Some(crop) = recipe.crop {
        edits.push((
            "Crop".into(),
            format!(
                "L {:.3}, T {:.3}, R {:.3}, B {:.3}; angle {:.2}°{}",
                crop.left,
                crop.top,
                crop.right,
                crop.bottom,
                crop.angle_degrees,
                if crop.constrain_aspect {
                    "; constrained"
                } else {
                    ""
                }
            ),
        ));
    }
    if let Some(detail) = recipe.detail {
        for (name, value) in [
            ("Sharpening", detail.sharpening),
            ("Sharpen radius", detail.sharpen_radius),
            ("Sharpen detail", detail.sharpen_detail),
            ("Edge masking", detail.edge_masking),
            (
                "Luminance noise reduction",
                detail.luminance_noise_reduction,
            ),
            ("Luminance detail", detail.luminance_detail),
            ("Luminance contrast", detail.luminance_contrast),
            ("Colour noise reduction", detail.color_noise_reduction),
            ("Colour detail", detail.color_detail),
            ("Colour smoothness", detail.color_smoothness),
        ] {
            edits.push((name.into(), format!("{value:.2}")));
        }
    }
    if let Some(monochrome) = recipe.monochrome {
        for (name, value) in [
            "Red", "Orange", "Yellow", "Green", "Aqua", "Blue", "Purple", "Magenta",
        ]
        .into_iter()
        .zip(monochrome.mixer)
        {
            edits.push((format!("Monochrome {name}"), format!("{value:.2}")));
        }
    }
    if let Some(vignette) = recipe.vignette {
        edits.extend([
            ("Vignette amount".into(), format!("{:.2}", vignette.amount)),
            (
                "Vignette midpoint".into(),
                format!("{:.2}", vignette.midpoint),
            ),
            (
                "Vignette feather".into(),
                format!("{:.2}", vignette.feather),
            ),
            (
                "Vignette roundness".into(),
                format!("{:.2}", vignette.roundness),
            ),
        ]);
    }
    if let Some(curve) = recipe.parametric_curve {
        edits.push((
            "Parametric curve".into(),
            format!(
                "H {:.2}, L {:.2}, D {:.2}, S {:.2}",
                curve[0], curve[1], curve[2], curve[3]
            ),
        ));
    }
    if let Some(matrix) = recipe.upright_homography {
        edits.push((
            "Upright homography".into(),
            matrix.map(|value| format!("{value:.3}")).join(", "),
        ));
    }
    push_value!("Manual lens distortion", recipe.lens_distortion);
    for (index, spot) in recipe.heal_spots.iter().enumerate() {
        edits.push((
            format!("{:?} spot {}", spot.kind, index + 1),
            format!(
                "to {:.3},{:.3} from {:.3},{:.3}; radius {:.3}, opacity {:.0}%, feather {:.0}%",
                spot.destination[0],
                spot.destination[1],
                spot.source[0],
                spot.source[1],
                spot.radius,
                spot.opacity * 100.0,
                spot.feather * 100.0
            ),
        ));
    }
    if let Some(grain) = recipe.grain {
        edits.push((
            "Grain".into(),
            format!(
                "amount {:.2}, size {:.2}, roughness {:.2}",
                grain.amount, grain.size, grain.roughness
            ),
        ));
    }
    edits
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct GridPredictionState {
    visible_rows: Range<usize>,
    columns: usize,
    thumbnail_width: u32,
    preview_variant: RenderVariant,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct PendingGridDemand {
    node: NodeId,
    visible_rows: Range<usize>,
    columns: usize,
    thumbnail_width: u32,
    preview_variant: RenderVariant,
}

struct PathContextMenu {
    path: PathBuf,
    position: (f32, f32),
    exportable: bool,
}

fn merge_grid_demand(slot: &mut Option<PendingGridDemand>, next: PendingGridDemand) {
    match slot {
        Some(pending)
            if pending.node == next.node
                && pending.columns == next.columns
                && pending.thumbnail_width == next.thumbnail_width
                && pending.preview_variant == next.preview_variant
                && pending.visible_rows.start <= next.visible_rows.end
                && next.visible_rows.start <= pending.visible_rows.end =>
        {
            pending.visible_rows.start = pending.visible_rows.start.min(next.visible_rows.start);
            pending.visible_rows.end = pending.visible_rows.end.max(next.visible_rows.end);
        }
        _ => *slot = Some(next),
    }
}

impl PictureBrowser {
    fn raw_preview_variant(window: &Window, cx: &Context<Self>) -> RenderVariant {
        let display_size = window
            .display(cx)
            .map(|display| display.bounds().size)
            .unwrap_or_else(|| window.viewport_size());
        let scale = window.scale_factor();
        RenderVariant::Preview {
            max_width: (f32::from(display_size.width) * scale).ceil().max(1.0) as u32,
            max_height: (f32::from(display_size.height) * scale).ceil().max(1.0) as u32,
        }
    }

    fn new(config: &Config, cx: &mut Context<Self>) -> Self {
        let settings = &config.settings;
        let model = DirectoryModel::new(settings.root.clone());
        let root = model.root();
        let root_path = model.node(root).path.to_path_buf();
        let job_limiter = JobLimiter::new(settings.job_parallelism);
        let render_pipeline = PreviewRenderer::with_source_cache_capacity(
            settings.cache_root.clone(),
            settings.job_parallelism,
            settings.raw_parallelism,
            settings.image_cache_size,
        );
        let metadata_loader = MetadataLoader::new(
            settings.cache_root.clone(),
            settings.job_parallelism,
            job_limiter.clone(),
        );
        let render_job_limiter = job_limiter.clone();
        let scheduler_pipeline = render_pipeline.clone();
        let scheduler = RenderScheduler::new(
            settings.job_parallelism,
            Arc::new(move |key, priority, cancelled| {
                let Some(_permit) = render_job_limiter.acquire(
                    priority.class >= scheduler::PriorityClass::Visible,
                    cancelled,
                ) else {
                    return PreviewState::Error("render cancelled".into());
                };
                scheduler_pipeline.render_with_recipe(
                    RenderInvocation {
                        path: key.path.clone(),
                        variant: key.variant,
                        render_identity: key.render_identity,
                        recipe: key.recipe.as_deref(),
                        interactive: priority.class == scheduler::PriorityClass::Interactive,
                        durable_cache_write: priority.class < scheduler::PriorityClass::Visible,
                    },
                    cancelled,
                )
            }),
        );
        let mut this = Self {
            model,
            visible: vec![VisibleRow {
                node: root,
                depth: 0,
            }],
            expanded: HashSet::from([root]),
            selected: root,
            previews: PreviewCache::new(settings.image_cache_size),
            raw_previews: PreviewCache::new(settings.image_cache_size),
            thumbnails: PreviewCache::new(settings.image_cache_size.saturating_mul(16).max(64)),
            main_view: MainView::Dir(root_path, ImageSelection::default()),
            saved_selection: ImageSelection::default(),
            thumbnail_width: DEFAULT_THUMBNAIL_WIDTH,
            grid_columns: 1,
            scroll: UniformListScrollHandle::new(),
            thumbnail_scroll: UniformListScrollHandle::new(),
            tree_focus: cx.focus_handle(),
            main_focus: cx.focus_handle(),
            sidebar_width: 300.0,
            resizing_sidebar: false,
            pair_preference: settings.pair_preference.into(),
            work: WorkProgress::default(),
            render_pipeline,
            scheduler,
            directory_scanner: DirectoryScanner::new(settings.job_parallelism, job_limiter.clone()),
            grid_demand_epoch: 0,
            grid_prediction: None,
            scroll_direction: 1,
            idle_prefetch_batch: 0,
            prefetch_preview_variant: RenderVariant::Preview {
                max_width: 1920,
                max_height: 1080,
            },
            focused_prefetch_epoch: 0,
            image_prefetch_epoch: 0,
            pending_grid_demand: None,
            grid_publish_scheduled: false,
            show_job_queue: false,
            show_develop_inspector: false,
            show_view_menu: false,
            show_file_menu: false,
            settings: settings.clone(),
            settings_path: config.settings_path.clone(),
            settings_draft: None,
            settings_status: None,
            job_queue_snapshot: SchedulerSnapshot::default(),
            job_queue_scroll: UniformListScrollHandle::new(),
            scheduler_refresh_scheduled: false,
            path_context_menu: None,
            last_export_directory: None,
            system_clipboard: None,
            metadata_loader,
            metadata_load: None,
            scrollbar_drag: None,
            recipes: None,
            developed: true,
        };
        let catalog_work = this.work.start("Loading develop catalog…");
        let catalog_root = settings.root.clone();
        let loading = cx
            .background_executor()
            .spawn(async move { catalog::load_recipes(&catalog_root) });
        cx.spawn(async move |this, cx| {
            let result = loading.await;
            this.update(cx, |this, cx| {
                this.work.finish(catalog_work);
                this.recipes = Some(Arc::new(result.unwrap_or_else(|error| {
                    eprintln!("cannot load Pics catalog: {error:#}");
                    HashMap::new()
                })));
                cx.notify();
            })
            .ok();
        })
        .detach();
        this.ensure_directory_loaded(root, DirectoryScanPriority::Interactive, cx);
        this
    }

    fn ensure_directory_loaded(
        &mut self,
        node: NodeId,
        priority: DirectoryScanPriority,
        cx: &mut Context<Self>,
    ) {
        let node_path = self.model.node(node).path.clone();
        let Some((request, path)) = self.model.begin_load(node) else {
            self.directory_scanner.promote(&node_path, priority);
            return;
        };
        let scan_work = self
            .work
            .start(format!("Scanning {}…", path_display_name(&path)));
        cx.notify();
        let pair_preference = self.pair_preference;
        let scanner = self.directory_scanner.clone();
        let metadata_loader = self.metadata_loader.clone();
        let scan = cx.background_spawn(async move {
            let scan = scanner.scan(&path, pair_preference, priority)?;
            let (cached_metadata, missing_metadata) = metadata_loader
                .load_cached_directory(scan.images.clone(), Arc::new(AtomicBool::new(false)));
            Ok::<_, std::io::Error>((scan, cached_metadata, missing_metadata))
        });
        cx.spawn(async move |this, cx| {
            let result = scan.await;
            this.update(cx, |this, cx| {
                this.work.finish(scan_work);
                let loaded = match result {
                    Ok((scan, cached_metadata, missing_metadata)) => {
                        let loaded = this.model.finish_load(node, request, Ok(scan));
                        if loaded {
                            this.model.apply_cached_metadata(
                                node,
                                cached_metadata,
                                missing_metadata,
                            );
                        }
                        loaded
                    }
                    Err(error) => this.model.finish_load(node, request, Err(error)),
                };
                if loaded {
                    if this
                        .model
                        .loaded(node)
                        .is_some_and(|contents| contents.children.is_empty())
                    {
                        this.expanded.remove(&node);
                    }
                    this.rebuild_visible();
                    if node == this.selected {
                        this.start_metadata_enrichment(node, cx);
                    }
                    cx.notify();
                }
            })
            .ok();
        })
        .detach();
    }

    fn cancel_metadata_enrichment(&mut self) {
        if let Some((node, request, cancelled)) = self.metadata_load.take() {
            cancelled.store(true, Ordering::Release);
            self.model.cancel_metadata_load(node, request);
        }
    }

    fn start_metadata_enrichment(&mut self, node: NodeId, cx: &mut Context<Self>) {
        if self
            .metadata_load
            .as_ref()
            .is_some_and(|(active, _, _)| *active == node)
        {
            return;
        }
        self.cancel_metadata_enrichment();
        let Some((request, paths)) = self.model.begin_metadata_load(node) else {
            return;
        };
        let cancelled = Arc::new(AtomicBool::new(false));
        self.metadata_load = Some((node, request, cancelled.clone()));
        let metadata_work = self
            .work
            .start(format!("Reading metadata for {} images…", paths.len()));
        cx.notify();
        let loader = self.metadata_loader.clone();
        let task_cancelled = cancelled.clone();
        let load = cx.background_spawn(async move { loader.load_directory(paths, task_cancelled) });
        cx.spawn(async move |this, cx| {
            let metadata = load.await;
            this.update(cx, |this, cx| {
                this.work.finish(metadata_work);
                let is_active =
                    this.metadata_load
                        .as_ref()
                        .is_some_and(|(active_node, active_request, _)| {
                            *active_node == node && *active_request == request
                        });
                if !is_active {
                    cx.notify();
                    return;
                }
                this.metadata_load = None;
                if cancelled.load(Ordering::Acquire) {
                    this.model.cancel_metadata_load(node, request);
                } else {
                    this.model.finish_metadata_load(node, request, metadata);
                }
                cx.notify();
            })
            .ok();
        })
        .detach();
    }

    fn apply_capture_time_order(&mut self, cx: &mut Context<Self>) {
        let node = self.selected;
        let images = self
            .model
            .loaded(node)
            .map(|contents| contents.images.as_slice())
            .unwrap_or_default();
        let selection = match &self.main_view {
            MainView::Dir(_, selection) => selection,
            MainView::Image(_) => &self.saved_selection,
        };
        let selected = selection
            .indices
            .iter()
            .filter_map(|index| images.get(*index).cloned())
            .collect::<HashSet<_>>();
        let focused = selection
            .focused
            .and_then(|index| images.get(index).cloned());
        if !self.model.apply_pending_capture_order(node) {
            return;
        }
        let images = self
            .model
            .loaded(node)
            .map(|contents| contents.images.as_slice())
            .unwrap_or_default();
        let mut restored = ImageSelection::default();
        for (index, path) in images.iter().enumerate() {
            if selected.contains(path) {
                restored.indices.insert(index);
            }
            if focused.as_ref() == Some(path) {
                restored.focused = Some(index);
                restored.anchor = Some(index);
            }
        }
        self.saved_selection = restored.clone();
        if let MainView::Dir(_, selection) = &mut self.main_view {
            *selection = restored;
        }
        self.reset_grid_prediction();
        cx.notify();
    }

    fn render_scrollbar(
        &self,
        id: &'static str,
        target: ScrollbarTarget,
        handle: &UniformListScrollHandle,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        const MIN_THUMB_HEIGHT: f32 = 28.0;
        let base_handle = handle.0.borrow().base_handle.clone();
        let viewport = base_handle.bounds();
        let viewport_height = viewport.size.height;
        let max_offset = base_handle.max_offset().height;
        if viewport_height <= px(0.0) || max_offset <= px(0.0) {
            return div().id(id).into_any_element();
        }
        let content_height = viewport_height + max_offset;
        let thumb_height = (viewport_height * (viewport_height / content_height))
            .clamp(px(MIN_THUMB_HEIGHT), viewport_height);
        let travel = (viewport_height - thumb_height).max(px(0.0));
        let scroll_offset = (-base_handle.offset().y).clamp(px(0.0), max_offset);
        let thumb_top = if max_offset > px(0.0) {
            travel * (scroll_offset / max_offset)
        } else {
            px(0.0)
        };
        let entity = cx.entity();
        div()
            .id(id)
            .absolute()
            .top(thumb_top)
            .right_1()
            .w(px(8.0))
            .h(thumb_height)
            .rounded_lg()
            .bg(rgb(0x596273))
            .hover(|this| this.bg(rgb(0x788397)))
            .child(
                canvas(
                    |_, _, _| (),
                    move |thumb_bounds, _, window, _| {
                        window.on_mouse_event({
                            let entity = entity.clone();
                            move |event: &MouseDownEvent, _, _, cx| {
                                if !thumb_bounds.contains(&event.position) {
                                    return;
                                }
                                entity.update(cx, |this, _| {
                                    this.scrollbar_drag = Some(ScrollbarDrag {
                                        target,
                                        pointer_offset: event.position.y - thumb_bounds.origin.y,
                                    });
                                });
                            }
                        });
                        window.on_mouse_event({
                            let entity = entity.clone();
                            move |_: &MouseUpEvent, _, _, cx| {
                                entity.update(cx, |this, _| {
                                    if this
                                        .scrollbar_drag
                                        .is_some_and(|drag| drag.target == target)
                                    {
                                        this.scrollbar_drag = None;
                                    }
                                });
                            }
                        });
                        window.on_mouse_event(move |event: &MouseMoveEvent, _, _, cx| {
                            if !event.dragging() {
                                return;
                            }
                            let Some(drag) = entity
                                .read(cx)
                                .scrollbar_drag
                                .filter(|drag| drag.target == target)
                            else {
                                return;
                            };
                            let thumb_top =
                                (event.position.y - viewport.origin.y - drag.pointer_offset)
                                    .clamp(px(0.0), travel);
                            let offset = if travel > px(0.0) {
                                max_offset * (thumb_top / travel)
                            } else {
                                px(0.0)
                            };
                            base_handle.set_offset(point(px(0.0), -offset));
                            cx.notify(entity.entity_id());
                        });
                    },
                )
                .size_full(),
            )
            .into_any_element()
    }

    fn cache_for_variant(&self, variant: RenderVariant) -> &PreviewCache {
        match variant {
            RenderVariant::Full => &self.previews,
            RenderVariant::Thumbnail { .. } => &self.thumbnails,
            RenderVariant::Preview { .. } => &self.raw_previews,
        }
    }

    fn cache_for_variant_mut(&mut self, variant: RenderVariant) -> &mut PreviewCache {
        match variant {
            RenderVariant::Full => &mut self.previews,
            RenderVariant::Thumbnail { .. } => &mut self.thumbnails,
            RenderVariant::Preview { .. } => &mut self.raw_previews,
        }
    }

    fn render_identity_for(&self, path: &std::path::Path) -> RenderIdentity {
        self.developed
            .then(|| self.recipes.as_ref()?.get(path))
            .flatten()
            .map(|entry| RenderIdentity {
                mode: RenderMode::Developed,
                recipe_digest: entry.digest,
                renderer_model: catalog::RENDERER_MODEL_VERSION,
            })
            .unwrap_or_default()
    }

    fn replace_render_scope(
        &mut self,
        scope: DemandScopeId,
        requested: Vec<(PathBuf, RenderVariant, JobPriority)>,
        cx: &mut Context<Self>,
    ) {
        self.replace_render_scope_internal(scope, requested, true, cx);
    }

    fn replace_render_scope_internal(
        &mut self,
        scope: DemandScopeId,
        requested: Vec<(PathBuf, RenderVariant, JobPriority)>,
        track_progress: bool,
        cx: &mut Context<Self>,
    ) {
        let mut requests = HashMap::new();
        let mut demands = Vec::new();
        let Some(recipes) = self.recipes.clone() else {
            return;
        };
        for (path, variant, priority) in requested {
            let render_identity = self.render_identity_for(&path);
            if !self
                .cache_for_variant(variant)
                .needs_work_for(&path, variant, render_identity)
            {
                continue;
            }
            let request =
                self.cache_for_variant_mut(variant)
                    .begin_for(&path, variant, render_identity);
            let recipe = self
                .developed
                .then(|| recipes.get(&path))
                .flatten()
                .map(|entry| Arc::new(entry.recipe.clone()));
            let key = JobKey {
                path,
                variant,
                render_identity,
                recipe,
            };
            if let Some(request) = request {
                requests.insert(key.clone(), request);
            }
            demands.push(JobDemand {
                key,
                priority,
                subscribe: request.is_some(),
            });
        }

        let subscriptions = self.scheduler.replace_scope(scope, demands);
        let mut progress_started = false;
        for subscription in subscriptions {
            let Some(request) = requests.remove(&subscription.key) else {
                continue;
            };
            let JobKey {
                path,
                variant,
                render_identity,
                ..
            } = subscription.key;
            let render_work = track_progress.then(|| {
                self.work.start(render_activity_label(
                    &path,
                    variant,
                    render_identity.mode,
                    JobSnapshotStatus::Running,
                ))
            });
            if track_progress {
                progress_started = true;
            }
            cx.spawn(async move |this, cx| {
                let completion = subscription.receiver.await;
                this.update(cx, |this, cx| {
                    if let Some(render_work) = render_work {
                        this.work.finish(render_work);
                    }
                    match completion {
                        Ok(JobCompletion::Finished(result)) => {
                            if let PreviewState::Loaded(preview) = &result {
                                this.model
                                    .record_render_metadata(path.clone(), preview.metadata.clone());
                            }
                            this.cache_for_variant_mut(variant).finish_for(
                                &path,
                                render_identity,
                                request,
                                result,
                            );
                        }
                        Ok(JobCompletion::Cancelled) | Err(_) => {
                            this.cache_for_variant_mut(variant).cancel_for(
                                &path,
                                render_identity,
                                request,
                            );
                        }
                    }
                    cx.notify();
                })
                .ok();
            })
            .detach();
        }
        if progress_started {
            // Grid demand is commonly published from a deferred callback. GPUI
            // does not render mutations made there until explicitly notified;
            // without this, fast jobs can finish before the active state is
            // ever painted and the progress bar remains invisible.
            cx.notify();
        }
    }

    fn replace_background_scope(
        &mut self,
        scope: DemandScopeId,
        requested: Vec<(PathBuf, RenderVariant, JobPriority)>,
        cx: &mut Context<Self>,
    ) {
        self.replace_render_scope_internal(scope, requested, false, cx);
    }

    fn reset_grid_prediction(&mut self) {
        self.grid_demand_epoch = self.grid_demand_epoch.wrapping_add(1);
        self.focused_prefetch_epoch = self.focused_prefetch_epoch.wrapping_add(1);
        self.image_prefetch_epoch = self.image_prefetch_epoch.wrapping_add(1);
        self.grid_prediction = None;
        self.idle_prefetch_batch = 0;
        self.scheduler.clear_scope(NEAR_GRID_SCOPE);
        self.scheduler.clear_scope(IDLE_PREFETCH_SCOPE);
        self.scheduler.clear_scope(FOCUSED_PREFETCH_SCOPE);
        self.scheduler.clear_scope(IMAGE_PREFETCH_SCOPE);
        self.pending_grid_demand = None;
    }

    fn record_grid_demand(
        &mut self,
        visible_rows: Range<usize>,
        preview_variant: RenderVariant,
        window: &Window,
        cx: &mut Context<Self>,
    ) {
        let next = PendingGridDemand {
            node: self.selected,
            visible_rows,
            columns: self.grid_columns,
            thumbnail_width: self.thumbnail_width,
            preview_variant,
        };
        merge_grid_demand(&mut self.pending_grid_demand, next);
        if self.grid_publish_scheduled {
            return;
        }
        self.grid_publish_scheduled = true;
        cx.defer_in(window, |this, _, cx| {
            this.grid_publish_scheduled = false;
            let Some(pending) = this.pending_grid_demand.take() else {
                return;
            };
            if pending.node != this.selected || !matches!(this.main_view, MainView::Dir(_, _)) {
                return;
            }
            this.publish_grid_demand(pending, cx);
        });
    }

    fn publish_grid_demand(&mut self, pending: PendingGridDemand, cx: &mut Context<Self>) {
        let visible = visible_thumbnail_demands(
            self.selected_images(),
            pending.visible_rows.clone(),
            pending.columns,
            pending.thumbnail_width,
        );
        // Publish visible work first so a worker can never race ahead with
        // overscan merely because prediction was calculated first.
        self.replace_render_scope(GRID_SCOPE, visible, cx);
        self.update_grid_prediction(
            pending.visible_rows,
            pending.columns,
            pending.thumbnail_width,
            pending.preview_variant,
            cx,
        );
    }

    fn update_grid_prediction(
        &mut self,
        visible_rows: Range<usize>,
        columns: usize,
        thumbnail_width: u32,
        preview_variant: RenderVariant,
        cx: &mut Context<Self>,
    ) {
        let prediction = GridPredictionState {
            visible_rows: visible_rows.clone(),
            columns,
            thumbnail_width,
            preview_variant,
        };
        if self.grid_prediction.as_ref() == Some(&prediction) {
            return;
        }
        if let Some(previous) = &self.grid_prediction {
            self.scroll_direction = if visible_rows.start >= previous.visible_rows.start {
                1
            } else {
                -1
            };
        }
        let before = if self.scroll_direction < 0 { 3 } else { 1 };
        let after = if self.scroll_direction > 0 { 3 } else { 1 };
        let near_start = visible_rows.start.saturating_sub(before);
        let near_end = visible_rows.end.saturating_add(after);
        let mut near = Vec::new();
        let images = self.selected_images();
        for row in near_start..near_end {
            if visible_rows.contains(&row) {
                continue;
            }
            let distance = if row < visible_rows.start {
                visible_rows.start - row
            } else {
                row - visible_rows.end + 1
            };
            near.extend(
                visible_thumbnail_demands(
                    images,
                    row..row.saturating_add(1),
                    columns,
                    thumbnail_width,
                )
                .into_iter()
                .map(|(path, variant, _)| {
                    (
                        path,
                        variant,
                        JobPriority::near_viewport(-(distance.min(i32::MAX as usize) as i32)),
                    )
                }),
            );
        }
        self.grid_prediction = Some(prediction);
        self.prefetch_preview_variant = preview_variant;
        self.grid_demand_epoch = self.grid_demand_epoch.wrapping_add(1);
        self.idle_prefetch_batch = 0;
        self.scheduler.clear_scope(IDLE_PREFETCH_SCOPE);
        self.replace_background_scope(NEAR_GRID_SCOPE, near, cx);
        self.schedule_idle_prefetch(self.grid_demand_epoch, cx);
        self.schedule_focused_prefetch(cx);
    }

    fn schedule_idle_prefetch(&self, epoch: u64, cx: &mut Context<Self>) {
        let timer = cx.background_executor().timer(IDLE_PREFETCH_DELAY);
        cx.spawn(async move |this, cx| {
            timer.await;
            this.update(cx, |this, cx| this.run_idle_prefetch(epoch, cx))
                .ok();
        })
        .detach();
    }

    fn run_idle_prefetch(&mut self, epoch: u64, cx: &mut Context<Self>) {
        if epoch != self.grid_demand_epoch || !matches!(self.main_view, MainView::Dir(_, _)) {
            return;
        }
        let stats = self.scheduler.stats();
        if self.scheduler.has_foreground_work() || stats.queued > 0 || stats.running > 0 {
            self.schedule_idle_prefetch(epoch, cx);
            return;
        }
        let Some(prediction) = self.grid_prediction.clone() else {
            return;
        };
        let image_count = self.selected_images().len();
        let first = prediction
            .visible_rows
            .start
            .saturating_mul(prediction.columns)
            .min(image_count);
        let last = prediction
            .visible_rows
            .end
            .saturating_mul(prediction.columns)
            .min(image_count);
        let indices = speculative_indices(
            image_count,
            first..last,
            self.scroll_direction,
            self.idle_prefetch_batch,
            IDLE_PREFETCH_BATCH,
        );
        if indices.is_empty() {
            self.scheduler.clear_scope(IDLE_PREFETCH_SCOPE);
            return;
        }
        let thumbnail_variant = RenderVariant::Thumbnail {
            width: prediction.thumbnail_width,
        };
        let demands: Vec<_> = indices
            .into_iter()
            .enumerate()
            .map(|(rank, index)| {
                (
                    self.selected_images()[index].clone(),
                    thumbnail_variant,
                    JobPriority::speculative(-(rank.min(i32::MAX as usize) as i32)),
                )
            })
            .collect();
        self.replace_background_scope(IDLE_PREFETCH_SCOPE, demands, cx);
        self.idle_prefetch_batch += 1;
        self.schedule_idle_prefetch(epoch, cx);
    }

    fn schedule_focused_prefetch(&mut self, cx: &mut Context<Self>) {
        self.focused_prefetch_epoch = self.focused_prefetch_epoch.wrapping_add(1);
        let epoch = self.focused_prefetch_epoch;
        self.scheduler.clear_scope(FOCUSED_PREFETCH_SCOPE);
        self.schedule_focused_prefetch_epoch(epoch, cx);
    }

    fn schedule_focused_prefetch_epoch(&self, epoch: u64, cx: &mut Context<Self>) {
        let timer = cx.background_executor().timer(IDLE_PREFETCH_DELAY);
        cx.spawn(async move |this, cx| {
            timer.await;
            this.update(cx, |this, cx| {
                if epoch != this.focused_prefetch_epoch {
                    return;
                }
                if this.scheduler.has_foreground_work() {
                    this.schedule_focused_prefetch_epoch(epoch, cx);
                    return;
                }
                let focused = match &this.main_view {
                    MainView::Dir(_, selection) => selection.focused,
                    MainView::Image(_) => None,
                };
                let Some(focused) = focused else {
                    return;
                };
                let images = this.selected_images();
                let demands = [
                    focused.saturating_sub(1),
                    focused,
                    focused.saturating_add(1),
                ]
                .into_iter()
                .filter_map(|index| images.get(index))
                .filter(|path| formats::is_raw(path))
                .map(|path| {
                    (
                        path.clone(),
                        this.prefetch_preview_variant,
                        JobPriority::speculative(0),
                    )
                })
                .collect();
                this.replace_background_scope(FOCUSED_PREFETCH_SCOPE, demands, cx);
            })
            .ok();
        })
        .detach();
    }

    fn schedule_image_prefetch(
        &mut self,
        current_path: PathBuf,
        direction: isize,
        preview_variant: RenderVariant,
        cx: &mut Context<Self>,
    ) {
        self.image_prefetch_epoch = self.image_prefetch_epoch.wrapping_add(1);
        let epoch = self.image_prefetch_epoch;
        self.scheduler.clear_scope(IMAGE_PREFETCH_SCOPE);
        self.schedule_image_prefetch_epoch(epoch, current_path, direction, preview_variant, cx);
    }

    fn schedule_image_prefetch_epoch(
        &self,
        epoch: u64,
        current_path: PathBuf,
        direction: isize,
        preview_variant: RenderVariant,
        cx: &mut Context<Self>,
    ) {
        let timer = cx.background_executor().timer(IDLE_PREFETCH_DELAY);
        cx.spawn(async move |this, cx| {
            timer.await;
            this.update(cx, |this, cx| {
                if epoch != this.image_prefetch_epoch
                    || !matches!(this.main_view, MainView::Image(_))
                {
                    return;
                }
                if this.scheduler.has_foreground_work() {
                    this.schedule_image_prefetch_epoch(
                        epoch,
                        current_path,
                        direction,
                        preview_variant,
                        cx,
                    );
                    return;
                }
                let images = this.selected_images();
                let MainView::Image(state) = &this.main_view else {
                    return;
                };
                if state.path != current_path {
                    return;
                }
                let Some(current) = images.iter().position(|path| path == &current_path) else {
                    return;
                };
                let demands = directional_adjacent_indices(
                    current,
                    images.len(),
                    direction,
                    IMAGE_PREFETCH_COUNT,
                )
                .into_iter()
                .enumerate()
                .map(|(rank, index)| {
                    let path = images[index].clone();
                    let variant = if formats::is_raw(&path) {
                        preview_variant
                    } else {
                        RenderVariant::Full
                    };
                    (
                        path,
                        variant,
                        JobPriority::speculative(-(rank.min(i32::MAX as usize) as i32)),
                    )
                })
                .collect();
                this.replace_background_scope(IMAGE_PREFETCH_SCOPE, demands, cx);
            })
            .ok();
        })
        .detach();
    }

    fn select(&mut self, node: NodeId, cx: &mut Context<Self>) {
        let directory_changed = self.selected != node;
        self.scheduler.clear_scope(GRID_SCOPE);
        self.scheduler.clear_scope(IMAGE_SCOPE);
        self.reset_grid_prediction();
        self.cancel_metadata_enrichment();
        self.selected = node;
        self.main_view = MainView::Dir(
            self.model.node(node).path.to_path_buf(),
            ImageSelection::default(),
        );
        if directory_changed {
            self.thumbnail_scroll
                .scroll_to_item_strict(0, ScrollStrategy::Top);
        }
        self.saved_selection = ImageSelection::default();
        self.ensure_directory_loaded(node, DirectoryScanPriority::Interactive, cx);
        self.start_metadata_enrichment(node, cx);
        cx.notify();
    }

    fn select_row(&mut self, index: usize, strategy: ScrollStrategy, cx: &mut Context<Self>) {
        if let Some(row) = self.visible.get(index) {
            self.scroll.scroll_to_item(index, strategy);
            self.select(row.node, cx);
        }
    }

    fn selected_row(&self) -> usize {
        self.visible
            .iter()
            .position(|row| row.node == self.selected)
            .unwrap_or(0)
    }

    fn tree_up(&mut self, _: &TreeUp, _: &mut Window, cx: &mut Context<Self>) {
        let index = self.selected_row().saturating_sub(1);
        self.select_row(index, ScrollStrategy::Top, cx);
    }

    fn tree_down(&mut self, _: &TreeDown, _: &mut Window, cx: &mut Context<Self>) {
        let index = (self.selected_row() + 1).min(self.visible.len().saturating_sub(1));
        self.select_row(index, ScrollStrategy::Bottom, cx);
    }

    fn tree_left(&mut self, _: &TreeLeft, _: &mut Window, cx: &mut Context<Self>) {
        if self.expanded.contains(&self.selected) {
            self.toggle(self.selected, cx);
            return;
        }
        let index = self.selected_row();
        let depth = self.visible[index].depth;
        if let Some(parent) = self.visible[..index]
            .iter()
            .rposition(|row| row.depth < depth)
        {
            self.select_row(parent, ScrollStrategy::Top, cx);
        }
    }

    fn tree_right(&mut self, _: &TreeRight, _: &mut Window, cx: &mut Context<Self>) {
        if !self.expanded.contains(&self.selected) {
            self.toggle(self.selected, cx);
            return;
        }
        let index = self.selected_row();
        if self
            .visible
            .get(index + 1)
            .is_some_and(|row| row.depth > self.visible[index].depth)
        {
            self.select_row(index + 1, ScrollStrategy::Bottom, cx);
        }
    }

    fn tree_confirm(&mut self, _: &TreeConfirm, _: &mut Window, cx: &mut Context<Self>) {
        self.select(self.selected, cx);
    }

    fn resize_sidebar(&mut self, event: &MouseMoveEvent, cx: &mut Context<Self>) {
        if self.resizing_sidebar && event.dragging() {
            self.sidebar_width = f32::from(event.position.x).clamp(160.0, 640.0);
            cx.notify();
        }
    }

    fn toggle(&mut self, node: NodeId, cx: &mut Context<Self>) {
        if !self.expanded.remove(&node) {
            self.expanded.insert(node);
            self.ensure_directory_loaded(node, DirectoryScanPriority::Background, cx);
        }
        self.rebuild_visible();
        cx.notify();
    }

    fn open(&mut self, node: NodeId, cx: &mut Context<Self>) {
        if self
            .model
            .loaded(node)
            .is_some_and(|contents| contents.children.is_empty())
        {
            return;
        }
        if self.expanded.insert(node) {
            self.ensure_directory_loaded(node, DirectoryScanPriority::Background, cx);
            self.rebuild_visible();
            cx.notify();
        }
    }

    fn select_thumbnail(
        &mut self,
        index: usize,
        extend: bool,
        toggle: bool,
        cx: &mut Context<Self>,
    ) {
        let MainView::Dir(_, selection) = &mut self.main_view else {
            return;
        };
        selection.select(index, extend, toggle);
        self.saved_selection = selection.clone();
        self.schedule_focused_prefetch(cx);
    }

    fn show_image(&mut self, path: PathBuf, cx: &mut Context<Self>) {
        if let MainView::Dir(_, selection) = &self.main_view {
            self.saved_selection = selection.clone();
        }
        // Full-screen navigation owns background prediction. Invalidate grid
        // timers and remove their speculative work so the job list and worker
        // budget contain only relevant adjacent images.
        self.grid_demand_epoch = self.grid_demand_epoch.wrapping_add(1);
        self.focused_prefetch_epoch = self.focused_prefetch_epoch.wrapping_add(1);
        self.scheduler.clear_scope(NEAR_GRID_SCOPE);
        self.scheduler.clear_scope(IDLE_PREFETCH_SCOPE);
        self.scheduler.clear_scope(FOCUSED_PREFETCH_SCOPE);
        self.main_view = MainView::Image(ImageViewState::new(path.clone()));
        // Keep visible thumbnail demand alive. Returning to the grid can then
        // resume without rebuilding its visible cache entries.
        let identity = self.render_identity_for(&path);
        self.previews.mark_used_for(&path, identity);
        cx.notify();
    }

    fn open_selected_image(&mut self, _: &OpenImage, _: &mut Window, cx: &mut Context<Self>) {
        let index = match &self.main_view {
            MainView::Dir(_, selection) => selection.focused,
            MainView::Image(_) => None,
        };
        if let Some(path) = index.and_then(|index| self.selected_images().get(index).cloned()) {
            self.show_image(path, cx);
        }
    }

    fn move_thumbnail_selection(&mut self, delta: isize, extend: bool, cx: &mut Context<Self>) {
        let image_count = self.selected_images().len();
        if image_count == 0 {
            return;
        }
        let current = match &self.main_view {
            MainView::Dir(_, selection) => selection.focused.unwrap_or(0),
            MainView::Image(_) => return,
        };
        let target = current
            .saturating_add_signed(delta)
            .min(image_count.saturating_sub(1));
        self.select_thumbnail(target, extend, false, cx);
        self.thumbnail_scroll
            .scroll_to_item(target / self.grid_columns, ScrollStrategy::Center);
        cx.notify();
    }

    fn grid_left(&mut self, _: &GridLeft, _: &mut Window, cx: &mut Context<Self>) {
        self.move_thumbnail_selection(-1, false, cx);
    }

    fn grid_right(&mut self, _: &GridRight, _: &mut Window, cx: &mut Context<Self>) {
        self.move_thumbnail_selection(1, false, cx);
    }

    fn grid_up(&mut self, _: &GridUp, _: &mut Window, cx: &mut Context<Self>) {
        self.move_thumbnail_selection(-(self.grid_columns as isize), false, cx);
    }

    fn grid_down(&mut self, _: &GridDown, _: &mut Window, cx: &mut Context<Self>) {
        self.move_thumbnail_selection(self.grid_columns as isize, false, cx);
    }

    fn grid_extend_left(&mut self, _: &GridExtendLeft, _: &mut Window, cx: &mut Context<Self>) {
        self.move_thumbnail_selection(-1, true, cx);
    }

    fn grid_extend_right(&mut self, _: &GridExtendRight, _: &mut Window, cx: &mut Context<Self>) {
        self.move_thumbnail_selection(1, true, cx);
    }

    fn grid_extend_up(&mut self, _: &GridExtendUp, _: &mut Window, cx: &mut Context<Self>) {
        self.move_thumbnail_selection(-(self.grid_columns as isize), true, cx);
    }

    fn grid_extend_down(&mut self, _: &GridExtendDown, _: &mut Window, cx: &mut Context<Self>) {
        self.move_thumbnail_selection(self.grid_columns as isize, true, cx);
    }

    fn close_image(&mut self, _: &CloseImage, _: &mut Window, cx: &mut Context<Self>) {
        if matches!(self.main_view, MainView::Image(_)) {
            self.scheduler.clear_scope(IMAGE_SCOPE);
            self.scheduler.clear_scope(IMAGE_PREFETCH_SCOPE);
            self.image_prefetch_epoch = self.image_prefetch_epoch.wrapping_add(1);
            self.main_view = MainView::Dir(
                self.model.node(self.selected).path.to_path_buf(),
                self.saved_selection.clone(),
            );
            cx.notify();
        }
    }

    fn image_mouse_down(
        &mut self,
        event: &MouseDownEvent,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let MainView::Image(state) = &mut self.main_view else {
            return;
        };
        if event.click_count >= 2 {
            state.toggle_zoom();
        } else if state.zoom == ZoomLevel::ActualSize {
            state.drag_last = Some((f32::from(event.position.x), f32::from(event.position.y)));
        }
        cx.notify();
    }

    fn image_mouse_move(
        &mut self,
        event: &MouseMoveEvent,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if !event.dragging() {
            return;
        }
        let identity = match &self.main_view {
            MainView::Image(state) => self.render_identity_for(&state.path),
            MainView::Dir(_, _) => return,
        };
        let MainView::Image(state) = &mut self.main_view else {
            return;
        };
        if state.zoom != ZoomLevel::ActualSize {
            return;
        }
        let position = (f32::from(event.position.x), f32::from(event.position.y));
        let Some(previous) = state.drag_last.replace(position) else {
            return;
        };
        state.pan.0 += position.0 - previous.0;
        state.pan.1 += position.1 - previous.1;

        if let PreviewState::Loaded(preview) = self.previews.state_for(&state.path, identity) {
            let (image_width, image_height) = preview.metadata.output_dimensions;
            let viewport_width =
                (f32::from(window.viewport_size().width) - self.sidebar_width - 5.0).max(0.0);
            let viewport_height = (f32::from(window.viewport_size().height) - 44.0).max(0.0);
            let max_x = ((image_width as f32 - viewport_width) / 2.0).max(0.0);
            let max_y = ((image_height as f32 - viewport_height) / 2.0).max(0.0);
            state.pan.0 = state.pan.0.clamp(-max_x, max_x);
            state.pan.1 = state.pan.1.clamp(-max_y, max_y);
        }
        cx.notify();
    }

    fn image_mouse_up(&mut self, _: &MouseUpEvent, _: &mut Window, cx: &mut Context<Self>) {
        if let MainView::Image(state) = &mut self.main_view {
            state.drag_last = None;
            cx.notify();
        }
    }

    fn image_scroll_wheel(
        &mut self,
        event: &ScrollWheelEvent,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let direction = {
            let MainView::Image(state) = &mut self.main_view else {
                return;
            };
            if state.zoom != ZoomLevel::FitWindow {
                return;
            }
            cx.stop_propagation();
            state.wheel_delta_y += f32::from(event.delta.pixel_delta(px(20.0)).y);
            if state.wheel_delta_y <= -IMAGE_WHEEL_THRESHOLD_PX {
                1
            } else if state.wheel_delta_y >= IMAGE_WHEEL_THRESHOLD_PX {
                -1
            } else {
                return;
            }
        };

        let MainView::Image(state) = &self.main_view else {
            return;
        };
        let images = self.selected_images();
        let Some(current) = images.iter().position(|path| path == &state.path) else {
            return;
        };
        let Some(target) = adjacent_image_index(current, images.len(), direction) else {
            if let MainView::Image(state) = &mut self.main_view {
                state.wheel_delta_y = 0.0;
            }
            return;
        };
        let path = images[target].clone();
        let preview_variant = Self::raw_preview_variant(window, cx);
        let current_variant = if formats::is_raw(&path) {
            preview_variant
        } else {
            RenderVariant::Full
        };
        self.saved_selection.select(target, false, false);
        self.show_image(path.clone(), cx);
        // Promote before clearing the old speculative scope. This transfers
        // the in-flight cache subscription instead of cancelling the render
        // for the image the user just navigated to.
        self.replace_render_scope(
            IMAGE_SCOPE,
            vec![(path.clone(), current_variant, JobPriority::INTERACTIVE)],
            cx,
        );
        self.schedule_image_prefetch(path, direction, preview_variant, cx);
    }

    fn selected_images(&self) -> &[PathBuf] {
        self.model
            .loaded(self.selected)
            .map_or(&[], |contents| contents.images.as_slice())
    }

    fn render_thumbnail_rows(
        &mut self,
        range: Range<usize>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Vec<gpui::AnyElement> {
        self.record_grid_demand(
            range.clone(),
            Self::raw_preview_variant(window, cx),
            window,
            cx,
        );
        let first = range
            .start
            .saturating_mul(self.grid_columns)
            .min(self.selected_images().len());
        let last = range
            .end
            .saturating_mul(self.grid_columns)
            .min(self.selected_images().len());
        let visible_paths: Vec<_> = self.selected_images()[first..last]
            .iter()
            .cloned()
            .enumerate()
            .map(|(offset, path)| (first + offset, path))
            .collect();
        let selected = match &self.main_view {
            MainView::Dir(_, selection) => selection.indices.clone(),
            MainView::Image(_) => BTreeSet::new(),
        };
        range
            .map(|row| {
                let tile_height = self.thumbnail_width as f32 + 48.0;
                let mut row_element = div()
                    .h(px(tile_height + THUMBNAIL_GAP))
                    .flex()
                    .gap(px(THUMBNAIL_GAP))
                    .px_3()
                    .pt_3();
                for (index, path) in visible_paths.iter().filter(|(index, _)| {
                    *index >= row * self.grid_columns && *index < (row + 1) * self.grid_columns
                }) {
                    let index = *index;
                    let path = path.clone();
                    let identity = self.render_identity_for(&path);
                    let image_content = match self.thumbnails.state_for(&path, identity) {
                        PreviewState::Loaded(preview) => img(preview.image.clone())
                            .max_w_full()
                            .max_h_full()
                            .into_any_element(),
                        PreviewState::Error(_) => {
                            div().text_xs().child("Cannot decode").into_any_element()
                        }
                        _ => div().text_xs().child("Loading…").into_any_element(),
                    };
                    let name: SharedString = path
                        .file_name()
                        .unwrap_or(path.as_os_str())
                        .to_string_lossy()
                        .into_owned()
                        .into();
                    let has_develop_data = has_develop_recipe(self.recipes.as_deref(), &path);
                    let open_path = path.clone();
                    let context_path = path.clone();
                    row_element = row_element.child(
                        div()
                            .id(("thumbnail", index))
                            .w(px(self.thumbnail_width as f32))
                            .h(px(tile_height))
                            .flex_none()
                            .flex()
                            .flex_col()
                            .p_2()
                            .rounded_md()
                            .when(selected.contains(&index), |this| {
                                this.bg(rgb(0x334b6e))
                                    .border_1()
                                    .border_color(rgb(0x6f9ddd))
                            })
                            .hover(|this| this.bg(rgb(0x292d36)))
                            .child(
                                div()
                                    .w_full()
                                    .h(px(self.thumbnail_width as f32))
                                    .flex_none()
                                    .flex()
                                    .items_center()
                                    .justify_center()
                                    .child(image_content),
                            )
                            .child(
                                div()
                                    .mt_2()
                                    .w_full()
                                    .flex()
                                    .items_center()
                                    .gap_1()
                                    .text_xs()
                                    .child(div().min_w_0().flex_1().truncate().child(name))
                                    .when(has_develop_data, |this| {
                                        this.child(
                                            div()
                                                .flex_none()
                                                .px_1()
                                                .rounded_sm()
                                                .bg(rgb(0x36547a))
                                                .text_color(rgb(0xd7e7ff))
                                                .child("D"),
                                        )
                                    }),
                            )
                            .on_mouse_down(
                                MouseButton::Right,
                                cx.listener(move |this, event: &MouseDownEvent, window, cx| {
                                    window.focus(&this.main_focus);
                                    this.select_thumbnail(index, false, false, cx);
                                    this.path_context_menu = Some(PathContextMenu {
                                        path: context_path.clone(),
                                        position: (
                                            f32::from(event.position.x),
                                            f32::from(event.position.y),
                                        ),
                                        exportable: true,
                                    });
                                    cx.stop_propagation();
                                    cx.notify();
                                }),
                            )
                            .on_click(cx.listener(move |this, event: &ClickEvent, window, cx| {
                                let modifiers = window.modifiers();
                                window.focus(&this.main_focus);
                                this.select_thumbnail(
                                    index,
                                    modifiers.shift,
                                    modifiers.secondary(),
                                    cx,
                                );
                                if event.click_count() >= 2 {
                                    this.show_image(open_path.clone(), cx);
                                }
                                cx.notify();
                            })),
                    );
                }
                row_element.into_any_element()
            })
            .collect()
    }

    /// Recreates only the flat rows that are actually exposed by expanded
    /// branches. Collapsed descendants have no row or GPUI element state.
    fn rebuild_visible(&mut self) {
        self.visible = visible_rows(&self.model, &self.expanded);
    }

    fn render_rows(
        &mut self,
        range: Range<usize>,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Vec<gpui::AnyElement> {
        range
            .map(|row_ix| {
                let row = self.visible[row_ix];
                let node = self.model.node(row.node);
                let selected = row.node == self.selected;
                let expanded = self.expanded.contains(&row.node);
                let indicator = match &node.contents {
                    DirectoryContents::Loading(_) => "…",
                    DirectoryContents::Error(_) => "!",
                    DirectoryContents::Loaded(contents) if contents.children.is_empty() => "",
                    _ if expanded => "▾",
                    _ => "▸",
                };
                let name: SharedString = node.name.to_string().into();
                let select_node = row.node;
                let toggle_node = row.node;
                div()
                    .id(("directory", row_ix))
                    .flex()
                    .items_center()
                    .h(px(30.0))
                    .pl(px(10.0 + row.depth as f32 * 16.0))
                    .pr_2()
                    .text_sm()
                    .text_color(rgb(0xd7dae0))
                    .when(selected, |this| this.bg(rgb(0x334b6e)))
                    .hover(|this| this.bg(rgb(0x292d36)))
                    .child(
                        div()
                            .id(("toggle", row_ix))
                            .w(px(20.0))
                            .flex_none()
                            .text_color(rgb(0x8d96a5))
                            .child(indicator)
                            .on_mouse_down(
                                MouseButton::Left,
                                cx.listener(move |this, _, _, cx| {
                                    this.toggle(toggle_node, cx);
                                }),
                            ),
                    )
                    .child(div().min_w_0().truncate().child(name))
                    .on_mouse_down(
                        MouseButton::Right,
                        cx.listener(move |this, event: &MouseDownEvent, _, cx| {
                            this.path_context_menu = Some(PathContextMenu {
                                path: this.model.node(select_node).path.to_path_buf(),
                                position: (
                                    f32::from(event.position.x),
                                    f32::from(event.position.y),
                                ),
                                exportable: false,
                            });
                            cx.stop_propagation();
                            cx.notify();
                        }),
                    )
                    .on_click(cx.listener(move |this, event: &ClickEvent, window, cx| {
                        window.focus(&this.tree_focus);
                        this.select(select_node, cx);
                        if event.click_count() >= 2 {
                            this.open(select_node, cx);
                        }
                    }))
                    .into_any_element()
            })
            .collect()
    }

    fn toggle_job_queue(&mut self, _: &ToggleJobQueue, _: &mut Window, cx: &mut Context<Self>) {
        self.show_job_queue = !self.show_job_queue;
        self.show_view_menu = false;
        self.job_queue_snapshot = self.scheduler.snapshot();
        cx.notify();
    }

    fn open_settings(&mut self, _: &OpenSettings, _: &mut Window, cx: &mut Context<Self>) {
        self.settings_draft = Some(self.settings.clone());
        self.settings_status = None;
        self.show_file_menu = false;
        cx.notify();
    }

    fn save_settings(&mut self, cx: &mut Context<Self>) {
        let Some(draft) = self.settings_draft.clone() else {
            return;
        };
        match config::save(&self.settings_path, &draft) {
            Ok(()) => {
                self.pair_preference = draft.pair_preference.into();
                self.settings = draft;
                self.settings_draft = None;
                self.settings_status = Some(
                    "Settings saved. Restart Pics to apply service and library changes.".into(),
                );
            }
            Err(error) => {
                self.settings_status = Some(format!("Could not save settings: {error:#}"))
            }
        }
        cx.notify();
    }

    fn adjust_setting(&mut self, field: SettingsNumber, delta: isize, cx: &mut Context<Self>) {
        let Some(draft) = &mut self.settings_draft else {
            return;
        };
        let value = match field {
            SettingsNumber::ImageCacheSize => &mut draft.image_cache_size,
            SettingsNumber::JobParallelism => &mut draft.job_parallelism,
            SettingsNumber::RawParallelism => &mut draft.raw_parallelism,
        };
        *value = value.saturating_add_signed(delta).max(1);
        if draft.raw_parallelism > draft.job_parallelism {
            draft.raw_parallelism = draft.job_parallelism;
        }
        cx.notify();
    }

    fn choose_settings_directory(&mut self, choose_root: bool, cx: &mut Context<Self>) {
        let selection = cx.prompt_for_paths(PathPromptOptions {
            files: false,
            directories: true,
            multiple: false,
            prompt: Some(
                if choose_root {
                    "Choose picture library"
                } else {
                    "Choose cache directory"
                }
                .into(),
            ),
        });
        cx.spawn(async move |this, cx| {
            let path = match selection.await {
                Ok(Ok(Some(paths))) => paths.into_iter().next(),
                _ => None,
            };
            if let Some(path) = path {
                this.update(cx, |this, cx| {
                    if let Some(draft) = &mut this.settings_draft {
                        if choose_root {
                            draft.root = path;
                        } else {
                            draft.cache_root = path;
                        }
                    }
                    cx.notify();
                })
                .ok();
            }
        })
        .detach();
    }

    fn import_lightroom(&mut self, _: &ImportLightroom, _: &mut Window, cx: &mut Context<Self>) {
        self.show_file_menu = false;
        let selection = cx.prompt_for_paths(PathPromptOptions {
            files: true,
            directories: false,
            multiple: false,
            prompt: Some("Import Lightroom catalog".into()),
        });
        let root = self.model.node(self.model.root()).path.to_path_buf();
        let background = cx.background_executor().clone();
        cx.spawn(async move |this, cx| {
            let catalog_path = match selection.await {
                Ok(Ok(Some(paths))) => paths.into_iter().next(),
                _ => None,
            };
            let Some(catalog_path) = catalog_path else {
                return;
            };
            let work = this
                .update(cx, |this, cx| {
                    let id = this.work.start(format!(
                        "Importing Lightroom catalog {}…",
                        path_display_name(&catalog_path)
                    ));
                    cx.notify();
                    id
                })
                .ok()
                .unwrap_or_default();
            let import_root = root.clone();
            let result = background
                .spawn(async move {
                    catalog::import_lightroom(&import_root, &catalog_path)?;
                    catalog::load_recipes(&import_root)
                })
                .await;
            this.update(cx, |this, cx| {
                this.work.finish(work);
                match result {
                    Ok(recipes) => {
                        this.recipes = Some(Arc::new(recipes));
                        this.previews.clear();
                        this.raw_previews.clear();
                        this.thumbnails.clear();
                    }
                    Err(error) => eprintln!("cannot import Lightroom catalog: {error:#}"),
                }
                cx.notify();
            })
            .ok();
        })
        .detach();
    }

    fn copy_absolute_path(&mut self, path: &std::path::Path, cx: &mut Context<Self>) {
        let text = path.to_string_lossy().into_owned();
        cx.write_to_clipboard(ClipboardItem::new_string(text.clone()));
        if self.system_clipboard.is_none() {
            self.system_clipboard = arboard::Clipboard::new()
                .map_err(|error| {
                    eprintln!("cannot initialize clipboard fallback: {error}");
                    error
                })
                .ok();
        }
        if let Some(clipboard) = &mut self.system_clipboard
            && let Err(error) = clipboard.set_text(text)
        {
            eprintln!("cannot copy path through clipboard fallback: {error}");
        }
        self.path_context_menu = None;
        cx.notify();
    }

    fn export_image(&mut self, path: PathBuf, cx: &mut Context<Self>) {
        self.path_context_menu = None;
        let suggested_name = path
            .file_stem()
            .unwrap_or(path.as_os_str())
            .to_string_lossy()
            .into_owned()
            + ".png";
        let directory = export_dialog_directory(self.last_export_directory.as_deref(), &path);
        let destination = cx.prompt_for_new_path(directory, Some(&suggested_name));
        let pipeline = self.render_pipeline.clone();
        let render_identity = self.render_identity_for(&path);
        let recipe = self
            .developed
            .then(|| self.recipes.as_ref()?.get(&path))
            .flatten()
            .map(|entry| entry.recipe.clone());
        let background = cx.background_executor().clone();
        cx.notify();

        cx.spawn(async move |this, cx| {
            let destination = match destination.await {
                Ok(Ok(Some(destination))) => destination,
                Ok(Ok(None)) | Err(_) => return,
                Ok(Err(error)) => {
                    eprintln!("cannot open export save dialog: {error:#}");
                    return;
                }
            };
            let export_directory = destination.parent().map(std::path::Path::to_path_buf);
            let export_work = this
                .update(cx, |this, cx| {
                    this.last_export_directory = export_directory;
                    let export_work = this
                        .work
                        .start(format!("Exporting {}…", path_display_name(&destination)));
                    cx.notify();
                    export_work
                })
                .ok()
                .unwrap_or_default();
            let export = background.spawn(async move {
                let cancelled = Arc::new(AtomicBool::new(false));
                let rendered = pipeline.render_with_recipe(
                    RenderInvocation {
                        path,
                        variant: RenderVariant::Full,
                        render_identity,
                        recipe: recipe.as_ref(),
                        interactive: true,
                        durable_cache_write: false,
                    },
                    &cancelled,
                );
                save_preview_as_png(rendered, &destination)
            });
            let result = export.await;
            this.update(cx, |this, cx| {
                this.work.finish(export_work);
                if let Err(error) = result {
                    eprintln!("cannot export image: {error:#}");
                }
                cx.notify();
            })
            .ok();
        })
        .detach();
    }

    fn toggle_develop_inspector(
        &mut self,
        _: &ToggleDevelopInspector,
        _: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.show_develop_inspector = !self.show_develop_inspector;
        self.show_view_menu = false;
        cx.notify();
    }

    fn toggle_developed(&mut self, _: &ToggleDeveloped, _: &mut Window, cx: &mut Context<Self>) {
        self.developed = !self.developed;
        cx.notify();
    }

    fn inspected_image_path(&self) -> Option<PathBuf> {
        match &self.main_view {
            MainView::Image(state) => Some(state.path.clone()),
            MainView::Dir(_, selection) => selection
                .focused
                .and_then(|index| self.selected_images().get(index).cloned()),
        }
    }

    fn render_develop_inspector(&self) -> AnyElement {
        let selected = self.inspected_image_path();
        let recipe = selected
            .as_ref()
            .and_then(|path| self.recipes.as_ref()?.get(path));
        let title = selected
            .as_ref()
            .and_then(|path| path.file_name())
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| "No image selected".to_owned());
        let rows = recipe
            .map(|entry| format_develop_edits(&entry.recipe))
            .unwrap_or_default();
        let mut contents = div()
            .id("develop-edit-rows")
            .flex_1()
            .min_h_0()
            .overflow_y_scroll()
            .p_3()
            .flex()
            .flex_col()
            .gap_2();
        if self.recipes.is_none() {
            contents = contents.child("Loading catalog…");
        } else if selected.is_none() {
            contents = contents.child("Select an image to inspect its develop edits.");
        } else if recipe.is_none() {
            contents = contents.child("This image has no imported develop data.");
        } else if rows.is_empty() {
            contents = contents.child("The imported recipe contains no active edits.");
        } else {
            for (name, value) in rows {
                contents = contents.child(
                    div()
                        .pb_2()
                        .border_b_1()
                        .border_color(rgb(0x292d36))
                        .child(div().text_xs().text_color(rgb(0x8d96a5)).child(name))
                        .child(
                            div()
                                .mt_1()
                                .text_sm()
                                .text_color(rgb(0xd0d5df))
                                .child(value),
                        ),
                );
            }
        }
        div()
            .w(px(DEVELOP_PANEL_WIDTH))
            .h_full()
            .flex_none()
            .flex()
            .flex_col()
            .bg(rgb(0x15171c))
            .border_l_1()
            .border_color(rgb(0x30343d))
            .child(
                div()
                    .h(px(54.0))
                    .flex_none()
                    .px_3()
                    .flex()
                    .flex_col()
                    .justify_center()
                    .border_b_1()
                    .border_color(rgb(0x30343d))
                    .child("Develop edits")
                    .child(
                        div()
                            .truncate()
                            .text_xs()
                            .text_color(rgb(0x9da7b6))
                            .child(title),
                    ),
            )
            .child(contents)
            .into_any_element()
    }

    fn schedule_scheduler_refresh(&mut self, cx: &mut Context<Self>) {
        if self.scheduler_refresh_scheduled {
            return;
        }
        self.scheduler_refresh_scheduled = true;
        let timer = cx.background_executor().timer(SCHEDULER_REFRESH_INTERVAL);
        cx.spawn(async move |this, cx| {
            timer.await;
            this.update(cx, |this, cx| {
                this.scheduler_refresh_scheduled = false;
                this.job_queue_snapshot = this.scheduler.snapshot();
                cx.notify();
            })
            .ok();
        })
        .detach();
    }

    fn render_job_queue_rows(
        &mut self,
        range: Range<usize>,
        _: &mut Window,
        _: &mut Context<Self>,
    ) -> Vec<gpui::AnyElement> {
        range
            .filter_map(|index| {
                let job = self.job_queue_snapshot.jobs.get(index)?;
                let status = match job.status {
                    JobSnapshotStatus::Running => "RUN",
                    JobSnapshotStatus::Queued => "QUEUE",
                    JobSnapshotStatus::PendingSuccessor => "NEXT",
                };
                let file = job
                    .key
                    .path
                    .file_name()
                    .unwrap_or(job.key.path.as_os_str())
                    .to_string_lossy()
                    .into_owned();
                let variant = format!("{:?}", job.key.variant);
                let priority = format!(
                    "#{index} {status} · {:?} ({:+}) · {} scope{} · seq {}{}",
                    job.priority.class,
                    job.priority.rank,
                    job.scope_count,
                    if job.scope_count == 1 { "" } else { "s" },
                    job.sequence,
                    if job.cancelled { " · cancelling" } else { "" },
                );
                Some(
                    div()
                        .h(px(58.0))
                        .px_3()
                        .py_2()
                        .border_b_1()
                        .border_color(rgb(0x30343d))
                        .overflow_hidden()
                        .child(
                            div()
                                .text_sm()
                                .truncate()
                                .text_color(rgb(0xe1e5ec))
                                .child(file),
                        )
                        .child(
                            div()
                                .text_xs()
                                .truncate()
                                .text_color(rgb(0x9da7b6))
                                .child(priority),
                        )
                        .child(
                            div()
                                .text_xs()
                                .truncate()
                                .text_color(rgb(0x778292))
                                .child(variant),
                        )
                        .into_any_element(),
                )
            })
            .collect()
    }
}

fn visible_rows(model: &DirectoryModel, expanded: &HashSet<NodeId>) -> Vec<VisibleRow> {
    fn append(
        model: &DirectoryModel,
        expanded: &HashSet<NodeId>,
        node: NodeId,
        depth: usize,
        rows: &mut Vec<VisibleRow>,
    ) {
        rows.push(VisibleRow { node, depth });
        if let Some(contents) = expanded
            .contains(&node)
            .then(|| model.loaded(node))
            .flatten()
        {
            for child in &contents.children {
                append(model, expanded, *child, depth + 1, rows);
            }
        }
    }

    let mut rows = Vec::new();
    append(model, expanded, model.root(), 0, &mut rows);
    rows
}

fn visible_thumbnail_demands(
    images: &[PathBuf],
    row_range: Range<usize>,
    columns: usize,
    width: u32,
) -> Vec<(PathBuf, RenderVariant, JobPriority)> {
    let first = row_range.start.saturating_mul(columns).min(images.len());
    let last = row_range.end.saturating_mul(columns).min(images.len());
    let center = first.saturating_add(last).saturating_div(2);
    let variant = RenderVariant::Thumbnail { width };
    images[first..last]
        .iter()
        .enumerate()
        .map(|(offset, path)| {
            let index = first + offset;
            let distance = index.abs_diff(center).min(i32::MAX as usize) as i32;
            (path.clone(), variant, JobPriority::visible(-distance))
        })
        .collect()
}

fn speculative_indices(
    image_count: usize,
    visible: Range<usize>,
    direction: i8,
    batch: usize,
    batch_size: usize,
) -> Vec<usize> {
    let before_len = visible.start.min(image_count);
    let after_start = visible.end.min(image_count);
    let after_len = image_count.saturating_sub(after_start);
    let total = before_len.saturating_add(after_len);
    let start = batch.saturating_mul(batch_size).min(total);
    let end = start.saturating_add(batch_size).min(total);
    (start..end)
        .map(|position| {
            if direction < 0 {
                if position < before_len {
                    before_len - 1 - position
                } else {
                    after_start + position - before_len
                }
            } else if position < after_len {
                after_start + position
            } else {
                before_len - 1 - (position - after_len)
            }
        })
        .collect()
}

impl Render for PictureBrowser {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        self.job_queue_snapshot = self.scheduler.snapshot();
        let scheduler_jobs = self
            .job_queue_snapshot
            .jobs
            .iter()
            .filter(|job| !job.cancelled)
            .count();
        if self.show_job_queue || scheduler_jobs > 0 || self.work.active > 0 {
            self.schedule_scheduler_refresh(cx);
        }
        let inspector_width = if self.show_job_queue {
            JOB_QUEUE_PANEL_WIDTH
        } else {
            0.0
        } + if self.show_develop_inspector {
            DEVELOP_PANEL_WIDTH
        } else {
            0.0
        };
        let main_width =
            (f32::from(window.viewport_size().width) - self.sidebar_width - inspector_width - 29.0)
                .max(self.thumbnail_width as f32);
        self.grid_columns = ((main_width + THUMBNAIL_GAP)
            / (self.thumbnail_width as f32 + THUMBNAIL_GAP))
            .floor()
            .max(1.0) as usize;
        let tree_scrollbar =
            self.render_scrollbar("tree-scrollbar", ScrollbarTarget::Tree, &self.scroll, cx);
        let grid_scrollbar = self.render_scrollbar(
            "grid-scrollbar",
            ScrollbarTarget::Grid,
            &self.thumbnail_scroll,
            cx,
        );
        let tree = div()
            .id("directory-tree")
            .w(px(self.sidebar_width))
            .h_full()
            .flex_none()
            .track_focus(&self.tree_focus)
            .key_context("DirectoryTree")
            .on_action(cx.listener(Self::tree_up))
            .on_action(cx.listener(Self::tree_down))
            .on_action(cx.listener(Self::tree_left))
            .on_action(cx.listener(Self::tree_right))
            .on_action(cx.listener(Self::tree_confirm))
            .bg(rgb(0x181a20))
            .py_2()
            .relative()
            .child(
                uniform_list(
                    "directory-rows",
                    self.visible.len(),
                    cx.processor(Self::render_rows),
                )
                .size_full()
                .track_scroll(self.scroll.clone())
                .with_sizing_behavior(ListSizingBehavior::Auto),
            )
            .child(tree_scrollbar);

        let has_active_recipe = self.developed
            && match &self.main_view {
                MainView::Image(state) => {
                    self.render_identity_for(&state.path).mode == RenderMode::Developed
                }
                MainView::Dir(_, _) => self
                    .selected_images()
                    .iter()
                    .any(|path| self.render_identity_for(path).mode == RenderMode::Developed),
            };
        let (title, show_back, main_content) = match &self.main_view {
            MainView::Image(state) => {
                let path = state.path.clone();
                let context_path = path.clone();
                let zoom = state.zoom;
                let pan = state.pan;
                let variant = if zoom == ZoomLevel::FitWindow && formats::is_raw(&path) {
                    Self::raw_preview_variant(window, cx)
                } else {
                    RenderVariant::Full
                };
                self.replace_render_scope(
                    IMAGE_SCOPE,
                    vec![(path.clone(), variant, JobPriority::INTERACTIVE)],
                    cx,
                );
                let state = match variant {
                    RenderVariant::Preview { .. } => self
                        .raw_previews
                        .state_for(&path, self.render_identity_for(&path)),
                    RenderVariant::Full => self
                        .previews
                        .state_for(&path, self.render_identity_for(&path)),
                    RenderVariant::Thumbnail { .. } => unreachable!(),
                };
                let content = match state {
                    PreviewState::Loaded(preview) => match zoom {
                        ZoomLevel::FitWindow => img(preview.image.clone())
                            .max_w_full()
                            .max_h_full()
                            .into_any_element(),
                        ZoomLevel::ActualSize => {
                            let (width, height) = preview.metadata.output_dimensions;
                            let viewport_width = (f32::from(window.viewport_size().width)
                                - self.sidebar_width
                                - 5.0)
                                .max(0.0);
                            let viewport_height =
                                (f32::from(window.viewport_size().height) - 44.0).max(0.0);
                            img(preview.image.clone())
                                .absolute()
                                .left(px((viewport_width - width as f32) / 2.0 + pan.0))
                                .top(px((viewport_height - height as f32) / 2.0 + pan.1))
                                .w(px(width as f32))
                                .h(px(height as f32))
                                .into_any_element()
                        }
                    },
                    PreviewState::Error(error) => {
                        div().text_sm().child(error.to_string()).into_any_element()
                    }
                    _ => div().text_sm().child("Loading…").into_any_element(),
                };
                (
                    path.file_name()
                        .unwrap_or(path.as_os_str())
                        .to_string_lossy()
                        .into_owned(),
                    true,
                    div()
                        .id("image-viewport")
                        .size_full()
                        .relative()
                        .overflow_hidden()
                        .flex()
                        .items_center()
                        .justify_center()
                        .when(zoom == ZoomLevel::FitWindow, |this| this.p_6())
                        .cursor(match zoom {
                            ZoomLevel::FitWindow => CursorStyle::Arrow,
                            ZoomLevel::ActualSize => CursorStyle::OpenHand,
                        })
                        .on_mouse_down(MouseButton::Left, cx.listener(Self::image_mouse_down))
                        .on_mouse_down(
                            MouseButton::Right,
                            cx.listener(move |this, event: &MouseDownEvent, _, cx| {
                                this.path_context_menu = Some(PathContextMenu {
                                    path: context_path.clone(),
                                    position: (
                                        f32::from(event.position.x),
                                        f32::from(event.position.y),
                                    ),
                                    exportable: true,
                                });
                                cx.stop_propagation();
                                cx.notify();
                            }),
                        )
                        .on_mouse_move(cx.listener(Self::image_mouse_move))
                        .on_mouse_up(MouseButton::Left, cx.listener(Self::image_mouse_up))
                        .on_scroll_wheel(cx.listener(Self::image_scroll_wheel))
                        .child(content)
                        .into_any_element(),
                )
            }
            MainView::Dir(path, selection) => {
                let count = self.selected_images().len();
                let ordering_pending = self.model.loaded(self.selected).is_some_and(|contents| {
                    contents.metadata_status != DirectoryMetadataStatus::Complete
                        && !contents.images.is_empty()
                });
                let capture_order_available = self.model.has_pending_capture_order(self.selected);
                let ordering_suffix = if ordering_pending {
                    " · loading capture metadata…"
                } else {
                    ""
                };
                let grid_content = match &self.model.node(self.selected).contents {
                    DirectoryContents::Error(error) => div()
                        .p_6()
                        .text_sm()
                        .child(error.to_string())
                        .into_any_element(),
                    DirectoryContents::Unloaded | DirectoryContents::Loading(_) => div()
                        .p_6()
                        .text_sm()
                        .child("Loading directory…")
                        .into_any_element(),
                    DirectoryContents::Loaded(_) if count == 0 => div()
                        .p_6()
                        .text_sm()
                        .child("No images in this directory")
                        .into_any_element(),
                    DirectoryContents::Loaded(_) => uniform_list(
                        "thumbnail-rows",
                        count.div_ceil(self.grid_columns),
                        cx.processor(Self::render_thumbnail_rows),
                    )
                    .size_full()
                    .track_scroll(self.thumbnail_scroll.clone())
                    .with_sizing_behavior(ListSizingBehavior::Auto)
                    .into_any_element(),
                };
                let content = div()
                    .size_full()
                    .flex()
                    .flex_col()
                    .when(capture_order_available, |this| {
                        this.child(
                            div()
                                .flex_none()
                                .flex()
                                .items_center()
                                .justify_between()
                                .px_4()
                                .py_2()
                                .bg(rgb(0x242b37))
                                .border_b_1()
                                .border_color(rgb(0x3b4658))
                                .text_sm()
                                .text_color(rgb(0xc8ced8))
                                .child("Capture times are available. The current order was kept.")
                                .child(
                                    div()
                                        .id("apply-capture-time-order")
                                        .ml_4()
                                        .px_3()
                                        .py_1()
                                        .rounded_sm()
                                        .bg(rgb(0x3b4658))
                                        .hover(|this| this.bg(rgb(0x52617a)))
                                        .cursor_pointer()
                                        .child("Apply capture-time order")
                                        .on_click(cx.listener(|this, _, _, cx| {
                                            this.apply_capture_time_order(cx);
                                        })),
                                ),
                        )
                    })
                    .child(
                        div()
                            .flex_1()
                            .min_h_0()
                            .relative()
                            .child(grid_content)
                            .child(grid_scrollbar),
                    )
                    .into_any_element();
                (
                    format!(
                        "{} — {} selected{}",
                        path.display(),
                        selection.indices.len(),
                        ordering_suffix
                    ),
                    false,
                    content,
                )
            }
        };
        let main_view = div()
            .flex_1()
            .min_w_0()
            .h_full()
            .flex()
            .flex_col()
            .track_focus(&self.main_focus)
            .key_context("MainView")
            .on_action(cx.listener(Self::open_selected_image))
            .on_action(cx.listener(Self::close_image))
            .on_action(cx.listener(Self::grid_left))
            .on_action(cx.listener(Self::grid_right))
            .on_action(cx.listener(Self::grid_up))
            .on_action(cx.listener(Self::grid_down))
            .on_action(cx.listener(Self::grid_extend_left))
            .on_action(cx.listener(Self::grid_extend_right))
            .on_action(cx.listener(Self::grid_extend_up))
            .on_action(cx.listener(Self::grid_extend_down))
            .bg(rgb(0x101217))
            .child(
                div()
                    .h(px(44.0))
                    .flex_none()
                    .flex()
                    .items_center()
                    .px_4()
                    .border_b_1()
                    .border_color(rgb(0x30343d))
                    .text_sm()
                    .text_color(rgb(0xaeb4bf))
                    .when(show_back, |this| {
                        this.child(
                            div()
                                .id("back-to-directory")
                                .mr_3()
                                .px_2()
                                .py_1()
                                .rounded_sm()
                                .bg(rgb(0x30343d))
                                .hover(|this| this.bg(rgb(0x445066)))
                                .child("← Back")
                                .on_click(cx.listener(|this, _, window, cx| {
                                    this.close_image(&CloseImage, window, cx);
                                })),
                        )
                    })
                    .child(title)
                    .child(div().flex_1())
                    .child(
                        div()
                            .id("toggle-developed")
                            .px_2()
                            .py_1()
                            .rounded_sm()
                            .bg(rgb(if has_active_recipe {
                                0x36547a
                            } else {
                                0x30343d
                            }))
                            .hover(|this| this.bg(rgb(0x445066)))
                            .child(if has_active_recipe {
                                "Developed"
                            } else if self.developed {
                                "Baseline (no adjustments)"
                            } else {
                                "Baseline"
                            })
                            .on_click(cx.listener(|this, _, window, cx| {
                                this.toggle_developed(&ToggleDeveloped, window, cx);
                            })),
                    ),
            )
            .child(
                div()
                    .flex_1()
                    .min_h_0()
                    .text_color(rgb(0x8d96a5))
                    .child(main_content),
            );
        let splitter = div()
            .id("sidebar-splitter")
            .w(px(5.0))
            .h_full()
            .flex_none()
            .cursor(CursorStyle::ResizeLeftRight)
            .bg(rgb(0x30343d))
            .hover(|this| this.bg(rgb(0x5577aa)))
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(|this, _, _, cx| {
                    this.resizing_sidebar = true;
                    cx.notify();
                }),
            );
        let develop_inspector = self
            .show_develop_inspector
            .then(|| self.render_develop_inspector());
        let workspace = div()
            .w_full()
            .h_full()
            .flex_1()
            .min_h_0()
            .flex()
            .on_mouse_move(cx.listener(|this, event, _, cx| {
                this.resize_sidebar(event, cx);
            }))
            .on_mouse_up(
                MouseButton::Left,
                cx.listener(|this, _, _, cx| {
                    this.resizing_sidebar = false;
                    cx.notify();
                }),
            )
            .child(tree)
            .child(splitter)
            .child(main_view)
            .when_some(develop_inspector, |this, panel| this.child(panel))
            .when(self.show_job_queue, |this| {
                this.child(
                    div()
                        .w(px(JOB_QUEUE_PANEL_WIDTH))
                        .h_full()
                        .flex_none()
                        .flex()
                        .flex_col()
                        .bg(rgb(0x15171c))
                        .border_l_1()
                        .border_color(rgb(0x30343d))
                        .child(
                            div()
                                .h(px(42.0))
                                .flex_none()
                                .flex()
                                .items_center()
                                .justify_between()
                                .px_3()
                                .border_b_1()
                                .border_color(rgb(0x30343d))
                                .child("Job queue")
                                .child(div().text_xs().text_color(rgb(0x9da7b6)).child(format!(
                                    "{} running · {} queued",
                                    self.job_queue_snapshot.running, self.job_queue_snapshot.queued
                                ))),
                        )
                        .child(
                            uniform_list(
                                "job-queue-rows",
                                self.job_queue_snapshot.jobs.len(),
                                cx.processor(Self::render_job_queue_rows),
                            )
                            .flex_1()
                            .min_h_0()
                            .track_scroll(self.job_queue_scroll.clone())
                            .with_sizing_behavior(ListSizingBehavior::Auto),
                        ),
                )
            });
        let progress_active = self.work.active > 0 || scheduler_jobs > 0;
        let progress = if self.work.active > 0 {
            self.work.fraction().clamp(0.08, 1.0)
        } else {
            0.35
        };
        let progress_label = scheduler_progress_label(&self.job_queue_snapshot)
            .or_else(|| self.work.current_activity().map(str::to_owned))
            .unwrap_or_else(|| "Finishing background work…".to_owned());
        let menu_bar = div()
            .h(px(30.0))
            .w_full()
            .flex_none()
            .flex()
            .items_center()
            .px_2()
            .bg(rgb(0x20232a))
            .text_color(rgb(0xd8dde7))
            .border_b_1()
            .border_color(rgb(0x30343d))
            .child(
                div()
                    .id("file-menu-button")
                    .h(px(24.0))
                    .px_2()
                    .flex()
                    .items_center()
                    .rounded_sm()
                    .text_sm()
                    .hover(|this| this.bg(rgb(0x343943)))
                    .when(self.show_file_menu, |this| this.bg(rgb(0x343943)))
                    .child("File")
                    .on_click(cx.listener(|this, _, _, cx| {
                        this.show_file_menu = !this.show_file_menu;
                        this.show_view_menu = false;
                        cx.notify();
                    })),
            )
            .child(
                div()
                    .id("view-menu-button")
                    .h(px(24.0))
                    .px_2()
                    .flex()
                    .items_center()
                    .rounded_sm()
                    .text_sm()
                    .hover(|this| this.bg(rgb(0x343943)))
                    .when(self.show_view_menu, |this| this.bg(rgb(0x343943)))
                    .child("View")
                    .on_click(cx.listener(|this, _, _, cx| {
                        this.show_view_menu = !this.show_view_menu;
                        this.show_file_menu = false;
                        cx.notify();
                    })),
            );
        let content = div()
            .w_full()
            .flex_1()
            .min_h_0()
            .relative()
            .child(workspace)
            .when(progress_active, |this| {
                this.child(
                    div()
                        .absolute()
                        .left_0()
                        .right_0()
                        .bottom_0()
                        .h(px(22.0))
                        .bg(rgb(0x181a20))
                        .border_t_1()
                        .border_color(rgb(0x30343d))
                        .child(
                            div()
                                .absolute()
                                .left_0()
                                .top_0()
                                .bottom_0()
                                .w(relative(progress))
                                .bg(rgb(0x315b8f)),
                        )
                        .child(
                            div()
                                .absolute()
                                .inset_0()
                                .flex()
                                .items_center()
                                .justify_center()
                                .text_xs()
                                .text_color(rgb(0xd8dde7))
                                .child(progress_label),
                        ),
                )
            });
        let path_context_menu = self.path_context_menu.as_ref().map(|menu| {
            let viewport = window.viewport_size();
            let can_export = menu.exportable;
            let left = menu
                .position
                .0
                .min((f32::from(viewport.width) - 228.0).max(0.0));
            let top = menu
                .position
                .1
                .min((f32::from(viewport.height) - if can_export { 66.0 } else { 38.0 }).max(0.0));
            (menu.path.clone(), left, top, can_export)
        });
        let settings_dialog = self
            .settings_draft
            .clone()
            .map(|draft| render_settings_dialog(draft, self.settings_status.clone(), cx));
        div()
            .size_full()
            .relative()
            .flex()
            .flex_col()
            .key_context("PictureBrowser")
            .on_action(cx.listener(Self::toggle_job_queue))
            .on_action(cx.listener(Self::toggle_develop_inspector))
            .on_action(cx.listener(Self::toggle_developed))
            .on_action(cx.listener(Self::open_settings))
            .on_action(cx.listener(Self::import_lightroom))
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(|this, _, _, cx| {
                    if this.path_context_menu.take().is_some() {
                        cx.notify();
                    }
                }),
            )
            .on_mouse_down(
                MouseButton::Right,
                cx.listener(|this, _, _, cx| {
                    if this.path_context_menu.take().is_some() {
                        cx.notify();
                    }
                }),
            )
            .when(USE_IN_WINDOW_MENU, |this| this.child(menu_bar))
            .child(content)
            .when(USE_IN_WINDOW_MENU && self.show_file_menu, |this| {
                this.child(
                    div()
                        .absolute()
                        .left(px(8.0))
                        .top(px(29.0))
                        .w(px(230.0))
                        .p_1()
                        .occlude()
                        .bg(rgb(0x252932))
                        .text_color(rgb(0xd8dde7))
                        .border_1()
                        .border_color(rgb(0x30343d))
                        .child(
                            div()
                                .id("import-lightroom-menu-item")
                                .h(px(28.0))
                                .px_2()
                                .flex()
                                .items_center()
                                .rounded_sm()
                                .text_sm()
                                .hover(|this| this.bg(rgb(0x3b4658)))
                                .child("Import Lightroom Catalog…")
                                .on_click(cx.listener(|this, _, window, cx| {
                                    this.import_lightroom(&ImportLightroom, window, cx);
                                })),
                        )
                        .child(
                            div()
                                .id("settings-menu-item")
                                .h(px(28.0))
                                .px_2()
                                .flex()
                                .items_center()
                                .rounded_sm()
                                .text_sm()
                                .hover(|this| this.bg(rgb(0x3b4658)))
                                .child("Settings…")
                                .on_click(cx.listener(|this, _, window, cx| {
                                    this.open_settings(&OpenSettings, window, cx);
                                })),
                        ),
                )
            })
            .when(USE_IN_WINDOW_MENU && self.show_view_menu, |this| {
                this.child(
                    div()
                        .absolute()
                        .left(px(8.0))
                        .top(px(29.0))
                        .w(px(230.0))
                        .p_1()
                        .occlude()
                        .bg(rgb(0x252932))
                        .text_color(rgb(0xd8dde7))
                        .border_1()
                        .border_color(rgb(0x30343d))
                        .child(
                            div()
                                .id("toggle-job-queue-menu-item")
                                .h(px(28.0))
                                .px_2()
                                .flex()
                                .items_center()
                                .rounded_sm()
                                .text_sm()
                                .hover(|this| this.bg(rgb(0x3b4658)))
                                .child(if self.show_job_queue {
                                    "✓ Job Queue"
                                } else {
                                    "Job Queue"
                                })
                                .on_click(cx.listener(|this, _, window, cx| {
                                    this.toggle_job_queue(&ToggleJobQueue, window, cx);
                                })),
                        )
                        .child(
                            div()
                                .id("toggle-develop-inspector-menu-item")
                                .h(px(28.0))
                                .px_2()
                                .flex()
                                .items_center()
                                .rounded_sm()
                                .text_sm()
                                .hover(|this| this.bg(rgb(0x3b4658)))
                                .child(if self.show_develop_inspector {
                                    "✓ Develop Edits"
                                } else {
                                    "Develop Edits"
                                })
                                .on_click(cx.listener(|this, _, window, cx| {
                                    this.toggle_develop_inspector(
                                        &ToggleDevelopInspector,
                                        window,
                                        cx,
                                    );
                                })),
                        ),
                )
            })
            .when_some(path_context_menu, |this, (path, left, top, can_export)| {
                let export_path = path.clone();
                this.child(
                    div()
                        .absolute()
                        .left(px(left))
                        .top(px(top))
                        .w(px(220.0))
                        .p_1()
                        .occlude()
                        .bg(rgb(0x252932))
                        .text_color(rgb(0xd8dde7))
                        .border_1()
                        .border_color(rgb(0x414754))
                        .on_mouse_down(
                            MouseButton::Left,
                            cx.listener(|_, _, _, cx| cx.stop_propagation()),
                        )
                        .on_mouse_down(
                            MouseButton::Right,
                            cx.listener(|_, _, _, cx| cx.stop_propagation()),
                        )
                        .child(
                            div()
                                .id("copy-absolute-path-menu-item")
                                .h(px(28.0))
                                .px_2()
                                .flex()
                                .items_center()
                                .rounded_sm()
                                .text_sm()
                                .hover(|this| this.bg(rgb(0x3b4658)))
                                .child("Copy Absolute Path")
                                .on_mouse_down(
                                    MouseButton::Left,
                                    cx.listener(move |this, _, _, cx| {
                                        this.copy_absolute_path(&path, cx);
                                        cx.stop_propagation();
                                    }),
                                ),
                        )
                        .when(can_export, |this| {
                            this.child(
                                div()
                                    .id("export-png-menu-item")
                                    .h(px(28.0))
                                    .px_2()
                                    .flex()
                                    .items_center()
                                    .rounded_sm()
                                    .text_sm()
                                    .hover(|this| this.bg(rgb(0x3b4658)))
                                    .child("Export PNG…")
                                    .on_mouse_down(
                                        MouseButton::Left,
                                        cx.listener(move |this, _, _, cx| {
                                            this.export_image(export_path.clone(), cx);
                                            cx.stop_propagation();
                                        }),
                                    ),
                            )
                        }),
                )
            })
            .when_some(settings_dialog, |this, dialog| this.child(dialog))
    }
}

fn settings_number_row(
    label: &'static str,
    value: usize,
    field: SettingsNumber,
    cx: &mut Context<PictureBrowser>,
) -> gpui::Div {
    div()
        .flex()
        .items_center()
        .justify_between()
        .py_2()
        .border_b_1()
        .border_color(rgb(0x30343d))
        .child(label)
        .child(
            div()
                .flex()
                .items_center()
                .gap_2()
                .child(
                    div()
                        .id(SharedString::from(format!("settings-minus-{label}")))
                        .px_3()
                        .py_1()
                        .rounded_sm()
                        .bg(rgb(0x343943))
                        .hover(|this| this.bg(rgb(0x465064)))
                        .child("−")
                        .on_click(cx.listener(move |this, _, _, cx| {
                            this.adjust_setting(field, -1, cx);
                        })),
                )
                .child(div().w(px(42.0)).text_center().child(value.to_string()))
                .child(
                    div()
                        .id(SharedString::from(format!("settings-plus-{label}")))
                        .px_3()
                        .py_1()
                        .rounded_sm()
                        .bg(rgb(0x343943))
                        .hover(|this| this.bg(rgb(0x465064)))
                        .child("+")
                        .on_click(cx.listener(move |this, _, _, cx| {
                            this.adjust_setting(field, 1, cx);
                        })),
                ),
        )
}

fn render_settings_dialog(
    draft: AppSettings,
    status: Option<String>,
    cx: &mut Context<PictureBrowser>,
) -> gpui::Div {
    let preference = draft.pair_preference;
    div()
        .absolute()
        .inset_0()
        .occlude()
        .bg(gpui::rgba(0x00000099))
        .flex()
        .items_center()
        .justify_center()
        .child(
            div()
                .w(px(620.0))
                .max_h(px(680.0))
                .p_5()
                .rounded_md()
                .bg(rgb(0x252932))
                .text_color(rgb(0xd8dde7))
                .border_1()
                .border_color(rgb(0x4a5260))
                .child(div().text_xl().mb_2().child("Settings"))
                .child(
                    div()
                        .text_sm()
                        .text_color(rgb(0x9da7b6))
                        .mb_3()
                        .child("Worker, cache, and library changes take effect after restart."),
                )
                .child(
                    div()
                        .py_2()
                        .border_b_1()
                        .border_color(rgb(0x30343d))
                        .child("Picture library")
                        .child(
                            div()
                                .mt_1()
                                .flex()
                                .items_center()
                                .justify_between()
                                .child(
                                    div()
                                        .max_w(px(470.0))
                                        .truncate()
                                        .text_sm()
                                        .text_color(rgb(0xb7bfcc))
                                        .child(draft.root.to_string_lossy().into_owned()),
                                )
                                .child(
                                    div()
                                        .id("settings-browse-root")
                                        .px_3()
                                        .py_1()
                                        .rounded_sm()
                                        .bg(rgb(0x343943))
                                        .hover(|this| this.bg(rgb(0x465064)))
                                        .child("Browse…")
                                        .on_click(cx.listener(|this, _, _, cx| {
                                            this.choose_settings_directory(true, cx);
                                        })),
                                ),
                        ),
                )
                .child(settings_number_row(
                    "Image cache entries",
                    draft.image_cache_size,
                    SettingsNumber::ImageCacheSize,
                    cx,
                ))
                .child(settings_number_row(
                    "Worker threads",
                    draft.job_parallelism,
                    SettingsNumber::JobParallelism,
                    cx,
                ))
                .child(settings_number_row(
                    "RAW worker threads",
                    draft.raw_parallelism,
                    SettingsNumber::RawParallelism,
                    cx,
                ))
                .child(
                    div()
                        .flex()
                        .items_center()
                        .justify_between()
                        .py_2()
                        .border_b_1()
                        .border_color(rgb(0x30343d))
                        .child("RAW/JPEG pair preference")
                        .child(
                            div()
                                .id("settings-pair-preference")
                                .px_3()
                                .py_1()
                                .rounded_sm()
                                .bg(rgb(0x343943))
                                .hover(|this| this.bg(rgb(0x465064)))
                                .child(match preference {
                                    config::PairPreference::Raw => "RAW",
                                    config::PairPreference::Jpeg => "JPEG",
                                })
                                .on_click(cx.listener(|this, _, _, cx| {
                                    if let Some(draft) = &mut this.settings_draft {
                                        draft.pair_preference = match draft.pair_preference {
                                            config::PairPreference::Raw => {
                                                config::PairPreference::Jpeg
                                            }
                                            config::PairPreference::Jpeg => {
                                                config::PairPreference::Raw
                                            }
                                        };
                                    }
                                    cx.notify();
                                })),
                        ),
                )
                .child(
                    div().py_2().child("Cache directory").child(
                        div()
                            .mt_1()
                            .flex()
                            .items_center()
                            .justify_between()
                            .child(
                                div()
                                    .max_w(px(470.0))
                                    .truncate()
                                    .text_sm()
                                    .text_color(rgb(0xb7bfcc))
                                    .child(draft.cache_root.to_string_lossy().into_owned()),
                            )
                            .child(
                                div()
                                    .id("settings-browse-cache")
                                    .px_3()
                                    .py_1()
                                    .rounded_sm()
                                    .bg(rgb(0x343943))
                                    .hover(|this| this.bg(rgb(0x465064)))
                                    .child("Browse…")
                                    .on_click(cx.listener(|this, _, _, cx| {
                                        this.choose_settings_directory(false, cx);
                                    })),
                            ),
                    ),
                )
                .when_some(status, |this, status| {
                    this.child(
                        div()
                            .mt_2()
                            .text_sm()
                            .text_color(rgb(0xf2b8b5))
                            .child(status),
                    )
                })
                .child(
                    div()
                        .mt_4()
                        .flex()
                        .justify_end()
                        .gap_2()
                        .child(
                            div()
                                .id("settings-cancel")
                                .px_4()
                                .py_2()
                                .rounded_sm()
                                .bg(rgb(0x343943))
                                .hover(|this| this.bg(rgb(0x465064)))
                                .child("Cancel")
                                .on_click(cx.listener(|this, _, _, cx| {
                                    this.settings_draft = None;
                                    this.settings_status = None;
                                    cx.notify();
                                })),
                        )
                        .child(
                            div()
                                .id("settings-save")
                                .px_4()
                                .py_2()
                                .rounded_sm()
                                .bg(rgb(0x315b8f))
                                .hover(|this| this.bg(rgb(0x3d6fa9)))
                                .child("Save")
                                .on_click(cx.listener(|this, _, _, cx| this.save_settings(cx))),
                        ),
                ),
        )
}

fn export_dialog_directory<'a>(
    last_export_directory: Option<&'a std::path::Path>,
    source: &'a std::path::Path,
) -> &'a std::path::Path {
    last_export_directory
        .or_else(|| source.parent())
        .unwrap_or_else(|| std::path::Path::new("."))
}

fn save_preview_as_png(preview: PreviewState, destination: &std::path::Path) -> Result<()> {
    let PreviewState::Loaded(preview) = preview else {
        let PreviewState::Error(error) = preview else {
            anyhow::bail!("full-size render did not complete");
        };
        anyhow::bail!(error.to_string());
    };
    let (width, height) = preview.metadata.output_dimensions;
    let mut rgba = preview
        .image
        .as_bytes(0)
        .ok_or_else(|| anyhow::anyhow!("rendered image has no frame"))?
        .to_vec();
    bgra_to_rgba(&mut rgba);
    image::save_buffer_with_format(
        destination,
        &rgba,
        width,
        height,
        image::ColorType::Rgba8,
        image::ImageFormat::Png,
    )?;
    Ok(())
}

fn bgra_to_rgba(pixels: &mut [u8]) {
    for pixel in pixels.as_chunks_mut::<4>().0 {
        pixel.swap(0, 2);
    }
}

#[cfg(not(test))]
fn main() -> Result<()> {
    let config = config::from_env()?;
    rendering::assert_hevc_decoder_available()?;
    Application::new().run(move |cx: &mut App| {
        cx.bind_keys([
            KeyBinding::new("up", TreeUp, Some("DirectoryTree")),
            KeyBinding::new("down", TreeDown, Some("DirectoryTree")),
            KeyBinding::new("left", TreeLeft, Some("DirectoryTree")),
            KeyBinding::new("right", TreeRight, Some("DirectoryTree")),
            KeyBinding::new("enter", TreeConfirm, Some("DirectoryTree")),
            KeyBinding::new("enter", OpenImage, Some("MainView")),
            KeyBinding::new("escape", CloseImage, Some("MainView")),
            KeyBinding::new("left", GridLeft, Some("MainView")),
            KeyBinding::new("right", GridRight, Some("MainView")),
            KeyBinding::new("up", GridUp, Some("MainView")),
            KeyBinding::new("down", GridDown, Some("MainView")),
            KeyBinding::new("shift-left", GridExtendLeft, Some("MainView")),
            KeyBinding::new("shift-right", GridExtendRight, Some("MainView")),
            KeyBinding::new("shift-up", GridExtendUp, Some("MainView")),
            KeyBinding::new("shift-down", GridExtendDown, Some("MainView")),
            KeyBinding::new("ctrl-shift-j", ToggleJobQueue, None),
            KeyBinding::new("d", ToggleDeveloped, Some("MainView")),
        ]);
        cx.set_menus(vec![
            Menu {
                name: "File".into(),
                items: vec![
                    MenuItem::action("Import Lightroom Catalog…", ImportLightroom),
                    MenuItem::action("Settings…", OpenSettings),
                ],
            },
            Menu {
                name: "View".into(),
                items: vec![
                    MenuItem::action("Developed / Baseline", ToggleDeveloped),
                    MenuItem::action("Develop Edits", ToggleDevelopInspector),
                    MenuItem::action("Job Queue", ToggleJobQueue),
                ],
            },
        ]);
        let bounds = Bounds::centered(None, size(px(1100.0), px(720.0)), cx);
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                ..Default::default()
            },
            |_, cx| cx.new(|cx| PictureBrowser::new(&config, cx)),
        )
        .expect("failed to open window");
        cx.activate(true);
    });
    Ok(())
}

#[cfg(test)]
test_r::enable!();

#[cfg(test)]
#[path = "main/tests.rs"]
mod tests;
