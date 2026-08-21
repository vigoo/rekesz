use std::{
    collections::{BTreeSet, HashMap, HashSet},
    env,
    ffi::OsString,
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
    MouseDownEvent, MouseMoveEvent, MouseUpEvent, Render, ScrollStrategy, SharedString,
    UniformListScrollHandle, Window, WindowBounds, WindowOptions, actions, canvas, div, img, point,
    prelude::*, px, relative, rgb, size, uniform_list,
};

mod discovery;
mod formats;
mod jobs;
mod metadata;
mod preview;
mod rendering;
mod scheduler;

use discovery::{
    DirectoryContents, DirectoryMetadataStatus, DirectoryModel, DirectoryScanPriority,
    DirectoryScanner, NodeId, RawJpegPreference,
};
use jobs::JobLimiter;
use metadata::MetadataLoader;
use preview::{PreviewCache, PreviewState, RenderPipeline};
use rendering::RenderVariant;
use scheduler::{
    DemandScopeId, JobCompletion, JobDemand, JobKey, JobPriority, JobSnapshotStatus,
    RenderScheduler, SchedulerSnapshot,
};

actions!(pics, [ToggleJobQueue]);

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
const IDLE_PREFETCH_DELAY: Duration = Duration::from_millis(400);
const IDLE_PREFETCH_BATCH: usize = 64;
const JOB_QUEUE_PANEL_WIDTH: f32 = 430.0;
const SCHEDULER_REFRESH_INTERVAL: Duration = Duration::from_millis(100);
const USE_IN_WINDOW_MENU: bool = !cfg!(target_os = "macos");

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
}

impl ImageViewState {
    fn new(path: PathBuf) -> Self {
        Self {
            path,
            zoom: ZoomLevel::FitWindow,
            pan: (0.0, 0.0),
            drag_last: None,
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
}

impl WorkProgress {
    fn start(&mut self) {
        if self.active == 0 {
            self.completed = 0;
            self.total = 0;
        }
        self.active += 1;
        self.total += 1;
    }

    fn finish(&mut self) {
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
    scheduler: RenderScheduler,
    directory_scanner: DirectoryScanner,
    grid_demand_epoch: u64,
    grid_prediction: Option<GridPredictionState>,
    scroll_direction: i8,
    idle_prefetch_batch: usize,
    prefetch_preview_variant: RenderVariant,
    focused_prefetch_epoch: u64,
    pending_grid_demand: Option<PendingGridDemand>,
    grid_publish_scheduled: bool,
    show_job_queue: bool,
    show_view_menu: bool,
    job_queue_snapshot: SchedulerSnapshot,
    job_queue_scroll: UniformListScrollHandle,
    scheduler_refresh_scheduled: bool,
    tree_context_menu: Option<TreeContextMenu>,
    metadata_loader: MetadataLoader,
    metadata_load: Option<(NodeId, discovery::RequestId, Arc<AtomicBool>)>,
    scrollbar_drag: Option<ScrollbarDrag>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ScrollbarTarget {
    Tree,
    Grid,
}

#[derive(Clone, Copy, Debug)]
struct ScrollbarDrag {
    target: ScrollbarTarget,
    pointer_offset: gpui::Pixels,
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

struct TreeContextMenu {
    node: NodeId,
    position: (f32, f32),
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

    fn new(
        root: PathBuf,
        image_cache_size: usize,
        pair_preference: RawJpegPreference,
        cache_root: PathBuf,
        job_parallelism: usize,
        raw_parallelism: usize,
        cx: &mut Context<Self>,
    ) -> Self {
        let model = DirectoryModel::new(root);
        let root = model.root();
        let root_path = model.node(root).path.to_path_buf();
        let job_limiter = JobLimiter::new(job_parallelism);
        let render_pipeline = RenderPipeline::with_source_cache_capacity(
            cache_root.clone(),
            job_parallelism,
            raw_parallelism,
            image_cache_size,
        );
        let metadata_loader =
            MetadataLoader::new(cache_root.clone(), job_parallelism, job_limiter.clone());
        let render_job_limiter = job_limiter.clone();
        let scheduler = RenderScheduler::new(
            job_parallelism,
            Arc::new(move |key, priority, cancelled| {
                let Some(_permit) = render_job_limiter.acquire(
                    priority.class >= scheduler::PriorityClass::Visible,
                    cancelled,
                ) else {
                    return PreviewState::Error("render cancelled".into());
                };
                render_pipeline.render(
                    key.path.clone(),
                    key.variant,
                    priority.class == scheduler::PriorityClass::Interactive,
                    priority.class < scheduler::PriorityClass::Visible,
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
            previews: PreviewCache::new(image_cache_size),
            raw_previews: PreviewCache::new(image_cache_size),
            thumbnails: PreviewCache::new(image_cache_size.saturating_mul(16).max(64)),
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
            pair_preference,
            work: WorkProgress::default(),
            scheduler,
            directory_scanner: DirectoryScanner::new(job_parallelism, job_limiter.clone()),
            grid_demand_epoch: 0,
            grid_prediction: None,
            scroll_direction: 1,
            idle_prefetch_batch: 0,
            prefetch_preview_variant: RenderVariant::Preview {
                max_width: 1920,
                max_height: 1080,
            },
            focused_prefetch_epoch: 0,
            pending_grid_demand: None,
            grid_publish_scheduled: false,
            show_job_queue: false,
            show_view_menu: false,
            job_queue_snapshot: SchedulerSnapshot::default(),
            job_queue_scroll: UniformListScrollHandle::new(),
            scheduler_refresh_scheduled: false,
            tree_context_menu: None,
            metadata_loader,
            metadata_load: None,
            scrollbar_drag: None,
        };
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
        self.work.start();
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
                this.work.finish();
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
        self.work.start();
        cx.notify();
        let loader = self.metadata_loader.clone();
        let task_cancelled = cancelled.clone();
        let load = cx.background_spawn(async move { loader.load_directory(paths, task_cancelled) });
        cx.spawn(async move |this, cx| {
            let metadata = load.await;
            this.update(cx, |this, cx| {
                this.work.finish();
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
        for (path, variant, priority) in requested {
            if !self.cache_for_variant(variant).needs_work(&path, variant) {
                continue;
            }
            let request = self.cache_for_variant_mut(variant).begin(&path, variant);
            let key = JobKey { path, variant };
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
            let JobKey { path, variant } = subscription.key;
            if track_progress {
                self.work.start();
                progress_started = true;
            }
            cx.spawn(async move |this, cx| {
                let completion = subscription.receiver.await;
                this.update(cx, |this, cx| {
                    if track_progress {
                        this.work.finish();
                    }
                    match completion {
                        Ok(JobCompletion::Finished(result)) => {
                            if let PreviewState::Loaded(preview) = &result {
                                this.model
                                    .record_render_metadata(path.clone(), preview.metadata);
                            }
                            this.cache_for_variant_mut(variant)
                                .finish(&path, request, result);
                        }
                        Ok(JobCompletion::Cancelled) | Err(_) => {
                            this.cache_for_variant_mut(variant).cancel(&path, request);
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
        self.grid_prediction = None;
        self.idle_prefetch_batch = 0;
        self.scheduler.clear_scope(NEAR_GRID_SCOPE);
        self.scheduler.clear_scope(IDLE_PREFETCH_SCOPE);
        self.scheduler.clear_scope(FOCUSED_PREFETCH_SCOPE);
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
        self.main_view = MainView::Image(ImageViewState::new(path.clone()));
        // Keep thumbnail and prediction demand alive. The image request is
        // Interactive, so the scheduler promotes it above this retained work
        // and cooperatively preempts lower-priority running renders. Returning
        // to the grid can then resume instead of rebuilding an erased queue.
        self.previews.mark_used(&path);
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

        if let PreviewState::Loaded(preview) = self.previews.state(&state.path) {
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
                    let image_content = match self.thumbnails.state(&path) {
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
                    let open_path = path.clone();
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
                            .child(div().mt_2().truncate().text_xs().child(name))
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
                            this.tree_context_menu = Some(TreeContextMenu {
                                node: select_node,
                                position: (
                                    f32::from(event.position.x),
                                    f32::from(event.position.y),
                                ),
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
        let foreground_jobs = self.job_queue_snapshot.foreground_count();
        if self.show_job_queue || foreground_jobs > 0 || self.work.active > 0 {
            self.schedule_scheduler_refresh(cx);
        }
        let inspector_width = if self.show_job_queue {
            JOB_QUEUE_PANEL_WIDTH
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

        let (title, show_back, main_content) = match &self.main_view {
            MainView::Image(state) => {
                let path = state.path.clone();
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
                    RenderVariant::Preview { .. } => self.raw_previews.state(&path),
                    RenderVariant::Full => self.previews.state(&path),
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
                        .on_mouse_move(cx.listener(Self::image_mouse_move))
                        .on_mouse_up(MouseButton::Left, cx.listener(Self::image_mouse_up))
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
                    .child(title),
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
        let progress_active = self.work.active > 0 || foreground_jobs > 0;
        let progress = if self.work.active > 0 {
            self.work.fraction().clamp(0.08, 1.0)
        } else {
            0.35
        };
        let progress_label = format!(
            "Working… {} foreground job{}",
            foreground_jobs.max(self.work.active),
            if foreground_jobs.max(self.work.active) == 1 {
                ""
            } else {
                "s"
            }
        );
        let menu_bar = div()
            .h(px(30.0))
            .w_full()
            .flex_none()
            .flex()
            .items_center()
            .px_2()
            .bg(rgb(0x20232a))
            .border_b_1()
            .border_color(rgb(0x30343d))
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
        let tree_context_menu = self.tree_context_menu.as_ref().map(|menu| {
            let viewport = window.viewport_size();
            let left = menu
                .position
                .0
                .min((f32::from(viewport.width) - 228.0).max(0.0));
            let top = menu
                .position
                .1
                .min((f32::from(viewport.height) - 38.0).max(0.0));
            (self.model.node(menu.node).path.to_path_buf(), left, top)
        });
        div()
            .size_full()
            .relative()
            .flex()
            .flex_col()
            .key_context("PictureBrowser")
            .on_action(cx.listener(Self::toggle_job_queue))
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(|this, _, _, cx| {
                    if this.tree_context_menu.take().is_some() {
                        cx.notify();
                    }
                }),
            )
            .on_mouse_down(
                MouseButton::Right,
                cx.listener(|this, _, _, cx| {
                    if this.tree_context_menu.take().is_some() {
                        cx.notify();
                    }
                }),
            )
            .when(USE_IN_WINDOW_MENU, |this| this.child(menu_bar))
            .child(content)
            .when(USE_IN_WINDOW_MENU && self.show_view_menu, |this| {
                this.child(
                    div()
                        .absolute()
                        .left(px(8.0))
                        .top(px(29.0))
                        .w(px(230.0))
                        .p_1()
                        .bg(rgb(0x252932))
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
                        ),
                )
            })
            .when_some(tree_context_menu, |this, (path, left, top)| {
                this.child(
                    div()
                        .absolute()
                        .left(px(left))
                        .top(px(top))
                        .w(px(220.0))
                        .p_1()
                        .bg(rgb(0x252932))
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
                                .id("copy-tree-path-menu-item")
                                .h(px(28.0))
                                .px_2()
                                .flex()
                                .items_center()
                                .rounded_sm()
                                .text_sm()
                                .hover(|this| this.bg(rgb(0x3b4658)))
                                .child("Copy Absolute Path")
                                .on_click(cx.listener(move |this, _, _, cx| {
                                    cx.write_to_clipboard(ClipboardItem::new_string(
                                        path.to_string_lossy().into_owned(),
                                    ));
                                    this.tree_context_menu = None;
                                    cx.notify();
                                })),
                        ),
                )
            })
    }
}

const DEFAULT_IMAGE_CACHE_SIZE: usize = 8;
const RAW_MEMORY_RESERVE_BYTES: u64 = 1024 * 1024 * 1024;
const ESTIMATED_RAW_WORKING_SET_BYTES: u64 = 768 * 1024 * 1024;

struct Config {
    root: PathBuf,
    image_cache_size: usize,
    pair_preference: RawJpegPreference,
    cache_root: PathBuf,
    job_parallelism: usize,
    raw_parallelism: usize,
}

fn config_from_args(args: impl IntoIterator<Item = OsString>) -> Result<Config> {
    let mut args = args.into_iter();
    let _program = args.next();
    let mut requested_root = None;
    let mut image_cache_size = DEFAULT_IMAGE_CACHE_SIZE;
    let mut pair_preference = RawJpegPreference::Raw;
    let mut requested_cache_root = None;
    let mut job_parallelism = std::thread::available_parallelism()
        .map(|count| count.get().min(8))
        .unwrap_or(1);
    let mut requested_raw_parallelism = None;
    while let Some(argument) = args.next() {
        if argument == "--image-cache-size" {
            let value = args
                .next()
                .ok_or_else(|| anyhow::anyhow!("--image-cache-size requires a value"))?;
            image_cache_size = value
                .to_string_lossy()
                .parse::<usize>()
                .map_err(|_| anyhow::anyhow!("image cache size must be a positive integer"))?;
            if image_cache_size == 0 {
                anyhow::bail!("image cache size must be at least 1");
            }
        } else if argument == "--job-parallelism" || argument == "--render-parallelism" {
            let value = args
                .next()
                .ok_or_else(|| anyhow::anyhow!("--job-parallelism requires a value"))?;
            job_parallelism = value
                .to_string_lossy()
                .parse::<usize>()
                .map_err(|_| anyhow::anyhow!("job parallelism must be a positive integer"))?;
            if job_parallelism == 0 {
                anyhow::bail!("job parallelism must be at least 1");
            }
        } else if argument == "--raw-parallelism" {
            let value = args
                .next()
                .ok_or_else(|| anyhow::anyhow!("--raw-parallelism requires a value"))?;
            let raw_parallelism = value
                .to_string_lossy()
                .parse::<usize>()
                .map_err(|_| anyhow::anyhow!("RAW parallelism must be a positive integer"))?;
            if raw_parallelism == 0 {
                anyhow::bail!("RAW parallelism must be at least 1");
            }
            requested_raw_parallelism = Some(raw_parallelism);
        } else if argument == "--cache-root" {
            requested_cache_root =
                Some(PathBuf::from(args.next().ok_or_else(|| {
                    anyhow::anyhow!("--cache-root requires a directory")
                })?));
        } else if argument == "--cr3-jpeg-preference" {
            let value = args
                .next()
                .ok_or_else(|| anyhow::anyhow!("--cr3-jpeg-preference requires cr3 or jpeg"))?;
            pair_preference = match value.to_string_lossy().to_ascii_lowercase().as_str() {
                "cr3" | "raw" => RawJpegPreference::Raw,
                "jpg" | "jpeg" => RawJpegPreference::Jpeg,
                _ => anyhow::bail!("--cr3-jpeg-preference must be cr3 or jpeg"),
            };
        } else if requested_root.is_none() {
            requested_root = Some(PathBuf::from(argument));
        } else {
            anyhow::bail!("unexpected argument: {}", argument.to_string_lossy());
        }
    }
    let requested = requested_root.unwrap_or(env::current_dir()?);
    let root = if requested.is_absolute() {
        requested
    } else {
        env::current_dir()?.join(requested)
    };
    let cache_root = requested_cache_root.unwrap_or(
        directories::ProjectDirs::from("", "", "pics")
            .ok_or_else(|| anyhow::anyhow!("cannot determine the platform cache directory"))?
            .cache_dir()
            .to_path_buf(),
    );
    let cache_root = if cache_root.is_absolute() {
        cache_root
    } else {
        env::current_dir()?.join(cache_root)
    };
    let raw_parallelism = requested_raw_parallelism
        .unwrap_or_else(|| default_raw_parallelism(job_parallelism))
        .min(job_parallelism);
    Ok(Config {
        root,
        image_cache_size,
        pair_preference,
        cache_root,
        job_parallelism,
        raw_parallelism,
    })
}

fn default_raw_parallelism(job_parallelism: usize) -> usize {
    let system = sysinfo::System::new_all();
    recommended_raw_parallelism(
        system.cpus().len(),
        system.available_memory(),
        job_parallelism,
    )
}

fn recommended_raw_parallelism(
    logical_cpus: usize,
    available_memory_bytes: u64,
    job_parallelism: usize,
) -> usize {
    let cpu_limit = logical_cpus.max(1).div_ceil(2);
    let memory_limit = available_memory_bytes
        .saturating_sub(RAW_MEMORY_RESERVE_BYTES)
        .checked_div(ESTIMATED_RAW_WORKING_SET_BYTES)
        .unwrap_or(0)
        .max(1) as usize;
    cpu_limit.min(memory_limit).min(job_parallelism).max(1)
}

fn main() -> Result<()> {
    let config = config_from_args(env::args_os())?;
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
        ]);
        cx.set_menus(vec![Menu {
            name: "View".into(),
            items: vec![MenuItem::action("Job Queue", ToggleJobQueue)],
        }]);
        let bounds = Bounds::centered(None, size(px(1100.0), px(720.0)), cx);
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                ..Default::default()
            },
            |_, cx| {
                cx.new(|cx| {
                    PictureBrowser::new(
                        config.root.clone(),
                        config.image_cache_size,
                        config.pair_preference,
                        config.cache_root.clone(),
                        config.job_parallelism,
                        config.raw_parallelism,
                        cx,
                    )
                })
            },
        )
        .expect("failed to open window");
        cx.activate(true);
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use discovery::DirectoryScan;
    use futures::executor::block_on;
    use std::time::Instant;

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
        assert_eq!(visible_rows(&model, &expanded).len(), 3);
        expanded.remove(&child);
        assert_eq!(visible_rows(&model, &expanded).len(), 2);
        assert_eq!(model.loaded(child).unwrap().children.len(), 1);
    }

    #[test]
    fn parses_image_cache_size() {
        let config = config_from_args([
            OsString::from("pics"),
            OsString::from("--image-cache-size"),
            OsString::from("3"),
            OsString::from("/pictures"),
        ])
        .unwrap();
        assert_eq!(config.image_cache_size, 3);
        assert_eq!(config.root, PathBuf::from("/pictures"));
        assert_eq!(config.pair_preference, RawJpegPreference::Raw);
    }

    #[test]
    fn parses_jpeg_pair_preference() {
        let config = config_from_args([
            OsString::from("pics"),
            OsString::from("--cr3-jpeg-preference"),
            OsString::from("jpeg"),
        ])
        .unwrap();
        assert_eq!(config.pair_preference, RawJpegPreference::Jpeg);
    }

    #[test]
    fn parses_custom_cache_root() {
        let config = config_from_args([
            OsString::from("pics"),
            OsString::from("--cache-root"),
            OsString::from("relative-cache"),
        ])
        .unwrap();
        assert!(config.cache_root.is_absolute());
        assert!(config.cache_root.ends_with("relative-cache"));
    }

    #[test]
    fn parses_job_parallelism() {
        let config = config_from_args([
            OsString::from("pics"),
            OsString::from("--job-parallelism"),
            OsString::from("3"),
        ])
        .unwrap();
        assert_eq!(config.job_parallelism, 3);
    }

    #[test]
    fn parses_and_bounds_raw_parallelism() {
        let config = config_from_args([
            OsString::from("pics"),
            OsString::from("--job-parallelism"),
            OsString::from("3"),
            OsString::from("--raw-parallelism"),
            OsString::from("8"),
        ])
        .unwrap();
        assert_eq!(config.raw_parallelism, 3);
    }

    #[test]
    fn automatic_raw_parallelism_respects_cpu_memory_and_job_limits() {
        assert_eq!(recommended_raw_parallelism(16, 16 * 1024_u64.pow(3), 8), 8);
        assert_eq!(recommended_raw_parallelism(16, 2 * 1024_u64.pow(3), 8), 1);
        assert_eq!(recommended_raw_parallelism(4, 16 * 1024_u64.pow(3), 8), 2);
        assert_eq!(recommended_raw_parallelism(32, 16 * 1024_u64.pow(3), 3), 3);
    }

    #[test]
    fn rejects_zero_job_parallelism() {
        assert!(
            config_from_args([
                OsString::from("pics"),
                OsString::from("--job-parallelism"),
                OsString::from("0"),
            ])
            .is_err()
        );
    }

    #[test]
    fn rejects_zero_raw_parallelism() {
        assert!(
            config_from_args([
                OsString::from("pics"),
                OsString::from("--raw-parallelism"),
                OsString::from("0"),
            ])
            .is_err()
        );
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
        work.start();
        work.start();
        work.finish();
        assert_eq!(work.active, 1);
        assert_eq!(work.fraction(), 0.5);
        work.finish();
        work.start();
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
        let metadata_loader =
            MetadataLoader::new(cache.path().join("metadata"), 2, limiter.clone());
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
            RenderPipeline::with_source_cache_capacity(cache.path().join("renders"), 2, 2, 8);
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
            let pipeline = RenderPipeline::with_source_cache_capacity(
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
                RenderPipeline::with_source_cache_capacity(cache.path().to_path_buf(), 1, 1, 1);
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
}
