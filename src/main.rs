// A custom test harness does not apply libtest's implicit dead-code and
// unused-import allowances to the binary under test.
#![cfg_attr(test, allow(dead_code, unused_imports))]

use std::{
    cell::{Cell, RefCell},
    collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque},
    hash::{Hash, Hasher},
    ops::Range,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::Duration,
};

use anyhow::Result;
use gpui::{
    AnyElement, App, AppContext as _, Application, Bounds, ClickEvent, ClipboardItem, Context,
    CursorStyle, FocusHandle, KeyBinding, ListSizingBehavior, Menu, MenuItem, MouseButton,
    MouseDownEvent, MouseMoveEvent, MouseUpEvent, PathPromptOptions, Render, ScrollStrategy,
    ScrollWheelEvent, SharedString, StatefulInteractiveElement, UniformListScrollHandle, Window,
    WindowBounds, WindowOptions, actions, canvas, div, img, point, prelude::*, px, relative, rgb,
    size, uniform_list,
};

mod accessibility;
mod catalog;
mod config;
mod discovery;
mod face_engine;
mod formats;
mod jobs;
mod map;
mod metadata;
mod preview;
mod rendering;
mod scheduler;

static MAP_RENDER_BUSY: AtomicBool = AtomicBool::new(false);
static MAP_CLUSTER_BUSY: AtomicBool = AtomicBool::new(false);
const MAP_RENDER_OVERSCAN: u32 = 128;
const FACE_OS_MEMORY_RESERVE: u64 = 512 * 1024 * 1024;

fn admitted_face_workers(available: u64, per_worker: u64, requested: usize) -> usize {
    let reserve = FACE_OS_MEMORY_RESERVE.saturating_add(available / 5);
    let usable = available.saturating_sub(reserve);
    usize::try_from(usable / per_worker.max(1))
        .unwrap_or(usize::MAX)
        .min(requested)
}

struct MapRenderGuard;

impl Drop for MapRenderGuard {
    fn drop(&mut self) {
        MAP_RENDER_BUSY.store(false, Ordering::Release);
    }
}

struct MapClusterGuard;

impl Drop for MapClusterGuard {
    fn drop(&mut self) {
        MAP_CLUSTER_BUSY.store(false, Ordering::Release);
    }
}

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
    rekesz,
    [
        ToggleDeveloped,
        ToggleJobQueue,
        ToggleDevelopInspector,
        OpenCacheStatistics,
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
const MAP_CARD_SCOPE: DemandScopeId = DemandScopeId(7);
const PEOPLE_SCOPE: DemandScopeId = DemandScopeId(8);
#[cfg(test)]
static PEOPLE_QUERY_TEST_DELAY: std::sync::Mutex<Option<(PathBuf, Duration)>> =
    std::sync::Mutex::new(None);
const MAP_GALLERY_PAGE_SIZE: usize = 6;
const IDLE_PREFETCH_DELAY: Duration = Duration::from_millis(400);
const IDLE_PREFETCH_BATCH: usize = 64;
const IMAGE_PREFETCH_COUNT: usize = 4;
const JOB_QUEUE_PANEL_WIDTH: f32 = 430.0;
const DEVELOP_PANEL_WIDTH: f32 = 320.0;
const SCHEDULER_REFRESH_INTERVAL: Duration = Duration::from_millis(100);
const FULL_IDENTITY_VERIFICATION_INTERVAL: u64 = 1_440;
const USE_IN_WINDOW_MENU: bool = cfg!(target_os = "linux");
const IMAGE_WHEEL_THRESHOLD_PX: f32 = 30.0;
const HIDDEN_NODE: NodeId = NodeId(usize::MAX);

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct RefreshThrottle {
    in_flight: bool,
    dirty: bool,
}

impl RefreshThrottle {
    fn request(&mut self) -> bool {
        if self.in_flight {
            self.dirty = true;
            false
        } else {
            self.in_flight = true;
            true
        }
    }

    fn complete(&mut self) -> bool {
        self.in_flight = false;
        if self.dirty {
            self.dirty = false;
            self.in_flight = true;
            true
        } else {
            false
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct MaintenanceCoordinator {
    refill_active: Duration,
    refill_error: Duration,
    inventory_watch: Duration,
    inventory_error: Duration,
    throughput_window: Duration,
}

type MaintenanceBatch = Vec<(face_engine::FaceAnalyzeJob, i32)>;
type MaintenanceRefill = (MaintenanceBatch, Option<Duration>);

#[derive(Default)]
struct FaceInventoryProgress {
    stage: AtomicUsize,
    discovered: AtomicUsize,
    processed: AtomicUsize,
    scan_directories: AtomicUsize,
    scan_pending: AtomicUsize,
    scan_images: AtomicUsize,
    current_file: Mutex<Option<PathBuf>>,
}

impl FaceInventoryProgress {
    fn label(&self) -> Option<String> {
        let total = self.discovered.load(Ordering::Acquire);
        let processed = self.processed.load(Ordering::Acquire).min(total);
        let file = self
            .current_file
            .lock()
            .ok()
            .and_then(|path| path.clone())
            .and_then(|path| {
                path.file_name()
                    .map(|name| name.to_string_lossy().into_owned())
            })
            .unwrap_or_else(|| "preparing…".to_owned());
        match self.stage.load(Ordering::Acquire) {
            1 => {
                let directories = self.scan_directories.load(Ordering::Acquire);
                let pending = self.scan_pending.load(Ordering::Acquire);
                let images = self.scan_images.load(Ordering::Acquire);
                Some(format!(
                    "Scanning library · {directories} folders scanned · {pending} queued · {images} images found · {file}"
                ))
            }
            2 => Some(format!(
                "Checking image identity {processed}/{total} · {file}"
            )),
            3 => Some(format!(
                "Updating face-analysis inventory {processed}/{total} · {file}"
            )),
            4 => Some(format!(
                "Prioritizing pending face analysis across {total} images…"
            )),
            _ => None,
        }
    }

    fn fraction(&self) -> Option<f32> {
        if self.stage.load(Ordering::Acquire) == 1 {
            let scanned = self.scan_directories.load(Ordering::Acquire);
            let pending = self.scan_pending.load(Ordering::Acquire);
            return (scanned + pending > 0).then_some(scanned as f32 / (scanned + pending) as f32);
        }
        let total = self.discovered.load(Ordering::Acquire);
        let processed = self.processed.load(Ordering::Acquire).min(total);
        (total > 0 && matches!(self.stage.load(Ordering::Acquire), 2 | 3))
            .then_some(processed as f32 / total as f32)
    }
}

trait MaintenanceDriver: Send + Sync {
    fn inventory(
        &self,
        root: &Path,
        hidden: &HashSet<PathBuf>,
        pair_preference: discovery::RawJpegPreference,
        preferred_subtree: &Path,
        recent_assets: &[PathBuf],
        coordination: InventoryCoordination<'_>,
    ) -> Result<MaintenanceBatch>;

    fn refill(
        &self,
        root: &Path,
        preferred_subtree: &Path,
        recent_assets: &[PathBuf],
    ) -> Result<MaintenanceRefill>;
}

#[derive(Clone, Copy)]
struct InventoryCoordination<'a> {
    progress: &'a FaceInventoryProgress,
    admission: &'a scheduler::MaintenanceAdmission,
    cancelled: &'a AtomicBool,
    identity_mode: catalog::InventoryIdentityMode,
}

fn owns_inventory_progress(current_generation: u64, completed_generation: u64) -> bool {
    current_generation == completed_generation
}

fn claim_inventory_run(active: &mut bool) -> bool {
    if *active {
        false
    } else {
        *active = true;
        true
    }
}

struct ProductionMaintenanceDriver;

impl MaintenanceDriver for ProductionMaintenanceDriver {
    fn inventory(
        &self,
        root: &Path,
        hidden: &HashSet<PathBuf>,
        pair_preference: discovery::RawJpegPreference,
        preferred_subtree: &Path,
        recent_assets: &[PathBuf],
        coordination: InventoryCoordination<'_>,
    ) -> Result<MaintenanceBatch> {
        let InventoryCoordination {
            progress,
            admission,
            cancelled,
            identity_mode,
        } = coordination;
        // Keep the calibrated inference module's legacy inventory entry point
        // linked without changing that source file's release-bound digest.
        let _calibrated_inventory_entry_point = face_engine::sync_analysis_inventory;
        progress.stage.store(1, Ordering::Release);
        let paths = discovery::scan_library_inventory_with_progress(
            root,
            hidden,
            pair_preference,
            cancelled,
            |directories, pending, images, path| {
                progress
                    .scan_directories
                    .store(directories, Ordering::Release);
                progress.scan_pending.store(pending, Ordering::Release);
                progress.scan_images.store(images, Ordering::Release);
                if let Ok(mut current) = progress.current_file.lock() {
                    *current = Some(path.to_path_buf());
                }
                admission.checkpoint(cancelled);
            },
        )?;
        progress.discovered.store(paths.len(), Ordering::Release);
        progress.processed.store(0, Ordering::Release);
        progress.stage.store(2, Ordering::Release);
        catalog::sync_asset_inventory_with_progress(
            root,
            &paths,
            "yunet-2023mar+sface-2021dec",
            "sface-2021dec",
            identity_mode,
            |processed, total, path| {
                let previous = progress.processed.swap(processed, Ordering::AcqRel);
                if processed < previous {
                    progress.stage.store(3, Ordering::Release);
                }
                progress.discovered.store(total, Ordering::Release);
                if let Ok(mut current) = progress.current_file.lock() {
                    *current = Some(path.to_path_buf());
                }
            },
            || admission.checkpoint(cancelled),
        )?;
        progress.stage.store(4, Ordering::Release);
        face_engine::pending_analysis_jobs_prioritized(
            root,
            10_000,
            Some(preferred_subtree),
            recent_assets,
        )
    }

    fn refill(
        &self,
        root: &Path,
        preferred_subtree: &Path,
        recent_assets: &[PathBuf],
    ) -> Result<MaintenanceRefill> {
        Ok((
            face_engine::pending_analysis_jobs_prioritized(
                root,
                10_000,
                Some(preferred_subtree),
                recent_assets,
            )?,
            face_engine::next_analysis_wake(root)?,
        ))
    }
}

impl MaintenanceCoordinator {
    const PRODUCTION: Self = Self {
        refill_active: Duration::from_secs(2),
        refill_error: Duration::from_secs(30),
        inventory_watch: Duration::from_secs(60),
        inventory_error: Duration::from_secs(120),
        // The scheduler uses a sixty-second rolling window. The extra second
        // avoids waking on its inclusive boundary.
        throughput_window: Duration::from_secs(61),
    };

    fn refill_after(
        self,
        paused: bool,
        had_jobs: bool,
        retry_deadline: Option<Duration>,
        failed: bool,
    ) -> Option<Duration> {
        if paused {
            None
        } else if failed {
            Some(self.refill_error)
        } else if had_jobs {
            Some(self.refill_active)
        } else {
            retry_deadline
        }
    }

    fn inventory_after(self, paused: bool, failed: bool) -> Option<Duration> {
        (!paused).then_some(if failed {
            self.inventory_error
        } else {
            self.inventory_watch
        })
    }

    fn throughput_after(self, completed_last_minute: usize) -> Option<Duration> {
        (completed_last_minute > 0).then_some(self.throughput_window)
    }
}

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

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum DirectoryViewMode {
    #[default]
    Grid,
    Map,
    People,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum PeopleDecision {
    Confirm,
    Reject,
    NotFace,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum PeopleNameOperation {
    Rename { person_id: i64 },
    Create { face_ids: Vec<i64> },
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum PeopleTargetOperation {
    Merge {
        source_person_id: i64,
    },
    Move {
        source_person_id: i64,
        face_ids: Vec<i64>,
    },
    Assign {
        face_ids: Vec<i64>,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum PeopleOperationDialog {
    Name {
        operation: PeopleNameOperation,
        input: String,
        cursor: usize,
        selection_anchor: Option<usize>,
    },
    Target {
        operation: PeopleTargetOperation,
        target_person_id: Option<i64>,
    },
}

#[derive(Clone, Debug)]
enum PeopleMergePreviewLoad {
    Loading(i64),
    Loaded(Arc<catalog::people::MergePeoplePreview>),
    Error(i64, Arc<str>),
}

fn prepare_selected_face_ids(selection: &PeopleFaceSelection) -> Result<Vec<i64>, String> {
    let face_ids = selection
        .selected
        .iter()
        .map(|key| key.face_id)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    if face_ids.is_empty() {
        Err("Select at least one face first".to_owned())
    } else {
        Ok(face_ids)
    }
}

fn validate_people_operation_dialog(dialog: &PeopleOperationDialog) -> Result<(), String> {
    match dialog {
        PeopleOperationDialog::Name { input, .. } => {
            let count = input.trim().chars().count();
            if (1..=200).contains(&count) {
                Ok(())
            } else {
                Err("A person name must contain 1 to 200 characters".to_owned())
            }
        }
        PeopleOperationDialog::Target {
            operation,
            target_person_id,
        } => {
            let target = target_person_id.ok_or_else(|| "Choose a target person".to_owned())?;
            let source = match operation {
                PeopleTargetOperation::Merge { source_person_id }
                | PeopleTargetOperation::Move {
                    source_person_id, ..
                } => Some(*source_person_id),
                PeopleTargetOperation::Assign { .. } => None,
            };
            if source == Some(target) {
                Err("Source and target people must differ".to_owned())
            } else {
                Ok(())
            }
        }
    }
}

fn people_editor_range(cursor: usize, anchor: Option<usize>) -> Range<usize> {
    let anchor = anchor.unwrap_or(cursor);
    anchor.min(cursor)..anchor.max(cursor)
}

fn people_editor_replace(
    input: &mut String,
    cursor: &mut usize,
    anchor: &mut Option<usize>,
    replacement: &str,
) {
    let mut chars = input.chars().collect::<Vec<_>>();
    let range = people_editor_range(*cursor, *anchor);
    let available = 200usize.saturating_sub(chars.len().saturating_sub(range.len()));
    let replacement = replacement.chars().take(available).collect::<Vec<_>>();
    chars.splice(range.clone(), replacement.iter().copied());
    *cursor = range.start + replacement.len();
    *anchor = None;
    *input = chars.into_iter().collect();
}

fn people_editor_selected_text(input: &str, cursor: usize, anchor: Option<usize>) -> String {
    let range = people_editor_range(cursor, anchor);
    input.chars().skip(range.start).take(range.len()).collect()
}

fn people_editor_display(input: &str, cursor: usize, anchor: Option<usize>) -> String {
    let chars = input.chars().collect::<Vec<_>>();
    let range = people_editor_range(cursor, anchor);
    let mut output = String::new();
    for (index, character) in chars.iter().enumerate() {
        if index == cursor {
            output.push('▏');
        }
        if range.contains(&index) {
            output.push('‹');
            output.push(*character);
            output.push('›');
        } else {
            output.push(*character);
        }
    }
    if cursor == chars.len() {
        output.push('▏');
    }
    output
}

fn people_merge_preview_text(preview: &catalog::people::MergePeoplePreview) -> String {
    let resulting_confirmed = preview
        .source_confirmed_faces
        .saturating_add(preview.target_confirmed_faces)
        .saturating_sub(preview.confirmation_deduplications);
    format!(
        "Result: {resulting_confirmed} confirmed faces. {} duplicate confirmations and {} duplicate assignments collapse; {} external catalog link(s) move. {}",
        preview.confirmation_deduplications,
        preview.assignment_deduplications,
        preview.external_links_moved,
        if preview.source_representative_will_be_discarded {
            "The target example face is kept and the source example is discarded."
        } else if preview.resulting_representative_face_id.is_some() {
            "The available example face is kept."
        } else {
            "No example face will be set."
        }
    )
}

fn should_start_automatic_face_inventory(settings: &AppSettings) -> bool {
    settings.automatic_face_analysis && settings.face_workers > 0
}

fn validate_people_decision_batch(decisions: &[(i64, i64, PeopleDecision)]) -> Result<(), String> {
    let mut confirmations = HashMap::new();
    for (face_id, person_id, decision) in decisions {
        if *decision != PeopleDecision::Confirm {
            continue;
        }
        if let Some(previous) = confirmations.insert(*face_id, *person_id)
            && previous != *person_id
        {
            return Err(format!(
                "Face {face_id} has multiple selected identities. Select only one person to accept."
            ));
        }
    }
    Ok(())
}

fn can_start_people_review_save(undo_pending: bool) -> bool {
    !undo_pending
}

fn can_start_people_undo(undo_pending: bool, pending_saves: usize) -> bool {
    !undo_pending && pending_saves == 0
}

const PEOPLE_DECISION_CHUNK_SIZE: usize = 500;

fn coordinate_people_decision_chunks<T: Clone>(items: &[T]) -> Vec<Vec<T>> {
    items
        .chunks(PEOPLE_DECISION_CHUNK_SIZE)
        .map(<[T]>::to_vec)
        .collect()
}

fn people_decision_affects_row(
    decision: &(i64, i64, PeopleDecision),
    row_face_id: i64,
    row_person_id: i64,
) -> bool {
    decision.0 == row_face_id
        && (decision.2 == PeopleDecision::NotFace || decision.1 == row_person_id)
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct PeopleFaceKey {
    face_id: i64,
    person_id: i64,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct PeopleFaceSelection {
    selected: HashSet<PeopleFaceKey>,
    anchor: Option<PeopleFaceKey>,
    focus: Option<PeopleFaceKey>,
}

impl PeopleFaceSelection {
    fn click(&mut self, key: PeopleFaceKey, order: &[PeopleFaceKey], shift: bool, toggle: bool) {
        if shift {
            let anchor = self.anchor.unwrap_or(key);
            self.select_range(anchor, key, order, toggle);
        } else if toggle {
            if !self.selected.insert(key) {
                self.selected.remove(&key);
            }
            self.anchor = Some(key);
        } else {
            self.selected.clear();
            self.selected.insert(key);
            self.anchor = Some(key);
        }
        self.focus = Some(key);
    }

    fn move_focus(&mut self, order: &[PeopleFaceKey], delta: isize, extend: bool) {
        if order.is_empty() {
            return;
        }
        let current = self
            .focus
            .and_then(|key| order.iter().position(|candidate| *candidate == key))
            .unwrap_or(0);
        let next = current
            .saturating_add_signed(delta)
            .min(order.len().saturating_sub(1));
        let key = order[next];
        if extend {
            let anchor = self.anchor.unwrap_or(order[current]);
            self.select_range(anchor, key, order, false);
        } else {
            self.selected.clear();
            self.selected.insert(key);
            self.anchor = Some(key);
        }
        self.focus = Some(key);
    }

    fn select_all(&mut self, order: &[PeopleFaceKey]) {
        self.selected.extend(order.iter().copied());
        self.anchor = order.first().copied();
        self.focus = order.last().copied();
    }

    fn clear(&mut self) {
        self.selected.clear();
        self.anchor = None;
        self.focus = None;
    }

    fn select_range(
        &mut self,
        anchor: PeopleFaceKey,
        key: PeopleFaceKey,
        order: &[PeopleFaceKey],
        preserve: bool,
    ) {
        let Some(anchor_index) = order.iter().position(|candidate| *candidate == anchor) else {
            self.clear();
            self.selected.insert(key);
            self.anchor = Some(key);
            return;
        };
        let Some(key_index) = order.iter().position(|candidate| *candidate == key) else {
            return;
        };
        if !preserve {
            self.selected.clear();
        }
        let (start, end) = if anchor_index <= key_index {
            (anchor_index, key_index)
        } else {
            (key_index, anchor_index)
        };
        self.selected.extend(order[start..=end].iter().copied());
        self.anchor = Some(anchor);
    }
}

#[derive(Clone)]
enum PeoplePageLoad<T> {
    Loading,
    Loaded(T),
    Error(Arc<str>),
}

enum EitherPeoplePage {
    Overview(catalog::people::PersonPage),
    Faces(catalog::people::FaceReviewPage),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PeopleSpecialPage {
    Unnamed,
    Review,
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

#[derive(Clone)]
enum LightroomImportState {
    Running,
    Succeeded {
        images: u64,
        recipes: u64,
        faces: u64,
    },
    Failed(Arc<str>),
}

#[derive(Clone)]
struct LightroomImportDialog {
    path: PathBuf,
    state: LightroomImportState,
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
    hidden_subtrees: HashSet<PathBuf>,
    hidden_expanded: bool,
    recursive_grid: Option<RecursiveGridState>,
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
    face_analysis: face_engine::FaceAnalysisService,
    recognition_refresh: face_engine::RecognitionRefreshService,
    people_summary_rebuild: face_engine::PeopleSummaryRebuildService,
    maintenance_admission: scheduler::MaintenanceAdmission,
    maintenance_driver: Arc<dyn MaintenanceDriver>,
    face_inventory_progress: Arc<FaceInventoryProgress>,
    face_inventory_cancelled: Arc<AtomicBool>,
    face_inventory_generation: u64,
    face_inventory_active: bool,
    face_analysis_paused: bool,
    face_crop_disk_cache_enabled: Arc<AtomicBool>,
    face_analysis_refill_scheduled: bool,
    face_inventory_refresh_scheduled: bool,
    face_inventory_watch_scheduled: bool,
    maintenance_diagnostics_observed: (u64, u64, u64),
    maintenance_throughput_expiry_scheduled: bool,
    maintenance_diagnostics_throttle: RefreshThrottle,
    recent_face_assets: VecDeque<PathBuf>,
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
    show_cache_statistics: bool,
    disk_cache_usage: Option<Result<rendering::pipeline::DiskCacheUsage, Arc<str>>>,
    settings: AppSettings,
    settings_path: PathBuf,
    settings_draft: Option<AppSettings>,
    lightroom_import_dialog: Option<LightroomImportDialog>,
    settings_status: Option<String>,
    job_queue_snapshot: SchedulerSnapshot,
    face_job_queue_snapshot: Vec<face_engine::FaceAnalysisJobSnapshot>,
    recognition_job_queue_snapshot: Vec<face_engine::RecognitionRefreshJobSnapshot>,
    summary_job_queue_snapshot: Vec<face_engine::PeopleSummaryRebuildJobSnapshot>,
    job_queue_scroll: UniformListScrollHandle,
    scheduler_refresh_scheduled: bool,
    path_context_menu: Option<PathContextMenu>,
    last_export_directory: Option<PathBuf>,
    system_clipboard: Option<arboard::Clipboard>,
    metadata_loader: MetadataLoader,
    metadata_load: Option<(NodeId, discovery::RequestId, Arc<AtomicBool>)>,
    scrollbar_drag: Option<ScrollbarDrag>,
    recipes: Option<Arc<HashMap<PathBuf, catalog::RecipeEntry>>>,
    people_overview: PeoplePageLoad<catalog::people::PersonPage>,
    people_diagnostics: Option<Result<Arc<catalog::people::PeopleDiagnostics>, Arc<str>>>,
    people_faces: PeoplePageLoad<catalog::people::FaceReviewPage>,
    people_rendered_face_indices: BTreeSet<usize>,
    people_rendered_overview_indices: BTreeSet<usize>,
    people_overview_scroll: UniformListScrollHandle,
    people_query_generation: u64,
    people_query_cancelled: Arc<AtomicBool>,
    people_sort: catalog::people::PeopleSort,
    people_representative_offsets: HashMap<i64, u64>,
    people_whole_library: bool,
    selected_person: Option<i64>,
    selected_person_name: Option<String>,
    people_special_page: Option<PeopleSpecialPage>,
    pending_people_decisions: HashSet<(i64, i64)>,
    people_undo_pending: bool,
    people_face_selection: PeopleFaceSelection,
    people_grid_focus: FocusHandle,
    people_view_switch_focus: FocusHandle,
    people_primary_card_focus: FocusHandle,
    people_back_focus: FocusHandle,
    people_restore_grid_focus: bool,
    people_decision_error: Option<Arc<str>>,
    people_operation_dialog: Option<PeopleOperationDialog>,
    people_operation_focus: FocusHandle,
    people_operation_pending: bool,
    people_operation_status: Option<Arc<str>>,
    people_operation_targets: Vec<(i64, String)>,
    people_operation_target_scroll: UniformListScrollHandle,
    people_operation_target_cancelled: Arc<AtomicBool>,
    people_operation_target_page: usize,
    people_operation_target_total: u64,
    people_operation_target_loading: bool,
    people_operation_target_generation: u64,
    people_merge_preview: Option<PeopleMergePreviewLoad>,
    people_context_mode: bool,
    /// Explicit crop/context choices for faces on the current detail page.
    /// Entries equal to the grid default are removed, keeping this bounded by
    /// the loaded page rather than catalog size.
    people_face_context_overrides: HashMap<PeopleFaceKey, bool>,
    people_accessibility: accessibility::PeopleAccessibilityBridge,
    people_page: usize,
    people_face_scroll: UniformListScrollHandle,
    developed: bool,
    directory_view_mode: DirectoryViewMode,
    map_camera: map::Camera,
    map_fit_node: Option<NodeId>,
    map_pending_fit: Option<NodeId>,
    map_render: MapRenderState,
    map_render_epoch: u64,
    map_render_generation: Arc<std::sync::atomic::AtomicU64>,
    map_render_size: (u32, u32),
    map_view_origin: (f32, f32),
    map_drag_last: Option<(f32, f32)>,
    map_drag_moved: bool,
    map_clusters: RefCell<Option<(u64, Option<map::PhotoClusters>)>>,
    map_cluster_pending_key: Cell<Option<u64>>,
    map_cluster_generation: Arc<std::sync::atomic::AtomicU64>,
    map_network_stats: map::network::NetworkStatsSnapshot,
    map_card: Option<MapCard>,
    map_job: Option<MapJob>,
    map_cache_clearing: bool,
    hidden_subtrees_loaded: bool,
    hidden_save_in_flight: bool,
}

fn durable_render_cache_write(
    face_crop_policy: &AtomicBool,
    variant: RenderVariant,
    normal_background_durability: bool,
) -> bool {
    match variant {
        RenderVariant::FaceCrop { .. } => face_crop_policy.load(Ordering::Acquire),
        _ => normal_background_durability,
    }
}

enum MapRenderState {
    Empty,
    Loading,
    Loaded {
        image: Arc<gpui::RenderImage>,
        camera: map::Camera,
        size: (u32, u32),
        detail_tiles: usize,
        detail_error: Option<Arc<str>>,
    },
    Error(Arc<str>),
}

#[derive(Clone)]
struct MapCard {
    items: Vec<MapCardItem>,
    page: usize,
    total: usize,
    cluster_id: Option<usize>,
    direct_indices: Option<Vec<usize>>,
    coordinate: metadata::GeoCoordinate,
    expansion_zoom: Option<u8>,
}

#[derive(Clone)]
struct MapCardItem {
    path: PathBuf,
}

impl MapCard {
    fn new(
        coordinate: metadata::GeoCoordinate,
        items: Vec<MapCardItem>,
        total: usize,
        cluster_id: Option<usize>,
        expansion_zoom: Option<u8>,
    ) -> Option<Self> {
        (!items.is_empty()).then_some(Self {
            items,
            page: 0,
            total,
            cluster_id,
            direct_indices: None,
            coordinate,
            expansion_zoom,
        })
    }

    fn page_count(&self) -> usize {
        self.total.div_ceil(MAP_GALLERY_PAGE_SIZE)
    }

    fn page_items(&self) -> &[MapCardItem] {
        &self.items
    }

    fn navigate(&mut self, delta: isize) {
        self.page = ((self.page as isize + delta).rem_euclid(self.page_count() as isize)) as usize;
    }
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
    MapCacheGiB,
    FaceWorkers,
    FaceMemoryMiB,
}

#[derive(Clone, Copy, Debug)]
struct ScrollbarDrag {
    target: ScrollbarTarget,
    pointer_offset: gpui::Pixels,
}

#[derive(Clone, Copy)]
struct MapCacheStatistics {
    usage: u64,
    capacity: u64,
    network: map::network::NetworkStatsSnapshot,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct MapJob {
    epoch: u64,
    label: String,
    status: MapJobStatus,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MapJobStatus {
    Queued,
    Running,
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

fn map_screen_position(
    coordinate: metadata::GeoCoordinate,
    camera: map::Camera,
    viewport: (u32, u32),
) -> Option<(f32, f32)> {
    fn mercator(coordinate: metadata::GeoCoordinate) -> (f64, f64) {
        let latitude = coordinate
            .latitude()
            .clamp(-85.051_13, 85.051_13)
            .to_radians();
        (
            (coordinate.longitude() + 180.0) / 360.0,
            (1.0 - latitude.tan().asinh() / std::f64::consts::PI) / 2.0,
        )
    }
    let (point_x, point_y) = mercator(coordinate);
    let (center_x, center_y) = mercator(camera.center);
    let world_pixels = 256.0 * 2_f64.powf(camera.zoom);
    let wrapped_x = (point_x - center_x + 0.5).rem_euclid(1.0) - 0.5;
    let x = 0.5 + wrapped_x * world_pixels / f64::from(viewport.0.max(1));
    let y = 0.5 + (point_y - center_y) * world_pixels / f64::from(viewport.1.max(1));
    (x.is_finite() && y.is_finite()).then_some((x as f32, y as f32))
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct MapSurfaceGeometry {
    left: f32,
    top: f32,
    width: f32,
    height: f32,
}

fn map_surface_geometry(
    rendered: map::Camera,
    current: map::Camera,
    viewport: (u32, u32),
    raster: (u32, u32),
) -> MapSurfaceGeometry {
    let width = viewport.0.max(1) as f32;
    let height = viewport.1.max(1) as f32;
    let scale = 2_f64.powf(current.zoom - rendered.zoom) as f32;
    let (rendered_x, rendered_y) = map::mercator_project(rendered.center);
    let (current_x, current_y) = map::mercator_project(current.center);
    let wrapped_x = (rendered_x - current_x + 0.5).rem_euclid(1.0) - 0.5;
    let world = 256.0 * 2_f64.powf(current.zoom);
    MapSurfaceGeometry {
        left: width / 2.0 - raster.0 as f32 * scale / 2.0 + (wrapped_x * world) as f32,
        top: height / 2.0 - raster.1 as f32 * scale / 2.0
            + ((rendered_y - current_y) * world) as f32,
        width: raster.0 as f32 * scale,
        height: raster.1 as f32 * scale,
    }
}

fn zoom_map_camera(
    mut camera: map::Camera,
    zoom_delta: f64,
    anchor: (f64, f64),
    viewport: (u32, u32),
) -> map::Camera {
    let old_zoom = camera.zoom;
    camera.zoom_by(zoom_delta);
    let offset_x = anchor.0 - f64::from(viewport.0) / 2.0;
    let offset_y = anchor.1 - f64::from(viewport.1) / 2.0;
    let old_world = 256.0 * 2_f64.powf(old_zoom);
    let new_world = 256.0 * 2_f64.powf(camera.zoom);
    let (center_x, center_y) = map::mercator_project(camera.center);
    if let Some(center) = map::mercator_unproject(
        center_x + offset_x / old_world - offset_x / new_world,
        center_y + offset_y / old_world - offset_y / new_world,
    ) {
        camera.center = center;
    }
    camera
}

fn map_detail_status(
    zoom: f64,
    permission: config::OnlineMapPermission,
    loading: bool,
    completed: Option<(usize, Option<&str>)>,
) -> String {
    if let Some((tiles, error)) = completed {
        if tiles > 0 {
            return format!("Detailed map · {tiles} vector tiles");
        }
        if let Some(error) = error {
            return format!("Bundled map · online detail unavailable: {error}");
        }
    }
    if zoom <= map::BUNDLED_DETAIL_MAX_ZOOM {
        "Bundled map · zoom in for detail".to_owned()
    } else if permission != config::OnlineMapPermission::Enabled {
        "Bundled map · online detail disabled".to_owned()
    } else if loading {
        "Bundled map · loading online detail".to_owned()
    } else {
        "Bundled map · no supported detail in visible tiles".to_owned()
    }
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
        (RenderVariant::FaceCrop { .. }, _) => "face crop",
        (RenderVariant::FaceContext { .. }, _) => "face context",
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

fn map_job_progress_label(job: &MapJob) -> String {
    match job.status {
        MapJobStatus::Queued => job.label.replacen("Rendering", "Preparing", 1),
        MapJobStatus::Running => job.label.clone(),
    }
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

struct RecursiveGridState {
    root: NodeId,
    status: RecursiveGridStatus,
    cancelled: Arc<AtomicBool>,
}

enum RecursiveGridStatus {
    Loading,
    Loaded(RecursiveImages),
    Error(Arc<str>),
}

struct RecursiveImages {
    paths: Vec<PathBuf>,
    coordinates: HashMap<PathBuf, metadata::GeoCoordinate>,
}

struct PathContextMenu {
    path: PathBuf,
    position: (f32, f32),
    exportable: bool,
    directory: Option<NodeId>,
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
        rendering::configure_analysis_memory_budget(settings.face_worker_memory_bytes);
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
        let face_crop_disk_cache_enabled = Arc::new(AtomicBool::new(
            settings.face_crop_disk_cache == config::FaceCropDiskCachePolicy::Disk,
        ));
        let scheduler_face_crop_disk_cache = face_crop_disk_cache_enabled.clone();
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
                        durable_cache_write: durable_render_cache_write(
                            scheduler_face_crop_disk_cache.as_ref(),
                            key.variant,
                            priority.class < scheduler::PriorityClass::Visible,
                        ),
                    },
                    cancelled,
                )
            }),
        );
        let maintenance_admission = scheduler.maintenance_admission();
        let face_workers = admitted_face_workers(
            sysinfo::System::new_all().available_memory(),
            settings.face_worker_memory_bytes,
            settings.face_workers,
        );
        let recognition_refresh =
            face_engine::RecognitionRefreshService::new(1, maintenance_admission.clone());
        let face_analysis = face_engine::FaceAnalysisService::new_with_pipeline_and_recognition(
            face_workers,
            job_limiter.clone(),
            maintenance_admission.clone(),
            render_pipeline.pipeline.clone(),
            recognition_refresh.clone(),
        );
        let people_summary_rebuild =
            face_engine::PeopleSummaryRebuildService::new(maintenance_admission.clone());
        let mut this = Self {
            model,
            visible: vec![VisibleRow {
                node: root,
                depth: 0,
            }],
            expanded: HashSet::from([root]),
            hidden_subtrees: HashSet::new(),
            hidden_expanded: true,
            recursive_grid: None,
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
            face_analysis,
            recognition_refresh,
            people_summary_rebuild,
            maintenance_admission,
            maintenance_driver: Arc::new(ProductionMaintenanceDriver),
            face_inventory_progress: Arc::new(FaceInventoryProgress::default()),
            face_inventory_cancelled: Arc::new(AtomicBool::new(false)),
            face_inventory_generation: 0,
            face_inventory_active: false,
            face_analysis_paused: !settings.automatic_face_analysis || face_workers == 0,
            face_crop_disk_cache_enabled,
            face_analysis_refill_scheduled: false,
            face_inventory_refresh_scheduled: false,
            face_inventory_watch_scheduled: false,
            maintenance_diagnostics_observed: (0, 0, 0),
            maintenance_throughput_expiry_scheduled: false,
            maintenance_diagnostics_throttle: RefreshThrottle::default(),
            recent_face_assets: VecDeque::new(),
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
            show_cache_statistics: false,
            disk_cache_usage: None,
            settings: settings.clone(),
            settings_path: config.settings_path.clone(),
            settings_draft: None,
            lightroom_import_dialog: None,
            settings_status: (face_workers == 0).then(||
                "Automatic face analysis is paused: insufficient memory after OS and foreground reserves."
                    .to_owned()),
            job_queue_snapshot: SchedulerSnapshot::default(),
            face_job_queue_snapshot: Vec::new(),
            recognition_job_queue_snapshot: Vec::new(),
            summary_job_queue_snapshot: Vec::new(),
            job_queue_scroll: UniformListScrollHandle::new(),
            scheduler_refresh_scheduled: false,
            path_context_menu: None,
            last_export_directory: None,
            system_clipboard: None,
            metadata_loader,
            metadata_load: None,
            scrollbar_drag: None,
            recipes: None,
            people_overview: PeoplePageLoad::Loading,
            people_diagnostics: None,
            people_faces: PeoplePageLoad::Loading,
            people_rendered_face_indices: BTreeSet::new(),
            people_rendered_overview_indices: BTreeSet::new(),
            people_overview_scroll: UniformListScrollHandle::new(),
            people_query_generation: 0,
            people_query_cancelled: Arc::new(AtomicBool::new(false)),
            people_sort: catalog::people::PeopleSort::Name,
            people_representative_offsets: HashMap::new(),
            people_whole_library: true,
            selected_person: None,
            selected_person_name: None,
            people_special_page: None,
            pending_people_decisions: HashSet::new(),
            people_undo_pending: false,
            people_face_selection: PeopleFaceSelection::default(),
            people_grid_focus: cx.focus_handle(),
            people_view_switch_focus: cx.focus_handle(),
            people_primary_card_focus: cx.focus_handle(),
            people_back_focus: cx.focus_handle(),
            people_restore_grid_focus: false,
            people_decision_error: None,
            people_operation_dialog: None,
            people_operation_focus: cx.focus_handle(),
            people_operation_pending: false,
            people_operation_status: None,
            people_operation_targets: Vec::new(),
            people_operation_target_scroll: UniformListScrollHandle::new(),
            people_operation_target_cancelled: Arc::new(AtomicBool::new(false)),
            people_operation_target_page: 0,
            people_operation_target_total: 0,
            people_operation_target_loading: false,
            people_operation_target_generation: 0,
            people_merge_preview: None,
            people_context_mode: false,
            people_face_context_overrides: HashMap::new(),
            people_accessibility: accessibility::PeopleAccessibilityBridge::new(),
            people_page: 0,
            people_face_scroll: UniformListScrollHandle::new(),
            developed: true,
            directory_view_mode: DirectoryViewMode::Grid,
            map_camera: map::Camera::default(),
            map_fit_node: None,
            map_pending_fit: Some(root),
            map_render: MapRenderState::Empty,
            map_render_epoch: 0,
            map_render_generation: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            map_render_size: (1280, 800),
            map_view_origin: (0.0, 0.0),
            map_drag_last: None,
            map_drag_moved: false,
            map_clusters: RefCell::new(None),
            map_cluster_pending_key: Cell::new(None),
            map_cluster_generation: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            map_network_stats: Default::default(),
            map_card: None,
            map_job: None,
            map_cache_clearing: false,
            hidden_subtrees_loaded: false,
            hidden_save_in_flight: false,
        };
        let catalog_work = this.work.start("Loading develop catalog…");
        let catalog_root = settings.root.clone();
        let catalog_loading = cx.background_executor().spawn(async move {
            catalog::prepare_existing_catalog(&catalog_root)?;
            let recipes = catalog::load_recipes(&catalog_root)?;
            let hidden = catalog::load_hidden_subtrees(&catalog_root)?;
            Ok::<_, anyhow::Error>((recipes, hidden))
        });
        cx.spawn(async move |this, cx| {
            let catalog = catalog_loading.await;
            this.update(cx, |this, cx| {
                this.work.finish(catalog_work);
                match catalog {
                    Ok((recipes, hidden)) => {
                        this.recipes = Some(Arc::new(recipes));
                        for path in &hidden {
                            this.model.ensure_node(path.clone());
                        }
                        if this.hidden_subtrees != hidden {
                            let preserve_fit = this.map_pending_fit == Some(this.selected)
                                || this.map_fit_node == Some(this.selected);
                            let restart_recursive = this.recursive_grid.is_some();
                            if let Some(recursive) = this.recursive_grid.take() {
                                recursive.cancelled.store(true, Ordering::Release);
                            }
                            this.hidden_subtrees = hidden;
                            this.rebuild_visible();
                            if restart_recursive {
                                this.start_recursive_grid(cx);
                                if !preserve_fit {
                                    this.map_pending_fit = None;
                                }
                            }
                        }
                        this.hidden_subtrees_loaded = true;
                        if this.directory_view_mode == DirectoryViewMode::People {
                            this.refresh_people_page(cx);
                        }
                        #[cfg(not(test))]
                        this.start_people_summary_maintenance(cx);
                        // Recover durable recognition work immediately on every start,
                        // including catalogs whose automatic face analysis is disabled.
                        #[cfg(not(test))]
                        this.recognition_refresh
                            .enqueue(this.settings.root.clone(), 0);
                        #[cfg(not(test))]
                        if should_start_automatic_face_inventory(&this.settings)
                            && !this.face_analysis_paused
                        {
                            this.refresh_face_inventory(cx);
                        }
                    }
                    Err(error) => {
                        eprintln!("cannot prepare Rekesz catalog: {error:#}");
                        this.recipes = Some(Arc::new(HashMap::new()));
                        this.settings_status =
                            Some(format!("Could not prepare Rekesz catalog: {error:#}"));
                    }
                }
                if this.map_card.is_some() {
                    this.request_map_card_thumbnail(cx);
                }
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
                            if node == this.selected
                                && this.directory_view_mode == DirectoryViewMode::Map
                                && this.map_pending_fit == Some(node)
                                && this.model.loaded(node).is_some_and(|contents| {
                                    contents.metadata_status == DirectoryMetadataStatus::Complete
                                })
                            {
                                this.fit_map_to_selected_images();
                                this.map_pending_fit = None;
                                this.request_map_render(cx);
                            }
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
                    if should_start_automatic_face_inventory(&this.settings)
                        && !this.face_analysis_paused
                    {
                        this.schedule_live_face_inventory_refresh(cx);
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
                    if node == this.selected
                        && this.directory_view_mode == DirectoryViewMode::Map
                        && this.map_pending_fit == Some(node)
                    {
                        this.fit_map_to_selected_images();
                        this.map_pending_fit = None;
                        this.request_map_render(cx);
                    }
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
            RenderVariant::FaceCrop { .. } | RenderVariant::FaceContext { .. } => &self.thumbnails,
            RenderVariant::Preview { .. } => &self.raw_previews,
        }
    }

    fn cache_for_variant_mut(&mut self, variant: RenderVariant) -> &mut PreviewCache {
        match variant {
            RenderVariant::Full => &mut self.previews,
            RenderVariant::Thumbnail { .. } => &mut self.thumbnails,
            RenderVariant::FaceCrop { .. } | RenderVariant::FaceContext { .. } => {
                &mut self.thumbnails
            }
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
        let recipes = self.recipes.clone().unwrap_or_default();
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
                            let failed = matches!(result, PreviewState::Error(_));
                            if let PreviewState::Loaded(preview) = &result {
                                this.model
                                    .record_render_metadata(path.clone(), preview.metadata.clone());
                            }
                            this.cache_for_variant_mut(variant).finish_for(
                                &path,
                                render_identity,
                                variant,
                                request,
                                result,
                            );
                            if failed
                                && scope == PEOPLE_SCOPE
                                && !this.advance_failed_people_representative(&path, variant, cx)
                            {
                                this.refresh_people_render_demand(cx);
                            }
                        }
                        Ok(JobCompletion::Cancelled) | Err(_) => {
                            this.cache_for_variant_mut(variant).cancel_for(
                                &path,
                                render_identity,
                                variant,
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
        self.dismiss_map_card();
        let directory_changed = self.selected != node;
        if !directory_changed {
            cx.notify();
            return;
        }
        if directory_changed && let Some(recursive) = self.recursive_grid.take() {
            recursive.cancelled.store(true, Ordering::Release);
        }
        self.scheduler.clear_scope(GRID_SCOPE);
        self.scheduler.clear_scope(IMAGE_SCOPE);
        self.reset_grid_prediction();
        self.cancel_metadata_enrichment();
        self.selected = node;
        if directory_changed {
            self.map_pending_fit = Some(node);
        }
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
        if directory_changed && self.directory_view_mode == DirectoryViewMode::Map {
            if self.model.loaded(node).is_some_and(|contents| {
                contents.metadata_status == DirectoryMetadataStatus::Complete
            }) {
                self.fit_map_to_selected_images();
                self.map_pending_fit = None;
            } else {
                self.map_camera = map::Camera::default();
            }
            self.request_map_render(cx);
        }
        if directory_changed
            && self.directory_view_mode == DirectoryViewMode::People
            && !self.people_whole_library
        {
            self.people_page = 0;
            self.people_representative_offsets.clear();
            self.refresh_people_page(cx);
        }
        cx.notify();
    }

    fn select_row(&mut self, index: usize, strategy: ScrollStrategy, cx: &mut Context<Self>) {
        if let Some(row) = self.visible.get(index) {
            self.scroll.scroll_to_item(index, strategy);
            if row.node != HIDDEN_NODE {
                self.select(row.node, cx);
            }
        }
    }

    fn selected_row(&self) -> usize {
        self.visible
            .iter()
            .position(|row| row.node == self.selected)
            .unwrap_or(0)
    }

    fn tree_up(&mut self, _: &TreeUp, _: &mut Window, cx: &mut Context<Self>) {
        if let Some(index) = (0..self.selected_row())
            .rev()
            .find(|index| self.visible[*index].node != HIDDEN_NODE)
        {
            self.select_row(index, ScrollStrategy::Top, cx);
        }
    }

    fn tree_down(&mut self, _: &TreeDown, _: &mut Window, cx: &mut Context<Self>) {
        if let Some(index) = (self.selected_row() + 1..self.visible.len())
            .find(|index| self.visible[*index].node != HIDDEN_NODE)
        {
            self.select_row(index, ScrollStrategy::Bottom, cx);
        }
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
            if self.visible[parent].node == HIDDEN_NODE {
                self.hidden_expanded = false;
                self.rebuild_visible();
                cx.notify();
            } else {
                self.select_row(parent, ScrollStrategy::Top, cx);
            }
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
        if node == HIDDEN_NODE {
            self.hidden_expanded = !self.hidden_expanded;
            self.rebuild_visible();
            cx.notify();
            return;
        }
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
        self.scheduler.clear_scope(MAP_CARD_SCOPE);
        self.recent_face_assets.retain(|recent| recent != &path);
        self.recent_face_assets.push_front(path.clone());
        self.recent_face_assets.truncate(32);
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
            if self.directory_view_mode == DirectoryViewMode::Map && self.map_card.is_some() {
                self.request_map_card_thumbnail(cx);
            }
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
        if let Some(RecursiveGridState {
            root,
            status: RecursiveGridStatus::Loaded(images),
            ..
        }) = &self.recursive_grid
            && *root == self.selected
        {
            return &images.paths;
        }
        self.model
            .loaded(self.selected)
            .map_or(&[], |contents| contents.images.as_slice())
    }

    fn selected_coordinates(&self) -> Vec<metadata::GeoCoordinate> {
        if let Some(RecursiveGridState {
            root,
            status: RecursiveGridStatus::Loaded(images),
            ..
        }) = &self.recursive_grid
            && *root == self.selected
        {
            return images
                .paths
                .iter()
                .filter_map(|path| images.coordinates.get(path).copied())
                .collect();
        }
        self.selected_images()
            .iter()
            .filter_map(|path| self.model.coordinate(self.selected, path))
            .collect()
    }

    fn selected_located_images(&self) -> Vec<(usize, metadata::GeoCoordinate)> {
        if let Some(RecursiveGridState {
            root,
            status: RecursiveGridStatus::Loaded(images),
            ..
        }) = &self.recursive_grid
            && *root == self.selected
        {
            return images
                .paths
                .iter()
                .enumerate()
                .filter_map(|(index, path)| {
                    images
                        .coordinates
                        .get(path)
                        .copied()
                        .map(|value| (index, value))
                })
                .collect();
        }
        self.selected_images()
            .iter()
            .enumerate()
            .filter_map(|(index, path)| {
                self.model
                    .coordinate(self.selected, path)
                    .map(|value| (index, value))
            })
            .collect()
    }

    fn fit_map_to_selected_images(&mut self) {
        if let Some(camera) =
            map::Camera::fit_viewport(&self.selected_coordinates(), self.map_render_size, 48.0)
        {
            self.map_camera = camera;
        } else {
            self.map_camera = map::Camera::default();
        }
        self.map_fit_node = Some(self.selected);
    }

    fn request_map_render(&mut self, cx: &mut Context<Self>) {
        if self.map_cache_clearing {
            return;
        }
        self.map_render_epoch = self.map_render_epoch.wrapping_add(1);
        let epoch = self.map_render_epoch;
        self.map_render_generation.store(epoch, Ordering::Release);
        if !matches!(self.map_render, MapRenderState::Loaded { .. }) {
            self.map_render = MapRenderState::Loading;
        }
        let camera = self.map_camera;
        let cache_path = self.settings.cache_root.join("map/ambient.sqlite3");
        let cache_capacity = self.settings.map_cache_bytes;
        let online_access =
            map::OnlineMapAccess::from_permission(self.settings.online_map_permission);
        self.map_job = Some(MapJob {
            epoch,
            label: if online_access.is_some() && camera.zoom > map::BUNDLED_DETAIL_MAX_ZOOM {
                "Rendering map and downloading visible tiles…".to_owned()
            } else {
                "Rendering map…".to_owned()
            },
            status: MapJobStatus::Queued,
        });
        cx.notify();
        let render_width = self
            .map_render_size
            .0
            .saturating_add(MAP_RENDER_OVERSCAN.saturating_mul(2));
        let render_height = self
            .map_render_size
            .1
            .saturating_add(MAP_RENDER_OVERSCAN.saturating_mul(2));
        let generation = self.map_render_generation.clone();
        let debounce = cx.background_executor().timer(Duration::from_millis(90));
        let retry_executor = cx.background_executor().clone();
        let (running_sender, running_receiver) = futures::channel::oneshot::channel();
        let rendering = cx.background_executor().spawn(async move {
            debounce.await;
            if generation.load(Ordering::Acquire) != epoch {
                return Ok::<_, String>(None);
            }
            let _render_guard = loop {
                if generation.load(Ordering::Acquire) != epoch {
                    return Ok(None);
                }
                if MAP_RENDER_BUSY
                    .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
                {
                    break MapRenderGuard;
                }
                retry_executor.timer(Duration::from_millis(16)).await;
            };
            if generation.load(Ordering::Acquire) != epoch {
                return Ok(None);
            }
            let _ = running_sender.send(());
            let rendered = map::renderer::render_map(
                render_width,
                render_height,
                camera,
                cache_path,
                cache_capacity,
                online_access,
                || generation.load(Ordering::Acquire) != epoch,
            )?;
            if generation.load(Ordering::Acquire) != epoch {
                return Ok(None);
            }
            Ok(Some((
                Arc::new(gpui::RenderImage::new(smallvec::smallvec![
                    image::Frame::new(rendered.image)
                ])),
                rendered.network,
                rendered.detail_tiles,
                rendered.detail_error.map(Arc::from),
            )))
        });
        cx.spawn(async move |this, cx| {
            if running_receiver.await.is_ok() {
                this.update(cx, |this, cx| {
                    if let Some(job) = &mut this.map_job
                        && job.epoch == epoch
                    {
                        job.status = MapJobStatus::Running;
                        cx.notify();
                    }
                })
                .ok();
            }
        })
        .detach();
        cx.spawn(async move |this, cx| {
            let result = rendering.await;
            this.update(cx, |this, cx| {
                if this.map_render_epoch != epoch || matches!(result, Ok(None)) {
                    return;
                }
                if this.map_job.as_ref().is_some_and(|job| job.epoch == epoch) {
                    this.map_job = None;
                }
                this.map_render = match result {
                    Ok(Some((image, stats, detail_tiles, detail_error))) => {
                        this.map_network_stats.hits =
                            this.map_network_stats.hits.saturating_add(stats.hits);
                        this.map_network_stats.misses =
                            this.map_network_stats.misses.saturating_add(stats.misses);
                        this.map_network_stats.downloads = this
                            .map_network_stats
                            .downloads
                            .saturating_add(stats.downloads);
                        this.map_network_stats.errors =
                            this.map_network_stats.errors.saturating_add(stats.errors);
                        this.map_network_stats.not_modified = this
                            .map_network_stats
                            .not_modified
                            .saturating_add(stats.not_modified);
                        MapRenderState::Loaded {
                            image,
                            camera,
                            size: (render_width, render_height),
                            detail_tiles,
                            detail_error,
                        }
                    }
                    Ok(None) => return,
                    Err(error) => MapRenderState::Error(error.into()),
                };
                cx.notify();
            })
            .ok();
        })
        .detach();
    }

    fn set_directory_view_mode(&mut self, mode: DirectoryViewMode, cx: &mut Context<Self>) {
        self.dismiss_map_card();
        self.directory_view_mode = mode;
        if mode != DirectoryViewMode::Map {
            self.map_render_epoch = self.map_render_epoch.wrapping_add(1);
            self.map_render_generation
                .store(self.map_render_epoch, Ordering::Release);
            self.map_job = None;
        }
        let metadata_complete = self.recursive_grid.as_ref().is_some_and(|recursive| {
            recursive.root == self.selected
                && matches!(recursive.status, RecursiveGridStatus::Loaded(_))
        }) || self
            .model
            .loaded(self.selected)
            .is_some_and(|contents| contents.metadata_status == DirectoryMetadataStatus::Complete);
        if mode == DirectoryViewMode::Map
            && metadata_complete
            && self.map_pending_fit == Some(self.selected)
        {
            self.fit_map_to_selected_images();
            self.map_pending_fit = None;
        }
        if mode == DirectoryViewMode::Map {
            self.request_map_render(cx);
        }
        if mode == DirectoryViewMode::People {
            self.refresh_people_page(cx);
        } else {
            self.people_query_generation = self.people_query_generation.wrapping_add(1);
            self.people_query_cancelled.store(true, Ordering::Release);
            self.scheduler.clear_scope(PEOPLE_SCOPE);
        }
        cx.notify();
    }

    fn refresh_face_inventory(&mut self, cx: &mut Context<Self>) {
        if !claim_inventory_run(&mut self.face_inventory_active) {
            return;
        }
        let root = self.settings.root.clone();
        let hidden = self.hidden_subtrees.clone();
        let pair_preference = self.pair_preference;
        let preferred_subtree = self.model.node(self.selected).path.clone();
        let recent_assets = self.recent_face_assets.iter().cloned().collect::<Vec<_>>();
        let maintenance_driver = self.maintenance_driver.clone();
        let admission = self.maintenance_admission.clone();
        self.face_inventory_cancelled.store(true, Ordering::Release);
        self.face_inventory_cancelled = Arc::new(AtomicBool::new(false));
        let cancelled = self.face_inventory_cancelled.clone();
        self.face_inventory_generation = self.face_inventory_generation.wrapping_add(1);
        let generation = self.face_inventory_generation;
        let identity_mode = if generation.is_multiple_of(FULL_IDENTITY_VERIFICATION_INTERVAL) {
            catalog::InventoryIdentityMode::VerifyContents
        } else {
            catalog::InventoryIdentityMode::MetadataFirst
        };
        let work_id = self.work.start("Updating face-analysis inventory…");
        let progress = Arc::new(FaceInventoryProgress::default());
        progress.scan_pending.store(1, Ordering::Release);
        progress.stage.store(1, Ordering::Release);
        self.face_inventory_progress = progress.clone();
        let worker_progress = progress.clone();
        let work = cx.background_executor().spawn(async move {
            let result = admission.run_maintenance(-100, &cancelled, || {
                maintenance_driver.inventory(
                    &root,
                    &hidden,
                    pair_preference,
                    &preferred_subtree,
                    &recent_assets,
                    InventoryCoordination {
                        progress: &worker_progress,
                        admission: &admission,
                        cancelled: &cancelled,
                        identity_mode,
                    },
                )
            });
            result.unwrap_or_else(|| Err(anyhow::anyhow!("face inventory cancelled")))
        });
        cx.spawn(async move |this, cx| {
            let result = work.await;
            this.update(cx, |this, cx| {
                this.work.finish(work_id);
                if !owns_inventory_progress(this.face_inventory_generation, generation) {
                    cx.notify();
                    return;
                }
                this.face_inventory_active = false;
                progress.stage.store(0, Ordering::Release);
                match result {
                    Ok(jobs) if !this.face_analysis_paused => {
                        for (job, rank) in jobs {
                            this.face_analysis.enqueue(job, rank);
                        }
                        this.recognition_refresh
                            .enqueue(this.settings.root.clone(), 0);
                        if let Some(delay) = MaintenanceCoordinator::PRODUCTION.refill_after(
                            this.face_analysis_paused,
                            true,
                            None,
                            false,
                        ) {
                            this.schedule_face_analysis_refill(delay, cx);
                        }
                        this.schedule_face_inventory_watch(cx);
                    }
                    Ok(_) => {}
                    Err(error) => {
                        this.settings_status =
                            Some(format!("Could not update face inventory: {error:#}"));
                        if let Some(delay) = MaintenanceCoordinator::PRODUCTION
                            .inventory_after(this.face_analysis_paused, true)
                        {
                            this.schedule_face_inventory_watch_after(delay, cx);
                        }
                    }
                }
                cx.notify();
            })?;
            Ok::<_, anyhow::Error>(())
        })
        .detach();
    }

    #[cfg(not(test))]
    fn start_people_summary_maintenance(&mut self, cx: &mut Context<Self>) {
        let root = self.settings.root.clone();
        let status = cx
            .background_executor()
            .spawn(async move { catalog::people::people_summary_rebuild_status(&root) });
        cx.spawn(async move |this, cx| {
            let result = status.await;
            this.update(cx, |this, cx| {
                match result {
                    Ok(status) if status.ready_to_enqueue() => {
                        this.people_summary_rebuild
                            .enqueue(this.settings.root.clone(), 0);
                    }
                    Ok(_) => {}
                    Err(error) => {
                        eprintln!("Could not refill People summary maintenance: {error:#}");
                    }
                }
                let timer = cx.background_executor().timer(Duration::from_secs(2));
                cx.spawn(async move |this, cx| {
                    timer.await;
                    this.update(cx, |this, cx| {
                        this.start_people_summary_maintenance(cx);
                    })
                    .ok();
                })
                .detach();
            })
            .ok();
        })
        .detach();
    }

    fn schedule_live_face_inventory_refresh(&mut self, cx: &mut Context<Self>) {
        if self.face_inventory_refresh_scheduled {
            return;
        }
        self.face_inventory_refresh_scheduled = true;
        let timer = cx.background_executor().timer(Duration::from_secs(1));
        cx.spawn(async move |this, cx| {
            timer.await;
            this.update(cx, |this, cx| {
                this.face_inventory_refresh_scheduled = false;
                if should_start_automatic_face_inventory(&this.settings)
                    && !this.face_analysis_paused
                {
                    this.refresh_face_inventory(cx);
                }
            })?;
            Ok::<_, anyhow::Error>(())
        })
        .detach();
    }

    fn schedule_face_inventory_watch(&mut self, cx: &mut Context<Self>) {
        if let Some(delay) =
            MaintenanceCoordinator::PRODUCTION.inventory_after(self.face_analysis_paused, false)
        {
            self.schedule_face_inventory_watch_after(delay, cx);
        }
    }

    fn schedule_face_inventory_watch_after(&mut self, delay: Duration, cx: &mut Context<Self>) {
        if self.face_inventory_watch_scheduled || self.face_analysis_paused {
            return;
        }
        self.face_inventory_watch_scheduled = true;
        let timer = cx.background_executor().timer(delay);
        cx.spawn(async move |this, cx| {
            timer.await;
            this.update(cx, |this, cx| {
                this.face_inventory_watch_scheduled = false;
                if should_start_automatic_face_inventory(&this.settings)
                    && !this.face_analysis_paused
                {
                    this.refresh_face_inventory(cx);
                }
            })?;
            Ok::<_, anyhow::Error>(())
        })
        .detach();
    }

    #[cfg(test)]
    fn schedule_face_analysis_refill(&mut self, _: Duration, _: &mut Context<Self>) {}

    #[cfg(not(test))]
    fn schedule_face_analysis_refill(&mut self, delay: Duration, cx: &mut Context<Self>) {
        if self.face_analysis_paused || self.face_analysis_refill_scheduled {
            return;
        }
        self.face_analysis_refill_scheduled = true;
        let root = self.settings.root.clone();
        let preferred_subtree = self.model.node(self.selected).path.clone();
        let recent_assets = self.recent_face_assets.iter().cloned().collect::<Vec<_>>();
        let maintenance_driver = self.maintenance_driver.clone();
        let executor = cx.background_executor().clone();
        let timer_executor = executor.clone();
        let refill = executor.spawn(async move {
            timer_executor.timer(delay).await;
            maintenance_driver.refill(&root, &preferred_subtree, &recent_assets)
        });
        cx.spawn(async move |this, cx| {
            let result = refill.await;
            this.update(cx, |this, cx| {
                this.face_analysis_refill_scheduled = false;
                match result {
                    Ok((jobs, next_wake)) if !this.face_analysis_paused => {
                        let had_jobs = !jobs.is_empty();
                        for (job, rank) in jobs {
                            this.face_analysis.enqueue(job, rank);
                        }
                        this.recognition_refresh
                            .enqueue(this.settings.root.clone(), 0);
                        if let Some(delay) = MaintenanceCoordinator::PRODUCTION.refill_after(
                            this.face_analysis_paused,
                            had_jobs,
                            next_wake,
                            false,
                        ) {
                            this.schedule_face_analysis_refill(delay, cx);
                        }
                    }
                    Ok(_) => {}
                    Err(error) => {
                        this.settings_status =
                            Some(format!("Could not refill face analysis: {error:#}"));
                        if let Some(delay) = MaintenanceCoordinator::PRODUCTION.refill_after(
                            this.face_analysis_paused,
                            false,
                            None,
                            true,
                        ) {
                            this.schedule_face_analysis_refill(delay, cx);
                        }
                    }
                }
                cx.notify();
            })?;
            Ok::<_, anyhow::Error>(())
        })
        .detach();
    }

    fn refresh_people_page(&mut self, cx: &mut Context<Self>) {
        self.people_query_generation = self.people_query_generation.wrapping_add(1);
        self.people_query_cancelled.store(true, Ordering::Release);
        self.people_query_cancelled = Arc::new(AtomicBool::new(false));
        self.people_rendered_face_indices.clear();
        self.people_rendered_overview_indices.clear();
        self.scheduler.clear_scope(PEOPLE_SCOPE);
        let generation = self.people_query_generation;
        let cancelled = self.people_query_cancelled.clone();
        let root = self.settings.root.clone();
        let subtree =
            (!self.people_whole_library).then(|| self.model.node(self.selected).path.to_path_buf());
        let offset = self.people_page.saturating_mul(30) as u64;
        let selected_person = self.selected_person;
        let special_page = self.people_special_page;
        let sort = self.people_sort;
        let representative_offsets = self.people_representative_offsets.clone();
        if selected_person.is_some() || special_page.is_some() {
            self.people_face_selection.clear();
            self.people_faces = PeoplePageLoad::Loading;
        } else {
            self.people_overview = PeoplePageLoad::Loading;
            self.people_diagnostics = None;
            let diagnostics_root = root.clone();
            let diagnostics = cx
                .background_executor()
                .spawn(async move { catalog::people::diagnostics(&diagnostics_root) });
            cx.spawn(async move |this, cx| {
                let result = diagnostics.await;
                this.update(cx, |this, cx| {
                    if generation == this.people_query_generation
                        && this.directory_view_mode == DirectoryViewMode::People
                    {
                        this.people_diagnostics = Some(result.map(Arc::new).map_err(|error| {
                            Arc::from(format!("Could not load People diagnostics: {error:#}"))
                        }));
                        cx.notify();
                    }
                })?;
                Ok::<_, anyhow::Error>(())
            })
            .detach();
        }
        self.scheduler.clear_scope(PEOPLE_SCOPE);
        let foreground = self.maintenance_admission.begin_foreground_activity();
        let query = cx.background_executor().spawn(async move {
            let _foreground = foreground;
            #[cfg(test)]
            if let Some(delay) = PEOPLE_QUERY_TEST_DELAY
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .take_if(|(delayed_root, _)| delayed_root == &root)
                .map(|(_, delay)| delay)
            {
                std::thread::sleep(delay);
            }
            if let Some(special_page) = special_page {
                match special_page {
                    PeopleSpecialPage::Unnamed => {
                        catalog::people::list_unnamed_faces_page_cancellable(
                            &root,
                            subtree.as_deref(),
                            offset,
                            30,
                            cancelled.clone(),
                        )
                    }
                    PeopleSpecialPage::Review => {
                        catalog::people::list_review_queue_page_cancellable(
                            &root,
                            subtree.as_deref(),
                            offset,
                            30,
                            cancelled.clone(),
                        )
                    }
                }
                .map(EitherPeoplePage::Faces)
            } else {
                selected_person.map_or_else(
                    || {
                        catalog::people::list_people_page_sorted_cancellable(
                            &root,
                            subtree.as_deref(),
                            sort,
                            offset,
                            30,
                            representative_offsets.clone(),
                            cancelled.clone(),
                        )
                        .map(EitherPeoplePage::Overview)
                    },
                    |person_id| {
                        catalog::people::list_person_faces_page_cancellable(
                            &root,
                            person_id,
                            subtree.as_deref(),
                            offset,
                            30,
                            cancelled.clone(),
                        )
                        .map(EitherPeoplePage::Faces)
                    },
                )
            }
        });
        cx.spawn(async move |this, cx| {
            let result = query.await;
            this.update(cx, |this, cx| {
                if generation != this.people_query_generation {
                    return;
                }
                if this.directory_view_mode != DirectoryViewMode::People {
                    return;
                }
                match result {
                    Ok(EitherPeoplePage::Overview(page)) => {
                        this.people_overview = PeoplePageLoad::Loaded(page);
                    }
                    Ok(EitherPeoplePage::Faces(page)) => {
                        let loaded = page
                            .items
                            .iter()
                            .map(|face| PeopleFaceKey {
                                face_id: face.face_id,
                                person_id: face.person_id,
                            })
                            .collect::<HashSet<_>>();
                        this.people_face_context_overrides
                            .retain(|key, _| loaded.contains(key));
                        this.people_faces = PeoplePageLoad::Loaded(page);
                    }
                    Err(error) => {
                        let error: Arc<str> = format!("{error:#}").into();
                        if selected_person.is_some() || special_page.is_some() {
                            this.people_faces = PeoplePageLoad::Error(error);
                        } else {
                            this.people_overview = PeoplePageLoad::Error(error);
                        }
                    }
                }
                this.refresh_people_render_demand(cx);
                this.people_restore_grid_focus = true;
                cx.notify();
            })?;
            Ok::<_, anyhow::Error>(())
        })
        .detach();
        cx.notify();
    }

    fn retry_failed_people_analysis(&mut self, cx: &mut Context<Self>) {
        let root = self.settings.root.clone();
        let work = cx
            .background_executor()
            .spawn(async move { catalog::people::retry_failed_analysis(&root) });
        cx.spawn(async move |this, cx| {
            let result = work.await;
            this.update(cx, |this, cx| {
                match result {
                    Ok(retried) => {
                        this.people_operation_status = Some(Arc::from(format!(
                            "Retrying {} analysis and {} recognition job(s)",
                            retried.analysis_jobs, retried.recognition_jobs
                        )));
                        if retried.recognition_jobs > 0 {
                            this.recognition_refresh
                                .enqueue(this.settings.root.clone(), 100);
                        }
                        this.refresh_face_inventory(cx);
                        this.refresh_people_page(cx);
                    }
                    Err(error) => {
                        this.people_operation_status = Some(Arc::from(format!(
                            "Could not retry failed People analysis: {error:#}"
                        )));
                    }
                }
                cx.notify();
            })?;
            Ok::<_, anyhow::Error>(())
        })
        .detach();
    }

    fn refresh_people_render_demand(&mut self, cx: &mut Context<Self>) {
        if self.directory_view_mode != DirectoryViewMode::People {
            self.scheduler.clear_scope(PEOPLE_SCOPE);
            return;
        }
        let requests = if self.selected_person.is_some() || self.people_special_page.is_some() {
            let PeoplePageLoad::Loaded(page) = &self.people_faces else {
                self.scheduler.clear_scope(PEOPLE_SCOPE);
                return;
            };
            page.items
                .iter()
                .enumerate()
                .filter(|(index, _)| self.people_rendered_face_indices.contains(index))
                .map(|(_, face)| {
                    let key = PeopleFaceKey {
                        face_id: face.face_id,
                        person_id: face.person_id,
                    };
                    let variant = if self.people_face_uses_context(key) {
                        RenderVariant::face_context(320)
                    } else {
                        RenderVariant::face_crop(face.face_id, face.bounds, 320)
                    };
                    (face.path.clone(), variant, JobPriority::visible(0))
                })
                .collect()
        } else {
            let PeoplePageLoad::Loaded(page) = &self.people_overview else {
                self.scheduler.clear_scope(PEOPLE_SCOPE);
                return;
            };
            let visible = page
                .items
                .iter()
                .enumerate()
                .filter(|(index, _)| self.people_rendered_overview_indices.contains(index))
                .filter_map(|(_, person)| {
                    person
                        .representative
                        .as_ref()
                        .map(|face| (person.id, face.clone()))
                })
                .collect::<Vec<_>>();
            // A failed preview can outlive the subscription that observed its
            // completion when another card refreshes the shared People scope.
            // Advance cached terminal errors here as well, otherwise that card
            // would remain permanently stuck on an unavailable candidate.
            let failed_people = visible
                .iter()
                .filter_map(|(person_id, face)| {
                    matches!(
                        self.thumbnails.state_for_variant(
                            &face.path,
                            self.render_identity_for(&face.path),
                            RenderVariant::face_crop(face.face_id, face.bounds, 320),
                        ),
                        PreviewState::Error(_)
                    )
                    .then_some(*person_id)
                })
                .collect::<Vec<_>>();
            if !failed_people.is_empty() {
                for person_id in failed_people {
                    *self
                        .people_representative_offsets
                        .entry(person_id)
                        .or_default() += 1;
                }
                self.refresh_people_page(cx);
                return;
            }
            visible
                .into_iter()
                .map(|(_, face)| {
                    let variant = RenderVariant::face_crop(face.face_id, face.bounds, 320);
                    (face.path, variant, JobPriority::visible(0))
                })
                .collect()
        };
        self.replace_render_scope(PEOPLE_SCOPE, requests, cx);
    }

    fn advance_failed_people_representative(
        &mut self,
        path: &std::path::Path,
        variant: RenderVariant,
        cx: &mut Context<Self>,
    ) -> bool {
        let RenderVariant::FaceCrop { face_id, .. } = variant else {
            return false;
        };
        let PeoplePageLoad::Loaded(page) = &self.people_overview else {
            return false;
        };
        let Some(person_id) = page.items.iter().find_map(|person| {
            person
                .representative
                .as_ref()
                .filter(|face| face.face_id == face_id && face.path == path)
                .map(|_| person.id)
        }) else {
            return false;
        };
        *self
            .people_representative_offsets
            .entry(person_id)
            .or_default() += 1;
        self.refresh_people_page(cx);
        true
    }

    fn change_people_page(&mut self, delta: isize, cx: &mut Context<Self>) {
        self.people_page = self.people_page.saturating_add_signed(delta);
        self.people_face_selection.clear();
        self.people_representative_offsets.clear();
        self.refresh_people_page(cx);
        cx.notify();
    }

    fn visible_people_face_keys(&self) -> Vec<PeopleFaceKey> {
        match &self.people_faces {
            PeoplePageLoad::Loaded(page) => page
                .items
                .iter()
                .map(|face| PeopleFaceKey {
                    face_id: face.face_id,
                    person_id: face.person_id,
                })
                .collect(),
            _ => Vec::new(),
        }
    }

    fn people_face_uses_context(&self, key: PeopleFaceKey) -> bool {
        self.people_face_context_overrides
            .get(&key)
            .copied()
            .unwrap_or(self.people_context_mode)
    }

    fn toggle_people_face_context(&mut self, key: PeopleFaceKey, cx: &mut Context<Self>) {
        let next = !self.people_face_uses_context(key);
        if next == self.people_context_mode {
            self.people_face_context_overrides.remove(&key);
        } else {
            self.people_face_context_overrides.insert(key, next);
        }
        self.refresh_people_render_demand(cx);
        cx.notify();
    }

    fn set_people_context_default(&mut self, context: bool, cx: &mut Context<Self>) {
        self.people_context_mode = context;
        self.people_face_context_overrides.clear();
        self.refresh_people_render_demand(cx);
        cx.notify();
    }

    fn people_has_next_page(&self) -> bool {
        let total = if self.selected_person.is_some() || self.people_special_page.is_some() {
            match &self.people_faces {
                PeoplePageLoad::Loaded(page) => page.total,
                _ => 0,
            }
        } else {
            match &self.people_overview {
                PeoplePageLoad::Loaded(page) => page.total,
                _ => 0,
            }
        };
        ((self.people_page as u64) + 1).saturating_mul(30) < total
    }

    fn current_people_face_path(&self, path: &Path) -> bool {
        matches!(&self.people_faces, PeoplePageLoad::Loaded(page) if page.items.iter().any(|face| face.path == path))
    }

    fn people_face_action_enabled(
        &self,
        face_id: i64,
        person_id: i64,
        decision: PeopleDecision,
    ) -> bool {
        if self
            .pending_people_decisions
            .iter()
            .any(|(pending, _)| *pending == face_id)
        {
            return false;
        }
        matches!(&self.people_faces, PeoplePageLoad::Loaded(page) if page.items.iter().any(|face| {
            face.face_id == face_id && face.person_id == person_id && match decision {
                PeopleDecision::Confirm => face.kind == catalog::people::FaceAssignmentKind::Suggested && person_id != 0,
                PeopleDecision::Reject => matches!(face.kind, catalog::people::FaceAssignmentKind::Suggested | catalog::people::FaceAssignmentKind::Confirmed) && person_id != 0,
                PeopleDecision::NotFace => face.kind != catalog::people::FaceAssignmentKind::Ignored,
            }
        }))
    }

    fn people_accessibility_controls(&self) -> Vec<accessibility::ControlAccessibilityItem> {
        use accessibility::{ControlAccessibilityItem as Item, PeopleAccessibilityAction as A};
        let button = |action, label: &str, disabled, busy| Item {
            action,
            label: label.to_owned(),
            role: accesskit::Role::Button,
            disabled,
            busy,
            toggled: None,
        };
        let switch = |action, label: &str, toggled| Item {
            action,
            label: label.to_owned(),
            role: accesskit::Role::Switch,
            disabled: false,
            busy: false,
            toggled: Some(toggled),
        };
        let mut controls = Vec::new();
        if self.people_operation_dialog.is_some() {
            controls.push(button(
                A::CancelDialog,
                "Cancel People operation",
                self.people_operation_pending,
                self.people_operation_pending,
            ));
            controls.push(button(
                A::ConfirmDialog,
                "Confirm People operation",
                self.people_operation_pending,
                self.people_operation_pending,
            ));
            if matches!(
                self.people_operation_dialog,
                Some(PeopleOperationDialog::Target { .. })
            ) {
                let (source, selected) = match &self.people_operation_dialog {
                    Some(PeopleOperationDialog::Target {
                        operation,
                        target_person_id,
                    }) => {
                        let source = match operation {
                            PeopleTargetOperation::Merge { source_person_id }
                            | PeopleTargetOperation::Move {
                                source_person_id, ..
                            } => Some(*source_person_id),
                            PeopleTargetOperation::Assign { .. } => None,
                        };
                        (source, *target_person_id)
                    }
                    _ => unreachable!(),
                };
                controls.extend(
                    self.people_operation_targets
                        .iter()
                        .filter(|(id, _)| Some(*id) != source)
                        .map(|(id, name)| Item {
                            action: A::SelectTarget(*id),
                            label: format!("Select {name} as target"),
                            role: accesskit::Role::RadioButton,
                            disabled: self.people_operation_target_loading,
                            busy: false,
                            toggled: Some(selected == Some(*id)),
                        }),
                );
                controls.push(button(
                    A::PreviousTargetPage,
                    "Previous target page",
                    self.people_operation_target_page == 0,
                    self.people_operation_target_loading,
                ));
                controls.push(button(
                    A::NextTargetPage,
                    "Next target page",
                    ((self.people_operation_target_page + 1) as u64 * 100)
                        >= self.people_operation_target_total,
                    self.people_operation_target_loading,
                ));
            }
            return controls;
        }
        let detail = self.selected_person.is_some() || self.people_special_page.is_some();
        let page_error = if detail {
            matches!(self.people_faces, PeoplePageLoad::Error(_))
        } else {
            matches!(self.people_overview, PeoplePageLoad::Error(_))
        };
        if page_error {
            controls.push(button(A::RetryPage, "Retry loading People", false, false));
        }
        controls.push(button(
            A::PreviousPage,
            "Previous People page",
            self.people_page == 0,
            false,
        ));
        controls.push(button(
            A::NextPage,
            "Next People page",
            !self.people_has_next_page(),
            false,
        ));
        if detail {
            controls.push(button(A::Back, "Back to People", false, false));
            let blocked = self.people_undo_pending || !self.pending_people_decisions.is_empty();
            controls.push(button(A::UndoReview, "Undo last review", blocked, blocked));
            if let Some(person_id) = self.selected_person {
                controls.push(button(
                    A::RenamePerson(person_id),
                    "Rename person",
                    self.people_operation_pending,
                    self.people_operation_pending,
                ));
                controls.push(button(
                    A::MergePerson(person_id),
                    "Merge person",
                    self.people_operation_pending,
                    self.people_operation_pending,
                ));
            }
            if let PeoplePageLoad::Loaded(page) = &self.people_faces {
                for face in &page.items {
                    let pending = self
                        .pending_people_decisions
                        .iter()
                        .any(|(id, _)| *id == face.face_id);
                    let name = face.path.file_name().map_or_else(
                        || face.path.display().to_string(),
                        |v| v.to_string_lossy().into_owned(),
                    );
                    controls.push(button(
                        A::OpenPhoto(face.path.clone()),
                        &format!("Open {name}"),
                        false,
                        false,
                    ));
                    if face.kind != catalog::people::FaceAssignmentKind::Ignored {
                        controls.push(button(
                            A::MarkNotFace {
                                face_id: face.face_id,
                                person_id: face.person_id,
                            },
                            &format!("Mark {name} as not a face"),
                            pending,
                            pending,
                        ));
                    }
                }
            }
            if !self.people_face_selection.selected.is_empty() {
                let selected_busy = self.people_undo_pending
                    || self.people_face_selection.selected.iter().any(|key| {
                        self.pending_people_decisions
                            .iter()
                            .any(|(pending_id, _)| *pending_id == key.face_id)
                    });
                let eligible = |decision| {
                    let mut faces = self
                        .people_face_selection
                        .selected
                        .iter()
                        .filter(|key| {
                            self.people_face_action_enabled(key.face_id, key.person_id, decision)
                        })
                        .map(|key| (key.face_id, key.person_id))
                        .collect::<Vec<_>>();
                    faces.sort_unstable();
                    faces
                };
                let confirm = eligible(PeopleDecision::Confirm);
                let reject = eligible(PeopleDecision::Reject);
                let not_face = eligible(PeopleDecision::NotFace);
                controls.push(button(
                    A::BulkDecision {
                        decision: PeopleDecision::Confirm,
                        faces: confirm.clone(),
                    },
                    "Accept selected faces",
                    self.people_undo_pending || confirm.is_empty(),
                    selected_busy,
                ));
                controls.push(button(
                    A::BulkDecision {
                        decision: PeopleDecision::Reject,
                        faces: reject.clone(),
                    },
                    "Reject selected faces",
                    self.people_undo_pending || reject.is_empty(),
                    selected_busy,
                ));
                controls.push(button(
                    A::BulkDecision {
                        decision: PeopleDecision::NotFace,
                        faces: not_face.clone(),
                    },
                    "Mark selected items as not faces",
                    self.people_undo_pending || not_face.is_empty(),
                    selected_busy,
                ));
                controls.push(button(
                    A::CreatePerson,
                    "Create person from selected faces",
                    false,
                    false,
                ));
                if let Some(person_id) = self.selected_person {
                    controls.push(button(
                        A::MoveSelected(person_id),
                        "Move selected faces",
                        false,
                        false,
                    ));
                    controls.push(button(
                        A::UseAsRepresentative,
                        "Use selected face as example",
                        self.people_face_selection.selected.len() != 1,
                        false,
                    ));
                }
                controls.push(button(
                    A::ClearSelection,
                    "Clear face selection",
                    false,
                    false,
                ));
            }
        } else {
            controls.push(button(
                A::OpenReviewQueue,
                "Open review queue",
                false,
                false,
            ));
            controls.push(button(
                A::OpenUnnamedFaces,
                "Open unnamed faces",
                false,
                false,
            ));
            controls.push(switch(
                A::UseWholeLibrary(true),
                "Use whole library scope",
                self.people_whole_library,
            ));
            controls.push(switch(
                A::UseWholeLibrary(false),
                "Use selected subtree scope",
                !self.people_whole_library,
            ));
            controls.push(button(
                A::CycleSort,
                &format!("Sort People by {}", people_sort_label(self.people_sort)),
                false,
                false,
            ));
            controls.push(switch(
                A::ToggleAnalysisPause,
                "Pause automatic face analysis",
                self.face_analysis_paused,
            ));
            if self.people_diagnostics.as_ref().is_some_and(|result| {
                result
                    .as_ref()
                    .is_ok_and(|d| d.analysis_errors + d.recognition_refresh_errors > 0)
            }) {
                controls.push(button(
                    A::RetryFailedAnalysis,
                    "Retry failed face analysis",
                    false,
                    false,
                ));
            }
        }
        controls
    }

    fn publish_people_accessibility(&self, window: &Window, cx: &mut Context<Self>) {
        let snapshot = if self.people_operation_dialog.is_some() {
            accessibility::people_dialog_snapshot("People operation")
        } else if self.selected_person.is_some() || self.people_special_page.is_some() {
            let items = match &self.people_faces {
                PeoplePageLoad::Loaded(page) => page
                    .items
                    .iter()
                    .map(|face| {
                        let key = PeopleFaceKey {
                            face_id: face.face_id,
                            person_id: face.person_id,
                        };
                        let file_name = face.path.file_name().map_or_else(
                            || face.path.display().to_string(),
                            |name| name.to_string_lossy().into_owned(),
                        );
                        let pending = self
                            .pending_people_decisions
                            .iter()
                            .any(|(face_id, _)| *face_id == face.face_id);
                        let decision = match face.kind {
                            catalog::people::FaceAssignmentKind::Suggested => "suggested",
                            catalog::people::FaceAssignmentKind::Confirmed => "confirmed",
                            catalog::people::FaceAssignmentKind::Rejected => "rejected",
                            catalog::people::FaceAssignmentKind::Unassigned => "unassigned",
                            catalog::people::FaceAssignmentKind::Ignored => "not a face",
                        };
                        (
                            file_name,
                            accessibility::FaceAccessibilityItem {
                                face_id: face.face_id,
                                person_id: face.person_id,
                                file_name: "",
                                person_name: face.person_name.as_deref(),
                                decision,
                                selected: self.people_face_selection.selected.contains(&key),
                                context: self.people_face_uses_context(key),
                                pending,
                                can_accept: face.kind
                                    == catalog::people::FaceAssignmentKind::Suggested,
                                can_reject: matches!(
                                    face.kind,
                                    catalog::people::FaceAssignmentKind::Suggested
                                        | catalog::people::FaceAssignmentKind::Confirmed
                                ),
                            },
                        )
                    })
                    .collect::<Vec<_>>(),
                _ => Vec::new(),
            };
            let borrowed = items
                .iter()
                .map(|(file_name, item)| accessibility::FaceAccessibilityItem {
                    file_name,
                    ..*item
                })
                .collect::<Vec<_>>();
            accessibility::people_faces_snapshot(&borrowed, self.people_context_mode)
        } else {
            let items = match &self.people_overview {
                PeoplePageLoad::Loaded(page) => page
                    .items
                    .iter()
                    .map(|person| accessibility::PersonAccessibilityItem {
                        id: person.id,
                        name: &person.name,
                        confirmed: person.confirmed_faces,
                        suggested: person.suggested_faces,
                    })
                    .collect::<Vec<_>>(),
                _ => Vec::new(),
            };
            accessibility::people_overview_snapshot(&items)
        };
        let mut snapshot = snapshot;
        snapshot.generation = self.people_query_generation;
        snapshot.append_controls(&self.people_accessibility_controls());
        self.people_accessibility
            .update(snapshot, window.is_window_active());
        for (generation, action) in self.people_accessibility.take_actions() {
            cx.defer_in(window, move |this, window, cx| {
                if generation != this.people_query_generation
                    || this.directory_view_mode != DirectoryViewMode::People
                    || !matches!(this.main_view, MainView::Dir(_, _))
                {
                    return;
                }
                match action {
                    accessibility::PeopleAccessibilityAction::OpenPerson(person_id) => {
                        if let PeoplePageLoad::Loaded(page) = &this.people_overview
                            && let Some(person) =
                                page.items.iter().find(|person| person.id == person_id)
                        {
                            this.selected_person_name = Some(person.name.clone());
                        }
                        this.selected_person = Some(person_id);
                        this.people_special_page = None;
                        this.people_page = 0;
                        this.set_people_context_default(false, cx);
                        this.refresh_people_page(cx);
                    }
                    accessibility::PeopleAccessibilityAction::SelectFace { face_id, person_id } => {
                        this.select_people_face(
                            PeopleFaceKey { face_id, person_id },
                            false,
                            false,
                            cx,
                        );
                    }
                    accessibility::PeopleAccessibilityAction::ToggleFaceContext {
                        face_id,
                        person_id,
                    } => {
                        this.toggle_people_face_context(PeopleFaceKey { face_id, person_id }, cx);
                    }
                    accessibility::PeopleAccessibilityAction::AcceptFace { face_id, person_id } => {
                        if this.people_face_action_enabled(
                            face_id,
                            person_id,
                            PeopleDecision::Confirm,
                        ) {
                            this.apply_people_decision(
                                face_id,
                                person_id,
                                PeopleDecision::Confirm,
                                cx,
                            );
                        }
                    }
                    accessibility::PeopleAccessibilityAction::RejectFace { face_id, person_id } => {
                        if this.people_face_action_enabled(
                            face_id,
                            person_id,
                            PeopleDecision::Reject,
                        ) {
                            this.apply_people_decision(
                                face_id,
                                person_id,
                                PeopleDecision::Reject,
                                cx,
                            );
                        }
                    }
                    accessibility::PeopleAccessibilityAction::ToggleGridContext => {
                        this.set_people_context_default(!this.people_context_mode, cx);
                    }
                    accessibility::PeopleAccessibilityAction::Back => {
                        this.selected_person = None;
                        this.selected_person_name = None;
                        this.people_special_page = None;
                        this.set_people_context_default(false, cx);
                        this.people_page = 0;
                        this.refresh_people_page(cx);
                    }
                    accessibility::PeopleAccessibilityAction::PreviousPage
                        if this.people_page > 0 =>
                    {
                        this.change_people_page(-1, cx);
                    }
                    accessibility::PeopleAccessibilityAction::NextPage
                        if this.people_has_next_page() =>
                    {
                        this.change_people_page(1, cx);
                    }
                    accessibility::PeopleAccessibilityAction::RenamePerson(person_id)
                        if this.selected_person == Some(person_id)
                            && !this.people_operation_pending =>
                    {
                        this.open_people_name_dialog(
                            PeopleNameOperation::Rename { person_id },
                            this.selected_person_name.clone().unwrap_or_default(),
                            window,
                            cx,
                        );
                    }
                    accessibility::PeopleAccessibilityAction::MergePerson(person_id)
                        if this.selected_person == Some(person_id)
                            && !this.people_operation_pending =>
                    {
                        this.open_people_target_dialog(
                            PeopleTargetOperation::Merge {
                                source_person_id: person_id,
                            },
                            window,
                            cx,
                        );
                    }
                    accessibility::PeopleAccessibilityAction::UndoReview
                        if can_start_people_undo(
                            this.people_undo_pending,
                            this.pending_people_decisions.len(),
                        ) =>
                    {
                        this.undo_people_decision_batch(cx);
                    }
                    accessibility::PeopleAccessibilityAction::OpenPhoto(path)
                        if this.current_people_face_path(&path) =>
                    {
                        this.show_image(path, cx);
                    }
                    accessibility::PeopleAccessibilityAction::MarkNotFace {
                        face_id,
                        person_id,
                    } if this.people_face_action_enabled(
                        face_id,
                        person_id,
                        PeopleDecision::NotFace,
                    ) =>
                    {
                        this.apply_people_decision(face_id, person_id, PeopleDecision::NotFace, cx);
                    }
                    accessibility::PeopleAccessibilityAction::OpenReviewQueue => {
                        this.selected_person = None;
                        this.people_special_page = Some(PeopleSpecialPage::Review);
                        this.people_page = 0;
                        this.refresh_people_page(cx);
                    }
                    accessibility::PeopleAccessibilityAction::OpenUnnamedFaces => {
                        this.selected_person = None;
                        this.people_special_page = Some(PeopleSpecialPage::Unnamed);
                        this.people_page = 0;
                        this.refresh_people_page(cx);
                    }
                    accessibility::PeopleAccessibilityAction::UseWholeLibrary(value) => {
                        this.people_whole_library = value;
                        this.people_page = 0;
                        this.people_representative_offsets.clear();
                        this.refresh_people_page(cx);
                    }
                    accessibility::PeopleAccessibilityAction::CycleSort => {
                        this.people_sort = next_people_sort(this.people_sort);
                        this.people_page = 0;
                        this.people_representative_offsets.clear();
                        this.refresh_people_page(cx);
                    }
                    accessibility::PeopleAccessibilityAction::ToggleAnalysisPause => {
                        this.face_analysis_paused = !this.face_analysis_paused;
                        this.face_analysis.set_paused(this.face_analysis_paused);
                        this.recognition_refresh
                            .set_paused(this.face_analysis_paused);
                        this.people_summary_rebuild
                            .set_paused(this.face_analysis_paused);
                        if !this.face_analysis_paused {
                            this.refresh_face_inventory(cx);
                        }
                        cx.notify();
                    }
                    accessibility::PeopleAccessibilityAction::RetryFailedAnalysis => {
                        this.retry_failed_people_analysis(cx);
                    }
                    accessibility::PeopleAccessibilityAction::RetryPage => {
                        this.refresh_people_page(cx);
                    }
                    accessibility::PeopleAccessibilityAction::SelectTarget(person_id)
                        if !this.people_operation_target_loading
                            && this
                                .people_operation_targets
                                .iter()
                                .any(|(id, _)| *id == person_id) =>
                    {
                        this.select_people_operation_target(person_id, cx);
                    }
                    accessibility::PeopleAccessibilityAction::BulkDecision { decision, faces }
                        if !this.people_undo_pending
                            && !faces.is_empty()
                            && faces.iter().all(|(face_id, person_id)| {
                                this.people_face_selection
                                    .selected
                                    .contains(&PeopleFaceKey {
                                        face_id: *face_id,
                                        person_id: *person_id,
                                    })
                                    && this
                                        .people_face_action_enabled(*face_id, *person_id, decision)
                            }) =>
                    {
                        this.apply_people_decision_batch(
                            faces
                                .into_iter()
                                .map(|(face_id, person_id)| (face_id, person_id, decision))
                                .collect(),
                            cx,
                        );
                    }
                    accessibility::PeopleAccessibilityAction::CreatePerson
                        if !this.people_face_selection.selected.is_empty() =>
                    {
                        if let Ok(face_ids) = prepare_selected_face_ids(&this.people_face_selection)
                        {
                            this.open_people_name_dialog(
                                PeopleNameOperation::Create { face_ids },
                                String::new(),
                                window,
                                cx,
                            );
                        }
                    }
                    accessibility::PeopleAccessibilityAction::MoveSelected(person_id)
                        if this.selected_person == Some(person_id) =>
                    {
                        if let Ok(face_ids) = this.selected_confirmed_face_ids(person_id) {
                            this.open_people_target_dialog(
                                PeopleTargetOperation::Move {
                                    source_person_id: person_id,
                                    face_ids,
                                },
                                window,
                                cx,
                            );
                        }
                    }
                    accessibility::PeopleAccessibilityAction::UseAsRepresentative => {
                        this.choose_selected_people_representative(cx);
                    }
                    accessibility::PeopleAccessibilityAction::ClearSelection => {
                        this.people_face_selection.clear();
                        cx.notify();
                    }
                    accessibility::PeopleAccessibilityAction::CancelDialog => {
                        this.people_operation_dialog = None;
                        this.people_operation_pending = false;
                        cx.notify();
                    }
                    accessibility::PeopleAccessibilityAction::ConfirmDialog
                        if !this.people_operation_pending =>
                    {
                        this.submit_people_operation(cx);
                    }
                    accessibility::PeopleAccessibilityAction::PreviousTargetPage
                        if this.people_operation_target_page > 0 =>
                    {
                        this.load_people_operation_target_page(
                            this.people_operation_target_page - 1,
                            cx,
                        );
                    }
                    accessibility::PeopleAccessibilityAction::NextTargetPage
                        if ((this.people_operation_target_page + 1) as u64 * 100)
                            < this.people_operation_target_total =>
                    {
                        this.load_people_operation_target_page(
                            this.people_operation_target_page + 1,
                            cx,
                        );
                    }
                    _ => {}
                }
            });
        }
    }

    fn select_people_face(
        &mut self,
        key: PeopleFaceKey,
        shift: bool,
        toggle: bool,
        cx: &mut Context<Self>,
    ) {
        let order = self.visible_people_face_keys();
        self.people_face_selection.click(key, &order, shift, toggle);
        cx.notify();
    }

    fn handle_people_selection_key(
        &mut self,
        event: &gpui::KeyDownEvent,
        modifiers: gpui::Modifiers,
        cx: &mut Context<Self>,
    ) {
        let order = self.visible_people_face_keys();
        let delta = match event.keystroke.key.as_str() {
            "left" => Some(-1),
            "right" => Some(1),
            "up" => Some(-3),
            "down" => Some(3),
            _ => None,
        };
        if let Some(delta) = delta {
            self.people_face_selection
                .move_focus(&order, delta, modifiers.shift);
        } else if event.keystroke.key == "a" && modifiers.secondary() {
            self.people_face_selection.select_all(&order);
        } else if event.keystroke.key == "escape" {
            self.people_face_selection.clear();
        } else {
            return;
        }
        cx.notify();
    }

    fn apply_selected_people_decision(&mut self, decision: PeopleDecision, cx: &mut Context<Self>) {
        let selected = self.people_face_selection.selected.clone();
        let eligible = match &self.people_faces {
            PeoplePageLoad::Loaded(page) => page
                .items
                .iter()
                .filter(|face| {
                    selected.contains(&PeopleFaceKey {
                        face_id: face.face_id,
                        person_id: face.person_id,
                    }) && match decision {
                        PeopleDecision::Confirm => {
                            face.kind == catalog::people::FaceAssignmentKind::Suggested
                                && face.person_id != 0
                        }
                        PeopleDecision::Reject => {
                            matches!(
                                face.kind,
                                catalog::people::FaceAssignmentKind::Suggested
                                    | catalog::people::FaceAssignmentKind::Confirmed
                            ) && face.person_id != 0
                        }
                        PeopleDecision::NotFace => true,
                    }
                })
                .map(|face| PeopleFaceKey {
                    face_id: face.face_id,
                    person_id: face.person_id,
                })
                .collect::<Vec<_>>(),
            _ => Vec::new(),
        };
        let mut ignored_faces = HashSet::new();
        let mut decisions = Vec::new();
        for key in eligible {
            if decision == PeopleDecision::NotFace && !ignored_faces.insert(key.face_id) {
                continue;
            }
            decisions.push((key.face_id, key.person_id, decision));
        }
        self.apply_people_decision_batch(decisions, cx);
        cx.notify();
    }

    fn open_people_name_dialog(
        &mut self,
        operation: PeopleNameOperation,
        input: String,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.people_operation_status = None;
        let cursor = input.chars().count();
        self.people_operation_dialog = Some(PeopleOperationDialog::Name {
            operation,
            input,
            cursor,
            selection_anchor: None,
        });
        self.people_operation_focus.focus(window);
        cx.notify();
    }

    fn open_people_target_dialog(
        &mut self,
        operation: PeopleTargetOperation,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.people_operation_status = None;
        self.people_operation_targets.clear();
        self.people_merge_preview = None;
        self.people_operation_target_cancelled
            .store(true, Ordering::Release);
        self.people_operation_target_cancelled = Arc::new(AtomicBool::new(false));
        self.people_operation_target_page = 0;
        self.people_operation_target_total = 0;
        self.people_operation_dialog = Some(PeopleOperationDialog::Target {
            operation,
            target_person_id: None,
        });
        self.people_operation_focus.focus(window);
        self.load_people_operation_target_page(0, cx);
    }

    fn load_people_operation_target_page(&mut self, page: usize, cx: &mut Context<Self>) {
        self.people_operation_target_cancelled
            .store(true, Ordering::Release);
        self.people_operation_target_cancelled = Arc::new(AtomicBool::new(false));
        self.people_operation_target_generation =
            self.people_operation_target_generation.wrapping_add(1);
        let generation = self.people_operation_target_generation;
        let cancelled = self.people_operation_target_cancelled.clone();
        self.people_operation_target_loading = true;
        self.people_operation_status = None;
        let root = self.settings.root.clone();
        let scope =
            (!self.people_whole_library).then(|| self.model.node(self.selected).path.to_path_buf());
        const TARGET_PAGE_SIZE: usize = 100;
        let work = cx.background_executor().spawn(async move {
            catalog::people::list_people_page_sorted_cancellable(
                &root,
                scope.as_deref(),
                catalog::people::PeopleSort::Name,
                page.saturating_mul(TARGET_PAGE_SIZE) as u64,
                TARGET_PAGE_SIZE,
                HashMap::new(),
                cancelled,
            )
        });
        cx.spawn(async move |this, cx| {
            let result = work.await;
            this.update(cx, |this, cx| {
                match result {
                    Ok(result_page)
                        if matches!(
                            this.people_operation_dialog,
                            Some(PeopleOperationDialog::Target { .. })
                        ) && this.people_operation_target_generation == generation =>
                    {
                        this.people_operation_targets = result_page
                            .items
                            .into_iter()
                            .map(|person| (person.id, person.name))
                            .collect();
                        this.people_operation_target_page = page;
                        this.people_operation_target_total = result_page.total;
                    }
                    Ok(_) => {}
                    Err(error) if this.people_operation_target_generation == generation => {
                        this.people_operation_status =
                            Some(Arc::from(format!("Could not load people: {error:#}")));
                    }
                    Err(_) => {}
                }
                if this.people_operation_target_generation == generation {
                    this.people_operation_target_loading = false;
                }
                cx.notify();
            })?;
            Ok::<_, anyhow::Error>(())
        })
        .detach();
        cx.notify();
    }

    fn select_people_operation_target(&mut self, target: i64, cx: &mut Context<Self>) {
        let source = match &mut self.people_operation_dialog {
            Some(PeopleOperationDialog::Target {
                operation,
                target_person_id,
            }) => {
                *target_person_id = Some(target);
                match operation {
                    PeopleTargetOperation::Merge { source_person_id } => Some(*source_person_id),
                    PeopleTargetOperation::Move { .. } | PeopleTargetOperation::Assign { .. } => {
                        None
                    }
                }
            }
            _ => return,
        };
        let Some(source) = source else {
            self.people_merge_preview = None;
            cx.notify();
            return;
        };
        self.people_merge_preview = Some(PeopleMergePreviewLoad::Loading(target));
        let root = self.settings.root.clone();
        let work = cx
            .background_executor()
            .spawn(async move { catalog::people::preview_people_merge(&root, source, target) });
        cx.spawn(async move |this, cx| {
            let result = work.await;
            this.update(cx, |this, cx| {
                let current = matches!(
                    this.people_operation_dialog,
                    Some(PeopleOperationDialog::Target {
                        target_person_id: Some(id), ..
                    }) if id == target
                );
                if current {
                    this.people_merge_preview = Some(match result {
                        Ok(preview) => PeopleMergePreviewLoad::Loaded(Arc::new(preview)),
                        Err(error) => PeopleMergePreviewLoad::Error(
                            target,
                            Arc::from(format!("Could not preview merge: {error:#}")),
                        ),
                    });
                }
                cx.notify();
            })?;
            Ok::<_, anyhow::Error>(())
        })
        .detach();
        cx.notify();
    }

    fn handle_people_operation_key(&mut self, event: &gpui::KeyDownEvent, cx: &mut Context<Self>) {
        if event.keystroke.key == "escape" {
            if !self.people_operation_pending {
                self.dismiss_people_operation();
            }
        } else if event.keystroke.key == "enter" {
            self.submit_people_operation(cx);
        } else if let Some(PeopleOperationDialog::Name {
            input,
            cursor,
            selection_anchor,
            ..
        }) = &mut self.people_operation_dialog
        {
            let secondary = event.keystroke.modifiers.control || event.keystroke.modifiers.platform;
            let shift = event.keystroke.modifiers.shift;
            let old_cursor = *cursor;
            let len = input.chars().count();
            if secondary && event.keystroke.key == "a" {
                *selection_anchor = Some(0);
                *cursor = len;
            } else if secondary && matches!(event.keystroke.key.as_str(), "c" | "x") {
                let selected = people_editor_selected_text(input, *cursor, *selection_anchor);
                if !selected.is_empty() {
                    cx.write_to_clipboard(ClipboardItem::new_string(selected));
                    if event.keystroke.key == "x" {
                        people_editor_replace(input, cursor, selection_anchor, "");
                    }
                }
            } else if secondary && event.keystroke.key == "v" {
                if let Some(text) = cx.read_from_clipboard().and_then(|item| item.text()) {
                    people_editor_replace(input, cursor, selection_anchor, &text);
                }
            } else if matches!(
                event.keystroke.key.as_str(),
                "left" | "right" | "home" | "end"
            ) {
                *cursor = match event.keystroke.key.as_str() {
                    "left" => cursor.saturating_sub(1),
                    "right" => cursor.saturating_add(1).min(len),
                    "home" => 0,
                    "end" => len,
                    _ => *cursor,
                };
                if shift {
                    selection_anchor.get_or_insert(old_cursor);
                } else {
                    *selection_anchor = None;
                }
            } else if matches!(event.keystroke.key.as_str(), "backspace" | "delete") {
                if people_editor_range(*cursor, *selection_anchor).is_empty() {
                    if event.keystroke.key == "backspace" && *cursor > 0 {
                        *selection_anchor = Some(*cursor - 1);
                    } else if event.keystroke.key == "delete" && *cursor < len {
                        *selection_anchor = Some(*cursor + 1);
                    }
                }
                people_editor_replace(input, cursor, selection_anchor, "");
            } else if !event.keystroke.modifiers.control
                && !event.keystroke.modifiers.platform
                && let Some(text) = &event.keystroke.key_char
            {
                people_editor_replace(input, cursor, selection_anchor, text);
            }
        }
        cx.notify();
    }

    fn dismiss_people_operation(&mut self) {
        self.people_operation_target_cancelled
            .store(true, Ordering::Release);
        self.people_operation_dialog = None;
        self.people_merge_preview = None;
        self.people_restore_grid_focus = true;
    }

    fn submit_people_operation(&mut self, cx: &mut Context<Self>) {
        if self.people_operation_pending {
            return;
        }
        let Some(dialog) = self.people_operation_dialog.clone() else {
            return;
        };
        if let Err(error) = validate_people_operation_dialog(&dialog) {
            self.people_operation_status = Some(Arc::from(error));
            cx.notify();
            return;
        }
        if let PeopleOperationDialog::Target {
            operation: PeopleTargetOperation::Merge { .. },
            target_person_id: Some(target),
        } = &dialog
            && !matches!(
                self.people_merge_preview,
                Some(PeopleMergePreviewLoad::Loaded(ref preview))
                    if preview.target_person_id == *target
            )
        {
            self.people_operation_status =
                Some(Arc::from("Wait for the merge consequence preview"));
            cx.notify();
            return;
        }
        self.people_operation_pending = true;
        self.people_operation_status = None;
        let root = self.settings.root.clone();
        let foreground = self.maintenance_admission.begin_foreground_activity();
        let work = cx.background_executor().spawn(async move {
            let _foreground = foreground;
            let result: anyhow::Result<(String, Option<(i64, String)>)> = match dialog {
                PeopleOperationDialog::Name {
                    operation, input, ..
                } => match operation {
                    PeopleNameOperation::Rename { person_id } => {
                        catalog::people::rename_person(&root, person_id, &input)?;
                        Ok((
                            "Person renamed".to_owned(),
                            Some((person_id, input.trim().to_owned())),
                        ))
                    }
                    PeopleNameOperation::Create { face_ids } => {
                        let person_id =
                            catalog::people::create_person_from_faces(&root, &input, &face_ids)?;
                        Ok((
                            format!("Created {}", input.trim()),
                            Some((person_id, input.trim().to_owned())),
                        ))
                    }
                },
                PeopleOperationDialog::Target {
                    operation,
                    target_person_id,
                } => {
                    let target = target_person_id.expect("validated target");
                    match operation {
                        PeopleTargetOperation::Merge { source_person_id } => {
                            catalog::people::merge_people(&root, source_person_id, target)?;
                            Ok(("People merged".to_owned(), None))
                        }
                        PeopleTargetOperation::Move {
                            source_person_id,
                            face_ids,
                        } => {
                            catalog::people::move_confirmed_faces(
                                &root,
                                source_person_id,
                                target,
                                &face_ids,
                            )?;
                            Ok((format!("Moved {} face(s)", face_ids.len()), None))
                        }
                        PeopleTargetOperation::Assign { face_ids } => {
                            catalog::people::assign_faces_to_person(&root, target, &face_ids)?;
                            Ok((format!("Assigned {} face(s)", face_ids.len()), None))
                        }
                    }
                }
            };
            result
        });
        cx.spawn(async move |this, cx| {
            let result = work.await;
            this.update(cx, |this, cx| {
                this.people_operation_pending = false;
                match result {
                    Ok((message, selected)) => {
                        this.people_operation_dialog = None;
                        this.people_operation_status = Some(Arc::from(message));
                        this.people_face_selection.clear();
                        if let Some((person_id, name)) = selected {
                            this.selected_person = Some(person_id);
                            this.selected_person_name = Some(name);
                            this.people_special_page = None;
                        } else if this.selected_person.is_some() {
                            this.selected_person = None;
                            this.selected_person_name = None;
                        }
                        this.people_page = 0;
                        this.refresh_people_page(cx);
                    }
                    Err(error) => {
                        this.people_operation_status =
                            Some(Arc::from(format!("Could not update people: {error:#}")));
                    }
                }
                cx.notify();
            })?;
            Ok::<_, anyhow::Error>(())
        })
        .detach();
        cx.notify();
    }

    fn choose_selected_people_representative(&mut self, cx: &mut Context<Self>) {
        let Some(person_id) = self.selected_person else {
            return;
        };
        let face_ids = match self.selected_confirmed_face_ids(person_id) {
            Ok(face_ids) => face_ids,
            Err(error) => {
                self.people_operation_status = Some(Arc::from(error));
                cx.notify();
                return;
            }
        };
        if face_ids.len() != 1 {
            self.people_operation_status = Some(Arc::from("Select exactly one confirmed face"));
            cx.notify();
            return;
        }
        let root = self.settings.root.clone();
        let face_id = face_ids[0];
        self.people_operation_pending = true;
        let work = cx.background_executor().spawn(async move {
            catalog::people::choose_person_representative(&root, person_id, face_id)
        });
        cx.spawn(async move |this, cx| {
            let result = work.await;
            this.update(cx, |this, cx| {
                this.people_operation_pending = false;
                match result {
                    Ok(()) => {
                        this.people_operation_status = Some(Arc::from("Representative updated"));
                        this.refresh_people_page(cx);
                    }
                    Err(error) => {
                        this.people_operation_status = Some(Arc::from(format!(
                            "Could not set representative: {error:#}"
                        )));
                    }
                }
                cx.notify();
            })?;
            Ok::<_, anyhow::Error>(())
        })
        .detach();
    }

    fn selected_confirmed_face_ids(&self, person_id: i64) -> Result<Vec<i64>, String> {
        let selected = &self.people_face_selection.selected;
        let Some(page) = (match &self.people_faces {
            PeoplePageLoad::Loaded(page) => Some(page),
            _ => None,
        }) else {
            return Err("Faces are still loading".to_owned());
        };
        let face_ids = page
            .items
            .iter()
            .filter(|face| {
                face.person_id == person_id
                    && face.kind == catalog::people::FaceAssignmentKind::Confirmed
                    && selected.contains(&PeopleFaceKey {
                        face_id: face.face_id,
                        person_id: face.person_id,
                    })
            })
            .map(|face| face.face_id)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        if face_ids.is_empty() {
            Err("Select at least one confirmed face for this person".to_owned())
        } else {
            Ok(face_ids)
        }
    }

    fn render_people_operation_targets(
        &mut self,
        rows: Range<usize>,
        _: &mut Window,
        cx: &mut Context<Self>,
    ) -> Vec<AnyElement> {
        let (source, selected) = match &self.people_operation_dialog {
            Some(PeopleOperationDialog::Target {
                operation,
                target_person_id,
            }) => {
                let source = match operation {
                    PeopleTargetOperation::Merge { source_person_id }
                    | PeopleTargetOperation::Move {
                        source_person_id, ..
                    } => Some(*source_person_id),
                    PeopleTargetOperation::Assign { .. } => None,
                };
                (source, *target_person_id)
            }
            _ => return Vec::new(),
        };
        rows.filter_map(|row| self.people_operation_targets.get(row).cloned())
            .map(|(id, name)| {
                if Some(id) == source {
                    return div().h(px(0.0)).into_any_element();
                }
                div()
                    .id(("people-target", id as usize))
                    .tab_index(0)
                    .h(px(42.0))
                    .p_2()
                    .mb_1()
                    .rounded_sm()
                    .bg(rgb(if selected == Some(id) {
                        0x46658c
                    } else {
                        0x30343d
                    }))
                    .cursor_pointer()
                    .child(name)
                    .on_key_down(cx.listener(move |this, event: &gpui::KeyDownEvent, _, cx| {
                        if matches!(event.keystroke.key.as_str(), "enter" | "space") {
                            this.select_people_operation_target(id, cx);
                            cx.stop_propagation();
                        }
                    }))
                    .on_click(cx.listener(move |this, _, _, cx| {
                        this.select_people_operation_target(id, cx);
                    }))
                    .into_any_element()
            })
            .collect()
    }

    fn apply_people_decision(
        &mut self,
        face_id: i64,
        person_id: i64,
        decision: PeopleDecision,
        cx: &mut Context<Self>,
    ) {
        self.apply_people_decision_batch(vec![(face_id, person_id, decision)], cx);
    }

    fn apply_people_decision_batch(
        &mut self,
        decisions: Vec<(i64, i64, PeopleDecision)>,
        cx: &mut Context<Self>,
    ) -> bool {
        if !can_start_people_review_save(self.people_undo_pending) {
            self.people_decision_error = Some(Arc::from("Wait for undo to finish"));
            cx.notify();
            return false;
        }
        if let Err(error) = validate_people_decision_batch(&decisions) {
            self.people_decision_error = Some(Arc::from(error));
            cx.notify();
            return false;
        }
        let busy_faces = self
            .pending_people_decisions
            .iter()
            .map(|(face_id, _)| *face_id)
            .collect::<HashSet<_>>();
        let decisions = decisions
            .into_iter()
            .filter(|(face_id, _, _)| !busy_faces.contains(face_id))
            .collect::<Vec<_>>();
        if decisions.is_empty() {
            return false;
        }
        for (face_id, person_id, _) in &decisions {
            self.pending_people_decisions.insert((*face_id, *person_id));
        }
        self.people_decision_error = None;
        let mut snapshots = Vec::new();
        if let PeoplePageLoad::Loaded(page) = &mut self.people_faces {
            for face in &mut page.items {
                if let Some((_, _, decision)) = decisions.iter().find(|decision| {
                    people_decision_affects_row(decision, face.face_id, face.person_id)
                }) {
                    snapshots.push((face.face_id, face.person_id, face.kind));
                    face.kind = match decision {
                        PeopleDecision::Confirm => catalog::people::FaceAssignmentKind::Confirmed,
                        PeopleDecision::Reject => catalog::people::FaceAssignmentKind::Rejected,
                        PeopleDecision::NotFace => catalog::people::FaceAssignmentKind::Ignored,
                    };
                }
            }
        }
        let root = self.settings.root.clone();
        let query_identity = (
            self.people_query_generation,
            self.selected_person,
            self.people_special_page,
            self.people_page,
        );
        let batch_prefix = format!(
            "ui-{}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |duration| duration.as_nanos()),
            self.people_query_generation
        );
        let chunks = coordinate_people_decision_chunks(&decisions);
        let foreground = self.maintenance_admission.begin_foreground_activity();
        let work = cx.background_executor().spawn(async move {
            let _foreground = foreground;
            chunks
                .into_iter()
                .enumerate()
                .map(|(index, chunk)| {
                    let commands = chunk
                        .iter()
                        .map(
                            |(face_id, person_id, decision)| catalog::people::DecisionCommand {
                                face_id: *face_id,
                                person_id: (*decision != PeopleDecision::NotFace)
                                    .then_some(*person_id),
                                action: match decision {
                                    PeopleDecision::Confirm => {
                                        catalog::people::DecisionAction::Confirm
                                    }
                                    PeopleDecision::Reject => {
                                        catalog::people::DecisionAction::Reject
                                    }
                                    PeopleDecision::NotFace => {
                                        catalog::people::DecisionAction::Ignore
                                    }
                                },
                            },
                        )
                        .collect();
                    let result = catalog::people::apply_decisions(
                        &root,
                        &catalog::people::DecisionBatch {
                            id: format!("{batch_prefix}/chunk/{index}"),
                            commands,
                        },
                    );
                    (chunk, result)
                })
                .collect::<Vec<_>>()
        });
        cx.spawn(async move |this, cx| {
            let result = work.await;
            this.update(cx, |this, cx| {
                for (face_id, person_id, _) in &decisions {
                    this.pending_people_decisions
                        .remove(&(*face_id, *person_id));
                }
                let completion_is_current = query_identity
                    == (
                        this.people_query_generation,
                        this.selected_person,
                        this.people_special_page,
                        this.people_page,
                    );
                let mut failed = Vec::new();
                let mut successful = Vec::new();
                let mut errors = Vec::new();
                for (chunk, chunk_result) in result {
                    match chunk_result {
                        Ok(_) => successful.extend(chunk),
                        Err(error) => {
                            errors.push(format!("{error:#}"));
                            failed.extend(chunk);
                        }
                    }
                }
                if !successful.is_empty() {
                    // The transaction has committed its durable refresh rows;
                    // wake recognition now instead of waiting for the poller.
                    this.recognition_refresh
                        .enqueue(this.settings.root.clone(), 0);
                }
                if completion_is_current
                    && let PeoplePageLoad::Loaded(page) = &mut this.people_faces
                {
                    for (face_id, person_id, old_kind) in snapshots {
                        let must_rollback = failed
                            .iter()
                            .any(|decision| people_decision_affects_row(decision, face_id, person_id));
                        if must_rollback
                            && let Some(face) = page.items.iter_mut().find(|face| {
                                face.face_id == face_id && face.person_id == person_id
                            })
                        {
                            face.kind = old_kind;
                        }
                    }
                }
                for (face_id, person_id, decision) in successful {
                    if !completion_is_current {
                        continue;
                    }
                    if decision == PeopleDecision::NotFace {
                        this.people_face_selection
                            .selected
                            .retain(|key| key.face_id != face_id);
                    } else {
                        this.people_face_selection.selected.remove(&PeopleFaceKey {
                            face_id,
                            person_id,
                        });
                    }
                }
                if !completion_is_current {
                    return;
                } else if failed.is_empty() {
                    this.people_decision_error = None;
                } else {
                    this.people_decision_error = Some(Arc::from(format!(
                        "Saved {} face decision(s); {} remain selected for retry across {} failed chunk(s): {}",
                        decisions.len().saturating_sub(failed.len()),
                        failed.len(),
                        errors.len(),
                        errors.join(" · ")
                    )));
                }
                this.refresh_people_render_demand(cx);
                this.people_restore_grid_focus = true;
                cx.notify();
            })?;
            Ok::<_, anyhow::Error>(())
        })
        .detach();
        cx.notify();
        true
    }

    fn undo_people_decision_batch(&mut self, cx: &mut Context<Self>) {
        if !can_start_people_undo(
            self.people_undo_pending,
            self.pending_people_decisions.len(),
        ) {
            self.people_decision_error = Some(Arc::from(
                "Wait for the current review save to finish before undoing",
            ));
            cx.notify();
            return;
        }
        self.people_undo_pending = true;
        self.people_decision_error = None;
        let root = self.settings.root.clone();
        let work = cx
            .background_executor()
            .spawn(async move { catalog::people::undo_most_recent_decision_batch(&root) });
        cx.spawn(async move |this, cx| {
            let result = work.await;
            this.update(cx, |this, cx| {
                this.people_undo_pending = false;
                match result {
                    Ok(Some(_)) => {
                        this.recognition_refresh
                            .enqueue(this.settings.root.clone(), 0);
                        this.refresh_people_page(cx);
                    }
                    Ok(None) => {
                        this.people_decision_error =
                            Some(Arc::from("There is no review batch to undo"));
                    }
                    Err(error) => {
                        this.people_decision_error =
                            Some(Arc::from(format!("Could not undo review batch: {error:#}")));
                    }
                }
                cx.notify();
            })?;
            Ok::<_, anyhow::Error>(())
        })
        .detach();
    }

    fn people_grid_columns(&self, window: &Window) -> usize {
        const CARD_WIDTH: f32 = 220.0;
        const CARD_GAP: f32 = 12.0;
        const VIEW_PADDING: f32 = 40.0;
        let inspector_width = if self.show_job_queue {
            JOB_QUEUE_PANEL_WIDTH
        } else {
            0.0
        } + if self.show_develop_inspector {
            DEVELOP_PANEL_WIDTH
        } else {
            0.0
        };
        let available = (f32::from(window.viewport_size().width)
            - self.sidebar_width
            - inspector_width
            - VIEW_PADDING)
            .max(CARD_WIDTH);
        people_grid_column_count(available, CARD_WIDTH, CARD_GAP)
    }

    fn render_people_face_rows(
        &mut self,
        rows: Range<usize>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Vec<AnyElement> {
        let columns = self.people_grid_columns(window);
        if self.people_restore_grid_focus {
            self.people_restore_grid_focus = false;
            self.people_grid_focus.focus(window);
        }
        let previous_visible = self.people_rendered_face_indices.clone();
        self.people_rendered_face_indices.clear();
        for row in rows.clone() {
            self.people_rendered_face_indices
                .extend(row.saturating_mul(columns)..row.saturating_add(1).saturating_mul(columns));
        }
        if self.people_rendered_face_indices != previous_visible {
            cx.spawn(async move |this, cx| {
                this.update(cx, |this, cx| this.refresh_people_render_demand(cx))?;
                Ok::<_, anyhow::Error>(())
            })
            .detach();
        }
        let PeoplePageLoad::Loaded(page) = &self.people_faces else {
            return Vec::new();
        };
        rows.map(|row| {
            div()
                .flex()
                .gap_3()
                .w_full()
                .h(px(350.0))
                .children(
                    page.items[row.saturating_mul(columns)
                        ..(row.saturating_add(1) * columns).min(page.items.len())]
                        .iter()
                        .map(|face| people_review_tile(self, face, cx)),
                )
                .into_any_element()
        })
        .collect()
    }

    fn render_people_overview_rows(
        &mut self,
        rows: Range<usize>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Vec<AnyElement> {
        let columns = self.people_grid_columns(window);
        let previous = self.people_rendered_overview_indices.clone();
        self.people_rendered_overview_indices.clear();
        for row in rows.clone() {
            self.people_rendered_overview_indices
                .extend(row.saturating_mul(columns)..row.saturating_add(1).saturating_mul(columns));
        }
        if previous != self.people_rendered_overview_indices {
            cx.spawn(async move |this, cx| {
                this.update(cx, |this, cx| this.refresh_people_render_demand(cx))?;
                Ok::<_, anyhow::Error>(())
            })
            .detach();
        }
        let PeoplePageLoad::Loaded(page) = &self.people_overview else {
            return Vec::new();
        };
        rows.map(|row| {
            div()
                .flex()
                .gap_3()
                .children(
                    page.items[row.saturating_mul(columns)
                        ..(row.saturating_add(1) * columns).min(page.items.len())]
                        .iter()
                        .enumerate()
                        .map(|(column, person)| {
                            let person_index = row * columns + column;
                            let person_id = person.id;
                            let person_name = person.name.clone();
                            let keyboard_name = person_name.clone();
                            let image = person.representative.as_ref().map_or_else(
                                || {
                                    div()
                                        .size_full()
                                        .child("No confirmed face")
                                        .into_any_element()
                                },
                                |face| {
                                    people_face_image(
                                        &self.thumbnails,
                                        &face.path,
                                        self.render_identity_for(&face.path),
                                        RenderVariant::face_crop(face.face_id, face.bounds, 320),
                                        None,
                                    )
                                },
                            );
                            div()
                                .id(("person-card", person_id as usize))
                                .debug_selector(move || format!("#person-card-{person_id}"))
                                .tab_index(0)
                                .when(person_index == 0, |card| {
                                    card.track_focus(&self.people_primary_card_focus)
                                })
                                .w(px(220.))
                                .p_4()
                                .rounded_md()
                                .bg(rgb(0x252932))
                                .cursor_pointer()
                                .child(
                                    div()
                                        .w_full()
                                        .h(px(170.))
                                        .mb_3()
                                        .overflow_hidden()
                                        .child(image),
                                )
                                .child(div().text_lg().child(person.name.clone()))
                                .child(format!(
                                    "{} confirmed · {} suggestions",
                                    person.confirmed_faces, person.suggested_faces
                                ))
                                .on_click(cx.listener(move |this, _, _, cx| {
                                    this.selected_person = Some(person_id);
                                    this.people_special_page = None;
                                    this.selected_person_name = Some(person_name.clone());
                                    this.people_page = 0;
                                    this.people_context_mode = false;
                                    this.people_face_context_overrides.clear();
                                    this.refresh_people_page(cx);
                                }))
                                .on_key_down(cx.listener(
                                    move |this, event: &gpui::KeyDownEvent, _, cx| {
                                        if matches!(event.keystroke.key.as_str(), "enter" | "space")
                                        {
                                            this.selected_person = Some(person_id);
                                            this.people_special_page = None;
                                            this.selected_person_name = Some(keyboard_name.clone());
                                            this.people_page = 0;
                                            this.people_context_mode = false;
                                            this.people_face_context_overrides.clear();
                                            this.refresh_people_page(cx);
                                            cx.stop_propagation();
                                        }
                                    },
                                ))
                        }),
                )
                .into_any_element()
        })
        .collect()
    }

    fn render_people_view(&self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        self.publish_people_accessibility(window, cx);
        let page_size = 30_u64;
        let columns = self.people_grid_columns(window);
        if self.selected_person.is_some() || self.people_special_page.is_some() {
            let content = match &self.people_faces {
                PeoplePageLoad::Loading => people_loading_skeletons(12),
                PeoplePageLoad::Error(error) => people_error_panel(error, cx),
                PeoplePageLoad::Loaded(page) if page.items.is_empty() => div()
                    .p_8()
                    .text_color(rgb(0xaeb4bf))
                    .child("No faces on this page.")
                    .into_any_element(),
                PeoplePageLoad::Loaded(page) => uniform_list(
                    "people-face-rows",
                    page.items.len().div_ceil(columns),
                    cx.processor(Self::render_people_face_rows),
                )
                .h_full()
                .track_scroll(self.people_face_scroll.clone())
                .with_sizing_behavior(ListSizingBehavior::Auto)
                .into_any_element(),
            };
            let total = match &self.people_faces {
                PeoplePageLoad::Loaded(page) => page.total,
                _ => 0,
            };
            let selected_count = self.people_face_selection.selected.len();
            return div()
                .size_full()
                .flex()
                .flex_col()
                .tab_index(0)
                .track_focus(&self.people_grid_focus)
                .on_key_down(cx.listener(|this, event: &gpui::KeyDownEvent, _, cx| {
                    this.handle_people_selection_key(event, event.keystroke.modifiers, cx);
                }))
                .p_5()
                .child(
                    div()
                        .flex()
                        .items_center()
                        .mb_4()
                        .child(
                            div()
                                .id("people-back")
                                .debug_selector(|| "#people-back".to_owned())
                                .tab_index(0)
                                .track_focus(&self.people_back_focus)
                                .cursor_pointer()
                                .child(format!(
                                    "← People / {}",
                                    self.selected_person_name.as_deref().unwrap_or(
                                        match self.people_special_page {
                                            Some(PeopleSpecialPage::Unnamed) => "Unnamed faces",
                                            Some(PeopleSpecialPage::Review) => "Review queue",
                                            None => "Person",
                                        }
                                    )
                                ))
                                .on_click(cx.listener(|this, _, _, cx| {
                                    this.selected_person = None;
                                    this.selected_person_name = None;
                                    this.people_special_page = None;
                                    this.people_context_mode = false;
                                    this.people_face_context_overrides.clear();
                                    this.people_page = 0;
                                    this.refresh_people_page(cx);
                                }))
                                .on_key_down(cx.listener(
                                    |this, event: &gpui::KeyDownEvent, _, cx| {
                                        if matches!(event.keystroke.key.as_str(), "enter" | "space")
                                        {
                                            this.selected_person = None;
                                            this.selected_person_name = None;
                                            this.people_special_page = None;
                                            this.people_context_mode = false;
                                            this.people_face_context_overrides.clear();
                                            this.people_page = 0;
                                            this.refresh_people_page(cx);
                                            cx.stop_propagation();
                                        }
                                    },
                                )),
                        )
                        .child(div().flex_1())
                        .child(people_page_controls(self.people_page, total, page_size, cx))
                        .children(self.selected_person.map(|person_id| {
                            div()
                                .flex()
                                .gap_2()
                                .ml_3()
                                .child(
                                    div()
                                        .id("rename-person")
                                        .tab_index(0)
                                        .px_3()
                                        .py_1()
                                        .rounded_sm()
                                        .bg(rgb(0x30343d))
                                        .cursor_pointer()
                                        .child("Rename")
                                        .on_key_down(cx.listener(
                                            move |this, event: &gpui::KeyDownEvent, window, cx| {
                                                if matches!(
                                                    event.keystroke.key.as_str(),
                                                    "enter" | "space"
                                                ) {
                                                    this.open_people_name_dialog(
                                                        PeopleNameOperation::Rename { person_id },
                                                        this.selected_person_name
                                                            .clone()
                                                            .unwrap_or_default(),
                                                        window,
                                                        cx,
                                                    );
                                                    cx.stop_propagation();
                                                }
                                            },
                                        ))
                                        .on_click(cx.listener(move |this, _, window, cx| {
                                            this.open_people_name_dialog(
                                                PeopleNameOperation::Rename { person_id },
                                                this.selected_person_name
                                                    .clone()
                                                    .unwrap_or_default(),
                                                window,
                                                cx,
                                            );
                                        })),
                                )
                                .child(
                                    div()
                                        .id("merge-person")
                                        .tab_index(0)
                                        .px_3()
                                        .py_1()
                                        .rounded_sm()
                                        .bg(rgb(0x30343d))
                                        .cursor_pointer()
                                        .child("Merge…")
                                        .on_key_down(cx.listener(
                                            move |this, event: &gpui::KeyDownEvent, window, cx| {
                                                if matches!(
                                                    event.keystroke.key.as_str(),
                                                    "enter" | "space"
                                                ) {
                                                    this.open_people_target_dialog(
                                                        PeopleTargetOperation::Merge {
                                                            source_person_id: person_id,
                                                        },
                                                        window,
                                                        cx,
                                                    );
                                                    cx.stop_propagation();
                                                }
                                            },
                                        ))
                                        .on_click(cx.listener(move |this, _, window, cx| {
                                            this.open_people_target_dialog(
                                                PeopleTargetOperation::Merge {
                                                    source_person_id: person_id,
                                                },
                                                window,
                                                cx,
                                            );
                                        })),
                                )
                        }))
                        .child(
                            div()
                                .id("people-undo-review")
                                .tab_index(0)
                                .ml_3()
                                .px_3()
                                .py_1()
                                .rounded_sm()
                                .bg(rgb(0x30343d))
                                .cursor_pointer()
                                .child(if self.people_undo_pending {
                                    "Undoing…"
                                } else if !self.pending_people_decisions.is_empty() {
                                    "Saving review…"
                                } else {
                                    "Undo last review"
                                })
                                .on_key_down(cx.listener(
                                    |this, event: &gpui::KeyDownEvent, _, cx| {
                                        if matches!(event.keystroke.key.as_str(), "enter" | "space")
                                        {
                                            this.undo_people_decision_batch(cx);
                                        }
                                    },
                                ))
                                .on_click(cx.listener(|this, _, _, cx| {
                                    this.undo_people_decision_batch(cx);
                                })),
                        )
                        .child(
                            div()
                                .id("people-context-toggle")
                                .debug_selector(|| "#people-context-toggle".to_owned())
                                .tab_index(0)
                                .ml_3()
                                .px_3()
                                .py_1()
                                .rounded_sm()
                                .bg(rgb(0x30343d))
                                .cursor_pointer()
                                .child(if self.people_context_mode {
                                    "Show face crops"
                                } else {
                                    "Show full context"
                                })
                                .on_click(cx.listener(|this, _, _, cx| {
                                    this.set_people_context_default(!this.people_context_mode, cx);
                                }))
                                .on_key_down(cx.listener(
                                    |this, event: &gpui::KeyDownEvent, _, cx| {
                                        if matches!(event.keystroke.key.as_str(), "enter" | "space")
                                        {
                                            this.set_people_context_default(
                                                !this.people_context_mode,
                                                cx,
                                            );
                                            cx.stop_propagation();
                                        }
                                    },
                                )),
                        ),
                )
                .children(self.people_decision_error.as_ref().map(|error| {
                    div()
                        .mb_3()
                        .p_3()
                        .rounded_sm()
                        .bg(rgb(0x613e42))
                        .child(error.to_string())
                }))
                .children(self.people_operation_status.as_ref().map(|status| {
                    div()
                        .mb_3()
                        .p_3()
                        .rounded_sm()
                        .bg(rgb(0x31445b))
                        .child(status.to_string())
                }))
                .children(
                    (selected_count > 0).then(|| {
                        people_selection_toolbar(selected_count, self.selected_person, cx)
                    }),
                )
                .child(div().flex_1().overflow_hidden().child(content))
                .children(
                    self.people_operation_dialog
                        .as_ref()
                        .map(|dialog| people_operation_dialog(self, dialog, cx)),
                )
                .into_any_element();
        }

        let content = match &self.people_overview {
            PeoplePageLoad::Loading => people_loading_skeletons(10),
            PeoplePageLoad::Error(error) => people_error_panel(error, cx),
            PeoplePageLoad::Loaded(page)
                if page.items.is_empty() && page.unnamed_faces == 0 && page.review_faces == 0 =>
            {
                div()
                    .p_8()
                    .text_color(rgb(0xaeb4bf))
                    .child("No people found in this scope.")
                    .into_any_element()
            }
            PeoplePageLoad::Loaded(page) => div()
                .h_full()
                .flex()
                .flex_col()
                .gap_3()
                .child(
                    div()
                        .flex()
                        .gap_3()
                        .child(people_special_overview_card(true, page.review_faces, cx))
                        .child(people_special_overview_card(false, page.unnamed_faces, cx)),
                )
                .child(
                    uniform_list(
                        "people-overview-rows",
                        page.items.len().div_ceil(columns),
                        cx.processor(Self::render_people_overview_rows),
                    )
                    .h_full()
                    .track_scroll(self.people_overview_scroll.clone())
                    .with_sizing_behavior(ListSizingBehavior::Auto),
                )
                .into_any_element(),
        };
        let total = match &self.people_overview {
            PeoplePageLoad::Loaded(page) => page.total,
            _ => 0,
        };
        let maintenance = self.face_analysis.stats();
        let index_status = match &self.people_overview {
            PeoplePageLoad::Loading => "Loading catalog index…".to_owned(),
            PeoplePageLoad::Error(_) => "Catalog index unavailable".to_owned(),
            PeoplePageLoad::Loaded(page) if page.analysis_errors > 0 => format!(
                "{} pending · {} analysis errors",
                page.analysis_pending, page.analysis_errors
            ),
            PeoplePageLoad::Loaded(page) => {
                format!(
                    "Catalog ready · {} pending analysis · {} queued / {} running",
                    page.analysis_pending, maintenance.queued, maintenance.running
                )
            }
        };
        let diagnostics = self.people_diagnostics.as_ref().map(|result| match result {
            Ok(diagnostics) => div()
                .flex()
                .items_center()
                .gap_3()
                .text_sm()
                .text_color(rgb(0xaeb4bf))
                .child(format!(
                    "Schema {} · detector {} · recognizer {} · policy v{} {} ({}) · {} indexed assets · {} faces · {} stale · {} reconciliation conflicts · {} failed",
                    diagnostics.schema_version,
                    diagnostics.detector_model_id,
                    diagnostics.recognizer_model_id,
                    diagnostics.recognition_policy_version,
                    if diagnostics.recognition_policy_active { "active" } else { "inactive" },
                    &diagnostics.recognition_policy_digest[..12],
                    diagnostics.indexed_assets,
                    diagnostics.active_faces,
                    diagnostics.stale_artifacts,
                    diagnostics.reconciliation_conflicts,
                    diagnostics.analysis_errors + diagnostics.recognition_refresh_errors
                ))
                .children(
                    ((diagnostics.analysis_errors + diagnostics.recognition_refresh_errors) > 0)
                        .then(|| {
                            div()
                                .id("people-retry-failed-analysis")
                                .tab_index(0)
                                .px_2()
                                .py_1()
                                .rounded_sm()
                                .bg(rgb(0x3f5570))
                                .cursor_pointer()
                                .child("Retry failed analysis")
                                .on_key_down(cx.listener(
                                    |this, event: &gpui::KeyDownEvent, _, cx| {
                                        if matches!(
                                            event.keystroke.key.as_str(),
                                            "enter" | "space"
                                        ) {
                                            this.retry_failed_people_analysis(cx);
                                            cx.stop_propagation();
                                        }
                                    },
                                ))
                                .on_click(cx.listener(|this, _, _, cx| {
                                    this.retry_failed_people_analysis(cx);
                                }))
                        }),
                )
                .into_any_element(),
            Err(error) => div()
                .text_sm()
                .text_color(rgb(0xffa0a0))
                .child(error.to_string())
                .into_any_element(),
        });
        div()
            .size_full()
            .p_5()
            .child(
                div()
                    .flex()
                    .items_center()
                    .mb_4()
                    .child(
                        div()
                            .id("people-library-scope")
                            .debug_selector(|| "#people-library-scope".to_owned())
                            .tab_index(0)
                            .px_3()
                            .py_1()
                            .rounded_sm()
                            .bg(rgb(if self.people_whole_library {
                                0x46658c
                            } else {
                                0x30343d
                            }))
                            .cursor_pointer()
                            .child("Whole library")
                            .on_click(cx.listener(|this, _, _, cx| {
                                this.people_whole_library = true;
                                this.people_page = 0;
                                this.people_representative_offsets.clear();
                                this.refresh_people_page(cx);
                            }))
                            .on_key_down(cx.listener(|this, event: &gpui::KeyDownEvent, _, cx| {
                                if matches!(event.keystroke.key.as_str(), "enter" | "space") {
                                    this.people_whole_library = true;
                                    this.people_page = 0;
                                    this.people_representative_offsets.clear();
                                    this.refresh_people_page(cx);
                                    cx.stop_propagation();
                                }
                            })),
                    )
                    .child(
                        div()
                            .id("people-subtree-scope")
                            .debug_selector(|| "#people-subtree-scope".to_owned())
                            .tab_index(0)
                            .ml_2()
                            .px_3()
                            .py_1()
                            .rounded_sm()
                            .bg(rgb(if self.people_whole_library {
                                0x30343d
                            } else {
                                0x46658c
                            }))
                            .cursor_pointer()
                            .child("Selected subtree")
                            .on_click(cx.listener(|this, _, _, cx| {
                                this.people_whole_library = false;
                                this.people_page = 0;
                                this.people_representative_offsets.clear();
                                this.refresh_people_page(cx);
                            }))
                            .on_key_down(cx.listener(|this, event: &gpui::KeyDownEvent, _, cx| {
                                if matches!(event.keystroke.key.as_str(), "enter" | "space") {
                                    this.people_whole_library = false;
                                    this.people_page = 0;
                                    this.people_representative_offsets.clear();
                                    this.refresh_people_page(cx);
                                    cx.stop_propagation();
                                }
                            })),
                    )
                    .child(
                        div()
                            .id("people-sort")
                            .debug_selector(|| "#people-sort".to_owned())
                            .tab_index(0)
                            .ml_3()
                            .px_3()
                            .py_1()
                            .rounded_sm()
                            .bg(rgb(0x30343d))
                            .cursor_pointer()
                            .child(format!("Sort: {}", people_sort_label(self.people_sort)))
                            .on_click(cx.listener(|this, _, _, cx| {
                                this.people_sort = match this.people_sort {
                                    catalog::people::PeopleSort::Name => {
                                        catalog::people::PeopleSort::Confirmed
                                    }
                                    catalog::people::PeopleSort::Confirmed => {
                                        catalog::people::PeopleSort::Review
                                    }
                                    catalog::people::PeopleSort::Review => {
                                        catalog::people::PeopleSort::RecentlyChanged
                                    }
                                    catalog::people::PeopleSort::RecentlyChanged => {
                                        catalog::people::PeopleSort::Name
                                    }
                                };
                                this.people_page = 0;
                                this.people_representative_offsets.clear();
                                this.refresh_people_page(cx);
                            }))
                            .on_key_down(cx.listener(|this, event: &gpui::KeyDownEvent, _, cx| {
                                if matches!(event.keystroke.key.as_str(), "enter" | "space") {
                                    this.people_sort = match this.people_sort {
                                        catalog::people::PeopleSort::Name => {
                                            catalog::people::PeopleSort::Confirmed
                                        }
                                        catalog::people::PeopleSort::Confirmed => {
                                            catalog::people::PeopleSort::Review
                                        }
                                        catalog::people::PeopleSort::Review => {
                                            catalog::people::PeopleSort::RecentlyChanged
                                        }
                                        catalog::people::PeopleSort::RecentlyChanged => {
                                            catalog::people::PeopleSort::Name
                                        }
                                    };
                                    this.people_page = 0;
                                    this.people_representative_offsets.clear();
                                    this.refresh_people_page(cx);
                                    cx.stop_propagation();
                                }
                            })),
                    )
                    .child(
                        div()
                            .ml_3()
                            .text_sm()
                            .text_color(rgb(0x8b929e))
                            .child(index_status),
                    )
                    .child(
                        div()
                            .id("people-analysis-pause")
                            .tab_index(0)
                            .ml_3()
                            .px_3()
                            .py_1()
                            .rounded_sm()
                            .bg(rgb(0x30343d))
                            .cursor_pointer()
                            .child(if self.face_analysis_paused {
                                "Resume analysis"
                            } else {
                                "Pause analysis"
                            })
                            .on_click(cx.listener(|this, _, _, cx| {
                                this.face_analysis_paused = !this.face_analysis_paused;
                                this.face_analysis.set_paused(this.face_analysis_paused);
                                this.recognition_refresh
                                    .set_paused(this.face_analysis_paused);
                                this.people_summary_rebuild
                                    .set_paused(this.face_analysis_paused);
                                if !this.face_analysis_paused {
                                    this.refresh_face_inventory(cx);
                                }
                                cx.notify();
                            }))
                            .on_key_down(cx.listener(|this, event: &gpui::KeyDownEvent, _, cx| {
                                if matches!(event.keystroke.key.as_str(), "enter" | "space") {
                                    this.face_analysis_paused = !this.face_analysis_paused;
                                    this.face_analysis.set_paused(this.face_analysis_paused);
                                    this.recognition_refresh
                                        .set_paused(this.face_analysis_paused);
                                    this.people_summary_rebuild
                                        .set_paused(this.face_analysis_paused);
                                    if !this.face_analysis_paused {
                                        this.refresh_face_inventory(cx);
                                    }
                                    cx.notify();
                                    cx.stop_propagation();
                                }
                            })),
                    )
                    .child(div().flex_1())
                    .child(people_page_controls(self.people_page, total, page_size, cx)),
            )
            .children(diagnostics)
            .child(content)
            .into_any_element()
    }

    fn set_online_map_permission(
        &mut self,
        permission: config::OnlineMapPermission,
        cx: &mut Context<Self>,
    ) {
        let mut settings = self.settings.clone();
        settings.online_map_permission = permission;
        match config::save(&self.settings_path, &settings) {
            Ok(()) => {
                self.settings = settings;
                if let Some(draft) = &mut self.settings_draft {
                    draft.online_map_permission = permission;
                }
                self.settings_status = None;
                if self.directory_view_mode == DirectoryViewMode::Map {
                    self.request_map_render(cx);
                }
            }
            Err(error) => {
                self.settings_status =
                    Some(format!("Could not save online map setting: {error:#}"));
            }
        }
        cx.notify();
    }

    fn render_map_view(&self, cx: &mut Context<Self>) -> AnyElement {
        let selected_images = self.selected_images();
        let total = selected_images.len();
        let can_show_recursively = self.recursive_grid.is_none()
            && self
                .model
                .loaded(self.selected)
                .is_some_and(|contents| !contents.children.is_empty());
        let located_images = self.selected_located_images();
        let coordinates = located_images
            .iter()
            .map(|(_, coordinate)| *coordinate)
            .collect::<Vec<_>>();
        let located = coordinates.len();
        let mut cluster_hasher = std::collections::hash_map::DefaultHasher::new();
        self.selected.hash(&mut cluster_hasher);
        for (index, coordinate) in &located_images {
            index.hash(&mut cluster_hasher);
            coordinate.latitude().to_bits().hash(&mut cluster_hasher);
            coordinate.longitude().to_bits().hash(&mut cluster_hasher);
        }
        let cluster_key = cluster_hasher.finish();
        let rebuild_clusters = self
            .map_clusters
            .borrow()
            .as_ref()
            .is_none_or(|(cached_key, _)| *cached_key != cluster_key);
        if rebuild_clusters && self.map_cluster_pending_key.get() != Some(cluster_key) {
            self.map_cluster_pending_key.set(Some(cluster_key));
            self.map_cluster_generation
                .store(cluster_key, Ordering::Release);
            let generation = Arc::clone(&self.map_cluster_generation);
            let retry_executor = cx.background_executor().clone();
            let clustering = cx.background_executor().spawn(async move {
                loop {
                    if generation.load(Ordering::Acquire) != cluster_key {
                        return None;
                    }
                    if MAP_CLUSTER_BUSY
                        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                        .is_ok()
                    {
                        break;
                    }
                    retry_executor.timer(Duration::from_millis(16)).await;
                }
                let _guard = MapClusterGuard;
                if generation.load(Ordering::Acquire) != cluster_key {
                    return None;
                }
                Some(map::PhotoClusters::new(&coordinates))
            });
            cx.spawn(async move |this, cx| {
                let result = clustering.await;
                this.update(cx, |this, cx| {
                    if this.map_cluster_pending_key.get() != Some(cluster_key)
                        || this.map_cluster_generation.load(Ordering::Acquire) != cluster_key
                    {
                        return;
                    }
                    let Some(result) = result else {
                        return;
                    };
                    this.map_cluster_pending_key.set(None);
                    *this.map_clusters.borrow_mut() = Some((cluster_key, Some(result)));
                    cx.notify();
                })
                .ok();
            })
            .detach();
        }
        let markers = self
            .map_clusters
            .borrow()
            .as_ref()
            .filter(|(cached_key, _)| *cached_key == cluster_key)
            .and_then(|(_, clusters)| {
                let clusters = clusters.as_ref()?;
                let zoom = self.map_camera.zoom.floor().clamp(0.0, 20.0) as u8;
                Some(
                    self.map_camera
                        .visible_bounds(self.map_render_size)
                        .into_iter()
                        .flat_map(|bounds| clusters.markers(bounds, zoom))
                        .collect::<Vec<_>>(),
                )
            })
            .unwrap_or_default();
        let camera = self.map_camera;
        let completed_detail = match &self.map_render {
            MapRenderState::Loaded {
                detail_tiles,
                detail_error,
                ..
            } => Some((*detail_tiles, detail_error.as_deref())),
            _ => None,
        };
        let status = map_detail_status(
            camera.zoom,
            self.settings.online_map_permission,
            matches!(self.map_render, MapRenderState::Loading),
            completed_detail,
        );
        div()
            .id("map-view")
            .debug_selector(|| "#map-view".to_owned())
            .size_full()
            .relative()
            .overflow_hidden()
            .bg(rgb(0x17242b))
            .cursor(if self.map_drag_last.is_some() {
                CursorStyle::ClosedHand
            } else {
                CursorStyle::OpenHand
            })
            .on_mouse_move(cx.listener(Self::map_mouse_move))
            .on_mouse_up(MouseButton::Left, cx.listener(Self::map_mouse_up))
            .on_mouse_down(MouseButton::Left, cx.listener(Self::map_mouse_down))
            .on_scroll_wheel(cx.listener(Self::map_scroll_wheel))
            .child(match &self.map_render {
                MapRenderState::Loaded {
                    image,
                    camera: rendered_camera,
                    size: rendered_size,
                    ..
                } => {
                    let geometry = map_surface_geometry(
                        *rendered_camera,
                        camera,
                        self.map_render_size,
                        *rendered_size,
                    );
                    img(image.clone())
                        .absolute()
                        .left(px(geometry.left))
                        .top(px(geometry.top))
                        .w(px(geometry.width))
                        .h(px(geometry.height))
                        .into_any_element()
                }
                MapRenderState::Loading => div()
                    .absolute()
                    .inset_0()
                    .flex()
                    .items_center()
                    .justify_center()
                    .child("Rendering bundled map…")
                    .into_any_element(),
                MapRenderState::Error(error) => div()
                    .absolute()
                    .inset_0()
                    .flex()
                    .items_center()
                    .justify_center()
                    .child(format!("Basic map renderer unavailable: {error}"))
                    .into_any_element(),
                MapRenderState::Empty => div().into_any_element(),
            })
            .children(markers.into_iter().enumerate().filter_map(|(marker_position, marker)| {
                let (x, y) =
                    map_screen_position(marker.coordinate, camera, self.map_render_size)?;
                if !(-0.05..=1.05).contains(&x) || !(-0.05..=1.05).contains(&y) {
                    return None;
                }
                let photo_indices = marker.photo_indices.clone();
                let cluster_id = marker.cluster_id;
                let expansion_zoom = marker.expansion_zoom;
                let count = marker.count;
                let coordinate = marker.coordinate;
                let marker_id = cluster_id.map_or_else(
                    || {
                        photo_indices
                            .iter()
                            .map(usize::to_string)
                            .collect::<Vec<_>>()
                            .join("-")
                    },
                    |id| format!("cluster-{id}"),
                );
                Some(
                    div()
                        .id(SharedString::from(format!("map-marker-{marker_id}")))
                        .debug_selector(move || format!("#map-marker-{marker_position}"))
                        .absolute()
                        .left(relative(x))
                        .top(relative(y))
                        .min_w(px(28.0))
                        .h(px(28.0))
                        .px_2()
                        .rounded_full()
                        .bg(rgb(0xd46a4c))
                        .border_2()
                        .border_color(rgb(0xffded4))
                        .text_color(rgb(0xffffff))
                        .text_sm()
                        .flex()
                        .items_center()
                        .justify_center()
                        .cursor_pointer()
                        .block_mouse_except_scroll()
                        .hover(|this| this.bg(rgb(0xf07b58)))
                        .child(if count > 1 {
                            count.to_string()
                        } else {
                            String::new()
                        })
                        .on_mouse_down(MouseButton::Left, |_, _, cx| cx.stop_propagation())
                        .on_mouse_up(
                            MouseButton::Left,
                            cx.listener(move |this, _, _, cx| {
                                this.activate_map_marker(
                                    coordinate,
                                    photo_indices.clone(),
                                    cluster_id,
                                    count,
                                    expansion_zoom,
                                    cx,
                                );
                                cx.stop_propagation();
                            }),
                        ),
                )
            }))
            .when(located == 0 && !can_show_recursively, |this| {
                this.child(
                    div()
                        .absolute()
                        .inset_0()
                        .flex()
                        .items_center()
                        .justify_center()
                        .text_color(rgb(0x7897a3))
                        .child("No location data for this selection"),
                )
            })
            .when(total == 0 && can_show_recursively, |this| {
                this.child(
                    div()
                        .absolute()
                        .inset_0()
                        .flex()
                        .flex_col()
                        .items_center()
                        .justify_center()
                        .gap_3()
                        .text_color(rgb(0x9fb0b8))
                        .child("No images directly in this directory")
                        .child(
                            div()
                                .id("map-show-images-recursively")
                                .debug_selector(|| "#map-show-images-recursively".to_owned())
                                .px_4()
                                .py_2()
                                .rounded_sm()
                                .bg(rgb(0x315b8f))
                                .text_color(rgb(0xffffff))
                                .hover(|this| this.bg(rgb(0x3d6fa9)))
                                .cursor_pointer()
                                .child("Show images from all subdirectories")
                                .on_click(cx.listener(|this, _, _, cx| {
                                    this.start_recursive_grid(cx);
                                })),
                        ),
                )
            })
            .when(
                self.settings.online_map_permission == config::OnlineMapPermission::Unknown,
                |this| {
                    this.child(
                        div()
                            .absolute()
                            .inset_0()
                            .occlude()
                            .flex()
                            .items_center()
                            .justify_center()
                            .bg(gpui::rgba(0x081015aa))
                            .child(
                                div()
                                    .w(px(520.0))
                                    .p_5()
                                    .rounded_md()
                                    .bg(rgb(0x1b252d))
                                    .text_color(rgb(0xe3e8ec))
                                    .child("Enable online map detail?")
                                    .child(
                                        div()
                                            .mt_2()
                                            .text_sm()
                                            .text_color(rgb(0xaebbc3))
                                            .child("OpenFreeMap receives tile coordinates for the area you view. Rekesz sends no photos, filenames, or photo metadata."),
                                    )
                                    .child(
                                        div()
                                            .mt_4()
                                            .flex()
                                            .gap_2()
                                            .child(map_control("enable-online-map", "Enable Online Detail").on_mouse_up(
                                                MouseButton::Left,
                                                cx.listener(|this, _, _, cx| {
                                                    this.set_online_map_permission(config::OnlineMapPermission::Enabled, cx);
                                                    cx.stop_propagation();
                                                }),
                                            ))
                                            .child(map_control("keep-map-offline", "Keep Offline").on_mouse_up(
                                                MouseButton::Left,
                                                cx.listener(|this, _, _, cx| {
                                                    this.set_online_map_permission(config::OnlineMapPermission::Disabled, cx);
                                                    cx.stop_propagation();
                                                }),
                                            )),
                                    ),
                            ),
                    )
                },
            )
            .child(
                div()
                    .absolute()
                    .left(px(12.0))
                    .top(px(12.0))
                    .px_3()
                    .py_2()
                    .rounded_sm()
                    .bg(gpui::rgba(0x101820dd))
                    .text_sm()
                    .text_color(rgb(0xc8d4db))
                    .child(format!("Located {located} of {total} images"))
                    .child(div().text_color(rgb(0x8fa3ad)).child(status)),
            )
            .child(
                div()
                    .absolute()
                    .right(px(12.0))
                    .top(px(12.0))
                    .flex()
                    .gap_1()
                    .child(map_control("map-zoom-out", "−").on_mouse_down(
                        MouseButton::Left,
                        cx.listener(|this, _, _, cx| {
                            this.dismiss_map_card();
                            this.map_camera.zoom_by(-1.0);
                            this.map_fit_node = None;
                            this.map_pending_fit = None;
                            this.request_map_render(cx);
                            cx.stop_propagation();
                            cx.notify();
                        }),
                    ))
                    .child(map_control("map-zoom-in", "+").on_mouse_down(
                        MouseButton::Left,
                        cx.listener(|this, _, _, cx| {
                            this.dismiss_map_card();
                            this.map_camera.zoom_by(1.0);
                            this.map_fit_node = None;
                            this.map_pending_fit = None;
                            this.request_map_render(cx);
                            cx.stop_propagation();
                            cx.notify();
                        }),
                    ))
                    .child(map_control("map-fit", "Fit").on_mouse_down(
                        MouseButton::Left,
                        cx.listener(|this, _, _, cx| {
                            this.dismiss_map_card();
                            this.fit_map_to_selected_images();
                            this.map_pending_fit = None;
                            this.request_map_render(cx);
                            cx.stop_propagation();
                            cx.notify();
                        }),
                    )),
            )
            .child(
                div()
                    .absolute()
                    .left(px(12.0))
                    .bottom(px(10.0))
                    .text_xs()
                    .text_color(rgb(0x9fb0b8))
                    .child(map::ATTRIBUTION),
            )
            .when_some(self.map_card.clone(), |this, card| {
                let (anchor_x, anchor_y) =
                    map_screen_position(card.coordinate, camera, self.map_render_size)
                        .unwrap_or((0.5, 0.5));
                let popup_left = (anchor_x * self.map_render_size.0 as f32 + 18.0)
                    .clamp(8.0, (self.map_render_size.0 as f32 - 368.0).max(8.0));
                let popup_top = (anchor_y * self.map_render_size.1 as f32 - 24.0)
                    .clamp(8.0, (self.map_render_size.1 as f32 - 310.0).max(8.0));
                let tiles = card.page_items().iter().enumerate().map(|(position, item)| {
                    let path = item.path.clone();
                    let identity = self.render_identity_for(&path);
                    let thumbnail = match self.thumbnails.state_for(&path, identity) {
                        PreviewState::Loaded(preview) => img(preview.image.clone())
                            .size_full()
                            .object_fit(gpui::ObjectFit::Cover)
                            .into_any_element(),
                        PreviewState::Error(_) => div()
                            .size_full()
                            .flex()
                            .items_center()
                            .justify_center()
                            .text_xs()
                            .child("Unavailable")
                            .into_any_element(),
                        _ => div()
                            .size_full()
                            .flex()
                            .items_center()
                            .justify_center()
                            .text_xs()
                            .child("Loading…")
                            .into_any_element(),
                    };
                    div()
                        .id(SharedString::from(format!(
                            "map-gallery-{}",
                            item.path.display()
                        )))
                        .debug_selector(move || format!("#map-gallery-item-{position}"))
                        .w(px(104.0))
                        .h(px(104.0))
                        .flex_none()
                        .overflow_hidden()
                        .rounded_sm()
                        .border_1()
                        .border_color(rgb(0x495463))
                        .hover(|this| this.border_color(rgb(0xffded4)))
                        .cursor_pointer()
                        .child(thumbnail)
                        .on_mouse_up(
                            MouseButton::Left,
                            cx.listener(move |this, _, _, cx| {
                                this.show_image(path.clone(), cx);
                                cx.stop_propagation();
                            }),
                        )
                });
                this.child(
                    div()
                        .absolute()
                        .left(px(popup_left))
                        .top(px(popup_top))
                        .w(px(360.0))
                        .p_3()
                        .rounded_md()
                        .bg(gpui::rgba(0x151b22f4))
                        .text_color(rgb(0xe3e8ec))
                        .on_mouse_down(MouseButton::Left, |_, _, cx| cx.stop_propagation())
                        .child(
                            div()
                                .flex()
                                .items_center()
                                .justify_between()
                                .child(format!("{} photos here", card.total))
                                .child(map_control("map-card-close", "×").on_mouse_down(
                                    MouseButton::Left,
                                    cx.listener(|this, _, _, cx| {
                                        this.dismiss_map_card();
                                        cx.stop_propagation();
                                        cx.notify();
                                    }),
                                )),
                        )
                        .child(div().mt_2().flex().flex_wrap().gap_2().children(tiles))
                        .child(
                            div()
                                .flex()
                                .items_center()
                                .justify_between()
                                .gap_2()
                                .mt_2()
                                .when(
                                    card.expansion_zoom
                                        .is_some_and(|zoom| f64::from(zoom) > camera.zoom),
                                    |this| {
                                        this.child(
                                            map_control("map-card-zoom", "Zoom to cluster")
                                                .on_mouse_down(
                                                    MouseButton::Left,
                                                    cx.listener(|this, _, _, cx| {
                                                        this.zoom_to_map_card(cx);
                                                        cx.stop_propagation();
                                                    }),
                                                ),
                                        )
                                    },
                                )
                                .when(card.page_count() > 1, |this| {
                                    this.child(format!(
                                        "Page {} of {}",
                                        card.page + 1,
                                        card.page_count()
                                    ))
                                    .child(map_control("map-card-previous", "←").on_mouse_down(
                                        MouseButton::Left,
                                        cx.listener(|this, _, _, cx| {
                                            this.navigate_map_card(-1, cx);
                                            cx.stop_propagation();
                                        }),
                                    ))
                                    .child(map_control("map-card-next", "→").on_mouse_down(
                                        MouseButton::Left,
                                        cx.listener(|this, _, _, cx| {
                                            this.navigate_map_card(1, cx);
                                            cx.stop_propagation();
                                        }),
                                    ))
                                }),
                        ),
                )
            })
            .into_any_element()
    }

    fn activate_map_marker(
        &mut self,
        coordinate: metadata::GeoCoordinate,
        photo_indices: Vec<usize>,
        cluster_id: Option<usize>,
        count: usize,
        expansion_zoom: Option<u8>,
        cx: &mut Context<Self>,
    ) {
        let direct_indices = cluster_id.is_none().then_some(photo_indices.clone());
        let dense_indices = cluster_id
            .and_then(|cluster_id| {
                self.map_clusters
                    .borrow()
                    .as_ref()
                    .and_then(|(_, clusters)| clusters.as_ref())
                    .map(|clusters| {
                        clusters.cluster_leaves_page(cluster_id, MAP_GALLERY_PAGE_SIZE, 0)
                    })
            })
            .unwrap_or_else(|| {
                photo_indices
                    .into_iter()
                    .take(MAP_GALLERY_PAGE_SIZE)
                    .collect()
            });
        let items = self.map_card_items(dense_indices);
        self.map_card = MapCard::new(coordinate, items, count, cluster_id, expansion_zoom);
        if let Some(card) = &mut self.map_card {
            card.direct_indices = direct_indices;
        }
        if self.map_card.is_some() {
            self.request_map_card_thumbnail(cx);
        }
        cx.notify();
    }

    fn request_map_card_thumbnail(&mut self, cx: &mut Context<Self>) {
        let Some(card) = self.map_card.as_ref() else {
            self.scheduler.clear_scope(MAP_CARD_SCOPE);
            return;
        };
        let requested = card
            .page_items()
            .iter()
            .map(|item| {
                (
                    item.path.clone(),
                    RenderVariant::Thumbnail { width: 120 },
                    JobPriority::INTERACTIVE,
                )
            })
            .collect();
        self.replace_render_scope(MAP_CARD_SCOPE, requested, cx);
    }

    fn navigate_map_card(&mut self, delta: isize, cx: &mut Context<Self>) {
        if let Some(card) = &mut self.map_card {
            card.navigate(delta);
        }
        let page_request = self.map_card.as_ref().map(|card| {
            (
                card.cluster_id,
                card.direct_indices.clone(),
                card.page.saturating_mul(MAP_GALLERY_PAGE_SIZE),
            )
        });
        if let Some((cluster_id, direct_indices, offset)) = page_request {
            let dense_indices = if let Some(cluster_id) = cluster_id {
                self.map_clusters
                    .borrow()
                    .as_ref()
                    .and_then(|(_, clusters)| clusters.as_ref())
                    .map(|clusters| {
                        clusters.cluster_leaves_page(cluster_id, MAP_GALLERY_PAGE_SIZE, offset)
                    })
                    .unwrap_or_default()
            } else {
                direct_indices
                    .unwrap_or_default()
                    .into_iter()
                    .skip(offset)
                    .take(MAP_GALLERY_PAGE_SIZE)
                    .collect()
            };
            let items = self.map_card_items(dense_indices);
            if let Some(card) = &mut self.map_card {
                card.items = items;
            }
        }
        self.request_map_card_thumbnail(cx);
        cx.notify();
    }

    fn map_card_items(&self, dense_indices: Vec<usize>) -> Vec<MapCardItem> {
        let selected_images = self.selected_images();
        let located_images = self.selected_located_images();
        dense_indices
            .into_iter()
            .filter_map(|dense| located_images.get(dense))
            .filter_map(|(index, _)| {
                selected_images
                    .get(*index)
                    .cloned()
                    .map(|path| MapCardItem { path })
            })
            .collect()
    }

    fn zoom_to_map_card(&mut self, cx: &mut Context<Self>) {
        let Some(card) = &self.map_card else {
            return;
        };
        let Some(expansion_zoom) = card.expansion_zoom else {
            return;
        };
        self.map_camera.center = card.coordinate;
        self.map_camera.zoom = f64::from(expansion_zoom)
            .max(self.map_camera.zoom + 1.0)
            .min(20.0);
        self.map_fit_node = None;
        self.map_pending_fit = None;
        self.dismiss_map_card();
        self.request_map_render(cx);
        cx.notify();
    }

    fn dismiss_map_card(&mut self) {
        self.map_card = None;
        self.scheduler.clear_scope(MAP_CARD_SCOPE);
    }

    fn map_mouse_down(&mut self, event: &MouseDownEvent, _: &mut Window, cx: &mut Context<Self>) {
        self.dismiss_map_card();
        self.map_drag_last = Some((f32::from(event.position.x), f32::from(event.position.y)));
        self.map_drag_moved = false;
        cx.stop_propagation();
        cx.notify();
    }

    fn map_mouse_move(&mut self, event: &MouseMoveEvent, _: &mut Window, cx: &mut Context<Self>) {
        if !event.dragging() {
            return;
        }
        let position = (f32::from(event.position.x), f32::from(event.position.y));
        let Some(previous) = self.map_drag_last.replace(position) else {
            return;
        };
        let delta_x = position.0 - previous.0;
        let delta_y = position.1 - previous.1;
        if delta_x == 0.0 && delta_y == 0.0 {
            return;
        }
        self.map_drag_moved = true;
        let world_pixels = 256.0 * 2_f64.powf(self.map_camera.zoom);
        let (center_x, center_y) = map::mercator_project(self.map_camera.center);
        if let Some(center) = map::mercator_unproject(
            center_x - f64::from(delta_x) / world_pixels,
            center_y - f64::from(delta_y) / world_pixels,
        ) {
            self.map_camera.center = center;
        }
        self.map_fit_node = None;
        self.map_pending_fit = None;
        cx.notify();
    }

    fn map_mouse_up(&mut self, _: &MouseUpEvent, _: &mut Window, cx: &mut Context<Self>) {
        self.finish_map_drag(cx);
    }

    fn finish_map_drag(&mut self, cx: &mut Context<Self>) {
        if self.map_drag_last.take().is_some() {
            if self.map_drag_moved {
                self.request_map_render(cx);
            }
            self.map_drag_moved = false;
            cx.notify();
        }
    }

    fn map_scroll_wheel(
        &mut self,
        event: &ScrollWheelEvent,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let delta = f32::from(event.delta.pixel_delta(px(20.0)).y);
        if delta.abs() < 1.0 {
            return;
        }
        cx.stop_propagation();
        let zoom_delta = (-f64::from(delta) / 120.0).clamp(-1.0, 1.0);
        let anchor = (
            f64::from(event.position.x) - f64::from(self.map_view_origin.0),
            f64::from(event.position.y) - f64::from(self.map_view_origin.1),
        );
        self.map_camera =
            zoom_map_camera(self.map_camera, zoom_delta, anchor, self.map_render_size);
        self.map_fit_node = None;
        self.map_pending_fit = None;
        self.request_map_render(cx);
        cx.notify();
    }

    fn start_recursive_grid(&mut self, cx: &mut Context<Self>) {
        let node = self.selected;
        let Some(contents) = self.model.loaded(node) else {
            return;
        };
        if !contents.images.is_empty() || contents.children.is_empty() {
            return;
        }
        let root = self.model.node(node).path.to_path_buf();
        let cancelled = Arc::new(AtomicBool::new(false));
        self.recursive_grid = Some(RecursiveGridState {
            root: node,
            status: RecursiveGridStatus::Loading,
            cancelled: cancelled.clone(),
        });
        self.map_pending_fit = Some(node);
        self.map_card = None;
        self.scheduler.clear_scope(MAP_CARD_SCOPE);
        self.reset_grid_prediction();
        if let MainView::Dir(_, selection) = &mut self.main_view {
            *selection = ImageSelection::default();
        }
        let work = self.work.start(format!(
            "Scanning subdirectories of {}…",
            path_display_name(&root)
        ));
        let scanner = self.directory_scanner.clone();
        let metadata_loader = self.metadata_loader.clone();
        let pair_preference = self.pair_preference;
        let hidden = self.hidden_subtrees.clone();
        let task_cancelled = cancelled.clone();
        let loading = cx.background_executor().spawn(async move {
            collect_recursive_images(
                &scanner,
                &metadata_loader,
                root,
                pair_preference,
                &hidden,
                task_cancelled,
            )
        });
        cx.spawn(async move |this, cx| {
            let result = loading.await;
            this.update(cx, |this, cx| {
                this.work.finish(work);
                let is_active = this.recursive_grid.as_ref().is_some_and(|active| {
                    active.root == node && Arc::ptr_eq(&active.cancelled, &cancelled)
                });
                if !is_active || cancelled.load(Ordering::Acquire) {
                    cx.notify();
                    return;
                }
                if let Some(active) = &mut this.recursive_grid {
                    active.status = match result {
                        Ok(images) => RecursiveGridStatus::Loaded(images),
                        Err(error) => RecursiveGridStatus::Error(error.to_string().into()),
                    };
                }
                if this.directory_view_mode == DirectoryViewMode::Map
                    && this.map_pending_fit == Some(node)
                {
                    this.fit_map_to_selected_images();
                    this.map_pending_fit = None;
                    this.request_map_render(cx);
                }
                this.reset_grid_prediction();
                cx.notify();
            })
            .ok();
        })
        .detach();
        cx.notify();
    }

    fn stop_recursive_grid(&mut self, cx: &mut Context<Self>) {
        if let Some(recursive) = self.recursive_grid.take() {
            recursive.cancelled.store(true, Ordering::Release);
            self.map_fit_node = None;
            self.map_pending_fit = None;
            self.map_card = None;
            self.scheduler.clear_scope(MAP_CARD_SCOPE);
            if self.directory_view_mode == DirectoryViewMode::Map {
                self.fit_map_to_selected_images();
                self.request_map_render(cx);
            }
            self.reset_grid_prediction();
            if let MainView::Dir(_, selection) = &mut self.main_view {
                *selection = ImageSelection::default();
            }
            cx.notify();
        }
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
                            .debug_selector(move || format!("#thumbnail-{index}"))
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
                                        directory: None,
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
        self.visible = visible_rows(
            &self.model,
            &self.expanded,
            &self.hidden_subtrees,
            self.hidden_expanded,
        );
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
                if row.node == HIDDEN_NODE {
                    return div()
                        .id("hidden-subtrees-node")
                        .flex()
                        .items_center()
                        .h(px(30.0))
                        .pl(px(10.0))
                        .pr_2()
                        .text_sm()
                        .text_color(rgb(0xaeb8c8))
                        .hover(|this| this.bg(rgb(0x292d36)))
                        .child(
                            div()
                                .w(px(20.0))
                                .flex_none()
                                .text_color(rgb(0x8d96a5))
                                .child(if self.hidden_expanded { "▾" } else { "▸" }),
                        )
                        .child("Hidden")
                        .on_click(cx.listener(|this, _, _, cx| {
                            this.hidden_expanded = !this.hidden_expanded;
                            this.rebuild_visible();
                            cx.notify();
                        }))
                        .into_any_element();
                }
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
                    .debug_selector(move || format!("#directory-{row_ix}"))
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
                                directory: Some(select_node),
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
        self.face_job_queue_snapshot = self.face_analysis.snapshot();
        self.recognition_job_queue_snapshot = self.recognition_refresh.snapshot();
        self.summary_job_queue_snapshot = self.people_summary_rebuild.snapshot();
        if self.show_job_queue {
            self.refresh_maintenance_diagnostics(cx);
        }
        cx.notify();
    }

    fn refresh_maintenance_diagnostics(&mut self, cx: &mut Context<Self>) {
        if !self.maintenance_diagnostics_throttle.request() {
            return;
        }
        self.start_maintenance_diagnostics_query(cx);
    }

    fn start_maintenance_diagnostics_query(&mut self, cx: &mut Context<Self>) {
        let root = self.settings.root.clone();
        let diagnostics = cx
            .background_executor()
            .spawn(async move { catalog::people::diagnostics(&root) });
        cx.spawn(async move |this, cx| {
            let result = diagnostics.await;
            this.update(cx, |this, cx| {
                this.people_diagnostics = Some(
                    result
                        .map(Arc::new)
                        .map_err(|error| Arc::from(format!("Diagnostics failed: {error:#}"))),
                );
                let trailing = this.maintenance_diagnostics_throttle.complete();
                if trailing {
                    this.start_maintenance_diagnostics_query(cx);
                }
                cx.notify();
            })?;
            Ok::<_, anyhow::Error>(())
        })
        .detach();
    }

    fn schedule_maintenance_throughput_expiry(&mut self, cx: &mut Context<Self>) {
        if self.maintenance_throughput_expiry_scheduled {
            return;
        }
        let Some(delay) = MaintenanceCoordinator::PRODUCTION
            .throughput_after(self.face_analysis.stats().completed_last_minute)
        else {
            return;
        };
        self.maintenance_throughput_expiry_scheduled = true;
        let timer = cx.background_executor().timer(delay);
        cx.spawn(async move |this, cx| {
            timer.await;
            this.update(cx, |this, cx| {
                this.maintenance_throughput_expiry_scheduled = false;
                if MaintenanceCoordinator::PRODUCTION
                    .throughput_after(this.face_analysis.stats().completed_last_minute)
                    .is_some()
                {
                    this.schedule_maintenance_throughput_expiry(cx);
                }
                cx.notify();
            })?;
            Ok::<_, anyhow::Error>(())
        })
        .detach();
    }

    fn open_settings(&mut self, _: &OpenSettings, _: &mut Window, cx: &mut Context<Self>) {
        self.settings_draft = Some(self.settings.clone());
        self.show_cache_statistics = false;
        self.settings_status = None;
        self.show_file_menu = false;
        cx.notify();
    }

    fn open_cache_statistics(
        &mut self,
        _: &OpenCacheStatistics,
        _: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.show_cache_statistics(cx);
    }

    fn show_cache_statistics(&mut self, cx: &mut Context<Self>) {
        self.show_cache_statistics = true;
        self.disk_cache_usage = None;
        self.settings_draft = None;
        self.show_file_menu = false;
        cx.notify();
        let pipeline = self.render_pipeline.clone();
        let usage = cx
            .background_executor()
            .spawn(async move { pipeline.disk_cache_usage().map_err(Arc::<str>::from) });
        cx.spawn(async move |this, cx| {
            let usage = usage.await;
            this.update(cx, |this, cx| {
                this.disk_cache_usage = Some(usage);
                cx.notify();
            })
            .ok();
        })
        .detach();
    }

    fn map_cache_usage(&self) -> u64 {
        let path = self.settings.cache_root.join("map/ambient.sqlite3");
        [
            path.clone(),
            path.with_extension("sqlite3-wal"),
            path.with_extension("sqlite3-shm"),
        ]
        .iter()
        .filter_map(|path| std::fs::metadata(path).ok())
        .map(|metadata| metadata.len())
        .sum()
    }

    fn clear_map_cache(&mut self, cx: &mut Context<Self>) {
        let path = self.settings.cache_root.join("map/ambient.sqlite3");
        let epoch = self.map_render_epoch.wrapping_add(1);
        self.map_render_epoch = epoch;
        self.map_render_generation.store(epoch, Ordering::Release);
        self.map_network_stats = Default::default();
        self.map_render = MapRenderState::Empty;
        self.map_job = None;
        self.map_cache_clearing = true;
        self.settings_status = Some("Clearing downloaded map tiles…".to_owned());
        let retry_executor = cx.background_executor().clone();
        let clearing = cx.background_executor().spawn(async move {
            let _render_guard = loop {
                if MAP_RENDER_BUSY
                    .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
                {
                    break MapRenderGuard;
                }
                retry_executor.timer(Duration::from_millis(16)).await;
            };
            for candidate in [
                path.clone(),
                path.with_extension("sqlite3-wal"),
                path.with_extension("sqlite3-shm"),
            ] {
                match std::fs::remove_file(candidate) {
                    Ok(()) => {}
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                    Err(error) => return Err(format!("Could not clear map cache: {error}")),
                }
            }
            Ok::<_, String>(())
        });
        cx.spawn(async move |this, cx| {
            let result = clearing.await;
            this.update(cx, |this, cx| {
                this.map_cache_clearing = false;
                this.settings_status = Some(match result {
                    Ok(()) => "Downloaded map tiles cleared.".to_owned(),
                    Err(error) => error,
                });
                if this.directory_view_mode == DirectoryViewMode::Map {
                    this.request_map_render(cx);
                }
                cx.notify();
            })
            .ok();
        })
        .detach();
    }

    fn clear_render_disk_cache(&mut self, cx: &mut Context<Self>) {
        let renderer = self.render_pipeline.clone();
        self.settings_status = Some("Clearing persistent image cache…".to_owned());
        let clearing = cx
            .background_executor()
            .spawn(async move { renderer.clear_disk_cache() });
        cx.spawn(async move |this, cx| {
            let result = clearing.await;
            this.update(cx, |this, cx| {
                this.settings_status = Some(match result {
                    Ok(()) => "Persistent image cache cleared.".to_owned(),
                    Err(error) => format!("Could not clear persistent image cache: {error}"),
                });
                this.show_cache_statistics(cx);
                cx.notify();
            })?;
            Ok::<_, anyhow::Error>(())
        })
        .detach();
    }

    fn save_settings(&mut self, cx: &mut Context<Self>) {
        let Some(draft) = self.settings_draft.clone() else {
            return;
        };
        match config::save(&self.settings_path, &draft) {
            Ok(()) => {
                let purge_face_crops = self.settings.face_crop_disk_cache
                    == config::FaceCropDiskCachePolicy::Disk
                    && draft.face_crop_disk_cache == config::FaceCropDiskCachePolicy::MemoryOnly;
                let cache_size_changed = draft.image_cache_size != self.settings.image_cache_size;
                // Publish the stricter policy before starting the purge. Any
                // scheduler work racing the clear immediately stops both disk
                // reads and writes for face crops and cannot repopulate it.
                self.face_crop_disk_cache_enabled.store(
                    draft.face_crop_disk_cache == config::FaceCropDiskCachePolicy::Disk,
                    Ordering::Release,
                );
                self.previews.set_capacity(draft.image_cache_size);
                self.raw_previews.set_capacity(draft.image_cache_size);
                self.thumbnails
                    .set_capacity(draft.image_cache_size.saturating_mul(16).max(64));
                self.render_pipeline
                    .set_source_cache_capacity(draft.image_cache_size);
                self.pair_preference = draft.pair_preference.into();
                self.settings = draft;
                self.settings_draft = None;
                self.settings_status = Some(if cache_size_changed {
                    "Settings saved. Image cache size applied immediately; restart Rekesz to apply service and library changes."
                } else {
                    "Settings saved. Restart Rekesz to apply service and library changes."
                }.into());
                if self.directory_view_mode == DirectoryViewMode::Map {
                    self.request_map_render(cx);
                }
                // Face-crop cache keys deliberately contain no separate privacy-policy
                // generation. Purge the shared derivative cache on the stricter
                // transition so a crop written under Disk can never be read later.
                if purge_face_crops {
                    self.clear_render_disk_cache(cx);
                }
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
            SettingsNumber::FaceWorkers => &mut draft.face_workers,
            SettingsNumber::FaceMemoryMiB => {
                let mib = 1024 * 1024;
                let current = draft.face_worker_memory_bytes.div_ceil(mib).max(64);
                draft.face_worker_memory_bytes =
                    current.saturating_add_signed(delta as i64).max(64) * mib;
                cx.notify();
                return;
            }
            SettingsNumber::MapCacheGiB => {
                let gib = 1024_u64.pow(3);
                let current = draft.map_cache_bytes.div_ceil(gib).max(1);
                draft.map_cache_bytes = current.saturating_add_signed(delta as i64).max(1) * gib;
                cx.notify();
                return;
            }
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
            this.update(cx, |this, cx| {
                this.lightroom_import_dialog = Some(LightroomImportDialog {
                    path: catalog_path.clone(),
                    state: LightroomImportState::Running,
                });
                cx.notify();
            })
            .ok();
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
            let import_catalog = catalog_path.clone();
            let result = background
                .spawn(async move {
                    let report = catalog::import_lightroom(&import_root, &import_catalog)?;
                    let recipes = catalog::load_recipes(&import_root)?;
                    Ok::<_, anyhow::Error>((report, recipes))
                })
                .await;
            this.update(cx, |this, cx| {
                this.work.finish(work);
                match result {
                    Ok((report, recipes)) => {
                        this.recipes = Some(Arc::new(recipes));
                        this.lightroom_import_dialog = Some(LightroomImportDialog {
                            path: catalog_path.clone(),
                            state: LightroomImportState::Succeeded {
                                images: report.matched_files,
                                recipes: report.imported_recipes,
                                faces: report.imported_faces,
                            },
                        });
                        if this.directory_view_mode == DirectoryViewMode::People {
                            this.refresh_people_page(cx);
                        }
                        this.previews.clear();
                        this.raw_previews.clear();
                        this.thumbnails.clear();
                        if this.map_card.is_some() {
                            this.request_map_card_thumbnail(cx);
                        }
                    }
                    Err(error) => {
                        let message: Arc<str> = Arc::from(format!("{error:#}"));
                        eprintln!("cannot import Lightroom catalog: {message}");
                        this.lightroom_import_dialog = Some(LightroomImportDialog {
                            path: catalog_path.clone(),
                            state: LightroomImportState::Failed(message),
                        });
                    }
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

    fn set_subtree_hidden(&mut self, path: PathBuf, hidden: bool, cx: &mut Context<Self>) {
        self.path_context_menu = None;
        if !self.hidden_subtrees_loaded {
            if !self.hidden_save_in_flight {
                self.hidden_save_in_flight = true;
                let root = self.model.node(self.model.root()).path.to_path_buf();
                let loading = cx
                    .background_executor()
                    .spawn(async move { catalog::load_hidden_subtrees(&root) });
                cx.spawn(async move |this, cx| {
                    let result = loading.await;
                    this.update(cx, |this, cx| {
                        this.hidden_save_in_flight = false;
                        match result {
                            Ok(hidden) => {
                                this.hidden_subtrees = hidden;
                                this.hidden_subtrees_loaded = true;
                                this.rebuild_visible();
                                this.settings_status =
                                    Some("Hidden folders loaded; repeat your change.".to_owned());
                            }
                            Err(error) => {
                                this.settings_status =
                                    Some(format!("Could not load hidden folders: {error:#}"));
                            }
                        }
                        cx.notify();
                    })
                    .ok();
                })
                .detach();
            }
            self.settings_status = Some("Loading hidden folders; try again shortly.".to_owned());
            cx.notify();
            return;
        }
        if self.hidden_save_in_flight {
            self.settings_status =
                Some("A hidden-folder change is already being saved.".to_owned());
            cx.notify();
            return;
        }
        let mut desired = self.hidden_subtrees.clone();
        if hidden {
            if desired.iter().any(|ancestor| path.starts_with(ancestor)) {
                cx.notify();
                return;
            }
            desired.retain(|descendant| !descendant.starts_with(&path));
            desired.insert(path);
        } else {
            desired.remove(&path);
        }
        let root = self.model.node(self.model.root()).path.to_path_buf();
        self.hidden_save_in_flight = true;
        let saving = cx
            .background_executor()
            .spawn(async move { catalog::save_hidden_subtrees(&root, &desired).map(|()| desired) });
        cx.spawn(async move |this, cx| {
            let result = saving.await;
            this.update(cx, |this, cx| {
                this.hidden_save_in_flight = false;
                match result {
                    Ok(desired) => {
                        let preserve_fit = this.map_pending_fit == Some(this.selected)
                            || this.map_fit_node == Some(this.selected);
                        let restart_recursive = this.recursive_grid.is_some();
                        if let Some(recursive) = this.recursive_grid.take() {
                            recursive.cancelled.store(true, Ordering::Release);
                            this.reset_grid_prediction();
                        }
                        this.map_card = None;
                        this.scheduler.clear_scope(MAP_CARD_SCOPE);
                        this.hidden_subtrees = desired;
                        this.rebuild_visible();
                        if restart_recursive {
                            this.start_recursive_grid(cx);
                            if !preserve_fit {
                                this.map_pending_fit = None;
                            }
                        }
                    }
                    Err(error) => {
                        this.settings_status =
                            Some(format!("Could not save hidden folders: {error:#}"));
                    }
                }
                cx.notify();
            })
            .ok();
        })
        .detach();
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
        if self.map_card.is_some()
            && self.directory_view_mode == DirectoryViewMode::Map
            && matches!(self.main_view, MainView::Dir(_, _))
        {
            self.request_map_card_thumbnail(cx);
        }
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
        cx: &mut Context<Self>,
    ) -> Vec<gpui::AnyElement> {
        range
            .filter_map(|index| {
                if let Some(map_job) = &self.map_job
                    && index == 0
                {
                    let status = match map_job.status {
                        MapJobStatus::Queued => "QUEUE",
                        MapJobStatus::Running => "RUN",
                    };
                    return Some(
                        div()
                            .id("map-job-queue-row")
                            .debug_selector(|| "#map-job-queue-row".to_owned())
                            .h(px(58.0))
                            .px_3()
                            .py_2()
                            .border_b_1()
                            .border_color(rgb(0x30343d))
                            .overflow_hidden()
                            .child(
                                div()
                                    .text_sm()
                                    .text_color(rgb(0xe1e5ec))
                                    .child("Visible map tiles"),
                            )
                            .child(
                                div()
                                    .text_xs()
                                    .truncate()
                                    .text_color(rgb(0x9da7b6))
                                    .child(format!("{status} · {}", map_job.label)),
                            )
                            .into_any_element(),
                    );
                }
                let scheduler_index = index.saturating_sub(usize::from(self.map_job.is_some()));
                if scheduler_index >= self.job_queue_snapshot.jobs.len() {
                    let mut maintenance_index = scheduler_index - self.job_queue_snapshot.jobs.len();
                    if self.face_inventory_progress.stage.load(Ordering::Acquire) != 0 {
                        if maintenance_index == 0 {
                            let label = self
                                .face_inventory_progress
                                .label()
                                .unwrap_or_else(|| "Face inventory maintenance".to_owned());
                            let cancelled = self
                                .face_inventory_cancelled
                                .load(Ordering::Acquire);
                            return Some(
                                div()
                                    .id("face-inventory-job-queue-row")
                                    .debug_selector(|| {
                                        "#face-inventory-job-queue-row".to_owned()
                                    })
                                    .h(px(76.0))
                                    .px_3()
                                    .py_2()
                                    .border_b_1()
                                    .border_color(rgb(0x30343d))
                                    .cursor_pointer()
                                    .child("Face-analysis inventory")
                                    .child(
                                        div()
                                            .text_xs()
                                            .text_color(rgb(0x9da7b6))
                                            .child(format!(
                                                "RUN · maintenance rank -100{} · {label} · click to cancel",
                                                if cancelled { " · cancelling" } else { "" }
                                            )),
                                    )
                                    .on_click(cx.listener(|this, _, _, cx| {
                                        this.face_inventory_cancelled
                                            .store(true, Ordering::Release);
                                        cx.notify();
                                    }))
                                    .into_any_element(),
                            );
                        }
                        maintenance_index -= 1;
                    }
                    let face_snapshot = &self.face_job_queue_snapshot;
                    if let Some(job) = face_snapshot.get(maintenance_index) {
                        let key = job.job.clone();
                        let file = key
                            .root
                            .file_name()
                            .unwrap_or(key.root.as_os_str())
                            .to_string_lossy()
                            .into_owned();
                        return Some(
                            div()
                                .id(("face-analysis-job-queue-row", job.job.asset_id as usize))
                                .debug_selector(|| "#face-analysis-job-queue-row".to_owned())
                                .h(px(76.0))
                                .px_3()
                                .py_2()
                                .border_b_1()
                                .border_color(rgb(0x30343d))
                                .cursor_pointer()
                                .child(format!("Analyze asset {} · {file}", job.job.asset_id))
                                .child(div().text_xs().text_color(rgb(0x9da7b6)).child(format!(
                                    "{:?} · {:?} · rank {}{} · click to cancel",
                                    job.status,
                                    job.stage,
                                    job.rank,
                                    if job.cancelled { " · cancelling" } else { "" }
                                )))
                                .on_click(cx.listener(move |this, _, _, cx| {
                                    this.face_analysis.cancel(&key);
                                    cx.notify();
                                }))
                                .into_any_element(),
                        );
                    }
                    if maintenance_index < face_snapshot.len() {
                        return None;
                    }
                    let mut service_index = maintenance_index - face_snapshot.len();
                    let recognition = self.recognition_refresh.stats();
                    let recognition_snapshot = &self.recognition_job_queue_snapshot;
                    if let Some(job) = recognition_snapshot.get(service_index) {
                        let root = job.job.root.clone();
                        return Some(
                            div()
                                .id("recognition-job-queue-row")
                                .debug_selector(|| "#recognition-job-queue-row".to_owned())
                                .h(px(76.0))
                                .px_3()
                                .py_2()
                                .border_b_1()
                                .border_color(rgb(0x30343d))
                                .cursor_pointer()
                                .child("Recognition maintenance")
                                .child(div().text_xs().text_color(rgb(0x9da7b6)).child(format!(
                                    "{:?} · rank {}{} · {} complete / {} failed · click to cancel{}",
                                    job.status,
                                    job.rank,
                                    if job.cancelled { " · cancelling" } else { "" },
                                    recognition.completed,
                                    recognition.failed,
                                    recognition.last_error.as_deref().map_or_else(
                                        String::new,
                                        |error| format!(" · last error: {error}"),
                                    )
                                )))
                                .on_click(cx.listener(move |this, _, _, cx| {
                                    this.recognition_refresh.cancel(&root);
                                    cx.notify();
                                }))
                                .into_any_element(),
                        );
                    }
                    service_index = service_index.saturating_sub(recognition_snapshot.len());
                    if service_index != 0 {
                        return None;
                    }
                    let summary_job = self.summary_job_queue_snapshot.get(service_index)?;
                    let summary = self.people_summary_rebuild.stats();
                    let root = summary_job.job.root.clone();
                    return Some(
                        div()
                            .id("people-summary-job-queue-row")
                            .debug_selector(|| "#people-summary-job-queue-row".to_owned())
                            .h(px(76.0))
                            .px_3()
                            .py_2()
                            .border_b_1()
                            .border_color(rgb(0x30343d))
                            .cursor_pointer()
                            .child("People summary rebuild")
                            .child(div().text_xs().text_color(rgb(0x9da7b6)).child(format!(
                                "{:?} · rank {}{} · {} complete / {} failed · click to cancel{}",
                                summary_job.status,
                                summary_job.rank,
                                if summary_job.cancelled {
                                    " · cancelling"
                                } else {
                                    ""
                                },
                                summary.completed,
                                summary.failed,
                                summary.last_error.as_deref().map_or_else(
                                    String::new,
                                    |error| format!(" · last error: {error}"),
                                )
                            )))
                            .on_click(cx.listener(move |this, _, _, cx| {
                                this.people_summary_rebuild.cancel(&root);
                                cx.notify();
                            }))
                            .into_any_element(),
                    );
                }
                let job = self.job_queue_snapshot.jobs.get(scheduler_index)?;
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
                        .h(px(76.0))
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

fn people_grid_column_count(available: f32, card_width: f32, gap: f32) -> usize {
    ((available + gap) / (card_width + gap)).floor().max(1.0) as usize
}

fn visible_rows(
    model: &DirectoryModel,
    expanded: &HashSet<NodeId>,
    hidden_subtrees: &HashSet<PathBuf>,
    hidden_expanded: bool,
) -> Vec<VisibleRow> {
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
    fn append_visible(
        model: &DirectoryModel,
        expanded: &HashSet<NodeId>,
        hidden_subtrees: &HashSet<PathBuf>,
        node: NodeId,
        depth: usize,
        rows: &mut Vec<VisibleRow>,
    ) {
        if node != model.root()
            && hidden_subtrees
                .iter()
                .any(|hidden| model.node(node).path.starts_with(hidden))
        {
            return;
        }
        rows.push(VisibleRow { node, depth });
        if let Some(contents) = expanded
            .contains(&node)
            .then(|| model.loaded(node))
            .flatten()
        {
            for child in &contents.children {
                append_visible(model, expanded, hidden_subtrees, *child, depth + 1, rows);
            }
        }
    }
    append_visible(model, expanded, hidden_subtrees, model.root(), 0, &mut rows);
    if !hidden_subtrees.is_empty() {
        rows.push(VisibleRow {
            node: HIDDEN_NODE,
            depth: 0,
        });
        if hidden_expanded {
            let mut hidden = hidden_subtrees.iter().collect::<Vec<_>>();
            hidden.sort();
            for path in hidden {
                if let Some(node) = model.node_id(path) {
                    append(model, expanded, node, 1, &mut rows);
                }
            }
        }
    }
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

fn people_loading_skeletons(count: usize) -> AnyElement {
    div()
        .flex()
        .flex_wrap()
        .gap_3()
        .children((0..count).map(|index| {
            div()
                .id(("people-skeleton", index))
                .w(px(220.0))
                .h(px(245.0))
                .rounded_md()
                .bg(rgb(0x252932))
        }))
        .into_any_element()
}

fn people_sort_label(sort: catalog::people::PeopleSort) -> &'static str {
    match sort {
        catalog::people::PeopleSort::Name => "Name",
        catalog::people::PeopleSort::Confirmed => "Confirmed",
        catalog::people::PeopleSort::Review => "Review",
        catalog::people::PeopleSort::RecentlyChanged => "Recent",
    }
}

fn next_people_sort(sort: catalog::people::PeopleSort) -> catalog::people::PeopleSort {
    match sort {
        catalog::people::PeopleSort::Name => catalog::people::PeopleSort::Confirmed,
        catalog::people::PeopleSort::Confirmed => catalog::people::PeopleSort::Review,
        catalog::people::PeopleSort::Review => catalog::people::PeopleSort::RecentlyChanged,
        catalog::people::PeopleSort::RecentlyChanged => catalog::people::PeopleSort::Name,
    }
}

fn people_error_panel(error: &Arc<str>, cx: &mut Context<PictureBrowser>) -> AnyElement {
    div()
        .p_6()
        .rounded_md()
        .bg(rgb(0x3d292d))
        .child(div().child(format!("Could not load People: {error}")))
        .child(
            div()
                .id("people-retry")
                .tab_index(0)
                .mt_3()
                .px_3()
                .py_1()
                .rounded_sm()
                .bg(rgb(0x613e42))
                .cursor_pointer()
                .child("Retry")
                .on_key_down(cx.listener(|this, event: &gpui::KeyDownEvent, _, cx| {
                    if matches!(event.keystroke.key.as_str(), "enter" | "space") {
                        this.refresh_people_page(cx);
                        cx.stop_propagation();
                    }
                }))
                .on_click(cx.listener(|this, _, _, cx| this.refresh_people_page(cx))),
        )
        .into_any_element()
}

fn people_page_controls(
    page: usize,
    total: u64,
    page_size: u64,
    cx: &mut Context<PictureBrowser>,
) -> AnyElement {
    let has_previous = page > 0;
    let has_next = ((page as u64).saturating_add(1)).saturating_mul(page_size) < total;
    div()
        .flex()
        .items_center()
        .child(
            div()
                .id("people-page-previous")
                .debug_selector(|| "#people-page-previous".to_owned())
                .tab_index(if has_previous { 0 } else { -1 })
                .px_2()
                .text_color(rgb(if has_previous { 0xd7dbe2 } else { 0x676d78 }))
                .when(has_previous, |element| {
                    element
                        .cursor_pointer()
                        .on_click(cx.listener(|this, _, _, cx| {
                            this.change_people_page(-1, cx);
                        }))
                        .on_key_down(cx.listener(|this, event: &gpui::KeyDownEvent, _, cx| {
                            if matches!(event.keystroke.key.as_str(), "enter" | "space") {
                                this.change_people_page(-1, cx);
                                cx.stop_propagation();
                            }
                        }))
                })
                .child("Previous"),
        )
        .child(div().px_2().child(format!("Page {}", page + 1)))
        .child(
            div()
                .id("people-page-next")
                .debug_selector(|| "#people-page-next".to_owned())
                .tab_index(if has_next { 0 } else { -1 })
                .px_2()
                .text_color(rgb(if has_next { 0xd7dbe2 } else { 0x676d78 }))
                .when(has_next, |element| {
                    element
                        .cursor_pointer()
                        .on_click(cx.listener(|this, _, _, cx| {
                            this.change_people_page(1, cx);
                        }))
                        .on_key_down(cx.listener(|this, event: &gpui::KeyDownEvent, _, cx| {
                            if matches!(event.keystroke.key.as_str(), "enter" | "space") {
                                this.change_people_page(1, cx);
                                cx.stop_propagation();
                            }
                        }))
                })
                .child("Next"),
        )
        .into_any_element()
}

fn people_face_image(
    cache: &PreviewCache,
    path: &std::path::Path,
    identity: RenderIdentity,
    variant: RenderVariant,
    context_bounds: Option<[f32; 4]>,
) -> AnyElement {
    match cache.state_for_variant(path, identity, variant) {
        PreviewState::Loaded(preview) => {
            let image =
                img(preview.image.clone())
                    .size_full()
                    .object_fit(if context_bounds.is_some() {
                        gpui::ObjectFit::Contain
                    } else {
                        gpui::ObjectFit::Cover
                    });
            let Some(canonical_bounds) = context_bounds else {
                return image.into_any_element();
            };
            let bounds = face_overlay_bounds(&preview.metadata, canonical_bounds);
            let (image_width, image_height) = preview.metadata.output_dimensions;
            let aspect = image_width as f32 / image_height.max(1) as f32;
            let (offset_x, offset_y, scale_x, scale_y) = if aspect >= 1.0 {
                (0.0, (1.0 - 1.0 / aspect) * 0.5, 1.0, 1.0 / aspect)
            } else {
                ((1.0 - aspect) * 0.5, 0.0, aspect, 1.0)
            };
            div()
                .relative()
                .size_full()
                .child(image)
                .child(
                    div()
                        .absolute()
                        .left(relative(offset_x + bounds[0] * scale_x))
                        .top(relative(offset_y + bounds[1] * scale_y))
                        .w(relative((bounds[2] - bounds[0]) * scale_x))
                        .h(relative((bounds[3] - bounds[1]) * scale_y))
                        .border_2()
                        .border_color(rgb(0x61d18b)),
                )
                .into_any_element()
        }
        PreviewState::Error(_) => div().child("Unavailable").into_any_element(),
        _ => div().child("Loading…").into_any_element(),
    }
}

fn people_review_tile(
    browser: &PictureBrowser,
    face: &catalog::people::FaceReviewItem,
    cx: &mut Context<PictureBrowser>,
) -> AnyElement {
    let face_id = face.face_id;
    let person_id = face.person_id;
    let selection_key = PeopleFaceKey { face_id, person_id };
    let selected = browser
        .people_face_selection
        .selected
        .contains(&selection_key);
    let row_key = format!("{face_id}-{person_id}");
    let path = face.path.clone();
    let keyboard_path = path.clone();
    let file_name = path.file_name().map_or_else(
        || path.display().to_string(),
        |name| name.to_string_lossy().into_owned(),
    );
    let pending = browser
        .pending_people_decisions
        .iter()
        .any(|(pending_face_id, _)| *pending_face_id == face_id);
    let tile_selector = format!("#people-face-tile-{row_key}");
    let context_mode = browser.people_face_uses_context(selection_key);
    let variant = if context_mode {
        RenderVariant::face_context(320)
    } else {
        RenderVariant::face_crop(face_id, face.bounds, 320)
    };
    let mut tile = div()
        .id(SharedString::from(format!("people-face-tile-{row_key}")))
        .debug_selector(move || tile_selector)
        .tab_group()
        .tab_index(0)
        .w(px(220.0))
        .p_3()
        .rounded_md()
        .bg(rgb(if selected { 0x303f54 } else { 0x252932 }))
        .border_2()
        .border_color(rgb(if selected { 0x70a5df } else { 0x3b4658 }))
        .on_click(cx.listener(move |this, _: &ClickEvent, window, cx| {
            let modifiers = window.modifiers();
            this.people_grid_focus.focus(window);
            this.select_people_face(selection_key, modifiers.shift, modifiers.secondary(), cx);
        }))
        .on_key_down(
            cx.listener(move |this, event: &gpui::KeyDownEvent, window, cx| {
                if matches!(event.keystroke.key.as_str(), "enter" | "space") {
                    let modifiers = window.modifiers();
                    this.select_people_face(
                        selection_key,
                        modifiers.shift,
                        modifiers.secondary(),
                        cx,
                    );
                    cx.stop_propagation();
                }
            }),
        )
        .child(
            div()
                .relative()
                .w(px(190.0))
                .h(px(190.0))
                .overflow_hidden()
                .rounded_sm()
                .child(people_face_image(
                    &browser.thumbnails,
                    &face.path,
                    browser.render_identity_for(&face.path),
                    variant,
                    context_mode.then_some(face.bounds),
                ))
                .child(
                    div()
                        .id(SharedString::from(format!("toggle-face-context-{row_key}")))
                        .debug_selector(move || {
                            format!("#toggle-face-context-{face_id}-{person_id}")
                        })
                        .tab_index(0)
                        .absolute()
                        .top_2()
                        .right_2()
                        .px_2()
                        .py_1()
                        .rounded_sm()
                        .bg(rgb(0x30343d))
                        .cursor_pointer()
                        .child(if context_mode { "Crop" } else { "Context" })
                        .on_key_down(cx.listener(move |this, event: &gpui::KeyDownEvent, _, cx| {
                            if matches!(event.keystroke.key.as_str(), "enter" | "space") {
                                this.toggle_people_face_context(selection_key, cx);
                                cx.stop_propagation();
                            }
                        }))
                        .on_click(cx.listener(move |this, _, _, cx| {
                            this.toggle_people_face_context(selection_key, cx);
                            cx.stop_propagation();
                        })),
                ),
        )
        .child(
            div()
                .mt_2()
                .text_sm()
                .text_color(rgb(0xaeb4bf))
                .child(format!(
                    "{}{:?} · {}{}",
                    face.person_name
                        .as_ref()
                        .map_or_else(String::new, |name| format!("Suggested: {name} · ")),
                    face.origin,
                    face.provenance,
                    face.confidence.map_or_else(String::new, |confidence| {
                        format!(" · {:.0}%", confidence * 100.0)
                    })
                )),
        )
        .child(div().text_xs().text_color(rgb(0x8f98a8)).child(file_name));
    if face.kind == catalog::people::FaceAssignmentKind::Suggested {
        tile = tile.child(
            div()
                .flex()
                .mt_2()
                .gap_2()
                .child(
                    div()
                        .id(SharedString::from(format!("accept-face-{row_key}")))
                        .debug_selector(move || format!("#accept-face-{face_id}-{person_id}"))
                        .tab_index(0)
                        .px_3()
                        .py_1()
                        .rounded_sm()
                        .bg(rgb(if pending { 0x3b4658 } else { 0x365f48 }))
                        .when(!pending, |button| button.cursor_pointer())
                        .child(if pending { "Saving…" } else { "Accept" })
                        .on_key_down(cx.listener(move |this, event: &gpui::KeyDownEvent, _, cx| {
                            if !pending && matches!(event.keystroke.key.as_str(), "enter" | "space")
                            {
                                this.apply_people_decision(
                                    face_id,
                                    person_id,
                                    PeopleDecision::Confirm,
                                    cx,
                                );
                            }
                        }))
                        .on_click(cx.listener(move |this, _, _, cx| {
                            if !pending {
                                this.apply_people_decision(
                                    face_id,
                                    person_id,
                                    PeopleDecision::Confirm,
                                    cx,
                                );
                            }
                            cx.stop_propagation();
                        })),
                )
                .child(
                    div()
                        .id(SharedString::from(format!("reject-face-{row_key}")))
                        .debug_selector(move || format!("#reject-face-{face_id}-{person_id}"))
                        .tab_index(0)
                        .px_3()
                        .py_1()
                        .rounded_sm()
                        .bg(rgb(if pending { 0x3b4658 } else { 0x613e42 }))
                        .when(!pending, |button| button.cursor_pointer())
                        .child(if pending { "Saving…" } else { "Reject" })
                        .on_key_down(cx.listener(move |this, event: &gpui::KeyDownEvent, _, cx| {
                            if !pending && matches!(event.keystroke.key.as_str(), "enter" | "space")
                            {
                                this.apply_people_decision(
                                    face_id,
                                    person_id,
                                    PeopleDecision::Reject,
                                    cx,
                                );
                            }
                        }))
                        .on_click(cx.listener(move |this, _, _, cx| {
                            if !pending {
                                this.apply_people_decision(
                                    face_id,
                                    person_id,
                                    PeopleDecision::Reject,
                                    cx,
                                );
                            }
                            cx.stop_propagation();
                        })),
                ),
        );
    } else if face.kind == catalog::people::FaceAssignmentKind::Confirmed {
        tile = tile
            .child(div().mt_2().text_color(rgb(0x75b58b)).child("Confirmed"))
            .child(
                div()
                    .id(SharedString::from(format!("reject-face-{row_key}")))
                    .debug_selector(move || format!("#reject-face-{face_id}-{person_id}"))
                    .tab_index(0)
                    .mt_2()
                    .px_3()
                    .py_1()
                    .rounded_sm()
                    .bg(rgb(if pending { 0x3b4658 } else { 0x613e42 }))
                    .when(!pending, |button| button.cursor_pointer())
                    .child(if pending {
                        "Saving…"
                    } else {
                        "Reject identity"
                    })
                    .on_key_down(cx.listener(move |this, event: &gpui::KeyDownEvent, _, cx| {
                        if !pending && matches!(event.keystroke.key.as_str(), "enter" | "space") {
                            this.apply_people_decision(
                                face_id,
                                person_id,
                                PeopleDecision::Reject,
                                cx,
                            );
                        }
                    }))
                    .on_click(cx.listener(move |this, _, _, cx| {
                        if !pending {
                            this.apply_people_decision(
                                face_id,
                                person_id,
                                PeopleDecision::Reject,
                                cx,
                            );
                        }
                        cx.stop_propagation();
                    })),
            );
    } else {
        let state = match face.kind {
            catalog::people::FaceAssignmentKind::Rejected => "Rejected identity",
            catalog::people::FaceAssignmentKind::Ignored => "Ignored — not a face",
            _ => "Unassigned",
        };
        tile = tile.child(div().mt_2().text_color(rgb(0xaeb4bf)).child(state));
    }
    tile.child(
        div()
            .flex()
            .mt_2()
            .gap_2()
            .child(
                div()
                    .id(SharedString::from(format!("open-face-photo-{row_key}")))
                    .tab_index(0)
                    .px_2()
                    .py_1()
                    .rounded_sm()
                    .bg(rgb(0x30343d))
                    .cursor_pointer()
                    .child("Open photo")
                    .on_key_down(cx.listener(move |this, event: &gpui::KeyDownEvent, _, cx| {
                        if matches!(event.keystroke.key.as_str(), "enter" | "space") {
                            this.show_image(keyboard_path.clone(), cx);
                        }
                    }))
                    .on_click(cx.listener(move |this, _, _, cx| {
                        this.show_image(path.clone(), cx);
                        cx.stop_propagation();
                    })),
            )
            .children(
                (face.kind != catalog::people::FaceAssignmentKind::Ignored).then(|| {
                    div()
                        .id(SharedString::from(format!("ignore-face-{row_key}")))
                        .debug_selector(move || format!("#ignore-face-{face_id}-{person_id}"))
                        .tab_index(0)
                        .px_2()
                        .py_1()
                        .rounded_sm()
                        .bg(rgb(if pending { 0x3b4658 } else { 0x613e42 }))
                        .when(!pending, |button| button.cursor_pointer())
                        .child(if pending { "Saving…" } else { "Not a face" })
                        .on_key_down(cx.listener(move |this, event: &gpui::KeyDownEvent, _, cx| {
                            if !pending && matches!(event.keystroke.key.as_str(), "enter" | "space")
                            {
                                this.apply_people_decision(
                                    face_id,
                                    person_id,
                                    PeopleDecision::NotFace,
                                    cx,
                                );
                            }
                        }))
                        .on_click(cx.listener(move |this, _, _, cx| {
                            if !pending {
                                this.apply_people_decision(
                                    face_id,
                                    person_id,
                                    PeopleDecision::NotFace,
                                    cx,
                                );
                            }
                            cx.stop_propagation();
                        }))
                }),
            ),
    )
    .into_any_element()
}

fn people_operation_dialog(
    browser: &PictureBrowser,
    dialog: &PeopleOperationDialog,
    cx: &mut Context<PictureBrowser>,
) -> AnyElement {
    let (title, body) = match dialog {
        PeopleOperationDialog::Name {
            operation, input, ..
        } => {
            let (title, consequence) = match operation {
                PeopleNameOperation::Rename { .. } => {
                    ("Rename person", "Enter a new name.".to_owned())
                }
                PeopleNameOperation::Create { face_ids } => (
                    "Create person",
                    format!(
                        "Assign {} selected face(s) to the new person.",
                        face_ids.len()
                    ),
                ),
            };
            let _ = input;
            (title, consequence)
        }
        PeopleOperationDialog::Target { operation, .. } => {
            let body = match operation {
                PeopleTargetOperation::Merge { .. } => {
                    "Choose the surviving person. Exact consequences are calculated before confirmation.".to_owned()
                }
                PeopleTargetOperation::Move { face_ids, .. } => format!(
                    "Choose the target for {} confirmed face(s). This removes those confirmations from the current person.",
                    face_ids.len()
                ),
                PeopleTargetOperation::Assign { face_ids } => format!(
                    "Choose the existing person for {} selected face(s).",
                    face_ids.len()
                ),
            };
            ("Choose target person", body)
        }
    };
    let panel = div()
        .w(px(560.0))
        .max_h(px(620.0))
        .overflow_hidden()
        .p_5()
        .rounded_md()
        .bg(rgb(0x252932))
        .border_1()
        .border_color(rgb(0x52749a))
        .child(div().text_lg().mb_3().child(title))
        .child(div().mb_3().text_color(rgb(0xaeb4bf)).child(body))
        .children(
            matches!(dialog, PeopleOperationDialog::Name { .. }).then(|| {
                div()
                    .p_3()
                    .mb_3()
                    .rounded_sm()
                    .border_1()
                    .border_color(rgb(0x6d91b8))
                    .child(match dialog {
                        PeopleOperationDialog::Name {
                            input,
                            cursor,
                            selection_anchor,
                            ..
                        } => people_editor_display(input, *cursor, *selection_anchor),
                        _ => String::new(),
                    })
            }),
        )
        .children(
            matches!(dialog, PeopleOperationDialog::Target { .. }).then(|| {
                uniform_list(
                    "people-target-list",
                    browser.people_operation_targets.len(),
                    cx.processor(PictureBrowser::render_people_operation_targets),
                )
                .h(px(380.0))
                .track_scroll(browser.people_operation_target_scroll.clone())
                .with_sizing_behavior(ListSizingBehavior::Auto)
            }),
        )
        .children(
            matches!(dialog, PeopleOperationDialog::Target { .. }).then(|| {
                let page = browser.people_operation_target_page;
                let has_next =
                    (page + 1).saturating_mul(100) < browser.people_operation_target_total as usize;
                div()
                    .flex()
                    .items_center()
                    .justify_between()
                    .mb_3()
                    .child(
                        div()
                            .id("people-target-previous")
                            .debug_selector(|| "#people-target-previous".to_owned())
                            .tab_index(0)
                            .px_3()
                            .py_1()
                            .rounded_sm()
                            .bg(rgb(0x30343d))
                            .child("Previous")
                            .on_key_down(cx.listener(
                                move |this, event: &gpui::KeyDownEvent, _, cx| {
                                    if page > 0
                                        && !this.people_operation_target_loading
                                        && matches!(event.keystroke.key.as_str(), "enter" | "space")
                                    {
                                        this.load_people_operation_target_page(page - 1, cx);
                                        cx.stop_propagation();
                                    }
                                },
                            ))
                            .on_click(cx.listener(move |this, _, _, cx| {
                                if page > 0 && !this.people_operation_target_loading {
                                    this.load_people_operation_target_page(page - 1, cx);
                                }
                            })),
                    )
                    .child(format!(
                        "Page {} · {} people{}",
                        page + 1,
                        browser.people_operation_target_total,
                        if browser.people_operation_target_loading {
                            " · Loading…"
                        } else {
                            ""
                        }
                    ))
                    .child(
                        div()
                            .id("people-target-next")
                            .debug_selector(|| "#people-target-next".to_owned())
                            .tab_index(0)
                            .px_3()
                            .py_1()
                            .rounded_sm()
                            .bg(rgb(0x30343d))
                            .child("Next")
                            .on_key_down(cx.listener(
                                move |this, event: &gpui::KeyDownEvent, _, cx| {
                                    if has_next
                                        && !this.people_operation_target_loading
                                        && matches!(event.keystroke.key.as_str(), "enter" | "space")
                                    {
                                        this.load_people_operation_target_page(page + 1, cx);
                                        cx.stop_propagation();
                                    }
                                },
                            ))
                            .on_click(cx.listener(move |this, _, _, cx| {
                                if has_next && !this.people_operation_target_loading {
                                    this.load_people_operation_target_page(page + 1, cx);
                                }
                            })),
                    )
            }),
        )
        .children(browser.people_operation_status.as_ref().map(|error| {
            div()
                .mb_3()
                .text_color(rgb(0xffa0a0))
                .child(error.to_string())
        }))
        .children(browser.people_merge_preview.as_ref().map(|preview| {
            let text = match preview {
                PeopleMergePreviewLoad::Loading(target) => {
                    format!("Calculating merge consequences for target {target}…")
                }
                PeopleMergePreviewLoad::Loaded(preview) => people_merge_preview_text(preview),
                PeopleMergePreviewLoad::Error(target, error) => {
                    format!("Target {target}: {error}")
                }
            };
            div()
                .mb_3()
                .p_3()
                .rounded_sm()
                .bg(rgb(0x31445b))
                .child(text)
        }))
        .child(
            div()
                .flex()
                .justify_end()
                .gap_2()
                .child(
                    div()
                        .id("cancel-people-operation")
                        .debug_selector(|| "#cancel-people-operation".to_owned())
                        .tab_index(0)
                        .px_3()
                        .py_2()
                        .rounded_sm()
                        .bg(rgb(0x30343d))
                        .when(!browser.people_operation_pending, |button| {
                            button.cursor_pointer()
                        })
                        .child(if browser.people_operation_pending {
                            "Saving…"
                        } else {
                            "Cancel"
                        })
                        .on_key_down(cx.listener(|this, event: &gpui::KeyDownEvent, _, cx| {
                            if !this.people_operation_pending
                                && matches!(event.keystroke.key.as_str(), "enter" | "space")
                            {
                                this.dismiss_people_operation();
                                cx.stop_propagation();
                                cx.notify();
                            }
                        }))
                        .on_click(cx.listener(|this, _, _, cx| {
                            if !this.people_operation_pending {
                                this.dismiss_people_operation();
                            }
                            cx.notify();
                        })),
                )
                .child(
                    div()
                        .id("confirm-people-operation")
                        .debug_selector(|| "#confirm-people-operation".to_owned())
                        .tab_index(0)
                        .px_3()
                        .py_2()
                        .rounded_sm()
                        .bg(rgb(0x365f48))
                        .when(!browser.people_operation_pending, |button| {
                            button.cursor_pointer()
                        })
                        .child(if browser.people_operation_pending {
                            "Saving…"
                        } else {
                            "Confirm"
                        })
                        .on_key_down(cx.listener(|this, event: &gpui::KeyDownEvent, _, cx| {
                            if !this.people_operation_pending
                                && matches!(event.keystroke.key.as_str(), "enter" | "space")
                            {
                                this.submit_people_operation(cx);
                                cx.stop_propagation();
                            }
                        }))
                        .on_click(cx.listener(|this, _, _, cx| {
                            this.submit_people_operation(cx);
                        })),
                ),
        );
    div()
        .absolute()
        .inset_0()
        .flex()
        .items_center()
        .justify_center()
        .bg(gpui::rgba(0x000000aa))
        .track_focus(&browser.people_operation_focus)
        .on_key_down(cx.listener(|this, event: &gpui::KeyDownEvent, _, cx| {
            this.handle_people_operation_key(event, cx);
            cx.stop_propagation();
        }))
        .child(panel)
        .into_any_element()
}

fn people_special_overview_card(
    review: bool,
    count: u64,
    cx: &mut Context<PictureBrowser>,
) -> AnyElement {
    let (id, title, suffix) = if review {
        ("people-review-queue", "Review queue", "suggestions")
    } else {
        ("people-unnamed", "Unnamed faces", "faces")
    };
    div()
        .id(id)
        .debug_selector(move || format!("#{id}"))
        .tab_index(0)
        .w(px(220.))
        .p_4()
        .rounded_md()
        .bg(rgb(0x2d3542))
        .cursor_pointer()
        .child(div().text_lg().child(title))
        .child(format!("{count} {suffix}"))
        .on_click(cx.listener(move |this, _, _, cx| {
            this.selected_person = None;
            this.people_special_page = Some(if review {
                PeopleSpecialPage::Review
            } else {
                PeopleSpecialPage::Unnamed
            });
            this.people_page = 0;
            this.refresh_people_page(cx);
        }))
        .on_key_down(cx.listener(move |this, event: &gpui::KeyDownEvent, _, cx| {
            if matches!(event.keystroke.key.as_str(), "enter" | "space") {
                this.selected_person = None;
                this.people_special_page = Some(if review {
                    PeopleSpecialPage::Review
                } else {
                    PeopleSpecialPage::Unnamed
                });
                this.people_page = 0;
                this.refresh_people_page(cx);
                cx.stop_propagation();
            }
        }))
        .into_any_element()
}

fn people_selection_toolbar(
    selected_count: usize,
    selected_person: Option<i64>,
    cx: &mut Context<PictureBrowser>,
) -> AnyElement {
    let action = |id: &'static str,
                  label: &'static str,
                  color: u32,
                  decision: PeopleDecision,
                  cx: &mut Context<PictureBrowser>| {
        div()
            .id(id)
            .debug_selector(move || format!("#{id}"))
            .tab_index(0)
            .px_3()
            .py_2()
            .rounded_sm()
            .bg(rgb(color))
            .cursor_pointer()
            .child(label)
            .on_key_down(
                cx.listener(move |this, event: &gpui::KeyDownEvent, window, cx| {
                    if matches!(event.keystroke.key.as_str(), "enter" | "space") {
                        this.people_grid_focus.focus(window);
                        this.apply_selected_people_decision(decision, cx);
                        cx.stop_propagation();
                    }
                }),
            )
            .on_click(cx.listener(move |this, _, window, cx| {
                this.people_grid_focus.focus(window);
                this.apply_selected_people_decision(decision, cx);
                cx.stop_propagation();
            }))
    };
    div()
        .id("people-selection-actions")
        .debug_selector(|| "#people-selection-actions".to_owned())
        .flex()
        .flex_wrap()
        .items_center()
        .gap_2()
        .mb_3()
        .p_2()
        .rounded_md()
        .bg(rgb(0x202a38))
        .border_1()
        .border_color(rgb(0x52749a))
        .child(format!("{selected_count} selected"))
        .child(action(
            "accept-selected-faces",
            "Accept",
            0x365f48,
            PeopleDecision::Confirm,
            cx,
        ))
        .child(action(
            "reject-selected-faces",
            "Reject",
            0x613e42,
            PeopleDecision::Reject,
            cx,
        ))
        .child(action(
            "ignore-selected-faces",
            "Not a face",
            0x613e42,
            PeopleDecision::NotFace,
            cx,
        ))
        .child(
            div()
                .id("create-person-from-selected")
                .debug_selector(|| "#create-person-from-selected".to_owned())
                .tab_index(0)
                .px_3()
                .py_2()
                .rounded_sm()
                .bg(rgb(0x3f5570))
                .cursor_pointer()
                .child("Create person…")
                .on_key_down(cx.listener(|this, event: &gpui::KeyDownEvent, window, cx| {
                    if matches!(event.keystroke.key.as_str(), "enter" | "space") {
                        match prepare_selected_face_ids(&this.people_face_selection) {
                            Ok(face_ids) => this.open_people_name_dialog(
                                PeopleNameOperation::Create { face_ids },
                                String::new(),
                                window,
                                cx,
                            ),
                            Err(error) => {
                                this.people_operation_status = Some(Arc::from(error));
                                cx.notify();
                            }
                        }
                        cx.stop_propagation();
                    }
                }))
                .on_click(cx.listener(|this, _, window, cx| {
                    match prepare_selected_face_ids(&this.people_face_selection) {
                        Ok(face_ids) => this.open_people_name_dialog(
                            PeopleNameOperation::Create { face_ids },
                            String::new(),
                            window,
                            cx,
                        ),
                        Err(error) => {
                            this.people_operation_status = Some(Arc::from(error));
                            cx.notify();
                        }
                    }
                    cx.stop_propagation();
                })),
        )
        .children(selected_person.is_none().then(|| {
            div()
                .id("assign-existing-person")
                .debug_selector(|| "#assign-existing-person".to_owned())
                .tab_index(0)
                .px_3()
                .py_2()
                .rounded_sm()
                .bg(rgb(0x3f5570))
                .cursor_pointer()
                .child("Assign to existing…")
                .on_key_down(cx.listener(|this, event: &gpui::KeyDownEvent, window, cx| {
                    if matches!(event.keystroke.key.as_str(), "enter" | "space") {
                        match prepare_selected_face_ids(&this.people_face_selection) {
                            Ok(face_ids) => this.open_people_target_dialog(
                                PeopleTargetOperation::Assign { face_ids },
                                window,
                                cx,
                            ),
                            Err(error) => {
                                this.people_operation_status = Some(Arc::from(error));
                                cx.notify();
                            }
                        }
                        cx.stop_propagation();
                    }
                }))
                .on_click(cx.listener(|this, _, window, cx| {
                    match prepare_selected_face_ids(&this.people_face_selection) {
                        Ok(face_ids) => this.open_people_target_dialog(
                            PeopleTargetOperation::Assign { face_ids },
                            window,
                            cx,
                        ),
                        Err(error) => {
                            this.people_operation_status = Some(Arc::from(error));
                            cx.notify();
                        }
                    }
                    cx.stop_propagation();
                }))
        }))
        .children(selected_person.map(|person_id| {
            div()
                .flex()
                .gap_2()
                .child(
                    div()
                        .id("move-selected-faces")
                        .debug_selector(|| "#move-selected-faces".to_owned())
                        .tab_index(0)
                        .px_3()
                        .py_2()
                        .rounded_sm()
                        .bg(rgb(0x3f5570))
                        .cursor_pointer()
                        .child("Move to…")
                        .on_key_down(cx.listener(
                            move |this, event: &gpui::KeyDownEvent, window, cx| {
                                if matches!(event.keystroke.key.as_str(), "enter" | "space") {
                                    match this.selected_confirmed_face_ids(person_id) {
                                        Ok(face_ids) => this.open_people_target_dialog(
                                            PeopleTargetOperation::Move {
                                                source_person_id: person_id,
                                                face_ids,
                                            },
                                            window,
                                            cx,
                                        ),
                                        Err(error) => {
                                            this.people_operation_status = Some(Arc::from(error));
                                            cx.notify();
                                        }
                                    }
                                    cx.stop_propagation();
                                }
                            },
                        ))
                        .on_click(cx.listener(move |this, _, window, cx| {
                            match this.selected_confirmed_face_ids(person_id) {
                                Ok(face_ids) => this.open_people_target_dialog(
                                    PeopleTargetOperation::Move {
                                        source_person_id: person_id,
                                        face_ids,
                                    },
                                    window,
                                    cx,
                                ),
                                Err(error) => {
                                    this.people_operation_status = Some(Arc::from(error));
                                    cx.notify();
                                }
                            }
                            cx.stop_propagation();
                        })),
                )
                .child(
                    div()
                        .id("set-person-representative")
                        .debug_selector(|| "#set-person-representative".to_owned())
                        .tab_index(0)
                        .px_3()
                        .py_2()
                        .rounded_sm()
                        .bg(rgb(0x3f5570))
                        .cursor_pointer()
                        .child("Use as example")
                        .on_key_down(cx.listener(|this, event: &gpui::KeyDownEvent, _, cx| {
                            if matches!(event.keystroke.key.as_str(), "enter" | "space") {
                                this.choose_selected_people_representative(cx);
                                cx.stop_propagation();
                            }
                        }))
                        .on_click(cx.listener(|this, _, _, cx| {
                            this.choose_selected_people_representative(cx);
                            cx.stop_propagation();
                        })),
                )
        }))
        .child(
            div()
                .id("clear-selected-faces")
                .debug_selector(|| "#clear-selected-faces".to_owned())
                .tab_index(0)
                .px_3()
                .py_2()
                .rounded_sm()
                .bg(rgb(0x30343d))
                .cursor_pointer()
                .child("Clear")
                .on_key_down(cx.listener(|this, event: &gpui::KeyDownEvent, _, cx| {
                    if matches!(event.keystroke.key.as_str(), "enter" | "space") {
                        this.people_face_selection.clear();
                        cx.stop_propagation();
                        cx.notify();
                    }
                }))
                .on_click(cx.listener(|this, _, _, cx| {
                    this.people_face_selection.clear();
                    cx.stop_propagation();
                    cx.notify();
                })),
        )
        .into_any_element()
}

#[cfg(test)]
fn people_representative_faces<'a>(
    people: &'a catalog::people::PeopleCatalog,
    person_id: i64,
    visible_face_ids: &HashSet<i64>,
) -> Vec<&'a catalog::people::FaceRecord> {
    let Some(person) = people.people.iter().find(|person| person.id == person_id) else {
        return Vec::new();
    };
    let confirmed = |face_id| {
        people.assignments.iter().any(|assignment| {
            assignment.person_id == person_id
                && assignment.face_id == face_id
                && assignment.kind == catalog::people::FaceAssignmentKind::Confirmed
        })
    };
    let preferred = person
        .representative_face_id
        .filter(|face_id| visible_face_ids.contains(face_id) && confirmed(*face_id))
        .and_then(|face_id| people.faces.iter().find(|face| face.id == face_id));
    let mut candidates = people
        .assignments
        .iter()
        .filter(|assignment| {
            assignment.person_id == person_id
                && assignment.kind == catalog::people::FaceAssignmentKind::Confirmed
                && visible_face_ids.contains(&assignment.face_id)
        })
        .filter_map(|assignment| {
            people
                .faces
                .iter()
                .find(|face| face.id == assignment.face_id)
                .map(|face| (assignment, face))
        })
        .filter(|(_, face)| {
            let area = (face.bounds[2] - face.bounds[0]) * (face.bounds[3] - face.bounds[1]);
            !face.ignored && area >= 0.0025
        })
        .collect::<Vec<_>>();
    candidates.sort_by(
        |(left_assignment, left_face), (right_assignment, right_face)| {
            let left_local = left_assignment.provenance == "rekesz";
            let right_local = right_assignment.provenance == "rekesz";
            right_local.cmp(&left_local).then_with(|| {
                let area = |face: &catalog::people::FaceRecord| {
                    (face.bounds[2] - face.bounds[0]) * (face.bounds[3] - face.bounds[1])
                };
                area(right_face).total_cmp(&area(left_face))
            })
        },
    );
    let mut faces = candidates
        .into_iter()
        .map(|(_, face)| face)
        .collect::<Vec<_>>();
    if let Some(preferred) = preferred
        && !preferred.ignored
        && let Some(index) = faces.iter().position(|face| face.id == preferred.id)
    {
        let preferred = faces.remove(index);
        faces.insert(0, preferred);
    }
    faces
}

fn face_overlay_bounds(
    metadata: &rendering::RenderMetadata,
    canonical_bounds: [f32; 4],
) -> [f32; 4] {
    metadata.face_transform.map_or_else(
        || {
            rendering::oriented_face_bounds(
                canonical_bounds,
                metadata.orientation_applied.unwrap_or(1),
            )
        },
        |transform| transform.bounds(canonical_bounds),
    )
}

fn collect_recursive_images(
    scanner: &DirectoryScanner,
    metadata_loader: &MetadataLoader,
    root: PathBuf,
    pair_preference: RawJpegPreference,
    hidden_subtrees: &HashSet<PathBuf>,
    cancelled: Arc<AtomicBool>,
) -> Result<RecursiveImages, std::io::Error> {
    let mut pending = vec![root];
    let mut visited = HashSet::new();
    let mut images = Vec::new();
    while let Some(directory) = pending.pop() {
        if cancelled.load(Ordering::Acquire) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Interrupted,
                "recursive scan cancelled",
            ));
        }
        if !visited.insert(directory.clone())
            || hidden_subtrees
                .iter()
                .any(|hidden| directory.starts_with(hidden))
        {
            continue;
        }
        let scan = scanner.scan(
            &directory,
            pair_preference,
            DirectoryScanPriority::Background,
        )?;
        images.extend(scan.images);
        pending.extend(scan.directories);
    }
    let mut metadata = metadata_loader.load_directory(images, cancelled);
    discovery::sort_metadata(&mut metadata);
    let coordinates = metadata
        .iter()
        .filter_map(|entry| {
            entry
                .coordinate
                .map(|coordinate| (entry.path.clone(), coordinate))
        })
        .collect();
    Ok(RecursiveImages {
        paths: metadata.into_iter().map(|entry| entry.path).collect(),
        coordinates,
    })
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
        self.face_job_queue_snapshot = self.face_analysis.snapshot();
        self.recognition_job_queue_snapshot = self.recognition_refresh.snapshot();
        self.summary_job_queue_snapshot = self.people_summary_rebuild.snapshot();
        let face_jobs = self.face_analysis.stats();
        let recognition_jobs = self.recognition_refresh.stats();
        let summary_jobs = self.people_summary_rebuild.stats();
        let maintenance_delta = (
            face_jobs.completed + recognition_jobs.completed + summary_jobs.completed,
            face_jobs.failed + recognition_jobs.failed + summary_jobs.failed,
            face_jobs.cancelled + recognition_jobs.cancelled + summary_jobs.cancelled,
        );
        if maintenance_delta != self.maintenance_diagnostics_observed {
            self.maintenance_diagnostics_observed = maintenance_delta;
            self.refresh_maintenance_diagnostics(cx);
            self.schedule_maintenance_throughput_expiry(cx);
        }
        let face_throughput = face_jobs.completed_last_minute;
        let face_instrumentation = self.face_analysis.instrumentation();
        let durable_face_summary = self
            .people_diagnostics
            .as_ref()
            .and_then(|result| result.as_ref().ok())
            .map_or_else(
                || {
                    format!(
                        "{face_throughput} assets/min · {}/{} inference sessions · summary {}/{}",
                        face_instrumentation.active_inference,
                        face_instrumentation.engines_constructed,
                        summary_jobs.queued,
                        summary_jobs.running
                    )
                },
                |diagnostics| {
                    format!(
                        "{} analyzed · {} pending · {} errors · {} reconciliation conflicts · {face_throughput} assets/min · {}/{} inference sessions (peak {}) · summary {}/{}",
                        diagnostics.indexed_assets,
                        diagnostics.analysis_pending,
                        diagnostics.analysis_errors,
                        diagnostics.reconciliation_conflicts,
                        face_instrumentation.active_inference,
                        face_instrumentation.engines_constructed,
                        face_instrumentation.max_active_inference,
                        summary_jobs.queued,
                        summary_jobs.running,
                    )
                },
            );
        let scheduler_jobs = self
            .job_queue_snapshot
            .jobs
            .iter()
            .filter(|job| !job.cancelled)
            .count()
            + usize::from(self.map_job.is_some())
            + face_jobs.queued
            + face_jobs.running
            + recognition_jobs.queued
            + recognition_jobs.running
            + summary_jobs.queued
            + summary_jobs.running;
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
        let menu_height = if USE_IN_WINDOW_MENU { 29.0 } else { 0.0 };
        let map_width = main_width.round().max(1.0) as u32;
        let map_height = (f32::from(window.viewport_size().height) - menu_height - 44.0)
            .round()
            .max(1.0) as u32;
        self.map_view_origin = (self.sidebar_width + 5.0, menu_height + 44.0);
        if self.directory_view_mode == DirectoryViewMode::Map
            && self.map_render_size != (map_width, map_height)
        {
            self.map_render_size = (map_width, map_height);
            if self.map_fit_node == Some(self.selected) {
                self.fit_map_to_selected_images();
            }
            cx.defer_in(window, |this, _, cx| this.request_map_render(cx));
        }
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
        if self.directory_view_mode != DirectoryViewMode::People
            || !matches!(self.main_view, MainView::Dir(_, _))
        {
            self.people_accessibility.deactivate();
        }
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
                    RenderVariant::Thumbnail { .. }
                    | RenderVariant::FaceCrop { .. }
                    | RenderVariant::FaceContext { .. } => unreachable!(),
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
                        .debug_selector(|| "#image-viewport".to_owned())
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
                                    directory: None,
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
                let recursive = self
                    .recursive_grid
                    .as_ref()
                    .filter(|recursive| recursive.root == self.selected);
                let recursive_active = recursive.is_some();
                let ordering_pending = !recursive_active
                    && self.model.loaded(self.selected).is_some_and(|contents| {
                        contents.metadata_status != DirectoryMetadataStatus::Complete
                            && !contents.images.is_empty()
                    });
                let capture_order_available =
                    !recursive_active && self.model.has_pending_capture_order(self.selected);
                let ordering_suffix = if ordering_pending {
                    " · loading capture metadata…"
                } else {
                    ""
                };
                let thumbnail_grid = || {
                    uniform_list(
                        "thumbnail-rows",
                        count.div_ceil(self.grid_columns),
                        cx.processor(Self::render_thumbnail_rows),
                    )
                    .size_full()
                    .track_scroll(self.thumbnail_scroll.clone())
                    .with_sizing_behavior(ListSizingBehavior::Auto)
                    .into_any_element()
                };
                let grid_content = if let Some(recursive) = recursive {
                    match &recursive.status {
                        RecursiveGridStatus::Loading => div()
                            .p_6()
                            .text_sm()
                            .child("Scanning subdirectories and ordering images…")
                            .into_any_element(),
                        RecursiveGridStatus::Error(error) => div()
                            .p_6()
                            .text_sm()
                            .text_color(rgb(0xf2b8b5))
                            .child(format!("Could not load recursive view: {error}"))
                            .into_any_element(),
                        RecursiveGridStatus::Loaded(_) if count == 0 => div()
                            .p_6()
                            .text_sm()
                            .child("No images in this subtree")
                            .into_any_element(),
                        RecursiveGridStatus::Loaded(_) => thumbnail_grid(),
                    }
                } else {
                    match &self.model.node(self.selected).contents {
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
                        DirectoryContents::Loaded(contents)
                            if count == 0 && !contents.children.is_empty() =>
                        {
                            div()
                                .size_full()
                                .flex()
                                .flex_col()
                                .items_center()
                                .justify_center()
                                .gap_3()
                                .text_sm()
                                .child("No images directly in this directory")
                                .child(
                                    div()
                                        .id("show-images-recursively")
                                        .debug_selector(|| "#show-images-recursively".to_owned())
                                        .px_4()
                                        .py_2()
                                        .rounded_sm()
                                        .bg(rgb(0x315b8f))
                                        .hover(|this| this.bg(rgb(0x3d6fa9)))
                                        .cursor_pointer()
                                        .child("Show images from all subdirectories")
                                        .on_click(cx.listener(|this, _, _, cx| {
                                            this.start_recursive_grid(cx);
                                        })),
                                )
                                .into_any_element()
                        }
                        DirectoryContents::Loaded(_) if count == 0 => div()
                            .p_6()
                            .text_sm()
                            .child("No images in this directory")
                            .into_any_element(),
                        DirectoryContents::Loaded(_) => thumbnail_grid(),
                    }
                };
                if self.directory_view_mode != DirectoryViewMode::People {
                    self.people_accessibility.deactivate();
                }
                let directory_content = match self.directory_view_mode {
                    DirectoryViewMode::Grid => grid_content,
                    DirectoryViewMode::Map => self.render_map_view(cx),
                    DirectoryViewMode::People => self.render_people_view(window, cx),
                };
                let content = div()
                    .size_full()
                    .flex()
                    .flex_col()
                    .when(recursive_active, |this| {
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
                                .child(format!("Recursive view · {count} images"))
                                .child(
                                    div()
                                        .id("exit-recursive-grid")
                                        .debug_selector(|| "#exit-recursive-grid".to_owned())
                                        .px_3()
                                        .py_1()
                                        .rounded_sm()
                                        .bg(rgb(0x3b4658))
                                        .hover(|this| this.bg(rgb(0x52617a)))
                                        .cursor_pointer()
                                        .child("Exit recursive view")
                                        .on_click(cx.listener(|this, _, _, cx| {
                                            this.stop_recursive_grid(cx);
                                        })),
                                ),
                        )
                    })
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
                                        .debug_selector(|| "#apply-capture-time-order".to_owned())
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
                            .child(directory_content)
                            .when(
                                self.directory_view_mode == DirectoryViewMode::Grid,
                                |this| this.child(grid_scrollbar),
                            ),
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
            .relative()
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
                    .when(
                        !show_back
                            && self.settings.map_switcher_placement
                                == config::MapSwitcherPlacement::Header,
                        |this| {
                            this.child(
                                div()
                                    .flex()
                                    .mr_3()
                                    .rounded_sm()
                                    .bg(rgb(0x252932))
                                    .child(
                                        div()
                                            .id("show-grid-view")
                                            .debug_selector(|| "#show-grid-view".to_owned())
                                            .tab_index(0)
                                            .px_3()
                                            .py_1()
                                            .rounded_sm()
                                            .bg(rgb(
                                                if self.directory_view_mode
                                                    == DirectoryViewMode::Grid
                                                {
                                                    0x46658c
                                                } else {
                                                    0x30343d
                                                },
                                            ))
                                            .child("Grid")
                                            .on_key_down(cx.listener(
                                                |this, event: &gpui::KeyDownEvent, _, cx| {
                                                    if matches!(
                                                        event.keystroke.key.as_str(),
                                                        "enter" | "space"
                                                    ) {
                                                        this.set_directory_view_mode(
                                                            DirectoryViewMode::Grid,
                                                            cx,
                                                        );
                                                        cx.stop_propagation();
                                                    }
                                                },
                                            ))
                                            .on_click(cx.listener(|this, _, _, cx| {
                                                this.set_directory_view_mode(
                                                    DirectoryViewMode::Grid,
                                                    cx,
                                                );
                                            })),
                                    )
                                    .child(
                                        div()
                                            .id("show-map-view")
                                            .debug_selector(|| "#show-map-view".to_owned())
                                            .tab_index(0)
                                            .px_3()
                                            .py_1()
                                            .rounded_sm()
                                            .bg(rgb(
                                                if self.directory_view_mode
                                                    == DirectoryViewMode::Map
                                                {
                                                    0x46658c
                                                } else {
                                                    0x30343d
                                                },
                                            ))
                                            .child("Map")
                                            .on_key_down(cx.listener(
                                                |this, event: &gpui::KeyDownEvent, _, cx| {
                                                    if matches!(
                                                        event.keystroke.key.as_str(),
                                                        "enter" | "space"
                                                    ) {
                                                        this.set_directory_view_mode(
                                                            DirectoryViewMode::Map,
                                                            cx,
                                                        );
                                                        cx.stop_propagation();
                                                    }
                                                },
                                            ))
                                            .on_click(cx.listener(|this, _, _, cx| {
                                                this.set_directory_view_mode(
                                                    DirectoryViewMode::Map,
                                                    cx,
                                                );
                                            })),
                                    ),
                            )
                            .child(
                                div()
                                    .id("show-people-view")
                                    .debug_selector(|| "#show-people-view".to_owned())
                                    .tab_index(0)
                                    .track_focus(&self.people_view_switch_focus)
                                    .px_3()
                                    .py_1()
                                    .rounded_sm()
                                    .bg(rgb(
                                        if self.directory_view_mode == DirectoryViewMode::People {
                                            0x46658c
                                        } else {
                                            0x30343d
                                        },
                                    ))
                                    .child("People")
                                    .on_key_down(cx.listener(
                                        |this, event: &gpui::KeyDownEvent, _, cx| {
                                            if matches!(
                                                event.keystroke.key.as_str(),
                                                "enter" | "space"
                                            ) {
                                                this.set_directory_view_mode(
                                                    DirectoryViewMode::People,
                                                    cx,
                                                );
                                                cx.stop_propagation();
                                            }
                                        },
                                    ))
                                    .on_click(cx.listener(|this, _, _, cx| {
                                        this.set_directory_view_mode(DirectoryViewMode::People, cx);
                                    })),
                            )
                        },
                    )
                    .child(
                        div()
                            .id("toggle-developed")
                            .debug_selector(|| "#toggle-developed".to_owned())
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
            )
            .when(
                !show_back
                    && self.settings.map_switcher_placement
                        == config::MapSwitcherPlacement::Floating,
                |this| {
                    this.child(
                        div()
                            .absolute()
                            .bottom(px(18.0))
                            .left_1_2()
                            .flex()
                            .p_1()
                            .rounded_md()
                            .bg(gpui::rgba(0x202630ee))
                            .child(
                                map_control("floating-grid-view", "Grid").on_click(cx.listener(
                                    |this, _, _, cx| {
                                        this.set_directory_view_mode(DirectoryViewMode::Grid, cx);
                                    },
                                )),
                            )
                            .child(
                                map_control("floating-map-view", "Map").on_click(cx.listener(
                                    |this, _, _, cx| {
                                        this.set_directory_view_mode(DirectoryViewMode::Map, cx);
                                    },
                                )),
                            )
                            .child(map_control("floating-people-view", "People").on_click(
                                cx.listener(|this, _, _, cx| {
                                    this.set_directory_view_mode(DirectoryViewMode::People, cx);
                                }),
                            )),
                    )
                },
            );
        let splitter = div()
            .id("sidebar-splitter")
            .debug_selector(|| "#sidebar-splitter".to_owned())
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
                    this.finish_map_drag(cx);
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
                                .child(
                                    div()
                                        .text_xs()
                                        .text_color(rgb(0x778292))
                                        .child(durable_face_summary.clone()),
                                )
                                .child(div().text_xs().text_color(rgb(0x9da7b6)).child(format!(
                                    "{} running · {} queued",
                                    self.job_queue_snapshot.running
                                        + self.face_analysis.stats().running
                                        + self.recognition_refresh.stats().running
                                        + self.people_summary_rebuild.stats().running
                                        + usize::from(self.map_job.as_ref().is_some_and(|job| {
                                            job.status == MapJobStatus::Running
                                        })),
                                    self.job_queue_snapshot.queued
                                        + self.face_analysis.stats().queued
                                        + self.recognition_refresh.stats().queued
                                        + self.people_summary_rebuild.stats().queued
                                        + usize::from(self.map_job.as_ref().is_some_and(|job| {
                                            job.status == MapJobStatus::Queued
                                        }))
                                )))
                                .child(
                                    div()
                                        .id("job-queue-face-analysis-toggle")
                                        .debug_selector(|| {
                                            "#job-queue-face-analysis-toggle".to_owned()
                                        })
                                        .tab_index(0)
                                        .ml_2()
                                        .px_2()
                                        .py_1()
                                        .rounded_sm()
                                        .bg(rgb(0x30343d))
                                        .cursor_pointer()
                                        .child(if self.face_analysis_paused {
                                            "Resume faces"
                                        } else {
                                            "Pause faces"
                                        })
                                        .on_click(cx.listener(|this, _, _, cx| {
                                            this.face_analysis_paused = !this.face_analysis_paused;
                                            this.face_analysis
                                                .set_paused(this.face_analysis_paused);
                                            this.recognition_refresh
                                                .set_paused(this.face_analysis_paused);
                                            this.people_summary_rebuild
                                                .set_paused(this.face_analysis_paused);
                                            if !this.face_analysis_paused {
                                                this.refresh_face_inventory(cx);
                                            }
                                            cx.notify();
                                        })),
                                ),
                        )
                        .child(
                            uniform_list(
                                "job-queue-rows",
                                self.job_queue_snapshot.jobs.len()
                                    + usize::from(self.map_job.is_some())
                                    + usize::from(
                                        self.face_inventory_progress.stage.load(Ordering::Acquire)
                                            != 0,
                                    )
                                    + self.face_job_queue_snapshot.len()
                                    + self.recognition_job_queue_snapshot.len()
                                    + self.summary_job_queue_snapshot.len(),
                                cx.processor(Self::render_job_queue_rows),
                            )
                            .flex_1()
                            .min_h_0()
                            .track_scroll(self.job_queue_scroll.clone())
                            .with_sizing_behavior(ListSizingBehavior::Auto),
                        ),
                )
            });
        let progress_active = self.work.active > 0 || scheduler_jobs > 0 || self.map_job.is_some();
        let progress = if let Some(inventory) = self.face_inventory_progress.fraction() {
            inventory.clamp(0.005, 1.0)
        } else if self.work.active > 0 {
            self.work.fraction().clamp(0.08, 1.0)
        } else {
            0.35
        };
        let progress_label = self
            .map_job
            .as_ref()
            .map(map_job_progress_label)
            .or_else(|| self.face_inventory_progress.label())
            .or_else(|| scheduler_progress_label(&self.job_queue_snapshot))
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
                    .debug_selector(|| "#file-menu-button".to_owned())
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
                    .debug_selector(|| "#view-menu-button".to_owned())
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
                        .debug_selector(|| "#background-progress".to_owned())
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
            let can_hide = menu.directory.is_some_and(|node| node != self.model.root());
            let left = menu
                .position
                .0
                .min((f32::from(viewport.width) - 228.0).max(0.0));
            let top = menu.position.1.min(
                (f32::from(viewport.height)
                    - (38.0
                        + if can_export { 28.0 } else { 0.0 }
                        + if can_hide { 28.0 } else { 0.0 }))
                .max(0.0),
            );
            let is_hidden_root = self.hidden_subtrees.contains(&menu.path);
            (
                menu.path.clone(),
                left,
                top,
                can_export,
                can_hide,
                is_hidden_root,
            )
        });
        let settings_dialog = self
            .settings_draft
            .clone()
            .map(|draft| render_settings_dialog(draft, self.settings_status.clone(), cx));
        let cache_statistics_dialog = self.show_cache_statistics.then(|| {
            render_cache_statistics_dialog(
                self.render_pipeline.stats(),
                self.previews.stats(),
                self.raw_previews.stats(),
                self.thumbnails.stats(),
                self.disk_cache_usage.clone(),
                MapCacheStatistics {
                    usage: self.map_cache_usage(),
                    capacity: self.settings.map_cache_bytes,
                    network: self.map_network_stats,
                },
                cx,
            )
        });
        let lightroom_import_dialog = self
            .lightroom_import_dialog
            .clone()
            .map(|dialog| render_lightroom_import_dialog(dialog, cx));
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
            .on_action(cx.listener(Self::open_cache_statistics))
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
                                .debug_selector(|| "#import-lightroom-menu-item".to_owned())
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
                                .id("cache-statistics-menu-item")
                                .debug_selector(|| "#cache-statistics-menu-item".to_owned())
                                .h(px(28.0))
                                .px_2()
                                .flex()
                                .items_center()
                                .rounded_sm()
                                .text_sm()
                                .hover(|this| this.bg(rgb(0x3b4658)))
                                .child("Cache Statistics…")
                                .on_click(cx.listener(|this, _, window, cx| {
                                    this.open_cache_statistics(&OpenCacheStatistics, window, cx);
                                })),
                        )
                        .child(
                            div()
                                .id("settings-menu-item")
                                .debug_selector(|| "#settings-menu-item".to_owned())
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
            .when_some(
                path_context_menu,
                |this, (path, left, top, can_export, is_directory, is_hidden_root)| {
                    let export_path = path.clone();
                    let hidden_path = path.clone();
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
                                    .debug_selector(|| "#copy-absolute-path-menu-item".to_owned())
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
                            .when(is_directory, |this| {
                                this.child(
                                    div()
                                        .id("toggle-hidden-subtree-menu-item")
                                        .debug_selector(|| {
                                            "#toggle-hidden-subtree-menu-item".to_owned()
                                        })
                                        .h(px(28.0))
                                        .px_2()
                                        .flex()
                                        .items_center()
                                        .rounded_sm()
                                        .text_sm()
                                        .hover(|this| this.bg(rgb(0x3b4658)))
                                        .child(if is_hidden_root {
                                            "Unhide Subtree"
                                        } else {
                                            "Hide Subtree"
                                        })
                                        .on_mouse_down(
                                            MouseButton::Left,
                                            cx.listener(move |this, _, _, cx| {
                                                this.set_subtree_hidden(
                                                    hidden_path.clone(),
                                                    !is_hidden_root,
                                                    cx,
                                                );
                                                cx.stop_propagation();
                                            }),
                                        ),
                                )
                            })
                            .when(can_export, |this| {
                                this.child(
                                    div()
                                        .id("export-png-menu-item")
                                        .debug_selector(|| "#export-png-menu-item".to_owned())
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
                },
            )
            .when_some(settings_dialog, |this, dialog| this.child(dialog))
            .when_some(cache_statistics_dialog, |this, dialog| this.child(dialog))
            .when_some(lightroom_import_dialog, |this, dialog| this.child(dialog))
    }
}

fn render_lightroom_import_dialog(
    dialog: LightroomImportDialog,
    cx: &mut Context<PictureBrowser>,
) -> gpui::Div {
    let running = matches!(dialog.state, LightroomImportState::Running);
    let status = match &dialog.state {
        LightroomImportState::Running => "Reading and importing the Lightroom catalog…".to_owned(),
        LightroomImportState::Succeeded {
            images,
            recipes,
            faces,
        } => format!(
            "Import finished: {images} matched images, {recipes} develop recipes, {faces} faces."
        ),
        LightroomImportState::Failed(error) => format!("Import failed: {error}"),
    };
    div()
        .absolute()
        .inset_0()
        .occlude()
        .bg(gpui::rgba(0x000000aa))
        .flex()
        .items_center()
        .justify_center()
        .child(
            div()
                .id("lightroom-import-dialog")
                .w(px(620.0))
                .p_5()
                .rounded_md()
                .border_1()
                .border_color(rgb(0x4a5260))
                .bg(rgb(0x252932))
                .text_color(rgb(0xd8dde7))
                .child(div().text_xl().mb_2().child("Import Lightroom Catalog"))
                .child(
                    div()
                        .text_sm()
                        .text_color(rgb(0x9da7b6))
                        .mb_4()
                        .child(dialog.path.to_string_lossy().into_owned()),
                )
                .child(status)
                .when(running, |panel| {
                    panel.child(
                        div()
                            .relative()
                            .w_full()
                            .h(px(12.0))
                            .mt_4()
                            .rounded_sm()
                            .overflow_hidden()
                            .bg(rgb(0x181a20))
                            .child(div().h_full().w(relative(0.65)).bg(rgb(0x315b8f))),
                    )
                })
                .when(!running, |panel| {
                    panel.child(
                        div()
                            .id("close-lightroom-import-dialog")
                            .mt_4()
                            .px_4()
                            .py_2()
                            .rounded_sm()
                            .bg(rgb(0x315b8f))
                            .cursor_pointer()
                            .child("Close")
                            .on_click(cx.listener(|this, _, _, cx| {
                                this.lightroom_import_dialog = None;
                                cx.notify();
                            })),
                    )
                }),
        )
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
                        .debug_selector(move || format!("#settings-minus-{label}"))
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
                        .debug_selector(move || format!("#settings-plus-{label}"))
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

fn map_control(id: &'static str, label: &'static str) -> gpui::Stateful<gpui::Div> {
    div()
        .id(id)
        .debug_selector(|| format!("#{id}"))
        .min_w(px(34.0))
        .px_2()
        .py_1()
        .rounded_sm()
        .bg(rgb(0x303b43))
        .text_color(rgb(0xe0e7ea))
        .text_center()
        .cursor_pointer()
        .hover(|this| this.bg(rgb(0x465761)))
        .child(label)
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
                .id("settings-dialog-panel")
                .w(px(620.0))
                .max_h(px(860.0))
                .p_5()
                .rounded_md()
                .bg(rgb(0x252932))
                .text_color(rgb(0xd8dde7))
                .border_1()
                .border_color(rgb(0x4a5260))
                .child(
                    div()
                        .flex()
                        .items_center()
                        .mb_2()
                        .child(div().text_xl().child("Settings"))
                        .child(div().flex_1())
                        .child(
                            div()
                                .id("settings-cancel")
                                .debug_selector(|| "#settings-cancel".to_owned())
                                .px_3()
                                .py_1()
                                .rounded_sm()
                                .bg(rgb(0x343943))
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
                                .debug_selector(|| "#settings-save".to_owned())
                                .ml_2()
                                .px_3()
                                .py_1()
                                .rounded_sm()
                                .bg(rgb(0x315b8f))
                                .child("Save")
                                .on_click(cx.listener(|this, _, _, cx| this.save_settings(cx))),
                        ),
                )
                .child(
                    div()
                        .text_sm()
                        .text_color(rgb(0x9da7b6))
                        .mb_3()
                        .child(
                            "Image cache size applies immediately. Worker, cache directory, and library changes require a restart.",
                        ),
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
                                        .debug_selector(|| "#settings-browse-root".to_owned())
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
                .child(settings_number_row(
                    "Face-analysis workers",
                    draft.face_workers,
                    SettingsNumber::FaceWorkers,
                    cx,
                ))
                .child(settings_number_row(
                    "Face worker memory allowance (MiB)",
                    draft.face_worker_memory_bytes.div_ceil(1024 * 1024) as usize,
                    SettingsNumber::FaceMemoryMiB,
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
                        .child("Automatic face analysis")
                        .child(
                            div()
                                .id("settings-automatic-face-analysis")
                                .px_3()
                                .py_1()
                                .rounded_sm()
                                .bg(rgb(0x343943))
                                .cursor_pointer()
                                .child(if draft.automatic_face_analysis {
                                    "Enabled"
                                } else {
                                    "Disabled"
                                })
                                .on_click(cx.listener(|this, _, _, cx| {
                                    if let Some(draft) = &mut this.settings_draft {
                                        draft.automatic_face_analysis =
                                            !draft.automatic_face_analysis;
                                    }
                                    cx.notify();
                                })),
                        ),
                )
                .child(
                    div()
                        .flex()
                        .items_center()
                        .justify_between()
                        .py_2()
                        .border_b_1()
                        .border_color(rgb(0x30343d))
                        .child("Face-crop disk cache")
                        .child(
                            div()
                                .id("settings-face-crop-disk-cache")
                                .px_3()
                                .py_1()
                                .rounded_sm()
                                .bg(rgb(0x343943))
                                .cursor_pointer()
                                .child(match draft.face_crop_disk_cache {
                                    config::FaceCropDiskCachePolicy::MemoryOnly => "Memory only",
                                    config::FaceCropDiskCachePolicy::Disk => "Local disk",
                                })
                                .on_click(cx.listener(|this, _, _, cx| {
                                    if let Some(draft) = &mut this.settings_draft {
                                        draft.face_crop_disk_cache = match draft.face_crop_disk_cache {
                                            config::FaceCropDiskCachePolicy::MemoryOnly => config::FaceCropDiskCachePolicy::Disk,
                                            config::FaceCropDiskCachePolicy::Disk => config::FaceCropDiskCachePolicy::MemoryOnly,
                                        };
                                    }
                                    cx.notify();
                                })),
                        ),
                )
                .child(settings_number_row(
                    "Downloaded map cache (GiB)",
                    draft.map_cache_bytes.div_ceil(1024_u64.pow(3)) as usize,
                    SettingsNumber::MapCacheGiB,
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
                        .child(
                            div()
                                .max_w(px(430.0))
                                .child("Online map detail")
                                .child(
                                    div()
                                        .mt_1()
                                        .text_xs()
                                        .text_color(rgb(0x9da7b6))
                                        .child("When enabled, only visible tile coordinates are sent to OpenFreeMap. Photos and metadata never leave Rekesz."),
                                ),
                        )
                        .child(
                            div()
                                .id("settings-online-map")
                                .debug_selector(|| "#settings-online-map".to_owned())
                                .px_3()
                                .py_1()
                                .rounded_sm()
                                .bg(rgb(0x343943))
                                .hover(|this| this.bg(rgb(0x465064)))
                                .child(match draft.online_map_permission {
                                    config::OnlineMapPermission::Enabled => "Enabled",
                                    config::OnlineMapPermission::Unknown
                                    | config::OnlineMapPermission::Disabled => "Keep offline",
                                })
                                .on_click(cx.listener(|this, _, _, cx| {
                                    let permission = if this.settings.online_map_permission
                                        == config::OnlineMapPermission::Enabled
                                    {
                                        config::OnlineMapPermission::Disabled
                                    } else {
                                        config::OnlineMapPermission::Enabled
                                    };
                                    this.set_online_map_permission(permission, cx);
                                })),
                        ),
                )
                .child(
                    div()
                        .flex()
                        .items_center()
                        .justify_between()
                        .py_2()
                        .border_b_1()
                        .border_color(rgb(0x30343d))
                        .child("Grid/Map switcher")
                        .child(
                            div()
                                .id("settings-map-switcher-placement")
                                .debug_selector(|| {
                                    "#settings-map-switcher-placement".to_owned()
                                })
                                .px_3()
                                .py_1()
                                .rounded_sm()
                                .bg(rgb(0x343943))
                                .hover(|this| this.bg(rgb(0x465064)))
                                .child(match draft.map_switcher_placement {
                                    config::MapSwitcherPlacement::Header => "Top header",
                                    config::MapSwitcherPlacement::Floating => "Bottom floating",
                                })
                                .on_click(cx.listener(|this, _, _, cx| {
                                    let placement = match this.settings.map_switcher_placement {
                                        config::MapSwitcherPlacement::Header => config::MapSwitcherPlacement::Floating,
                                        config::MapSwitcherPlacement::Floating => config::MapSwitcherPlacement::Header,
                                    };
                                    let mut settings = this.settings.clone();
                                    settings.map_switcher_placement = placement;
                                    if let Err(error) = config::save(&this.settings_path, &settings) {
                                        this.settings_status = Some(format!("Could not save switcher placement: {error:#}"));
                                    } else {
                                        this.settings = settings;
                                        if let Some(draft) = &mut this.settings_draft {
                                            draft.map_switcher_placement = placement;
                                        }
                                    }
                                    cx.notify();
                                })),
                        ),
                )
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
                                .debug_selector(|| "#settings-pair-preference".to_owned())
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
                                    .debug_selector(|| "#settings-browse-cache".to_owned())
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
                                .id("settings-cancel-bottom")
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
                                .id("settings-save-bottom")
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

fn render_cache_statistics_dialog(
    pipeline: rendering::pipeline::PipelineStats,
    previews: preview::PreviewCacheStats,
    raw_previews: preview::PreviewCacheStats,
    thumbnails: preview::PreviewCacheStats,
    disk_usage: Option<Result<rendering::pipeline::DiskCacheUsage, Arc<str>>>,
    map_cache: MapCacheStatistics,
    cx: &mut Context<PictureBrowser>,
) -> gpui::Div {
    let requests = pipeline.cache_hits + pipeline.cache_misses;
    let hit_rate = if requests == 0 {
        "—".to_owned()
    } else {
        format!(
            "{:.1}%",
            pipeline.cache_hits as f64 * 100.0 / requests as f64
        )
    };
    let row = |label: &'static str, value: String| {
        div()
            .flex()
            .justify_between()
            .py_1()
            .child(label)
            .child(div().text_color(rgb(0xb7bfcc)).child(value))
    };
    let memory = |name: &'static str, stats: preview::PreviewCacheStats| {
        row(
            name,
            format!(
                "{} / {} loaded · ~{} · {} loading · {} errors",
                stats.loaded,
                stats.capacity,
                format_byte_size(stats.estimated_memory_bytes),
                stats.loading,
                stats.errors
            ),
        )
    };
    let total_memory_bytes = previews
        .estimated_memory_bytes
        .saturating_add(raw_previews.estimated_memory_bytes)
        .saturating_add(thumbnails.estimated_memory_bytes);
    let map_cache_fraction = cache_capacity_fraction(map_cache.usage, map_cache.capacity);
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
                .w(px(520.0))
                .p_5()
                .rounded_md()
                .bg(rgb(0x252932))
                .text_color(rgb(0xd8dde7))
                .border_1()
                .border_color(rgb(0x4a5260))
                .child(div().text_xl().mb_3().child("Cache Statistics"))
                .child(memory("Image previews", previews))
                .child(memory("RAW previews", raw_previews))
                .child(memory("Thumbnails", thumbnails))
                .child(row(
                    "Estimated image memory",
                    format!("~{}", format_byte_size(total_memory_bytes)),
                ))
                .child(div().my_2().border_t_1().border_color(rgb(0x30343d)))
                .child(render_disk_cache_usage(disk_usage))
                .child(
                    div()
                        .id("clear-persistent-image-cache")
                        .debug_selector(|| "#clear-persistent-image-cache".to_owned())
                        .mt_2()
                        .px_3()
                        .py_2()
                        .rounded_sm()
                        .bg(rgb(0x593b3b))
                        .hover(|this| this.bg(rgb(0x754949)))
                        .child("Clear Persistent Image Cache")
                        .on_click(cx.listener(|this, _, _, cx| this.clear_render_disk_cache(cx))),
                )
                .child(div().my_2().border_t_1().border_color(rgb(0x30343d)))
                .child(row(
                    "Downloaded map tiles",
                    format!(
                        "{} / {}",
                        format_byte_size(map_cache.usage),
                        format_byte_size(map_cache.capacity)
                    ),
                ))
                .child(
                    div()
                        .id("map-cache-capacity-bar")
                        .debug_selector(|| "#map-cache-capacity-bar".to_owned())
                        .h(px(12.0))
                        .w_full()
                        .my_1()
                        .relative()
                        .overflow_hidden()
                        .rounded_sm()
                        .bg(rgb(0x15171c))
                        .child(
                            div()
                                .absolute()
                                .top_0()
                                .bottom_0()
                                .left_0()
                                .w(relative(map_cache_fraction))
                                .debug_selector(|| "#map-cache-capacity-fill".to_owned())
                                .bg(if map_cache_fraction >= 0.9 {
                                    rgb(0xc06052)
                                } else {
                                    rgb(0x315b8f)
                                }),
                        ),
                )
                .child(row("Map cache hits", map_cache.network.hits.to_string()))
                .child(row(
                    "Map cache misses",
                    map_cache.network.misses.to_string(),
                ))
                .child(row(
                    "Map downloads",
                    map_cache.network.downloads.to_string(),
                ))
                .child(row(
                    "Map revalidations",
                    map_cache.network.not_modified.to_string(),
                ))
                .child(row("Map errors", map_cache.network.errors.to_string()))
                .child(
                    div()
                        .id("clear-downloaded-map-tiles")
                        .debug_selector(|| "#clear-downloaded-map-tiles".to_owned())
                        .mt_2()
                        .px_3()
                        .py_2()
                        .rounded_sm()
                        .bg(rgb(0x593b3b))
                        .hover(|this| this.bg(rgb(0x754949)))
                        .child("Clear Downloaded Map Tiles")
                        .on_click(cx.listener(|this, _, _, cx| this.clear_map_cache(cx))),
                )
                .child(div().my_2().border_t_1().border_color(rgb(0x30343d)))
                .child(row(
                    "Persistent cache hits",
                    pipeline.cache_hits.to_string(),
                ))
                .child(row(
                    "Persistent cache misses",
                    pipeline.cache_misses.to_string(),
                ))
                .child(row("Hit rate", hit_rate))
                .child(row("Writes", pipeline.cache_writes.to_string()))
                .child(row("Bypasses", pipeline.cache_bypasses.to_string()))
                .child(row(
                    "Read failures",
                    pipeline.cache_read_failures.to_string(),
                ))
                .child(row(
                    "Write failures",
                    pipeline.failed_cache_writes.to_string(),
                ))
                .child(row(
                    "Average lookup time",
                    format!("{} µs", pipeline.cache_lookup.average_nanos() / 1_000),
                ))
                .child(
                    div().mt_4().flex().justify_end().child(
                        div()
                            .id("cache-statistics-close")
                            .debug_selector(|| "#cache-statistics-close".to_owned())
                            .px_4()
                            .py_2()
                            .rounded_sm()
                            .bg(rgb(0x315b8f))
                            .hover(|this| this.bg(rgb(0x3d6fa9)))
                            .child("Close")
                            .on_click(cx.listener(|this, _, _, cx| {
                                this.show_cache_statistics = false;
                                cx.notify();
                            })),
                    ),
                ),
        )
}

fn cache_capacity_fraction(usage: u64, capacity: u64) -> f32 {
    if capacity == 0 {
        return 0.0;
    }
    (usage as f64 / capacity as f64).clamp(0.0, 1.0) as f32
}

fn render_disk_cache_usage(
    usage: Option<Result<rendering::pipeline::DiskCacheUsage, Arc<str>>>,
) -> gpui::Div {
    let Some(usage) = usage else {
        return div()
            .py_2()
            .text_color(rgb(0xb7bfcc))
            .child("Calculating disk cache usage…");
    };
    let Ok(usage) = usage else {
        return div()
            .py_2()
            .text_color(rgb(0xf2b8b5))
            .child(format!("Could not read disk cache: {}", usage.unwrap_err()));
    };
    let fraction = if usage.max_bytes == 0 {
        0.0
    } else {
        (usage.used_bytes as f32 / usage.max_bytes as f32).clamp(0.0, 1.0)
    };
    div()
        .py_1()
        .child(div().flex().justify_between().child("Disk cache").child(
            div().text_color(rgb(0xb7bfcc)).child(format!(
                "{} of {} ({:.1}%)",
                format_byte_size(usage.used_bytes),
                format_byte_size(usage.max_bytes),
                fraction * 100.0
            )),
        ))
        .child(
            div()
                .mt_2()
                .h(px(14.0))
                .w_full()
                .rounded_sm()
                .bg(rgb(0x181a20))
                .border_1()
                .border_color(rgb(0x4a5260))
                .child(
                    div()
                        .h_full()
                        .w(relative(fraction))
                        .rounded_sm()
                        .bg(rgb(0x3d6fa9)),
                ),
        )
        .child(
            div()
                .mt_1()
                .text_sm()
                .text_color(rgb(0x9da7b6))
                .child(format!("{} cached derivatives", usage.entries)),
        )
}

fn format_byte_size(bytes: u64) -> String {
    const GIB: u64 = 1024 * 1024 * 1024;
    const MIB: u64 = 1024 * 1024;
    if bytes >= GIB {
        format!("{:.2} GiB", bytes as f64 / GIB as f64)
    } else {
        format!("{:.1} MiB", bytes as f64 / MIB as f64)
    }
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
        bind_picture_browser_keys(cx);
        cx.set_menus(vec![
            Menu {
                name: "File".into(),
                items: vec![
                    MenuItem::action("Import Lightroom Catalog…", ImportLightroom),
                    MenuItem::action("Cache Statistics…", OpenCacheStatistics),
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

fn bind_picture_browser_keys(cx: &mut App) {
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
}

#[cfg(test)]
test_r::enable!();

#[cfg(test)]
#[path = "main/tests.rs"]
mod tests;
