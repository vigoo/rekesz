//! The import view: a Lightroom-style three-pane module for bringing photos
//! from external sources (volumes, folders, devices) into the library.

use std::collections::{BTreeSet, HashSet};
use std::ops::Range;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use super::*;
use crate::sources::SourceBacking;

const IMPORT_SOURCE_POLL: Duration = Duration::from_secs(3);
const IMPORT_PROGRESS_POLL: Duration = Duration::from_millis(250);
const MAX_RECENT_SOURCES: usize = 5;
/// Device files staged for preview per listing; the rest render as
/// placeholders until the user navigates or imports.
const MAX_STAGED_PREVIEWS: usize = 200;

/// One location in the source navigation stack; `stack[0]` is always the
/// source root.
#[derive(Clone)]
pub(crate) struct ImportLocation {
    /// Slash-separated path from the source root ("" at the root). A logical
    /// identity, never joined as a filesystem path.
    relative: String,
    backing: SourceBacking,
}

pub(crate) struct ImportViewState {
    pub(crate) sources: Vec<sources::SourceEntry>,
    pub(crate) selected: Option<sources::SourceId>,
    pub(crate) stack: Vec<ImportLocation>,
    pub(crate) listing: ImportListing,
    /// Cancellation for the running listing scan; replaced on every load.
    pub(crate) scan_cancel: Arc<AtomicBool>,
    /// Photos of the current view (current folder or recursive subtree),
    /// already paired and ordered.
    pub(crate) photos: Vec<import::ScannedPhoto>,
    /// Directories of the current folder (folder mode), for navigation.
    pub(crate) subdirectories: Vec<(String, SourceBacking)>,
    /// Live progress of the running source scan, shared with the job.
    pub(crate) scan_progress: Arc<Mutex<import::ScanProgress>>,
    /// Downloaded device files (MTP objects, PTP camera files), keyed by
    /// their backing reference; the source of thumbnails and capture dates
    /// for device-backed photos.
    pub(crate) staged: HashMap<SourceBacking, PathBuf>,
    /// Throttled cache for the footer's whole-directory thumbnail count.
    pub(crate) thumbnail_progress: std::cell::Cell<Option<(std::time::Instant, usize, usize)>>,
    /// Generation of the running whole-directory thumbnail pass; a listing
    /// change supersedes it.
    pub(crate) directory_pass_generation: u64,
    /// Next photo index the whole-directory pass will request.
    pub(crate) directory_pass_cursor: usize,
    /// Photos of the current pass that have ever reached a terminal
    /// thumbnail state; a set (not a counter) so repeated progress scans
    /// cannot compound, and monotonic so LRU eviction cannot move the bar
    /// back.
    pub(crate) directory_pass_done: std::cell::RefCell<std::collections::HashSet<SourceBacking>>,
    pub(crate) staging_generation: u64,
    pub(crate) staging_cancel: Arc<AtomicBool>,
    /// Non-fatal scan findings (unreadable folders, truncation).
    pub(crate) scan_notice: Option<String>,
    /// Keys of photos whose primary file already exists in the library.
    pub(crate) duplicates: HashSet<String>,
    pub(crate) selection: HashSet<String>,
    pub(crate) generation: u64,
    /// Epoch for listing loads; distinct from `generation` so a source
    /// selection's listing and its library index never invalidate each other.
    pub(crate) index_generation: u64,
    pub(crate) index_cancel: Arc<AtomicBool>,
    pub(crate) library_index: Option<Arc<import::duplicates::LibraryIndex>>,
    pub(crate) index_files: u64,
    pub(crate) index_building: bool,
    pub(crate) recursive: bool,
    pub(crate) preset: import::DestinationPreset,
    /// Absolute folder the layout applies within; `None` = library root.
    pub(crate) destination_base: Option<PathBuf>,
    pub(crate) destination_picker_open: bool,
    pub(crate) picker_dir: PathBuf,
    pub(crate) picker_entries: Option<Vec<String>>,
    pub(crate) picker_generation: u64,
    pub(crate) creating_folder: bool,
    pub(crate) new_folder_name: String,
    pub(crate) new_folder_focus: FocusHandle,
    pub(crate) layout_menu_open: bool,
    pub(crate) include_duplicates: bool,
    pub(crate) import_scroll: UniformListScrollHandle,
    pub(crate) job: Option<ImportJobHandle>,
    pub(crate) job_progress: import::ImportProgress,
    pub(crate) report: Option<ImportReportDialog>,
    pub(crate) notice: Option<String>,
    pub(crate) poll_scheduled: bool,
}

pub(crate) struct ImportJobHandle {
    progress: Arc<Mutex<import::ImportProgress>>,
    cancel: Arc<AtomicBool>,
    work_id: u64,
}

#[derive(Clone)]
pub(crate) struct ImportReportDialog {
    status: &'static str,
    report: import::ImportReport,
}

pub(crate) enum ImportListing {
    Loading,
    Loaded,
    Error(Arc<str>),
}

struct LoadedListing {
    photos: Vec<import::ScannedPhoto>,
    subdirectories: Vec<(String, SourceBacking)>,
    findings: Vec<String>,
}

impl ImportViewState {
    pub(crate) fn new(settings: &config::AppSettings, cx: &mut Context<PictureBrowser>) -> Self {
        Self {
            sources: Vec::new(),
            selected: None,
            stack: Vec::new(),
            listing: ImportListing::Loading,
            scan_cancel: Arc::new(AtomicBool::new(false)),
            photos: Vec::new(),
            subdirectories: Vec::new(),
            scan_progress: Arc::new(Mutex::new(import::ScanProgress::default())),
            scan_notice: None,
            staged: HashMap::new(),
            thumbnail_progress: std::cell::Cell::new(None),
            directory_pass_generation: 0,
            directory_pass_cursor: 0,
            directory_pass_done: std::cell::RefCell::new(std::collections::HashSet::new()),
            staging_generation: 0,
            staging_cancel: Arc::new(AtomicBool::new(false)),
            duplicates: HashSet::new(),
            selection: HashSet::new(),
            generation: 0,
            index_generation: 0,
            index_cancel: Arc::new(AtomicBool::new(false)),
            library_index: None,
            index_files: 0,
            index_building: false,
            recursive: false,
            preset: settings.import_destination_preset,
            destination_base: None,
            destination_picker_open: false,
            picker_dir: PathBuf::new(),
            picker_entries: None,
            picker_generation: 0,
            creating_folder: false,
            new_folder_name: String::new(),
            new_folder_focus: cx.focus_handle(),
            layout_menu_open: false,
            include_duplicates: settings.import_include_duplicates,
            import_scroll: UniformListScrollHandle::new(),
            job: None,
            job_progress: import::ImportProgress::default(),
            report: None,
            notice: None,
            poll_scheduled: false,
        }
    }

    fn current(&self) -> Option<&ImportLocation> {
        self.stack.last()
    }

    /// Slash-separated path of the current location from the source root.
    #[cfg(test)]
    pub(crate) fn current_relative(&self) -> Option<String> {
        self.stack.last().map(|location| location.relative.clone())
    }

    /// Whether the current source listing finished without an error.
    #[cfg(test)]
    pub(crate) fn listing_loaded(&self) -> bool {
        matches!(self.listing, ImportListing::Loaded)
    }

    fn selected_source(&self) -> Option<&sources::SourceEntry> {
        let id = self.selected.as_ref()?;
        self.sources.iter().find(|entry| &entry.id == id)
    }

    fn selected_count(&self) -> usize {
        self.photos
            .iter()
            .filter(|photo| self.selection.contains(&photo.key()))
            .count()
    }
}

impl PictureBrowser {
    pub(crate) fn import_photos(
        &mut self,
        _: &ImportPhotos,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.show_file_menu = false;
        self.open_import_view(window, cx);
    }

    fn open_import_view(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if matches!(self.main_view, MainView::Image(_)) {
            self.close_image(&CloseImage, window, cx);
        }
        if self.directory_view_mode != DirectoryViewMode::Import {
            self.set_directory_view_mode(DirectoryViewMode::Import, cx);
        }
    }

    /// Re-enumerate sources off the UI thread and reschedule the poll.
    pub(crate) fn refresh_import_sources(&mut self, cx: &mut Context<Self>) {
        let recents = self.settings.import_recent_sources.clone();
        let task =
            cx.background_spawn(async move { sources::SourceRegistry::new().refresh(&recents) });
        cx.spawn(async move |this, cx| {
            let refreshed = task.await;
            this.update(cx, |this, cx| {
                let previous = this.import_view.selected_source().cloned();
                let selected_gone = this
                    .import_view
                    .selected
                    .as_ref()
                    .is_some_and(|id| !refreshed.iter().any(|entry| &entry.id == id));
                let mut sources = refreshed;
                if selected_gone {
                    // Removable-volume subfolders are merged into their
                    // volume's entry by the source refresh; keep the selection
                    // alive as long as the folder still exists on disk.
                    let still_there = matches!(&previous, Some(entry) if match &entry.backing {
                        SourceBacking::Path(path) => path.exists(),
                        SourceBacking::Mtp(_) | SourceBacking::Ptp(_) => false,
                    });
                    if let (true, Some(entry)) = (still_there, previous.clone()) {
                        sources.push(entry);
                    } else {
                        this.import_view.selected = None;
                        this.import_view.stack.clear();
                        this.import_view.photos.clear();
                        this.import_view.selection.clear();
                        this.import_view.listing = ImportListing::Loading;
                    }
                }
                this.import_view.sources = sources;
                cx.notify();
            })
            .ok();
        })
        .detach();
        self.schedule_import_source_poll(cx);
    }

    fn schedule_import_source_poll(&mut self, cx: &mut Context<Self>) {
        if self.import_view.poll_scheduled || self.import_view.job.is_some() {
            return;
        }
        self.import_view.poll_scheduled = true;
        cx.spawn(async move |this, cx| {
            cx.background_executor().timer(IMPORT_SOURCE_POLL).await;
            this.update(cx, |this, cx| {
                this.import_view.poll_scheduled = false;
                if this.directory_view_mode == DirectoryViewMode::Import
                    && this.import_view.job.is_none()
                {
                    this.refresh_import_sources(cx);
                }
            })
            .ok();
        })
        .detach();
    }

    pub(crate) fn select_import_source(&mut self, id: sources::SourceId, cx: &mut Context<Self>) {
        let Some(entry) = self
            .import_view
            .sources
            .iter()
            .find(|entry| entry.id == id)
            .cloned()
        else {
            return;
        };
        self.import_view.notice = None;
        self.import_view
            .staging_cancel
            .store(true, Ordering::Release);
        self.import_view.scan_cancel.store(true, Ordering::Release);
        self.import_view.selected = Some(id);
        self.import_view.stack = vec![ImportLocation {
            relative: String::new(),
            backing: entry.backing.clone(),
        }];
        self.import_view
            .import_scroll
            .scroll_to_item_strict(0, ScrollStrategy::Top);
        self.import_view.selection.clear();
        self.import_view.photos.clear();
        if matches!(entry.kind, sources::SourceKind::Folder) {
            self.remember_import_recent(entry.backing.root_path(), cx);
        }
        self.rebuild_library_index(cx);
        self.reload_import_listing(cx);
    }

    fn remember_import_recent(&mut self, path: Option<PathBuf>, cx: &mut Context<Self>) {
        let Some(path) = path else { return };
        let recents = &mut self.settings.import_recent_sources;
        recents.retain(|recent| recent != &path);
        recents.insert(0, path);
        recents.truncate(MAX_RECENT_SOURCES);
        if let Err(error) = config::save(&self.settings_path, &self.settings) {
            eprintln!("cannot save import recents: {error:#}");
        }
        cx.notify();
    }

    pub(crate) fn import_browse_folder(&mut self, cx: &mut Context<Self>) {
        let selection = cx.prompt_for_paths(PathPromptOptions {
            files: false,
            directories: true,
            multiple: false,
            prompt: Some("Choose a folder to import from".into()),
        });
        cx.spawn(async move |this, cx| {
            let path = match selection.await {
                Ok(Ok(Some(paths))) => paths.into_iter().next(),
                _ => None,
            };
            let Some(path) = path else { return };
            this.update(cx, |this, cx| {
                if this.directory_view_mode != DirectoryViewMode::Import {
                    return;
                }
                // Canonicalize before building the id: the source refresh
                // re-derives ids from canonicalized recents, and a mismatch
                // would drop the selection on the next poll.
                let path = sources::canonical_folder(&path);
                let id = sources::SourceId::new(format!("folder:{}", path.display()));
                this.import_view.sources.retain(|entry| entry.id != id);
                let label = path
                    .file_name()
                    .unwrap_or_else(|| path.as_os_str())
                    .to_string_lossy()
                    .into_owned();
                this.import_view.sources.push(sources::SourceEntry {
                    id: id.clone(),
                    label,
                    kind: sources::SourceKind::Folder,
                    backing: SourceBacking::Path(path),
                    capacity: None,
                });
                this.select_import_source(id, cx);
            })
            .ok();
        })
        .detach();
    }

    /// Rebuild the duplicate index over the library root in the background.
    /// Runs on its own epoch so concurrent listing loads cannot discard it.
    fn rebuild_library_index(&mut self, cx: &mut Context<Self>) {
        self.import_view.index_cancel.store(true, Ordering::Release);
        let cancel = Arc::new(AtomicBool::new(false));
        self.import_view.index_cancel = Arc::clone(&cancel);
        self.import_view.library_index = None;
        self.import_view.index_files = 0;
        self.import_view.index_building = true;
        self.import_view.index_generation = self.import_view.index_generation.wrapping_add(1);
        let generation = self.import_view.index_generation;
        let root = self.settings.root.clone();
        let task = cx.background_spawn(async move {
            import::duplicates::LibraryIndex::build(&root, &cancel, &mut |_| {})
                .map(Arc::new)
                .map_err(|_| ())
        });
        cx.spawn(async move |this, cx| {
            let result = task.await;
            this.update(cx, |this, cx| {
                if this.import_view.index_generation != generation {
                    return;
                }
                this.import_view.index_building = false;
                if let Ok(index) = result {
                    this.import_view.index_files = index.files;
                    this.import_view.library_index = Some(index);
                }
                this.mark_import_duplicates();
                cx.notify();
            })
            .ok();
        })
        .detach();
    }

    /// (Re)load the photos of the current location: the folder listing, or a
    /// recursive subtree scan when that mode is on.
    pub(crate) fn reload_import_listing(&mut self, cx: &mut Context<Self>) {
        self.import_view.scan_cancel.store(true, Ordering::Release);
        let scan_cancel = Arc::new(AtomicBool::new(false));
        self.import_view.scan_cancel = Arc::clone(&scan_cancel);
        self.import_view.listing = ImportListing::Loading;
        self.import_view.photos.clear();
        self.import_view.subdirectories.clear();
        self.import_view.scan_notice = None;
        self.import_view.selection.clear();
        self.import_view.generation = self.import_view.generation.wrapping_add(1);
        let generation = self.import_view.generation;
        let Some(location) = self.import_view.current().cloned() else {
            return;
        };
        let recursive = self.import_view.recursive;
        let preference = self.pair_preference;
        let Some(access) = self.import_sources_registry().access_for(&location.backing) else {
            self.import_view.listing = ImportListing::Error("This source cannot be read.".into());
            cx.notify();
            return;
        };
        let cancel = scan_cancel;
        let relative = location.relative.clone();
        let backing = location.backing.clone();
        let scan_progress = Arc::new(Mutex::new(import::ScanProgress::default()));
        self.import_view.scan_progress = Arc::clone(&scan_progress);
        let progress_target = Arc::clone(&scan_progress);
        let scan = cx.background_spawn(async move {
            if recursive {
                import::scan_source(
                    access.as_ref(),
                    &backing,
                    preference,
                    &cancel,
                    &mut |progress| {
                        *progress_target.lock().expect("import scan progress lock") = progress;
                    },
                )
                .map(|outcome| {
                    let mut findings = Vec::new();
                    if outcome.skipped_directories > 0 {
                        findings.push(format!(
                            "{} folder{} could not be read",
                            outcome.skipped_directories,
                            if outcome.skipped_directories == 1 {
                                ""
                            } else {
                                "s"
                            },
                        ));
                    }
                    if outcome.truncated {
                        findings.push("the listing was truncated".to_owned());
                    }
                    LoadedListing {
                        photos: outcome.photos,
                        subdirectories: Vec::new(),
                        findings,
                    }
                })
            } else {
                let entries = access.list_dir(&backing)?;
                let mut subdirectories = Vec::new();
                let mut files = Vec::new();
                for entry in entries {
                    if entry.is_dir {
                        subdirectories.push((entry.name, entry.backing));
                    } else if crate::discovery::is_supported_image(std::path::Path::new(
                        &entry.name,
                    )) {
                        files.push(import::ScannedFile {
                            backing: entry.backing,
                            name: entry.name,
                            size: entry.size,
                            modified: entry.modified,
                        });
                    }
                }
                let photos = import::pair_photos(&relative, files, preference);
                Ok(LoadedListing {
                    photos,
                    subdirectories,
                    findings: Vec::new(),
                })
            }
        });
        cx.spawn(async move |this, cx| {
            let result = scan.await;
            this.update(cx, |this, cx| {
                if this.import_view.generation != generation {
                    return;
                }
                match result {
                    Ok(loaded) => {
                        this.import_view.photos = loaded.photos;
                        this.import_view.subdirectories = loaded.subdirectories;
                        if loaded.findings.is_empty() {
                            this.import_view.scan_notice = None;
                        } else {
                            this.import_view.scan_notice = Some(loaded.findings.join(" · "));
                        }
                        this.import_view.listing = ImportListing::Loaded;
                    }
                    Err(error) => {
                        let message = match error {
                            import::ScanError::Cancelled => "cancelled".to_owned(),
                            import::ScanError::Failed(message) => message.to_string(),
                        };
                        this.import_view.listing = ImportListing::Error(message.into());
                    }
                }
                this.mark_import_duplicates();
                this.import_select_new_only(cx);
                this.schedule_import_staging(cx);
                this.import_view.directory_pass_generation =
                    this.import_view.directory_pass_generation.wrapping_add(1);
                this.import_view.directory_pass_cursor = 0;
                this.import_view.directory_pass_done.borrow_mut().clear();
                this.schedule_directory_thumbnail_pass(cx);
                cx.notify();
            })
            .ok();
        })
        .detach();
        cx.notify();
    }

    fn import_sources_registry(&self) -> sources::SourceRegistry {
        sources::SourceRegistry::new()
    }

    fn mark_import_duplicates(&mut self) {
        self.import_view.duplicates.clear();
        let Some(index) = &self.import_view.library_index else {
            return;
        };
        for photo in &self.import_view.photos {
            if index.contains(&photo.primary.name, photo.primary.size) {
                self.import_view.duplicates.insert(photo.key());
            }
        }
    }

    /// Preselect every non-duplicate photo of the current view.
    pub(crate) fn import_select_new_only(&mut self, cx: &mut Context<Self>) {
        let duplicates = &self.import_view.duplicates;
        self.import_view.selection = self
            .import_view
            .photos
            .iter()
            .filter(|photo| !duplicates.contains(&photo.key()))
            .map(|photo| photo.key())
            .collect();
        cx.notify();
    }

    /// Download device-backed photos of the current view into the cache so
    /// the render pipeline can produce thumbnails and capture dates.
    fn schedule_import_staging(&mut self, cx: &mut Context<Self>) {
        self.import_view
            .staging_cancel
            .store(true, Ordering::Release);
        let cancel = Arc::new(AtomicBool::new(false));
        self.import_view.staging_cancel = Arc::clone(&cancel);
        self.import_view.staging_generation = self.import_view.staging_generation.wrapping_add(1);
        let generation = self.import_view.staging_generation;
        let Some(location) = self.import_view.current().cloned() else {
            return;
        };
        let Some(access) = self.import_sources_registry().access_for(&location.backing) else {
            return;
        };
        let mut pending = Vec::new();
        for photo in &self.import_view.photos {
            if pending.len() >= MAX_STAGED_PREVIEWS {
                break;
            }
            let mut files = vec![&photo.primary];
            if let Some(companion) = &photo.companion {
                files.push(companion);
            }
            for file in files {
                if !matches!(file.backing, SourceBacking::Path(_))
                    && !self.import_view.staged.contains_key(&file.backing)
                {
                    pending.push(file.clone());
                }
            }
        }
        if pending.is_empty() {
            return;
        }
        let cache_root = self.settings.cache_root.clone();
        cx.spawn(async move |this, cx| {
            for file in pending {
                if cancel.load(Ordering::Acquire) {
                    return;
                }
                let backing = match &file.backing {
                    SourceBacking::Path(_) => continue,
                    backing => backing.clone(),
                };
                let task = cx.background_spawn({
                    let access = Arc::clone(&access);
                    let cache_root = cache_root.clone();
                    let cancel = Arc::clone(&cancel);
                    let file = file.clone();
                    async move {
                        import::staging::stage_file(access.as_ref(), &file, &cache_root, &cancel)
                    }
                });
                let staged = task.await;
                this.update(cx, |this, cx| {
                    if this.import_view.staging_generation != generation {
                        return;
                    }
                    if let Ok(Some(path)) = staged {
                        this.import_view.staged.insert(backing, path);
                    }
                    cx.notify();
                })
                .ok();
            }
        })
        .detach();
    }

    pub(crate) fn import_toggle_photo(&mut self, key: &str, cx: &mut Context<Self>) {
        if !self.import_view.selection.remove(key) {
            self.import_view.selection.insert(key.to_owned());
        }
        cx.notify();
    }

    pub(crate) fn import_select_all(&mut self, cx: &mut Context<Self>) {
        self.import_view.selection = self
            .import_view
            .photos
            .iter()
            .map(|photo| photo.key())
            .collect();
        cx.notify();
    }

    pub(crate) fn import_select_none(&mut self, cx: &mut Context<Self>) {
        self.import_view.selection.clear();
        cx.notify();
    }

    fn import_navigate_into(
        &mut self,
        name: String,
        backing: SourceBacking,
        cx: &mut Context<Self>,
    ) {
        let relative = self
            .import_view
            .current()
            .map(|location| location.relative.clone())
            .unwrap_or_default();
        let child_relative = if relative.is_empty() {
            name
        } else {
            format!("{relative}/{name}")
        };
        self.import_view.stack.push(ImportLocation {
            relative: child_relative,
            backing,
        });
        self.import_view
            .import_scroll
            .scroll_to_item_strict(0, ScrollStrategy::Top);
        self.reload_import_listing(cx);
    }

    pub(crate) fn import_navigate_up(&mut self, cx: &mut Context<Self>) {
        if self.import_view.stack.len() > 1 {
            self.import_view.stack.pop();
            self.reload_import_listing(cx);
        }
    }

    pub(crate) fn import_navigate_to(&mut self, depth: usize, cx: &mut Context<Self>) {
        if depth < self.import_view.stack.len() {
            self.import_view.stack.truncate(depth + 1);
            self.reload_import_listing(cx);
        }
    }

    pub(crate) fn import_toggle_recursive(&mut self, cx: &mut Context<Self>) {
        self.import_view.recursive = !self.import_view.recursive;
        self.reload_import_listing(cx);
    }

    pub(crate) fn import_toggle_include_duplicates(&mut self, cx: &mut Context<Self>) {
        self.import_view.include_duplicates = !self.import_view.include_duplicates;
        self.settings.import_include_duplicates = self.import_view.include_duplicates;
        if let Err(error) = config::save(&self.settings_path, &self.settings) {
            eprintln!("cannot save import duplicate policy: {error:#}");
        }
        cx.notify();
    }

    fn destination_root(&self) -> PathBuf {
        self.import_view
            .destination_base
            .clone()
            .unwrap_or_else(|| self.settings.root.clone())
    }

    fn toggle_destination_picker(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.import_view.destination_picker_open = !self.import_view.destination_picker_open;
        if self.import_view.destination_picker_open {
            self.import_view.creating_folder = false;
            self.import_view.new_folder_name.clear();
            self.import_view.picker_dir = self.destination_root();
            self.load_picker_entries(cx);
        } else {
            // The input (if rendered) is going away; drop focus so MainView
            // shortcuts keep working.
            window.blur();
        }
        cx.notify();
    }

    fn navigate_picker(&mut self, dir: PathBuf, window: &mut Window, cx: &mut Context<Self>) {
        self.import_view.picker_dir = dir;
        self.import_view.creating_folder = false;
        window.blur();
        self.load_picker_entries(cx);
    }

    fn load_picker_entries(&mut self, cx: &mut Context<Self>) {
        self.import_view.picker_entries = None;
        self.import_view.picker_generation = self.import_view.picker_generation.wrapping_add(1);
        let generation = self.import_view.picker_generation;
        let dir = self.import_view.picker_dir.clone();
        let task = cx.background_spawn(async move {
            let mut entries = Vec::new();
            if let Ok(reader) = std::fs::read_dir(&dir) {
                for entry in reader.filter_map(|entry| entry.ok()) {
                    let Ok(file_type) = entry.file_type() else {
                        continue;
                    };
                    if !file_type.is_dir() {
                        continue;
                    }
                    let name = entry.file_name().to_string_lossy().into_owned();
                    if name.starts_with('.') || name == crate::catalog::CATALOG_DIR {
                        continue;
                    }
                    entries.push(name);
                }
            }
            entries.sort_by_key(|name| name.to_lowercase());
            entries
        });
        cx.spawn(async move |this, cx| {
            let entries = task.await;
            this.update(cx, |this, cx| {
                if this.import_view.picker_generation == generation {
                    this.import_view.picker_entries = Some(entries);
                }
                cx.notify();
            })
            .ok();
        })
        .detach();
    }

    fn create_destination_folder(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let name = self.import_view.new_folder_name.trim().to_owned();
        if name.is_empty() || name.contains(['/', '\\']) || name.chars().all(|c| c == '.') {
            return;
        }
        let target = self.import_view.picker_dir.join(&name);
        if let Err(error) = std::fs::create_dir_all(&target) {
            self.import_view.notice = Some(format!("Cannot create folder: {error}"));
            cx.notify();
            return;
        }
        self.import_view.creating_folder = false;
        self.import_view.new_folder_name.clear();
        window.blur();
        self.load_picker_entries(cx);
        cx.notify();
    }

    fn pick_destination(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.import_view.destination_base = Some(self.import_view.picker_dir.clone());
        self.import_view.destination_picker_open = false;
        window.blur();
        cx.notify();
    }

    fn import_new_folder_key_down(&mut self, event: &gpui::KeyDownEvent, cx: &mut Context<Self>) {
        // Enter is handled by the ImportNewFolderConfirm action binding so it
        // takes precedence over the MainView keymap.
        let mut handled = false;
        if event.keystroke.key == "backspace" {
            self.import_view.new_folder_name.pop();
            handled = true;
        } else if !event.keystroke.modifiers.control
            && !event.keystroke.modifiers.platform
            && let Some(text) = &event.keystroke.key_char
            && text
                .chars()
                .all(|c| !c.is_control() && c != '/' && c != '\\')
        {
            self.import_view.new_folder_name.push_str(text);
            handled = true;
        }
        if handled {
            // Consumed keys must not reach the root plain-key shortcut
            // listener (a typed "d" would otherwise toggle the view).
            cx.stop_propagation();
        }
        cx.notify();
    }

    pub(crate) fn start_import(&mut self, cx: &mut Context<Self>) {
        if self.import_view.job.is_some() || self.import_view.current().is_none() {
            return;
        }
        // Free the device for the import job: no staging downloads in flight.
        self.import_view
            .staging_cancel
            .store(true, Ordering::Release);
        self.import_view.scan_cancel.store(true, Ordering::Release);
        let Some(source) = self.import_view.selected_source().cloned() else {
            return;
        };
        let duplicates = self.import_view.duplicates.clone();
        let mut photos = self
            .import_view
            .photos
            .iter()
            .filter(|photo| self.import_view.selection.contains(&photo.key()))
            .filter(|photo| {
                self.import_view.include_duplicates || !duplicates.contains(&photo.key())
            })
            .cloned()
            .collect::<Vec<_>>();
        // Device-backed files with a staged copy are imported from disk: the
        // capture-date lookup then reads the staged file's EXIF, and the copy
        // itself never touches the device twice.
        for photo in &mut photos {
            let staged = &self.import_view.staged;
            if let Some(path) = staged.get(&photo.primary.backing) {
                photo.primary.backing = SourceBacking::Path(path.clone());
            }
            if let Some(companion) = &mut photo.companion
                && let Some(path) = staged.get(&companion.backing)
            {
                companion.backing = SourceBacking::Path(path.clone());
            }
        }
        if photos.is_empty() {
            return;
        }
        let Some(access) = self.import_sources_registry().access_for(&source.backing) else {
            self.import_view.notice = Some("This source cannot be read.".into());
            cx.notify();
            return;
        };
        // Preview staging swaps device backings for cache paths, so the
        // import reads a mix of device objects and staged files.
        let access = sources::staged_import_access(access);
        let root = self.settings.root.clone();
        let destination_base = self
            .import_view
            .destination_base
            .clone()
            .filter(|base| base.starts_with(&root))
            .unwrap_or_else(|| root.clone());
        let plan = import::ImportPlan {
            library_root: root,
            destination_base,
            photos,
            preset: self.import_view.preset,
        };
        let progress = Arc::new(Mutex::new(import::ImportProgress::default()));
        let cancel = Arc::new(AtomicBool::new(false));
        let work_id = self.work.start("Importing photos…");
        self.import_view.job_progress = import::ImportProgress::default();
        self.import_view.job = Some(ImportJobHandle {
            progress: Arc::clone(&progress),
            cancel: Arc::clone(&cancel),
            work_id,
        });
        self.poll_import_progress(cx);
        let run = cx.background_spawn(async move {
            import::run_import(&plan, access.as_ref(), progress, &cancel)
        });
        cx.spawn(async move |this, cx| {
            let status = run.await;
            this.update(cx, |this, cx| {
                let Some(job) = this.import_view.job.take() else {
                    return;
                };
                this.work.finish(job.work_id);
                let (status_label, report) = match status {
                    import::ImportStatus::Completed(report) => ("Import finished.", report),
                    import::ImportStatus::Cancelled(report) => ("Import cancelled.", report),
                };
                this.refresh_imported_directories(&report.directories, cx);
                // Leaving the view cancels the import and must not resurrect
                // a stale report dialog on the next Import entry.
                if this.directory_view_mode == DirectoryViewMode::Import {
                    this.import_view.report = Some(ImportReportDialog {
                        status: status_label,
                        report,
                    });
                }
                this.import_view.job_progress = import::ImportProgress::default();
                this.refresh_import_sources(cx);
                cx.notify();
            })
            .ok();
        })
        .detach();
        cx.notify();
    }

    /// Make the freshly written files show up in the browser: ensure every
    /// directory on each destination chain is loaded by the discovery model.
    fn refresh_imported_directories(&mut self, directories: &[PathBuf], cx: &mut Context<Self>) {
        let root = self.settings.root.clone();
        let mut chains = BTreeSet::new();
        for directory in directories {
            let Ok(relative) = directory.strip_prefix(&root) else {
                continue;
            };
            let mut chain = root.clone();
            chains.insert(chain.clone());
            for component in relative.components() {
                chain.push(component);
                chains.insert(chain.clone());
            }
        }
        for path in chains {
            let node = self.model.ensure_node(path);
            self.ensure_directory_loaded(node, discovery::DirectoryScanPriority::Background, cx);
        }
    }

    pub(crate) fn cancel_import(&mut self, cx: &mut Context<Self>) {
        if let Some(job) = &self.import_view.job {
            job.cancel.store(true, Ordering::Release);
        }
        cx.notify();
    }

    /// Leave import mode: a running import is cancelled (the engine removes
    /// partial files; completed files stay), and the library view — with the
    /// directory tree — comes back.
    pub(crate) fn exit_import_view(&mut self, cx: &mut Context<Self>) {
        self.cancel_import(cx);
        // Leaving the view also stops source scans, staging downloads, and
        // the library-index walk — no device reads for a view nobody is
        // looking at.
        self.import_view.scan_cancel.store(true, Ordering::Release);
        self.import_view
            .staging_cancel
            .store(true, Ordering::Release);
        self.import_view.index_cancel.store(true, Ordering::Release);
        if self.directory_view_mode == DirectoryViewMode::Import {
            self.set_directory_view_mode(DirectoryViewMode::Grid, cx);
        }
    }

    pub(crate) fn dismiss_import_report(&mut self, cx: &mut Context<Self>) {
        self.import_view.report = None;
        cx.notify();
    }

    fn poll_import_progress(&mut self, cx: &mut Context<Self>) {
        let Some(job) = self.import_view.job.as_ref() else {
            return;
        };
        let progress = Arc::clone(&job.progress);
        let work_id = job.work_id;
        cx.spawn(async move |this, cx| {
            cx.background_executor().timer(IMPORT_PROGRESS_POLL).await;
            this.update(cx, |this, cx| {
                // Only re-arm while the same job is still running, so a stale
                // chain from a finished import cannot linger.
                let same_job = this
                    .import_view
                    .job
                    .as_ref()
                    .is_some_and(|job| job.work_id == work_id);
                if let (true, Ok(snapshot)) = (same_job, progress.lock()) {
                    this.import_view.job_progress = snapshot.clone();
                    this.poll_import_progress(cx);
                }
                cx.notify();
            })
            .ok();
        })
        .detach();
    }
}

impl PictureBrowser {
    pub(crate) fn render_import_view(&self, cx: &mut Context<Self>) -> AnyElement {
        let sources_rail = self.render_import_sources_rail(cx);
        let content = self.render_import_content(cx);
        let options_rail = self.render_import_options_rail(cx);
        // Panes row, then an in-flow progress footer so it never covers the
        // options rail; the report dialog stays an overlay.
        let panes = div()
            .flex_1()
            .min_h_0()
            .flex()
            .child(sources_rail)
            .child(content)
            .child(options_rail);
        let mut view = div()
            .id("import-view")
            .size_full()
            .relative()
            .flex()
            .flex_col()
            .bg(rgb(0x1b1f27))
            .text_color(rgb(0xd8dde7))
            .child(panes);
        if self.import_view.job.is_some() || !self.import_thumbnails_complete() {
            view = view.child(self.render_import_progress_footer(cx));
        }
        if let Some(dialog) = self.import_view.report.clone() {
            view = view.child(render_import_report_dialog(dialog, cx));
        }
        view.into_any_element()
    }

    fn render_import_sources_rail(&self, cx: &mut Context<Self>) -> AnyElement {
        let import_running = self.import_view.job.is_some();
        let mut rail = div()
            .w(px(270.0))
            .flex_none()
            .h_full()
            .flex()
            .flex_col()
            .border_r_1()
            .border_color(rgb(0x30343d))
            .bg(rgb(0x20242c))
            .child(
                div()
                    .flex_none()
                    .flex()
                    .px_3()
                    .py_2()
                    .border_b_1()
                    .border_color(rgb(0x30343d))
                    .child(
                        div()
                            .id("import-exit")
                            .px_2()
                            .py_1()
                            .rounded_sm()
                            .text_xs()
                            .bg(if import_running {
                                rgb(0x6b3535)
                            } else {
                                rgb(0x30343d)
                            })
                            .hover(|this| {
                                this.bg(if import_running {
                                    rgb(0x7d4040)
                                } else {
                                    rgb(0x445066)
                                })
                            })
                            .cursor_pointer()
                            .child(if import_running {
                                "← Cancel import"
                            } else {
                                "← Back to library"
                            })
                            .on_click(cx.listener(|this, _, _, cx| {
                                this.exit_import_view(cx);
                            })),
                    ),
            )
            .child(
                div()
                    .flex()
                    .items_center()
                    .justify_between()
                    .px_3()
                    .py_2()
                    .child(div().text_sm().child("Sources"))
                    .child(
                        div()
                            .id("import-refresh")
                            .px_2()
                            .py_1()
                            .rounded_sm()
                            .text_xs()
                            .bg(rgb(0x30343d))
                            .hover(|this| this.bg(rgb(0x3b4658)))
                            .cursor_pointer()
                            .child("Refresh")
                            .on_click(cx.listener(|this, _, _, cx| {
                                this.refresh_import_sources(cx);
                            })),
                    ),
            );
        for (index, entry) in self.import_view.sources.iter().enumerate() {
            let selected = self
                .import_view
                .selected
                .as_ref()
                .is_some_and(|id| *id == entry.id);
            let id = entry.id.clone();
            let kind_label: SharedString = match &entry.kind {
                sources::SourceKind::Volume { removable: true } => "Removable volume".into(),
                sources::SourceKind::Volume { .. } => "Volume".into(),
                sources::SourceKind::Device { .. } => "Device".into(),
                sources::SourceKind::Folder => "Folder".into(),
            };
            let capacity: SharedString = entry
                .capacity
                .map(|capacity| {
                    format!(
                        "{} free of {}",
                        format_byte_size(capacity.available),
                        format_byte_size(capacity.total)
                    )
                    .into()
                })
                .unwrap_or_else(|| kind_label.clone());
            rail = rail.child(
                div()
                    .id(("import-source", index))
                    .px_3()
                    .py_2()
                    .cursor_pointer()
                    .when(selected, |this| this.bg(rgb(0x334b6e)))
                    .hover(|this| this.bg(rgb(0x292d36)))
                    .child(div().text_sm().truncate().child(entry.label.clone()))
                    .child(
                        div()
                            .text_xs()
                            .text_color(rgb(0x9da7b6))
                            .truncate()
                            .child(capacity),
                    )
                    .on_click(cx.listener(move |this, _, _, cx| {
                        this.select_import_source(id.clone(), cx);
                    })),
            );
        }
        rail = rail.child(
            div()
                .id("import-browse")
                .px_3()
                .py_2()
                .text_sm()
                .text_color(rgb(0x9dc0f0))
                .cursor_pointer()
                .hover(|this| this.bg(rgb(0x292d36)))
                .child("Browse for Folder…")
                .on_click(cx.listener(|this, _, _, cx| {
                    this.import_browse_folder(cx);
                })),
        );
        if let Some(notice) = self.import_view.notice.clone() {
            rail = rail.child(
                div()
                    .px_3()
                    .py_2()
                    .text_xs()
                    .text_color(rgb(0xf2b8b5))
                    .child(notice),
            );
        }
        rail.into_any_element()
    }

    fn render_import_content(&self, cx: &mut Context<Self>) -> AnyElement {
        let has_selection = self.import_view.selected.is_some();
        let mut header = div()
            .flex_none()
            .flex()
            .items_center()
            .gap_2()
            .px_3()
            .py_2()
            .bg(rgb(0x242b37))
            .border_b_1()
            .border_color(rgb(0x3b4658))
            .text_sm()
            .child(self.render_import_breadcrumb(cx));
        if self.import_view.stack.len() > 1 {
            header = header.child(
                div()
                    .id("import-navigate-up")
                    .debug_selector(|| "#import-navigate-up".to_owned())
                    .px_2()
                    .py_1()
                    .rounded_sm()
                    .text_xs()
                    .bg(rgb(0x30343d))
                    .hover(|this| this.bg(rgb(0x3b4658)))
                    .cursor_pointer()
                    .child("Up")
                    .on_click(cx.listener(|this, _, _, cx| {
                        this.import_navigate_up(cx);
                    })),
            );
        }
        header = header
            .child(div().flex_1())
            .child(self.render_import_toggle(
                cx,
                "import-recursive",
                if self.import_view.recursive {
                    "✓ All subdirectories"
                } else {
                    "All subdirectories"
                },
                |this, cx| this.import_toggle_recursive(cx),
            ))
            .child(
                self.render_import_toggle(cx, "import-select-all", "Select all", |this, cx| {
                    this.import_select_all(cx)
                }),
            )
            .child(
                self.render_import_toggle(cx, "import-select-new", "Select new", |this, cx| {
                    this.import_select_new_only(cx)
                }),
            )
            .child(
                self.render_import_toggle(cx, "import-select-none", "None", |this, cx| {
                    this.import_select_none(cx)
                }),
            );
        let body = match (&self.import_view.listing, self.import_view.photos.len()) {
            _ if !has_selection => div()
                .p_6()
                .text_sm()
                .text_color(rgb(0x9da7b6))
                .child("Choose a source on the left, or browse for a folder.")
                .into_any_element(),
            (_, _) if self.import_view.job.is_some() => div()
                .p_6()
                .text_sm()
                .text_color(rgb(0x9da7b6))
                .child("Importing…")
                .into_any_element(),
            (_, 0) => match &self.import_view.listing {
                ImportListing::Loading => div()
                    .p_6()
                    .text_sm()
                    .child("Reading source…")
                    .into_any_element(),
                ImportListing::Error(error) => div()
                    .p_6()
                    .text_sm()
                    .text_color(rgb(0xf2b8b5))
                    .child(error.to_string())
                    .into_any_element(),
                ImportListing::Loaded => div()
                    .p_6()
                    .text_sm()
                    .text_color(rgb(0x9da7b6))
                    .child("No importable photos here")
                    .into_any_element(),
            },
            (ImportListing::Loading, _) => {
                let progress = self
                    .import_view
                    .scan_progress
                    .lock()
                    .map(|progress| progress.clone())
                    .unwrap_or_default();
                div()
                    .p_6()
                    .text_sm()
                    .text_color(rgb(0x9da7b6))
                    .child(if self.import_view.recursive {
                        format!(
                            "Scanning source… {} folders, {} photos so far{}",
                            progress.directories,
                            progress.photos,
                            if progress.current.is_empty() {
                                String::new()
                            } else {
                                format!(" · at {}", progress.current)
                            }
                        )
                    } else {
                        "Reading source…".to_owned()
                    })
                    .into_any_element()
            }
            (ImportListing::Error(error), _) => div()
                .p_6()
                .text_sm()
                .text_color(rgb(0xf2b8b5))
                .child(error.to_string())
                .into_any_element(),
            (ImportListing::Loaded, _) => {
                let columns = self.grid_columns.max(1);
                let rows = self.import_view.photos.len().div_ceil(columns);
                let grid =
                    uniform_list("import-rows", rows, cx.processor(Self::render_import_rows))
                        .flex_1()
                        .min_h_0()
                        .track_scroll(self.import_view.import_scroll.clone())
                        .with_sizing_behavior(ListSizingBehavior::Auto);
                div()
                    .size_full()
                    .flex()
                    .flex_col()
                    .when_some(self.render_import_subdirectories(cx), |this, strip| {
                        this.child(strip)
                    })
                    .when_some(self.import_view.scan_notice.clone(), |this, notice| {
                        this.child(
                            div()
                                .flex_none()
                                .px_3()
                                .py_1()
                                .text_xs()
                                .text_color(rgb(0xf5e0a9))
                                .bg(rgb(0x2b2a20))
                                .child(notice),
                        )
                    })
                    .child(grid)
                    .into_any_element()
            }
        };
        div()
            .flex_1()
            .min_w_0()
            .h_full()
            .flex()
            .flex_col()
            .child(header)
            .child(div().flex_1().min_h_0().overflow_hidden().child(body))
            .into_any_element()
    }

    fn render_import_breadcrumb(&self, cx: &mut Context<Self>) -> AnyElement {
        let Some(source) = self.import_view.selected_source().cloned() else {
            return div().child("Import").into_any_element();
        };
        let mut breadcrumb = div().flex().items_center().gap_1().min_w_0();
        for (depth, location) in self.import_view.stack.iter().enumerate() {
            let label: SharedString = if depth == 0 {
                source.label.clone().into()
            } else {
                location
                    .relative
                    .rsplit('/')
                    .next()
                    .unwrap_or_default()
                    .to_owned()
                    .into()
            };
            let is_current = depth + 1 == self.import_view.stack.len();
            breadcrumb = breadcrumb
                .when(depth > 0, |this| {
                    this.child(
                        div()
                            .text_color(rgb(0x9da7b6))
                            .text_xs()
                            .child("/".to_owned()),
                    )
                })
                .child(if is_current {
                    div()
                        .truncate()
                        .max_w(px(280.0))
                        .child(label)
                        .into_any_element()
                } else {
                    div()
                        .id(("import-crumb", depth))
                        .debug_selector(move || format!("#import-crumb-{depth}"))
                        .cursor_pointer()
                        .hover(|this| this.underline())
                        .truncate()
                        .max_w(px(200.0))
                        .child(label)
                        .on_click(cx.listener(move |this, _, _, cx| {
                            this.import_navigate_to(depth, cx);
                        }))
                        .into_any_element()
                });
        }
        breadcrumb.into_any_element()
    }

    fn render_import_toggle(
        &self,
        cx: &mut Context<Self>,
        id: &'static str,
        label: &'static str,
        action: impl Fn(&mut Self, &mut Context<Self>) + 'static,
    ) -> AnyElement {
        div()
            .id(id)
            .debug_selector(move || format!("#{id}"))
            .px_2()
            .py_1()
            .rounded_sm()
            .text_xs()
            .bg(rgb(0x30343d))
            .hover(|this| this.bg(rgb(0x3b4658)))
            .cursor_pointer()
            .child(label)
            .on_click(cx.listener(move |this, _, _, cx| action(this, cx)))
            .into_any_element()
    }

    fn render_import_rows(
        &mut self,
        range: Range<usize>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Vec<AnyElement> {
        let columns = self.grid_columns.max(1);
        let count = self.import_view.photos.len();
        let first = (range.start * columns).min(count);
        let last = (range.end * columns).min(count);
        let visible = self.import_view.photos[first..last]
            .iter()
            .map(|photo| {
                (
                    photo.key(),
                    photo.primary.backing.clone(),
                    photo.primary.name.clone(),
                    photo.primary.size,
                    photo.companion.as_ref().map(|file| file.name.clone()),
                )
            })
            .collect::<Vec<_>>();
        // Record the demand instead of replacing the scope here: GPUI calls
        // this processor with a bogus 0..1 range for measurement, and a
        // synchronous full-scope replacement from that call would cancel
        // every in-flight decode outside the measured rows. The deferred
        // publish coalesces both invocations, exactly like the library grid.
        self.record_import_demand(first..last, window, cx);
        let thumbnail_width = self.thumbnail_width;
        let tile_height = thumbnail_width as f32 + 48.0;
        range
            .map(|row| {
                let mut row_element = div()
                    .h(px(tile_height + THUMBNAIL_GAP))
                    .flex()
                    .gap(px(THUMBNAIL_GAP))
                    .px_3()
                    .pt_3();
                for (offset, (key, backing, name, size, companion)) in visible
                    .iter()
                    .enumerate()
                    .skip_while(|(offset, _)| first + offset < row * columns)
                    .take_while(|(offset, _)| first + offset < (row + 1) * columns)
                {
                    let _ = offset;
                    let selected = self.import_view.selection.contains(key);
                    let duplicate = self.import_view.duplicates.contains(key);
                    let image_content = match backing {
                        SourceBacking::Path(path) => {
                            let identity = self.render_identity_for(path);
                            match self.thumbnails.state_for(path, identity) {
                                PreviewState::Loaded(preview) => img(preview.image.clone())
                                    .max_w_full()
                                    .max_h_full()
                                    .into_any_element(),
                                PreviewState::Error(_) => {
                                    div().text_xs().child("Cannot decode").into_any_element()
                                }
                                _ => div().text_xs().child("Loading…").into_any_element(),
                            }
                        }
                        SourceBacking::Mtp(_) | SourceBacking::Ptp(_) => {
                            match self.import_view.staged.get(backing) {
                                Some(path) => {
                                    let identity = self.render_identity_for(path);
                                    match self.thumbnails.state_for(path, identity) {
                                        PreviewState::Loaded(preview) => img(preview.image.clone())
                                            .max_w_full()
                                            .max_h_full()
                                            .into_any_element(),
                                        PreviewState::Error(_) => div()
                                            .text_xs()
                                            .child("Cannot decode")
                                            .into_any_element(),
                                        _ => div().text_xs().child("Loading…").into_any_element(),
                                    }
                                }
                                None => div().text_xs().child("Device file").into_any_element(),
                            }
                        }
                    };
                    let toggle_key = key.clone();
                    row_element = row_element.child(
                        div()
                            .id(("import-tile", first + offset))
                            .debug_selector(move || format!("#import-tile-{}", first + offset))
                            .w(px(thumbnail_width as f32))
                            .h(px(tile_height))
                            .flex_none()
                            .flex()
                            .flex_col()
                            .p_2()
                            .rounded_md()
                            .when(selected, |this| {
                                this.bg(rgb(0x334b6e))
                                    .border_1()
                                    .border_color(rgb(0x6f9ddd))
                            })
                            .when(!selected, |this| {
                                this.border_1().border_color(rgb(0x30343d))
                            })
                            .when(duplicate, |this| {
                                this.opacity(if selected { 1.0 } else { 0.55 })
                            })
                            .hover(|this| this.bg(rgb(0x292d36)))
                            .child(
                                div()
                                    .w_full()
                                    .h(px(thumbnail_width as f32))
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
                                    .child(div().min_w_0().flex_1().truncate().child(name.clone()))
                                    .when(duplicate, |this| {
                                        this.child(
                                            div()
                                                .flex_none()
                                                .px_1()
                                                .rounded_sm()
                                                .bg(rgb(0x6b5320))
                                                .text_color(rgb(0xf5e0a9))
                                                .child("Duplicate"),
                                        )
                                    })
                                    .when_some(companion.as_ref(), |this, _companion| {
                                        this.child(
                                            div()
                                                .flex_none()
                                                .px_1()
                                                .rounded_sm()
                                                .bg(rgb(0x36547a))
                                                .text_color(rgb(0xd7e7ff))
                                                .child("+ pair"),
                                        )
                                    })
                                    .child(
                                        div()
                                            .flex_none()
                                            .text_color(rgb(0x9da7b6))
                                            .child(format_byte_size(*size)),
                                    ),
                            )
                            .child(
                                div()
                                    .absolute()
                                    .top(px(6.0))
                                    .left(px(6.0))
                                    .px_1()
                                    .rounded_sm()
                                    .text_xs()
                                    .when(selected, |this| {
                                        this.bg(rgb(0x6f9ddd)).text_color(rgb(0x10131a)).child("✓")
                                    })
                                    .when(!selected, |this| {
                                        this.bg(rgb(0x181a20)).text_color(rgb(0x59637a)).child("□")
                                    }),
                            )
                            .on_click(cx.listener(move |this, _, _, cx| {
                                this.import_toggle_photo(&toggle_key, cx);
                            })),
                    );
                }
                row_element.into_any_element()
            })
            .collect()
    }

    pub(crate) fn render_import_options_rail(&self, cx: &mut Context<Self>) -> AnyElement {
        let root = self.settings.root.clone();
        let base = self
            .import_view
            .destination_base
            .clone()
            .unwrap_or_else(|| root.clone());
        let base_display = base
            .strip_prefix(&root)
            .ok()
            .map(|relative| relative.display().to_string())
            .filter(|display| !display.is_empty())
            .unwrap_or_else(|| "Library root".to_owned());
        let include_duplicates = self.import_view.include_duplicates;
        let count: usize = self
            .import_view
            .photos
            .iter()
            .filter(|photo| self.import_view.selection.contains(&photo.key()))
            .filter(|photo| {
                include_duplicates || !self.import_view.duplicates.contains(&photo.key())
            })
            .count();
        let bytes: u64 = self
            .import_view
            .photos
            .iter()
            .filter(|photo| self.import_view.selection.contains(&photo.key()))
            .filter(|photo| {
                include_duplicates || !self.import_view.duplicates.contains(&photo.key())
            })
            .map(|photo| photo.primary.size + photo.companion.as_ref().map_or(0, |f| f.size))
            .sum();
        let running = self.import_view.job.is_some();
        let index_status: SharedString = if self.import_view.index_building {
            "Indexing library…".into()
        } else {
            format!("Library index: {} files", self.import_view.index_files).into()
        };
        div()
            .id("import-options-rail")
            .w(px(320.0))
            .flex_none()
            .h_full()
            .flex()
            .flex_col()
            .gap_3()
            .p_3()
            .border_l_1()
            .border_color(rgb(0x30343d))
            .bg(rgb(0x20242c))
            .overflow_y_scroll()
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap_2()
                    .child(div().text_sm().child("Destination folder"))
                    .child(
                        div()
                            .text_xs()
                            .text_color(rgb(0x9da7b6))
                            .truncate()
                            .child(base_display),
                    )
                    .child(
                        div()
                            .id("import-destination-picker")
                            .debug_selector(|| "#import-destination-picker".to_owned())
                            .px_2()
                            .py_1()
                            .rounded_sm()
                            .text_xs()
                            .bg(rgb(0x30343d))
                            .hover(|this| this.bg(rgb(0x3b4658)))
                            .cursor_pointer()
                            .child(if self.import_view.destination_picker_open {
                                "Hide folder picker"
                            } else {
                                "Choose or create folder…"
                            })
                            .on_click(cx.listener(|this, _, window, cx| {
                                this.toggle_destination_picker(window, cx);
                            })),
                    )
                    .when(self.import_view.destination_picker_open, |this| {
                        this.child(self.render_destination_picker(cx))
                    }),
            )
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap_2()
                    .child(div().text_sm().child("Folder layout"))
                    .child(self.render_layout_combobox(cx))
                    .child(self.render_import_checkbox(
                        cx,
                        "import-duplicates",
                        "Import duplicates",
                        include_duplicates,
                        |this, cx| this.import_toggle_include_duplicates(cx),
                    ))
                    .child(
                        div()
                            .text_xs()
                            .text_color(rgb(0x9da7b6))
                            .child(index_status),
                    ),
            )
            .child(div().flex_1())
            .child(if count > 0 && !running {
                div()
                    .id("import-start")
                    .px_4()
                    .py_2()
                    .rounded_sm()
                    .bg(rgb(0x315b8f))
                    .hover(|this| this.bg(rgb(0x3d6fa9)))
                    .cursor_pointer()
                    .text_sm()
                    .child(format!(
                        "Import {} photo{} ({})",
                        count,
                        if count == 1 { "" } else { "s" },
                        format_byte_size(bytes)
                    ))
                    .on_click(cx.listener(|this, _, _, cx| {
                        this.start_import(cx);
                    }))
                    .into_any_element()
            } else {
                div()
                    .px_4()
                    .py_2()
                    .rounded_sm()
                    .bg(rgb(0x242b37))
                    .text_color(rgb(0x59637a))
                    .text_sm()
                    .child(if running {
                        "Import running…".to_owned()
                    } else {
                        "Import 0 photos".to_owned()
                    })
                    .into_any_element()
            })
            .into_any_element()
    }

    /// Inline folder picker constrained to the library root, with creation
    /// of new folders.
    fn render_destination_picker(&self, cx: &mut Context<Self>) -> AnyElement {
        let root = self.settings.root.clone();
        let at_root = self.import_view.picker_dir == root;
        let relative = self
            .import_view
            .picker_dir
            .strip_prefix(&root)
            .ok()
            .map(|relative| relative.display().to_string())
            .unwrap_or_default();
        // A plain flex column on purpose: a `max_h` + overflow clip here would
        // let taffy shrink the rows to fit the cap, and the truncated row text
        // then clips itself down to nothing — the picker renders as an empty
        // bordered box (GPUI 0.2.2). Long listings just grow; the options rail
        // scrolls.
        let mut panel = div()
            .id("import-destination-panel")
            .debug_selector(|| "#import-destination-panel".to_owned())
            .rounded_md()
            .border_1()
            .border_color(rgb(0x3b4658))
            .bg(rgb(0x181c24))
            .p_2()
            .flex()
            .flex_col()
            .gap_1()
            .child(
                div()
                    .text_xs()
                    .text_color(rgb(0x9da7b6))
                    .truncate()
                    .child(if at_root {
                        "Library root".to_owned()
                    } else {
                        relative
                    }),
            );
        if !at_root {
            panel = panel
                .child(
                    self.render_picker_row(cx, "picker-up", "..", |this, window, cx| {
                        let parent = this
                            .import_view
                            .picker_dir
                            .parent()
                            .unwrap_or(&this.settings.root)
                            .to_path_buf();
                        this.navigate_picker(parent, window, cx);
                    }),
                )
                .child(self.render_picker_row(
                    cx,
                    "picker-root",
                    "Library root",
                    |this, window, cx| {
                        let root = this.settings.root.clone();
                        this.navigate_picker(root, window, cx);
                    },
                ));
        }
        match &self.import_view.picker_entries {
            None => {
                panel = panel.child(div().text_xs().text_color(rgb(0x9da7b6)).child("Reading…"));
            }
            Some(entries) if entries.is_empty() => {
                panel = panel.child(
                    div()
                        .text_xs()
                        .text_color(rgb(0x9da7b6))
                        .child("No subfolders"),
                );
            }
            Some(entries) => {
                for (index, name) in entries.iter().enumerate() {
                    let child_dir = self.import_view.picker_dir.join(name);
                    panel = panel.child(self.render_picker_row_id(
                        cx,
                        ("picker-entry", index),
                        name,
                        move |this, window, cx| {
                            this.navigate_picker(child_dir.clone(), window, cx);
                        },
                    ));
                }
            }
        }
        if self.import_view.creating_folder {
            let draft = self.import_view.new_folder_name.clone();
            panel = panel.child(
                div()
                    .flex()
                    .items_center()
                    .gap_1()
                    .child(
                        div()
                            .id("import-new-folder-input")
                            .key_context("ImportNewFolder")
                            .track_focus(&self.import_view.new_folder_focus)
                            .flex_1()
                            .min_w_0()
                            .px_2()
                            .py_1()
                            .rounded_sm()
                            .bg(rgb(0x10131a))
                            .border_1()
                            .border_color(rgb(0x3b4658))
                            .text_sm()
                            .on_action(cx.listener(
                                |this, _: &ImportNewFolderConfirm, window, cx| {
                                    this.create_destination_folder(window, cx);
                                },
                            ))
                            .on_key_down(cx.listener(|this, event, _, cx| {
                                this.import_new_folder_key_down(event, cx);
                            }))
                            .child(div().child(if draft.is_empty() {
                                "Folder name".to_owned()
                            } else {
                                draft
                            })),
                    )
                    .child(
                        div()
                            .id("import-new-folder-create")
                            .debug_selector(|| "#import-new-folder-create".to_owned())
                            .px_2()
                            .py_1()
                            .rounded_sm()
                            .text_xs()
                            .bg(rgb(0x315b8f))
                            .hover(|this| this.bg(rgb(0x3d6fa9)))
                            .cursor_pointer()
                            .child("Create")
                            .on_click(cx.listener(|this, _, window, cx| {
                                this.create_destination_folder(window, cx);
                            })),
                    ),
            );
        } else {
            panel = panel.child(self.render_picker_row(
                cx,
                "picker-new-folder",
                "New folder…",
                |this, window, cx| {
                    this.import_view.creating_folder = true;
                    this.import_view.new_folder_name.clear();
                    window.focus(&this.import_view.new_folder_focus);
                    cx.notify();
                },
            ));
        }
        panel = panel.child(
            div()
                .id("picker-use-folder")
                .debug_selector(|| "#picker-use-folder".to_owned())
                .mt_1()
                .px_2()
                .py_1()
                .rounded_sm()
                .text_xs()
                .bg(rgb(0x315b8f))
                .hover(|this| this.bg(rgb(0x3d6fa9)))
                .cursor_pointer()
                .child("Use this folder")
                .on_click(cx.listener(|this, _, window, cx| {
                    this.pick_destination(window, cx);
                })),
        );
        panel.into_any_element()
    }

    fn render_picker_row(
        &self,
        cx: &mut Context<Self>,
        id: &'static str,
        label: &str,
        action: impl Fn(&mut Self, &mut Window, &mut Context<Self>) + 'static,
    ) -> AnyElement {
        self.render_picker_row_id(cx, (id, 0), label, action)
    }

    fn render_picker_row_id(
        &self,
        cx: &mut Context<Self>,
        id: (&'static str, usize),
        label: &str,
        action: impl Fn(&mut Self, &mut Window, &mut Context<Self>) + 'static,
    ) -> AnyElement {
        div()
            .id(id)
            .debug_selector(move || format!("#{}-{}", id.0, id.1))
            .px_2()
            .py_1()
            .rounded_sm()
            .text_xs()
            .hover(|this| this.bg(rgb(0x292d36)))
            .cursor_pointer()
            .truncate()
            .child(label.to_owned())
            .on_click(cx.listener(move |this, _, window, cx| action(this, window, cx)))
            .into_any_element()
    }

    /// Combobox-style layout selector: a trigger button with an inline
    /// option list.
    fn render_layout_combobox(&self, cx: &mut Context<Self>) -> AnyElement {
        let current = self.import_view.preset;
        let current_label = import::destination::ALL_PRESETS
            .iter()
            .find(|(preset, _)| *preset == current)
            .map(|(_, label)| *label)
            .unwrap_or("");
        let mut container = div().flex().flex_col().gap_1().child(
            div()
                .id("import-layout-button")
                .debug_selector(|| "#import-layout-button".to_owned())
                .flex()
                .items_center()
                .justify_between()
                .px_2()
                .py_1()
                .rounded_sm()
                .text_xs()
                .bg(rgb(0x30343d))
                .hover(|this| this.bg(rgb(0x3b4658)))
                .cursor_pointer()
                .child(div().truncate().child(current_label.to_owned()))
                .child(div().text_color(rgb(0x9da7b6)).child("▾"))
                .on_click(cx.listener(|this, _, _, cx| {
                    this.import_view.layout_menu_open = !this.import_view.layout_menu_open;
                    cx.notify();
                })),
        );
        if self.import_view.layout_menu_open {
            let options = import::destination::ALL_PRESETS
                .iter()
                .enumerate()
                .map(|(index, (preset, label))| {
                    let is_current = *preset == current;
                    let preset = *preset;
                    div()
                        .id(("import-layout-option", index))
                        .debug_selector(move || format!("#import-layout-option-{index}"))
                        .px_2()
                        .py_1()
                        .text_xs()
                        .rounded_sm()
                        .when(is_current, |this| this.bg(rgb(0x334b6e)))
                        .hover(|this| this.bg(rgb(0x292d36)))
                        .cursor_pointer()
                        .child(if is_current {
                            format!("✓ {label}")
                        } else {
                            (*label).to_owned()
                        })
                        .on_click(cx.listener(move |this, _, _, cx| {
                            this.import_view.preset = preset;
                            this.import_view.layout_menu_open = false;
                            this.settings.import_destination_preset = preset;
                            if let Err(error) = config::save(&this.settings_path, &this.settings) {
                                eprintln!("cannot save import destination preset: {error:#}");
                            }
                            cx.notify();
                        }))
                        .into_any_element()
                })
                .collect::<Vec<_>>();
            container = container.child(
                div()
                    .rounded_sm()
                    .border_1()
                    .border_color(rgb(0x3b4658))
                    .bg(rgb(0x181c24))
                    .p_1()
                    .flex()
                    .flex_col()
                    .gap_0p5()
                    .children(options),
            );
        }
        container.into_any_element()
    }

    fn render_import_checkbox(
        &self,
        cx: &mut Context<Self>,
        id: &'static str,
        label: &str,
        checked: bool,
        action: impl Fn(&mut Self, &mut Context<Self>) + 'static,
    ) -> AnyElement {
        div()
            .id(id)
            .debug_selector(move || format!("#{id}"))
            .flex()
            .items_center()
            .gap_2()
            .cursor_pointer()
            .hover(|this| this.text_color(rgb(0xd8dde7)))
            .child(
                div()
                    .w(px(14.0))
                    .h(px(14.0))
                    .rounded_xs()
                    .border_1()
                    .border_color(rgb(0x59637a))
                    .when(checked, |this| {
                        this.bg(rgb(0x315b8f)).border_color(rgb(0x6f9ddd))
                    })
                    .flex()
                    .items_center()
                    .justify_center()
                    .text_xs()
                    .child(if checked {
                        "✓".to_owned()
                    } else {
                        String::new()
                    }),
            )
            .child(div().text_xs().child(label.to_owned()))
            .on_click(cx.listener(move |this, _, _, cx| action(this, cx)))
            .into_any_element()
    }

    fn render_import_progress_footer(&self, cx: &mut Context<Self>) -> gpui::Div {
        if self.import_view.job.is_none() {
            return self.render_thumbnail_progress_footer();
        }
        let progress = &self.import_view.job_progress;
        let fraction = if progress.total_bytes > 0 {
            (progress.copied_bytes as f32 / progress.total_bytes as f32).clamp(0.0, 1.0)
        } else {
            0.0
        };
        div()
            .flex_none()
            .flex()
            .items_center()
            .gap_3()
            .px_4()
            .py_2()
            .bg(rgb(0x242b37))
            .border_t_1()
            .border_color(rgb(0x3b4658))
            .text_sm()
            .child(format!(
                "{} / {} files · {} / {}",
                progress.copied_files,
                progress.total_files,
                format_byte_size(progress.copied_bytes),
                format_byte_size(progress.total_bytes),
            ))
            .child(
                div()
                    .flex_1()
                    .h(px(8.0))
                    .rounded_sm()
                    .overflow_hidden()
                    .bg(rgb(0x181a20))
                    .child(div().h_full().w(relative(fraction)).bg(rgb(0x3d6fa9))),
            )
            .child(
                div()
                    .text_xs()
                    .text_color(rgb(0x9da7b6))
                    .truncate()
                    .max_w(px(320.0))
                    .child(progress.current.clone()),
            )
            .child(
                div()
                    .id("import-cancel")
                    .px_3()
                    .py_1()
                    .rounded_sm()
                    .text_xs()
                    .bg(rgb(0x6b3535))
                    .hover(|this| this.bg(rgb(0x7d4040)))
                    .cursor_pointer()
                    .child("Cancel")
                    .on_click(cx.listener(|this, _, _, cx| {
                        this.cancel_import(cx);
                    })),
            )
    }

    /// Thumbnail-generation progress for the whole current directory:
    /// rendered previews over renderable photos (path-backed or staged).
    /// Throttled: the footer calls this every frame and the scan is O(photos).
    pub(crate) fn import_thumbnail_progress(&self) -> (usize, usize) {
        const PROGRESS_REFRESH: Duration = Duration::from_millis(250);
        if let Some((at, rendered, total)) = self.import_view.thumbnail_progress.get()
            && at.elapsed() < PROGRESS_REFRESH
        {
            return (rendered, total);
        }
        let staged = &self.import_view.staged;
        // Monotonic per-pass accounting: a file that has EVER been rendered
        // (or terminally failed) stays counted as done even if the LRU cache
        // evicts it, so the bar never moves backwards and completion stays
        // reachable for directories larger than the cache. Done is a SET of
        // files, not an added counter — the footer rescans every 250 ms, and
        // an additive high-water mark would re-count every terminal file on
        // each scan (the target racing into the thousands). The target is
        // the count of files with a renderable path, which staging only
        // ever grows towards its bounded budget.
        let mut ever_rendered = self.import_view.directory_pass_done.borrow_mut();
        let mut total = 0;
        for photo in &self.import_view.photos {
            for file in std::iter::once(&photo.primary).chain(photo.companion.as_ref()) {
                let display_path = match &file.backing {
                    SourceBacking::Path(path) => Some(path.clone()),
                    other => staged.get(other).cloned(),
                };
                let Some(path) = display_path else {
                    continue;
                };
                total += 1;
                let identity = self.render_identity_for(&path);
                if matches!(
                    self.thumbnails.state_for(&path, identity),
                    PreviewState::Loaded(_) | PreviewState::Error(_)
                ) {
                    ever_rendered.insert(file.backing.clone());
                }
            }
        }
        let rendered = ever_rendered.len().min(total);
        drop(ever_rendered);
        self.import_view
            .thumbnail_progress
            .set(Some((std::time::Instant::now(), rendered, total)));
        (rendered, total)
    }

    /// Stage the visible-row demand and publish it once per frame — the
    /// uniform_list processor runs for both a measurement range and the real
    /// range, and a synchronous replace-all from the measurement call would
    /// cancel decodes outside the measured rows.
    fn record_import_demand(
        &mut self,
        rows: Range<usize>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        match &mut self.pending_import_demand {
            Some(pending) => {
                pending.start = pending.start.min(rows.start);
                pending.end = pending.end.max(rows.end);
            }
            None => self.pending_import_demand = Some(rows),
        }
        if self.import_demand_scheduled {
            return;
        }
        self.import_demand_scheduled = true;
        cx.defer_in(window, |this, _, cx| {
            this.import_demand_scheduled = false;
            let Some(rows) = this.pending_import_demand.take() else {
                return;
            };
            if this.directory_view_mode != DirectoryViewMode::Import {
                return;
            }
            this.publish_import_demand(rows, cx);
        });
    }

    fn publish_import_demand(&mut self, rows: Range<usize>, cx: &mut Context<Self>) {
        let columns = self.grid_columns.max(1);
        let count = self.import_view.photos.len();
        let first = (rows.start.saturating_mul(columns)).min(count);
        let last = (rows.end.saturating_mul(columns)).min(count);
        let overscan = 2 * columns;
        let near_first = first.saturating_sub(overscan);
        let near_last = (last + overscan).min(count);
        let center = first.saturating_add(last).saturating_div(2);
        let mut requests = Vec::new();
        for index in near_first..near_last {
            let photo = &self.import_view.photos[index];
            let path = match &photo.primary.backing {
                SourceBacking::Path(path) => Some(path.clone()),
                other => self.import_view.staged.get(other).cloned(),
            };
            let Some(path) = path else {
                continue;
            };
            let on_screen = (first..last).contains(&index);
            let priority = if on_screen {
                let distance = index.abs_diff(center).min(i32::MAX as usize) as i32;
                JobPriority::visible(-distance)
            } else {
                JobPriority::near_viewport(0)
            };
            requests.push((
                path,
                RenderVariant::Thumbnail {
                    width: self.thumbnail_width,
                },
                priority,
            ));
        }
        self.replace_render_scope(IMPORT_SCOPE, requests, cx);
    }

    /// Render the whole current directory's thumbnails at speculative
    /// priority, a batch at a time: the footer's whole-directory progress
    /// implies a whole-directory pass. Speculative-class work yields to
    /// everything the user is looking at (Visible import tiles, face
    /// analysis), so this fills in at idle speed.
    fn schedule_directory_thumbnail_pass(&mut self, cx: &mut Context<Self>) {
        if self.import_view.directory_pass_cursor >= self.import_view.photos.len() {
            return;
        }
        let generation = self.import_view.directory_pass_generation;
        cx.spawn(async move |this, cx| {
            cx.background_executor()
                .timer(Duration::from_millis(80))
                .await;
            loop {
                let Some(batch) = this
                    .update(cx, |this, _| {
                        if this.import_view.directory_pass_generation != generation {
                            return None;
                        }
                        let photos = &this.import_view.photos;
                        let start = this.import_view.directory_pass_cursor;
                        if start >= photos.len() {
                            return None;
                        }
                        let end = (start + 16).min(photos.len());
                        let mut requests = Vec::new();
                        for photo in &photos[start..end] {
                            for file in
                                std::iter::once(&photo.primary).chain(photo.companion.as_ref())
                            {
                                if let SourceBacking::Path(path) = &file.backing {
                                    requests.push((
                                        path.clone(),
                                        RenderVariant::Thumbnail {
                                            width: this.thumbnail_width,
                                        },
                                        JobPriority::speculative(0),
                                    ));
                                }
                            }
                        }
                        this.import_view.directory_pass_cursor = end;
                        Some(requests)
                    })
                    .ok()
                    .flatten()
                else {
                    return;
                };
                this.update(cx, |this, cx| {
                    this.replace_render_scope(IMPORT_SCOPE, batch, cx);
                    cx.notify();
                })
                .ok();
                // Return to the executor between batches so user demand and
                // the UI stay responsive.
                cx.background_executor()
                    .timer(Duration::from_millis(20))
                    .await;
            }
        })
        .detach();
    }

    fn import_thumbnails_complete(&self) -> bool {
        let (rendered, total) = self.import_thumbnail_progress();
        total == 0 || rendered >= total
    }

    fn render_thumbnail_progress_footer(&self) -> gpui::Div {
        let (rendered, total) = self.import_thumbnail_progress();
        let fraction = if total > 0 {
            (rendered as f32 / total as f32).clamp(0.0, 1.0)
        } else {
            0.0
        };
        div()
            .flex_none()
            .flex()
            .items_center()
            .gap_3()
            .px_4()
            .py_2()
            .bg(rgb(0x242b37))
            .border_t_1()
            .border_color(rgb(0x3b4658))
            .text_sm()
            .child(format!("Rendering thumbnails · {} / {}", rendered, total))
            .child(
                div()
                    .flex_1()
                    .h(px(8.0))
                    .rounded_sm()
                    .overflow_hidden()
                    .bg(rgb(0x181a20))
                    .child(div().h_full().w(relative(fraction)).bg(rgb(0x3d6fa9))),
            )
    }
}

fn render_import_report_dialog(
    dialog: ImportReportDialog,
    cx: &mut Context<PictureBrowser>,
) -> gpui::Div {
    let report = &dialog.report;
    let mut failed_names = report
        .failed
        .iter()
        .map(|failure| failure.name.clone())
        .collect::<Vec<_>>();
    failed_names.truncate(6);
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
                .id("import-report-dialog")
                .w(px(560.0))
                .p_5()
                .rounded_md()
                .border_1()
                .border_color(rgb(0x4a5260))
                .bg(rgb(0x252932))
                .text_color(rgb(0xd8dde7))
                .child(div().text_xl().mb_2().child(dialog.status))
                .child(div().text_sm().mb_2().child(format!(
                    "Copied {} file{} ({}) into {} destination folder{}.",
                    report.copied_files,
                    if report.copied_files == 1 { "" } else { "s" },
                    format_byte_size(report.copied_bytes),
                    report.directories.len(),
                    if report.directories.len() == 1 {
                        ""
                    } else {
                        "s"
                    },
                )))
                .when(report.renamed_files > 0, |this| {
                    this.child(div().text_sm().mb_2().child(format!(
                        "{} file{} renamed to avoid overwriting.",
                        report.renamed_files,
                        if report.renamed_files == 1 { "" } else { "s" },
                    )))
                })
                .when(!report.failed.is_empty(), |this| {
                    this.child(
                        div()
                            .text_sm()
                            .text_color(rgb(0xf2b8b5))
                            .mb_2()
                            .child(format!(
                                "{} file{} failed:",
                                report.failed.len(),
                                if report.failed.len() == 1 { "" } else { "s" },
                            )),
                    )
                    .children(failed_names.iter().map(|name| {
                        div()
                            .text_xs()
                            .text_color(rgb(0xf2b8b5))
                            .child(name.clone())
                    }))
                    .when_some(
                        report.failed.first().map(|failure| failure.message.clone()),
                        |this, message| {
                            this.child(
                                div()
                                    .text_xs()
                                    .text_color(rgb(0x9da7b6))
                                    .mb_2()
                                    .child(message.to_string()),
                            )
                        },
                    )
                })
                .child(
                    div()
                        .id("close-import-report")
                        .mt_4()
                        .px_4()
                        .py_2()
                        .rounded_sm()
                        .bg(rgb(0x315b8f))
                        .cursor_pointer()
                        .child("Close")
                        .on_click(cx.listener(|this, _, _, cx| {
                            this.dismiss_import_report(cx);
                        })),
                ),
        )
}

impl PictureBrowser {
    /// Horizontal strip of folder tiles for navigating path sources.
    fn render_import_subdirectories(&self, cx: &mut Context<Self>) -> Option<AnyElement> {
        if self.import_view.recursive || self.import_view.subdirectories.is_empty() {
            return None;
        }
        let tiles = self
            .import_view
            .subdirectories
            .iter()
            .enumerate()
            .map(|(index, (name, backing))| {
                let label: SharedString = name.clone().into();
                let folder_name = name.clone();
                let folder_backing = backing.clone();
                div()
                    .id(("import-folder", index))
                    .debug_selector(move || format!("#import-folder-{index}"))
                    .px_3()
                    .py_2()
                    .mr_2()
                    .mb_2()
                    .rounded_md()
                    .border_1()
                    .border_color(rgb(0x30343d))
                    .bg(rgb(0x242b37))
                    .hover(|this| this.bg(rgb(0x292d36)))
                    .cursor_pointer()
                    .flex_none()
                    .child(div().text_sm().max_w(px(220.0)).truncate().child(label))
                    .child(
                        div()
                            .text_xs()
                            .text_color(rgb(0x9da7b6))
                            .child("Folder".to_owned()),
                    )
                    .on_click(cx.listener(move |this, _, _, cx| {
                        this.import_navigate_into(folder_name.clone(), folder_backing.clone(), cx);
                    }))
                    .into_any_element()
            })
            .collect::<Vec<_>>();
        Some(
            div()
                .flex_none()
                .flex()
                .flex_wrap()
                .p_3()
                .border_b_1()
                .border_color(rgb(0x30343d))
                .children(tiles)
                .into_any_element(),
        )
    }
}

impl std::fmt::Debug for ImportViewState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ImportViewState")
            .field("selected", &self.selected)
            .field("photos", &self.photos.len())
            .field("selected_count", &self.selected_count())
            .finish_non_exhaustive()
    }
}
