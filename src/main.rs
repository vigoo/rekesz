use std::{
    collections::{BTreeSet, HashSet},
    env,
    ffi::OsString,
    ops::Range,
    path::PathBuf,
};

use anyhow::Result;
use gpui::{
    App, AppContext as _, Application, Bounds, ClickEvent, Context, CursorStyle, FocusHandle,
    KeyBinding, ListSizingBehavior, MouseButton, MouseMoveEvent, Render, ScrollStrategy,
    SharedString, UniformListScrollHandle, Window, WindowBounds, WindowOptions, actions, div, img,
    prelude::*, px, relative, rgb, size, uniform_list,
};

mod discovery;
mod formats;
mod preview;
mod rendering;

use discovery::{DirectoryContents, DirectoryModel, NodeId, RawJpegPreference, scan_directory};
use preview::{PreviewCache, PreviewState, decode_path};

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

enum MainView {
    Image(PathBuf),
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
}

impl PictureBrowser {
    fn new(
        root: PathBuf,
        image_cache_size: usize,
        pair_preference: RawJpegPreference,
        cx: &mut Context<Self>,
    ) -> Self {
        let model = DirectoryModel::new(root);
        let root = model.root();
        let root_path = model.node(root).path.to_path_buf();
        let mut this = Self {
            model,
            visible: vec![VisibleRow {
                node: root,
                depth: 0,
            }],
            expanded: HashSet::from([root]),
            selected: root,
            previews: PreviewCache::new(image_cache_size),
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
        };
        this.ensure_directory_loaded(root, cx);
        this
    }

    fn ensure_directory_loaded(&mut self, node: NodeId, cx: &mut Context<Self>) {
        let Some((request, path)) = self.model.begin_load(node) else {
            return;
        };
        self.work.start();
        cx.notify();
        let pair_preference = self.pair_preference;
        let scan = cx.background_spawn(async move { scan_directory(&path, pair_preference) });
        cx.spawn(async move |this, cx| {
            let result = scan.await;
            this.update(cx, |this, cx| {
                this.work.finish();
                if this.model.finish_load(node, request, result) {
                    if this
                        .model
                        .loaded(node)
                        .is_some_and(|contents| contents.children.is_empty())
                    {
                        this.expanded.remove(&node);
                    }
                    this.rebuild_visible();
                    cx.notify();
                }
            })
            .ok();
        })
        .detach();
    }

    fn ensure_image(
        &mut self,
        path: PathBuf,
        thumbnail_width: Option<u32>,
        cx: &mut Context<Self>,
    ) {
        let cache = if thumbnail_width.is_some() {
            &mut self.thumbnails
        } else {
            &mut self.previews
        };
        let Some(request) = cache.begin(&path) else {
            return;
        };
        self.work.start();
        cx.notify();
        let source_path = path.clone();
        let decode = cx.background_spawn(async move { decode_path(source_path, thumbnail_width) });
        cx.spawn(async move |this, cx| {
            let result = decode.await;
            this.update(cx, |this, cx| {
                this.work.finish();
                if let PreviewState::Loaded(preview) = &result {
                    this.model
                        .record_render_metadata(path.clone(), preview.metadata);
                }
                let cache = if thumbnail_width.is_some() {
                    &mut this.thumbnails
                } else {
                    &mut this.previews
                };
                if cache.finish(&path, request, result) {
                    cx.notify();
                }
            })
            .ok();
        })
        .detach();
    }

    fn select(&mut self, node: NodeId, cx: &mut Context<Self>) {
        self.selected = node;
        self.main_view = MainView::Dir(
            self.model.node(node).path.to_path_buf(),
            ImageSelection::default(),
        );
        self.saved_selection = ImageSelection::default();
        self.ensure_directory_loaded(node, cx);
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
            self.ensure_directory_loaded(node, cx);
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
            self.ensure_directory_loaded(node, cx);
            self.rebuild_visible();
            cx.notify();
        }
    }

    fn select_thumbnail(&mut self, index: usize, extend: bool, toggle: bool) {
        let MainView::Dir(_, selection) = &mut self.main_view else {
            return;
        };
        selection.select(index, extend, toggle);
        self.saved_selection = selection.clone();
    }

    fn show_image(&mut self, path: PathBuf, cx: &mut Context<Self>) {
        if let MainView::Dir(_, selection) = &self.main_view {
            self.saved_selection = selection.clone();
        }
        self.main_view = MainView::Image(path.clone());
        self.previews.mark_used(&path);
        self.ensure_image(path, None, cx);
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
        self.select_thumbnail(target, extend, false);
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
            self.main_view = MainView::Dir(
                self.model.node(self.selected).path.to_path_buf(),
                self.saved_selection.clone(),
            );
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
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Vec<gpui::AnyElement> {
        let images = self.selected_images().to_vec();
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
                for (index, path) in images
                    .iter()
                    .enumerate()
                    .skip(row * self.grid_columns)
                    .take(self.grid_columns)
                {
                    let path = path.clone();
                    self.ensure_image(path.clone(), Some(self.thumbnail_width), cx);
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

impl Render for PictureBrowser {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let main_width = (f32::from(window.viewport_size().width) - self.sidebar_width - 29.0)
            .max(self.thumbnail_width as f32);
        self.grid_columns = ((main_width + THUMBNAIL_GAP)
            / (self.thumbnail_width as f32 + THUMBNAIL_GAP))
            .floor()
            .max(1.0) as usize;
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
            .child(
                uniform_list(
                    "directory-rows",
                    self.visible.len(),
                    cx.processor(Self::render_rows),
                )
                .size_full()
                .track_scroll(self.scroll.clone())
                .with_sizing_behavior(ListSizingBehavior::Auto),
            );

        let (title, show_back, main_content) = match &self.main_view {
            MainView::Image(path) => {
                let path = path.clone();
                self.ensure_image(path.clone(), None, cx);
                let content = match self.previews.state(&path) {
                    PreviewState::Loaded(preview) => img(preview.image.clone())
                        .max_w_full()
                        .max_h_full()
                        .into_any_element(),
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
                        .size_full()
                        .flex()
                        .items_center()
                        .justify_center()
                        .p_6()
                        .child(content)
                        .into_any_element(),
                )
            }
            MainView::Dir(path, selection) => {
                let count = self.selected_images().len();
                let content = match &self.model.node(self.selected).contents {
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
                (
                    format!("{} — {} selected", path.display(), selection.indices.len()),
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
            .child(main_view);
        let progress = self.work.fraction();
        let progress_label = format!(
            "Working… {} of {} tasks complete",
            self.work.completed, self.work.total
        );
        div()
            .size_full()
            .flex()
            .flex_col()
            .child(workspace)
            .when(self.work.active > 0, |this| {
                this.child(
                    div()
                        .h(px(22.0))
                        .flex_none()
                        .relative()
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
            })
    }
}

const DEFAULT_IMAGE_CACHE_SIZE: usize = 8;

struct Config {
    root: PathBuf,
    image_cache_size: usize,
    pair_preference: RawJpegPreference,
}

fn config_from_args(args: impl IntoIterator<Item = OsString>) -> Result<Config> {
    let mut args = args.into_iter();
    let _program = args.next();
    let mut requested_root = None;
    let mut image_cache_size = DEFAULT_IMAGE_CACHE_SIZE;
    let mut pair_preference = RawJpegPreference::Raw;
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
    Ok(Config {
        root,
        image_cache_size,
        pair_preference,
    })
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
        ]);
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
}
