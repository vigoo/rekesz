use std::{collections::HashSet, env, ffi::OsString, ops::Range, path::PathBuf};

use anyhow::Result;
use gpui::{
    App, AppContext as _, Application, Bounds, Context, CursorStyle, FocusHandle, KeyBinding,
    ListSizingBehavior, MouseButton, MouseMoveEvent, Render, ScrollStrategy, SharedString,
    UniformListScrollHandle, Window, WindowBounds, WindowOptions, actions, div, img, prelude::*,
    px, rgb, size, uniform_list,
};

mod discovery;
mod preview;

use discovery::{DirectoryContents, DirectoryModel, NodeId, scan_directory};
use preview::{PreviewCache, PreviewState, decode_first};

actions!(
    directory_tree,
    [TreeUp, TreeDown, TreeLeft, TreeRight, TreeConfirm]
);

#[derive(Clone, Copy)]
struct VisibleRow {
    node: NodeId,
    depth: usize,
}

struct PictureBrowser {
    model: DirectoryModel,
    visible: Vec<VisibleRow>,
    expanded: HashSet<NodeId>,
    selected: NodeId,
    previews: PreviewCache,
    scroll: UniformListScrollHandle,
    tree_focus: FocusHandle,
    sidebar_width: f32,
    resizing_sidebar: bool,
}

impl PictureBrowser {
    fn new(root: PathBuf, image_cache_size: usize, cx: &mut Context<Self>) -> Self {
        let model = DirectoryModel::new(root);
        let root = model.root();
        let mut this = Self {
            model,
            visible: vec![VisibleRow {
                node: root,
                depth: 0,
            }],
            expanded: HashSet::from([root]),
            selected: root,
            previews: PreviewCache::new(image_cache_size),
            scroll: UniformListScrollHandle::new(),
            tree_focus: cx.focus_handle(),
            sidebar_width: 300.0,
            resizing_sidebar: false,
        };
        this.ensure_directory_loaded(root, cx);
        this
    }

    fn ensure_directory_loaded(&mut self, node: NodeId, cx: &mut Context<Self>) {
        let Some((request, path)) = self.model.begin_load(node) else {
            if self.selected == node {
                self.ensure_preview(node, cx);
            }
            return;
        };
        cx.notify();
        let scan = cx.background_spawn(async move { scan_directory(&path) });
        cx.spawn(async move |this, cx| {
            let result = scan.await;
            this.update(cx, |this, cx| {
                if this.model.finish_load(node, request, result) {
                    this.rebuild_visible();
                    if this.selected == node {
                        this.ensure_preview(node, cx);
                    }
                    cx.notify();
                }
            })
            .ok();
        })
        .detach();
    }

    fn ensure_preview(&mut self, node: NodeId, cx: &mut Context<Self>) {
        let Some(contents) = self.model.loaded(node) else {
            return;
        };
        let Some(request) = self.previews.begin(node) else {
            return;
        };
        let images = contents.images.clone();
        cx.notify();
        let decode = cx.background_spawn(async move { decode_first(images) });
        cx.spawn(async move |this, cx| {
            let result = decode.await;
            this.update(cx, |this, cx| {
                if this.previews.finish(node, request, result) {
                    cx.notify();
                }
            })
            .ok();
        })
        .detach();
    }

    fn select(&mut self, node: NodeId, cx: &mut Context<Self>) {
        self.selected = node;
        self.previews.mark_used(node);
        self.ensure_directory_loaded(node, cx);
        self.ensure_preview(node, cx);
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
                    .on_click(cx.listener(move |this, _, window, cx| {
                        window.focus(&this.tree_focus);
                        this.select(select_node, cx);
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
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
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

        let selected_node = self.model.node(self.selected);
        let preview_state = match &selected_node.contents {
            DirectoryContents::Error(error) => PreviewState::Error(error.clone()),
            DirectoryContents::Unloaded | DirectoryContents::Loading(_) => PreviewState::Loading(0),
            DirectoryContents::Loaded(_) => self.previews.state(self.selected),
        };
        let (file_name, preview_content) = match preview_state {
            PreviewState::Loaded(preview) => (
                preview.file_name.to_string(),
                img(preview.image.clone())
                    .max_w_full()
                    .max_h_full()
                    .into_any_element(),
            ),
            PreviewState::Loading(_) => (
                "Loading…".into(),
                div().text_sm().child("Loading…").into_any_element(),
            ),
            PreviewState::Error(error) => {
                let message = error.to_string();
                (
                    "Image error".into(),
                    div().text_sm().child(message).into_any_element(),
                )
            }
            PreviewState::Missing => (
                "No image".into(),
                div()
                    .text_sm()
                    .child("No image in this directory")
                    .into_any_element(),
            ),
            PreviewState::Unloaded => (
                "Loading…".into(),
                div().text_sm().child("Loading…").into_any_element(),
            ),
        };
        let preview = div()
            .flex_1()
            .min_w_0()
            .h_full()
            .flex()
            .flex_col()
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
                    .child(file_name),
            )
            .child(
                div()
                    .flex_1()
                    .min_h_0()
                    .flex()
                    .items_center()
                    .justify_center()
                    .p_6()
                    .text_color(rgb(0x8d96a5))
                    .child(preview_content),
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
        div()
            .size_full()
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
            .child(preview)
    }
}

const DEFAULT_IMAGE_CACHE_SIZE: usize = 8;

struct Config {
    root: PathBuf,
    image_cache_size: usize,
}

fn config_from_args(args: impl IntoIterator<Item = OsString>) -> Result<Config> {
    let mut args = args.into_iter();
    let _program = args.next();
    let mut requested_root = None;
    let mut image_cache_size = DEFAULT_IMAGE_CACHE_SIZE;
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
        ]);
        let bounds = Bounds::centered(None, size(px(1100.0), px(720.0)), cx);
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                ..Default::default()
            },
            |_, cx| {
                cx.new(|cx| PictureBrowser::new(config.root.clone(), config.image_cache_size, cx))
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
    }
}
