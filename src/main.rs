use std::{
    collections::HashSet,
    env,
    path::{Path, PathBuf},
};

use anyhow::{Context as _, Result, bail};
use gpui::{
    App, Application, Bounds, Context, MouseButton, Render, SharedString, Window, WindowBounds,
    WindowOptions, div, img, prelude::*, px, rgb, size,
};

mod discovery;
mod preview;

use discovery::Catalog;
use preview::{Preview, PreviewCache};

#[derive(Clone)]
struct DirectoryRow {
    path: PathBuf,
    name: SharedString,
    depth: usize,
    has_children: bool,
}

struct PictureBrowser {
    catalog: Catalog,
    selected: PathBuf,
    preview: Preview,
    previews: PreviewCache,
    expanded: HashSet<PathBuf>,
}

impl PictureBrowser {
    fn new(root: PathBuf) -> Self {
        let catalog = Catalog::scan(root.clone());
        let mut expanded = HashSet::new();
        expanded.insert(root.clone());
        let mut previews = PreviewCache::default();
        let preview = previews.get(&root, &catalog);
        Self {
            selected: root,
            preview,
            previews,
            catalog,
            expanded,
        }
    }

    fn rows(&self) -> Vec<DirectoryRow> {
        let mut rows = Vec::new();
        self.collect_rows(self.catalog.root(), 0, &mut rows);
        rows
    }

    fn collect_rows(&self, path: &Path, depth: usize, rows: &mut Vec<DirectoryRow>) {
        let Some(directory) = self.catalog.directory(path) else {
            return;
        };
        let name = if depth == 0 {
            path.to_string_lossy().into_owned()
        } else {
            path.file_name()
                .unwrap_or(path.as_os_str())
                .to_string_lossy()
                .into_owned()
        };
        rows.push(DirectoryRow {
            path: path.to_path_buf(),
            name: name.into(),
            depth,
            has_children: !directory.children.is_empty(),
        });

        if self.expanded.contains(path) {
            for child in &directory.children {
                self.collect_rows(child, depth + 1, rows);
            }
        }
    }

    fn select(&mut self, path: PathBuf, cx: &mut Context<Self>) {
        self.selected = path.clone();
        self.preview = self.previews.get(&path, &self.catalog);
        cx.notify();
    }

    fn toggle(&mut self, path: PathBuf, cx: &mut Context<Self>) {
        if !self.expanded.remove(&path) {
            self.expanded.insert(path);
        }
        cx.notify();
    }
}

impl Render for PictureBrowser {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let rows = self.rows();
        let selected = self.selected.clone();

        let tree = div()
            .id("directory-tree")
            .w(px(300.0))
            .h_full()
            .flex_none()
            .overflow_y_scroll()
            .bg(rgb(0x181a20))
            .border_r_1()
            .border_color(rgb(0x30343d))
            .py_2()
            .children(rows.into_iter().enumerate().map(|(row_ix, row)| {
                let select_path = row.path.clone();
                let toggle_path = row.path.clone();
                let is_selected = row.path == selected;
                div()
                    .id(("directory", row_ix))
                    .flex()
                    .items_center()
                    .h(px(30.0))
                    .pl(px(10.0 + row.depth as f32 * 16.0))
                    .pr_2()
                    .text_sm()
                    .text_color(rgb(0xd7dae0))
                    .when(is_selected, |this| this.bg(rgb(0x334b6e)))
                    .hover(|this| this.bg(rgb(0x292d36)))
                    .child(
                        div()
                            .id(("toggle", row_ix))
                            .w(px(20.0))
                            .flex_none()
                            .text_color(rgb(0x8d96a5))
                            .child(if row.has_children {
                                if self.expanded.contains(&row.path) {
                                    "▾"
                                } else {
                                    "▸"
                                }
                            } else {
                                ""
                            })
                            .on_mouse_down(
                                MouseButton::Left,
                                cx.listener(move |this, _, _, cx| {
                                    this.toggle(toggle_path.clone(), cx);
                                }),
                            ),
                    )
                    .child(div().min_w_0().truncate().child(row.name))
                    .on_click(cx.listener(move |this, _, _, cx| {
                        this.select(select_path.clone(), cx);
                    }))
            }));

        let title = self.selected.to_string_lossy().into_owned();
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
                    .child(title),
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
                    .child(if let Some(image) = self.preview.image.clone() {
                        img(image).max_w_full().max_h_full().into_any_element()
                    } else {
                        div()
                            .text_sm()
                            .child(
                                self.preview
                                    .error
                                    .clone()
                                    .unwrap_or_else(|| "No image in this directory".into()),
                            )
                            .into_any_element()
                    }),
            );

        div().size_full().flex().child(tree).child(preview)
    }
}

fn root_from_args() -> Result<PathBuf> {
    let requested = env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or(env::current_dir()?);
    let root = requested
        .canonicalize()
        .with_context(|| format!("cannot open {}", requested.display()))?;
    if !root.is_dir() {
        bail!("{} is not a directory", root.display());
    }
    Ok(root)
}

fn main() -> Result<()> {
    let root = root_from_args()?;
    Application::new().run(move |cx: &mut App| {
        let bounds = Bounds::centered(None, size(px(1100.0), px(720.0)), cx);
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                ..Default::default()
            },
            |_, cx| cx.new(|_| PictureBrowser::new(root.clone())),
        )
        .expect("failed to open window");
        cx.activate(true);
    });
    Ok(())
}
