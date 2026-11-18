//! Shared UI design tokens and small element builders (ui-redesign-plan
//! Stage 0; see `design/ui-guidelines.md`).
//!
//! Colors, type sizes, and control idioms are defined once here so every
//! surface stays consistent. Never hard-code a color or size in a render
//! function when a token exists — add a token instead.

// The module deliberately ships the full documented token set up front;
// redesign stages adopt the not-yet-referenced tokens one surface at a
// time (ui-redesign-plan.md stages 2-7).
#![allow(dead_code)]

use gpui::{Div, SharedString, Stateful, div, prelude::*, px, rgb};

/// Photo canvas (the darkest background; images sit on this).
pub const CANVAS_DEEP: u32 = 0x0b0d10;
/// Window canvas behind panels.
pub const CANVAS: u32 = 0x101318;
/// Raised panel background (sidebars, rails, inspector).
pub const PANEL: u32 = 0x171b22;
/// Inset panel background (headers, docks).
pub const PANEL_SUNK: u32 = 0x14171d;
pub const BORDER: u32 = 0x23262e;
pub const BORDER_STRONG: u32 = 0x2c3340;
pub const TEXT: u32 = 0xcdd3dc;
pub const TEXT_DIM: u32 = 0x8b93a0;
pub const TEXT_FAINT: u32 = 0x5c6470;
pub const TEXT_EMPHASIS: u32 = 0xeef2f8;
pub const ACCENT: u32 = 0x4f7cd9;
pub const ACCENT_SURFACE: u32 = 0x243045;
pub const ACCENT_BORDER: u32 = 0x3c5f9e;
pub const ACCENT_TEXT: u32 = 0xcfe0f8;
pub const INFO_SURFACE: u32 = 0x1a2130;
pub const INFO_BORDER: u32 = 0x31456b;
pub const INFO_TEXT: u32 = 0x8fb0e8;
pub const INFO_HOVER: u32 = 0x22304a;
pub const SUCCESS: u32 = 0x3e7d4f;
pub const PRIMARY_SURFACE: u32 = 0x2c4a7c;
pub const PRIMARY_BORDER: u32 = 0x3c5f9e;
pub const PRIMARY_TEXT: u32 = 0xeaf1fb;
pub const DEFAULT_SURFACE: u32 = 0x1d222b;
pub const DANGER_SURFACE: u32 = 0x5a2d28;
pub const DANGER_BORDER: u32 = 0x7c3f36;
pub const DANGER_TEXT: u32 = 0xf2d7d3;
/// Neutral chip/button label color.
pub const CHIP_TEXT: u32 = 0xaeb6c2;

/// Visual weight of a [`button`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ButtonVariant {
    /// The one advancing action of the current view.
    Primary,
    /// Standard secondary action.
    Default,
    /// Borderless quiet action.
    Subtle,
    /// Destructive or cancelling action.
    Danger,
}

/// A labeled button: id, `debug_selector`, tokens, hover state, and label are
/// applied; the caller attaches `.on_click` / `.on_key_down` and extra
/// spacing.
pub fn button(
    id: &'static str,
    label: impl Into<SharedString>,
    variant: ButtonVariant,
) -> Stateful<Div> {
    let (surface, border, text, hover) = match variant {
        ButtonVariant::Primary => (PRIMARY_SURFACE, PRIMARY_BORDER, PRIMARY_TEXT, 0x35568c),
        ButtonVariant::Default => (DEFAULT_SURFACE, BORDER_STRONG, TEXT, 0x252b35),
        ButtonVariant::Subtle => (PANEL, BORDER, TEXT_DIM, DEFAULT_SURFACE),
        ButtonVariant::Danger => (DANGER_SURFACE, DANGER_BORDER, DANGER_TEXT, 0x6b382f),
    };
    let button = div()
        .id(id)
        .debug_selector(move || format!("#{id}"))
        .px_3()
        .py_1()
        .rounded_md()
        .text_sm()
        .bg(rgb(surface))
        .text_color(rgb(text))
        .hover(move |this| this.bg(rgb(hover)))
        .child(label.into());
    // Subtle is the borderless variant (guidelines §2).
    if variant == ButtonVariant::Subtle {
        button
    } else {
        button.border_1().border_color(rgb(border))
    }
}

/// Visual weight of a [`chip`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChipVariant {
    /// Neutral state chip (counts, scopes).
    Default,
    /// Informational accent chip (offers, hints).
    Info,
}

/// A non-interactive state chip. Chips carry state, not actions; an
/// element that acts on click is a `button`.
pub fn chip(text: impl Into<SharedString>, variant: ChipVariant) -> Div {
    let (surface, border, text_color) = match variant {
        ChipVariant::Default => (DEFAULT_SURFACE, BORDER_STRONG, CHIP_TEXT),
        ChipVariant::Info => (INFO_SURFACE, INFO_BORDER, INFO_TEXT),
    };
    div()
        .px_2()
        .py_1()
        .rounded_md()
        .text_xs()
        .bg(rgb(surface))
        .border_1()
        .border_color(rgb(border))
        .text_color(rgb(text_color))
        .child(text.into())
}

/// The standard header row height including its bottom border; right
/// panels and dropdowns align against it.
pub const HEADER_HEIGHT: f32 = 44.0;

/// The standard 44px header row: full window width, panels sit below it.
pub fn header_row() -> Div {
    div()
        .h(px(HEADER_HEIGHT))
        .flex_none()
        .flex()
        .items_center()
        .px_4()
        .bg(rgb(PANEL))
        .border_b_1()
        .border_color(rgb(BORDER_STRONG))
        .text_sm()
        .text_color(rgb(TEXT))
}
