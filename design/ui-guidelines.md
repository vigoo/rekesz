# Rekesz UI Guidelines

How UI is built in rekesz. Binding for every new surface and every change to
an existing one — for humans and agents alike. Established 2026-11-15 during
the UI redesign (`ui-redesign-plan.md`); codifies what the screenshot-tour
review found and what user feedback has enforced since 2026-10.

## 1. Design tokens

All colors, type, and spacing come from the shared token module (`src/ui.rs`,
Stage 0 of the redesign plan). Never hard-code a color, font size, or padding
in a render function when a token exists; adding a token is the answer when
none fits.

- **Type scale** — gpui's built-ins are the tokens: `text_xs` (11–12px,
  captions/chips) · `text_sm` (13–14px, body/controls) · 15–17px section
  titles (`text_lg`) · larger dialog/page titles. Two weights only: regular
  + semibold. One accent family; dim text `#8b93a0`, faint `#5c6470`, body
  `#cdd3dc`, emphasized `#eef2f8`.
- **Spacing**: multiples of 4; panel padding 12–16; related controls 8–12
  apart; groups separated by 16–24, never by color alone.
- **Surfaces**: canvas `#0b0d10`–`#101318`, panels `#14171d`–`#171b22`,
  borders `#23262e`/`#2c3340`. Photos always sit on the darkest canvas.
- **Accent** `#4f7cd9` (selection, primary). Success `#3e7d4f`, danger
  `#a04b42`. The photo canvas never tints by accent.

## 2. Control idioms

- **Buttons** have four variants: `primary` (one per view — the action you'd
  reach for first), `default`, `subtle` (borderless), `danger`. If two
  actions compete for `primary`, the design is wrong — pick the one that
  advances the task.
- **Chips** (small rounded rect, 11–12px) carry *state*, not actions:
  counts, filters, scopes (`person filter ✕` is a chip with a dismiss
  affordance — a chip that does something on click is a button).
- **Toggles are checkboxes or segmented pairs** (Enabled/Disabled), never
  cycling pill buttons. Pickers are comboboxes. This was explicit feedback.
- **Switcher buttons group by function**: browsing modes (Grid/Map/People)
  form one segmented group; unrelated toggles (Developed) sit apart. After
  any header change, re-check every adjacent grouping visually — this was
  flagged twice.
- **Banners are scoped chips**: contextual offers (capture-time ordering,
  "N unreviewed suggestions — open triage") render as a single inline chip
  row belonging to the surface that understands them — never above panels
  that can't act on them.
- **Dialogs are self-contained**: all controls inside the frame, one button
  pair, visible draggable scrollbar when content overflows (wheel-only
  scrolling is a defect).
- **Tooltips** carry the fields that cannot be read off the item at a glance
  (dimensions, capture time, format, flags) — see the grid tile tooltip as
  the reference.

## 3. Layout rules

- **Header spans the full window width**; right panels live entirely in the
  workspace row below it. A panel that renders its own header beside the app
  header (or clips it) is a bug (C4).
- **Overlays must not occlude interactive elements.** In-flow layout beats
  floating bars; when an overlay is unavoidable (status dock, drawer), it
  anchors to an edge and never covers the primary action.
- **Footers are in-flow and span their container**; progress bars measure
  the whole task set (`N / M` of everything visible), with per-file names as
  a subtitle — per-file slices were explicit feedback.
- **Full-window modes hide library chrome deliberately** (Import hides the
  tree) and always carry an explicit exit control (`← Back to library` /
  `← Cancel import` while a job runs).
- **Modes own their banners**: a banner shown in a mode that cannot act on
  it (capture-time order in Map/People/Import) is a defect (C3).

## 4. Status text and text-in-tight-spaces

- **Status labels are precise, never optimistic.** Name the actual work,
  stage, and remaining queue; fallback text must never imply completion
  ("Importing… 7/12 files · Scan-008.jpg", not "Finishing…").
- **Dense telemetry belongs behind an affordance** (ⓘ popover, drawer
  footer), not in a header line end users read (C8). Users who want
  diagnostics can open the activity drawer — precision is preserved there,
  in full.
- **Every line in a fixed-height row truncates**: titles use light `#e1e5ec`
  + `truncate()`, status lines use muted `#9da7b6` + `truncate()`. Status
  lines that embed paths or errors are never safe untruncated.
- **Queue rows name their items** (file path), not opaque ids + folder name.

## 5. Keyboard and focus

- Plain-letter bindings (`d`, `y`, `n`, `e`, `space`) go through the **root
  key listener**; text inputs must `stop_propagation` (gpui dispatches
  keybindings before key listeners and consumes keys even for no-op
  actions).
- Every review/triage surface states its keys on screen (`Y accept · N
  reject · E not a face · ⌘Z undo`) — keyboard-first UI must be
  discoverable.
- `Esc` always means "back/out of the current sub-state"; destructive
  confirmations never bind to a single keypress.

## 6. Selectors and testability

- Every interactive element gets a `debug_selector` with a stable kebab-case
  name (`#import-start`, `#thumbnail-{index}`, `#people-face-tile-{f}-{p}`).
  Tests, the screenshot tour, and future automation all key off these.
- New surfaces: state transitions, keyboard behavior, and (for geometry)
  recorded-bounds assertions are part of the definition of done — see
  `ui-redesign-plan.md` § "UI testing strategy" for the per-area matrix.
  Geometry-dependent layouts get bounds tests, not eyeball checks.
- **`debug_bounds(..).is_none()` is only reliable for elements that never
  rendered in the window's lifetime.** gpui 0.2.2 keeps TWO frame buffers;
  inserts go into `next_frame`, `draw()` swaps them, and `Frame::clear()`
  never touches `debug_bounds` — so queries return the union of every draw
  with the same parity: a removed element's entry resurfaces on
  same-parity frames, and opposite-parity frames can report absence for an
  element that IS rendered. Both directions are unreliable. Presence
  asserts are sound; for absence, either (a) use a fresh window already in
  the target state (the element never painted there), or (b) assert
  behaviorally: click the element's last recorded position and verify the
  handler no longer fires (see `gpui_recursive_banner_stays_in_grid_mode`,
  and the settings-search stray-click check).
- **Occlude-dialog remounts swallow clicks on every second instance in the
  TEST harness** (gpui 0.2.2; isolated with a minimal overlay). Mouse
  dispatch runs against the rendered frame; after an occlude overlay
  closes, the next dialog instance's body hitboxes alternate with the
  frame buffers — instance 1's clicks land, instance 2's vanish, instance
  3's land again. Real usage repaints continuously and is unaffected. In
  tests, open+close the dialog once more ("burn" the dead instance)
  before clicking into a reopened dialog — see
  `gpui_settings_dialog_cancels_and_saves_painted_control_changes`.
- Status-quo risk: if a test asserts on visible text that the redesign
  rewords, update the test in the same stage — never delete an assertion to
  make a restyle pass.

## 7. Platform conventions

- The photo canvas is dark everywhere; chrome tokens may follow the system
  theme later, behind tokens only.
- Menus: macOS gets the real menu bar (the `set_menus` data), Linux keeps
  the in-window bar, Windows uses native menus; accelerators use Cmd/Ctrl
  per platform.
- Native file pickers everywhere; dialogs stay centered cards on
  Windows/Linux, sheet-style is optional on macOS.
- Fonts resolve per platform (SF Pro / Segoe UI / system sans on Linux);
  never ship a bundled font as the default.

## 8. Process for UI changes

1. Check this doc + the relevant mockup (`design/mockups/`) before writing
   code.
2. Implement with tokens + selectors.
3. Land the UI tests required by the plan's testing matrix.
4. Re-run the screenshot tour for affected steps
   (`scripts/screenshot-tour.sh real|scratch`) and compare against the
   mockup — the tour is the pixel-level smoke test.
5. Two review sub-agents per stage (advisory-review process); all findings
   addressed before moving on.
