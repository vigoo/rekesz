# Repository Instructions

## Rust quality gates

After every source or configuration change:

1. Run `cargo fmt --all -- --check`.
2. Run `cargo clippy --all-targets --all-features -- -D warnings`.
3. Run the relevant tests. For changes affecting Rust behavior, run `cargo test`.

Do not consider the work complete while any required command fails. If formatting is
needed, run `cargo fmt --all` and then rerun the formatting check. Fix Clippy
warnings in the code; do not suppress them with broad `#[allow]` attributes unless
the exception is necessary and documented.

## UI work

UI changes follow `design/ui-guidelines.md` — design tokens, control idioms,
layout rules, keyboard conventions, `debug_selector` coverage, and the testing
requirements for every surface you touch (per-area matrix in
`ui-redesign-plan.md` § "UI testing strategy"). A UI change without tests for
the surfaces it modifies is incomplete.
