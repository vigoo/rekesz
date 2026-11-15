#!/usr/bin/env bash
# Install (or remove) the Rekesz desktop entry and app icon for the current
# user, so Linux desktops show the icon in launchers and the taskbar.
#
# The window advertises the app id "rekesz" (see main.rs); desktops match it
# to rekesz.desktop, whose Icon= names the PNGs installed into the hicolor
# theme. Regenerate the PNGs with scripts/generate-app-icon.py.
#
# Usage: scripts/install-desktop-entry.sh [path/to/rekesz]
#        scripts/install-desktop-entry.sh --uninstall
#
# Without a path, Exec= runs `rekesz` from PATH, falling back to this
# checkout's target/release/rekesz.
set -euo pipefail
cd "$(dirname "$0")/.."

data_home="${XDG_DATA_HOME:-$HOME/.local/share}"
entry="$data_home/applications/rekesz.desktop"
icons="$data_home/icons/hicolor"
sizes=(16 22 24 32 48 64 96 128 256 512)

refresh_caches() {
    # Only refresh a GTK icon cache that already exists: creating one would
    # hide icons other programs install later without updating it.
    if [[ -f "$icons/icon-theme.cache" ]] && command -v gtk-update-icon-cache >/dev/null; then
        gtk-update-icon-cache --force --quiet --ignore-theme-index "$icons" || true
    fi
    # KDE resolves desktop entries through its sycoca cache.
    if command -v kbuildsycoca6 >/dev/null; then
        kbuildsycoca6 >/dev/null 2>&1 || true
    fi
}

if [[ "${1:-}" == "--uninstall" ]]; then
    rm -fv "$entry"
    for size in "${sizes[@]}"; do
        rm -fv "$icons/${size}x${size}/apps/rekesz.png"
    done
    refresh_caches
    exit 0
fi

binary="${1:-}"
if [[ -z "$binary" ]]; then
    if command -v rekesz >/dev/null; then
        exec_line="rekesz"
    elif [[ -x target/release/rekesz ]]; then
        binary="target/release/rekesz"
    else
        echo "error: rekesz is not on PATH and target/release/rekesz is missing;" \
            "run cargo build --release or pass the binary path" >&2
        exit 1
    fi
fi
if [[ -n "$binary" ]]; then
    binary="$(realpath "$binary")"
    if [[ ! -x "$binary" ]]; then
        echo "error: $binary is not an executable file" >&2
        exit 1
    fi
    # Desktop entries need escaping for quotes, $, `, \ and %; refuse those.
    if [[ ! "$binary" =~ ^[A-Za-z0-9/._+\ -]+$ ]]; then
        echo "error: unsupported characters in binary path: $binary" >&2
        exit 1
    fi
    exec_line="$binary"
    if [[ "$binary" == *" "* ]]; then
        exec_line="\"$binary\""
    fi
fi

mkdir -p "$(dirname "$entry")"
sed "s|^Exec=.*|Exec=$exec_line|" assets/linux/rekesz.desktop >"$entry.tmp"
mv "$entry.tmp" "$entry"
chmod 644 "$entry"
echo "installed $entry (Exec=$exec_line)"
for size in "${sizes[@]}"; do
    install -Dm644 "assets/icon/png/rekesz-$size.png" "$icons/${size}x${size}/apps/rekesz.png"
done
echo "installed icons into $icons (${sizes[*]} px)"

if command -v desktop-file-validate >/dev/null; then
    desktop-file-validate "$entry"
fi
refresh_caches
