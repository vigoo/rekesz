#!/usr/bin/env bash
# Screenshot tour driver: runs the app with the in-app tour harness
# (REKESZ_SHOT_TOUR=1), captures a screenshot of every settled surface with
# spectacle, and — for the "real" suite — proves the real catalog was never
# touched.
#
# Usage: scripts/screenshot-tour.sh [real|scratch]
#
#   real    The library under review (REKESZ_SHOT_REAL_ROOT, default
#           /mnt/pictures/Pictures). The catalog is copied to a scratch dir
#           via REKESZ_CATALOG_DIR, so the app never opens the real catalog;
#           import-copy actions are refused in-app; the script checksums the
#           real catalog before and after and fails loudly on any change.
#   scratch A disposable synthetic library for transient states (empty views,
#           banners, a real import job, an active job queue). It never
#           references the real library.
#
# The app window takes focus for the duration (~2-5 min per suite); do not
# steal focus mid-run or captures may show the wrong window.

set -uo pipefail

SUITE="${1:-real}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$REPO_ROOT/target/release/rekesz"
SEED="$REPO_ROOT/scripts/seed-shot-library.py"
REAL_ROOT="${REKESZ_SHOT_REAL_ROOT:-/mnt/pictures/Pictures}"
SCRATCH="$HOME/.cache/rekesz-shot-tour/$SUITE"
OUT_DIR="$REPO_ROOT/design/screenshots/$SUITE"
TOTAL_TIMEOUT=1500

die() { echo "FATAL: $*" >&2; exit 1; }
log() { echo "[tour] $*"; }

# --- Preconditions ----------------------------------------------------------

[[ $SUITE == real || $SUITE == scratch ]] || die "usage: screenshot-tour.sh [real|scratch]"
[[ -x $BIN ]] || die "missing $BIN — run: cargo build --release"
# The captures must reflect the current sources: build (cargo's own
# freshness check; ~1s when up to date).
( cd "$REPO_ROOT" && cargo build --release >/dev/null 2>&1 ) \
    || die "cargo build --release failed"
pgrep -x rekesz >/dev/null && die "rekesz is already running; close it first"
command -v spectacle >/dev/null || die "spectacle not found"
command -v python3 >/dev/null || die "python3 not found"
mkdir -p "$OUT_DIR"

real_catalog() { echo "$REAL_ROOT/.rekesz/catalog.sqlite3"; }

# The safety proof: main catalog checksum + full listing (incl. WAL sidecars)
# of the real .rekesz dir, snapshotted before and after the run.
snapshot_real_catalog() {
    local dir="$REAL_ROOT/.rekesz"
    sha256sum "$(real_catalog)" | cut -d' ' -f1
    ( cd "$dir" && stat -c '%n|%s|%Y|%i' * 2>/dev/null | sort )
}

if [[ $SUITE == real ]]; then
    [[ -d $REAL_ROOT ]] || die "real library root not found: $REAL_ROOT"
    [[ -f $(real_catalog) ]] || die "real catalog not found: $(real_catalog)"
    BEFORE=$(snapshot_real_catalog) || die "cannot read the real catalog"
fi

# --- Scratch setup ----------------------------------------------------------

rm -rf "$SCRATCH"
mkdir -p "$SCRATCH/config/rekesz" "$SCRATCH/cache"
LOG="$SCRATCH/tour.log"
python3 - "$SCRATCH/config/rekesz/settings.json" <<PYEOF
import json, pathlib, sys
scratch = pathlib.Path(sys.argv[1]).resolve().parents[2]  # <suite>/config/rekesz/settings.json -> <suite>
config = {
    "root": "$([[ $SUITE == real ]] && echo "$REAL_ROOT" || echo "$SCRATCH/library")",
    "image_cache_size": 512,
    "pair_preference": "jpeg",
    "cache_root": str(scratch / "cache"),
    "job_parallelism": 4,
    "raw_parallelism": 1,
    "automatic_face_analysis": False,
    "online_map_permission": "disabled",
    "import_recent_sources": [],
}
pathlib.Path(sys.argv[1]).write_text(json.dumps(config, indent=2) + "\n")
PYEOF

declare -a TOUR_ENV=(
    XDG_CONFIG_HOME="$SCRATCH/config"
    XDG_CACHE_HOME="$SCRATCH/cache"
    REKESZ_SHOT_TOUR=1
    REKESZ_SHOT_SUITE="$SUITE"
    REKESZ_TRACE=1
)

if [[ $SUITE == real ]]; then
    # Consistent catalog copy via sqlite3 .backup (single file, WAL-aware);
    # fall back to copying db+wal+shm while the app is confirmed not running.
    mkdir -p "$SCRATCH/catalog"
    if command -v sqlite3 >/dev/null; then
        sqlite3 "$(real_catalog)" ".backup '$SCRATCH/catalog/catalog.sqlite3'" \
            || die "sqlite3 .backup of the real catalog failed"
    else
        cp -a "$REAL_ROOT/.rekesz/"catalog.sqlite3* "$SCRATCH/catalog/" \
            || die "copying the real catalog failed"
        [[ -f $SCRATCH/catalog/catalog.sqlite3 ]] || die "catalog copy missing"
    fi
    [[ -s $SCRATCH/catalog/catalog.sqlite3 ]] || die "catalog copy is empty"

    # Demo directory: the one with the most develop recipes (fallback: most
    # images two levels deep), so develop-mode captures have real content.
    DEMO_DIR=$(python3 - "$SCRATCH/catalog/catalog.sqlite3" "$REAL_ROOT" <<'PYEOF'
import json, pathlib, sqlite3, sys
db, root = sys.argv[1], pathlib.Path(sys.argv[2])
def decode(blob):
    # Catalog relative paths are 'p' + JSON component array (portable) or
    # 'u' + raw unix bytes.
    data = bytes(blob)
    if data[:1] == b"p":
        return "/".join(json.loads(data[1:].decode("utf-8")))
    if data[:1] == b"u":
        data = data[1:]
    return data.decode("utf-8", "replace")
try:
    connection = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    counts = {}
    for (blob,) in connection.execute(
        "SELECT assets.relative_path FROM develop_recipes JOIN assets ON assets.id=develop_recipes.asset_id"
    ):
        rel = decode(blob)
        parent = str(pathlib.PurePosixPath(rel).parent)
        counts[parent] = counts.get(parent, 0) + 1
    if counts:
        top = max(counts.items(), key=lambda item: item[1])[0]
        print(root / top)
except Exception as error:
    print(f"catalog demo-dir heuristic failed: {error}", file=sys.stderr)
PYEOF
    )
    if [[ -z $DEMO_DIR || ! -d $DEMO_DIR ]]; then
        log "catalog heuristic failed; falling back to an image-count scan"
        DEMO_DIR=""
        best=0
        while IFS= read -r candidate; do
            count=$(find "$candidate" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.cr3' -o -iname '*.nef' -o -iname '*.arw' \) 2>/dev/null | wc -l)
            if (( count > best )); then best=$count; DEMO_DIR=$candidate; fi
        done < <(find "$REAL_ROOT" -maxdepth 2 -type d 2>/dev/null)
    fi
    [[ -n $DEMO_DIR && -d $DEMO_DIR ]] || die "no demo directory found under $REAL_ROOT"
    log "demo directory: $DEMO_DIR"
    TOUR_ENV+=(
        REKESZ_CATALOG_DIR="$SCRATCH/catalog"
        REKESZ_SHOT_DIR="$DEMO_DIR"
    )
    # The import listing shows the demo directory as a folder source (its
    # photos are already in the library, so Duplicate badges render).
    python3 - "$SCRATCH/config/rekesz/settings.json" "$DEMO_DIR" <<'PYEOF'
import json, pathlib, sys
path, demo = pathlib.Path(sys.argv[1]), sys.argv[2]
config = json.loads(path.read_text())
config["import_recent_sources"] = [demo]
path.write_text(json.dumps(config, indent=2) + "\n")
PYEOF
else
    python3 "$SEED" "$SCRATCH/library" "$SCRATCH/import-source" "$SCRATCH/library/Imported" \
        || die "seeding the scratch library failed"
    TOUR_ENV+=(
        REKESZ_SHOT_EMPTY_DIR="$SCRATCH/library/Empty Album"
        REKESZ_SHOT_DEST="$SCRATCH/library/Imported"
    )
    python3 - "$SCRATCH/config/rekesz/settings.json" "$SCRATCH/import-source" <<'PYEOF'
import json, pathlib, sys
path, source = pathlib.Path(sys.argv[1]), sys.argv[2]
config = json.loads(path.read_text())
config["import_recent_sources"] = [source]
path.write_text(json.dumps(config, indent=2) + "\n")
PYEOF
fi

# --- Launch + capture loop --------------------------------------------------

log "launching rekesz (suite=$SUITE) — keep this window focused"
coproc REKESZ { env "${TOUR_ENV[@]}" "$BIN" "$([[ $SUITE == real ]] && echo "$REAL_ROOT" || echo "$SCRATCH/library")" 2>"$LOG"; }

exec {OUT_FD}<&"${REKESZ[0]}"
exec {IN_FD}>&"${REKESZ[1]}"

manifest="$OUT_DIR/manifest.tsv"
: > "$manifest"
captured=0
skipped=0
done_seen=0

cleanup() {
    kill "$REKESZ_PID" 2>/dev/null
    wait "$REKESZ_PID" 2>/dev/null
}
trap cleanup EXIT

while IFS= read -r -t "$TOTAL_TIMEOUT" line <&"$OUT_FD"; do
    case "$line" in
        REKESZ_SHOT_SKIP:*)
            id="${line#REKESZ_SHOT_SKIP:}"
            printf '%s\t%s\t%s\n' "$id" "skipped" "-" >> "$manifest"
            skipped=$((skipped + 1))
            log "skip  $id"
            printf 'NEXT\n' >&"$IN_FD"
            ;;
        REKESZ_SHOT_READY:*)
            id="${line#REKESZ_SHOT_READY:}"
            # Next line carries the settle hint in milliseconds.
            IFS= read -r -t 10 settle_line <&"$OUT_FD"
            settle_ms=1500
            [[ $settle_line == REKESZ_SHOT_SETTLE:* ]] && settle_ms="${settle_line#REKESZ_SHOT_SETTLE:}"
            settle_s=$(python3 -c "print(max(0.4, $settle_ms / 1000.0))")
            sleep "$settle_s"
            file="$OUT_DIR/$id.png"
            spectacle -b -a -n -o "$file" >/dev/null 2>&1
            if [[ ! -s $file ]]; then
                sleep 0.7
                spectacle -b -a -n -o "$file" >/dev/null 2>&1
            fi
            if [[ -s $file ]]; then
                printf '%s\t%s\t%s\n' "$id" "captured" "$file" >> "$manifest"
                captured=$((captured + 1))
                log "shot  $id"
            else
                printf '%s\t%s\t%s\n' "$id" "failed" "-" >> "$manifest"
                log "MISS  $id"
            fi
            printf 'NEXT\n' >&"$IN_FD"
            ;;
        REKESZ_SHOT_DONE)
            done_seen=1
            log "tour finished"
            break
            ;;
    esac
done

if (( done_seen == 0 )); then
    log "warning: marker stream ended without REKESZ_SHOT_DONE — check $LOG"
    tail -5 "$LOG" 2>/dev/null || true
fi

# --- Safety proof -----------------------------------------------------------

if [[ $SUITE == real ]]; then
    AFTER=$(snapshot_real_catalog) || die "cannot re-read the real catalog"
    if [[ $BEFORE != "$AFTER" ]]; then
        diff <(echo "$BEFORE") <(echo "$AFTER") | head -20 >&2
        die "THE REAL CATALOG CHANGED DURING THE RUN — investigate before continuing"
    fi
    log "verified: real catalog untouched ($(echo "$BEFORE" | head -1 | cut -c1-16)… checksum stable)"
fi

log "done: $captured captured, $skipped skipped — manifest: $manifest"
