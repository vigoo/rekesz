#!/usr/bin/env bash
set -euo pipefail

artifact_dir="${1:?usage: check-people-artifact-privacy.sh ARTIFACT_DIRECTORY}"
report="$artifact_dir/reference-resources.json"
reference_metadata="$artifact_dir/reference.txt"
entry_list="$(mktemp 2>/dev/null)" || {
    printf 'privacy gate: could not allocate traversal state\n' >&2
    exit 1
}
trap 'rm -f "$entry_list" >/dev/null 2>&1 || true' EXIT

if [[ -L "$artifact_dir" || ! -d "$artifact_dir" ]]; then
    printf 'privacy gate: artifact root must be a real directory\n' >&2
    exit 1
fi
if [[ ! -f "$report" ]]; then
    printf 'privacy gate: missing aggregate resource report\n' >&2
    exit 1
fi
if [[ -L "$reference_metadata" || ! -f "$reference_metadata" ]]; then
    printf 'privacy gate: missing or unsupported reference metadata\n' >&2
    exit 1
fi

validate_build_mapping() {
    local artifact="$1"
    local profile="$2"
    [[ -s "$artifact" ]] || return 1
    awk '
        !/^(lightroom-inventory|rekesz)=target\/(debug|release)(\/[A-Za-z0-9_-]+)+$/ { invalid=1; next }
        {
            split($0, fields, "=")
            key=tolower(fields[1])
            seen[key]++
        }
        END { exit !invalid && seen["lightroom-inventory"] == 1 && seen["rekesz"] == 1 ? 0 : 1 }
    ' "$artifact" && awk -F= -v profile="$profile" '$2 !~ ("^target/" profile "/") { exit 1 }' "$artifact"
}

validate_test_artifact_manifest() {
    local artifact="$1"
    [[ -s "$artifact" ]] || return 1
    awk '
        /^revision=[0-9a-f]+$/ && length($0) == 49 { revision++; next }
        /^build_input_sha256=[0-9a-f]+$/ && length($0) == 83 { input++; next }
        /^worktree_status_sha256=[0-9a-f]+$/ && length($0) == 87 { status++; next }
        /^[0-9a-f]+  target\/(debug|release)(\/[A-Za-z0-9_-]+)+$/ && index($0, "  target/") == 65 {
            path=substr($0, 67)
            digests[path]++
            next
        }
        /^[0-9]+ target\/(debug|release)(\/[A-Za-z0-9_-]+)+$/ {
            path=$0
            sub(/^[0-9]+ /, "", path)
            timestamps[path]++
            next
        }
        { invalid=1 }
        END {
            paths=0
            for (path in digests) {
                paths++
                if (digests[path] != 1 || timestamps[path] != 1) invalid=1
            }
            for (path in timestamps) if (digests[path] != 1 || timestamps[path] != 1) invalid=1
            exit !invalid && revision == 1 && input == 1 && status == 1 && paths > 0 ? 0 : 1
        }
    ' "$artifact"
}

validate_resource_report() {
    python3 -c '
import json, math, re, sys
required = {
 "schema", "reference_tag", "face_count", "wall_seconds", "cpu_percent", "peak_rss_kib",
 "cancellation_latency_ms", "running_jobs_observed", "queued_jobs_observed",
 "inference_started", "inference_completed", "inference_overlapped_people_workload",
 "people_workload_started_at_inference", "people_workload_finished_at_inference",
 "people_workload_in_progress_at_inference", "durable_pending_after_cancel",
 "partial_faces_after_cancel", "partial_embeddings_after_cancel", "partial_landmarks_after_cancel",
 "thresholds"
}
thresholds = {"max_cpu_percent", "max_rss_kib", "max_cancellation_latency_ms"}
integer_fields = {"schema", "face_count", "running_jobs_observed", "queued_jobs_observed", "inference_started", "inference_completed", "people_workload_in_progress_at_inference", "durable_pending_after_cancel", "partial_faces_after_cancel", "partial_embeddings_after_cancel", "partial_landmarks_after_cancel"}
number_fields = {"wall_seconds", "cpu_percent", "peak_rss_kib", "cancellation_latency_ms"}
boolean_fields = {"inference_overlapped_people_workload", "people_workload_started_at_inference", "people_workload_finished_at_inference"}
try:
 data = json.load(open(sys.argv[1], encoding="utf-8"))
 ok = isinstance(data, dict) and set(data) == required
 ok = ok and data.get("schema") == 1 and not isinstance(data.get("schema"), bool)
 ok = ok and isinstance(data.get("reference_tag"), str) and re.fullmatch(r"[A-Za-z0-9._-]+", data["reference_tag"]) is not None
 ok = ok and all(isinstance(data.get(k), int) and not isinstance(data.get(k), bool) and data[k] >= 0 for k in integer_fields)
 ok = ok and all(isinstance(data.get(k), (int, float)) and not isinstance(data.get(k), bool) and math.isfinite(data[k]) and data[k] >= 0 for k in number_fields)
 ok = ok and all(isinstance(data.get(k), bool) for k in boolean_fields)
 value = data.get("thresholds")
 ok = ok and isinstance(value, dict) and set(value) == thresholds
 ok = ok and all(isinstance(value.get(k), (int, float)) and not isinstance(value.get(k), bool) and math.isfinite(value[k]) and value[k] >= 0 for k in thresholds)
except (OSError, UnicodeError, json.JSONDecodeError):
 ok = False
sys.exit(0 if ok else 1)
' "$1"
}

validate_reference_metadata() {
    awk -F= '
        !/^[a-z0-9_]+=[^[:cntrl:]]*$/ { invalid=1; next }
        {
            allowed=" reference_tag revision worktree_dirty worktree_diff_sha256 worktree_status_sha256 kernel cpu_count max_rss_kib max_cpu_percent max_cancel_latency_ms "
            if (index(allowed, " " $1 " ") == 0) invalid=1
            seen[$1]++
        }
        $1 == "reference_tag" && $2 !~ /^[A-Za-z0-9._-]+$/ { invalid=1 }
        $1 == "revision" && $2 !~ /^[0-9a-f]{40}$/ { invalid=1 }
        ($1 == "worktree_diff_sha256" || $1 == "worktree_status_sha256") && $2 !~ /^[0-9a-f]{64}$/ { invalid=1 }
        $1 == "worktree_dirty" && $2 !~ /^(true|false)$/ { invalid=1 }
        $1 == "kernel" && $2 == "" { invalid=1 }
        $1 == "cpu_count" && ($2 !~ /^[0-9]+$/ || $2 + 0 < 1) { invalid=1 }
        $1 == "max_rss_kib" && ($2 !~ /^[0-9]+$/ || $2 + 0 < 1) { invalid=1 }
        ($1 == "max_cpu_percent" || $1 == "max_cancel_latency_ms") && ($2 !~ /^[0-9]+([.][0-9]+)?$/ || $2 + 0 <= 0) { invalid=1 }
        END {
            required="reference_tag revision worktree_dirty worktree_diff_sha256 worktree_status_sha256 kernel cpu_count max_rss_kib max_cpu_percent max_cancel_latency_ms"
            count=split(required, keys, " ")
            for (i=1; i<=count; i++) if (seen[keys[i]] != 1) invalid=1
            exit invalid ? 1 : 0
        }
    ' "$1"
}

validate_cross_artifact_evidence() {
    python3 -c '
import json, sys
from decimal import Decimal, InvalidOperation

try:
 reference = dict(line.rstrip("\n").split("=", 1) for line in open(sys.argv[1], encoding="utf-8"))
 report = json.load(open(sys.argv[2], encoding="utf-8"))
 ok = reference["reference_tag"] == report["reference_tag"]
 for reference_key, report_key in (
     ("max_rss_kib", "max_rss_kib"),
     ("max_cpu_percent", "max_cpu_percent"),
     ("max_cancel_latency_ms", "max_cancellation_latency_ms"),
 ):
     ok = ok and Decimal(reference[reference_key]) == Decimal(str(report["thresholds"][report_key]))
 if len(sys.argv) == 4:
     manifest = dict(line.rstrip("\n").split("=", 1) for line in open(sys.argv[3], encoding="utf-8") if "=" in line)
     ok = ok and reference["revision"] == manifest["revision"]
     ok = ok and reference["worktree_status_sha256"] == manifest["worktree_status_sha256"]
except (OSError, UnicodeError, ValueError, KeyError, TypeError, InvalidOperation, json.JSONDecodeError):
 ok = False
sys.exit(0 if ok else 1)
' "$@"
}

if ! find "$artifact_dir" -mindepth 1 ! -type d -print0 >"$entry_list" 2>/dev/null; then
    printf 'privacy gate: artifact traversal failed\n' >&2
    exit 1
fi

while IFS= read -r -d '' entry; do
    if [[ -L "$entry" || ! -f "$entry" ]]; then
        printf 'privacy gate: unsupported artifact entry found\n' >&2
        exit 1
    fi
    case "$entry" in
        "$artifact_dir/debug-test-executables.txt")
            if ! validate_build_mapping "$entry" debug 2>/dev/null; then
                printf 'privacy gate: malformed build mapping\n' >&2
                exit 1
            fi
            continue
            ;;
        "$artifact_dir/release-test-executables.txt")
            if ! validate_build_mapping "$entry" release 2>/dev/null; then
                printf 'privacy gate: malformed build mapping\n' >&2
                exit 1
            fi
            continue
            ;;
        "$artifact_dir/test-artifacts.before.txt" | \
        "$artifact_dir/test-artifacts.current.txt" | \
        "$artifact_dir/test-artifacts.after.txt")
            if ! validate_test_artifact_manifest "$entry" 2>/dev/null; then
                printf 'privacy gate: malformed test artifact manifest\n' >&2
                exit 1
            fi
            continue
            ;;
    esac
    case "$entry" in
        "$report")
            if ! validate_resource_report "$entry" 2>/dev/null; then
                printf 'privacy gate: aggregate resource report violates its schema\n' >&2
                exit 1
            fi
            ;;
        "$artifact_dir/reference.txt")
            if ! validate_reference_metadata "$entry" 2>/dev/null; then
                printf 'privacy gate: reference metadata violates its schema\n' >&2
                exit 1
            fi
            ;;
    esac
    relative="${entry#"$artifact_dir"/}"
    case "$relative" in
        reference-resources.json | reference.txt | \
        *.log | *.resources.txt | *.build.log | *.build-resources.txt) ;;
        *) printf 'privacy gate: unapproved retained artifact type\n' >&2; exit 1 ;;
    esac
    case "${entry,,}" in
        *.jpg | *.jpeg | *.png | *.gif | *.bmp | *.tif | *.tiff | *.webp | *.avif | \
        *.heic | *.heif | *.dng | *.cr2 | *.cr3 | *.nef | *.arw | *.raf | *.rw2 | \
        *.orf | *.pef | *.x3f | *.raw | *.zip | *.gz | *.bz2 | *.xz | *.zst | *.tar)
            printf 'privacy gate: source-media or container artifact found\n' >&2
            exit 1
            ;;
    esac
    if ! magic="$(LC_ALL=C od -An -tx1 -N16 "$entry" 2>/dev/null | tr -d ' \n')"; then
        printf 'privacy gate: artifact header audit failed\n' >&2
        exit 1
    fi
    case "$magic" in
        25504446* | 1f8b* | 425a68* | fd377a585a00* | 28b52ffd* | \
        504b0304* | 504b0506* | 504b0708* | 89504e47* | ffd8ff* | 47494638* | \
        424d* | 49492a00* | 4d4d002a* | 377abcaf271c* | 52617221* | \
        53514c69746520666f726d6174203300* | 1a45dfa3* | 464c56* | 4f676753* | \
        000001ba* | 000001b3*)
            printf 'privacy gate: disguised binary/container artifact found\n' >&2
            exit 1
            ;;
    esac
    if [[ "${magic:8:8}" == 66747970 || "$magic" == 52494646* ]]; then
        printf 'privacy gate: disguised media/container artifact found\n' >&2
        exit 1
    fi
    if ! entry_size="$(stat -c %s "$entry" 2>/dev/null)" || [[ ! "$entry_size" =~ ^[0-9]+$ ]]; then
        printf 'privacy gate: artifact size audit failed\n' >&2
        exit 1
    fi
    tar_marker=""
    if (( entry_size >= 262 )); then
        if ! tar_marker="$(LC_ALL=C od -An -tc -j257 -N5 "$entry" 2>/dev/null | tr -d ' \n')"; then
            printf 'privacy gate: artifact container audit failed\n' >&2
            exit 1
        fi
    fi
    if (( entry_size >= 262 )) && [[ "$tar_marker" == ustar ]]; then
        printf 'privacy gate: disguised tar artifact found\n' >&2
        exit 1
    fi
    if ! nul_count="$(LC_ALL=C tr -cd '\000' <"$entry" 2>/dev/null | wc -c)"; then
        printf 'privacy gate: artifact encoding audit failed\n' >&2
        exit 1
    fi
    if ! iconv -f UTF-8 -t UTF-8 "$entry" >/dev/null 2>&1 || (( nul_count != 0 )); then
        printf 'privacy gate: non-UTF-8 or binary artifact cannot be audited\n' >&2
        exit 1
    fi
    if rg --ignore-case --quiet \
        '(^|["{,[:space:]])((original|file|asset|photo|source)[_ -]?(path|filename|name)|path|name|identity|user[_ -]?name|person[_ -]?(id|name)|subject[_ -]?(id|name)|(face[_ -]?)?embedding|(feature[_ -]?)?vector|(face[_ -]?)?landmarks|crop|face[_ -]?(crop|image|bytes)|image[_ -]?bytes)["[:space:]]*[=:]|(original|file|asset|photo|source|person|subject|face|user|identity)[_ -]+(path|filename|name|identifier|embedding|landmarks)[[:space:]]' \
        "$entry" >/dev/null 2>&1; then
        printf 'privacy gate: biometric or identifying content found\n' >&2
        exit 1
    else
        rg_status=$?
        if (( rg_status != 1 )); then
            printf 'privacy gate: identifying-content audit failed\n' >&2
            exit 1
        fi
    fi
    if rg --ignore-case --quiet \
        '((^|[^[:alnum:]_.:/-])/{1,}[^\x00-\x20"'"'"'(),;]+|file://|[A-Za-z]:\\\\|[^[:space:]"'"'"']+\.(jpe?g|png|gif|bmp|tiff?|webp|avif|heic|heif|dng|cr[23]|nef|arw|raf|rw2|orf|pef|x3f|raw)([[:space:]"'"'"']|$))' \
        "$entry" >/dev/null 2>&1; then
        printf 'privacy gate: filesystem or source-image identifier found\n' >&2
        exit 1
    else
        rg_status=$?
        if (( rg_status != 1 )); then
            printf 'privacy gate: filesystem-identifier audit failed\n' >&2
            exit 1
        fi
    fi
done <"$entry_list"

manifest_count=0
for manifest in test-artifacts.before.txt test-artifacts.current.txt test-artifacts.after.txt; do
    [[ -e "$artifact_dir/$manifest" ]] && ((manifest_count += 1))
done
if (( manifest_count != 0 )); then
    if (( manifest_count != 3 )) || \
        ! cmp -s "$artifact_dir/test-artifacts.before.txt" "$artifact_dir/test-artifacts.current.txt" || \
        ! cmp -s "$artifact_dir/test-artifacts.before.txt" "$artifact_dir/test-artifacts.after.txt"; then
        printf 'privacy gate: build evidence manifests are incomplete or inconsistent\n' >&2
        exit 1
    fi
fi

cross_evidence=("$reference_metadata" "$report")
if (( manifest_count == 3 )); then
    cross_evidence+=("$artifact_dir/test-artifacts.before.txt")
fi
if ! validate_cross_artifact_evidence "${cross_evidence[@]}" 2>/dev/null; then
    printf 'privacy gate: retained evidence is internally inconsistent\n' >&2
    exit 1
fi

printf 'privacy gate: approved UTF-8 release text formats passed aggregate-field checks; build manifests passed strict schemas\n'
