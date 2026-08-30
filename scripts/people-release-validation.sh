#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/people-resource-limit.sh"

: "${PEOPLE_REFERENCE_TAG:?set PEOPLE_REFERENCE_TAG to the reviewed host/configuration tag}"

artifact_dir="${PEOPLE_RELEASE_ARTIFACT_DIR:-target/people-release-validation}"
artifact_dir="$(./scripts/prepare-people-release-artifacts.sh "$artifact_dir")"

finalize_privacy() {
    local release_status=$?
    trap - EXIT
    local privacy_status=0
    ./scripts/check-people-artifact-privacy.sh "$artifact_dir" || privacy_status=$?
    if (( release_status != 0 )); then
        exit "$release_status"
    fi
    exit "$privacy_status"
}
trap finalize_privacy EXIT

if [[ -z "$(git status --porcelain=v1)" ]]; then
    worktree_dirty=false
else
    worktree_dirty=true
fi
worktree_diff_sha256="$({ git diff --binary; git diff --cached --binary; } | sha256sum | cut -d' ' -f1)"
worktree_status_sha256="$(git status --porcelain=v1 -z | sha256sum | cut -d' ' -f1)"

export PEOPLE_MAX_RSS_KIB="${PEOPLE_MAX_RSS_KIB:-2097152}"
export PEOPLE_MAX_CPU_PERCENT="${PEOPLE_MAX_CPU_PERCENT:-400}"
export PEOPLE_MAX_CANCEL_LATENCY_MS="${PEOPLE_MAX_CANCEL_LATENCY_MS:-100}"
export PEOPLE_RESOURCE_ARTIFACT="$artifact_dir/reference-resources.json"

{
    printf 'reference_tag=%s\n' "$PEOPLE_REFERENCE_TAG"
    printf 'revision=%s\n' "$(git rev-parse HEAD)"
    printf 'worktree_dirty=%s\n' "$worktree_dirty"
    printf 'worktree_diff_sha256=%s\n' "$worktree_diff_sha256"
    printf 'worktree_status_sha256=%s\n' "$worktree_status_sha256"
    printf 'kernel=%s\n' "$(uname -srmo)"
    printf 'cpu_count=%s\n' "$(getconf _NPROCESSORS_ONLN)"
    printf 'max_rss_kib=%s\n' "$PEOPLE_MAX_RSS_KIB"
    printf 'max_cpu_percent=%s\n' "$PEOPLE_MAX_CPU_PERCENT"
    printf 'max_cancel_latency_ms=%s\n' "$PEOPLE_MAX_CANCEL_LATENCY_MS"
} >"$artifact_dir/reference.txt"

# Build/quality work is recorded separately and is deliberately not described as
# application runtime. Rust compiler/linker resource use is host/toolchain input.
cargo build --release --bin people-resource-supervisor --bin people-test-artifacts \
    --bin sface-calibration-embeddings
export PEOPLE_RESOURCE_SUPERVISOR="${CARGO_TARGET_DIR:-target}/release/people-resource-supervisor"
artifact_mapper="${CARGO_TARGET_DIR:-target}/release/people-test-artifacts"
export SFACE_CALIBRATION_EXTRACTOR="${CARGO_TARGET_DIR:-target}/release/sface-calibration-embeddings"

run_build_gate() {
    local gate="$1"
    shift
    PEOPLE_MAX_CPU_PERCENT=100000 run_people_resource_monitored \
        "$artifact_dir/${gate}.build-resources.txt" "$artifact_dir/${gate}.build.log" \
        9223372036854775807 "$@"
}

run_build_gate quality-fmt cargo fmt --all -- --check
run_build_gate quality-clippy cargo clippy --all-targets --all-features -- -D warnings
debug_mapping="$artifact_dir/debug-test-executables.txt"
release_mapping="$artifact_dir/release-test-executables.txt"
run_build_gate compile-debug-tests /bin/bash -o pipefail -c \
    'cargo test --quiet --no-run --message-format=json | "$1" >"$2"' \
    people-test-discovery "$artifact_mapper" "$debug_mapping"
run_build_gate compile-release-tests /bin/bash -o pipefail -c \
    'cargo test --quiet --release --no-run --message-format=json | "$1" >"$2"' \
    people-test-discovery "$artifact_mapper" "$release_mapping"

normalize_test_mapping() {
    local mapping="$1"
    local normalized="$mapping.relative"
    : >"$normalized"
    while IFS='=' read -r name path; do
        printf '%s=%s\n' "$name" "$(realpath --relative-to="$PWD" "$path")" >>"$normalized"
    done <"$mapping"
    mv "$normalized" "$mapping"
}
normalize_test_mapping "$debug_mapping"
normalize_test_mapping "$release_mapping"

mapped_test_path() {
    local mapping="$1"
    local name="$2"
    awk -F= -v name="$name" '$1 == name { print substr($0, index($0, "=") + 1) }' "$mapping"
}

debug_lightroom="$(mapped_test_path "$debug_mapping" lightroom-inventory)"
debug_rekesz="$(mapped_test_path "$debug_mapping" rekesz)"
release_rekesz="$(mapped_test_path "$release_mapping" rekesz)"
[[ -x "$debug_lightroom" && -x "$debug_rekesz" && -x "$release_rekesz" ]]

test_artifact_manifest() {
    local output="$1"
    write_people_mapped_test_artifact_manifest "$output" "$debug_mapping" "$release_mapping"
    sha256sum "$SFACE_CALIBRATION_EXTRACTOR" >>"$output"
    stat -c '%Y %n' "$SFACE_CALIBRATION_EXTRACTOR" >>"$output"
}

test_artifact_manifest "$artifact_dir/test-artifacts.before.txt"

run_gate() {
    local gate="$1"
    shift
    local resources="$artifact_dir/${gate}.resources.txt"
    run_people_resource_monitored \
        "$resources" "$artifact_dir/${gate}.log" "$PEOPLE_MAX_RSS_KIB" "$@"
    if grep -Eq '(^|[[:space:]])Compiling[[:space:]]' "$artifact_dir/${gate}.log"; then
        printf 'runtime gate unexpectedly compiled code: %s\n' "$gate" >&2
        return 1
    fi
    test_artifact_manifest "$artifact_dir/test-artifacts.current.txt"
    cmp "$artifact_dir/test-artifacts.before.txt" "$artifact_dir/test-artifacts.current.txt"
}

run_exact_gate() {
    local gate="$1"
    local suffix="$2"
    local include_ignored="${3:-false}"
    local profile_stages="${4:-false}"
    if [[ "$profile_stages" == true ]]; then
        run_gate "$gate" env REKESZ_PEOPLE_PROFILE_STAGES=1 /bin/bash -c \
            'source "$1"; run_people_exact_test "$2" "$3" "$4"' \
            people-exact-test "$(dirname "$0")/people-resource-limit.sh" \
            "$release_rekesz" "$suffix" "$include_ignored"
        return
    fi
    run_gate "$gate" /bin/bash -c \
        'source "$1"; run_people_exact_test "$2" "$3" "$4"' \
        people-exact-test "$(dirname "$0")/people-resource-limit.sh" \
        "$release_rekesz" "$suffix" "$include_ignored"
}

# Direct prebuilt test-r binaries require --nocapture so they do not activate
# the Cargo-coordinated IPC capture worker. Adversarial helper diagnostics are
# operand-free, so the complete retained stream remains privacy-auditable.
run_gate quality-tests-lightroom "$debug_lightroom" --test-threads=1 --nocapture
run_gate quality-tests-rekesz "$debug_rekesz" --test-threads=1 --nocapture
run_gate recognition-calibration-report python3 scripts/verify-sface-policy-calibration.py
run_exact_gate offline-inference real_detect_and_embed_run_with_socket_syscalls_denied
run_exact_gate offline-dependency-audit face_engine_has_no_network_or_map_dependency
run_exact_gate migration-recovery-markers migration_rejects_disagreeing_version_markers_without_schema_changes
run_exact_gate migration-recovery-v2 v2_catalog_migrates_to_v3_with_paths_and_effective_views
run_exact_gate migration-recovery-v1 v1_catalog_migrates_to_current_with_path_backup_and_ann_schema
run_exact_gate migration-recovery-v3 v3_recognition_migration_rolls_back_all_columns_on_failure
run_exact_gate recognition-ann-correctness real_dimension_multi_query_is_exact_and_metric_prunes_ann_candidates
run_exact_gate recognition-ann-resource ann_resource_512d_cached_generation_overlap_stays_within_budget
run_exact_gate recognition-ann-catalog-ceiling ann_release_100k_512d_cached_overlap_throughput_and_resources true
run_exact_gate recognition-ann-cancellation index_waiter_cancels_within_one_hundred_ms_and_other_catalog_progresses
run_exact_gate recognition-ann-contention-cancellation model_build_contention_cancels_within_one_hundred_ms_without_blocking_other_keys
run_exact_gate gpui-people-performance people_gpui_reference_performance true
run_exact_gate reference-resource reference_people_resource_profile true
run_exact_gate catalog-60k generated_people_catalog_profile_60k true true
run_exact_gate catalog-250k generated_people_catalog_profile_250k true true
run_exact_gate long-soak long_reimport_restart_cancel_soak true
test_artifact_manifest "$artifact_dir/test-artifacts.after.txt"
cmp "$artifact_dir/test-artifacts.before.txt" "$artifact_dir/test-artifacts.after.txt"
./scripts/check-people-artifact-privacy.sh "$artifact_dir"
trap - EXIT
