#!/usr/bin/env bash

enforce_people_rss_resource_file() {
    local resources="$1"
    local budget_kib="$2"
    local measured_rss_kib
    measured_rss_kib="$(awk -F= '$1 == "max_rss_kib" { print $2 }' "$resources")"
    if [[ ! "$measured_rss_kib" =~ ^[0-9]+$ ]]; then
        printf 'resource report produced no valid max_rss_kib value\n' >&2
        return 1
    fi
    if [[ ! "$budget_kib" =~ ^[0-9]+$ ]] || (( measured_rss_kib > budget_kib )); then
        printf 'RSS budget exceeded: %s KiB > %s KiB\n' "$measured_rss_kib" "$budget_kib" >&2
        return 1
    fi
}

run_people_resource_monitored() {
    local resources="$1"
    local log="$2"
    local budget_kib="$3"
    shift 3
    "${PEOPLE_RESOURCE_SUPERVISOR:?set PEOPLE_RESOURCE_SUPERVISOR}" \
        "$resources" "$log" "$budget_kib" "${PEOPLE_MAX_CPU_PERCENT:-400}" -- "$@"
}

write_people_test_artifact_manifest() {
    local output="$1"
    shift
    local build_input_sha256
    local status_sha256
    build_input_sha256="$({
        git diff --binary
        git diff --cached --binary
        git ls-files --others --exclude-standard -z | sort -z | xargs -0 -r sha256sum
    } | sha256sum | cut -d' ' -f1)"
    status_sha256="$(git status --porcelain=v1 -z | sha256sum | cut -d' ' -f1)"
    {
        printf 'revision=%s\n' "$(git rev-parse HEAD)"
        printf 'build_input_sha256=%s\n' "$build_input_sha256"
        printf 'worktree_status_sha256=%s\n' "$status_sha256"
        find "$@" -maxdepth 1 -type f -executable -print0 | sort -z | xargs -0 -r sha256sum
        find "$@" -maxdepth 1 -type f -executable -printf '%T@ %p\n' | sort
    } >"$output"
}

write_people_mapped_test_artifact_manifest() {
    local output="$1"
    shift
    local paths=()
    local mapping
    for mapping in "$@"; do
        while IFS='=' read -r _ path; do
            [[ -n "$path" ]] && paths+=("$path")
        done <"$mapping"
    done
    local build_input_sha256
    build_input_sha256="$({
        git diff --binary
        git diff --cached --binary
        git ls-files --others --exclude-standard -z | sort -z | xargs -0 -r sha256sum
    } | sha256sum | cut -d' ' -f1)"
    {
        printf 'revision=%s\n' "$(git rev-parse HEAD)"
        printf 'build_input_sha256=%s\n' "$build_input_sha256"
        printf 'worktree_status_sha256=%s\n' "$(git status --porcelain=v1 -z | sha256sum | cut -d' ' -f1)"
        for path in "${paths[@]}"; do
            [[ -x "$path" ]] || return 1
            sha256sum "$path"
            stat -c '%Y %n' "$path"
        done
    } >"$output"
}

run_people_mapped_test_binaries() {
    local mapping="$1"
    local marker_dir="$2"
    local name path
    while IFS='=' read -r name path; do
        [[ -n "$name" && -x "$path" ]] || return 1
        "$path" || return $?
        : >"$marker_dir/$name.executed"
    done <"$mapping"
}

resolve_people_exact_test_name() {
    local binary="$1"
    local suffix="$2"
    local listing
    listing="$(mktemp)"
    "$binary" --list >"$listing" || {
        rm -f "$listing"
        return 1
    }
    local matches=()
    mapfile -t matches < <(awk -v suffix="$suffix" '
        $0 == suffix || (length($0) > length(suffix) + 1 && substr($0, length($0) - length(suffix) - 1) == "::" suffix) { print }
    ' "$listing")
    rm -f "$listing"
    if (( ${#matches[@]} != 1 )); then
        printf 'test suffix must resolve exactly once: %s (matches=%s)\n' "$suffix" "${#matches[@]}" >&2
        return 1
    fi
    # test-r's list display prefixes the binary target (`rekesz::`), while its
    # `--exact` matcher uses the canonical in-binary name after that prefix.
    printf '%s\n' "${matches[0]#*::}"
}

run_people_exact_test() {
    local binary="$1"
    local suffix="$2"
    local include_ignored="${3:-false}"
    local full_name
    full_name="$(resolve_people_exact_test_name "$binary" "$suffix")" || return
    local args=(--exact "$full_name" --test-threads=1 --nocapture)
    if [[ "$include_ignored" == true ]]; then
        args+=(--include-ignored)
    fi
    "$binary" "${args[@]}"
}
