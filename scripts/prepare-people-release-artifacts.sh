#!/usr/bin/env bash
set -euo pipefail

requested="${1:?usage: prepare-people-release-artifacts.sh ARTIFACT_DIRECTORY}"
repo_root="$(pwd -P)"
target_root="$repo_root/target"

case "$requested" in
    /*) candidate="$requested" ;;
    *) candidate="$repo_root/$requested" ;;
esac
if [[ "$requested" == *'/../'* || "$requested" == ../* || "$requested" == */.. || \
      "$requested" == *'/./'* || "$requested" == ./* || "$requested" == */. ]]; then
    printf 'refusing aliased release artifact directory\n' >&2
    exit 1
fi
candidate="${candidate%/}"
case "$candidate" in
    "$target_root"/*) ;;
    *) printf 'release artifact directory must be a dedicated target descendant\n' >&2; exit 1 ;;
esac

if ! mkdir -p "$target_root" 2>/dev/null; then
    printf 'release target root could not be prepared\n' >&2
    exit 1
fi
if [[ -L "$target_root" ]]; then
    printf 'release target root must not be a symlink\n' >&2
    exit 1
fi
relative="${candidate#"$target_root"/}"
current="$target_root"
IFS='/' read -r -a components <<<"$relative"
for component in "${components[@]}"; do
    [[ -n "$component" && "$component" != . && "$component" != .. ]] || exit 1
    current="$current/$component"
    if [[ -L "$current" ]]; then
        printf 'release artifact path component must not be a symlink\n' >&2
        exit 1
    fi
done
if [[ "$relative" == */* ]]; then
    printf 'release artifact directory must be a direct target leaf\n' >&2
    exit 1
fi
case "$relative" in
    people-release-validation | people-release-validation-[A-Za-z0-9_-]*) ;;
    *)
        printf 'release artifact directory must be a dedicated people-release-validation leaf\n' >&2
        exit 1
        ;;
esac
if ! mkdir -p "$candidate" 2>/dev/null; then
    printf 'release artifact directory could not be prepared\n' >&2
    exit 1
fi
canonical="$(realpath "$candidate" 2>/dev/null)" || {
    printf 'release artifact directory could not be canonicalized\n' >&2
    exit 1
}
[[ "$canonical" == "$candidate" ]] || {
    printf 'release artifact directory did not canonicalize to itself\n' >&2
    exit 1
}
if ! find "$candidate" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} + 2>/dev/null; then
    printf 'release artifact cleanup failed\n' >&2
    exit 1
fi
printf '%s\n' "$candidate"
