#!/usr/bin/env bash
# Refresh the SFace production policy's source bindings after a legitimate
# edit to one of the hash-pinned files (face_engine.rs, recognition_policy.rs,
# the verifier script, the extractor, the calibration README, or Cargo.lock).
#
# Recomputes the pinned digests in assets/face/sface_2021dec_policy_v1.json,
# updates PRODUCTION_SFACE_POLICY_DIGEST and the hardcoded digest comparison
# in src/catalog/recognition.rs, and re-signs assets/face/MANIFEST.sha256.
# Run the face-engine tests afterwards.
set -euo pipefail
cd "$(dirname "$0")/.."

policy="assets/face/sface_2021dec_policy_v1.json"
manifest="assets/face/MANIFEST.sha256"
recognition="src/catalog/recognition.rs"

declare -A bindings=(
  [verifier_source_sha256]="scripts/verify-sface-policy-calibration.py"
  [extractor_source_sha256]="src/bin/sface_calibration_embeddings.rs"
  [production_inference_source_sha256]="src/face_engine.rs"
  [production_policy_source_sha256]="src/catalog/recognition_policy.rs"
  [corpus_provenance_sha256]="assets/face/calibration/README.md"
  [dependency_lock_sha256]="Cargo.lock"
)

for key in "${!bindings[@]}"; do
  old=$(python3 - "$policy" "$key" <<'EOF'
import json, sys
data = json.load(open(sys.argv[1]))
print(data["calibration_report"]["held_out_data"][sys.argv[2]])
EOF
)
  new=$(sha256sum "${bindings[$key]}" | cut -d' ' -f1)
  if [ "$old" != "$new" ]; then
    python3 - "$policy" "$key" "$old" "$new" <<'EOF'
import sys
path, key, old, new = sys.argv[1:]
text = open(path).read()
assert f'"{key}": "{old}"' in text, f"{key}: recorded digest not found"
open(path, "w").write(text.replace(f'"{key}": "{old}"', f'"{key}": "{new}"'))
EOF
    echo "refreshed $key (${bindings[$key]})"
    # The hardcoded comparison in recognition.rs carries the same digest.
    python3 - "$recognition" "$old" "$new" <<'EOF'
import sys
path, old, new = sys.argv[1:]
text = open(path).read()
open(path, "w").write(text.replace(f'!= "{old}"', f'!= "{new}"'))
EOF
  fi
done

# The policy JSON itself is digest-pinned in three places.
old_policy=$(sha256sum "$policy" | cut -d' ' -f1)
previous=$(grep "$(basename "$policy")" "$manifest" | cut -d' ' -f1)
if [ "$previous" != "$old_policy" ]; then
  digest_const="PRODUCTION_SFACE_POLICY_DIGEST"
  python3 - "$recognition" "$previous" "$old_policy" <<'EOF'
import sys
path, old, new = sys.argv[1:]
text = open(path).read()
assert old in text, "previous policy digest not found in recognition.rs"
open(path, "w").write(text.replace(old, new))
EOF
  python3 - "$manifest" "$previous" "$old_policy" <<'EOF'
import sys
path, old, new = sys.argv[1:]
text = open(path).read()
assert old in text, "previous policy digest not found in MANIFEST"
open(path, "w").write(text.replace(old, new))
EOF
  echo "refreshed policy digest ($policy)"
fi

echo "chain refreshed; run: cargo test -- face_engine:: catalog::"
