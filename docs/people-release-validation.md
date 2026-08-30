# People release, performance, soak, and privacy validation

Run the checked-in harness on the tagged reference machine. It runs all quality
gates, both ignored generated-catalog profiles, the tagged resource profile,
and the long soak; a failure in any command fails the release gate:

```sh
PEOPLE_REFERENCE_TAG=linux-x86_64-reference-2026q3 \
  ./scripts/people-release-validation.sh
```

The manual `People reference release validation` workflow runs the same script
on the `people-reference` self-hosted runner and retains its artifacts for 90
days. Do not use a generic hosted runner for release approval.

The harness writes command logs, resource-supervisor records, revision/host
metadata, and `reference-resources.json` beneath
`target/people-release-validation` (override with
`PEOPLE_RELEASE_ARTIFACT_DIR`). The Linux-only Rust supervisor starts every
measured command in a new process session, samples aggregate process-group RSS
from `/proc`, reaps descendants with `wait4`, and records the conservative
maximum of sampled group RSS and per-process kernel high-water RSS. It also
records aggregate descendant CPU time, forwards termination signals, and kills
and reaps the complete process group after a timeout or threshold violation.
This deliberately avoids a dependency on GNU `/usr/bin/time` and prevents
compiler/linker work from being confused with the runtime under review.

The JSON report is tagged with
`PEOPLE_REFERENCE_TAG` and machine-checks peak RSS, aggregate CPU utilization,
and cancellation-observation latency. Reviewed host-specific limits may be set
with `PEOPLE_MAX_RSS_KIB`, `PEOPLE_MAX_CPU_PERCENT`, and
`PEOPLE_MAX_CANCEL_LATENCY_MS`; defaults are 2 GiB, 400%, and 100 ms. Any
threshold violation fails the test after still writing the report, so failed
runs retain evidence. Threshold changes used for release approval must be
recorded with the artifact and reviewed; they are not query-latency overrides.
The resource profile keeps a real ONNX analysis job running at a production
stage with a second job queued while the 60k People workload executes. It
samples RSS and CPU during that overlap, then cancels and records cleanup
latency plus durable pending and partial face/landmark/embedding counts.
Production counters around the actual detector invocation must report at least
one start and completion, and the JSON overlap flag must be true; a pre-detect
stage barrier alone is not valid release evidence.

Record CPU, RAM, storage, OS, build revision, dataset size, elapsed query times,
peak RSS, and cache policy. The normal `cargo test` gate always exercises 10k
faces, with a deliberately conservative 500 ms debug-build page-query ceiling
that remains stable while the suite's real ONNX concurrency tests are active.
The two ignored release gates exercise 60k and 250k faces with fixed page-query
ceilings of 50 ms and 100 ms respectively. These are source-controlled release
budgets: do not override them through the environment or silently relax them.
The generated data distributes four faces per
asset and twenty faces per person, with a ten-percent review queue. The test
machine-checks result counts, fixed 30-row page/DTO bounds, every query against
the applicable ceiling, prompt cancellation, and indexed query plans using
`EXPLAIN QUERY PLAN`. A slower CI runner requires a reviewed source change with
its reference measurements; an environment override is intentionally not
available. The initial
budgets are: skeleton paint under 100 ms; bounded 60k-catalog page queries under
50 ms; no steady-state UI-thread task over 8 ms; cache/DTO memory reaches a
plateau while scrolling; and cancellation is observed between pipeline stages.
A budget miss blocks release until it is investigated and the baseline or code
is deliberately updated with recorded evidence.

The normal suite also runs a three-cycle accelerated reimport/restart/cancel
soak. A production stage barrier proves each Lightroom reimport overlaps a
running real face service. The cycle cancels and drops the service during that
overlap, recreates it, completes the durable job, and verifies durable state,
zero partial/duplicate artifacts, and preserved effective rejection. The
ignored long test runs the same assertions for four hours by default; set
`PEOPLE_SOAK_SECONDS` only to a reviewed release duration.
Record elapsed time and peak RSS. Also manually exercise
pause/resume and scope changes in the UI. Verify leases are reclaimed, decisions
remain unchanged, retries are capped, queues drain, and RSS plateaus. Save
aggregate counters only; paths, names, embeddings, landmarks, and crop bytes
must never appear in the report.

The release harness automatically runs real detection and embedding under a
Linux seccomp filter that rejects `socket` and `connect`, plus the dependency
audit proving the face engine has no map/network client. The normal integration
suite covers import, recognition, accept/reject/ignore, undo, restart, and
retry-failed flows with online map detail disabled. Finally,
`check-people-artifact-privacy.sh` fails the run if the retained aggregate JSON
contains identifying paths/names or biometric vectors, landmarks, crops, or
image bytes. A reference host may additionally deny the entire runner's network
namespace as defense in depth, but release approval does not rely on a manual
log inspection.

The automated recovery gates start from supported old schemas, force migration
failure after backup creation, open the versioned copy independently, and verify
SQLite integrity plus the original schema marker and decisions.

## GPUI accessibility contract

GPUI 0.2.2 does not expose an ARIA/accessibility-tree label API for custom
`div` controls. People controls therefore use the strongest semantics available
in this version: stable composite element IDs/debug selectors, complete visible
labels and state text, tab stops, and identical Enter/Space and pointer actions.
Review cards include both face and person identity in their stable selector so
two suggestions for one detected face remain independently addressable. The
grid focus is restored after refreshes, saved decisions, undo, and operation
dialog dismissal. When GPUI exposes platform accessibility nodes, replacing
the debug-selector metadata with native roles/names is a release requirement,
not an optional visual cleanup.

This limitation was verified against GPUI 0.2.2's public `Element`,
`InteractiveElement`, `StatefulInteractiveElement`, and `Window` APIs: the
crate exposes focus handles, tab stops, debug selectors, visible text, and
keyboard/pointer listeners, but no platform accessibility node, role, name,
selected, disabled, or busy-state setter. Stable debug selectors are test
metadata only and are not represented as native assistive-technology nodes.

The `gpui-people-performance` release gate seeds 100,100 detected faces (a
100,000-face detail queue plus 100 overview representatives) and
retains its aggregate JSON in
`gpui-people-performance.log`. It measures an actual first painted selector,
all four overview sorts across whole-library early/late pages and both pages of
a distinct 50-person selected subtree, detail pages through offset 99,960,
long-distance cache/scroll plateau, bounded visible crop demand, painted
multiselect, and durable completion of a 30-face decision batch. It enforces the 100 ms skeleton, 8 ms
steady frame, 64 MiB scroll-growth, bounded-render/cache, and one-second bulk
budgets in the optimized test binary.

## Current retained reference

The current 2026-08-30 run is under
`target/people-release-validation-final10-20260830` and is tagged
`codex-production-readiness-2026-08-30-final10-15m-soak`. It ran on Linux 6.6.152,
x86-64, with 16 logical CPUs. The complete before/current/after artifact
manifests match, proving that the measured test executables and all build inputs
were unchanged across runtime validation.

The run passed formatting, strict Clippy, both complete debug test binaries,
offline inference and dependency audits, all supported migration/recovery
gates, reproducible recognition-calibration verification, ANN
correctness/resource/cancellation gates, the 100,100-face GPUI journey, 60k and
250k catalog profiles, a reviewed 15-minute reimport/restart/cancel soak, and
the artifact privacy scan. Key retained measurements are:

- full Rekesz suite: 421 passed, 12 ignored, 0 failed;
- calibration v3: operating point 0.70/0.70/0.20, 6 true positives and 10
  true negatives, with no observed false reject, false accept, unknown accept,
  or misidentification;
- 60k and 250k catalog profiles: all source-controlled query, rebuild,
  cancellation, plan, and storage bounds passed; exact timings and supervised
  RSS are retained in `catalog-60k.log`, `catalog-250k.log`, and their sibling
  `.resources.txt` files;
- GPUI at 100,100 faces: skeleton, 8 ms frame, bounded render/cache, scroll RSS,
  and durable 30-face decision budgets passed; exact results are retained in
  `gpui-people-performance.log` and `.resources.txt`;
- 100k×512 ANN: build/query, correctness, overlap RSS, and all 100 ms stage
  cancellation budgets passed; exact results are retained in
  `recognition-ann-catalog-ceiling.log` and `.resources.txt`;
- overlapped inference/People resource profile: CPU, RSS, and cancellation
  budgets passed with no partial faces, landmarks, or embeddings; exact values
  are retained in `reference-resources.json` and `reference-resource.resources.txt`;
- reviewed soak: the selected 900-second test passed; exact test and supervised
  wall time, RSS, and CPU are retained in `long-soak.log` and
  `long-soak.resources.txt`;
- privacy scan: no identifying or biometric payload in retained aggregate
  reports.

The user explicitly selected a 15-minute rather than four-hour soak for this
run. It exercises the same lifecycle and durability invariants, but provides
less statistical confidence against rare races and slow resource leaks. The
historical runs, including the four-hour `final3` run and the superseded
`final5`–`final9` runs, predate parts of the current schema, calibration,
privacy validator, test layout, or GPUI journey and are not release evidence
for this tree.

These artifacts are local build outputs and are not expected to be committed.
For a later release, create a fresh tagged artifact directory and retain it in
the reference-runner artifact store rather than reusing these measurements.
