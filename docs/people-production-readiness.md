# People and local face recognition: production-readiness record

Date: 2026-08-30

Reference plan: [`people-production-plan.md`](../people-production-plan.md)

Retained validation tag: `codex-production-readiness-2026-08-30-final10-15m-soak`

Retained local artifacts:
`target/people-release-validation-final10-20260830`

This record maps the plan's production requirements to implementation and
retained verification evidence. The artifact directory is intentionally local;
release CI must retain the equivalent directory in its artifact store.

## User-visible result

- The People overview is paged and virtualized. Each ordinary person card uses
  a deterministic representative from effective confirmed faces and shows
  confirmed/review counts. Review and unnamed queues are first-class cards.
- Person and queue details use a virtualized face-thumbnail grid. Face crop is
  the default; an individual or grid-wide context mode displays the full image
  with the face rectangle. Only visible/overscan thumbnails are demanded.
- Pointer and keyboard selection support replacement, toggle, ranges, focus
  movement, select loaded results, and clear. The action bar performs bulk
  Accept, Reject identity, and Not a face operations.
- Decisions paint optimistically, commit through the background catalog worker,
  and roll back only the affected faces on failure. Decision journaling supports
  undo. Decisions and undo survive restart and Lightroom reimport.
- Face analysis is automatic maintenance work in the unified queue. It can be
  paused, resumed, cancelled, retried, and inspected. Visible rendering retains
  reserved foreground capacity.
- Accepted embeddings immediately participate in later recognition generations;
  person-specific rejections suppress that candidate; Not a face excludes the
  detection from review, recognition, and exemplars.

## Requirement matrix

| Plan requirement | Implementation evidence | Verification evidence | Status |
| --- | --- | --- | --- |
| Canonical schema and supported migrations | Versioned schema-v6 catalog migrations, application/version checks, backup-before-migration, foreign keys, typed People repository, atomic fingerprint/embedding/landmark/analysis writes | Fresh/v1/v2/v3/v4/v5 and disagreeing-marker recovery gates; injected rollback, pre-fingerprint pending backfill, and independent-backup integrity tests | Pass |
| Lightroom import/reimport precedence | Provenance-bearing faces/assignments, local override retention, vanished-face retention, conservative IoU/landmark reconciliation, conflict diagnostics | Synthetic and real Lightroom fixture import/reimport tests; idempotence, local confirm/reject/ignore/rename survival, invalid/non-finite geometry, virtual-copy and rollback coverage | Pass |
| Bounded People repository reads | Paged detail APIs plus schema-v5 epoch summaries, component closure, Unicode sort keys, ranked representatives, exact live fallback while rebuilding | Query-plan checks, authorizer-denied base-table reads on fresh summaries, randomized differential tests, 60k/250k profiles and churn/GC/preemption tests | Pass |
| No catalog work on GPUI thread | Generation-guarded async repository requests and typed background writes; stale results are discarded | Delayed/out-of-order request regressions, UI responsiveness/performance gate, optimistic rollback tests | Pass |
| Face crop/context rendering | Explicit face crop/context variants, revisioned cache identity, canonical coordinate transforms, shared JPEG/RAW/HEIF decoder and decoded-source cache | All eight orientations, crop/context, overlay, padding, decoder fixture, cache identity and no-redecode tests | Pass |
| Representative confirmed face | Deterministic effective-confirmation ranking, quality exclusions, user-selected representative protection, visible-only crop demand | Ranking unit/catalog tests, painted fallback/exhaustion tests, and retained 100,100-face GPUI gate | Pass |
| Virtualized review UI | Paged overview and uniform-list face grid with bounded DTO and thumbnail caches | Painted GPUI interaction tests; retained 100,100-face gate covers overview/detail, late pages, every sort in whole/subtree scopes, cache eviction, and bulk selection | Pass |
| Single and bulk review | Durable confirm/reject/not-face batches, idempotent chunks, per-face failure handling, optimistic deltas, decision journal and undo | Mouse/keyboard multiselect, 30-face bulk, scoped rollback, restart/reimport persistence and undo tests | Pass |
| Unified background jobs | Shared priority/admission/cancellation/metrics queue for rendering, face analysis, prototypes, and People-summary rebuilds | Foreground-overtakes-maintenance, reserved-capacity, pause/resume/cancel, durable claim/reclaim, queue-row painted cancellation and retry tests | Pass |
| Whole-library inventory | Discovery-backed portable catalog assets with supported-format, hidden-tree, pair preference, deletion and bounded-memory full-content SHA-256 source-identity synchronization | Inventory synchronization/change tests, including same-size/same-mtime byte replacement and immediate derived-vector invalidation; all eligible assets scheduled rather than only Lightroom-linked assets | Pass |
| Local detector/embedder | Bundled YuNet/SFace pipeline, letterboxing, NMS, five-point alignment, L2-normalized/versioned embeddings, fingerprint-keyed decode and transaction-time source revalidation | Model hashes/licenses/shapes, synthetic fixtures, orientation/RAW/HEIF path, metadata-preserving replacement race, invalid-vector and cancellation-at-stage tests | Pass |
| Recognition affects future results | Hash-bound production calibration, generation-scoped exemplars/prototypes, rejection constraints, absolute score plus runner-up margin, certified exact best/second identity over ANN seeds | Fresh production confirm→prototype→suggestion, identity-aware calibration reproduction/tamper tests, generation invalidation, rejection/ignore, exact-vs-brute-force and realistic 512D pruning tests | Pass |
| ANN scale and resource bounds | Cancellable HNSW build with deterministic exact certification, per-catalog/model/generation single-flight, bounded overlap | 100k×512 release ceiling, panic/poison/cancel retry, waiter/contention/stage cancellation; exact retained peak in `recognition-ann-catalog-ceiling.resources.txt` | Pass |
| Summary rebuild concurrency | Durable singleton epochs, leased typed scheduler job, TEMP shadow build, bounded keyset live copies/GC, owner/source/epoch CAS and foreground progress hook | Churn, writer-lock, active-generation GC race, expired reclaim, active cancel, 1M-VM mid-SQL preemption and retry convergence | Pass |
| Offline and private operation | Face engine has no network/map client; real ONNX inference runs under socket/connect-denying seccomp; normal diagnostics are aggregate-only | Offline inference/dependency gates and final artifact privacy scan | Pass |
| Diagnostics and recovery | Job progress/errors/stages, explicit transient/deterministic/exhausted/manual failure classes, five-attempt capped backoff, terminal manual retry, migration backup/recovery, settings for analysis and resources | No-wake terminal errors, retry exhaustion/manual reset, decoder-version reactivation, GPUI queue/diagnostics, lease and migration recovery gates | Pass |
| Production performance and soak | Source-controlled latency/resource limits with Linux descendant-aware supervisor and immutable executable/build-input manifests | Full retained release run, 60k/250k profiles, 100,100 GPUI gate, resource profile, user-selected 15-minute soak, privacy scan | Pass |

## Retained release measurements

The reference machine reported Linux 6.6.152, x86-64, and 16 logical CPUs.
Configured ceilings were 2,097,152 KiB RSS, 400% aggregate CPU, and 100 ms
cancellation latency.

| Gate | Retained result |
| --- | --- |
| Formatting | Passed |
| Strict Clippy, all targets/features | Passed with `-D warnings` |
| Complete test binaries | Lightroom inventory: 3 passed; Rekesz: 421 passed, 12 ignored, 0 failed |
| Recognition calibration | Hash-pinned identity-aware v3 report reproduced; operating point 0.70/0.70/0.20; 6 TP, 10 TN, 0 observed false rejects/accepts, unknown accepts, or misidentifications; small-holdout upper-bound limitation recorded |
| 60k catalog | All rebuild, bounded-read, plan, cancellation, and supervised-RSS budgets passed; exact values retained in `catalog-60k.log` and `.resources.txt` |
| 250k catalog/churn | All rebuild, bounded-read, churn/plateau, cancellation, and supervised-RSS budgets passed; exact values retained in `catalog-250k.log` and `.resources.txt` |
| GPUI 100,100 | Skeleton, 8 ms frame, bounded render/cache, scroll-RSS, and durable 30-face batch budgets passed; exact values retained in `gpui-people-performance.log` and `.resources.txt` |
| 100k×512 ANN | Build/query, exact-certification, overlapping-generation RSS, and all 100 ms cancellation budgets passed; exact values retained in `recognition-ann-catalog-ceiling.log` and `.resources.txt` |
| Overlapped real inference | CPU, RSS, and cancellation budgets passed with zero partial artifacts; exact values retained in `reference-resources.json` and `reference-resource.resources.txt` |
| User-selected 15-minute soak | The 900-second invariant test passed under the resource supervisor; exact wall time, RSS, and CPU retained in `long-soak.log` and `.resources.txt`; less rare-race/leak confidence than four hours |
| Offline/privacy | Real inference succeeded with socket/connect denied; retained aggregate privacy scan passed |
| Artifact integrity | Before/current/after executable and build-input manifests are identical |

## Phase reviews

Each implementation phase received an adversarial review before the next phase
was accepted. Findings were fixed and the relevant tests rerun. The final
cross-phase review is recorded in this document only after it reports no open
correctness, durability, responsiveness, resource, privacy, or test-evidence
finding.

## Manual acceptance journey

Automated coverage is the release authority, but a human acceptance pass should
still verify the visual and ergonomic result:

1. Open a copy or disposable Rekesz catalog that references a representative
   JPEG/RAW/HEIF library; never point test automation at the production catalog.
2. Open People and verify immediate skeleton paint followed by representative
   confirmed face crops and counts.
3. Open a person and scroll rapidly. Verify face crops, toggle several tiles and
   the whole grid to context, and confirm the face rectangle matches each photo.
4. Exercise click, Ctrl/Cmd-click, Shift-click, arrows, Shift+arrows,
   Ctrl/Cmd+A, and Escape. Apply all three bulk actions and verify optimistic
   paint, persistence after restart, and undo.
5. Import/reimport a Lightroom catalog copy. Confirm that local decisions and
   names remain authoritative and that retained/conflict counts are visible.
6. Let automatic analysis run, then create visible render demand. Confirm face
   work yields, and exercise pause, resume, individual cancel, retry failed, and
   application restart.
7. Accept a suggested identity, wait for prototype refresh, and confirm later
   compatible detections can receive it. Reject a person for a face and confirm
   that candidate stays suppressed; mark a detection Not a face and confirm it
   disappears from review and training.
8. Inspect People diagnostics and the job queue for useful aggregate status and
   errors, with no names, image paths, landmarks, vectors, or crop bytes in
   diagnostic logs.

### Automatic representative policy

Only effective confirmed, active, non-ignored faces with normalized area of at
least 0.0025 are candidates. A manual representative remains first. Automatic
candidates then sort, deterministically, by local Rekesz confirmation before
imported confirmation, detection confidence, usable normalized face area,
five-landmark frontal geometry, face `updated_at` recency, and finally face ID.
The frontal score rewards level eyes and mouth, a centered nose, and symmetric
eye/mouth spans; missing, malformed, non-finite, or implausibly out-of-bounds
landmarks score zero. Rekesz does not currently persist a calibrated sharpness
measurement, so the policy does not invent a blur score from area or detector
confidence. A crop decode/render failure advances to the next ranked candidate;
exhaustion produces the stable placeholder. The summary and exact live-fallback
queries use the same ordering.
