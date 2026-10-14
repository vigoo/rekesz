# Background Render Scheduler Plan

This document is the implementation plan and progress tracker for prioritized,
cancellable background image work. It is intentionally kept beside the code so
future scheduling, cache, metadata, and rendering changes preserve the same
invariants.

## Goals and invariants

- The GPUI thread never performs filesystem I/O, image decoding, cache encoding,
  or blocking waits.
- Opening an image is more important than all thumbnail and speculative work.
- Visible thumbnails are more important than off-screen or predicted work.
- Changing a viewport or directory removes obsolete demand immediately.
- Selecting a directory promotes its queued discovery ahead of background tree
  expansion, with discovery capacity reserved for interactive navigation.
- Equal render requests execute once and may have multiple interested consumers.
- Parallel render-job count is configurable and strictly bounded.
- Queue operations are short, bounded mutex operations; workers never hold the
  scheduler mutex while rendering or delivering results.
- Stale/cancelled results never replace newer UI state.
- In-memory and filesystem cache semantics remain independent of job scheduling.
- Speculative work must never keep the foreground progress bar active.

## Job model

`JobKey` contains the source path and complete `RenderVariant`. The current
render recipe and source identity are resolved by the rendering backend and its
filesystem-cache key. When per-image recipes become mutable, the recipe identity
must move into `JobKey` as well.

Each job contains:

- a set of demand scopes and their priorities;
- zero or more result subscribers;
- queued/running state;
- a cooperative cancellation token;
- an effective priority equal to the strongest live demand;
- monotonic sequencing for deterministic ordering inside equal priorities.

The scheduler uses an indexed priority queue so a queued job can be promoted,
demoted, or removed without inserting duplicate heap entries.

## Demand scopes

- `ImageView`: the currently opened fit-window preview or 1:1 render.
- `ThumbnailViewport`: thumbnails required by the current virtual-grid range.
- `ThumbnailNearViewport` (phase 3): bounded overscan around the visible range.
- `IdlePrefetch` (phase 3): disposable predicted work.
- Directory discovery uses an equivalent priority-aware bounded queue: selected
  directories are interactive and tree expansion is background work.
- Future scopes may cover cache maintenance.

Replacing a scope is atomic from the scheduler's perspective. Removed interests
are detached; jobs with no remaining interest are removed when queued or marked
cancelled when running. Subscribers removed with a scope receive a cancellation
completion so their UI cache returns from `Loading` to `Unloaded`.

## Priority policy

From strongest to weakest:

1. `Interactive`: opened RAW preview, full 1:1 render, explicit user action.
2. `Visible`: thumbnails in the current virtual viewport, center-first.
3. `NearViewport`: overscan, biased by scroll direction.
4. `Speculative`: idle prediction and gradual directory pre-generation.
5. `Maintenance`: persistent-cache writes and cleanup.

Equal-priority work is ordered deterministically. Queued background work ages
up to `NearViewport`, which prevents starvation among retained background work
without ever allowing speculative demand to overtake genuinely visible work.

## Cancellation semantics

- Queued jobs with no demand are removed and never execute.
- Running jobs receive a cancellation token and their stale result is discarded.
- New demand for a cancelled, still-running key becomes a successor generation;
  it starts after the retiring generation and is not attached to stale work.
- Cancellation is checked around renderer/cache stages initially.
- Rawler decoding and demosaicing are not currently preemptible internally. A
  later change to our rawler fork can add checkpoints inside expensive loops.
- At least one execution slot should eventually be reserved for interactive work;
  this is required for strong latency guarantees when low-priority RAW work is
  already running (phase 4 resource policy).

## Pipeline evolution

### Phase 1 — scheduler foundation

- [x] Application-owned scheduler and worker lifecycle
- [x] Stable job keys and semantic priorities
- [x] Indexed priority queue with promotion/demotion/removal
- [x] Configurable top-level render parallelism
- [x] Single-flight execution with multiple subscribers
- [x] Queued cancellation and running cancellation token
- [x] Scheduler statistics snapshot
- [x] Non-blocking UI shutdown with asynchronous worker reaping

### Phase 2 — UI demand integration

- [x] Replace direct GPUI image-render spawning with scheduler submission
- [x] Image-view scope uses `Interactive`
- [x] Virtual-grid callback publishes visible thumbnail demand
- [x] Grid center-first ordering
- [x] Scroll changes cancel obsolete queued thumbnails
- [x] Directory changes clear the previous grid scope immediately
- [x] Preview-cache request IDs still reject stale completions
- [x] Foreground progress tracks subscribed interactive/visible requests
- [x] CLI option for render parallelism

### Phase 3 — idle prediction

- [x] Near-viewport bounded overscan
- [x] Scroll-direction tracking and directional bias
- [x] Idle timer before speculative submission
- [x] Bounded speculative queue and gradual radius expansion
- [x] Neighbor RAW-preview prediction for focused/adjacent images
- [x] Speculative jobs excluded from foreground progress
- [x] Priority aging and starvation tests

### Phase 4 — staged execution and resource policy

- [x] Separate cache-lookup I/O, CPU render, and cache-write stages
- [x] Return pixels before low-priority persistent-cache writes complete
- [x] Separate bounded directory/metadata I/O capacity
- [x] Reserve interactive execution capacity
- [x] Audit rawler's Rayon use and bound nested parallelism
- [x] Use a dedicated bounded Rayon pool via `ThreadPool::install`
- [x] Add a separately configurable concurrency limit for RAW working sets

The stage policy is implemented by `RenderPipeline`: cache reads have their own
small permit pool, decode/transform work has a general CPU permit pool plus a
separate RAW-working-set pool, and cache persistence uses a bounded maintenance
channel with one writer.
The UI-visible result is constructed immediately after rendering; a saturated
write queue drops optional persistence rather than delaying pixels or retaining
an unbounded number of decoded images.

An implementation audit found Rawler's decode, demosaic, and pixel operations
use Rayon throughout. Running
multiple scheduler workers against Rayon's unrestricted global pool would
multiply runnable work. RAW calls are therefore installed into a dedicated pool
whose size is derived from hardware concurrency and permitted RAW concurrency.
This provides a deterministic oversubscription bound; further camera/OS-specific
throughput measurements can tune the formula without changing pipeline APIs.

### Phase 5 — deeper cancellation and observability

- [x] Cancellation checkpoints throughout our photographic transforms
- [x] Cooperative cancellation support in the rawler fork
- [x] Queue latency, execution time, cancellation, and cache-hit metrics
- [x] Optional diagnostics UI/logging
- [x] Disk-cache LRU cleanup using cache-entry access information

### End-to-end review repairs

- [x] Aggregate all virtual-list callbacks before one viewport publication
- [x] Publish visible demand before near/speculative demand
- [x] Remove demand-triggered unconditional repainting
- [x] Overlay progress without changing virtual-grid geometry
- [x] Report current foreground activity instead of cumulative task churn
- [x] Allow the reserved worker to serve cancellable visible work while keeping
  near/speculative work out of the foreground slot
- [x] Subscribe background work into bounded memory caches without activating
  foreground progress
- [x] Apply bounded backpressure to speculative cache writes instead of silently
  discarding completed background renders
- [x] Run filesystem-cache LRU scans periodically rather than after every write
- [x] Avoid cloning the complete directory image list from each grid callback
- [x] Preserve cache-result subscriptions when a thumbnail moves between live
  speculative/near/visible scopes
- [x] Promote selected-directory discovery and reserve discovery capacity so
  background expansion cannot occupy every scanner slot
- [x] Preserve grid/near/speculative demand while entering image mode; enqueue
  the opened image at `Interactive` instead of erasing the existing queue
- [x] Cooperatively preempt one lower-priority running render when new visible or
  interactive work would otherwise wait, retaining and requeueing its demand
- [x] Treat disjoint virtual-list callbacks as a newer viewport instead of
  unioning every row traversed during a fast scroll
- [x] Split directory discovery into immediate modification-time enumeration and
  cancellable capture-time enrichment
- [x] Treat adjacent XMP `DateTimeOriginal` as authoritative and skip source-file
  metadata extraction when it is valid
- [x] Persist capture-time metadata using source and XMP size/mtime identities
- [x] Apply cached capture-time ordering before first display; newly discovered
  capture times are cached without moving the live grid and instead expose an
  explicit apply-order action that preserves path-based selection
- [x] Generalize `--render-parallelism` to `--job-parallelism` (with the former
  retained as an alias) and enforce one shared application-wide admission cap
  across render, directory, and metadata work

On the 515-image `/mnt/reference-pictures/manual-benchmark/2018/vacation` benchmark,
the old synchronous scan spent 2.24–2.39 seconds extracting capture metadata.
The two-stage implementation publishes the modification-time grid in about
21 ms, completes a fresh authoritative-XMP enrichment in about 119 ms, and
reloads the persistent metadata cache in about 70 ms (debug build, warm mount).

Set `REKESZ_DIAGNOSTICS=1` to print aggregate scheduler and cache metrics at
shutdown. Scheduler snapshots retain queue/execution latency and completion /
cancellation counts; the render pipeline separately counts cache hits, misses,
intentional bypasses, read failures, completed/rejected/failed writes, and writes
discarded because the maintenance queue was saturated.

View → Job Queue opens a live, read-only scheduler panel. The application always
registers the GPUI platform menu model: macOS renders it as the native system
menu, while Linux and Windows use the in-window menu fallback because GPUI 0.2
only retains the platform menu model there and does not render an OS menu bar.
It displays running jobs first, followed by queued jobs in actual dispatch order,
including aged effective priority, rank, demand-scope count, sequence, render
variant, and cancellation state. The same snapshots drive foreground progress,
so scope promotion cannot make visible work disappear from the progress bar.

Disk entries have portable `.access` sidecars refreshed on hits and writes. The
maintenance writer enforces a 10 GiB default budget by removing the oldest PNG,
metadata, and access-sidecar group together.

The `vigoo/dnglab` fork now exposes a cache-key-independent cancellation token,
a generic cancellable decoder boundary, internal Canon CR3/CRX tile, plane, and
integration checkpoints, and cancellable RAW-development stage boundaries. Rekesz
pins that revision and shares the scheduler's atomic cancellation flag directly
with rawler. Other decoder implementations receive safe before/after checks and
can add deeper format-specific checkpoints incrementally. Image/HEIF decode,
resize, and orientation calls remain bracketed non-preemptible library calls.

### Render and metadata performance pass

- [x] RAW source decode/development is single-flight across render variants.
- [x] Recently developed RAW sources use the configurable image-cache capacity,
  allowing thumbnails, display previews, and 1:1 renders to derive multiple
  outputs without repeating RAW development.
- [x] The intermediate-source cache is bounded and uses source identity in its
  key, so changed files cannot reuse stale developed pixels.
- [x] JPEG thumbnail/preview requests use decoder-level 1/8, 1/4, or 1/2 IDCT
  scaling before their final exact resize; full JPEG rendering is unchanged.
- [x] Capture metadata is stored in one SQLite/WAL index with batched
  transactional writes instead of hundreds of per-image JSON files.
- [x] Display previews are persisted as QOI for low-overhead lossless decoding;
  compact thumbnails remain PNG.
- [x] Pipeline diagnostics separately time source identity, cache lookup, source
  decode, derivative transform, GPUI image conversion, and cache persistence.
- [x] Ignored release benchmarks cover every scheduled action, cache cold/hit
  behavior, RAW cancellation handoff, and RAW throughput at parallelism 1/2/4.

The representative release benchmark shows the main intended reuse path: after
a RAW thumbnail develops its source, a 1920x1080 preview takes about 125–130 ms
instead of 1.6–1.9 seconds, and the subsequent full render takes about 108–114
ms. Four distinct NEF thumbnails take about 7.71 s at parallelism 1, 4.25 s at
parallelism 2, and 2.73 s at parallelism 4 when RAW concurrency is configured to
match. `--raw-parallelism` independently controls the maximum number
of simultaneous RAW working sets and is capped by `--job-parallelism`. Without
an override, sysinfo supplies logical CPU count and currently available memory;
the default uses at most half the logical CPUs, reserves 1 GiB for the system,
budgets 768 MiB per RAW job, and always retains an interactive RAW permit when
more than one slot is available. These constants are deliberately conservative
estimates and can be refined later using measured peak resident memory. General
job parallelism defaults to the logical CPU count capped at eight, so the RAW
estimate can exceed the former fixed two-job ceiling while preventing an
unbounded metadata or ordinary-image burst on very large machines.

## Detailed testing plan

### Pure priority and queue tests

- Higher semantic class always precedes lower classes.
- Center-nearer visible thumbnail precedes farther visible thumbnails.
- Equal priorities have deterministic submission ordering.
- Promotion mutates the existing queue item instead of duplicating it.
- Demotion changes pop order.
- Removing the final demand removes a queued job.
- Removing one of several demands retains the job at the next effective priority.

### Scheduler concurrency tests with a controlled fake renderer

- Configured parallelism is never exceeded.
- An interactive job overtakes already queued visible thumbnails.
- Two consumers of one `JobKey` cause exactly one execution and both complete.
- Cancelling one consumer does not cancel another consumer's shared job.
- Cancelling the final queued consumer prevents execution.
- Cancelling a running job discards its result.
- Worker panic/failure becomes an error completion and does not kill capacity.
- Shutdown wakes idle workers and joins all threads.
- Enqueuing and reprioritizing remain safe while jobs complete concurrently.

### UI/model integration tests

- Entering image mode publishes only the required `Interactive` variant.
- Switching fit/1:1 replaces preview/full demand correctly.
- Returning to fit mode reuses memory/disk cache without unnecessary rendering.
- A virtual-grid range publishes only its visible image keys.
- Changing the visible range removes obsolete thumbnail demand.
- Switching directories clears the old grid scope before new work is submitted.
- A cancelled completion returns the matching preview-cache entry to `Unloaded`.
- A stale request ID cannot overwrite a newer render.
- Memory-cache hits do not enqueue scheduler jobs.

### Rendering/cache integration tests

- Filesystem-cache hit completes through the scheduler without source decoding.
- Cache miss renders once, persists, and serves a later scheduler instance.
- Different variants/recipes do not deduplicate incorrectly.
- Larger RAW previews satisfy smaller display demand, but not the reverse.
- Cancelled work does not publish stale pixels or corrupt persistent entries.
- Corrupt filesystem entries fall back to generation.

### End-to-end/manual scenarios

- Queue a screen of RAW thumbnails, immediately open one, and confirm its preview
  starts before remaining queued thumbnails.
- Rapidly scroll through thousands of images and confirm old queued work drains
  without visible-grid starvation or unbounded queue growth.
- Switch directories repeatedly during RAW generation and confirm no wrong image
  appears and foreground progress settles.
- Compare `--job-parallelism 1`, `2`, and CPU-count values for correctness,
  responsiveness, throughput, and peak memory.
- Move the window between standard-DPI and HiDPI displays while work is queued.
- Restart the app and verify scheduler-driven persistent-cache hits.

## Current progress

- Planning and architecture: complete.
- Phase 1: complete.
- Phase 2: complete.
- Phase 3: complete and adversarially reviewed.
- Phase 4: complete and adversarially reviewed.
- Phase 5: complete and adversarially reviewed.

### Implemented automated validation

- [x] Semantic priority classes have strict ordering.
- [x] Interactive work overtakes queued visible work.
- [x] Duplicate demand executes once and notifies both consumers.
- [x] Clearing a scope removes queued obsolete work and cancels its receiver.
- [x] Configured parallelism is a hard upper bound.
- [x] Cancelled preview-cache requests return to `Unloaded`.
- [x] Virtual-grid demand contains only the supplied visible row range.
- [x] Visible thumbnails are prioritized center-first.
- [x] CLI parsing covers render parallelism and rejects zero.
- [x] Speculative batches remain bounded for extremely large directories.
- [x] Geometry/display changes invalidate stale grid prediction.
- [x] Background aging never overtakes true visible work.
- [x] A cancelled running generation can be followed by fresh demand for the
  same key without concurrent duplicate execution or a lost request.
- [x] Configured scheduler parallelism remains a hard upper bound.
- [x] Background work cannot consume the reserved interactive worker.
- [x] Existing persistent-cache and full CR3 decode regressions pass through the
  staged backend.
- [x] Priority-aware permits preserve interactive capacity below the scheduler.
- [x] Waiting stage admission observes cooperative cancellation.
- [x] Deferred persistence rejects a render when its captured source identity
  no longer matches the file at commit time.
- [x] A deterministic staged thumbnail test covers generation and deferred
  persistent-cache visibility.
- [x] Existing RAW decode, persistent cache, LRU, discovery, selection, and zoom
  regression tests continue to pass.
- [x] Photographic transforms stop at bounded cancellation checkpoints.
- [x] Disk-cache cleanup evicts the least-recently-accessed complete entry.
- [x] The CR3 integration passes the scheduler cancellation flag into rawler.

### Remaining cancellation boundary

Queued work is fully cancellable. Running work is marked cancelled and its
result is discarded. Canon CR3/CRX cancellation is checked at tile, plane, and
integration-row boundaries, while RAW development checks between processing
stages. Entropy decoding within one plane, individual demosaic/calibration
operators, and third-party image/HEIF operations remain monolithic boundaries,
so cancellation is cooperative rather than instantaneous. Other RAW decoders
currently use the generic before/after boundary and can gain format-specific
internal checkpoints without changing Rekesz. Resource reservation ensures these
bounded operations do not consume the interactive slot when parallelism is
greater than one.
