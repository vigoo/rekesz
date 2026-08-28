//! Resource-aware orchestration for one render invocation.
//!
//! This is the sole rendering entry point used by UI adapters. It owns permit
//! ordering, RAW decoded-source reuse, cache publication policy, cancellation,
//! and stage metrics; pixel algorithms remain in lower rendering modules.

use std::{
    path::PathBuf,
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc,
    },
    thread,
    time::Instant,
};

use crate::{
    catalog::DevelopRecipe,
    rendering::{
        RenderBackend, RenderIdentity, RenderRequest, RenderVariant, RenderedImage, SourceIdentity,
        cache::{DecodedSourceCache, DecodedSourceKey},
        plan::{RenderPlan, RenderStage},
    },
};

pub struct RenderInvocation<'a> {
    pub path: PathBuf,
    pub variant: RenderVariant,
    pub render_identity: RenderIdentity,
    pub recipe: Option<&'a DevelopRecipe>,
    pub interactive: bool,
    pub durable_cache_write: bool,
}

#[cfg(test)]
mod tests;

/// Blocking render stages and their independent resource limits. The scheduler
/// owns prioritization; this type owns cache I/O, weighted decode permits, and
/// disposable low-priority persistence.
#[derive(Clone)]
pub struct RenderPipeline {
    backend: Arc<RenderBackend>,
    decoded_sources: Arc<DecodedSourceCache>,
    cache_reads: Arc<PermitPool>,
    cpu: Arc<PermitPool>,
    raw_cpu: Arc<PermitPool>,
    cache_writer: Arc<CacheWriter>,
    raw_pool: Arc<rayon::ThreadPool>,
    metrics: Arc<PipelineMetrics>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PipelineStats {
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub cache_writes: u64,
    pub dropped_cache_writes: u64,
    pub rejected_cache_writes: u64,
    pub failed_cache_writes: u64,
    pub cache_bypasses: u64,
    pub cache_read_failures: u64,
    pub decoded_source_hits: u64,
    pub source_identity: StageTiming,
    pub cache_lookup: StageTiming,
    pub source_decode: StageTiming,
    pub derivative_transform: StageTiming,
    pub cache_encode_and_write: StageTiming,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct StageTiming {
    pub calls: u64,
    pub total_nanos: u64,
}

pub(crate) use crate::rendering::cache::DiskCacheUsage;

impl StageTiming {
    pub fn average_nanos(self) -> u64 {
        self.total_nanos.checked_div(self.calls).unwrap_or(0)
    }
}

#[derive(Default)]
struct AtomicStageTiming {
    calls: AtomicU64,
    total_nanos: AtomicU64,
}

impl AtomicStageTiming {
    fn record(&self, started: Instant) {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.total_nanos.fetch_add(
            started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64,
            Ordering::Relaxed,
        );
    }

    fn snapshot(&self) -> StageTiming {
        StageTiming {
            calls: self.calls.load(Ordering::Relaxed),
            total_nanos: self.total_nanos.load(Ordering::Relaxed),
        }
    }
}

#[derive(Default)]
struct PipelineMetrics {
    cache_hits: AtomicU64,
    cache_misses: AtomicU64,
    cache_writes: AtomicU64,
    dropped_cache_writes: AtomicU64,
    rejected_cache_writes: AtomicU64,
    failed_cache_writes: AtomicU64,
    cache_bypasses: AtomicU64,
    cache_read_failures: AtomicU64,
    source_identity: AtomicStageTiming,
    cache_lookup: AtomicStageTiming,
    source_decode: AtomicStageTiming,
    derivative_transform: AtomicStageTiming,
    cache_encode_and_write: AtomicStageTiming,
}

impl PipelineMetrics {
    fn snapshot(&self) -> PipelineStats {
        PipelineStats {
            cache_hits: self.cache_hits.load(Ordering::Relaxed),
            cache_misses: self.cache_misses.load(Ordering::Relaxed),
            cache_writes: self.cache_writes.load(Ordering::Relaxed),
            dropped_cache_writes: self.dropped_cache_writes.load(Ordering::Relaxed),
            rejected_cache_writes: self.rejected_cache_writes.load(Ordering::Relaxed),
            failed_cache_writes: self.failed_cache_writes.load(Ordering::Relaxed),
            cache_bypasses: self.cache_bypasses.load(Ordering::Relaxed),
            cache_read_failures: self.cache_read_failures.load(Ordering::Relaxed),
            decoded_source_hits: 0,
            source_identity: self.source_identity.snapshot(),
            cache_lookup: self.cache_lookup.snapshot(),
            source_decode: self.source_decode.snapshot(),
            derivative_transform: self.derivative_transform.snapshot(),
            cache_encode_and_write: self.cache_encode_and_write.snapshot(),
        }
    }
}

struct CacheWrite {
    request: RenderRequest,
    identity: SourceIdentity,
    rendered: Arc<RenderedImage>,
}

struct CacheWriter {
    sender: Mutex<Option<mpsc::SyncSender<CacheWrite>>>,
    worker: Mutex<Option<thread::JoinHandle<()>>>,
    metrics: Arc<PipelineMetrics>,
}

impl CacheWriter {
    fn write(&self, mut write: CacheWrite, durable: bool, cancelled: &AtomicBool) {
        loop {
            let result = self
                .sender
                .lock()
                .unwrap()
                .as_ref()
                .map(|sender| sender.try_send(write));
            match result {
                Some(Ok(())) => return,
                Some(Err(mpsc::TrySendError::Full(returned))) if durable => {
                    if cancelled.load(Ordering::Acquire) {
                        return;
                    }
                    write = returned;
                    thread::sleep(std::time::Duration::from_millis(10));
                }
                Some(Err(mpsc::TrySendError::Full(_))) => {
                    self.metrics
                        .dropped_cache_writes
                        .fetch_add(1, Ordering::Relaxed);
                    return;
                }
                Some(Err(mpsc::TrySendError::Disconnected(_))) | None => return,
            }
        }
    }
}

impl Drop for CacheWriter {
    fn drop(&mut self) {
        self.sender.get_mut().unwrap().take();
        if let Some(worker) = self.worker.get_mut().unwrap().take() {
            let _ = worker.join();
        }
        if crate::scheduler::diagnostics_enabled() {
            let stats = self.metrics.snapshot();
            eprintln!(
                "rekesz cache: hits={}, misses={}, bypasses={}, read-failures={}, writes={}, dropped={}, rejected={}, write-failures={}; average-us: identity={}, cache-lookup={}, source-decode={}, derivative-transform={}, cache-write={}",
                stats.cache_hits,
                stats.cache_misses,
                stats.cache_bypasses,
                stats.cache_read_failures,
                stats.cache_writes,
                stats.dropped_cache_writes,
                stats.rejected_cache_writes,
                stats.failed_cache_writes,
                stats.source_identity.average_nanos() / 1_000,
                stats.cache_lookup.average_nanos() / 1_000,
                stats.source_decode.average_nanos() / 1_000,
                stats.derivative_transform.average_nanos() / 1_000,
                stats.cache_encode_and_write.average_nanos() / 1_000,
            );
        }
    }
}

struct PermitPool {
    state: Mutex<PermitState>,
    wake: Condvar,
    capacity: usize,
    interactive_reserve: usize,
}

struct PermitState {
    available: usize,
    waiting_interactive: usize,
}

struct Permit<'a> {
    pool: &'a PermitPool,
    units: usize,
}

impl PermitPool {
    fn new(capacity: usize, interactive_reserve: usize) -> Self {
        Self {
            state: Mutex::new(PermitState {
                available: capacity,
                waiting_interactive: 0,
            }),
            wake: Condvar::new(),
            capacity,
            interactive_reserve: interactive_reserve.min(capacity),
        }
    }

    fn acquire(
        &self,
        requested: usize,
        interactive: bool,
        cancelled: &AtomicBool,
    ) -> Option<Permit<'_>> {
        let units = requested.min(self.capacity).max(1);
        let mut state = self.state.lock().unwrap();
        if interactive {
            state.waiting_interactive += 1;
        }
        loop {
            let admissible = if interactive {
                state.available >= units
            } else {
                state.waiting_interactive == 0
                    && state.available >= units.saturating_add(self.interactive_reserve)
            };
            if admissible {
                if interactive {
                    state.waiting_interactive -= 1;
                }
                state.available -= units;
                return Some(Permit { pool: self, units });
            }
            if cancelled.load(Ordering::Acquire) {
                if interactive {
                    state.waiting_interactive -= 1;
                    self.wake.notify_all();
                }
                return None;
            }
            state = self
                .wake
                .wait_timeout(state, std::time::Duration::from_millis(20))
                .unwrap()
                .0;
        }
    }
}

impl Drop for Permit<'_> {
    fn drop(&mut self) {
        let mut state = self.pool.state.lock().unwrap();
        state.available = (state.available + self.units).min(self.pool.capacity);
        drop(state);
        self.pool.wake.notify_all();
    }
}

impl RenderPipeline {
    pub fn with_source_cache_capacity(
        cache_root: PathBuf,
        parallelism: usize,
        raw_parallelism: usize,
        source_cache_capacity: usize,
    ) -> Self {
        assert!(raw_parallelism > 0, "RAW parallelism must be positive");
        let backend = Arc::new(RenderBackend::new(cache_root));
        let metrics = Arc::new(PipelineMetrics::default());
        let (cache_writes, receiver) = mpsc::sync_channel::<CacheWrite>(2);
        let writer_backend = backend.clone();
        let writer_metrics = metrics.clone();
        let worker = thread::Builder::new()
            .name("rekesz-cache-writer".into())
            .spawn(move || {
                while let Ok(write) = receiver.recv() {
                    let started = Instant::now();
                    match writer_backend.store_cache(
                        &write.request,
                        &write.identity,
                        &write.rendered,
                    ) {
                        Ok(true) => {
                            writer_metrics.cache_writes.fetch_add(1, Ordering::Relaxed);
                        }
                        Ok(false) => {
                            writer_metrics
                                .rejected_cache_writes
                                .fetch_add(1, Ordering::Relaxed);
                        }
                        Err(_) => {
                            writer_metrics
                                .failed_cache_writes
                                .fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    writer_metrics.cache_encode_and_write.record(started);
                }
            })
            .expect("failed to start cache writer");
        let cache_writer = Arc::new(CacheWriter {
            sender: Mutex::new(Some(cache_writes)),
            worker: Mutex::new(Some(worker)),
            metrics: metrics.clone(),
        });
        Self {
            backend,
            decoded_sources: Arc::new(DecodedSourceCache::new(source_cache_capacity)),
            cache_reads: Arc::new(PermitPool::new(
                parallelism.clamp(1, 4) + usize::from(parallelism > 1),
                usize::from(parallelism > 1),
            )),
            cpu: Arc::new(PermitPool::new(
                parallelism.max(1),
                usize::from(parallelism > 1),
            )),
            raw_cpu: Arc::new(PermitPool::new(
                raw_parallelism,
                usize::from(raw_parallelism > 1),
            )),
            cache_writer,
            // Rawler uses Rayon internally. Install RAW work in a dedicated,
            // bounded pool so multiple scheduler workers do not each expand
            // into the unrestricted global pool.
            raw_pool: Arc::new(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(
                        std::thread::available_parallelism()
                            .map(usize::from)
                            .unwrap_or(1),
                    )
                    .thread_name(|index| format!("rekesz-raw-{index}"))
                    .build()
                    .expect("failed to start RAW processing pool"),
            ),
            metrics,
        }
    }

    pub fn stats(&self) -> PipelineStats {
        PipelineStats {
            decoded_source_hits: self.decoded_sources.hits(),
            ..self.metrics.snapshot()
        }
    }

    pub fn disk_cache_usage(&self) -> Result<DiskCacheUsage, String> {
        self.backend.disk_cache_usage()
    }

    pub fn set_source_cache_capacity(&self, capacity: usize) {
        self.decoded_sources.set_capacity(capacity);
    }

    #[cfg(test)]
    pub fn render(
        &self,
        path: PathBuf,
        variant: RenderVariant,
        interactive: bool,
        durable_cache_write: bool,
        cancelled: &Arc<AtomicBool>,
    ) -> Result<Arc<RenderedImage>, Arc<str>> {
        self.render_with_recipe(
            RenderInvocation {
                path,
                variant,
                render_identity: RenderIdentity::default(),
                recipe: None,
                interactive,
                durable_cache_write,
            },
            cancelled,
        )
    }

    pub fn render_with_recipe(
        &self,
        invocation: RenderInvocation<'_>,
        cancelled: &Arc<AtomicBool>,
    ) -> Result<Arc<RenderedImage>, Arc<str>> {
        let RenderInvocation {
            path,
            variant,
            render_identity,
            recipe,
            interactive,
            durable_cache_write,
        } = invocation;
        let plan = RenderPlan::compile(path.clone(), variant, render_identity, recipe)
            .map_err(Arc::from)?;
        let request = plan.request().clone();
        if cancelled.load(Ordering::Acquire) {
            return Err("render cancelled".into());
        }
        let (identity, cached) = {
            let Some(_permit) = self.cache_reads.acquire(1, interactive, cancelled) else {
                return Err("render cancelled".into());
            };
            if cancelled.load(Ordering::Acquire) {
                return Err("render cancelled".into());
            }
            let started = Instant::now();
            let identity = self.backend.source_identity(&request).unwrap_or(None);
            self.metrics.source_identity.record(started);
            let started = Instant::now();
            let cached = self.backend.lookup_cache(&request, identity.as_ref());
            self.metrics.cache_lookup.record(started);
            (identity, cached)
        };
        match cached {
            Ok(Some(rendered)) => {
                self.metrics.cache_hits.fetch_add(1, Ordering::Relaxed);
                if cancelled.load(Ordering::Acquire) {
                    return Err("render cancelled".into());
                }
                return Ok(Arc::new(rendered));
            }
            Ok(None) if variant == RenderVariant::Full => {
                self.metrics.cache_bypasses.fetch_add(1, Ordering::Relaxed);
            }
            Ok(None) => {
                self.metrics.cache_misses.fetch_add(1, Ordering::Relaxed);
            }
            Err(_) => {
                self.metrics
                    .cache_read_failures
                    .fetch_add(1, Ordering::Relaxed);
            }
        }
        if cancelled.load(Ordering::Acquire) {
            return Err("render cancelled".into());
        }
        let rendered = {
            let Some(_permit) = self.cpu.acquire(1, interactive, cancelled) else {
                return Err("render cancelled".into());
            };
            let _raw_permit = if plan.is_raw_decode() {
                let Some(permit) = self.raw_cpu.acquire(1, interactive, cancelled) else {
                    return Err("render cancelled".into());
                };
                Some(permit)
            } else {
                None
            };
            if cancelled.load(Ordering::Acquire) {
                return Err("render cancelled".into());
            }
            let decode_started = Instant::now();
            let decoded = if plan.caches_decoded_source() {
                let key = identity.clone().map(|identity| DecodedSourceKey {
                    path: path.clone(),
                    identity,
                });
                match key {
                    Some(key) => self.decoded_sources.get_or_decode(key, cancelled, || {
                        self.raw_pool
                            .install(|| self.backend.decode_source(&plan, cancelled))
                    }),
                    None => self
                        .raw_pool
                        .install(|| self.backend.decode_source(&plan, cancelled)),
                }
            } else {
                self.backend.decode_source(&plan, cancelled)
            };
            self.metrics.source_decode.record(decode_started);
            let transform_started = Instant::now();
            let rendered =
                decoded.and_then(|decoded| self.backend.render_source(&plan, decoded, cancelled));
            self.metrics.derivative_transform.record(transform_started);
            rendered
        };
        match rendered {
            Ok(rendered) => {
                if cancelled.load(Ordering::Acquire) {
                    return Err("render cancelled".into());
                }
                // Never delay pixels behind PNG encoding or filesystem writes.
                // If the tiny maintenance queue is full, this cache write is
                // disposable and a future request may regenerate it.
                let rendered = Arc::new(rendered);
                if plan.includes(RenderStage::CacheWriteEligible)
                    && let Some(identity) = identity
                {
                    self.cache_writer.write(
                        CacheWrite {
                            request,
                            identity,
                            rendered: rendered.clone(),
                        },
                        durable_cache_write,
                        cancelled,
                    );
                }
                Ok(rendered)
            }
            Err(error) => Err(format!("Cannot decode {}: {error}", path.display()).into()),
        }
    }
}
