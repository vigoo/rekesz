use std::{
    collections::{HashMap, VecDeque},
    path::PathBuf,
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc,
    },
    thread,
    time::Instant,
};

use gpui::RenderImage;
use image::{DynamicImage, Frame};

use crate::discovery::RequestId;
use crate::{
    formats::is_raw,
    rendering::{
        RenderBackend, RenderMetadata, RenderRequest, RenderVariant, RenderedImage, SourceIdentity,
    },
};

#[derive(Clone)]
pub enum PreviewState {
    Unloaded,
    Loading(RequestId),
    Loaded(Arc<DecodedPreview>),
    Error(Arc<str>),
}

pub struct DecodedPreview {
    pub image: Arc<RenderImage>,
    pub metadata: RenderMetadata,
}

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

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct DecodedSourceKey {
    path: PathBuf,
    identity: SourceIdentity,
}

enum DecodedSourceEntry {
    Loading,
    Ready(Arc<crate::rendering::source::DecodedSource>),
}

struct DecodedSourceCache {
    entries: Mutex<HashMap<DecodedSourceKey, DecodedSourceEntry>>,
    order: Mutex<VecDeque<DecodedSourceKey>>,
    wake: Condvar,
    capacity: usize,
    hits: AtomicU64,
}

impl DecodedSourceCache {
    fn new(capacity: usize) -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
            order: Mutex::new(VecDeque::new()),
            wake: Condvar::new(),
            capacity: capacity.max(1),
            hits: AtomicU64::new(0),
        }
    }

    fn get_or_decode(
        &self,
        key: DecodedSourceKey,
        cancelled: &AtomicBool,
        decode: impl FnOnce() -> image::ImageResult<Arc<crate::rendering::source::DecodedSource>>,
    ) -> image::ImageResult<Arc<crate::rendering::source::DecodedSource>> {
        let mut decode = Some(decode);
        loop {
            let mut entries = self.entries.lock().unwrap();
            match entries.get(&key) {
                Some(DecodedSourceEntry::Ready(source)) => {
                    self.hits.fetch_add(1, Ordering::Relaxed);
                    return Ok(source.clone());
                }
                Some(DecodedSourceEntry::Loading) => {
                    if cancelled.load(Ordering::Acquire) {
                        return Err(cancelled_image_error());
                    }
                    drop(
                        self.wake
                            .wait_timeout(entries, std::time::Duration::from_millis(20))
                            .unwrap(),
                    );
                }
                None => {
                    entries.insert(key.clone(), DecodedSourceEntry::Loading);
                    drop(entries);
                    let result = decode.take().expect("decode closure used once")();
                    let mut entries = self.entries.lock().unwrap();
                    match &result {
                        Ok(source) => {
                            entries.insert(key.clone(), DecodedSourceEntry::Ready(source.clone()));
                            let mut order = self.order.lock().unwrap();
                            order.retain(|candidate| candidate != &key);
                            order.push_back(key.clone());
                            while order.len() > self.capacity {
                                if let Some(oldest) = order.pop_front()
                                    && oldest != key
                                {
                                    entries.remove(&oldest);
                                }
                            }
                        }
                        Err(_) => {
                            entries.remove(&key);
                        }
                    }
                    drop(entries);
                    self.wake.notify_all();
                    return result;
                }
            }
        }
    }
}

fn cancelled_image_error() -> image::ImageError {
    image::ImageError::Decoding(image::error::DecodingError::new(
        image::error::ImageFormatHint::Unknown,
        "render cancelled",
    ))
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
    pub ui_image_conversion: StageTiming,
    pub cache_encode_and_write: StageTiming,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct StageTiming {
    pub calls: u64,
    pub total_nanos: u64,
}

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
    ui_image_conversion: AtomicStageTiming,
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
            ui_image_conversion: self.ui_image_conversion.snapshot(),
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
        if std::env::var_os("PICS_DIAGNOSTICS").is_some() {
            let stats = self.metrics.snapshot();
            eprintln!(
                "pics cache: hits={}, misses={}, bypasses={}, read-failures={}, writes={}, dropped={}, rejected={}, write-failures={}; average-us: identity={}, cache-lookup={}, source-decode={}, derivative-transform={}, ui-conversion={}, cache-write={}",
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
                stats.ui_image_conversion.average_nanos() / 1_000,
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
            .name("pics-cache-writer".into())
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
                    .thread_name(|index| format!("pics-raw-{index}"))
                    .build()
                    .expect("failed to start RAW processing pool"),
            ),
            metrics,
        }
    }

    #[cfg(test)]
    pub fn stats(&self) -> PipelineStats {
        PipelineStats {
            decoded_source_hits: self.decoded_sources.hits.load(Ordering::Relaxed),
            ..self.metrics.snapshot()
        }
    }

    pub fn render(
        &self,
        path: PathBuf,
        variant: RenderVariant,
        interactive: bool,
        durable_cache_write: bool,
        cancelled: &Arc<AtomicBool>,
    ) -> PreviewState {
        let request = RenderRequest::new(path.clone(), variant);
        if cancelled.load(Ordering::Acquire) {
            return PreviewState::Error("render cancelled".into());
        }
        let (identity, cached) = {
            let Some(_permit) = self.cache_reads.acquire(1, interactive, cancelled) else {
                return PreviewState::Error("render cancelled".into());
            };
            if cancelled.load(Ordering::Acquire) {
                return PreviewState::Error("render cancelled".into());
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
                    return PreviewState::Error("render cancelled".into());
                }
                return decoded_preview(&rendered, &self.metrics);
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
            return PreviewState::Error("render cancelled".into());
        }
        let rendered = {
            let Some(_permit) = self.cpu.acquire(1, interactive, cancelled) else {
                return PreviewState::Error("render cancelled".into());
            };
            let _raw_permit = if is_raw(&path) {
                let Some(permit) = self.raw_cpu.acquire(1, interactive, cancelled) else {
                    return PreviewState::Error("render cancelled".into());
                };
                Some(permit)
            } else {
                None
            };
            if cancelled.load(Ordering::Acquire) {
                return PreviewState::Error("render cancelled".into());
            }
            let decode_started = Instant::now();
            let decoded = if is_raw(&path) {
                let key = identity.clone().map(|identity| DecodedSourceKey {
                    path: path.clone(),
                    identity,
                });
                match key {
                    Some(key) => self.decoded_sources.get_or_decode(key, cancelled, || {
                        self.raw_pool
                            .install(|| self.backend.decode_source(&request, cancelled))
                    }),
                    None => self
                        .raw_pool
                        .install(|| self.backend.decode_source(&request, cancelled)),
                }
            } else {
                self.backend.decode_source(&request, cancelled)
            };
            self.metrics.source_decode.record(decode_started);
            let transform_started = Instant::now();
            let rendered =
                decoded.and_then(|decoded| self.backend.render_source(variant, decoded, cancelled));
            self.metrics.derivative_transform.record(transform_started);
            rendered
        };
        match rendered {
            Ok(rendered) => {
                if cancelled.load(Ordering::Acquire) {
                    return PreviewState::Error("render cancelled".into());
                }
                // Never delay pixels behind PNG encoding or filesystem writes.
                // If the tiny maintenance queue is full, this cache write is
                // disposable and a future request may regenerate it.
                let rendered = Arc::new(rendered);
                if variant != RenderVariant::Full
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
                decoded_preview(&rendered, &self.metrics)
            }
            Err(error) => {
                PreviewState::Error(format!("Cannot decode {}: {error}", path.display()).into())
            }
        }
    }
}

fn decoded_preview(rendered: &RenderedImage, metrics: &PipelineMetrics) -> PreviewState {
    let started = Instant::now();
    let image = Arc::new(render_image(&rendered.image));
    metrics.ui_image_conversion.record(started);
    PreviewState::Loaded(Arc::new(DecodedPreview {
        image,
        metadata: rendered.metadata,
    }))
}

/// Independent resource cache. Future thumbnails and metadata can follow the
/// same lifecycle without becoming part of the tree projection.
pub struct PreviewCache {
    entries: HashMap<PathBuf, PreviewState>,
    last_used: HashMap<PathBuf, u64>,
    capacity: usize,
    next_request: RequestId,
    access_clock: u64,
}

impl PreviewCache {
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "preview cache capacity must be positive");
        Self {
            entries: HashMap::new(),
            last_used: HashMap::new(),
            capacity,
            next_request: 0,
            access_clock: 0,
        }
    }

    pub fn state(&self, path: &std::path::Path) -> PreviewState {
        self.entries
            .get(path)
            .cloned()
            .unwrap_or(PreviewState::Unloaded)
    }

    pub fn begin(&mut self, path: &std::path::Path, variant: RenderVariant) -> Option<RequestId> {
        if matches!(self.state(path), PreviewState::Loaded(ref preview) if variant_satisfies(preview.metadata.variant, variant))
        {
            self.touch(path);
            return None;
        }
        if matches!(self.state(path), PreviewState::Loaded(_)) {
            self.entries.remove(path);
            self.last_used.remove(path);
        }
        if !matches!(self.state(path), PreviewState::Unloaded) {
            return None;
        }
        self.next_request += 1;
        let request = self.next_request;
        self.touch(path);
        self.entries
            .insert(path.to_path_buf(), PreviewState::Loading(request));
        Some(request)
    }

    pub fn needs_work(&self, path: &std::path::Path, variant: RenderVariant) -> bool {
        match self.state(path) {
            PreviewState::Loaded(preview) => !variant_satisfies(preview.metadata.variant, variant),
            PreviewState::Unloaded | PreviewState::Loading(_) => true,
            PreviewState::Error(_) => false,
        }
    }

    pub fn cancel(&mut self, path: &std::path::Path, request: RequestId) -> bool {
        if !matches!(self.state(path), PreviewState::Loading(active) if active == request) {
            return false;
        }
        self.entries.remove(path);
        self.last_used.remove(path);
        true
    }

    pub fn finish(
        &mut self,
        path: &std::path::Path,
        request: RequestId,
        result: PreviewState,
    ) -> bool {
        if !matches!(self.state(path), PreviewState::Loading(active) if active == request) {
            return false;
        }
        let is_loaded = matches!(result, PreviewState::Loaded(_));
        self.entries.insert(path.to_path_buf(), result);
        if is_loaded {
            self.evict_over_capacity();
        } else {
            self.last_used.remove(path);
        }
        true
    }

    pub fn mark_used(&mut self, path: &std::path::Path) {
        if matches!(self.state(path), PreviewState::Loaded(_)) {
            self.touch(path);
        }
    }

    fn touch(&mut self, path: &std::path::Path) {
        self.access_clock += 1;
        self.last_used.insert(path.to_path_buf(), self.access_clock);
    }

    fn evict_over_capacity(&mut self) {
        while self
            .entries
            .values()
            .filter(|state| matches!(state, PreviewState::Loaded(_)))
            .count()
            > self.capacity
        {
            let evicted = self
                .entries
                .iter()
                .filter(|(_, state)| matches!(state, PreviewState::Loaded(_)))
                .min_by_key(|(id, _)| self.last_used.get(*id).copied().unwrap_or(0))
                .map(|(id, _)| id.clone());
            if let Some(evicted) = evicted {
                self.entries.remove(&evicted);
                self.last_used.remove(&evicted);
            }
        }
    }
}

fn variant_satisfies(cached: RenderVariant, requested: RenderVariant) -> bool {
    match (cached, requested) {
        (
            RenderVariant::Preview {
                max_width: cached_width,
                max_height: cached_height,
            },
            RenderVariant::Preview {
                max_width,
                max_height,
            },
        ) => cached_width >= max_width && cached_height >= max_height,
        _ => cached == requested,
    }
}

fn render_image(decoded: &DynamicImage) -> RenderImage {
    let mut buffer = decoded.to_rgba8();
    for pixel in buffer.as_chunks_mut::<4>().0 {
        pixel.swap(0, 2);
    }
    RenderImage::new(smallvec::smallvec![Frame::new(buffer)])
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgba};
    use std::{path::Path, sync::mpsc, time::Duration};

    fn image_state() -> PreviewState {
        image_state_for(RenderVariant::Full)
    }

    fn image_state_for(variant: RenderVariant) -> PreviewState {
        let buffer = ImageBuffer::from_pixel(1, 1, Rgba([0, 0, 0, 255]));
        PreviewState::Loaded(Arc::new(DecodedPreview {
            image: Arc::new(RenderImage::new(smallvec::smallvec![Frame::new(buffer)])),
            metadata: RenderMetadata {
                source_pipeline: crate::rendering::SourcePipeline::Standard,
                photographic: None,
                source_dimensions: (1, 1),
                output_dimensions: (1, 1),
                variant,
                orientation_applied: None,
            },
        }))
    }

    #[test]
    fn a_larger_raw_preview_satisfies_a_smaller_display_request() {
        let mut cache = PreviewCache::new(1);
        let path = Path::new("raw.cr3");
        let large = RenderVariant::Preview {
            max_width: 3840,
            max_height: 2160,
        };
        let request = cache.begin(path, large).unwrap();
        cache.finish(path, request, image_state_for(large));

        assert!(
            cache
                .begin(
                    path,
                    RenderVariant::Preview {
                        max_width: 1920,
                        max_height: 1080,
                    }
                )
                .is_none()
        );
        assert!(
            cache
                .begin(
                    path,
                    RenderVariant::Preview {
                        max_width: 5120,
                        max_height: 2880,
                    }
                )
                .is_some()
        );
    }

    fn load(cache: &mut PreviewCache, path: &std::path::Path) {
        let request = cache.begin(path, RenderVariant::Full).unwrap();
        assert!(cache.finish(path, request, image_state()));
    }

    #[test]
    fn evicts_the_least_recently_used_loaded_image() {
        let mut cache = PreviewCache::new(2);
        let first = Path::new("first");
        let second = Path::new("second");
        let third = Path::new("third");
        load(&mut cache, first);
        load(&mut cache, second);
        cache.mark_used(first);
        load(&mut cache, third);

        assert!(matches!(cache.state(first), PreviewState::Loaded(_)));
        assert!(matches!(cache.state(second), PreviewState::Unloaded));
        assert!(matches!(cache.state(third), PreviewState::Loaded(_)));
    }

    #[test]
    fn cheap_negative_results_do_not_consume_image_capacity() {
        let mut cache = PreviewCache::new(1);
        let missing = Path::new("missing");
        let image = Path::new("image");
        let request = cache.begin(missing, RenderVariant::Full).unwrap();
        cache.finish(missing, request, PreviewState::Error("missing".into()));
        load(&mut cache, image);

        assert!(matches!(cache.state(missing), PreviewState::Error(_)));
        assert!(matches!(cache.state(image), PreviewState::Loaded(_)));
    }

    #[test]
    fn completion_order_does_not_override_usage_order() {
        let mut cache = PreviewCache::new(1);
        let first = Path::new("first");
        let second = Path::new("second");
        let first_request = cache.begin(first, RenderVariant::Full).unwrap();
        let second_request = cache.begin(second, RenderVariant::Full).unwrap();
        cache.finish(second, second_request, image_state());
        cache.finish(first, first_request, image_state());

        assert!(matches!(cache.state(first), PreviewState::Unloaded));
        assert!(matches!(cache.state(second), PreviewState::Loaded(_)));
    }

    #[test]
    fn cancelling_a_matching_request_restores_unloaded_state() {
        let mut cache = PreviewCache::new(1);
        let path = Path::new("cancelled");
        let request = cache.begin(path, RenderVariant::Full).unwrap();
        assert!(cache.cancel(path, request));
        assert!(matches!(cache.state(path), PreviewState::Unloaded));
        assert!(!cache.cancel(path, request));
    }

    #[test]
    fn interactive_permits_bypass_the_background_reserve() {
        let pool = Arc::new(PermitPool::new(4, 2));
        let cancellation = AtomicBool::new(false);
        let background = pool.acquire(2, false, &cancellation).unwrap();
        let (sender, receiver) = mpsc::channel();
        let contender = {
            let pool = pool.clone();
            std::thread::spawn(move || {
                let cancellation = AtomicBool::new(false);
                let _permit = pool.acquire(2, false, &cancellation).unwrap();
                sender.send("background").unwrap();
            })
        };
        assert!(receiver.recv_timeout(Duration::from_millis(50)).is_err());
        let interactive = pool.acquire(2, true, &cancellation).unwrap();
        drop(interactive);
        assert!(receiver.recv_timeout(Duration::from_millis(50)).is_err());
        drop(background);
        assert_eq!(
            receiver.recv_timeout(Duration::from_secs(1)).unwrap(),
            "background"
        );
        contender.join().unwrap();
    }

    #[test]
    fn permit_wait_is_cooperatively_cancellable() {
        let pool = Arc::new(PermitPool::new(1, 0));
        let cancellation = Arc::new(AtomicBool::new(false));
        let held = pool.acquire(1, false, &cancellation).unwrap();
        let waiter = {
            let pool = pool.clone();
            let cancellation = cancellation.clone();
            std::thread::spawn(move || pool.acquire(1, false, &cancellation).is_none())
        };
        std::thread::sleep(Duration::from_millis(30));
        cancellation.store(true, Ordering::Release);
        assert!(waiter.join().unwrap());
        drop(held);
    }

    #[test]
    fn durable_background_writes_wait_for_bounded_queue_capacity() {
        let directory = tempfile::tempdir().unwrap();
        let source = directory.path().join("source.png");
        std::fs::write(&source, b"identity").unwrap();
        let backend = RenderBackend::new(directory.path().join("cache"));
        let request = RenderRequest::new(source, RenderVariant::Thumbnail { width: 4 });
        let identity = backend.source_identity(&request).unwrap().unwrap();
        let rendered = Arc::new(RenderedImage {
            image: DynamicImage::ImageRgba8(ImageBuffer::from_pixel(1, 1, Rgba([1, 2, 3, 255]))),
            metadata: RenderMetadata {
                source_pipeline: crate::rendering::SourcePipeline::Standard,
                photographic: None,
                source_dimensions: (1, 1),
                output_dimensions: (1, 1),
                variant: request.variant,
                orientation_applied: None,
            },
        });
        let make_write = || CacheWrite {
            request: request.clone(),
            identity: identity.clone(),
            rendered: rendered.clone(),
        };
        let (sender, receiver) = mpsc::sync_channel(1);
        assert!(sender.try_send(make_write()).is_ok());
        let writer = Arc::new(CacheWriter {
            sender: Mutex::new(Some(sender)),
            worker: Mutex::new(None),
            metrics: Arc::new(PipelineMetrics::default()),
        });
        let cancellation = Arc::new(AtomicBool::new(false));
        let waiter = {
            let writer = writer.clone();
            let cancellation = cancellation.clone();
            let write = make_write();
            std::thread::spawn(move || writer.write(write, true, &cancellation))
        };
        std::thread::sleep(Duration::from_millis(30));
        assert!(!waiter.is_finished());
        receiver.recv().unwrap();
        waiter.join().unwrap();
        receiver.recv_timeout(Duration::from_secs(1)).unwrap();
    }

    #[test]
    fn staged_pipeline_persists_a_generated_thumbnail() {
        let directory = tempfile::tempdir().unwrap();
        let source = directory.path().join("source.png");
        DynamicImage::ImageRgba8(ImageBuffer::from_pixel(8, 8, Rgba([1, 2, 3, 255])))
            .save(&source)
            .unwrap();
        let pipeline =
            RenderPipeline::with_source_cache_capacity(directory.path().join("cache"), 2, 2, 1);
        let variant = RenderVariant::Thumbnail { width: 4 };
        assert!(matches!(
            pipeline.render(
                source.clone(),
                variant,
                true,
                false,
                &Arc::new(AtomicBool::new(false))
            ),
            PreviewState::Loaded(_)
        ));
        assert_eq!(pipeline.stats().cache_misses, 1);
        let request = RenderRequest::new(source, variant);
        let identity = pipeline.backend.source_identity(&request).unwrap().unwrap();
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        while std::time::Instant::now() < deadline {
            if pipeline
                .backend
                .lookup_cache(&request, Some(&identity))
                .unwrap()
                .is_some()
            {
                return;
            }
            std::thread::yield_now();
        }
        panic!("deferred cache write did not become visible");
    }

    #[test]
    fn decodes_the_cr3_regression_sample() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test/IMG_0052.CR3");
        if path.exists() {
            let cache_root = tempfile::tempdir().unwrap();
            let pipeline = RenderPipeline::with_source_cache_capacity(
                cache_root.path().to_path_buf(),
                2,
                2,
                1,
            );
            let result = pipeline.render(
                path.clone(),
                RenderVariant::Thumbnail { width: 64 },
                true,
                false,
                &Arc::new(AtomicBool::new(false)),
            );
            if let PreviewState::Error(error) = result {
                panic!("CR3 sample did not decode: {error}");
            }
            let result = pipeline.render(
                path,
                RenderVariant::Preview {
                    max_width: 320,
                    max_height: 240,
                },
                true,
                false,
                &Arc::new(AtomicBool::new(false)),
            );
            assert!(matches!(result, PreviewState::Loaded(_)));
            assert_eq!(pipeline.stats().decoded_source_hits, 1);
        }
    }

    #[test]
    fn cancels_cr3_work_through_the_rawler_token() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test/IMG_0052.CR3");
        if !path.exists() {
            return;
        }
        let cache_root = tempfile::tempdir().unwrap();
        let pipeline =
            RenderPipeline::with_source_cache_capacity(cache_root.path().to_path_buf(), 2, 2, 1);
        let cancellation = Arc::new(AtomicBool::new(false));
        let worker_cancellation = cancellation.clone();
        let worker = std::thread::spawn(move || {
            pipeline.render(
                path,
                RenderVariant::Thumbnail { width: 64 },
                true,
                false,
                &worker_cancellation,
            )
        });
        std::thread::sleep(Duration::from_millis(10));
        cancellation.store(true, Ordering::Release);
        assert!(matches!(worker.join().unwrap(), PreviewState::Error(_)));
    }
}
