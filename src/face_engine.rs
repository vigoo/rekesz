use std::{
    collections::{BTreeSet, HashMap},
    hash::{Hash, Hasher},
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicBool, Ordering},
    },
};

use anyhow::{Context, Result, bail};
use image::imageops::FilterType;
use image::{DynamicImage, GenericImageView, Rgb, RgbImage};
use ort::{session::Session, value::Tensor};
use rusqlite::{Connection, OptionalExtension, params};
use sha2::{Digest, Sha256};

use crate::{
    catalog,
    config::FaceGpuPolicy,
    jobs::JobLimiter,
    rendering::{
        DECODER_MODEL_ID, RenderIdentity, RenderVariant,
        pipeline::{RenderInvocation, RenderPipeline},
    },
    scheduler::{
        MaintenanceAdmission, MaintenanceJobStatus, MaintenanceScheduler, MaintenanceStats,
    },
};

const YUNET: &[u8] = include_bytes!("../assets/face/face_detection_yunet_2023mar.onnx");
const SFACE: &[u8] = include_bytes!("../assets/face/face_recognition_sface_2021dec.onnx");
const YUNET_LICENSE: &[u8] = include_bytes!("../assets/face/LICENSE-YUNET");
const SFACE_LICENSE: &[u8] = include_bytes!("../assets/face/LICENSE-SFACE");
const YUNET_SHA256: &str = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4";
const SFACE_SHA256: &str = "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79";
const YUNET_LICENSE_SHA256: &str =
    "c83b8120c50ccbd4c4f96edf53141bdd566ebb8f8e9227e415326aa1b1aba958";
const SFACE_LICENSE_SHA256: &str =
    "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30";
const ANALYSIS_MODEL_ID: &str = "yunet-2023mar+sface-2021dec";
const ANALYSIS_MAX_DIMENSION: u32 = 2048;
const ANALYSIS_MAX_SOURCE_BYTES: u64 = 512 * 1024 * 1024;
const MAX_AUTOMATIC_ANALYSIS_ATTEMPTS: i64 = 5;

/// Targets whose bundled ONNX Runtime build ships the CUDA execution
/// provider; mirrors the target-scoped `ort/cuda` dependency in Cargo.toml.
#[cfg_attr(test, allow(dead_code))]
const CUDA_EP_COMPILED: bool = cfg!(all(
    any(target_os = "linux", target_os = "windows"),
    target_arch = "x86_64"
));

/// Which device the bundled face models run on.
#[cfg_attr(test, allow(dead_code))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum FaceComputeDevice {
    Cpu,
    Cuda,
}

#[cfg_attr(test, allow(dead_code))]
impl FaceComputeDevice {
    fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
        }
    }
}

/// The automatic device probe's verdict, process-wide, so every analysis
/// worker lands on the same device instead of each re-attempting (and
/// re-diagnosing) a failing CUDA session. The probe is unreferenced in test
/// builds, which pin the device to CPU for deterministic embeddings.
#[cfg_attr(test, allow(dead_code))]
static AUTO_COMPUTE_DEVICE: OnceLock<FaceComputeDevice> = OnceLock::new();

#[cfg(test)]
pub(crate) static ORT_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
static FORCED_SETUP_FAILURE: std::sync::LazyLock<Mutex<Option<(PathBuf, i64)>>> =
    std::sync::LazyLock::new(|| Mutex::new(None));

#[cfg(test)]
pub(crate) fn install_face_engine_setup_failure(root: PathBuf, asset_id: i64) {
    *FORCED_SETUP_FAILURE.lock().unwrap() = Some((root, asset_id));
}

#[cfg(test)]
fn take_face_engine_setup_failure(job: &FaceAnalyzeJob) -> bool {
    let mut forced = FORCED_SETUP_FAILURE.lock().unwrap();
    if forced.as_ref() == Some(&(job.root.clone(), job.asset_id)) {
        forced.take();
        true
    } else {
        false
    }
}

/// Sealed marker carried through every local inference session. Face inference
/// has no transport capability; the Linux integration test additionally runs
/// the real model with socket syscalls denied by the kernel.
#[derive(Clone, Copy)]
struct OfflineNetworkCapability;

impl OfflineNetworkCapability {
    fn enter(self) {
        let Self = self;
    }
}

fn defer_identity_mismatch(
    connection: &Connection,
    asset_id: i64,
    source_size: i64,
    source_modified_ns: i64,
    source_fingerprint: &[u8],
    message: &str,
) -> Result<()> {
    connection.execute(
        "INSERT INTO face_analysis_state(asset_id, detector_model_id, recognizer_model_id,
             decoder_model_id, source_size, source_modified_ns, source_fingerprint,
             status, attempts, error, failure_class, retry_after, updated_at)
         VALUES(?1, ?2, 'sface-2021dec', ?3, ?4, ?5, ?6, 'error', 0, ?7,
                'deterministic', NULL, datetime('now'))
         ON CONFLICT(asset_id) DO UPDATE SET status='error', error=excluded.error,
           detector_model_id=excluded.detector_model_id,
           recognizer_model_id=excluded.recognizer_model_id,
           decoder_model_id=excluded.decoder_model_id,
           source_size=excluded.source_size, source_modified_ns=excluded.source_modified_ns,
           source_fingerprint=excluded.source_fingerprint,
           failure_class=excluded.failure_class, retry_after=NULL,
           worker_id=NULL, lease_expires_at=NULL,
           updated_at=excluded.updated_at WHERE face_analysis_state.status!='running'",
        params![
            asset_id,
            ANALYSIS_MODEL_ID,
            DECODER_MODEL_ID,
            source_size,
            source_modified_ns,
            source_fingerprint,
            message
        ],
    )?;
    Ok(())
}

struct ClaimGuard<'a> {
    connection: &'a Connection,
    asset_id: i64,
    worker_id: &'a str,
    source_size: i64,
    source_modified_ns: i64,
    source_fingerprint: Vec<u8>,
}

struct AnalysisCandidate {
    asset_id: i64,
    relative_path: Vec<u8>,
}

#[derive(Clone, Copy)]
struct AnalyzedSourceIdentity<'a> {
    size: i64,
    modified_ns: i64,
    fingerprint: &'a [u8],
    path: Option<&'a Path>,
}

struct AnalysisFailure<'a> {
    category: &'a str,
    class: AnalysisFailureClass,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AnalysisFailureClass {
    Transient,
    Deterministic,
}

impl AnalysisFailureClass {
    fn as_db(self) -> &'static str {
        match self {
            Self::Transient => "transient",
            Self::Deterministic => "deterministic",
        }
    }
}

fn record_analysis_failure(
    connection: &Connection,
    asset_id: i64,
    worker_id: &str,
    source: AnalyzedSourceIdentity<'_>,
    failure: AnalysisFailure<'_>,
) -> Result<()> {
    connection.execute(
        "UPDATE face_analysis_state SET status='error', error=?1,
            attempts=attempts+1, worker_id=NULL, lease_expires_at=NULL,
            failure_class=CASE WHEN ?6='deterministic' THEN 'deterministic'
              WHEN attempts+1>=?7 THEN 'exhausted' ELSE 'transient' END,
            retry_after=CASE WHEN ?6='deterministic' OR attempts+1>=?7 THEN NULL
              ELSE datetime('now', printf('+%d seconds',
                min(3600, 1 << min(10, attempts+1)))) END,
            updated_at=datetime('now') WHERE asset_id=?2 AND worker_id=?3
            AND source_size=?4 AND source_modified_ns=?5 AND source_fingerprint=?8",
        params![
            failure.category,
            asset_id,
            worker_id,
            source.size,
            source.modified_ns,
            failure.class.as_db(),
            MAX_AUTOMATIC_ANALYSIS_ATTEMPTS,
            source.fingerprint
        ],
    )?;
    Ok(())
}

struct AnalysisStageGuard {
    stages: Arc<Mutex<HashMap<FaceAnalyzeJob, FaceAnalysisStage>>>,
    job: FaceAnalyzeJob,
}

/// Injectable boundary observer used by the production analysis control flow.
/// The live implementation is a no-op; deterministic tests use it to stop a
/// real claimed job exactly between expensive stages.
pub(crate) trait ProductionAnalysisStages: Send + Sync {
    fn reached(&self, _stage: FaceAnalysisStage, _cancelled: &Arc<AtomicBool>) {}
    fn inference_started(&self) {}
    fn inference_completed(&self) {}
}

struct LiveAnalysisStages;

impl ProductionAnalysisStages for LiveAnalysisStages {}

struct InstrumentedAnalysisStages {
    observer: Arc<dyn ProductionAnalysisStages>,
    instrumentation: Arc<FaceAnalysisInstrumentation>,
}

impl ProductionAnalysisStages for InstrumentedAnalysisStages {
    fn reached(&self, stage: FaceAnalysisStage, cancelled: &Arc<AtomicBool>) {
        self.observer.reached(stage, cancelled);
    }

    fn inference_started(&self) {
        self.instrumentation
            .inference_started
            .fetch_add(1, Ordering::AcqRel);
        self.observer.inference_started();
    }

    fn inference_completed(&self) {
        self.instrumentation
            .inference_completed
            .fetch_add(1, Ordering::AcqRel);
        self.observer.inference_completed();
    }
}

#[derive(Clone, Copy)]
struct AnalysisProgress<'a> {
    display: Option<(
        &'a Mutex<HashMap<FaceAnalyzeJob, FaceAnalysisStage>>,
        &'a FaceAnalyzeJob,
    )>,
    hooks: &'a dyn ProductionAnalysisStages,
}

impl Drop for AnalysisStageGuard {
    fn drop(&mut self) {
        self.stages.lock().unwrap().remove(&self.job);
    }
}

impl Drop for ClaimGuard<'_> {
    fn drop(&mut self) {
        let _ = self.connection.execute(
            "UPDATE face_analysis_state SET status='pending', worker_id=NULL,
             lease_expires_at=NULL, updated_at=datetime('now') WHERE asset_id=?1
             AND status='running' AND worker_id=?2 AND source_size=?3
             AND source_modified_ns=?4 AND source_fingerprint=?5",
            params![
                self.asset_id,
                self.worker_id,
                self.source_size,
                self.source_modified_ns,
                self.source_fingerprint
            ],
        );
    }
}

#[derive(Clone, Debug, Eq)]
pub struct FaceAnalyzeJob {
    pub root: PathBuf,
    pub asset_id: i64,
    /// Asset's library-relative path, for queue display; not part of the
    /// job's identity.
    pub label: Option<Arc<str>>,
}

impl PartialEq for FaceAnalyzeJob {
    fn eq(&self, other: &Self) -> bool {
        self.root == other.root && self.asset_id == other.asset_id
    }
}

impl Hash for FaceAnalyzeJob {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.root.hash(state);
        self.asset_id.hash(state);
    }
}

/// Typed background service for face analysis. It shares the application's
/// admission limiter, so maintenance inference leaves a slot for foreground
/// work. One engine/session pair is created per active worker and the scheduler
/// parallelism is therefore also the explicit inference-session cap.
pub struct FaceAnalysisService {
    scheduler: MaintenanceScheduler<FaceAnalyzeJob>,
    paused: Arc<AtomicBool>,
    enabled: bool,
    stages: Arc<Mutex<HashMap<FaceAnalyzeJob, FaceAnalysisStage>>>,
    instrumentation: Arc<FaceAnalysisInstrumentation>,
}

/// Runtime counters for validating and diagnosing the bounded inference pool.
/// These counters instrument the production worker pool itself (one
/// session per worker, up to `parallelism` concurrent inferences), and are
/// intentionally cheap enough to leave on.
#[derive(Default)]
pub struct FaceAnalysisInstrumentation {
    engines_constructed: std::sync::atomic::AtomicUsize,
    active_inference: std::sync::atomic::AtomicUsize,
    max_active_inference: std::sync::atomic::AtomicUsize,
    inference_started: std::sync::atomic::AtomicUsize,
    inference_completed: std::sync::atomic::AtomicUsize,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct FaceAnalysisInstrumentationSnapshot {
    pub engines_constructed: usize,
    pub active_inference: usize,
    pub max_active_inference: usize,
    pub inference_started: usize,
    pub inference_completed: usize,
}

struct ActiveInferenceGuard<'a>(&'a FaceAnalysisInstrumentation);

impl Drop for ActiveInferenceGuard<'_> {
    fn drop(&mut self) {
        self.0.active_inference.fetch_sub(1, Ordering::AcqRel);
    }
}

impl FaceAnalysisInstrumentation {
    fn enter(&self) -> ActiveInferenceGuard<'_> {
        let active = self.active_inference.fetch_add(1, Ordering::AcqRel) + 1;
        self.max_active_inference
            .fetch_max(active, Ordering::AcqRel);
        ActiveInferenceGuard(self)
    }

    fn snapshot(&self) -> FaceAnalysisInstrumentationSnapshot {
        FaceAnalysisInstrumentationSnapshot {
            engines_constructed: self.engines_constructed.load(Ordering::Acquire),
            active_inference: self.active_inference.load(Ordering::Acquire),
            max_active_inference: self.max_active_inference.load(Ordering::Acquire),
            inference_started: self.inference_started.load(Ordering::Acquire),
            inference_completed: self.inference_completed.load(Ordering::Acquire),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FaceAnalysisStage {
    Queued,
    Initializing,
    Claiming,
    Decoding,
    Detecting,
    Embedding,
    Reconciling,
    Committing,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FaceAnalysisJobSnapshot {
    pub job: FaceAnalyzeJob,
    pub status: MaintenanceJobStatus,
    pub stage: FaceAnalysisStage,
    pub rank: i32,
    pub cancelled: bool,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct RecognitionRefreshJob {
    pub root: PathBuf,
}

/// Runs durable recognition refresh batches under the same foreground admission
/// policy as face inference. Each submitted catalog is a deduplicated job and
/// processes one bounded batch, allowing periodic refill to yield between batches.
pub struct RecognitionRefreshService {
    scheduler: Arc<MaintenanceScheduler<RecognitionRefreshJob>>,
    paused: Arc<AtomicBool>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RecognitionRefreshJobSnapshot {
    pub job: RecognitionRefreshJob,
    pub status: MaintenanceJobStatus,
    pub rank: i32,
    pub cancelled: bool,
}

impl Clone for RecognitionRefreshService {
    fn clone(&self) -> Self {
        Self {
            scheduler: self.scheduler.clone(),
            paused: self.paused.clone(),
        }
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct PeopleSummaryRebuildJob {
    pub root: PathBuf,
}

/// Typed scheduler integration for the durable schema-v5 People summary queue.
/// The queue is authoritative: periodic callers use `refill`, so mutations made
/// only by SQLite triggers are observed without opening the People view.
pub struct PeopleSummaryRebuildService {
    scheduler: MaintenanceScheduler<PeopleSummaryRebuildJob>,
    paused: Arc<AtomicBool>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PeopleSummaryRebuildJobSnapshot {
    pub job: PeopleSummaryRebuildJob,
    pub status: MaintenanceJobStatus,
    pub rank: i32,
    pub cancelled: bool,
}

impl PeopleSummaryRebuildService {
    pub fn new(admission: MaintenanceAdmission) -> Self {
        let paused = Arc::new(AtomicBool::new(false));
        let executor_paused = paused.clone();
        let executor_admission = admission.clone();
        let scheduler = MaintenanceScheduler::new_with_admission(
            1,
            admission,
            Arc::new(move |job: &PeopleSummaryRebuildJob, cancelled| {
                if executor_paused.load(Ordering::Acquire)
                    || !executor_admission.checkpoint(cancelled)
                {
                    return Ok(());
                }
                let preempt_admission = executor_admission.clone();
                catalog::people::rebuild_people_summary_with_preemption(
                    &job.root,
                    cancelled.clone(),
                    &|cancelled| executor_admission.checkpoint(cancelled),
                    Arc::new(move || preempt_admission.is_foreground_blocked()),
                )?;
                let _ = executor_admission.checkpoint(cancelled);
                Ok(())
            }),
        );
        Self { scheduler, paused }
    }

    pub fn enqueue(&self, root: PathBuf, rank: i32) -> bool {
        !self.paused.load(Ordering::Acquire)
            && self
                .scheduler
                .submit(PeopleSummaryRebuildJob { root }, rank)
    }

    pub fn set_paused(&self, paused: bool) {
        self.paused.store(paused, Ordering::Release);
        if paused {
            self.scheduler.cancel_all();
        }
    }

    pub fn cancel(&self, root: &Path) -> bool {
        self.scheduler.cancel(&PeopleSummaryRebuildJob {
            root: root.to_path_buf(),
        })
    }

    pub fn stats(&self) -> MaintenanceStats {
        self.scheduler.stats()
    }

    pub fn snapshot(&self) -> Vec<PeopleSummaryRebuildJobSnapshot> {
        self.scheduler
            .snapshot()
            .into_iter()
            .map(|snapshot| PeopleSummaryRebuildJobSnapshot {
                job: snapshot.key,
                status: snapshot.status,
                rank: snapshot.rank,
                cancelled: snapshot.cancelled,
            })
            .collect()
    }
}

impl RecognitionRefreshService {
    pub fn new(parallelism: usize, admission: MaintenanceAdmission) -> Self {
        let paused = Arc::new(AtomicBool::new(false));
        let executor_paused = paused.clone();
        let executor_admission = admission.clone();
        let worker_sequence = Arc::new(std::sync::atomic::AtomicU64::new(1));
        let scheduler = Arc::new(MaintenanceScheduler::new_with_admission(
            parallelism.clamp(1, 2),
            admission,
            Arc::new(move |job: &RecognitionRefreshJob, cancelled| {
                if executor_paused.load(Ordering::Acquire)
                    || !executor_admission.checkpoint(cancelled)
                {
                    return Ok(());
                }
                let sequence = worker_sequence.fetch_add(1, Ordering::Relaxed);
                let worker_id = format!("recognition-{}-{sequence}", std::process::id());
                catalog::recognition::ensure_production_calibration(&job.root)?;
                loop {
                    catalog::recognition::process_pending_refresh_batch(
                        &job.root,
                        &worker_id,
                        &executor_admission,
                        cancelled,
                    )?;
                    if executor_paused.load(Ordering::Acquire)
                        || !executor_admission.checkpoint(cancelled)
                        || !catalog::recognition::has_immediately_eligible_refresh_work(&job.root)?
                    {
                        break;
                    }
                    // Make every durable batch a cooperative scheduling boundary.
                    std::thread::yield_now();
                }
                let _ = executor_admission.checkpoint(cancelled);
                Ok(())
            }),
        ));
        Self { scheduler, paused }
    }

    pub fn enqueue(&self, root: PathBuf, rank: i32) -> bool {
        !self.paused.load(Ordering::Acquire)
            && self.scheduler.submit(RecognitionRefreshJob { root }, rank)
    }

    pub fn set_paused(&self, paused: bool) {
        self.paused.store(paused, Ordering::Release);
        if paused {
            self.scheduler.cancel_all();
        }
    }

    pub fn stats(&self) -> MaintenanceStats {
        self.scheduler.stats()
    }

    pub fn snapshot(&self) -> Vec<RecognitionRefreshJobSnapshot> {
        self.scheduler
            .snapshot()
            .into_iter()
            .map(|snapshot| RecognitionRefreshJobSnapshot {
                job: snapshot.key,
                status: snapshot.status,
                rank: snapshot.rank,
                cancelled: snapshot.cancelled,
            })
            .collect()
    }

    pub fn cancel(&self, root: &Path) -> bool {
        self.scheduler.cancel(&RecognitionRefreshJob {
            root: root.to_path_buf(),
        })
    }
}

impl FaceAnalysisService {
    /// Uses the application's shared renderer, including its bounded decoded
    /// source cache and JPEG/RAW/HEIF orientation-aware decode paths.
    #[cfg(test)]
    pub fn new_with_pipeline(
        parallelism: usize,
        limiter: JobLimiter,
        admission: MaintenanceAdmission,
        pipeline: RenderPipeline,
    ) -> Self {
        Self::new_with_pipeline_and_stages(
            parallelism,
            limiter,
            admission,
            pipeline,
            FaceGpuPolicy::Off,
            Arc::new(LiveAnalysisStages),
            None,
        )
    }

    pub fn new_with_pipeline_and_recognition(
        parallelism: usize,
        limiter: JobLimiter,
        admission: MaintenanceAdmission,
        pipeline: RenderPipeline,
        recognition: RecognitionRefreshService,
        gpu_policy: FaceGpuPolicy,
    ) -> Self {
        let wake = Arc::new(move |root: PathBuf| {
            recognition.enqueue(root, 0);
        });
        Self::new_with_pipeline_and_stages(
            parallelism,
            limiter,
            admission,
            pipeline,
            gpu_policy,
            Arc::new(LiveAnalysisStages),
            Some(wake),
        )
    }

    pub(crate) fn new_with_pipeline_and_stages(
        parallelism: usize,
        limiter: JobLimiter,
        admission: MaintenanceAdmission,
        pipeline: RenderPipeline,
        gpu_policy: FaceGpuPolicy,
        analysis_stages: Arc<dyn ProductionAnalysisStages>,
        recognition_wake: Option<Arc<dyn Fn(PathBuf) + Send + Sync>>,
    ) -> Self {
        let paused = Arc::new(AtomicBool::new(false));
        let executor_paused = paused.clone();
        let executor_admission = admission.clone();
        let stages = Arc::new(Mutex::new(HashMap::new()));
        let executor_stages = stages.clone();
        let instrumentation = Arc::new(FaceAnalysisInstrumentation::default());
        let executor_instrumentation = instrumentation.clone();
        let executor_analysis_stages: Arc<dyn ProductionAnalysisStages> =
            Arc::new(InstrumentedAnalysisStages {
                observer: analysis_stages,
                instrumentation: instrumentation.clone(),
            });
        // One stateful engine per worker: the scheduler parallelism is also
        // the explicit inference-session cap, and the shared maintenance
        // admission (widened to the worker count) keeps that many executors
        // running at once.
        let scheduler = MaintenanceScheduler::new_with_admission_factory(
            parallelism,
            admission,
            Arc::new(move |_| {
                let executor_paused = executor_paused.clone();
                let executor_admission = executor_admission.clone();
                let limiter = limiter.clone();
                let pipeline = pipeline.clone();
                let gpu_policy = gpu_policy;
                let stages = executor_stages.clone();
                let instrumentation = executor_instrumentation.clone();
                let analysis_stages = executor_analysis_stages.clone();
                let recognition_wake = recognition_wake.clone();
                let mut engine = None;
                Box::new(move |job: &FaceAnalyzeJob, cancelled: &Arc<AtomicBool>| {
                    let _stage_guard = AnalysisStageGuard {
                        stages: stages.clone(),
                        job: job.clone(),
                    };
                    let job_started = std::time::Instant::now();
                    let job_label = job
                        .label
                        .clone()
                        .unwrap_or_else(|| format!("asset {}", job.asset_id).into());
                    crate::trace::slow_event(&format!("face analysis: {job_label} starting"));
                    if executor_paused.load(Ordering::Acquire) {
                        return Ok(());
                    }
                    let Some(_permit) = limiter.acquire(false, cancelled) else {
                        return Ok(());
                    };
                    if !executor_admission.checkpoint(cancelled) {
                        return Ok(());
                    }
                    if engine.is_none() {
                        set_analysis_stage(&stages, job, FaceAnalysisStage::Initializing);
                        #[cfg(test)]
                        let forced_failure = take_face_engine_setup_failure(job);
                        #[cfg(not(test))]
                        let forced_failure = false;
                        let initialized = if forced_failure {
                            Err(anyhow::anyhow!("injected face-engine setup failure"))
                        } else {
                            FaceEngine::with_gpu_policy(gpu_policy)
                        };
                        engine = Some(match initialized {
                            Ok(engine) => engine,
                            Err(error) => {
                                persist_setup_failure(&job.root, job.asset_id, &error)?;
                                return Err(error);
                            }
                        });
                        instrumentation
                            .engines_constructed
                            .fetch_add(1, Ordering::AcqRel);
                    }
                    set_analysis_stage(&stages, job, FaceAnalysisStage::Claiming);
                    let _active = instrumentation.enter();
                    let result = analyze_asset(
                        &job.root,
                        job.asset_id,
                        engine.as_mut().expect("face engine initialized"),
                        &pipeline,
                        &executor_admission,
                        cancelled,
                        AnalysisProgress {
                            display: Some((&stages, job)),
                            hooks: analysis_stages.as_ref(),
                        },
                    );
                    let job_elapsed = job_started.elapsed();
                    match &result {
                        Ok(()) => {
                            crate::trace::slow_event(&format!(
                                "face analysis: {job_label} completed in {}",
                                crate::trace::secs(job_elapsed),
                            ));
                        }
                        Err(error) => {
                            crate::trace::slow_event(&format!(
                                "face analysis: {job_label} failed after {}: {error}",
                                crate::trace::secs(job_elapsed),
                            ));
                        }
                    }
                    if result.is_ok()
                        && !cancelled.load(Ordering::Acquire)
                        && let Some(wake) = &recognition_wake
                    {
                        wake(job.root.clone());
                    }
                    result
                })
            }),
        );
        Self {
            scheduler,
            paused,
            enabled: parallelism > 0,
            stages,
            instrumentation,
        }
    }

    pub fn enqueue(&self, job: FaceAnalyzeJob, rank: i32) -> bool {
        if !self.enabled || self.paused.load(Ordering::Acquire) {
            return false;
        }
        let submitted = self.scheduler.submit(job.clone(), rank);
        if submitted {
            self.stages
                .lock()
                .unwrap()
                .entry(job)
                .or_insert(FaceAnalysisStage::Queued);
        }
        submitted
    }

    pub fn set_paused(&self, paused: bool) {
        self.paused.store(paused, Ordering::Release);
        if paused {
            self.scheduler.cancel_all();
            self.stages.lock().unwrap().clear();
        }
    }

    pub fn stats(&self) -> MaintenanceStats {
        self.scheduler.stats()
    }

    pub fn instrumentation(&self) -> FaceAnalysisInstrumentationSnapshot {
        self.instrumentation.snapshot()
    }

    pub fn snapshot(&self) -> Vec<FaceAnalysisJobSnapshot> {
        let scheduler = self.scheduler.snapshot();
        let stages = self.stages.lock().unwrap();
        scheduler
            .into_iter()
            .map(|snapshot| FaceAnalysisJobSnapshot {
                stage: stages
                    .get(&snapshot.key)
                    .copied()
                    .unwrap_or(FaceAnalysisStage::Queued),
                job: snapshot.key,
                status: snapshot.status,
                rank: snapshot.rank,
                cancelled: snapshot.cancelled,
            })
            .collect()
    }

    pub fn cancel(&self, job: &FaceAnalyzeJob) -> bool {
        let durable = snooze_analysis_job(&job.root, job.asset_id).unwrap_or(false);
        let cancelled_status = self.scheduler.cancel_and_status(job);
        let was_running = cancelled_status == Some(MaintenanceJobStatus::Running);
        if !was_running {
            self.stages.lock().unwrap().remove(job);
        }
        durable || cancelled_status.is_some()
    }
}

pub fn snooze_analysis_job(root: &Path, asset_id: i64) -> Result<bool> {
    let path = catalog::catalog_path(root);
    if !path.is_file() {
        return Ok(false);
    }
    let connection = Connection::open(path)?;
    Ok(connection.execute(
        "UPDATE face_analysis_state SET status='error', error='manual-cancelled',
           failure_class='manual', worker_id=NULL, lease_expires_at=NULL, retry_after=NULL,
           updated_at=datetime('now') WHERE asset_id=?1",
        [asset_id],
    )? > 0)
}

fn set_analysis_stage(
    stages: &Mutex<HashMap<FaceAnalyzeJob, FaceAnalysisStage>>,
    job: &FaceAnalyzeJob,
    stage: FaceAnalysisStage,
) {
    let subject = job
        .label
        .as_deref()
        .map(str::to_owned)
        .unwrap_or_else(|| format!("asset {}", job.asset_id));
    crate::trace::event(&format!("face analysis: {subject} → {stage:?}"));
    stages.lock().unwrap().insert(job.clone(), stage);
}

fn reach_analysis_stage(
    display: Option<(
        &Mutex<HashMap<FaceAnalyzeJob, FaceAnalysisStage>>,
        &FaceAnalyzeJob,
    )>,
    hooks: &dyn ProductionAnalysisStages,
    stage: FaceAnalysisStage,
    cancelled: &Arc<AtomicBool>,
) -> bool {
    if let Some((stages, job)) = display {
        set_analysis_stage(stages, job, stage);
    }
    hooks.reached(stage, cancelled);
    !cancelled.load(Ordering::Acquire)
}

#[cfg(test)]
pub fn pending_analysis_jobs(root: &Path, limit: usize) -> Result<Vec<FaceAnalyzeJob>> {
    Ok(pending_analysis_jobs_prioritized(root, limit, None, &[])?
        .into_iter()
        .map(|(job, _)| job)
        .collect())
}

pub fn pending_analysis_jobs_prioritized(
    root: &Path,
    limit: usize,
    preferred_subtree: Option<&Path>,
    recent_assets: &[PathBuf],
) -> Result<Vec<(FaceAnalyzeJob, i32)>> {
    if limit == 0 {
        return Ok(Vec::new());
    }
    let path = catalog::catalog_path(root);
    if !path.is_file() {
        return Ok(Vec::new());
    }
    let connection = Connection::open(path)?;
    let preferred_scope = preferred_subtree
        .and_then(|path| path.strip_prefix(root).ok())
        .filter(|path| !path.as_os_str().is_empty())
        .map(catalog::scope_key)
        .transpose()?;
    let preferred_len = preferred_scope.as_ref().map_or(0, Vec::len) as i64;
    let recent_paths = recent_assets
        .iter()
        .filter_map(|path| path.strip_prefix(root).ok())
        .map(catalog::encode_relative_path)
        .collect::<Result<Vec<_>>>()?;
    connection.execute_batch(
        "CREATE TEMP TABLE IF NOT EXISTS recent_analysis_assets(
             relative_path BLOB PRIMARY KEY, rank INTEGER NOT NULL);
         DELETE FROM recent_analysis_assets;",
    )?;
    for (position, path) in recent_paths.iter().enumerate() {
        connection.execute(
            "INSERT OR IGNORE INTO recent_analysis_assets VALUES(?1, ?2)",
            params![path, 100_i32.saturating_sub(position as i32)],
        )?;
    }
    let mut query = connection.prepare(
        "SELECT asset.id, asset.relative_path,
                CASE WHEN ?4 IS NOT NULL AND substr(asset.scope_key, 1, ?5)=?4
                     THEN 200 ELSE coalesce(recent.rank, 0) END AS rank
         FROM analysis_asset_inventory inventory
         JOIN assets asset ON asset.id=inventory.asset_id
         LEFT JOIN recent_analysis_assets recent ON recent.relative_path=asset.relative_path
         LEFT JOIN face_analysis_state state ON state.asset_id=asset.id
         WHERE (state.asset_id IS NULL OR NOT(state.status='error'
                   AND coalesce(state.error, '')='manual-cancelled'))
           AND (state.asset_id IS NULL OR NOT(state.status='error'
                   AND coalesce(state.error, '') LIKE 'filesystem identity differs from inventory%'
                   AND state.retry_after>datetime('now')))
           AND (state.asset_id IS NULL
            OR state.detector_model_id<>?1
            OR state.recognizer_model_id<>'sface-2021dec'
            OR state.decoder_model_id<>?3
            OR state.source_size<>inventory.source_size
            OR state.source_modified_ns<>inventory.source_modified_ns
            OR state.source_fingerprint<>inventory.source_fingerprint
            OR state.status='pending'
            OR (state.status='error' AND state.failure_class='transient' AND
                (state.retry_after IS NULL OR state.retry_after<=datetime('now')))
            OR (state.status='running' AND
                (state.lease_expires_at IS NULL OR state.lease_expires_at<=datetime('now'))))
         ORDER BY rank DESC, asset.id LIMIT ?2",
    )?;
    query
        .query_map(
            params![
                ANALYSIS_MODEL_ID,
                limit as i64,
                DECODER_MODEL_ID,
                preferred_scope,
                preferred_len
            ],
            |row| {
                Ok((
                    FaceAnalyzeJob {
                        root: root.to_path_buf(),
                        asset_id: row.get(0)?,
                        label: row
                            .get::<_, Vec<u8>>(1)
                            .ok()
                            .and_then(|encoded| crate::catalog::decode_relative_path(&encoded).ok())
                            .map(|path| Arc::from(path.to_string_lossy().into_owned().as_str())),
                    },
                    row.get::<_, i32>(2)?,
                ))
            },
        )?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map(|mut jobs| {
            jobs.sort_by_key(|(job, rank)| (std::cmp::Reverse(*rank), job.asset_id));
            jobs
        })
        .map_err(Into::into)
}

/// Delay until durable retry or abandoned-lease recovery becomes eligible.
/// `None` means the queue is fully drained and no polling wake is necessary.
pub fn next_analysis_wake(root: &Path) -> Result<Option<std::time::Duration>> {
    let path = catalog::catalog_path(root);
    if !path.is_file() {
        return Ok(None);
    }
    let connection = Connection::open(path)?;
    let seconds = connection
        .query_row(
            "SELECT min(max(1, unixepoch(deadline)-unixepoch('now'))) FROM (
               SELECT retry_after AS deadline FROM face_analysis_state
                 WHERE status='error' AND retry_after>datetime('now')
                   AND failure_class='transient'
               UNION ALL
               SELECT lease_expires_at FROM face_analysis_state
                 WHERE status='running' AND lease_expires_at>datetime('now'))",
            [],
            |row| row.get::<_, Option<i64>>(0),
        )?
        .map(|seconds| std::time::Duration::from_secs(seconds.clamp(1, 3600) as u64));
    Ok(seconds)
}

pub fn sync_analysis_inventory(
    root: &Path,
    eligible_paths: &[PathBuf],
) -> Result<catalog::AssetInventorySync> {
    catalog::sync_asset_inventory(root, eligible_paths, ANALYSIS_MODEL_ID, "sface-2021dec")
}

fn persist_setup_failure(root: &Path, asset_id: i64, _: &anyhow::Error) -> Result<()> {
    let path = catalog::catalog_path(root);
    if !path.is_file() {
        return Ok(());
    }
    let connection = Connection::open(path)?;
    connection.execute(
        "WITH decision(next_attempts,source_size,source_modified_ns,source_fingerprint)
         AS MATERIALIZED (
           SELECT CASE WHEN state.detector_model_id<>?1
                            OR state.recognizer_model_id<>'sface-2021dec'
                            OR state.decoder_model_id<>?2
                            OR state.source_size<>inventory.source_size
                            OR state.source_modified_ns<>inventory.source_modified_ns
                            OR state.source_fingerprint<>inventory.source_fingerprint
                       THEN 1 ELSE state.attempts+1 END,
                  inventory.source_size,inventory.source_modified_ns,
                  inventory.source_fingerprint
             FROM face_analysis_state state
             JOIN analysis_asset_inventory inventory ON inventory.asset_id=state.asset_id
            WHERE state.asset_id=?4
              AND NOT(state.status='running' AND state.lease_expires_at>datetime('now'))
              AND (state.detector_model_id<>?1
                OR state.recognizer_model_id<>'sface-2021dec'
                OR state.decoder_model_id<>?2
                OR state.source_size<>inventory.source_size
                OR state.source_modified_ns<>inventory.source_modified_ns
                OR state.source_fingerprint<>inventory.source_fingerprint
                OR state.status='pending'
                OR (state.status='error' AND state.failure_class='transient'
                    AND (state.retry_after IS NULL OR state.retry_after<=datetime('now')))
                OR (state.status='running'
                    AND (state.lease_expires_at IS NULL
                         OR state.lease_expires_at<=datetime('now')))))
         UPDATE face_analysis_state SET status='error',
           attempts=(SELECT next_attempts FROM decision),
           detector_model_id=?1, recognizer_model_id='sface-2021dec', decoder_model_id=?2,
           source_size=(SELECT source_size FROM decision),
           source_modified_ns=(SELECT source_modified_ns FROM decision),
           source_fingerprint=(SELECT source_fingerprint FROM decision),
           worker_id=NULL, lease_expires_at=NULL,
           failure_class=CASE WHEN (SELECT next_attempts FROM decision)>=?5
                              THEN 'exhausted' ELSE 'transient' END,
           retry_after=CASE WHEN (SELECT next_attempts FROM decision)>=?5 THEN NULL
             ELSE datetime('now', printf('+%d seconds', min(3600,
               1 << min(10, (SELECT next_attempts FROM decision))))) END,
           error=?3, updated_at=datetime('now') WHERE asset_id=?4
             AND EXISTS(SELECT 1 FROM decision)",
        params![
            ANALYSIS_MODEL_ID,
            DECODER_MODEL_ID,
            "face-engine-unavailable",
            asset_id,
            MAX_AUTOMATIC_ANALYSIS_ATTEMPTS
        ],
    )?;
    Ok(())
}

#[derive(Clone, Debug, PartialEq)]
pub struct DetectedFace {
    pub bounds: [f32; 4],
    pub landmarks: [[f32; 2]; 5],
    pub confidence: f32,
}

pub struct FaceEngine {
    detector: Session,
    recognizer: Session,
    network: OfflineNetworkCapability,
}

impl FaceEngine {
    /// Builds the engine, honoring the configured GPU policy. `Auto` probes
    /// CUDA once per process and falls back to CPU when the driver, CUDA
    /// runtime, or cuDNN userland libraries are missing; `Off` pins CPU.
    pub fn with_gpu_policy(policy: FaceGpuPolicy) -> Result<Self> {
        let device = resolve_compute_device(policy);
        let engine = Self::build(device)?;
        crate::trace::event(&format!(
            "face engine: {} inference sessions (policy {policy:?})",
            device.as_str(),
        ));
        Ok(engine)
    }

    fn build(device: FaceComputeDevice) -> Result<Self> {
        validate_packaged_face_assets()?;
        Ok(Self {
            detector: build_inference_session(YUNET, device, "YuNet face detector")?,
            recognizer: build_inference_session(SFACE, device, "SFace recognizer")?,
            network: OfflineNetworkCapability,
        })
    }

    pub fn detect(&mut self, image: &DynamicImage) -> Result<Vec<DetectedFace>> {
        self.network.enter();
        let rgb = image.to_rgb8();
        let (width, height) = rgb.dimensions();
        if width == 0 || height == 0 {
            return Ok(Vec::new());
        }
        let scale = (640.0 / width as f32).min(640.0 / height as f32);
        let resized_width = (width as f32 * scale).round().clamp(1.0, 640.0) as u32;
        let resized_height = (height as f32 * scale).round().clamp(1.0, 640.0) as u32;
        let rgb =
            image::imageops::resize(&rgb, resized_width, resized_height, FilterType::Triangle);
        let padded_width = 640;
        let padded_height = 640;
        let plane = padded_width as usize * padded_height as usize;
        let mut input = vec![0.0_f32; plane * 3];
        for (x, y, pixel) in rgb.enumerate_pixels() {
            let index = y as usize * padded_width as usize + x as usize;
            input[index] = f32::from(pixel[2]);
            input[plane + index] = f32::from(pixel[1]);
            input[plane * 2 + index] = f32::from(pixel[0]);
        }
        let tensor = Tensor::from_array((
            [1_usize, 3, padded_height as usize, padded_width as usize],
            input,
        ))?;
        let output = self.detector.run(ort::inputs![tensor])?;
        let names = [
            "cls_8", "cls_16", "cls_32", "obj_8", "obj_16", "obj_32", "bbox_8", "bbox_16",
            "bbox_32", "kps_8", "kps_16", "kps_32",
        ];
        let mut values = Vec::with_capacity(names.len());
        for name in names {
            values.push(
                output[name]
                    .try_extract_tensor::<f32>()
                    .with_context(|| format!("read YuNet output {name}"))?
                    .1,
            );
        }
        let mut candidates = Vec::new();
        for (level, stride) in [8_u32, 16, 32].into_iter().enumerate() {
            let columns = padded_width / stride;
            let rows = padded_height / stride;
            for row in 0..rows {
                for column in 0..columns {
                    let index = (row * columns + column) as usize;
                    let confidence = (values[level][index].clamp(0.0, 1.0)
                        * values[level + 3][index].clamp(0.0, 1.0))
                    .sqrt();
                    if confidence < 0.9 {
                        continue;
                    }
                    let boxes = values[level + 6];
                    let points = values[level + 9];
                    let cx = (column as f32 + boxes[index * 4]) * stride as f32;
                    let cy = (row as f32 + boxes[index * 4 + 1]) * stride as f32;
                    let box_width = boxes[index * 4 + 2].exp() * stride as f32;
                    let box_height = boxes[index * 4 + 3].exp() * stride as f32;
                    let mut landmarks = [[0.0; 2]; 5];
                    for (point_index, point) in landmarks.iter_mut().enumerate() {
                        point[0] =
                            (points[index * 10 + point_index * 2] + column as f32) * stride as f32;
                        point[1] =
                            (points[index * 10 + point_index * 2 + 1] + row as f32) * stride as f32;
                    }
                    candidates.push(PixelFace {
                        bounds: [
                            cx - box_width / 2.0,
                            cy - box_height / 2.0,
                            cx + box_width / 2.0,
                            cy + box_height / 2.0,
                        ],
                        landmarks,
                        confidence,
                    });
                }
            }
        }
        candidates.sort_by(|left, right| right.confidence.total_cmp(&left.confidence));
        let mut kept: Vec<PixelFace> = Vec::new();
        for face in candidates {
            if kept
                .iter()
                .all(|other| intersection_over_union(face.bounds, other.bounds) <= 0.3)
            {
                kept.push(face);
            }
        }
        Ok(kept
            .into_iter()
            .filter_map(|face| face.normalized(width, height, scale))
            .collect())
    }

    pub fn embedding(&mut self, image: &DynamicImage, face: &DetectedFace) -> Result<Vec<f32>> {
        self.network.enter();
        let aligned = align_face(image, face)?;
        let plane = 112 * 112;
        let mut input = vec![0.0_f32; plane * 3];
        for (x, y, pixel) in aligned.enumerate_pixels() {
            let index = (y * 112 + x) as usize;
            input[index] = f32::from(pixel[0]);
            input[plane + index] = f32::from(pixel[1]);
            input[plane * 2 + index] = f32::from(pixel[2]);
        }
        let tensor = Tensor::from_array(([1_usize, 3, 112, 112], input))?;
        let output = self.recognizer.run(ort::inputs![tensor])?;
        let mut embedding = output[0].try_extract_tensor::<f32>()?.1.to_vec();
        normalize(&mut embedding)?;
        Ok(embedding)
    }
}

/// Picks the inference device for the configured policy. `Auto` benchmarks
/// both devices once per process (CUDA only when the whole stack — driver,
/// CUDA runtime, cuDNN, execution provider — initializes) and keeps the
/// faster one: these models are small enough that a latency-bound GPU can
/// lose to the CPU. Any failure falls back to CPU with a diagnostic. Face
/// tests are pinned to CPU so their embeddings stay deterministic regardless
/// of the host GPU.
fn resolve_compute_device(policy: FaceGpuPolicy) -> FaceComputeDevice {
    #[cfg(test)]
    {
        let _ = policy;
        FaceComputeDevice::Cpu
    }
    #[cfg(not(test))]
    match policy {
        FaceGpuPolicy::Off => FaceComputeDevice::Cpu,
        FaceGpuPolicy::Auto => {
            if !CUDA_EP_COMPILED {
                FaceComputeDevice::Cpu
            } else {
                *AUTO_COMPUTE_DEVICE.get_or_init(benchmark_auto_device)
            }
        }
    }
}

/// One-time `Auto` probe: time the YuNet detection pass on each device and
/// keep the winner. Detection is the heavier of the two models (640×640
/// input versus 112×112), so it is a representative proxy for photo cost.
#[cfg(not(test))]
fn benchmark_auto_device() -> FaceComputeDevice {
    let describe = |result: &Result<std::time::Duration>| match result {
        Ok(duration) => format!("{}ms", duration.as_millis()),
        Err(error) => format!("failed ({error:#})"),
    };
    let cpu = benchmark_device(FaceComputeDevice::Cpu);
    let cuda = benchmark_device(FaceComputeDevice::Cuda);
    match (&cpu, &cuda) {
        (Ok(cpu_time), Ok(cuda_time)) => {
            let device = if cuda_time < cpu_time {
                FaceComputeDevice::Cuda
            } else {
                FaceComputeDevice::Cpu
            };
            crate::trace::slow_event(&format!(
                "face engine: auto device benchmark · cpu {} · cuda {} per detection → {}",
                describe(&cpu),
                describe(&cuda),
                device.as_str(),
            ));
            device
        }
        _ => {
            crate::trace::slow_event(&format!(
                "face engine: auto device benchmark · cpu {} · cuda {} → cpu fallback",
                describe(&cpu),
                describe(&cuda),
            ));
            FaceComputeDevice::Cpu
        }
    }
}

/// Average steady-state duration of one detection pass on `device`.
#[cfg(not(test))]
fn benchmark_device(device: FaceComputeDevice) -> Result<std::time::Duration> {
    // Enough warmup for cuDNN plan caching and a measure of GPU clock ramp;
    // the comparison measures steady state, the regime a 150k-photo backlog
    // actually runs in.
    const WARMUP_RUNS: usize = 8;
    const TIMED_RUNS: usize = 6;
    let mut engine = FaceEngine::build(device)?;
    // Structured synthetic frame: detection cost is input-shape dominated,
    // so the content only needs to exercise the full session path.
    let frame = DynamicImage::ImageRgb8(image::RgbImage::from_fn(640, 480, |x, y| {
        Rgb([
            (x * 255 / 640) as u8,
            (y * 255 / 480) as u8,
            ((x + y) % 256) as u8,
        ])
    }));
    for _ in 0..WARMUP_RUNS {
        engine.detect(&frame)?;
    }
    let started = std::time::Instant::now();
    for _ in 0..TIMED_RUNS {
        engine.detect(&frame)?;
    }
    Ok(started.elapsed() / TIMED_RUNS as u32)
}

fn build_inference_session(
    model: &[u8],
    device: FaceComputeDevice,
    label: &str,
) -> Result<Session> {
    // Bounded per-worker ORT pools: each session gets two intra-op threads,
    // so N workers use 2N ORT threads; the models are invoked sequentially
    // within each job.
    let builder = Session::builder()?
        .with_intra_threads(2)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?
        .with_inter_threads(1)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    // Registration errors must propagate (error_on_failure): ort otherwise
    // swallows a failed CUDA EP registration — its log line is compiled out
    // without the tracing feature — and silently commits the session on CPU.
    #[cfg(all(
        any(target_os = "linux", target_os = "windows"),
        target_arch = "x86_64"
    ))]
    let mut builder = if device == FaceComputeDevice::Cuda {
        use ort::ep::cuda::ConvAlgorithmSearch;
        builder
            .with_execution_providers([ort::ep::CUDA::default()
                .with_device_id(0)
                // The analysis input shapes are fixed, so the exhaustive
                // first-run algorithm benchmarking buys nothing.
                .with_conv_algorithm_search(ConvAlgorithmSearch::Heuristic)
                .build()
                .error_on_failure()])
            .map_err(|error| anyhow::anyhow!(error.to_string()))?
    } else {
        builder
    };
    #[cfg(not(all(
        any(target_os = "linux", target_os = "windows"),
        target_arch = "x86_64"
    )))]
    let mut builder = builder;
    let _ = device;
    builder
        .commit_from_memory(model)
        .with_context(|| format!("load bundled {label}"))
}

pub fn validate_packaged_face_assets() -> Result<()> {
    for (name, bytes, expected) in [
        ("YuNet", YUNET, YUNET_SHA256),
        ("SFace", SFACE, SFACE_SHA256),
    ] {
        let actual = hex::encode(Sha256::digest(bytes));
        if actual != expected {
            bail!(
                "packaged {name} face model failed SHA-256 validation (expected {expected}, got {actual}); reinstall Rekesz"
            );
        }
    }
    for (name, bytes, expected) in [
        ("YuNet license", YUNET_LICENSE, YUNET_LICENSE_SHA256),
        ("SFace license", SFACE_LICENSE, SFACE_LICENSE_SHA256),
    ] {
        let actual = hex::encode(Sha256::digest(bytes));
        if actual != expected {
            bail!(
                "packaged {name} failed SHA-256 validation (expected {expected}, got {actual}); reinstall Rekesz"
            );
        }
    }
    catalog::recognition::validate_production_calibration_artifact()
        .context("validate packaged SFace production policy")?;
    Ok(())
}

/// Indexes a small, bounded batch so startup remains responsive. Repeated app
/// sessions continue where the durable analysis state left off.
#[cfg(test)]
pub fn index_batch(root: &Path, limit: usize) -> Result<usize> {
    let catalog_path = catalog::catalog_path(root);
    if !catalog_path.is_file() || limit == 0 {
        return Ok(0);
    }
    let mut engine = FaceEngine::with_gpu_policy(FaceGpuPolicy::Auto)?;
    let mut indexed = 0;
    while indexed < limit && index_next_asset(root, &catalog_path, &mut engine)? {
        indexed += 1;
    }
    Ok(indexed)
}

#[cfg(test)]
fn index_next_asset(root: &Path, catalog_path: &Path, engine: &mut FaceEngine) -> Result<bool> {
    let connection = Connection::open(catalog_path)?;
    let candidate = connection
        .query_row(
            "SELECT asset.id, asset.relative_path FROM assets asset \
             WHERE NOT EXISTS (SELECT 1 FROM face_analysis_state state \
                               WHERE state.asset_id=asset.id AND state.status IN ('complete', 'error')) \
             ORDER BY asset.id LIMIT 1",
            [],
            |row| Ok((row.get::<_, i64>(0)?, row.get::<_, Vec<u8>>(1)?)),
        )
        .optional()?;
    let Some((asset_id, relative_path)) = candidate else {
        return Ok(false);
    };
    let pipeline = RenderPipeline::with_source_cache_capacity(
        root.join(catalog::CATALOG_DIR).join("analysis-cache"),
        1,
        1,
        1,
    );
    analyze_asset_with_candidate(
        root,
        AnalysisCandidate {
            asset_id,
            relative_path,
        },
        engine,
        &pipeline,
        &MaintenanceAdmission::new(),
        &Arc::new(AtomicBool::new(false)),
        AnalysisProgress {
            display: None,
            hooks: &LiveAnalysisStages,
        },
    )?;
    Ok(true)
}

fn analyze_asset(
    root: &Path,
    asset_id: i64,
    engine: &mut FaceEngine,
    pipeline: &RenderPipeline,
    admission: &MaintenanceAdmission,
    cancelled: &Arc<AtomicBool>,
    progress: AnalysisProgress<'_>,
) -> Result<()> {
    let catalog_path = catalog::catalog_path(root);
    let connection = Connection::open(&catalog_path)?;
    let relative_path = connection
        .query_row(
            "SELECT relative_path FROM assets WHERE id=?1",
            [asset_id],
            |row| row.get::<_, Vec<u8>>(0),
        )
        .optional()?;
    let Some(relative_path) = relative_path else {
        return Ok(());
    };
    analyze_asset_with_candidate(
        root,
        AnalysisCandidate {
            asset_id,
            relative_path,
        },
        engine,
        pipeline,
        admission,
        cancelled,
        progress,
    )
}

fn claim_analysis_state(
    connection: &Connection,
    asset_id: i64,
    worker_id: &str,
    source: AnalyzedSourceIdentity<'_>,
) -> Result<bool> {
    connection.execute(
        "INSERT INTO face_analysis_state(asset_id, detector_model_id, recognizer_model_id,
           decoder_model_id, source_size, source_modified_ns, source_fingerprint, status,
           attempts, worker_id, lease_expires_at, failure_class, updated_at)
         VALUES(?1, ?2, 'sface-2021dec', ?3, ?4, ?5, ?6, 'running', 0, ?7,
                datetime('now', '+30 minutes'), NULL, datetime('now'))
         ON CONFLICT(asset_id) DO UPDATE SET status='running', worker_id=excluded.worker_id,
           attempts=CASE WHEN face_analysis_state.detector_model_id<>excluded.detector_model_id
                    OR face_analysis_state.recognizer_model_id<>excluded.recognizer_model_id
                    OR face_analysis_state.decoder_model_id<>excluded.decoder_model_id
                    OR face_analysis_state.source_size<>excluded.source_size
                    OR face_analysis_state.source_modified_ns<>excluded.source_modified_ns
                    OR face_analysis_state.source_fingerprint<>excluded.source_fingerprint
                  THEN 0 ELSE face_analysis_state.attempts END,
           detector_model_id=excluded.detector_model_id,
           recognizer_model_id=excluded.recognizer_model_id,
           decoder_model_id=excluded.decoder_model_id, source_size=excluded.source_size,
           source_modified_ns=excluded.source_modified_ns,
           source_fingerprint=excluded.source_fingerprint,
           lease_expires_at=datetime('now', '+30 minutes'), error=NULL,
           failure_class=NULL, retry_after=NULL, updated_at=datetime('now')
         WHERE face_analysis_state.detector_model_id<>excluded.detector_model_id
            OR face_analysis_state.recognizer_model_id<>excluded.recognizer_model_id
            OR face_analysis_state.decoder_model_id<>excluded.decoder_model_id
            OR face_analysis_state.source_size<>excluded.source_size
            OR face_analysis_state.source_modified_ns<>excluded.source_modified_ns
            OR face_analysis_state.source_fingerprint<>excluded.source_fingerprint
            OR face_analysis_state.status='pending'
            OR (face_analysis_state.status='error' AND face_analysis_state.failure_class='transient'
                AND (face_analysis_state.retry_after IS NULL
                     OR face_analysis_state.retry_after<=datetime('now')))
            OR (face_analysis_state.status='running' AND
                (face_analysis_state.lease_expires_at IS NULL
                 OR face_analysis_state.lease_expires_at<=datetime('now'))) ",
        params![
            asset_id,
            ANALYSIS_MODEL_ID,
            DECODER_MODEL_ID,
            source.size,
            source.modified_ns,
            source.fingerprint,
            worker_id
        ],
    )?;
    Ok(connection.changes() != 0)
}

fn analyze_asset_with_candidate(
    root: &Path,
    candidate: AnalysisCandidate,
    engine: &mut FaceEngine,
    pipeline: &RenderPipeline,
    admission: &MaintenanceAdmission,
    cancelled: &Arc<AtomicBool>,
    progress: AnalysisProgress<'_>,
) -> Result<()> {
    let AnalysisCandidate {
        asset_id,
        relative_path,
    } = candidate;
    if cancelled.load(Ordering::Acquire) {
        return Ok(());
    }
    let catalog_path = catalog::catalog_path(root);
    let connection = Connection::open(&catalog_path)?;
    let path = root.join(catalog::decode_relative_path(&relative_path)?);
    let metadata = std::fs::metadata(&path)
        .with_context(|| format!("read source identity for {}", path.display()))?;
    let source_size = metadata.len().min(i64::MAX as u64) as i64;
    let source_modified_ns = catalog::modified_ns(&metadata);
    // The inventory pass verified this file's content fingerprint moments
    // ago; when its recorded size and mtime still match, reuse that
    // fingerprint instead of re-reading (and re-hashing) the whole file per
    // analysis. Any mismatch falls back to the full content hash, and the
    // existing inventory-currency check defers the asset.
    let inventory_row: Option<(i64, i64, Vec<u8>)> = connection
        .query_row(
            "SELECT source_size, source_modified_ns, source_fingerprint
             FROM analysis_asset_inventory WHERE asset_id=?1",
            [asset_id],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .optional()?;
    // A NULL/malformed stored fingerprint degrades to the hashing fallback
    // instead of failing the asset permanently.
    let identity_matches_inventory =
        inventory_row
            .as_ref()
            .and_then(|(stored_size, stored_modified, fingerprint)| {
                let fingerprint: [u8; 32] = fingerprint.as_slice().try_into().ok()?;
                (*stored_size == source_size && *stored_modified == source_modified_ns)
                    .then_some(fingerprint)
            });
    let (source_identity, inventory_current) = if let Some(fingerprint) = identity_matches_inventory
    {
        (
            catalog::SourceContentIdentity {
                size: source_size,
                modified_ns: source_modified_ns,
                fingerprint,
            },
            true,
        )
    } else {
        let identity = catalog::stable_content_identity(&path)?;
        let inventory_current: bool = connection.query_row(
            "SELECT EXISTS(SELECT 1 FROM analysis_asset_inventory WHERE asset_id=?1
               AND source_size=?2 AND source_modified_ns=?3 AND decoder_model_id=?4
               AND source_fingerprint=?5)",
            params![
                asset_id,
                identity.size,
                identity.modified_ns,
                DECODER_MODEL_ID,
                identity.fingerprint.as_slice()
            ],
            |row| row.get(0),
        )?;
        (identity, inventory_current)
    };
    let source_fingerprint = source_identity.fingerprint;
    if source_size as u64 > ANALYSIS_MAX_SOURCE_BYTES {
        defer_identity_mismatch(
            &connection,
            asset_id,
            source_size,
            source_modified_ns,
            &source_fingerprint,
            "source exceeds the face-analysis byte budget",
        )?;
        return Ok(());
    }
    if !inventory_current {
        defer_identity_mismatch(
            &connection,
            asset_id,
            source_size,
            source_modified_ns,
            &source_fingerprint,
            "filesystem identity differs from inventory; waiting for inventory refresh",
        )?;
        return Ok(());
    }
    let worker_id = format!(
        "{:?}-{}",
        std::thread::current().id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |value| value.as_nanos())
    );
    let claimed = claim_analysis_state(
        &connection,
        asset_id,
        &worker_id,
        AnalyzedSourceIdentity {
            size: source_size,
            modified_ns: source_modified_ns,
            fingerprint: &source_fingerprint,
            path: Some(&path),
        },
    )?;
    if !claimed {
        return Ok(());
    }
    let _claim_guard = ClaimGuard {
        connection: &connection,
        asset_id,
        worker_id: &worker_id,
        source_size,
        source_modified_ns,
        source_fingerprint: source_fingerprint.to_vec(),
    };
    if !admission.checkpoint(cancelled) {
        return Ok(());
    }
    if !reach_analysis_stage(
        progress.display,
        progress.hooks,
        FaceAnalysisStage::Decoding,
        cancelled,
    ) {
        return Ok(());
    }
    let analysis = pipeline
        .render_analysis(
            RenderInvocation {
                path: path.clone(),
                // A dedicated bounded analysis derivative avoids retaining a
                // full-resolution raster in the inference job. Normalized
                // face coordinates remain invariant under this resize.
                variant: RenderVariant::Preview {
                    max_width: ANALYSIS_MAX_DIMENSION,
                    max_height: ANALYSIS_MAX_DIMENSION,
                },
                render_identity: RenderIdentity {
                    recipe_digest: source_fingerprint,
                    ..RenderIdentity::default()
                },
                recipe: None,
                interactive: false,
                durable_cache_write: false,
            },
            cancelled,
        )
        .map_err(|error| anyhow::anyhow!("decode {} for face analysis: {error}", path.display()))
        .and_then(|image| {
            anyhow::ensure!(
                u64::from(image.image.width()) * u64::from(image.image.height())
                    <= u64::from(ANALYSIS_MAX_DIMENSION).pow(2),
                "analysis decoder exceeded its pixel budget"
            );
            if !admission.checkpoint(cancelled) {
                return Ok(None);
            }
            if cancelled.load(Ordering::Acquire) {
                return Ok(None);
            }
            let inverse = image
                .metadata
                .face_transform
                .and_then(|transform| transform.inverse())
                .context("decoder returned a non-invertible face coordinate transform")?;
            if !reach_analysis_stage(
                progress.display,
                progress.hooks,
                FaceAnalysisStage::Detecting,
                cancelled,
            ) {
                return Ok(None);
            }
            progress.hooks.inference_started();
            let detections = engine.detect(&image.image)?;
            progress.hooks.inference_completed();
            if cancelled.load(Ordering::Acquire) {
                return Ok(None);
            }
            let mut results = Vec::with_capacity(detections.len());
            if !reach_analysis_stage(
                progress.display,
                progress.hooks,
                FaceAnalysisStage::Embedding,
                cancelled,
            ) {
                return Ok(None);
            }
            for detection in detections {
                if !admission.checkpoint(cancelled) {
                    return Ok(None);
                }
                if cancelled.load(Ordering::Acquire) {
                    return Ok(None);
                }
                let embedding = engine.embedding(&image.image, &detection)?;
                let canonical = DetectedFace {
                    bounds: inverse
                        .bounds(detection.bounds)
                        .map(|value| value.clamp(0.0, 1.0)),
                    landmarks: detection.landmarks.map(|point| {
                        inverse
                            .mapped_point(point)
                            .map(|value| value.clamp(0.0, 1.0))
                    }),
                    confidence: detection.confidence,
                };
                if canonical.bounds[0] < canonical.bounds[2]
                    && canonical.bounds[1] < canonical.bounds[3]
                {
                    results.push((canonical, embedding));
                }
            }
            Ok(Some(results))
        });
    match analysis {
        Ok(Some(results)) => {
            if !cancelled.load(Ordering::Acquire) {
                let current = catalog::stable_content_identity(&path)?;
                if current.size != source_size
                    || current.modified_ns != source_modified_ns
                    || current.fingerprint != source_fingerprint
                {
                    connection.execute(
                        "UPDATE face_analysis_state SET status='pending', worker_id=NULL,
                         lease_expires_at=NULL, updated_at=datetime('now') WHERE asset_id=?1
                         AND worker_id=?2 AND source_size=?3 AND source_modified_ns=?4
                         AND source_fingerprint=?5",
                        params![
                            asset_id,
                            worker_id,
                            source_size,
                            source_modified_ns,
                            source_fingerprint.as_slice()
                        ],
                    )?;
                    return Ok(());
                }
                if !admission.checkpoint(cancelled) {
                    return Ok(());
                }
                if !reach_analysis_stage(
                    progress.display,
                    progress.hooks,
                    FaceAnalysisStage::Reconciling,
                    cancelled,
                ) || !reach_analysis_stage(
                    progress.display,
                    progress.hooks,
                    FaceAnalysisStage::Committing,
                    cancelled,
                ) {
                    return Ok(());
                }
                let stored = store_analysis(
                    &connection,
                    asset_id,
                    AnalyzedSourceIdentity {
                        size: source_size,
                        modified_ns: source_modified_ns,
                        fingerprint: &source_fingerprint,
                        path: Some(&path),
                    },
                    &worker_id,
                    &results,
                    cancelled,
                );
                if !cancelled.load(Ordering::Acquire) {
                    stored?;
                }
            }
        }
        Ok(None) => {}
        Err(error) => {
            let (category, class) = analysis_error_category(&error);
            record_analysis_failure(
                &connection,
                asset_id,
                &worker_id,
                AnalyzedSourceIdentity {
                    size: source_size,
                    modified_ns: source_modified_ns,
                    fingerprint: &source_fingerprint,
                    path: Some(&path),
                },
                AnalysisFailure { category, class },
            )?;
        }
    }
    if cancelled.load(Ordering::Acquire) {
        connection.execute(
            "UPDATE face_analysis_state SET status='pending', worker_id=NULL, lease_expires_at=NULL, \
             updated_at=datetime('now') WHERE asset_id=?1 AND status='running'
             AND worker_id=?2 AND source_size=?3 AND source_modified_ns=?4",
            params![asset_id, worker_id, source_size, source_modified_ns],
        )?;
    }
    Ok(())
}

fn analysis_error_category(error: &anyhow::Error) -> (&'static str, AnalysisFailureClass) {
    let message = error.to_string().to_ascii_lowercase();
    if message.contains("unsupported") {
        ("unsupported-format", AnalysisFailureClass::Deterministic)
    } else if message.contains("corrupt")
        || message.contains("invalid image")
        || message.contains("pixel budget")
        || message.contains("non-invertible")
    {
        ("source-decode-failed", AnalysisFailureClass::Deterministic)
    } else if message.contains("model") || message.contains("onnx") {
        ("inference-model-failed", AnalysisFailureClass::Transient)
    } else if message.contains("decode") || message.contains("source identity") {
        ("source-io-failed", AnalysisFailureClass::Transient)
    } else {
        ("analysis-failed", AnalysisFailureClass::Transient)
    }
}

fn store_analysis(
    connection: &Connection,
    asset_id: i64,
    source: AnalyzedSourceIdentity<'_>,
    worker_id: &str,
    results: &[(DetectedFace, Vec<f32>)],
    cancelled: &AtomicBool,
) -> Result<()> {
    let AnalyzedSourceIdentity {
        size: source_size,
        modified_ns: source_modified_ns,
        fingerprint: source_fingerprint,
        path: source_path,
    } = source;
    connection.execute("BEGIN IMMEDIATE", [])?;
    let result = (|| {
        if let Some(source_path) = source_path {
            let current = catalog::stable_content_identity(source_path)?;
            if current.size != source_size
                || current.modified_ns != source_modified_ns
                || current.fingerprint.as_slice() != source_fingerprint
            {
                bail!("source identity changed before face-analysis commit");
            }
        }
        let identity_is_current: bool = connection.query_row(
            "SELECT EXISTS(SELECT 1 FROM analysis_asset_inventory
               WHERE asset_id=?1 AND source_size=?2 AND source_modified_ns=?3
                 AND decoder_model_id=?4 AND source_fingerprint=?5)",
            params![
                asset_id,
                source_size,
                source_modified_ns,
                DECODER_MODEL_ID,
                source_fingerprint
            ],
            |row| row.get(0),
        )?;
        if !identity_is_current {
            bail!("source identity changed before face-analysis commit");
        }
        let owns_claim: bool = connection.query_row(
            "SELECT EXISTS(SELECT 1 FROM face_analysis_state WHERE asset_id=?1
               AND status='running' AND worker_id=?2 AND detector_model_id=?3
               AND recognizer_model_id='sface-2021dec' AND decoder_model_id=?4
               AND source_size=?5 AND source_modified_ns=?6
               AND source_fingerprint=?7
               AND lease_expires_at>datetime('now'))",
            params![
                asset_id,
                worker_id,
                ANALYSIS_MODEL_ID,
                DECODER_MODEL_ID,
                source_size,
                source_modified_ns,
                source_fingerprint
            ],
            |row| row.get(0),
        )?;
        if !owns_claim {
            bail!("face-analysis claim is stale or owned by another worker");
        }
        let existing = {
            let mut query = connection.prepare(
                "SELECT face.id, face.left, face.top, face.right, face.bottom,
                        face.provenance, embedding.landmarks
                   FROM faces face
                   LEFT JOIN face_embeddings embedding
                     ON embedding.face_id=face.id AND embedding.model_id='sface-2021dec'
                  WHERE face.asset_id=?1
                    AND face.provenance IN ('lightroom', 'rekesz')
                    AND face.retired=0",
            )?;
            let rows = query.query_map([asset_id], |row| {
                let provenance = row.get::<_, String>(5)?;
                let landmarks = match row.get::<_, Option<Vec<u8>>>(6)? {
                    None => StoredLandmarks::Missing,
                    Some(bytes) => decode_stored_landmarks(&bytes)
                        .map_or(StoredLandmarks::Invalid, StoredLandmarks::Valid),
                };
                Ok(ReconciliationFace {
                    id: row.get(0)?,
                    bounds: [row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?],
                    provenance: FaceProvenance::parse(&provenance),
                    landmarks,
                })
            })?;
            rows.collect::<rusqlite::Result<Vec<_>>>()?
        };
        let detections = results.iter().map(|(face, _)| face).collect::<Vec<_>>();
        let reconciled = conservative_reconciliation(&detections, &existing);
        let mut reconciliation_conflicts = 0_i64;
        let mut matched_existing = Vec::new();
        for (index, (face, embedding)) in results.iter().enumerate() {
            if cancelled.load(Ordering::Acquire) {
                bail!("face analysis cancelled before commit");
            }
            let face_id = if let ReconciliationMatch::Reuse(face_id) = &reconciled[index] {
                let face_id = *face_id;
                matched_existing.push(face_id);
                connection.execute(
                    "UPDATE faces SET left=?1, top=?2, right=?3, bottom=?4,
                       detector_model_id=?5, detection_confidence=?6, retired=0,
                       updated_at=datetime('now') WHERE id=?7 AND provenance='rekesz'",
                    params![
                        face.bounds[0],
                        face.bounds[1],
                        face.bounds[2],
                        face.bounds[3],
                        ANALYSIS_MODEL_ID,
                        face.confidence,
                        face_id
                    ],
                )?;
                face_id
            } else if let ReconciliationMatch::Conflict(face_ids) = &reconciled[index] {
                reconciliation_conflicts += 1;
                matched_existing.extend(face_ids.iter().copied());
                continue;
            } else {
                connection.execute(
                    "INSERT INTO faces(asset_id, provenance, source_key, left, top, right, bottom, \
                                       origin, ignored, retired, detector_model_id,
                                       detection_confidence, updated_at) \
                     VALUES(?1, 'rekesz', 'detected:' || lower(hex(randomblob(16))),
                            ?2, ?3, ?4, ?5, 'rekesz-detected', 0, 0, ?6, ?7, datetime('now'))",
                    params![asset_id, face.bounds[0], face.bounds[1], face.bounds[2],
                        face.bounds[3], ANALYSIS_MODEL_ID, face.confidence],
                )?;
                connection.last_insert_rowid()
            };
            if !matched_existing.contains(&face_id) {
                matched_existing.push(face_id);
            }
            let bytes: Vec<u8> = embedding
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect();
            let landmarks: Vec<u8> = face
                .landmarks
                .iter()
                .flatten()
                .flat_map(|value| value.to_le_bytes())
                .collect();
            connection.execute(
                "INSERT INTO face_embeddings(face_id, model_id, dimensions, vector, landmarks,
                                               source_size, source_modified_ns,
                                               source_fingerprint, created_at) \
                 VALUES(?1, 'sface-2021dec', ?2, ?3, ?4, ?5, ?6, ?7, datetime('now')) \
                 ON CONFLICT(face_id, model_id) DO UPDATE SET dimensions=excluded.dimensions, \
                    vector=excluded.vector, landmarks=excluded.landmarks,
                    source_size=excluded.source_size,
                    source_modified_ns=excluded.source_modified_ns,
                    source_fingerprint=excluded.source_fingerprint,
                    created_at=excluded.created_at",
                params![
                    face_id,
                    embedding.len() as i64,
                    bytes,
                    landmarks,
                    source_size,
                    source_modified_ns,
                    source_fingerprint
                ],
            )?;
            let confirmed_people = effective_confirmed_people(connection, face_id)?;
            catalog::recognition::enqueue_refresh_for_models_in_connection(
                connection,
                &["sface-2021dec".to_owned()],
                &confirmed_people,
                &[face_id],
            )?;
        }
        let stale = existing
            .iter()
            .map(|face| face.id)
            .filter(|face_id| !matched_existing.contains(face_id))
            .collect::<Vec<_>>();
        for face_id in stale {
            // Retain imported data and durable local decisions. Only derived
            // Rekesz inference artifacts become stale with a new detector run.
            connection.execute(
                "DELETE FROM face_assignments WHERE face_id=?1
                   AND provenance='rekesz' AND kind='suggested'",
                [face_id],
            )?;
            let confirmed_people = effective_confirmed_people(connection, face_id)?;
            let embedding_models = {
                let mut query =
                    connection.prepare("SELECT model_id FROM face_embeddings WHERE face_id=?1")?;
                query
                    .query_map([face_id], |row| row.get::<_, String>(0))?
                    .collect::<rusqlite::Result<Vec<_>>>()?
            };
            for model_id in embedding_models {
                catalog::recognition::enqueue_refresh_for_models_in_connection(
                    connection,
                    &[model_id],
                    &confirmed_people,
                    &[face_id],
                )?;
            }
            connection.execute("DELETE FROM face_embeddings WHERE face_id=?1", [face_id])?;
            connection.execute(
                "UPDATE faces SET retired=1, updated_at=datetime('now')
                   WHERE id=?1 AND provenance='rekesz' AND origin!='rekesz-manual'
                     AND NOT EXISTS (SELECT 1 FROM face_assignments assignment
                         WHERE assignment.face_id=faces.id AND assignment.provenance='rekesz'
                           AND assignment.kind='confirmed')",
                [face_id],
            )?;
        }
        if reconciliation_conflicts == 0 {
            connection.execute(
                "DELETE FROM face_reconciliation_conflicts WHERE asset_id=?1",
                [asset_id],
            )?;
        } else {
            connection.execute(
                "INSERT INTO face_reconciliation_conflicts(asset_id, detector_model_id,
                    conflict_count, updated_at)
                 VALUES(?1, ?2, ?3, datetime('now'))
                 ON CONFLICT(asset_id) DO UPDATE SET detector_model_id=excluded.detector_model_id,
                    conflict_count=excluded.conflict_count, updated_at=excluded.updated_at",
                params![asset_id, ANALYSIS_MODEL_ID, reconciliation_conflicts],
            )?;
        }
        connection.execute(
            "INSERT INTO face_analysis_state(asset_id, detector_model_id, source_size, \
                                              source_modified_ns, source_fingerprint,
                                              status, attempts, error, failure_class, updated_at) \
             VALUES(?1, 'yunet-2023mar+sface-2021dec', ?2, ?3, ?4, 'complete', 0, NULL, NULL, datetime('now')) \
             ON CONFLICT(asset_id) DO UPDATE SET status='complete', error=NULL, failure_class=NULL, \
                attempts=0, detector_model_id=excluded.detector_model_id, \
                source_size=excluded.source_size, source_modified_ns=excluded.source_modified_ns, \
                source_fingerprint=excluded.source_fingerprint,
                worker_id=NULL, lease_expires_at=NULL, retry_after=NULL, updated_at=excluded.updated_at",
            params![asset_id, source_size, source_modified_ns, source_fingerprint],
        )?;
        Ok::<_, anyhow::Error>(())
    })();
    if result.is_ok() {
        connection.execute("COMMIT", [])?;
    } else {
        let _ = connection.execute("ROLLBACK", []);
    }
    result
}

fn effective_confirmed_people(connection: &Connection, face_id: i64) -> Result<Vec<i64>> {
    let mut query = connection.prepare(
        "SELECT person_id FROM effective_confirmations WHERE face_id=?1 ORDER BY person_id",
    )?;
    Ok(query
        .query_map([face_id], |row| row.get::<_, i64>(0))?
        .collect::<rusqlite::Result<Vec<_>>>()?)
}

#[cfg(test)]
pub(crate) fn store_test_analysis(
    root: &Path,
    asset_id: i64,
    bounds: [f32; 4],
    landmarks: [[f32; 2]; 5],
) -> Result<()> {
    let connection = Connection::open(catalog::catalog_path(root))?;
    let identity: (i64, i64, Vec<u8>, Vec<u8>) = connection.query_row(
        "SELECT inventory.source_size, inventory.source_modified_ns,
                inventory.source_fingerprint, asset.relative_path
           FROM analysis_asset_inventory inventory JOIN assets asset ON asset.id=inventory.asset_id
          WHERE inventory.asset_id=?1",
        [asset_id],
        |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
    )?;
    let source_path = root.join(catalog::decode_relative_path(&identity.3)?);
    connection.execute(
        "UPDATE face_analysis_state SET status='running', worker_id='catalog-integration-test',
         lease_expires_at=datetime('now', '+30 minutes') WHERE asset_id=?1",
        [asset_id],
    )?;
    store_analysis(
        &connection,
        asset_id,
        AnalyzedSourceIdentity {
            size: identity.0,
            modified_ns: identity.1,
            fingerprint: &identity.2,
            path: Some(&source_path),
        },
        "catalog-integration-test",
        &[(
            DetectedFace {
                bounds,
                landmarks,
                confidence: 0.95,
            },
            vec![1.0, 0.0],
        )],
        &AtomicBool::new(false),
    )
}

#[cfg(test)]
pub(crate) fn record_test_analysis_failure(
    root: &Path,
    asset_id: i64,
    deterministic: bool,
) -> Result<()> {
    let connection = Connection::open(catalog::catalog_path(root))?;
    let (size, modified_ns, fingerprint): (i64, i64, Vec<u8>) = connection.query_row(
        "SELECT source_size,source_modified_ns,source_fingerprint
           FROM analysis_asset_inventory WHERE asset_id=?1",
        [asset_id],
        |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
    )?;
    connection.execute(
        "UPDATE face_analysis_state SET retry_after=datetime('now','-1 second')
           WHERE asset_id=?1 AND failure_class='transient'",
        [asset_id],
    )?;
    if !claim_analysis_state(
        &connection,
        asset_id,
        "catalog-failure-test",
        AnalyzedSourceIdentity {
            size,
            modified_ns,
            fingerprint: &fingerprint,
            path: None,
        },
    )? {
        bail!("test analysis row was not claimable");
    }
    record_analysis_failure(
        &connection,
        asset_id,
        "catalog-failure-test",
        AnalyzedSourceIdentity {
            size,
            modified_ns,
            fingerprint: &fingerprint,
            path: None,
        },
        AnalysisFailure {
            category: if deterministic {
                "unsupported-format"
            } else {
                "analysis-failed"
            },
            class: if deterministic {
                AnalysisFailureClass::Deterministic
            } else {
                AnalysisFailureClass::Transient
            },
        },
    )
}

struct PixelFace {
    bounds: [f32; 4],
    landmarks: [[f32; 2]; 5],
    confidence: f32,
}

impl PixelFace {
    fn normalized(mut self, width: u32, height: u32, scale: f32) -> Option<DetectedFace> {
        let width = width as f32;
        let height = height as f32;
        self.bounds = [
            (self.bounds[0] / scale / width).clamp(0.0, 1.0),
            (self.bounds[1] / scale / height).clamp(0.0, 1.0),
            (self.bounds[2] / scale / width).clamp(0.0, 1.0),
            (self.bounds[3] / scale / height).clamp(0.0, 1.0),
        ];
        if self.bounds[0] >= self.bounds[2] || self.bounds[1] >= self.bounds[3] {
            return None;
        }
        for point in &mut self.landmarks {
            point[0] /= scale * width;
            point[1] /= scale * height;
        }
        Some(DetectedFace {
            bounds: self.bounds,
            landmarks: self.landmarks,
            confidence: self.confidence,
        })
    }
}

#[cfg(test)]
pub fn cosine_similarity(left: &[f32], right: &[f32]) -> Option<f32> {
    (left.len() == right.len() && !left.is_empty()).then(|| {
        left.iter()
            .zip(right)
            .map(|(left, right)| left * right)
            .sum()
    })
}

fn normalize(values: &mut [f32]) -> Result<()> {
    let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    if !norm.is_finite() || norm <= f32::EPSILON {
        bail!("face recognizer returned a zero or non-finite embedding");
    }
    values.iter_mut().for_each(|value| *value /= norm);
    Ok(())
}

fn intersection_over_union(left: [f32; 4], right: [f32; 4]) -> f32 {
    let width = (left[2].min(right[2]) - left[0].max(right[0])).max(0.0);
    let height = (left[3].min(right[3]) - left[1].max(right[1])).max(0.0);
    let intersection = width * height;
    let left_area = (left[2] - left[0]).max(0.0) * (left[3] - left[1]).max(0.0);
    let right_area = (right[2] - right[0]).max(0.0) * (right[3] - right[1]).max(0.0);
    intersection / (left_area + right_area - intersection).max(f32::EPSILON)
}

#[derive(Clone, Copy)]
struct ReconciliationFace {
    id: i64,
    bounds: [f32; 4],
    provenance: FaceProvenance,
    landmarks: StoredLandmarks,
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum FaceProvenance {
    Lightroom,
    Rekesz,
}

impl FaceProvenance {
    fn parse(value: &str) -> Self {
        if value == "lightroom" {
            Self::Lightroom
        } else {
            Self::Rekesz
        }
    }
}

#[derive(Clone, Copy)]
enum StoredLandmarks {
    Missing,
    Valid([[f32; 2]; 5]),
    Invalid,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum ReconciliationMatch {
    Reuse(i64),
    Conflict(Vec<i64>),
    New,
}

fn decode_stored_landmarks(bytes: &[u8]) -> Option<[[f32; 2]; 5]> {
    if bytes.len() != 5 * 2 * std::mem::size_of::<f32>() {
        return None;
    }
    let mut result = [[0.0_f32; 2]; 5];
    for (value, chunk) in result.iter_mut().flatten().zip(bytes.as_chunks::<4>().0) {
        *value = f32::from_le_bytes(*chunk);
        if !value.is_finite() {
            return None;
        }
    }
    Some(result)
}

fn landmarks_are_consistent(detection: &DetectedFace, existing: &ReconciliationFace) -> bool {
    let existing_landmarks = match existing.landmarks {
        StoredLandmarks::Missing => return existing.provenance == FaceProvenance::Lightroom,
        StoredLandmarks::Valid(landmarks) => landmarks,
        StoredLandmarks::Invalid => return false,
    };
    let width = (existing.bounds[2] - existing.bounds[0]).max(f32::EPSILON);
    let height = (existing.bounds[3] - existing.bounds[1]).max(f32::EPSILON);
    let diagonal = width.hypot(height);
    let normalized_rms = detection
        .landmarks
        .iter()
        .zip(existing_landmarks)
        .map(|(left, right)| (left[0] - right[0]).powi(2) + (left[1] - right[1]).powi(2))
        .sum::<f32>()
        / 5.0;
    let normalized_rms = normalized_rms.sqrt() / diagonal;
    normalized_rms.is_finite() && normalized_rms <= 0.15
}

// Calibration policy: 0.55 IoU tolerates the small detector jitter observed
// across supported decoders without joining adjacent faces. The independent
// quarter-diagonal center guard protects unequal/aspect-shifted boxes whose IoU
// alone can still be misleading. Boundary tests below lock both inclusive
// thresholds and representative scale/aspect cases.
const RECONCILIATION_MIN_IOU: f32 = 0.55;
const RECONCILIATION_MAX_CENTER_DISTANCE: f32 = 0.25;

/// Reuses an existing row only when the overlap establishes an unambiguous
/// one-to-one relationship in both directions. A greedy best-match can attach
/// durable decisions to the wrong face when two detections overlap or their
/// detector ordering changes, so ambiguous rows deliberately become new faces.
fn conservative_reconciliation(
    detections: &[&DetectedFace],
    existing: &[ReconciliationFace],
) -> Vec<ReconciliationMatch> {
    let detection_candidates = detections
        .iter()
        .map(|detection| {
            existing
                .iter()
                .enumerate()
                .filter(|(_, face)| {
                    intersection_over_union(detection.bounds, face.bounds) >= RECONCILIATION_MIN_IOU
                        && normalized_center_distance(detection.bounds, face.bounds)
                            <= RECONCILIATION_MAX_CENTER_DISTANCE
                        && landmarks_are_consistent(detection, face)
                })
                .map(|(index, _)| index)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let cross_provenance_conflicts = detection_candidates
        .iter()
        .filter(|candidates| {
            candidates
                .iter()
                .any(|index| existing[*index].provenance == FaceProvenance::Lightroom)
                && candidates
                    .iter()
                    .any(|index| existing[*index].provenance == FaceProvenance::Rekesz)
        })
        .flatten()
        .copied()
        .collect::<BTreeSet<_>>();

    detection_candidates
        .iter()
        .enumerate()
        .map(|(detection_index, candidates)| {
            if candidates
                .iter()
                .any(|index| cross_provenance_conflicts.contains(index))
            {
                return ReconciliationMatch::Conflict(
                    candidates.iter().map(|index| existing[*index].id).collect(),
                );
            }
            let [existing_index] = candidates.as_slice() else {
                return ReconciliationMatch::New;
            };
            let reverse_matches = detection_candidates
                .iter()
                .enumerate()
                .filter(|(_, other)| other.contains(existing_index))
                .map(|(index, _)| index)
                .collect::<Vec<_>>();
            if reverse_matches.as_slice() == [detection_index] {
                ReconciliationMatch::Reuse(existing[*existing_index].id)
            } else {
                ReconciliationMatch::New
            }
        })
        .collect()
}

fn normalized_center_distance(left: [f32; 4], right: [f32; 4]) -> f32 {
    let left_center = [(left[0] + left[2]) * 0.5, (left[1] + left[3]) * 0.5];
    let right_center = [(right[0] + right[2]) * 0.5, (right[1] + right[3]) * 0.5];
    let scale = (right[2] - right[0])
        .max(f32::EPSILON)
        .hypot((right[3] - right[1]).max(f32::EPSILON));
    (left_center[0] - right_center[0]).hypot(left_center[1] - right_center[1]) / scale
}

fn align_face(image: &DynamicImage, face: &DetectedFace) -> Result<RgbImage> {
    const TARGET: [[f32; 2]; 5] = [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ];
    let (width, height) = image.dimensions();
    let source = face
        .landmarks
        .map(|point| [point[0] * width as f32, point[1] * height as f32]);
    let transform = similarity_transform(&source, &TARGET)?;
    let determinant = transform[0] * transform[0] + transform[1] * transform[1];
    if determinant <= f32::EPSILON {
        bail!("degenerate face landmark transform");
    }
    let source = image.to_rgb8();
    let mut aligned = RgbImage::new(112, 112);
    for y in 0..112 {
        for x in 0..112 {
            let dx = x as f32 - transform[2];
            let dy = y as f32 - transform[3];
            let source_x = (transform[0] * dx + transform[1] * dy) / determinant;
            let source_y = (-transform[1] * dx + transform[0] * dy) / determinant;
            aligned.put_pixel(x, y, bilinear(&source, source_x, source_y));
        }
    }
    Ok(aligned)
}

fn similarity_transform(source: &[[f32; 2]; 5], target: &[[f32; 2]; 5]) -> Result<[f32; 4]> {
    let source_mean = mean(source);
    let target_mean = mean(target);
    let (mut denominator, mut a, mut b) = (0.0, 0.0, 0.0);
    for (source, target) in source.iter().zip(target) {
        let (sx, sy) = (source[0] - source_mean[0], source[1] - source_mean[1]);
        let (tx, ty) = (target[0] - target_mean[0], target[1] - target_mean[1]);
        denominator += sx * sx + sy * sy;
        a += sx * tx + sy * ty;
        b += sx * ty - sy * tx;
    }
    if denominator <= f32::EPSILON {
        bail!("degenerate face landmarks");
    }
    a /= denominator;
    b /= denominator;
    Ok([
        a,
        b,
        target_mean[0] - a * source_mean[0] + b * source_mean[1],
        target_mean[1] - b * source_mean[0] - a * source_mean[1],
    ])
}

fn mean(points: &[[f32; 2]; 5]) -> [f32; 2] {
    let sum = points.iter().fold([0.0, 0.0], |sum, point| {
        [sum[0] + point[0], sum[1] + point[1]]
    });
    [sum[0] / 5.0, sum[1] / 5.0]
}

fn bilinear(image: &RgbImage, x: f32, y: f32) -> Rgb<u8> {
    if x < 0.0 || y < 0.0 || x >= image.width() as f32 || y >= image.height() as f32 {
        return Rgb([0, 0, 0]);
    }
    let (x0, y0) = (x.floor() as u32, y.floor() as u32);
    let (x1, y1) = (
        (x0 + 1).min(image.width() - 1),
        (y0 + 1).min(image.height() - 1),
    );
    let (fx, fy) = (x - x0 as f32, y - y0 as f32);
    let mut result = [0_u8; 3];
    for (channel, output) in result.iter_mut().enumerate() {
        let top = f32::from(image.get_pixel(x0, y0)[channel]) * (1.0 - fx)
            + f32::from(image.get_pixel(x1, y0)[channel]) * fx;
        let bottom = f32::from(image.get_pixel(x0, y1)[channel]) * (1.0 - fx)
            + f32::from(image.get_pixel(x1, y1)[channel]) * fx;
        *output = (top * (1.0 - fy) + bottom * fy).round() as u8;
    }
    Rgb(result)
}

#[cfg(test)]
mod tests;
