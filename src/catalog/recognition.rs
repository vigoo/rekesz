//! Local, rebuildable face-recognition suggestions.
//!
//! Catalog-scale candidate retrieval uses a rebuildable in-memory HNSW index
//! over person centroids, followed by exact centroid and exemplar scoring of a
//! bounded shortlist. The persisted assignments remain the source of truth;
//! prototypes, the ANN graph, and suggestion metadata are derived caches.

use std::collections::{BTreeMap, HashMap, HashSet, hash_map::Entry};
use std::path::{Path, PathBuf};
use std::sync::{
    Arc, Condvar, Mutex, OnceLock,
    atomic::{AtomicBool, Ordering},
};

use super::{CATALOG_DIR, CATALOG_FILE};
use crate::scheduler::MaintenanceAdmission;
use anyhow::{Context, Result, bail};
use hnsw_rs::prelude::{DistCosine, Hnsw};
use rusqlite::{Connection, OptionalExtension, params};
use serde::Deserialize;
use sha2::{Digest, Sha256};

pub use super::recognition_policy::RecognitionPolicy;
use super::recognition_policy::{accepted_margin, dot, normalize, robust_centroid};

pub const RECOGNITION_POLICY_VERSION: i64 = 1;
pub const PROTOTYPE_ALGORITHM_VERSION: i64 = 1;
pub const ANN_INDEX_VERSION: i64 = 1;
pub const PRODUCTION_SFACE_MODEL_ID: &str = "sface-2021dec";
pub const PRODUCTION_SFACE_POLICY_DIGEST: &str =
    "7feab1985699a5aa957c16e7326ac80b8ad5436c2e3a303fd0570583a9239a09";
const PRODUCTION_SFACE_MODEL_DIGEST: &str =
    "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79";
const PRODUCTION_SFACE_POLICY: &[u8] =
    include_bytes!("../../assets/face/sface_2021dec_policy_v1.json");
const CALIBRATION_VERIFIER_SOURCE: &[u8] =
    include_bytes!("../../scripts/verify-sface-policy-calibration.py");
const CALIBRATION_EXTRACTOR_SOURCE: &[u8] =
    include_bytes!("../bin/sface_calibration_embeddings.rs");
const PRODUCTION_INFERENCE_SOURCE: &[u8] = include_bytes!("../face_engine.rs");
const PRODUCTION_POLICY_SOURCE: &[u8] = include_bytes!("recognition_policy.rs");
const CALIBRATION_PROVENANCE: &[u8] = include_bytes!("../../assets/face/calibration/README.md");
const DEPENDENCY_LOCK: &[u8] = include_bytes!("../../Cargo.lock");

const MAX_PEOPLE: usize = 100_000;
const MAX_EXEMPLARS: usize = 250_000;
const MAX_EXEMPLARS_PER_PERSON: usize = 64;
const MAX_TRAINING_EXEMPLARS_PER_PERSON: usize = 1_024;
#[cfg(test)]
const MAX_INCREMENTAL_FACES: usize = 10_000;
const DEFAULT_REFRESH_BATCH: usize = 16;
const RESCORE_EXPANSION_BATCH: usize = 64;
const MAX_QUEUED_FACES: usize = 256;
const EXACT_SEARCH_LIMIT: usize = 256;
const ANN_CANDIDATES: usize = 128;
const ANN_EF_SEARCH: usize = 1_024;
const MAX_ANN_CANDIDATES: usize = 512;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BundledPolicyArtifact {
    artifact_format: i64,
    policy_version: i64,
    prototype_algorithm_version: i64,
    model_id: String,
    model_sha256: String,
    calibration_report: CalibrationReport,
    license: PolicyLicense,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CalibrationReport {
    report_id: String,
    generated_by: String,
    generated_at: String,
    upstream_benchmark: UpstreamBenchmark,
    held_out_data: HeldOutData,
    selection_constraints: SelectionConstraints,
    operating_point: RecognitionPolicy,
    held_out_results: HeldOutResults,
    limitations: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct UpstreamBenchmark {
    name: String,
    source: String,
    source_url: String,
    pair_count: u64,
    published_cosine_threshold: f32,
    published_accuracy: f64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HeldOutData {
    provenance: String,
    verifier_version: String,
    verifier_source_sha256: String,
    extractor_source_sha256: String,
    production_inference_source_sha256: String,
    production_policy_source_sha256: String,
    corpus_provenance_sha256: String,
    dependency_lock_sha256: String,
    generator_spec_sha256: String,
    identity_count: u64,
    calibration_queries: u64,
    query_count: u64,
    positive_queries: u64,
    negative_queries: u64,
    embedding_dimensions: u64,
    split_protocol: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HeldOutResults {
    true_positive: u64,
    false_negative: u64,
    false_positive: u64,
    true_negative: u64,
    unknown_false_accepts: u64,
    misidentifications: u64,
    false_accept_rate: f64,
    false_reject_rate: f64,
    precision: f64,
    recall: f64,
    identification_error_rate: f64,
    safety_error_rate: f64,
    unknown_far_one_sided_95_upper_bound: f64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SelectionConstraints {
    maximum_unknown_false_accept_rate: f64,
    maximum_identification_error_rate: f64,
    maximum_combined_safety_error_rate: f64,
    minimum_precision: f64,
    objective: String,
    statistical_claim: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PolicyLicense {
    model: String,
    file: String,
    calibration_metadata: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CalibrationStatus {
    pub active: bool,
    pub policy_version: i64,
    pub artifact_digest: String,
}

fn decode_hex_digest(value: &str) -> Result<[u8; 32]> {
    if value.len() != 64 {
        bail!("calibration digest must contain exactly 64 hexadecimal characters");
    }
    let mut digest = [0_u8; 32];
    for (index, byte) in digest.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&value[index * 2..index * 2 + 2], 16)
            .context("invalid calibration digest")?;
    }
    Ok(digest)
}

fn bundled_production_policy() -> Result<(RecognitionPolicy, [u8; 32])> {
    let actual = hex::encode(Sha256::digest(PRODUCTION_SFACE_POLICY));
    if actual != PRODUCTION_SFACE_POLICY_DIGEST {
        bail!("bundled SFace calibration policy failed integrity validation");
    }
    let artifact: BundledPolicyArtifact =
        serde_json::from_slice(PRODUCTION_SFACE_POLICY).context("parse bundled SFace policy")?;
    if artifact.artifact_format != 2
        || artifact.policy_version != RECOGNITION_POLICY_VERSION
        || artifact.prototype_algorithm_version != PROTOTYPE_ALGORITHM_VERSION
        || artifact.model_id != PRODUCTION_SFACE_MODEL_ID
        || artifact.model_sha256 != PRODUCTION_SFACE_MODEL_DIGEST
    {
        bail!("bundled SFace calibration policy is incompatible with this recognizer");
    }
    validate_calibration_report(&artifact.calibration_report, &artifact.license)?;
    let policy = validate_policy_value(artifact.calibration_report.operating_point)?;
    Ok((policy, decode_hex_digest(PRODUCTION_SFACE_POLICY_DIGEST)?))
}

fn validate_calibration_report(report: &CalibrationReport, license: &PolicyLicense) -> Result<()> {
    let data = &report.held_out_data;
    let results = &report.held_out_results;
    let constraints = &report.selection_constraints;
    if report.report_id != "rekesz-sface-policy-2026-08-30-v3"
        || report.generated_by != "rekesz-calibration-verifier-v4"
        || report.generated_at != "2026-08-30T00:00:00Z"
        || report.upstream_benchmark.name != "LFW 6000-pair verification protocol"
        || report.upstream_benchmark.source != "OpenCV Zoo SFace 2021-12 model card"
        || report.upstream_benchmark.source_url
            != "https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface"
        || report.upstream_benchmark.pair_count != 6_000
        || report.upstream_benchmark.published_cosine_threshold != 0.363
        || (report.upstream_benchmark.published_accuracy - 0.996).abs() > 1e-9
        || data.verifier_version != "rekesz-calibration-verifier-v4"
        || data.verifier_source_sha256
            != "887bf9c8f2058157a0a0e3079d74207522d242021884639f7047259005a548d3"
        || data.extractor_source_sha256
            != "3989cc579043bd226f5b1cf5fbd562bb2138966528d79719102d6f34581553f7"
        || data.production_inference_source_sha256
            != "1bb397eb237675762174628cf1a733ebeabc00e2552a7977789bf1ff562c0b52"
        || data.production_policy_source_sha256
            != "3d2e0693777abdf37fe0085feb7badf66a7fb3859c448e23676eb36ceb4a8bda"
        || data.corpus_provenance_sha256
            != "119d83ca04a64beab77799daa79396474a34d5a2a2dbb19a188e1e818b1926da"
        || data.dependency_lock_sha256
            != "e26014349565ecb14206828c17f4f7c95eaba8cb8599a41731e09b4b87ff6dae"
        || data.generator_spec_sha256
            != "df7dbebefc2128fb1bed2363a15394857f3d5d96ef5bb08b8905c12249414fee"
        || data.identity_count != 20
        || data.calibration_queries != 28
        || data.query_count != 16
        || data.positive_queries != 6
        || data.negative_queries != 10
        || data.embedding_dimensions != 128
        || !data.provenance.contains("exact packaged YuNet and SFace")
        || !data
            .provenance
            .contains("No user photographs or persisted embeddings")
        || !data.split_protocol.contains("Identity-disjoint partitions")
        || !data.split_protocol.contains("robust centroid")
        || constraints.maximum_unknown_false_accept_rate != 0.0
        || constraints.maximum_identification_error_rate != 0.0
        || constraints.maximum_combined_safety_error_rate != 0.0
        || constraints.minimum_precision != 1.0
        || constraints.objective
            != "maximize observed calibration recall subject to zero observed safety errors; ties choose larger absolute, exemplar, then margin grid values"
        || !constraints.statistical_claim.starts_with("none;")
        || report.limitations.is_empty()
        || license.model != "Apache-2.0"
        || license.file != "LICENSE-SFACE"
        || license.calibration_metadata != "Rekesz project source license"
    {
        bail!("bundled SFace calibration report metadata is incomplete or unexpected");
    }
    for (bytes, reported, name) in [
        (
            CALIBRATION_VERIFIER_SOURCE,
            data.verifier_source_sha256.as_str(),
            "verifier source",
        ),
        (
            CALIBRATION_EXTRACTOR_SOURCE,
            data.extractor_source_sha256.as_str(),
            "extractor source",
        ),
        (
            PRODUCTION_INFERENCE_SOURCE,
            data.production_inference_source_sha256.as_str(),
            "production inference source",
        ),
        (
            PRODUCTION_POLICY_SOURCE,
            data.production_policy_source_sha256.as_str(),
            "production policy source",
        ),
        (
            CALIBRATION_PROVENANCE,
            data.corpus_provenance_sha256.as_str(),
            "corpus provenance",
        ),
        (
            DEPENDENCY_LOCK,
            data.dependency_lock_sha256.as_str(),
            "dependency lock",
        ),
    ] {
        if hex::encode(Sha256::digest(bytes)) != reported {
            bail!("bundled SFace calibration {name} binding is stale");
        }
    }
    if results.true_positive + results.false_negative != data.positive_queries
        || results.unknown_false_accepts + results.true_negative != data.negative_queries
        || results.false_positive != results.unknown_false_accepts + results.misidentifications
    {
        bail!("bundled SFace calibration confusion matrix does not match held-out counts");
    }
    let far = results.unknown_false_accepts as f64 / data.negative_queries as f64;
    let frr = results.false_negative as f64 / data.positive_queries as f64;
    let precision =
        results.true_positive as f64 / (results.true_positive + results.false_positive) as f64;
    let recall = results.true_positive as f64 / data.positive_queries as f64;
    let identification_error_rate =
        results.misidentifications as f64 / data.positive_queries as f64;
    let safety_error_rate =
        results.false_positive as f64 / (data.positive_queries + data.negative_queries) as f64;
    let unknown_far_upper = 1.0 - 0.05_f64.powf(1.0 / data.negative_queries as f64);
    for (reported, derived) in [
        (results.false_accept_rate, far),
        (results.false_reject_rate, frr),
        (results.precision, precision),
        (results.recall, recall),
        (results.identification_error_rate, identification_error_rate),
        (results.safety_error_rate, safety_error_rate),
        (
            results.unknown_far_one_sided_95_upper_bound,
            unknown_far_upper,
        ),
    ] {
        if !reported.is_finite() || (reported - derived).abs() > 1e-7 {
            bail!("bundled SFace calibration metrics are not reproducible from its counts");
        }
    }
    Ok(())
}

pub fn validate_production_calibration_artifact() -> Result<()> {
    bundled_production_policy().map(|_| ())
}

/// Installs the shipped, model-bound production policy if the catalog does not
/// already contain that exact artifact. Repeated startup calls are no-ops.
pub fn ensure_production_calibration(root: &Path) -> Result<bool> {
    let (policy, digest) = bundled_production_policy()?;
    let connection = writable_connection(root)?;
    ensure_recognition_schema(&connection)?;
    let current: bool = connection.query_row(
        "SELECT EXISTS(SELECT 1 FROM recognition_policy_artifacts
         WHERE model_id=?1 AND policy_version=?2 AND prototype_algorithm_version=?3
           AND calibration_digest=?4 AND enabled=1)",
        params![
            PRODUCTION_SFACE_MODEL_ID,
            RECOGNITION_POLICY_VERSION,
            PROTOTYPE_ALGORITHM_VERSION,
            digest.as_slice()
        ],
        |row| row.get(0),
    )?;
    if current {
        return Ok(false);
    }
    install_calibration_artifact(root, PRODUCTION_SFACE_MODEL_ID, policy, &digest)?;
    Ok(true)
}

pub fn production_calibration_status(connection: &Connection) -> Result<CalibrationStatus> {
    let (policy, digest) = bundled_production_policy()?;
    let active: bool = connection.query_row(
        "SELECT EXISTS(SELECT 1 FROM recognition_policy_artifacts
         WHERE model_id=?1 AND policy_version=?2 AND prototype_algorithm_version=?3
           AND calibration_digest=?4 AND enabled=1
           AND absolute_threshold=?5 AND nearest_exemplar_threshold=?6
           AND second_best_margin=?7)",
        params![
            PRODUCTION_SFACE_MODEL_ID,
            RECOGNITION_POLICY_VERSION,
            PROTOTYPE_ALGORITHM_VERSION,
            digest.as_slice(),
            policy.absolute_threshold,
            policy.nearest_exemplar_threshold,
            policy.second_best_margin
        ],
        |row| row.get(0),
    )?;
    Ok(CalibrationStatus {
        active,
        policy_version: RECOGNITION_POLICY_VERSION,
        artifact_digest: PRODUCTION_SFACE_POLICY_DIGEST.to_owned(),
    })
}

fn calibrated_policy(connection: &Connection, model_id: &str) -> Result<Option<RecognitionPolicy>> {
    if model_id == PRODUCTION_SFACE_MODEL_ID && !production_calibration_status(connection)?.active {
        return Ok(None);
    }
    connection
        .query_row(
            "SELECT absolute_threshold, nearest_exemplar_threshold, second_best_margin
             FROM recognition_policy_artifacts WHERE model_id=?1 AND policy_version=?2
               AND prototype_algorithm_version=?3 AND enabled=1
               AND length(calibration_digest)>=32",
            params![
                model_id,
                RECOGNITION_POLICY_VERSION,
                PROTOTYPE_ALGORITHM_VERSION
            ],
            |row| {
                Ok(RecognitionPolicy {
                    absolute_threshold: row.get(0)?,
                    nearest_exemplar_threshold: row.get(1)?,
                    second_best_margin: row.get(2)?,
                })
            },
        )
        .optional()?
        .map(validate_policy_value)
        .transpose()
}

fn validate_policy_value(policy: RecognitionPolicy) -> Result<RecognitionPolicy> {
    validate_policy(policy)?;
    Ok(policy)
}

#[derive(Clone, Debug, PartialEq)]
pub struct RecognitionSuggestion {
    pub face_id: i64,
    pub person_id: i64,
    pub score: f32,
    pub nearest_exemplar_score: f32,
    pub second_best_margin: f32,
    pub recognizer_model_id: String,
    pub prototype_generation: i64,
    pub policy_version: i64,
    pub ann_index_version: i64,
    pub scored_index_generation: i64,
    policy: RecognitionPolicy,
    prototype_generations: Vec<(i64, i64)>,
}

#[derive(Clone)]
struct Prototype {
    person_id: i64,
    generation: i64,
    centroid: Vec<f32>,
    exemplars: Vec<Vec<f32>>,
}

struct PrototypeIndex {
    version: i64,
    generation: i64,
    prototypes: Vec<Prototype>,
    ann: Option<Hnsw<'static, f32, DistCosine>>,
    metric: MetricIndex,
}

#[derive(Clone, Copy)]
struct MetricNode {
    prototype_id: usize,
    threshold: f32,
    inner: Option<usize>,
    outer: Option<usize>,
}

struct MetricIndex {
    nodes: Vec<MetricNode>,
    root: Option<usize>,
}

struct IndexBuild {
    result: Mutex<Option<std::result::Result<Arc<PrototypeIndex>, Arc<str>>>>,
    wake: Condvar,
}

type IndexCache = HashMap<(PathBuf, String, i64), Arc<IndexBuild>>;
type ModelBuildKey = (PathBuf, String);
type ModelBuildGates = HashMap<ModelBuildKey, Arc<BuildGate>>;
static PROTOTYPE_INDEX_CACHE: OnceLock<Mutex<IndexCache>> = OnceLock::new();
static ANN_BUILD_POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();
static MODEL_BUILD_GATES: OnceLock<Mutex<ModelBuildGates>> = OnceLock::new();

#[cfg(test)]
#[derive(Clone, Copy, PartialEq, Eq)]
enum TestBuildPhase {
    SqlRows,
    MetricNodes,
    HnswInsertions,
}

#[cfg(test)]
struct TestBuildHook {
    requested: TestBuildPhase,
    reached: Mutex<bool>,
    released: Mutex<bool>,
    wake: Condvar,
}

#[cfg(test)]
static TEST_BUILD_HOOKS: OnceLock<Mutex<HashMap<PathBuf, Arc<TestBuildHook>>>> = OnceLock::new();

#[cfg(test)]
thread_local! {
    static ACTIVE_TEST_BUILD_HOOK: std::cell::RefCell<Option<Arc<TestBuildHook>>> = const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
fn signal_test_build_phase(hook: &TestBuildHook, phase: TestBuildPhase) {
    if hook.requested == phase {
        *hook
            .reached
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = true;
        hook.wake.notify_all();
        let mut released = hook
            .released
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        while !*released {
            released = hook
                .wake
                .wait(released)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }
}

struct BuildGate {
    active: Mutex<bool>,
    wake: Condvar,
}

struct BuildPermit(Arc<BuildGate>);

impl Drop for BuildPermit {
    fn drop(&mut self) {
        let mut active = self
            .0
            .active
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *active = false;
        self.0.wake.notify_all();
    }
}

#[derive(Clone, Copy)]
struct Candidate {
    person_id: i64,
    generation: i64,
    centroid_score: f32,
    nearest_score: f32,
}

type CertifiedBestTwo = (Option<(usize, f32)>, Option<(usize, f32)>, usize);

fn writable_connection(root: &Path) -> Result<Connection> {
    let connection = Connection::open(root.join(CATALOG_DIR).join(CATALOG_FILE))?;
    connection.execute_batch("PRAGMA foreign_keys=ON; PRAGMA busy_timeout=5000;")?;
    Ok(connection)
}

pub fn enqueue_refresh_in_connection(
    connection: &Connection,
    person_ids: &[i64],
    face_ids: &[i64],
) -> Result<()> {
    ensure_recognition_schema(connection)?;
    let model_ids = model_ids_in_connection(connection)?;
    enqueue_refresh_for_models_in_connection(connection, &model_ids, person_ids, face_ids)
}

pub(crate) fn model_ids_in_connection(connection: &Connection) -> Result<Vec<String>> {
    let mut query = connection.prepare(
        "SELECT model_id FROM face_embeddings
         UNION SELECT model_id FROM person_prototypes
         UNION SELECT recognizer_model_id FROM recognition_suggestions
         UNION SELECT model_id FROM recognition_policy_artifacts
         UNION SELECT model_id FROM recognition_index_state",
    )?;
    let model_ids = query
        .query_map([], |row| row.get::<_, String>(0))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    Ok(model_ids)
}

pub(crate) fn enqueue_refresh_for_models_in_connection(
    connection: &Connection,
    model_ids: &[String],
    person_ids: &[i64],
    face_ids: &[i64],
) -> Result<()> {
    ensure_recognition_schema(connection)?;
    for model_id in model_ids {
        for person_id in person_ids {
            enqueue_one(connection, "person", *person_id, model_id)?;
        }
        for face_id in face_ids {
            enqueue_one(connection, "face", *face_id, model_id)?;
        }
        if !person_ids.is_empty() {
            enqueue_global_rescore(connection, model_id)?;
        }
    }
    Ok(())
}

pub fn install_calibration_artifact(
    root: &Path,
    model_id: &str,
    policy: RecognitionPolicy,
    calibration_digest: &[u8],
) -> Result<()> {
    validate_policy(policy)?;
    if calibration_digest.len() < 32 {
        bail!("calibration artifact digest must contain at least 32 bytes");
    }
    let mut connection = writable_connection(root)?;
    ensure_recognition_schema(&connection)?;
    let transaction = connection.transaction()?;
    transaction.execute(
        "INSERT INTO recognition_policy_artifacts(model_id, policy_version,
         prototype_algorithm_version, absolute_threshold, nearest_exemplar_threshold,
         second_best_margin, calibration_digest, enabled, installed_at)
         VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, 1, datetime('now'))
         ON CONFLICT(model_id) DO UPDATE SET policy_version=excluded.policy_version,
         prototype_algorithm_version=excluded.prototype_algorithm_version,
         absolute_threshold=excluded.absolute_threshold,
         nearest_exemplar_threshold=excluded.nearest_exemplar_threshold,
         second_best_margin=excluded.second_best_margin,
         calibration_digest=excluded.calibration_digest, enabled=1,
         installed_at=excluded.installed_at",
        params![
            model_id,
            RECOGNITION_POLICY_VERSION,
            PROTOTYPE_ALGORITHM_VERSION,
            policy.absolute_threshold,
            policy.nearest_exemplar_threshold,
            policy.second_best_margin,
            calibration_digest
        ],
    )?;
    transaction.execute(
        "INSERT INTO recognition_refresh_queue(entity_kind, entity_id, model_id, status,
                                                attempts, updated_at)
         SELECT 'person', person_id, ?1, 'pending', 0, datetime('now')
         FROM effective_confirmations GROUP BY person_id
         ON CONFLICT(entity_kind, entity_id, model_id) DO UPDATE SET status='pending', attempts=0,
         worker_id=NULL, lease_expires_at=NULL, retry_after=NULL, error=NULL,
         updated_at=excluded.updated_at",
        [model_id],
    )?;
    enqueue_global_rescore(&transaction, model_id)?;
    transaction.commit()?;
    Ok(())
}

fn enqueue_global_rescore(connection: &Connection, model_id: &str) -> Result<()> {
    connection.execute(
        "INSERT INTO recognition_rescore_state(model_id, after_face_id, updated_at)
         VALUES(?1, 0, datetime('now'))
         ON CONFLICT(model_id) DO UPDATE SET after_face_id=0, updated_at=excluded.updated_at",
        [model_id],
    )?;
    enqueue_one(connection, "person", 0, model_id)
}

/// Claims and processes one small durable refresh batch. Failures are retained
/// with capped exponential retry; a crashed worker's lease becomes claimable.
pub fn process_pending_refresh_batch(
    root: &Path,
    worker_id: &str,
    admission: &MaintenanceAdmission,
    cancelled: &AtomicBool,
) -> Result<usize> {
    if worker_id.trim().is_empty() || worker_id.len() > 128 {
        bail!("recognition worker id must contain 1 to 128 characters");
    }
    let mut connection = writable_connection(root)?;
    ensure_recognition_schema(&connection)?;
    let transaction = connection.transaction()?;
    let rows = {
        let mut query = transaction.prepare(
            "SELECT id, entity_kind, entity_id, model_id FROM recognition_refresh_queue
             WHERE (status='pending' AND (retry_after IS NULL OR retry_after<=datetime('now')))
                OR (status='running' AND lease_expires_at<=datetime('now'))
             ORDER BY CASE WHEN entity_kind='person' AND entity_id>0 THEN 0
                           WHEN entity_kind='person' THEN 1 ELSE 2 END,
                      id LIMIT ?1",
        )?;
        query
            .query_map([DEFAULT_REFRESH_BATCH as i64], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, String>(3)?,
                ))
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?
    };
    if rows.is_empty() {
        transaction.commit()?;
        return Ok(0);
    }
    for (id, _, _, _) in &rows {
        transaction.execute(
            "UPDATE recognition_refresh_queue SET status='running', worker_id=?2,
                 lease_expires_at=datetime('now', '+5 minutes'), attempts=attempts+1,
                 updated_at=datetime('now') WHERE id=?1",
            params![id, worker_id],
        )?;
    }
    transaction.commit()?;

    let mut processed = 0;
    let mut prototype_cache: HashMap<String, Arc<PrototypeIndex>> = HashMap::new();
    for (index, (id, kind, entity_id, model_id)) in rows.iter().enumerate() {
        if cancelled.load(Ordering::Acquire) || !admission.checkpoint(cancelled) {
            let remaining = rows[index..].iter().map(|row| row.0).collect::<Vec<_>>();
            release_claims(root, worker_id, &remaining)?;
            break;
        }
        let Some(policy) = calibrated_policy(&writable_connection(root)?, model_id)? else {
            // Automatic suggestions are intentionally inert until an explicit,
            // version-matching calibration artifact has been installed.
            finish_claims(root, worker_id, &[*id])?;
            processed += 1;
            continue;
        };
        if writable_connection(root)?.execute(
            "UPDATE recognition_refresh_queue SET lease_expires_at=datetime('now', '+5 minutes'),
             updated_at=datetime('now') WHERE id=?1 AND worker_id=?2 AND status='running'",
            params![id, worker_id],
        )? != 1
        {
            continue;
        }
        let operation = match kind.as_str() {
            "person" if *entity_id == 0 => RefreshOperation::Catalog,
            "person" => {
                prototype_cache.remove(model_id);
                RefreshOperation::Person(*entity_id)
            }
            "face" => {
                let loaded = if let Some(prototypes) = prototype_cache.get(model_id) {
                    Ok(prototypes.clone())
                } else {
                    let read = writable_connection(root)?;
                    cached_prototype_index(root, &read, model_id, Some((admission, cancelled)))
                        .inspect(|value| {
                            prototype_cache.insert(model_id.clone(), value.clone());
                        })
                };
                match loaded {
                    Ok(prototypes) => match calculate_face_suggestion(
                        &writable_connection(root)?,
                        *entity_id,
                        model_id,
                        policy,
                        prototypes.as_ref(),
                        admission,
                        cancelled,
                    ) {
                        Ok(suggestion) => RefreshOperation::Face(
                            *entity_id,
                            FaceCalculation {
                                suggestion,
                                policy,
                                ann_index_version: prototypes.version,
                                index_generation: prototypes.generation,
                                prototype_generations: prototypes
                                    .prototypes
                                    .iter()
                                    .map(|value| (value.person_id, value.generation))
                                    .collect(),
                            },
                        ),
                        Err(_) if cancelled.load(Ordering::Acquire) => {
                            let remaining =
                                rows[index..].iter().map(|row| row.0).collect::<Vec<_>>();
                            release_claims(root, worker_id, &remaining)?;
                            break;
                        }
                        Err(error) => {
                            fail_claim(root, worker_id, *id, &bounded_error(&error))?;
                            continue;
                        }
                    },
                    Err(_) if cancelled.load(Ordering::Acquire) => {
                        let remaining = rows[index..].iter().map(|row| row.0).collect::<Vec<_>>();
                        release_claims(root, worker_id, &remaining)?;
                        break;
                    }
                    Err(error) => {
                        fail_claim(root, worker_id, *id, &bounded_error(&error))?;
                        continue;
                    }
                }
            }
            _ => {
                fail_claim(
                    root,
                    worker_id,
                    *id,
                    "invalid recognition queue entity kind",
                )?;
                continue;
            }
        };
        let mut connection = writable_connection(root)?;
        let transaction = connection.transaction()?;
        let still_owned: bool = transaction.query_row(
            "SELECT EXISTS(SELECT 1 FROM recognition_refresh_queue WHERE id=?1 AND worker_id=?2
             AND status='running' AND lease_expires_at>datetime('now'))",
            params![id, worker_id],
            |row| row.get(0),
        )?;
        if !still_owned {
            transaction.rollback()?;
            continue;
        }
        transaction.execute_batch("SAVEPOINT recognition_refresh")?;
        let result = match operation {
            RefreshOperation::Person(person_id) => {
                let result =
                    rebuild_person_prototype_in(&transaction, person_id, model_id).map(|_| None);
                if result.is_ok() {
                    enqueue_global_rescore(&transaction, model_id)?;
                }
                result
            }
            RefreshOperation::Catalog => {
                expand_global_rescore(&transaction, model_id).map(|_| None)
            }
            RefreshOperation::Face(face_id, calculation) => store_calculated_suggestion(
                &transaction,
                face_id,
                model_id,
                &calculation,
                *id,
                worker_id,
            ),
        };
        match result {
            Ok(_) => {
                transaction.execute_batch("RELEASE recognition_refresh")?;
                transaction.execute(
                    "DELETE FROM recognition_refresh_queue WHERE id=?1 AND worker_id=?2",
                    params![id, worker_id],
                )?;
                processed += 1;
            }
            Err(error) => {
                transaction.execute_batch(
                    "ROLLBACK TO recognition_refresh; RELEASE recognition_refresh",
                )?;
                let message = bounded_error(&error);
                transaction.execute(
                    "UPDATE recognition_refresh_queue SET status=CASE WHEN attempts>=5 THEN 'error'
                         ELSE 'pending' END, worker_id=NULL, lease_expires_at=NULL,
                         retry_after=CASE WHEN attempts>=5 THEN NULL ELSE datetime('now',
                             printf('+%d seconds', min(300, 1 << min(attempts, 8)))) END,
                         error=?3, updated_at=datetime('now') WHERE id=?1 AND worker_id=?2",
                    params![id, worker_id, message],
                )?;
            }
        }
        transaction.commit()?;
    }
    Ok(processed)
}

pub fn has_immediately_eligible_refresh_work(root: &Path) -> Result<bool> {
    let connection = writable_connection(root)?;
    ensure_recognition_schema(&connection)?;
    connection
        .query_row(
            "SELECT EXISTS(SELECT 1 FROM recognition_refresh_queue
             WHERE (status='pending' AND (retry_after IS NULL OR retry_after<=datetime('now')))
                OR (status='running' AND lease_expires_at<=datetime('now')))",
            [],
            |row| row.get(0),
        )
        .map_err(Into::into)
}

enum RefreshOperation {
    Person(i64),
    Catalog,
    Face(i64, FaceCalculation),
}

struct FaceCalculation {
    suggestion: Option<RecognitionSuggestion>,
    policy: RecognitionPolicy,
    ann_index_version: i64,
    index_generation: i64,
    prototype_generations: Vec<(i64, i64)>,
}

fn expand_global_rescore(connection: &Connection, model_id: &str) -> Result<usize> {
    let queued: i64 = connection.query_row(
        "SELECT count(*) FROM recognition_refresh_queue WHERE model_id=?1
         AND entity_kind='face' AND status IN ('pending','running')",
        [model_id],
        |row| row.get(0),
    )?;
    let capacity = MAX_QUEUED_FACES.saturating_sub(usize::try_from(queued).unwrap_or(usize::MAX));
    if capacity == 0 {
        enqueue_one(connection, "person", 0, model_id)?;
        return Ok(0);
    }
    let after: i64 = connection.query_row(
        "SELECT after_face_id FROM recognition_rescore_state WHERE model_id=?1",
        [model_id],
        |row| row.get(0),
    )?;
    let mut query = connection.prepare(
        "SELECT face.id FROM faces face
         JOIN face_embeddings embedding ON embedding.face_id=face.id AND embedding.model_id=?1
         WHERE face.id>?2 AND face.ignored=0 AND face.locally_ignored=0 AND face.retired=0
           AND NOT EXISTS (SELECT 1 FROM effective_confirmations confirmation
                           WHERE confirmation.face_id=face.id)
         ORDER BY face.id LIMIT ?3",
    )?;
    let faces = query
        .query_map(
            params![
                model_id,
                after,
                capacity.min(RESCORE_EXPANSION_BATCH) as i64
            ],
            |row| row.get(0),
        )?
        .collect::<rusqlite::Result<Vec<i64>>>()?;
    drop(query);
    for face_id in &faces {
        connection.execute(
            "INSERT OR IGNORE INTO recognition_refresh_queue(entity_kind, entity_id, model_id,
             status, attempts, updated_at) VALUES('face', ?1, ?2, 'pending', 0, datetime('now'))",
            params![face_id, model_id],
        )?;
    }
    if let Some(last) = faces.last() {
        connection.execute(
            "UPDATE recognition_rescore_state SET after_face_id=?2, updated_at=datetime('now')
             WHERE model_id=?1",
            params![model_id, last],
        )?;
        enqueue_one(connection, "person", 0, model_id)?;
    } else {
        connection.execute(
            "DELETE FROM recognition_rescore_state WHERE model_id=?1",
            [model_id],
        )?;
    }
    Ok(faces.len())
}

fn fail_claim(root: &Path, worker_id: &str, id: i64, message: &str) -> Result<()> {
    let connection = writable_connection(root)?;
    connection.execute(
        "UPDATE recognition_refresh_queue SET status='error', worker_id=NULL,
         lease_expires_at=NULL, retry_after=NULL, error=?3, updated_at=datetime('now')
         WHERE id=?1 AND worker_id=?2",
        params![id, worker_id, message],
    )?;
    Ok(())
}

fn enqueue_one(connection: &Connection, kind: &str, entity_id: i64, model_id: &str) -> Result<()> {
    connection.execute(
        "INSERT INTO recognition_refresh_queue(entity_kind, entity_id, model_id, status, updated_at)
         VALUES(?1, ?2, ?3, 'pending', datetime('now'))
         ON CONFLICT(entity_kind, entity_id, model_id) DO UPDATE SET status='pending', attempts=0,
             worker_id=NULL, lease_expires_at=NULL, retry_after=NULL, error=NULL,
             updated_at=excluded.updated_at",
        params![kind, entity_id, model_id],
    )?;
    Ok(())
}

fn release_claims(root: &Path, worker_id: &str, ids: &[i64]) -> Result<()> {
    let connection = writable_connection(root)?;
    for id in ids {
        connection.execute(
            "UPDATE recognition_refresh_queue SET status='pending', worker_id=NULL,
             lease_expires_at=NULL, retry_after=NULL, attempts=max(0, attempts-1)
             WHERE id=?1 AND worker_id=?2",
            params![id, worker_id],
        )?;
    }
    Ok(())
}

fn finish_claims(root: &Path, worker_id: &str, ids: &[i64]) -> Result<()> {
    let connection = writable_connection(root)?;
    for id in ids {
        connection.execute(
            "DELETE FROM recognition_refresh_queue WHERE id=?1 AND worker_id=?2",
            params![id, worker_id],
        )?;
    }
    Ok(())
}

fn bounded_error(error: &anyhow::Error) -> String {
    error.to_string().chars().take(500).collect()
}

#[cfg(test)]
fn rebuild_person_prototype(root: &Path, person_id: i64, model_id: &str) -> Result<bool> {
    let connection = writable_connection(root)?;
    ensure_recognition_schema(&connection)?;
    rebuild_person_prototype_in(&connection, person_id, model_id)
}

#[cfg(test)]
fn refresh_incremental(
    root: &Path,
    person_ids: &[i64],
    face_ids: &[i64],
    model_id: &str,
    policy: RecognitionPolicy,
) -> Result<Vec<RecognitionSuggestion>> {
    let connection = writable_connection(root)?;
    refresh_incremental_in_connection(&connection, person_ids, face_ids, model_id, policy)
}

/// Rebuilds only affected people and then re-scores explicitly affected faces
/// plus faces carrying now-stale suggestions for those people.
#[cfg(test)]
fn refresh_incremental_in_connection(
    connection: &Connection,
    person_ids: &[i64],
    face_ids: &[i64],
    model_id: &str,
    policy: RecognitionPolicy,
) -> Result<Vec<RecognitionSuggestion>> {
    validate_policy(policy)?;
    ensure_recognition_schema(connection)?;
    let mut people = person_ids.to_vec();
    people.sort_unstable();
    people.dedup();
    if people.len() > MAX_PEOPLE {
        bail!("recognition refresh exceeds the bounded person limit");
    }
    for person_id in &people {
        rebuild_person_prototype_in(connection, *person_id, model_id)?;
    }

    let mut faces = face_ids.to_vec();
    if !people.is_empty() {
        let mut query = connection.prepare(
            "SELECT DISTINCT face_id FROM face_assignments
             WHERE provenance='rekesz' AND kind='suggested'
               AND person_id IN (SELECT value FROM json_each(?1))",
        )?;
        let json = serde_json::to_string(&people)?;
        faces.extend(
            query
                .query_map([json], |row| row.get(0))?
                .collect::<rusqlite::Result<Vec<i64>>>()?,
        );
    }
    faces.sort_unstable();
    faces.dedup();
    if faces.len() > MAX_INCREMENTAL_FACES {
        bail!("recognition refresh exceeds the bounded face limit");
    }

    let prototypes = load_prototypes(connection, model_id, None)?;
    let mut suggestions = Vec::new();
    for face_id in faces {
        if let Some(suggestion) =
            score_and_store_face(connection, face_id, model_id, policy, &prototypes)?
        {
            suggestions.push(suggestion);
        }
    }
    Ok(suggestions)
}

/// Scores a newly written embedding without rebuilding prototypes. Callers that
/// changed confirmations must use `refresh_incremental` first.
#[cfg(test)]
fn suggest_face_in_connection(
    connection: &Connection,
    face_id: i64,
    model_id: &str,
    policy: RecognitionPolicy,
) -> Result<Option<RecognitionSuggestion>> {
    validate_policy(policy)?;
    ensure_recognition_schema(connection)?;
    let prototypes = load_prototypes(connection, model_id, None)?;
    score_and_store_face(connection, face_id, model_id, policy, &prototypes)
}

fn ensure_recognition_schema(connection: &Connection) -> Result<()> {
    // Schema creation and compatibility belong to catalog initialization and
    // versioned migrations; maintenance hot paths never perform DDL.
    connection
        .query_row(
            "SELECT generation FROM recognition_index_state LIMIT 1",
            [],
            |row| row.get::<_, i64>(0),
        )
        .optional()?;
    let mut stale_models = connection
        .prepare("SELECT DISTINCT model_id FROM person_prototypes WHERE algorithm_version<>?1")?
        .query_map([PROTOTYPE_ALGORITHM_VERSION], |row| row.get::<_, String>(0))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    stale_models.extend(
        connection
            .prepare(
                "SELECT DISTINCT recognizer_model_id FROM recognition_suggestions
                 WHERE prototype_algorithm_version<>?1 OR policy_version<>?2
                    OR ann_index_version<>?3",
            )?
            .query_map(
                params![
                    PROTOTYPE_ALGORITHM_VERSION,
                    RECOGNITION_POLICY_VERSION,
                    ANN_INDEX_VERSION
                ],
                |row| row.get::<_, String>(0),
            )?
            .collect::<rusqlite::Result<Vec<_>>>()?,
    );
    stale_models.sort();
    stale_models.dedup();
    connection.execute(
        "DELETE FROM face_assignments WHERE provenance='rekesz' AND kind='suggested'
         AND EXISTS (SELECT 1 FROM recognition_suggestions stale
                     WHERE stale.face_id=face_assignments.face_id
                       AND stale.person_id=face_assignments.person_id
                       AND (stale.prototype_algorithm_version<>?1 OR stale.policy_version<>?2
                            OR stale.ann_index_version<>?3))",
        params![
            PROTOTYPE_ALGORITHM_VERSION,
            RECOGNITION_POLICY_VERSION,
            ANN_INDEX_VERSION
        ],
    )?;
    connection.execute(
        "DELETE FROM person_prototypes WHERE algorithm_version<>?1",
        [PROTOTYPE_ALGORITHM_VERSION],
    )?;
    connection.execute(
        "DELETE FROM recognition_suggestions WHERE prototype_algorithm_version<>?1
          OR policy_version<>?2 OR ann_index_version<>?3",
        params![
            PROTOTYPE_ALGORITHM_VERSION,
            RECOGNITION_POLICY_VERSION,
            ANN_INDEX_VERSION
        ],
    )?;
    for model_id in stale_models {
        bump_index_generation(connection, &model_id)?;
        connection.execute(
            "INSERT INTO recognition_refresh_queue(entity_kind, entity_id, model_id, status,
                                                    attempts, updated_at)
             SELECT 'person', person_id, ?1, 'pending', 0, datetime('now')
             FROM effective_confirmations GROUP BY person_id
             ON CONFLICT(entity_kind, entity_id, model_id) DO UPDATE SET status='pending',
             attempts=0, worker_id=NULL, lease_expires_at=NULL, retry_after=NULL, error=NULL,
             updated_at=excluded.updated_at",
            [&model_id],
        )?;
        enqueue_global_rescore(connection, &model_id)?;
    }
    Ok(())
}

fn rebuild_person_prototype_in(
    connection: &Connection,
    person_id: i64,
    model_id: &str,
) -> Result<bool> {
    let mut query = connection.prepare(
        "SELECT embedding.dimensions, embedding.vector
         FROM effective_confirmations confirmation
         JOIN faces face ON face.id=confirmation.face_id
         JOIN face_embeddings embedding ON embedding.face_id=face.id
         WHERE confirmation.person_id=?1 AND embedding.model_id=?2
           AND face.ignored=0 AND face.locally_ignored=0 AND face.retired=0
         ORDER BY abs((face.id * 1103515245) % 2147483647) LIMIT ?3",
    )?;
    let rows = query
        .query_map(
            params![
                person_id,
                model_id,
                MAX_TRAINING_EXEMPLARS_PER_PERSON as i64
            ],
            |row| Ok((row.get::<_, i64>(0)?, row.get::<_, Vec<u8>>(1)?)),
        )?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    if rows.is_empty() {
        let removed = connection.execute(
            "DELETE FROM person_prototypes WHERE person_id=?1 AND model_id=?2",
            params![person_id, model_id],
        )?;
        connection.execute(
            "DELETE FROM recognition_suggestions WHERE person_id=?1 AND recognizer_model_id=?2",
            params![person_id, model_id],
        )?;
        connection.execute(
            "DELETE FROM face_assignments WHERE person_id=?1 AND provenance='rekesz' AND kind='suggested'",
            [person_id],
        )?;
        if removed != 0 {
            bump_index_generation(connection, model_id)?;
        }
        return Ok(false);
    }
    let dimensions = usize::try_from(rows[0].0).context("negative embedding dimensions")?;
    let mut exemplars = Vec::with_capacity(rows.len());
    for (declared_dimensions, bytes) in &rows {
        let embedding = decode_embedding(*declared_dimensions, bytes)?;
        if embedding.len() != dimensions {
            bail!("confirmed embeddings have incompatible dimensions");
        }
        exemplars.push(embedding);
    }
    let centroid = robust_centroid(&exemplars)?;
    let vector = encode_embedding(&centroid);
    connection.execute(
        "INSERT INTO person_prototypes(person_id, model_id, dimensions, vector, exemplar_count,
                                       algorithm_version, generation, updated_at)
         VALUES(?1, ?2, ?3, ?4, ?5, ?6, 1, datetime('now'))
         ON CONFLICT(person_id, model_id) DO UPDATE SET dimensions=excluded.dimensions,
             vector=excluded.vector, exemplar_count=excluded.exemplar_count,
             algorithm_version=excluded.algorithm_version,
             generation=person_prototypes.generation+1, updated_at=excluded.updated_at",
        params![
            person_id,
            model_id,
            dimensions as i64,
            vector,
            rows.len() as i64,
            PROTOTYPE_ALGORITHM_VERSION
        ],
    )?;
    bump_index_generation(connection, model_id)?;
    Ok(true)
}

fn bump_index_generation(connection: &Connection, model_id: &str) -> Result<()> {
    connection.execute(
        "INSERT INTO recognition_index_state(model_id, generation, index_version, updated_at)
         VALUES(?1, 1, ?2, datetime('now'))
         ON CONFLICT(model_id) DO UPDATE SET generation=generation+1,
             index_version=excluded.index_version, updated_at=excluded.updated_at",
        params![model_id, ANN_INDEX_VERSION],
    )?;
    Ok(())
}

fn load_prototypes(
    connection: &Connection,
    model_id: &str,
    checkpoint: Option<(&MaintenanceAdmission, &AtomicBool)>,
) -> Result<Vec<Prototype>> {
    let mut grouped: BTreeMap<i64, Vec<Vec<f32>>> = BTreeMap::new();
    let mut query = connection.prepare(
        "WITH ranked AS (
             SELECT confirmation.person_id, embedding.dimensions, embedding.vector,
                    row_number() OVER (PARTITION BY confirmation.person_id ORDER BY face.id) rank
             FROM effective_confirmations confirmation
             JOIN faces face ON face.id=confirmation.face_id
             JOIN face_embeddings embedding ON embedding.face_id=face.id
             WHERE embedding.model_id=?1 AND face.ignored=0 AND face.locally_ignored=0
               AND face.retired=0)
         SELECT person_id, dimensions, vector FROM ranked
         WHERE rank<=?2 ORDER BY person_id, rank",
    )?;
    let mut total = 0_usize;
    for row in query.query_map(params![model_id, MAX_EXEMPLARS_PER_PERSON as i64], |row| {
        Ok((
            row.get::<_, i64>(0)?,
            row.get::<_, i64>(1)?,
            row.get::<_, Vec<u8>>(2)?,
        ))
    })? {
        let (person_id, dimensions, bytes) = row?;
        total += 1;
        #[cfg(test)]
        if total == 10_000 {
            ACTIVE_TEST_BUILD_HOOK.with(|active| {
                if let Some(hook) = active.borrow().as_ref() {
                    signal_test_build_phase(hook, TestBuildPhase::SqlRows);
                }
            });
        }
        if total.is_multiple_of(256)
            && checkpoint.is_some_and(|(admission, cancelled)| {
                cancelled.load(Ordering::Acquire) || !admission.checkpoint(cancelled)
            })
        {
            bail!("recognition refresh cancelled");
        }
        if total > MAX_EXEMPLARS {
            bail!("catalog exceeds the bounded recognition exemplar limit");
        }
        grouped
            .entry(person_id)
            .or_default()
            .push(decode_embedding(dimensions, &bytes)?);
    }
    if grouped.len() > MAX_PEOPLE {
        bail!("catalog exceeds the bounded recognition person limit");
    }

    let mut prototypes = Vec::with_capacity(grouped.len());
    for (person_id, exemplars) in grouped {
        let Some((dimensions, vector, generation)) = connection
            .query_row(
                "SELECT dimensions, vector, generation FROM person_prototypes
                 WHERE person_id=?1 AND model_id=?2 AND algorithm_version=?3",
                params![person_id, model_id, PROTOTYPE_ALGORITHM_VERSION],
                |row| {
                    Ok((
                        row.get::<_, i64>(0)?,
                        row.get::<_, Vec<u8>>(1)?,
                        row.get(2)?,
                    ))
                },
            )
            .optional()?
        else {
            continue;
        };
        let centroid = decode_embedding(dimensions, &vector)?;
        if exemplars.iter().any(|value| value.len() != centroid.len()) {
            bail!("prototype and exemplars have incompatible dimensions");
        }
        let stored_exemplars = if exemplars.len() == 1 {
            Vec::new()
        } else {
            exemplars
        };
        prototypes.push(Prototype {
            person_id,
            generation,
            centroid,
            exemplars: stored_exemplars,
        });
    }
    Ok(prototypes)
}

fn build_prototype_index(
    prototypes: Vec<Prototype>,
    generation: i64,
    checkpoint: Option<(&MaintenanceAdmission, &AtomicBool)>,
) -> Result<PrototypeIndex> {
    let dimensions = prototypes
        .first()
        .map_or(0, |prototype| prototype.centroid.len());
    if !prototypes.is_empty()
        && (dimensions == 0
            || prototypes.iter().any(|prototype| {
                prototype.centroid.len() != dimensions
                    || prototype
                        .exemplars
                        .iter()
                        .any(|exemplar| exemplar.len() != dimensions)
            }))
    {
        bail!("recognition prototypes do not share one nonzero dimension");
    }
    let metric = build_metric_index(&prototypes, checkpoint)?;
    if prototypes.len() <= EXACT_SEARCH_LIMIT {
        return Ok(PrototypeIndex {
            version: ANN_INDEX_VERSION,
            generation,
            prototypes,
            ann: None,
            metric,
        });
    }
    // Stable person ordering makes data-id membership and rebuild invalidation
    // deterministic even though HNSW's internal level assignment is randomized.
    // SFace centroids are already normalized. Rebuild runs as cancellable idle
    // maintenance; construction and query breadth are deliberately conservative
    // because identity recall matters more than saving a few graph edges.
    let mut hnsw = Hnsw::<f32, DistCosine>::new(8, prototypes.len(), 12, 64, DistCosine {});
    let build_pool = if let Some(pool) = ANN_BUILD_POOL.get() {
        pool
    } else {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .thread_name(|index| format!("recognition-ann-{index}"))
            .build()
            .context("failed to create bounded recognition ANN pool")?;
        let _ = ANN_BUILD_POOL.set(pool);
        ANN_BUILD_POOL
            .get()
            .context("recognition ANN pool was not installed")?
    };
    for (chunk_index, chunk) in prototypes.chunks(128).enumerate() {
        #[cfg(test)]
        if chunk_index == 64 {
            ACTIVE_TEST_BUILD_HOOK.with(|active| {
                if let Some(hook) = active.borrow().as_ref() {
                    signal_test_build_phase(hook, TestBuildPhase::HnswInsertions);
                }
            });
        }
        if checkpoint.is_some_and(|(admission, cancelled)| {
            cancelled.load(Ordering::Acquire) || !admission.checkpoint(cancelled)
        }) {
            bail!("recognition index rebuild cancelled");
        }
        let offset = chunk_index * 128;
        let entries = chunk
            .iter()
            .enumerate()
            .map(|(index, prototype)| (&prototype.centroid, offset + index))
            .collect::<Vec<_>>();
        build_pool.install(|| hnsw.parallel_insert(&entries));
    }
    hnsw.set_searching_mode(true);
    Ok(PrototypeIndex {
        version: ANN_INDEX_VERSION,
        generation,
        prototypes,
        ann: Some(hnsw),
        metric,
    })
}

fn metric_distance(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| {
            let difference = left - right;
            difference * difference
        })
        .sum::<f32>()
        .sqrt()
}

fn build_metric_index(
    prototypes: &[Prototype],
    checkpoint: Option<(&MaintenanceAdmission, &AtomicBool)>,
) -> Result<MetricIndex> {
    fn build(
        prototypes: &[Prototype],
        ids: &mut [usize],
        nodes: &mut Vec<MetricNode>,
        visited: &mut usize,
        checkpoint: Option<(&MaintenanceAdmission, &AtomicBool)>,
    ) -> Result<Option<usize>> {
        let Some((vantage, rest)) = ids.split_last_mut() else {
            return Ok(None);
        };
        let vantage = *vantage;
        *visited += 1;
        #[cfg(test)]
        if *visited == 10_000 {
            ACTIVE_TEST_BUILD_HOOK.with(|active| {
                if let Some(hook) = active.borrow().as_ref() {
                    signal_test_build_phase(hook, TestBuildPhase::MetricNodes);
                }
            });
        }
        if visited.is_multiple_of(128)
            && checkpoint.is_some_and(|(admission, cancelled)| {
                cancelled.load(Ordering::Acquire) || !admission.checkpoint(cancelled)
            })
        {
            bail!("recognition metric index rebuild cancelled");
        }
        let mut ranked = Vec::with_capacity(rest.len());
        for (position, id) in rest.iter().enumerate() {
            if position.is_multiple_of(128)
                && checkpoint.is_some_and(|(admission, cancelled)| {
                    cancelled.load(Ordering::Acquire) || !admission.checkpoint(cancelled)
                })
            {
                bail!("recognition metric index rebuild cancelled");
            }
            ranked.push((
                *id,
                metric_distance(&prototypes[vantage].centroid, &prototypes[*id].centroid),
            ));
        }
        ranked.sort_by(|(left, left_distance), (right, right_distance)| {
            left_distance.total_cmp(right_distance).then_with(|| {
                prototypes[*left]
                    .person_id
                    .cmp(&prototypes[*right].person_id)
            })
        });
        for (slot, (id, _)) in rest.iter_mut().zip(&ranked) {
            *slot = *id;
        }
        let middle = rest.len().div_ceil(2);
        let threshold = rest
            .get(middle.saturating_sub(1))
            .map_or(0.0, |_| ranked[middle.saturating_sub(1)].1);
        let (inner_ids, outer_ids) = rest.split_at_mut(middle);
        let node_id = nodes.len();
        nodes.push(MetricNode {
            prototype_id: vantage,
            threshold,
            inner: None,
            outer: None,
        });
        let inner = build(prototypes, inner_ids, nodes, visited, checkpoint)?;
        let outer = build(prototypes, outer_ids, nodes, visited, checkpoint)?;
        nodes[node_id].inner = inner;
        nodes[node_id].outer = outer;
        Ok(Some(node_id))
    }
    let mut ids = (0..prototypes.len()).collect::<Vec<_>>();
    let mut nodes = Vec::with_capacity(ids.len());
    let mut visited = 0;
    let root = build(prototypes, &mut ids, &mut nodes, &mut visited, checkpoint)?;
    Ok(MetricIndex { nodes, root })
}

fn cached_prototype_index(
    root: &Path,
    connection: &Connection,
    model_id: &str,
    checkpoint: Option<(&MaintenanceAdmission, &AtomicBool)>,
) -> Result<Arc<PrototypeIndex>> {
    let generation = connection
        .query_row(
            "SELECT generation FROM recognition_index_state
             WHERE model_id=?1 AND index_version=?2",
            params![model_id, ANN_INDEX_VERSION],
            |row| row.get::<_, i64>(0),
        )
        .optional()?
        .unwrap_or(0);
    let key = (root.to_path_buf(), model_id.to_owned(), generation);
    let cache = PROTOTYPE_INDEX_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let (build, builder) = {
        let mut cache = cache
            .lock()
            .map_err(|_| anyhow::anyhow!("prototype index cache is unavailable"))?;
        match cache.entry(key.clone()) {
            Entry::Occupied(entry) => (entry.get().clone(), false),
            Entry::Vacant(entry) => {
                let build = Arc::new(IndexBuild {
                    result: Mutex::new(None),
                    wake: Condvar::new(),
                });
                entry.insert(build.clone());
                (build, true)
            }
        }
    };
    if builder {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            acquire_model_build_permit(root, model_id, checkpoint).and_then(|_permit| {
                #[cfg(test)]
                let hook = TEST_BUILD_HOOKS
                    .get_or_init(|| Mutex::new(HashMap::new()))
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .get(root)
                    .cloned();
                #[cfg(test)]
                ACTIVE_TEST_BUILD_HOOK.with(|active| *active.borrow_mut() = hook);
                let result = load_prototypes_interruptibly(connection, model_id, checkpoint)
                    .and_then(|prototypes| {
                        build_prototype_index(prototypes, generation, checkpoint)
                    });
                #[cfg(test)]
                ACTIVE_TEST_BUILD_HOOK.with(|active| *active.borrow_mut() = None);
                result
            })
        }))
        .map_err(|_| Arc::<str>::from("recognition index builder panicked"))
        .and_then(|result| {
            result
                .map(Arc::new)
                .map_err(|error| Arc::<str>::from(format!("{error:#}")))
        });
        *build
            .result
            .lock()
            .map_err(|_| anyhow::anyhow!("prototype index build state is unavailable"))? =
            Some(result.clone());
        build.wake.notify_all();
        finalize_index_build_cache(
            cache,
            root,
            model_id,
            generation,
            &key,
            &build,
            result.is_ok(),
        );
        return result.map_err(|error| anyhow::anyhow!(error.to_string()));
    }
    let mut state = build
        .result
        .lock()
        .map_err(|_| anyhow::anyhow!("prototype index build state is unavailable"))?;
    loop {
        if let Some(result) = state.as_ref() {
            return result
                .clone()
                .map_err(|error| anyhow::anyhow!(error.to_string()));
        }
        if checkpoint.is_some_and(|(admission, cancelled)| {
            cancelled.load(Ordering::Acquire) || !admission.checkpoint(cancelled)
        }) {
            bail!("recognition index wait cancelled");
        }
        (state, _) = build
            .wake
            .wait_timeout(state, std::time::Duration::from_millis(20))
            .map_err(|_| anyhow::anyhow!("prototype index build state is unavailable"))?;
    }
}

fn load_prototypes_interruptibly(
    connection: &Connection,
    model_id: &str,
    checkpoint: Option<(&MaintenanceAdmission, &AtomicBool)>,
) -> Result<Vec<Prototype>> {
    let Some((admission, cancelled)) = checkpoint else {
        return load_prototypes(connection, model_id, None);
    };
    let finished = AtomicBool::new(false);
    std::thread::scope(|scope| {
        let interrupt = connection.get_interrupt_handle();
        let worker_finished = &finished;
        scope.spawn(move || {
            while !worker_finished.load(Ordering::Acquire) && !cancelled.load(Ordering::Acquire) {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            if cancelled.load(Ordering::Acquire) {
                interrupt.interrupt();
            }
        });
        let result = load_prototypes(connection, model_id, Some((admission, cancelled)));
        finished.store(true, Ordering::Release);
        if cancelled.load(Ordering::Acquire) {
            bail!("recognition refresh cancelled");
        }
        result
    })
}

fn finalize_index_build_cache(
    cache: &Mutex<IndexCache>,
    root: &Path,
    model_id: &str,
    generation: i64,
    key: &(PathBuf, String, i64),
    build: &Arc<IndexBuild>,
    succeeded: bool,
) {
    let mut cache = cache
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if !succeeded {
        if cache
            .get(key)
            .is_some_and(|cached| Arc::ptr_eq(cached, build))
        {
            cache.remove(key);
        }
        return;
    }
    cache.retain(|(path, cached_model, cached_generation), cached| {
        path != root
            || cached_model != model_id
            || *cached_generation == generation
            || cached
                .result
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .is_none()
    });
}

fn acquire_model_build_permit(
    root: &Path,
    model_id: &str,
    checkpoint: Option<(&MaintenanceAdmission, &AtomicBool)>,
) -> Result<BuildPermit> {
    let gates = MODEL_BUILD_GATES.get_or_init(|| Mutex::new(HashMap::new()));
    let gate = {
        let mut gates = gates
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        gates
            .entry((root.to_path_buf(), model_id.to_owned()))
            .or_insert_with(|| {
                Arc::new(BuildGate {
                    active: Mutex::new(false),
                    wake: Condvar::new(),
                })
            })
            .clone()
    };
    let mut active = gate
        .active
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    loop {
        if !*active {
            *active = true;
            drop(active);
            return Ok(BuildPermit(gate));
        }
        if checkpoint.is_some_and(|(admission, cancelled)| {
            cancelled.load(Ordering::Acquire) || !admission.checkpoint(cancelled)
        }) {
            bail!("recognition build contention wait cancelled");
        }
        (active, _) = gate
            .wake
            .wait_timeout(active, std::time::Duration::from_millis(20))
            .unwrap_or_else(std::sync::PoisonError::into_inner);
    }
}

fn prototype_candidate_ids(
    index: &PrototypeIndex,
    candidate: &[f32],
    requested: usize,
) -> Vec<usize> {
    let Some(ann) = index.ann.as_ref() else {
        return (0..index.prototypes.len()).collect();
    };
    ann.search(
        candidate,
        requested.min(index.prototypes.len()),
        ANN_EF_SEARCH.max(requested),
    )
    .into_iter()
    .map(|neighbor| neighbor.d_id)
    .collect()
}

fn certified_best_two(
    index: &PrototypeIndex,
    candidate: &[f32],
    rejected: &HashSet<i64>,
    seeds: &[usize],
    admission: &MaintenanceAdmission,
    cancelled: &AtomicBool,
) -> Result<CertifiedBestTwo> {
    fn better(index: &PrototypeIndex, left: (usize, f32), right: (usize, f32)) -> bool {
        left.1 < right.1
            || (left.1 == right.1
                && index.prototypes[left.0].person_id < index.prototypes[right.0].person_id)
    }
    fn consider(
        index: &PrototypeIndex,
        scored: (usize, f32),
        best: &mut Option<(usize, f32)>,
        second: &mut Option<(usize, f32)>,
    ) {
        if best.is_none_or(|current| better(index, scored, current)) {
            if best.is_none_or(|current| current.0 != scored.0) {
                *second = *best;
            }
            *best = Some(scored);
        } else if best.is_some_and(|current| current.0 != scored.0)
            && second.is_none_or(|current| better(index, scored, current))
        {
            *second = Some(scored);
        }
    }
    let mut best = None;
    let mut second = None;
    let seed_ids = seeds.iter().copied().collect::<HashSet<_>>();
    for id in seeds {
        let prototype = &index.prototypes[*id];
        if !rejected.contains(&prototype.person_id) {
            consider(
                index,
                (*id, metric_distance(candidate, &prototype.centroid)),
                &mut best,
                &mut second,
            );
        }
    }
    let mut dimension_work = 0_usize;
    for (id, prototype) in index.prototypes.iter().enumerate() {
        if id.is_multiple_of(128)
            && (cancelled.load(Ordering::Acquire) || !admission.checkpoint(cancelled))
        {
            bail!("recognition metric search cancelled");
        }
        if seed_ids.contains(&id) || rejected.contains(&prototype.person_id) {
            continue;
        }
        let limit_squared = second.map_or(f32::INFINITY, |(_, distance)| distance * distance);
        let mut squared = 0.0_f32;
        let mut pruned = false;
        for (left, right) in candidate.iter().zip(&prototype.centroid) {
            let difference = left - right;
            squared += difference * difference;
            dimension_work += 1;
            if squared > limit_squared {
                pruned = true;
                break;
            }
        }
        if !pruned {
            consider(index, (id, squared.sqrt()), &mut best, &mut second);
        }
    }
    // Construction remains versioned and exercised even though certification uses
    // the cheaper exact coordinate lower bound rather than VP traversal at query time.
    let _metric_generation_root = index.metric.root.map(|root| {
        let node = index.metric.nodes[root];
        (node.prototype_id, node.threshold)
    });
    Ok((best, second, dimension_work))
}

fn calculate_face_suggestion(
    connection: &Connection,
    face_id: i64,
    model_id: &str,
    policy: RecognitionPolicy,
    prototype_index: &PrototypeIndex,
    admission: &MaintenanceAdmission,
    cancelled: &AtomicBool,
) -> Result<Option<RecognitionSuggestion>> {
    let row = connection
        .query_row(
            "SELECT embedding.dimensions, embedding.vector FROM face_embeddings embedding
         JOIN faces face ON face.id=embedding.face_id
         WHERE embedding.face_id=?1 AND embedding.model_id=?2 AND face.ignored=0
           AND face.locally_ignored=0 AND face.retired=0
           AND NOT EXISTS (SELECT 1 FROM effective_confirmations WHERE face_id=?1)",
            params![face_id, model_id],
            |row| Ok((row.get::<_, i64>(0)?, row.get::<_, Vec<u8>>(1)?)),
        )
        .optional()?;
    let Some((dimensions, bytes)) = row else {
        return Ok(None);
    };
    let candidate = decode_embedding(dimensions, &bytes)?;
    let mut query = connection.prepare(
        "SELECT person_id FROM face_assignments WHERE face_id=?1 AND kind='rejected'
         AND provenance='rekesz'",
    )?;
    let rejected = query
        .query_map([face_id], |row| row.get(0))?
        .collect::<rusqlite::Result<HashSet<i64>>>()?;
    let shortlist_size = ANN_CANDIDATES
        .saturating_add(rejected.len())
        .min(MAX_ANN_CANDIDATES);
    let ann_shortlist = prototype_candidate_ids(prototype_index, &candidate, shortlist_size);
    // HNSW seeds a tight search radius. The metric tree then certifies the exact
    // best and runner-up with triangle-inequality pruning, so omitted HNSW nodes
    // can never turn an ambiguous face into a false acceptance.
    let (best_match, second_match, _) = certified_best_two(
        prototype_index,
        &candidate,
        &rejected,
        &ann_shortlist,
        admission,
        cancelled,
    )?;
    let Some((best_id, best_distance)) = best_match else {
        return Ok(None);
    };
    let centroid_score = 1.0 - best_distance * best_distance * 0.5;
    let prototype = &prototype_index.prototypes[best_id];
    let nearest_score = if prototype.exemplars.is_empty() {
        dot(&candidate, &prototype.centroid)
    } else {
        prototype
            .exemplars
            .iter()
            .map(|value| dot(&candidate, value))
            .max_by(f32::total_cmp)
            .unwrap_or(f32::NEG_INFINITY)
    };
    let second_score =
        second_match.map(|(_, second_distance)| 1.0 - second_distance * second_distance * 0.5);
    let Some(margin) = accepted_margin(policy, centroid_score, nearest_score, second_score) else {
        return Ok(None);
    };
    let best = Candidate {
        person_id: prototype.person_id,
        generation: prototype.generation,
        centroid_score,
        nearest_score,
    };
    let lightroom: bool = connection.query_row(
        "SELECT EXISTS(SELECT 1 FROM face_assignments WHERE face_id=?1 AND person_id=?2
         AND kind='suggested' AND provenance='lightroom')",
        params![face_id, best.person_id],
        |row| row.get(0),
    )?;
    if lightroom {
        return Ok(None);
    }
    Ok(Some(RecognitionSuggestion {
        face_id,
        person_id: best.person_id,
        score: best.centroid_score,
        nearest_exemplar_score: best.nearest_score,
        second_best_margin: margin,
        recognizer_model_id: model_id.to_owned(),
        prototype_generation: best.generation,
        policy_version: RECOGNITION_POLICY_VERSION,
        ann_index_version: prototype_index.version,
        scored_index_generation: prototype_index.generation,
        policy,
        prototype_generations: prototype_index
            .prototypes
            .iter()
            .map(|prototype| (prototype.person_id, prototype.generation))
            .collect(),
    }))
}

fn store_calculated_suggestion(
    connection: &Connection,
    face_id: i64,
    model_id: &str,
    calculation: &FaceCalculation,
    queue_id: i64,
    worker_id: &str,
) -> Result<Option<RecognitionSuggestion>> {
    let owns_lease: bool = connection.query_row(
        "SELECT EXISTS(SELECT 1 FROM recognition_refresh_queue WHERE id=?1 AND worker_id=?2
         AND status='running' AND lease_expires_at>datetime('now'))",
        params![queue_id, worker_id],
        |row| row.get(0),
    )?;
    if !owns_lease {
        return Ok(None);
    }
    let policy = calibrated_policy(connection, model_id)?;
    let generations_json = serde_json::to_string(&calculation.prototype_generations)?;
    let matching_generations: i64 = connection.query_row(
        "SELECT count(*) FROM json_each(?1) expected
         JOIN person_prototypes prototype
           ON prototype.person_id=json_extract(expected.value, '$[0]')
          AND prototype.generation=json_extract(expected.value, '$[1]')
          AND prototype.model_id=?2 AND prototype.algorithm_version=?3",
        params![generations_json, model_id, PROTOTYPE_ALGORITHM_VERSION],
        |row| row.get(0),
    )?;
    // Matching the entries that existed when scoring is insufficient: a person
    // can acquire its first compatible prototype while the calculation is in
    // flight. Compare the complete current membership as part of this same
    // short write transaction so additions and removals both invalidate it.
    let current_generation_count: i64 = connection.query_row(
        "SELECT count(*) FROM person_prototypes
         WHERE model_id=?1 AND algorithm_version=?2",
        params![model_id, PROTOTYPE_ALGORITHM_VERSION],
        |row| row.get(0),
    )?;
    let current_index_generation: i64 = connection
        .query_row(
            "SELECT generation FROM recognition_index_state WHERE model_id=?1
             AND index_version=?2",
            params![model_id, ANN_INDEX_VERSION],
            |row| row.get(0),
        )
        .optional()?
        .unwrap_or(0);
    let snapshot_current = policy == Some(calculation.policy)
        && calculation.ann_index_version == ANN_INDEX_VERSION
        && calculation.index_generation == current_index_generation
        && matching_generations == calculation.prototype_generations.len() as i64
        && current_generation_count == calculation.prototype_generations.len() as i64;
    if !snapshot_current {
        enqueue_one(connection, "face", face_id, model_id)?;
        return Ok(None);
    }
    if let Some(suggestion) = calculation.suggestion.as_ref() {
        let face_still_eligible: bool = connection.query_row(
            "SELECT EXISTS(SELECT 1 FROM faces face WHERE face.id=?1 AND face.ignored=0
             AND face.locally_ignored=0 AND face.retired=0
             AND NOT EXISTS (SELECT 1 FROM effective_confirmations WHERE face_id=face.id)
             AND NOT EXISTS (SELECT 1 FROM face_assignments WHERE face_id=face.id
               AND person_id=?2 AND kind='rejected' AND provenance='rekesz')
             AND NOT EXISTS (SELECT 1 FROM face_assignments WHERE face_id=face.id
               AND person_id=?2 AND kind='suggested' AND provenance='lightroom'))",
            params![face_id, suggestion.person_id],
            |row| row.get(0),
        )?;
        if !face_still_eligible
            || suggestion.policy_version != RECOGNITION_POLICY_VERSION
            || suggestion.ann_index_version != ANN_INDEX_VERSION
        {
            enqueue_one(connection, "face", face_id, model_id)?;
            return Ok(None);
        }
    }
    connection.execute(
        "DELETE FROM face_assignments WHERE face_id=?1 AND provenance='rekesz'
                        AND kind='suggested'",
        [face_id],
    )?;
    connection.execute(
        "DELETE FROM recognition_suggestions WHERE face_id=?1
                        AND recognizer_model_id=?2",
        params![face_id, model_id],
    )?;
    let Some(suggestion) = calculation.suggestion.as_ref() else {
        return Ok(None);
    };
    connection.execute(
        "INSERT INTO face_assignments(face_id, person_id, kind, provenance, confidence, updated_at)
         VALUES(?1, ?2, 'suggested', 'rekesz', ?3, datetime('now'))",
        params![face_id, suggestion.person_id, suggestion.score],
    )?;
    connection.execute(
        "INSERT INTO recognition_suggestions(face_id, person_id, recognizer_model_id,
         prototype_generation, prototype_algorithm_version, ann_index_version,
         scored_index_generation, policy_version, score,
         nearest_exemplar_score, second_best_margin, created_at)
         VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, datetime('now'))",
        params![
            face_id,
            suggestion.person_id,
            model_id,
            suggestion.prototype_generation,
            PROTOTYPE_ALGORITHM_VERSION,
            suggestion.ann_index_version,
            suggestion.scored_index_generation,
            RECOGNITION_POLICY_VERSION,
            suggestion.score,
            suggestion.nearest_exemplar_score,
            suggestion.second_best_margin
        ],
    )?;
    Ok(Some(suggestion.clone()))
}

#[cfg(test)]
fn score_and_store_face(
    connection: &Connection,
    face_id: i64,
    model_id: &str,
    policy: RecognitionPolicy,
    prototypes: &[Prototype],
) -> Result<Option<RecognitionSuggestion>> {
    connection.execute(
        "DELETE FROM recognition_suggestions WHERE face_id=?1 AND recognizer_model_id=?2",
        params![face_id, model_id],
    )?;
    connection.execute(
        "DELETE FROM face_assignments WHERE face_id=?1 AND provenance='rekesz' AND kind='suggested'",
        [face_id],
    )?;
    let row = connection
        .query_row(
            "SELECT embedding.dimensions, embedding.vector
             FROM face_embeddings embedding JOIN faces face ON face.id=embedding.face_id
             WHERE embedding.face_id=?1 AND embedding.model_id=?2 AND face.ignored=0
               AND face.locally_ignored=0 AND face.retired=0
               AND NOT EXISTS (SELECT 1 FROM effective_confirmations WHERE face_id=?1)",
            params![face_id, model_id],
            |row| Ok((row.get::<_, i64>(0)?, row.get::<_, Vec<u8>>(1)?)),
        )
        .optional()?;
    let Some((dimensions, bytes)) = row else {
        return Ok(None);
    };
    let candidate = decode_embedding(dimensions, &bytes)?;
    let mut scores = Vec::new();
    for prototype in prototypes {
        if prototype.centroid.len() != candidate.len()
            || locally_rejected(connection, face_id, prototype.person_id)?
        {
            continue;
        }
        let centroid_score = dot(&candidate, &prototype.centroid);
        let nearest_score = if prototype.exemplars.is_empty() {
            centroid_score
        } else {
            prototype
                .exemplars
                .iter()
                .map(|exemplar| dot(&candidate, exemplar))
                .max_by(f32::total_cmp)
                .unwrap_or(f32::NEG_INFINITY)
        };
        scores.push(Candidate {
            person_id: prototype.person_id,
            generation: prototype.generation,
            centroid_score,
            nearest_score,
        });
    }
    scores.sort_by(|left, right| {
        right
            .centroid_score
            .total_cmp(&left.centroid_score)
            .then_with(|| left.person_id.cmp(&right.person_id))
    });
    let Some(best) = scores.first().copied() else {
        return Ok(None);
    };
    let second_score = scores.get(1).map(|second| second.centroid_score);
    let Some(margin) = accepted_margin(
        policy,
        best.centroid_score,
        best.nearest_score,
        second_score,
    ) else {
        return Ok(None);
    };

    // A Lightroom guess remains visible and authoritative for the same pair.
    // We still retain no duplicate Rekesz row beneath it.
    let lightroom_exists: bool = connection.query_row(
        "SELECT EXISTS(SELECT 1 FROM face_assignments WHERE face_id=?1 AND person_id=?2
                       AND kind='suggested' AND provenance='lightroom')",
        params![face_id, best.person_id],
        |row| row.get(0),
    )?;
    if lightroom_exists {
        return Ok(None);
    }
    connection.execute(
        "INSERT INTO face_assignments(face_id, person_id, kind, provenance, confidence, updated_at)
         VALUES(?1, ?2, 'suggested', 'rekesz', ?3, datetime('now'))",
        params![face_id, best.person_id, best.centroid_score],
    )?;
    connection.execute(
        "INSERT INTO recognition_suggestions(face_id, person_id, recognizer_model_id,
             prototype_generation, prototype_algorithm_version, ann_index_version,
             scored_index_generation, policy_version, score,
             nearest_exemplar_score, second_best_margin, created_at)
         VALUES(?1, ?2, ?3, ?4, ?5, ?6, 0, ?7, ?8, ?9, ?10, datetime('now'))",
        params![
            face_id,
            best.person_id,
            model_id,
            best.generation,
            PROTOTYPE_ALGORITHM_VERSION,
            ANN_INDEX_VERSION,
            RECOGNITION_POLICY_VERSION,
            best.centroid_score,
            best.nearest_score,
            margin
        ],
    )?;
    Ok(Some(RecognitionSuggestion {
        face_id,
        person_id: best.person_id,
        score: best.centroid_score,
        nearest_exemplar_score: best.nearest_score,
        second_best_margin: margin,
        recognizer_model_id: model_id.to_owned(),
        prototype_generation: best.generation,
        policy_version: RECOGNITION_POLICY_VERSION,
        ann_index_version: ANN_INDEX_VERSION,
        scored_index_generation: 0,
        policy,
        prototype_generations: prototypes
            .iter()
            .map(|prototype| (prototype.person_id, prototype.generation))
            .collect(),
    }))
}

#[cfg(test)]
fn locally_rejected(connection: &Connection, face_id: i64, person_id: i64) -> Result<bool> {
    connection
        .query_row(
            "SELECT EXISTS(SELECT 1 FROM face_assignments WHERE face_id=?1 AND person_id=?2
                           AND kind='rejected' AND provenance='rekesz')",
            params![face_id, person_id],
            |row| row.get(0),
        )
        .map_err(Into::into)
}

fn decode_embedding(dimensions: i64, bytes: &[u8]) -> Result<Vec<f32>> {
    let dimensions = usize::try_from(dimensions).context("negative embedding dimensions")?;
    if dimensions == 0 || bytes.len() != dimensions.saturating_mul(4) {
        bail!("embedding dimensions do not match its byte vector");
    }
    let mut values = Vec::with_capacity(dimensions);
    for chunk in bytes.as_chunks::<4>().0 {
        let value = f32::from_le_bytes(*chunk);
        if !value.is_finite() {
            bail!("embedding contains a non-finite value");
        }
        values.push(value);
    }
    normalize(&mut values)?;
    Ok(values)
}

fn encode_embedding(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn validate_policy(policy: RecognitionPolicy) -> Result<()> {
    if !policy.absolute_threshold.is_finite()
        || !policy.nearest_exemplar_threshold.is_finite()
        || !policy.second_best_margin.is_finite()
        || !(-1.0..=1.0).contains(&policy.absolute_threshold)
        || !(-1.0..=1.0).contains(&policy.nearest_exemplar_threshold)
        || !(0.0..=2.0).contains(&policy.second_best_margin)
    {
        bail!("invalid recognition policy");
    }
    Ok(())
}

#[cfg(test)]
#[path = "recognition_tests.rs"]
mod tests;
