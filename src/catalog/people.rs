use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use anyhow::{Context, Result, bail};
use rusqlite::{
    Connection, OptionalExtension, Transaction, TransactionBehavior, functions::FunctionFlags,
    params,
};
use serde::{Deserialize, Serialize};

use super::{
    ImportReport, decode_relative_path, encode_relative_path, resolve_relative_source, scope_key,
};

/// A deterministic pose score for the five SFace landmarks (eyes, nose, mouth
/// corners). It deliberately measures only geometry: image sharpness is not
/// persisted today, so it must never be guessed from confidence or face area.
fn frontal_landmark_quality(bytes: Option<&[u8]>, bounds: [f64; 4]) -> f64 {
    let Some(bytes) = bytes.filter(|bytes| bytes.len() == 5 * 2 * size_of::<f32>()) else {
        return 0.0;
    };
    let mut points = [[0.0_f64; 2]; 5];
    for (index, coordinate) in points.iter_mut().flatten().enumerate() {
        let start = index * size_of::<f32>();
        let value = f32::from_le_bytes(bytes[start..start + 4].try_into().unwrap());
        if !value.is_finite() {
            return 0.0;
        }
        *coordinate = f64::from(value);
    }
    let width = bounds[2] - bounds[0];
    let height = bounds[3] - bounds[1];
    if width <= 0.0 || height <= 0.0 {
        return 0.0;
    }
    let normalize = |point: [f64; 2]| {
        [
            (point[0] - bounds[0]) / width,
            (point[1] - bounds[1]) / height,
        ]
    };
    let points = points.map(normalize);
    if points
        .iter()
        .flatten()
        .any(|value| !(-0.25..=1.25).contains(value))
    {
        return 0.0;
    }
    let eye_tilt = (points[0][1] - points[1][1]).abs();
    let mouth_tilt = (points[3][1] - points[4][1]).abs();
    let eye_mid = (points[0][0] + points[1][0]) * 0.5;
    let mouth_mid = (points[3][0] + points[4][0]) * 0.5;
    let symmetry = (points[2][0] - eye_mid).abs() + (points[2][0] - mouth_mid).abs();
    let eye_span = (points[1][0] - points[0][0]).abs();
    let mouth_span = (points[4][0] - points[3][0]).abs();
    let scale_penalty = (0.32 - eye_span).abs() + (0.22 - mouth_span).abs();
    (1.0 - 2.0 * eye_tilt - 2.0 * mouth_tilt - symmetry - scale_penalty).clamp(0.0, 1.0)
}

fn install_representative_quality_function(connection: &Connection) -> Result<()> {
    connection.create_scalar_function(
        "rekesz_frontal_quality",
        5,
        FunctionFlags::SQLITE_DETERMINISTIC | FunctionFlags::SQLITE_INNOCUOUS,
        |context| {
            Ok(frontal_landmark_quality(
                context.get_raw(0).as_blob_or_null()?,
                [
                    context.get(1)?,
                    context.get(2)?,
                    context.get(3)?,
                    context.get(4)?,
                ],
            ))
        },
    )?;
    Ok(())
}

#[cfg(test)]
#[derive(Clone)]
pub(crate) struct PeopleSummaryProgressHook {
    pub root: PathBuf,
    pub threshold_steps: u64,
    pub steps: Arc<std::sync::atomic::AtomicU64>,
    pub barrier: Arc<(std::sync::Mutex<(bool, bool)>, std::sync::Condvar)>,
}

#[cfg(test)]
pub(crate) static PEOPLE_SUMMARY_PROGRESS_HOOK: std::sync::Mutex<
    Option<PeopleSummaryProgressHook>,
> = std::sync::Mutex::new(None);
#[cfg(test)]
pub(crate) static PEOPLE_SUMMARY_PROGRESS_TEST_LOCK: std::sync::Mutex<()> =
    std::sync::Mutex::new(());

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FaceOrigin {
    LightroomDetected,
    LightroomManual,
    RekeszDetected,
    RekeszManual,
    Unknown,
}

impl FaceOrigin {
    fn as_str(self) -> &'static str {
        match self {
            Self::LightroomDetected => "lightroom-detected",
            Self::LightroomManual => "lightroom-manual",
            Self::RekeszDetected => "rekesz-detected",
            Self::RekeszManual => "rekesz-manual",
            Self::Unknown => "unknown",
        }
    }

    fn parse(value: &str) -> Self {
        match value {
            "lightroom-detected" => Self::LightroomDetected,
            "lightroom-manual" => Self::LightroomManual,
            "rekesz-detected" => Self::RekeszDetected,
            "rekesz-manual" => Self::RekeszManual,
            _ => Self::Unknown,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FaceAssignmentKind {
    Unassigned,
    Confirmed,
    Suggested,
    Rejected,
    Ignored,
}

impl FaceAssignmentKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Unassigned => "unassigned",
            Self::Confirmed => "confirmed",
            Self::Suggested => "suggested",
            Self::Rejected => "rejected",
            Self::Ignored => "ignored",
        }
    }

    fn parse(value: &str) -> Result<Self> {
        match value {
            "unassigned" => Ok(Self::Unassigned),
            "confirmed" => Ok(Self::Confirmed),
            "suggested" => Ok(Self::Suggested),
            "rejected" => Ok(Self::Rejected),
            "ignored" => Ok(Self::Ignored),
            _ => bail!("invalid face assignment kind {value}"),
        }
    }
}

#[cfg(test)]
#[derive(Clone, Debug, PartialEq)]
pub struct FaceRecord {
    pub id: i64,
    pub path: PathBuf,
    /// Normalized rectangle in the unoriented source-image coordinate space.
    pub bounds: [f32; 4],
    pub origin: FaceOrigin,
    pub ignored: bool,
    pub cluster: Option<String>,
}

#[cfg(test)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PersonRecord {
    pub id: i64,
    pub name: String,
    pub confirmed_faces: i64,
    pub suggested_faces: i64,
    pub representative_face_id: Option<i64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PersonSummary {
    pub id: i64,
    pub name: String,
    pub confirmed_faces: i64,
    pub suggested_faces: i64,
    pub representative_face_id: Option<i64>,
    pub representative: Option<RepresentativeFace>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RepresentativeFace {
    pub face_id: i64,
    pub path: PathBuf,
    pub bounds: [f32; 4],
}

#[derive(Clone, Debug, PartialEq)]
pub struct PersonPage {
    pub items: Vec<PersonSummary>,
    pub total: u64,
    pub offset: u64,
    pub unnamed_faces: u64,
    pub review_faces: u64,
    pub analysis_pending: u64,
    pub analysis_errors: u64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum PeopleSort {
    #[default]
    Name,
    Confirmed,
    Review,
    RecentlyChanged,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FaceReviewItem {
    pub face_id: i64,
    pub person_id: i64,
    pub person_name: Option<String>,
    pub path: PathBuf,
    pub bounds: [f32; 4],
    pub origin: FaceOrigin,
    pub kind: FaceAssignmentKind,
    pub provenance: String,
    pub confidence: Option<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FaceReviewPage {
    pub items: Vec<FaceReviewItem>,
    pub total: u64,
    pub offset: u64,
}

pub const MAX_DECISIONS_PER_BATCH: usize = 500;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecisionAction {
    Confirm,
    Reject,
    Ignore,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecisionCommand {
    pub face_id: i64,
    pub person_id: Option<i64>,
    pub action: DecisionAction,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecisionBatch {
    /// Stable caller-generated key used to make retries idempotent.
    pub id: String,
    pub commands: Vec<DecisionCommand>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecisionBatchResult {
    pub batch_id: String,
    pub applied: bool,
    pub faces: Vec<i64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct LocalFaceState {
    ignored: bool,
    assignments: Vec<(i64, String, Option<f32>)>,
    #[serde(default)]
    representatives: Vec<(i64, Option<i64>, bool)>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MergePeoplePreview {
    pub source_person_id: i64,
    pub target_person_id: i64,
    pub source_confirmed_faces: u64,
    pub target_confirmed_faces: u64,
    pub confirmation_deduplications: u64,
    pub assignment_deduplications: u64,
    pub source_representative_face_id: Option<i64>,
    pub target_representative_face_id: Option<i64>,
    pub resulting_representative_face_id: Option<i64>,
    pub source_representative_will_be_discarded: bool,
    pub external_links_moved: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PeopleDiagnostics {
    pub schema_version: i64,
    pub detector_model_id: String,
    pub recognizer_model_id: String,
    pub recognition_policy_active: bool,
    pub recognition_policy_version: i64,
    pub recognition_policy_digest: String,
    pub indexed_assets: u64,
    pub active_faces: u64,
    pub analysis_pending: u64,
    pub analysis_errors: u64,
    pub reconciliation_conflicts: u64,
    pub recognition_refresh_errors: u64,
    pub stale_artifacts: u64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RetryFailedResult {
    pub analysis_jobs: u64,
    pub recognition_jobs: u64,
}

#[cfg(test)]
#[derive(Clone, Debug, PartialEq)]
pub struct FaceAssignment {
    pub face_id: i64,
    pub person_id: i64,
    pub kind: FaceAssignmentKind,
    pub provenance: String,
    pub confidence: Option<f32>,
}

#[cfg(test)]
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PeopleCatalog {
    pub people: Vec<PersonRecord>,
    pub faces: Vec<FaceRecord>,
    pub assignments: Vec<FaceAssignment>,
}

#[derive(Clone, Debug)]
pub(super) struct ImportedPerson {
    external_key: Vec<u8>,
    name: String,
}

#[derive(Clone, Debug)]
struct ImportedAssignment {
    person_key: Vec<u8>,
    kind: FaceAssignmentKind,
}

#[derive(Clone, Debug)]
pub(super) struct ImportedFace {
    source_key: String,
    relative_path: Vec<u8>,
    bounds: [f32; 4],
    origin: FaceOrigin,
    ignored: bool,
    cluster_key: Option<String>,
    assignments: Vec<ImportedAssignment>,
}

#[derive(Clone, Debug, Default)]
pub(super) struct PeopleImport {
    source_namespace: String,
    people: BTreeMap<Vec<u8>, ImportedPerson>,
    pub(super) faces: Vec<ImportedFace>,
}

pub(super) fn collect_lightroom_people(
    root: &Path,
    lightroom: &Connection,
    source_namespace: &str,
    report: &mut ImportReport,
) -> Result<PeopleImport> {
    if !table_exists(lightroom, "AgLibraryFace")? {
        return Ok(PeopleImport::default());
    }
    if source_namespace.is_empty() {
        bail!("Lightroom People import could not find a stable catalog identity");
    }
    for required in [
        "AgLibraryKeyword",
        "AgLibraryKeywordFace",
        "AgLibraryFaceCluster",
    ] {
        if !table_exists(lightroom, required)? {
            bail!("Lightroom face schema is incomplete: missing {required}");
        }
    }

    report.lightroom_faces = count(lightroom, "SELECT count(*) FROM AgLibraryFace")?;
    report.orphan_face_assignments = count(
        lightroom,
        "SELECT count(*) FROM AgLibraryKeywordFace assignment \
         LEFT JOIN AgLibraryFace face ON face.id_local=assignment.face \
         WHERE face.id_local IS NULL",
    )?;
    if table_exists(lightroom, "AgLibraryFaceData")? {
        report.incomplete_lightroom_embeddings = count(
            lightroom,
            "SELECT count(*) FROM AgLibraryFaceData \
             WHERE length(CAST(data AS BLOB)) != 1024",
        )?;
    }

    let mut query = lightroom
        .prepare(
            "SELECT face.id_local, image.id_local, image.masterImage, root.absolutePath, \
                folder.pathFromRoot, file.baseName, file.extension, \
                face.tl_x, face.tl_y, face.br_x, face.br_y, face.imageOrientation, \
                face.origination, coalesce(face.ignored, 0), face.cluster, \
                keyword.id_global, keyword.name, keyword.keywordType, \
                assignment.userPick, assignment.userReject \
         FROM AgLibraryFace face \
         JOIN Adobe_images image ON image.id_local=face.image \
         JOIN AgLibraryFile file ON file.id_local=image.rootFile \
         JOIN AgLibraryFolder folder ON folder.id_local=file.folder \
         JOIN AgLibraryRootFolder root ON root.id_local=folder.rootFolder \
         LEFT JOIN AgLibraryKeywordFace assignment ON assignment.face=face.id_local \
         LEFT JOIN AgLibraryKeyword keyword ON keyword.id_local=assignment.tag \
         ORDER BY face.id_local, assignment.id_local",
        )
        .context("unsupported Lightroom face schema")?;
    let mut rows = query.query([])?;
    let mut planned = PeopleImport {
        source_namespace: source_namespace.to_owned(),
        ..PeopleImport::default()
    };
    let mut face_positions = HashMap::<i64, usize>::new();
    let mut clusters = BTreeSet::new();
    while let Some(row) = rows.next()? {
        let source_id: i64 = row.get(0)?;
        let face_index = if let Some(index) = face_positions.get(&source_id).copied() {
            index
        } else {
            if row.get::<_, Option<i64>>(2)?.is_some() {
                report.skipped_virtual_copy_faces += 1;
                continue;
            }
            let absolute_root: String = row.get(3)?;
            let folder: String = row.get(4)?;
            let base: String = row.get(5)?;
            let extension: String = row.get(6)?;
            let Some(relative) = resolve_relative_source(
                root,
                &absolute_root,
                &folder,
                &format!("{base}.{extension}"),
            ) else {
                *report
                    .face_import_warnings
                    .entry("face source is missing or outside the opened root".to_owned())
                    .or_default() += 1;
                continue;
            };
            let display_bounds = [row.get(7)?, row.get(8)?, row.get(9)?, row.get(10)?];
            let orientation: String = row.get(11)?;
            let Some(bounds) = displayed_bounds_to_source(display_bounds, &orientation) else {
                *report
                    .face_import_warnings
                    .entry(format!("unsupported Lightroom orientation {orientation}"))
                    .or_default() += 1;
                continue;
            };
            if !valid_bounds(bounds) {
                *report
                    .face_import_warnings
                    .entry("invalid Lightroom face bounds".to_owned())
                    .or_default() += 1;
                continue;
            }
            let raw_origin: f64 = row.get(12)?;
            let origin = if (raw_origin - 1.0).abs() < f64::EPSILON {
                FaceOrigin::LightroomDetected
            } else if (raw_origin - 3.0).abs() < f64::EPSILON {
                FaceOrigin::LightroomManual
            } else {
                *report
                    .face_import_warnings
                    .entry(format!("unknown Lightroom face origin {raw_origin}"))
                    .or_default() += 1;
                FaceOrigin::Unknown
            };
            let cluster = row
                .get::<_, Option<i64>>(14)?
                .map(|value| value.to_string());
            if let Some(cluster) = &cluster {
                clusters.insert(cluster.clone());
            }
            planned.faces.push(ImportedFace {
                source_key: format!("{source_namespace}:{source_id}"),
                relative_path: encode_relative_path(&relative)?,
                bounds,
                origin,
                ignored: row.get::<_, i64>(13)? != 0,
                cluster_key: cluster,
                assignments: Vec::new(),
            });
            let index = planned.faces.len() - 1;
            face_positions.insert(source_id, index);
            report.imported_faces += 1;
            index
        };

        let Some(keyword_type) = row.get::<_, Option<String>>(17)? else {
            report.unnamed_faces += 1;
            continue;
        };
        if keyword_type != "person" {
            *report
                .face_import_warnings
                .entry("face assignment references a non-person keyword".to_owned())
                .or_default() += 1;
            continue;
        }
        let mut external_key = source_namespace.as_bytes().to_vec();
        external_key.push(b':');
        external_key.extend_from_slice(row.get_ref(15)?.as_bytes()?);
        let name = row.get::<_, Option<String>>(16)?.unwrap_or_default();
        if name.trim().is_empty() {
            *report
                .face_import_warnings
                .entry("person keyword has an empty name".to_owned())
                .or_default() += 1;
            continue;
        }
        planned
            .people
            .entry(external_key.clone())
            .or_insert_with(|| ImportedPerson {
                external_key: external_key.clone(),
                name,
            });
        let kind = if row.get::<_, Option<i64>>(19)? == Some(1) {
            FaceAssignmentKind::Rejected
        } else if row.get::<_, Option<i64>>(18)? == Some(1) {
            report.confirmed_face_assignments += 1;
            FaceAssignmentKind::Confirmed
        } else {
            report.suggested_face_assignments += 1;
            FaceAssignmentKind::Suggested
        };
        planned.faces[face_index]
            .assignments
            .push(ImportedAssignment {
                person_key: external_key,
                kind,
            });
    }
    report.imported_face_clusters = clusters.len() as u64;
    Ok(planned)
}

pub(super) fn write_people_import(
    transaction: &Transaction<'_>,
    import: &PeopleImport,
) -> Result<()> {
    // A Lightroom catalog without the People tables is an unsupported source,
    // not an authoritative empty People catalog. Never mutate existing data.
    if import.source_namespace.is_empty() {
        return Ok(());
    }
    let namespace_prefix = format!("{}:", import.source_namespace);
    let confirmed_before = lightroom_confirmed_memberships(transaction, &namespace_prefix)?;
    let recognition_models = super::recognition::model_ids_in_connection(transaction)?;
    adopt_legacy_lightroom_keys(transaction, import)?;
    transaction.execute("DROP TABLE IF EXISTS temp.lightroom_seen_faces", [])?;
    transaction.execute(
        "CREATE TEMP TABLE lightroom_seen_faces(source_key TEXT PRIMARY KEY)",
        [],
    )?;

    let mut people = HashMap::<Vec<u8>, i64>::new();
    for person in import.people.values() {
        let existing = transaction
            .query_row(
                "SELECT person_id FROM external_people \
                 WHERE provenance='lightroom' AND external_key=?1",
                [&person.external_key],
                |row| row.get(0),
            )
            .optional()?;
        let person_id = if let Some(id) = existing {
            transaction.execute(
                "UPDATE people SET display_name=?1, updated_at=datetime('now') \
                 WHERE id=?2 AND locally_renamed=0",
                params![person.name, id],
            )?;
            id
        } else {
            transaction.execute(
                "INSERT INTO people(display_name, locally_renamed, created_at, updated_at) \
                 VALUES(?1, 0, datetime('now'), datetime('now'))",
                [&person.name],
            )?;
            let id = transaction.last_insert_rowid();
            transaction.execute(
                "INSERT INTO external_people(person_id, provenance, external_key) \
                 VALUES(?1, 'lightroom', ?2)",
                params![id, person.external_key],
            )?;
            id
        };
        people.insert(person.external_key.clone(), person_id);
    }

    for face in &import.faces {
        let source_key = &face.source_key;
        transaction.execute(
            "INSERT INTO lightroom_seen_faces(source_key) VALUES(?1)",
            [source_key],
        )?;
        let asset_id = asset_id(transaction, &face.relative_path)?;
        transaction.execute(
            "INSERT INTO faces(asset_id, provenance, source_key, left, top, right, bottom, \
                               origin, ignored, retired, updated_at) \
             VALUES(?1, 'lightroom', ?2, ?3, ?4, ?5, ?6, ?7, ?8, 0, datetime('now')) \
             ON CONFLICT(provenance, source_key) DO UPDATE SET \
                 asset_id=excluded.asset_id, left=excluded.left, top=excluded.top, \
                 right=excluded.right, bottom=excluded.bottom, origin=excluded.origin, \
                 ignored=excluded.ignored, retired=0, updated_at=excluded.updated_at",
            params![
                asset_id,
                source_key,
                face.bounds[0],
                face.bounds[1],
                face.bounds[2],
                face.bounds[3],
                face.origin.as_str(),
                i64::from(face.ignored),
            ],
        )?;
        let face_id: i64 = transaction.query_row(
            "SELECT id FROM faces WHERE provenance='lightroom' AND source_key=?1",
            [source_key],
            |row| row.get(0),
        )?;
        transaction.execute(
            "DELETE FROM face_assignments WHERE face_id=?1 AND provenance='lightroom'",
            [face_id],
        )?;
        transaction.execute(
            "DELETE FROM face_cluster_memberships WHERE face_id=?1 AND provenance='lightroom'",
            [face_id],
        )?;
        if let Some(cluster) = &face.cluster_key {
            transaction.execute(
                "INSERT INTO face_cluster_memberships(face_id, provenance, cluster_key) \
                 VALUES(?1, 'lightroom', ?2)",
                params![face_id, cluster],
            )?;
        }
        for assignment in &face.assignments {
            let Some(person_id) = people.get(&assignment.person_key) else {
                continue;
            };
            transaction.execute(
                "INSERT INTO face_assignments(face_id, person_id, kind, provenance, confidence, updated_at) \
                 VALUES(?1, ?2, ?3, 'lightroom', NULL, datetime('now'))",
                params![face_id, person_id, assignment.kind.as_str()],
            )?;
        }
    }
    transaction.execute(
        "DELETE FROM face_assignments WHERE provenance='lightroom'
           AND face_id IN (SELECT id FROM faces WHERE provenance='lightroom'
             AND substr(source_key, 1, length(?1))=?1
             AND source_key NOT IN (SELECT source_key FROM lightroom_seen_faces))",
        [&namespace_prefix],
    )?;
    transaction.execute(
        "DELETE FROM face_cluster_memberships WHERE provenance='lightroom'
           AND face_id IN (SELECT id FROM faces WHERE provenance='lightroom'
             AND substr(source_key, 1, length(?1))=?1
             AND source_key NOT IN (SELECT source_key FROM lightroom_seen_faces))",
        [&namespace_prefix],
    )?;
    transaction.execute(
        "DELETE FROM faces WHERE provenance='lightroom' \
         AND substr(source_key, 1, length(?1))=?1 \
         AND source_key NOT IN (SELECT source_key FROM lightroom_seen_faces) \
         AND id NOT IN (SELECT face_id FROM face_assignments WHERE provenance='rekesz')",
        [&namespace_prefix],
    )?;
    transaction.execute(
        "DELETE FROM people WHERE locally_renamed=0 \
         AND id IN (SELECT person_id FROM external_people WHERE provenance='lightroom') \
         AND id NOT IN (SELECT person_id FROM face_assignments)",
        [],
    )?;
    let confirmed_after = lightroom_confirmed_memberships(transaction, &namespace_prefix)?;
    let changed = confirmed_before
        .symmetric_difference(&confirmed_after)
        .copied()
        .collect::<Vec<_>>();
    if !changed.is_empty() {
        let person_ids = changed
            .iter()
            .map(|(person_id, _)| *person_id)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        let face_ids = changed
            .iter()
            .map(|(_, face_id)| *face_id)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        super::recognition::enqueue_refresh_for_models_in_connection(
            transaction,
            &recognition_models,
            &person_ids,
            &face_ids,
        )?;
    }
    transaction.execute("DROP TABLE lightroom_seen_faces", [])?;
    Ok(())
}

fn lightroom_confirmed_memberships(
    connection: &Connection,
    namespace_prefix: &str,
) -> Result<BTreeSet<(i64, i64)>> {
    let mut query = connection.prepare(
        "SELECT assignment.person_id, assignment.face_id
           FROM face_assignments assignment
           JOIN faces face ON face.id=assignment.face_id
          WHERE assignment.provenance='lightroom' AND assignment.kind='confirmed'
            AND face.provenance='lightroom'
            AND substr(face.source_key, 1, length(?1))=?1",
    )?;
    Ok(query
        .query_map([namespace_prefix], |row| Ok((row.get(0)?, row.get(1)?)))?
        .collect::<rusqlite::Result<BTreeSet<_>>>()?)
}

fn adopt_legacy_lightroom_keys(transaction: &Transaction<'_>, import: &PeopleImport) -> Result<()> {
    let namespaced_faces: i64 = transaction.query_row(
        "SELECT count(*) FROM faces WHERE provenance='lightroom' AND instr(source_key, ':')>0",
        [],
        |row| row.get(0),
    )?;
    if namespaced_faces > 0 {
        transaction.execute(
            "UPDATE faces SET retired=1
             WHERE provenance='lightroom' AND instr(source_key, ':')=0",
            [],
        )?;
        return Ok(());
    }
    let adopted_faces = adopt_matching_legacy_faces(transaction, import)?;
    let mut statement = transaction.prepare(
        "SELECT person_id, external_key FROM external_people
         WHERE provenance='lightroom' AND instr(CAST(external_key AS TEXT), ':')=0",
    )?;
    let legacy_people = statement
        .query_map([], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, Vec<u8>>(1)?))
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    drop(statement);
    for (person_id, key) in legacy_people {
        let mut assignments =
            transaction.prepare("SELECT face_id FROM face_assignments WHERE person_id=?1")?;
        let linked_to_adopted_face = assignments
            .query_map([person_id], |row| row.get::<_, i64>(0))?
            .collect::<rusqlite::Result<Vec<_>>>()?
            .into_iter()
            .any(|face_id| adopted_faces.contains(&face_id));
        let mut namespaced = import.source_namespace.as_bytes().to_vec();
        namespaced.push(b':');
        namespaced.extend_from_slice(&key);
        if linked_to_adopted_face && import.people.contains_key(&namespaced) {
            transaction.execute(
                "UPDATE external_people SET external_key=?1
                 WHERE person_id=?2 AND provenance='lightroom'",
                params![namespaced, person_id],
            )?;
        }
    }
    transaction.execute(
        "UPDATE faces SET retired=1
         WHERE provenance='lightroom' AND instr(source_key, ':')=0",
        [],
    )?;
    Ok(())
}

fn adopt_matching_legacy_faces(
    transaction: &Transaction<'_>,
    import: &PeopleImport,
) -> Result<BTreeSet<i64>> {
    let imported = import
        .faces
        .iter()
        .filter_map(|face| face.source_key.rsplit_once(':').map(|(_, id)| (id, face)))
        .collect::<HashMap<_, _>>();
    let mut statement = transaction.prepare(
        "SELECT face.id, face.source_key, asset.relative_path,
                face.left, face.top, face.right, face.bottom
         FROM faces face JOIN assets asset ON asset.id=face.asset_id
         WHERE face.provenance='lightroom' AND instr(face.source_key, ':')=0",
    )?;
    let mut rows = statement.query([])?;
    let mut matches = Vec::new();
    while let Some(row) = rows.next()? {
        let face_id: i64 = row.get(0)?;
        let source_key: String = row.get(1)?;
        let Some(candidate) = imported.get(source_key.as_str()) else {
            continue;
        };
        let relative_path: Vec<u8> = row.get(2)?;
        let bounds: [f32; 4] = [row.get(3)?, row.get(4)?, row.get(5)?, row.get(6)?];
        let geometry_matches = bounds
            .iter()
            .zip(candidate.bounds)
            .all(|(legacy, current)| (legacy - current).abs() <= 0.000_01);
        if relative_path == candidate.relative_path && geometry_matches {
            matches.push((face_id, candidate.source_key.clone()));
        }
    }
    drop(rows);
    drop(statement);
    let mut adopted = BTreeSet::new();
    for (face_id, source_key) in matches {
        transaction.execute(
            "UPDATE faces SET source_key=?1 WHERE id=?2 AND provenance='lightroom'",
            params![source_key, face_id],
        )?;
        adopted.insert(face_id);
    }
    Ok(adopted)
}

#[cfg(test)]
pub fn load(root: &Path) -> Result<PeopleCatalog> {
    let path = super::catalog_path(root);
    if !path.is_file() {
        return Ok(PeopleCatalog::default());
    }
    let mut catalog = PeopleCatalog::default();
    let mut offset = 0;
    loop {
        let page = list_people_page(root, None, offset, 200)?;
        let count = page.items.len();
        catalog
            .people
            .extend(page.items.into_iter().map(|person| PersonRecord {
                id: person.id,
                name: person.name,
                confirmed_faces: person.confirmed_faces,
                suggested_faces: person.suggested_faces,
                representative_face_id: person.representative_face_id,
            }));
        if count < 200 {
            break;
        }
        offset += count as u64;
    }
    let connection = Connection::open_with_flags(
        path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )?;
    connection.busy_timeout(Duration::from_secs(5))?;
    super::validate_schema(&connection)?;
    let mut faces = connection.prepare(
        "SELECT face.id, asset.relative_path, face.left, face.top, face.right, face.bottom, \
                face.origin, face.ignored, cluster.cluster_key \
         FROM faces face JOIN assets asset ON asset.id=face.asset_id \
         LEFT JOIN face_cluster_memberships cluster ON cluster.face_id=face.id \
             AND cluster.provenance='lightroom' \
         WHERE face.retired=0 \
         ORDER BY face.id",
    )?;
    let rows = faces.query_map([], |row| {
        Ok((
            row.get::<_, i64>(0)?,
            row.get::<_, Vec<u8>>(1)?,
            [row.get(2)?, row.get(3)?, row.get(4)?, row.get(5)?],
            row.get::<_, String>(6)?,
            row.get::<_, i64>(7)? != 0,
            row.get::<_, Option<String>>(8)?,
        ))
    })?;
    for row in rows {
        let (id, relative, bounds, origin, ignored, cluster) = row?;
        catalog.faces.push(FaceRecord {
            id,
            path: root.join(decode_relative_path(&relative)?),
            bounds,
            origin: FaceOrigin::parse(&origin),
            ignored,
            cluster,
        });
    }
    for person in &catalog.people {
        let mut offset = 0;
        loop {
            let page = list_person_faces_page(root, person.id, None, offset, 500)?;
            let count = page.items.len();
            catalog
                .assignments
                .extend(page.items.into_iter().map(|item| FaceAssignment {
                    face_id: item.face_id,
                    person_id: item.person_id,
                    kind: item.kind,
                    provenance: item.provenance,
                    confidence: item.confidence,
                }));
            if count < 500 {
                break;
            }
            offset += count as u64;
        }
    }
    Ok(catalog)
}

#[cfg(test)]
pub fn list_people_page(
    root: &Path,
    subtree: Option<&Path>,
    offset: u64,
    limit: usize,
) -> Result<PersonPage> {
    list_people_page_sorted(root, subtree, PeopleSort::Name, offset, limit)
}

#[cfg(test)]
pub fn list_people_page_sorted(
    root: &Path,
    subtree: Option<&Path>,
    sort: PeopleSort,
    offset: u64,
    limit: usize,
) -> Result<PersonPage> {
    list_people_page_sorted_cancellable(
        root,
        subtree,
        sort,
        offset,
        limit,
        HashMap::new(),
        Arc::new(AtomicBool::new(false)),
    )
}

#[cfg(test)]
pub fn list_fresh_people_page_with_authoritative_reads_denied(root: &Path) -> Result<PersonPage> {
    use rusqlite::hooks::{AuthAction, AuthContext, Authorization};
    let connection = Connection::open_with_flags(
        super::catalog_path(root),
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
    )?;
    connection.authorizer(Some(|context: AuthContext<'_>| match context.action {
        AuthAction::Read { table_name, .. }
            if table_name == "faces"
                || table_name == "face_assignments"
                || table_name == "recognition_suggestions" =>
        {
            Authorization::Deny
        }
        _ => Authorization::Allow,
    }))?;
    let deep = super::scope_key(Path::new("profile/2024/08/30/event"))?;
    let mut last = None;
    for scope in [None, Some(deep.as_slice())] {
        for sort in [
            PeopleSort::Name,
            PeopleSort::Confirmed,
            PeopleSort::Review,
            PeopleSort::RecentlyChanged,
        ] {
            for offset in [0, 30] {
                last = Some(list_people_page_with_connection(
                    &connection,
                    root,
                    scope,
                    sort,
                    offset,
                    30,
                    &HashMap::from([(1, 50)]),
                )?);
            }
        }
    }
    last.context("authorizer overview matrix was empty")
}

#[cfg(test)]
pub fn rebuild_people_summary(root: &Path, cancelled: Arc<AtomicBool>) -> Result<u64> {
    if cancelled.load(Ordering::Acquire) {
        return Err(crate::scheduler::MaintenanceYield("People summary rebuild cancelled").into());
    }
    let worker_id = format!(
        "people-summary-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    );
    rebuild_people_summary_managed(root, cancelled, &worker_id, None, None, Arc::new(|| false))
}

#[cfg(test)]
pub fn rebuild_people_summary_with_checkpoint(
    root: &Path,
    cancelled: Arc<AtomicBool>,
    checkpoint: &dyn Fn(&AtomicBool) -> bool,
) -> Result<u64> {
    rebuild_people_summary_with_preemption(root, cancelled, checkpoint, Arc::new(|| false))
}

pub fn rebuild_people_summary_with_preemption(
    root: &Path,
    cancelled: Arc<AtomicBool>,
    checkpoint: &dyn Fn(&AtomicBool) -> bool,
    preempt_requested: Arc<dyn Fn() -> bool + Send + Sync>,
) -> Result<u64> {
    let worker_id = format!(
        "people-summary-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    );
    rebuild_people_summary_managed(
        root,
        cancelled,
        &worker_id,
        None,
        Some(checkpoint),
        preempt_requested,
    )
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PeopleSummaryRebuildStatus {
    pub source_epoch: u64,
    pub published_epoch: u64,
    pub queue_status: Option<String>,
    pub requested_epoch: Option<u64>,
    pub attempts: u32,
    pub worker_id: Option<String>,
    pub lease_expires_at: Option<String>,
    pub retry_after: Option<String>,
    pub error: Option<String>,
    pub lease_expired: bool,
    pub retry_ready: bool,
}

impl PeopleSummaryRebuildStatus {
    pub fn needs_work(&self) -> bool {
        self.published_epoch != self.source_epoch || self.queue_status.is_some()
    }

    pub fn ready_to_enqueue(&self) -> bool {
        if !self.needs_work() {
            return false;
        }
        match self.queue_status.as_deref() {
            Some("running") => self.lease_expired,
            Some("pending" | "error") => self.retry_ready,
            Some(_) => false,
            None => true,
        }
    }
}

pub fn people_summary_rebuild_status(root: &Path) -> Result<PeopleSummaryRebuildStatus> {
    let path = super::catalog_path(root);
    if !path.is_file() {
        return Ok(PeopleSummaryRebuildStatus::default());
    }
    let connection = Connection::open(path)?;
    connection
        .query_row(
            "SELECT e.source_epoch,e.published_epoch,q.status,q.requested_epoch,
                    coalesce(q.attempts,0),q.worker_id,q.lease_expires_at,q.retry_after,q.error,
                    coalesce(q.lease_expires_at<datetime('now'),0),
                    coalesce(q.retry_after IS NULL OR q.retry_after<=datetime('now'),1)
             FROM people_summary_epoch e LEFT JOIN people_summary_rebuild_queue q ON q.id=e.id
             WHERE e.id=1",
            [],
            |row| {
                let source_epoch = row.get::<_, i64>(0)?;
                let published_epoch = row.get::<_, i64>(1)?;
                let requested_epoch = row.get::<_, Option<i64>>(3)?;
                let attempts = row.get::<_, i64>(4)?;
                Ok((
                    source_epoch,
                    published_epoch,
                    row.get(2)?,
                    requested_epoch,
                    attempts,
                    row.get(5)?,
                    row.get(6)?,
                    row.get(7)?,
                    row.get(8)?,
                    row.get(9)?,
                    row.get(10)?,
                ))
            },
        )
        .and_then(
            |(
                source,
                published,
                queue_status,
                requested,
                attempts,
                worker,
                lease,
                retry,
                error,
                lease_expired,
                retry_ready,
            )| {
                Ok(PeopleSummaryRebuildStatus {
                    source_epoch: source
                        .try_into()
                        .map_err(|_| rusqlite::Error::InvalidQuery)?,
                    published_epoch: published
                        .try_into()
                        .map_err(|_| rusqlite::Error::InvalidQuery)?,
                    queue_status,
                    requested_epoch: requested
                        .map(|epoch| epoch.try_into().map_err(|_| rusqlite::Error::InvalidQuery))
                        .transpose()?,
                    attempts: attempts
                        .try_into()
                        .map_err(|_| rusqlite::Error::InvalidQuery)?,
                    worker_id: worker,
                    lease_expires_at: lease,
                    retry_after: retry,
                    error,
                    lease_expired,
                    retry_ready,
                })
            },
        )
        .context("failed to read People summary rebuild status")
}

fn rebuild_people_summary_managed(
    root: &Path,
    cancelled: Arc<AtomicBool>,
    worker_id: &str,
    before_publish: Option<&dyn Fn()>,
    checkpoint: Option<&dyn Fn(&AtomicBool) -> bool>,
    preempt_requested: Arc<dyn Fn() -> bool + Send + Sync>,
) -> Result<u64> {
    let query_interrupted = Arc::new(AtomicBool::new(false));
    match rebuild_people_summary_inner(
        root,
        cancelled,
        worker_id,
        before_publish,
        checkpoint,
        preempt_requested,
        query_interrupted.clone(),
    ) {
        Ok(epoch) => Ok(epoch),
        Err(error) => {
            let yielded = error
                .downcast_ref::<crate::scheduler::MaintenanceYield>()
                .is_some()
                || query_interrupted.load(Ordering::Acquire);
            let persisted_error = (!yielded).then(|| format!("{error:#}"));
            if !format!("{error:#}").contains("already leased")
                && let Ok(mut connection) = Connection::open(super::catalog_path(root))
            {
                let requeued = connection.execute(
                    "UPDATE people_summary_rebuild_queue SET status='pending', requested_epoch=(
                       SELECT source_epoch FROM people_summary_epoch WHERE id=1), worker_id=NULL,
                       claimed_epoch=NULL,
                       lease_expires_at=NULL,
                       retry_after=CASE WHEN ?1 IS NULL THEN NULL ELSE
                         datetime('now',printf('+%d seconds',min(300,1 << min(attempts,8)))) END,
                       error=?1, updated_at=datetime('now')
                     WHERE id=1 AND status='running' AND worker_id=?2",
                    params![persisted_error, worker_id],
                )
                .with_context(|| "failed to requeue People summary rebuild")?;
                let _ = requeued;
                if let Ok(published) = connection.query_row(
                    "SELECT published_epoch FROM people_summary_epoch WHERE id=1",
                    [],
                    |row| row.get::<_, i64>(0),
                ) {
                    let _ = gc_people_summary_epochs(
                        &mut connection,
                        published,
                        &AtomicBool::new(false),
                        None,
                    );
                }
            }
            if yielded {
                Err(crate::scheduler::MaintenanceYield("People summary rebuild preempted").into())
            } else {
                Err(error)
            }
        }
    }
}

fn rebuild_people_summary_inner(
    root: &Path,
    cancelled: Arc<AtomicBool>,
    worker_id: &str,
    before_publish: Option<&dyn Fn()>,
    checkpoint: Option<&dyn Fn(&AtomicBool) -> bool>,
    preempt_requested: Arc<dyn Fn() -> bool + Send + Sync>,
    query_interrupted: Arc<AtomicBool>,
) -> Result<u64> {
    let cooperate = || -> Result<()> {
        if cancelled.load(Ordering::Acquire)
            || preempt_requested()
            || checkpoint.is_some_and(|checkpoint| !checkpoint(&cancelled))
        {
            return Err(
                crate::scheduler::MaintenanceYield("People summary rebuild cancelled").into(),
            );
        }
        Ok(())
    };
    let path = super::catalog_path(root);
    let mut connection = Connection::open(path)?;
    install_representative_quality_function(&connection)?;
    connection.busy_timeout(Duration::from_secs(5))?;
    // Shadow generations can exceed RAM at deep 250k catalogs. Keep SQLite's
    // connection-private TEMP tables file-backed so maintenance remains within
    // the release RSS ceiling.
    connection.execute_batch("PRAGMA temp_store=FILE;")?;
    if let Some(epoch) = connection.query_row(
        "SELECT published_epoch FROM people_summary_epoch WHERE id=1 AND published_epoch=source_epoch AND published_epoch>0",
        [], |row| row.get::<_,i64>(0),
    ).optional()? {
        return epoch.try_into().context("negative People summary epoch");
    }
    let query_cancelled = cancelled.clone();
    let query_preempt = preempt_requested.clone();
    #[cfg(test)]
    let progress_hook = PEOPLE_SUMMARY_PROGRESS_HOOK
        .lock()
        .unwrap()
        .as_ref()
        .filter(|hook| hook.root == root)
        .cloned();
    connection.progress_handler(
        1_000,
        Some(move || {
            #[cfg(test)]
            if let Some(hook) = &progress_hook {
                let steps = hook.steps.fetch_add(1_000, Ordering::AcqRel) + 1_000;
                if steps >= hook.threshold_steps {
                    let (lock, wake) = &*hook.barrier;
                    let mut state = lock.lock().unwrap();
                    if !state.0 {
                        state.0 = true;
                        wake.notify_all();
                        while !state.1 {
                            state = wake.wait(state).unwrap();
                        }
                    }
                }
            }
            let interrupted = query_cancelled.load(Ordering::Acquire) || query_preempt();
            if interrupted {
                query_interrupted.store(true, Ordering::Release);
            }
            interrupted
        }),
    )?;
    let epoch: i64 = connection.query_row(
        "SELECT source_epoch FROM people_summary_epoch WHERE id=1",
        [],
        |row| row.get(0),
    )?;
    let claimed = connection.execute(
        "UPDATE people_summary_rebuild_queue SET status='running', attempts=attempts+1,
         worker_id=?1, claimed_epoch=requested_epoch,
         lease_expires_at=datetime('now','+5 minutes'), updated_at=datetime('now')
         WHERE id=1 AND (status='pending' OR (status='running' AND lease_expires_at<datetime('now')))", [worker_id],
    )?;
    if claimed != 1 {
        bail!("People summary rebuild is already leased");
    }
    connection.execute_batch(
        "CREATE TEMP TABLE build_ancestors AS SELECT * FROM asset_scope_ancestors WHERE 0;
         CREATE TEMP TABLE build_effective AS SELECT * FROM effective_face_assignments WHERE 0;
         CREATE TEMP TABLE build_people AS SELECT * FROM people_scope_summary WHERE 0;
         CREATE TEMP TABLE build_totals AS SELECT * FROM people_scope_totals WHERE 0;
         CREATE TEMP TABLE build_representatives AS SELECT * FROM people_scope_representatives WHERE 0;
         CREATE TEMP TABLE build_faces AS SELECT * FROM people_scope_faces WHERE 0;",
    )?;
    #[cfg(test)]
    let mut stage_started = std::time::Instant::now();
    #[cfg(test)]
    let report_stages = std::env::var_os("REKESZ_PEOPLE_PROFILE_STAGES").is_some();
    #[cfg(test)]
    macro_rules! report_stage {
        ($name:literal) => {{
            if report_stages {
                eprintln!(
                    "People summary stage {}: {:?}",
                    $name,
                    stage_started.elapsed()
                );
            }
            stage_started = std::time::Instant::now();
        }};
    }
    cooperate()?;
    connection.execute(
        "INSERT INTO temp.build_effective SELECT * FROM effective_face_assignments",
        [],
    )?;
    connection.execute_batch(
        "CREATE INDEX temp.build_effective_face_kind ON build_effective(face_id,kind);
         CREATE INDEX temp.build_effective_person_kind ON build_effective(person_id,kind);",
    )?;
    #[cfg(test)]
    report_stage!("effective");
    let assets = {
        let mut statement = connection.prepare("SELECT id, relative_path FROM assets")?;
        statement
            .query_map([], |row| {
                Ok((row.get::<_, i64>(0)?, row.get::<_, Vec<u8>>(1)?))
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?
    };
    for asset_batch in assets.chunks(500) {
        cooperate()?;
        let transaction = connection.transaction()?;
        {
            let mut insert = transaction.prepare(
                "INSERT OR IGNORE INTO temp.build_ancestors(build_epoch,asset_id,scope_key)
                 VALUES(?1,?2,?3)",
            )?;
            for (asset_id, encoded) in asset_batch {
                if cancelled.load(Ordering::Acquire) {
                    return Err(crate::scheduler::MaintenanceYield(
                        "People summary rebuild cancelled",
                    )
                    .into());
                }
                let relative = decode_relative_path(encoded)?;
                insert.execute(params![epoch, *asset_id, Vec::<u8>::new()])?;
                let mut ancestor = PathBuf::new();
                for component in relative.parent().into_iter().flat_map(Path::components) {
                    if let std::path::Component::Normal(component) = component {
                        ancestor.push(component);
                        insert.execute(params![epoch, *asset_id, super::scope_key(&ancestor)?])?;
                    }
                }
            }
        }
        transaction.commit()?;
    }
    #[cfg(test)]
    report_stage!("ancestors");
    cooperate()?;
    connection.execute(
        "INSERT INTO temp.build_people
         SELECT ?1, ancestor.scope_key, person.id, person.display_name, person.display_name, person.updated_at,
                sum(assignment.kind='confirmed'), sum(assignment.kind='suggested')
         FROM temp.build_effective assignment JOIN faces face ON face.id=assignment.face_id
         JOIN people person ON person.id=assignment.person_id
         JOIN temp.build_ancestors ancestor ON ancestor.asset_id=face.asset_id AND ancestor.build_epoch=?1
         WHERE assignment.kind IN ('confirmed','suggested') AND face.retired=0
           AND face.ignored=0 AND face.locally_ignored=0
         GROUP BY ancestor.scope_key,person.id",
        [epoch],
    )?;
    connection
        .execute_batch("CREATE INDEX temp.build_people_person ON build_people(person_id);")?;
    #[cfg(test)]
    report_stage!("people");
    cooperate()?;
    let names = {
        let mut statement = connection.prepare("SELECT id,display_name FROM people")?;
        statement
            .query_map([], |row| {
                Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?
    };
    for batch in names.chunks(500) {
        cooperate()?;
        let transaction = connection.transaction()?;
        {
            let mut update = transaction.prepare(
                "UPDATE temp.build_people SET sort_key=?1
                 WHERE build_epoch=?2 AND person_id=?3",
            )?;
            for (person_id, name) in batch {
                update.execute(params![name.to_lowercase(), epoch, person_id])?;
            }
        }
        transaction.commit()?;
    }
    #[cfg(test)]
    report_stage!("names");
    cooperate()?;
    connection.execute(
        "INSERT INTO temp.build_faces
         SELECT ?1,a.scope_key,'review',f.id,e.person_id,p.display_name,e.kind,e.provenance,e.confidence,
                asset.relative_path,f.left,f.top,f.right,f.bottom,f.origin
         FROM temp.build_effective e JOIN faces f ON f.id=e.face_id JOIN assets asset ON asset.id=f.asset_id
         JOIN people p ON p.id=e.person_id JOIN temp.build_ancestors a ON a.asset_id=f.asset_id AND a.build_epoch=?1
         WHERE e.kind='suggested' AND f.retired=0 AND f.ignored=0 AND f.locally_ignored=0",
        [epoch],
    )?;
    #[cfg(test)]
    report_stage!("review faces");
    cooperate()?;
    cooperate()?;
    connection.execute(
        "INSERT INTO temp.build_faces
         SELECT ?1,a.scope_key,'unnamed',f.id,0,NULL,'unassigned',f.origin,NULL,
                asset.relative_path,f.left,f.top,f.right,f.bottom,f.origin
         FROM faces f JOIN assets asset ON asset.id=f.asset_id JOIN temp.build_ancestors a ON a.asset_id=f.asset_id AND a.build_epoch=?1
         WHERE f.retired=0 AND f.ignored=0 AND f.locally_ignored=0 AND NOT EXISTS (
           SELECT 1 FROM temp.build_effective e WHERE e.face_id=f.id AND e.kind='confirmed')",
        [epoch],
    )?;
    #[cfg(test)]
    report_stage!("unnamed faces");
    cooperate()?;
    cooperate()?;
    connection.execute(
        "INSERT INTO temp.build_totals
         WITH scopes AS (SELECT DISTINCT scope_key FROM temp.build_ancestors WHERE build_epoch=?1),
         active AS (SELECT a.scope_key,count(DISTINCT f.id) n FROM temp.build_ancestors a
           JOIN faces f ON f.asset_id=a.asset_id WHERE f.retired=0 AND f.ignored=0 AND f.locally_ignored=0 GROUP BY a.scope_key),
         confirmed AS (SELECT a.scope_key,count(DISTINCT e.face_id) n FROM temp.build_effective e
           JOIN faces f ON f.id=e.face_id JOIN temp.build_ancestors a ON a.asset_id=f.asset_id AND a.build_epoch=?1
           WHERE e.kind='confirmed' AND f.retired=0 AND f.ignored=0 AND f.locally_ignored=0 GROUP BY a.scope_key),
         suggested AS (SELECT a.scope_key,count(*) n FROM temp.build_effective e JOIN faces f ON f.id=e.face_id
           JOIN temp.build_ancestors a ON a.asset_id=f.asset_id AND a.build_epoch=?1
           WHERE e.kind='suggested' AND f.retired=0 AND f.ignored=0 AND f.locally_ignored=0 GROUP BY a.scope_key),
         pending AS (SELECT a.scope_key,count(*) n FROM face_analysis_state s JOIN temp.build_ancestors a ON a.asset_id=s.asset_id AND a.build_epoch=?1
           WHERE s.status IN ('pending','running') GROUP BY a.scope_key),
         errors AS (SELECT a.scope_key,count(*) n FROM face_analysis_state s JOIN temp.build_ancestors a ON a.asset_id=s.asset_id AND a.build_epoch=?1
           WHERE s.status='error' GROUP BY a.scope_key)
         SELECT ?1,scopes.scope_key,coalesce(active.n,0),coalesce(confirmed.n,0),coalesce(suggested.n,0),
                coalesce(pending.n,0),coalesce(errors.n,0) FROM scopes
         LEFT JOIN active USING(scope_key) LEFT JOIN confirmed USING(scope_key) LEFT JOIN suggested USING(scope_key)
         LEFT JOIN pending USING(scope_key) LEFT JOIN errors USING(scope_key)", [epoch],
    )?;
    #[cfg(test)]
    report_stage!("totals");
    cooperate()?;
    cooperate()?;
    connection.execute(
        "INSERT INTO temp.build_representatives
         SELECT ?1,X'',person_id,rank,face_id,asset_id,relative_path,left,top,right,bottom FROM (
           SELECT assignment.person_id,face.id face_id,face.asset_id,asset.relative_path,
             face.left,face.top,face.right,face.bottom,
             row_number() OVER(PARTITION BY assignment.person_id ORDER BY
               (face.id=person.representative_face_id AND person.representative_is_manual=1) DESC,
               (assignment.provenance='rekesz') DESC,
               coalesce(face.detection_confidence,0) DESC,
               ((face.right-face.left)*(face.bottom-face.top)) DESC,
               rekesz_frontal_quality(embedding.landmarks,face.left,face.top,face.right,face.bottom) DESC,
               face.updated_at DESC,face.id)-1 rank
           FROM temp.build_effective assignment JOIN faces face ON face.id=assignment.face_id
           JOIN assets asset ON asset.id=face.asset_id JOIN people person ON person.id=assignment.person_id
           LEFT JOIN face_embeddings embedding ON embedding.face_id=face.id
             AND embedding.model_id='sface-2021dec'
           WHERE assignment.kind='confirmed' AND face.retired=0
             AND face.ignored=0 AND face.locally_ignored=0
             AND ((face.right-face.left)*(face.bottom-face.top))>=0.0025)", [epoch],
    )?;
    #[cfg(test)]
    report_stage!("representatives");
    cooperate()?;
    for (shadow, live) in [
        ("build_ancestors", "asset_scope_ancestors"),
        ("build_people", "people_scope_summary"),
        ("build_totals", "people_scope_totals"),
        ("build_representatives", "people_scope_representatives"),
        ("build_faces", "people_scope_faces"),
    ] {
        loop {
            let transaction =
                connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
            let deleted = transaction.execute(
                &format!(
                    "DELETE FROM {live} WHERE rowid IN (
                       SELECT rowid FROM {live} WHERE build_epoch=?1 LIMIT 500)"
                ),
                [epoch],
            )?;
            transaction.commit()?;
            if deleted < 500 {
                break;
            }
        }
        // Use a rowid cursor rather than OFFSET. OFFSET repeatedly walks all
        // preceding rows and made copying a million-row deep-scope shadow
        // quadratic even though each writer transaction remained bounded.
        let mut copied_through = 0_i64;
        loop {
            cooperate()?;
            let batch_end: Option<i64> = connection.query_row(
                &format!(
                    "SELECT max(rowid) FROM (
                       SELECT rowid FROM temp.{shadow}
                       WHERE rowid>?1 ORDER BY rowid LIMIT 500)"
                ),
                [copied_through],
                |row| row.get(0),
            )?;
            let Some(batch_end) = batch_end else {
                break;
            };
            let transaction =
                connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
            transaction.execute(
                &format!(
                    "INSERT INTO {live} SELECT * FROM temp.{shadow}
                     WHERE rowid>?1 AND rowid<=?2 ORDER BY rowid"
                ),
                params![copied_through, batch_end],
            )?;
            let heartbeat = transaction.execute(
                "UPDATE people_summary_rebuild_queue SET lease_expires_at=datetime('now','+5 minutes'),
                 updated_at=datetime('now') WHERE id=1 AND status='running' AND worker_id=?1",
                [worker_id],
            )?;
            if heartbeat != 1 {
                bail!("People summary rebuild lease was reclaimed");
            }
            transaction.commit()?;
            copied_through = batch_end;
        }
    }
    #[cfg(test)]
    if report_stages {
        eprintln!(
            "People summary stage live copy: {:?}",
            stage_started.elapsed()
        );
    }
    cooperate()?;
    if let Some(before_publish) = before_publish {
        before_publish();
    }
    let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
    let published = transaction.execute(
        "UPDATE people_summary_epoch SET published_epoch=?1 WHERE id=1 AND source_epoch=?1
         AND EXISTS(SELECT 1 FROM people_summary_rebuild_queue q WHERE q.id=1
                    AND q.status='running' AND q.worker_id=?2 AND q.claimed_epoch=?1
                    AND q.requested_epoch=?1)",
        params![epoch, worker_id],
    )?;
    if published != 1 {
        transaction.execute(
            "UPDATE people_summary_rebuild_queue SET status='pending',
             requested_epoch=(SELECT source_epoch FROM people_summary_epoch WHERE id=1),
             worker_id=NULL,claimed_epoch=NULL,lease_expires_at=NULL,retry_after=NULL,
             error='source changed during rebuild',updated_at=datetime('now')
             WHERE id=1 AND status='running' AND worker_id=?1",
            [worker_id],
        )?;
        transaction.commit()?;
        bail!("People summary changed during rebuild");
    }
    transaction.execute("DELETE FROM people_summary_rebuild_queue WHERE id=1", [])?;
    transaction.commit()?;

    // Readers select by epoch. Reclaim obsolete generations independently in
    // bounded writer transactions so maintenance never creates a long pause.
    let _ = gc_people_summary_epochs(&mut connection, epoch, &cancelled, None);
    epoch.try_into().context("negative People summary epoch")
}

fn gc_people_summary_epochs(
    connection: &mut Connection,
    keep_epoch: i64,
    cancelled: &AtomicBool,
    checkpoint: Option<&dyn Fn()>,
) -> Result<()> {
    for table in [
        "people_scope_summary",
        "people_scope_totals",
        "people_scope_representatives",
        "people_scope_faces",
        "asset_scope_ancestors",
    ] {
        loop {
            if cancelled.load(Ordering::Acquire) {
                bail!("People summary cleanup cancelled");
            }
            let transaction =
                connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
            let deleted = transaction.execute(
                &format!(
                    "DELETE FROM {table} WHERE rowid IN (
                       SELECT rowid FROM {table} WHERE build_epoch<>?1
                         AND build_epoch NOT IN (
                           SELECT claimed_epoch FROM people_summary_rebuild_queue
                            WHERE status='running' AND claimed_epoch IS NOT NULL)
                       LIMIT 500)"
                ),
                [keep_epoch],
            )?;
            transaction.commit()?;
            if let Some(checkpoint) = checkpoint {
                checkpoint();
            }
            if deleted < 500 {
                break;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
pub fn gc_people_summary_epochs_with_checkpoint(
    root: &Path,
    keep_epoch: i64,
    checkpoint: &dyn Fn(),
) -> Result<()> {
    let mut connection = Connection::open(super::catalog_path(root))?;
    gc_people_summary_epochs(
        &mut connection,
        keep_epoch,
        &AtomicBool::new(false),
        Some(checkpoint),
    )
}

#[cfg(test)]
pub fn rebuild_people_summary_before_publish(
    root: &Path,
    before_publish: &dyn Fn(),
) -> Result<u64> {
    rebuild_people_summary_managed(
        root,
        Arc::new(AtomicBool::new(false)),
        "people-summary-race-test",
        Some(before_publish),
        None,
        Arc::new(|| false),
    )
}

pub fn list_people_page_sorted_cancellable(
    root: &Path,
    subtree: Option<&Path>,
    sort: PeopleSort,
    offset: u64,
    limit: usize,
    representative_offsets: HashMap<i64, u64>,
    cancelled: Arc<AtomicBool>,
) -> Result<PersonPage> {
    if cancelled.load(Ordering::Acquire) {
        bail!("People query interrupted");
    }
    let path = super::catalog_path(root);
    if !path.is_file() {
        return Ok(PersonPage {
            items: Vec::new(),
            total: 0,
            offset,
            unnamed_faces: 0,
            review_faces: 0,
            analysis_pending: 0,
            analysis_errors: 0,
        });
    }
    let connection = Connection::open_with_flags(
        path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )?;
    install_representative_quality_function(&connection)?;
    connection.busy_timeout(Duration::from_secs(5))?;
    let query_cancelled = cancelled.clone();
    connection.progress_handler(1_000, Some(move || query_cancelled.load(Ordering::Acquire)))?;
    super::validate_schema(&connection)?;
    let prefix = subtree
        .map(|subtree| {
            subtree
                .strip_prefix(root)
                .with_context(|| {
                    format!(
                        "scope {} is outside library {}",
                        subtree.display(),
                        root.display()
                    )
                })
                .and_then(scope_key)
        })
        .transpose()?;
    list_people_page_with_connection(
        &connection,
        root,
        prefix.as_deref(),
        sort,
        offset,
        limit,
        &representative_offsets,
    )
}

fn list_people_page_with_connection(
    connection: &Connection,
    root: &Path,
    prefix: Option<&[u8]>,
    sort: PeopleSort,
    offset: u64,
    limit: usize,
    representative_offsets: &HashMap<i64, u64>,
) -> Result<PersonPage> {
    let summary_scope = prefix.unwrap_or_default();
    let fresh_epoch: Option<i64> = connection
        .query_row(
            "SELECT published_epoch FROM people_summary_epoch
         WHERE id=1 AND published_epoch=source_epoch AND published_epoch>0",
            [],
            |row| row.get(0),
        )
        .optional()?;
    if let Some(epoch) = fresh_epoch {
        return list_people_summary_page(
            connection,
            root,
            summary_scope,
            epoch,
            sort,
            (offset, limit),
            representative_offsets,
        );
    }
    type RepresentativeCandidate = (i64, Vec<u8>, [f32; 4], bool, f64, f64, f64, String);
    struct Aggregate {
        name: String,
        updated_at: String,
        preferred_face_id: Option<i64>,
        confirmed_faces: i64,
        suggested_faces: i64,
        candidates: Vec<RepresentativeCandidate>,
    }

    let limit = limit.clamp(1, 500);
    let path_matches = "(?1 IS NULL OR substr(asset.scope_key, 1, length(?1))=?1)";
    let sql = format!(
        "SELECT assignment.person_id, person.display_name, person.updated_at,
                person.representative_face_id, person.representative_is_manual,
                assignment.kind, assignment.provenance,
                face.id, asset.relative_path, face.left, face.top, face.right, face.bottom,
                coalesce(face.detection_confidence, 0),
                rekesz_frontal_quality(embedding.landmarks,face.left,face.top,face.right,face.bottom),
                face.updated_at
         FROM effective_face_assignments assignment
         JOIN people person ON person.id=assignment.person_id
         JOIN faces face ON face.id=assignment.face_id
         JOIN assets asset ON asset.id=face.asset_id
         LEFT JOIN face_embeddings embedding ON embedding.face_id=face.id
           AND embedding.model_id='sface-2021dec'
         WHERE assignment.kind IN ('confirmed', 'suggested')
           AND face.retired=0 AND face.ignored=0 AND face.locally_ignored=0 AND {path_matches}"
    );
    let mut query = connection.prepare(&sql)?;
    let mut rows = query.query([prefix])?;
    let mut people = HashMap::<i64, Aggregate>::new();
    let mut confirmed_face_ids = HashSet::<i64>::new();
    let mut review_faces = 0_i64;
    while let Some(row) = rows.next()? {
        let person_id: i64 = row.get(0)?;
        let kind: String = row.get(5)?;
        let face_id: i64 = row.get(7)?;
        let aggregate = people.entry(person_id).or_insert_with(|| Aggregate {
            name: row.get(1).unwrap_or_default(),
            updated_at: row.get(2).unwrap_or_default(),
            preferred_face_id: row
                .get::<_, bool>(4)
                .unwrap_or(false)
                .then(|| row.get(3).unwrap_or(None))
                .flatten(),
            confirmed_faces: 0,
            suggested_faces: 0,
            candidates: Vec::new(),
        });
        if kind == "confirmed" {
            aggregate.confirmed_faces += 1;
            confirmed_face_ids.insert(face_id);
            let bounds = [row.get(9)?, row.get(10)?, row.get(11)?, row.get(12)?];
            let area = f64::from((bounds[2] - bounds[0]) * (bounds[3] - bounds[1]));
            if area >= 0.0025 {
                let candidate = (
                    face_id,
                    Vec::new(),
                    bounds,
                    row.get::<_, String>(6)? == "rekesz",
                    row.get(13)?,
                    area,
                    row.get(14)?,
                    row.get(15)?,
                );
                let preferred = aggregate.preferred_face_id.unwrap_or(-1);
                let compare = |left: &RepresentativeCandidate, right: &RepresentativeCandidate| {
                    (right.0 == preferred)
                        .cmp(&(left.0 == preferred))
                        .then_with(|| right.3.cmp(&left.3))
                        .then_with(|| right.4.total_cmp(&left.4))
                        .then_with(|| right.5.total_cmp(&left.5))
                        .then_with(|| right.6.total_cmp(&left.6))
                        .then_with(|| right.7.cmp(&left.7))
                        .then_with(|| left.0.cmp(&right.0))
                };
                let keep = representative_offsets
                    .get(&person_id)
                    .copied()
                    .and_then(|value| usize::try_from(value).ok())
                    .unwrap_or(0)
                    .saturating_add(1);
                if aggregate.candidates.len() < keep
                    || aggregate
                        .candidates
                        .last()
                        .is_some_and(|worst| compare(&candidate, worst).is_lt())
                {
                    let mut candidate = candidate;
                    candidate.1 = row.get(8)?;
                    aggregate.candidates.push(candidate);
                    aggregate.candidates.sort_unstable_by(compare);
                    aggregate.candidates.truncate(keep);
                }
            }
        } else {
            aggregate.suggested_faces += 1;
            review_faces += 1;
        }
    }

    let active_faces: i64 = connection.query_row(
        &format!(
            "SELECT count(*) FROM faces face JOIN assets asset ON asset.id=face.asset_id
             WHERE face.retired=0 AND face.ignored=0 AND face.locally_ignored=0
               AND {path_matches}"
        ),
        [prefix],
        |row| row.get(0),
    )?;
    let unnamed_faces = active_faces.saturating_sub(
        confirmed_face_ids
            .len()
            .try_into()
            .context("confirmed face count overflow")?,
    );
    let analysis_pending: i64 = connection.query_row(
        &format!(
            "SELECT count(*) FROM face_analysis_state analysis
             JOIN assets asset ON asset.id=analysis.asset_id
             WHERE analysis.status IN ('pending', 'running') AND {path_matches}"
        ),
        [prefix],
        |row| row.get(0),
    )?;
    let analysis_errors: i64 = connection.query_row(
        &format!(
            "SELECT count(*) FROM face_analysis_state analysis
             JOIN assets asset ON asset.id=analysis.asset_id
             WHERE analysis.status='error' AND {path_matches}"
        ),
        [prefix],
        |row| row.get(0),
    )?;

    let total = people.len() as u64;
    let mut aggregates = people.into_iter().collect::<Vec<_>>();
    aggregates.sort_unstable_by(|(left_id, left), (right_id, right)| match sort {
        PeopleSort::Name => left
            .name
            .to_lowercase()
            .cmp(&right.name.to_lowercase())
            .then_with(|| left_id.cmp(right_id)),
        PeopleSort::Confirmed => right
            .confirmed_faces
            .cmp(&left.confirmed_faces)
            .then_with(|| left.name.to_lowercase().cmp(&right.name.to_lowercase()))
            .then_with(|| left_id.cmp(right_id)),
        PeopleSort::Review => right
            .suggested_faces
            .cmp(&left.suggested_faces)
            .then_with(|| left.name.to_lowercase().cmp(&right.name.to_lowercase()))
            .then_with(|| left_id.cmp(right_id)),
        PeopleSort::RecentlyChanged => right
            .updated_at
            .cmp(&left.updated_at)
            .then_with(|| left_id.cmp(right_id)),
    });

    let start = usize::try_from(offset)
        .unwrap_or(usize::MAX)
        .min(aggregates.len());
    let end = start.saturating_add(limit).min(aggregates.len());
    let mut items = Vec::with_capacity(end.saturating_sub(start));
    for (person_id, aggregate) in aggregates.drain(start..end) {
        let candidate_offset = representative_offsets
            .get(&person_id)
            .copied()
            .and_then(|value| usize::try_from(value).ok())
            .unwrap_or(0);
        let representative = aggregate
            .candidates
            .get(candidate_offset)
            .map(|candidate| {
                Ok::<_, anyhow::Error>(RepresentativeFace {
                    face_id: candidate.0,
                    path: root.join(decode_relative_path(&candidate.1)?),
                    bounds: candidate.2,
                })
            })
            .transpose()?;
        let representative_face_id = representative.as_ref().map(|face| face.face_id);
        items.push(PersonSummary {
            id: person_id,
            name: aggregate.name,
            confirmed_faces: aggregate.confirmed_faces,
            suggested_faces: aggregate.suggested_faces,
            representative_face_id,
            representative,
        });
    }
    Ok(PersonPage {
        items,
        total,
        offset,
        unnamed_faces: unnamed_faces.try_into().context("negative unnamed count")?,
        review_faces: review_faces.try_into().context("negative review count")?,
        analysis_pending: analysis_pending
            .try_into()
            .context("negative pending count")?,
        analysis_errors: analysis_errors.try_into().context("negative error count")?,
    })
}

fn list_people_summary_page(
    connection: &Connection,
    root: &Path,
    scope: &[u8],
    epoch: i64,
    sort: PeopleSort,
    page: (u64, usize),
    representative_offsets: &HashMap<i64, u64>,
) -> Result<PersonPage> {
    let (offset, limit) = page;
    let ordering = match sort {
        PeopleSort::Name => "sort_key,person_id",
        PeopleSort::Confirmed => "confirmed_faces DESC,sort_key,person_id",
        PeopleSort::Review => "suggested_faces DESC,sort_key,person_id",
        PeopleSort::RecentlyChanged => "updated_at DESC,person_id",
    };
    let sql = format!(
        "SELECT person_id,display_name,confirmed_faces,suggested_faces
         FROM people_scope_summary WHERE build_epoch=?1 AND scope_key=?2
         ORDER BY {ordering} LIMIT ?3 OFFSET ?4"
    );
    let mut statement = connection.prepare(&sql)?;
    let mut items = statement
        .query_map(
            params![epoch, scope, limit.clamp(1, 500) as i64, offset as i64],
            |row| {
                Ok(PersonSummary {
                    id: row.get(0)?,
                    name: row.get(1)?,
                    confirmed_faces: row.get(2)?,
                    suggested_faces: row.get(3)?,
                    representative_face_id: None,
                    representative: None,
                })
            },
        )?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    let mut representative = connection.prepare(
        "SELECT representative.face_id,representative.relative_path,representative.left,
                representative.top,representative.right,representative.bottom
         FROM people_scope_representatives representative
         JOIN asset_scope_ancestors ancestor ON ancestor.build_epoch=representative.build_epoch
              AND ancestor.asset_id=representative.asset_id AND ancestor.scope_key=?2
         WHERE representative.build_epoch=?1 AND representative.scope_key=X''
           AND representative.person_id=?3
         ORDER BY representative.rank LIMIT 1 OFFSET ?4",
    )?;
    for person in &mut items {
        let rank = representative_offsets.get(&person.id).copied().unwrap_or(0) as i64;
        person.representative = representative
            .query_row(params![epoch, scope, person.id, rank], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, Vec<u8>>(1)?,
                    [row.get(2)?, row.get(3)?, row.get(4)?, row.get(5)?],
                ))
            })
            .optional()?
            .map(|(face_id, path, bounds)| -> Result<_> {
                Ok(RepresentativeFace {
                    face_id,
                    path: root.join(decode_relative_path(&path)?),
                    bounds,
                })
            })
            .transpose()?;
        person.representative_face_id = person.representative.as_ref().map(|face| face.face_id);
    }
    let (active, confirmed, review, pending, errors): (i64, i64, i64, i64, i64) = connection
        .query_row(
            "SELECT active_faces,confirmed_faces,suggested_rows,analysis_pending,analysis_errors
         FROM people_scope_totals WHERE build_epoch=?1 AND scope_key=?2",
            params![epoch, scope],
            |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                ))
            },
        )
        .optional()?
        .unwrap_or((0, 0, 0, 0, 0));
    let total: i64 = connection.query_row(
        "SELECT count(*) FROM people_scope_summary WHERE build_epoch=?1 AND scope_key=?2",
        params![epoch, scope],
        |row| row.get(0),
    )?;
    Ok(PersonPage {
        items,
        total: total as u64,
        offset,
        unnamed_faces: active.saturating_sub(confirmed) as u64,
        review_faces: review as u64,
        analysis_pending: pending as u64,
        analysis_errors: errors as u64,
    })
}
#[cfg(test)]
pub fn list_person_faces_page(
    root: &Path,
    person_id: i64,
    subtree: Option<&Path>,
    offset: u64,
    limit: usize,
) -> Result<FaceReviewPage> {
    list_person_faces_page_cancellable(
        root,
        person_id,
        subtree,
        offset,
        limit,
        Arc::new(AtomicBool::new(false)),
    )
}

pub fn list_person_faces_page_cancellable(
    root: &Path,
    person_id: i64,
    subtree: Option<&Path>,
    offset: u64,
    limit: usize,
    cancelled: Arc<AtomicBool>,
) -> Result<FaceReviewPage> {
    let path = super::catalog_path(root);
    if !path.is_file() {
        return Ok(FaceReviewPage {
            items: Vec::new(),
            total: 0,
            offset,
        });
    }
    let connection = Connection::open_with_flags(
        path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )?;
    connection.busy_timeout(Duration::from_secs(5))?;
    let query_cancelled = cancelled.clone();
    connection.progress_handler(1_000, Some(move || query_cancelled.load(Ordering::Acquire)))?;
    super::validate_schema(&connection)?;
    let prefix = subtree
        .map(|subtree| {
            subtree
                .strip_prefix(root)
                .with_context(|| {
                    format!(
                        "scope {} is outside library {}",
                        subtree.display(),
                        root.display()
                    )
                })
                .and_then(scope_key)
        })
        .transpose()?;
    let limit = limit.clamp(1, 500);
    let path_matches = "(?2 IS NULL OR substr(asset.scope_key, 1, length(?2))=?2)";
    let count_sql = format!(
        "SELECT count(*) FROM effective_face_assignments assignment
         JOIN faces face ON face.id=assignment.face_id
         JOIN assets asset ON asset.id=face.asset_id
         WHERE assignment.person_id=?1 AND face.retired=0 AND face.ignored=0 AND face.locally_ignored=0
           AND {path_matches}"
    );
    let total: i64 =
        connection.query_row(&count_sql, params![person_id, prefix], |row| row.get(0))?;
    let sql = format!(
        "SELECT face.id, assignment.person_id, person.display_name, asset.relative_path,
                face.left, face.top, face.right, face.bottom, face.origin,
                assignment.kind, assignment.provenance, assignment.confidence
         FROM effective_face_assignments assignment
         JOIN people person ON person.id=assignment.person_id
         JOIN faces face ON face.id=assignment.face_id
         JOIN assets asset ON asset.id=face.asset_id
         WHERE assignment.person_id=?1 AND face.retired=0 AND face.ignored=0 AND face.locally_ignored=0
           AND {path_matches}
         ORDER BY (assignment.kind='suggested') DESC,
                  coalesce(assignment.confidence, 0) DESC, face.id
         LIMIT ?3 OFFSET ?4"
    );
    let mut query = connection.prepare(&sql)?;
    let rows = query.query_map(
        params![person_id, prefix, limit as i64, offset as i64],
        |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, Vec<u8>>(3)?,
                [row.get(4)?, row.get(5)?, row.get(6)?, row.get(7)?],
                row.get::<_, String>(8)?,
                row.get::<_, String>(9)?,
                row.get::<_, String>(10)?,
                row.get::<_, Option<f32>>(11)?,
            ))
        },
    )?;
    let mut items = Vec::new();
    for row in rows {
        let (
            face_id,
            person_id,
            person_name,
            relative,
            bounds,
            origin,
            kind,
            provenance,
            confidence,
        ) = row?;
        items.push(FaceReviewItem {
            face_id,
            person_id,
            person_name: Some(person_name),
            path: root.join(decode_relative_path(&relative)?),
            bounds,
            origin: FaceOrigin::parse(&origin),
            kind: FaceAssignmentKind::parse(&kind)?,
            provenance,
            confidence,
        });
    }
    Ok(FaceReviewPage {
        items,
        total: total
            .try_into()
            .context("SQLite returned a negative count")?,
        offset,
    })
}

pub fn apply_decisions(root: &Path, batch: &DecisionBatch) -> Result<DecisionBatchResult> {
    if batch.id.trim().is_empty() || batch.id.len() > 128 {
        bail!("decision batch id must contain 1 to 128 characters");
    }
    if batch.commands.is_empty() || batch.commands.len() > MAX_DECISIONS_PER_BATCH {
        bail!("decision batch must contain 1 to {MAX_DECISIONS_PER_BATCH} commands");
    }
    let mut keys = BTreeSet::new();
    for command in &batch.commands {
        let expected_person = command.action != DecisionAction::Ignore;
        if command.person_id.is_some() != expected_person {
            bail!(
                "face {} has an invalid person for its action",
                command.face_id
            );
        }
        if !keys.insert((command.face_id, command.person_id)) {
            bail!(
                "face {} occurs more than once in the batch",
                command.face_id
            );
        }
    }

    let mut connection = writable_connection(root)?;
    let transaction = connection.transaction()?;
    super::begin_people_summary_bulk(&transaction)?;
    let already_applied: bool = transaction.query_row(
        "SELECT EXISTS(SELECT 1 FROM face_decision_journal WHERE batch_id=?1)",
        [&batch.id],
        |row| row.get(0),
    )?;
    if already_applied {
        let stored_count: i64 = transaction.query_row(
            "SELECT count(*) FROM face_decision_journal WHERE batch_id=?1",
            [&batch.id],
            |row| row.get(0),
        )?;
        let mut same_commands = stored_count == batch.commands.len() as i64;
        for command in &batch.commands {
            same_commands &= transaction.query_row(
                "SELECT EXISTS(SELECT 1 FROM face_decision_journal
                 WHERE batch_id=?1 AND face_id=?2 AND person_id IS ?3 AND next_state=?4)",
                params![
                    batch.id,
                    command.face_id,
                    command.person_id,
                    decision_action_name(command.action)
                ],
                |row| row.get::<_, bool>(0),
            )?;
        }
        if !same_commands {
            bail!(
                "decision batch id {} was reused with different commands",
                batch.id
            );
        }
        return Ok(DecisionBatchResult {
            batch_id: batch.id.clone(),
            applied: false,
            faces: batch
                .commands
                .iter()
                .map(|command| command.face_id)
                .collect(),
        });
    }
    for command in &batch.commands {
        validate_decision(&transaction, command)?;
    }
    let mut recognition_people = BTreeSet::new();
    let mut recognition_faces = BTreeSet::new();
    for command in &batch.commands {
        let previous = local_face_state(&transaction, command.face_id)?;
        let mut affected_people = face_person_ids(&transaction, command.face_id)?;
        recognition_people.extend(affected_people.iter().copied());
        recognition_faces.insert(command.face_id);
        apply_decision(&transaction, command)?;
        affected_people.extend(face_person_ids(&transaction, command.face_id)?);
        recognition_people.extend(affected_people.iter().copied());
        if let Some(person_id) = command.person_id {
            affected_people.insert(person_id);
        }
        touch_people(&transaction, &affected_people)?;
        transaction.execute(
            "INSERT INTO face_decision_journal(batch_id, face_id, person_id, previous_state,
                                               next_state, created_at)
             VALUES(?1, ?2, ?3, ?4, ?5, datetime('now'))",
            params![
                batch.id,
                command.face_id,
                command.person_id,
                serde_json::to_string(&previous)?,
                decision_action_name(command.action)
            ],
        )?;
    }
    refresh_recognition_models(
        &transaction,
        &recognition_people.into_iter().collect::<Vec<_>>(),
        &recognition_faces.into_iter().collect::<Vec<_>>(),
    )?;
    super::finish_people_summary_bulk(&transaction)?;
    transaction.commit()?;
    Ok(DecisionBatchResult {
        batch_id: batch.id.clone(),
        applied: true,
        faces: batch
            .commands
            .iter()
            .map(|command| command.face_id)
            .collect(),
    })
}

fn refresh_recognition_models(
    connection: &Connection,
    person_ids: &[i64],
    face_ids: &[i64],
) -> Result<()> {
    super::recognition::enqueue_refresh_in_connection(connection, person_ids, face_ids)
}

pub fn undo_most_recent_decision_batch(root: &Path) -> Result<Option<String>> {
    let mut connection = writable_connection(root)?;
    let transaction = connection.transaction()?;
    super::begin_people_summary_bulk(&transaction)?;
    let batch_id: Option<String> = transaction
        .query_row(
            "SELECT batch_id FROM face_decision_journal
             WHERE previous_state LIKE '{%' ORDER BY id DESC LIMIT 1",
            [],
            |row| row.get(0),
        )
        .optional()?;
    let Some(batch_id) = batch_id else {
        return Ok(None);
    };
    let operation_id = batch_id
        .rsplit_once("/chunk/")
        .filter(|(_, chunk)| !chunk.is_empty() && chunk.bytes().all(|byte| byte.is_ascii_digit()))
        .map(|(operation, _)| operation.to_owned());
    let mut statement = transaction.prepare(
        "SELECT face_id, person_id, previous_state FROM face_decision_journal
         WHERE batch_id=?1 OR (
             ?2 IS NOT NULL
             AND substr(batch_id, 1, length(?2)+7)=?2 || '/chunk/'
             AND substr(batch_id, length(?2)+8)<>''
             AND substr(batch_id, length(?2)+8) NOT GLOB '*[^0-9]*')
         ORDER BY id DESC",
    )?;
    let states = statement
        .query_map(params![batch_id, operation_id], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, Option<i64>>(1)?,
                row.get::<_, String>(2)?,
            ))
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    drop(statement);
    let mut recognition_people = BTreeSet::new();
    let mut recognition_faces = BTreeSet::new();
    for (face_id, person_id, encoded) in states {
        let mut affected_people = face_person_ids(&transaction, face_id)?;
        recognition_people.extend(affected_people.iter().copied());
        recognition_faces.insert(face_id);
        let state: LocalFaceState = serde_json::from_str(&encoded)
            .with_context(|| format!("decision batch {batch_id} cannot be undone"))?;
        restore_local_face_state(&transaction, face_id, &state)?;
        affected_people.extend(face_person_ids(&transaction, face_id)?);
        recognition_people.extend(affected_people.iter().copied());
        affected_people.extend(state.representatives.iter().map(|entry| entry.0));
        if let Some(person_id) = person_id {
            affected_people.insert(person_id);
        }
        touch_people(&transaction, &affected_people)?;
    }
    refresh_recognition_models(
        &transaction,
        &recognition_people.into_iter().collect::<Vec<_>>(),
        &recognition_faces.into_iter().collect::<Vec<_>>(),
    )?;
    transaction.execute(
        "DELETE FROM face_decision_journal
         WHERE batch_id=?1 OR (
             ?2 IS NOT NULL
             AND substr(batch_id, 1, length(?2)+7)=?2 || '/chunk/'
             AND substr(batch_id, length(?2)+8)<>''
             AND substr(batch_id, length(?2)+8) NOT GLOB '*[^0-9]*')",
        params![batch_id, operation_id],
    )?;
    super::finish_people_summary_bulk(&transaction)?;
    transaction.commit()?;
    Ok(Some(operation_id.unwrap_or(batch_id)))
}

fn validate_decision(transaction: &Transaction<'_>, command: &DecisionCommand) -> Result<()> {
    let face_exists: bool = transaction.query_row(
        "SELECT EXISTS(SELECT 1 FROM faces WHERE id=?1 AND retired=0)",
        [command.face_id],
        |row| row.get(0),
    )?;
    if !face_exists {
        bail!("face {} does not exist", command.face_id);
    }
    if let Some(person_id) = command.person_id {
        let person_exists: bool = transaction.query_row(
            "SELECT EXISTS(SELECT 1 FROM people WHERE id=?1)",
            [person_id],
            |row| row.get(0),
        )?;
        if !person_exists {
            bail!("person {person_id} does not exist");
        }
    }
    Ok(())
}

fn local_face_state(transaction: &Transaction<'_>, face_id: i64) -> Result<LocalFaceState> {
    let ignored = transaction.query_row(
        "SELECT locally_ignored != 0 FROM faces WHERE id=?1",
        [face_id],
        |row| row.get(0),
    )?;
    let mut query = transaction.prepare(
        "SELECT person_id, kind, confidence FROM face_assignments
         WHERE face_id=?1 AND provenance='rekesz' ORDER BY person_id, kind",
    )?;
    let assignments = query
        .query_map([face_id], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))?
        .collect::<rusqlite::Result<_>>()?;
    let mut representatives_query = transaction.prepare(
        "SELECT id, representative_face_id, representative_is_manual != 0
         FROM people WHERE representative_face_id=?1 ORDER BY id",
    )?;
    let representatives = representatives_query
        .query_map([face_id], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))?
        .collect::<rusqlite::Result<_>>()?;
    Ok(LocalFaceState {
        ignored,
        assignments,
        representatives,
    })
}

fn restore_local_face_state(
    transaction: &Transaction<'_>,
    face_id: i64,
    state: &LocalFaceState,
) -> Result<()> {
    transaction.execute(
        "DELETE FROM face_assignments WHERE face_id=?1 AND provenance='rekesz'",
        [face_id],
    )?;
    for (person_id, kind, confidence) in &state.assignments {
        transaction.execute(
            "INSERT INTO face_assignments(face_id, person_id, kind, provenance, confidence, updated_at)
             VALUES(?1, ?2, ?3, 'rekesz', ?4, datetime('now'))",
            params![face_id, person_id, kind, confidence],
        )?;
    }
    transaction.execute(
        "UPDATE faces SET locally_ignored=?2, updated_at=datetime('now') WHERE id=?1",
        params![face_id, state.ignored],
    )?;
    for (person_id, representative_face_id, manual) in &state.representatives {
        transaction.execute(
            "UPDATE people SET representative_face_id=?2, representative_is_manual=?3
             WHERE id=?1",
            params![person_id, representative_face_id, manual],
        )?;
    }
    invalidate_unavailable_representatives(transaction, face_id)?;
    Ok(())
}

fn face_person_ids(transaction: &Transaction<'_>, face_id: i64) -> Result<BTreeSet<i64>> {
    let mut query = transaction.prepare(
        "SELECT person_id FROM face_assignments WHERE face_id=?1
         UNION SELECT id FROM people WHERE representative_face_id=?1",
    )?;
    Ok(query
        .query_map([face_id], |row| row.get(0))?
        .collect::<rusqlite::Result<_>>()?)
}

fn touch_people(transaction: &Transaction<'_>, people: &BTreeSet<i64>) -> Result<()> {
    for person_id in people {
        transaction.execute(
            "UPDATE people SET updated_at=datetime('now') WHERE id=?1",
            [person_id],
        )?;
    }
    Ok(())
}

fn invalidate_unavailable_representatives(
    transaction: &Transaction<'_>,
    face_id: i64,
) -> Result<()> {
    transaction.execute(
        "UPDATE people SET representative_face_id=NULL, representative_is_manual=0
         WHERE representative_face_id=?1 AND NOT EXISTS (
             SELECT 1 FROM effective_face_assignments assignment
             JOIN faces face ON face.id=assignment.face_id
             WHERE assignment.person_id=people.id AND assignment.face_id=?1
               AND assignment.kind='confirmed' AND face.ignored=0
               AND face.locally_ignored=0 AND face.retired=0)",
        [face_id],
    )?;
    Ok(())
}

fn apply_decision(transaction: &Transaction<'_>, command: &DecisionCommand) -> Result<()> {
    match command.action {
        DecisionAction::Confirm => {
            let person_id = command
                .person_id
                .context("confirmation requires a person")?;
            transaction.execute(
                "DELETE FROM face_assignments WHERE face_id=?1 AND provenance='rekesz'
                 AND (kind='confirmed' OR (kind='rejected' AND person_id=?2))",
                params![command.face_id, person_id],
            )?;
            transaction.execute(
                "INSERT INTO face_assignments(face_id, person_id, kind, provenance, confidence, updated_at)
                 VALUES(?1, ?2, 'confirmed', 'rekesz', NULL, datetime('now'))",
                params![command.face_id, person_id],
            )?;
        }
        DecisionAction::Reject => {
            let person_id = command.person_id.context("rejection requires a person")?;
            transaction.execute(
                "DELETE FROM face_assignments WHERE face_id=?1 AND person_id=?2
                 AND provenance='rekesz' AND kind IN ('suggested', 'confirmed', 'rejected')",
                params![command.face_id, person_id],
            )?;
            transaction.execute(
                "INSERT INTO face_assignments(face_id, person_id, kind, provenance, confidence, updated_at)
                 VALUES(?1, ?2, 'rejected', 'rekesz', NULL, datetime('now'))",
                params![command.face_id, person_id],
            )?;
        }
        DecisionAction::Ignore => {
            transaction.execute(
                "UPDATE faces SET locally_ignored=1, updated_at=datetime('now') WHERE id=?1",
                [command.face_id],
            )?;
        }
    }
    if let Some(person_id) = command.person_id {
        transaction.execute(
            "UPDATE people SET updated_at=datetime('now') WHERE id=?1",
            [person_id],
        )?;
    }
    invalidate_unavailable_representatives(transaction, command.face_id)?;
    Ok(())
}

fn decision_action_name(action: DecisionAction) -> &'static str {
    match action {
        DecisionAction::Confirm => "confirmed",
        DecisionAction::Reject => "rejected",
        DecisionAction::Ignore => "ignored",
    }
}

#[cfg(test)]
pub fn confirm(root: &Path, face_id: i64, person_id: i64) -> Result<()> {
    let mut connection = writable_connection(root)?;
    let transaction = connection.transaction()?;
    let previous_state: Option<String> = transaction
        .query_row(
            "SELECT kind FROM effective_face_assignments WHERE face_id=?1 AND person_id=?2",
            params![face_id, person_id],
            |row| row.get(0),
        )
        .optional()?;
    transaction.execute(
        "DELETE FROM face_assignments WHERE face_id=?1 AND provenance='rekesz' AND kind='confirmed'",
        [face_id],
    )?;
    transaction.execute(
        "DELETE FROM face_assignments WHERE face_id=?1 AND person_id=?2 \
         AND (kind='suggested' OR (kind='rejected' AND provenance='rekesz'))",
        params![face_id, person_id],
    )?;
    transaction.execute(
        "INSERT INTO face_assignments(face_id, person_id, kind, provenance, confidence, updated_at) \
         VALUES(?1, ?2, 'confirmed', 'rekesz', NULL, datetime('now')) \
         ON CONFLICT(face_id, person_id, kind, provenance) DO UPDATE SET updated_at=excluded.updated_at",
        params![face_id, person_id],
    )?;
    transaction.execute(
        "INSERT INTO face_decision_journal(batch_id, face_id, person_id, previous_state, next_state, created_at) \
         VALUES(lower(hex(randomblob(16))), ?1, ?2, ?3, 'confirmed', datetime('now'))",
        params![face_id, person_id, previous_state],
    )?;
    transaction.execute(
        "UPDATE people SET updated_at=datetime('now') WHERE id=?1",
        [person_id],
    )?;
    transaction.commit()?;
    Ok(())
}

#[cfg(test)]
pub fn reject(root: &Path, face_id: i64, person_id: i64) -> Result<()> {
    let mut connection = writable_connection(root)?;
    let transaction = connection.transaction()?;
    let previous_state: Option<String> = transaction
        .query_row(
            "SELECT kind FROM effective_face_assignments WHERE face_id=?1 AND person_id=?2",
            params![face_id, person_id],
            |row| row.get(0),
        )
        .optional()?;
    transaction.execute(
        "DELETE FROM face_assignments WHERE face_id=?1 AND person_id=?2
         AND provenance='rekesz' AND kind IN ('suggested', 'confirmed')",
        params![face_id, person_id],
    )?;
    transaction.execute(
        "INSERT INTO face_assignments(face_id, person_id, kind, provenance, confidence, updated_at) \
         VALUES(?1, ?2, 'rejected', 'rekesz', NULL, datetime('now')) \
         ON CONFLICT(face_id, person_id, kind, provenance) DO UPDATE SET updated_at=excluded.updated_at",
        params![face_id, person_id],
    )?;
    transaction.execute(
        "INSERT INTO face_decision_journal(batch_id, face_id, person_id, previous_state, next_state, created_at) \
         VALUES(lower(hex(randomblob(16))), ?1, ?2, ?3, 'rejected', datetime('now'))",
        params![face_id, person_id, previous_state],
    )?;
    transaction.execute(
        "UPDATE people SET updated_at=datetime('now') WHERE id=?1",
        [person_id],
    )?;
    transaction.commit()?;
    Ok(())
}

#[cfg(test)]
pub fn ignore_face(root: &Path, face_id: i64) -> Result<()> {
    let mut connection = writable_connection(root)?;
    let transaction = connection.transaction()?;
    transaction.execute(
        "UPDATE people SET updated_at=datetime('now')
         WHERE id IN (SELECT person_id FROM effective_face_assignments WHERE face_id=?1)",
        [face_id],
    )?;
    let changed = transaction.execute(
        "UPDATE faces SET locally_ignored=1, updated_at=datetime('now') WHERE id=?1",
        [face_id],
    )?;
    if changed == 0 {
        anyhow::bail!("face {face_id} does not exist");
    }
    transaction.commit()?;
    Ok(())
}

pub fn rename_person(root: &Path, person_id: i64, name: &str) -> Result<()> {
    let name = validated_person_name(name)?;
    let connection = writable_connection(root)?;
    let changed = connection.execute(
        "UPDATE people SET display_name=?2, locally_renamed=1, updated_at=datetime('now')
         WHERE id=?1",
        params![person_id, name],
    )?;
    if changed == 0 {
        bail!("person {person_id} does not exist");
    }
    Ok(())
}

pub fn create_person_from_faces(root: &Path, name: &str, face_ids: &[i64]) -> Result<i64> {
    let name = validated_person_name(name)?;
    let face_ids = validated_face_selection(face_ids)?;
    let mut connection = writable_connection(root)?;
    let transaction = connection.transaction()?;
    validate_faces(&transaction, &face_ids)?;
    transaction.execute(
        "INSERT INTO people(display_name, locally_renamed, created_at, updated_at)
         VALUES(?1, 1, datetime('now'), datetime('now'))",
        [name],
    )?;
    let person_id = transaction.last_insert_rowid();
    for face_id in &face_ids {
        apply_decision(
            &transaction,
            &DecisionCommand {
                face_id: *face_id,
                person_id: Some(person_id),
                action: DecisionAction::Confirm,
            },
        )?;
    }
    refresh_recognition_models(&transaction, &[person_id], &face_ids)?;
    transaction.commit()?;
    Ok(person_id)
}

pub fn assign_faces_to_person(root: &Path, person_id: i64, face_ids: &[i64]) -> Result<()> {
    let face_ids = validated_face_selection(face_ids)?;
    let mut connection = writable_connection(root)?;
    let transaction = connection.transaction()?;
    validate_people(&transaction, &[person_id])?;
    validate_faces(&transaction, &face_ids)?;
    for face_id in &face_ids {
        apply_decision(
            &transaction,
            &DecisionCommand {
                face_id: *face_id,
                person_id: Some(person_id),
                action: DecisionAction::Confirm,
            },
        )?;
    }
    refresh_recognition_models(&transaction, &[person_id], &face_ids)?;
    transaction.commit()?;
    Ok(())
}

pub fn preview_people_merge(
    root: &Path,
    source_person_id: i64,
    target_person_id: i64,
) -> Result<MergePeoplePreview> {
    if source_person_id == target_person_id {
        bail!("cannot merge a person into itself");
    }
    let mut connection = writable_connection(root)?;
    let transaction = connection.transaction()?;
    validate_people(&transaction, &[source_person_id, target_person_id])?;
    let confirmed = |person_id| -> Result<u64> {
        let value: i64 = transaction.query_row(
            "SELECT count(*) FROM effective_face_assignments
             WHERE person_id=?1 AND kind='confirmed'",
            [person_id],
            |row| row.get(0),
        )?;
        value.try_into().context("negative confirmation count")
    };
    let count: i64 = transaction.query_row(
        "SELECT count(*) FROM face_assignments source
         JOIN face_assignments target ON target.face_id=source.face_id
              AND target.person_id=?2 AND target.kind=source.kind
              AND target.provenance=source.provenance
         WHERE source.person_id=?1",
        params![source_person_id, target_person_id],
        |row| row.get(0),
    )?;
    let confirmation_deduplications: i64 = transaction.query_row(
        "SELECT count(*) FROM effective_face_assignments source
         JOIN effective_face_assignments target ON target.face_id=source.face_id
              AND target.person_id=?2 AND target.kind='confirmed'
         WHERE source.person_id=?1 AND source.kind='confirmed'",
        params![source_person_id, target_person_id],
        |row| row.get(0),
    )?;
    let representatives = transaction.query_row(
        "SELECT source.representative_face_id, target.representative_face_id
         FROM people source JOIN people target ON target.id=?2 WHERE source.id=?1",
        params![source_person_id, target_person_id],
        |row| Ok((row.get::<_, Option<i64>>(0)?, row.get::<_, Option<i64>>(1)?)),
    )?;
    let external_links: i64 = transaction.query_row(
        "SELECT count(*) FROM external_people WHERE person_id=?1",
        [source_person_id],
        |row| row.get(0),
    )?;
    let source_confirmed_faces = confirmed(source_person_id)?;
    let target_confirmed_faces = confirmed(target_person_id)?;
    transaction.commit()?;
    Ok(MergePeoplePreview {
        source_person_id,
        target_person_id,
        source_confirmed_faces,
        target_confirmed_faces,
        confirmation_deduplications: confirmation_deduplications
            .try_into()
            .context("negative confirmation deduplication count")?,
        assignment_deduplications: count
            .try_into()
            .context("negative assignment deduplication count")?,
        source_representative_face_id: representatives.0,
        target_representative_face_id: representatives.1,
        resulting_representative_face_id: representatives.1.or(representatives.0),
        source_representative_will_be_discarded: representatives.0.is_some()
            && representatives.1.is_some(),
        external_links_moved: external_links
            .try_into()
            .context("negative external link count")?,
    })
}

pub fn merge_people(root: &Path, source_person_id: i64, target_person_id: i64) -> Result<()> {
    if source_person_id == target_person_id {
        bail!("cannot merge a person into itself");
    }
    let mut connection = writable_connection(root)?;
    let transaction = connection.transaction()?;
    validate_people(&transaction, &[source_person_id, target_person_id])?;
    let affected_faces = {
        let mut query = transaction
            .prepare("SELECT DISTINCT face_id FROM face_assignments WHERE person_id IN (?1, ?2)")?;
        query
            .query_map(params![source_person_id, target_person_id], |row| {
                row.get(0)
            })?
            .collect::<rusqlite::Result<Vec<i64>>>()?
    };
    transaction.execute(
        "INSERT OR IGNORE INTO face_assignments(face_id, person_id, kind, provenance,
                                                confidence, updated_at)
         SELECT face_id, ?2, kind, provenance, confidence, datetime('now')
         FROM face_assignments WHERE person_id=?1",
        params![source_person_id, target_person_id],
    )?;
    transaction.execute(
        "DELETE FROM face_assignments WHERE person_id=?1",
        [source_person_id],
    )?;
    transaction.execute(
        "UPDATE OR IGNORE external_people SET person_id=?2 WHERE person_id=?1",
        params![source_person_id, target_person_id],
    )?;
    transaction.execute(
        "DELETE FROM external_people WHERE person_id=?1",
        [source_person_id],
    )?;
    transaction.execute(
        "UPDATE people SET
             representative_is_manual=CASE WHEN representative_face_id IS NULL
                 THEN (SELECT representative_is_manual FROM people WHERE id=?1)
                 ELSE representative_is_manual END,
             representative_face_id=coalesce(representative_face_id,
                 (SELECT representative_face_id FROM people WHERE id=?1)),
             updated_at=datetime('now') WHERE id=?2",
        params![source_person_id, target_person_id],
    )?;
    transaction.execute("DELETE FROM people WHERE id=?1", [source_person_id])?;
    refresh_recognition_models(
        &transaction,
        &[source_person_id, target_person_id],
        &affected_faces,
    )?;
    transaction.commit()?;
    Ok(())
}

pub fn move_confirmed_faces(
    root: &Path,
    source_person_id: i64,
    target_person_id: i64,
    face_ids: &[i64],
) -> Result<()> {
    if source_person_id == target_person_id {
        bail!("source and target people must differ");
    }
    let face_ids = validated_face_selection(face_ids)?;
    let mut connection = writable_connection(root)?;
    let transaction = connection.transaction()?;
    validate_people(&transaction, &[source_person_id, target_person_id])?;
    validate_faces(&transaction, &face_ids)?;
    for face_id in &face_ids {
        let confirmed: bool = transaction.query_row(
            "SELECT EXISTS(SELECT 1 FROM effective_face_assignments
                           WHERE face_id=?1 AND person_id=?2 AND kind='confirmed')",
            params![face_id, source_person_id],
            |row| row.get(0),
        )?;
        if !confirmed {
            bail!("face {face_id} is not confirmed for person {source_person_id}");
        }
        apply_decision(
            &transaction,
            &DecisionCommand {
                face_id: *face_id,
                person_id: Some(target_person_id),
                action: DecisionAction::Confirm,
            },
        )?;
    }
    transaction.execute(
        "UPDATE people SET updated_at=datetime('now') WHERE id IN (?1, ?2)",
        params![source_person_id, target_person_id],
    )?;
    refresh_recognition_models(
        &transaction,
        &[source_person_id, target_person_id],
        &face_ids,
    )?;
    transaction.commit()?;
    Ok(())
}

pub fn choose_person_representative(root: &Path, person_id: i64, face_id: i64) -> Result<()> {
    let connection = writable_connection(root)?;
    let confirmed: bool = connection.query_row(
        "SELECT EXISTS(SELECT 1 FROM effective_face_assignments assignment
                       JOIN faces face ON face.id=assignment.face_id
                       WHERE assignment.person_id=?1 AND assignment.face_id=?2
                         AND assignment.kind='confirmed' AND face.ignored=0
                         AND face.locally_ignored=0)",
        params![person_id, face_id],
        |row| row.get(0),
    )?;
    if !confirmed {
        bail!("face {face_id} is not an available confirmation for person {person_id}");
    }
    connection.execute(
        "UPDATE people SET representative_face_id=?2, representative_is_manual=1,
                           updated_at=datetime('now') WHERE id=?1",
        params![person_id, face_id],
    )?;
    Ok(())
}

fn validated_person_name(name: &str) -> Result<&str> {
    let name = name.trim();
    if name.is_empty() || name.chars().count() > 200 {
        bail!("person name must contain 1 to 200 characters");
    }
    Ok(name)
}

fn validated_face_selection(face_ids: &[i64]) -> Result<Vec<i64>> {
    if face_ids.is_empty() || face_ids.len() > MAX_DECISIONS_PER_BATCH {
        bail!("face selection must contain 1 to {MAX_DECISIONS_PER_BATCH} faces");
    }
    let unique = face_ids.iter().copied().collect::<BTreeSet<_>>();
    if unique.len() != face_ids.len() {
        bail!("face selection contains duplicates");
    }
    Ok(unique.into_iter().collect())
}

fn validate_faces(transaction: &Transaction<'_>, face_ids: &[i64]) -> Result<()> {
    for face_id in face_ids {
        let exists: bool = transaction.query_row(
            "SELECT EXISTS(SELECT 1 FROM faces WHERE id=?1 AND retired=0)",
            [face_id],
            |row| row.get(0),
        )?;
        if !exists {
            bail!("face {face_id} does not exist");
        }
    }
    Ok(())
}

fn validate_people(transaction: &Transaction<'_>, person_ids: &[i64]) -> Result<()> {
    for person_id in person_ids {
        let exists: bool = transaction.query_row(
            "SELECT EXISTS(SELECT 1 FROM people WHERE id=?1)",
            [person_id],
            |row| row.get(0),
        )?;
        if !exists {
            bail!("person {person_id} does not exist");
        }
    }
    Ok(())
}

pub fn list_review_queue_page_cancellable(
    root: &Path,
    subtree: Option<&Path>,
    offset: u64,
    limit: usize,
    cancelled: Arc<AtomicBool>,
) -> Result<FaceReviewPage> {
    list_special_face_page(root, subtree, offset, limit, cancelled, false)
}

pub fn list_unnamed_faces_page_cancellable(
    root: &Path,
    subtree: Option<&Path>,
    offset: u64,
    limit: usize,
    cancelled: Arc<AtomicBool>,
) -> Result<FaceReviewPage> {
    list_special_face_page(root, subtree, offset, limit, cancelled, true)
}

fn list_special_face_page(
    root: &Path,
    subtree: Option<&Path>,
    offset: u64,
    limit: usize,
    cancelled: Arc<AtomicBool>,
    unnamed: bool,
) -> Result<FaceReviewPage> {
    list_special_face_page_inner(root, subtree, offset, limit, cancelled, unnamed, false)
}

fn list_special_face_page_inner(
    root: &Path,
    subtree: Option<&Path>,
    offset: u64,
    limit: usize,
    cancelled: Arc<AtomicBool>,
    unnamed: bool,
    deny_authoritative_reads: bool,
) -> Result<FaceReviewPage> {
    let path = super::catalog_path(root);
    if !path.is_file() {
        return Ok(FaceReviewPage {
            items: Vec::new(),
            total: 0,
            offset,
        });
    }
    let connection = Connection::open_with_flags(
        path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )?;
    #[cfg(test)]
    if deny_authoritative_reads {
        use rusqlite::hooks::{AuthAction, AuthContext, Authorization};
        connection.authorizer(Some(|context: AuthContext<'_>| match context.action {
            AuthAction::Read { table_name, .. }
                if table_name == "faces"
                    || table_name == "face_assignments"
                    || table_name == "recognition_suggestions" =>
            {
                Authorization::Deny
            }
            _ => Authorization::Allow,
        }))?;
    }
    #[cfg(not(test))]
    let _ = deny_authoritative_reads;
    connection.busy_timeout(Duration::from_secs(5))?;
    let query_cancelled = cancelled.clone();
    connection.progress_handler(1_000, Some(move || query_cancelled.load(Ordering::Acquire)))?;
    super::validate_schema(&connection)?;
    let prefix = subtree
        .map(|subtree| {
            subtree
                .strip_prefix(root)
                .context("People scope is outside the library")
                .and_then(scope_key)
        })
        .transpose()?;
    let fresh_epoch: Option<i64> = connection
        .query_row(
            "SELECT published_epoch FROM people_summary_epoch WHERE id=1
         AND published_epoch=source_epoch AND published_epoch>0",
            [],
            |row| row.get(0),
        )
        .optional()?;
    if let Some(epoch) = fresh_epoch {
        let scope = prefix.as_deref().unwrap_or_default();
        let queue_kind = if unnamed { "unnamed" } else { "review" };
        let total: i64 = connection.query_row(
            "SELECT count(*) FROM people_scope_faces WHERE build_epoch=?1 AND scope_key=?2 AND queue_kind=?3",
            params![epoch,scope,queue_kind], |row| row.get(0),
        )?;
        let mut statement = connection.prepare(
            "SELECT face_id,person_id,person_name,assignment_kind,provenance,confidence,relative_path,
                    left,top,right,bottom,origin FROM people_scope_faces
             WHERE build_epoch=?1 AND scope_key=?2 AND queue_kind=?3
             ORDER BY face_id,person_id LIMIT ?4 OFFSET ?5",
        )?;
        let rows = statement.query_map(
            params![
                epoch,
                scope,
                queue_kind,
                limit.clamp(1, 500) as i64,
                offset as i64
            ],
            |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, Option<String>>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, String>(4)?,
                    row.get::<_, Option<f32>>(5)?,
                    row.get::<_, Vec<u8>>(6)?,
                    [row.get(7)?, row.get(8)?, row.get(9)?, row.get(10)?],
                    row.get::<_, String>(11)?,
                ))
            },
        )?;
        let mut items = Vec::new();
        for row in rows {
            let (
                face_id,
                person_id,
                person_name,
                kind,
                provenance,
                confidence,
                path,
                bounds,
                origin,
            ) = row?;
            items.push(FaceReviewItem {
                face_id,
                person_id,
                person_name,
                path: root.join(decode_relative_path(&path)?),
                bounds,
                origin: FaceOrigin::parse(&origin),
                kind: FaceAssignmentKind::parse(&kind)?,
                provenance,
                confidence,
            });
        }
        return Ok(FaceReviewPage {
            items,
            total: total as u64,
            offset,
        });
    }
    let limit = limit.clamp(1, 500);
    let path_matches = "(?1 IS NULL OR substr(asset.scope_key, 1, length(?1))=?1)";
    let predicate = if unnamed {
        "NOT EXISTS (SELECT 1 FROM effective_face_assignments confirmed
                     WHERE confirmed.face_id=face.id AND confirmed.kind='confirmed')"
    } else {
        "assignment.kind='suggested'"
    };
    let joins = if unnamed {
        ""
    } else {
        "JOIN effective_face_assignments assignment ON assignment.face_id=face.id"
    };
    let count_sql = format!(
        "SELECT count(*) FROM faces face {joins}
         JOIN assets asset ON asset.id=face.asset_id
         WHERE face.retired=0 AND face.ignored=0 AND face.locally_ignored=0
           AND {path_matches} AND {predicate}"
    );
    let total: i64 = connection.query_row(&count_sql, [prefix.as_deref()], |row| row.get(0))?;
    let assignment_columns = if unnamed {
        "0, NULL, 'unassigned', face.origin, NULL"
    } else {
        "assignment.person_id, person.display_name, assignment.kind, assignment.provenance, assignment.confidence"
    };
    let people_join = if unnamed {
        ""
    } else {
        "JOIN people person ON person.id=assignment.person_id"
    };
    let ordering = if unnamed {
        "face.id"
    } else {
        "face.id, assignment.person_id"
    };
    let sql = format!(
        "SELECT face.id, {assignment_columns}, asset.relative_path,
                face.left, face.top, face.right, face.bottom, face.origin
         FROM faces face {joins} {people_join}
         JOIN assets asset ON asset.id=face.asset_id
         WHERE face.retired=0 AND face.ignored=0 AND face.locally_ignored=0
           AND {path_matches} AND {predicate}
         ORDER BY {ordering} LIMIT ?2 OFFSET ?3"
    );
    let mut query = connection.prepare(&sql)?;
    let rows = query.query_map(params![prefix, limit as i64, offset as i64], |row| {
        Ok((
            row.get::<_, i64>(0)?,
            row.get::<_, i64>(1)?,
            row.get::<_, Option<String>>(2)?,
            row.get::<_, String>(3)?,
            row.get::<_, String>(4)?,
            row.get::<_, Option<f32>>(5)?,
            row.get::<_, Vec<u8>>(6)?,
            [row.get(7)?, row.get(8)?, row.get(9)?, row.get(10)?],
            row.get::<_, String>(11)?,
        ))
    })?;
    let mut items = Vec::new();
    for row in rows {
        let (
            face_id,
            person_id,
            person_name,
            kind,
            provenance,
            confidence,
            relative,
            bounds,
            origin,
        ) = row?;
        items.push(FaceReviewItem {
            face_id,
            person_id,
            person_name,
            path: root.join(decode_relative_path(&relative)?),
            bounds,
            origin: FaceOrigin::parse(&origin),
            kind: FaceAssignmentKind::parse(&kind)?,
            provenance,
            confidence,
        });
    }
    Ok(FaceReviewPage {
        items,
        total: total.try_into().context("negative face count")?,
        offset,
    })
}

#[cfg(test)]
pub fn list_fresh_special_pages_with_authoritative_reads_denied(root: &Path) -> Result<()> {
    for subtree in [None, Some(root.join("profile/2024/08/30/event"))] {
        for unnamed in [false, true] {
            for offset in [0, 30] {
                list_special_face_page_inner(
                    root,
                    subtree.as_deref(),
                    offset,
                    30,
                    Arc::new(AtomicBool::new(false)),
                    unnamed,
                    true,
                )?;
            }
        }
    }
    Ok(())
}

pub fn diagnostics(root: &Path) -> Result<PeopleDiagnostics> {
    let connection = writable_connection(root)?;
    let schema_version = connection.query_row("PRAGMA user_version", [], |row| row.get(0))?;
    let count = |sql: &str| -> Result<u64> {
        let value: i64 = connection.query_row(sql, [], |row| row.get(0))?;
        value
            .try_into()
            .context("SQLite returned a negative diagnostics count")
    };
    let detector_model_id = connection
        .query_row(
            "SELECT detector_model_id FROM face_analysis_state
             WHERE detector_model_id IS NOT NULL ORDER BY updated_at DESC LIMIT 1",
            [],
            |row| row.get(0),
        )
        .optional()?
        .unwrap_or_else(|| "yunet-2023mar".to_owned());
    let recognizer_model_id = connection
        .query_row(
            "SELECT recognizer_model_id FROM face_analysis_state
             WHERE recognizer_model_id IS NOT NULL ORDER BY updated_at DESC LIMIT 1",
            [],
            |row| row.get(0),
        )
        .optional()?
        .unwrap_or_else(|| "sface-2021dec".to_owned());
    let calibration = super::recognition::production_calibration_status(&connection)?;
    Ok(PeopleDiagnostics {
        schema_version,
        detector_model_id,
        recognizer_model_id,
        recognition_policy_active: calibration.active,
        recognition_policy_version: calibration.policy_version,
        recognition_policy_digest: calibration.artifact_digest,
        indexed_assets: count("SELECT count(*) FROM face_analysis_state WHERE status='complete'")?,
        active_faces: count("SELECT count(*) FROM faces WHERE retired=0")?,
        analysis_pending: count(
            "SELECT count(*) FROM face_analysis_state WHERE status IN ('pending','running')",
        )?,
        analysis_errors: count("SELECT count(*) FROM face_analysis_state WHERE status='error'")?,
        reconciliation_conflicts: count(
            "SELECT coalesce(sum(conflict_count), 0) FROM face_reconciliation_conflicts",
        )?,
        recognition_refresh_errors: count(
            "SELECT count(*) FROM recognition_refresh_queue WHERE status='error'",
        )?,
        stale_artifacts: count(
            "SELECT (SELECT count(*) FROM face_embeddings embedding
                     JOIN faces face ON face.id=embedding.face_id WHERE face.retired!=0)
                  + (SELECT count(*) FROM recognition_suggestions suggestion
                     JOIN faces face ON face.id=suggestion.face_id WHERE face.retired!=0)",
        )?,
    })
}

pub fn retry_failed_analysis(root: &Path) -> Result<RetryFailedResult> {
    let mut connection = writable_connection(root)?;
    let transaction = connection.transaction()?;
    let analysis_jobs = transaction.execute(
        "UPDATE face_analysis_state SET status='pending', attempts=0, worker_id=NULL,
             lease_expires_at=NULL, retry_after=NULL, error=NULL, failure_class=NULL,
             updated_at=datetime('now')
         WHERE status='error'",
        [],
    )?;
    let recognition_jobs = transaction.execute(
        "UPDATE recognition_refresh_queue SET status='pending', attempts=0, worker_id=NULL,
             lease_expires_at=NULL, retry_after=NULL, error=NULL, updated_at=datetime('now')
         WHERE status='error'",
        [],
    )?;
    transaction.commit()?;
    Ok(RetryFailedResult {
        analysis_jobs: analysis_jobs as u64,
        recognition_jobs: recognition_jobs as u64,
    })
}

fn writable_connection(root: &Path) -> Result<Connection> {
    let path = super::catalog_path(root);
    if !path.is_file() {
        bail!("Rekesz catalog does not exist");
    }
    let connection = Connection::open(path)?;
    // Interactive People edits should wait out bounded maintenance writes
    // instead of surfacing a transient "database is locked" error.
    connection.busy_timeout(Duration::from_secs(30))?;
    super::validate_schema(&connection)?;
    connection.execute("PRAGMA foreign_keys=ON", [])?;
    Ok(connection)
}

fn asset_id(transaction: &Transaction<'_>, relative_path: &[u8]) -> Result<i64> {
    let decoded = decode_relative_path(relative_path)?;
    transaction.execute(
        "INSERT INTO assets(relative_path, scope_key) VALUES(?1, ?2) \
         ON CONFLICT(relative_path) DO UPDATE SET scope_key=excluded.scope_key",
        params![relative_path, scope_key(&decoded)?],
    )?;
    Ok(transaction.query_row(
        "SELECT id FROM assets WHERE relative_path=?1",
        [relative_path],
        |row| row.get(0),
    )?)
}

fn table_exists(connection: &Connection, name: &str) -> Result<bool> {
    Ok(connection.query_row(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name=?1)",
        [name],
        |row| row.get(0),
    )?)
}

fn count(connection: &Connection, sql: &str) -> Result<u64> {
    let value: i64 = connection.query_row(sql, [], |row| row.get(0))?;
    value.try_into().context("SQLite returned a negative count")
}

fn displayed_bounds_to_source(bounds: [f32; 4], orientation: &str) -> Option<[f32; 4]> {
    let corner = |name| match name {
        b'A' => Some([0.0_f32, 0.0_f32]),
        b'B' => Some([1.0, 0.0]),
        b'C' => Some([1.0, 1.0]),
        b'D' => Some([0.0, 1.0]),
        _ => None,
    };
    let bytes = orientation.as_bytes();
    if bytes.len() != 2 {
        return None;
    }
    let origin = corner(bytes[0])?;
    let right = corner(bytes[1])?;
    let u = [right[0] - origin[0], right[1] - origin[1]];
    if (u[0].abs() + u[1].abs() - 1.0).abs() > f32::EPSILON {
        return None;
    }
    let candidates = [[-u[1], u[0]], [u[1], -u[0]]];
    let v = candidates.into_iter().find(|candidate| {
        let point = [origin[0] + candidate[0], origin[1] + candidate[1]];
        (0.0..=1.0).contains(&point[0]) && (0.0..=1.0).contains(&point[1])
    })?;
    // Lightroom's rectangle is expressed in the displayed orientation. The
    // basis above maps source coordinates to that display, so use its inverse.
    let source = |x: f32, y: f32| {
        let delta = [x - origin[0], y - origin[1]];
        [
            delta[0] * u[0] + delta[1] * u[1],
            delta[0] * v[0] + delta[1] * v[1],
        ]
    };
    let points = [
        source(bounds[0], bounds[1]),
        source(bounds[2], bounds[1]),
        source(bounds[2], bounds[3]),
        source(bounds[0], bounds[3]),
    ];
    Some([
        points
            .iter()
            .map(|point| point[0])
            .fold(f32::INFINITY, f32::min),
        points
            .iter()
            .map(|point| point[1])
            .fold(f32::INFINITY, f32::min),
        points
            .iter()
            .map(|point| point[0])
            .fold(f32::NEG_INFINITY, f32::max),
        points
            .iter()
            .map(|point| point[1])
            .fold(f32::NEG_INFINITY, f32::max),
    ])
}

fn valid_bounds(bounds: [f32; 4]) -> bool {
    bounds.iter().all(|value| value.is_finite())
        && bounds[0] >= 0.0
        && bounds[1] >= 0.0
        && bounds[2] <= 1.0
        && bounds[3] <= 1.0
        && bounds[0] < bounds[2]
        && bounds[1] < bounds[3]
}

#[cfg(test)]
#[path = "people_tests.rs"]
mod tests;
