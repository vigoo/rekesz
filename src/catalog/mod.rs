use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fs,
    io::Read,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, anyhow, bail};
use rusqlite::{Connection, OpenFlags, OptionalExtension, Transaction, params};
use serde::{Deserialize, Serialize};

pub(crate) mod people;
pub(crate) mod recognition;
mod recognition_policy;

pub const CATALOG_DIR: &str = ".rekesz";
const CATALOG_FILE: &str = "catalog.sqlite3";
const SCHEMA_VERSION: i64 = 6;
const CONTENT_IDENTITY_BUFFER_BYTES: usize = 64 * 1024;
const SQLITE_APPLICATION_ID: i64 = 0x5049_4353;
pub const RENDERER_MODEL_VERSION: u32 = 6;

#[cfg(test)]
thread_local! {
    static LIGHTROOM_TRANSACTION_BEGUN_HOOK: std::cell::RefCell<Option<Box<dyn FnOnce()>>> =
        const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
fn install_lightroom_transaction_begun_hook(hook: impl FnOnce() + 'static) {
    LIGHTROOM_TRANSACTION_BEGUN_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

#[cfg(test)]
fn notify_lightroom_transaction_begun() {
    LIGHTROOM_TRANSACTION_BEGUN_HOOK.with(|slot| {
        if let Some(hook) = slot.borrow_mut().take() {
            hook();
        }
    });
}

#[derive(Clone, Copy)]
struct ImportCapabilities {
    basic_tone: bool,
    color: bool,
    white_balance: bool,
    tone_curve: bool,
    crop: bool,
    detail: bool,
    presence: bool,
    monochrome: bool,
    finishing: bool,
    geometry: bool,
    healing: bool,
    grain: bool,
}

impl ImportCapabilities {
    #[cfg(test)]
    const ALL: Self = Self {
        basic_tone: true,
        color: true,
        white_balance: true,
        tone_curve: true,
        crop: true,
        detail: true,
        presence: true,
        monochrome: true,
        finishing: true,
        geometry: true,
        healing: true,
        grain: true,
    };

    const PHASE_B: Self = Self {
        basic_tone: true,
        color: true,
        white_balance: true,
        tone_curve: true,
        crop: true,
        detail: true,
        presence: true,
        monochrome: true,
        finishing: true,
        geometry: true,
        healing: true,
        grain: true,
    };
}

const CURRENT_IMPORT_CAPABILITIES: ImportCapabilities = ImportCapabilities::PHASE_B;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct DevelopRecipe {
    pub exposure_ev: Option<f32>,
    pub contrast: Option<f32>,
    pub highlights: Option<f32>,
    pub shadows: Option<f32>,
    pub whites: Option<f32>,
    pub blacks: Option<f32>,
    pub saturation: Option<f32>,
    pub vibrance: Option<f32>,
    pub clarity: Option<f32>,
    pub white_balance: Option<WhiteBalance>,
    pub white_balance_increment: Option<WhiteBalanceIncrement>,
    pub tone_curve: Option<Vec<[f32; 2]>>,
    pub tone_curve_red: Option<Vec<[f32; 2]>>,
    pub tone_curve_green: Option<Vec<[f32; 2]>>,
    pub tone_curve_blue: Option<Vec<[f32; 2]>>,
    pub crop: Option<Crop>,
    pub detail: Option<DetailRecipe>,
    pub texture: Option<f32>,
    pub dehaze: Option<f32>,
    pub monochrome: Option<MonochromeRecipe>,
    pub vignette: Option<VignetteRecipe>,
    pub parametric_curve: Option<[f32; 4]>,
    pub upright_homography: Option<[f32; 9]>,
    pub lens_distortion: Option<f32>,
    pub heal_spots: Vec<HealSpot>,
    pub grain: Option<GrainRecipe>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct WhiteBalance {
    pub temperature: f32,
    pub tint: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct WhiteBalanceIncrement {
    pub temperature: f32,
    pub tint: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Crop {
    pub left: f32,
    pub top: f32,
    pub right: f32,
    pub bottom: f32,
    pub angle_degrees: f32,
    pub constrain_aspect: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct DetailRecipe {
    pub sharpening: f32,
    pub sharpen_radius: f32,
    pub sharpen_detail: f32,
    pub edge_masking: f32,
    pub luminance_noise_reduction: f32,
    pub luminance_detail: f32,
    pub luminance_contrast: f32,
    pub color_noise_reduction: f32,
    pub color_detail: f32,
    pub color_smoothness: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct MonochromeRecipe {
    pub mixer: [f32; 8],
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct VignetteRecipe {
    pub amount: f32,
    pub midpoint: f32,
    pub feather: f32,
    pub roundness: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct HealSpot {
    pub kind: HealKind,
    pub destination: [f32; 2],
    pub source: [f32; 2],
    pub radius: f32,
    pub opacity: f32,
    pub feather: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum HealKind {
    Heal,
    Clone,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct GrainRecipe {
    pub amount: f32,
    pub size: f32,
    pub roughness: f32,
    pub seed: u64,
}

impl DevelopRecipe {
    pub fn is_empty(&self) -> bool {
        self == &Self::default()
    }

    pub fn digest(&self) -> Result<[u8; 32]> {
        let encoded = serde_json::to_vec(self)?;
        Ok(*blake3::hash(&encoded).as_bytes())
    }

    pub fn validate(&self) -> Result<()> {
        if self.white_balance.is_some() && self.white_balance_increment.is_some() {
            bail!("absolute and incremental white balance are mutually exclusive");
        }
        for (name, value, minimum, maximum) in [
            ("exposure", self.exposure_ev, -5.0, 5.0),
            ("contrast", self.contrast, -100.0, 100.0),
            ("highlights", self.highlights, -100.0, 100.0),
            ("shadows", self.shadows, -100.0, 100.0),
            ("whites", self.whites, -100.0, 100.0),
            ("blacks", self.blacks, -100.0, 100.0),
            ("saturation", self.saturation, -100.0, 100.0),
            ("vibrance", self.vibrance, -100.0, 100.0),
            ("clarity", self.clarity, -100.0, 100.0),
            ("texture", self.texture, -100.0, 100.0),
            ("dehaze", self.dehaze, -100.0, 100.0),
        ] {
            if value.is_some_and(|value| !value.is_finite() || value < minimum || value > maximum) {
                bail!("invalid {name} value");
            }
        }
        if let Some(wb) = self.white_balance
            && (!wb.temperature.is_finite()
                || !(2_000.0..=50_000.0).contains(&wb.temperature)
                || !wb.tint.is_finite()
                || !(-150.0..=150.0).contains(&wb.tint))
        {
            bail!("invalid white balance");
        }
        if let Some(wb) = self.white_balance_increment
            && (!wb.temperature.is_finite()
                || !(-100.0..=100.0).contains(&wb.temperature)
                || !wb.tint.is_finite()
                || !(-100.0..=100.0).contains(&wb.tint))
        {
            bail!("invalid incremental white balance");
        }
        for curve in [
            &self.tone_curve,
            &self.tone_curve_red,
            &self.tone_curve_green,
            &self.tone_curve_blue,
        ] {
            if let Some(curve) = curve
                && (curve.len() < 2
                    || curve.iter().flatten().any(|value| !value.is_finite())
                    || curve.windows(2).any(|points| points[0][0] >= points[1][0]))
            {
                bail!("invalid tone curve");
            }
        }
        if let Some(crop) = self.crop
            && (!crop.left.is_finite()
                || !crop.top.is_finite()
                || !crop.right.is_finite()
                || !crop.bottom.is_finite()
                || !crop.angle_degrees.is_finite()
                || crop.left < 0.0
                || crop.top < 0.0
                || crop.right > 1.0
                || crop.bottom > 1.0
                || crop.left >= crop.right
                || crop.top >= crop.bottom)
        {
            bail!("invalid crop");
        }
        if let Some(detail) = self.detail
            && (!detail.sharpening.is_finite()
                || !(0.0..=150.0).contains(&detail.sharpening)
                || !detail.sharpen_radius.is_finite()
                || !(0.5..=3.0).contains(&detail.sharpen_radius)
                || [
                    detail.sharpen_detail,
                    detail.edge_masking,
                    detail.luminance_noise_reduction,
                    detail.luminance_detail,
                    detail.luminance_contrast,
                    detail.color_noise_reduction,
                    detail.color_detail,
                    detail.color_smoothness,
                ]
                .iter()
                .any(|value| !value.is_finite() || !(0.0..=100.0).contains(value)))
        {
            bail!("invalid detail controls");
        }
        if let Some(monochrome) = self.monochrome
            && monochrome
                .mixer
                .iter()
                .any(|value| !value.is_finite() || !(-100.0..=100.0).contains(value))
        {
            bail!("invalid monochrome mixer");
        }
        if let Some(vignette) = self.vignette
            && (!vignette.amount.is_finite()
                || !(-100.0..=100.0).contains(&vignette.amount)
                || !vignette.midpoint.is_finite()
                || !(0.0..=100.0).contains(&vignette.midpoint)
                || !vignette.feather.is_finite()
                || !(0.0..=100.0).contains(&vignette.feather)
                || !vignette.roundness.is_finite()
                || !(-100.0..=100.0).contains(&vignette.roundness))
        {
            bail!("invalid vignette");
        }
        if self.parametric_curve.is_some_and(|curve| {
            curve
                .iter()
                .any(|value| !value.is_finite() || !(-100.0..=100.0).contains(value))
        }) {
            bail!("invalid parametric curve");
        }
        if self.upright_homography.is_some_and(|matrix| {
            matrix.iter().any(|value| !value.is_finite())
                || homography_determinant(matrix).abs() < 1e-8
        }) {
            bail!("invalid upright homography");
        }
        if self
            .lens_distortion
            .is_some_and(|value| !value.is_finite() || !(-100.0..=100.0).contains(&value))
        {
            bail!("invalid lens distortion");
        }
        if self.heal_spots.iter().any(|spot| {
            spot.destination
                .iter()
                .chain(spot.source.iter())
                .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
                || !spot.radius.is_finite()
                || !(0.0..=1.0).contains(&spot.radius)
                || !spot.opacity.is_finite()
                || !(0.0..=1.0).contains(&spot.opacity)
                || !spot.feather.is_finite()
                || !(0.0..=1.0).contains(&spot.feather)
        }) {
            bail!("invalid heal spot");
        }
        if let Some(grain) = self.grain
            && [grain.amount, grain.size, grain.roughness]
                .iter()
                .any(|value| !value.is_finite() || !(0.0..=100.0).contains(value))
        {
            bail!("invalid grain");
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct ImportReport {
    pub catalog_path: PathBuf,
    pub lightroom_images: u64,
    pub matched_files: u64,
    pub imported_recipes: u64,
    pub empty_supported_recipes: u64,
    pub skipped_virtual_copies: u64,
    pub missing_or_outside_root: u64,
    pub conflicting_sources: u64,
    pub unsupported_property_occurrences: BTreeMap<String, u64>,
    pub fidelity_warnings: BTreeMap<String, u64>,
    pub lightroom_faces: u64,
    pub imported_faces: u64,
    pub confirmed_face_assignments: u64,
    pub suggested_face_assignments: u64,
    pub unnamed_faces: u64,
    pub imported_face_clusters: u64,
    pub skipped_virtual_copy_faces: u64,
    pub orphan_face_assignments: u64,
    pub incomplete_lightroom_embeddings: u64,
    pub face_import_warnings: BTreeMap<String, u64>,
}

#[derive(Clone, Debug)]
pub struct RecipeEntry {
    pub recipe: DevelopRecipe,
    pub digest: [u8; 32],
}

pub fn load_recipes(root: &Path) -> Result<HashMap<PathBuf, RecipeEntry>> {
    let path = catalog_path(root);
    if !path.is_file() {
        return Ok(HashMap::new());
    }
    let connection = Connection::open_with_flags(
        path,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )?;
    validate_schema(&connection)?;
    let mut query = connection.prepare(
        "SELECT assets.relative_path, recipe.recipe_json, recipe.recipe_digest \
         FROM develop_recipes recipe JOIN assets ON assets.id=recipe.asset_id",
    )?;
    let rows = query.query_map([], |row| {
        Ok((
            row.get::<_, Vec<u8>>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, Vec<u8>>(2)?,
        ))
    })?;
    let mut recipes = HashMap::new();
    for row in rows {
        let (relative, json, digest) = row?;
        let relative_path = decode_relative_path(&relative)?;
        let digest: [u8; 32] = digest.try_into().map_err(|_| {
            anyhow!(
                "invalid recipe digest length for {}",
                relative_path.display()
            )
        })?;
        let recipe: DevelopRecipe = serde_json::from_str(&json)?;
        recipe.validate()?;
        let computed = recipe.digest()?;
        if computed != digest {
            bail!("recipe digest mismatch for {}", relative_path.display());
        }
        recipes.insert(root.join(relative_path), RecipeEntry { recipe, digest });
    }
    Ok(recipes)
}

pub fn load_hidden_subtrees(root: &Path) -> Result<HashSet<PathBuf>> {
    let path = catalog_path(root);
    if !path.is_file() {
        return Ok(HashSet::new());
    }
    let connection = Connection::open_with_flags(
        path,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )?;
    validate_schema(&connection)?;
    let table_exists: bool = connection.query_row(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='hidden_subtrees')",
        [],
        |row| row.get(0),
    )?;
    if !table_exists {
        return Ok(HashSet::new());
    }
    let mut query = connection.prepare("SELECT relative_path FROM hidden_subtrees")?;
    let rows = query.query_map([], |row| row.get::<_, Vec<u8>>(0))?;
    rows.map(|row| Ok(root.join(decode_relative_path(&row?)?)))
        .collect()
}

/// Validate and migrate an existing owned catalog before runtime readers or
/// maintenance workers are started. An unopened library without a catalog is
/// left untouched; the first operation that needs persistence may create it.
pub(crate) fn prepare_existing_catalog(root: &Path) -> Result<()> {
    let path = catalog_path(root);
    if !path.is_file() {
        return Ok(());
    }
    let mut connection = Connection::open(path)?;
    initialize_schema(&mut connection)
}

pub fn save_hidden_subtrees(root: &Path, hidden_subtrees: &HashSet<PathBuf>) -> Result<()> {
    let directory = root.join(CATALOG_DIR);
    fs::create_dir_all(&directory)?;
    let mut connection = Connection::open(catalog_path(root))?;
    initialize_schema(&mut connection)?;
    connection.execute(
        "CREATE TABLE IF NOT EXISTS hidden_subtrees(\
             relative_path BLOB PRIMARY KEY,\
             hidden_at TEXT NOT NULL\
         )",
        [],
    )?;
    let transaction = connection.transaction()?;
    transaction.execute("DELETE FROM hidden_subtrees", [])?;
    for subtree in hidden_subtrees {
        let relative = subtree.strip_prefix(root).with_context(|| {
            format!(
                "hidden subtree {} is outside library {}",
                subtree.display(),
                root.display()
            )
        })?;
        if relative.as_os_str().is_empty() {
            bail!("the library root cannot be hidden");
        }
        let encoded = encode_relative_path(relative)?;
        transaction.execute(
            "INSERT INTO hidden_subtrees(relative_path, hidden_at) VALUES (?1, datetime('now'))",
            [&encoded],
        )?;
    }
    transaction.commit()?;
    Ok(())
}

#[cfg(test)]
pub fn analyze_lightroom(root: &Path, lightroom_catalog: &Path) -> Result<ImportReport> {
    import_lightroom_inner(root, lightroom_catalog, false, CURRENT_IMPORT_CAPABILITIES)
}

pub fn import_lightroom(root: &Path, lightroom_catalog: &Path) -> Result<ImportReport> {
    import_lightroom_inner(root, lightroom_catalog, true, CURRENT_IMPORT_CAPABILITIES)
}

fn import_lightroom_inner(
    root: &Path,
    lightroom_catalog: &Path,
    apply: bool,
    capabilities: ImportCapabilities,
) -> Result<ImportReport> {
    let root = root
        .canonicalize()
        .with_context(|| format!("cannot resolve opened root {}", root.display()))?;
    let destination = catalog_path(&root);
    let lightroom = open_lightroom_immutable(lightroom_catalog)?;
    let source_namespace = lightroom_source_namespace(&lightroom, lightroom_catalog)?;
    let mut report = ImportReport {
        catalog_path: destination.clone(),
        ..ImportReport::default()
    };

    let mut planned = BTreeMap::<Vec<u8>, DevelopRecipe>::new();
    let mut query = lightroom.prepare(
        "SELECT root.absolutePath, folder.pathFromRoot, file.baseName, file.extension, \
                image.masterImage, coalesce(settings.processVersion, ''), \
                coalesce(settings.text, '') \
         FROM Adobe_images image \
         JOIN AgLibraryFile file ON file.id_local = image.rootFile \
         JOIN AgLibraryFolder folder ON folder.id_local = file.folder \
         JOIN AgLibraryRootFolder root ON root.id_local = folder.rootFolder \
         LEFT JOIN Adobe_imageDevelopSettings settings ON settings.image = image.id_local",
    )?;
    let mut rows = query.query([])?;
    while let Some(row) = rows.next()? {
        report.lightroom_images += 1;
        if row.get::<_, Option<i64>>(4)?.is_some() {
            report.skipped_virtual_copies += 1;
            continue;
        }
        let absolute_root: String = row.get(0)?;
        let folder: String = row.get(1)?;
        let base: String = row.get(2)?;
        let extension: String = row.get(3)?;
        let Some(relative) = resolve_relative_source(
            &root,
            &absolute_root,
            &folder,
            &format!("{base}.{extension}"),
        ) else {
            report.missing_or_outside_root += 1;
            continue;
        };
        report.matched_files += 1;
        let process_version: String = row.get(5)?;
        let text: String = row.get(6)?;
        let parsed = if text.trim().is_empty() {
            BTreeMap::new()
        } else {
            Parser::new(&text).parse_assignment()?
        };
        let (mut recipe, unsupported, warnings) =
            normalize_recipe(&process_version, &parsed, capabilities);
        for property in unsupported {
            *report
                .unsupported_property_occurrences
                .entry(property)
                .or_default() += 1;
        }
        for warning in warnings {
            *report.fidelity_warnings.entry(warning).or_default() += 1;
        }
        let relative_key = encode_relative_path(&relative)?;
        if let Some(grain) = &mut recipe.grain {
            let digest = blake3::hash(&relative_key);
            grain.seed = u64::from_le_bytes(digest.as_bytes()[..8].try_into().unwrap());
        }
        if recipe.is_empty() {
            report.empty_supported_recipes += 1;
        } else {
            if let Some(previous) = planned.insert(relative_key.clone(), recipe.clone())
                && previous != recipe
            {
                report.conflicting_sources += 1;
                planned.remove(&relative_key);
            }
            report.imported_recipes += 1;
        }
    }
    drop(rows);
    drop(query);
    let people =
        people::collect_lightroom_people(&root, &lightroom, &source_namespace, &mut report)?;

    if !apply {
        return Ok(report);
    }
    if report.conflicting_sources > 0 {
        bail!("refusing to import recipes with conflicting Lightroom source rows");
    }
    if destination.exists() {
        let mut rekesz = Connection::open(&destination)
            .with_context(|| format!("cannot open Rekesz catalog {}", destination.display()))?;
        initialize_schema(&mut rekesz)?;
        write_lightroom_import(&mut rekesz, &planned, &people)?;
    } else if planned.is_empty() && people.faces.is_empty() {
        // There is nothing to persist. In particular, do not create a Rekesz
        // catalog merely to record that every Lightroom image is unedited.
        return Ok(report);
    } else {
        let temporary = tempfile::NamedTempFile::new_in(&root)?;
        let temporary = temporary.into_temp_path();
        {
            let mut rekesz = Connection::open(&temporary)?;
            initialize_schema(&mut rekesz)?;
            write_lightroom_import(&mut rekesz, &planned, &people)?;
            rekesz.execute_batch("PRAGMA wal_checkpoint(TRUNCATE); PRAGMA journal_mode=DELETE;")?;
        }
        let parent = destination.parent().expect("catalog path has a parent");
        fs::create_dir_all(parent).with_context(|| {
            format!(
                "cannot create Rekesz catalog directory {}",
                parent.display()
            )
        })?;
        temporary
            .persist(&destination)
            .map_err(|error| error.error)
            .with_context(|| format!("cannot install Rekesz catalog {}", destination.display()))?;
    }
    Ok(report)
}

fn begin_people_summary_bulk(connection: &Connection) -> Result<()> {
    connection.execute(
        "UPDATE people_summary_bulk_state SET dirty=CASE WHEN depth=0 THEN 0 ELSE dirty END,
         depth=depth+1 WHERE id=1",
        [],
    )?;
    Ok(())
}

fn finish_people_summary_bulk(connection: &Connection) -> Result<()> {
    connection.execute_batch(
        "UPDATE people_summary_epoch SET source_epoch=source_epoch+1 WHERE id=1
           AND (SELECT depth=1 AND dirty=1 FROM people_summary_bulk_state WHERE id=1);
         INSERT INTO people_summary_rebuild_queue(id,status,requested_epoch,attempts,worker_id,
           lease_expires_at,retry_after,error,updated_at)
         SELECT 1,'pending',source_epoch,0,NULL,NULL,NULL,NULL,datetime('now')
           FROM people_summary_epoch
          WHERE id=1 AND (SELECT depth=1 AND dirty=1 FROM people_summary_bulk_state WHERE id=1)
         ON CONFLICT(id) DO UPDATE SET
           status=CASE WHEN people_summary_rebuild_queue.status='running'
                            AND people_summary_rebuild_queue.lease_expires_at>=datetime('now')
                       THEN 'running' ELSE 'pending' END,
           requested_epoch=excluded.requested_epoch,
           worker_id=CASE WHEN people_summary_rebuild_queue.status='running'
                               AND people_summary_rebuild_queue.lease_expires_at>=datetime('now')
                          THEN people_summary_rebuild_queue.worker_id ELSE NULL END,
           claimed_epoch=CASE WHEN people_summary_rebuild_queue.status='running'
                                   AND people_summary_rebuild_queue.lease_expires_at>=datetime('now')
                              THEN people_summary_rebuild_queue.claimed_epoch ELSE NULL END,
           lease_expires_at=CASE WHEN people_summary_rebuild_queue.status='running'
                                      AND people_summary_rebuild_queue.lease_expires_at>=datetime('now')
                                 THEN people_summary_rebuild_queue.lease_expires_at ELSE NULL END,
           retry_after=NULL,error=NULL,updated_at=excluded.updated_at;
         UPDATE people_summary_bulk_state SET depth=depth-1,
           dirty=CASE WHEN depth=1 THEN 0 ELSE dirty END WHERE id=1 AND depth>0;",
    )?;
    Ok(())
}

fn write_lightroom_import(
    rekesz: &mut Connection,
    planned: &BTreeMap<Vec<u8>, DevelopRecipe>,
    people: &people::PeopleImport,
) -> Result<()> {
    let transaction = rekesz.transaction()?;
    begin_people_summary_bulk(&transaction)?;
    #[cfg(test)]
    notify_lightroom_transaction_begun();
    transaction.execute("DELETE FROM develop_recipes", [])?;
    for (relative_key, recipe) in planned {
        let relative_path = decode_relative_path(relative_key)?;
        transaction.execute(
            "INSERT INTO assets(relative_path, scope_key) VALUES (?1, ?2) \
             ON CONFLICT(relative_path) DO UPDATE SET scope_key=excluded.scope_key",
            params![relative_key, scope_key(&relative_path)?],
        )?;
        let asset_id: i64 = transaction.query_row(
            "SELECT id FROM assets WHERE relative_path=?1",
            [&relative_key],
            |row| row.get(0),
        )?;
        let json = serde_json::to_string(&recipe)?;
        recipe.validate()?;
        let digest = recipe.digest()?;
        transaction.execute(
            "INSERT INTO develop_recipes(asset_id, recipe_json, recipe_digest, imported_at) \
             VALUES (?1, ?2, ?3, datetime('now'))",
            params![asset_id, json, digest.as_slice()],
        )?;
    }
    people::write_people_import(&transaction, people)?;
    transaction.execute(
        "DELETE FROM assets WHERE id NOT IN (SELECT asset_id FROM develop_recipes) \
         AND id NOT IN (SELECT asset_id FROM faces) \
         AND id NOT IN (SELECT asset_id FROM face_analysis_state)",
        [],
    )?;
    transaction.execute(
        "UPDATE catalog_meta SET renderer_model=?1, last_imported_at=datetime('now')",
        [RENDERER_MODEL_VERSION],
    )?;
    finish_people_summary_bulk(&transaction)?;
    transaction.commit()?;
    Ok(())
}

pub(crate) fn catalog_path(root: &Path) -> PathBuf {
    root.join(CATALOG_DIR).join(CATALOG_FILE)
}

pub(crate) fn scope_key(path: &Path) -> Result<Vec<u8>> {
    let mut key = Vec::new();
    for component in path.components() {
        let std::path::Component::Normal(value) = component else {
            bail!("scope path must be relative and contain only normal components");
        };
        let encoded = encode_relative_path(Path::new(value))?;
        let length = u32::try_from(encoded.len()).context("path component is too large")?;
        key.extend_from_slice(&length.to_be_bytes());
        key.extend_from_slice(&encoded);
    }
    Ok(key)
}

fn lightroom_source_namespace(connection: &Connection, catalog_path: &Path) -> Result<String> {
    let has_variables: bool = connection.query_row(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='Adobe_variables')",
        [],
        |row| row.get(0),
    )?;
    if has_variables {
        let provider_id: Option<Vec<u8>> = connection
            .query_row(
                "SELECT CAST(value AS BLOB) FROM Adobe_variables
                 WHERE name='Adobe_storeProviderID' AND value IS NOT NULL LIMIT 1",
                [],
                |row| row.get(0),
            )
            .optional()?;
        if let Some(provider_id) = provider_id.filter(|value| !value.is_empty()) {
            return Ok(blake3::hash(&provider_id).to_hex().to_string());
        }
        // Adobe_storeProviderID is absent from some real Lightroom Classic
        // catalogs. Adobe_variables.id_global is Lightroom's persistent object
        // identity and survives catalog moves, so it is a safe namespace
        // fallback without depending on a mutable file path.
        let global_id: Option<Vec<u8>> = connection
            .query_row(
                "SELECT CAST(id_global AS BLOB) FROM Adobe_variables
                 WHERE id_global IS NOT NULL AND length(CAST(id_global AS BLOB))>0
                 ORDER BY id_local LIMIT 1",
                [],
                |row| row.get(0),
            )
            .optional()?;
        if let Some(global_id) = global_id {
            return Ok(blake3::hash(&global_id).to_hex().to_string());
        }
    }

    // Lightroom catalog versions differ considerably in where (and whether)
    // they retain Adobe_storeProviderID. Prefer persistent object identifiers
    // from the catalog before falling back to its location.
    for table in [
        "AgLibraryRootFolder",
        "Adobe_images",
        "AgLibraryFile",
        "AgLibraryKeyword",
    ] {
        let has_global_id: bool = connection.query_row(
            "SELECT EXISTS(SELECT 1 FROM pragma_table_info(?1) WHERE name='id_global')",
            [table],
            |row| row.get(0),
        )?;
        if !has_global_id {
            continue;
        }
        let query = format!(
            "SELECT CAST(id_global AS BLOB) FROM {table} \
             WHERE id_global IS NOT NULL AND length(CAST(id_global AS BLOB))>0 \
             ORDER BY id_local LIMIT 1"
        );
        let global_id: Option<Vec<u8>> = connection
            .query_row(&query, [], |row| row.get(0))
            .optional()?;
        if let Some(global_id) = global_id {
            let mut identity = table.as_bytes().to_vec();
            identity.push(0);
            identity.extend(global_id);
            return Ok(blake3::hash(&identity).to_hex().to_string());
        }
    }

    // Very old or compacted catalogs can contain no persistent global IDs at
    // all. A canonical path still gives repeatable imports of that catalog and
    // is safer than rejecting otherwise valid People data.
    let database_path = lightroom_database_path(catalog_path)?;
    let stable_path = database_path.canonicalize().unwrap_or(database_path);
    Ok(blake3::hash(stable_path.as_os_str().as_encoded_bytes())
        .to_hex()
        .to_string())
}

fn initialize_schema(connection: &mut Connection) -> Result<()> {
    let application_id: i64 =
        connection.query_row("PRAGMA application_id", [], |row| row.get(0))?;
    let tables: i64 = connection.query_row(
        "SELECT count(*) FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'",
        [],
        |row| row.get(0),
    )?;
    if tables > 0 && application_id != SQLITE_APPLICATION_ID {
        bail!("refusing to initialize a database that is not a Rekesz catalog");
    }
    if tables > 0 {
        validate_owned_catalog(connection)?;
        connection.execute_batch("PRAGMA foreign_keys=ON")?;
        let version: i64 = connection.query_row("PRAGMA user_version", [], |row| row.get(0))?;
        match version {
            SCHEMA_VERSION => {}
            1 => {
                backup_catalog_before_migration(connection, version)?;
                migrate_v1_to_v3(connection)?;
                migrate_v3_to_v4(connection)?;
                migrate_v4_to_v5(connection)?;
                migrate_v5_to_v6(connection)?;
            }
            2 => {
                backup_catalog_before_migration(connection, version)?;
                migrate_v2_to_v3(connection)?;
                migrate_v3_to_v4(connection)?;
                migrate_v4_to_v5(connection)?;
                migrate_v5_to_v6(connection)?;
            }
            3 => {
                backup_catalog_before_migration(connection, version)?;
                migrate_v3_to_v4(connection)?;
                migrate_v4_to_v5(connection)?;
                migrate_v5_to_v6(connection)?;
            }
            4 => {
                backup_catalog_before_migration(connection, version)?;
                migrate_v4_to_v5(connection)?;
                migrate_v5_to_v6(connection)?;
            }
            5 => {
                backup_catalog_before_migration(connection, version)?;
                migrate_v5_to_v6(connection)?;
            }
            _ => {
                bail!("unsupported Rekesz catalog schema {version}; expected {SCHEMA_VERSION}")
            }
        }
        connection.execute_batch(asset_inventory_schema())?;
        ensure_analysis_decoder_columns(connection)?;
        connection.execute_batch("PRAGMA foreign_keys=ON; PRAGMA journal_mode=WAL;")?;
        return validate_schema(connection);
    }
    connection.execute_batch(&format!(
        "PRAGMA foreign_keys=ON;
         PRAGMA journal_mode=WAL;
         PRAGMA application_id={SQLITE_APPLICATION_ID};
         CREATE TABLE IF NOT EXISTS catalog_meta(
             id INTEGER PRIMARY KEY CHECK(id=1),
             schema_version INTEGER NOT NULL,
             renderer_model INTEGER NOT NULL,
             created_at TEXT NOT NULL,
             last_imported_at TEXT
         );
         CREATE TABLE IF NOT EXISTS assets(
             id INTEGER PRIMARY KEY,
             relative_path BLOB NOT NULL UNIQUE,
             scope_key BLOB NOT NULL
         );
         CREATE TABLE IF NOT EXISTS develop_recipes(
             asset_id INTEGER PRIMARY KEY REFERENCES assets(id) ON DELETE CASCADE,
             recipe_json TEXT NOT NULL,
             recipe_digest BLOB NOT NULL,
             imported_at TEXT NOT NULL
         );
         CREATE INDEX IF NOT EXISTS assets_scope_key ON assets(scope_key);
         {asset_inventory_schema}
         CREATE TABLE IF NOT EXISTS hidden_subtrees(
             relative_path BLOB PRIMARY KEY,
             hidden_at TEXT NOT NULL
         );
         {people_schema}
         {people_summary_schema}
         INSERT INTO catalog_meta(id, schema_version, renderer_model, created_at)
         SELECT 1, {SCHEMA_VERSION}, {RENDERER_MODEL_VERSION}, datetime('now')
         WHERE NOT EXISTS(SELECT 1 FROM catalog_meta);
         PRAGMA user_version={SCHEMA_VERSION};",
        people_schema = people_schema(),
        people_summary_schema = people_summary_schema(),
        asset_inventory_schema = asset_inventory_schema()
    ))?;
    validate_schema(connection)
}

fn backup_catalog_before_migration(connection: &Connection, source_version: i64) -> Result<()> {
    let database_path: String = connection.query_row(
        "SELECT file FROM pragma_database_list WHERE name='main'",
        [],
        |row| row.get(0),
    )?;
    if database_path.is_empty() {
        bail!("cannot back up an in-memory Rekesz catalog before migration");
    }
    let database_path = PathBuf::from(database_path);
    let file_name = database_path
        .file_name()
        .and_then(|name| name.to_str())
        .context("Rekesz catalog has a non-Unicode or missing file name")?;
    let attempt = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let backup_path = database_path.with_file_name(format!(
        "{file_name}.pre-v{source_version}-{attempt}-backup.sqlite3"
    ));
    let escaped = backup_path.to_string_lossy().replace('\'', "''");
    connection
        .execute_batch(&format!("VACUUM main INTO '{escaped}'"))
        .with_context(|| {
            format!(
                "cannot create pre-migration catalog backup {}",
                backup_path.display()
            )
        })?;
    let backup = Connection::open_with_flags(
        &backup_path,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )?;
    validate_migration_backup(connection, &backup, source_version).with_context(|| {
        format!(
            "pre-migration catalog backup is not recoverable: {}",
            backup_path.display()
        )
    })?;
    Ok(())
}

fn validate_migration_backup(
    source: &Connection,
    backup: &Connection,
    source_version: i64,
) -> Result<()> {
    validate_owned_catalog(backup)?;
    let integrity: String = backup.query_row("PRAGMA integrity_check", [], |row| row.get(0))?;
    if integrity != "ok" {
        bail!("SQLite integrity check failed: {integrity}");
    }
    let backup_version: i64 = backup.query_row("PRAGMA user_version", [], |row| row.get(0))?;
    let metadata_version: i64 = backup.query_row(
        "SELECT schema_version FROM catalog_meta WHERE id=1",
        [],
        |row| row.get(0),
    )?;
    if backup_version != source_version || metadata_version != source_version {
        bail!(
            "backup schema markers disagree: expected {source_version}, got {backup_version}/{metadata_version}"
        );
    }
    for table in [
        "people",
        "faces",
        "face_assignments",
        "face_decision_journal",
    ] {
        let present: bool = source.query_row(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name=?1)",
            [table],
            |row| row.get(0),
        )?;
        if present {
            let source_count: i64 =
                source.query_row(&format!("SELECT count(*) FROM {table}"), [], |row| {
                    row.get(0)
                })?;
            let backup_count: i64 =
                backup.query_row(&format!("SELECT count(*) FROM {table}"), [], |row| {
                    row.get(0)
                })?;
            if source_count != backup_count {
                bail!("backup did not preserve {table} rows");
            }
        }
    }
    Ok(())
}

fn asset_inventory_schema() -> &'static str {
    "CREATE TABLE IF NOT EXISTS analysis_asset_inventory(
         asset_id INTEGER PRIMARY KEY REFERENCES assets(id) ON DELETE CASCADE,
         source_size INTEGER NOT NULL,
         source_modified_ns INTEGER NOT NULL,
         source_fingerprint BLOB NOT NULL DEFAULT X'',
         decoder_model_id TEXT NOT NULL DEFAULT 'renderer-6',
         discovered_at TEXT NOT NULL
     );"
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct SourceContentIdentity {
    pub size: i64,
    pub modified_ns: i64,
    pub fingerprint: [u8; 32],
}

#[cfg(test)]
thread_local! {
    static CONTENT_IDENTITY_CHUNK_HOOK: std::cell::RefCell<Option<Box<dyn Fn()>>> =
        const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
pub(crate) fn install_content_identity_chunk_hook(hook: impl Fn() + 'static) {
    CONTENT_IDENTITY_CHUNK_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn modified_ns(metadata: &fs::Metadata) -> i64 {
    metadata
        .modified()
        .ok()
        .and_then(|value| value.duration_since(std::time::UNIX_EPOCH).ok())
        .map_or(0, |value| value.as_nanos().min(i64::MAX as u128) as i64)
}

/// Reads size, mtime, and full-content SHA-256 from one handle. Metadata is
/// sampled around hashing; an unstable source is retried twice, then rejected.
pub(crate) fn stable_content_identity(path: &Path) -> Result<SourceContentIdentity> {
    use sha2::Digest;
    use std::io::Read;

    for attempt in 0..3 {
        let mut file = fs::File::open(path)
            .with_context(|| format!("open source for content identity: {}", path.display()))?;
        let before = file.metadata()?;
        let mut hasher = sha2::Sha256::new();
        hasher.update(b"rekesz-content-identity-v1");
        let mut buffer = [0_u8; CONTENT_IDENTITY_BUFFER_BYTES];
        let mut first = true;
        loop {
            let read = file.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
            if first {
                first = false;
                #[cfg(test)]
                CONTENT_IDENTITY_CHUNK_HOOK.with(|slot| {
                    if let Some(hook) = slot.borrow().as_ref() {
                        hook();
                    }
                });
            }
        }
        let after = file.metadata()?;
        let before_identity = (before.len(), modified_ns(&before));
        let after_identity = (after.len(), modified_ns(&after));
        if before_identity == after_identity {
            return Ok(SourceContentIdentity {
                size: before.len().min(i64::MAX as u64) as i64,
                modified_ns: before_identity.1,
                fingerprint: hasher.finalize().into(),
            });
        }
        if attempt == 2 {
            bail!("source changed repeatedly while computing content identity");
        }
    }
    unreachable!()
}

fn ensure_analysis_decoder_columns(connection: &Connection) -> Result<()> {
    for table in ["analysis_asset_inventory", "face_analysis_state"] {
        let present: bool = connection.query_row(
            "SELECT EXISTS(SELECT 1 FROM pragma_table_info(?1) WHERE name='decoder_model_id')",
            [table],
            |row| row.get(0),
        )?;
        if !present {
            connection.execute(
                &format!("ALTER TABLE {table} ADD COLUMN decoder_model_id TEXT NOT NULL DEFAULT 'renderer-6'"),
                [],
            )?;
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct AssetInventorySync {
    pub present: usize,
    pub added: usize,
    pub invalidated: usize,
    pub removed: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum InventoryIdentityMode {
    MetadataFirst,
    VerifyContents,
}

/// Reconciles a complete, already-filtered discovery snapshot with the catalog.
/// Callers own supported-format, hidden-subtree, and RAW/JPEG-pair policy; this
/// boundary owns portable paths, stable asset ids, source identity, and removal.
pub(crate) fn sync_asset_inventory(
    root: &Path,
    eligible_paths: &[PathBuf],
    detector_model_id: &str,
    recognizer_model_id: &str,
) -> Result<AssetInventorySync> {
    sync_asset_inventory_with_progress(
        root,
        eligible_paths,
        detector_model_id,
        recognizer_model_id,
        InventoryIdentityMode::VerifyContents,
        |_, _, _| {},
        || true,
    )
}

pub(crate) fn sync_asset_inventory_with_progress(
    root: &Path,
    eligible_paths: &[PathBuf],
    detector_model_id: &str,
    recognizer_model_id: &str,
    identity_mode: InventoryIdentityMode,
    mut progress: impl FnMut(usize, usize, &Path),
    mut checkpoint: impl FnMut() -> bool,
) -> Result<AssetInventorySync> {
    fs::create_dir_all(root.join(CATALOG_DIR))?;
    let mut connection = Connection::open(catalog_path(root))?;
    initialize_schema(&mut connection)?;
    let mut stored = HashMap::<Vec<u8>, (i64, i64, [u8; 32])>::new();
    if identity_mode == InventoryIdentityMode::MetadataFirst {
        let mut query = connection.prepare(
            "SELECT asset.relative_path, inventory.source_size,
                    inventory.source_modified_ns, inventory.source_fingerprint
             FROM analysis_asset_inventory inventory
             JOIN assets asset ON asset.id=inventory.asset_id",
        )?;
        let rows = query.query_map([], |row| {
            Ok((
                row.get::<_, Vec<u8>>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, i64>(2)?,
                row.get::<_, Vec<u8>>(3)?,
            ))
        })?;
        for row in rows {
            let (path, size, modified, fingerprint) = row?;
            if let Ok(fingerprint) = fingerprint.try_into() {
                stored.insert(path, (size, modified, fingerprint));
            }
        }
    }
    let mut staged = Vec::with_capacity(eligible_paths.len());
    for (index, path) in eligible_paths.iter().enumerate() {
        if !checkpoint() {
            bail!("inventory synchronization cancelled");
        }
        progress(index, eligible_paths.len(), path);
        let relative = path
            .strip_prefix(root)
            .with_context(|| format!("inventory path is outside library: {}", path.display()))?;
        if relative.as_os_str().is_empty()
            || relative
                .components()
                .any(|component| !matches!(component, std::path::Component::Normal(_)))
        {
            bail!(
                "inventory path is not a portable relative asset: {}",
                path.display()
            );
        }
        let metadata = fs::metadata(path)
            .with_context(|| format!("read inventory identity for {}", path.display()))?;
        if metadata.is_file() {
            let size = metadata.len().min(i64::MAX as u64) as i64;
            let modified = modified_ns(&metadata);
            let encoded = encode_relative_path(relative)?;
            let identity = stored
                .get(&encoded)
                .filter(|(stored_size, stored_modified, _)| {
                    *stored_size == size && *stored_modified == modified
                })
                .map(|(_, _, fingerprint)| SourceContentIdentity {
                    size,
                    modified_ns: modified,
                    fingerprint: *fingerprint,
                })
                .map_or_else(|| stable_content_identity(path), Ok)?;
            staged.push((
                relative.to_path_buf(),
                identity.size,
                identity.modified_ns,
                identity.fingerprint,
            ));
        }
    }
    if let Some(last) = eligible_paths.last() {
        progress(eligible_paths.len(), eligible_paths.len(), last);
    }
    connection.execute_batch(
        "CREATE TEMP TABLE inventory_seen(asset_id INTEGER PRIMARY KEY) WITHOUT ROWID;",
    )?;
    let mut result = AssetInventorySync::default();
    let decoder_model_id = crate::rendering::DECODER_MODEL_ID;
    let staged_total = staged.len();
    // Keep maintenance write ownership short so interactive People edits can
    // acquire SQLite between batches.
    for chunk in staged.chunks(32) {
        if !checkpoint() {
            bail!("inventory synchronization cancelled");
        }
        let transaction = connection.transaction()?;
        begin_people_summary_bulk(&transaction)?;
        for (relative, source_size, source_modified_ns, source_fingerprint) in chunk {
            progress(result.present, staged_total, relative);
            let encoded = encode_relative_path(relative)?;
            let existed: bool = transaction.query_row(
                "SELECT EXISTS(SELECT 1 FROM assets WHERE relative_path=?1)",
                [&encoded],
                |row| row.get(0),
            )?;
            transaction.execute(
                "INSERT INTO assets(relative_path, scope_key) VALUES(?1, ?2)
             ON CONFLICT(relative_path) DO UPDATE SET scope_key=excluded.scope_key
             WHERE assets.scope_key<>excluded.scope_key",
                params![encoded, scope_key(relative)?],
            )?;
            let asset_id: i64 = transaction.query_row(
                "SELECT id FROM assets WHERE relative_path=?1",
                [&encoded],
                |row| row.get(0),
            )?;
            let changed: bool = transaction.query_row(
                "SELECT NOT EXISTS(SELECT 1 FROM analysis_asset_inventory inventory
               JOIN face_analysis_state state ON state.asset_id=inventory.asset_id
               WHERE inventory.asset_id=?1 AND inventory.source_size=?2
                 AND inventory.source_modified_ns=?3 AND state.detector_model_id=?4
                 AND state.recognizer_model_id=?5 AND state.decoder_model_id=?6
                 AND inventory.source_fingerprint=?7 AND state.source_fingerprint=?7
                 AND NOT(state.status='error' AND coalesce(state.error, '')
                         LIKE 'filesystem identity differs from inventory%'))",
                params![
                    asset_id,
                    source_size,
                    source_modified_ns,
                    detector_model_id,
                    recognizer_model_id,
                    decoder_model_id,
                    source_fingerprint.as_slice()
                ],
                |row| row.get(0),
            )?;
            transaction.execute(
                "INSERT INTO analysis_asset_inventory(asset_id, source_size, source_modified_ns,
                source_fingerprint, decoder_model_id, discovered_at)
             VALUES(?1, ?2, ?3, ?4, ?5, datetime('now')) ON CONFLICT(asset_id) DO UPDATE SET
             source_size=excluded.source_size, source_modified_ns=excluded.source_modified_ns,
             source_fingerprint=excluded.source_fingerprint,
             decoder_model_id=excluded.decoder_model_id, discovered_at=excluded.discovered_at",
                params![
                    asset_id,
                    source_size,
                    source_modified_ns,
                    source_fingerprint.as_slice(),
                    decoder_model_id
                ],
            )?;
            if changed {
                let face_ids = {
                    let mut statement = transaction.prepare(
                        "SELECT id FROM faces WHERE asset_id=?1 AND provenance='rekesz'",
                    )?;
                    statement
                        .query_map([asset_id], |row| row.get::<_, i64>(0))?
                        .collect::<rusqlite::Result<Vec<_>>>()?
                };
                if !face_ids.is_empty() {
                    let mut models = Vec::new();
                    let mut people = Vec::new();
                    for face_id in &face_ids {
                        let mut statement = transaction
                            .prepare("SELECT model_id FROM face_embeddings WHERE face_id=?1")?;
                        models.extend(
                            statement
                                .query_map([face_id], |row| row.get::<_, String>(0))?
                                .collect::<rusqlite::Result<Vec<_>>>()?,
                        );
                        let mut statement = transaction.prepare(
                            "SELECT person_id FROM effective_confirmations WHERE face_id=?1",
                        )?;
                        people.extend(
                            statement
                                .query_map([face_id], |row| row.get::<_, i64>(0))?
                                .collect::<rusqlite::Result<Vec<_>>>()?,
                        );
                    }
                    models.sort();
                    models.dedup();
                    people.sort_unstable();
                    people.dedup();
                    recognition::enqueue_refresh_for_models_in_connection(
                        &transaction,
                        &models,
                        &people,
                        &face_ids,
                    )?;
                    transaction.execute(
                        "DELETE FROM recognition_suggestions WHERE face_id IN
                       (SELECT id FROM faces WHERE asset_id=?1 AND provenance='rekesz')",
                        [asset_id],
                    )?;
                    transaction.execute(
                        "DELETE FROM face_embeddings WHERE face_id IN
                       (SELECT id FROM faces WHERE asset_id=?1 AND provenance='rekesz')",
                        [asset_id],
                    )?;
                }
                transaction.execute(
                "INSERT INTO face_analysis_state(asset_id, detector_model_id, recognizer_model_id,
                    source_size, source_modified_ns, source_fingerprint, decoder_model_id,
                    status, attempts, failure_class, updated_at)
                 VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, 'pending', 0, NULL, datetime('now'))
                 ON CONFLICT(asset_id) DO UPDATE SET detector_model_id=excluded.detector_model_id,
                    recognizer_model_id=excluded.recognizer_model_id,
                    decoder_model_id=excluded.decoder_model_id,
                    source_size=excluded.source_size, source_modified_ns=excluded.source_modified_ns,
                    source_fingerprint=excluded.source_fingerprint,
                    status='pending', attempts=0, worker_id=NULL, lease_expires_at=NULL,
                    retry_after=NULL, error=NULL, failure_class=NULL, updated_at=excluded.updated_at",
                params![asset_id, detector_model_id, recognizer_model_id, source_size,
                    source_modified_ns, source_fingerprint.as_slice(), decoder_model_id],
            )?;
                result.invalidated += usize::from(existed);
            }
            result.added += usize::from(!existed);
            result.present += 1;
            transaction.execute(
                "INSERT OR IGNORE INTO inventory_seen VALUES(?1)",
                [asset_id],
            )?;
        }
        finish_people_summary_bulk(&transaction)?;
        transaction.commit()?;
        if let Some((relative, ..)) = chunk.last() {
            progress(result.present, staged_total, relative);
        }
    }
    let transaction = connection.transaction()?;
    begin_people_summary_bulk(&transaction)?;
    result.removed = transaction.execute(
        "DELETE FROM analysis_asset_inventory WHERE asset_id NOT IN (SELECT asset_id FROM inventory_seen)",
        [],
    )?;
    transaction.execute(
        "DELETE FROM face_analysis_state WHERE asset_id NOT IN
           (SELECT asset_id FROM analysis_asset_inventory)",
        [],
    )?;
    transaction.execute(
        "DELETE FROM assets WHERE id NOT IN (SELECT asset_id FROM analysis_asset_inventory)
          AND id NOT IN (SELECT asset_id FROM develop_recipes)
          AND id NOT IN (SELECT asset_id FROM faces)",
        [],
    )?;
    finish_people_summary_bulk(&transaction)?;
    transaction.commit()?;
    if let Some(last) = eligible_paths.last() {
        progress(staged_total, staged_total, last);
    }
    Ok(result)
}

fn people_schema() -> &'static str {
    "CREATE TABLE IF NOT EXISTS people(
         id INTEGER PRIMARY KEY,
         display_name TEXT NOT NULL,
         locally_renamed INTEGER NOT NULL DEFAULT 0 CHECK(locally_renamed IN (0,1)),
         representative_face_id INTEGER REFERENCES faces(id) ON DELETE SET NULL,
         representative_is_manual INTEGER NOT NULL DEFAULT 0 CHECK(representative_is_manual IN (0,1)),
         created_at TEXT NOT NULL,
         updated_at TEXT NOT NULL
     );
     CREATE TABLE IF NOT EXISTS external_people(
         person_id INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,
         provenance TEXT NOT NULL,
         external_key BLOB NOT NULL,
         PRIMARY KEY(provenance, external_key)
     );
     CREATE TABLE IF NOT EXISTS faces(
         id INTEGER PRIMARY KEY,
         asset_id INTEGER NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
         provenance TEXT NOT NULL,
         source_key TEXT NOT NULL,
         left REAL NOT NULL, top REAL NOT NULL, right REAL NOT NULL, bottom REAL NOT NULL,
         origin TEXT NOT NULL,
         ignored INTEGER NOT NULL DEFAULT 0 CHECK(ignored IN (0,1)),
         locally_ignored INTEGER NOT NULL DEFAULT 0 CHECK(locally_ignored IN (0,1)),
         retired INTEGER NOT NULL DEFAULT 0 CHECK(retired IN (0,1)),
         detector_model_id TEXT,
         detection_confidence REAL,
         updated_at TEXT NOT NULL,
         UNIQUE(provenance, source_key),
         CHECK(left >= 0 AND top >= 0 AND right <= 1 AND bottom <= 1),
         CHECK(left < right AND top < bottom)
     );
     CREATE INDEX IF NOT EXISTS faces_asset ON faces(asset_id);
     CREATE TABLE IF NOT EXISTS face_assignments(
         face_id INTEGER NOT NULL REFERENCES faces(id) ON DELETE CASCADE,
         person_id INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,
         kind TEXT NOT NULL CHECK(kind IN ('confirmed','suggested','rejected')),
         provenance TEXT NOT NULL,
         confidence REAL,
         updated_at TEXT NOT NULL,
         PRIMARY KEY(face_id, person_id, kind, provenance)
     );
     CREATE INDEX IF NOT EXISTS face_assignments_person ON face_assignments(person_id, kind);
     CREATE UNIQUE INDEX IF NOT EXISTS one_local_confirmation_per_face
         ON face_assignments(face_id) WHERE kind='confirmed' AND provenance='rekesz';
     CREATE VIEW IF NOT EXISTS effective_face_assignments AS
         SELECT assignment.face_id, assignment.person_id, assignment.kind,
                assignment.provenance, assignment.confidence, assignment.updated_at
         FROM face_assignments assignment
         WHERE assignment.kind='confirmed' AND assignment.provenance='rekesz'
           AND EXISTS (SELECT 1 FROM faces face WHERE face.id=assignment.face_id AND face.retired=0)
         UNION ALL
         SELECT assignment.face_id, assignment.person_id, assignment.kind,
                assignment.provenance, assignment.confidence, assignment.updated_at
         FROM face_assignments assignment
         WHERE assignment.kind='confirmed' AND assignment.provenance!='rekesz'
           AND EXISTS (SELECT 1 FROM faces face WHERE face.id=assignment.face_id AND face.retired=0)
           AND NOT EXISTS (
               SELECT 1 FROM face_assignments local
               WHERE local.face_id=assignment.face_id AND local.provenance='rekesz'
                 AND (local.kind='confirmed' OR
                      (local.kind='rejected' AND local.person_id=assignment.person_id)))
         UNION ALL
         SELECT assignment.face_id, assignment.person_id, assignment.kind,
                assignment.provenance, assignment.confidence, assignment.updated_at
         FROM face_assignments assignment
         WHERE assignment.kind='suggested'
           AND (assignment.provenance!='rekesz' OR NOT EXISTS (
               SELECT 1 FROM recognition_suggestions suggestion
               WHERE suggestion.face_id=assignment.face_id
                 AND suggestion.person_id=assignment.person_id)
             OR EXISTS (
               SELECT 1 FROM recognition_suggestions suggestion
               JOIN recognition_index_state state
                 ON state.model_id=suggestion.recognizer_model_id
                AND state.index_version=suggestion.ann_index_version
                AND state.generation=suggestion.scored_index_generation
               WHERE suggestion.face_id=assignment.face_id
                 AND suggestion.person_id=assignment.person_id))
           AND EXISTS (SELECT 1 FROM faces face WHERE face.id=assignment.face_id AND face.retired=0)
           AND NOT EXISTS (
               SELECT 1 FROM face_assignments confirmation
               WHERE confirmation.face_id=assignment.face_id
                 AND confirmation.kind='confirmed' AND confirmation.provenance='rekesz')
           AND NOT EXISTS (
               SELECT 1 FROM face_assignments confirmation
               WHERE confirmation.face_id=assignment.face_id
                 AND confirmation.kind='confirmed' AND confirmation.provenance!='rekesz'
                 AND NOT EXISTS (
                     SELECT 1 FROM face_assignments local
                     WHERE local.face_id=confirmation.face_id AND local.provenance='rekesz'
                       AND (local.kind='confirmed' OR
                            (local.kind='rejected' AND local.person_id=confirmation.person_id))))
           AND NOT EXISTS (
               SELECT 1 FROM face_assignments rejection
               WHERE rejection.face_id=assignment.face_id
                 AND rejection.person_id=assignment.person_id
                 AND rejection.kind='rejected' AND rejection.provenance='rekesz')
           AND (assignment.provenance='lightroom' OR NOT EXISTS (
               SELECT 1 FROM face_assignments preferred
               WHERE preferred.face_id=assignment.face_id
                 AND preferred.person_id=assignment.person_id
                 AND preferred.kind='suggested' AND preferred.provenance='lightroom'));
     CREATE VIEW IF NOT EXISTS effective_confirmations AS
         SELECT assignment.face_id, assignment.person_id, assignment.provenance
         FROM face_assignments assignment
         WHERE assignment.kind='confirmed' AND assignment.provenance='rekesz'
           AND EXISTS (SELECT 1 FROM faces face WHERE face.id=assignment.face_id AND face.retired=0)
         UNION ALL
         SELECT assignment.face_id, assignment.person_id, assignment.provenance
         FROM face_assignments assignment
         WHERE assignment.kind='confirmed' AND assignment.provenance!='rekesz'
           AND EXISTS (SELECT 1 FROM faces face WHERE face.id=assignment.face_id AND face.retired=0)
           AND NOT EXISTS (
               SELECT 1 FROM face_assignments local
               WHERE local.face_id=assignment.face_id AND local.provenance='rekesz'
                 AND (local.kind='confirmed' OR
                      (local.kind='rejected' AND local.person_id=assignment.person_id)));
     CREATE TABLE IF NOT EXISTS face_cluster_memberships(
         face_id INTEGER NOT NULL REFERENCES faces(id) ON DELETE CASCADE,
         provenance TEXT NOT NULL,
         cluster_key TEXT NOT NULL,
         PRIMARY KEY(face_id, provenance)
     );
     CREATE INDEX IF NOT EXISTS face_clusters_lookup
         ON face_cluster_memberships(provenance, cluster_key);
     CREATE TABLE IF NOT EXISTS face_embeddings(
         face_id INTEGER NOT NULL REFERENCES faces(id) ON DELETE CASCADE,
         model_id TEXT NOT NULL,
         dimensions INTEGER NOT NULL,
         vector BLOB NOT NULL,
         landmarks BLOB,
         source_size INTEGER,
         source_modified_ns INTEGER,
         source_fingerprint BLOB,
         created_at TEXT NOT NULL,
         PRIMARY KEY(face_id, model_id)
     );
     CREATE TABLE IF NOT EXISTS face_analysis_state(
         asset_id INTEGER PRIMARY KEY REFERENCES assets(id) ON DELETE CASCADE,
         detector_model_id TEXT NOT NULL,
         recognizer_model_id TEXT NOT NULL DEFAULT 'sface-2021dec',
         decoder_model_id TEXT NOT NULL DEFAULT 'renderer-6',
         source_size INTEGER NOT NULL,
         source_modified_ns INTEGER NOT NULL,
         source_fingerprint BLOB NOT NULL DEFAULT X'',
         status TEXT NOT NULL CHECK(status IN ('pending','running','complete','error')),
         attempts INTEGER NOT NULL DEFAULT 0,
         worker_id TEXT,
         lease_expires_at TEXT,
         retry_after TEXT,
         error TEXT,
         failure_class TEXT CHECK(failure_class IS NULL OR failure_class IN
             ('transient','deterministic','exhausted','manual')),
         updated_at TEXT NOT NULL
     );
     CREATE INDEX IF NOT EXISTS face_analysis_eligibility
         ON face_analysis_state(status, retry_after, lease_expires_at);
     CREATE TABLE IF NOT EXISTS face_reconciliation_conflicts(
         asset_id INTEGER PRIMARY KEY REFERENCES assets(id) ON DELETE CASCADE,
         detector_model_id TEXT NOT NULL,
         conflict_count INTEGER NOT NULL CHECK(conflict_count>0),
         updated_at TEXT NOT NULL
     );
     CREATE TABLE IF NOT EXISTS person_prototypes(
         person_id INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,
         model_id TEXT NOT NULL,
         dimensions INTEGER NOT NULL,
         vector BLOB NOT NULL,
         exemplar_count INTEGER NOT NULL,
         algorithm_version INTEGER NOT NULL DEFAULT 1,
         generation INTEGER NOT NULL,
         updated_at TEXT NOT NULL,
         PRIMARY KEY(person_id, model_id)
     );
     CREATE TABLE IF NOT EXISTS recognition_suggestions(
         face_id INTEGER NOT NULL REFERENCES faces(id) ON DELETE CASCADE,
         person_id INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,
         recognizer_model_id TEXT NOT NULL,
         prototype_generation INTEGER NOT NULL,
         prototype_algorithm_version INTEGER NOT NULL,
         ann_index_version INTEGER NOT NULL DEFAULT 1,
         scored_index_generation INTEGER NOT NULL DEFAULT 0,
         policy_version INTEGER NOT NULL,
         score REAL NOT NULL,
         nearest_exemplar_score REAL NOT NULL,
         second_best_margin REAL NOT NULL,
         created_at TEXT NOT NULL,
         PRIMARY KEY(face_id, person_id, recognizer_model_id)
     );
     CREATE INDEX IF NOT EXISTS recognition_suggestions_person
         ON recognition_suggestions(person_id, recognizer_model_id);
     CREATE TABLE IF NOT EXISTS recognition_index_state(
         model_id TEXT PRIMARY KEY,
         generation INTEGER NOT NULL,
         index_version INTEGER NOT NULL,
         updated_at TEXT NOT NULL);
     CREATE TABLE IF NOT EXISTS recognition_refresh_queue(
         id INTEGER PRIMARY KEY,
         entity_kind TEXT NOT NULL CHECK(entity_kind IN ('person','face')),
         entity_id INTEGER NOT NULL,
         model_id TEXT NOT NULL,
         status TEXT NOT NULL CHECK(status IN ('pending','running','error')),
         attempts INTEGER NOT NULL DEFAULT 0,
         worker_id TEXT,
         lease_expires_at TEXT,
         retry_after TEXT,
         error TEXT,
         updated_at TEXT NOT NULL,
         UNIQUE(entity_kind, entity_id, model_id)
     );
     CREATE INDEX IF NOT EXISTS recognition_refresh_eligibility
         ON recognition_refresh_queue(status, retry_after, lease_expires_at);
     CREATE TABLE IF NOT EXISTS recognition_rescore_state(
         model_id TEXT PRIMARY KEY,
         after_face_id INTEGER NOT NULL,
         updated_at TEXT NOT NULL
     );
     CREATE TABLE IF NOT EXISTS recognition_policy_artifacts(
         model_id TEXT PRIMARY KEY,
         policy_version INTEGER NOT NULL,
         prototype_algorithm_version INTEGER NOT NULL,
         absolute_threshold REAL NOT NULL,
         nearest_exemplar_threshold REAL NOT NULL,
         second_best_margin REAL NOT NULL,
         calibration_digest BLOB NOT NULL,
         enabled INTEGER NOT NULL CHECK(enabled IN (0,1)),
         installed_at TEXT NOT NULL
     );
     CREATE TABLE IF NOT EXISTS face_decision_journal(
         id INTEGER PRIMARY KEY,
         batch_id TEXT NOT NULL,
         face_id INTEGER NOT NULL REFERENCES faces(id) ON DELETE CASCADE,
         person_id INTEGER REFERENCES people(id) ON DELETE CASCADE,
         previous_state TEXT,
         next_state TEXT NOT NULL,
         created_at TEXT NOT NULL
     );
     CREATE INDEX IF NOT EXISTS face_decision_journal_batch
         ON face_decision_journal(batch_id, id);"
}

fn people_summary_schema() -> String {
    let mut schema = "CREATE TABLE IF NOT EXISTS people_summary_epoch(
       id INTEGER PRIMARY KEY CHECK(id=1), source_epoch INTEGER NOT NULL CHECK(source_epoch>=0),
       published_epoch INTEGER NOT NULL CHECK(published_epoch>=0));
     INSERT OR IGNORE INTO people_summary_epoch VALUES(1,1,0);
     CREATE TABLE IF NOT EXISTS people_summary_bulk_state(
       id INTEGER PRIMARY KEY CHECK(id=1), depth INTEGER NOT NULL DEFAULT 0,
       dirty INTEGER NOT NULL DEFAULT 0 CHECK(dirty IN (0,1)));
     INSERT OR IGNORE INTO people_summary_bulk_state(id) VALUES(1);
     CREATE TABLE IF NOT EXISTS people_summary_rebuild_queue(
       id INTEGER PRIMARY KEY CHECK(id=1), status TEXT NOT NULL CHECK(status IN ('pending','running','error')),
       requested_epoch INTEGER NOT NULL, attempts INTEGER NOT NULL DEFAULT 0, worker_id TEXT,
       claimed_epoch INTEGER, lease_expires_at TEXT, retry_after TEXT, error TEXT,
       updated_at TEXT NOT NULL);
     INSERT OR IGNORE INTO people_summary_rebuild_queue(id,status,requested_epoch,updated_at)
       VALUES(1,'pending',1,datetime('now'));
     CREATE TABLE IF NOT EXISTS asset_scope_ancestors(
       build_epoch INTEGER NOT NULL, asset_id INTEGER NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
       scope_key BLOB NOT NULL, PRIMARY KEY(build_epoch,asset_id,scope_key));
     CREATE INDEX IF NOT EXISTS asset_scope_ancestors_scope
       ON asset_scope_ancestors(build_epoch,scope_key,asset_id);
     CREATE TABLE IF NOT EXISTS people_scope_summary(
       build_epoch INTEGER NOT NULL, scope_key BLOB NOT NULL, person_id INTEGER NOT NULL,
       display_name TEXT NOT NULL, sort_key TEXT NOT NULL, updated_at TEXT NOT NULL, confirmed_faces INTEGER NOT NULL,
       suggested_faces INTEGER NOT NULL, PRIMARY KEY(build_epoch,scope_key,person_id));
     CREATE INDEX IF NOT EXISTS people_scope_summary_name ON people_scope_summary
       (build_epoch,scope_key,sort_key,person_id);
     CREATE INDEX IF NOT EXISTS people_scope_summary_confirmed ON people_scope_summary
       (build_epoch,scope_key,confirmed_faces DESC,sort_key,person_id);
     CREATE INDEX IF NOT EXISTS people_scope_summary_review ON people_scope_summary
       (build_epoch,scope_key,suggested_faces DESC,sort_key,person_id);
     CREATE INDEX IF NOT EXISTS people_scope_summary_recent ON people_scope_summary
       (build_epoch,scope_key,updated_at DESC,person_id);
     CREATE TABLE IF NOT EXISTS people_scope_totals(
       build_epoch INTEGER NOT NULL, scope_key BLOB NOT NULL, active_faces INTEGER NOT NULL,
       confirmed_faces INTEGER NOT NULL, suggested_rows INTEGER NOT NULL,
       analysis_pending INTEGER NOT NULL, analysis_errors INTEGER NOT NULL,
       PRIMARY KEY(build_epoch,scope_key));
     CREATE TABLE IF NOT EXISTS people_scope_representatives(
       build_epoch INTEGER NOT NULL, scope_key BLOB NOT NULL, person_id INTEGER NOT NULL,
       rank INTEGER NOT NULL, face_id INTEGER NOT NULL, asset_id INTEGER NOT NULL,
       relative_path BLOB NOT NULL,
       left REAL NOT NULL, top REAL NOT NULL, right REAL NOT NULL, bottom REAL NOT NULL,
       PRIMARY KEY(build_epoch,scope_key,person_id,rank));
     CREATE TABLE IF NOT EXISTS people_scope_faces(
       build_epoch INTEGER NOT NULL, scope_key BLOB NOT NULL, queue_kind TEXT NOT NULL,
       face_id INTEGER NOT NULL, person_id INTEGER NOT NULL, person_name TEXT,
       assignment_kind TEXT NOT NULL, provenance TEXT NOT NULL, confidence REAL,
       relative_path BLOB NOT NULL, left REAL NOT NULL, top REAL NOT NULL, right REAL NOT NULL,
       bottom REAL NOT NULL, origin TEXT NOT NULL,
       PRIMARY KEY(build_epoch,scope_key,queue_kind,face_id,person_id));
     CREATE INDEX IF NOT EXISTS people_scope_faces_page ON people_scope_faces
       (build_epoch,scope_key,queue_kind,face_id,person_id);".to_owned();
    for table in [
        "face_assignments",
        "faces",
        "people",
        "assets",
        "recognition_suggestions",
        "recognition_index_state",
        "face_analysis_state",
        "face_embeddings",
    ] {
        for event in ["INSERT", "UPDATE", "DELETE"] {
            schema.push_str(&format!(
                "CREATE TRIGGER IF NOT EXISTS people_summary_dirty_{table}_{event} AFTER {event} ON {table} BEGIN
                 SELECT CASE WHEN (SELECT source_epoch FROM people_summary_epoch WHERE id=1)>=9223372036854775806
                   THEN RAISE(ABORT,'People summary epoch exhausted') END;
                 UPDATE people_summary_bulk_state SET dirty=1 WHERE id=1;
                 UPDATE people_summary_epoch SET source_epoch=source_epoch+1 WHERE id=1
                   AND (SELECT depth FROM people_summary_bulk_state WHERE id=1)=0;
                 INSERT INTO people_summary_rebuild_queue(id,status,requested_epoch,attempts,worker_id,
                   lease_expires_at,retry_after,error,updated_at)
                 SELECT 1,'pending',source_epoch,0,NULL,NULL,NULL,NULL,datetime('now') FROM people_summary_epoch WHERE id=1
                   AND (SELECT depth FROM people_summary_bulk_state WHERE id=1)=0
                 ON CONFLICT(id) DO UPDATE SET
                   status=CASE WHEN people_summary_rebuild_queue.status='running'
                                    AND people_summary_rebuild_queue.lease_expires_at>=datetime('now')
                               THEN 'running' ELSE 'pending' END,
                   requested_epoch=excluded.requested_epoch,
                   worker_id=CASE WHEN people_summary_rebuild_queue.status='running'
                                       AND people_summary_rebuild_queue.lease_expires_at>=datetime('now')
                                  THEN people_summary_rebuild_queue.worker_id ELSE NULL END,
                   claimed_epoch=CASE WHEN people_summary_rebuild_queue.status='running'
                                           AND people_summary_rebuild_queue.lease_expires_at>=datetime('now')
                                      THEN people_summary_rebuild_queue.claimed_epoch ELSE NULL END,
                   lease_expires_at=CASE WHEN people_summary_rebuild_queue.status='running'
                                              AND people_summary_rebuild_queue.lease_expires_at>=datetime('now')
                                         THEN people_summary_rebuild_queue.lease_expires_at ELSE NULL END,
                   retry_after=NULL,error=NULL,updated_at=excluded.updated_at; END;"
            ));
        }
    }
    schema
}

fn migrate_v4_to_v5(connection: &mut Connection) -> Result<()> {
    let transaction = connection.transaction()?;
    transaction.execute_batch(&people_summary_schema())?;
    let updated = transaction.execute(
        "UPDATE catalog_meta SET schema_version=5 WHERE id=1 AND schema_version=4",
        [],
    )?;
    if updated != 1 {
        bail!("Rekesz catalog metadata changed during v5 migration");
    }
    transaction.pragma_update(None, "user_version", 5)?;
    ensure_foreign_keys_valid(&transaction)?;
    transaction.commit()?;
    Ok(())
}

fn migrate_v5_to_v6(connection: &mut Connection) -> Result<()> {
    let transaction = connection.transaction()?;
    transaction.execute_batch(asset_inventory_schema())?;
    for (table, definition) in [
        ("analysis_asset_inventory", "source_fingerprint BLOB"),
        ("face_analysis_state", "source_fingerprint BLOB"),
        ("face_analysis_state", "failure_class TEXT"),
        ("face_embeddings", "source_fingerprint BLOB"),
    ] {
        let column = definition.split_whitespace().next().unwrap_or_default();
        let present: bool = transaction.query_row(
            "SELECT EXISTS(SELECT 1 FROM pragma_table_info(?1) WHERE name=?2)",
            params![table, column],
            |row| row.get(0),
        )?;
        if !present {
            transaction.execute(&format!("ALTER TABLE {table} ADD COLUMN {definition}"), [])?;
        }
    }
    transaction.execute_batch(
        "UPDATE face_analysis_state SET status='pending', attempts=0, worker_id=NULL,
             lease_expires_at=NULL, retry_after=NULL, error=NULL, failure_class=NULL,
             source_fingerprint=X'', updated_at=datetime('now');
         DELETE FROM recognition_suggestions;
         DELETE FROM person_prototypes;
         DELETE FROM face_embeddings WHERE face_id IN
             (SELECT id FROM faces WHERE provenance='rekesz');
         UPDATE face_embeddings SET source_fingerprint=NULL;",
    )?;
    // v6 representative ranking consumes persisted five-point landmarks, so
    // embedding changes must invalidate the durable People summary too.
    transaction.execute_batch(&people_summary_schema())?;
    let updated = transaction.execute(
        "UPDATE catalog_meta SET schema_version=6 WHERE id=1 AND schema_version=5",
        [],
    )?;
    if updated != 1 {
        bail!("Rekesz catalog metadata changed during v6 migration");
    }
    transaction.pragma_update(None, "user_version", 6)?;
    ensure_foreign_keys_valid(&transaction)?;
    transaction.commit()?;
    Ok(())
}

fn validate_owned_catalog(connection: &Connection) -> Result<()> {
    let application_id: i64 =
        connection.query_row("PRAGMA application_id", [], |row| row.get(0))?;
    if application_id != SQLITE_APPLICATION_ID {
        bail!("database is not a Rekesz catalog");
    }
    let rows: i64 =
        connection.query_row("SELECT count(*) FROM catalog_meta", [], |row| row.get(0))?;
    if rows != 1 {
        bail!("Rekesz catalog metadata is inconsistent with its schema");
    }
    Ok(())
}

fn migrate_v1_to_v3(connection: &mut Connection) -> Result<()> {
    let metadata_version: i64 = connection.query_row(
        "SELECT schema_version FROM catalog_meta WHERE id=1",
        [],
        |row| row.get(0),
    )?;
    if metadata_version != 1 {
        bail!(
            "Rekesz catalog version markers disagree: PRAGMA is 1 but metadata is {metadata_version}"
        );
    }
    let transaction = connection.transaction()?;
    transaction.execute_batch(people_schema())?;
    transaction.execute_batch(&people_summary_schema())?;
    transaction.execute("ALTER TABLE assets ADD COLUMN scope_key BLOB", [])?;
    transaction.execute("CREATE INDEX assets_scope_key ON assets(scope_key)", [])?;
    populate_path_keys(&transaction)?;
    install_scope_key_guards(&transaction)?;
    let updated = transaction.execute(
        "UPDATE catalog_meta SET schema_version=?1 WHERE id=1 AND schema_version=1",
        [3],
    )?;
    if updated != 1 {
        bail!("Rekesz catalog metadata changed during migration");
    }
    transaction.pragma_update(None, "user_version", 3)?;
    let migrated_version: i64 = transaction.query_row(
        "SELECT schema_version FROM catalog_meta WHERE id=1",
        [],
        |row| row.get(0),
    )?;
    if migrated_version != 3 {
        bail!("Rekesz catalog migration did not update its metadata");
    }
    ensure_foreign_keys_valid(&transaction)?;
    transaction.commit()?;
    Ok(())
}

fn migrate_v2_to_v3(connection: &mut Connection) -> Result<()> {
    let metadata_version: i64 = connection.query_row(
        "SELECT schema_version FROM catalog_meta WHERE id=1",
        [],
        |row| row.get(0),
    )?;
    if metadata_version != 2 {
        bail!(
            "Rekesz catalog version markers disagree: PRAGMA is 2 but metadata is {metadata_version}"
        );
    }
    let transaction = connection.transaction()?;
    transaction.execute_batch(
        "ALTER TABLE assets ADD COLUMN scope_key BLOB;
         CREATE INDEX assets_scope_key ON assets(scope_key);
         ALTER TABLE people ADD COLUMN representative_face_id INTEGER REFERENCES faces(id) ON DELETE SET NULL;
         ALTER TABLE people ADD COLUMN representative_is_manual INTEGER NOT NULL DEFAULT 0 CHECK(representative_is_manual IN (0,1));
         ALTER TABLE faces ADD COLUMN locally_ignored INTEGER NOT NULL DEFAULT 0 CHECK(locally_ignored IN (0,1));
         ALTER TABLE faces ADD COLUMN retired INTEGER NOT NULL DEFAULT 0 CHECK(retired IN (0,1));
         ALTER TABLE faces ADD COLUMN detector_model_id TEXT;
         ALTER TABLE faces ADD COLUMN detection_confidence REAL;
         ALTER TABLE face_embeddings ADD COLUMN source_size INTEGER;
         ALTER TABLE face_embeddings ADD COLUMN source_modified_ns INTEGER;
         ALTER TABLE face_analysis_state ADD COLUMN recognizer_model_id TEXT NOT NULL DEFAULT 'sface-2021dec';
         ALTER TABLE face_analysis_state ADD COLUMN worker_id TEXT;
         ALTER TABLE face_analysis_state ADD COLUMN lease_expires_at TEXT;
         ALTER TABLE face_analysis_state ADD COLUMN retry_after TEXT;
         CREATE INDEX face_analysis_eligibility
             ON face_analysis_state(status, retry_after, lease_expires_at);
         CREATE TABLE person_prototypes(
             person_id INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,
             model_id TEXT NOT NULL,
             dimensions INTEGER NOT NULL,
             vector BLOB NOT NULL,
             exemplar_count INTEGER NOT NULL,
             algorithm_version INTEGER NOT NULL DEFAULT 1,
             generation INTEGER NOT NULL,
             updated_at TEXT NOT NULL,
             PRIMARY KEY(person_id, model_id));
         CREATE TABLE face_decision_journal(
             id INTEGER PRIMARY KEY,
             batch_id TEXT NOT NULL,
             face_id INTEGER NOT NULL REFERENCES faces(id) ON DELETE CASCADE,
             person_id INTEGER REFERENCES people(id) ON DELETE CASCADE,
             previous_state TEXT,
             next_state TEXT NOT NULL,
             created_at TEXT NOT NULL);
         CREATE INDEX face_decision_journal_batch ON face_decision_journal(batch_id, id);
         DROP VIEW IF EXISTS effective_face_assignments;
         DROP VIEW IF EXISTS effective_confirmations;",
    )?;
    transaction.execute_batch(people_schema())?;
    populate_path_keys(&transaction)?;
    install_scope_key_guards(&transaction)?;
    let updated = transaction.execute(
        "UPDATE catalog_meta SET schema_version=3 WHERE id=1 AND schema_version=2",
        [],
    )?;
    if updated != 1 {
        bail!("Rekesz catalog metadata changed during v3 migration");
    }
    transaction.pragma_update(None, "user_version", 3)?;
    ensure_foreign_keys_valid(&transaction)?;
    transaction.commit()?;
    Ok(())
}

fn migrate_v3_to_v4(connection: &mut Connection) -> Result<()> {
    let metadata_version: i64 = connection.query_row(
        "SELECT schema_version FROM catalog_meta WHERE id=1",
        [],
        |row| row.get(0),
    )?;
    if metadata_version != 3 {
        bail!(
            "Rekesz catalog version markers disagree: PRAGMA is 3 but metadata is {metadata_version}"
        );
    }
    let transaction = connection.transaction()?;
    let has_ann_version: bool = transaction.query_row(
        "SELECT EXISTS(SELECT 1 FROM pragma_table_info('recognition_suggestions')
                       WHERE name='ann_index_version')",
        [],
        |row| row.get(0),
    )?;
    if !has_ann_version {
        transaction.execute(
            "ALTER TABLE recognition_suggestions
             ADD COLUMN ann_index_version INTEGER NOT NULL DEFAULT 0",
            [],
        )?;
    }
    let has_generation: bool = transaction.query_row(
        "SELECT EXISTS(SELECT 1 FROM pragma_table_info('recognition_suggestions')
                       WHERE name='scored_index_generation')",
        [],
        |row| row.get(0),
    )?;
    if !has_generation {
        transaction.execute(
            "ALTER TABLE recognition_suggestions
             ADD COLUMN scored_index_generation INTEGER NOT NULL DEFAULT 0",
            [],
        )?;
    }
    transaction.execute_batch(
        "CREATE TABLE IF NOT EXISTS recognition_index_state(
             model_id TEXT PRIMARY KEY,
             generation INTEGER NOT NULL,
             index_version INTEGER NOT NULL,
             updated_at TEXT NOT NULL);",
    )?;
    transaction.execute_batch(
        "DROP VIEW IF EXISTS effective_face_assignments;
         DROP VIEW IF EXISTS effective_confirmations;",
    )?;
    transaction.execute_batch(people_schema())?;
    let updated = transaction.execute(
        "UPDATE catalog_meta SET schema_version=4 WHERE id=1 AND schema_version=3",
        [],
    )?;
    if updated != 1 {
        bail!("Rekesz catalog metadata changed during v4 migration");
    }
    transaction.pragma_update(None, "user_version", 4)?;
    ensure_foreign_keys_valid(&transaction)?;
    transaction.commit()?;
    Ok(())
}

fn populate_path_keys(transaction: &Transaction<'_>) -> Result<()> {
    let paths = {
        let mut query = transaction.prepare("SELECT id, relative_path FROM assets")?;
        query
            .query_map([], |row| {
                Ok((row.get::<_, i64>(0)?, row.get::<_, Vec<u8>>(1)?))
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?
    };
    for (asset_id, encoded) in paths {
        let relative = decode_relative_path(&encoded)?;
        transaction.execute(
            "UPDATE assets SET scope_key=?1 WHERE id=?2",
            params![scope_key(&relative)?, asset_id],
        )?;
    }
    Ok(())
}

fn install_scope_key_guards(transaction: &Transaction<'_>) -> Result<()> {
    transaction.execute_batch(
        "CREATE TRIGGER assets_scope_key_insert_guard
         BEFORE INSERT ON assets WHEN NEW.scope_key IS NULL
         BEGIN SELECT RAISE(ABORT, 'assets.scope_key is required'); END;
         CREATE TRIGGER assets_scope_key_update_guard
         BEFORE UPDATE OF scope_key ON assets WHEN NEW.scope_key IS NULL
         BEGIN SELECT RAISE(ABORT, 'assets.scope_key is required'); END;",
    )?;
    Ok(())
}

fn ensure_foreign_keys_valid(transaction: &Transaction<'_>) -> Result<()> {
    let mut statement = transaction.prepare("PRAGMA foreign_key_check")?;
    if statement.query([])?.next()?.is_some() {
        bail!("Rekesz catalog contains a foreign-key violation");
    }
    Ok(())
}

#[cfg(test)]
pub(crate) fn initialize_test_catalog(root: &Path) -> Result<()> {
    let directory = root.join(CATALOG_DIR);
    fs::create_dir_all(&directory)?;
    let mut connection = Connection::open(directory.join(CATALOG_FILE))?;
    initialize_schema(&mut connection)
}

fn validate_schema(connection: &Connection) -> Result<()> {
    validate_owned_catalog(connection)?;
    let version: i64 = connection.query_row("PRAGMA user_version", [], |row| row.get(0))?;
    if version != SCHEMA_VERSION {
        bail!("unsupported Rekesz catalog schema {version}; expected {SCHEMA_VERSION}");
    }
    let (rows, metadata_version, renderer_model): (i64, i64, i64) = connection.query_row(
        "SELECT count(*), min(schema_version), min(renderer_model) FROM catalog_meta",
        [],
        |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
    )?;
    if rows != 1
        || metadata_version != SCHEMA_VERSION
        || renderer_model <= 0
        || renderer_model > i64::from(RENDERER_MODEL_VERSION)
    {
        bail!("Rekesz catalog metadata is inconsistent with its schema");
    }
    Ok(())
}

fn open_lightroom_immutable(path: &Path) -> Result<Connection> {
    let database_path = lightroom_database_path(path)?;
    let escaped = database_path
        .to_string_lossy()
        .replace('%', "%25")
        .replace('?', "%3F")
        .replace('#', "%23");
    let connection = Connection::open_with_flags(
        format!("file:{escaped}?mode=ro&immutable=1"),
        OpenFlags::SQLITE_OPEN_READ_ONLY
            | OpenFlags::SQLITE_OPEN_URI
            | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )?;
    connection.execute_batch("PRAGMA query_only=ON")?;
    Ok(connection)
}

fn lightroom_database_path(selected: &Path) -> Result<PathBuf> {
    let candidate = if selected
        .extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("lrcat-data"))
    {
        let name = selected
            .file_name()
            .and_then(|name| name.to_str())
            .and_then(|name| name.strip_suffix("-data"))
            .context("selected Lightroom .lrcat-data file has an invalid name")?;
        let sibling = selected.with_file_name(name);
        if !sibling.is_file() {
            bail!(
                "selected {} is Lightroom auxiliary data; select its missing sibling {} instead",
                selected.display(),
                sibling.display()
            );
        }
        sibling
    } else {
        selected.to_path_buf()
    };
    let mut file = fs::File::open(&candidate)
        .with_context(|| format!("cannot open Lightroom catalog {}", candidate.display()))?;
    let mut header = [0_u8; 16];
    let read = file
        .read(&mut header)
        .with_context(|| format!("cannot read Lightroom catalog {}", candidate.display()))?;
    if read >= 4 && header[..4] == *b"PK\x03\x04" {
        bail!(
            "{} is a compressed Lightroom backup; extract it and select the .lrcat file inside",
            candidate.display()
        );
    }
    if read != header.len() || header != *b"SQLite format 3\0" {
        bail!(
            "{} is not a Lightroom .lrcat SQLite database; select the main .lrcat file, not .lrcat-data, Previews.lrdata, or a backup archive",
            candidate.display()
        );
    }
    Ok(candidate)
}

fn resolve_relative_source(
    tree_root: &Path,
    lightroom_root: &str,
    folder: &str,
    filename: &str,
) -> Option<PathBuf> {
    let direct = Path::new(lightroom_root).join(folder).join(filename);
    if let Ok(candidate) = direct.canonicalize()
        && let Ok(relative) = candidate.strip_prefix(tree_root)
        && candidate.is_file()
    {
        return Some(relative.to_path_buf());
    }
    let tree_name = tree_root.file_name()?.to_string_lossy();
    let lr_name = lightroom_root
        .trim_end_matches(['/', '\\'])
        .rsplit(['/', '\\'])
        .next()?;
    if tree_name.eq_ignore_ascii_case(lr_name) {
        let candidate = tree_root.join(folder).join(filename).canonicalize().ok()?;
        if candidate.is_file() {
            return candidate
                .strip_prefix(tree_root)
                .ok()
                .map(Path::to_path_buf);
        }
    }
    None
}

#[cfg(unix)]
pub(crate) fn encode_relative_path(path: &Path) -> Result<Vec<u8>> {
    use std::os::unix::ffi::OsStrExt as _;
    if let Some(encoded) = encode_portable_path(path)? {
        return Ok(encoded);
    }
    let mut encoded = vec![b'u'];
    encoded.extend_from_slice(path.as_os_str().as_bytes());
    Ok(encoded)
}

#[cfg(unix)]
pub(crate) fn decode_relative_path(encoded: &[u8]) -> Result<PathBuf> {
    use std::os::unix::ffi::OsStrExt as _;
    if encoded.first() == Some(&b'p') {
        return decode_portable_path(encoded);
    }
    if encoded.first() != Some(&b'u') {
        bail!("catalog path was created on an incompatible platform");
    }
    Ok(PathBuf::from(std::ffi::OsStr::from_bytes(&encoded[1..])))
}

#[cfg(windows)]
pub(crate) fn encode_relative_path(path: &Path) -> Result<Vec<u8>> {
    use std::os::windows::ffi::OsStrExt as _;
    if let Some(encoded) = encode_portable_path(path)? {
        return Ok(encoded);
    }
    let mut encoded = vec![b'w'];
    for unit in path.as_os_str().encode_wide() {
        encoded.extend_from_slice(&unit.to_le_bytes());
    }
    Ok(encoded)
}

#[cfg(windows)]
pub(crate) fn decode_relative_path(encoded: &[u8]) -> Result<PathBuf> {
    use std::os::windows::ffi::OsStringExt as _;
    if encoded.first() == Some(&b'p') {
        return decode_portable_path(encoded);
    }
    if encoded.first() != Some(&b'w') || !(encoded.len() - 1).is_multiple_of(2) {
        bail!("catalog path was created on an incompatible platform");
    }
    let units = encoded[1..]
        .chunks_exact(2)
        .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]]))
        .collect::<Vec<_>>();
    Ok(PathBuf::from(std::ffi::OsString::from_wide(&units)))
}

fn encode_portable_path(path: &Path) -> Result<Option<Vec<u8>>> {
    let components = path
        .components()
        .map(|component| match component {
            std::path::Component::Normal(value) => value.to_str().map(str::to_owned),
            _ => None,
        })
        .collect::<Option<Vec<_>>>();
    let Some(components) = components else {
        return Ok(None);
    };
    let mut encoded = vec![b'p'];
    encoded.extend(serde_json::to_vec(&components)?);
    Ok(Some(encoded))
}

fn decode_portable_path(encoded: &[u8]) -> Result<PathBuf> {
    let components: Vec<String> = serde_json::from_slice(&encoded[1..])?;
    let mut path = PathBuf::new();
    for component in components {
        if component.is_empty() || component == "." || component == ".." {
            bail!("invalid portable relative path component");
        }
        path.push(component);
    }
    Ok(path)
}

#[derive(Clone, Debug, PartialEq)]
enum Value {
    Bool(bool),
    Number(f64),
    String(String),
    Table(Vec<(Option<String>, Value)>),
    Nil,
}

fn normalize_recipe(
    process_version: &str,
    values: &BTreeMap<String, Value>,
    capabilities: ImportCapabilities,
) -> (DevelopRecipe, Vec<String>, Vec<String>) {
    if !matches!(process_version, "6.7" | "11.0" | "15.4") {
        let mut unsupported = Vec::new();
        for (key, value) in values {
            flatten_property_paths(key, value, &mut unsupported);
        }
        unsupported.push(format!("UnsupportedProcessVersion:{process_version}"));
        return (
            DevelopRecipe::default(),
            unsupported,
            vec![format!("unsupported process version {process_version}")],
        );
    }
    // Every process version present in this catalog serializes its effective
    // basic-panel values in the 2012 fields. The older similarly named fields
    // are compatibility defaults, not the active recipe.
    let mut recipe = DevelopRecipe {
        exposure_ev: capabilities
            .basic_tone
            .then(|| number(values, "Exposure2012"))
            .flatten(),
        contrast: capabilities
            .basic_tone
            .then(|| number(values, "Contrast2012"))
            .flatten(),
        highlights: capabilities
            .basic_tone
            .then(|| number(values, "Highlights2012"))
            .flatten(),
        shadows: capabilities
            .basic_tone
            .then(|| number(values, "Shadows2012"))
            .flatten(),
        whites: capabilities
            .basic_tone
            .then(|| number(values, "Whites2012"))
            .flatten(),
        blacks: capabilities
            .basic_tone
            .then(|| number(values, "Blacks2012"))
            .flatten(),
        saturation: capabilities
            .color
            .then(|| number(values, "Saturation"))
            .flatten(),
        vibrance: capabilities
            .color
            .then(|| number(values, "Vibrance"))
            .flatten(),
        clarity: capabilities
            .basic_tone
            .then(|| number(values, "Clarity2012"))
            .flatten(),
        ..DevelopRecipe::default()
    };
    clear_zeroes(&mut recipe);
    let mut warnings = Vec::new();
    if capabilities.white_balance
        && string(values, "WhiteBalance").is_some_and(|value| value != "As Shot")
    {
        let temperature =
            number(values, "CustomTemperature").or_else(|| number(values, "Temperature"));
        let tint = number(values, "CustomTint").or_else(|| number(values, "Tint"));
        if let (Some(temperature), Some(tint)) = (temperature, tint) {
            recipe.white_balance = Some(WhiteBalance { temperature, tint });
        } else {
            warnings.push("custom white balance lacks absolute temperature/tint".to_owned());
        }
    }
    if capabilities.white_balance && recipe.white_balance.is_none() {
        let temperature = number(values, "CustomIncrementalTemperature")
            .or_else(|| number(values, "IncrementalTemperature"))
            .unwrap_or(0.0);
        let tint = number(values, "CustomIncrementalTint")
            .or_else(|| number(values, "IncrementalTint"))
            .unwrap_or(0.0);
        if temperature != 0.0 || tint != 0.0 {
            recipe.white_balance_increment = Some(WhiteBalanceIncrement { temperature, tint });
        }
    }
    if capabilities.tone_curve {
        recipe.tone_curve = curve(values, "ToneCurvePV2012").or_else(|| curve(values, "ToneCurve"));
        recipe.tone_curve_red =
            curve(values, "ToneCurvePV2012Red").or_else(|| curve(values, "ToneCurveRed"));
        recipe.tone_curve_green =
            curve(values, "ToneCurvePV2012Green").or_else(|| curve(values, "ToneCurveGreen"));
        recipe.tone_curve_blue =
            curve(values, "ToneCurvePV2012Blue").or_else(|| curve(values, "ToneCurveBlue"));
        if let Some(look_parameters) = nested_table(values, &["Look", "Parameters"]) {
            recipe.tone_curve = recipe
                .tone_curve
                .or_else(|| curve_entries(look_parameters, "ToneCurvePV2012"));
            recipe.tone_curve_red = recipe
                .tone_curve_red
                .or_else(|| curve_entries(look_parameters, "ToneCurvePV2012Red"));
            recipe.tone_curve_green = recipe
                .tone_curve_green
                .or_else(|| curve_entries(look_parameters, "ToneCurvePV2012Green"));
            recipe.tone_curve_blue = recipe
                .tone_curve_blue
                .or_else(|| curve_entries(look_parameters, "ToneCurvePV2012Blue"));
        }
    }
    let crop_values =
        ["CropLeft", "CropTop", "CropRight", "CropBottom"].map(|key| number(values, key));
    if capabilities.crop
        && let [Some(left), Some(top), Some(right), Some(bottom)] = crop_values
    {
        recipe.crop = Some(Crop {
            left,
            top,
            right,
            bottom,
            angle_degrees: number(values, "CropAngle").unwrap_or(0.0),
            constrain_aspect: boolean(values, "CropConstrainAspectRatio").unwrap_or(false),
        });
    }
    if capabilities.detail {
        let keys = [
            "Sharpness",
            "SharpenRadius",
            "SharpenDetail",
            "SharpenEdgeMasking",
            "LuminanceSmoothing",
            "LuminanceNoiseReductionDetail",
            "LuminanceNoiseReductionContrast",
            "ColorNoiseReduction",
            "ColorNoiseReductionDetail",
            "ColorNoiseReductionSmoothness",
        ];
        if keys.iter().any(|key| values.contains_key(*key)) {
            recipe.detail = Some(DetailRecipe {
                sharpening: number(values, keys[0]).unwrap_or(0.0),
                sharpen_radius: number(values, keys[1]).unwrap_or(1.0),
                sharpen_detail: number(values, keys[2]).unwrap_or(25.0),
                edge_masking: number(values, keys[3]).unwrap_or(0.0),
                luminance_noise_reduction: number(values, keys[4]).unwrap_or(0.0),
                luminance_detail: number(values, keys[5]).unwrap_or(50.0),
                luminance_contrast: number(values, keys[6]).unwrap_or(0.0),
                color_noise_reduction: number(values, keys[7]).unwrap_or(0.0),
                color_detail: number(values, keys[8]).unwrap_or(50.0),
                color_smoothness: number(values, keys[9]).unwrap_or(50.0),
            });
        }
    }
    if capabilities.presence {
        recipe.texture = number(values, "Texture").filter(|value| *value != 0.0);
        recipe.dehaze = number(values, "Dehaze").filter(|value| *value != 0.0);
    }
    if capabilities.monochrome && boolean(values, "ConvertToGrayscale") == Some(true) {
        recipe.monochrome = Some(MonochromeRecipe {
            mixer: [
                "GrayMixerRed",
                "GrayMixerOrange",
                "GrayMixerYellow",
                "GrayMixerGreen",
                "GrayMixerAqua",
                "GrayMixerBlue",
                "GrayMixerPurple",
                "GrayMixerMagenta",
            ]
            .map(|key| number(values, key).unwrap_or(0.0)),
        });
    }
    if capabilities.finishing
        && let Some(amount) = number(values, "PostCropVignetteAmount").filter(|value| *value != 0.0)
    {
        recipe.vignette = Some(VignetteRecipe {
            amount,
            midpoint: number(values, "PostCropVignetteMidpoint").unwrap_or(50.0),
            feather: number(values, "PostCropVignetteFeather").unwrap_or(50.0),
            roundness: number(values, "PostCropVignetteRoundness").unwrap_or(0.0),
        });
    }
    if capabilities.finishing {
        let curve = [
            "ParametricHighlights",
            "ParametricLights",
            "ParametricDarks",
            "ParametricShadows",
        ]
        .map(|key| number(values, key).unwrap_or(0.0));
        if curve.iter().any(|value| *value != 0.0) {
            recipe.parametric_curve = Some(curve);
        }
    }
    if capabilities.geometry {
        let stored_upright = [
            "UprightTransform_4",
            "UprightTransform_3",
            "UprightTransform_2",
            "UprightTransform_1",
        ]
        .into_iter()
        .find_map(|key| string(values, key).and_then(parse_homography));
        if stored_upright.is_some() {
            warnings.push(
                "Upright candidate matrices have no serialized active-mode selector; omitted"
                    .into(),
            );
        }
        recipe.lens_distortion =
            number(values, "LensManualDistortionAmount").filter(|value| *value != 0.0);
    }
    if capabilities.healing {
        recipe.heal_spots = parse_heal_spots(values.get("RetouchAreas"));
        if recipe.heal_spots.is_empty() {
            recipe.heal_spots = parse_heal_spots(values.get("RetouchInfo"));
        }
    }
    if capabilities.grain {
        let size = number(values, "GrainSize").unwrap_or(25.0);
        let roughness = number(values, "GrainFrequency");
        if size != 25.0 || roughness.is_some() {
            recipe.grain = Some(GrainRecipe {
                amount: ((size - 25.0).abs() / 32.0 * 25.0).clamp(0.0, 25.0),
                size,
                roughness: roughness.unwrap_or(50.0),
                seed: 0,
            });
            warnings.push("grain amount is inferred because Lightroom did not serialize it".into());
        }
    }

    let mut unsupported = Vec::new();
    if let Some(profile) = string(values, "CameraProfile")
        && !matches!(profile, "Embedded" | "Apple Embedded Color Profile")
    {
        warnings.push(format!(
            "camera profile '{profile}' has no compatible resolver; using the source/default profile"
        ));
    }
    if number(values, "LensProfileEnable").unwrap_or(0.0) != 0.0
        && (string(values, "LensProfileName").is_some()
            || string(values, "LensProfileFilename").is_some())
    {
        warnings.push(
            "lens profile has no compatible resolver; only manual distortion is imported"
                .to_owned(),
        );
    }
    for (key, value) in values {
        let supported = matches!(key.as_str(), "ProcessVersion" | "Version")
            || (capabilities.basic_tone
                && matches!(
                    key.as_str(),
                    "Exposure2012"
                        | "Contrast2012"
                        | "Highlights2012"
                        | "Shadows2012"
                        | "Whites2012"
                        | "Blacks2012"
                        | "Clarity2012"
                ))
            || (capabilities.color && matches!(key.as_str(), "Saturation" | "Vibrance"))
            || (capabilities.white_balance
                && matches!(
                    key.as_str(),
                    "WhiteBalance"
                        | "Temperature"
                        | "Tint"
                        | "CustomTemperature"
                        | "CustomTint"
                        | "IncrementalTemperature"
                        | "IncrementalTint"
                        | "CustomIncrementalTemperature"
                        | "CustomIncrementalTint"
                ))
            || (capabilities.tone_curve
                && matches!(
                    key.as_str(),
                    "ToneCurvePV2012"
                        | "ToneCurve"
                        | "ToneCurvePV2012Red"
                        | "ToneCurvePV2012Green"
                        | "ToneCurvePV2012Blue"
                        | "ToneCurveRed"
                        | "ToneCurveGreen"
                        | "ToneCurveBlue"
                ))
            || (capabilities.crop
                && matches!(
                    key.as_str(),
                    "CropLeft"
                        | "CropTop"
                        | "CropRight"
                        | "CropBottom"
                        | "CropAngle"
                        | "CropConstrainAspectRatio"
                ))
            || (capabilities.detail
                && matches!(
                    key.as_str(),
                    "Sharpness"
                        | "SharpenRadius"
                        | "SharpenDetail"
                        | "SharpenEdgeMasking"
                        | "LuminanceSmoothing"
                        | "LuminanceNoiseReductionDetail"
                        | "LuminanceNoiseReductionContrast"
                        | "ColorNoiseReduction"
                        | "ColorNoiseReductionDetail"
                        | "ColorNoiseReductionSmoothness"
                ))
            || (capabilities.presence && matches!(key.as_str(), "Texture" | "Dehaze"))
            || (capabilities.monochrome
                && (key == "ConvertToGrayscale" || key.starts_with("GrayMixer")))
            || (capabilities.finishing
                && (key.starts_with("PostCropVignette") || key.starts_with("Parametric")))
            || key == "CameraProfileDigest"
            || (key == "CameraProfile"
                && string(values, "CameraProfile").is_some_and(|profile| {
                    matches!(profile, "Embedded" | "Apple Embedded Color Profile")
                }))
            || (capabilities.geometry
                && (key.starts_with("Upright")
                    || key == "LensManualDistortionAmount"
                    || key.starts_with("LensProfile")
                    || key.starts_with("Perspective")))
            || (capabilities.healing && matches!(key.as_str(), "RetouchInfo" | "RetouchAreas"))
            || (capabilities.grain && matches!(key.as_str(), "GrainSize" | "GrainFrequency"));
        if !supported {
            flatten_property_paths(key, value, &mut unsupported);
        }
    }
    clear_lightroom_defaults(process_version, &mut recipe);
    (recipe, unsupported, warnings)
}

fn clear_lightroom_defaults(process_version: &str, recipe: &mut DevelopRecipe) {
    // Lightroom serializes these effective defaults even when the user has
    // made no edit. They are defaults of the supported PV2012-family process
    // versions, rather than authored recipe operations.
    if matches!(process_version, "6.7" | "11.0" | "15.4")
        && recipe
            .tone_curve
            .as_deref()
            .is_some_and(is_lightroom_default_tone_curve)
    {
        recipe.tone_curve = None;
    }
    if matches!(process_version, "6.7" | "11.0" | "15.4")
        && recipe.detail.is_some_and(is_lightroom_default_detail)
    {
        recipe.detail = None;
    }
}

fn is_lightroom_default_tone_curve(curve: &[[f32; 2]]) -> bool {
    const DEFAULT: [[f32; 2]; 7] = [
        [0.0, 0.0],
        [22.0 / 255.0, 16.0 / 255.0],
        [40.0 / 255.0, 35.0 / 255.0],
        [127.0 / 255.0, 127.0 / 255.0],
        [224.0 / 255.0, 230.0 / 255.0],
        [240.0 / 255.0, 246.0 / 255.0],
        [1.0, 1.0],
    ];
    curve.len() == DEFAULT.len()
        && curve.iter().zip(DEFAULT).all(|(actual, expected)| {
            actual
                .iter()
                .zip(expected)
                .all(|(actual, expected)| (actual - expected).abs() <= 1e-6)
        })
}

fn is_lightroom_default_detail(detail: DetailRecipe) -> bool {
    detail
        == DetailRecipe {
            sharpening: 40.0,
            sharpen_radius: 1.0,
            sharpen_detail: 25.0,
            edge_masking: 0.0,
            luminance_noise_reduction: 0.0,
            luminance_detail: 50.0,
            luminance_contrast: 0.0,
            color_noise_reduction: 25.0,
            color_detail: 50.0,
            color_smoothness: 50.0,
        }
}

fn flatten_property_paths(prefix: &str, value: &Value, output: &mut Vec<String>) {
    output.push(prefix.to_owned());
    let Value::Table(entries) = value else {
        return;
    };
    for (index, (key, value)) in entries.iter().enumerate() {
        let component = key.clone().unwrap_or_else(|| format!("[{index}]"));
        flatten_property_paths(&format!("{prefix}.{component}"), value, output);
    }
}

fn clear_zeroes(recipe: &mut DevelopRecipe) {
    for value in [
        &mut recipe.exposure_ev,
        &mut recipe.contrast,
        &mut recipe.highlights,
        &mut recipe.shadows,
        &mut recipe.whites,
        &mut recipe.blacks,
        &mut recipe.saturation,
        &mut recipe.vibrance,
        &mut recipe.clarity,
    ] {
        if value.is_some_and(|value| value == 0.0) {
            *value = None;
        }
    }
}

fn number(values: &BTreeMap<String, Value>, key: &str) -> Option<f32> {
    match values.get(key) {
        Some(Value::Number(value)) if value.is_finite() && (*value as f32).is_finite() => {
            Some(*value as f32)
        }
        _ => None,
    }
}

fn string<'a>(values: &'a BTreeMap<String, Value>, key: &str) -> Option<&'a str> {
    match values.get(key) {
        Some(Value::String(value)) => Some(value),
        _ => None,
    }
}

fn boolean(values: &BTreeMap<String, Value>, key: &str) -> Option<bool> {
    match values.get(key) {
        Some(Value::Bool(value)) => Some(*value),
        _ => None,
    }
}

fn curve(values: &BTreeMap<String, Value>, key: &str) -> Option<Vec<[f32; 2]>> {
    curve_value(values.get(key)?)
}

fn curve_entries(entries: &[(Option<String>, Value)], key: &str) -> Option<Vec<[f32; 2]>> {
    curve_value(
        entries
            .iter()
            .find_map(|(entry_key, value)| (entry_key.as_deref() == Some(key)).then_some(value))?,
    )
}

fn curve_value(value: &Value) -> Option<Vec<[f32; 2]>> {
    let Value::Table(entries) = value else {
        return None;
    };
    let numbers: Vec<f32> = entries
        .iter()
        .filter_map(|(_, value)| match value {
            Value::Number(value) => Some(*value as f32 / 255.0),
            _ => None,
        })
        .collect();
    if numbers.len() < 4 || !numbers.len().is_multiple_of(2) {
        return None;
    }
    let points: Vec<[f32; 2]> = numbers.as_chunks::<2>().0.to_vec();
    (points != [[0.0, 0.0], [1.0, 1.0]]).then_some(points)
}

fn parse_homography(value: &str) -> Option<[f32; 9]> {
    let values: Vec<f32> = value
        .split(',')
        .map(str::trim)
        .map(str::parse)
        .collect::<std::result::Result<_, _>>()
        .ok()?;
    let matrix: [f32; 9] = values.try_into().ok()?;
    let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    (matrix.iter().all(|value| value.is_finite())
        && matrix[8].abs() > 1e-6
        && matrix
            .iter()
            .zip(identity)
            .any(|(value, identity)| (value - identity).abs() > 1e-6))
    .then_some(matrix)
}

fn homography_determinant(m: [f32; 9]) -> f32 {
    m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
        + m[2] * (m[3] * m[7] - m[4] * m[6])
}

fn parse_heal_spots(value: Option<&Value>) -> Vec<HealSpot> {
    let Some(Value::Table(entries)) = value else {
        return Vec::new();
    };
    let mut spots = entries
        .iter()
        .filter_map(|(_, value)| parse_heal_spot(value))
        .collect::<Vec<_>>();
    spots.dedup();
    spots
}

fn parse_heal_spot(value: &Value) -> Option<HealSpot> {
    let Value::Table(entries) = value else {
        return None;
    };
    let number = |keys: &[&str]| {
        entries.iter().find_map(|(entry_key, value)| {
            if !keys.contains(&entry_key.as_deref()?) {
                return None;
            }
            match value {
                Value::Number(value) => Some(*value as f32),
                _ => None,
            }
        })
    };
    let text = |keys: &[&str]| {
        entries.iter().find_map(|(entry_key, value)| {
            if !keys.contains(&entry_key.as_deref()?) {
                return None;
            }
            match value {
                Value::String(value) => Some(value.as_str()),
                _ => None,
            }
        })
    };
    let kind = match text(&["spotType", "SpotType"])? {
        "heal" => HealKind::Heal,
        "clone" => HealKind::Clone,
        _ => return None,
    };
    Some(HealSpot {
        kind,
        destination: [
            number(&["centerX", "CenterX"])?,
            number(&["centerY", "CenterY"])?,
        ],
        source: [
            number(&["sourceX", "SourceX"])?,
            number(&["sourceY", "SourceY", "OffsetY"])?,
        ],
        radius: number(&["radius", "Radius"])?,
        opacity: number(&["opacity", "Opacity"]).unwrap_or(1.0),
        feather: number(&["Feather", "feather"]).unwrap_or(0.0) / 100.0,
    })
}

fn nested_table<'a>(
    values: &'a BTreeMap<String, Value>,
    path: &[&str],
) -> Option<&'a [(Option<String>, Value)]> {
    let mut current = values.get(*path.first()?)?;
    for component in &path[1..] {
        let Value::Table(entries) = current else {
            return None;
        };
        current = entries
            .iter()
            .find_map(|(key, value)| (key.as_deref() == Some(*component)).then_some(value))?;
    }
    let Value::Table(entries) = current else {
        return None;
    };
    Some(entries)
}

struct Parser<'a> {
    input: &'a [u8],
    offset: usize,
}

impl<'a> Parser<'a> {
    fn new(input: &'a str) -> Self {
        Self {
            input: input.as_bytes(),
            offset: 0,
        }
    }

    fn parse_assignment(mut self) -> Result<BTreeMap<String, Value>> {
        self.space();
        self.identifier()?;
        self.space();
        self.expect(b'=')?;
        let Value::Table(entries) = self.value()? else {
            bail!("recipe is not a table")
        };
        self.space();
        if self.offset != self.input.len() {
            bail!("unexpected trailing input at byte {}", self.offset);
        }
        Ok(entries
            .into_iter()
            .filter_map(|(key, value)| key.map(|key| (key, value)))
            .collect())
    }

    fn value(&mut self) -> Result<Value> {
        self.space();
        match self.peek() {
            Some(b'{') => self.table(),
            Some(b'"' | b'\'') => self.string().map(Value::String),
            Some(b'-' | b'+' | b'0'..=b'9') => self.number().map(Value::Number),
            Some(_) => match self.identifier()?.as_str() {
                "true" => Ok(Value::Bool(true)),
                "false" => Ok(Value::Bool(false)),
                "nil" => Ok(Value::Nil),
                value => Ok(Value::String(value.to_owned())),
            },
            None => bail!("expected value"),
        }
    }

    fn table(&mut self) -> Result<Value> {
        self.expect(b'{')?;
        let mut entries = Vec::new();
        loop {
            self.space();
            if self.consume(b'}') {
                break;
            }
            let checkpoint = self.offset;
            let key = if self.consume(b'[') {
                self.space();
                let key = self.string()?;
                self.space();
                self.expect(b']')?;
                self.space();
                self.expect(b'=')?;
                Some(key)
            } else if self.peek().is_some_and(identifier_start) {
                let candidate = self.identifier()?;
                self.space();
                if self.consume(b'=') {
                    Some(candidate)
                } else {
                    self.offset = checkpoint;
                    None
                }
            } else {
                None
            };
            entries.push((key, self.value()?));
            self.space();
            if !self.consume(b',') && !self.consume(b';') && self.peek() != Some(b'}') {
                bail!("expected separator at byte {}", self.offset);
            }
        }
        Ok(Value::Table(entries))
    }

    fn identifier(&mut self) -> Result<String> {
        self.space();
        let start = self.offset;
        if !self.peek().is_some_and(identifier_start) {
            bail!("expected identifier")
        }
        self.offset += 1;
        while self.peek().is_some_and(identifier_continue) {
            self.offset += 1;
        }
        Ok(String::from_utf8(self.input[start..self.offset].to_vec())?)
    }

    fn string(&mut self) -> Result<String> {
        let quote = self.next().ok_or_else(|| anyhow!("expected string"))?;
        let mut bytes = Vec::new();
        loop {
            match self.next() {
                Some(value) if value == quote => break,
                Some(b'\\') => {
                    bytes.push(self.next().ok_or_else(|| anyhow!("unterminated escape"))?)
                }
                Some(value) => bytes.push(value),
                None => bail!("unterminated string"),
            }
        }
        Ok(String::from_utf8(bytes)?)
    }

    fn number(&mut self) -> Result<f64> {
        let start = self.offset;
        while self
            .peek()
            .is_some_and(|v| matches!(v, b'+' | b'-' | b'.' | b'e' | b'E' | b'0'..=b'9'))
        {
            self.offset += 1;
        }
        Ok(std::str::from_utf8(&self.input[start..self.offset])?.parse()?)
    }
    fn space(&mut self) {
        while self.peek().is_some_and(|v| v.is_ascii_whitespace()) {
            self.offset += 1;
        }
    }
    fn expect(&mut self, value: u8) -> Result<()> {
        if self.consume(value) {
            Ok(())
        } else {
            bail!("expected token at byte {}", self.offset)
        }
    }
    fn consume(&mut self, value: u8) -> bool {
        if self.peek() == Some(value) {
            self.offset += 1;
            true
        } else {
            false
        }
    }
    fn peek(&self) -> Option<u8> {
        self.input.get(self.offset).copied()
    }
    fn next(&mut self) -> Option<u8> {
        let value = self.peek()?;
        self.offset += 1;
        Some(value)
    }
}

fn identifier_start(value: u8) -> bool {
    value.is_ascii_alphabetic() || value == b'_'
}
fn identifier_continue(value: u8) -> bool {
    value.is_ascii_alphanumeric() || value == b'_'
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
