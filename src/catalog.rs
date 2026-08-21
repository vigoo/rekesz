use std::{
    collections::{BTreeMap, HashMap},
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, anyhow, bail};
use rusqlite::{Connection, OpenFlags, params};
use serde::{Deserialize, Serialize};

pub const CATALOG_DIR: &str = ".pics";
const CATALOG_FILE: &str = "catalog.sqlite3";
const SCHEMA_VERSION: i64 = 1;
const SQLITE_APPLICATION_ID: i64 = 0x5049_4353;
pub const RENDERER_MODEL_VERSION: u32 = 5;

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

    if !apply {
        return Ok(report);
    }
    if report.conflicting_sources > 0 {
        bail!("refusing to import recipes with conflicting Lightroom source rows");
    }
    if planned.is_empty() {
        bail!("no recipes use actions supported by renderer model {RENDERER_MODEL_VERSION}");
    }
    if destination.exists() {
        let mut pics = Connection::open(&destination)
            .with_context(|| format!("cannot open Pics catalog {}", destination.display()))?;
        initialize_schema(&mut pics)?;
        write_planned_recipes(&mut pics, &planned)?;
    } else {
        let temporary = tempfile::NamedTempFile::new_in(&root)?;
        let temporary = temporary.into_temp_path();
        {
            let mut pics = Connection::open(&temporary)?;
            initialize_schema(&mut pics)?;
            write_planned_recipes(&mut pics, &planned)?;
            pics.execute_batch("PRAGMA wal_checkpoint(TRUNCATE); PRAGMA journal_mode=DELETE;")?;
        }
        let parent = destination.parent().expect("catalog path has a parent");
        fs::create_dir_all(parent).with_context(|| {
            format!("cannot create Pics catalog directory {}", parent.display())
        })?;
        temporary
            .persist(&destination)
            .map_err(|error| error.error)
            .with_context(|| format!("cannot install Pics catalog {}", destination.display()))?;
    }
    Ok(report)
}

fn write_planned_recipes(
    pics: &mut Connection,
    planned: &BTreeMap<Vec<u8>, DevelopRecipe>,
) -> Result<()> {
    let transaction = pics.transaction()?;
    transaction.execute("DELETE FROM develop_recipes", [])?;
    transaction.execute("DELETE FROM assets", [])?;
    for (relative_key, recipe) in planned {
        transaction.execute(
            "INSERT INTO assets(relative_path) VALUES (?1)",
            [&relative_key],
        )?;
        let asset_id = transaction.last_insert_rowid();
        let json = serde_json::to_string(&recipe)?;
        recipe.validate()?;
        let digest = recipe.digest()?;
        transaction.execute(
            "INSERT INTO develop_recipes(asset_id, recipe_json, recipe_digest, imported_at) \
             VALUES (?1, ?2, ?3, datetime('now'))",
            params![asset_id, json, digest.as_slice()],
        )?;
    }
    transaction.execute(
        "UPDATE catalog_meta SET renderer_model=?1, last_imported_at=datetime('now')",
        [RENDERER_MODEL_VERSION],
    )?;
    transaction.commit()?;
    Ok(())
}

fn catalog_path(root: &Path) -> PathBuf {
    root.join(CATALOG_DIR).join(CATALOG_FILE)
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
        bail!("refusing to initialize a database that is not a Pics catalog");
    }
    if tables > 0 {
        // Probe and reject an unsupported owned schema before any PRAGMA or DDL
        // can mutate it. Schema migrations must be explicit future operations.
        validate_schema(connection)?;
        connection.execute_batch("PRAGMA foreign_keys=ON; PRAGMA journal_mode=WAL;")?;
        return Ok(());
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
             relative_path BLOB NOT NULL UNIQUE
         );
         CREATE TABLE IF NOT EXISTS develop_recipes(
             asset_id INTEGER PRIMARY KEY REFERENCES assets(id) ON DELETE CASCADE,
             recipe_json TEXT NOT NULL,
             recipe_digest BLOB NOT NULL,
             imported_at TEXT NOT NULL
         );
         INSERT INTO catalog_meta(id, schema_version, renderer_model, created_at)
         SELECT 1, {SCHEMA_VERSION}, {RENDERER_MODEL_VERSION}, datetime('now')
         WHERE NOT EXISTS(SELECT 1 FROM catalog_meta);
         PRAGMA user_version={SCHEMA_VERSION};"
    ))?;
    validate_schema(connection)
}

fn validate_schema(connection: &Connection) -> Result<()> {
    let application_id: i64 =
        connection.query_row("PRAGMA application_id", [], |row| row.get(0))?;
    if application_id != SQLITE_APPLICATION_ID {
        bail!("database is not a Pics catalog");
    }
    let version: i64 = connection.query_row("PRAGMA user_version", [], |row| row.get(0))?;
    if version != SCHEMA_VERSION {
        bail!("unsupported Pics catalog schema {version}; expected {SCHEMA_VERSION}");
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
        bail!("Pics catalog metadata is inconsistent with its schema");
    }
    Ok(())
}

fn open_lightroom_immutable(path: &Path) -> Result<Connection> {
    let escaped = path
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
fn encode_relative_path(path: &Path) -> Result<Vec<u8>> {
    use std::os::unix::ffi::OsStrExt as _;
    if let Some(encoded) = encode_portable_path(path)? {
        return Ok(encoded);
    }
    let mut encoded = vec![b'u'];
    encoded.extend_from_slice(path.as_os_str().as_bytes());
    Ok(encoded)
}

#[cfg(unix)]
fn decode_relative_path(encoded: &[u8]) -> Result<PathBuf> {
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
fn encode_relative_path(path: &Path) -> Result<Vec<u8>> {
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
fn decode_relative_path(encoded: &[u8]) -> Result<PathBuf> {
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
    (recipe, unsupported, warnings)
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
mod tests {
    use super::*;

    fn lightroom_fixture(root: &Path, recipe: &str) -> tempfile::TempDir {
        let fixture = tempfile::tempdir().unwrap();
        let path = fixture.path().join("fixture.lrcat");
        let connection = Connection::open(path).unwrap();
        connection
            .execute_batch(
                "CREATE TABLE AgLibraryRootFolder(id_local INTEGER PRIMARY KEY, absolutePath TEXT);
                 CREATE TABLE AgLibraryFolder(id_local INTEGER PRIMARY KEY, rootFolder INTEGER, pathFromRoot TEXT);
                 CREATE TABLE AgLibraryFile(id_local INTEGER PRIMARY KEY, folder INTEGER, baseName TEXT, extension TEXT);
                 CREATE TABLE Adobe_images(id_local INTEGER PRIMARY KEY, rootFile INTEGER, masterImage INTEGER);
                 CREATE TABLE Adobe_imageDevelopSettings(image INTEGER, processVersion TEXT, text TEXT);",
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO AgLibraryRootFolder VALUES(1, ?1)",
                [format!("{}/", root.display())],
            )
            .unwrap();
        connection
            .execute("INSERT INTO AgLibraryFolder VALUES(1, 1, '')", [])
            .unwrap();
        connection
            .execute("INSERT INTO AgLibraryFile VALUES(1, 1, 'photo', 'CR3')", [])
            .unwrap();
        connection
            .execute("INSERT INTO Adobe_images VALUES(1, 1, NULL)", [])
            .unwrap();
        connection
            .execute(
                "INSERT INTO Adobe_imageDevelopSettings VALUES(1, '15.4', ?1)",
                [recipe],
            )
            .unwrap();
        fixture
    }

    #[test]
    fn canonical_digest_is_stable_and_changes_with_recipe() {
        let recipe = DevelopRecipe {
            exposure_ev: Some(1.0),
            ..DevelopRecipe::default()
        };
        assert_eq!(recipe.digest().unwrap(), recipe.clone().digest().unwrap());
        let changed = DevelopRecipe {
            exposure_ev: Some(2.0),
            ..DevelopRecipe::default()
        };
        assert_ne!(recipe.digest().unwrap(), changed.digest().unwrap());
    }

    #[test]
    fn normalization_keeps_only_supported_non_defaults() {
        let values = Parser::new(
            "s={Exposure2012=1,Contrast2012=0,Unknown=7,ToneCurvePV2012={0,0,255,255}}",
        )
        .parse_assignment()
        .unwrap();
        let (recipe, unsupported, warnings) =
            normalize_recipe("15.4", &values, ImportCapabilities::ALL);
        assert_eq!(recipe.exposure_ev, Some(1.0));
        assert_eq!(recipe.contrast, None);
        assert_eq!(recipe.tone_curve, None);
        assert_eq!(unsupported, ["Unknown"]);
        assert!(warnings.is_empty());
    }

    #[test]
    fn phase_b_normalization_is_stable_across_catalog_process_versions() {
        let values = Parser::new(
            "s={Exposure2012=1.25,WhiteBalance='Custom',CustomTemperature=7200,CustomTint=12,IncrementalTemperature=4,IncrementalTint=-3,ToneCurve={0,0,128,160,255,255},ToneCurvePV2012Red={0,0,255,240}}",
        )
        .parse_assignment()
        .unwrap();
        for version in ["6.7", "11.0", "15.4"] {
            let (recipe, unsupported, warnings) =
                normalize_recipe(version, &values, ImportCapabilities::ALL);
            assert_eq!(recipe.exposure_ev, Some(1.25));
            assert_eq!(recipe.white_balance.unwrap().temperature, 7200.0);
            assert!(recipe.white_balance_increment.is_none());
            assert_eq!(recipe.tone_curve.as_ref().unwrap().len(), 3);
            assert!(recipe.tone_curve_red.is_some());
            assert!(unsupported.is_empty(), "{version}: {unsupported:?}");
            assert!(warnings.is_empty(), "{version}: {warnings:?}");
        }
    }

    #[test]
    fn incremental_white_balance_is_used_only_without_an_absolute_family() {
        let incremental = Parser::new(
            "s={WhiteBalance='As Shot',CustomIncrementalTemperature=8,CustomIncrementalTint=-2}",
        )
        .parse_assignment()
        .unwrap();
        let (recipe, _, _) = normalize_recipe("15.4", &incremental, ImportCapabilities::ALL);
        assert!(recipe.white_balance.is_none());
        assert_eq!(recipe.white_balance_increment.unwrap().temperature, 8.0);

        let absolute = Parser::new(
            "s={WhiteBalance='Custom',CustomTemperature=7000,CustomTint=4,CustomIncrementalTemperature=8}",
        )
        .parse_assignment()
        .unwrap();
        let (recipe, _, _) = normalize_recipe("15.4", &absolute, ImportCapabilities::ALL);
        assert!(recipe.white_balance.is_some());
        assert!(recipe.white_balance_increment.is_none());
    }

    #[test]
    fn phase_c_controls_are_normalized_without_adobe_profile_payloads() {
        let values = Parser::new(
            "s={CameraProfile='Adobe Standard',CameraProfileDigest='secret',Sharpness=40,SharpenRadius=1,LuminanceSmoothing=20,ColorNoiseReduction=25,Texture=10,Dehaze=5,ConvertToGrayscale=true,GrayMixerRed=20,PostCropVignetteAmount=-15,ParametricHighlights=10}",
        )
        .parse_assignment()
        .unwrap();
        let (recipe, unsupported, warnings) =
            normalize_recipe("15.4", &values, ImportCapabilities::ALL);

        assert_eq!(recipe.detail.unwrap().sharpening, 40.0);
        assert_eq!(recipe.texture, Some(10.0));
        assert!(recipe.monochrome.is_some());
        assert_eq!(recipe.vignette.unwrap().amount, -15.0);
        assert_eq!(recipe.parametric_curve.unwrap()[0], 10.0);
        assert!(unsupported.contains(&"CameraProfile".to_owned()));
        assert!(
            warnings
                .iter()
                .any(|warning| warning.contains("no compatible resolver"))
        );
    }

    #[test]
    fn phase_d_omits_unselected_upright_and_normalizes_heal_manual_lens_and_grain() {
        let values = Parser::new(
            "s={UprightTransform_4='1,0,0.1,0,1,0,0,0,1',LensManualDistortionAmount=12,RetouchInfo={{spotType='heal',centerX=0.5,centerY=0.5,sourceX=0.25,sourceY=0.25,radius=0.1,opacity=0.8}},GrainSize=57,GrainFrequency=81}",
        )
        .parse_assignment()
        .unwrap();
        let (recipe, unsupported, warnings) =
            normalize_recipe("15.4", &values, ImportCapabilities::ALL);

        assert!(recipe.upright_homography.is_none());
        assert_eq!(recipe.lens_distortion, Some(12.0));
        assert_eq!(recipe.heal_spots.len(), 1);
        assert_eq!(recipe.heal_spots[0].kind, HealKind::Heal);
        assert_eq!(recipe.heal_spots[0].opacity, 0.8);
        assert_eq!(recipe.grain.unwrap().roughness, 81.0);
        assert!(unsupported.is_empty(), "{unsupported:?}");
        assert!(
            warnings
                .iter()
                .any(|warning| warning.contains("active-mode"))
        );
        assert!(
            warnings
                .iter()
                .any(|warning| warning.contains("grain amount"))
        );
    }

    #[test]
    fn retouch_areas_is_canonical_and_legacy_is_only_a_fallback() {
        let values = Parser::new(
            "s={RetouchAreas={{SpotType='clone',CenterX=0.5,CenterY=0.6,SourceX=0.2,SourceY=0.3,Radius=0.1,Opacity=0.7,Feather=25}},RetouchInfo={{spotType='heal',centerX=0.1,centerY=0.1,sourceX=0.9,sourceY=0.9,radius=0.2}}}",
        )
        .parse_assignment()
        .unwrap();
        let (recipe, _, _) = normalize_recipe("15.4", &values, ImportCapabilities::ALL);

        assert_eq!(recipe.heal_spots.len(), 1);
        assert_eq!(recipe.heal_spots[0].kind, HealKind::Clone);
        assert_eq!(recipe.heal_spots[0].destination, [0.5, 0.6]);
        assert_eq!(recipe.heal_spots[0].feather, 0.25);
    }

    #[test]
    fn active_unresolved_lens_profiles_warn_but_inactive_profiles_do_not() {
        let inactive = Parser::new("s={LensProfileEnable=0,LensProfileName='Adobe Lens Profile'}")
            .parse_assignment()
            .unwrap();
        let (_, _, warnings) = normalize_recipe("15.4", &inactive, ImportCapabilities::ALL);
        assert!(warnings.is_empty());

        let active = Parser::new("s={LensProfileEnable=1,LensProfileName='Adobe Lens Profile'}")
            .parse_assignment()
            .unwrap();
        let (recipe, _, warnings) = normalize_recipe("15.4", &active, ImportCapabilities::ALL);
        assert!(recipe.lens_distortion.is_none());
        assert!(
            warnings
                .iter()
                .any(|warning| warning.contains("lens profile"))
        );
    }

    #[test]
    fn singular_homography_is_rejected_by_recipe_validation() {
        let recipe = DevelopRecipe {
            upright_homography: Some([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            ..DevelopRecipe::default()
        };
        assert!(recipe.validate().is_err());
    }

    #[test]
    fn phase_d_inventory_assumptions_are_guarded() {
        let inventory: serde_json::Value =
            serde_json::from_str(include_str!("../lightroom-inventory.json")).unwrap();
        let properties = inventory["properties"].as_object().unwrap();

        assert_eq!(properties["LensProfileEnable"]["numeric_min"], 0.0);
        assert_eq!(properties["LensProfileEnable"]["numeric_max"], 0.0);
        assert_eq!(properties["UprightCenterMode"]["numeric_max"], 0.0);
        assert_eq!(properties["RetouchAreas"]["present"], 1);
        assert_eq!(properties["GrainSize"]["numeric_min"], 25.0);
        assert_eq!(properties["GrainSize"]["numeric_max"], 57.0);
        assert_eq!(properties["GrainFrequency"]["present"], 1);
        let common = |name: &str| {
            properties[name]["most_common_values"][0]["value"]
                .as_str()
                .unwrap()
        };
        let assert_only_common = |name: &str, expected: &str| {
            assert_eq!(properties[name]["distinct_values"], 1);
            assert_eq!(common(name), expected);
            assert_eq!(
                properties[name]["most_common_values"][0]["count"], properties[name]["present"],
                "{name} gained a non-default value; reevaluate its import policy"
            );
        };
        for empty_table in ["AILook", "PointColors", "RedEyeInfo", "LensBlur"] {
            assert_only_common(empty_table, "<table:0>");
        }
        assert_only_common("Look.Parameters.PointColors", "<table:0>");
        assert_only_common("Look.Parameters.LensBlur.Active", "false");
        for default_zero in [
            "ColorGradeGlobalHue",
            "ColorGradeGlobalLum",
            "ColorGradeGlobalSat",
            "ColorGradeHighlightLum",
            "ColorGradeMidtoneHue",
            "ColorGradeMidtoneLum",
            "ColorGradeMidtoneSat",
            "ColorGradeShadowLum",
        ] {
            assert_eq!(properties[default_zero]["numeric_min"], 0.0);
            assert_eq!(properties[default_zero]["numeric_max"], 0.0);
        }
        assert_eq!(properties["ColorGradeBlending"]["numeric_min"], 50.0);
        assert_eq!(properties["ColorGradeBlending"]["numeric_max"], 50.0);
        assert_eq!(properties["RetouchAreas.[0].Masks"]["present"], 1);
        assert_eq!(common("RetouchAreas.[0].Masks"), "<table:1>");
    }

    #[test]
    fn identity_upright_and_default_grain_are_not_stored() {
        let values = Parser::new("s={UprightTransform_4='1,0,0,0,1,0,0,0,1',GrainSize=25}")
            .parse_assignment()
            .unwrap();
        let (recipe, _, _) = normalize_recipe("15.4", &values, ImportCapabilities::ALL);
        assert!(recipe.upright_homography.is_none());
        assert!(recipe.grain.is_none());
    }

    #[test]
    fn existing_catalog_is_never_created_implicitly() {
        let directory = tempfile::tempdir().unwrap();
        assert!(load_recipes(directory.path()).unwrap().is_empty());
        assert!(!directory.path().join(CATALOG_DIR).exists());
    }

    #[test]
    fn import_is_read_only_minimal_and_reloadable() {
        let root = tempfile::tempdir().unwrap();
        fs::write(root.path().join("photo.CR3"), b"test").unwrap();
        let fixture = lightroom_fixture(
            root.path(),
            "s={Exposure2012=1.5,Contrast2012=0,UnknownAdobeThing=42}",
        );
        let lightroom_path = fixture.path().join("fixture.lrcat");
        let before = fs::read(&lightroom_path).unwrap();

        let report =
            import_lightroom_inner(root.path(), &lightroom_path, true, ImportCapabilities::ALL)
                .unwrap();

        assert_eq!(report.imported_recipes, 1);
        assert_eq!(
            report.unsupported_property_occurrences["UnknownAdobeThing"],
            1
        );
        assert_eq!(fs::read(&lightroom_path).unwrap(), before);
        assert!(!lightroom_path.with_extension("lrcat-wal").exists());
        assert!(!lightroom_path.with_extension("lrcat-shm").exists());
        let recipes = load_recipes(root.path()).unwrap();
        let entry = &recipes[&root.path().join("photo.CR3")];
        let expected = DevelopRecipe {
            exposure_ev: Some(1.5),
            ..DevelopRecipe::default()
        };
        assert_eq!(entry.digest, expected.digest().unwrap());

        let connection =
            Connection::open(root.path().join(CATALOG_DIR).join(CATALOG_FILE)).unwrap();
        let stored: String = connection
            .query_row("SELECT recipe_json FROM develop_recipes", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert!(!stored.contains("UnknownAdobeThing"));
        assert!(!stored.contains("ProcessVersion"));
    }

    #[test]
    fn failed_reimport_rolls_back_existing_recipes() {
        let root = tempfile::tempdir().unwrap();
        fs::write(root.path().join("photo.CR3"), b"test").unwrap();
        let fixture = lightroom_fixture(root.path(), "s={Exposure2012=1}");
        let lightroom_path = fixture.path().join("fixture.lrcat");
        import_lightroom_inner(root.path(), &lightroom_path, true, ImportCapabilities::ALL)
            .unwrap();
        let before = load_recipes(root.path()).unwrap()[&root.path().join("photo.CR3")].digest;
        Connection::open(&lightroom_path)
            .unwrap()
            .execute(
                "UPDATE Adobe_imageDevelopSettings SET text='not a recipe'",
                [],
            )
            .unwrap();

        assert!(
            import_lightroom_inner(root.path(), &lightroom_path, true, ImportCapabilities::ALL,)
                .is_err()
        );
        let after = load_recipes(root.path()).unwrap()[&root.path().join("photo.CR3")].digest;
        assert_eq!(after, before);
    }

    #[test]
    fn dry_run_and_failed_first_import_do_not_create_a_catalog() {
        let root = tempfile::tempdir().unwrap();
        fs::write(root.path().join("photo.CR3"), b"test").unwrap();
        let fixture = lightroom_fixture(root.path(), "s={Exposure2012=1}");
        let lightroom_path = fixture.path().join("fixture.lrcat");
        analyze_lightroom(root.path(), &lightroom_path).unwrap();
        assert!(!root.path().join(CATALOG_DIR).exists());

        Connection::open(&lightroom_path)
            .unwrap()
            .execute("UPDATE Adobe_imageDevelopSettings SET text='broken'", [])
            .unwrap();
        assert!(import_lightroom(root.path(), &lightroom_path).is_err());
        assert!(!root.path().join(CATALOG_DIR).exists());
    }

    #[test]
    fn source_resolution_rejects_parent_traversal() {
        let parent = tempfile::tempdir().unwrap();
        let root = parent.path().join("root");
        fs::create_dir(&root).unwrap();
        fs::write(parent.path().join("outside.CR3"), b"outside").unwrap();
        assert!(
            resolve_relative_source(&root, root.to_str().unwrap(), "..", "outside.CR3").is_none()
        );
    }

    #[cfg(unix)]
    #[test]
    fn relative_path_encoding_round_trips_non_unicode_and_backslashes() {
        use std::os::unix::ffi::OsStrExt as _;
        let path = Path::new(std::ffi::OsStr::from_bytes(b"dir/invalid-\xff\\name.CR3"));
        assert_eq!(
            decode_relative_path(&encode_relative_path(path).unwrap()).unwrap(),
            path
        );
    }

    #[test]
    fn unknown_process_version_imports_no_actions() {
        let values = Parser::new("s={Exposure2012=1}")
            .parse_assignment()
            .unwrap();
        let (recipe, unsupported, warnings) =
            normalize_recipe("99.0", &values, ImportCapabilities::ALL);
        assert!(recipe.is_empty());
        assert!(unsupported.contains(&"UnsupportedProcessVersion:99.0".to_owned()));
        assert_eq!(warnings, ["unsupported process version 99.0"]);
    }

    #[test]
    fn portable_unicode_paths_use_cross_platform_encoding() {
        let path = Path::new("2025/Kanada/photo.CR3");
        let encoded = encode_relative_path(path).unwrap();
        assert_eq!(encoded.first(), Some(&b'p'));
        assert_eq!(decode_portable_path(&encoded).unwrap(), path);
    }

    #[test]
    fn incompatible_owned_schema_is_not_modified() {
        let root = tempfile::tempdir().unwrap();
        fs::write(root.path().join("photo.CR3"), b"test").unwrap();
        let fixture = lightroom_fixture(root.path(), "s={Exposure2012=1}");
        let directory = root.path().join(CATALOG_DIR);
        fs::create_dir(&directory).unwrap();
        let path = directory.join(CATALOG_FILE);
        let connection = Connection::open(&path).unwrap();
        connection
            .execute_batch(&format!(
                "PRAGMA application_id={SQLITE_APPLICATION_ID};
                 PRAGMA user_version=2;
                 CREATE TABLE marker(value TEXT);
                 INSERT INTO marker VALUES('untouched');"
            ))
            .unwrap();
        drop(connection);
        let before = fs::read(&path).unwrap();

        assert!(
            import_lightroom_inner(
                root.path(),
                &fixture.path().join("fixture.lrcat"),
                true,
                ImportCapabilities::ALL,
            )
            .is_err()
        );
        assert_eq!(fs::read(path).unwrap(), before);
    }

    #[test]
    fn older_renderer_catalog_remains_readable_and_reimport_updates_revision() {
        let root = tempfile::tempdir().unwrap();
        fs::write(root.path().join("photo.CR3"), b"test").unwrap();
        let fixture = lightroom_fixture(root.path(), "s={Exposure2012=1}");
        let lightroom_path = fixture.path().join("fixture.lrcat");
        import_lightroom_inner(root.path(), &lightroom_path, true, ImportCapabilities::ALL)
            .unwrap();
        let path = catalog_path(root.path());
        let connection = Connection::open(&path).unwrap();
        connection
            .execute("UPDATE catalog_meta SET renderer_model=?1", [3_u32])
            .unwrap();
        drop(connection);

        assert_eq!(load_recipes(root.path()).unwrap().len(), 1);
        import_lightroom_inner(root.path(), &lightroom_path, true, ImportCapabilities::ALL)
            .unwrap();
        let connection = Connection::open(path).unwrap();
        let renderer_model: u32 = connection
            .query_row("SELECT renderer_model FROM catalog_meta", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(renderer_model, RENDERER_MODEL_VERSION);
    }

    #[test]
    fn conflicting_master_rows_are_reported_and_not_applied() {
        let root = tempfile::tempdir().unwrap();
        fs::write(root.path().join("photo.CR3"), b"test").unwrap();
        let fixture = lightroom_fixture(root.path(), "s={Exposure2012=1}");
        let path = fixture.path().join("fixture.lrcat");
        let connection = Connection::open(&path).unwrap();
        connection
            .execute("INSERT INTO Adobe_images VALUES(2, 1, NULL)", [])
            .unwrap();
        connection
            .execute(
                "INSERT INTO Adobe_imageDevelopSettings VALUES(2, '15.4', 's={Exposure2012=2}')",
                [],
            )
            .unwrap();
        drop(connection);
        let report =
            import_lightroom_inner(root.path(), &path, false, ImportCapabilities::ALL).unwrap();
        assert_eq!(report.conflicting_sources, 1);
        assert!(import_lightroom_inner(root.path(), &path, true, ImportCapabilities::ALL).is_err());
        assert!(!root.path().join(CATALOG_DIR).exists());
    }

    #[test]
    fn parser_rejects_trailing_garbage() {
        assert!(Parser::new("s={} garbage").parse_assignment().is_err());
    }
}
