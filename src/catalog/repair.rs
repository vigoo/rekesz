//! One-time repair of recipes imported by the pre-defaults-fix Lightroom
//! importer: re-derive the target recipe set from the .lrcat with the
//! production pipeline and diff it against the stored rows. The deletion
//! set is never derived from stored JSON alone — the flavor inputs that
//! decide whether a value is a serialized default (process version, blob
//! keys, rawness) only exist in the .lrcat.
//!
//! Invoked through the env-gated live-acceptance test harness rather than
//! the application: the shipped binary intentionally has no repair UI, so
//! this module is dead code outside test builds.
#![cfg_attr(not(test), allow(dead_code))]

use anyhow::{Context, Result, bail};
use rusqlite::Connection;
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use super::{DevelopRecipe, ImportReport, catalog_path, decode_relative_path};

/// Summary of what the production pipeline derives from the .lrcat snapshot,
/// mirroring the importer's counters so dry-run output reconciles with the
/// Stage 3 numbers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DerivationSummary {
    pub lightroom_images: u64,
    pub matched: u64,
    pub imported: u64,
    pub empty: u64,
    pub virtual_copies: u64,
    pub missing_or_outside_root: u64,
    pub conflicting_sources: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RepairReport {
    pub derivation: DerivationSummary,
    /// Stored rows whose re-derived recipe is `None` — the false flags.
    pub deleted: u64,
    /// Stored rows whose re-derived recipe differs — rewritten in place
    /// (defaults stripped, edits kept) with a recomputed digest.
    pub updated: u64,
    /// Stored rows whose re-derived recipe equals the stored one.
    pub kept_identical: u64,
    /// Stored rows the .lrcat snapshot cannot re-derive (file absent from
    /// the catalog, unresolved root, virtual copy). Never touched.
    pub kept_unprovable: u64,
    /// Derived recipes with no stored row — a future import's inserts, not
    /// the repair's job.
    pub not_yet_imported: u64,
    /// Assets removed by the same garbage-collection predicate the importer
    /// applies (only meaningful when applied).
    pub orphaned_assets_removed: u64,
    /// Backup written before the mutating transaction (only when applied).
    pub backup_path: Option<PathBuf>,
    pub applied: bool,
}

struct StoredRecipeRow {
    asset_id: i64,
    recipe_json: String,
    recipe_digest: Vec<u8>,
}

/// Re-derive every recipe from `lightroom_catalog` and, when `apply`, bring
/// the stored `develop_recipes` in line: delete defaults-only rows, update
/// rows whose content changed, and keep everything else — including rows the
/// snapshot cannot re-derive. People/face data is never touched and no rows
/// are inserted.
///
/// Runs read-only unless `apply`. In apply mode a validated backup is
/// written next to the catalog before the mutating transaction. The .lrcat
/// is staged (main + WAL copied) before opening, so derivation always sees
/// the freshest committed state and the original catalog is never opened.
/// Concurrent writers are refused: the pre-transaction checkpoint fails with
/// `busy` if anything else holds the database, and the caller must still
/// have closed the application.
pub fn repair_lightroom_defaults(
    root: &Path,
    lightroom_catalog: &Path,
    apply: bool,
) -> Result<RepairReport> {
    let root = root
        .canonicalize()
        .with_context(|| format!("cannot resolve opened root {}", root.display()))?;
    let destination = catalog_path(&root);
    if !destination.is_file() {
        bail!("no Rekesz catalog at {} to repair", destination.display());
    }

    // Stage the .lrcat (main + WAL sidecars) and open the private copy
    // read-write so SQLite recovers the WAL into it: derivation sees the
    // freshest committed state, and the original is never touched. The guard
    // is declared first so the staged files outlive the connection.
    let staging = stage_lightroom_catalog(lightroom_catalog)?;
    let lightroom = Connection::open(
        staging.path().join(
            lightroom_catalog
                .file_name()
                .context("Lightroom catalog path has no file name")?,
        ),
    )?;
    lightroom.busy_timeout(std::time::Duration::from_secs(30))?;

    let mut rekesz = connect_rekesz(&destination, apply)?;
    super::initialize_schema(&mut rekesz)?;

    // Every stored row must be a Lightroom import artifact. A NULL
    // imported_at would mean some other writer exists; refuse to guess.
    // (Unreachable on schema v7, which declares the column NOT NULL; this is
    // defense-in-depth against future schema drift.)
    let total_rows: i64 =
        rekesz.query_row("SELECT COUNT(*) FROM develop_recipes", [], |row| row.get(0))?;
    let imported_stamped: i64 = rekesz.query_row(
        "SELECT COUNT(*) FROM develop_recipes WHERE imported_at IS NOT NULL",
        [],
        |row| row.get(0),
    )?;
    if imported_stamped != total_rows {
        bail!(
            "{total_rows} stored recipes but only {imported_stamped} carry imported_at; \
             a non-import recipe writer exists, refusing to repair"
        );
    }
    // Pre-flight: every stored row must parse, validate, and carry a matching
    // digest before anything is decided or written.
    super::load_recipes(&root)?;

    let mut report = RepairReport::default();
    let mut import_report = ImportReport::default();
    let plan = super::plan_lightroom_recipes(
        &root,
        &lightroom,
        super::CURRENT_IMPORT_CAPABILITIES,
        &mut import_report,
    )?;
    drop(lightroom);
    let derivation = DerivationSummary {
        lightroom_images: import_report.lightroom_images,
        matched: import_report.matched_files,
        imported: import_report.imported_recipes,
        empty: import_report.empty_supported_recipes,
        virtual_copies: import_report.skipped_virtual_copies,
        missing_or_outside_root: import_report.missing_or_outside_root,
        conflicting_sources: import_report.conflicting_sources,
    };
    report.derivation = derivation;
    if derivation.conflicting_sources > 0 {
        bail!(
            "refusing to repair with {} conflicting Lightroom source rows",
            derivation.conflicting_sources
        );
    }

    let mut statement = rekesz.prepare(
        "SELECT recipe.asset_id, assets.relative_path, recipe.recipe_json, \
                recipe.recipe_digest \
         FROM develop_recipes recipe JOIN assets ON assets.id = recipe.asset_id",
    )?;
    let mut stored: HashMap<Vec<u8>, StoredRecipeRow> = HashMap::new();
    let mut rows = statement.query([])?;
    while let Some(row) = rows.next()? {
        let asset_id: i64 = row.get(0)?;
        let relative_key: Vec<u8> = row.get(1)?;
        stored.insert(
            relative_key,
            StoredRecipeRow {
                asset_id,
                recipe_json: row.get(2)?,
                recipe_digest: row.get(3)?,
            },
        );
    }
    drop(rows);
    drop(statement);

    let mut delete_ids = Vec::new();
    let mut update_ids = Vec::new();
    for (relative_key, row) in &stored {
        let derived = plan.recipes.get(relative_key);
        match derived {
            None => {
                if plan.matched_keys.contains(relative_key) {
                    delete_ids.push(row.asset_id);
                } else {
                    report.kept_unprovable += 1;
                }
            }
            Some(derived_recipe) => {
                let stored_recipe: DevelopRecipe = serde_json::from_str(&row.recipe_json)
                    .with_context(|| {
                        format!(
                            "stored recipe_json for {} failed to parse",
                            decode_relative_path(relative_key)
                                .map(|path| path.display().to_string())
                                .unwrap_or_else(|_| "unencodable path".to_owned())
                        )
                    })?;
                if &stored_recipe == derived_recipe
                    && row.recipe_digest == derived_recipe.digest()?
                {
                    report.kept_identical += 1;
                } else {
                    update_ids.push((row.asset_id, (*derived_recipe).clone()));
                }
            }
        }
    }
    report.deleted = delete_ids.len() as u64;
    report.updated = update_ids.len() as u64;
    report.not_yet_imported = plan
        .recipes
        .keys()
        .filter(|key| !stored.contains_key(*key))
        .count() as u64;
    report.applied = apply;
    if !apply {
        return Ok(report);
    }

    let backup_path = backup_catalog(&destination, &mut rekesz)?;
    report.backup_path = Some(backup_path);

    let transaction = rekesz.transaction()?;
    for chunk in delete_ids.chunks(512) {
        transaction.execute(
            &format!(
                "DELETE FROM develop_recipes WHERE asset_id IN ({})",
                vec!["?"; chunk.len()].join(",")
            ),
            rusqlite::params_from_iter(chunk.iter()),
        )?;
    }
    for (asset_id, derived_recipe) in &update_ids {
        derived_recipe
            .validate()
            .with_context(|| format!("derived recipe for asset {asset_id} failed validation"))?;
        let json = serde_json::to_string(derived_recipe)?;
        let digest = derived_recipe.digest()?;
        transaction.execute(
            "UPDATE develop_recipes SET recipe_json=?1, recipe_digest=?2 WHERE asset_id=?3",
            rusqlite::params![json, digest.as_slice(), asset_id],
        )?;
    }
    let orphaned_assets_removed = transaction.execute(
        "DELETE FROM assets WHERE id NOT IN (SELECT asset_id FROM develop_recipes) \
         AND id NOT IN (SELECT asset_id FROM faces) \
         AND id NOT IN (SELECT asset_id FROM face_analysis_state)",
        [],
    )? as u64;
    report.orphaned_assets_removed = orphaned_assets_removed;
    transaction.commit()?;

    // Post-apply verification: digests must validate, every updated row must
    // hold exactly the derived content, and the population must match the
    // decision exactly.
    let reloaded = super::load_recipes(&root)?;
    let expected_rows = total_rows - report.deleted as i64;
    if reloaded.len() as i64 != expected_rows {
        bail!(
            "post-repair recipe count {} does not match {} - {} deleted",
            reloaded.len(),
            total_rows,
            report.deleted
        );
    }
    // Per-key content check: every updated asset must hold exactly its
    // derived JSON and digest.
    {
        let mut statement = rekesz
            .prepare("SELECT recipe_json, recipe_digest FROM develop_recipes WHERE asset_id=?1")?;
        for (asset_id, derived_recipe) in &update_ids {
            let (json, digest): (String, Vec<u8>) = statement
                .query_row(rusqlite::params![asset_id], |row| {
                    Ok((row.get(0)?, row.get(1)?))
                })
                .with_context(|| {
                    format!("post-repair verification: asset {asset_id} lost its recipe row")
                })?;
            if json != serde_json::to_string(derived_recipe)?
                || digest != derived_recipe.digest()?
            {
                bail!(
                    "post-repair verification: asset {asset_id} does not hold its derived recipe"
                );
            }
        }
    }
    let integrity: String = rekesz.query_row("PRAGMA integrity_check", [], |row| row.get(0))?;
    if integrity != "ok" {
        bail!("post-repair integrity_check reported: {integrity}");
    }
    Ok(report)
}

fn connect_rekesz(destination: &Path, apply: bool) -> Result<Connection> {
    let flags = if apply {
        rusqlite::OpenFlags::default()
    } else {
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY
    };
    let rekesz = Connection::open_with_flags(destination, flags)
        .with_context(|| format!("cannot open Rekesz catalog {}", destination.display()))?;
    rekesz.busy_timeout(std::time::Duration::from_secs(5))?;
    Ok(rekesz)
}

/// Copy the Lightroom catalog (main database plus `-wal`/`-shm` sidecars)
/// into a private staging directory next to it. Opening the copy lets
/// SQLite recover the hot WAL, so derivation covers the freshest committed
/// state without ever opening the original catalog.
fn stage_lightroom_catalog(lightroom_catalog: &Path) -> Result<tempfile::TempDir> {
    let parent = lightroom_catalog
        .parent()
        .context("Lightroom catalog has no parent directory")?;
    let file_name = lightroom_catalog
        .file_name()
        .context("Lightroom catalog path has no file name")?;
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock before Unix epoch")?
        .as_nanos();
    let staging = tempfile::Builder::new()
        .prefix(&format!(".rekesz-repair-staging-{nanos}-"))
        .tempdir_in(parent)
        .context("cannot create Lightroom staging directory")?;
    std::fs::copy(lightroom_catalog, staging.path().join(file_name))
        .with_context(|| format!("cannot stage {}", lightroom_catalog.display()))?;
    for suffix in ["-wal", "-shm"] {
        let sidecar = PathBuf::from(format!("{}{suffix}", lightroom_catalog.display()));
        if sidecar.exists() {
            std::fs::copy(
                &sidecar,
                staging
                    .path()
                    .join(format!("{}{suffix}", file_name.to_string_lossy())),
            )
            .with_context(|| format!("cannot stage {}", sidecar.display()))?;
        }
    }
    Ok(staging)
}

/// WAL-checkpoint (refusing to proceed if another connection is busy), copy
/// the catalog next to its siblings using the established backup naming,
/// atomically rename it into place, and verify it loads. Restoring that one
/// file rolls back the repair.
fn backup_catalog(destination: &Path, rekesz: &mut Connection) -> Result<PathBuf> {
    let (busy, _log, _checkpointed): (i64, i64, i64) =
        rekesz.query_row("PRAGMA wal_checkpoint(TRUNCATE)", [], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?))
        })?;
    if busy != 0 {
        bail!(
            "WAL checkpoint is busy — another connection is using the catalog; \
             close the application and retry"
        );
    }
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock before Unix epoch")?
        .as_nanos();
    let final_path = destination.with_file_name(format!(
        "catalog.sqlite3.pre-lightroom-repair-{nanos}-backup.sqlite3"
    ));
    let temporary_path = final_path.with_extension("sqlite3.partial");
    std::fs::copy(destination, &temporary_path).with_context(|| {
        format!(
            "failed to back up {} to {}",
            destination.display(),
            temporary_path.display()
        )
    })?;
    // The integrity-check connection must be closed before the rename below:
    // Windows opens SQLite files without FILE_SHARE_DELETE, so renaming a
    // file with a live handle fails with a sharing violation (os error 32).
    let integrity: String = {
        let backup = Connection::open_with_flags(
            &temporary_path,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
        )?;
        backup.query_row("PRAGMA integrity_check", [], |row| row.get(0))?
    };
    if integrity != "ok" {
        let _ = std::fs::remove_file(&temporary_path);
        bail!("pre-repair backup failed integrity_check: {integrity}");
    }
    std::fs::rename(&temporary_path, &final_path).with_context(|| {
        format!(
            "cannot move backup {} into place at {}",
            temporary_path.display(),
            final_path.display()
        )
    })?;
    Ok(final_path)
}
