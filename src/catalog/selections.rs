use std::{
    collections::{BTreeMap, HashSet},
    path::{Component, Path, PathBuf},
    time::Duration,
};

use anyhow::{Context, Result, bail};
use rusqlite::{Connection, OpenFlags, params};

use super::{
    CATALOG_DIR, catalog_path, decode_relative_path, encode_relative_path, initialize_schema,
    validate_schema,
};

/// Production busy timeout for the interactive per-directory save. Editors
/// toggle on the UI thread while maintenance writers commit in the background,
/// so a contended save waits instead of failing immediately.
const SAVE_BUSY_TIMEOUT: Duration = Duration::from_secs(30);

/// Largest accepted selection name, counted in characters.
const NAME_MAX_CHARS: usize = 64;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AlbumSelection {
    pub name: String,
    /// Library-absolute member paths (matching `load_recipes` /
    /// `load_hidden_subtrees`). Paths whose file has disappeared stay here
    /// (dangling) — the grid filters them against the live listing.
    pub members: Vec<PathBuf>,
    pub created_at: String,
}

/// Whole-catalog read for the startup load. Members are returned as stored;
/// dangling paths are NOT filtered here.
pub fn load_album_selections(root: &Path) -> Result<BTreeMap<PathBuf, Vec<AlbumSelection>>> {
    let path = catalog_path(root);
    if !path.is_file() {
        return Ok(BTreeMap::new());
    }
    let connection = Connection::open_with_flags(
        path,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )?;
    validate_schema(&connection)?;
    let mut query = connection.prepare(
        "SELECT s.directory_relative_path, s.name, s.created_at, m.asset_relative_path
         FROM album_selections s
         LEFT JOIN album_selection_members m ON m.selection_id = s.id
         ORDER BY s.created_at, s.id, m.rowid",
    )?;
    let rows = query.query_map([], |row| {
        Ok((
            row.get::<_, Vec<u8>>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, Option<Vec<u8>>>(3)?,
        ))
    })?;
    let mut by_directory: BTreeMap<PathBuf, Vec<AlbumSelection>> = BTreeMap::new();
    for row in rows {
        let (directory, name, created_at, member) = row?;
        let directory_relative = decode_relative_path(&directory)
            .with_context(|| format!("cannot decode selection album path for {name:?}"))?;
        let album_selections = by_directory
            .entry(absolute_album_path(root, &directory_relative))
            .or_default();
        if album_selections
            .last()
            .map(|last| last.name != name)
            .unwrap_or(true)
        {
            album_selections.push(AlbumSelection {
                name,
                members: Vec::new(),
                created_at,
            });
        }
        let Some(member) = member else {
            continue;
        };
        let member = decode_relative_path(&member).with_context(|| {
            format!(
                "cannot decode member of selection {:?} in album {}",
                album_selections.last().expect("just pushed").name,
                directory_relative.display()
            )
        })?;
        album_selections
            .last_mut()
            .expect("just pushed")
            .members
            .push(absolute_album_path(root, &member));
    }
    Ok(by_directory)
}

/// The ONLY database writer for a directory: snapshot-and-replace in one
/// transaction. `dir` must be inside the library; the library root itself
/// (the empty relative path) is a valid album.
pub fn save_album_selections(root: &Path, dir: &Path, selections: &[AlbumSelection]) -> Result<()> {
    save_album_selections_with_busy_timeout(root, dir, selections, SAVE_BUSY_TIMEOUT)
}

pub(crate) fn save_album_selections_with_busy_timeout(
    root: &Path,
    dir: &Path,
    selections: &[AlbumSelection],
    busy_timeout: Duration,
) -> Result<()> {
    validate_selections(selections)?;
    let relative_dir = relative_album_path(root, dir)?;
    let directory = root.join(CATALOG_DIR);
    std::fs::create_dir_all(&directory)?;
    let mut connection = Connection::open(catalog_path(root))?;
    connection.busy_timeout(busy_timeout)?;
    initialize_schema(&mut connection)?;
    let transaction = connection.transaction()?;
    let encoded_dir = encode_relative_path(&relative_dir)?;
    // Members are deleted explicitly: PRAGMA foreign_keys=ON is per-connection,
    // so the cascade must never be load-bearing.
    transaction.execute(
        "DELETE FROM album_selection_members WHERE selection_id IN
             (SELECT id FROM album_selections WHERE directory_relative_path = ?1)",
        [&encoded_dir],
    )?;
    transaction.execute(
        "DELETE FROM album_selections WHERE directory_relative_path = ?1",
        [&encoded_dir],
    )?;
    for selection in selections {
        transaction.execute(
            "INSERT INTO album_selections(directory_relative_path, name, created_at) \
             VALUES (?1, ?2, ?3)",
            params![encoded_dir, selection.name, selection.created_at],
        )?;
        let selection_id = transaction.last_insert_rowid();
        for member in &selection.members {
            let relative_member = relative_album_path(root, member)?;
            if relative_member.as_os_str().is_empty() {
                bail!("a selection member cannot be the library root itself");
            }
            let encoded_member = encode_relative_path(&relative_member)?;
            // Path identity makes duplicates the same fact, not an error.
            transaction.execute(
                "INSERT OR IGNORE INTO album_selection_members\
                 (selection_id, asset_relative_path, added_at) \
                 VALUES (?1, ?2, datetime('now'))",
                params![selection_id, encoded_member],
            )?;
        }
    }
    transaction.commit()?;
    Ok(())
}

fn validate_selections(selections: &[AlbumSelection]) -> Result<()> {
    let mut seen = HashSet::new();
    for selection in selections {
        let name = selection.name.as_str();
        if name.is_empty() {
            bail!("selection name cannot be empty");
        }
        if name.trim() != name {
            bail!("selection name {name:?} must not have leading or trailing whitespace");
        }
        if name.chars().count() > NAME_MAX_CHARS {
            bail!("selection name {name:?} exceeds {NAME_MAX_CHARS} characters");
        }
        if !seen.insert(name) {
            bail!("duplicate selection name {name:?}");
        }
    }
    Ok(())
}

fn relative_album_path(root: &Path, path: &Path) -> Result<PathBuf> {
    let relative = path.strip_prefix(root).with_context(|| {
        format!(
            "selection album {} is outside library {}",
            path.display(),
            root.display()
        )
    })?;
    if relative.components().next().is_some()
        && !relative
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
    {
        bail!(
            "selection album {} is not a plain library path",
            path.display()
        );
    }
    Ok(relative.to_path_buf())
}

fn absolute_album_path(root: &Path, relative: &Path) -> PathBuf {
    if relative.as_os_str().is_empty() {
        root.to_path_buf()
    } else {
        root.join(relative)
    }
}
