// A custom test harness does not apply libtest's implicit dead-code and
// unused-import allowances to the binary under test.
#![cfg_attr(test, allow(dead_code, unused_imports))]

use std::{
    cmp::Reverse,
    collections::{BTreeMap, BinaryHeap},
    env,
    ffi::OsStr,
    fs,
    path::{Path, PathBuf},
    time::SystemTime,
};

use anyhow::{Context, Result, anyhow, bail};
use rusqlite::{Connection, OpenFlags};
use serde::Serialize;

const DEFAULT_CATALOG_DIR: &str = "./Lightroom";
const TOP_VALUES: usize = 12;

#[derive(Debug)]
struct Args {
    catalog: Option<PathBuf>,
    catalog_dir: PathBuf,
    output: Option<PathBuf>,
    check_sidecars: bool,
    path_maps: Vec<(String, PathBuf)>,
}

#[derive(Debug, Clone, PartialEq)]
enum DevelopValue {
    Bool(bool),
    Number(f64),
    String(String),
    Nil,
    Table(Vec<(Option<String>, DevelopValue)>),
}

#[derive(Debug, Default)]
struct PropertyAccumulator {
    present: u64,
    booleans: u64,
    numbers: u64,
    strings: u64,
    nils: u64,
    tables: u64,
    numeric_min: Option<f64>,
    numeric_max: Option<f64>,
    values: BTreeMap<String, u64>,
}

#[derive(Debug, Serialize)]
struct InventoryReport {
    catalog: String,
    catalog_modified_unix_seconds: Option<u64>,
    images: u64,
    root_files: u64,
    virtual_copies: u64,
    develop_recipes: u64,
    recipes_with_adjustments: u64,
    empty_recipes: u64,
    parse_failures: u64,
    parse_failure_examples: Vec<ParseFailure>,
    process_versions: Vec<ValueCount>,
    sidecars: Option<SidecarSummary>,
    properties: BTreeMap<String, PropertyReport>,
}

#[derive(Debug, Serialize)]
struct SidecarSummary {
    files_checked: u64,
    xmp_found: u64,
    files_missing: u64,
    paths_unavailable: u64,
}

#[derive(Debug, Serialize)]
struct ParseFailure {
    error: String,
    recipe: String,
}

#[derive(Debug, Serialize)]
struct PropertyReport {
    present: u64,
    types: TypeCounts,
    numeric_min: Option<f64>,
    numeric_max: Option<f64>,
    distinct_values: usize,
    most_common_values: Vec<ValueCount>,
}

#[derive(Debug, Serialize)]
struct TypeCounts {
    booleans: u64,
    numbers: u64,
    strings: u64,
    nils: u64,
    tables: u64,
}

#[derive(Debug, Serialize)]
struct ValueCount {
    value: String,
    count: u64,
}

#[cfg(not(test))]
fn main() -> Result<()> {
    let args = parse_args()?;
    let catalog = match args.catalog {
        Some(path) => path,
        None => newest_catalog(&args.catalog_dir)?,
    };
    let report = inventory(&catalog, args.check_sidecars, &args.path_maps)?;
    let json = serde_json::to_string_pretty(&report)?;

    if let Some(output) = args.output {
        fs::write(&output, format!("{json}\n"))
            .with_context(|| format!("failed to write {}", output.display()))?;
        eprintln!("wrote {}", output.display());
    } else {
        println!("{json}");
    }
    Ok(())
}

#[cfg(test)]
test_r::enable!();

fn parse_args() -> Result<Args> {
    let mut args = Args {
        catalog: None,
        catalog_dir: PathBuf::from(DEFAULT_CATALOG_DIR),
        output: None,
        check_sidecars: true,
        path_maps: Vec::new(),
    };
    let mut values = env::args_os().skip(1);
    while let Some(arg) = values.next() {
        match arg.to_str() {
            Some("--catalog") => args.catalog = Some(next_path(&mut values, "--catalog")?),
            Some("--catalog-dir") => {
                args.catalog_dir = next_path(&mut values, "--catalog-dir")?;
            }
            Some("--output") => args.output = Some(next_path(&mut values, "--output")?),
            Some("--no-sidecars") => args.check_sidecars = false,
            Some("--path-map") => {
                let mapping = values
                    .next()
                    .ok_or_else(|| anyhow!("--path-map requires FROM=TO"))?;
                let mapping = mapping.to_string_lossy();
                let (from, to) = mapping
                    .split_once('=')
                    .ok_or_else(|| anyhow!("--path-map requires FROM=TO"))?;
                args.path_maps.push((from.to_owned(), PathBuf::from(to)));
            }
            Some("-h" | "--help") => {
                println!(
                    "Usage: lightroom-inventory [--catalog PATH | --catalog-dir DIR] \
                     [--output PATH] [--no-sidecars] [--path-map FROM=TO]\n\n\
                     Catalogs are opened read-only and immutable. --path-map may be repeated."
                );
                std::process::exit(0);
            }
            _ => bail!("unknown argument: {}", arg.to_string_lossy()),
        }
    }
    Ok(args)
}

fn next_path(values: &mut impl Iterator<Item = std::ffi::OsString>, flag: &str) -> Result<PathBuf> {
    values
        .next()
        .map(PathBuf::from)
        .ok_or_else(|| anyhow!("{flag} requires a path"))
}

fn newest_catalog(root: &Path) -> Result<PathBuf> {
    let mut pending = vec![root.to_path_buf()];
    let mut newest: Option<(SystemTime, PathBuf)> = None;
    while let Some(dir) = pending.pop() {
        for entry in fs::read_dir(&dir)
            .with_context(|| format!("failed to read catalog directory {}", dir.display()))?
        {
            let entry = entry?;
            let path = entry.path();
            let file_type = entry.file_type()?;
            if file_type.is_dir() {
                if !path
                    .components()
                    .any(|part| part.as_os_str() == OsStr::new("Corrupt Catalogs"))
                {
                    pending.push(path);
                }
            } else if file_type.is_file()
                && path
                    .extension()
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("lrcat"))
            {
                let modified = entry.metadata()?.modified()?;
                if newest.as_ref().is_none_or(|(time, _)| modified > *time) {
                    newest = Some((modified, path));
                }
            }
        }
    }
    newest
        .map(|(_, path)| path)
        .ok_or_else(|| anyhow!("no .lrcat file found below {}", root.display()))
}

fn open_catalog(path: &Path) -> Result<Connection> {
    let escaped_path = path
        .to_string_lossy()
        .replace('%', "%25")
        .replace('?', "%3F")
        .replace('#', "%23");
    let uri = format!("file:{escaped_path}?mode=ro&immutable=1");
    Connection::open_with_flags(
        uri,
        OpenFlags::SQLITE_OPEN_READ_ONLY
            | OpenFlags::SQLITE_OPEN_URI
            | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .with_context(|| format!("failed to open {} read-only", path.display()))
}

fn inventory(
    path: &Path,
    check_sidecars: bool,
    path_maps: &[(String, PathBuf)],
) -> Result<InventoryReport> {
    let connection = open_catalog(path)?;
    connection.execute_batch("PRAGMA query_only = ON;")?;

    let (images, root_files, virtual_copies) = connection.query_row(
        "SELECT count(*), count(DISTINCT rootFile), sum(masterImage IS NOT NULL) FROM Adobe_images",
        [],
        |row| {
            Ok((
                row.get::<_, i64>(0)? as u64,
                row.get::<_, i64>(1)? as u64,
                row.get::<_, i64>(2)? as u64,
            ))
        },
    )?;

    let mut properties = BTreeMap::<String, PropertyAccumulator>::new();
    let mut process_versions = BTreeMap::<String, u64>::new();
    let mut recipes_with_adjustments = 0;
    let mut empty_recipes = 0;
    let mut parse_failures = 0;
    let mut parse_failure_examples = Vec::new();
    let mut develop_recipes = 0;
    let mut statement = connection.prepare(
        "SELECT coalesce(processVersion, ''), coalesce(hasDevelopAdjustmentsEx, 0), coalesce(text, '') \
         FROM Adobe_imageDevelopSettings",
    )?;
    let mut rows = statement.query([])?;
    while let Some(row) = rows.next()? {
        develop_recipes += 1;
        let process_version: String = row.get(0)?;
        *process_versions.entry(process_version).or_default() += 1;
        if row.get::<_, f64>(1)? > 0.0 {
            recipes_with_adjustments += 1;
        }
        let text: String = row.get(2)?;
        if text.trim().is_empty() {
            empty_recipes += 1;
            continue;
        }
        match Parser::new(&text).parse_assignment() {
            Ok(value) => flatten_value("", &value, &mut properties),
            Err(error) => {
                parse_failures += 1;
                if parse_failure_examples.len() < 8 {
                    parse_failure_examples.push(ParseFailure {
                        error: format!("{error:#}"),
                        recipe: text,
                    });
                }
            }
        }
    }

    let sidecars = check_sidecars
        .then(|| scan_sidecars(&connection, path_maps))
        .transpose()?;
    let modified = fs::metadata(path)
        .and_then(|metadata| metadata.modified())
        .ok()
        .and_then(|time| time.duration_since(SystemTime::UNIX_EPOCH).ok())
        .map(|duration| duration.as_secs());

    Ok(InventoryReport {
        catalog: path.display().to_string(),
        catalog_modified_unix_seconds: modified,
        images,
        root_files,
        virtual_copies,
        develop_recipes,
        recipes_with_adjustments,
        empty_recipes,
        parse_failures,
        parse_failure_examples,
        process_versions: sorted_counts(process_versions),
        sidecars,
        properties: properties
            .into_iter()
            .map(|(name, value)| (name, value.into_report()))
            .collect(),
    })
}

fn scan_sidecars(
    connection: &Connection,
    path_maps: &[(String, PathBuf)],
) -> Result<SidecarSummary> {
    let mut summary = SidecarSummary {
        files_checked: 0,
        xmp_found: 0,
        files_missing: 0,
        paths_unavailable: 0,
    };
    let mut statement = connection.prepare(
        "SELECT root.absolutePath, folder.pathFromRoot, file.baseName, file.extension \
         FROM AgLibraryFile file \
         JOIN AgLibraryFolder folder ON folder.id_local = file.folder \
         JOIN AgLibraryRootFolder root ON root.id_local = folder.rootFolder",
    )?;
    let rows = statement.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, String>(3)?,
        ))
    })?;
    for row in rows {
        let (root, folder, base, extension) = row?;
        let mapped_root = path_maps
            .iter()
            .find_map(|(from, to)| root.strip_prefix(from).map(|suffix| to.join(suffix)))
            .or_else(|| {
                let path = PathBuf::from(&root);
                path.is_absolute().then_some(path)
            });
        let Some(mapped_root) = mapped_root else {
            summary.paths_unavailable += 1;
            continue;
        };
        let image = mapped_root.join(folder).join(format!("{base}.{extension}"));
        if !image.exists() {
            summary.files_missing += 1;
            continue;
        }
        summary.files_checked += 1;
        let sidecar = image.with_extension("xmp");
        let upper_sidecar = image.with_extension("XMP");
        if sidecar.is_file() || upper_sidecar.is_file() {
            summary.xmp_found += 1;
        }
    }
    Ok(summary)
}

impl PropertyAccumulator {
    fn observe(&mut self, value: &DevelopValue) {
        self.present += 1;
        let display = match value {
            DevelopValue::Bool(value) => {
                self.booleans += 1;
                value.to_string()
            }
            DevelopValue::Number(value) => {
                self.numbers += 1;
                self.numeric_min = Some(self.numeric_min.map_or(*value, |old| old.min(*value)));
                self.numeric_max = Some(self.numeric_max.map_or(*value, |old| old.max(*value)));
                format!("{value}")
            }
            DevelopValue::String(value) => {
                self.strings += 1;
                value.clone()
            }
            DevelopValue::Nil => {
                self.nils += 1;
                "nil".to_owned()
            }
            DevelopValue::Table(values) => {
                self.tables += 1;
                format!("<table:{}>", values.len())
            }
        };
        *self.values.entry(display).or_default() += 1;
    }

    fn into_report(self) -> PropertyReport {
        PropertyReport {
            present: self.present,
            types: TypeCounts {
                booleans: self.booleans,
                numbers: self.numbers,
                strings: self.strings,
                nils: self.nils,
                tables: self.tables,
            },
            numeric_min: self.numeric_min,
            numeric_max: self.numeric_max,
            distinct_values: self.values.len(),
            most_common_values: sorted_counts(self.values)
                .into_iter()
                .take(TOP_VALUES)
                .collect(),
        }
    }
}

fn flatten_value(
    prefix: &str,
    value: &DevelopValue,
    properties: &mut BTreeMap<String, PropertyAccumulator>,
) {
    let DevelopValue::Table(entries) = value else {
        return;
    };
    for (index, (key, child)) in entries.iter().enumerate() {
        let component = key.clone().unwrap_or_else(|| format!("[{index}]"));
        let name = if prefix.is_empty() {
            component
        } else {
            format!("{prefix}.{component}")
        };
        properties.entry(name.clone()).or_default().observe(child);
        if matches!(child, DevelopValue::Table(_)) {
            flatten_value(&name, child, properties);
        }
    }
}

fn sorted_counts<T: IntoIterator<Item = (String, u64)>>(values: T) -> Vec<ValueCount> {
    let mut heap = BinaryHeap::new();
    for (value, count) in values {
        heap.push((count, Reverse(value)));
    }
    let mut result = Vec::with_capacity(heap.len());
    while let Some((count, Reverse(value))) = heap.pop() {
        result.push(ValueCount { value, count });
    }
    result
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

    fn parse_assignment(mut self) -> Result<DevelopValue> {
        self.space();
        self.identifier()?;
        self.space();
        self.expect(b'=')?;
        let result = self.value()?;
        self.space();
        if self.offset != self.input.len() {
            bail!("unexpected trailing input at byte {}", self.offset);
        }
        Ok(result)
    }

    fn value(&mut self) -> Result<DevelopValue> {
        self.space();
        match self.peek() {
            Some(b'{') => self.table(),
            Some(b'"' | b'\'') => self.string().map(DevelopValue::String),
            Some(b'-' | b'+' | b'0'..=b'9') => self.number().map(DevelopValue::Number),
            Some(_) => match self.identifier()?.as_str() {
                "true" => Ok(DevelopValue::Bool(true)),
                "false" => Ok(DevelopValue::Bool(false)),
                "nil" => Ok(DevelopValue::Nil),
                value => Ok(DevelopValue::String(value.to_owned())),
            },
            None => bail!("expected value at end of input"),
        }
    }

    fn table(&mut self) -> Result<DevelopValue> {
        self.expect(b'{')?;
        let mut values = Vec::new();
        loop {
            self.space();
            if self.consume(b'}') {
                break;
            }
            let checkpoint = self.offset;
            let key = if self.consume(b'[') {
                self.space();
                let key = match self.peek() {
                    Some(b'"' | b'\'') => self.string()?,
                    Some(b'-' | b'+' | b'0'..=b'9') => self.number()?.to_string(),
                    _ => bail!("expected bracketed table key at byte {}", self.offset),
                };
                self.space();
                self.expect(b']')?;
                self.space();
                self.expect(b'=')?;
                Some(key)
            } else if self.peek().is_some_and(is_identifier_start) {
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
            values.push((key, self.value()?));
            self.space();
            if !self.consume(b',') && !self.consume(b';') && self.peek() != Some(b'}') {
                bail!("expected table separator at byte {}", self.offset);
            }
        }
        Ok(DevelopValue::Table(values))
    }

    fn identifier(&mut self) -> Result<String> {
        self.space();
        let start = self.offset;
        if !self.peek().is_some_and(is_identifier_start) {
            bail!("expected identifier at byte {}", self.offset);
        }
        self.offset += 1;
        while self.peek().is_some_and(is_identifier_continue) {
            self.offset += 1;
        }
        Ok(String::from_utf8(self.input[start..self.offset].to_vec())?)
    }

    fn string(&mut self) -> Result<String> {
        let quote = self.next().ok_or_else(|| anyhow!("expected string"))?;
        let mut result = Vec::new();
        loop {
            match self.next() {
                Some(value) if value == quote => break,
                Some(b'\\') => match self.next() {
                    Some(b'n') => result.push(b'\n'),
                    Some(b'r') => result.push(b'\r'),
                    Some(b't') => result.push(b'\t'),
                    Some(value) => result.push(value),
                    None => bail!("unterminated escape"),
                },
                Some(value) => result.push(value),
                None => bail!("unterminated string"),
            }
        }
        Ok(String::from_utf8(result)?)
    }

    fn number(&mut self) -> Result<f64> {
        let start = self.offset;
        while self
            .peek()
            .is_some_and(|value| matches!(value, b'+' | b'-' | b'.' | b'e' | b'E' | b'0'..=b'9'))
        {
            self.offset += 1;
        }
        std::str::from_utf8(&self.input[start..self.offset])?
            .parse()
            .with_context(|| format!("invalid number at byte {start}"))
    }

    fn space(&mut self) {
        while self.peek().is_some_and(|value| value.is_ascii_whitespace()) {
            self.offset += 1;
        }
    }

    fn expect(&mut self, value: u8) -> Result<()> {
        if self.consume(value) {
            Ok(())
        } else {
            bail!("expected '{}' at byte {}", char::from(value), self.offset)
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

fn is_identifier_start(value: u8) -> bool {
    value.is_ascii_alphabetic() || value == b'_'
}

fn is_identifier_continue(value: u8) -> bool {
    value.is_ascii_alphanumeric() || value == b'_'
}

#[cfg(test)]
#[path = "lightroom_inventory/tests.rs"]
mod tests;
