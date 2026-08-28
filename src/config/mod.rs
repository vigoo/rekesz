use std::{fs, path::PathBuf};

#[cfg(test)]
use std::ffi::OsString;

use anyhow::{Context as _, Result, bail};
use clap::{Parser, ValueEnum};
use serde::{Deserialize, Serialize};

use crate::discovery::RawJpegPreference;

pub(crate) const DEFAULT_IMAGE_CACHE_SIZE: usize = 8;
const RAW_MEMORY_RESERVE_BYTES: u64 = 1024 * 1024 * 1024;
const ESTIMATED_RAW_WORKING_SET_BYTES: u64 = 768 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(rename_all = "lowercase")]
pub(crate) enum PairPreference {
    Raw,
    Jpeg,
}

impl From<PairPreference> for RawJpegPreference {
    fn from(value: PairPreference) -> Self {
        match value {
            PairPreference::Raw => Self::Raw,
            PairPreference::Jpeg => Self::Jpeg,
        }
    }
}

impl From<RawJpegPreference> for PairPreference {
    fn from(value: RawJpegPreference) -> Self {
        match value {
            RawJpegPreference::Raw => Self::Raw,
            RawJpegPreference::Jpeg => Self::Jpeg,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
pub(crate) struct AppSettings {
    pub root: PathBuf,
    pub image_cache_size: usize,
    pub pair_preference: PairPreference,
    pub cache_root: PathBuf,
    pub job_parallelism: usize,
    pub raw_parallelism: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct Config {
    pub settings: AppSettings,
    pub settings_path: PathBuf,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum CliPairPreference {
    #[value(alias = "cr3")]
    Raw,
    #[value(alias = "jpg")]
    Jpeg,
}

#[derive(Debug, Parser)]
#[command(version, about = "Browse and develop a directory of photographs")]
struct Cli {
    /// Picture-library root directory.
    root: Option<PathBuf>,
    #[arg(long)]
    image_cache_size: Option<usize>,
    #[arg(long, alias = "render-parallelism")]
    job_parallelism: Option<usize>,
    #[arg(long)]
    raw_parallelism: Option<usize>,
    #[arg(long)]
    cache_root: Option<PathBuf>,
    #[arg(long = "cr3-jpeg-preference", value_enum)]
    pair_preference: Option<CliPairPreference>,
}

#[cfg(test)]
pub(crate) fn from_args(args: impl IntoIterator<Item = OsString>) -> Result<Config> {
    from_args_with_dirs(args, project_dirs()?)
}

pub(crate) fn from_env() -> Result<Config> {
    from_cli(Cli::parse(), project_dirs()?)
}

#[cfg(test)]
fn from_args_with_dirs(
    args: impl IntoIterator<Item = OsString>,
    dirs: directories::ProjectDirs,
) -> Result<Config> {
    from_cli(Cli::try_parse_from(args)?, dirs)
}

fn from_cli(cli: Cli, dirs: directories::ProjectDirs) -> Result<Config> {
    let settings_path = dirs.config_dir().join("settings.json");
    let mut settings =
        if settings_path.exists() {
            serde_json::from_slice::<AppSettings>(&fs::read(&settings_path).with_context(|| {
                format!("cannot read settings from {}", settings_path.display())
            })?)
            .with_context(|| format!("invalid settings in {}", settings_path.display()))?
        } else {
            defaults(&dirs)?
        };

    if let Some(root) = cli.root {
        settings.root = absolute(root)?;
    }
    if let Some(value) = cli.image_cache_size {
        settings.image_cache_size = value;
    }
    if let Some(value) = cli.job_parallelism {
        settings.job_parallelism = value;
    }
    if let Some(value) = cli.raw_parallelism {
        settings.raw_parallelism = value;
    }
    if let Some(value) = cli.cache_root {
        settings.cache_root = absolute(value)?;
    }
    if let Some(value) = cli.pair_preference {
        settings.pair_preference = match value {
            CliPairPreference::Raw => PairPreference::Raw,
            CliPairPreference::Jpeg => PairPreference::Jpeg,
        };
    }
    validate(&mut settings)?;
    Ok(Config {
        settings,
        settings_path,
    })
}

pub(crate) fn save(path: &std::path::Path, settings: &AppSettings) -> Result<()> {
    let mut validated = settings.clone();
    validate(&mut validated)?;
    let parent = path
        .parent()
        .context("settings path has no parent directory")?;
    fs::create_dir_all(parent)
        .with_context(|| format!("cannot create settings directory {}", parent.display()))?;
    fs::write(path, serde_json::to_vec_pretty(&validated)?)
        .with_context(|| format!("cannot write settings to {}", path.display()))?;
    Ok(())
}

fn defaults(dirs: &directories::ProjectDirs) -> Result<AppSettings> {
    let job_parallelism = std::thread::available_parallelism()
        .map(|count| count.get().min(8))
        .unwrap_or(1);
    Ok(AppSettings {
        root: std::env::current_dir()?,
        image_cache_size: DEFAULT_IMAGE_CACHE_SIZE,
        pair_preference: PairPreference::Raw,
        cache_root: dirs.cache_dir().to_path_buf(),
        job_parallelism,
        raw_parallelism: default_raw_parallelism(job_parallelism),
    })
}

fn validate(settings: &mut AppSettings) -> Result<()> {
    if settings.image_cache_size == 0 {
        bail!("image cache size must be at least 1");
    }
    if settings.job_parallelism == 0 {
        bail!("job parallelism must be at least 1");
    }
    if settings.raw_parallelism == 0 {
        bail!("RAW parallelism must be at least 1");
    }
    settings.raw_parallelism = settings.raw_parallelism.min(settings.job_parallelism);
    settings.root = absolute(settings.root.clone())?;
    settings.cache_root = absolute(settings.cache_root.clone())?;
    Ok(())
}

fn absolute(path: PathBuf) -> Result<PathBuf> {
    if path.is_absolute() {
        Ok(path)
    } else {
        Ok(std::env::current_dir()?.join(path))
    }
}

fn project_dirs() -> Result<directories::ProjectDirs> {
    directories::ProjectDirs::from("", "", "pics")
        .context("cannot determine the platform configuration directory")
}

pub(crate) fn default_raw_parallelism(job_parallelism: usize) -> usize {
    let system = sysinfo::System::new_all();
    recommended_raw_parallelism(
        system.cpus().len(),
        system.available_memory(),
        job_parallelism,
    )
}

pub(crate) fn recommended_raw_parallelism(
    logical_cpus: usize,
    available_memory_bytes: u64,
    job_parallelism: usize,
) -> usize {
    let cpu_limit = logical_cpus.max(1).div_ceil(2);
    let memory_limit = available_memory_bytes
        .saturating_sub(RAW_MEMORY_RESERVE_BYTES)
        .checked_div(ESTIMATED_RAW_WORKING_SET_BYTES)
        .unwrap_or(0)
        .max(1) as usize;
    cpu_limit.min(memory_limit).min(job_parallelism).max(1)
}

#[cfg(test)]
mod tests;
