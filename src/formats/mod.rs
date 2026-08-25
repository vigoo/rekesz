use std::path::Path;

/// RAW containers supported by rawler. Camera support within a container is
/// determined by rawler's profiles and decoder capabilities.
const RAW_EXTENSIONS: &[&str] = &[
    "3fr", "ari", "arw", "cr2", "cr3", "crw", "dcr", "dcs", "dng", "erf", "iiq", "kdc", "mef",
    "mos", "mrw", "nef", "nrw", "orf", "pef", "raf", "rw2", "sr2", "srf", "srw",
];

pub fn is_raw(path: &Path) -> bool {
    has_extension(path, RAW_EXTENSIONS)
}

pub fn is_heif(path: &Path) -> bool {
    has_extension(path, &["heic", "heif"])
}

fn has_extension(path: &Path, expected: &[&str]) -> bool {
    path.extension().is_some_and(|actual| {
        expected
            .iter()
            .any(|expected| actual.eq_ignore_ascii_case(expected))
    })
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
