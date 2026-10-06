use std::{fs, path::Path};

use sha2::{Digest, Sha256};

/// Assets grouped by the manifest that pins them
/// (`<dir>/MANIFEST.sha256`).
const ASSET_GROUPS: [(&str, &str); 2] = [
    ("assets/face", "assets/face/MANIFEST.sha256"),
    ("assets/lens", "assets/lens/MANIFEST.sha256"),
];

const ASSETS: [(&str, &str); 5] = [
    (
        "assets/face/face_detection_yunet_2023mar.onnx",
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    ),
    (
        "assets/face/face_recognition_sface_2021dec.onnx",
        "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
    ),
    (
        "assets/face/LICENSE-YUNET",
        "c83b8120c50ccbd4c4f96edf53141bdd566ebb8f8e9227e415326aa1b1aba958",
    ),
    (
        "assets/face/LICENSE-SFACE",
        "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
    ),
    (
        "assets/lens/powershot_v1.json",
        "06cab9c3876a381cd1d7656970ba7b64b3c198409d851f30513d939ea50fdb10",
    ),
];

fn main() {
    let manifests: std::collections::HashMap<&str, String> = ASSET_GROUPS
        .iter()
        .map(|(dir, manifest_path)| {
            println!("cargo:rerun-if-changed={manifest_path}");
            let content = fs::read_to_string(Path::new(manifest_path)).unwrap_or_else(|error| {
                panic!("required packaged asset manifest {manifest_path} is missing: {error}")
            });
            (*dir, content)
        })
        .collect();
    for (path, expected) in ASSETS {
        println!("cargo:rerun-if-changed={path}");
        let bytes = fs::read(Path::new(path))
            .unwrap_or_else(|error| panic!("required packaged asset {path} is missing: {error}"));
        let actual = hex::encode(Sha256::digest(&bytes));
        assert_eq!(
            actual, expected,
            "packaged asset {path} failed pinned SHA-256 validation"
        );
        let file_name = Path::new(path).file_name().unwrap().to_string_lossy();
        let dir = Path::new(path).parent().unwrap().to_string_lossy();
        let manifest = manifests
            .get(dir.as_ref())
            .unwrap_or_else(|| panic!("no asset manifest declared for {path}"));
        assert!(
            manifest
                .lines()
                .any(|line| line == format!("{expected}  {file_name}")),
            "packaged asset manifest is missing the pinned entry for {file_name}"
        );
    }
}
