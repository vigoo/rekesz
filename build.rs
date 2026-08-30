use std::{fs, path::Path};

use sha2::{Digest, Sha256};

const ASSETS: [(&str, &str); 4] = [
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
];

fn main() {
    println!("cargo:rerun-if-changed=assets/face/MANIFEST.sha256");
    let manifest = fs::read_to_string("assets/face/MANIFEST.sha256")
        .expect("required packaged face asset manifest is missing");
    for (path, expected) in ASSETS {
        println!("cargo:rerun-if-changed={path}");
        let bytes = fs::read(Path::new(path)).unwrap_or_else(|error| {
            panic!("required packaged face asset {path} is missing: {error}")
        });
        let actual = format!("{:x}", Sha256::digest(&bytes));
        assert_eq!(
            actual, expected,
            "packaged face asset {path} failed pinned SHA-256 validation"
        );
        let file_name = Path::new(path).file_name().unwrap().to_string_lossy();
        assert!(
            manifest
                .lines()
                .any(|line| line == format!("{expected}  {file_name}")),
            "packaged face manifest is missing the pinned entry for {file_name}"
        );
    }
}
