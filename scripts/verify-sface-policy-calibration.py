#!/usr/bin/env python3
"""Reproduce the identity-disjoint policy from real YuNet/SFace image output."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
GENERATOR_VERSION = "rekesz-calibration-verifier-v4"
EXTRACTOR = ROOT / "src/bin/sface_calibration_embeddings.rs"
PRODUCTION_INFERENCE = ROOT / "src/face_engine.rs"
PRODUCTION_POLICY = ROOT / "src/catalog/recognition_policy.rs"
CORPUS_PROVENANCE = ROOT / "assets/face/calibration/README.md"
SHEETS = sorted((ROOT / "assets/face/calibration").glob("synthetic-identities-sheet-*.png"))
EXPECTED_SHEET_HASHES = [
    "6f8f403e26541fdaf3dc3974cc1a30b9dacc843f9e4b5fb52f27d80176da21d9",
    "5e9afe33b11623486ee02798a9e0c08d141af07e3aaace2136171fdc9eb3008e",
    "2484762005af25def5f526c3846308210ce5320926a304addd56444795c12424",
    "bce9a431760cb0a55deca12a6b1441ff8296e608442fa41fb7e9055a1cadf98d",
    "3abf50e73e07d434e376f76bf50e251dbbd6e3c66954e39cb2d0ef5d5e7a28ba",
]
SPEC = (
    "rekesz-sface-image-corpus-v4|fictional-identities=20|source-variants=3|"
    "grid=5x(4-columns-identities,3-rows-variants)|"
    "pixel-variants=row0-base,row0-bright+18,row1-base,row1-bright-18,row2-base|"
    "split=cal-known-0:8,unknown-8:12;holdout-known-12:18,unknown-18:20|"
    "variants-0:4=exemplars|variant-4=query|unknown=all-five-variants|"
    "robust-centroid=trim-lowest-quarter-v1|model-authentic-yunet+sface"
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def spec_digest() -> str:
    bound = {
        "protocol": SPEC,
        "extractor_sha256": digest(EXTRACTOR),
        "production_inference_sha256": digest(PRODUCTION_INFERENCE),
        "production_policy_sha256": digest(PRODUCTION_POLICY),
        "corpus_provenance_sha256": digest(CORPUS_PROVENANCE),
        "cargo_lock_sha256": digest(ROOT / "Cargo.lock"),
        "yunet_sha256": digest(ROOT / "assets/face/face_detection_yunet_2023mar.onnx"),
        "sface_sha256": digest(ROOT / "assets/face/face_recognition_sface_2021dec.onnx"),
        "sheet_sha256": EXPECTED_SHEET_HASHES,
    }
    return hashlib.sha256(
        json.dumps(bound, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def extract_embeddings():
    assert len(SHEETS) == 5, "expected exactly five calibration contact sheets"
    assert [digest(path) for path in SHEETS] == EXPECTED_SHEET_HASHES, (
        "calibration pixels changed; review and re-freeze the corpus"
    )
    extractor = os.environ.get("SFACE_CALIBRATION_EXTRACTOR")
    command = ([extractor, *map(str, SHEETS)] if extractor else
               ["cargo", "run", "--quiet", "--bin", "sface-calibration-embeddings",
                "--", *map(str, SHEETS)])
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    samples = json.loads(completed.stdout)
    assert len(samples) == 100
    assert min(sample["detector_confidence"] for sample in samples) >= 0.9
    assert {len(sample["embedding"]) for sample in samples} == {128}
    return {
        identity: {
            sample["variant"]: sample["embedding"]
            for sample in samples
            if sample["identity"] == identity
        }
        for identity in range(20)
    }


def dot(left, right):
    return sum(a * b for a, b in zip(left, right, strict=True))


def normalize(vector):
    norm = sum(value * value for value in vector) ** 0.5
    return [value / norm for value in vector]


def robust_centroid(exemplars):
    initial = normalize([sum(values) for values in zip(*exemplars, strict=True)])
    if len(exemplars) < 4:
        return initial
    ranked = sorted(exemplars, key=lambda value: dot(value, initial), reverse=True)
    retained = len(ranked) - len(ranked) // 4
    return normalize([sum(values) for values in zip(*ranked[:retained], strict=True)])


def score(query, people):
    candidates = []
    for identity, exemplars in people.items():
        centroid = robust_centroid(exemplars)
        candidates.append((dot(query, centroid), identity,
                           max(dot(query, exemplar) for exemplar in exemplars)))
    candidates.sort(reverse=True)
    best, runner = candidates[0], candidates[1]
    return best[1], best[0], best[2], best[0] - runner[0]


def corpus(embeddings, known, unknown):
    people = {identity: [embeddings[identity][variant] for variant in range(4)]
              for identity in known}
    cases = []
    for identity in known:
        cases.append((identity, score(embeddings[identity][4], people)))
    for identity in unknown:
        for variant in range(5):
            cases.append((None, score(embeddings[identity][variant], people)))
    return cases


def confusion(cases, point):
    counts = dict(true_positive=0, false_negative=0, false_positive=0,
                  true_negative=0, unknown_false_accepts=0, misidentifications=0)
    for expected, scored in cases:
        predicted, absolute, exemplar, margin = scored
        accepted = absolute >= point[0] and exemplar >= point[1] and margin >= point[2]
        if expected is None:
            if accepted:
                counts["false_positive"] += 1
                counts["unknown_false_accepts"] += 1
            else:
                counts["true_negative"] += 1
        elif accepted and predicted == expected:
            counts["true_positive"] += 1
        else:
            counts["false_negative"] += 1
            if accepted:
                counts["false_positive"] += 1
                counts["misidentifications"] += 1
    return counts


def metrics(counts):
    positives = counts["true_positive"] + counts["false_negative"]
    negatives = counts["unknown_false_accepts"] + counts["true_negative"]
    predicted = counts["true_positive"] + counts["false_positive"]
    return {
        "false_accept_rate": counts["unknown_false_accepts"] / negatives,
        "false_reject_rate": counts["false_negative"] / positives,
        "precision": counts["true_positive"] / predicted if predicted else 1.0,
        "recall": counts["true_positive"] / positives,
        "identification_error_rate": counts["misidentifications"] / positives,
        "safety_error_rate": counts["false_positive"] / (positives + negatives),
        "unknown_far_one_sided_95_upper_bound": 1.0 - 0.05 ** (1.0 / negatives),
    }


def select_point(cases):
    eligible = []
    for absolute in (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
        for nearest in (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
            for margin in (0.04, 0.08, 0.12, 0.16, 0.20, 0.24, 0.28, 0.30):
                point = (absolute, nearest, margin)
                values = metrics(confusion(cases, point))
                if (values["false_accept_rate"] == 0.0
                        and values["identification_error_rate"] == 0.0
                        and values["safety_error_rate"] == 0.0
                        and values["precision"] == 1.0):
                    eligible.append((values["recall"], point))
    assert eligible, "no zero-observed-error calibration point"
    return max(eligible, key=lambda item: (item[0], *item[1]))[1]


def reproduce():
    embeddings = extract_embeddings()
    calibration_ids = set(range(0, 12))
    holdout_ids = set(range(12, 20))
    assert calibration_ids.isdisjoint(holdout_ids)
    calibration = corpus(embeddings, range(0, 8), range(8, 12))
    point = select_point(calibration)
    holdout = corpus(embeddings, range(12, 18), range(18, 20))
    # Prove selection depends on calibration observations, rather than merely
    # reproducing constants copied from the report.
    adversarial = [*calibration, (8, (9, 0.99, 0.99, 0.99))]
    try:
        select_point(adversarial)
    except AssertionError:
        pass
    else:
        raise AssertionError("wrong-person calibration case did not invalidate the policy")
    return point, calibration, holdout


def main() -> None:
    policy_path = ROOT / "assets/face/sface_2021dec_policy_v1.json"
    report = json.loads(policy_path.read_text(encoding="utf-8"))["calibration_report"]
    data = report["held_out_data"]
    point, calibration, holdout = reproduce()
    counts = confusion(holdout, point)
    derived = metrics(counts)
    generated = {
        "generator_spec_sha256": spec_digest(),
        "operating_point": point,
        "calibration_queries": len(calibration),
        "holdout_queries": len(holdout),
        "positive_queries": counts["true_positive"] + counts["false_negative"],
        "negative_queries": counts["unknown_false_accepts"] + counts["true_negative"],
        **counts,
        **derived,
    }
    if "--print-generated" in sys.argv:
        print(json.dumps(generated, indent=2, sort_keys=True))
        return
    assert data["verifier_version"] == GENERATOR_VERSION
    assert digest(Path(__file__)) == data["verifier_source_sha256"]
    assert digest(EXTRACTOR) == data["extractor_source_sha256"]
    assert digest(PRODUCTION_INFERENCE) == data["production_inference_source_sha256"]
    assert digest(PRODUCTION_POLICY) == data["production_policy_source_sha256"]
    assert digest(CORPUS_PROVENANCE) == data["corpus_provenance_sha256"]
    assert digest(ROOT / "Cargo.lock") == data["dependency_lock_sha256"]
    assert generated["generator_spec_sha256"] == data["generator_spec_sha256"]
    expected = report["operating_point"]
    assert point == (expected["absolute_threshold"], expected["nearest_exemplar_threshold"],
                     expected["second_best_margin"])
    assert data["calibration_queries"] == len(calibration)
    assert data["query_count"] == len(holdout)
    assert data["positive_queries"] == generated["positive_queries"]
    assert data["negative_queries"] == generated["negative_queries"]
    results = report["held_out_results"]
    assert all(results[name] == value for name, value in counts.items())
    for name, value in derived.items():
        assert abs(results[name] - value) <= 1e-7, (name, results[name], value)
    # Aggregate-only output: embeddings and source identifiers never enter logs.
    print(json.dumps({"report_id": report["report_id"], "operating_point": point,
                      **counts, **derived}, sort_keys=True))


if __name__ == "__main__":
    main()
