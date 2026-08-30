# Offline face models

Rekesz performs face analysis locally. These files are loaded directly from this
directory; the application does not send photographs or face embeddings to a
network service.

| File | Upstream model | SHA-256 |
| --- | --- | --- |
| `face_detection_yunet_2023mar.onnx` | OpenCV Zoo YuNet (2023 March) | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` |
| `face_recognition_sface_2021dec.onnx` | OpenCV Zoo SFace (2021 December) | `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79` |

`sface_2021dec_policy_v1.json` is the versioned production suggestion policy
for that exact SFace model hash. Rekesz validates the policy file's compiled-in
SHA-256, model hash, schema and thresholds before installing it in a catalog.
Its provenance and validation basis are embedded in the artifact. The policy
uses the model's Apache-2.0 license in `LICENSE-SFACE`; the policy metadata is
part of Rekesz itself.

The five hash-pinned images under `calibration/` contain 20 AI-generated,
fictional identities with three pose/expression/lighting source portraits each.
The calibration verifier crops all 60 cells, derives deterministic brightness
variants at the pixel level, and runs 100 samples through the exact bundled
YuNet and SFace models. Four outputs per known identity exercise production's
outlier-trimming robust centroid; the third portrait remains the query.
Embeddings remain ephemeral; only aggregate confusion counts and metrics are
retained. This small synthetic corpus is a policy-regression gate, not evidence
of demographic or population performance. Detailed, hash-bound prompt/tool
limitations are in `calibration/README.md`.

The adjacent `LICENSE-YUNET` and `LICENSE-SFACE` files contain the upstream
model licenses. Runtime inference uses ONNX Runtime and is fully offline.

## Synthetic test portraits

`fixtures/synthetic-*.png` are AI-generated, entirely fictional portraits
created for Rekesz's detector/recognizer integration tests. They do not contain
private library photographs or knowingly depict real people. The fixtures are
used only for broad model outcomes: valid landmarks, multi-face/NMS behavior,
and same-identity similarity above different-identity similarity. Tests do not
lock down brittle pixel-exact scores.

Face crops default to the bounded in-memory preview cache and are not written to
disk. Users may explicitly enable the local face-crop disk cache in Settings;
those entries use the normal render-cache budget, statistics, and clear action.
Full-photo context thumbnails use the normal
local thumbnail cache and contain no baked-in face rectangle; the rectangle is
painted by the UI. Clearing Rekesz's image cache removes those context images.
