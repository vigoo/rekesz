# People analysis operations and privacy

People analysis runs entirely in the Rekesz process using the packaged YuNet
and SFace ONNX models. Detection pixels, face crops, landmarks, embeddings,
names, and decisions are never submitted to a network service. Online map
detail is a separate opt-in feature and receives only map tile coordinates.
People analysis remains functional with network access denied.

The model files and their license files live in `assets/face`. Rekesz verifies
their pinned SHA-256 digests before creating an inference session and reports a
packaged-asset error if validation fails. Release packaging must include both
ONNX files, both license files, `sface_2021dec_policy_v1.json`, and
`assets/face/README.md`. The shipped policy is bound to the exact SFace model
hash and has its own pinned SHA-256 in `MANIFEST.sha256`. Rekesz validates its
format, model and algorithm versions, thresholds, model hash, and content hash
before idempotently activating it. People diagnostics show the active policy
version and a non-sensitive digest prefix.

Recognition refresh work is durable. Startup resumes expired or pending work;
completed face analysis, identity decisions, and undo wake it immediately.
Workers claim at most 16 rows at once, then yield and check pause/cancellation
and foreground admission before claiming another batch. Retry-delayed rows do
not cause a busy loop and are picked up by the maintenance coordinator later.
Transient analysis failures retry automatically with capped exponential
backoff and become terminal after five attempts. Unsupported or corrupt input
is terminal immediately. Terminal rows have no timer wake; “Retry failed
analysis” explicitly resets them. A detector, recognizer, or decoder version
change also makes the asset eligible again.

## Recognition calibration evidence

The policy and calibration report are one canonical, hash-pinned JSON artifact
at `assets/face/sface_2021dec_policy_v1.json`. It records the upstream SFace LFW
6,000-pair reference, identity-disjoint split protocol, generator/version and
corpus hashes, verifier/extractor/production-source/dependency hashes, sample counts,
confusion matrix, FAR, FRR, precision,
recall, and the combined centroid/exemplar/margin operating point. The verifier
hash-checks five contact sheets, crops their 60 prompt-fictional face images,
derives 100 deterministic pixel samples, and runs the exact packaged YuNet
detector and SFace embedder. Four model outputs per known identity exercise the
same lowest-quarter-trimming robust centroid used by production. It scores
positive and unknown queries through the same centroid/exemplar/runner-up
semantics as production, selects thresholds only on identity-disjoint calibration
data, and then evaluates the frozen point on untouched holdout identities. Run
`scripts/verify-sface-policy-calibration.py` to reproduce threshold selection,
the confusion matrix and metrics, and model/pixel/spec binding.

The held-out evidence uses actual SFace outputs but only 20 prompt-fictional
identities; it is not a demographic validation corpus. The image tool exposed
no reproducible generator model or seed, so pixel hashes—not regeneration—are
the canonical corpus identity. The published
SFace/LFW result is only an upstream anchor. Rekesz therefore uses the policy
only for human-review suggestions and never to confirm an identity. A later
consented-corpus policy requires a new policy version and artifact digest rather
than silently changing thresholds. With ten unknown holdout queries, zero
observed accepts has a one-sided 95% upper bound of about 25.9%, so it does not
establish a population FAR target. The selected 0.70 absolute/exemplar values
are tested grid ceilings, not statistically identified boundaries. Accepted matches to the wrong known identity
count as both a missed correct identity and a safety false accept.

Source provenance uses a durable full-content SHA-256 identity computed with a
bounded 64 KiB buffer, in addition to size and mtime.
The identity is stored with inventory, analysis state, and embeddings and is
revalidated inside the commit transaction. Thus an in-place byte replacement
that preserves size and mtime invalidates derived vectors and cannot publish a
stale analysis. Schema-v6 migration returns pre-fingerprint analysis to pending
and discards derived vectors; the next inventory pass backfills identities.

The persistent settings file supports these People keys:

- `automatic_face_analysis`: starts/resumes idle analysis when true. Imported
  Lightroom People remain readable when false.
- `face_workers`: maximum concurrent inference sessions. It is also capped by
  the global job parallelism and the configured per-worker memory allowance.
- `face_worker_memory_bytes`: conservative memory allowance used to cap worker
  concurrency; the minimum is 64 MiB.
- `face_crop_disk_cache`: `memory-only` (privacy-preserving default) or `disk`.
  Context thumbnails continue to follow the normal local render-cache policy.

Before every schema-migration attempt, Rekesz creates and validates a fresh,
uniquely named `catalog.sqlite3.pre-vN-<attempt>-backup.sqlite3` beside the
owned catalog. If migration
fails, close Rekesz, retain the failed catalog for diagnosis, copy the backup
back to `catalog.sqlite3`, and reopen with software that supports schema N.
Rekesz opens Lightroom catalogs read-only with SQLite immutable mode and never
uses them as migration or recovery targets.

Diagnostics must remain aggregate-only: schema/model identifiers, indexed and
pending counts, error counts, and stale-artifact counts. Logs and support
bundles must not contain names, source paths, crops, landmarks, or embeddings.
“Retry failed analysis” resets only failed owned-catalog jobs; it does not alter
identity decisions or the Lightroom catalog.
