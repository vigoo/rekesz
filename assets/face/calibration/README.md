# Fictional calibration-image provenance

These five contact sheets were generated specifically for Rekesz on 2026-08-30
with OpenAI's image-generation tool. The generation request required four
fictional adults per sheet, one identity per column, and three portrait variants
per identity: frontal/neutral, turned pose, and changed lighting/expression. It
also requested diversity in apparent age, skin tone, gender presentation,
hairstyle, eyewear, and facial hair without naming or referencing any real
person.

The tool did not expose a reproducible seed or underlying image-model identifier,
so this repository does not claim that the pixels can be regenerated. The five
SHA-256-bound PNG outputs are the canonical corpus. “Fictional” records the prompt
intent and lack of a known real-person reference; it is not a biometric or legal
proof that an output cannot coincidentally resemble a real person. No user or
catalog photograph was used.

The corpus is a small engineering regression fixture. It is not representative
demographic evidence and carries no claim of population-level accuracy or
fairness. Rekesz retains only aggregate calibration results; generated SFace
embeddings are ephemeral verifier process data.

Each sheet is arranged as four identity columns by three variant rows. The
calibration extractor additionally applies deterministic brightness changes to
the first two rows before inference, yielding four actual-pixel SFace exemplars
per known identity and one untouched query. This exercises production's
four-or-more-exemplar robust-centroid branch.
