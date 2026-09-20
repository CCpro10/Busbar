# Frozen decision fixtures

`semif-authored144.jsonl` is an unchanged copy of SemIf's project-authored fixture:

- Source: https://github.com/TheoLeeCJ/SemIf/blob/ca3ba65f142967030ecb453346e94d6f476a69df/benchmarks/data/authored144.jsonl
- Revision: `ca3ba65f142967030ecb453346e94d6f476a69df`.
- Licence: MIT, copyright 2026 TheoLeeCJ; retained in `SemIf-LICENSE`.
- 144 rows, 36 source groups, four variants per group. Labels were independently
  model-reviewed by the source project, not human-adjudicated. This is a compact
  synthetic benchmark, not evidence of production-wide accuracy.
- Busbar uses the original states, questions, option descriptions, order and gold
  labels, with its own chat prompt. No model training, prompt tuning or temperature
  fitting on this fixture. Reports retain source IDs, provenance and input SHA-256.

The reported `mean_family_balanced_accuracy` matches the source's averaging rule:
within each family, average recall across represented gold labels; then average
across families. Multiclass Brier is the **sum** of squared class errors (range 0–2),
not the normalized variant used by some other leaderboards. NLL uses a 1e-12 floor;
ECE uses ten equal-width confidence bins. Accuracy and probability calibration are
separate measurements.

The independent [Typed Decision Bench](https://blobfish.ai/benchmarks/typed-decision-bench)
also reports coverage, Brier, log loss and calibration. Its 5,387-item leaderboard
uses different data and normalization; our 144-row results must not be inserted
into that ranking. We do not claim a live Jev API run.
