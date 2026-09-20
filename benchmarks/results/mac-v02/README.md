# M4 Pro / 48 GiB, 2026-09-21

See [the complete report](../../../docs/mac-v02.md) for conditions, interpretation and commands.

- `quality-*-bf16.json`: all 144 frozen SemIf rows, original labels, Busbar prompt, temperature 1.
- `benchmark-*-bf16.json`: matched 2048-token prefix / 8 questions / 3 measured repetitions.
  State-only APC and exact question repetition are separate strategies. These are synthetic
  latency workloads, not labeled accuracy measurements.
- `profile-*.json`: native MLX FP16, 4096-token prefix / 16 questions / 30 paired repetitions.
  Each pair runs both projections in seeded random order; it does not enforce an equal number
  of selected-first and full-first pairs. Head microbenchmarks and intrusive phase timers
  are separate from ordinary backend timing. The raw scope text predates this clarification.
- `correctness-*-fp32*.json`: actual MLX / HF FP32, all candidate scores, cache and batch checks.
- `bf16-diagnostic-cases.jsonl` and `diagnostic-*-fresh.json`: six cases selected **after** the
  quality run for large backend differences. Compare by original ID with the cached quality
  reports. They diagnose numerical behavior, not held-out quality. The source fixture and
  licence are retained in `../../data/`.
- `http-metal-3b-final.json`: 17 real HTTP requests including 64-question fanout and lifecycle.
- `http-metal-token-limit.json`: real 8191-token accepted / 8192-token rejected input boundary.

Files are copied from completed runs without rewriting measured values. The early quality
report's precision counter excludes the plugin's private attention wrapper; the later latency
and HTTP reports include all 3,085,938,688 BF16 parameters. This read-only inspector change did
not alter weights or inference. Initial mixed-precision exploratory runs are excluded.
`SHA256SUMS` covers the JSON/JSONL files. Package versions and pinned model revisions are in
the reports; the complete Metal environment is locked in `../../../requirements-metal.lock`.
