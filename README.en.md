# Busbar

<p align="center">
  <a href="README.md"><kbd>简体中文</kbd></a> &nbsp; <a href="README.en.md"><kbd>English</kbd></a>
</p>

**Make repeated, structured decisions over a shared context.**

Busbar is an open-source LLM decision runtime that runs on Apple Silicon Macs. Submit a context once and Busbar compiles it into a reusable `ContextSnapshot`. Later questions reuse the model's computed prefix and return boolean decisions, choices, or numeric scores with probability distributions.

For example, one support ticket can answer “Was the customer charged twice?”, “Which team should handle this?”, and “How urgent is it?”. Your application defines the questions and candidate descriptions and performs any business actions; Busbar handles model execution, context reuse, and structured results. It suits ticket routing, rule checks, and candidate selection. Use a generation interface for long-form writing or open-ended chat.

```text
Application context → tokenization + one prefill → ContextSnapshot
                                                    ├─ Boolean: Is a refund needed?
                                                    ├─ Choice: Which team should handle it?
                                                    └─ Score: What is the priority?
```

## Two integration paths

| | Quick integration | Trainable selection head |
|---|---|---|
| Where scores come from | Existing vocabulary rows for labels such as A/B/C/D | Encode each candidate description, then score it with a shared head |
| Training required | No | Yes: labeled data; only the small head is trained, with the backbone frozen |
| When to use it | The instruction model already understands the task; start here | You have task data and want to learn your own decision rules or preferences |
| Backends | MLX, HF/PyTorch, vLLM Metal | Native MLX |
| Application interface | Shared Boolean / Choice / Score APIs and snapshot lifecycle | Same |

Native MLX/HF can project only the vocabulary rows needed for candidate labels. The trainable path uses NanoJev-style candidate scoring: `zᵢ = wᵀ LayerNorm(hᵢ)`, followed by softmax over candidates. Both paths reuse context, but K candidates require K suffix paths with the selection head. **A smaller output layer alone does not establish better speed or accuracy.** See the [selection-head guide](docs/trainable-heads.md) (Chinese) for encoding, training, and trade-offs.

## Benchmark against official Jev (partial coverage)

On 2026-09-21, we compared Busbar with the official hosted model `jev-1.13.0` on **the same 4,370 items** from [Typed Decision Bench v0.3](https://blobfish.ai/benchmarks/typed-decision-bench) (TDB). TDB is a third-party public benchmark, not TypeSafe's private four-workflow evaluation. The Jev column uses upstream's published per-item responses; we did not call the Jev API again for this comparison.

Busbar ran **Qwen3.5-4B, BF16, native MLX, vocabulary readout** on an M4 Pro Mac with 48 GiB RAM. **These results do not include the trainable head or ongoing training improvements.** Both systems were rescored using the upstream scoring functions on the identical subset of item IDs.

| Suite | Items | Jev DS ↑ | Busbar DS ↑ | Jev accuracy | Busbar accuracy |
|---|---:|---:|---:|---:|---:|
| data-and-operations | 722 | 82.75 | 78.40 | 78.67% | 77.42% |
| real-time-and-agents | 1,199 | 75.81 | 70.47 | 65.47% | 58.22% |
| retrieval-and-knowledge | 890 | 82.49 | 77.54 | 79.33% | 71.12% |
| safety-and-quality | 1,162 | 88.91 | 85.83 | 85.89% | 80.72% |
| workflow-control | 397 | 85.27 | 81.63 | 81.86% | 76.07% |
| **Same-subset total** | **4,370** | **82.66** | **78.32** | **77.39%** | **71.62%** |

DS is DecisionScore (0–100; higher is better), aggregating `1 − normalized Brier` per item to measure probability quality. It is not accuracy. On this subset, Busbar trails Jev by **4.34 DS points** and **5.77 percentage points of accuracy**.

**Partial coverage: these are not full-leaderboard results.**

- TDB contains 5,387 items. Of these, 359 are published as IDs only, without redistributable text, leaving 5,028 runnable items.
- Busbar answered 4,370: **81.12%** of the full roster and **86.91%** of items with public text. The remaining 658 contain 28–64 candidates, exceeding this run's 26-candidate limit, and are excluded.
- Jev is also scored only on those 4,370 items. This table cannot establish a rank on the full TDB leaderboard or attribute the gap solely to architecture, training, or execution backend.
- Local Busbar inference and the hosted Jev API have different timing boundaries; we do not derive a speedup ratio from them.

See the [scoring summary JSON](benchmarks/results/tdb-2026-09-21/compare-4370.json) for per-suite/task metrics, the upstream scorer revision, and source-result SHA256 hashes; the [run configuration](benchmarks/results/tdb-2026-09-21/busbar-qwen35-full.manifest.json); and the [detailed comparison and reproduction steps](docs/jev-comparison.md) (Chinese). Items and Jev responses come from the [public TDB dataset](https://huggingface.co/datasets/SamuelChien821/typed-decision-bench). The linked results directory publishes aggregates and configuration only, not item text or raw per-item responses.

## Get started on a Mac

Requires Apple Silicon, macOS 15+, Python 3.11–3.13, and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/CCpro10/Busbar.git
cd Busbar
uv sync --locked --extra mlx
uv run busbar run examples/returns.json
```

The default is Qwen3-0.6B at a pinned revision, requiring roughly 1.2 GB of weights on first download. Inference and inputs stay on your machine. Results include each candidate's score and probability, the derived decision value, actual computed/reused token counts, and timings. Larger models have been verified on an M4 Pro Mac with 48 GiB RAM:

```bash
uv run busbar run examples/returns.json --model openbmb/MiniCPM5-2B --dtype bfloat16
uv run busbar run examples/returns.json --model Qwen/Qwen3.5-4B --dtype bfloat16
```

MiniCPM5 has approximately 2.52B parameters; Qwen3.5 loads the text component of its official checkpoint. Both are unquantized and need more memory than the default 0.6B model. See the [backend guide](docs/backends.md) (Chinese) for installation, download sizes, the HF reference path, the separate vLLM Metal environment, and known limitations.

## Python or HTTP integration

```python
from busbar import ContextSpec, DecisionRequest, Runtime
from busbar.backends import load_backend

runtime = Runtime(load_backend("mlx"))
context = runtime.compile_context(
    ContextSpec(
        namespace="ticket-1", state={"message": "I paid once, but my card was charged twice."}
    )
)
result = runtime.decide(
    DecisionRequest(
        namespace="ticket-1",
        snapshot_id=context.snapshot.id,
        questions={
            "billing": {
                "type": "boolean",
                "question": "Does this request need the payments or billing team?",
            }
        },
    )
)
print(result.decisions["billing"].model_dump())
# Reuse context.snapshot.id for later questions; release it when finished.
runtime.delete_context(context.snapshot.id, namespace="ticket-1")
```

For a resident model and cache across HTTP requests, run `uv run busbar serve --port 8787` and open the [local API documentation](http://127.0.0.1:8787/docs). First call `POST /v1/contexts`, then pass its ID to `POST /v1/decisions`. A request can mix 1–64 questions. Contexts support creation, reuse, versioning, listing/filtering, deletion, and namespace-scoped cleanup.

Choice supports 2–26 options with IDs and descriptions. Score supports 2–26 described, strictly increasing numeric levels and returns their probability-weighted expectation. Probabilities are normalized within the candidate set and are not business-calibrated; entropy is not correctness. The [API guide](docs/api.md) (Chinese) covers fields, lifecycle, error codes, and memory limits.

## Train your own selection head

```bash
uv run busbar train-head examples/trainable_support/train.jsonl \
  --validation examples/trainable_support/validation.jsonl \
  --output models/heads/support-v1 --epochs 40

uv run busbar evaluate examples/trainable_support/test.jsonl --format native \
  --head models/heads/support-v1 --output local-results/support-v1-test.json

uv run busbar serve --head models/heads/support-v1 --port 8787
```

Each version stores the small head's weights, training history, data-split provenance, and file hashes without duplicating the backbone. You can continue training from an existing head, verify artifacts on load, and roll back to earlier versions. New versions use new directories; existing versions are not overwritten. Switch to `load_backend("mlx", head="models/heads/support-v1")` without changing the decision API. See the [training guide](docs/trainable-heads.md) (Chinese) for data formats, validation-based selection, inherited leakage checks, and failure handling.

## Evidence and current boundaries

The repository includes contract regressions, real-model numerical tests, real HTTP validation, and raw comparisons with SemIf and NanoJev. The [Jev comparison](docs/jev-comparison.md) uses upstream scoring on the same 4,370 public TDB items. The [validation guide](docs/validation.md) distinguishes implementation correctness, task quality, end-to-end latency, and output-head microbenchmarks. Reports for [0.3](docs/mac-v03.md) and [0.4](docs/mac-v04.md) retain settings, per-item results, comparisons, and failures. These detailed documents are currently in Chinese.

The server targets local use: one process holds one model, model operations are serialized, and paths within a request can be batched. KV state is in memory and must be rebuilt after a restart. Namespaces do not provide authentication. MiniCPM5/vLLM Metal has not passed the existing cache numerical-consistency gate; native MLX is a starting point for integration. Linux serving engines, continuous batching, quantization, and backbone fine-tuning are not implemented.

Busbar is an independent implementation inspired by Jev-style structured decisions. It is not an official TypeSafe Jev implementation or a NanoJev checkpoint loader. It depends on [MLX-LM](https://github.com/ml-explore/mlx-lm), [Transformers](https://github.com/huggingface/transformers), and the selected model.

## Documentation and contributing

The following guides are in Chinese:

- [API and runtime](docs/api.md): integration, complete lifecycle, and error handling.
- [Backends and models](docs/backends.md): installation, platform differences, and support scope.
- [Trainable selection heads](docs/trainable-heads.md): scoring, data, training, evaluation, and rollback.
- [Architecture and extensions](docs/design.md): module responsibilities, data flow, invariants, and backend contracts.
- [Development guide](CONTRIBUTING.md): environment, style, testing, and contribution requirements.
- [Changelog](CHANGELOG.md) and [0.4.1 quality review](docs/quality-v041.md): changes, fixes, and validation.

License: [Apache License 2.0](LICENSE).
