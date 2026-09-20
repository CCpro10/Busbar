"""Ablate context reuse, fanout batching and projection with real model outputs retained."""

import importlib.metadata
import platform
import statistics
import subprocess
import time

from .runtime import Runtime
from .schemas import ContextSpec, DecisionRequest


def benchmark_context(runtime: Runtime, target: int) -> ContextSpec:
    """Construct a synthetic context near the target length without truncating tokenized input."""
    if not 128 <= target <= runtime.backend.max_tokens - 256:
        raise ValueError("context target must be at least 128 and leave 256 tokens for suffixes")
    base = "Every item is unused and delivered seven days ago. Returns are allowed within 30 days."
    # Repeating a neutral field gives a reproducible workload; this is not an accuracy dataset.
    repeats = max(0, target - len(runtime.compiler.context(ContextSpec(state=base))))
    spec = ContextSpec(state=base + " x" * repeats)
    for _ in range(5):
        gap = target - len(runtime.compiler.context(spec))
        if gap == 0:
            return spec
        repeats = max(0, repeats + gap)
        spec = ContextSpec(state=base + " x" * repeats)
    return spec


def _invoke(runtime, request, serial):
    """Time complete fanout work, including request preparation."""
    started = time.perf_counter()
    if serial:
        responses = [
            runtime.decide(request.model_copy(update={"questions": {name: question}}))
            for name, question in request.questions.items()
        ]
    else:
        responses = [runtime.decide(request)]
    return {
        "elapsed_ms": (time.perf_counter() - started) * 1000,
        "computed_tokens": sum(x.computed_tokens for x in responses),
        "reused_prefix_tokens": sum(x.reused_prefix_tokens for x in responses),
        "batches": sum(x.batches for x in responses),
        "responses": [x.model_dump() for x in responses],
    }


def _decisions(run):
    """Flatten serial and batch results into one identically keyed comparison map."""
    return {
        name: value
        for response in run["responses"]
        for name, value in response["decisions"].items()
    }


def benchmark(backend, context_tokens=4096, questions=16, repeats=3):
    """Measure actual cold/warm requests; never infer speedups from FLOP counts."""
    if not 1 <= questions <= 64 or not 1 <= repeats <= 20:
        raise ValueError("questions must be 1–64 and repeats 1–20")
    runtime = Runtime(backend)
    spec = benchmark_context(runtime, context_tokens)
    definitions = {
        f"item_{i:02}": {
            "type": "choice",
            "question": f"Should item {i:02} be accepted for return under the stated policy?",
            "options": [
                {
                    "id": "accept",
                    "description": "Accept the return: the item satisfies the policy.",
                },
                {"id": "reject", "description": "Reject the return: the item violates the policy."},
            ],
        }
        for i in range(questions)
    }
    # Compile kernels on a small independent snapshot before timing the experiment.
    warm = runtime.compile_context(
        ContextSpec(state="An unused item was delivered seven days ago.")
    )
    warm_request = DecisionRequest(snapshot_id=warm.snapshot.id, questions=definitions)
    runtime.decide(warm_request)
    runtime.clear()
    compiled = runtime.compile_context(spec)
    request = DecisionRequest(snapshot_id=compiled.snapshot.id, questions=definitions)
    strategies = {
        "fresh_sequential": ("fresh", "selected", True),
        "fresh_batched": ("fresh", "selected", False),
        "cached_sequential": ("cached", "selected", True),
        "cached_batched": ("cached", "selected", False),
        "cached_batched_full": ("cached", "full", False),
    }
    runs = {name: [] for name in strategies}
    # Warm each measured shape, then alternate strategy order to reduce fixed-order bias.
    for mode, projection, serial in strategies.values():
        _invoke(
            runtime, request.model_copy(update={"mode": mode, "projection": projection}), serial
        )
    for iteration in range(repeats):
        names = list(strategies)
        if iteration % 2:
            names.reverse()
        for name in names:
            mode, projection, serial = strategies[name]
            run = _invoke(
                runtime, request.model_copy(update={"mode": mode, "projection": projection}), serial
            )
            runs[name].append(run)
    cold = []
    for _ in range(repeats):
        runtime.clear()
        started = time.perf_counter()
        compiled = runtime.compile_context(spec)
        request = request.model_copy(update={"snapshot_id": compiled.snapshot.id})
        run = _invoke(runtime, request, False)
        run["elapsed_ms"] = (time.perf_counter() - started) * 1000
        run["prefill_ms"] = compiled.elapsed_ms
        run["prefill_tokens"] = compiled.prefill_tokens
        run["computed_tokens"] += compiled.prefill_tokens
        cold.append(run)
    runs["cold_compile_and_fanout"] = cold
    reference = _decisions(runs["fresh_sequential"][0])
    summary = {}
    for name, measurements in runs.items():
        observed = [_decisions(run) for run in measurements]
        summary[name] = {
            "median_ms": statistics.median(run["elapsed_ms"] for run in measurements),
            "min_ms": min(run["elapsed_ms"] for run in measurements),
            "max_ms": max(run["elapsed_ms"] for run in measurements),
            "computed_tokens_per_run": measurements[0]["computed_tokens"],
            "argmax_changes_vs_first_fresh": sum(
                row[key]["selected"] != reference[key]["selected"]
                for row in observed
                for key in reference
            ),
            "comparisons": len(reference) * len(observed),
            "max_probability_delta_vs_first_fresh": max(
                abs(a - b)
                for row in observed
                for key in reference
                for a, b in zip(
                    row[key]["probabilities"], reference[key]["probabilities"], strict=True
                )
            ),
        }
    summary["warm_speedup_vs_fresh_sequential"] = (
        summary["fresh_sequential"]["median_ms"] / summary["cached_batched"]["median_ms"]
    )
    versions = {}
    for package in ("mlx", "mlx-lm", "torch", "transformers", "busbar-runtime"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    hardware = {"platform": platform.platform(), "machine": platform.machine()}
    if platform.system() == "Darwin":
        hardware["chip"] = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
        hardware["memory_bytes"] = int(
            subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
        )
    return {
        "schema_version": 1,
        "model": backend.identity,
        "hardware": hardware,
        "versions": versions,
        "target_prefix_tokens": context_tokens,
        "actual_prefix_tokens": compiled.snapshot.token_count,
        "suffix_tokens": [
            len(runtime.compiler.question(q).token_ids) for q in request.questions.values()
        ],
        "questions": questions,
        "repeats": repeats,
        "cache_bytes": compiled.snapshot.cache_bytes,
        "scope": (
            "resident model; kernel warm-up excluded; "
            "synthetic latency workload, not a quality benchmark"
        ),
        "summary": summary,
        "runs": runs,
    }
