"""Synchronized MLX phase timings plus ordinary end-to-end projection comparisons."""

import random
import statistics
import time

from .backends.base import length_batches
from .benchmark import benchmark_context
from .runtime import Runtime
from .schemas import ChoiceQuestion


def distribution(values):
    """Preserve individual observations and use nearest-rank p95, not a fitted percentile."""
    import math

    ordered = sorted(values)
    return {
        "median_ms": statistics.median(values),
        "p95_ms": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "samples_ms": values,
    }


def staged_score(backend, sequences, labels, prefix, projection):
    """Force GPU completion at each boundary; report these as intrusive diagnostic timings."""
    mx = backend.mx
    phases = dict(
        kv_branch_ms=0.0, input_ms=0.0, transformer_ms=0.0, projection_ms=0.0, readout_ms=0.0
    )
    for indices in length_batches(sequences, backend.batch_size):
        mark = time.perf_counter()
        cache = [layer.merge([layer] * len(indices)) for layer in prefix.cache]
        mx.eval([layer.state for layer in cache])
        phases["kv_branch_ms"] += (time.perf_counter() - mark) * 1000
        mark = time.perf_counter()
        inputs = mx.array([sequences[i] for i in indices])
        mx.eval(inputs)
        phases["input_ms"] += (time.perf_counter() - mark) * 1000
        mark = time.perf_counter()
        hidden = backend.model.model(inputs, cache=cache)[:, -1, :]
        mx.eval(hidden)
        phases["transformer_ms"] += (time.perf_counter() - mark) * 1000
        slots = [labels[i] for i in indices]
        mark = time.perf_counter()
        projected = backend._project(hidden, slots, projection)
        mx.eval(projected)
        phases["projection_ms"] += (time.perf_counter() - mark) * 1000
        mark = time.perf_counter()
        values = [value.astype(mx.float32).tolist() for value in projected]
        phases["readout_ms"] += (time.perf_counter() - mark) * 1000
        if not values:
            raise RuntimeError("profiling returned no model output")
    return phases, hidden, slots


def profile(backend, context_tokens=4096, questions=16, repeats=30):
    """Separate scheduling noise from the actual vocabulary head and KV branch work."""
    if (
        backend.identity["backend"] != "mlx"
        or getattr(backend, "default_projection", None) == "head"
    ):
        raise ValueError("vocabulary phase profiling requires the native MLX vocabulary scorer")
    if not 1 <= questions <= 64 or not 3 <= repeats <= 100:
        raise ValueError("profile needs 1–64 questions and 3–100 repeats")
    mx = backend.mx
    runtime = Runtime(backend)
    spec = benchmark_context(runtime, context_tokens)
    prefix_ids = runtime.compiler.context(spec)
    prefix = backend.prefill(prefix_ids)
    compiled = [
        runtime.compiler.question(
            ChoiceQuestion(
                type="choice",
                question=f"Should item {index:02d} be accepted for return?",
                options=[
                    {"id": "yes", "description": "Accept the return."},
                    {"id": "no", "description": "Reject the return."},
                ],
            )
        )
        for index in range(questions)
    ]
    sequences = [q.paths[0].token_ids for q in compiled]
    labels = [q.paths[0].label_ids for q in compiled]
    for projection in ("selected", "full"):
        backend.score(sequences, labels, prefix, projection)
        staged_score(backend, sequences, labels, prefix, projection)
    uninstrumented = {"selected": [], "full": []}
    staged = {"selected": [], "full": []}
    outputs = {}
    rng = random.Random(217)
    mx.reset_peak_memory()
    for _ in range(repeats):
        order = ["selected", "full"]
        rng.shuffle(order)
        for projection in order:
            mark = time.perf_counter()
            output = backend.score(sequences, labels, prefix, projection)
            uninstrumented[projection].append((time.perf_counter() - mark) * 1000)
            outputs[projection] = output.logits
            phases, hidden, slots = staged_score(backend, sequences, labels, prefix, projection)
            staged[projection].append(phases)
    # Use actual, already materialized hidden states. This isolates head execution
    # from Transformer graph fusion and does not claim an end-to-end speedup.
    rows = backend.head.weight[mx.array(slots[0])]
    mx.eval(rows)
    if any(ids != slots[0] for ids in slots):
        raise ValueError("head microbenchmark requires a homogeneous candidate set")
    methods = {
        "selected_rowwise": lambda: backend._project(hidden, slots, "selected"),
        "selected_single_matmul": lambda: hidden @ rows.T,
        "full_vocabulary": lambda: backend._project(hidden, slots, "full"),
    }
    head_samples = {key: [] for key in methods}
    for function in methods.values():
        mx.eval(function())
    for _ in range(max(30, repeats)):
        order = list(methods)
        rng.shuffle(order)
        for name in order:
            mark = time.perf_counter()
            mx.eval(methods[name]())
            head_samples[name].append((time.perf_counter() - mark) * 1000)
    summary = {
        "end_to_end": {name: distribution(values) for name, values in uninstrumented.items()},
        "staged_medians_ms": {
            name: {phase: statistics.median(row[phase] for row in values) for phase in values[0]}
            for name, values in staged.items()
        },
        "head_only": {name: distribution(values) for name, values in head_samples.items()},
        "max_selected_full_logit_delta": max(
            abs(a - b)
            for left, right in zip(outputs["selected"], outputs["full"], strict=True)
            for a, b in zip(left, right, strict=True)
        ),
        "peak_mlx_active_bytes": mx.get_peak_memory(),
    }
    return {
        "schema_version": 1,
        "model": backend.identity,
        "prefix_tokens": len(prefix_ids),
        "retained_kv_bytes": prefix.nbytes,
        "questions": questions,
        "batch_size": backend.batch_size,
        "repeats": repeats,
        "summary": summary,
        "staged_runs": staged,
        "scope": (
            "Resident model, warmed shapes, seed 217 random paired order. "
            "Staged timers add GPU barriers; use ordinary end-to-end samples for latency claims. "
            "Head microbenchmark uses the final real batch. No other model runs concurrently."
        ),
    }
