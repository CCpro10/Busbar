"""Opt-in real-checkpoint gates for cache isolation, projection and cross-backend agreement."""

import json
import os
from pathlib import Path

import pytest

from busbar import ContextSpec, DecisionRequest, Runtime
from busbar.backends import load_backend

pytestmark = [
    pytest.mark.model,
    pytest.mark.skipif(
        os.environ.get("BUSBAR_RUN_MODEL_TESTS") != "1", reason="real-model tests are opt-in"
    ),
]


def _assert_close(left, right, tolerance):
    """Check every candidate, not merely the winning label."""
    assert left.decisions.keys() == right.decisions.keys()
    for key in left.decisions:
        a, b = left.decisions[key], right.decisions[key]
        assert a.selected == b.selected
        assert a.logits == pytest.approx(b.logits, abs=tolerance, rel=0)
        assert a.probabilities == pytest.approx(b.probabilities, abs=tolerance, rel=0)


def test_real_cache_branches_selected_projection_and_hf_reference():
    """One pinned model must preserve decisions across backends and repeated branch fanout."""
    example = json.loads((Path(__file__).parents[1] / "examples/returns.json").read_text())
    spec = ContextSpec.model_validate(example["context"])
    mlx = Runtime(load_backend("mlx", dtype="float32"))
    compiled = mlx.compile_context(spec)
    request = DecisionRequest(
        snapshot_id=compiled.snapshot.id, namespace=spec.namespace, questions=example["questions"]
    )
    fresh = mlx.decide(request.model_copy(update={"mode": "fresh", "projection": "full"}))
    cached = mlx.decide(request)
    full_cached = mlx.decide(request.model_copy(update={"projection": "full"}))
    _assert_close(cached, full_cached, 0.001)
    _assert_close(cached, fresh, 0.002)
    before_bytes = mlx.stats()["cache_bytes"]
    # Equal-length suffixes enter the actual batched cache path rather than serial fallback.
    fanout = request.model_copy(
        update={"questions": {f"branch{i}": request.questions["route"] for i in range(16)}}
    )
    batch = mlx.decide(fanout)
    assert batch.batches == 2
    for decision in batch.decisions.values():
        assert decision.logits == pytest.approx(cached.decisions["route"].logits, abs=0.002)
    reversed_questions = dict(reversed(list(request.questions.items())))
    reordered = mlx.decide(request.model_copy(update={"questions": reversed_questions}))
    _assert_close(reordered, cached, 0.002)
    _assert_close(mlx.decide(request), cached, 0.00001)
    assert mlx.stats()["cache_bytes"] == before_bytes
    assert mlx.compile_context(spec).reused

    hf = Runtime(load_backend("hf", dtype="float32"))
    hf_context = hf.compile_context(spec)
    hf_request = request.model_copy(update={"snapshot_id": hf_context.snapshot.id})
    reference = hf.decide(hf_request.model_copy(update={"mode": "fresh", "projection": "full"}))
    _assert_close(fresh, reference, 0.005)
    _assert_close(hf.decide(hf_request), reference, 0.002)

    report = {
        "checkpoint": mlx.backend.identity,
        "gate": "real float32 model, same prompts, every candidate compared",
        "tolerances": {"mlx_projection": 0.001, "mlx_cache": 0.002, "mlx_vs_hf": 0.005},
        "mlx_fresh_full": fresh.model_dump(),
        "mlx_cached_selected": cached.model_dump(),
        "hf_fresh_full": reference.model_dump(),
        "fanout_16_batches": batch.batches,
        "cache_bytes_unchanged": before_bytes,
    }
    destination = os.environ.get("BUSBAR_MODEL_REPORT")
    if destination:
        with Path(destination).open("x") as file:
            json.dump(report, file, indent=2)
            file.write("\n")
