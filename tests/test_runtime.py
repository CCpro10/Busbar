"""Regression coverage for snapshot identity, cache lifetime and decision semantics."""

import math
from concurrent.futures import ThreadPoolExecutor

import pytest
from pydantic import ValidationError

from busbar.compiler import read_decision
from busbar.runtime import CapacityError, Runtime, SnapshotNotFound
from busbar.schemas import ContextSpec, DecisionRequest, ScoreQuestion


def test_context_lifecycle_and_cross_request_reuse(runtime, backend, questions):
    """Identical state reuses prefill; edits version it; old versions can be restored or deleted."""
    original = runtime.compile_context({"state": {"b": 2, "a": 1}})
    same = runtime.compile_context({"state": {"a": 1, "b": 2}})
    assert original.snapshot.id == same.snapshot.id and same.reused and same.prefill_tokens == 0
    assert len(backend.prefills) == 1
    request = DecisionRequest(snapshot_id=original.snapshot.id, questions=questions)
    cached = runtime.decide(request)
    fresh = runtime.decide(request.model_copy(update={"mode": "fresh"}))
    assert cached.decisions == fresh.decisions
    assert fresh.computed_tokens - cached.computed_tokens == 3 * original.snapshot.token_count
    assert cached.reused_prefix_tokens == 3 * original.snapshot.token_count
    newer = runtime.replace_context(original.snapshot.id, ContextSpec(state={"a": 2}))
    assert newer.snapshot.id != original.snapshot.id
    assert runtime.decide(request).decisions == cached.decisions
    assert len(runtime.list_contexts(id_prefix=original.snapshot.id[:12])) == 1
    assert runtime.delete_context(original.snapshot.id)
    assert not runtime.delete_context(original.snapshot.id)
    with pytest.raises(SnapshotNotFound):
        runtime.decide(request)
    assert runtime.clear() == 1
    assert runtime.stats()["cache_bytes"] == 0 and runtime.list_contexts() == []


def test_namespace_isolation(runtime, questions):
    """Even identical states cannot alias another namespace's context handle."""
    a = runtime.compile_context({"state": "same", "namespace": "a"})
    b = runtime.compile_context({"state": "same", "namespace": "b"})
    assert a.snapshot.id != b.snapshot.id
    with pytest.raises(SnapshotNotFound):
        runtime.decide(
            DecisionRequest(snapshot_id=a.snapshot.id, namespace="b", questions=questions)
        )
    assert not runtime.delete_context(a.snapshot.id, "b")
    assert runtime.clear("a") == 1
    assert runtime.get_context(b.snapshot.id, "b")


def test_lru_byte_budget_and_failed_admission_preserve_old_state(backend):
    """Oversized replacement cannot destroy the existing context; LRU eviction accounts bytes."""
    runtime = Runtime(backend, max_contexts=2, max_cache_bytes=2000)
    a = runtime.compile_context({"state": "a"})
    b = runtime.compile_context({"state": "b"})
    runtime.compile_context({"state": "a"})
    with pytest.raises(CapacityError):
        runtime.replace_context(a.snapshot.id, ContextSpec(state="large" * 300))
    assert runtime.get_context(a.snapshot.id) and runtime.get_context(b.snapshot.id)
    c = runtime.compile_context({"state": "c"})
    assert {x.id for x in runtime.list_contexts()} == {a.snapshot.id, c.snapshot.id}
    assert runtime.stats()["cache_bytes"] == a.snapshot.cache_bytes + c.snapshot.cache_bytes
    assert runtime.stats()["evictions"] == 1


def test_ttl_and_invalid_fanout_do_not_mutate_or_execute(backend, questions, monkeypatch):
    """Expired IDs fail clearly; an overlong question cannot partially execute the fanout."""
    now = [100.0]
    monkeypatch.setattr("busbar.runtime.time.monotonic", lambda: now[0])
    runtime = Runtime(backend, ttl_seconds=10)
    context = runtime.compile_context({"state": "hello"})
    with pytest.raises(ValueError, match="token limit"):
        runtime.decide(
            DecisionRequest(
                snapshot_id=context.snapshot.id,
                questions={
                    **questions,
                    "oversized": {"type": "boolean", "question": "x" * 8192},
                },
            )
        )
    assert backend.forwards == []
    now[0] = 110
    with pytest.raises(SnapshotNotFound):
        runtime.get_context(context.snapshot.id)
    assert runtime.stats()["cache_bytes"] == 0
    assert runtime.stats()["expirations"] == 1


def test_concurrent_context_creation_is_idempotent(runtime, backend):
    """Simultaneous callers sharing one context trigger one physical prefill."""
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: runtime.compile_context({"state": "shared"}), range(8)))
    assert len({x.snapshot.id for x in results}) == 1
    assert len(backend.prefills) == 1 and sum(not x.reused for x in results) == 1


def test_score_expectation_and_probability_semantics():
    """The public Score value is a bounded expectation over scale values, not the winning index."""
    question = ScoreQuestion(
        type="score",
        question="Rate",
        levels=[
            {"value": 0, "description": "low"},
            {"value": 1, "description": "medium"},
            {"value": 2, "description": "high"},
        ],
    )
    result = read_decision(question, [math.log(0.1), math.log(0.6), math.log(0.3)])
    assert result.value == pytest.approx(1.2)
    assert result.selected == "1.0" and sum(result.probabilities) == pytest.approx(1)
    with pytest.raises(ValueError):
        read_decision(question, [float("nan"), 0, 1])


@pytest.mark.parametrize(
    "invalid",
    [
        {},
        {"x": {"type": "boolean", "question": " "}},
        {"x": {"type": "choice", "question": "Q", "options": [{"id": "a", "description": "a"}]}},
        {
            "x": {
                "type": "choice",
                "question": "Q",
                "options": [
                    {"id": "a", "description": "a"},
                    {"id": "a", "description": "b"},
                ],
            }
        },
        {
            "x": {
                "type": "score",
                "question": "Q",
                "levels": [
                    {"value": 2, "description": "a"},
                    {"value": 1, "description": "b"},
                ],
            }
        },
        {"x": {"type": "boolean", "question": "Q", "temperature": float("inf")}},
    ],
)
def test_invalid_decisions_are_rejected(invalid):
    """Reject empty, ambiguous and nonfinite requests before model execution."""
    with pytest.raises(ValidationError):
        DecisionRequest(snapshot_id="a" * 64, questions=invalid)
