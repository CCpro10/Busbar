"""Real artifact checks: no vocabulary projection, isolated caches and reorderable options."""

import json
import os
from pathlib import Path

import pytest

from busbar import DecisionRequest, Runtime
from busbar.backends import load_backend
from busbar.datasets import read_dataset
from busbar.schemas import Option

pytestmark = [
    pytest.mark.model,
    pytest.mark.skipif(not os.environ.get("BUSBAR_TEST_HEAD"), reason="trained artifact is opt-in"),
]


def test_trained_head_reloads_and_preserves_every_candidate_across_branches():
    """Same state/options yield the same learned scores after reordering, fanout and cache reuse."""
    backend = load_backend(head=os.environ["BUSBAR_TEST_HEAD"])

    class UnusableVocabulary:
        """Raise if inference accidentally falls back to either vocabulary projection path."""

        def __getattr__(self, name):
            """Catch selected-row access to the vocabulary weights."""
            raise AssertionError(f"vocabulary head must not be used: {name}")

        def __call__(self, *args):
            """Catch complete vocabulary matrix projection."""
            raise AssertionError("vocabulary head must not be called")

    backend.head = UnusableVocabulary()
    runtime = Runtime(backend)
    source = Path(__file__).parents[1] / "examples/trainable_support/test.jsonl"
    rows = read_dataset(source)[:3]
    spec = rows[0].context
    context = runtime.compile_context(spec)
    request = DecisionRequest(
        namespace=spec.namespace,
        snapshot_id=context.snapshot.id,
        questions={row.question.type: row.question for row in rows},
    )
    cached = runtime.decide(request)
    fresh = runtime.decide(request.model_copy(update={"mode": "fresh"}))
    tolerance = 0.002 if backend.identity["dtype"] == "float32" else 0.03
    for name in request.questions:
        assert cached.decisions[name].selected == fresh.decisions[name].selected
        assert cached.decisions[name].logits == pytest.approx(
            fresh.decisions[name].logits, abs=tolerance, rel=0
        )
    assert cached.projection == "head"
    assert cached.backend_details["vocabulary_projection"] is False
    retained_bytes = runtime.stats()["cache_bytes"]
    question = request.questions["choice"]
    reordered_question = question.model_copy(
        update={
            "options": tuple(
                option.model_copy(update={"id": f"renamed-{index}"})
                for index, option in enumerate(reversed(question.options))
            )
        }
    )
    reordered = runtime.decide(
        request.model_copy(update={"questions": {"choice": reordered_question}})
    )
    assert list(reversed(reordered.decisions["choice"].logits)) == pytest.approx(
        cached.decisions["choice"].logits, abs=tolerance, rel=0
    )
    fanout = runtime.decide(
        request.model_copy(update={"questions": {f"q{i}": question for i in range(16)}})
    )
    for decision in fanout.decisions.values():
        assert decision.logits == pytest.approx(
            cached.decisions["choice"].logits, abs=tolerance, rel=0
        )
    expanded = question.model_copy(
        update={
            "options": question.options
            + tuple(
                Option(id=f"other{i}", description=f"Unrelated support category number {i}")
                for i in range(16 - len(question.options))
            )
        }
    )
    expanded_result = runtime.decide(request.model_copy(update={"questions": {"choice": expanded}}))
    assert len(expanded_result.decisions["choice"].probabilities) == 16
    assert expanded_result.decisions["choice"].logits[: len(question.options)] == pytest.approx(
        cached.decisions["choice"].logits, abs=tolerance, rel=0
    )
    assert runtime.decide(request).decisions == cached.decisions
    assert runtime.stats()["cache_bytes"] == retained_bytes
    with pytest.raises(ValueError, match="projection"):
        runtime.decide(request.model_copy(update={"projection": "full"}))
    report = {
        "model": backend.identity,
        "cached": cached.model_dump(),
        "fresh": fresh.model_dump(),
        "reordered": reordered.model_dump(),
        "fanout": fanout.model_dump(),
        "expanded_16_candidates": expanded_result.model_dump(),
        "vocabulary_tripwire": "passed",
        "cache_bytes_unchanged": retained_bytes,
        "tolerance": tolerance,
    }
    if os.environ.get("BUSBAR_HEAD_REPORT"):
        with Path(os.environ["BUSBAR_HEAD_REPORT"]).open("x") as file:
            json.dump(report, file, indent=2)
            file.write("\n")
