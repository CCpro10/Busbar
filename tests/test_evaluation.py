"""Evaluation denominators and confidence metrics must preserve failed and incorrect rows."""

import json

import pytest

from busbar.evaluation import evaluate, metrics


def test_probability_metrics_penalize_confident_errors_and_missing_answers():
    """A highly confident wrong answer scores worse than an uncertain correct answer."""
    rows = [
        {"gold_id": "yes", "gold_index": 0, "correct": True, "probabilities": [0.6, 0.4]},
        {"gold_id": "no", "gold_index": 1, "correct": False, "probabilities": [0.99, 0.01]},
    ]
    result = metrics(rows)
    assert result["accuracy"] == 0.5 and result["balanced_accuracy"] == 0.5
    assert result["brier"] == pytest.approx((0.32 + 1.9602) / 2)
    assert result["ece_10_equal_width"] == pytest.approx((0.4 + 0.99) / 2)
    missing = metrics(rows + [{"gold_id": "no", "gold_index": 1}])
    assert missing["accuracy"] == pytest.approx(1 / 3)
    assert missing["coverage"] == pytest.approx(2 / 3)
    assert "brier" not in missing


def test_complete_fixture_preserves_ids_labels_and_source_hash(backend, tmp_path):
    """All source rows survive state grouping; duplicate source IDs cannot be hidden."""
    rows = [
        {
            "id": f"row{i}",
            "family": "routing",
            "group_id": "g",
            "state": "same",
            "question": f"Question {i}?",
            "label": i % 2,
            "options": [{"id": "yes", "description": "Yes"}, {"id": "no", "description": "No"}],
        }
        for i in range(4)
    ]
    source = tmp_path / "fixture.jsonl"
    source.write_text("\n".join(map(json.dumps, rows)))
    result = evaluate(backend, source)
    assert result["summary"]["rows"] == 4 and result["summary"]["coverage"] == 1
    assert {p["id"] for p in result["predictions"]} == {p["id"] for p in rows}
    assert len(result["input_sha256"]) == 64 and len(result["timings"]) == 1
    source.write_text("\n".join(map(json.dumps, rows + rows[:1])))
    with pytest.raises(ValueError, match="unique IDs"):
        evaluate(backend, source)
