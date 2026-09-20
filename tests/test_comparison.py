"""Cross-project comparisons must preserve labels, ordering and failed-row denominators."""

import importlib.util
from pathlib import Path

import pytest


def comparison_module(name="compare_projects"):
    """Load the standalone benchmark without importing any optional model framework."""
    path = Path(__file__).parents[1] / f"benchmarks/{name}.py"
    spec = importlib.util.spec_from_file_location("busbar_compare", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cross_project_option_order_and_incomplete_distribution():
    """Reordered semantic IDs must stay aligned with the original frozen gold position."""
    compare = comparison_module()
    row = {
        "id": "sample",
        "family": "routing",
        "group_id": "source",
        "label": 1,
        "options": [{"id": "z", "description": "No"}, {"id": "a", "description": "Yes"}],
    }
    result = compare.prediction(row, [0.1, 0.9])
    assert result["selected"] == "a" and result["correct"] and result["gold_index"] == 1
    for probabilities in ([1.0], [float("nan"), 0], [-0.2, 1.2]):
        with pytest.raises(ValueError, match="incomplete"):
            compare.prediction(row, probabilities)
    summary, _ = compare.summarize(
        [result, {"id": "failed", "family": "routing", "gold_id": "z", "correct": False}]
    )
    assert summary["accuracy"] == 0.5 and summary["coverage"] == 0.5
    assert "nll" not in summary


def test_controlled_comparison_rejects_changed_prompt_or_candidate_order():
    """Matching accuracy and row count cannot establish that two runs saw the same inputs."""
    import copy

    paired = comparison_module("summarize_comparison").paired
    row = {
        "id": "r1",
        "gold_id": "yes",
        "option_ids": ["yes", "no"],
        "prompt_sha256": "prompt-a",
        "input_tokens": 100,
        "selected": "yes",
        "probabilities": [0.75, 0.25],
        "logits": [1.1, 0.0],
    }
    report = {"input_sha256": "fixture", "model": {"revision": "a" * 40}, "predictions": [row]}
    assert paired(report, copy.deepcopy(report))["argmax_disagreements"] == 0
    for field, value in (("prompt_sha256", "prompt-b"), ("option_ids", ["no", "yes"])):
        changed = copy.deepcopy(report)
        changed["predictions"][0][field] = value
        with pytest.raises(AssertionError):
            paired(report, changed)
