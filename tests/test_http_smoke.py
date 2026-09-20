"""A working protocol must never conceal a failed repeated-inference numerical gate."""

import importlib.util
from pathlib import Path


def test_repeated_choice_flip_remains_a_reported_failure():
    """Record the observed MiniCPM Metal tie flip while keeping the strict original tolerance."""
    path = Path(__file__).parents[1] / "examples/http_smoke.py"
    spec = importlib.util.spec_from_file_location("http_smoke", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    first = {"condition": {"selected": 2, "probabilities": [0.375, 0.007, 0.618]}}
    second = {"condition": {"selected": 0, "probabilities": [0.4965, 0.007, 0.4965]}}
    actual = module.repeat_checks(first, second)["condition"]
    assert not actual["passed"] and not actual["selected_unchanged"]
    assert actual["max_probability_delta"] > actual["probability_tolerance"]
    assert module.repeat_checks(first, first)["condition"]["passed"]
