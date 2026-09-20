"""Regression scenarios for input snapshots, artifact publication and early validation."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from busbar.backends.base import BackendOutputError
from busbar.cli import main
from busbar.datasets import dataset_provenance, parse_dataset
from busbar.evaluation import evaluate
from busbar.storage import FileSnapshot, write_json


def categorical_row():
    """Return one valid external record whose label can change independently of its ID."""
    return {
        "id": "ticket",
        "group_id": "g",
        "family": "route",
        "state": "same",
        "question": "Which team?",
        "label": 0,
        "options": [{"id": "a", "description": "Returns"}, {"id": "b", "description": "Delivery"}],
    }


def test_evaluation_hash_and_labels_come_from_the_same_read(backend, tmp_path, monkeypatch):
    """Replacing a file after its first read cannot attach an old digest to new predictions."""
    source = tmp_path / "data.jsonl"
    original = json.dumps(categorical_row()).encode()
    changed = json.dumps({**categorical_row(), "label": 1}).encode()
    source.write_bytes(original)
    read_bytes = Path.read_bytes
    reads = []

    def replace_after_read(path):
        """Simulate an editor publishing new data immediately after the reader captured bytes."""
        content = read_bytes(path)
        if path == source:
            reads.append(content)
            source.write_bytes(changed)
        return content

    monkeypatch.setattr(Path, "read_bytes", replace_after_read)
    report = evaluate(backend, source)
    assert reads == [original]
    assert report["predictions"][0]["gold_id"] == "a"
    assert report["input_sha256"] == hashlib.sha256(original).hexdigest()


def test_training_provenance_retains_captured_data_and_detects_edits(tmp_path):
    """An edit cannot change provenance after labels were parsed; publication detects the drift."""
    source = tmp_path / "train.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": "a",
                "group_id": "a",
                "context": {"state": "x"},
                "question": {"type": "boolean", "question": "Q"},
                "label": True,
            }
        )
    )
    captured = FileSnapshot.read(source)
    rows = parse_dataset(captured)
    source.write_text("changed during training")
    assert (
        dataset_provenance(captured, rows)["sha256"] == hashlib.sha256(captured.payload).hexdigest()
    )
    with pytest.raises(ValueError, match="changed during training"):
        captured.assert_unchanged()


def test_last_malformed_external_row_never_executes_a_model(backend, tmp_path):
    """Missing provenance keys used to fail only after warm-up and inference had already run."""
    bad = {**categorical_row(), "id": "last"}
    del bad["family"]
    source = tmp_path / "data.jsonl"
    source.write_text(json.dumps(categorical_row()) + "\n" + json.dumps(bad))
    with pytest.raises(ValueError, match="data.jsonl:2"):
        evaluate(backend, source)
    assert backend.prefills == [] and backend.forwards == []


def test_external_metadata_preserves_human_readable_names(backend, tmp_path):
    """Family/group metadata need not follow the restricted IDs used as runtime question keys."""
    row = {**categorical_row(), "group_id": "来源/客服 1", "family": "客服 分流"}
    source = tmp_path / "external.jsonl"
    source.write_text(json.dumps(row))
    report = evaluate(backend, source)
    assert report["families"]["客服 分流"]["rows"] == 1
    assert report["predictions"][0]["group_id"] == row["group_id"]


def test_nonfinite_provenance_is_rejected_before_inference(backend, tmp_path):
    """Metadata that cannot be published as JSON must fail before spending model compute."""
    source = tmp_path / "external.jsonl"
    source.write_text(json.dumps({**categorical_row(), "provenance": {"score": float("nan")}}))
    with pytest.raises(ValueError, match="external.jsonl:1"):
        evaluate(backend, source)
    assert backend.prefills == [] and backend.forwards == []


def test_failed_evaluation_releases_its_current_snapshot(runtime, backend, tmp_path, monkeypatch):
    """The current group cannot retain model KV when inference aborts an evaluation."""
    source = tmp_path / "external.jsonl"
    source.write_text(json.dumps(categorical_row()))
    score = backend.score

    def fail_after_warmup(*args):
        """Allow the independent warm-up, then fail the first evaluated request."""
        if backend.forwards:
            raise BackendOutputError("simulated scorer failure")
        return score(*args)

    monkeypatch.setattr(backend, "score", fail_after_warmup)
    monkeypatch.setattr("busbar.evaluation.Runtime", lambda selected: runtime)
    with pytest.raises(BackendOutputError, match="simulated"):
        evaluate(backend, source)
    assert runtime.stats()["contexts"] == 0 and runtime.stats()["cache_bytes"] == 0


@pytest.mark.parametrize("payload", [[], {"context": {"state": "x"}, "questions": {}}])
def test_cli_rejects_invalid_one_shot_input_before_loading(tmp_path, monkeypatch, payload, capsys):
    """Malformed CLI input must not trigger model downloads or a costly prefill."""
    source = tmp_path / "request.json"
    source.write_text(json.dumps(payload))
    monkeypatch.setattr("busbar.cli.load_backend", lambda *a, **kw: pytest.fail("model loaded"))
    with pytest.raises(SystemExit) as error:
        main(["run", str(source)])
    assert error.value.code == 2 and "busbar:" in capsys.readouterr().err


def test_report_publication_is_complete_and_has_one_winner(tmp_path):
    """Concurrent writers must never overwrite a version, interleave bytes or leave temp files."""
    target = tmp_path / "result.json"

    def publish(value):
        """Return which contender acquired the filename without suppressing unrelated failures."""
        try:
            write_json(target, value)
            return value
        except FileExistsError:
            return None

    contenders = [{"winner": "a", "rows": [1] * 1000}, {"winner": "b", "rows": [2] * 1000}]
    with ThreadPoolExecutor(max_workers=2) as pool:
        winners = [value for value in pool.map(publish, contenders) if value is not None]
    assert len(winners) == 1 and json.loads(target.read_bytes()) == winners[0]
    assert list(tmp_path.iterdir()) == [target]


def test_failed_json_publication_leaves_no_partial_target(tmp_path, monkeypatch):
    """Serialization and filesystem failures cannot publish a misleading ready JSON file."""
    target = tmp_path / "manifest.json"
    with pytest.raises(ValueError):
        write_json(target, {"invalid": float("nan")})
    assert not list(tmp_path.iterdir())

    def fail_link(source, destination):
        """Verify staging is complete, then simulate a failed atomic publication."""
        assert json.loads(Path(source).read_text()) == {"valid": True}
        assert not Path(destination).exists()
        raise OSError("simulated I/O failure")

    monkeypatch.setattr("busbar.storage.os.link", fail_link)
    with pytest.raises(OSError, match="simulated"):
        write_json(target, {"valid": True})
    assert not list(tmp_path.iterdir())
