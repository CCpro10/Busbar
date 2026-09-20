"""Candidate-path, dataset and artifact boundaries without optional accelerator imports."""

import copy
import json

import pytest

from busbar.backends import load_backend
from busbar.compiler import CandidateCompiler
from busbar.datasets import LabeledDecision, assert_disjoint, file_sha256, read_dataset
from busbar.evaluation import evaluate
from busbar.head_artifact import HeadManifest, check_compatibility, read_manifest, write_json
from busbar.runtime import Runtime
from busbar.schemas import ContextSpec, DecisionRequest


def labeled(kind="choice"):
    """Supply a valid typed record with nondefault namespace and explicit instructions."""
    question = {"type": kind, "question": "Is this appropriate?"}
    label = True
    if kind == "choice":
        question["options"] = [{"id": "a", "description": "Yes"}, {"id": "b", "description": "No"}]
        label = "a"
    if kind == "score":
        question["levels"] = [
            {"value": 0, "description": "Low"},
            {"value": 2, "description": "High"},
        ]
        label = 2
    return {
        "id": kind,
        "group_id": "ticket1",
        "context": {"state": "hello", "namespace": "team", "instructions": "Be literal."},
        "question": question,
        "label": label,
    }


def candidate_backend(backend):
    """Use the real runtime with a scalar-output deterministic backbone boundary."""
    backend.tokenizer.eos_token_id = 0
    backend.compiler = CandidateCompiler(backend.tokenizer)
    backend.default_projection = "head"
    backend.identity["head_id"] = "a" * 64
    return backend


def test_candidate_paths_keep_ids_and_order_out_of_model_input(backend):
    """Applications can rename/reorder choices; each semantic candidate retains exact token IDs."""
    compiler = candidate_backend(backend).compiler
    row = LabeledDecision.model_validate(labeled())
    original = compiler.question(row.question)
    changed = row.question.model_copy(
        update={
            "options": tuple(
                option.model_copy(update={"id": f"new-{index}"})
                for index, option in enumerate(reversed(row.question.options))
            )
        }
    )
    assert original.paths == tuple(reversed(compiler.question(changed).paths))
    assert all(path.label_ids == (0,) and path.token_ids[-1] == 0 for path in original.paths)


@pytest.mark.parametrize("attributes", [{}, {"eos_token_id": None}, {"eos_token_id": True}])
def test_candidate_encoder_requires_a_real_eos_token(attributes):
    """Unsupported tokenizers fail with an input error instead of AttributeError or token True."""
    from types import SimpleNamespace

    with pytest.raises(ValueError, match="EOS token"):
        CandidateCompiler(SimpleNamespace(**attributes))


def test_candidate_fanout_regroups_three_types_and_accounts_every_path(backend, questions):
    """A cached question with K options reuses K prefix branches without altering the snapshot."""
    runtime = Runtime(candidate_backend(backend))
    context = runtime.compile_context(ContextSpec(state="unused"))
    request = DecisionRequest(snapshot_id=context.snapshot.id, questions=questions)
    cached = runtime.decide(request)
    fresh = runtime.decide(request.model_copy(update={"mode": "fresh"}))
    assert cached.projection == "head" and cached.decisions == fresh.decisions
    assert len(backend.forwards[0][0]) == 6
    assert cached.reused_prefix_tokens == 6 * context.snapshot.token_count
    assert fresh.computed_tokens - cached.computed_tokens == cached.reused_prefix_tokens
    assert runtime.stats()["cache_bytes"] == context.snapshot.cache_bytes
    backend.identity["head_id"] = "b" * 64
    assert (
        Runtime(backend).compile_context(ContextSpec(state="unused")).snapshot.id
        != context.snapshot.id
    )


def test_last_candidate_overflow_fails_before_any_forward(backend):
    """Preflight must inspect later candidate paths as well as the first one."""
    runtime = Runtime(candidate_backend(backend))
    context = runtime.compile_context({"state": "x"})
    question = labeled()["question"]
    question["options"][-1]["description"] = "x" * 8192
    with pytest.raises(ValueError, match="token limit"):
        runtime.decide(DecisionRequest(snapshot_id=context.snapshot.id, questions={"q": question}))
    assert backend.forwards == []


@pytest.mark.parametrize(
    "kind,label",
    [("boolean", "true"), ("choice", 0), ("score", True), ("score", 1), ("score", float("nan"))],
)
def test_typed_labels_reject_ambiguous_targets(kind, label):
    """Boolean, application ID and numeric-level supervision cannot silently interchange."""
    with pytest.raises(ValueError):
        LabeledDecision.model_validate({**labeled(kind), "label": label})


def test_split_leakage_cannot_be_hidden_by_renaming_or_reordering():
    """All questions for a ticket stay in one split; renamed duplicate inputs also fail."""
    original = LabeledDecision.model_validate(labeled())
    duplicate = copy.deepcopy(labeled())
    duplicate.update(id="renamed", group_id="another")
    duplicate["context"]["namespace"] = "different"
    duplicate["question"]["options"].reverse()
    duplicate["question"]["options"][0]["id"] = "c"
    with pytest.raises(ValueError, match="semantic inputs"):
        assert_disjoint([original], [LabeledDecision.model_validate(duplicate)])
    duplicate["context"]["state"] = "different"
    duplicate["group_id"] = original.group_id
    with pytest.raises(ValueError, match="group IDs"):
        assert_disjoint([original], [LabeledDecision.model_validate(duplicate)])


def test_dataset_errors_identify_source_line(tmp_path):
    """Empty and duplicate-ID files fail before model loading or partial training."""
    path = tmp_path / "data.jsonl"
    path.write_text("")
    with pytest.raises(ValueError, match="empty"):
        read_dataset(path)
    path.write_text((json.dumps(labeled()) + "\n") * 2)
    with pytest.raises(ValueError, match="data.jsonl:2: duplicate"):
        read_dataset(path)


def test_artifact_integrity_and_encoder_compatibility(tmp_path):
    """Incomplete, changed and incompatible head versions are never accepted as ready."""
    with pytest.raises(ValueError, match="incomplete"):
        read_manifest(tmp_path)
    weights = tmp_path / "head.safetensors"
    weights.write_bytes(b"integrity fixture; tensor validation lives in MLX tests")
    write_json(tmp_path / "training.json", {"datasets": {}})
    manifest = HeadManifest(
        model="test",
        revision="a" * 40,
        dtype="float32",
        hidden_size=8,
        weights_sha256=file_sha256(weights),
        training_sha256=file_sha256(tmp_path / "training.json"),
    )
    write_json(tmp_path / "manifest.json", manifest.model_dump())
    assert read_manifest(tmp_path) == manifest
    for kwargs in ({"model": "wrong"}, {"revision": "b" * 40}, {"dtype": "bfloat16"}):
        with pytest.raises(ValueError, match="mismatch"):
            check_compatibility(manifest, **kwargs)
    with pytest.raises(ValueError):
        HeadManifest.model_validate({**manifest.model_dump(), "encoder_version": "future"})
    with pytest.raises(FileExistsError):
        write_json(tmp_path / "manifest.json", {})
    weights.write_bytes(b"changed")
    with pytest.raises(ValueError, match="integrity"):
        read_manifest(tmp_path)


@pytest.mark.parametrize("name", ["hf", "vllm-metal"])
def test_unsupported_head_backend_is_rejected_before_optional_imports(name):
    """There is no silent fallback to vocabulary scores when a trained head is requested."""
    with pytest.raises(ValueError, match="native MLX"):
        load_backend(name, head="anything")


def test_vocabulary_backends_reject_head_projection_without_executing():
    """Invalid projection names cannot silently execute the old selected-row path."""
    from busbar.backends.hf import HFBackend
    from busbar.backends.mlx import MLXBackend

    for backend_type in (HFBackend, MLXBackend):
        backend = backend_type.__new__(backend_type)
        with pytest.raises(ValueError, match="projection"):
            backend.score([], [], None, "head")


def test_native_evaluation_preserves_instructions_and_prevents_leakage(tmp_path, backend):
    """The exact typed API input is scored; known fitting data needs an explicit override."""
    rows = [labeled(kind) for kind in ("boolean", "choice", "score")]
    source = tmp_path / "test.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    backend = candidate_backend(backend)
    backend.training_splits = {"train": {"ids": ["boolean"], "groups": [], "fingerprints": []}}
    with pytest.raises(ValueError, match="overlaps"):
        evaluate(backend, source, data_format="native")
    assert backend.prefills == []
    report = evaluate(backend, source, data_format="native", allow_training_data=True)
    assert report["summary"]["rows"] == 3 and report["summary"]["coverage"] == 1
    assert set(report["families"]) == {"boolean", "choice", "score"}
    assert any("not held-out" in item for item in report["limitations"])
    assert "Be literal." in backend.tokenizer.decode(backend.prefills[-1])
