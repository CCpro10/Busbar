"""Small real-gradient tests on Apple Silicon; no model download or generated mock scores."""

import sys

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="native MLX training")


def test_real_optimizer_updates_head_and_reload_preserves_scores(tmp_path):
    """Learn a separable task, mask padding and round-trip the exact trained parameters."""
    mx = pytest.importorskip("mlx.core")
    from busbar.backends.mlx_head import CandidateHead, load_head
    from busbar.head_artifact import HeadArtifact, HeadManifest
    from busbar.training import TrainingConfig, fit_head

    mx.random.seed(4)
    head = CandidateHead(4)
    features = mx.array(
        [
            [[2.0, -1.0, 0.0, 0.0], [-1.0, 2.0, 0.0, 0.0], [100.0, 0.0, 0.0, 0.0]],
            [[-1.0, 2.0, 0.0, 0.0], [2.0, -1.0, 0.0, 0.0], [100.0, 0.0, 0.0, 0.0]],
        ]
    )
    data = (features, mx.array([[True, True, False]] * 2), mx.array([0, 1]))
    report = fit_head(head, data, data, TrainingConfig(epochs=15, learning_rate=0.05))
    assert report["parameter_delta_l2"] > 0 and report["trainable_parameters"] == 12
    assert report["selected_validation"]["accuracy"] == 1
    assert report["selected_validation"]["nll"] < report["history"][0]["validation"]["nll"]
    head.save_weights(str(tmp_path / "head.safetensors"))
    manifest = HeadManifest(
        model="test",
        revision="a" * 40,
        dtype="float32",
        hidden_size=4,
        weights_sha256="b" * 64,
        training_sha256="c" * 64,
    )
    weights = (tmp_path / "head.safetensors").read_bytes()
    loaded = load_head(HeadArtifact(manifest, weights, {}))
    assert head(features).tolist() == loaded(features).tolist()
    wrong = manifest.model_copy(update={"hidden_size": 8})
    with pytest.raises(ValueError, match="tensors"):
        load_head(HeadArtifact(wrong, weights, {}))
    mx.save_safetensors(str(tmp_path / "head.safetensors"), {"extra": mx.array([float("nan")])})
    with pytest.raises(ValueError, match="tensors"):
        load_head(HeadArtifact(manifest, (tmp_path / "head.safetensors").read_bytes(), {}))


def test_training_preflight_never_calls_backbone_for_overlong_input(backend):
    """A bad final example cannot leave a partially encoded training run."""
    pytest.importorskip("mlx.core")
    from types import SimpleNamespace

    from busbar.datasets import LabeledDecision
    from busbar.training import extract_features

    backend.tokenizer.eos_token_id = 0
    backend.model = SimpleNamespace(
        model=SimpleNamespace(embed_tokens=SimpleNamespace(weight=SimpleNamespace(shape=(10, 4))))
    )
    backend.max_tokens = 10
    row = LabeledDecision(
        id="a",
        group_id="a",
        context={"state": "x"},
        question={"type": "boolean", "question": "Q"},
        label=True,
    )
    with pytest.raises(ValueError, match="token limit"):
        extract_features(backend, [row], 1)
    assert not backend.prefills


def test_verified_head_uses_captured_weights_after_files_change(tmp_path):
    """A replacement between verification and tensor decoding cannot change the loaded model."""
    mx = pytest.importorskip("mlx.core")
    from busbar.backends.mlx_head import CandidateHead, load_head
    from busbar.head_artifact import publish_head, read_artifact

    head = CandidateHead(4)
    features = mx.array([[2.0, -1.0, 0.0, 1.0]])
    expected = head(features).tolist()
    provenance = {
        "file": "train.jsonl",
        "sha256": "a" * 64,
        "rows": 1,
        "ids": ["ticket-a"],
        "groups": ["ticket"],
        "fingerprints": ["b" * 64],
    }
    publish_head(
        tmp_path,
        head,
        {"model": "test", "revision": "c" * 40, "dtype": "float32"},
        4,
        {"datasets": {"train": provenance, "validation": provenance}},
    )
    artifact = read_artifact(tmp_path)
    (tmp_path / "head.safetensors").write_bytes(b"replacement weights")
    (tmp_path / "training.json").write_text("{}")
    assert load_head(artifact)(features).tolist() == expected
    assert artifact.training_splits["train"] == provenance
    with pytest.raises(ValueError, match="integrity failure"):
        read_artifact(tmp_path)


def test_training_refuses_publication_when_a_source_changes(tmp_path, backend, monkeypatch):
    """Editing data while fitting cannot produce a loadable version with misleading provenance."""
    pytest.importorskip("mlx.core")
    import json
    from types import SimpleNamespace

    from busbar.head_artifact import read_artifact
    from busbar.training import train_head

    sources = [tmp_path / "train.jsonl", tmp_path / "validation.jsonl"]
    for index, path in enumerate(sources):
        path.write_text(
            json.dumps(
                {
                    "id": str(index),
                    "group_id": str(index),
                    "context": {"state": index},
                    "question": {"type": "boolean", "question": "Q"},
                    "label": True,
                }
            )
        )
    backend.model = SimpleNamespace(
        model=SimpleNamespace(embed_tokens=SimpleNamespace(weight=SimpleNamespace(shape=(10, 4))))
    )

    def edit_while_fitting(*args):
        """Represent a concurrent editor without relying on nondeterministic thread timing."""
        sources[0].write_text("modified after feature extraction")
        return None, {"best_epoch": 1}

    monkeypatch.setattr("busbar.backends.load_backend", lambda *args, **kwargs: backend)
    monkeypatch.setattr("busbar.training._fit_features", edit_while_fitting)
    output = tmp_path / "new-head"
    with pytest.raises(ValueError, match="changed during training"):
        train_head(*sources, output)
    assert not (output / "manifest.json").exists()
    with pytest.raises(ValueError, match="incomplete head"):
        read_artifact(output)
