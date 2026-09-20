"""Small real-gradient tests on Apple Silicon; no model download or generated mock scores."""

import sys

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="native MLX training")


def test_real_optimizer_updates_head_and_reload_preserves_scores(tmp_path):
    """Learn a separable task, mask padding and round-trip the exact trained parameters."""
    mx = pytest.importorskip("mlx.core")
    from busbar.backends.mlx_head import CandidateHead, load_head
    from busbar.head_artifact import HeadManifest
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
    loaded = load_head(tmp_path, manifest)
    assert head(features).tolist() == loaded(features).tolist()
    wrong = manifest.model_copy(update={"hidden_size": 8})
    with pytest.raises(ValueError, match="tensors"):
        load_head(tmp_path, wrong)
    mx.save_safetensors(str(tmp_path / "head.safetensors"), {"extra": mx.array([float("nan")])})
    with pytest.raises(ValueError, match="tensors"):
        load_head(tmp_path, manifest)


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
