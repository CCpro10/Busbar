"""Small native weight conversions must preserve the requested reference precision."""

import platform

import pytest

from busbar.backends.mlx import _model_classes

if platform.system() != "Darwin" or platform.machine() != "arm64":
    pytest.skip("MLX requires Apple Silicon", allow_module_level=True)
mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")


def test_fp32_conversion_promotes_before_folding_offset_norm():
    """BF16 checkpoint RMSNorm offsets must not round away when loading an FP32 reference."""
    config = {
        "model_type": "qwen3_5",
        "text_config": {
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "vocab_size": 32,
            "linear_num_key_heads": 1,
            "linear_num_value_heads": 1,
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
        },
    }
    weights = {
        "model.language_model.norm.weight": mx.array([0.003], dtype=mx.bfloat16),
        "model.language_model.layers.0.linear_attn.conv1d.weight": mx.zeros((24, 1, 4)),
    }
    cls, args = _model_classes(config, dtype="float32")
    converted = cls(args.from_dict(config)).sanitize(weights.copy())
    actual = converted["language_model.model.norm.weight"].item()
    expected = weights["model.language_model.norm.weight"].astype(mx.float32).item() + 1
    assert actual == expected and actual != 1.0
    # The production BF16 path must still use the original upstream conversion.
    cls, args = _model_classes(config, dtype="bfloat16")
    native = cls(args.from_dict(config)).sanitize(weights.copy())
    assert native["language_model.model.norm.weight"].dtype == mx.bfloat16


def test_recurrent_normalization_uses_sum_epsilon():
    """The pinned upstream revision fixes a Qwen3.5 epsilon error in release 0.31.3."""
    from mlx_lm.models.gated_delta import normalize_qk

    values = mx.full((1, 2, 128), 1e-4)
    q, k = normalize_qk(values, values, inv_scale=128**-0.5, eps=1e-6)
    expected = values * mx.rsqrt(mx.sum(values * values, axis=-1, keepdims=True) + 1e-6)
    assert mx.max(mx.abs(k - expected)).item() < 1e-6
    assert mx.max(mx.abs(q - expected * 128**-0.5)).item() < 1e-6
