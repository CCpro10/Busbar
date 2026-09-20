"""Apple GPU execution using MLX-LM's native model and branchable KV caches."""

import re
from pathlib import Path

from .base import BackendResult, PrefixState, length_batches


def _model_classes(config, *, dtype):
    """Use the loader's extension point to promote weights before FP32 sanitization.

    Qwen3.5 folds ``1 + norm.weight`` into RMSNorm while loading. Casting only
    afterwards would retain BF16 rounding and invalidate the FP32 HF reference.
    BF16 production loading keeps the upstream conversion unchanged.
    """
    import importlib

    import mlx.core as mx

    family = config.get("model_type")
    if family not in ("qwen2", "qwen3", "qwen3_5", "llama") or any(
        config.get(key) or config.get("text_config", {}).get(key)
        for key in ("quantization", "quantization_config", "model_file")
    ):
        raise ValueError("MLX supports native, dense, unquantized Qwen2/3/3.5 and Llama")
    module = importlib.import_module(f"mlx_lm.models.{family}")
    if dtype != "float32":
        return module.Model, module.ModelArgs

    class Float32Model(module.Model):
        """Preserve checkpoint precision before architecture-specific arithmetic."""

        def sanitize(self, weights):
            """Promote source weights before invoking the unchanged native sanitizer."""
            weights = {key: value.astype(mx.float32) for key, value in weights.items()}
            return super().sanitize(weights)

    return Float32Model, module.ModelArgs


class MLXBackend:
    """Dense ChatML models, including hybrid Qwen3.5 attention and recurrent state."""

    def __init__(
        self, model: str, revision: str, *, batch_size=8, max_tokens=8192, dtype="float16"
    ):
        """Pin checkpoint/tokenizer revisions and reject unsupported or quantized head layouts."""
        import mlx.core as mx
        from huggingface_hub import snapshot_download
        from mlx.utils import tree_map_with_path
        from mlx_lm.utils import DEFAULT_ALLOW_PATTERNS, load_model, load_tokenizer

        if not mx.metal.is_available():
            raise RuntimeError("MLX backend requires an Apple Silicon Mac with Metal")
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("revision must be a full immutable Hugging Face commit SHA")
        if (
            dtype not in ("float16", "bfloat16", "float32")
            or not 1 <= batch_size <= 32
            or max_tokens < 1
        ):
            raise ValueError("invalid dtype, batch size or context limit")
        self.mx = mx
        path = Path(
            snapshot_download(model, revision=revision, allow_patterns=DEFAULT_ALLOW_PATTERNS)
        )
        self.model, config = load_model(
            path, get_model_classes=lambda config: _model_classes(config, dtype=dtype)
        )
        self.tokenizer = load_tokenizer(
            path, {"trust_remote_code": False}, eos_token_ids=config.get("eos_token_id")
        )
        # MLX-LM sanitizes multimodal checkpoints into their native text model.
        # Preserve architecture-specific FP32 parameters such as GDN's A_log.
        cast = getattr(self.model, "cast_predicate", lambda _: True)
        self.model.update(
            tree_map_with_path(
                lambda path, value: value.astype(getattr(mx, dtype)) if cast(path) else value,
                self.model.parameters(),
            )
        )
        self.model = getattr(self.model, "language_model", self.model)
        text_config = config.get("text_config", config)
        self.model.eval()
        mx.eval(self.model.parameters())
        self.head = (
            self.model.model.embed_tokens
            if text_config.get("tie_word_embeddings")
            else self.model.lm_head
        )
        self.tied = bool(text_config.get("tie_word_embeddings"))
        self.batch_size = batch_size
        self.max_tokens = min(max_tokens, text_config["max_position_embeddings"])
        self.identity = {
            "backend": "mlx",
            "model": model,
            "revision": revision,
            "tokenizer_revision": revision,
            "dtype": dtype,
        }
        self._rows: dict[tuple[int, ...], object] = {}

    def prefill(self, token_ids) -> PrefixState:
        """Prefill in bounded chunks; skip all vocabulary projection at this stage."""
        from mlx_lm.models.cache import make_prompt_cache

        mx = self.mx
        cache = make_prompt_cache(self.model)
        for start in range(0, len(token_ids), 256):
            hidden = self.model.model(mx.array([token_ids[start : start + 256]]), cache=cache)
            mx.eval(hidden, [layer.state for layer in cache])
        return PrefixState(cache, sum(layer.nbytes for layer in cache))

    def _project(self, hidden, labels, projection):
        """Gather weights before projection; the full path remains a correctness reference."""
        mx = self.mx
        if projection == "full":
            vocabulary = self.head.as_linear(hidden) if self.tied else self.head(hidden)
            return [vocabulary[i, mx.array(ids)] for i, ids in enumerate(labels)]
        selected = []
        for i, ids in enumerate(labels):
            if ids not in self._rows:
                self._rows[ids] = self.head.weight[mx.array(ids)]
                mx.eval(self._rows[ids])
            selected.append(hidden[i] @ self._rows[ids].T)
        return selected

    def score(self, sequences, labels, prefix, projection) -> BackendResult:
        """Merge into new batch caches; never pass the stored prefix into a mutating forward."""
        mx = self.mx
        results = [None] * len(sequences)
        batches = 0
        for indices in length_batches(sequences, self.batch_size):
            # Native merge allocates independent buffers; branches cannot mutate the snapshot.
            cache = (
                None
                if prefix is None
                else [layer.merge([layer] * len(indices)) for layer in prefix.cache]
            )
            inputs = mx.array([sequences[i] for i in indices])
            hidden = self.model.model(inputs, cache=cache)[:, -1, :]
            projected = self._project(hidden, [labels[i] for i in indices], projection)
            mx.eval(projected)
            for index, values in zip(indices, projected, strict=True):
                results[index] = values.astype(mx.float32).tolist()
            batches += 1
        return BackendResult(results, sum(map(len, sequences)), batches)
