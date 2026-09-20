"""Apple GPU execution using MLX-LM's native model and branchable KV caches."""

import re

from .base import BackendResult, PrefixState, length_batches


class MLXBackend:
    """Dense Qwen2/3 adapter with native prefix reuse and suffix batching."""

    def __init__(
        self, model: str, revision: str, *, batch_size=8, max_tokens=8192, dtype="float16"
    ):
        """Pin checkpoint/tokenizer revisions and reject unsupported or quantized head layouts."""
        import mlx.core as mx
        from mlx_lm import load

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
        self.model, self.tokenizer, config = load(
            model,
            revision=revision,
            tokenizer_config={"trust_remote_code": False},
            return_config=True,
        )
        if config.get("model_type") not in ("qwen2", "qwen3") or config.get("quantization"):
            raise ValueError("MLX supports dense, unquantized Qwen2/3 checkpoints only")
        self.model.set_dtype(getattr(mx, dtype))
        self.model.eval()
        mx.eval(self.model.parameters())
        self.head = (
            self.model.model.embed_tokens
            if config.get("tie_word_embeddings")
            else self.model.lm_head
        )
        self.tied = bool(config.get("tie_word_embeddings"))
        self.batch_size = batch_size
        self.max_tokens = min(max_tokens, config["max_position_embeddings"])
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
        from mlx_lm.models.cache import BatchKVCache

        mx = self.mx
        results = [None] * len(sequences)
        batches = 0
        for indices in length_batches(sequences, self.batch_size):
            # Native merge allocates independent buffers; branches cannot mutate the snapshot.
            cache = (
                None
                if prefix is None
                else [BatchKVCache.merge([layer] * len(indices)) for layer in prefix.cache]
            )
            inputs = mx.array([sequences[i] for i in indices])
            hidden = self.model.model(inputs, cache=cache)[:, -1, :]
            projected = self._project(hidden, [labels[i] for i in indices], projection)
            mx.eval(projected)
            for index, values in zip(indices, projected, strict=True):
                results[index] = values.astype(mx.float32).tolist()
            batches += 1
        return BackendResult(results, sum(map(len, sequences)), batches)
