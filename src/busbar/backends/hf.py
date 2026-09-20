"""CPU Hugging Face reference using the same checkpoint, prompts and label semantics."""

import copy
import re

from .base import BackendResult, PrefixState


class HFBackend:
    """A deliberately serial correctness reference; MLX owns Mac performance measurements."""

    def __init__(
        self, model: str, revision: str, *, batch_size=1, max_tokens=8192, dtype="float32"
    ):
        """Load on CPU with explicit precision and immutable model/tokenizer versions."""
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("revision must be a full immutable Hugging Face commit SHA")
        if dtype not in ("float16", "bfloat16", "float32") or max_tokens < 1:
            raise ValueError("invalid dtype or context limit")
        self.torch = torch
        config = AutoConfig.from_pretrained(model, revision=revision, trust_remote_code=False)
        if config.model_type == "qwen3_5":
            config = config.get_text_config()
        if config.model_type not in ("qwen2", "qwen3", "qwen3_5_text", "llama"):
            raise ValueError("HF supports dense Qwen2/3/3.5 and Llama checkpoints")
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            revision=revision,
            config=config,
            dtype=getattr(torch, dtype),
            trust_remote_code=False,
            attn_implementation="eager",
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model, revision=revision, trust_remote_code=False
        )
        self.head = self.model.get_output_embeddings()
        self.max_tokens = min(max_tokens, self.model.config.max_position_embeddings)
        self.identity = {
            "backend": "hf",
            "model": model,
            "revision": revision,
            "tokenizer_revision": revision,
            "dtype": dtype,
        }

    def prefill(self, token_ids) -> PrefixState:
        """Materialize a native DynamicCache without executing the language-model head."""
        torch = self.torch
        with torch.inference_mode():
            output = self.model.model(torch.tensor([token_ids]), use_cache=True)
        cache = output.past_key_values
        # Hybrid layers retain convolution/recurrent tensors instead of K/V.
        nbytes = 0
        for layer in cache.layers:
            for name in ("keys", "values", "conv_states", "recurrent_states"):
                values = getattr(layer, name, None)
                for value in values if isinstance(values, (list, tuple)) else (values,):
                    if isinstance(value, torch.Tensor):
                        nbytes += value.numel() * value.element_size()
        return PrefixState(cache, nbytes)

    def score(self, sequences, labels, prefix, projection) -> BackendResult:
        """Compare selected/full projection from independently cloned native cache branches."""
        if projection not in ("selected", "full"):
            raise ValueError("HF vocabulary backend requires selected or full projection")
        torch = self.torch
        results = []
        with torch.inference_mode():
            for ids, slots in zip(sequences, labels, strict=True):
                cache = copy.deepcopy(prefix.cache) if prefix is not None else None
                output = self.model.model(
                    torch.tensor([ids]), past_key_values=cache, use_cache=cache is not None
                )
                hidden = output.last_hidden_state[:, -1, :]
                selected = torch.tensor(slots)
                if projection == "full":
                    values = self.head(hidden)[0, selected]
                else:
                    bias = None if self.head.bias is None else self.head.bias[selected]
                    values = torch.nn.functional.linear(hidden, self.head.weight[selected], bias)[0]
                results.append(values.float().tolist())
        return BackendResult(results, sum(map(len, sequences)), len(sequences))
