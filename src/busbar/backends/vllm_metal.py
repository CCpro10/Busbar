"""vLLM Metal execution through its public LLM API and engine-owned prefix cache."""

import importlib.metadata
import platform
import re
import uuid
from dataclasses import dataclass

from .base import BackendResult, PrefixState


class PrecisionInspector:
    """Use vLLM's worker extension API for a read-only precision check, with named RPC."""

    def busbar_precision(self):
        """Report scalar dtype counts without copying weights or enabling pickle RPC."""
        from collections import Counter

        from mlx.utils import tree_flatten

        runner = self.model_runner
        counts = Counter()
        tensors = tree_flatten(runner.model.parameters())
        # The pinned plugin hides original attention modules behind _inner;
        # MLX's public parameters() walk excludes those private wrapper members.
        for layer in runner.model.layers:
            inner = getattr(layer.self_attn, "_inner", None)
            if inner is not None:
                tensors.extend(tree_flatten(inner.parameters()))
        seen = set()
        for _, value in tensors:
            if id(value) in seen:
                continue
            seen.add(id(value))
            counts[str(value.dtype)] += value.size
        return {"parameter_dtypes": dict(counts), "kv_dtype": str(runner.kv_cache_dtype)}


@dataclass(frozen=True)
class EnginePrefix:
    """Tokens plus a private APC salt; physical KV lifetime belongs to the vLLM engine."""

    tokens: tuple[int, ...]
    salt: str


class VLLMMetalBackend:
    """Read candidate log probabilities from one generated token, using native paged APC.

    The engine computes the full vocabulary head. Candidate probabilities have the
    same softmax ratios as raw logits, but the returned scores have a shared offset.
    Individual snapshot deletion revokes access; cached physical blocks remain in
    the engine's bounded pool until eviction or explicit global reset.
    """

    default_projection = "full"

    def __init__(
        self,
        model: str,
        revision: str,
        *,
        batch_size=8,
        max_tokens=8192,
        dtype="bfloat16",
        memory_fraction=0.35,
    ):
        """Create a private engine in the separately installed, pinned Metal environment."""
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise RuntimeError("vllm-metal requires an Apple Silicon Mac")
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("revision must be a full immutable Hugging Face commit SHA")
        if dtype != "bfloat16" or not 1 <= batch_size <= 32 or max_tokens < 2:
            raise ValueError(
                "vllm-metal validates bfloat16 weights/KV only, batch 1–32 and token limit >= 2"
            )
        if not 0 < memory_fraction <= 0.8:
            raise ValueError("memory fraction must be in (0, 0.8]")
        from transformers import AutoConfig
        from vllm import LLM, SamplingParams
        from vllm.platforms import current_platform
        from vllm_metal.platform import MetalPlatform

        if not isinstance(current_platform, MetalPlatform):
            raise RuntimeError("vLLM Metal plugin is not active; use the documented environment")
        config = AutoConfig.from_pretrained(model, revision=revision, trust_remote_code=False)
        if config.model_type not in ("qwen2", "qwen3", "llama") or getattr(
            config, "quantization_config", None
        ):
            raise ValueError(
                "Busbar Metal validates dense, unquantized Qwen2/3 and Llama checkpoints"
            )
        engine_limit = min(max_tokens, config.max_position_embeddings)
        # Unlike native readout, vLLM requires room for its one generated token.
        # Expose the true input capacity so Runtime rejects overflow before inference.
        self.max_tokens = engine_limit - 1
        self.batch_size = batch_size
        self.SamplingParams = SamplingParams
        self.engine = LLM(
            model=model,
            revision=revision,
            tokenizer_revision=revision,
            dtype=dtype,
            max_model_len=engine_limit,
            max_num_seqs=batch_size,
            max_num_batched_tokens=2048,
            gpu_memory_utilization=memory_fraction,
            enable_prefix_caching=True,
            enforce_eager=True,
            trust_remote_code=False,
            disable_log_stats=True,
            worker_extension_cls="busbar.backends.vllm_metal.PrecisionInspector",
            seed=0,
        )
        self.tokenizer = self.engine.get_tokenizer()
        actual_precision = self.engine.collective_rpc("busbar_precision", timeout=30)
        if not actual_precision or any(
            set(item["parameter_dtypes"]) != {f"mlx.core.{dtype}"}
            or item["kv_dtype"] != f"mlx.core.{dtype}"
            for item in actual_precision
        ):
            raise ValueError(
                "vLLM Metal did not cast checkpoint weights to the requested dtype; "
                "use --dtype bfloat16 for the validated checkpoints"
            )
        self.identity = {
            "backend": "vllm-metal",
            "model": model,
            "revision": revision,
            "tokenizer_revision": revision,
            "dtype": dtype,
            "actual_precision": actual_precision,
            "vllm": importlib.metadata.version("vllm"),
            "vllm_metal": importlib.metadata.version("vllm-metal"),
            "cache_owner": "engine; snapshot cache_bytes counts no physical KV",
            "memory_fraction": memory_fraction,
            "readout": "full vocabulary logprobs; one generated token per decision",
        }

    def prefill(self, token_ids) -> PrefixState:
        """Prime APC with a one-token request; vLLM retains only complete prefix blocks."""
        prefix = EnginePrefix(tuple(token_ids), uuid.uuid4().hex)
        self.engine.generate(
            [{"prompt_token_ids": list(prefix.tokens), "cache_salt": prefix.salt}],
            self.SamplingParams(temperature=0, max_tokens=1, detokenize=False),
            use_tqdm=False,
        )
        return PrefixState(prefix, 0)

    def reset_cache(self):
        """Clear this private engine's APC for reproducible cold measurements."""
        if not self.engine.reset_prefix_cache():
            raise RuntimeError("vLLM refused to reset its prefix cache")

    def score(self, sequences, labels, prefix, projection) -> BackendResult:
        """Read every declared candidate; never replace missing labels by top-k guesses."""
        if projection != "full":
            raise ValueError(
                "vllm-metal uses full vocabulary projection; request projection='full'"
            )
        prompts, parameters = [], []
        for sequence, slots in zip(sequences, labels, strict=True):
            tokens = (prefix.cache.tokens if prefix else ()) + tuple(sequence)
            # Fresh requests get distinct salts, so neither prior requests nor sibling
            # questions can silently turn a fresh baseline into an APC benchmark.
            salt = prefix.cache.salt if prefix else uuid.uuid4().hex
            prompts.append({"prompt_token_ids": list(tokens), "cache_salt": salt})
            parameters.append(
                self.SamplingParams(
                    temperature=0.0,
                    max_tokens=1,
                    logprob_token_ids=list(slots),
                    detokenize=False,
                    seed=0,
                )
            )
        outputs = self.engine.generate(prompts, parameters, use_tqdm=False)
        if len(outputs) != len(labels):
            raise RuntimeError("vLLM returned an incomplete decision batch")
        scores, computed, reused, all_cached = [], 0, 0, 0
        for output, slots, prompt in zip(outputs, labels, prompts, strict=True):
            if output.prompt_token_ids != prompt["prompt_token_ids"]:
                raise RuntimeError("vLLM changed the prompt or reordered the batch")
            row = output.outputs[0].logprobs
            if not row or any(token not in row[0] for token in slots):
                raise RuntimeError("vLLM omitted requested candidate log probabilities")
            scores.append([float(row[0][token].logprob) for token in slots])
            cached = output.num_cached_tokens
            if cached is None or not 0 <= cached <= len(prompt["prompt_token_ids"]):
                raise RuntimeError("vLLM did not return valid APC accounting")
            all_cached += cached
            reused += min(cached, len(prefix.cache.tokens)) if prefix else 0
            computed += len(prompt["prompt_token_ids"]) - cached
        return BackendResult(
            scores,
            computed,
            1,
            reused_tokens=reused,
            details={
                "engine_cached_tokens": all_cached,
                "generated_tokens": len(outputs),
                "batches_meaning": "one engine submission; internal scheduling is engine-owned",
                "logit_space": "vocabulary_logprobs",
            },
        )
