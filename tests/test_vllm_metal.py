"""Contract tests for engine-owned APC and exact candidate readout, without a GPU."""

import sys
from types import SimpleNamespace

import pytest

from busbar.backends.vllm_metal import VLLMMetalBackend
from busbar.schemas import ContextSpec


def test_model_loading_reserves_output_capacity_and_verifies_real_dtype(monkeypatch):
    """An exact input limit must leave one output slot; requested dtype is not evidence."""
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    monkeypatch.setattr("platform.machine", lambda: "arm64")
    monkeypatch.setattr("importlib.metadata.version", lambda _: "test")
    configuration = {}
    precision = {"parameter_dtypes": {"mlx.core.bfloat16": 100}, "kv_dtype": "mlx.core.bfloat16"}

    class MetalPlatform:
        """Stand in for the plugin without loading its accelerator libraries."""

    def create_engine(**kwargs):
        """Capture the actual engine limit while avoiding GPU work in the contract suite."""
        configuration.update(kwargs)
        return SimpleNamespace(
            get_tokenizer=lambda: object(),
            collective_rpc=lambda *args, **kwargs: [precision],
        )

    monkeypatch.setitem(
        sys.modules, "vllm", SimpleNamespace(LLM=create_engine, SamplingParams=object)
    )
    monkeypatch.setitem(
        sys.modules, "vllm.platforms", SimpleNamespace(current_platform=MetalPlatform())
    )
    monkeypatch.setitem(
        sys.modules, "vllm_metal.platform", SimpleNamespace(MetalPlatform=MetalPlatform)
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoConfig=SimpleNamespace(
                from_pretrained=lambda *a, **k: SimpleNamespace(
                    model_type="qwen2", max_position_embeddings=8192
                )
            )
        ),
    )
    adapter = VLLMMetalBackend("test", "a" * 40, max_tokens=8192)
    assert adapter.max_tokens == 8191 and configuration["max_model_len"] == 8192
    precision["parameter_dtypes"] = {"mlx.core.float16": 100}
    with pytest.raises(ValueError, match="did not cast"):
        VLLMMetalBackend("test", "a" * 40)
    with pytest.raises(ValueError, match="bfloat16"):
        VLLMMetalBackend("test", "a" * 40, dtype="float16")


class Engine:
    """Expose exact request salts, candidate IDs and reported cache hits."""

    def __init__(self):
        """Keep deterministic state for namespace and reset assertions."""
        self.calls = []
        self.cached = 0
        self.resets = 0

    def generate(self, prompts, parameters, use_tqdm=False):
        """Return candidate log probabilities, including low-probability alternatives."""
        self.calls.append((prompts, parameters))
        if not isinstance(parameters, list):
            return []
        return [
            SimpleNamespace(
                prompt_token_ids=prompt["prompt_token_ids"],
                num_cached_tokens=self.cached,
                outputs=[
                    SimpleNamespace(
                        logprobs=[
                            {
                                token: SimpleNamespace(logprob=-100.0 - i)
                                for i, token in enumerate(parameter.logprob_token_ids)
                            }
                        ]
                    )
                ],
            )
            for prompt, parameter in zip(prompts, parameters, strict=True)
        ]

    def reset_prefix_cache(self):
        """Report a successful public engine cache reset."""
        self.resets += 1
        return True


def backend():
    """Skip heavyweight model loading while exercising the production adapter methods."""
    result = VLLMMetalBackend.__new__(VLLMMetalBackend)
    result.engine = Engine()
    result.SamplingParams = SimpleNamespace
    return result


def test_cached_and_fresh_salts_and_real_token_accounting():
    """APC hits may extend into a repeated question; reported prefix reuse must be capped."""
    adapter = backend()
    prefix = adapter.prefill(tuple(range(20)))
    adapter.engine.cached = 22
    output = adapter.score([(90, 91, 92)], [(32, 33, 34)], prefix, "full")
    assert output.computed_tokens == 1
    assert output.reused_tokens == 20
    assert output.details["engine_cached_tokens"] == 22
    assert output.logits == [[-100, -101, -102]]
    prompts, params = adapter.engine.calls[-1]
    assert prompts[0]["prompt_token_ids"] == list(range(20)) + [90, 91, 92]
    assert prompts[0]["cache_salt"] == prefix.cache.salt
    assert params[0].logprob_token_ids == [32, 33, 34]
    # Distinct salts isolate both fresh siblings and independent snapshot lifetimes.
    adapter.engine.cached = 0
    fresh = adapter.score([(1, 2), (1, 2)], [(32, 33)] * 2, None, "full")
    assert fresh.computed_tokens == 4 and fresh.reused_tokens == 0
    salts = [p["cache_salt"] for p in adapter.engine.calls[-1][0]]
    assert len(set(salts + [prefix.cache.salt])) == 3
    assert adapter.prefill(tuple(range(20))).cache.salt != prefix.cache.salt


def test_unsupported_projection_and_missing_candidate_fail_closed():
    """A full-head backend cannot silently advertise selected-row execution."""
    adapter = backend()
    with pytest.raises(ValueError, match="full vocabulary"):
        adapter.score([(1,)], [(32, 33)], None, "selected")
    adapter.engine.generate = lambda *args, **kwargs: [
        SimpleNamespace(
            prompt_token_ids=[1],
            num_cached_tokens=0,
            outputs=[SimpleNamespace(logprobs=[{32: SimpleNamespace(logprob=-1)}])],
        )
    ]
    with pytest.raises(RuntimeError, match="omitted"):
        adapter.score([(1,)], [(32, 33)], None, "full")


def test_auto_projection_and_global_clear(runtime, backend, questions):
    """The existing HTTP/runtime defaults resolve to backend capability and clear engine APC."""
    backend.default_projection = "full"
    resets = []
    backend.reset_cache = lambda: resets.append(True)
    snapshot = runtime.compile_context(ContextSpec(state="sample"))
    result = runtime.decide({"snapshot_id": snapshot.snapshot.id, "questions": questions})
    assert result.projection == "full"
    runtime.clear("default")
    assert not resets
    runtime.clear()
    assert resets == [True]


def test_engine_reported_cache_counts_are_not_replaced_by_assumed_hits(runtime, backend, questions):
    """An evicted engine prefix must not be advertised as a complete warm-cache hit."""
    from busbar.backends.base import BackendResult

    original = backend.score

    def with_engine_counts(*args):
        """Model a partial block hit reported by an engine-managed cache."""
        result = original(*args)
        return BackendResult(
            result.logits,
            99,
            1,
            reused_tokens=16,
            details={"engine_cached_tokens": 20, "logit_space": "vocabulary_logprobs"},
        )

    backend.score = with_engine_counts
    key = runtime.compile_context(ContextSpec(state="sample")).snapshot.id
    result = runtime.decide({"snapshot_id": key, "questions": questions})
    assert result.reused_prefix_tokens == 16 and result.computed_tokens == 99
    assert result.backend_details["engine_cached_tokens"] == 20
    assert all(row.logit_space == "vocabulary_logprobs" for row in result.decisions.values())
