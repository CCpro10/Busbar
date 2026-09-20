"""A warm benchmark must distinguish shared-state reuse from caching the questions too."""

from busbar.backends.base import BackendResult
from busbar.benchmark import benchmark


def test_engine_state_only_and_exact_repeat_have_distinct_work(backend):
    """Fully cached questions cannot accidentally inflate the reported state-only speedup."""
    backend.identity["backend"] = "vllm-metal"
    backend.default_projection = "full"
    seen = set()
    score = backend.score
    backend.reset_cache = seen.clear

    def apc_score(sequences, labels, prefix, projection):
        """Model a native engine retaining full question prefixes between calls."""
        result = score(sequences, labels, prefix, projection)
        computed = reused = 0
        for sequence in sequences:
            prompt = (prefix.cache if prefix else ()) + sequence
            if prefix and prompt in seen:
                computed += 1
            else:
                computed += len(sequence)
            reused += len(prefix.cache) if prefix else 0
            seen.add(prompt)
        return BackendResult(result.logits, computed, 1, reused_tokens=reused)

    backend.score = apc_score
    report = benchmark(backend, context_tokens=512, questions=2, repeats=2)
    state = report["summary"]["cached_batched"]["computed_tokens_per_run"]
    repeat = report["summary"]["cached_exact_repeat"]["computed_tokens_per_run"]
    assert state == sum(report["suffix_tokens"])
    assert repeat == 2 and repeat < state
    assert all(run["computed_tokens"] == state for run in report["runs"]["cached_batched"])
