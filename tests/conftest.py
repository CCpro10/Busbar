"""A deterministic backend for lifecycle tests; real accelerator tests live separately."""

import pytest

from busbar.backends.base import BackendResult, PrefixState
from busbar.runtime import Runtime


class Tokenizer:
    """Represent each character as one token so boundary regressions are easy to detect."""

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        """Mimic Qwen's closed-message boundary, without claiming to mimic model semantics."""
        text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        return text + ("<|im_start|>assistant\n" if add_generation_prompt else "")

    def encode(self, text, add_special_tokens=False):
        """Use a reversible encoding for exact-prefix tests."""
        return list(map(ord, text))

    def decode(self, ids):
        """Invert the deterministic test encoding."""
        return "".join(map(chr, ids))


class FakeBackend:
    """Expose submitted tokens to prove cache accounting without heavyweight dependencies."""

    tokenizer = Tokenizer()
    max_tokens = 8192

    def __init__(self):
        """Keep test observations isolated across runtime instances."""
        self.identity = {"model": "test", "revision": "a" * 40, "dtype": "float32"}
        self.prefills = []
        self.forwards = []

    def prefill(self, token_ids):
        """Retain immutable prefix data with a simple exact byte accounting model."""
        self.prefills.append(token_ids)
        return PrefixState(token_ids, len(token_ids) * 2)

    def score(self, sequences, labels, prefix, projection):
        """Produce identical logits for fresh and cached reconstruction of each full input."""
        self.forwards.append((sequences, prefix))
        logits = []
        for sequence, slots in zip(sequences, labels, strict=True):
            combined = (prefix.cache if prefix else ()) + sequence
            score = sum(combined) % 7
            logits.append([float(score + i) for i in range(len(slots))])
        return BackendResult(logits, sum(map(len, sequences)), len(sequences))


@pytest.fixture
def backend():
    """Return an isolated deterministic backend."""
    return FakeBackend()


@pytest.fixture
def runtime(backend):
    """Return the actual production runtime over a deterministic execution boundary."""
    return Runtime(backend)


@pytest.fixture
def questions():
    """Exercise all public decision types in one fanout."""
    return {
        "yes": {"type": "boolean", "question": "Is the item unused?"},
        "route": {
            "type": "choice",
            "question": "Which team?",
            "options": [
                {"id": "returns", "description": "Returns"},
                {"id": "delivery", "description": "Delivery"},
            ],
        },
        "score": {
            "type": "score",
            "question": "How suitable?",
            "levels": [
                {"value": 0, "description": "Not suitable"},
                {"value": 2, "description": "Suitable"},
            ],
        },
    }
