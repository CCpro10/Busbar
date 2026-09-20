"""Compiler boundaries must fail closed before invalid candidate labels reach a backend."""

import pytest

from busbar.compiler import Compiler
from busbar.schemas import BooleanQuestion, ContextSpec


def test_suffix_token_cache_and_context_mutation(runtime, backend, questions):
    """Warm requests reuse question tokens; caller mutation cannot rewrite a snapshot."""
    state = {"answer": "original"}
    context = runtime.compile_context(ContextSpec(state=state))
    state["answer"] = "changed"
    stored_tokens = backend.prefills[0]
    runtime.decide({"snapshot_id": context.snapshot.id, "questions": questions})
    before = runtime.compiler._suffix.cache_info()
    runtime.decide({"snapshot_id": context.snapshot.id, "questions": questions})
    after = runtime.compiler._suffix.cache_info()
    assert after.hits == before.hits + len(questions)
    assert backend.prefills == [stored_tokens]
    assert "original" in backend.tokenizer.decode(stored_tokens)
    assert "changed" not in backend.tokenizer.decode(stored_tokens)


def test_multitoken_label_is_rejected_before_execution(backend, monkeypatch):
    """A tokenizer that splits A cannot silently produce a wrong categorical head."""
    encode = backend.tokenizer.encode

    def split_letter(text, add_special_tokens=False):
        """Simulate a model whose answer slots are not single-token labels."""
        return [1, 2] if text == "A" else encode(text, add_special_tokens=add_special_tokens)

    monkeypatch.setattr(backend.tokenizer, "encode", split_letter)
    compiler = Compiler(backend.tokenizer)
    with pytest.raises(ValueError, match="single token"):
        compiler.question(BooleanQuestion(type="boolean", question="Q"))


def test_empty_context_is_valid_but_nonfinite_state_is_not(runtime):
    """No additional evidence is a valid context; invalid numeric JSON is not."""
    assert runtime.compile_context({"state": None}).snapshot.token_count > 0
    with pytest.raises(ValueError):
        runtime.compile_context({"state": {"value": float("nan")}})
