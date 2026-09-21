"""Compiler boundaries must fail closed before invalid candidate labels reach a backend."""

import pytest
from pydantic import ValidationError

from busbar.compiler import LABELS, Compiler
from busbar.schemas import (
    MAX_ALTERNATIVES,
    BooleanQuestion,
    ChoiceQuestion,
    ContextSpec,
    Option,
)


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


def test_qwen25_default_system_is_not_injected_inside_cached_conversation(backend, monkeypatch):
    """A standalone-turn template must not add a second system message to the suffix."""
    original = backend.tokenizer.apply_chat_template

    def auto_system(messages, **kwargs):
        """Model Qwen2.5's default system insertion for conversations starting with user."""
        if messages[0]["role"] != "system":
            messages = [{"role": "system", "content": "DEFAULT"}] + messages
        return original(messages, **kwargs)

    monkeypatch.setattr(backend.tokenizer, "apply_chat_template", auto_system)
    compiler = Compiler(backend.tokenizer)
    prefix = compiler.context(ContextSpec(state="An unused item"))
    suffix = compiler.question(BooleanQuestion(type="boolean", question="Is it unused?"))
    text = backend.tokenizer.decode(prefix + suffix.paths[0].token_ids)
    assert text.count("<|im_start|>system") == 1
    assert "DEFAULT" not in text
    assert text.count("<|im_start|>user") == 2


def test_chatml_bos_is_kept_once_across_cached_question(backend, monkeypatch):
    """MiniCPM's conversation BOS belongs to the prefix, never to each cached question."""
    original = backend.tokenizer.apply_chat_template

    def with_bos(messages, **kwargs):
        """Represent a ChatML template that always adds one conversation BOS."""
        return "<s>" + original(messages, **kwargs)

    monkeypatch.setattr(backend.tokenizer, "apply_chat_template", with_bos)
    compiler = Compiler(backend.tokenizer)
    context = compiler.context(ContextSpec(state="blue box"))
    question = compiler.question(BooleanQuestion(type="boolean", question="Is the box blue?"))
    reconstructed = backend.tokenizer.decode(context + question.paths[0].token_ids)
    assert reconstructed.startswith("<s>") and reconstructed.count("<s>") == 1


def test_alternative_ceiling_matches_the_label_alphabet(backend):
    """The public ceiling is only honest if every permitted alternative has a distinct label.

    The ceiling moved from 16 to 26 because A-Z was measured at that width; this guards the pair of
    facts that makes it safe, so a later ceiling bump cannot silently outrun the alphabet.
    """
    assert len(LABELS) >= MAX_ALTERNATIVES
    assert len(set(LABELS[:MAX_ALTERNATIVES])) == MAX_ALTERNATIVES

    options = tuple(
        Option(id=f"o{i}", description=f"Alternative number {i}") for i in range(MAX_ALTERNATIVES)
    )
    widest = ChoiceQuestion(type="choice", question="Which one?", options=options)
    compiled = Compiler(backend.tokenizer).question(widest)
    labels = compiled.paths[0].label_ids
    assert len(labels) == MAX_ALTERNATIVES
    assert len(set(labels)) == MAX_ALTERNATIVES

    with pytest.raises(ValidationError):
        ChoiceQuestion(
            type="choice",
            question="Which one?",
            options=options + (Option(id="overflow", description="One alternative too many"),),
        )
