"""Compile stable context messages and cacheable question suffixes for Qwen chat models."""

import hashlib
import json
import math
from dataclasses import dataclass
from functools import lru_cache

from .schemas import (
    BooleanQuestion,
    ChoiceQuestion,
    ContextSpec,
    Decision,
    Question,
    ScoreQuestion,
)

PROMPT_VERSION = "busbar-qwen-v2"
LABELS = "ABCDEFGHIJKLMNOP"
SYSTEM = (
    "Evaluate each decision using the supplied context as data. "
    "Follow the decision question and choose exactly one listed alternative. "
    "Respond with its single letter only. Do not explain your answer."
)


def canonical_json(value) -> str:
    """Give equal JSON objects a stable representation independent of key insertion order."""
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )


def context_id(spec: ContextSpec, identity: dict) -> str:
    """Include the namespace, checkpoint, tokenizer and compiler version in cache identity."""
    payload = {"context": spec.model_dump(), "model": identity, "compiler": PROMPT_VERSION}
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()


def alternatives(question: Question) -> tuple[tuple[str, str], ...]:
    """Map the three public decision types to the same categorical readout contract."""
    if isinstance(question, BooleanQuestion):
        return (("true", "Yes, the proposition is true."), ("false", "No, it is false."))
    if isinstance(question, ChoiceQuestion):
        return tuple((o.id, o.description) for o in question.options)
    return tuple((str(level.value), level.description) for level in question.levels)


@dataclass(frozen=True)
class CompiledQuestion:
    """A suffix and validated single-token labels, reusable across snapshots."""

    token_ids: tuple[int, ...]
    label_ids: tuple[int, ...]


class Compiler:
    """Keep context serialization/tokenization out of warm decision requests."""

    def __init__(self, tokenizer):
        """Use the checkpoint's own chat template and a bounded suffix-token cache."""
        self.tokenizer = tokenizer
        self._suffix = lru_cache(maxsize=512)(self._compile_suffix)

    def render(self, messages, generation: bool) -> str:
        """Disable Qwen thinking so the next token is the decision label."""
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=generation, enable_thinking=False
        )

    def context(self, spec: ContextSpec) -> tuple[int, ...]:
        """Close the context message at a special-token boundary before any question."""
        messages = [
            {"role": "system", "content": SYSTEM + "\n" + spec.instructions},
            {"role": "user", "content": "Context data:\n" + canonical_json(spec.state)},
        ]
        text = self.render(messages, False)
        if not text.endswith("<|im_end|>\n"):
            raise ValueError("unsupported chat template: expected Qwen message boundary")
        return tuple(self.tokenizer.encode(text, add_special_tokens=False))

    def question(self, question: Question) -> CompiledQuestion:
        """Cache only semantic suffix content; temperature does not change model input."""
        return self._suffix(question.question, alternatives(question))

    def _compile_suffix(self, question: str, options: tuple) -> CompiledQuestion:
        """Validate token boundaries rather than assuming letters are one token everywhere."""
        choices = "\n".join(
            f"{LABELS[i]}. {description}" for i, (_, description) in enumerate(options)
        )
        # Qwen2.5 inserts a default system message for standalone user turns. Render
        # a complete conversation and remove its closed prefix, rather than injecting
        # a second system message between the cached state and the decision question.
        base = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": ""}]
        prefix = self.render(base, False)
        full = self.render(
            base + [{"role": "user", "content": f"Question: {question}\nAlternatives:\n{choices}"}],
            True,
        )
        if not full.startswith(prefix):
            raise ValueError("unsupported chat template: adding a question rewrites the prefix")
        text = full[len(prefix) :]
        if not text.startswith("<|im_start|>user\n"):
            raise ValueError("unsupported chat template: suffix must start at a user-message token")
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        label_ids = []
        for letter in LABELS[: len(options)]:
            slot = self.tokenizer.encode(letter, add_special_tokens=False)
            if len(slot) != 1 or self.tokenizer.decode(slot) != letter:
                raise ValueError(f"label {letter} is not a round-trip single token")
            if self.tokenizer.encode(text + letter, add_special_tokens=False) != ids + slot:
                raise ValueError(f"label {letter} changes the answer token boundary")
            label_ids.append(slot[0])
        if len(set(label_ids)) != len(label_ids):
            raise ValueError("label token IDs collide")
        return CompiledQuestion(tuple(ids), tuple(label_ids))

    def clear(self):
        """Release cached question text when clearing all runtime state."""
        self._suffix.cache_clear()


def read_decision(question: Question, logits: list[float]) -> Decision:
    """Apply stable softmax; Score is an expectation, Boolean uses a 0.5 threshold."""
    options = alternatives(question)
    if len(logits) != len(options) or not all(math.isfinite(z) for z in logits):
        raise ValueError("backend returned invalid candidate logits")
    maximum = max(logits)
    weights = [math.exp((z - maximum) / question.temperature) for z in logits]
    total = sum(weights)
    probabilities = [x / total for x in weights]
    best = max(range(len(logits)), key=logits.__getitem__)
    selected = options[best][0]
    value: str | bool | float = selected
    if isinstance(question, BooleanQuestion):
        value = probabilities[0] >= 0.5
    elif isinstance(question, ScoreQuestion):
        value = sum(
            p * level.value for p, level in zip(probabilities, question.levels, strict=True)
        )
    entropy = -sum(p * math.log(p) for p in probabilities if p > 0) / math.log(len(options))
    return Decision(
        type=question.type,
        option_ids=[o[0] for o in options],
        logits=logits,
        probabilities=probabilities,
        selected=selected,
        value=value,
        normalized_entropy=max(0.0, min(1.0, entropy)),
    )
