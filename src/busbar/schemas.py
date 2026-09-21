"""Validated public contracts; probabilities are conditional, never calibrated confidence."""

import json
import math
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, JsonValue, model_validator


def nonblank_text(value: str) -> str:
    """Questions and candidate descriptions must contain meaning, without stripping their text."""
    if not value.strip():
        raise ValueError("text must not be blank")
    return value


Name = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")]
Text = Annotated[str, Field(min_length=1, max_length=8192), AfterValidator(nonblank_text)]

# The public ceiling on alternatives per question. Vocabulary readout needs one single-token
# label per alternative, so the compiler's label alphabet must cover this count; `compiler`
# asserts that on import instead of repeating the number. Raising it requires a label scheme
# measured at that width, not just a longer alphabet: accuracy is a property of how the model
# reads the labels, which tokenization alone does not establish.
MAX_ALTERNATIVES = 26


class Contract(BaseModel):
    """Reject misspelled fields and keep public records immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class ContextSpec(Contract):
    """A namespace-scoped context; changing any content creates a new snapshot."""

    namespace: Name = "default"
    state: JsonValue
    instructions: str = Field(default="", max_length=8192)

    @model_validator(mode="after")
    def validate_size(self):
        """Reject oversized or nonfinite JSON before tokenization or model execution."""
        payload = json.dumps(self.model_dump(), ensure_ascii=False, allow_nan=False)
        if len(payload.encode("utf-8")) > 256_000:
            raise ValueError("context JSON exceeds 256000 UTF-8 bytes")
        return self


class Option(Contract):
    """An application-owned option identifier and its semantic description."""

    id: Name
    description: Text


class BaseQuestion(Contract):
    """Shared question text and optional, explicit temperature scaling."""

    question: Text
    temperature: float = Field(default=1.0, ge=0.01, le=100.0)


class BooleanQuestion(BaseQuestion):
    """Return the conditional probability that a proposition is true."""

    type: Literal["boolean"]


class ChoiceQuestion(BaseQuestion):
    """Choose exactly one of 2–26 mutually exclusive alternatives."""

    type: Literal["choice"]
    options: tuple[Option, ...] = Field(min_length=2, max_length=MAX_ALTERNATIVES)

    @model_validator(mode="after")
    def unique_ids(self):
        """Duplicate option IDs would lose probability mass in the response map."""
        if len({o.id for o in self.options}) != len(self.options):
            raise ValueError("option IDs must be unique")
        return self


class ScoreLevel(Contract):
    """A numeric scale point with explicit semantic meaning."""

    value: float
    description: Text


class ScoreQuestion(BaseQuestion):
    """Return the expectation of an explicitly described ordinal scale."""

    type: Literal["score"]
    levels: tuple[ScoreLevel, ...] = Field(min_length=2, max_length=MAX_ALTERNATIVES)

    @model_validator(mode="after")
    def increasing_levels(self):
        """Require a finite, strictly increasing scale to avoid ambiguous aggregation."""
        values = [x.value for x in self.levels]
        if any(not math.isfinite(x) for x in values) or any(
            a >= b for a, b in zip(values, values[1:], strict=False)
        ):
            raise ValueError("score values must be finite and strictly increasing")
        return self


Question = Annotated[BooleanQuestion | ChoiceQuestion | ScoreQuestion, Field(discriminator="type")]
Questions = Annotated[dict[Name, Question], Field(min_length=1, max_length=64)]


class DecisionInput(Contract):
    """Validate a complete one-shot input before loading or prefilling any model."""

    context: ContextSpec
    questions: Questions


class DecisionRequest(Contract):
    """Fan out questions from a previously compiled snapshot."""

    snapshot_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    namespace: Name = "default"
    questions: Questions
    mode: Literal["cached", "fresh"] = "cached"
    projection: Literal["auto", "selected", "full", "head"] = "auto"


class Snapshot(Contract):
    """Public cache metadata; source state and KV arrays are deliberately omitted."""

    id: str
    namespace: str
    token_count: int
    cache_bytes: int
    model: str
    revision: str
    created_at: float
    expires_at: float


class CompileResult(Contract):
    """Compilation timing distinguishes token-cache/KV hits from new prefill work."""

    snapshot: Snapshot
    reused: bool
    prefill_tokens: int
    elapsed_ms: float


class Decision(Contract):
    """Native label logits, normalized probabilities and type-specific derived value."""

    type: str
    option_ids: list[str]
    logits: list[float]
    probabilities: list[float]
    selected: str
    value: str | bool | float
    normalized_entropy: float
    logit_space: str = "raw_logits"


class DecisionResult(Contract):
    """Actual submitted-token counters and wall timings make cache behavior observable."""

    snapshot_id: str
    decisions: dict[str, Decision]
    mode: str
    projection: str
    prefix_tokens: int
    reused_prefix_tokens: int
    computed_tokens: int
    batches: int
    compile_ms: float
    inference_ms: float
    total_ms: float
    backend_details: dict = Field(default_factory=dict)
    probability_status: str = "conditional label probabilities; uncalibrated"
