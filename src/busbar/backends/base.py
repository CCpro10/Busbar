"""The backend boundary owns physical KV state; the runtime owns semantic snapshot lifetime."""

from dataclasses import dataclass, field
from typing import Any, Protocol

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
MODEL_REVISIONS = {
    DEFAULT_MODEL: DEFAULT_REVISION,
    "Qwen/Qwen2.5-3B-Instruct": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
    "Qwen/Qwen3.5-4B": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    "openbmb/MiniCPM5-2B": "12a3808a956f869c767195e9266b59c4d21d92e2",
}


@dataclass(frozen=True)
class PrefixState:
    """Opaque immutable-by-contract prefix state with accounted retained bytes."""

    cache: Any
    nbytes: int


@dataclass(frozen=True)
class BackendResult:
    """Logits stay in caller order even when the backend groups questions by length."""

    logits: list[list[float]]
    computed_tokens: int
    batches: int
    # Engine-managed APC can reuse question tokens as well as context tokens.
    # None preserves exact native-KV accounting for the MLX/HF backends.
    reused_tokens: int | None = None
    details: dict = field(default_factory=dict)


class Backend(Protocol):
    """A real implementation must branch without mutating the supplied prefix."""

    tokenizer: Any
    identity: dict
    max_tokens: int

    def prefill(self, token_ids: tuple[int, ...]) -> PrefixState:
        """Prepare one prefix; the backend documents native KV versus engine-managed APC."""
        ...

    def score(
        self,
        sequences: list[tuple[int, ...]],
        labels: list[tuple[int, ...]],
        prefix: PrefixState | None,
        projection: str,
    ) -> BackendResult:
        """Compute suffixes from independent branches, or full sequences with no prefix."""
        ...


def length_batches(sequences, max_batch: int):
    """Group equal lengths to avoid padding, position ambiguity and unused readout positions."""
    groups: dict[int, list[int]] = {}
    for index, sequence in enumerate(sequences):
        groups.setdefault(len(sequence), []).append(index)
    for indices in groups.values():
        for start in range(0, len(indices), max_batch):
            yield indices[start : start + max_batch]
