"""Typed decision datasets shared by head training and held-out evaluation."""

import hashlib
import json
from pathlib import Path

from pydantic import (
    ConfigDict,
    Field,
    JsonValue,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    model_validator,
)

from .compiler import alternatives, canonical_json
from .schemas import (
    MAX_ALTERNATIVES,
    ChoiceQuestion,
    ContextSpec,
    Contract,
    Name,
    Option,
    Question,
    Text,
)
from .storage import FileSnapshot
from .storage import file_sha256 as file_sha256


class LabeledDecision(Contract):
    """One supervised decision; related examples must share a group_id across all splits."""

    id: Name
    group_id: Name
    context: ContextSpec
    question: Question
    label: StrictStr | StrictBool | StrictInt | StrictFloat

    @property
    def gold_index(self) -> int:
        """Resolve application labels, never option positions, into the categorical target."""
        if self.question.type == "boolean":
            if not isinstance(self.label, bool):
                raise ValueError("boolean label must be a JSON boolean")
            return 0 if self.label else 1
        if self.question.type == "choice":
            ids = [item.id for item in self.question.options]
            if not isinstance(self.label, str) or self.label not in ids:
                raise ValueError("choice label must be an option ID")
            return ids.index(self.label)
        values = [level.value for level in self.question.levels]
        if isinstance(self.label, (bool, str)) or self.label not in values:
            raise ValueError("score label must equal a declared numeric level")
        return values.index(self.label)

    @model_validator(mode="after")
    def validate_target(self):
        """Reject duplicate candidates and ambiguous supervision before loading a model."""
        _ = self.gold_index
        descriptions = [text.strip() for _, text in alternatives(self.question)]
        if not all(descriptions) or len(set(descriptions)) != len(descriptions):
            raise ValueError("training candidates must have distinct, nonblank descriptions")
        if self.question.temperature != 1:
            raise ValueError("labeled datasets require temperature=1; calibration is separate")
        return self

    def fingerprint(self) -> str:
        """Detect exact semantic duplicates despite namespace, ID or candidate-order changes."""
        question = self.question.model_dump(exclude={"temperature"})
        if self.question.type == "choice":
            question["options"] = sorted(option.description for option in self.question.options)
        context = self.context.model_dump(exclude={"namespace"})
        return hashlib.sha256(canonical_json([context, question]).encode()).hexdigest()


class SemIfExample(Contract):
    """Validate the external categorical format before executing any evaluation request."""

    # Upstream datasets may attach metadata beyond the fields consumed by Busbar.
    model_config = ConfigDict(extra="ignore")
    id: Name
    group_id: Text
    family: Text
    state: JsonValue
    question: Text
    options: tuple[Option, ...] = Field(min_length=2, max_length=MAX_ALTERNATIVES)
    label: StrictInt = Field(ge=0)
    provenance: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_decision(self):
        """Catch malformed options, labels and oversized states before even warming a model."""
        self.as_question()
        ContextSpec(state=self.state)
        json.dumps(self.provenance, allow_nan=False)
        if self.label >= len(self.options):
            raise ValueError("gold label is outside the candidate set")
        return self

    def as_question(self) -> ChoiceQuestion:
        """Normalize the external format to the existing public decision contract."""
        return ChoiceQuestion(type="choice", question=self.question, options=self.options)


def parse_dataset(source: FileSnapshot, row_type=LabeledDecision) -> list:
    """Parse captured JSONL with source line numbers and unique IDs for both supported formats."""
    rows, ids = [], set()
    for line_number, line in enumerate(source.payload.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = row_type.model_validate(json.loads(line))
            if row.id in ids:
                raise ValueError(f"duplicate ID {row.id}; dataset requires unique IDs")
        except ValueError as error:
            raise ValueError(f"{source.path.name}:{line_number}: {error}") from error
        rows.append(row)
        ids.add(row.id)
    if not rows:
        raise ValueError(f"{source.path.name}: dataset is empty")
    return rows


def read_dataset(source: Path) -> list[LabeledDecision]:
    """Convenience API for callers that only need validated native records."""
    return parse_dataset(FileSnapshot.read(source))


def dataset_provenance(source: FileSnapshot, rows) -> dict:
    """Tie split membership to the exact bytes that produced the training records."""
    return {
        "file": source.path.name,
        "sha256": source.sha256,
        "rows": len(rows),
        "ids": sorted(row.id for row in rows),
        "groups": sorted({row.group_id for row in rows}),
        "fingerprints": sorted({row.fingerprint() for row in rows}),
    }


def assert_disjoint(left, right):
    """Reject row, group and exact-input leakage; authors still own semantic split quality."""
    checks = (
        ("IDs", lambda row: row.id),
        ("group IDs", lambda row: row.group_id),
        ("semantic inputs", lambda row: row.fingerprint()),
    )
    for name, key in checks:
        if {key(row) for row in left} & {key(row) for row in right}:
            raise ValueError(f"dataset splits overlap in {name}")
