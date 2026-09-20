"""Typed decision datasets shared by head training and held-out evaluation."""

import hashlib
import json
from pathlib import Path

from pydantic import StrictBool, StrictFloat, StrictInt, StrictStr, model_validator

from .compiler import alternatives, canonical_json
from .schemas import ContextSpec, Contract, Name, Question


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


def read_dataset(source: Path) -> list[LabeledDecision]:
    """Parse a nonempty JSONL file with line-numbered failures and unique example IDs."""
    rows, ids = [], set()
    for line_number, line in enumerate(source.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = LabeledDecision.model_validate(json.loads(line))
            if row.id in ids:
                raise ValueError(f"duplicate ID {row.id}")
        except ValueError as error:
            raise ValueError(f"{source.name}:{line_number}: {error}") from error
        rows.append(row)
        ids.add(row.id)
    if not rows:
        raise ValueError(f"{source.name}: dataset is empty")
    return rows


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


def file_sha256(path: Path) -> str:
    """Hash artifact and dataset bytes without loading large files into memory."""
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()
