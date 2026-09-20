"""Portable metadata and integrity checks for immutable, independently versioned selection heads."""

import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field

from .compiler import CandidateCompiler, canonical_json
from .datasets import file_sha256
from .schemas import Contract

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Revision = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]


class HeadManifest(Contract):
    """Bind a head to its exact frozen feature encoder, weight bytes and training report."""

    schema_version: Literal[1] = 1
    architecture: Literal["layernorm-linear-v1"] = "layernorm-linear-v1"
    encoder_version: Literal["busbar-candidate-v1"] = CandidateCompiler.version
    model: str = Field(min_length=1)
    revision: Revision
    dtype: Literal["float16", "bfloat16", "float32"]
    hidden_size: int = Field(ge=1, le=65536)
    weights_sha256: Sha256
    training_sha256: Sha256

    @property
    def identity(self) -> str:
        """Invalidate snapshots when either learned parameters or encoder metadata changes."""
        return hashlib.sha256(canonical_json(self.model_dump()).encode()).hexdigest()


class DatasetProvenance(Contract):
    """Retain all fitting/selection data membership, including inherited warm-start datasets."""

    file: str
    sha256: Sha256
    rows: int = Field(ge=1)
    ids: list[str] = Field(min_length=1)
    groups: list[str] = Field(min_length=1)
    fingerprints: list[Sha256] = Field(min_length=1)


def read_training_splits(directory: Path) -> dict:
    """Validate provenance without importing MLX; corrupted reports fail with a clear error."""
    report = json.loads((directory / "training.json").read_bytes())
    splits = report.get("datasets") if isinstance(report, dict) else None
    if not isinstance(splits, dict) or not {"train", "validation"} <= set(splits):
        raise ValueError("head training report requires train and validation provenance")
    return {
        key: DatasetProvenance.model_validate(value).model_dump() for key, value in splits.items()
    }


def overlaps_provenance(rows, split):
    """Match row/group identity and exact semantic inputs against previously seen data."""
    return bool(
        set(split["ids"]) & {row.id for row in rows}
        or set(split["groups"]) & {row.group_id for row in rows}
        or set(split["fingerprints"]) & {row.fingerprint() for row in rows}
    )


def read_manifest(directory: Path) -> HeadManifest:
    """Accept only a complete artifact; fixed file names prevent manifest path traversal."""
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"incomplete head artifact: {manifest_path} is missing")
    manifest = HeadManifest.model_validate_json(manifest_path.read_bytes())
    for filename, expected in (
        ("head.safetensors", manifest.weights_sha256),
        ("training.json", manifest.training_sha256),
    ):
        path = directory / filename
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"head artifact integrity failure: {filename}")
    return manifest


def check_compatibility(manifest: HeadManifest, *, model=None, revision=None, dtype=None):
    """An explicit caller override must agree with the encoder used during training."""
    for name, value in (("model", model), ("revision", revision), ("dtype", dtype)):
        if value is not None and value != getattr(manifest, name):
            raise ValueError(
                f"head {name} mismatch: expected {getattr(manifest, name)}, got {value}"
            )


def write_json(path: Path, value):
    """Create artifact files exclusively; existing versions are never silently replaced."""
    with path.open("x") as file:
        json.dump(value, file, indent=2, ensure_ascii=False, allow_nan=False)
        file.write("\n")
