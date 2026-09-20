"""Portable metadata and integrity checks for immutable, independently versioned selection heads."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field

from .compiler import CandidateCompiler, canonical_json
from .schemas import Contract
from .storage import FileSnapshot, file_sha256
from .storage import write_json as write_json

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


def _parse_training_splits(payload: bytes) -> dict:
    """Validate the already-verified report bytes without reopening a mutable file path."""
    report = json.loads(payload)
    splits = report.get("datasets") if isinstance(report, dict) else None
    if not isinstance(splits, dict) or not {"train", "validation"} <= set(splits):
        raise ValueError("head training report requires train and validation provenance")
    return {
        key: DatasetProvenance.model_validate(value).model_dump() for key, value in splits.items()
    }


@dataclass(frozen=True)
class HeadArtifact:
    """A coherent in-memory version: decoding cannot race with on-disk replacement."""

    manifest: HeadManifest
    weights: bytes
    training_splits: dict


def read_artifact(directory: Path) -> HeadArtifact:
    """Verify the exact bytes later consumed by the tensor loader and leakage checks."""
    manifest, files = _verified_files(directory)
    return HeadArtifact(
        manifest, files["head.safetensors"], _parse_training_splits(files["training.json"])
    )


def read_training_splits(directory: Path) -> dict:
    """Inspect provenance only after verifying the complete artifact's integrity."""
    return read_artifact(directory).training_splits


def overlaps_provenance(rows, split):
    """Match row/group identity and exact semantic inputs against previously seen data."""
    return bool(
        set(split["ids"]) & {row.id for row in rows}
        or set(split["groups"]) & {row.group_id for row in rows}
        or set(split["fingerprints"]) & {row.fingerprint() for row in rows}
    )


def _verified_files(directory: Path) -> tuple[HeadManifest, dict[str, bytes]]:
    """Fixed filenames and digest checks bind each captured file to one manifest version."""
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"incomplete head artifact: {manifest_path} is missing")
    manifest = HeadManifest.model_validate_json(manifest_path.read_bytes())
    files = {}
    for filename, expected in (
        ("head.safetensors", manifest.weights_sha256),
        ("training.json", manifest.training_sha256),
    ):
        path = directory / filename
        if not path.is_file():
            raise ValueError(f"head artifact integrity failure: {filename}")
        snapshot = FileSnapshot.read(path)
        if snapshot.sha256 != expected:
            raise ValueError(f"head artifact integrity failure: {filename}")
        files[filename] = snapshot.payload
    return manifest, files


def read_manifest(directory: Path) -> HeadManifest:
    """Inspect metadata after checking all file hashes; does not import accelerator code."""
    return _verified_files(directory)[0]


def check_compatibility(manifest: HeadManifest, *, model=None, revision=None, dtype=None):
    """An explicit caller override must agree with the encoder used during training."""
    for name, value in (("model", model), ("revision", revision), ("dtype", dtype)):
        if value is not None and value != getattr(manifest, name):
            raise ValueError(
                f"head {name} mismatch: expected {getattr(manifest, name)}, got {value}"
            )


def publish_head(
    directory: Path, head, identity: dict, hidden_size: int, report: dict
) -> HeadManifest:
    """Publish the manifest last to mark an exclusively owned version as ready."""
    head.save_weights(str(directory / "head.safetensors"))
    write_json(directory / "training.json", report)
    manifest = HeadManifest(
        model=identity["model"],
        revision=identity["revision"],
        dtype=identity["dtype"],
        hidden_size=hidden_size,
        weights_sha256=file_sha256(directory / "head.safetensors"),
        training_sha256=file_sha256(directory / "training.json"),
    )
    write_json(directory / "manifest.json", manifest.model_dump())
    return manifest
