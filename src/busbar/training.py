"""Offline, head-only training: immutable backbone features and validation-selected artifacts."""

import time
from importlib.metadata import version
from pathlib import Path

from pydantic import Field

from .compiler import CandidateCompiler
from .datasets import assert_disjoint, dataset_provenance, parse_dataset
from .head_artifact import (
    check_compatibility,
    overlaps_provenance,
    publish_head,
    read_artifact,
)
from .mlx_training import extract_features as extract_features
from .mlx_training import fit_head as fit_head
from .schemas import Contract
from .storage import FileSnapshot


class TrainingConfig(Contract):
    """Bound optimizer work and resident feature storage before allocating accelerator arrays."""

    epochs: int = Field(default=30, ge=1, le=10000)
    learning_rate: float = Field(default=0.003, gt=0, le=1)
    train_batch_size: int = Field(default=32, ge=1, le=4096)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    feature_cache_mib: int = Field(default=256, ge=1)


def _ancestor_splits(artifact, validation) -> dict:
    """Preserve inherited leakage barriers when continuing from an earlier head version."""
    if artifact is None:
        return {}
    ancestors = {}
    for name, split in artifact.training_splits.items():
        if name.split(".")[-1] == "train" and overlaps_provenance(validation, split):
            raise ValueError("validation overlaps training data from the initial head")
        key = (
            name
            if name.startswith("ancestor.")
            else f"ancestor.{artifact.manifest.identity}.{name}"
        )
        ancestors[key] = split
    return ancestors


def _fit_features(backend, train, validation, head, config):
    """Extract once, then optimize only the small head on the explicit train/validation slices."""
    import mlx.core as mx

    from .backends.mlx_head import CandidateHead

    features = extract_features(backend, train + validation, config.feature_cache_mib)
    training = tuple(value[: len(train)] for value in features)
    held_out = tuple(value[len(train) :] for value in features)
    mx.random.seed(config.seed)
    head = head if head is not None else CandidateHead(features[0].shape[-1])
    return head, fit_head(head, training, held_out, config)


def train_head(
    train_source: Path,
    validation_source: Path,
    output: Path,
    *,
    model=None,
    revision=None,
    dtype=None,
    init_head: Path | None = None,
    config: TrainingConfig | None = None,
    **backend_options,
):
    """Train a new immutable artifact; manifest.json is written last as its readiness marker."""
    from .backends import load_backend
    from .backends.mlx_head import load_head

    config = config or TrainingConfig()
    train_input, validation_input = (
        FileSnapshot.read(train_source),
        FileSnapshot.read(validation_source),
    )
    train, validation = parse_dataset(train_input), parse_dataset(validation_input)
    assert_disjoint(train, validation)
    if output.exists():
        raise ValueError("head output already exists; choose a new version directory")
    artifact = read_artifact(init_head) if init_head else None
    previous = artifact.manifest if artifact else None
    ancestors = _ancestor_splits(artifact, validation)
    if previous:
        check_compatibility(previous, model=model, revision=revision, dtype=dtype)
        model, revision, dtype = previous.model, previous.revision, previous.dtype
    head = load_head(artifact) if artifact else None
    backend = load_backend("mlx", model, revision, dtype=dtype or "float32", **backend_options)
    hidden_size = backend.model.model.embed_tokens.weight.shape[1]
    if previous and previous.hidden_size != hidden_size:
        raise ValueError("initial head hidden size does not match backbone")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    head, fitted = _fit_features(backend, train, validation, head, config)
    if previous is None and fitted["best_epoch"] == 0:
        raise ValueError("no validation improvement over random initialization; no head published")
    train_input.assert_unchanged()
    validation_input.assert_unchanged()
    report = {
        "schema_version": 1,
        "method": "frozen-backbone-head-only",
        "base": backend.identity,
        "config": config.model_dump(),
        "init_head_id": previous.identity if previous else None,
        "optimizer_resumed": False,
        "encoder_version": CandidateCompiler.version,
        "datasets": {
            **ancestors,
            "train": dataset_provenance(train_input, train),
            "validation": dataset_provenance(validation_input, validation),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "versions": {name: version(name) for name in ("mlx", "mlx-lm", "transformers")},
        **fitted,
    }
    manifest = publish_head(output, head, backend.identity, hidden_size, report)
    return {
        "artifact": str(output.resolve()),
        "head_id": manifest.identity,
        **fitted,
        "elapsed_seconds": report["elapsed_seconds"],
    }
