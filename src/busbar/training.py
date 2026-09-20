"""Offline, head-only training: immutable backbone features and validation-selected artifacts."""

import math
import random
import sys
import time
from importlib.metadata import version
from pathlib import Path

from pydantic import Field

from .compiler import CandidateCompiler
from .datasets import assert_disjoint, file_sha256, read_dataset
from .head_artifact import (
    HeadManifest,
    check_compatibility,
    overlaps_provenance,
    read_manifest,
    read_training_splits,
    write_json,
)
from .schemas import Contract


class TrainingConfig(Contract):
    """Bound optimizer work and resident feature storage before allocating accelerator arrays."""

    epochs: int = Field(default=30, ge=1, le=10000)
    learning_rate: float = Field(default=0.003, gt=0, le=1)
    train_batch_size: int = Field(default=32, ge=1, le=4096)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    feature_cache_mib: int = Field(default=256, ge=1)


def extract_features(backend, rows, feature_cache_mib):
    """Preflight every path, then encode with a frozen body and disposable KV branches."""
    import mlx.core as mx

    compiler = CandidateCompiler(backend.tokenizer)
    size = backend.model.model.embed_tokens.weight.shape[1]
    compiled = [(compiler.context(row.context), compiler.question(row.question)) for row in rows]
    max_candidates = max(len(question.paths) for _, question in compiled)
    # Include both the per-row arrays and their stacked copy; model/KV memory is separate.
    required = 2 * len(rows) * max_candidates * size * 4
    if required > feature_cache_mib * 1024 * 1024:
        raise ValueError(f"feature cache needs at least {math.ceil(required / 2**20)} MiB")
    if any(
        len(prefix) + len(path.token_ids) > backend.max_tokens
        for prefix, question in compiled
        for path in question.paths
    ):
        raise ValueError("training input exceeds token limit; no examples were truncated")
    backend.model.freeze()
    features, masks = [None] * len(rows), [None] * len(rows)
    groups = {}
    for index, (prefix, _) in enumerate(compiled):
        groups.setdefault(prefix, []).append(index)
    completed = 0
    for prefix_tokens, indices in groups.items():
        prefix = backend.prefill(prefix_tokens)
        for index in indices:
            paths = compiled[index][1].paths
            vectors = [None] * len(paths)
            for positions, hidden in backend.hidden_batches([p.token_ids for p in paths], prefix):
                hidden = mx.stop_gradient(hidden.astype(mx.float32))
                mx.eval(hidden)
                for position, vector in zip(positions, hidden, strict=True):
                    vectors[position] = vector
            features[index] = mx.stack(
                vectors + [mx.zeros((size,))] * (max_candidates - len(paths))
            )
            masks[index] = [i < len(paths) for i in range(max_candidates)]
            mx.eval(features[index])
            completed += 1
            if completed % 24 == 0 or completed == len(rows):
                print(f"features {completed}/{len(rows)}", file=sys.stderr, flush=True)
        del prefix
    result = (
        mx.stack(features),
        mx.array(masks),
        mx.array([row.gold_index for row in rows]),
    )
    mx.eval(result)
    return result


def _logits(head, features, mask):
    """Exclude padding from both the training objective and quality measurements."""
    import mlx.core as mx

    return mx.where(mask, head(features), -1e9)


def feature_metrics(head, data):
    """Measure categorical accuracy and NLL without fitting calibration on validation examples."""
    import mlx.core as mx
    import mlx.nn as nn

    features, mask, labels = data
    logits = _logits(head, features, mask)
    nll = float(nn.losses.cross_entropy(logits, labels, reduction="mean").item())
    accuracy = float(mx.mean(mx.argmax(logits, axis=-1) == labels).item())
    if not math.isfinite(nll):
        raise ValueError("nonfinite training metrics; artifact was not published")
    return {"rows": len(labels), "accuracy": accuracy, "nll": nll}


def fit_head(head, train, validation, config: TrainingConfig):
    """Optimize only the head; choose a checkpoint by validation NLL, including epoch zero."""
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten

    def loss(module, features, mask, labels):
        """One mutually exclusive categorical target per variable-length candidate set."""
        return nn.losses.cross_entropy(_logits(module, features, mask), labels, reduction="mean")

    gradient = nn.value_and_grad(head, loss)
    optimizer = optim.Adam(learning_rate=config.learning_rate)
    shuffle = random.Random(config.seed)
    initial = dict(tree_flatten(head.parameters()))
    history = [
        {
            "epoch": 0,
            "train": feature_metrics(head, train),
            "validation": feature_metrics(head, validation),
        }
    ]
    best, best_epoch, best_loss = initial, 0, history[0]["validation"]["nll"]
    for epoch in range(1, config.epochs + 1):
        order = list(range(len(train[2])))
        shuffle.shuffle(order)
        for start in range(0, len(order), config.train_batch_size):
            indices = mx.array(order[start : start + config.train_batch_size])
            value, grads = gradient(head, *(item[indices] for item in train))
            optimizer.update(head, grads)
            mx.eval(head.parameters(), optimizer.state, value)
            if not math.isfinite(float(value.item())):
                raise ValueError("nonfinite training loss; artifact was not published")
        record = {
            "epoch": epoch,
            "train": feature_metrics(head, train),
            "validation": feature_metrics(head, validation),
        }
        history.append(record)
        if record["validation"]["nll"] < best_loss:
            best = dict(tree_flatten(head.parameters()))
            best_epoch, best_loss = epoch, record["validation"]["nll"]
        if epoch == 1 or epoch % 10 == 0 or epoch == config.epochs:
            print(
                f"epoch {epoch}: train NLL={record['train']['nll']:.4f}, "
                f"validation NLL={record['validation']['nll']:.4f}",
                file=sys.stderr,
                flush=True,
            )
    head.load_weights(list(best.items()), strict=True)
    head.eval()
    delta = sum(float(mx.sum((best[key] - value) ** 2).item()) for key, value in initial.items())
    return {
        "history": history,
        "best_epoch": best_epoch,
        "parameter_delta_l2": math.sqrt(delta),
        "trainable_parameters": sum(value.size for value in best.values()),
        "selected_train": feature_metrics(head, train),
        "selected_validation": feature_metrics(head, validation),
    }


def _provenance(source, rows):
    """Record hashes and split membership so evaluation can reject accidental training reuse."""
    return {
        "file": source.name,
        "sha256": file_sha256(source),
        "rows": len(rows),
        "ids": sorted(row.id for row in rows),
        "groups": sorted({row.group_id for row in rows}),
        "fingerprints": sorted({row.fingerprint() for row in rows}),
    }


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
    import mlx.core as mx

    from .backends import load_backend
    from .backends.mlx_head import CandidateHead, load_head

    config = config or TrainingConfig()
    input_hashes = {path: file_sha256(path) for path in (train_source, validation_source)}
    train, validation = read_dataset(train_source), read_dataset(validation_source)
    assert_disjoint(train, validation)
    if output.exists():
        raise ValueError("head output already exists; choose a new version directory")
    previous = read_manifest(init_head) if init_head else None
    ancestors = {}
    if previous:
        check_compatibility(previous, model=model, revision=revision, dtype=dtype)
        model, revision, dtype = previous.model, previous.revision, previous.dtype
        for name, split in read_training_splits(init_head).items():
            if name.split(".")[-1] == "train" and overlaps_provenance(validation, split):
                raise ValueError("validation overlaps training data from the initial head")
            key = name if name.startswith("ancestor.") else f"ancestor.{previous.identity}.{name}"
            ancestors[key] = split
    backend = load_backend("mlx", model, revision, dtype=dtype or "float32", **backend_options)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    features = extract_features(backend, train + validation, config.feature_cache_mib)
    train_features = tuple(value[: len(train)] for value in features)
    validation_features = tuple(value[len(train) :] for value in features)
    mx.random.seed(config.seed)
    hidden_size = features[0].shape[-1]
    if previous and previous.hidden_size != hidden_size:
        raise ValueError("initial head hidden size does not match backbone")
    head = load_head(init_head, previous) if previous else CandidateHead(hidden_size)
    fitted = fit_head(head, train_features, validation_features, config)
    if previous is None and fitted["best_epoch"] == 0:
        raise ValueError("no validation improvement over random initialization; no head published")
    if any(file_sha256(path) != expected for path, expected in input_hashes.items()):
        raise ValueError("dataset changed during training; artifact was not published")
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
            "train": _provenance(train_source, train),
            "validation": _provenance(validation_source, validation),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "versions": {name: version(name) for name in ("mlx", "mlx-lm", "transformers")},
        **fitted,
    }
    # Exclusive directory ownership plus a last-written manifest keeps failed runs unloadable.
    head.save_weights(str(output / "head.safetensors"))
    write_json(output / "training.json", report)
    manifest = HeadManifest(
        model=backend.identity["model"],
        revision=backend.identity["revision"],
        dtype=backend.identity["dtype"],
        hidden_size=hidden_size,
        weights_sha256=file_sha256(output / "head.safetensors"),
        training_sha256=file_sha256(output / "training.json"),
    )
    write_json(output / "manifest.json", manifest.model_dump())
    return {
        "artifact": str(output.resolve()),
        "head_id": manifest.identity,
        **fitted,
        "elapsed_seconds": report["elapsed_seconds"],
    }
