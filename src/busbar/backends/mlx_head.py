"""Trainable, candidate-conditioned readout over an unchanged native MLX backbone."""

import io
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from ..compiler import CandidateCompiler
from ..head_artifact import HeadArtifact, check_compatibility, read_artifact
from .mlx import MLXBackend


class CandidateHead(nn.Module):
    """A shared scalar compatibility function; its size is independent of the option count."""

    def __init__(self, hidden_size: int):
        """Keep head arithmetic in FP32; a common scalar bias would cancel in softmax."""
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.scalar = nn.Linear(hidden_size, 1, bias=False)

    def __call__(self, hidden):
        """Score either [candidates, hidden] inference inputs or [batch, candidates, hidden]."""
        return self.scalar(self.norm(hidden.astype(mx.float32))).squeeze(-1)


def load_head(artifact: HeadArtifact):
    """Validate every tensor's name, shape, precision and finiteness before model use."""
    manifest = artifact.manifest
    try:
        values = mx.load(io.BytesIO(artifact.weights), format="safetensors")
    except (ValueError, RuntimeError) as error:
        raise ValueError(f"cannot read selection head tensors: {error}") from error
    size = manifest.hidden_size
    expected = {"norm.weight": (size,), "norm.bias": (size,), "scalar.weight": (1, size)}
    if set(values) != set(expected) or any(
        tuple(values[key].shape) != shape
        or values[key].dtype != mx.float32
        or not bool(mx.all(mx.isfinite(values[key])).item())
        for key, shape in expected.items()
    ):
        raise ValueError(
            "invalid selection head tensors: names, shapes, FP32 or finite check failed"
        )
    head = CandidateHead(size)
    head.load_weights(list(values.items()), strict=True)
    head.eval()
    return head


class MLXHeadBackend(MLXBackend):
    """Execute candidate branches through the backbone and only the learned selection head."""

    default_projection = "head"
    projections = ("head",)

    def __init__(self, directory: Path, *, model=None, revision=None, dtype=None, **kwargs):
        """Validate artifact compatibility before loading the expensive frozen base model."""
        artifact = read_artifact(directory)
        manifest = artifact.manifest
        check_compatibility(manifest, model=model, revision=revision, dtype=dtype)
        self.decision_head = load_head(artifact)
        self.training_splits = artifact.training_splits
        super().__init__(manifest.model, manifest.revision, dtype=manifest.dtype, **kwargs)
        if self.model.model.embed_tokens.weight.shape[1] != manifest.hidden_size:
            raise ValueError("head hidden size does not match backbone")
        self.model.freeze()
        self.compiler = CandidateCompiler(self.tokenizer)
        self.identity.update(
            readout="candidate_head",
            head_id=manifest.identity,
            head_architecture=manifest.architecture,
            encoder_version=manifest.encoder_version,
        )

    def _project(self, hidden, labels, projection):
        """Read one scalar per candidate; the vocabulary matrix is never multiplied."""
        if any(tuple(slots) != (0,) for slots in labels):
            raise ValueError("candidate head requires one scalar readout per path")
        scores = self.decision_head(hidden)
        return [scores[index : index + 1] for index in range(len(labels))]

    def score(self, sequences, labels, prefix, projection):
        """Expose learned logit semantics while retaining native cache and token accounting."""
        output = super().score(sequences, labels, prefix, projection)
        output.details.update(
            logit_space="trained_candidate_logits",
            probability_status="learned candidate probabilities; uncalibrated",
            candidate_paths=len(sequences),
            vocabulary_projection=False,
        )
        return output
