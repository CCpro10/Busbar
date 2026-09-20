"""Load optional accelerator dependencies only when their backend is selected."""

from .base import DEFAULT_MODEL, MODEL_REVISIONS
from .base import DEFAULT_REVISION as DEFAULT_REVISION


def load_backend(name="mlx", model=None, revision=None, *, head=None, **kwargs):
    """Construct a supported backend without importing MLX on Linux or Torch on Mac by default."""
    if head is not None:
        if name != "mlx":
            raise ValueError("trained selection heads currently require the native MLX backend")
        from pathlib import Path

        from .mlx_head import MLXHeadBackend

        return MLXHeadBackend(Path(head), model=model, revision=revision, **kwargs)
    model = model or DEFAULT_MODEL
    revision = revision or MODEL_REVISIONS.get(model)
    if revision is None:
        raise ValueError("provide --revision with an immutable checkpoint SHA for this model")
    if name == "vllm-metal":
        from .vllm_metal import VLLMMetalBackend

        return VLLMMetalBackend(model, revision, **kwargs)
    if name == "mlx":
        from .mlx import MLXBackend

        return MLXBackend(model, revision, **kwargs)
    if name == "hf":
        from .hf import HFBackend

        return HFBackend(model, revision, **kwargs)
    raise ValueError(f"unsupported backend: {name}")
