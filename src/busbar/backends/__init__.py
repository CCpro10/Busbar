"""Load optional accelerator dependencies only when their backend is selected."""

from .base import DEFAULT_MODEL, DEFAULT_REVISION


def load_backend(name="mlx", model=DEFAULT_MODEL, revision=DEFAULT_REVISION, **kwargs):
    """Construct a supported backend without importing MLX on Linux or Torch on Mac by default."""
    if name == "mlx":
        from .mlx import MLXBackend

        return MLXBackend(model, revision, **kwargs)
    if name == "hf":
        from .hf import HFBackend

        return HFBackend(model, revision, **kwargs)
    raise ValueError(f"unsupported backend: {name}")
