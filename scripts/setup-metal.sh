#!/bin/bash
# Keep native extension ABI pins separate from the ordinary MLX/HF environment.
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "vLLM Metal requires an Apple Silicon Mac running macOS 15 or later." >&2
  exit 1
fi
busbar_root="$(cd "$(dirname "$0")/.." && pwd)"
busbar_env="${1:-$busbar_root/.venv-metal}"
if [[ ! -x "$busbar_env/bin/python" ]]; then
  uv venv --python 3.12 "$busbar_env"
fi
uv pip install --python "$busbar_env/bin/python" \
  -r "$busbar_root/requirements-metal.lock" -e "$busbar_root"
uv pip check --python "$busbar_env/bin/python"
echo "Ready: $busbar_env/bin/busbar serve --backend vllm-metal --model openbmb/MiniCPM5-2B"
