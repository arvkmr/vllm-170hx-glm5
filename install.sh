#!/usr/bin/env bash
# Build the pinned vNext engine beside, never over, the vLLM 0.26 install.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT="$HERE"
# shellcheck source=versions.env
source "$HERE/versions.env"

ROOT=${VLLM_NEXT_ROOT:-$HOME/vllm_glm53_dflash2}
SRC=${VLLM_NEXT_SRC:-$ROOT/vllm-src}
VENV=${VLLM_NEXT_VENV:-$ROOT/venv}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
[ -x "$CUDA_HOME/bin/nvcc" ] || {
  echo "install: CUDA compiler not found at $CUDA_HOME/bin/nvcc" >&2
  exit 1
}
case $("$CUDA_HOME/bin/nvcc" --version) in
  *"release 13."*) ;;
  *)
    echo "install: CUDA 13.x is required; $CUDA_HOME/bin/nvcc reports a different release" >&2
    exit 1
    ;;
esac
UV=${UV:-$(command -v uv || true)}
if [ -z "$UV" ] && [ -x "$HOME/.local/bin/uv" ]; then UV=$HOME/.local/bin/uv; fi
[ -n "$UV" ] || { echo "install: uv not found (expected uv or $HOME/.local/bin/uv)" >&2; exit 1; }

mkdir -p "$ROOT"
if [ ! -d "$SRC/.git" ]; then
  git clone --filter=blob:none --branch "$VLLM_BRANCH" "$VLLM_REPO" "$SRC"
fi
if ! git -C "$SRC" rev-parse --verify -q "$VLLM_COMMIT^{commit}" >/dev/null; then
  git -C "$SRC" fetch origin "$VLLM_BRANCH"
fi
HEAD=$(git -C "$SRC" rev-parse HEAD)
if [ "$HEAD" != "$VLLM_COMMIT" ]; then
  if [ -n "$(git -C "$SRC" status --porcelain)" ]; then
    echo "install: source is at $HEAD with local changes; refusing to overwrite it" >&2
    exit 1
  fi
  git -C "$SRC" checkout -B glm53-dflash2-local "$VLLM_COMMIT"
fi
git -C "$SRC" submodule update --init --recursive --depth 1
python3 "$HERE/apply_engine_patch.py" --source "$SRC" --helper "$PROJECT/glm52_mla_fp8.py"

if [ ! -x "$VENV/bin/python" ]; then
  "$UV" venv --python "$PYTHON_VERSION" "$VENV"
fi
PIP=("$UV" pip install --python "$VENV/bin/python" --extra-index-url https://flashinfer.ai/whl/)
"${PIP[@]}" "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" "torchaudio==$TORCHAUDIO_VERSION"
VLLM_USE_PRECOMPILED=1 \
VLLM_PRECOMPILED_WHEEL_COMMIT="$VLLM_PRECOMPILED_WHEEL_COMMIT" \
CUDA_HOME="$CUDA_HOME" \
  "${PIP[@]}" -e "$SRC"
"${PIP[@]}" -r "$SRC/requirements/cuda.txt" ninja 'huggingface_hub[hf_xet]>=1.0'

"$VENV/bin/python" - <<'PY'
import importlib
import torch
import vllm
import vllm._custom_ops  # noqa: F401
from vllm.v1.attention.backends.mla import triton_mla_sparse  # noqa: F401
from vllm import _glm52_mla_fp8  # noqa: F401
for name in ("triton", "tilelang", "flashinfer"):
    importlib.import_module(name)
print(f"install: vllm={vllm.__version__} torch={torch.__version__} cuda={torch.version.cuda}")
print("install: pinned engine and packed-fp8 reader import successfully")
PY

PATCH_SHA=$(sha256sum "$PROJECT/glm52_mla_fp8.py" | cut -d' ' -f1)
printf 'engine=%s\npatch=%s\n' "$VLLM_COMMIT" "$PATCH_SHA" >"$ROOT/install.stamp"
echo "install: complete at $ROOT"
