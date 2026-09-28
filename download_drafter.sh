#!/usr/bin/env bash
# Download only the 2B DFlash2 drafter; the target checkpoint already exists.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=versions.env
source "$HERE/versions.env"

if [ "${DFLASH_LICENSE_ACK:-0}" != 1 ]; then
  echo "download: set DFLASH_LICENSE_ACK=1 after reviewing CC-BY-NC-ND-4.0; commercial use needs an Inco license" >&2
  exit 1
fi
ROOT=${VLLM_NEXT_ROOT:-$HOME/vllm_glm53_dflash2}
VENV=${VLLM_NEXT_VENV:-$ROOT/venv}
DEST=${DFLASH_MODEL:-$HOME/models/GLM-5.3-DFlash2}
[ -x "$VENV/bin/hf" ] || { echo "download: run ./install.sh first" >&2; exit 1; }
mkdir -p "$DEST"
"$VENV/bin/hf" download "$DFLASH_REPO" --revision "$DFLASH_REVISION" --local-dir "$DEST"
printf '%s\n' "$DFLASH_REVISION" >"$DEST/.pinned-revision"
echo "download: pinned $DFLASH_REPO@$DFLASH_REVISION in $DEST"
