#!/usr/bin/env bash
# Restore the PP=9 production setup saved on 2026-10-05 (before PP=8 on
# GPUs 0-5,7,8). Stops GLM, puts the PP=9 scripts back, starts GLM again.
#   ~/vllm_install/vllm_next/rollback-pp9-20261005/rollback.sh
# Set NO_START=1 to only restore the files.
set -euo pipefail
SNAP=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DIR=$(dirname "$SNAP")
cd "$DIR"
./stop.sh || true
for f in serve.sh serve-uncensored.sh preflight.py start.sh stop.sh server-alert.py glm-watchdog.sh; do
  cp -p "$SNAP/$f" "$DIR/$f"
done
echo "rollback: PP=9 scripts restored from $SNAP"
if [ "${NO_START:-0}" != 1 ]; then
  SERVE_SCRIPT=${SERVE_SCRIPT:-serve-uncensored.sh} ./start.sh
fi
