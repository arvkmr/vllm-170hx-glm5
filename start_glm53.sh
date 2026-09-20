#!/usr/bin/env bash
# Start the GLM-5.3 server detached (survives the launching shell) and wait
# for it to come up. Companion to stop_glm53.sh; the actual vllm invocation
# and all config live in serve_glm53.sh.
#
# Defaults target 12 cards: PP=12, 409600 max-model-len (400K per stream),
# 20 GiB/rank fp8_ds_mla KV (estimated 4,422,272 slots, 10.80x capacity at
# full context), MTP k=3, hybrid full cudagraphs, PP topk relay + canonical
# topk + deterministic moe_align. Env knobs (all pass through to serve script):
#   SPEC_TOKENS=0            disable MTP speculative decoding
#   MAX_LEN=32768            shorter context
#   KV_CACHE_MEM=""          use GPU_UTIL fraction instead of fixed KV budget
#   GLM52_PP_SPEC_GLOBAL_SER=1  fully serialize PP batches (slow, escape hatch)
#   GLM52_PROF=1             per-span indexer timing to the log (diagnostic)
#   WAIT=0                   don't block waiting for health
#
# Cold start is ~10 min: ~2 min weight load (warm NFS cache) + 33 CUDA graph
# captures at ~17 s each.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# Reject invalid topology before detaching or waiting through a model load.
source "$SCRIPT_DIR/glm_profile.sh"
export VLLM_INSTALL_DIR=${VLLM_INSTALL_DIR:-/home/user/vllm_install}
cd "$VLLM_INSTALL_DIR"

if pgrep -f '[b]in/vllm serve' > /dev/null; then
  echo "vllm serve is already running (pid $(pgrep -f '[b]in/vllm serve' | head -1))." >&2
  echo "Run ./stop_glm53.sh first." >&2
  exit 1
fi

# Model lives on local SSD (rsynced from the NAS "fast" export); fail early
# with a clear message instead of a confusing loader error 2 minutes in.
MODEL=${MODEL:-/home/user/models/GLM-5.3-AWQ-g64}
if [ ! -e "$MODEL/config.json" ]; then
  echo "Model not found at $MODEL" >&2
  echo "Re-sync it: rsync -a /home/user/srv/fast/models/GLM-5.3-AWQ-g64/ $MODEL/" >&2
  exit 1
fi

mkdir -p logs
LOG=${LOG:-logs/glm53.$(date +%Y%m%d-%H%M%S).log}
setsid nohup "$SCRIPT_DIR/serve_glm53.sh" > "$LOG" 2>&1 < /dev/null &
disown
ln -sf "$(basename "$LOG")" logs/glm53.latest.log
echo "launched; log: $LOG (symlinked as logs/glm53.latest.log)"

if [ "${WAIT:-1}" != "1" ]; then
  exit 0
fi

echo -n "waiting for health (warm start ~10 min; first boot with cold compile cache ~20 min)"
for _ in $(seq 1 180); do
  if curl -fsS -m 2 "http://localhost:${PORT:-8000}/health" > /dev/null 2>&1; then
    echo
    echo "server healthy."
    grep -hE "KV cache size|Maximum concurrency" "$LOG" | tail -2 || true
    exit 0
  fi
  if ! pgrep -f '[b]in/vllm serve' > /dev/null; then
    echo
    echo "server died during startup; last log lines:" >&2
    tail -25 "$LOG" >&2
    exit 1
  fi
  sleep 10
  echo -n .
done
echo
echo "timed out after 30 min; still starting? check $LOG" >&2
exit 1
