#!/usr/bin/env bash
# Start the GLM-5.2 server detached (survives the launching shell) and wait
# for it to come up. Companion to stop_glm52.sh; the actual vllm invocation
# and all config live in serve_glm52.sh.
#
# Defaults are the validated config: 204800 max-model-len, 5.5 GiB/rank KV
# (442,943 KV tokens), MTP k=3. Env knobs (all pass through to serve script):
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
cd /home/user/vllm_install

if pgrep -f '[b]in/vllm serve' > /dev/null; then
  echo "vllm serve is already running (pid $(pgrep -f '[b]in/vllm serve' | head -1))." >&2
  echo "Run ./stop_glm52.sh first." >&2
  exit 1
fi

# The NFS "fast" mount is not restored on boot; fail early with a clear
# message instead of a confusing loader error 2 minutes in.
MODEL=${MODEL:-/home/user/srv/fast/models/GLM-5.2-AWQ-g64}
if [ ! -e "$MODEL/config.json" ]; then
  echo "Model not found at $MODEL" >&2
  echo "Is the NFS 'fast' export mounted? (not auto-mounted on boot)" >&2
  exit 1
fi

mkdir -p logs
LOG=${LOG:-logs/glm52.$(date +%Y%m%d-%H%M%S).log}
setsid nohup ./serve_glm52.sh > "$LOG" 2>&1 < /dev/null &
disown
ln -sf "$(basename "$LOG")" logs/glm52.latest.log
echo "launched; log: $LOG (symlinked as logs/glm52.latest.log)"

if [ "${WAIT:-1}" != "1" ]; then
  exit 0
fi

echo -n "waiting for health (cold start ~10 min)"
for _ in $(seq 1 90); do
  if curl -s -m 2 localhost:8000/health > /dev/null 2>&1; then
    echo
    echo "server healthy."
    grep -hE "KV cache size|Maximum concurrency" "$LOG" | tail -2
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
echo "timed out after 15 min; still starting? check $LOG" >&2
exit 1
