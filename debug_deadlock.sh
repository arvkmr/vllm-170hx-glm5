#!/usr/bin/env bash
# Reproduce the PP+spec concurrency deadlock and capture per-rank stacks.
#
# Phase 1: single-stream speed (safe -- validated correct).
# Phase 2: fire 2 concurrent requests; if they don't finish in REPRO_WAIT s,
#          py-spy dump every VLLM worker + the engine core so we can see the
#          exact line each rank is wedged on, then leave the server as-is
#          (caller decides whether to restart).
set -uo pipefail
cd /home/user/vllm_install
PYSPY=.venv/bin/py-spy
OUT=logs/deadlock_stacks_$(date +%H%M%S).txt
REPRO_WAIT=${REPRO_WAIT:-45}

req() {
  curl -s -m 600 localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d "{
   \"model\":\"glm-5.2\",\"messages\":[{\"role\":\"user\",\"content\":\"$1\"}],
   \"max_tokens\":48,\"min_tokens\":48,\"temperature\":0.0,
   \"chat_template_kwargs\":{\"enable_thinking\":false}}" \
   | python3 -c "import json,sys
try: print('DONE:', json.load(sys.stdin)['usage']['completion_tokens'], 'tok')
except Exception: print('FAILED/EMPTY')"
}

echo "===== Phase 1: single-stream speed (baseline no-MTP: ~24 tok/s) ====="
.venv/bin/python bench_glm52.py --gen 128 --conc 1 1 | tail -2

echo
echo "===== Phase 2: 2-way concurrency repro ====="
req "Write a haiku about mountains." > /tmp/claude-1000/req1.out 2>&1 &
P1=$!
req "Write a haiku about rivers." > /tmp/claude-1000/req2.out 2>&1 &
P2=$!

waited=0
while kill -0 $P1 2>/dev/null || kill -0 $P2 2>/dev/null; do
  sleep 3; waited=$((waited+3))
  if [ $waited -ge $REPRO_WAIT ]; then
    echo "STUCK after ${waited}s -- dumping stacks to $OUT"
    {
      for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' '); do
        name=$(cat /proc/$p/comm 2>/dev/null)
        case "$name" in VLLM*|pt_main_thread|python*) ;; *) continue;; esac
        echo "############ pid $p ($name) ############"
        sudo -n "$PYSPY" dump --pid "$p" 2>&1 || "$PYSPY" dump --pid "$p" 2>&1
        echo
      done
      ec=$(pgrep -f 'EngineCore' | head -1)
      # EngineCore proc name is python; find it via cmdline of vllm serve children
      for p in $(pgrep -P "$(pgrep -f '[b]in/vllm serve' | head -1)" 2>/dev/null); do
        echo "############ engine-core-candidate pid $p ############"
        "$PYSPY" dump --pid "$p" 2>&1
        echo
      done
    } > "$OUT"
    echo "stacks written: $OUT"
    exit 2
  fi
done
echo "both finished (no hang):"
cat /tmp/claude-1000/req1.out /tmp/claude-1000/req2.out
