#!/usr/bin/env bash
# Stop the GLM-5.2 server and wait until the GPUs are genuinely free.
#
# Two traps this works around, both of which caused spurious
# "Free memory on device cuda:N ... is less than desired GPU memory utilization"
# failures on the next launch:
#
#  1. Killing the `vllm serve` parent does NOT reap its 8 PP workers. They keep
#     running (and keep loading weights) as orphans.
#  2. vLLM renames workers to "VLLM::Worker_PPn" -- uppercase, and with no
#     "vllm serve" in the cmdline -- so pattern-matching on the launch command
#     misses them entirely. Drive off the pids nvidia-smi reports instead, which
#     is the ground truth for "who is holding VRAM".
#
# Also note: the driver frees VRAM asynchronously, so require several
# consecutive clear samples before declaring success.
#
# Do NOT use `pkill -f "vllm serve"` -- that pattern matches the calling shell's
# own command line and kills it.
set -uo pipefail

TIMEOUT=${TIMEOUT:-120}
NEED_CLEAR=${NEED_CLEAR:-3}   # consecutive clear samples required
deadline=$(( SECONDS + TIMEOUT ))
clear_count=0

# Ask the parent to go down gracefully first.
parents=$(pgrep -f '[b]in/vllm serve' || true)
[ -n "$parents" ] && kill -TERM $parents 2>/dev/null

while [ $SECONDS -lt $deadline ]; do
  # Ground truth: who currently holds GPU memory.
  mapfile -t gpu_pids < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ')

  ours=()
  for p in "${gpu_pids[@]:-}"; do
    [ -z "$p" ] && continue
    cmd=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null)
    comm=$(cat "/proc/$p/comm" 2>/dev/null)
    # Match our stack case-insensitively, including the renamed workers.
    if printf '%s %s' "$cmd" "$comm" | grep -qiE 'vllm|multiprocessing\.spawn|resource_tracker'; then
      ours+=("$p")
    fi
  done

  if [ ${#ours[@]} -eq 0 ]; then
    busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits |
           awk '$1 > 1024 {n++} END {print n+0}')
    if [ "$busy" -eq 0 ]; then
      clear_count=$(( clear_count + 1 ))
      if [ "$clear_count" -ge "$NEED_CLEAR" ]; then
        echo "GPUs clear."
        nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
        exit 0
      fi
    else
      clear_count=0
    fi
    sleep 2
    continue
  fi

  clear_count=0
  kill -9 "${ours[@]}" 2>/dev/null
  sleep 2
done

echo "WARNING: GPU memory still held after ${TIMEOUT}s:" >&2
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader >&2
exit 1
