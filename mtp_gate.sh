#!/usr/bin/env bash
# Full MTP validation gate, in order of importance:
#   1. Correctness: greedy output token-identical to the SPEC_TOKENS=0 baseline
#      (mtp_verify.py). If this fails nothing else matters.
#   2. Acceptance rate: drafts must be accepted well above 0% or the draft
#      weights/wiring are wrong even if output is preserved (rejection sampling
#      masks a garbage draft as pure overhead).
#   3. Speed: single-stream decode vs the 24 tok/s no-MTP baseline.
set -uo pipefail
cd /home/user/vllm_install

BASELINE=${BASELINE:-/tmp/claude-1000/-home-user-vllm-install/a6225a47-979a-4817-9582-5541f9209d66/scratchpad/baseline_nomtp.json}

echo "===== 1. CORRECTNESS (must be token-identical) ====="
.venv/bin/python mtp_verify.py check "$BASELINE"
verdict=$?
echo

if [ $verdict -ne 0 ]; then
  echo "GATE FAILED at correctness -- skipping perf measurements."
  exit 1
fi

echo "===== 2. ACCEPTANCE RATE ====="
.venv/bin/python mtp_accept.py 4 128
echo

echo "===== 3. SINGLE-STREAM SPEED (baseline: ~24 tok/s decode) ====="
.venv/bin/python bench_glm52.py --gen 128 --conc 1 1
