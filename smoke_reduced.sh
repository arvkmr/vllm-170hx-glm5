#!/usr/bin/env bash
# One-GPU integration smoke of the sm_80 DSA path: the real GLM-5.3 config cut
# to a few layers, dummy weights, the production attention/KV settings. It
# finds engine blockers in minutes instead of a 10-GPU real-weight boot; it
# says nothing about output quality. Localhost only, its own port (8002).
#
#   CUDA_VISIBLE_DEVICES=0 ./smoke_reduced.sh        # serves in the foreground
#   CUDA_VISIBLE_DEVICES=0,1 PP=2 DRAFT=~/vllm_glm53_dflash2/smoke-drafter \
#     ./smoke_reduced.sh                              # PP + DFlash2 path
# DRAFT must be a drafter copy whose target_layer_ids fit inside LAYERS.
set -euo pipefail
ROOT=${VLLM_NEXT_ROOT:-$HOME/vllm_glm53_dflash2}
VENV=${VLLM_NEXT_VENV:-$ROOT/venv}
MODEL=${MODEL:-$HOME/models/GLM-5.3-Int4-Int8Mix-AWQ-g64}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export CUDA_HOME PATH="$VENV/bin:$CUDA_HOME/bin:$PATH"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-$ROOT/cache/vllm}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$ROOT/cache/torchinductor}
export OMP_NUM_THREADS=1
export VLLM_GLM5_DETERMINISTIC_MOE_ALIGN=1 VLLM_GLM5_MOE_MASK_PADDING=1
export GLM53_TOPK_CANON=1 VLLM_GLM5_TOPK_CANONICAL=0 VLLM_GLM5_TOPK_TIEFIX=0 VLLM_GLM5_TOPK_SORTED=0
export VLLM_GLM5_DECODE_IDX_GLUE=0 VLLM_GLM5_DECODE_KERNELS=0
export VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=${VLLM_SPARSE_INDEXER_MAX_LOGITS_MB:-128}

SPEC=()
if [ -n "${DRAFT:-}" ]; then
  SPEC=(--speculative-config "{\"method\":\"dflash\",\"model\":\"$DRAFT\",\"num_speculative_tokens\":${SPEC_TOKENS:-7},\"kv_cache_dtype\":\"auto\"}")
fi

# The fork's PP + spec-decode path is built on async scheduling. With it off,
# the engine's batch queue pulls the drafter's placeholder drafts into a
# request whose prefill is still in flight on the pipeline, before its sampled
# token is back; the next step then carries k drafts in only k query rows and
# the worker asserts (seen 2026-09-27, PP=10 and PP=2 DFlash2). ASYNC_SCHED=0
# restores --no-async-scheduling for A/B only.
# EAGER=0 drops --enforce-eager; COMPILATION_CONFIG (JSON) is then passed
# through, else the engine's default graph mode applies.
GRAPH=(--enforce-eager)
if [ "${EAGER:-1}" != 1 ]; then
  GRAPH=()
  if [ -n "${COMPILATION_CONFIG:-}" ]; then GRAPH=(--compilation-config "$COMPILATION_CONFIG"); fi
fi
SCHED_ARGS=()
if [ "${ASYNC_SCHED:-1}" != 1 ]; then SCHED_ARGS=(--no-async-scheduling); fi

exec "$VENV/bin/vllm" serve "$MODEL" \
  "${SPEC[@]}" \
  --pipeline-parallel-size "${PP:-1}" \
  --load-format dummy \
  --hf-overrides "{\"num_hidden_layers\": ${LAYERS:-8}}" \
  --served-model-name smoke \
  --host 127.0.0.1 --port "${PORT:-8002}" \
  --attention-backend TRITON_MLA_SPARSE "${SCHED_ARGS[@]}" \
  --dtype bfloat16 --kv-cache-dtype fp8_ds_mla --block-size "${BLOCK_SIZE:-128}" \
  "${GRAPH[@]}" \
  --max-model-len "${MAX_LEN:-16384}" --max-num-seqs "${MAX_SEQS:-4}" \
  --max-num-batched-tokens "${MAX_BATCHED:-1024}" \
  --gpu-memory-utilization "${GPU_UTIL:-0.85}" \
  ${EXTRA_ARGS:-}
