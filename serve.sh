#!/usr/bin/env bash
# Foreground GLM-5.3 + DFlash2 server. Defaults to a conservative 32K smoke run.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=${VLLM_NEXT_ROOT:-$HOME/vllm_glm53_dflash2}
SRC=${VLLM_NEXT_SRC:-$ROOT/vllm-src}
VENV=${VLLM_NEXT_VENV:-$ROOT/venv}
MODEL=${MODEL:-$HOME/models/GLM-5.3-Int4-Int8Mix-AWQ-g64}
DFLASH_MODEL=${DFLASH_MODEL:-$HOME/models/GLM-5.3-DFlash2}
PROFILE=${PROFILE:-smoke}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export CUDA_HOME
export PATH="$VENV/bin:$CUDA_HOME/bin:$PATH"

case "$PROFILE" in
  smoke)
    MAX_LEN=${MAX_LEN:-32768}; MAX_SEQS=${MAX_SEQS:-1}; MAX_BATCHED=${MAX_BATCHED:-1024}; EAGER=${EAGER:-1} ;;
  production)
    MAX_LEN=${MAX_LEN:-1048576}; MAX_SEQS=${MAX_SEQS:-8}; MAX_BATCHED=${MAX_BATCHED:-2048}; EAGER=${EAGER:-0} ;;
  agent)
    # Validated 2026-09-28 (PP=10, DFlash2 k=7): FULL decode graphs, canonical
    # top-k, fork PP hop optimizations, 4 concurrent sequences, full 1M
    # context (drafter KV window-bounded, see --block-size below).
    # MAX_BATCHED 512: each running request's drafter window groups reserve
    # window + (10 stages x max_num_batched_tokens) tokens of 16-token blocks,
    # each costing a whole 128-token pool block; 512 halves that (~900 vs
    # ~1666 pool blocks per running request) and is what v0.26 measured as
    # better agent TTFT.
    MAX_LEN=${MAX_LEN:-1048576}; MAX_SEQS=${MAX_SEQS:-4}; MAX_BATCHED=${MAX_BATCHED:-512}; EAGER=${EAGER:-0}
    ENABLE_FORK_PP_OPT=${ENABLE_FORK_PP_OPT:-1}; HOST=${HOST:-0.0.0.0}; PORT=${PORT:-8000} ;;
  *) echo "serve: PROFILE must be smoke, agent or production" >&2; exit 2 ;;
esac

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9}
# PP_SIZE x TP_SIZE must equal the visible GPU count; consecutive devices form
# a TP group (0,1), (2,3), ... which are the same-host-bridge pairs here.
export PP_SIZE=${PP_SIZE:-10}
export VLLM_PP_LAYER_PARTITION=${VLLM_PP_LAYER_PARTITION:-10,8,8,8,8,8,8,8,8,4}
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-$ROOT/cache/vllm}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$ROOT/cache/torchinductor}
export FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE:-$ROOT/cache}
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# Correctness/repeatability work already carried by the selected fork.
export VLLM_GLM5_DETERMINISTIC_MOE_ALIGN=${VLLM_GLM5_DETERMINISTIC_MOE_ALIGN:-1}
export VLLM_GLM5_MOE_MASK_PADDING=${VLLM_GLM5_MOE_MASK_PADDING:-1}
# Canonical top-k (what v0.26 validated as GLM52_TOPK_DET/PREFILL_TOPK_DET=
# canon). NOT the fork's TIEFIX/SORTED in-place post-processing: on this
# DSA target it corrupted decode (2026-09-28: decode-vs-prefill logprob
# disagreement 61/600 positions, up to 13 nats, stray "0"/"1"/"!" tokens at
# temperature > 0; canonical 5/600, noise floor). Each kernel is correct in
# isolation; the in-place rewrite is not safe as consumed here.
export GLM53_TOPK_CANON=${GLM53_TOPK_CANON:-1}
# v0.26's W8A16 Marlin lm_head, ported but OFF: on the real weight it saves
# only 0.27 ms per projection at 8 rows (1.16 -> 0.89 ms, ~0.5 ms/step) with
# 98.4% top-1 agreement vs BF16 on random activations. Set 8 to enable.
export GLM52_LMHEAD_BITS=${GLM52_LMHEAD_BITS-0}
# The fork's own canonical path is a torch sort over the full max_model_len
# row (~48 ms/step at 950K); GLM53_TOPK_CANON uses v0.26's Triton kernel.
export VLLM_GLM5_TOPK_CANONICAL=${VLLM_GLM5_TOPK_CANONICAL:-0}
export VLLM_GLM5_TOPK_TIEFIX=${VLLM_GLM5_TOPK_TIEFIX:-0}
export VLLM_GLM5_TOPK_SORTED=${VLLM_GLM5_TOPK_SORTED:-0}
export VLLM_GLM5_FLA_PIN_AUTOTUNE=${VLLM_GLM5_FLA_PIN_AUTOTUNE:-1}
export VLLM_GLM5_INDEXER_GATHER_CLAMP=${VLLM_GLM5_INDEXER_GATHER_CLAMP:-1}
export VLLM_GLM5_INDEXER_DECODE_ROWS=${VLLM_GLM5_INDEXER_DECODE_ROWS:-1}
export VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=${VLLM_SPARSE_INDEXER_MAX_LOGITS_MB:-128}

# The fork's fused cache-write glue assumes its BF16 reference path. Keep that
# family off until the packed-fp8 path has a dedicated equivalence benchmark.
export VLLM_GLM5_DECODE_IDX_GLUE=0
export VLLM_GLM5_DECODE_KERNELS=${VLLM_GLM5_DECODE_KERNELS:-0}

# Pipeline optimizations have kill switches and are enabled only after the
# eager smoke/correctness gate, not merely because the fork provides them.
if [ "${ENABLE_FORK_PP_OPT:-0}" = 1 ]; then
  export VLLM_PP_SPREAD_DECODES=${VLLM_PP_SPREAD_DECODES:-1}
  export VLLM_PP_PACKED_HOP=${VLLM_PP_PACKED_HOP:-1}
  export VLLM_PP_HOP_NO_METADATA=${VLLM_PP_HOP_NO_METADATA:-1}
  export VLLM_PP_SPLIT_DRAFT_EVENT=${VLLM_PP_SPLIT_DRAFT_EVENT:-1}
else
  export VLLM_PP_SPREAD_DECODES=0 VLLM_PP_PACKED_HOP=0
  export VLLM_PP_HOP_NO_METADATA=0 VLLM_PP_SPLIT_DRAFT_EVENT=0
fi
export VLLM_GLM5_PP_FOLD_DRAFT_FC=${VLLM_GLM5_PP_FOLD_DRAFT_FC:-0}
export VLLM_PP_DRAFT_TAIL_STAGE=${VLLM_PP_DRAFT_TAIL_STAGE:--1}

mkdir -p "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR"
PREFLIGHT=("$VENV/bin/python" "$HERE/preflight.py" --target "$MODEL" --draft "$DFLASH_MODEL" --engine "$SRC" --host)
if [ "${ALLOW_BUSY_GPUS:-0}" != 1 ]; then PREFLIGHT+=(--require-idle); fi
"${PREFLIGHT[@]}"

SPEC_TOKENS=${SPEC_TOKENS:-7}
# The drafter's attention is dense (head 128, non-causal, sliding window); no
# dense backend reads the target's packed fp8_ds_mla rows, so it keeps its own
# BF16 cache ("auto") while the target stays packed-FP8.
DRAFT_KV_DTYPE=${DRAFT_KV_DTYPE:-auto}
SPEC=(--speculative-config "{\"method\":\"dflash\",\"model\":\"$DFLASH_MODEL\",\"num_speculative_tokens\":$SPEC_TOKENS,\"kv_cache_dtype\":\"$DRAFT_KV_DTYPE\"}")
# NO_SPEC=1 serves the target alone (bisection/A-B only).
if [ "${NO_SPEC:-0}" = 1 ]; then SPEC=(); fi
GRAPH=(--enforce-eager)
if [ "$EAGER" != 1 ]; then
  # FULL decode graphs + PIECEWISE prefill. 2026-09-28, PP=10 + DFlash2: once
  # the TIEFIX/SORTED top-k post-processing was replaced by canonical top-k,
  # FULL is clean (greedy 12-run probe 0 token-0, decode-vs-prefill 1/600) at
  # ~85-88 ms/step vs PIECEWISE ~90-94 and eager ~237. COMPILATION_CONFIG
  # (JSON) overrides it.
  # (A JSON default cannot sit inside ${VAR:-...}: bash ends it at the first
  # "}" and appends the rest literally, corrupting any override.)
  CC_DEFAULT='{"cudagraph_mode":"FULL_AND_PIECEWISE"}'
  GRAPH=(--compilation-config "${COMPILATION_CONFIG:-$CC_DEFAULT}")
fi

echo "serve: profile=$PROFILE model=$MODEL draft=$DFLASH_MODEL k=$SPEC_TOKENS max_len=$MAX_LEN partition=$VLLM_PP_LAYER_PARTITION" >&2
# The fork's PP + spec-decode path is built on async scheduling. With it off,
# the engine's batch queue pulls the drafter's placeholder drafts into a
# request whose prefill is still in flight on the pipeline, before its sampled
# token is back; the next step then carries k drafts in only k query rows and
# the worker asserts (seen 2026-09-27, PP=10 and PP=2 DFlash2). ASYNC_SCHED=0
# restores --no-async-scheduling for A/B only.
# --block-size 128 (was 64): lets the DFlash drafter's 16-token BF16 pages
# (64 KiB) ride inside the 128-token packed-fp8 MLA pages (82 KiB) at
# disjoint block ids (apply_engine_patch.py, local-cmp170hx-dsa-dflash-kv), so
# drafter KV is bounded by its 2048-token window instead of full length.
SCHED_ARGS=()
if [ "${ASYNC_SCHED:-1}" != 1 ]; then SCHED_ARGS=(--no-async-scheduling); fi
# v0.26 fix carried over (upstream_writeup_router_precision_and_boot_lottery.md
# section 2): Inductor re-benchmarks its RMSNorm reductions on every boot, so
# compiled outputs differ boot to boot. Pin the norms to vLLM's CUDA kernels.
KERNEL_ARGS=()
if [ "${PIN_NORMS:-1}" = 1 ]; then
  KERNEL_ARGS=(--kernel-config '{"ir_op_priority":{"rms_norm":["vllm_c"],"fused_add_rms_norm":["vllm_c"]}}')
fi

exec "$VENV/bin/vllm" serve "$MODEL" \
  "${SPEC[@]}" \
  --served-model-name "${SERVED_MODEL_NAME:-glm-5.3}" \
  --host "${HOST:-127.0.0.1}" --port "${PORT:-8001}" \
  --pipeline-parallel-size "$PP_SIZE" --tensor-parallel-size "${TP_SIZE:-1}" \
  --attention-backend TRITON_MLA_SPARSE "${SCHED_ARGS[@]}" \
  --dtype bfloat16 --kv-cache-dtype fp8_ds_mla --block-size "${BLOCK_SIZE:-128}" \
  "${GRAPH[@]}" "${KERNEL_ARGS[@]}" \
  --max-model-len "$MAX_LEN" --max-num-seqs "$MAX_SEQS" \
  --max-num-batched-tokens "$MAX_BATCHED" \
  --gpu-memory-utilization "${GPU_UTIL:-0.93}" \
  --reasoning-parser glm47 --enable-auto-tool-choice --tool-call-parser glm47 \
  ${EXTRA_ARGS:-}
