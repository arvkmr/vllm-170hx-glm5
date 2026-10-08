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
    # MAX_LEN 512K since 2026-10-03 (PP=9): the pool is ~1.14M tokens, so no
    # single request may take 1M. The gateway's largest window is 512K.
    MAX_LEN=${MAX_LEN:-524288}; MAX_SEQS=${MAX_SEQS:-4}; MAX_BATCHED=${MAX_BATCHED:-512}; EAGER=${EAGER:-0}
    # Private since 2026-09-30: the vision proxy (~/vision_sidecar/proxy.py)
    # owns the public 0.0.0.0:8000 and forwards here.
    ENABLE_FORK_PP_OPT=${ENABLE_FORK_PP_OPT:-1}; HOST=${HOST:-127.0.0.1}; PORT=${PORT:-8002} ;;
  *) echo "serve: PROFILE must be smoke, agent or production" >&2; exit 2 ;;
esac

# PP_SIZE x TP_SIZE must equal the visible GPU count; consecutive devices form
# a TP group (0,1), (2,3), ... which are the same-host-bridge pairs here.
# PP=9 since 2026-10-03: serial 1322421041986 was pulled after falling off the
# bus twice. Set PP_SIZE=10 again once a tenth card is in.
export PP_SIZE=${PP_SIZE:-9}
export TP_SIZE=${TP_SIZE:-1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((PP_SIZE * TP_SIZE - 1)))}
case "$PP_SIZE" in
  # 9 stages need the top-k PP relay (stages 19, 28, 37, 55, 64, 72 begin on
  # skip-topk layers). Nine MoE layers on stages 1-6 bound the KV pool.
  9) DEFAULT_PARTITION=10,9,9,9,9,9,9,8,6 ;;
  *) DEFAULT_PARTITION=10,8,8,8,8,8,8,8,8,4 ;;
esac
export VLLM_PP_LAYER_PARTITION=${VLLM_PP_LAYER_PARTITION:-$DEFAULT_PARTITION}
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
# Of the fork's sm_80 decode kernels only the fused MoE router is used: gate
# GEMV + sigmoid top-k + Marlin alignment in one launch (~11 us vs ~39 us per
# MoE layer). apply_engine_patch.py widens its gate to this 256 x 6144 router;
# expert ids and alignment are identical to the unfused chain. The other
# families are off (mHC/KDA belong to GLM-5.3-Flash; moe_routing is 288-expert
# only), and DECODE_MOE_MAX_TOKENS=0 keeps the MoE padding mask at every batch
# size -- with the decode kernels on, the runner otherwise skips it at <= 8 rows.
export VLLM_GLM5_DECODE_KERNELS=${VLLM_GLM5_DECODE_KERNELS:-1}
export VLLM_GLM5_DECODE_MOE_ROUTE_V2=${VLLM_GLM5_DECODE_MOE_ROUTE_V2:-1}
export VLLM_GLM5_DECODE_MOE_ROUTING=0 VLLM_GLM5_DECODE_MOE_MAX_TOKENS=0
export VLLM_GLM5_DECODE_MHC=0 VLLM_GLM5_DECODE_MHC_V2=0
export VLLM_GLM5_DECODE_KDA=0 VLLM_GLM5_DECODE_KDA_V2=0
# Enqueue the shared experts on the aux stream after the routed experts so the
# two actually overlap (enqueue order only; numerics unchanged). ~1 ms/step.
export VLLM_GLM5_SHARED_EXPERT_REORDER=${VLLM_GLM5_SHARED_EXPERT_REORDER:-1}
# The fork's sm_80 thin-M BF16 GEMM (indexer, dense layers 0-2, drafter,
# lm_head) is OFF: on this model it saves only ~0.75 ms/step and DFlash2
# acceptance measured 4% lower with it (3.22 vs 3.36 tok/step, 8 prompts).
# apply_engine_patch.py still keeps its two losing shapes on cuBLAS.
export VLLM_GLM5_THIN_GEMM=${VLLM_GLM5_THIN_GEMM:-0}

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
  --pipeline-parallel-size "$PP_SIZE" --tensor-parallel-size "$TP_SIZE" \
  --attention-backend TRITON_MLA_SPARSE "${SCHED_ARGS[@]}" \
  --dtype bfloat16 --kv-cache-dtype fp8_ds_mla --block-size "${BLOCK_SIZE:-128}" \
  "${GRAPH[@]}" "${KERNEL_ARGS[@]}" \
  --max-model-len "$MAX_LEN" --max-num-seqs "$MAX_SEQS" \
  --max-num-batched-tokens "$MAX_BATCHED" \
  --gpu-memory-utilization "${GPU_UTIL:-0.93}" \
  --reasoning-parser glm47 --enable-auto-tool-choice --tool-call-parser glm47 \
  ${EXTRA_ARGS:-}
