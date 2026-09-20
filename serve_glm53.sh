#!/usr/bin/env bash
# Serve GLM-5.3 (753B DSA MoE, AWQ-INT4 g64) across 12 GPUs with pipeline
# parallelism (PP=12).
#
# GLM-5.3 uses DeepSeek Sparse Attention. Upstream vLLM only has Hopper/Blackwell
# sparse-MLA backends (FLASHMLA_SPARSE) and DeepGEMM's fp8_mqa_logits, none of
# which build for sm_80. This box runs the TRITON_MLA_SPARSE backend from vLLM
# PR #38476, hand-applied into the venv's site-packages (see logs/ and the
# backup at .venv/lib/python3.12/site-packages/.glm52-backup).
set -euo pipefail

# Run from a neutral directory: launching from inside the installed vllm package
# makes vllm/tokenizers/ shadow the real `tokenizers` package on sys.path.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/glm_profile.sh"
VLLM_INSTALL_DIR=${VLLM_INSTALL_DIR:-/home/user/vllm_install}
cd "$VLLM_INSTALL_DIR"

VENV=${VENV:-$VLLM_INSTALL_DIR/.venv}
# Local SSD copy (rsynced 2026-09-02 from the NAS "fast" export at
# /home/user/srv/fast/models/GLM-5.3-AWQ-g64). Loading locally avoids the
# NFS mount, which is NOT restored on boot.
MODEL=${MODEL:-/home/user/models/GLM-5.3-AWQ-g64}
PORT=${PORT:-8000}

# THE concurrency lever, and the easiest one to lose: stock vLLM merges every
# ready decode into one lockstep batch, which leaves pipeline depth at 1 and the
# GPUs ~10% busy (conc-16 ~80 tok/s). The scheduler patch in
# vllm/v1/core/sched/scheduler.py splits decodes into pipelined groups, but it
# reads this env var and DEFAULTS TO 0 (off) -- so a serve script that forgets
# to export it silently runs unpatched-equivalent. That is exactly what happened
# between 2026-08-09 and 2026-08-12: the patch was in site-packages the whole
# time, unset here, and conc-16 sat at ~80 (~120 measured with a same-prompt
# harness) against the 190 reference.
# Reference conc-16: 211 tok/s spec-off, ~190 spec-on hybrid, ~80 = cap missing.
export GLM52_PP_DECODE_BATCH_CAP=${GLM52_PP_DECODE_BATCH_CAP:-2}

# ...but 2 is only right when enough requests are decoding to fill the pipe.
# The cap trades pipeline depth (small cap = more groups) against MoE batch
# efficiency (large cap = fewer, fatter batches amortising Marlin's fixed
# floor), and which side wins depends on the NUMBER of decoding requests, not
# on context length. Measured, aggregate decode tok/s:
#                     cap=2   cap=1
#   conc=4  (512 ctx)  78.8   101.3
#   conc=8  (512 ctx) 163.6   135.5
#   conc=16 (512 ctx) 189.7   142.7
#   4 x 64K            90.8   100.3
#   4 x 192K           45.8    67.7
# So patch_pp_adaptive_cap.py drops the cap to 1 while fewer than this many
# requests are decoding. Best of both, measured with it on: 4x192K 71.1
# (+55% vs constant cap=2), 4x64K 99.3, conc=16 202.7 (better than either
# constant, and finally at the 203 ledger reference). 0 = constant cap.
export GLM52_PP_DECODE_ADAPTIVE=${GLM52_PP_DECODE_ADAPTIVE:-$PP}

# glm_profile.sh sets the topology and fixed KV budget: 20 GiB/rank for
# the built-in PP=12 split, estimated 4,422,272 fp8 KV slots. This profile
# needs on-host validation; KV_CACHE_MEM=12884901888 restores 12 GiB.
# KV_CACHE_MEM="" uses vLLM profiling with GPU_UTIL (default 0.93).
# A fresh card previously reported ~59.6 GiB free out of 63.39 GiB.

# bf16, not the checkpoint's declared float16. The PR #38476 Triton kernels emit
# bf16 unconditionally (`.to(tl.bfloat16)` in triton_mla_sparse_kernel.py and a
# hardcoded torch.bfloat16 output buffer), so running fp16 dies in _v_up_proj
# with "expected scalar type BFloat16 but found Half" -- the attention output is
# bf16 while W_UV is fp16. bf16 is what the backend is designed for.

# The Ampere sparse-MLA path. The gist sets VLLM_ATTENTION_BACKEND, but 0.26.0
# dropped that env var (it only logs "Unknown vLLM environment variable
# detected" and is ignored) -- the knob is now the --attention-backend flag
# below. Auto-selection would also land here, since the sm_80 priority list in
# platforms/cuda.py was patched to include TRITON_MLA_SPARSE, but pin it
# explicitly so a backend-priority change upstream can't silently move us.
BACKEND=${BACKEND:-TRITON_MLA_SPARSE}

# Keep the compile cache local so restarts skip torch.compile warmup.
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-/home/user/vllm_install/.vllm_cache}

# 12 PP workers: leave OMP single-threaded or the workers thrash.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

# torch.compile shells out to `ninja`, which only exists in the venv. Calling
# $VENV/bin/vllm by absolute path does not put it on PATH, so add it here or
# CUDA graph capture dies with FileNotFoundError: 'ninja'.
export PATH="$VENV/bin:$PATH"

# flashinfer JIT-builds its sampling kernels with $CUDA_HOME/bin/nvcc on the
# first sampler call -- which happens inside profile_run, ~6 min into startup.
# sampling.cuh includes <curand.h>, and the CUDA 13.3 toolkit installed on
# 2026-08-11 carries the compiler but none of the math-library headers (the
# CUDA 12.4 apt toolkit used to supply them from /usr/include, which nvcc
# searches by default; that tree was moved to /var/backups/cuda-12.4-orphan-*).
# curand/cusparse/cusolver/cufft headers were copied in from the venv's
# nvidia-cu13 wheels; check here so a future toolkit swap fails in one second
# with a clear message instead of six minutes into a load with a ninja dump.
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
if [ ! -e "$CUDA_HOME/include/curand.h" ]; then
  echo "$CUDA_HOME/include/curand.h is missing: the CUDA toolkit has no math-" >&2
  echo "library headers, so flashinfer's sampling JIT will fail in profile_run." >&2
  echo "Fix: cp \$VENV/lib/python3.12/site-packages/nvidia/cu13/include/{curand,cusparse,cusolver,cufft}*.h $CUDA_HOME/include/" >&2
  echo "(or install the matching cuda-libraries-dev package)." >&2
  exit 1
fi

# MTP speculative decoding. The checkpoint carries a grafted MTP head at layer
# 78 (num_nextn_predict_layers=1) and vLLM maps model_type `glm_moe_dsa` ->
# DeepSeekMTPModel, so `method: mtp` reuses this checkpoint as its own draft.
#
# Getting this working under PP>1 needed three patches to the venv (all backed
# up in .venv/.../.glm52-backup/), because upstream never exercises MTP+PP:
#   1. deepseek_mtp.py: DeepSeekMTP now declares SupportsPP. The draft config
#      inherits the target's pipeline_parallel_size, so without the flag
#      verify_with_parallel_config rejects it outright. The declaration is
#      nominal (NemotronHMTP does the same) -- the MTP module is never split,
#      it is built only on the last PP rank.
#   2. gpu_model_runner.py: the drafter is created only on the last PP rank, but
#      initialize_kv_cache / cudagraph-dispatcher init / attn-metadata build /
#      _dummy_run all dereferenced it on every rank, gated only on
#      `speculative_config` -> AttributeError on non-last ranks. Guarded each with
#      `hasattr(self, "drafter")`, matching load_model's existing idiom. The
#      runtime drafting paths needed nothing: execute_model already returns
#      early on non-last ranks before reaching them.
#   3. deepseek_mtp.py load_weights: under PP the proposer cannot share the
#      target's embed_tokens (it lives on rank 0, the drafter on the last rank), and
#      the loader dropped the checkpoint's top-level embedding as a
#      non-spec-layer weight -- leaving the draft on uninitialised embeddings.
#      Now loaded explicitly when pp world_size > 1.
#
# The missing `shared_head.head` in the graft is fine: the proposer assigns the
# target lm_head into it (`_maybe_share_lm_head`), logged as "Detected MTP
# model. Sharing target model lm_head weights with the draft model."
#
# Verify correctness with mtp_verify.py: at temperature 0 speculation must be
# token-identical to SPEC_TOKENS=0. Set SPEC_TOKENS=0 to disable.
SPEC_ARGS=()
if [ "$SPEC_TOKENS" -gt 0 ]; then
  SPEC_ARGS=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$SPEC_TOKENS}")
fi

# Sync scheduling, as the gist uses. MTP+PP is made to work on this path by
# patch_mtp_pp.py. Do NOT switch to --async-scheduling with MTP: the PP+async
# token broadcast hard-asserts one sampled token per request
# ("PP+async expects sampled_token_ids to have shape [num_reqs, 1]"), which
# speculative decoding violates by construction.
ASYNC_SCHED_ARG=${ASYNC_SCHED_ARG:---no-async-scheduling}

# Loading the target and then the draft on the last rank can leave the allocator
# fragmented -- the OOM there reported 6.74 GiB "reserved but unallocated".
# Expandable segments let those reservations be reused instead of stranded.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# Optional fixed KV budget (see KV_CACHE_MEM note above).
KV_ARGS=()
if [ -n "${KV_CACHE_MEM:-}" ]; then
  KV_ARGS=(--kv-cache-memory "$KV_CACHE_MEM")
fi

# KV_CACHE_DTYPE: `auto` (bf16, 1152 B/token) or `fp8_ds_mla` (656 B/token:
# 512 e4m3 NoPE + 4 fp32 group scales + 64 bf16 RoPE, left unquantized).
# Needs patch_fp8_kv.py, which adds the sm_80 reader -- Triton cannot name
# fp8e4nv below sm_89, so the kernel decodes e4m3 out of raw bytes. The write
# side is stock vLLM and already worked on Ampere.
# On the PP=12 binding ranks (7 MLA layers + 2 indexer caches) this takes the
# total from 8,328 to 4,856 B/token, so the same KV_CACHE_MEM holds 1.71x the
# tokens. The indexer caches are already fp8 and do not shrink.
# DEFAULT fp8_ds_mla as of 2026-08-12. Needle 5/5 at 261,687 and 3/3 at
# 1,040,056 tokens; costs ~7% on prefill and 2-5% on decode, and buys 1.69x
# the KV tokens. At the estimated 20 GiB PP=12 default, `auto` holds 2,578,624
# slots while fp8_ds_mla holds 4,422,272 (before scheduler overhead).
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-fp8_ds_mla}

# GLM52_LMHEAD_BITS: runtime lm_head quantization to Marlin WxA16 g64 at
# load (patch_lmhead_quant.py). At k=3 MTP the shared lm_head is projected
# 4x/step (7.6 GB of bf16 reads on the last rank's critical path). DEFAULT 8
# (W8A16: 64.1 vs 65.9 ms/step, quality-lossless on greedy canaries).
# 4 saves ~1.3 ms more but flipped a greedy code token (plain-RTN INT4 is
# too coarse for lm_head); 0 disables.
export GLM52_LMHEAD_BITS=${GLM52_LMHEAD_BITS-8}

# MOE_BACKEND: force the WNA16 MoE kernel backend (marlin | humming |
# emulation). Default auto = marlin (first supported in priority order).
# humming's "indexed" gemm is the small-batch candidate; see
# glm52-decode-profiling memory note.
# KERNEL_CFG_JSON: extra --kernel-config entries (JSON object body without braces), e.g.
#   KERNEL_CFG_JSON='"ir_op_priority":{"rms_norm":["vllm_c"],"fused_add_rms_norm":["vllm_c"]}'
# routes RMSNorm to vLLM's CUDA kernels inside the compiled graph (Inductor's native
# RMSNorm re-benchmarks its reduction config per boot -> per-boot ULP lottery, see
# TYPO_INVESTIGATION.md 16.7).
# DEFAULT ON (2026-09-09): closes the per-boot output lottery -- two boots with it are
# bit-identical (run 37); set KERNEL_CFG_JSON="" to get Inductor's native RMSNorm back.
# GLM52_GATE_FP32=1 (DEFAULT ON, 2026-09-09): MoE router logits from a true fp32 GEMM. On
# sm_80 GateLinear falls through to a bf16 F.linear and casts the bf16-ROUNDED logits to fp32
# although GLM-5 requires fp32 routing; the rounding grid (~0.03 at logit 4) amplifies ULP-level
# perturbations into expert flips -> garbage tokens. Needle probe: 22/22 correct vs 6/22 without,
# throughput unchanged (patch_precision_knobs.py; weights pre-cast once at load).
export GLM52_GATE_FP32=${GLM52_GATE_FP32-1}
KERNEL_CFG_JSON=${KERNEL_CFG_JSON-'"ir_op_priority":{"rms_norm":["vllm_c"],"fused_add_rms_norm":["vllm_c"]}'}
KERNEL_ARGS=()
KC=""
[ -n "${MOE_BACKEND:-}" ] && KC="\"moe_backend\":\"$MOE_BACKEND\""
if [ -n "${KERNEL_CFG_JSON:-}" ]; then KC="${KC:+$KC,}$KERNEL_CFG_JSON"; fi
[ -n "$KC" ] && KERNEL_ARGS=(--kernel-config "{$KC}")

# EAGER=1 skips ALL CUDA graph capture (boots ~9 min faster; decode ~2x
# slower) -- use for crash-repro / debug restarts where perf is irrelevant.
# GRAPH_SIZES overrides cudagraph_capture_sizes (JSON list); default trims
# to <=64 (decode batches are tiny with the batch cap; oversize falls back
# to eager). GRAPH_SIZES=full restores the stock 33-size list.
# GRAPH_MODE overrides cudagraph_mode (e.g. PIECEWISE to keep partition
# graphs but run attention/custom ops eager -- bisection + fallback mode).
#
# DSA + FULL cudagraphs (2026-08-11): the indexer declares NEVER support
# (patch_dsa_no_fullcg.py), so the DEFAULT resolves to PIECEWISE -- FULL
# capture sizes the decode logits buffer at the capture dummy's seq len and
# silently truncates the sparse top-k beyond it (plus the now-clamped OOB
# -inf spray, see patch_mqa_store_clamp.py).
#   GLM52_DSA_FULLCG=1        hybrid: FULL graphs again, but a deterministic
#                             per-batch gate (patch_dsa_ctx_gate.py) drops to
#                             piecewise whenever any request's context exceeds
#                             GLM52_DSA_FULLCG_MAXLEN (default 2048 = topk,
#                             below which the decode top-k is exact by the
#                             identity shortcut). Benchmarked: hybrid == FULL
#                             at short ctx (46 single / 190 conc16), piecewise
#                             correctness beyond 2048.
#   plain default (guarded)   piecewise everywhere: 29.5 single / 132 conc8 /
#                             181 conc16.
# DEFAULT flipped to hybrid on 2026-08-12. Measured this session, same box,
# same build: piecewise 27.4 single / 121 conc16, hybrid 41.6 single / 117
# conc16 -- i.e. hybrid buys ~52% single-stream and costs nothing. It needs all
# three patches applied (patch_dsa_no_fullcg.py guard, patch_dsa_ctx_gate.py
# gate, patch_mqa_store_clamp.py clamp); without the clamp, FULL graphs at
# diverse conc-16 crash rank 1 within ~2 runs. Set GLM52_DSA_FULLCG=0 to fall
# back to piecewise everywhere if a graph-related corruption is ever suspected.
export GLM52_DSA_FULLCG=${GLM52_DSA_FULLCG:-1}

# GLM52_DSA_FULLCG_MAXLEN=0 (2026-08-12): the ctx gate above is now REDUNDANT
# and was costing 1.7x on long-context decode. Its whole purpose is to avoid a
# capture-sized decode logits buffer truncating the sparse top-k beyond the
# captured width -- but patch_dsa_fullwidth_capture.py already fixes that at the
# source, passing profile_seq_lens = max_model_len for EVERY full capture
# (_glm52_capture_seqlen in gpu_model_runner.py). With full-width capture the
# buffer is never undersized, so dropping to piecewise past 2048 buys nothing
# and just gives up cudagraphs on exactly the batches that need them.
# Measured this session, fp8 KV, same build, ms/step single-stream:
#   gate on (2048):  4K 112.9 | 32K 116.6 | 128K 124.7
#   gate off (0):    4K  66.9 | 32K  66.9 | 128K  74.8
# i.e. 1.74x at 32K, and it lands back on the tuning ledger (65.6 @ 64K,
# 75.0 @ 128K). Correctness re-validated: needle 5/5 at 261,057 tokens with
# FULL graphs active the whole way, no illegal-access or Xid.
# Set to 2048 to restore the conservative gate if graph corruption is suspected;
# it only matters when patch_dsa_fullwidth_capture.py is NOT applied.
export GLM52_DSA_FULLCG_MAXLEN=${GLM52_DSA_FULLCG_MAXLEN:-0}

# GLM52_TOPK_DET: decode-topk determinism mode (patch_topk_canon.py).
# The stock topk kernels break exact score ties nondeterministically (fp8
# indexer logits tie constantly); under MTP + PP pipelining the resulting
# selection lottery cascades into temp-0 token flips at long context (the
# 2026-08-13/14 copy-corruption investigation, NOTES_longctx_copy_fidelity.md).
#   canon  (DEFAULT) canonical tie-break kernel: (score desc, index asc),
#          ~0.24 ms worst-case per indexer layer call, capture-safe.
#   torch  stable torch.topk overwrite (validation baseline, ~2x slower)
#   ""     stock lottery behavior (debug only)
# Validated 2026-08-14: 0/8 vs 3/8 corrupted generations on the 100K
# overlap probe; quiet-repeat accept sequences bit-stable.
export GLM52_TOPK_DET=${GLM52_TOPK_DET-canon}

# GLM52_PREFILL_TOPK_DET: deterministic prefill topk selection
# (patch_prefill_topk_canon.py + patch_prefill_topk_relfix.py). canon =
# windowed canonical tie-break kernel, ~3.8ms per 2048x131072 call (~2%
# of a 120K prefill). Keeps KV builds reproducible; validated in the
# session-5 root-cause campaign. "" = stock lottery (debug only).
export GLM52_PREFILL_TOPK_DET=${GLM52_PREFILL_TOPK_DET-canon}

# GLM52_CAP_PREFILL_GUARD: the 08-15 mitigation (suspend decode splitting
# during prefills). Root cause fixed 08-16 (PP topk relay,
# patch_pp_topk_relay.py): 0/21 accuracy validated guard-OFF in both
# eager and cudagraph modes, so the guard is retired from production.
# Set 1 to re-arm if corruption is ever suspected again.
export GLM52_CAP_PREFILL_GUARD=${GLM52_CAP_PREFILL_GUARD-0}

# PP TOPK RELAY (patch_pp_topk_relay.py): THE session-5 root-cause fix.
# 'shared' (skip_topk) indexer layers at PP stage starts consumed the
# per-rank topk_indices_buffer holding the PREVIOUS batch's selections
# (another request's under co-flight + MTP) -> build-time cache
# poisoning -> the long-context copy corruption. The fix ships the last
# full-indexer selections across each PP hop via IntermediateTensors and
# seeds the receiving rank's buffer. Perf cost ~= 0 (measured).
# GLM52_PP_TOPK_RELAY=0 disables (DEBUG ONLY -- the bug returns).
export GLM52_PP_TOPK_RELAY=${GLM52_PP_TOPK_RELAY-1}

# GLM52_MOE_ALIGN_DET: deterministic moe_align token ordering
# (patch_moe_align_det.py). The CUDA op's atomic arrival order makes the
# fused Marlin MoE bitwise-nondeterministic per call -- the second lottery
# behind the long-context copy corruption (the first was topk tie-breaks).
# Verified: with this on, fused_marlin_moe is bit-stable across repeated
# calls, quiet and under transfer contention. Set 0 to restore the CUDA op.
# GLM52_SPLIT_MOE=1 restores the experimental per-row GEMV decode path.
# The later GLM-5.3 investigation measured ~40% lower throughput without
# fixing corruption. Both launchers default to fused Marlin; keep the
# router, norm, topk relay, and deterministic alignment fixes enabled.
export GLM52_SPLIT_MOE=${GLM52_SPLIT_MOE:-0}

export GLM52_MOE_ALIGN_DET=${GLM52_MOE_ALIGN_DET-1}

# GLM52_META_SNAPSHOT: metadata clone-at-build (patch_meta_snapshot.py).
# MUST STAY 0: cloning decode metadata breaks FULL cudagraph replays
# catastrophically (7/7 garbage) -- graphs receive fresh metadata only
# through the persistent-buffer aliasing the clone severs. Failed detour
# from the layer-3 hunt, kept only as a historical repro knob.
export GLM52_META_SNAPSHOT=${GLM52_META_SNAPSHOT-0}
# Re-compensate the AWQ smoothing fold on mlp.gate / indexer.wq_b (checkpoint
# defect, TYPO_INVESTIGATION.md §14.4). Set GLM52_ROUTER_FIX= (empty) to disable.
export GLM52_WEIGHT_HASH=${GLM52_WEIGHT_HASH-1}
export GLM52_ROUTER_FIX=${GLM52_ROUTER_FIX-/home/user/vllm_install/router_fix_glm53_s.pt}

EAGER_ARGS=()
if [ "${EAGER:-0}" = "1" ]; then
  EAGER_ARGS=(--enforce-eager)
else
  CC_JSON="{\"cudagraph_capture_sizes\":${GRAPH_SIZES:-[1,2,4,8,16,24,32,48,64]}"
  if [ "${GRAPH_SIZES:-}" = "full" ]; then
    CC_JSON="{"
  fi
  if [ -n "${GRAPH_MODE:-}" ]; then
    [ "$CC_JSON" != "{" ] && CC_JSON="$CC_JSON,"
    CC_JSON="$CC_JSON\"cudagraph_mode\":\"$GRAPH_MODE\""
  fi
  # INDUCTOR_CFG_JSON='{"deterministic": true}' merges extra torch._inductor config
  # (applied by vLLM inside the compile call; part of the compile-cache key).
  # CUSTOM_OPS_JSON='["all"]' selects vLLM CUDA custom ops inside the compiled graph
  # instead of Inductor-generated norm/activation kernels (compile-cache key changes).
  if [ -n "${CUSTOM_OPS_JSON:-}" ]; then
    [ "$CC_JSON" != "{" ] && CC_JSON="$CC_JSON,"
    CC_JSON="$CC_JSON\"custom_ops\":$CUSTOM_OPS_JSON"
  fi
  if [ -n "${INDUCTOR_CFG_JSON:-}" ]; then
    [ "$CC_JSON" != "{" ] && CC_JSON="$CC_JSON,"
    CC_JSON="$CC_JSON\"inductor_compile_config\":$INDUCTOR_CFG_JSON"
  fi
  CC_JSON="$CC_JSON}"
  if [ "$CC_JSON" != "{}" ]; then
    EAGER_ARGS=(--compilation-config "$CC_JSON")
  fi
fi

# PROFILER=torch mounts /start_profile + /stop_profile (per-rank chrome
# traces into PROFILER_DIR). PROFILER=cuda wraps steps in
# cudaProfilerStart/Stop for `nsys profile --capture-range=cudaProfilerApi`.
# (VLLM_TORCH_PROFILER_DIR no longer exists in 0.26 -- silently ignored.)
PROF_ARGS=()
if [ "${PROFILER:-}" = "torch" ]; then
  PROF_ARGS=(--profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${PROFILER_DIR:-/home/user/vllm_install/profiles}\",\"torch_profiler_with_stack\":false}")
elif [ -n "${PROFILER:-}" ]; then
  PROF_ARGS=(--profiler-config "{\"profiler\":\"$PROFILER\"}")
fi

# GLM52_PP_SPEC_GLOBAL_SER=1 restores the pre-patch-12 global batch
# serialization under PP+spec (lockstep batches, no cross-batch pipelining).
# Default off: per-request serialization is sufficient since the drafter RPC
# was retired (patch 11), and pipelining is what concurrency throughput needs.

# Diagnostic knobs cost real throughput and are easy to leave set in a shell:
# the tripwire is ~-15%, IDXVAL>=2 adds a sync per prefill-indexer call, and
# global spec serialization removes cross-batch pipelining entirely. Warn
# loudly rather than silently serving a slow config.
for _diag in GLM52_PP_TRIPWIRE GLM52_IDXVAL GLM52_PP_SPEC_GLOBAL_SER GLM52_PROF; do
  _val="${!_diag:-}"
  if [ -n "$_val" ] && [ "$_val" != "0" ]; then
    echo "WARNING: diagnostic $_diag=$_val is set -- expect reduced throughput." >&2
  fi
done

echo "config: PP=$PP TP=$TP world=$_world_size spec=$SPEC_TOKENS decode_batch_cap=$GLM52_PP_DECODE_BATCH_CAP" \
     "fullcg=$GLM52_DSA_FULLCG adaptive_cap=$GLM52_PP_DECODE_ADAPTIVE lmhead_bits=$GLM52_LMHEAD_BITS" \
     "max_len=${MAX_LEN:-409600} kv_bytes=${KV_CACHE_MEM:-unset}" \
     "kv_dtype=$KV_CACHE_DTYPE fullcg_maxlen=$GLM52_DSA_FULLCG_MAXLEN" \
     "relay=$GLM52_PP_TOPK_RELAY topk_det=$GLM52_TOPK_DET prefill_det=$GLM52_PREFILL_TOPK_DET" \
     "moe_align_det=$GLM52_MOE_ALIGN_DET guard=$GLM52_CAP_PREFILL_GUARD split_moe=$GLM52_SPLIT_MOE" \
     "router_fix=${GLM52_ROUTER_FIX:-off}" "kernel_cfg=${KC:-none}" "gate_fp32=$GLM52_GATE_FP32" \
     "partition=${VLLM_PP_LAYER_PARTITION:-auto}" >&2

# The new budget exceeds the old kernel's signed 32-bit byte addressing.
# Check the INSTALLED kernel, since editing the repository alone does not
# update site-packages. Check profiled/custom budgets too.
if [[ "$KV_CACHE_DTYPE" == fp8* ]]; then
  if ! "$VENV/bin/python" -c 'from vllm._glm52_mla_fp8 import KV_ADDRESS_BITS; assert KV_ADDRESS_BITS >= 64'; then
    echo "Install the updated FP8 kernel before serving: $VENV/bin/python $SCRIPT_DIR/patch_fp8_kv.py" >&2
    exit 1
  fi
fi

exec "$VENV/bin/vllm" serve "$MODEL" \
  ${SPEC_ARGS[@]+"${SPEC_ARGS[@]}"} \
  ${KV_ARGS[@]+"${KV_ARGS[@]}"} \
  ${PROF_ARGS[@]+"${PROF_ARGS[@]}"} \
  ${KERNEL_ARGS[@]+"${KERNEL_ARGS[@]}"} \
  ${EAGER_ARGS[@]+"${EAGER_ARGS[@]}"} \
  --served-model-name glm-5.3 \
  --pipeline-parallel-size "$PP" \
  --tensor-parallel-size "$TP" \
  --attention-backend "$BACKEND" \
  ${ASYNC_SCHED_ARG} \
  --max-model-len "${MAX_LEN:-409600}" \
  --gpu-memory-utilization "${GPU_UTIL:-0.93}" \
  --max-num-seqs "${MAX_SEQS:-32}" \
  --kv-cache-dtype "$KV_CACHE_DTYPE" \
  ${PREFIX_CACHE:+ } $( [ "${PREFIX_CACHE:-1}" = "0" ] && echo --no-enable-prefix-caching ) \
  --block-size "${BLOCK_SIZE:-64}" \
  --dtype "${DTYPE:-bfloat16}" \
  --trust-remote-code \
  --reasoning-parser "${REASONING_PARSER:-glm47}" \
  --enable-auto-tool-choice \
  --tool-call-parser "${TOOL_PARSER:-glm47}" \
  --host 0.0.0.0 \
  --port "$PORT"
