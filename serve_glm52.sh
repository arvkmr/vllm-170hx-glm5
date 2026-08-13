#!/usr/bin/env bash
# Serve GLM-5.2 (753B DSA MoE, AWQ-INT4 g64) across 8 GPUs with pipeline
# parallelism (PP=8).
#
# GLM-5.2 uses DeepSeek Sparse Attention. Upstream vLLM only has Hopper/Blackwell
# sparse-MLA backends (FLASHMLA_SPARSE) and DeepGEMM's fp8_mqa_logits, none of
# which build for sm_80. This box runs the TRITON_MLA_SPARSE backend from vLLM
# PR #38476, hand-applied into the venv's site-packages (see logs/ and the
# backup at .venv/lib/python3.12/site-packages/.glm52-backup).
set -euo pipefail

# Run from a neutral directory: launching from inside the installed vllm package
# makes vllm/tokenizers/ shadow the real `tokenizers` package on sys.path.
cd /home/user/vllm_install

VENV=/home/user/vllm_install/.venv
# On the NFS "fast" export, which reads at ~690 MB/s -- unlike
# the "share" export used for MiniMax, this one is quick enough to load from
# directly (~10 min for 390 GB). The mount is NOT restored on boot; remount
# before running if this path is empty.
MODEL=${MODEL:-/home/user/srv/fast/models/GLM-5.2-AWQ-g64}
PORT=${PORT:-8000}

# PP=8, not TP=8: the GPUs sit on PCIe gen2 x4, so TP's per-layer all-reduces
# would dominate. PP only ships one activation tensor per stage boundary.
PP=${PP:-8}
TP=${TP:-1}

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
export GLM52_PP_DECODE_ADAPTIVE=${GLM52_PP_DECODE_ADAPTIVE:-8}

# KV budget: a fixed --kv-cache-memory per rank (DEFAULT, 5.5 GiB) beats the
# GPU_UTIL fraction, which must clear the free-memory check on EVERY rank
# while per-rank slack differs hugely (ranks 1-6 had 7.58 GiB available at
# full utilization vs the 3.8 GiB the 0.93 fraction granted). 5.5 GiB leaves
# ~2 GiB headroom on the tight ranks for long-context indexer transients and
# yields 442,943 KV tokens (2.16x concurrency at the full 200K context).
# Set KV_CACHE_MEM="" to fall back to the GPU_UTIL fraction; if you do:
# a *fresh* card reports only ~59.6 GiB free of 63.39 (CUDA context + driver
# reserve), so ~0.93 is the practical maximum for GPU_UTIL.
# DEFAULT raised to 7.5 GiB on 2026-08-12: 604,032 KV tokens (was 442,943 at
# 5.5 GiB), 2.30x concurrency at the full 262,144 context. Validated by a 5/5
# needle sweep at 261,687 tokens with no OOM. Ranks 1-6 are the binding stages
# and had ~5.4 GiB free above this reservation when idle, so the old ~2 GiB
# pad for long-context indexer transients is now closer to ~1.5 GiB -- if a
# heavy concurrent long-prefill workload ever OOMs a middle rank, this is the
# first number to walk back.
# DEFAULT raised again 2026-08-12 to 7.69 GiB, together with the fp8 KV cache
# below: 7,876 B/token x 64 x 16,384 blocks = exactly **1,048,576 KV tokens**,
# i.e. 4.00x concurrency at the 262,144 default context, or the model's full
# 1M context in a single stream if you set MAX_LEN=1048576. Booting at 1M also
# needs patch_fp8_kv.py edit 5 (it skips a 3.5 GiB profile-run reserve for a
# dense-MHA prefill path this backend can never reach; without it 1M OOMs by
# ~20 MB).
KV_CACHE_MEM=${KV_CACHE_MEM-8258584576}

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

# 8 PP workers on 10 cores: leave OMP single-threaded or the workers thrash.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

# torch.compile shells out to `ninja`, which only exists in the venv. Calling
# $VENV/bin/vllm by absolute path does not put it on PATH, so add it here or
# CUDA graph capture dies with FileNotFoundError: 'ninja'.
export PATH="$VENV/bin:$PATH"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

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
# Getting this working under PP=8 needed three patches to the venv (all backed
# up in .venv/.../.glm52-backup/), because upstream never exercises MTP+PP:
#   1. deepseek_mtp.py: DeepSeekMTP now declares SupportsPP. The draft config
#      inherits the target's pipeline_parallel_size, so without the flag
#      verify_with_parallel_config rejects it outright. The declaration is
#      nominal (NemotronHMTP does the same) -- the MTP module is never split,
#      it is built only on the last PP rank.
#   2. gpu_model_runner.py: the drafter is created only on the last PP rank, but
#      initialize_kv_cache / cudagraph-dispatcher init / attn-metadata build /
#      _dummy_run all dereferenced it on every rank, gated only on
#      `speculative_config` -> AttributeError on ranks 0-6. Guarded each with
#      `hasattr(self, "drafter")`, matching load_model's existing idiom. The
#      runtime drafting paths needed nothing: execute_model already returns
#      early on non-last ranks before reaching them.
#   3. deepseek_mtp.py load_weights: under PP the proposer cannot share the
#      target's embed_tokens (it lives on rank 0, the drafter on rank 7), and
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
SPEC_TOKENS=${SPEC_TOKENS:-3}
SPEC_ARGS=()
if [ "$SPEC_TOKENS" -gt 0 ]; then
  SPEC_ARGS=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$SPEC_TOKENS}")

  # The drafter lives entirely on the LAST PP rank, so rank 7 pays for its own
  # target layers *plus* the whole MTP module (a 256-expert MoE layer ~6 GiB and
  # a 1.9 GiB embedding table, since under PP it cannot share the target's).
  # With the default even split [9,10,10,10,10,10,10,9] that rank OOMs while
  # loading the draft. Give it two fewer layers and hand them to rank 0, which
  # is the lightest stage anyway -- its first 3 layers are dense, not MoE.
  # Sums to 78 (num_hidden_layers).
  export VLLM_PP_LAYER_PARTITION=${VLLM_PP_LAYER_PARTITION:-11,10,10,10,10,10,10,7}
fi

# Sync scheduling, as the gist uses. MTP+PP is made to work on this path by
# patch_mtp_pp.py. Do NOT switch to --async-scheduling with MTP: the PP+async
# token broadcast hard-asserts one sampled token per request
# ("PP+async expects sampled_token_ids to have shape [num_reqs, 1]"), which
# speculative decoding violates by construction.
ASYNC_SCHED_ARG=${ASYNC_SCHED_ARG:---no-async-scheduling}

# Loading the target and then the draft on rank 7 leaves the allocator badly
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
# On the binding rank (11 MLA layers + 5 indexer caches) this takes the total
# from 13,332 to 7,876 B/token, so the same KV_CACHE_MEM holds 1.69x the
# tokens. The indexer caches are already fp8 and do not shrink.
# DEFAULT fp8_ds_mla as of 2026-08-12. Needle 5/5 at 261,687 and 3/3 at
# 1,040,056 tokens; costs ~7% on prefill and 2-5% on decode, and buys 1.69x
# the KV tokens. Set KV_CACHE_DTYPE=auto for the bf16 cache (604,032 tokens
# at the KV_CACHE_MEM above, so 2.30x at 262K instead of 4.00x).
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-fp8_ds_mla}

# GLM52_LMHEAD_BITS: runtime lm_head quantization to Marlin WxA16 g64 at
# load (patch_lmhead_quant.py). At k=3 MTP the shared lm_head is projected
# 4x/step (7.6 GB of bf16 reads on rank 7's critical path). DEFAULT 8
# (W8A16: 64.1 vs 65.9 ms/step, quality-lossless on greedy canaries).
# 4 saves ~1.3 ms more but flipped a greedy code token (plain-RTN INT4 is
# too coarse for lm_head); 0 disables.
export GLM52_LMHEAD_BITS=${GLM52_LMHEAD_BITS-8}

# MOE_BACKEND: force the WNA16 MoE kernel backend (marlin | humming |
# emulation). Default auto = marlin (first supported in priority order).
# humming's "indexed" gemm is the small-batch candidate; see
# glm52-decode-profiling memory note.
KERNEL_ARGS=()
if [ -n "${MOE_BACKEND:-}" ]; then
  KERNEL_ARGS=(--kernel-config "{\"moe_backend\":\"$MOE_BACKEND\"}")
fi

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

echo "config: PP=$PP spec=$SPEC_TOKENS decode_batch_cap=$GLM52_PP_DECODE_BATCH_CAP" \
     "fullcg=$GLM52_DSA_FULLCG adaptive_cap=$GLM52_PP_DECODE_ADAPTIVE lmhead_bits=$GLM52_LMHEAD_BITS" \
     "max_len=${MAX_LEN:-262144} kv_bytes=${KV_CACHE_MEM:-unset}" \
     "kv_dtype=$KV_CACHE_DTYPE fullcg_maxlen=$GLM52_DSA_FULLCG_MAXLEN" >&2

exec "$VENV/bin/vllm" serve "$MODEL" \
  "${SPEC_ARGS[@]}" \
  "${KV_ARGS[@]}" \
  "${PROF_ARGS[@]}" \
  "${KERNEL_ARGS[@]}" \
  "${EAGER_ARGS[@]}" \
  --served-model-name glm-5.2 \
  --pipeline-parallel-size "$PP" \
  --tensor-parallel-size "$TP" \
  --attention-backend "$BACKEND" \
  ${ASYNC_SCHED_ARG} \
  --max-model-len "${MAX_LEN:-262144}" \
  --gpu-memory-utilization "${GPU_UTIL:-0.93}" \
  --max-num-seqs "${MAX_SEQS:-32}" \
  --kv-cache-dtype "$KV_CACHE_DTYPE" \
  --block-size "${BLOCK_SIZE:-64}" \
  --dtype "${DTYPE:-bfloat16}" \
  --trust-remote-code \
  --reasoning-parser "${REASONING_PARSER:-glm47}" \
  --enable-auto-tool-choice \
  --tool-call-parser "${TOOL_PARSER:-glm47}" \
  --host 0.0.0.0 \
  --port "$PORT"
