# Router precision, RMSNorm autotune, and an AWQ fold defect: three sources of garbage tokens at long context

Context: GLM-5.3 (DSA MoE, AWQ W4 g64) on 8x sm_80 (CMP 170HX), PP=8, vLLM 0.26.0, torch 2.11,
`torch.compile` + piecewise cudagraphs. Symptom: rare garbage tokens in tool-call arguments and
documents at 60-120K context (a ' fg' where a number belongs, ' half' / ' incomplete' where only
a digit is plausible), worse under concurrency, and different from boot to boot. Everything
below was measured with a cache-hot greedy probe at a fixed 85K-token position (top-10
logprobs compared bit-for-bit against the run's own quiet baseline), with decode requests
co-scheduled ahead of the probe to reproduce the concurrency case.

## 1. `GateLinear` rounds the router logits to bf16 on every non-Hopper GPU

**Where.** `vllm/model_executor/layers/fused_moe/router/gate_linear.py`. All specialized tiers
(cuteDSL, DSV3 kernel, fp32 kernel, cuBLAS bf16-in/fp32-out) are gated on SM90/SM100. On any
other GPU the forward falls through to `ReplicatedLinear.forward` (a bf16 `F.linear`) and only
then casts to `out_dtype`. `_get_moe_router_dtype` in `deepseek_v2.py` requests float32 for
DeepSeek-V3-family / GLM-5 configs, so Hopper users get fp32 logits and Ampere/Ada users get
bf16-rounded logits cast to fp32, silently.

**Why it matters.** A bf16 ULP at a logit of ~4 is 0.03. Top-8 selection among 256 experts is
decided by differences far smaller than that for a few percent of tokens. On synthetic data
with GLM's shapes (hidden 6144, 256 experts), bf16 vs fp32 logits differ by 1.1e-3 mean /
1.4e-2 max and the top-8 SET differs for 4.6% of rows from the rounding alone. The rounding grid
also amplifies any ULP-level upstream perturbation (batch composition, boot-to-boot kernel
choice) by ~100x at rounding boundaries, which is exactly how tiny numerics became a routing
cascade (expert-set flips from layers 7-11, late-layer sparse-attention selections overlapping
only 15-38% between two ULP-perturbed runs, needle lost).

**Fix.** Compute the router GEMM in fp32 when `out_dtype` is float32 and no specialized kernel
is available. `GateLinear` already has `force_fp32_compute` (stores the weight in fp32 so the
fallback runs in fp32; `nemotron_h.py` sets it) -- `deepseek_v2.py` just never asks for it.
Here: `patch_precision_knobs.py`, `GLM52_GATE_FP32=1` (`x.float() @ W.float()`, weights
pre-cast once at load). Throughput unchanged (36-38 tok/s single stream).

**Evidence** (needle probe at 85K, one boot per arm, bit-reproducible boots, see section 2):

| arm | correct top token | garbage tokens |
|---|---|---|
| baseline | 6 / 22 | ' incomplete', ' half' |
| fp32 router | 22 / 22 (p = 1.000) | none |
| bf16 indexer query (fp8 -> bf16 scoring) | 4 / 22 | ' mid', ' half', ' trailed' |
| both | 21 / 22 | none |

Tails still differ per batch composition (all 12 co-flight probes are distinct states) -- that
is the rounding-level variation a quantized checkpoint will always have. What disappears is the
wrong top token.

## 2. Inductor re-benchmarks RMSNorm reductions per boot -> outputs differ from boot to boot

**Where.** With Inductor, vLLM lowers `rms_norm` / `fused_add_rms_norm` natively
(`CudaPlatform.get_default_ir_op_priority` returns `["native"]` whenever compiling; note that
`custom_ops=["all"]` does NOT change this). Inductor autotunes the generated row-reduction
kernels (R0_BLOCK in {1024, 2048, 4096, 8192} for a 6144-wide row) by timing, and a warm compile
cache does not stop it: in a two-boot experiment on a fresh cache root the second boot rewrote
28 of 196 `.best_config` files, all XBLOCK=1 reductions. A different R0_BLOCK is a different fp32
summation order.

**How it was localized.** An opaque `mutates_args` custom op (`_glm52_tap.py`, `patch_tap.py`)
inserted at 20 points of the layer code hashes intermediates of the *compiled* graph on real
steps. Layer 0 was bit-identical across two boots through the attention output, the o_proj
output and the fused residual sum; the first differing tensor was the normalized output of
`post_attention_layernorm`. Excluded on the way: uninitialized/OOB memory (compute-sanitizer
memcheck clean; poisoning every free allocator block with NaN and torch's deterministic NaN fill
change nothing), cuBLAS (identical across processes for all layer-0 shapes and alignments),
weights, KV/workspace buffers, VRAM. Eager boots were bit-identical all along.

**Fix / workaround.** Route the norms to vLLM's CUDA kernels inside the compiled graph:
`--kernel-config '{"ir_op_priority":{"rms_norm":["vllm_c"],"fused_add_rms_norm":["vllm_c"]}}'`
(serve scripts: `KERNEL_CFG_JSON`, default on). With it, two boots are bit-identical at all 168
tapped boundaries and in the probe, and the warm boot rewrites 0 autotune files. This is worth a
vLLM issue (the compile-cache key does not cover Inductor's runtime autotune; batch-invariant
mode should pin it) and possibly a PyTorch one (why a warm Inductor cache re-benchmarks).
Deterministic is not the same as correct: the frozen state is one draw of the same cascade;
section 1 is what removes the garbage.

## 3. AWQ smoothing fold left the router and indexer uncompensated

**Where.** The checkpoint's recipe folded per-channel smoothing scales s into
`post_attention_layernorm`, `q_a_layernorm` and `kv_a_layernorm` (LN' = LN/s) and multiplied the
smoothed consumers (experts, q_b_proj, ...) by s. Two consumers of the same norms were on the
ignore list and were left IDENTICAL to the original: `mlp.gate.weight` (75 layers) and
`indexer.wq_b.weight` (18 layers). They see x/s instead of x: routing agreement with the
original router was 8-57% per layer; with the fix 99.8-99.9%. Ignoring the gate is standard
practice for DeepSeek-family AWQ quants, so other checkpoints from the same flow may carry it --
check with the same test (gate bit-identical to the original while its feeding norm differs).

**Fix.** `make_router_fix.py --awq DIR --bf16 DIR --out s.pt` recovers s = LN_bf16 / LN_awq;
`patch_router_fix.py` + `GLM52_ROUTER_FIX=s.pt` multiplies the gate / wq_b input columns by s at
load. No throughput cost.

## 4. Batch composition (the concurrency case)

With the fused Marlin MoE, a decode row's result depends on the batch it shares (block size 8
vs 64 changes the K-reduction order), so the same request gives a different, deterministic
state for each set of co-scheduled requests. `patch_split_moe.py` (`GLM52_SPLIT_MOE=1`) routes
decode rows through a per-row GEMV and makes the state composition-invariant, at ~40% decode
throughput; it is wired but off by default because fixes 1 and 3 remove the garbage while
leaving only rounding-level variation.

## Tooling that made this tractable

- `patch_tap.py` / `_glm52_tap.py`: graph taps for compiled-graph fingerprints (needs a fresh
  `VLLM_CACHE_ROOT`: vLLM's compile cache key ignores model-file edits).
- `patch_step_trace.py` / `_glm52_steptrace.py`: per-step batch trace, weight fingerprints,
  allocator poison, deterministic fill, op-boundary hashes -- all env-gated and inert.
- `patch_mp_executable.py`: spawn workers through a wrapper interpreter, e.g.
  compute-sanitizer (`--target-processes all` does not reach vLLM's workers).
- `copyfid_glm52.py`: verbatim-copy fidelity vs context length under concurrency.
