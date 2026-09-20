# vllm-170hx-glm5

Serving **GLM-5.2 and GLM-5.3** (753B DeepSeek-Sparse-Attention MoE, AWQ INT4 g64) on
**12× NVIDIA CMP 170HX** (GA100, sm_80, 64 GB unlocked mining cards) with **vLLM 0.26.0**.

Patches, custom Triton kernels, benchmarks and serving scripts. Everything here targets a
configuration upstream does not support: DSA needs sparse-MLA kernels that only exist for
Hopper/Blackwell, and sm_80 has no fp8 hardware at all.

**Twelve-card starting profile:** 20 GiB/rank of fp8 cache is estimated to provide
**4,422,272 aggregate KV-token slots**, up from 2,653,376 with the previous 12 GiB budget
(**+66.7%**). That is 10.80× 409,600-token or 16.87× 262,144-token capacity before scheduler
reserves. The twelve-card profile has not been run on hardware yet; the original eight-card
setup demonstrated the full **1,048,576-token per-request context**.

---

## Hardware this targets

| | |
|---|---|
| GPUs | 12× CMP 170HX, GA100 sm_80, 70 SMs, 64 GB (unlocked) |
| Interconnect | PCIe gen2 x4 (~1.5 GB/s), no NVLink |
| Measured HBM read | 1.70 TB/s |
| Measured bf16 dense GEMM | 187 TFLOP/s (8192³) |

PCIe gen2 x4 is the binding constraint: **use PP=12, TP=1.** A PP hop ships one activation
tensor per stage boundary (~0.1 ms); TP's per-layer all-reduces would dominate.

The default layer partition is `6,7,7,7,7,7,7,7,7,6,6,4`. The final
rank is deliberately three target layers lighter than a regular heavy stage because it also
owns the complete MTP draft MoE, its private embedding, and the shared output head. Rank 0
ends before full-indexer layer 6, avoiding a four-indexer cache and long-prefill hotspot.
The same partition is retained with `SPEC_TOKENS=0` so turning speculation off does not
change the cache bottleneck. Topology and memory defaults live in `glm_profile.sh`.

---

## Performance

The numbers below are the measured **original 8-card baseline**, retained for regression
comparison. The 12-card defaults are topology- and memory-derived and should be remeasured on
the target host with `bench_matrix.py` before treating them as a performance claim. vLLM 0.26.0,
MTP k=3, fp8 KV cache, and all patches were used for the baseline.
Decode is reported as **ms/step** — tok/s swings with MTP acceptance (`min_tokens` forces
generation past EOS and the filler is trivially predictable, which inflates acceptance), so
step time is the stable metric.

### Single stream

| context | prefill tok/s | decode ms/step |
|---|---|---|
| 512 | — | 67 |
| 32K | 2,132 | 66 |
| 128K | 2,476 | 76 |
| 192K | 2,457 | 80 |
| 256K | 2,398 | 88 |
| 1015K | 1,707 (TTFT 604 s) | 164 |

Decode is nearly flat across a 380× context range — that is DSA's top-2048 sparse attention
working as intended. Prefill is *not* flat: at 1M the indexer's O(L²) term dominates and
per-token throughput roughly halves.

### Concurrent (aggregate decode tok/s)

| workload | KV used | tok/s |
|---|---|---|
| 4 × 64K | 260K | 99 |
| 4 × 192K | 786K | 71 |
| conc 8, 512 ctx | — | 154 |
| conc 16, 512 ctx | — | 203 |

On the original eight-card layout, 4×192K did not fit without the fp8 cache. The twelve-card
bf16 profile is estimated to provide 2,578,624 aggregate KV-token slots.

### KV cache capacity

The binding ranks in the twelve-stage MTP partition carry 7 MLA layers + 2 indexer caches:

| dtype | bytes/token on binding rank | 12 GiB/rank | 20 GiB/rank (default) |
|---|---|---|---|
| bf16 | 8,328 | 1,547,136 | 2,578,624 |
| `fp8_ds_mla` | **4,856** | **2,653,376** | **4,422,272** |

These are block-aligned estimates: `floor(bytes / (bytes_per_token × 64)) × 64`.
The final stage's MTP cache is included in the check and does not bind. vLLM takes the
smallest block count across PP ranks and shrinks the other allocations accordingly, so
unused memory on a light rank cannot compensate for a full heavy rank. See the
[v0.26.0 allocator](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/v1/core/kv_cache_utils.py).
The server's startup cache report is authoritative; null blocks, speculative lookahead,
partial blocks and generation need headroom beyond the prompt working set.

**Why 20 GiB:** the old tight middle stages owned ten MoE layers; the twelve-card split
owns at most seven. At approximately 5 GiB of resident weights per layer, that frees
about 15 GiB on those ranks. Increasing the original 7.69 GiB cache to 20 GiB uses
12.31 GiB of this estimate, leaving a margin for graph capture and long-prefill transients.
This is a sizing estimate from the old measurements, not an on-host memory profile.
Use `KV_CACHE_MEM=12884901888` to restore the conservative 12 GiB budget, or
`KV_CACHE_MEM="" GPU_UTIL=0.90` to profile automatically. A fixed budget bypasses vLLM's
KV memory sizing; changing `GPU_UTIL` alone does not shrink it.

**Required kernel update:** 4.42M slots put a single 656-byte MLA cache beyond 2 GiB.
The reader now widens slot IDs to int64 **before** stride multiplication. Both launchers
check `KV_ADDRESS_BITS` in the installed kernel, so a stale site-packages copy fails before
loading the model. Re-run `patch_fp8_kv.py` after copying these sources. The GPU test
`test_mla_fp8.py` exercises the reader and stock writer around the 2 GiB boundary and
at the final slot, using a compact reference instead of decoding the entire pool.

In the eight-card measurements, fp8 costs ~7% on prefill and 2–5% on decode. Needle-in-haystack: **5/5 at 261,687 tokens**
and **3/3 at 1,040,056 tokens**.

---

## Setup

1. vLLM **0.26.0** in a venv (`.venv/` here), PyTorch 2.11 + CUDA 13, Triton 3.6.
2. Hand-apply vLLM PR **#38476** (Triton sparse-MLA backend for sm_80). `patch(1)` mis-applies
   it against 0.26.0 even with `--fuzz=3` — it silently puts the `cuda.py` hunk in the SM100
   priority list instead of SM80, and 3 of 6 `sparse_attn_indexer.py` hunks fail. Apply by hand.
3. Run the patch scripts below. Each is idempotent, validates its anchors before writing
   anything, and backs the original up to `site-packages/.glm52-backup/`.
4. Edit the paths in `serve_glm52.sh` (model dir, venv) and `./start_glm52.sh`.

The PR alone is not sufficient on 0.26.0. Four further fixes are needed, all folded into the
scripts here: `VLLM_ATTENTION_BACKEND` no longer exists (use `--attention-backend`), the backend
must declare `get_supported_kernel_block_sizes() -> [64]`, the PR's metadata class is missing
fields the CUDA MLA layer requires, and `--dtype bfloat16` is mandatory despite the checkpoint
declaring float16.

### Patches

| script | what it does |
|---|---|
| `patch_mtp_pp.py` | 15 patches making MTP speculative decoding work under pipeline parallelism |
| `patch_fp8_kv.py` | `fp8_ds_mla` KV cache on sm_80 (+ skips a 3.5 GiB unreachable reserve so 1M fits) |
| `patch_idx_prefill_v2.py` | query-blocked DSA prefill logits kernel, 1.28x, bit-exact |
| `patch_mqa_v2.py` | rewritten DSA decode logits kernel, 2.8x |
| `patch_pp_adaptive_cap.py` | adaptive decode batch cap, +55% on long-context concurrency |
| `patch_dsa_fullwidth_capture.py` | full-width seq lens for FULL cudagraph capture (correctness) |
| `patch_dsa_ctx_gate.py` | context gate for FULL graphs — **redundant with the above; keep disabled** |
| `patch_mqa_store_clamp.py` | fixes an OOB `-inf` store in the sm80 paged MQA kernel under FULL capture |
| `patch_lmhead_quant.py` | runtime lm_head quantization to Marlin W8A16 |
| `patch_idx_prefill_buf.py` | caps the indexer prefill buffer (3.17 GB at 600K ctx OOMs capture) |
| `patch_precision_knobs.py` | **fp32 MoE router logits** (`GLM52_GATE_FP32=1`, default on) -- on non-Hopper GPUs the gate silently rounds routing logits to bf16; this was the garbage-token amplifier. Also `GLM52_IDX_Q_BF16` (bf16 indexer query; measured useless, off) |
| `patch_router_fix.py` + `make_router_fix.py` | re-compensates an AWQ smoothing fold the recipe left off the router gate and indexer `wq_b` (`GLM52_ROUTER_FIX=<s.pt>`, default on); the generator recovers the scales from the AWQ + bf16 checkpoints |
| `patch_split_moe.py` + `_glm52_moe_split.py` | batch-composition-invariant decode rows through the Marlin-packed GEMV (`GLM52_SPLIT_MOE=1`); wired, **off by default** (~40% decode cost) |
| `patch_step_trace.py` + `_glm52_steptrace.py` | env-gated diagnostics (step trace, weight fingerprints, allocator poison, deterministic fill, op-boundary hashes) and the fp32 gate weight pre-cast hook |
| `patch_tap.py` + `_glm52_tap.py` | graph taps: fingerprint intermediates of the *compiled* graph (`GLM52_TAP=<layers>`, needs a fresh `VLLM_CACHE_ROOT`) |
| `patch_mp_executable.py` | spawn workers through a wrapper interpreter (`GLM52_MP_EXECUTABLE`), e.g. compute-sanitizer |

### Correctness fixes (silent long-context corruption + MTP acceptance)

A multi-week investigation into rare, silent, plausible-token corruption at long
context (temp 0, verbatim-copy tasks, ~1-in-a-million tokens, concurrency- and
MTP-dependent) root-caused **three independent bugs**, each with its own patch
and writeup. None of them are sm80-specific — they apply to any DSA model with
shared indexer layers served under pipeline parallelism on stock vLLM.

| script | what it fixes |
|---|---|
| `patch_pp_topk_relay.py` | **The main one.** PP rank-boundary stale `topk_indices_buffer`: shared (skip-topk) indexer layers at the start of each PP stage attend with the *previous batch's* selections because the sharing full-indexer layer ran on the previous rank and the buffer is per-rank. Fix: relay the selections across the hop inside `IntermediateTensors` and seed the receiving rank's buffer. ≈0 cost. |
| `patch_topk_det.py` + `patch_topk_canon.py` + `glm52_topk_canon.py` | Decode-topk tie lottery: fp8 indexer logits tie constantly and the stock kernels break ties nondeterministically → temp-0 selection flips. `GLM52_TOPK_DET=canon` = canonical (score desc, index asc) tie-break kernel, ~0.24 ms/layer, capture-safe. |
| `patch_prefill_topk_canon.py` + `patch_prefill_topk_relfix.py` | Same lottery on the prefill path (makes KV builds bit-reproducible), windowed canon kernel ~3.8 ms per 2048×131072 call. |
| `patch_moe_align_det.py` | Deterministic `moe_align_block_size` (atomics arrival-order → run-to-run wobble). Measured ≈0 cost; kept as free hygiene. |
| `patch_bt_selfpad.py` | Block-table row tails keep the previous tenant's block ids; the indexer expand path copies full row width and can walk into another request's pages. Self-pads the tail. |
| `patch_reorder_decodes.py` | The DSA metadata builder never declares its decode-first reorder requirement, so spec-decode tokens can be misclassified into the prefill bucket. |
| `patch_precision_knobs.py` (`GLM52_GATE_FP32`) | **Router logits rounded to bf16 on every non-SM90 GPU** (`GateLinear` fallback). Needle probe at 85K under concurrency: 22/22 correct with fp32 routing vs 6/22 without, garbage tokens gone, throughput unchanged. |
| `KERNEL_CFG_JSON` (serve default) | Inductor re-benchmarks its RMSNorm reduction kernels per boot, so compiled boots differed at ULP level from layer 0 on. Routing the norms to vLLM's CUDA kernels (`ir_op_priority`) makes boots bit-identical. |
| `patch_router_fix.py` + `make_router_fix.py` | AWQ smoothing scales folded into the norms but never compensated on `mlp.gate` / `indexer.wq_b` (routing agreement with the original 8-57% -> 99.8%). |

Full analyses, evidence, and upstream-ready framing:

- `upstream_writeup_pp_stale_indexer_buffer.md` — the PP boundary bug (mechanism,
  per-rank cachemap proof, fix design, validation: 0/21 vs 14/21 pre-fix)
- `upstream_writeup_topk_lottery.md` — the tie-break lottery
- `upstream_writeup_block_table_tails.md` — the stale row tails
- `upstream_writeup_attr_sniffed_mtp_buffer_sharing.md` — a trap discovered while
  fixing the above: vLLM's drafter loader attr-sniffs `topk_indices_buffer` on the
  target model and silently rebinds the drafter's buffer, halving MTP acceptance
- `upstream_writeup_router_precision_and_boot_lottery.md` — the September 2026
  round: bf16-rounded router logits on non-Hopper GPUs (the garbage-token amplifier),
  per-boot Inductor RMSNorm autotune (boot-to-boot nondeterminism, how it was localized
  with graph taps), the AWQ fold defect, and batch-composition sensitivity

**Quick self-check for GLM-5.2 operators**: if you serve with `PP > 1`, look at
your stage boundaries. Full-indexer layers sit at 0,1,2 then every 4th layer
(2+4k); a stage that *starts on any other layer* has this bug — including the
default even split (78 layers / PP=2 splits at L39: affected). Symptoms are
subtle: occasional single-token "typos" in long verbatim reproduction, degraded
long-context quality under concurrent load, and depressed MTP acceptance — the
model just "feels stupider" than it should at long context. Also grep your boot
log for `"Sharing target model topk_indices_buffer"` — if present, your MTP
drafter has been hijacked by the attr-sniffing trap (see the fourth writeup).

### Custom kernels

| file | |
|---|---|
| `glm52_mla_fp8.py` | sparse MLA attention over the packed 656 B/token `fp8_ds_mla` cache |
| `glm52_idx_prefill_v2.py` | DSA prefill logits with BLOCK_M query blocking |
| `mqa_logits_v2.py` | DSA decode logits, batch-major persistent tiles |
| `glm52_moe_gemv_v2.py` | Marlin-packed MoE GEMV — **correct but 3-5x slower than Marlin, not wired in** |

---

## Usage

```bash
./start_glm52.sh          # detached launch + health wait; cold start ~13 min
./stop_glm52.sh           # kills workers by the pids nvidia-smi reports
./start_glm53.sh          # GLM-5.3: same stack, same knobs (serve_glm53.sh / stop_glm53.sh)
```

Defaults are the twelve-card topology profile: PP=12, TP=1, MTP k=3, fp8 KV, 409,600 context
at an estimated 10.80x cache capacity. The router/norm/topk correctness fixes remain enabled;
both launchers now default to fused Marlin (`GLM52_SPLIT_MOE=0`), matching the later GLM-5.3
findings. The new partition, cache size, kernel addressing and adaptive threshold require
on-host validation.
Override with env vars:

| env | default | |
|---|---|---|
| `PP` / `TP` | `12` / `1` | built-in PP=8 fallback; TP=1 required by this profile |
| `CUDA_VISIBLE_DEVICES` | `0,...,PP-1` | derives from PP; duplicate, empty and mismatched lists are rejected |
| `VLLM_PP_LAYER_PARTITION` | `6,7,7,7,7,7,7,7,7,6,6,4` | MTP/indexer-aware 78-layer split; validated for entry count and sum at launch |
| `MAX_LEN` | `409600` | `262144` gives an estimated 16.87x cache capacity; `1048576` enables the full per-request context |
| `KV_CACHE_DTYPE` | `fp8_ds_mla` | `auto` for the bf16 cache |
| `KV_CACHE_MEM` | `21474836480` | 20 GiB/rank estimate for the built-in PP=12 split; empty uses `GPU_UTIL`; custom partitions default to profiling |
| `SPEC_TOKENS` | `3` | `0` disables MTP |
| `GLM52_PP_DECODE_BATCH_CAP` | `2` | decodes per scheduled batch |
| `GLM52_PP_DECODE_ADAPTIVE` | `12` | below this many decoders, drop the cap to 1; defaults to `PP`, `0` disables |
| `GLM52_DSA_FULLCG_MAXLEN` | `0` | `2048` restores the (redundant) piecewise gate |
| `GLM52_GATE_FP32` | `1` | fp32 MoE router logits; `0` = stock bf16 fallback (garbage tokens at long context) |
| `GLM52_ROUTER_FIX` | `router_fix_glm5x_s.pt` | AWQ fold compensation for the router/indexer; build the file with `make_router_fix.py`; empty disables |
| `KERNEL_CFG_JSON` | `ir_op_priority rms_norm/fused_add_rms_norm -> vllm_c` | bit-identical boots; `""` = Inductor native norms (per-boot lottery) |
| `GLM52_SPLIT_MOE` | `0` | `1` = composition-invariant decode rows, ~40% slower decode |

Every knob is echoed in a `config:` line at boot, and the script warns loudly if a diagnostic
env (`GLM52_PROF`, `GLM52_IDXVAL`, tripwire) is left set.

### Validate the twelve-card profile on the host

Keep the updated source files together, including `glm_profile.sh`. In the serving venv:

```bash
python patch_fp8_kv.py       # installs the 64-bit reader; existing patch anchors are idempotent
python test_mla_fp8.py       # GPU numerical checks, including >2 GiB writer/reader addresses
python -m unittest -v test_launch_profiles  # CPU-only launcher and benchmark checks
./start_glm53.sh
python bench_glm52.py --model glm-5.3
python bench_matrix.py --model glm-5.3 --only conc
```

Check the boot log's KV capacity and each GPU's free memory, then run the existing
needle/copy-fidelity probes and the concurrent benchmark before treating 20 GiB as validated.
The default matrix skips its 1M case on a 409,600-token server. To test that case, restart
with `MAX_LEN=1048576` and pass `--max-model-len 1048576` to `bench_matrix.py`.
Both benchmarks accept `--base-url`; use `--model glm-5.2` for the other launcher. Request
failures now fail the benchmark rather than reporting throughput from a partial workload.

### Two settings that silently cost ~1.7x

Both were found by measurement, not by reading code, and both look fine in the logs:

- **`GLM52_PP_DECODE_BATCH_CAP` unset.** The scheduler patch defaults to `0` (off) inside
  site-packages, so forgetting to export it runs unpatched-equivalent: all ready decodes merge
  into one lockstep batch, pipeline depth 1, GPUs ~10% busy.
- **`GLM52_DSA_FULLCG_MAXLEN=2048`.** Forces PIECEWISE cudagraphs beyond 2048 tokens of
  context, costing 1.74x on long-context decode (117 → 67 ms/step at 32K). It guards against a
  capture-sized logits buffer truncating the sparse window — which `patch_dsa_fullwidth_capture.py`
  already fixes properly. Keep it at `0`.

---

## Benchmarks and tests

| script | |
|---|---|
| `bench_matrix.py` | prefill + decode matrix, single and concurrent |
| `needle_glm52.py` | needle-in-haystack accuracy and long-context decode |
| `conc_workload.py` | concurrency with diverse prompts and staggered starts |
| `bench_glm52.py` | same-prompt 512-token throughput |
| `test_mla_fp8.py` | fp8 MLA kernel vs an fp32 reference |
| `test_idx_prefill_v2.py` | prefill logits v2 vs v1, bit-exactness |
| `test_moe_gemv.py` | MoE GEMV vs `moe_wna16_marlin_gemm` on identical packed buffers |
| `idx_bench.py`, `idx_prefill_bench.py` | DSA indexer microbenchmarks |
| `copyfid_glm52.py` | verbatim-copy fidelity vs context length, placement and concurrency (temp 0) |
| `test_moe_split.py` | split-MoE GEMV path vs fused Marlin on identical rows |

**Benchmarking traps on this setup**, all of which produced wrong conclusions before being found:

- Never time prefill and decode in one pass for multi-stream long context. Prefills serialize,
  so stream 1's decode window contains every later stream's prefill. This made 4×192K look like
  4.4 tok/s when it is 71. `bench_matrix.py` uses a two-pass split.
- Compare **ms/step**, not tok/s. `min_tokens` pushes generation past EOS into trivially
  predictable filler, driving MTP acceptance toward 100%.
- The same-prompt harness (`bench_glm52.py`) overstates concurrency badly versus diverse
  prompts — identical greedy prompts decode in prefix-cache lockstep. Do not compare its numbers
  against `conc_workload.py`'s.
- The first sweep after a cold start reads ~2x low.

---

## Known limits

- **Prefill at 1M is ~10 minutes.** The DSA indexer is O(L²) and becomes ~37% of prefill at 1M.
  Treat 1M as a load-once batch mode, not interactive.
- **MoE is the remaining decode bottleneck.** Marlin WNA16 is ~42% of GPU kernel time and runs
  at 49% of the memory roof at conc=8 shapes. A perfect replacement is worth ~1.27x overall;
  the GEMV attempt here reached only 0.30x of Marlin and is not wired in.
- Low-concurrency workloads cannot fill a twelve-stage pipeline. At decode cap=2, use
  24 distinct decoding requests to provide 12 groups. `bench_matrix.py` defaults to
  24 × 128K (3.15M prompt tokens, within the estimated 4.42M fp8 pool); the adaptive
  cap uses one request per group below 12 decoders.
- Model, router-fix and cache paths are specific to the original box; override them before
  use elsewhere. `VLLM_INSTALL_DIR` and `VENV` relocate the serving runtime.

## License

Patches are derived from vLLM (Apache-2.0) and carry the same license.
