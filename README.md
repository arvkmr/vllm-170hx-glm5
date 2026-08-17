# vllm-170hx-glm5

Serving **GLM-5.2** (753B DeepSeek-Sparse-Attention MoE, AWQ INT4 g64) on **8× NVIDIA CMP 170HX**
(GA100, sm_80, 64 GB unlocked mining cards) with **vLLM 0.26.0**.

Patches, custom Triton kernels, benchmarks and serving scripts. Everything here targets a
configuration upstream does not support: DSA needs sparse-MLA kernels that only exist for
Hopper/Blackwell, and sm_80 has no fp8 hardware at all.

**Headline:** the full **1,048,576-token context** runs on this box, or 4 concurrent 262K
streams, at 66–80 ms/step single-stream decode.

---

## Hardware this targets

| | |
|---|---|
| GPUs | 8× CMP 170HX, GA100 sm_80, 70 SMs, 64 GB (unlocked) |
| Interconnect | PCIe gen2 x4 (~1.5 GB/s), no NVLink |
| Measured HBM read | 1.70 TB/s |
| Measured bf16 dense GEMM | 187 TFLOP/s (8192³) |

PCIe gen2 x4 is the binding constraint: **use PP=8, never TP.** A PP hop ships one activation
tensor per stage boundary (~0.1 ms); TP's per-layer all-reduces would dominate.

---

## Performance

All numbers measured on the above box, vLLM 0.26.0, MTP k=3, fp8 KV cache, all patches applied.
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

4×192K holds 786K KV tokens, which does not fit without the fp8 cache (bf16 caps at 604K here).

### KV cache capacity

The binding PP rank carries 11 MLA layers + 5 indexer caches:

| dtype | bytes/token | KV tokens @ 7.69 GiB/rank |
|---|---|---|
| bf16 | 13,332 | 604,032 |
| `fp8_ds_mla` | **7,876** | **1,048,576** |

fp8 costs ~7% on prefill and 2–5% on decode. Needle-in-haystack: **5/5 at 261,687 tokens**
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
| `patch_mtp_pp.py` | 15 patches making MTP speculative decoding work under PP=8 |
| `patch_fp8_kv.py` | `fp8_ds_mla` KV cache on sm_80 (+ skips a 3.5 GiB unreachable reserve so 1M fits) |
| `patch_idx_prefill_v2.py` | query-blocked DSA prefill logits kernel, 1.28x, bit-exact |
| `patch_mqa_v2.py` | rewritten DSA decode logits kernel, 2.8x |
| `patch_pp_adaptive_cap.py` | adaptive decode batch cap, +55% on long-context concurrency |
| `patch_dsa_fullwidth_capture.py` | full-width seq lens for FULL cudagraph capture (correctness) |
| `patch_dsa_ctx_gate.py` | context gate for FULL graphs — **redundant with the above; keep disabled** |
| `patch_mqa_store_clamp.py` | fixes an OOB `-inf` store in the sm80 paged MQA kernel under FULL capture |
| `patch_lmhead_quant.py` | runtime lm_head quantization to Marlin W8A16 |
| `patch_idx_prefill_buf.py` | caps the indexer prefill buffer (3.17 GB at 600K ctx OOMs capture) |

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

Full analyses, evidence, and upstream-ready framing:

- `upstream_writeup_pp_stale_indexer_buffer.md` — the PP boundary bug (mechanism,
  per-rank cachemap proof, fix design, validation: 0/21 vs 14/21 pre-fix)
- `upstream_writeup_topk_lottery.md` — the tie-break lottery
- `upstream_writeup_block_table_tails.md` — the stale row tails
- `upstream_writeup_attr_sniffed_mtp_buffer_sharing.md` — a trap discovered while
  fixing the above: vLLM's drafter loader attr-sniffs `topk_indices_buffer` on the
  target model and silently rebinds the drafter's buffer, halving MTP acceptance

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
```

Defaults are the validated configuration: PP=8, MTP k=3, fp8 KV, 262,144 context at 4.00x
concurrency. Override with env vars:

| env | default | |
|---|---|---|
| `MAX_LEN` | `262144` | `1048576` for the full 1M context (1.00x concurrency, ~10 min TTFT) |
| `KV_CACHE_DTYPE` | `fp8_ds_mla` | `auto` for the bf16 cache |
| `KV_CACHE_MEM` | `8258584576` | bytes/rank; this value gives exactly 1,048,576 KV tokens |
| `SPEC_TOKENS` | `3` | `0` disables MTP |
| `GLM52_PP_DECODE_BATCH_CAP` | `2` | decodes per scheduled batch |
| `GLM52_PP_DECODE_ADAPTIVE` | `8` | below this many decoders, drop the cap to 1; `0` disables |
| `GLM52_DSA_FULLCG_MAXLEN` | `0` | `2048` restores the (redundant) piecewise gate |

Every knob is echoed in a `config:` line at boot, and the script warns loudly if a diagnostic
env (`GLM52_PROF`, `GLM52_IDXVAL`, tripwire) is left set.

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
- **Concurrency tops out around 2x single-stream at 4 streams**, because 4 groups fill only half
  of the 8-stage pipeline.
- Paths in the scripts are absolute and specific to this box; edit before use.

## License

Patches are derived from vLLM (Apache-2.0) and carry the same license.
