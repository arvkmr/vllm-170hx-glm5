# GLM-5.3 + DFlash2 on 8-10x CMP 170HX (sm_80), vLLM 0.30

Serving recipe for the 78-layer GLM-5.3 DSA MoE (`GLM-5.3-Int4-Int8Mix-AWQ-g64`)
with the `incoai/GLM-5.3-DFlash2` speculative drafter on ten 64-GiB CMP 170HX
cards (Ampere, sm_80, PCIe), pipeline parallel 8, 9 or 10 (`PP_SIZE`).

It pins the Morrowmake CMP 170HX vLLM fork (`versions.env`) and applies a small,
fail-closed set of source patches (`apply_engine_patch.py`) that the fork
lacks for this target on Ampere.

| | |
|---|---|
| Engine | `Morrowmake/vllm-cmp170hx` @ `ampere-glm53` (vLLM 0.30.1rc1), pinned commit |
| Torch | 2.13.0 + CUDA 13.0 |
| Parallelism | PP=8 (default), 9 or 10, TP=1; see [Pipeline depth](#pipeline-depth) |
| KV cache | packed `fp8_ds_mla` (656 B/token/layer) + fp8 indexer keys, block 128 |
| Context | 1,048,576 tokens per request at every PP size; KV pool depends on PP (below) |
| Speculation | DFlash2, k=7, ~3.3-3.9 accepted tokens/step |
| Decode (+250 MHz VF, 1350 MHz ceiling)| ~81-83 ms/step, ~41-49 tok/s single stream (FULL decode graphs); 174 tok/s aggregate at 8 streams |
| Prefill | ~1.8-2.0K tok/s for one 8K prompt, ~2.4K tok/s at 32K or 8 concurrent |

Underclocking leads to 3-5% lost performance with 10-20% power savings

## Performance


Measured 2026-09-29 on the `agent` profile (10 cards; the core profile
applies +250 MHz VF offset and a 1350 MHz SM ceiling, with 1300 MHz on the one
marginal card; 170 W power cap; SM clocks sampled under load at 1350 / 1290
MHz). C8 needs 8 seats, so every row was measured with `MAX_SEQS=8`, which
leaves 1,824,985 KV tokens.

Decode: C concurrent greedy streams, 512 tokens each (`ignore_eos`), a
different prompt per stream (prose, code, technical). Prefill: C concurrent
requests of ~7.8K unique prompt tokens (no prefix-cache hits), `max_tokens=1`.

| Concurrency | Per-stream decode | Aggregate decode | Spec tokens/step | Prefill TTFT mean / max | Aggregate prefill |
|---|---|---|---|---|---|
| C1 | 48.7 tok/s | 48.0 tok/s | 3.92 | 4.3 s / 4.3 s | 1,833 tok/s |
| C4 | 32.6 tok/s (min 27.8) | 108.4 tok/s | 3.46 | 9.9 s / 14.1 s | 2,225 tok/s |
| C8 | 27.1 tok/s (min 22.4) | 174.1 tok/s | 3.31 | 16.2 s / 26.6 s | 2,351 tok/s |

Previous table (2026-09-28, before the decode kernel work below): C1 39.7,
C4 29.0 per stream / 102.1 aggregate, C8 22.5 / 152.2 tok/s; prefill
unchanged within noise.

Per-stream tok/s from a single prompt swings with DFlash2 acceptance (3.55
vs 3.92 tokens/step above for the same prompt, because any numerical change
alters the greedy text). For A/B work use `accept_bench.py`, which runs all
eight prompts for 384 tokens each and reports ms/step, the kernel-side cost:

| Build (`MAX_SEQS=8`) | ms/step | tokens/step | tok/s |
|---|---|---|---|
| Decode kernel work off (`GLM52_MLA_WIDE=0 GLM52_MLA_HEAD_BMM=0 VLLM_GLM5_DECODE_KERNELS=0 VLLM_GLM5_SHARED_EXPERT_REORDER=0`) | 92.0 | 3.26 | 35.4 |
| Default | **82.7** | 3.41 | **41.3** |

A single 31K-token prompt prefills in 13.1 s (2,372 tok/s). Prefill is
chunked at `max_num_batched_tokens=512` through a 10-stage pipeline, so it
gains only ~25-35% from concurrency. Decode throughput scales with
concurrency. Speculative acceptance dips slightly as streams are added.

Reproduce with `bench_conc.py` against a running server:

```bash
MAX_SEQS=8 ./start.sh
python3 bench_conc.py 8000 decode 8 512     # C8 decode
python3 bench_conc.py 8000 prefill 8 8192   # C8 prefill
```

```bash
python3 accept_bench.py 8000 384            # 8-prompt ms/step + acceptance
```

## What the patches do

`apply_engine_patch.py` refuses any other engine commit, applies every edit by
exact anchor, parses each modified file, and is idempotent. `preflight.py`
checks the markers before launch.

- **Packed FP8 MLA reader** (`glm52_mla_fp8.py`): the fork's Triton sparse-MLA
  backend gains an `fp8_ds_mla` path that decodes e4m3 bytes in software.
  Ampere Triton cannot name `tl.float8e4nv`.
- **Software FP8 in the DSA kernels**: the fused norm/RoPE and fused-Q kernels
  write e4m3 bit patterns (round-to-nearest-even, saturating; equal to the
  hardware cast for all 2^32 float32 inputs) into real uint8 buffers. A uint8
  *view* of a float8 tensor hid the write from `torch.compile` and produced
  stray token-0 outputs under graphs.
- **Indexer on Ampere**: the DSA sparse indexer uses the fork's Triton
  fp8 MQA-logits kernels (DeepGEMM is SM90+), with byte-packed decode Q and
  weights packed to the same rows.
- **Canonical top-k** (`glm52_topk_canon.py`, `GLM53_TOPK_CANON=1`): native
  value-exact selection, then an out-of-place canonical rebuild (score desc,
  index asc) for decode and prefill. The fork's `VLLM_GLM5_TOPK_TIEFIX` /
  `SORTED` post-processing corrupted decode on this target (decode-vs-prefill
  logprob gaps up to 13 nats, stray digit tokens at temperature > 0); its own
  canonical path is a torch sort over the full `max_model_len` row (+48 ms/step).
- **DFlash over PP for the DSA model**: aux hidden states are relayed across
  stages with the fork's `EagleModelMixin` layout, and the last stage keeps
  the embedding the drafter aliases.
- **Window-bounded drafter KV**: each of the drafter's 6 sliding-window layers
  co-owns one MLA tensor on the last stage at disjoint block ids, its 16-token
  BF16 page padded to the 128-token MLA page. Without it the drafter is
  allocated full-length and caps KV at ~0.96M tokens. Requires `--block-size 128`.
- **Top-k PP relay** (`local-cmp170hx-dsv32-topk-pp-relay`): a skip-topk
  layer reuses the selections of the last index-producing layer. When a stage
  starts on a skip-topk layer, the upstream stage sends its
  `topk_indices_buffer` rows (int32, `[tokens, 2048]`, cloned) with the hidden
  state, and this stage seeds its buffer from them. PP=8 and PP=9 need it,
  because 78 layers cannot be split into 8 or 9 producer-aligned stages
  without a ~60 GiB stage.
- **Optional W8 lm_head** (`glm52_lmhead_quant.py`, `GLM52_LMHEAD_BITS=8`):
  off by default (~0.5 ms/step gain, small top-1 changes).

### Decode kernel work (2026-09-29)

From a per-rank critical-path profile of a C1 step. Each has a kill switch;
none reduces precision (all fp32 accumulation, pinned configs, bitwise
reproducible boot to boot).

| Change | Switch | Saved |
|---|---|---|
| Wide-head sparse-MLA decode launch: 64 heads per program instead of 16, so each topk row is gathered and e4m3-decoded once, not 4x; splits by row count (3-32 rows). Kernel 127-181 -> 85-92 us at 8 rows; error vs fp32 identical | `GLM52_MLA_WIDE` | ~5-6 ms |
| Fork's fused sm_80 MoE router (gate GEMV + sigmoid top-k + Marlin alignment, one launch, ~11 us vs ~46 us per layer), gate widened from GLM-5.3-Flash's 288x4096 to this 256x6144 router. Expert ids and the Marlin alignment identical to the unfused chain on real router weights; logits differ by fp32 rounding only | `VLLM_GLM5_DECODE_KERNELS`, `VLLM_GLM5_DECODE_MOE_ROUTE_V2` | ~2-3 ms |
| Shared experts enqueued on the aux stream after the routed experts, so they overlap (fork flag; enqueue order only) | `VLLM_GLM5_SHARED_EXPERT_REORDER` | ~1 ms |
| Per-head Triton kernel for MLA's absorbed W_UK / W_UV bmms (`glm52_mla_bmm.py`), weights made contiguous at load; 1.3-1.45 TB/s vs cuBLAS 0.8-0.9 | `GLM52_MLA_HEAD_BMM` | ~0.6 ms |

`serve.sh` keeps the MoE padding mask at every batch size with the decode
kernels on (`VLLM_GLM5_DECODE_MOE_MAX_TOKENS=0`), and leaves the fork's
thin-M BF16 GEMM off (`VLLM_GLM5_THIN_GEMM`): ~0.75 ms/step, but DFlash2
acceptance measured 4% lower with it.

Measured and not worth it on this host: TP=2 x PP=5 (NCCL all-reduce of one
decode hidden state is ~100 us on Gen2 x4, ~4.5% net, and it crashed); plain
PP=8 (stages must start on index-producing layers, which forces 12-layer
stages that do not fit); an HBM overclock (NDIV70 corrupts output, no ECC);
a Triton W8A16 GEMM for the dense INT8 layers (slower than Marlin, which is
within ~35% of the read floor at these sizes). The MoE experts (~56% of the
step) already stream at the HBM bandwidth limit.

## Install

```bash
./install.sh                                  # builds ~/vllm_glm53_dflash2 (engine, venv)
DFLASH_LICENSE_ACK=1 ./download_drafter.sh    # drafter is CC-BY-NC-ND-4.0
```

`install.sh` requires a CUDA 13 toolkit at `$CUDA_HOME` (default
`/usr/local/cuda`), clones the pinned fork, installs the matching precompiled
wheel and applies the patches. Nothing outside `~/vllm_glm53_dflash2` changes.

## Run

```bash
./start.sh                    # production: agent profile on 127.0.0.1:8002, background
                              # logs/serve.log, logs/vllm.pid
./smoke.sh                    # health + one chat request
./stop.sh                     # stops only the recorded process group
PROFILE=agent ./serve.sh      # foreground equivalent
PP_SIZE=10 ./start.sh         # pipeline depth: 8 (default), 9 or 10
```

### Pipeline depth

| `PP_SIZE` | GPUs (`nvidia-smi` index) | Partition | `GPU_UTIL` | KV pool (tokens) |
|---|---|---|---|---|
| 8 (default) | 0-5, 7, 8 | `11,10,10,10,10,10,9,8` | 0.96 | 1,025,573 |
| 9 | 0-8 | `10,9,9,9,9,9,9,8,6` | 0.93 | 1,600,895 |
| 10 | 0-9 | `10,8,8,8,8,8,8,8,8,4` | 0.93 | ~2.09M |

- `serve.sh` sets `CUDA_DEVICE_ORDER=PCI_BUS_ID`, so the indices match
  `nvidia-smi`. PP=8 leaves GPU 6 free. Pass `CUDA_VISIBLE_DEVICES` to choose
  other cards.
- PP=8 and PP=9 have stages that start on skip-topk layers. They need the
  top-k PP relay. Preflight checks for its marker and refuses those
  partitions without it.
- `CUDA_VISIBLE_DEVICES`, `VLLM_PP_LAYER_PARTITION`, `GPU_UTIL` and `MAX_LEN`
  all override the defaults. Any other `PP_SIZE` (e.g. `PP_SIZE=5 TP_SIZE=2`)
  must set `VLLM_PP_LAYER_PARTITION`.
- Preflight checks only the selected cards, so `--require-idle` ignores
  work on cards outside the set.

| Profile | Context | Seqs | Batched tokens | Graphs | Bind |
|---|---|---|---|---|---|
| `smoke` (`serve.sh` default) | 32K | 1 | 1024 | eager | 127.0.0.1:8001 |
| `agent` (`start.sh` default) | 1,048,576 | 4 | 512 | FULL + PIECEWISE | 127.0.0.1:8002 |
| `production` | 1,048,576 | 8 | 2048 | FULL + PIECEWISE | 127.0.0.1:8001 |

Useful switches (environment): `PP_SIZE`, `TP_SIZE`, `MODEL`, `DFLASH_MODEL`, `MAX_LEN`, `MAX_SEQS`,
`MAX_BATCHED`, `EAGER=1`, `COMPILATION_CONFIG` (JSON), `SPEC_TOKENS`,
`NO_SPEC=1`, `ENABLE_FORK_PP_OPT`, `ASYNC_SCHED=0` (A/B only),
`PIN_NORMS=0`, `ALLOW_BUSY_GPUS=1` (skip the idle-GPU preflight).

Operational notes baked into `serve.sh`:

- Async scheduling is required: with `--no-async-scheduling`, PP + DFlash2
  asserts on the first short prompt.
- `--block-size 128` is required for the window-bounded drafter KV.
- Each running request reserves the drafter window plus
  10 stages x `max_num_batched_tokens` of drafter blocks, so the agent profile
  uses 512 batched tokens.
- RMSNorm is pinned to vLLM's CUDA kernels so compiled outputs are identical
  from boot to boot.

## Validate

```bash
python3 -m unittest -v test_vllm_next.py                          # static, no GPU
CUDA_VISIBLE_DEVICES=0 ~/vllm_glm53_dflash2/venv/bin/python test_fp8_kernel.py
CUDA_VISIBLE_DEVICES=0 ~/vllm_glm53_dflash2/venv/bin/python test_dsa_sm80.py
CUDA_VISIBLE_DEVICES=0 ~/vllm_glm53_dflash2/venv/bin/python test_topk_canon_sm80.py
./smoke_reduced.sh            # 8-layer dummy-weight model on 1-2 GPUs (fast engine smoke)
python3 probe_bang.py 8000 30 # 30 greedy repeats: must be identical, no token 0
```

`probe_bang.py` reports every generated token id 0 (`!`) with its logprob. A
logprob of -11.95 (= -ln(vocab size)) means an all-zero hidden row reached the
LM head: silent corruption, not a model choice.

## Faulty GPUs

One card was found to compute all-zero MoE output
rows (about 1 per 3,000 decode steps) and occasional garbage top-k indices at
its default clock under these kernels, while passing plain GEMM and memory
tests (`gpu_consistency.py`). To find such a card:

1. Run `probe_bang.py`; injected tokens with logprob -11.95 indicate the fault.
2. Instrument or bisect by swapping two devices in `CUDA_VISIBLE_DEVICES`; the
   fault follows the physical card, not the pipeline stage.
3. Cap that card's SM clock with `gpu-clock-cap.service` (1300 MHz removed the
   fault entirely at no measurable decode cost), or replace it.
