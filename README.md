# GLM-5.3 + DFlash2 on 10x CMP 170HX (sm_80), vLLM 0.30

Serving recipe for the 78-layer GLM-5.3 DSA MoE (`GLM-5.3-Int4-Int8Mix-AWQ-g64`)
with the `incoai/GLM-5.3-DFlash2` speculative drafter on ten 64-GiB CMP 170HX
cards (Ampere, sm_80, PCIe), pipeline parallel 10.

It pins the Morrowmake CMP 170HX vLLM fork (`versions.env`) and applies a small,
fail-closed set of source patches (`apply_engine_patch.py`) that the fork
lacks for this target on Ampere.

| | |
|---|---|
| Engine | `Morrowmake/vllm-cmp170hx` @ `ampere-glm53` (vLLM 0.30.1rc1), pinned commit |
| Torch | 2.13.0 + CUDA 13.0 |
| Parallelism | PP=10, TP=1, partition `10,8,8,8,8,8,8,8,8,4` |
| KV cache | packed `fp8_ds_mla` (656 B/token/layer) + fp8 indexer keys, block 128 |
| Context | 1,048,576 tokens; ~1.87M tokens of KV reported (2.09M pool) |
| Speculation | DFlash2, k=7, ~3.5-3.9 accepted tokens/step |
| Decode (default clocks)| ~85-89 ms/step, ~41-44 tok/s single stream (FULL decode graphs) |
| Decode (+250 MHz VF)| ~90-94 ms/step, ~39-41 tok/s single stream (FULL decode graphs); 152 tok/s aggregate at 8 streams |
| Prefill | ~1.8-2.0K tok/s for one 8K prompt, ~2.4K tok/s at 32K or 8 concurrent |

Underclocking leads to 3-5% lost performance with 10-20% power savings

## Performance


Measured 2026-09-28 on the `agent` profile (10 cards; the core profile
applies +250 MHz VF offset and a 1350 MHz SM ceiling, with 1300 MHz on the one
marginal card; 180 W power cap). C8 needs 8 seats, so every row was measured
with `MAX_SEQS=8`. That leaves 1,832,659 KV tokens against 1,874,240 at the
default 4 seats. C1 and C4 matched the default profile within 4%.

Decode: C concurrent greedy streams, 512 tokens each (`ignore_eos`), a
different prompt per stream (prose, code, technical). Prefill: C concurrent
requests of ~7.8K unique prompt tokens (no prefix-cache hits), `max_tokens=1`.

| Concurrency | Per-stream decode | Aggregate decode | Spec tokens/step | Prefill TTFT mean / max | Aggregate prefill |
|---|---|---|---|---|---|
| C1 | 39.7 tok/s | 39.2 tok/s | 3.55 | 4.3 s / 4.3 s | 1,836 tok/s |
| C4 | 29.0 tok/s (min 26.4) | 102.1 tok/s | 3.47 | 9.9 s / 14.6 s | 2,140 tok/s |
| C8 | 22.5 tok/s (min 20.0) | 152.2 tok/s | 3.21 | 15.1 s / 25.5 s | 2,460 tok/s |

A single 31K-token prompt prefills in 13.1 s (2,386 tok/s). Prefill is
chunked at `max_num_batched_tokens=512` through a 10-stage pipeline, so it
gains only ~25-35% from concurrency. Decode throughput scales with
concurrency. Speculative acceptance dips slightly as streams are added.

Reproduce with `bench_conc.py` against a running server:

```bash
MAX_SEQS=8 PROFILE=agent ./start.sh
python3 bench_conc.py 8000 decode 8 512     # C8 decode
python3 bench_conc.py 8000 prefill 8 8192   # C8 prefill
```

The default `agent` profile (4 seats) measured C1 39.6 tok/s decode and
1,973 tok/s prefill, and C4 98.1 tok/s aggregate decode and 2,227 tok/s
prefill. Those are within run-to-run noise of the rows above.

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
- **Optional W8 lm_head** (`glm52_lmhead_quant.py`, `GLM52_LMHEAD_BITS=8`):
  off by default (~0.5 ms/step gain, small top-1 changes).

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
PROFILE=agent ./start.sh      # background; logs/serve.log, logs/vllm.pid
./smoke.sh                    # health + one chat request
./stop.sh                     # stops only the recorded process group
PROFILE=agent ./serve.sh      # foreground equivalent
```

| Profile | Context | Seqs | Batched tokens | Graphs | Bind |
|---|---|---|---|---|---|
| `smoke` (default) | 32K | 1 | 1024 | eager | 127.0.0.1:8001 |
| `agent` | 1,048,576 | 4 | 512 | FULL + PIECEWISE | 0.0.0.0:8000 |
| `production` | 1,048,576 | 8 | 2048 | FULL + PIECEWISE | 127.0.0.1:8001 |

Useful switches (environment): `MODEL`, `DFLASH_MODEL`, `MAX_LEN`, `MAX_SEQS`,
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

CMP 170HX cards have no ECC. One card was found to compute all-zero MoE output
rows (about 1 per 3,000 decode steps) and occasional garbage top-k indices at
its default clock under these kernels, while passing plain GEMM and memory
tests (`gpu_consistency.py`). To find such a card:

1. Run `probe_bang.py`; injected tokens with logprob -11.95 indicate the fault.
2. Instrument or bisect by swapping two devices in `CUDA_VISIBLE_DEVICES`; the
   fault follows the physical card, not the pipeline stage.
3. Cap that card's SM clock with `gpu-clock-cap.service` (1300 MHz removed the
   fault entirely at no measurable decode cost), or replace it.
