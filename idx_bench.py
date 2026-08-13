#!/usr/bin/env python3
"""Standalone microbench for the DSA decode indexer kernels (no server).

Times fp8_paged_mqa_logits_triton (+ persistent_topk when available) across
a context sweep at serving-realistic shapes (B requests x next_n=4 spec
rows, 32 heads x 128 dim, block 64), and projects the per-step cost across
LAYERS_PER_RANK layers. Compare kernels with --impl {v1,v2}.

Usage: idx_bench.py [--ctx 4096,32768,131072,262144] [--batch 1,4,16]
                    [--impl v1] [--device 0]
"""

import argparse
import time

import torch

from vllm.v1.attention.ops.mqa_logits_triton import (
    fp8_paged_mqa_logits_triton,
)

LAYERS_PER_RANK = 10
NEXT_N = 4
HEADS = 32
DIM = 128
BLOCK = 64


def make_inputs(batch, ctx, max_blocks, device):
    torch.manual_seed(0)
    q = (torch.randn(batch, NEXT_N, HEADS, DIM, device=device) / 8).to(
        torch.float8_e4m3fn
    )
    n_blocks = (ctx + BLOCK - 1) // BLOCK
    kv_f = torch.randn(n_blocks * batch, BLOCK, DIM, device=device) / 8
    kv_fp8 = kv_f.to(torch.float8_e4m3fn)
    kv_scale = torch.rand(n_blocks * batch, BLOCK, device=device) * 0.05 + 0.01
    # Cache layout (indexer_k_quant_and_cache): per block, FP8 K bytes
    # (BLOCK*DIM) followed by fp32 scales (BLOCK*4); nominal shape
    # (NB, BLOCK, 1, DIM+4) uint8. Build the flat layout and reshape.
    nb = n_blocks * batch
    kv_flat = torch.empty(nb, BLOCK * (DIM + 4), dtype=torch.uint8,
                          device=device)
    kv_flat[:, : BLOCK * DIM] = kv_fp8.view(torch.uint8).reshape(nb, -1)
    kv_flat[:, BLOCK * DIM:] = (
        kv_scale.to(torch.float32).view(torch.uint8).reshape(nb, -1)
    )
    kv_cache = kv_flat.view(nb, BLOCK, 1, DIM + 4)
    block_table = torch.zeros(
        batch, max_blocks, dtype=torch.int32, device=device
    )
    for b in range(batch):
        block_table[b, :n_blocks] = torch.arange(
            b * n_blocks, (b + 1) * n_blocks, dtype=torch.int32
        )
    context_lens = torch.full((batch,), ctx, dtype=torch.int32, device=device)
    weights = torch.rand(
        batch * NEXT_N, HEADS, dtype=torch.float32, device=device
    )
    return q, kv_cache, weights, context_lens, block_table


def bench(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", default="4096,32768,131072,262144")
    ap.add_argument("--batch", default="1,4,16")
    ap.add_argument("--impl", default="v1", choices=["v1", "v2"])
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--max-len", type=int, default=262144)
    ap.add_argument("--check", action="store_true",
                    help="verify v2 against v1 (small shapes) and exit")
    ap.add_argument("--topk", action="store_true",
                    help="bench persistent_topk vs baked max_seq_len and exit")
    a = ap.parse_args()
    torch.cuda.set_device(a.device)
    dev = f"cuda:{a.device}"
    max_blocks = (a.max_len + BLOCK - 1) // BLOCK

    if a.topk:
        # Does a baked (capture-time) max_seq_len make persistent_topk slow
        # at small runtime lengths? This is what full-width capture bakes.
        import vllm._custom_ops as ops  # noqa: F401
        rows = 64
        for runtime_len in (1024, 3300, 32768):
            width = max(runtime_len, 2048)
            logits = torch.randn(rows, width, device=dev)
            lengths = torch.full((rows,), runtime_len, dtype=torch.int32,
                                 device=dev)
            out = torch.empty(rows, 2048, dtype=torch.int32, device=dev)
            ws = torch.empty(16 << 20, dtype=torch.uint8, device=dev)
            for baked_max in (runtime_len, 262144):
                def fn():
                    torch.ops._C.persistent_topk(
                        logits, lengths, out, ws, 2048, baked_max)
                ms = bench(fn)
                print(f"  rows={rows} runtime_len={runtime_len:>6} "
                      f"baked_max={baked_max:>6}: {ms:.3f} ms "
                      f"(x{LAYERS_PER_RANK} layers = {ms*LAYERS_PER_RANK:.1f}"
                      f" ms/step/rank)")
        return

    if a.check:
        from mqa_logits_v2 import fp8_paged_mqa_logits_v2
        for batch, ctx in [(1, 4096), (3, 5000), (16, 8192), (2, 131072)]:
            q, kv, w, lens, bt = make_inputs(batch, ctx, max_blocks, dev)
            # vary lens so causal bounds differ per request
            lens = lens - torch.arange(batch, device=dev, dtype=torch.int32) * 7
            ref = fp8_paged_mqa_logits_triton(
                q, kv, w, lens, bt, max_model_len=a.max_len,
                clean_logits=False, skip_le=0)
            got = fp8_paged_mqa_logits_v2(
                q, kv, w, lens, bt, max_model_len=a.max_len)
            for b in range(batch):
                L = int(lens[b])
                r = ref[b * NEXT_N:(b + 1) * NEXT_N, :L]
                g = got[b * NEXT_N:(b + 1) * NEXT_N, :L]
                inf_match = ((r == float("-inf")) == (g == float("-inf"))).all()
                fin = r != float("-inf")
                close = torch.allclose(r[fin], g[fin], rtol=1e-3, atol=1e-3)
                assert inf_match and close, (
                    f"MISMATCH batch={batch} ctx={ctx} req={b}: "
                    f"inf_match={bool(inf_match)} close={close} "
                    f"maxdiff={(r[fin]-g[fin]).abs().max().item():.4e}")
            print(f"  check OK batch={batch} ctx={ctx}")
        print("v2 == v1 on all shapes")
        return

    if a.impl == "v2":
        from mqa_logits_v2 import fp8_paged_mqa_logits_v2 as impl
    else:
        impl = None  # use wrapper below

    print(f"impl={a.impl} heads={HEADS} dim={DIM} next_n={NEXT_N} "
          f"layers/rank={LAYERS_PER_RANK}")
    print(f"{'batch':>5} {'ctx':>8} {'kernel ms':>10} {'step ms/rank':>13} "
          f"{'GB/s(K once)':>12}")
    for batch in map(int, a.batch.split(",")):
        for ctx in map(int, a.ctx.split(",")):
            q, kv, w, lens, bt = make_inputs(batch, ctx, max_blocks, dev)
            if a.impl == "v1":
                def fn():
                    fp8_paged_mqa_logits_triton(
                        q, kv, w, lens, bt,
                        max_model_len=a.max_len, clean_logits=False,
                        skip_le=0,
                    )
            else:
                def fn():
                    impl(q, kv, w, lens, bt, max_model_len=a.max_len)
            ms = bench(fn)
            step = ms * LAYERS_PER_RANK
            gbs = batch * ctx * (DIM + 4) / (ms / 1e3) / 1e9
            print(f"{batch:>5} {ctx:>8} {ms:>10.3f} {step:>13.1f} "
                  f"{gbs:>12.0f}")


if __name__ == "__main__":
    main()
