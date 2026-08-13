#!/usr/bin/env python3
"""Standalone bench for the sparse MLA decode attention (sm80 Triton).

Isolates the gather-locality question: does split-kernel cost grow with the
SPREAD of the 2048 top-k indices across the KV buffer (TLB/L2 locality),
independent of the fixed candidate count? Also measures whether sorting
the indices (exact: softmax over a set is order-invariant, the merge is a
sum) recovers the loss.

Patterns per ctx: contig (best case: last 2048 positions), spread (uniform
over ctx, sorted), shuffled (uniform, random order -- what topk actually
emits today, bucket-ordered-ish).
"""

import argparse
import time

import torch

from vllm.v1.attention.ops.triton_mla_sparse_kernel import (
    triton_mla_sparse_attention,
)

TOKENS = 4       # B=1, MTP next_n=4
HEADS = 128
DIM = 576
TOPK = 2048


def bench(fn, iters=50, warmup=10):
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
    ap.add_argument("--ctx", default="32768,196608,409600")
    ap.add_argument("--device", type=int, default=0)
    a = ap.parse_args()
    torch.cuda.set_device(a.device)
    dev = f"cuda:{a.device}"
    torch.manual_seed(0)

    q = torch.randn(TOKENS, HEADS, DIM, device=dev, dtype=torch.bfloat16)

    print(f"tokens={TOKENS} heads={HEADS} dim={DIM} topk={TOPK}")
    print(f"{'ctx':>8} {'pattern':>9} {'ms':>8}  (x80 layers = ms/step)")
    for ctx in map(int, a.ctx.split(",")):
        kv = torch.randn(ctx, 1, DIM, device=dev, dtype=torch.bfloat16)
        pats = {}
        base = torch.arange(ctx - TOPK, ctx, dtype=torch.int32, device=dev)
        pats["contig"] = base
        spread = torch.linspace(0, ctx - 1, TOPK, device=dev).to(torch.int32)
        pats["spread"] = spread
        pats["shuffled"] = spread[torch.randperm(TOPK, device=dev)]
        for name, idx in pats.items():
            indices = idx.view(1, 1, TOPK).expand(TOKENS, 1, TOPK).contiguous()
            fn = lambda: triton_mla_sparse_attention(
                q, kv, indices, sm_scale=0.135)
            ms = bench(fn)
            print(f"{ctx:>8} {name:>9} {ms:>8.3f}  {ms*80:>7.1f}")


if __name__ == "__main__":
    main()
