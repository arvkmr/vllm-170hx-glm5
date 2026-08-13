#!/usr/bin/env python3
"""Microbench for the DSA *prefill* indexer at long context (no server).

Companion to idx_bench.py, which covers the decode path. The prefill path is
different work: per chunk it materializes a full [M, N] fp32 logits matrix
(M query rows x N keys) and then runs a top-2048 over each row.

The question this answers: vLLM bounds that matrix with
VLLM_SPARSE_INDEXER_MAX_LOGITS_MB (default 512), so the query chunk is
M = 134217728 / N. At 262K context that is 512 rows; at 1M it collapses to
128. Since every chunk re-reads the whole N-token K workspace, cost per query
token should fall as M grows -- this measures by how much, which is the
difference between raising a config knob and writing a fused kernel.

Reported as ns per (query token x 1K keys) so numbers are comparable across
both M and N.
"""

import argparse

import torch
import triton

from vllm import _custom_ops as ops
from vllm.v1.attention.ops.mqa_logits_triton import fp8_mqa_logits_triton

HEADS = 32
DIM = 128
TOPK = 2048
# What the default 512 MB logits budget allows, in elements.
DEFAULT_BUDGET_ELEMS = 512 * 1024 * 1024 // 4


def make_inputs(m: int, n: int, device):
    torch.manual_seed(0)
    q = (torch.randn(m, HEADS, DIM, device=device) / 8).to(torch.float8_e4m3fn)
    k = (torch.randn(n, DIM, device=device) / 8).to(torch.float8_e4m3fn)
    k_scales = (torch.rand(n, device=device) * 0.05 + 0.01).float()
    weights = torch.randn(m, HEADS, device=device).float().abs()
    # Causal bounds: row i seeds at 0 and ends at (n - m + i + 1), i.e. the
    # chunk sits at the end of an n-token context, as during prefill.
    ke = torch.arange(n - m + 1, n + 1, device=device, dtype=torch.int32)
    ks = torch.zeros(m, device=device, dtype=torch.int32)
    return q, k, k_scales, weights, ks, ke


def bench(m: int, n: int, device, do_topk: bool):
    q, k, k_scales, weights, ks, ke = make_inputs(m, n, device)
    logits = fp8_mqa_logits_triton(q, (k, k_scales), weights, ks, ke, clean_logits=False)
    topk_out = torch.empty(m, TOPK, dtype=torch.int32, device=device)

    t_logits = triton.testing.do_bench(
        lambda: fp8_mqa_logits_triton(
            q, (k, k_scales), weights, ks, ke, clean_logits=False
        ),
        warmup=25,
        rep=100,
    )
    t_topk = 0.0
    if do_topk:
        t_topk = triton.testing.do_bench(
            lambda: ops.top_k_per_row_prefill(
                logits, ks, ke, topk_out, m, logits.stride(0), logits.stride(1), TOPK
            ),
            warmup=10,
            rep=50,
        )
    del logits, topk_out, q, k, k_scales, weights
    torch.cuda.empty_cache()
    return t_logits, t_topk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", default="262144,524288,1048576")
    ap.add_argument("--rows", default="128,256,512,1024",
                    help="query rows per indexer chunk (M)")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--no-topk", action="store_true",
                    help="skip top_k_per_row_prefill (halves peak memory)")
    args = ap.parse_args()

    device = f"cuda:{args.device}"
    torch.cuda.set_device(args.device)
    free, total = torch.cuda.mem_get_info(args.device)
    print(f"device {args.device}: {free / 2**30:.1f} GiB free of {total / 2**30:.1f}")
    print(f"topk={TOPK} heads={HEADS} dim={DIM}\n")

    for n in [int(x) for x in args.ctx.split(",")]:
        native_m = max(1, DEFAULT_BUDGET_ELEMS // n)
        print(f"context N={n:,}  (default 512MB budget allows M={native_m})")
        base = None
        for m in [int(x) for x in args.rows.split(",")]:
            need = m * n * 4 * (2 if not args.no_topk else 1) + n * 132
            if need > free * 0.85:
                print(f"  M={m:<5} skipped, needs {need / 2**30:.1f} GiB")
                continue
            try:
                t_log, t_topk = bench(m, n, device, not args.no_topk)
            except torch.OutOfMemoryError:
                print(f"  M={m:<5} OOM")
                torch.cuda.empty_cache()
                continue
            # Normalize: ns per query token per 1K keys.
            unit = 1e6 / (m * n / 1000)
            per_log, per_topk = t_log * unit, t_topk * unit
            base = base if base is not None else per_log + per_topk
            tag = f"  ({(base / (per_log + per_topk)):.2f}x vs M={args.rows.split(',')[0]})"
            print(
                f"  M={m:<5} logits {t_log:8.2f} ms  topk {t_topk:7.2f} ms   "
                f"per-tok-per-1Kkeys: logits {per_log:6.1f} ns  "
                f"topk {per_topk:5.1f} ns{tag}"
            )
        print()


if __name__ == "__main__":
    main()
