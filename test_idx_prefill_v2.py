#!/usr/bin/env python3
"""Correctness + speed of the v2 prefill indexer logits kernel vs v1.

v1 and v2 compute the same math but not in the same order (v1 does one
[H, D] x [D, N] dot per query row in bf16; v2 does one [BLOCK_M*H, D] x
[D, N] dot in fp16 for BLOCK_M rows at once). Both decode e4m3 exactly --
4 significant bits land in bf16 and fp16 alike -- so the only difference is
fp32 accumulation order, and agreement should be at fp32-rounding level.

Only the [ks, ke) window is compared: outside it both write -inf, and the
indexer's top-k never reads there.
"""

import argparse
import sys

import torch
import triton

sys.path.insert(0, "/home/user/vllm_install")

from vllm.v1.attention.ops.mqa_logits_triton import fp8_mqa_logits_triton  # noqa: E402

from glm52_idx_prefill_v2 import fp8_mqa_logits_v2  # noqa: E402

HEADS, DIM = 32, 128


def make_inputs(m, n, device, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    q = (torch.randn(m, HEADS, DIM, device=device, generator=g) / 8).to(
        torch.float8_e4m3fn
    )
    k = (torch.randn(n, DIM, device=device, generator=g) / 8).to(torch.float8_e4m3fn)
    k_scales = (torch.rand(n, device=device, generator=g) * 0.05 + 0.01).float()
    weights = torch.randn(m, HEADS, device=device, generator=g).float().abs()
    # Chunk of m queries sitting at the end of an n-token context.
    ke = torch.arange(n - m + 1, n + 1, device=device, dtype=torch.int32)
    ks = torch.zeros(m, device=device, dtype=torch.int32)
    return q, k, k_scales, weights, ks, ke


def check(m, n, device, seed=0, ks_offset=0):
    q, k, k_scales, weights, ks, ke = make_inputs(m, n, device, seed)
    if ks_offset:
        # Sliding-window-ish case: a nonzero row start, so the early-exit and
        # masking paths get exercised on both ends.
        ks = ks + ks_offset
    a = fp8_mqa_logits_triton(q, (k, k_scales), weights, ks, ke, clean_logits=False)
    b = fp8_mqa_logits_v2(q, (k, k_scales), weights, ks, ke, clean_logits=False)

    idx = torch.arange(n, device=device)
    win = (idx[None, :] >= ks[:, None]) & (idx[None, :] < ke[:, None])
    av, bv = a[win], b[win]
    # Both must put -inf everywhere outside the window.
    outside_ok = bool(torch.isinf(a[~win]).all() and torch.isinf(b[~win]).all())
    rel = ((av - bv).abs().max() / av.abs().max().clamp(min=1e-6)).item()
    l2 = ((av - bv).norm() / av.norm().clamp(min=1e-6)).item()
    nan = bool(torch.isnan(bv).any())
    ok = rel < 2e-3 and outside_ok and not nan
    print(
        f"  M={m:<5} N={n:<8} ks+{ks_offset:<7} max_rel={rel:.2e} l2={l2:.2e} "
        f"outside_inf={outside_ok} {'OK' if ok else 'FAIL'}"
    )
    del a, b, q, k
    torch.cuda.empty_cache()
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--ctx", default="262144,1048576")
    args = ap.parse_args()
    device = f"cuda:{args.device}"
    torch.cuda.set_device(args.device)
    free, _ = torch.cuda.mem_get_info(args.device)
    print(f"device {args.device}: {free / 2**30:.1f} GiB free\n")

    print("correctness (v2 vs v1, inside [ks, ke)):")
    cases = [
        (1, 8192, 0, 0),
        (8, 8192, 0, 1),
        (128, 65536, 0, 2),
        (128, 65536, 1000, 3),
        (129, 65536, 0, 4),  # M not a multiple of BLOCK_M
        (255, 32768, 777, 5),
        (512, 131072, 0, 6),
    ]
    ok = all([check(m, n, device, seed, off) for m, n, off, seed in cases])

    print("\nspeed (per query token x 1K keys; lower is better):")
    for n in [int(x) for x in args.ctx.split(",")]:
        native_m = max(1, (512 * 1024 * 1024 // 4) // n)
        for m in sorted({128, 256, native_m}):
            need = m * n * 4 * 2 + n * 132
            if need > free * 0.8:
                continue
            q, k, k_scales, weights, ks, ke = make_inputs(m, n, device)
            t1 = triton.testing.do_bench(
                lambda: fp8_mqa_logits_triton(
                    q, (k, k_scales), weights, ks, ke, clean_logits=False
                ),
                warmup=25,
                rep=100,
            )
            t2 = triton.testing.do_bench(
                lambda: fp8_mqa_logits_v2(
                    q, (k, k_scales), weights, ks, ke, clean_logits=False
                ),
                warmup=25,
                rep=100,
            )
            unit = 1e6 / (m * n / 1000)
            # 32 heads x 128 dims x 2 flop, per (query, key) pair.
            tflops = lambda ms: m * n * HEADS * DIM * 2 / (ms * 1e-3) / 1e12  # noqa: E731
            print(
                f"  N={n:<8} M={m:<5} v1 {t1 * unit:6.1f} ns ({tflops(t1):5.1f} TF/s)  "
                f"v2 {t2 * unit:6.1f} ns ({tflops(t2):5.1f} TF/s)   {t1 / t2:.2f}x"
            )
            del q, k, k_scales, weights
            torch.cuda.empty_cache()

    print("\n" + ("ALL PASS" if ok else "FAILURES"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
