#!/usr/bin/env python3
"""Validation for the split-MoE dispatch (_glm52_moe_split.py).

Four gates, run at production dtypes (bf16) and shapes (K=6144, N2=4096,
I=2048, g64 uint4 asymmetric, top_k 8):

1. GEMM parity: the CUDA GEMV vs ops.moe_wna16_marlin_gemm on identical
   Marlin-packed buffers, scored against exact fp math on the dequantized
   reference — including the 2-shard N=4096 w13 shape and the w2 shape with
   routed-weight folding.
2. Whole-layer parity: align -> gemv1 -> silu*mul -> gemv2 -> sum vs the
   same pipeline through the Marlin kernels (fused_marlin_moe's structure).
3. BATCH-INVARIANCE (the property production needs): decode rows computed
   alone vs co-batched with 2048 junk "prefill" rows must be BIT-IDENTICAL
   through the gemv path. Also demonstrates that fused Marlin FAILS this
   (moe_block_size flips 8 -> 64), which is the corruption mechanism.
4. Speed at decode shapes (M=4/8/16, E=256) vs fused Marlin.

Needs a free GPU (several GB at E=256) — stop the server first.
"""

import argparse
import sys

import torch
import triton

sys.path.insert(0, "/home/user/vllm_install")

from vllm import _custom_ops as ops  # noqa: E402
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (  # noqa: E402
    moe_align_block_size,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (  # noqa: E402
    marlin_make_workspace_new,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (  # noqa: E402
    awq_marlin_quantize,
)
from vllm.scalar_type import scalar_types  # noqa: E402

import _glm52_moe_split as split  # noqa: E402

DEV = "cuda:0"
QTYPE = scalar_types.uint4
K, N2, I, GROUP, TOPK = 6144, 4096, 2048, 64, 8


def build_experts(E, k, n, dtype, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    qw, sc, zp, ref = [], [], [], []
    for _ in range(E):
        w = (torch.randn(k, n, device=DEV, generator=g) / (k**0.5)).to(dtype)
        w_ref, mq, ms, mz = awq_marlin_quantize(w, QTYPE, GROUP)
        ref.append(w_ref)
        qw.append(mq)
        sc.append(ms)
        zp.append(mz)
    return (torch.stack(qw), torch.stack(sc), torch.stack(zp),
            torch.stack(ref))


def marlin_gemm(x, qw, sc, zp, sorted_ids, expert_ids, num_pad, topk_w,
                block, top_k, mul, M_rows, n, k, workspace):
    return ops.moe_wna16_marlin_gemm(
        x, None, qw, None, sc, None, None, zp, None, None, workspace,
        sorted_ids, expert_ids, num_pad, topk_w,
        moe_block_size=block, top_k=top_k, mul_topk_weights=mul,
        b_q_type=QTYPE, size_m=M_rows, size_n=n, size_k=k,
        is_k_full=True, use_atomic_add=False, use_fp32_reduce=True,
        is_zp_float=False,
    )


def gemv(ext, x, qw, sc, zp, sorted_ids, expert_ids, tw, k, n, block, nvt,
         mul, x_row_div):
    y = torch.empty(nvt, n, dtype=torch.float32, device=DEV)
    ext.moe_gemv_marlin(x, qw, sc, zp, y, sorted_ids, expert_ids, tw,
                        k, n, GROUP, block, nvt, mul, x_row_div)
    return y


def gemv_layer(ext, x, w13, s13, z13, w2, s2, z2, topk_w, topk_ids,
               block=8):
    """The exact pipeline moe_gemv_forward runs (inlined so the test does
    not depend on forward-context)."""
    M = x.shape[0]
    E = w13.shape[0]
    nvt = M * TOPK
    sorted_ids, expert_ids, _ = moe_align_block_size(topk_ids, block, E)
    tw = topk_w.reshape(-1).float()
    y1 = gemv(ext, x, w13, s13, z13, sorted_ids, expert_ids, tw,
              K, N2, block, nvt, False, TOPK)
    inter = (torch.nn.functional.silu(y1[:, :I]) * y1[:, I:]).to(x.dtype)
    y2 = gemv(ext, inter, w2, s2, z2, sorted_ids, expert_ids, tw,
              I, K, block, nvt, True, 1)
    return y2.view(M, TOPK, K).sum(dim=1).to(x.dtype), y2


def marlin_layer(x, w13, s13, z13, w2, s2, z2, topk_w, topk_ids, workspace,
                 block):
    M = x.shape[0]
    E = w13.shape[0]
    nvt = M * TOPK
    sorted_ids, expert_ids, num_pad = moe_align_block_size(
        topk_ids, block, E)
    c1 = marlin_gemm(x, w13, s13, z13, sorted_ids, expert_ids, num_pad,
                     topk_w, block, TOPK, False, M, N2, K,
                     workspace).view(nvt, N2)
    c2 = torch.empty(nvt, I, dtype=x.dtype, device=DEV)
    torch.ops._C.silu_and_mul(c2, c1)
    c3 = marlin_gemm(c2, w2, s2, z2, sorted_ids, expert_ids, num_pad,
                     topk_w, block, 1, True, nvt, K, I,
                     workspace).view(nvt, K)
    return c3.view(M, TOPK, K).sum(dim=1).to(x.dtype), c3


def rel(a, b):
    return ((a.float() - b.float()).norm()
            / b.float().norm().clamp(min=1e-9)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", type=int, default=64)
    ap.add_argument("--dtype", choices=["bfloat16", "float16"],
                    default="bfloat16")
    ap.add_argument("--speed", action="store_true")
    a = ap.parse_args()
    dtype = getattr(torch, a.dtype)
    torch.cuda.set_device(0)
    E = a.experts
    ext = split._get_ext()
    ws = marlin_make_workspace_new(torch.device(DEV), 4)
    g = torch.Generator(device=DEV).manual_seed(7)

    print(f"experts={E} dtype={a.dtype}")
    print("building packed experts (w13 + w2)...")
    w13, s13, z13, ref13 = build_experts(E, K, N2, dtype, seed=0)
    w2, s2, z2, ref2 = build_experts(E, I, K, dtype, seed=1)
    ok = True

    # ---- gate 1: per-GEMM parity vs exact fp math -------------------------
    print("\n[1] per-GEMM parity (vs exact fp on dequantized reference):")
    M = 8
    x = (torch.randn(M, K, device=DEV, generator=g) / 4).to(dtype)
    topk_ids = torch.randint(0, E, (M, TOPK), device=DEV,
                             dtype=torch.int32, generator=g)
    topk_w = torch.rand(M, TOPK, device=DEV, generator=g).float()
    nvt = M * TOPK
    flat = topk_ids.view(-1)
    for block in (8, 16):
        sorted_ids, expert_ids, num_pad = moe_align_block_size(
            topk_ids, block, E)
        tw = topk_w.reshape(-1).float()
        # w13 shape
        y_g = gemv(ext, x, w13, s13, z13, sorted_ids, expert_ids, tw,
                   K, N2, block, nvt, False, TOPK)
        y_m = marlin_gemm(x, w13, s13, z13, sorted_ids, expert_ids,
                          num_pad, topk_w, block, TOPK, False, M, N2, K,
                          ws).view(nvt, N2)
        gold = torch.stack([
            x[i // TOPK].float() @ ref13[flat[i]].float()
            for i in range(nvt)])
        rg, rm = rel(y_g, gold), rel(y_m, gold)
        p = rg <= max(3 * rm, 1e-3)
        ok &= p
        print(f"  w13 blk={block}: gemv_err={rg:.2e} marlin_err={rm:.2e} "
              f"{'OK' if p else 'FAIL'}")
        # w2 shape with routed weights
        x2 = (torch.randn(nvt, I, device=DEV, generator=g) / 4).to(dtype)
        y_g = gemv(ext, x2, w2, s2, z2, sorted_ids, expert_ids, tw,
                   I, K, block, nvt, True, 1)
        y_m = marlin_gemm(x2, w2, s2, z2, sorted_ids, expert_ids,
                          num_pad, topk_w, block, 1, True, nvt, K, I,
                          ws).view(nvt, K)
        gold = torch.stack([
            x2[i].float() @ ref2[flat[i]].float() * topk_w.view(-1)[i]
            for i in range(nvt)])
        rg, rm = rel(y_g, gold), rel(y_m, gold)
        p = rg <= max(3 * rm, 1e-3)
        ok &= p
        print(f"  w2  blk={block}: gemv_err={rg:.2e} marlin_err={rm:.2e} "
              f"{'OK' if p else 'FAIL'}")

    # ---- gate 2: whole-layer parity --------------------------------------
    print("\n[2] whole-layer parity (gemv pipeline vs marlin pipeline):")
    out_g, _ = gemv_layer(ext, x, w13, s13, z13, w2, s2, z2, topk_w,
                          topk_ids)
    out_m, _ = marlin_layer(x, w13, s13, z13, w2, s2, z2, topk_w,
                            topk_ids, ws, block=8)
    d = rel(out_g, out_m)
    p = d < 5e-2
    ok &= p
    print(f"  M={M}: rel delta gemv vs marlin = {d:.2e} "
          f"{'OK' if p else 'FAIL'}")

    # ---- gate 3: batch invariance (bitwise) -------------------------------
    print("\n[3] batch invariance (decode rows alone vs +2048 junk rows):")
    M_dec, M_junk = 8, 2048
    xd = (torch.randn(M_dec, K, device=DEV, generator=g) / 4).to(dtype)
    xj = (torch.randn(M_junk, K, device=DEV, generator=g) / 4).to(dtype)
    tid_d = torch.randint(0, E, (M_dec, TOPK), device=DEV,
                          dtype=torch.int32, generator=g)
    tid_j = torch.randint(0, E, (M_junk, TOPK), device=DEV,
                          dtype=torch.int32, generator=g)
    tw_d = torch.rand(M_dec, TOPK, device=DEV, generator=g).float()
    tw_j = torch.rand(M_junk, TOPK, device=DEV, generator=g).float()
    x_all = torch.cat([xd, xj])
    tid_all = torch.cat([tid_d, tid_j])
    tw_all = torch.cat([tw_d, tw_j])

    out_alone, _ = gemv_layer(ext, xd, w13, s13, z13, w2, s2, z2, tw_d,
                              tid_d)
    out_co, _ = gemv_layer(ext, x_all, w13, s13, z13, w2, s2, z2, tw_all,
                           tid_all)
    bit_ok = torch.equal(out_alone, out_co[:M_dec])
    ok &= bit_ok
    print(f"  gemv:   decode rows bit-identical = {bit_ok} "
          f"{'OK' if bit_ok else 'FAIL'}")

    # fused-marlin behavior for contrast (block flips 8 -> 64 with M);
    # informational, not a gate.
    m_alone, _ = marlin_layer(xd, w13, s13, z13, w2, s2, z2, tw_d, tid_d,
                              ws, block=8)
    m_co, _ = marlin_layer(x_all, w13, s13, z13, w2, s2, z2, tw_all,
                           tid_all, ws, block=64)
    m_bit = torch.equal(m_alone, m_co[:M_dec])
    m_delta = rel(m_alone, m_co[:M_dec])
    print(f"  marlin: decode rows bit-identical = {m_bit} "
          f"(rel delta {m_delta:.2e}) <- the corruption mechanism")

    # repeat-call determinism of the gemv path
    out_rep, _ = gemv_layer(ext, x_all, w13, s13, z13, w2, s2, z2, tw_all,
                            tid_all)
    det = torch.equal(out_co, out_rep)
    ok &= det
    print(f"  gemv:   repeat-call bit-identical = {det} "
          f"{'OK' if det else 'FAIL'}")

    # ---- gate 4: speed at decode shapes -----------------------------------
    if a.speed:
        print("\n[4] speed at decode shapes (whole layer, ms):")
        for M_s in (4, 8, 16):
            xs = (torch.randn(M_s, K, device=DEV, generator=g) / 4).to(dtype)
            tid = torch.randint(0, E, (M_s, TOPK), device=DEV,
                                dtype=torch.int32, generator=g)
            tws = torch.rand(M_s, TOPK, device=DEV, generator=g).float()
            t_g = triton.testing.do_bench(
                lambda: gemv_layer(ext, xs, w13, s13, z13, w2, s2, z2,
                                   tws, tid))
            t_m = triton.testing.do_bench(
                lambda: marlin_layer(xs, w13, s13, z13, w2, s2, z2, tws,
                                     tid, ws, block=8))
            print(f"  M={M_s:<3} marlin {t_m:7.3f}  gemv {t_g:7.3f}  "
                  f"{t_m / t_g:.2f}x")

    print("\n" + ("ALL PASS" if ok else "FAILURES"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
