#!/usr/bin/env python3
"""Correctness + speed of the Marlin-packed MoE GEMV (glm52_moe_decode.py)
against vLLM's `moe_wna16_marlin_gemm` on the *same* packed buffers.

Both kernels read identical [E, K/16, N*16/8] int32 weights, marlin-permuted
scales and packed zero points, so any disagreement is the GEMV's index algebra,
not a quantization difference. Reference weights (`w_ref`) are also carried so
the two can be scored against exact fp arithmetic rather than each other.

Shapes default to GLM-5.2 production: K=6144, N=2048 (per gate/up shard),
group 64, INT4 asymmetric. Expert count is small so the packed weights fit
alongside a running server.
"""

import argparse
import sys

import torch
import triton

sys.path.insert(0, "/home/user/vllm_install")

from vllm import _custom_ops as ops  # noqa: E402
from vllm.model_executor.layers.quantization.utils.marlin_utils import (  # noqa: E402
    marlin_make_workspace_new,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (  # noqa: E402
    awq_marlin_quantize,
)
from vllm.scalar_type import scalar_types  # noqa: E402

from glm52_moe_decode import moe_marlin_gemv  # noqa: E402

DEV = "cuda:0"
QTYPE = scalar_types.uint4


def build_experts(E, K, N, group, seed=0):
    """Marlin-pack E experts; returns stacked packed tensors + fp references."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    qw, sc, zp, ref = [], [], [], []
    for _ in range(E):
        w = (torch.randn(K, N, device=DEV, generator=g) / (K**0.5)).half()
        w_ref, mq, ms, mz = awq_marlin_quantize(w, QTYPE, group)
        ref.append(w_ref)
        qw.append(mq)
        sc.append(ms)
        zp.append(mz)
    return (torch.stack(qw), torch.stack(sc), torch.stack(zp), torch.stack(ref))


def align(topk_ids, block, E):
    """moe_align_block_size writes into caller-allocated buffers (returns None)."""
    M, top_k = topk_ids.shape
    max_slots = M * top_k + E * (block - 1)
    n_blocks = (max_slots + block - 1) // block
    sorted_ids = torch.empty(n_blocks * block, dtype=torch.int32, device=DEV)
    sorted_ids.fill_(M * top_k)  # pad marker: >= num_valid_tokens
    expert_ids = torch.empty(n_blocks, dtype=torch.int32, device=DEV)
    num_pad = torch.empty(1, dtype=torch.int32, device=DEV)
    ops.moe_align_block_size(
        topk_ids, E, block, sorted_ids, expert_ids, num_pad, None
    )
    return sorted_ids, expert_ids, num_pad


def run(M, E, K, N, group, top_k, block, tok, mul_routed, verbose=True):
    qw, sc, zp, ref = build_experts(E, K, N, group)
    g = torch.Generator(device=DEV).manual_seed(1)
    x = (torch.randn(M, K, device=DEV, generator=g) / 4).half()
    topk_ids = torch.randint(0, E, (M, top_k), device=DEV, dtype=torch.int32,
                             generator=g)
    topk_w = torch.rand(M, top_k, device=DEV, generator=g).float()
    sorted_ids, expert_ids, num_pad = align(topk_ids, block, E)

    # -- reference: vLLM Marlin
    workspace = marlin_make_workspace_new(torch.device(DEV), 4)
    out_marlin = ops.moe_wna16_marlin_gemm(
        x, None, qw, None, sc, None, None, zp, None, None, workspace,
        sorted_ids, expert_ids, num_pad, topk_w,
        moe_block_size=block, top_k=top_k, mul_topk_weights=mul_routed,
        b_q_type=QTYPE, size_m=M, size_n=N, size_k=K,
        is_k_full=True, use_atomic_add=False, use_fp32_reduce=True,
        is_zp_float=False,
    ).view(M * top_k, N)

    # -- candidate: the GEMV, same buffers
    y = torch.zeros(sorted_ids.shape[0], N, dtype=torch.float32, device=DEV)
    moe_marlin_gemv(
        x, qw, sc, zp, sorted_ids, expert_ids, topk_w.view(-1),
        num_valid_tokens=M * top_k, N=N, K=K, G=group, top_k=top_k,
        mul_routed_weight=mul_routed, x_row_div=top_k, out=y,
        tok_block=block, tok=tok,
    )
    out_gemv = y[: M * top_k].half()

    # Score both against exact fp math on the dequantized reference weights.
    flat = topk_ids.view(-1)
    gold = torch.empty(M * top_k, N, device=DEV, dtype=torch.float32)
    for i in range(M * top_k):
        gold[i] = x[i // top_k].float() @ ref[flat[i]].float()
    if mul_routed:
        gold = gold * topk_w.view(-1, 1)

    def rel(a):
        return ((a.float() - gold).norm() / gold.norm()).item()

    r_marlin, r_gemv = rel(out_marlin), rel(out_gemv)
    delta = ((out_gemv.float() - out_marlin.float()).norm()
             / out_marlin.float().norm().clamp(min=1e-6)).item()
    ok = r_gemv <= max(3 * r_marlin, 1e-3)
    if verbose:
        print(f"  M={M:<4} E={E:<3} K={K:<5} N={N:<5} g={group} topk={top_k} "
              f"blk={block} tok={tok} mul={int(mul_routed)}  "
              f"marlin_err={r_marlin:.2e} gemv_err={r_gemv:.2e} "
              f"delta={delta:.2e}  {'OK' if ok else 'FAIL'}")

    t_m = triton.testing.do_bench(lambda: ops.moe_wna16_marlin_gemm(
        x, None, qw, None, sc, None, None, zp, None, None, workspace,
        sorted_ids, expert_ids, num_pad, topk_w,
        moe_block_size=block, top_k=top_k, mul_topk_weights=mul_routed,
        b_q_type=QTYPE, size_m=M, size_n=N, size_k=K, is_k_full=True,
        use_atomic_add=False, use_fp32_reduce=True, is_zp_float=False))
    t_g = triton.testing.do_bench(lambda: moe_marlin_gemv(
        x, qw, sc, zp, sorted_ids, expert_ids, topk_w.view(-1),
        num_valid_tokens=M * top_k, N=N, K=K, G=group, top_k=top_k,
        mul_routed_weight=mul_routed, x_row_div=top_k, out=y,
        tok_block=block, tok=tok))
    del qw, sc, zp, ref, y
    torch.cuda.empty_cache()
    return ok, t_m, t_g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--speed", action="store_true")
    a = ap.parse_args()
    torch.cuda.set_device(0)
    free, _ = torch.cuda.mem_get_info(0)
    print(f"free {free / 2**30:.1f} GiB\n")

    E = a.experts
    print("correctness (gemv vs marlin, both vs exact fp reference):")
    cases = [
        # M, E, K,    N,    group, top_k, block, tok, mul
        (1,  E, 6144, 2048, 64, 1, 16, 4, False),
        (4,  E, 6144, 2048, 64, 1, 16, 4, False),
        (4,  E, 6144, 2048, 64, 8, 16, 16, False),  # w1 shape, top_k=8
        (8,  E, 2048, 6144, 64, 1, 16, 4, True),    # w2 shape, routed weight
        (2,  E, 6144, 2048, 64, 8, 16, 16, True),
        # TOK must cover the busiest block or tokens are silently dropped:
        # M=4 x topk=8 = 32 expanded tokens over 8 experts overflows TOK=4.
        (4,  E, 6144, 2048, 64, 8, 8, 8, False),
        (16, E, 6144, 2048, 64, 8, 16, 16, False),
        (16, E, 6144, 2048, 64, 8, 8, 8, False),
    ]
    results = [run(*c) for c in cases]
    ok = all(r[0] for r in results)

    if a.speed:
        print("\nspeed (ms):")
        for c in cases:
            _, t_m, t_g = run(*c, verbose=False)
            print(f"  M={c[0]:<4} topk={c[5]}  marlin {t_m:7.3f}  "
                  f"gemv {t_g:7.3f}   {t_m / t_g:.2f}x")

    print("\n" + ("ALL PASS" if ok else "FAILURES"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
