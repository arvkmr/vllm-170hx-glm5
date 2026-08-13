"""Small-batch (decode) W4A16 MoE GEMM that reads Marlin-packed weights.

Why: at decode batch sizes (M <= 8) vLLM's Marlin MoE kernel reaches only
62-69% of HBM bandwidth on GA100 (22% occupancy, ~0.45 ms fixed floor per
call), which is ~5 ms of every 7.4 ms pipeline stage on this box. This kernel
streams the SAME Marlin-packed buffers (no repacking, no extra memory) with a
GEMV-style schedule: grid = (expert-token-block, 64-col chunk), each program
loops over K.

Marlin layout algebra (derived by index tracing, verified exact):
per (16k x 64n) tile chunk = 128 consecutive int32 words, word w nibble i
holds the logical position
    k = 8*i0 + 2*pk + i2          i = i2*4 + i1*2 + i0
    n = 16*j + 8*i1 + 2*q + pn    w = q*32 + pn*16 + pk*4 + j
Scales ([K/G, N] halves): within each 64-col group the mapping is the
self-inverse 8x8 transpose  marlin_col(n) = 8*(n%8) + n//8.
Packed zero points ([K/G, N/8] int32): logical col n lives at word n%8,
nibble (n//8)%2*4 + n//16 within the group's 8 words.

KEY PERFORMANCE IDEA: n(w,i) does not depend on the k-tile, so the kernel
accumulates in STORED order (a [128, 8] register tensor) across the whole K
loop -- the per-iteration work is one coalesced 128-word load plus an L1-hot
gather of 16 activation values -- and the 7-D unpermute runs ONCE at the end
on the accumulator. Scales/zeros are gathered into stored order once per
group (every 4 k-tiles).

Dequant: (nib - zp) * scale (asymmetric INT4 g64; the GLM-5.2 checkpoint is
compressed-tensors "pack-quantized" with symmetric=false).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def moe_marlin_gemv_kernel(
    x_ptr,                 # [M, K] activations (half)
    w_ptr,                 # marlin packed [E, K/16, N*16/8] int32
    s_ptr,                 # marlin scales [E, K/G, N] half
    z_ptr,                 # marlin packed zp [E, K/G, N/8] int32
    y_ptr,                 # out [num_token_slots, N] float32
    sorted_token_ids_ptr,  # [num_slots * TOK_BLOCK] (moe_align output)
    expert_ids_ptr,        # [num_slots]
    topk_w_ptr,            # [M * top_k] router weights (float32)
    K: tl.constexpr, N: tl.constexpr, G: tl.constexpr,
    TOK_BLOCK: tl.constexpr,
    TOK: tl.constexpr,             # compile-time max tokens per block
    num_valid_tokens,
    stride_we, stride_se, stride_ze,
    MUL_ROUTED: tl.constexpr,
    x_row_div: tl.constexpr,
    NUM_KT: tl.constexpr,          # K // 16
    GT: tl.constexpr,              # G // 16 (k-tiles per scale group)
):
    slot = tl.program_id(0)
    chunk = tl.program_id(1)
    e = tl.load(expert_ids_ptr + slot)
    if e < 0:
        return

    toks = tl.load(sorted_token_ids_ptr + slot * TOK_BLOCK + tl.arange(0, TOK))
    tok_mask = toks < num_valid_tokens
    x_rows = tl.where(tok_mask, toks // x_row_div, 0)

    wp = w_ptr + e * stride_we + chunk * 128
    sp = s_ptr + e * stride_se + chunk * 64
    zp = z_ptr + e * stride_ze + chunk * 8
    NW: tl.constexpr = (N * 16) // 8           # words per k-tile row

    # Everything below stays in STORED order [w:128, i:8] with
    # w = (q, pn, pk, j) and i = (i2, i1, i0); logical coords are
    # k = 8*i0 + 2*pk + i2 and n = 16*j + 8*i1 + 2*q + pn. All expansions
    # from contiguous loads are pure reshape/broadcast/permute -- no gathers.
    acc = tl.zeros((TOK, 128, 8), dtype=tl.float32)

    for gt in range(0, NUM_KT // GT):          # scale groups
        # scales: marlin col = 8*(2q+pn) + 2j+i1 -> S[q,pn,j,i1]
        sv = tl.load(sp + gt * N + tl.arange(0, 64))
        sv = tl.reshape(sv, (4, 2, 4, 2))                       # (q,pn,j,i1)
        # -> stored dims (q,pn,pk,j | i2,i1,i0): broadcast pk,i2,i0
        sv = sv[:, :, None, :, None, :, None]                   # q,pn,1,j,1,i1,1
        scale = tl.reshape(
            tl.broadcast_to(sv, (4, 2, 4, 4, 2, 2, 2)), (128, 8)
        ).to(tl.float32)
        # zeros: word = 2q+pn, nibble = i1*4 + j -> Z[(q,pn),(i1,j)]
        zwords = tl.load(zp + gt * (N // 8) + tl.arange(0, 8))
        znib = (zwords[:, None] >> (4 * tl.arange(0, 8))[None, :]) & 0xF
        zv = tl.reshape(znib, (4, 2, 2, 4))                     # (q,pn,i1,j)
        zv = tl.permute(zv, (0, 1, 3, 2))                       # (q,pn,j,i1)
        zv = zv[:, :, None, :, None, :, None]
        zero = tl.reshape(
            tl.broadcast_to(zv, (4, 2, 4, 4, 2, 2, 2)), (128, 8)
        ).to(tl.float32)
        sz = scale * zero
        for tt in tl.static_range(GT):         # k-tiles within group
            t = gt * GT + tt
            words = tl.load(wp + t * NW + tl.arange(0, 128))         # [128]
            nib = ((words[:, None] >> (4 * tl.arange(0, 8))[None, :])
                   & 0xF).to(tl.float32)                             # [128,8]
            wdeq = nib * scale - sz                                  # [128,8]
            # activations: contiguous [TOK,16] load; k-bits (i0, pk, i2)
            xk = tl.load(x_ptr + x_rows[:, None] * K + t * 16
                         + tl.arange(0, 16)[None, :],
                         mask=tok_mask[:, None], other=0.0)          # [TOK,16]
            xk = tl.reshape(xk, (TOK, 2, 4, 2))                 # (i0,pk,i2)
            # insert singleton dims: (TOK, q1, pn1, i0, j1, pk, i11, i2)
            xk = xk[:, None, None, :, None, :, None, :]
            # -> (TOK, q, pn, pk, j, i2, i1, i0)
            xk = tl.permute(xk, (0, 1, 2, 5, 4, 7, 6, 3))
            xr = tl.reshape(
                tl.broadcast_to(xk, (TOK, 4, 2, 4, 4, 2, 2, 2)),
                (TOK, 128, 8)).to(tl.float32)
            acc += xr * wdeq[None, :, :]

    # unpermute the accumulator ONCE: [TOK,128,8] -> [TOK,16,64] -> sum k
    a = tl.reshape(acc, (TOK, 4, 2, 4, 4, 2, 2, 2))
    a = tl.permute(a, (0, 7, 3, 5, 4, 6, 1, 2))   # (TOK, i0,pk,i2, j,i1,q,pn)
    a = tl.reshape(a, (TOK, 16, 64))
    y = tl.sum(a, axis=1)                          # [TOK, 64]

    if MUL_ROUTED:
        rw = tl.load(topk_w_ptr + toks, mask=tok_mask, other=0.0)
        y = y * rw[:, None]

    nn = chunk * 64 + tl.arange(0, 64)
    tl.store(y_ptr + toks[:, None] * N + nn[None, :], y,
             mask=tok_mask[:, None])


def moe_marlin_gemv(x, w_packed, scales, zp_packed, sorted_token_ids,
                    expert_ids, topk_weights, num_valid_tokens, N, K, G,
                    top_k, mul_routed_weight, x_row_div, out=None,
                    tok_block=16, tok=4, num_warps=4, num_stages=3):
    num_slots = expert_ids.shape[0]
    if out is None:
        out = torch.zeros(sorted_token_ids.shape[0], N,
                          dtype=torch.float32, device=x.device)
    grid = (num_slots, N // 64)
    moe_marlin_gemv_kernel[grid](
        x, w_packed, scales, zp_packed, out,
        sorted_token_ids, expert_ids, topk_weights,
        K=K, N=N, G=G, TOK_BLOCK=tok_block, TOK=tok,
        num_valid_tokens=num_valid_tokens,
        stride_we=w_packed.stride(0), stride_se=scales.stride(0),
        stride_ze=zp_packed.stride(0),
        MUL_ROUTED=mul_routed_weight, x_row_div=x_row_div,
        NUM_KT=K // 16, GT=G // 16,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out
