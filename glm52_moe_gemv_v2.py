"""v2 Marlin-packed MoE GEMV: per-k-tile unpermute + tl.dot.

v1 (glm52_moe_decode.py) is index-exact but ~10x slower than vLLM's Marlin.
Its stated "KEY PERFORMANCE IDEA" -- accumulate in Marlin's stored order and
run the 7-D unpermute ONCE at the end -- is what costs it: the accumulator has
to be `[TOK, 128, 8]`, which is 16x the `[TOK, 64]` output tile (64 KB per
program at TOK=16). That spills to local memory, and the spill costs far more
than the permutes it saves. It also forces the inner product to be a broadcast
multiply, leaving the tensor cores idle.

v2 inverts that: unpermute each k-tile's dequantized weights to `[16, 64]`
(k, n) immediately, and accumulate into a `[TOK, 64]` fp32 tile (4 KB). The
inner product then becomes exactly the shape tl.dot wants --
`[TOK, 16] x [16, 64]` -- so it runs on the MMA units instead of the ALUs.

Layout algebra is unchanged from v1 (derived and verified there):
    per (16k x 64n) chunk = 128 int32 words, word w nibble i holds
        k = 8*i0 + 2*pk + i2        i = i2*4 + i1*2 + i0
        n = 16*j + 8*i1 + 2*q + pn  w = q*32 + pn*16 + pk*4 + j
so stored dims (q,pn,pk,j | i2,i1,i0) permute to (i0,pk,i2 | j,i1,q,pn) to
give (k, n). v1 applied that to the accumulator; v2 applies it to the weights.

TOK must equal TOK_BLOCK: the kernel reads the first TOK entries of each
block, so a smaller TOK silently drops tokens (v1 shipped tok=4/block=16,
which breaks as soon as one expert gets more than 4 tokens). TOK is also the
MMA's M dimension, so it must be >= 16 -- use block_size 16.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def moe_marlin_gemv_v2_kernel(
    x_ptr,                 # [M, K] activations (half)
    w_ptr,                 # marlin packed [E, K/16, N*16/8] int32
    s_ptr,                 # marlin scales [E, K/G, N] half
    z_ptr,                 # marlin packed zp [E, K/G, N/8] int32
    y_ptr,                 # out [num_token_slots, N] float32
    sorted_token_ids_ptr,  # [num_slots * TOK_BLOCK]
    expert_ids_ptr,        # [num_slots]
    topk_w_ptr,            # [M * top_k] router weights (float32)
    K: tl.constexpr, N: tl.constexpr, G: tl.constexpr,
    TOK_BLOCK: tl.constexpr,
    TOK: tl.constexpr,
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
    NW: tl.constexpr = (N * 16) // 8

    # [TOK, 64] instead of v1's [TOK, 128, 8] -- 16x smaller, stays in registers.
    acc = tl.zeros((TOK, 64), dtype=tl.float32)

    for gt in range(0, NUM_KT // GT):
        # scales: marlin col = 8*(2q+pn) + 2j+i1 -> S[q,pn,j,i1]
        sv = tl.load(sp + gt * N + tl.arange(0, 64))
        sv = tl.reshape(sv, (4, 2, 4, 2))
        sv = sv[:, :, None, :, None, :, None]
        scale = tl.reshape(
            tl.broadcast_to(sv, (4, 2, 4, 4, 2, 2, 2)), (128, 8)
        ).to(tl.float32)
        zwords = tl.load(zp + gt * (N // 8) + tl.arange(0, 8))
        znib = (zwords[:, None] >> (4 * tl.arange(0, 8))[None, :]) & 0xF
        zv = tl.reshape(znib, (4, 2, 2, 4))
        zv = tl.permute(zv, (0, 1, 3, 2))
        zv = zv[:, :, None, :, None, :, None]
        zero = tl.reshape(
            tl.broadcast_to(zv, (4, 2, 4, 4, 2, 2, 2)), (128, 8)
        ).to(tl.float32)
        sz = scale * zero

        for tt in tl.static_range(GT):
            t = gt * GT + tt
            words = tl.load(wp + t * NW + tl.arange(0, 128))
            nib = ((words[:, None] >> (4 * tl.arange(0, 8))[None, :])
                   & 0xF).to(tl.float32)
            wdeq = nib * scale - sz                               # [128, 8]

            # stored (q,pn,pk,j,i2,i1,i0) -> (i0,pk,i2, j,i1,q,pn) = (k, n).
            w8 = tl.reshape(wdeq, (4, 2, 4, 4, 2, 2, 2))
            w8 = tl.permute(w8, (6, 2, 4, 3, 5, 0, 1))
            w_kn = tl.reshape(w8, (16, 64)).to(tl.float16)

            xk = tl.load(x_ptr + x_rows[:, None] * K + t * 16
                         + tl.arange(0, 16)[None, :],
                         mask=tok_mask[:, None], other=0.0)        # [TOK, 16]
            acc = tl.dot(xk, w_kn, acc)

    y = acc
    if MUL_ROUTED:
        rw = tl.load(topk_w_ptr + toks, mask=tok_mask, other=0.0)
        y = y * rw[:, None]

    nn = chunk * 64 + tl.arange(0, 64)
    tl.store(y_ptr + toks[:, None] * N + nn[None, :], y,
             mask=tok_mask[:, None])


def moe_marlin_gemv_v2(x, w_packed, scales, zp_packed, sorted_token_ids,
                       expert_ids, topk_weights, num_valid_tokens, N, K, G,
                       top_k, mul_routed_weight, x_row_div, out=None,
                       tok_block=16, tok=16, num_warps=4, num_stages=3):
    assert tok == tok_block, (
        f"TOK ({tok}) must equal TOK_BLOCK ({tok_block}); a smaller TOK "
        f"silently drops tokens past the first TOK in a block"
    )
    assert tok >= 16, "TOK is the MMA M dimension; needs >= 16"
    num_slots = expert_ids.shape[0]
    if out is None:
        out = torch.zeros(sorted_token_ids.shape[0], N,
                          dtype=torch.float32, device=x.device)
    grid = (num_slots, N // 64)
    moe_marlin_gemv_v2_kernel[grid](
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
