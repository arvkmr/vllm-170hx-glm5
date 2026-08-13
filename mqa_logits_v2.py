#!/usr/bin/env python3
"""v2 paged MQA logits kernel for the DSA indexer (sm80).

Fixes vs the PR #38476 fallback at GLM-5.2 shapes (32 heads x 128 dim,
block 64, next_n=4 MTP rows):
  1. batch-major persistent grid (B, TILES) with runtime loop bounds --
     graph-safe, no empty-CTA storm (v1: (rows, max_blocks) grid).
  2. K decoded once per tile and shared by all spec rows via MMA (v1:
     each row re-read + re-decoded K).
  3. e4m3 bit-trick decode instead of LUT gathers.
  4. tl.range software pipelining (while loops are never pipelined).
  5. Occupancy: no persistent q tile at all -- q bytes are re-loaded
     (L1-hot) and decoded per pass inside the loop; the [BLOCK_H, N] f32
     accumulator is halved by SPLIT row-passes per K tile. ncu showed the
     single-pass version pinned at 168 regs / 18.75% occupancy.

Exact same contract as fp8_paged_mqa_logits_triton (fp32 logits, -inf
outside causal/valid, rows with context_len <= skip_le untouched).
next_n must be a power of two (1, 2, 4); others fall back to v1.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _decode_e4m3_bittrick(u8):
    """e4m3 byte -> fp16 without LUT gathers: sign to bit 15, exp+mant to
    bits 13..7, reinterpret as fp16, scale by 2^8. Exact for all normal
    and denormal e4m3; the NaN encodings (0x7f/0xff) decode to +-496
    instead of the LUT's +-480 -- the indexer k-quantizer saturates to
    +-448 and never emits them."""
    u = u8.to(tl.uint16)
    bits = ((u & 0x80) << 8) | ((u & 0x7F) << 7)
    return bits.to(tl.float16, bitcast=True) * 256.0


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=nw, num_stages=ns)
        for nw in (4, 8)
        for ns in (2, 3)
    ],
    key=["num_heads", "head_dim", "block_size", "next_n"],
)
@triton.jit
def _paged_mqa_logits_v2_kernel(
    q_ptr,            # [B, next_n, H, D] uint8 (fp8 bytes)
    kv_fp8_ptr,       # [NB, BLOCK, D] uint8
    kv_scale_ptr,     # [NB, BLOCK] f32
    weights_ptr,      # [B*next_n, H] f32
    context_lens_ptr,  # [B] i32
    block_tables_ptr,  # [B, max_blocks] i32
    logits_ptr,        # [B*next_n, L_MAX] f32
    stride_q_b, stride_q_n, stride_q_h, stride_q_d,
    stride_kvf_block, stride_kvf_s, stride_kvf_d,
    stride_kvs_block, stride_kvs_s,
    stride_w_t, stride_w_h,
    stride_bt_b, stride_bt_k,
    stride_l_t, stride_l_n,
    L_MAX,
    SKIP_LE,
    TILES,
    next_n: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,   # block_size (pow2)
):
    BLOCK_H2: tl.constexpr = next_n * num_heads

    batch_id = tl.program_id(0)
    tile_id = tl.program_id(1)

    context_len = tl.load(context_lens_ptr + batch_id)
    if context_len <= SKIP_LE:
        return

    n_blocks = (context_len + block_size - 1) // block_size

    offs_r = tl.arange(0, BLOCK_H2)
    offs_d = tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, BLOCK_N)
    rp_rows = tl.arange(0, next_n)
    mask_d = offs_d < head_dim
    mask_n = offs_n < block_size
    q_base = q_ptr + batch_id * stride_q_b
    r_h = offs_r % num_heads
    r_n = offs_r // num_heads

    # load q ONCE as bytes (16KB tile); decode happens inside the loop so
    # the fp16 operand's register lifetime overlaps K's (occupancy).
    q_byte = tl.load(
        q_base
        + r_n[:, None] * stride_q_n
        + r_h[:, None] * stride_q_h
        + offs_d[None, :] * stride_q_d,
        mask=mask_d[None, :],
        other=0,
    )
    w = tl.load(
        weights_ptr
        + (batch_id * next_n + r_n) * stride_w_t
        + r_h * stride_w_h,
    )
    q_off = context_len - next_n + rp_rows

    # Two consecutive KV blocks per iteration: [2*BLOCK_N, D] K tile, one
    # [BLOCK_H2, D] x [D, 2*BLOCK_N] dot -- half the iterations, wider MMA.
    offs_n2 = tl.arange(0, 2 * BLOCK_N)
    n_pairs = (n_blocks + 1) // 2
    n_iter = (n_pairs - tile_id + TILES - 1) // TILES
    for it in tl.range(0, n_iter, num_stages=2):
        pblk = tile_id + it * TILES
        blk = 2 * pblk
        bidx_a = tl.load(
            block_tables_ptr + batch_id * stride_bt_b + blk * stride_bt_k
        )
        nb = tl.minimum(blk + 1, n_blocks - 1)
        bidx_b = tl.load(
            block_tables_ptr + batch_id * stride_bt_b + nb * stride_bt_k
        )
        row_base = tl.where(
            offs_n2 < BLOCK_N,
            bidx_a.to(tl.int64) * stride_kvf_block,
            bidx_b.to(tl.int64) * stride_kvf_block,
        )
        row_in_blk = offs_n2 % BLOCK_N
        k_byte = tl.load(
            kv_fp8_ptr
            + row_base[:, None]
            + row_in_blk[:, None] * stride_kvf_s
            + offs_d[None, :] * stride_kvf_d,
            mask=mask_d[None, :],
            other=0,
        )
        k = _decode_e4m3_bittrick(k_byte)
        srow_base = tl.where(
            offs_n2 < BLOCK_N,
            bidx_a.to(tl.int64) * stride_kvs_block,
            bidx_b.to(tl.int64) * stride_kvs_block,
        )
        k_scale = tl.load(
            kv_scale_ptr + srow_base + row_in_blk * stride_kvs_s,
        )
        k_offset = blk * block_size + offs_n2  # [2*BLOCK_N]

        q = _decode_e4m3_bittrick(q_byte)

        # [BLOCK_H2, D] x [D, BLOCK_N] -> [BLOCK_H2, BLOCK_N]
        s = tl.dot(q, tl.trans(k)) * k_scale[None, :]
        s = tl.where(s > 0, s, 0.0) * w[:, None]

        s3 = tl.reshape(s, (next_n, num_heads, 2 * BLOCK_N))
        out = tl.sum(s3, axis=1)  # [next_n, 2*BLOCK_N]

        valid = (
            (k_offset[None, :] < context_len)
            & (k_offset[None, :] <= q_off[:, None])
        )
        out = tl.where(valid, out, float("-inf"))

        tl.store(
            logits_ptr
            + (batch_id * next_n + rp_rows)[:, None] * stride_l_t
            + k_offset[None, :] * stride_l_n,
            out,
            mask=(k_offset[None, :] < context_len)
            & (k_offset[None, :] < L_MAX),
        )


def fp8_paged_mqa_logits_v2(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
    skip_le: int = 0,
    tiles: int | None = None,
) -> torch.Tensor:
    B, next_n, num_heads, head_dim = q.shape
    if next_n & (next_n - 1):
        # non-pow2 next_n (spec k=2): keep the reference path
        from vllm.v1.attention.ops.mqa_logits_triton import (
            fp8_paged_mqa_logits_triton,
        )
        return fp8_paged_mqa_logits_triton(
            q, kv_cache, weights, context_lens, block_tables,
            max_model_len=max_model_len, clean_logits=False,
            skip_le=skip_le,
        )
    _, block_size, one, d_plus_4 = kv_cache.shape
    assert one == 1 and d_plus_4 == head_dim + 4

    num_blocks = kv_cache.shape[0]
    kv_flat = kv_cache.view(num_blocks, -1)
    k_end = block_size * head_dim
    kv_byte = kv_flat[:, :k_end].as_strided(
        (num_blocks, block_size, head_dim),
        (kv_flat.stride(0), head_dim, 1),
    )
    kv_scale = kv_flat[:, k_end:].view(torch.float32)
    q_byte = q.view(torch.uint8)

    logits = torch.empty(
        (B * next_n, max_model_len), dtype=torch.float32, device=q.device
    )
    if tiles is None:
        # Swept on CMP 170HX (70 SMs): B=8@196K best at 256, B=1 flat.
        tiles = max(32, 2048 // max(B, 1))
    grid = (B, tiles)
    _paged_mqa_logits_v2_kernel[grid](
        q_byte, kv_byte, kv_scale, weights,
        context_lens, block_tables, logits,
        q_byte.stride(0), q_byte.stride(1), q_byte.stride(2),
        q_byte.stride(3),
        kv_byte.stride(0), kv_byte.stride(1), kv_byte.stride(2),
        kv_scale.stride(0), kv_scale.stride(1),
        weights.stride(0), weights.stride(1),
        block_tables.stride(0), block_tables.stride(1),
        logits.stride(0), logits.stride(1),
        max_model_len,
        skip_le,
        tiles,
        next_n=next_n,
        num_heads=num_heads,
        head_dim=head_dim,
        block_size=block_size,
        BLOCK_D=triton.next_power_of_2(head_dim),
        BLOCK_N=triton.next_power_of_2(block_size),
    )
    return logits
