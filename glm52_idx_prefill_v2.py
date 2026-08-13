#!/usr/bin/env python3
"""v2 prefill MQA logits kernel for the DSA indexer (sm80).

The prefill twin of mqa_logits_v2.py. That rewrite fixed the *decode* kernel
(2.8x); the prefill kernel in vllm/v1/attention/ops/mqa_logits_triton.py never
got the same treatment and has the identical flaw:

    grid = (M, cdiv(N, BLOCK_N))        # one program per query token
    s = tl.dot(q, tl.trans(k))          # [32 heads, 128] x [128, BLOCK_N]

With one program per query row the MMA's M dimension is just num_heads (32),
and every query row re-reads the whole K tile on its own. The fingerprint is
that cost per query token is *flat* in the chunk size M -- measured 137 ns vs
124 ns per (token x 1K keys) going from M=128 to M=512, i.e. no economy of
scale at all, because queries never share a K tile.

That leaves the kernel at ~60 TFLOP/s.

The fix: BLOCK_M query rows per program, so one [BLOCK_M*H, D] x [D, BLOCK_N]
MMA covers BLOCK_M queries and they share each K tile instead of re-reading it
one row at a time. Row r of the dot is (m = r // H, h = r % H); the per-head
relu/weight reduction is a reshape + sum over axis 1, exactly as in decode v2.

Measured on CMP 170HX at N=262144: 137 -> 105 ns per (query token x 1K keys)
at M=128, and 124 -> 97 ns at M=512. **1.28-1.31x, bit-exact vs v1** (max rel
0.0 -- same operand dtypes, and the fp32 accumulation happens to land
identically). Autotune settles on BLOCK_M=2, BLOCK_N=128, 2 warps.

Two things that did NOT work, recorded so they are not retried:

  * Decoding e4m3 in-kernel with the bit-trick, to skip the wrapper's
    `k_fp8.to(bfloat16)` (which re-converts the whole N-token workspace on
    every sub-chunk call, ~256 MB temporary each time). That is a *2x
    regression* -- 105 -> 263 ns. The comment in v1 warning that in-kernel
    decode costs ~2x at prefill was measured against a LUT gather, but the
    conclusion holds for the bit-trick too: the decode contends with the
    matmul for ALU and registers, and the wrapper's one-shot pass is only
    ~2% of call time anyway. Keep the pre-decode.
  * Doing the head reduction as a second MMA with a weight-selector matrix,
    to dodge the reshape's layout change: 100 -> 103 ns, slightly worse, and
    it gives up bit-exactness. reshape + sum is fine here.

The remaining gap to the card's 187 TFLOP/s dense-GEMM ceiling is structural,
not tiling: the contraction is only K=128 and every output element carries a
32-way cross-lane head reduction. Closing it needs less work, not better
blocking.

Contract is identical to fp8_mqa_logits_triton: fp32 [M, N] logits, -inf
outside each row's [ks, ke), and with clean_logits=False the caller's
untouched region is still written (the indexer top-k reads only [ks, ke)).
"""

import torch

from vllm.triton_utils import tl, triton


_MAX_BLOCK_M = 8

_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": bm, "BLOCK_N": bn}, num_warps=nw, num_stages=ns)
    for bm in (1, 2, 4, 8)
    for bn in (32, 64, 128)
    for nw in (2, 4, 8)
    for ns in (2, 3)
    # [BLOCK_M*32, BLOCK_N] fp32 accumulator; cap it so register pressure
    # doesn't collapse occupancy the way it did for the decode v2.
    if bm * 32 * bn <= 16384
]


# M_BUCKET, not M, and not omitted entirely. v1 keys on (heads, dim) alone
# because BLOCK_N is its only knob and per-program work is N-independent. That
# does not carry over: BLOCK_M's optimum depends on how many query rows there
# actually are, so keying without it lets whichever shape happens to run first
# pin the config -- a warmup or test at M=1 selects BLOCK_M=1 and silently
# turns this back into v1. Keying on M itself would re-tune on every chunk,
# since chunked prefill varies M constantly. Bucketing to a power of two
# capped at the largest BLOCK_M gives at most four tunings.
@triton.autotune(
    configs=_AUTOTUNE_CONFIGS, key=["num_heads", "head_dim", "M_BUCKET"]
)
@triton.jit
def _fp8_mqa_logits_v2_kernel(
    q_ptr,  # [M, H, D] bf16 (wrapper pre-decodes; see docstring)
    k_ptr,  # [N, D]    bf16
    k_scale_ptr,  # [N]       f32
    weights_ptr,  # [M, H]    f32
    ks_ptr,  # [M]       i32
    ke_ptr,  # [M]       i32
    logits_ptr,  # [M, N]    f32
    stride_q_m,
    stride_q_h,
    stride_q_d,
    stride_k_n,
    stride_k_d,
    stride_w_m,
    stride_w_h,
    stride_l_m,
    stride_l_n,
    M,
    N,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_D: tl.constexpr,
    M_BUCKET: tl.constexpr,  # autotune key only; see decorator
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    m_block = tl.program_id(0)
    n_block = tl.program_id(1)

    offs_m = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    offs_n = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    ks = tl.load(ks_ptr + offs_m, mask=mask_m, other=0)
    ke = tl.load(ke_ptr + offs_m, mask=mask_m, other=0)

    # Block-level early exit. v1 could test one row; here the tile is skipped
    # only if it misses *every* row in the block. ke is nondecreasing in m
    # under causal prefill, so the widened range costs almost nothing, and
    # rows that individually miss still get -inf from `valid` below.
    n_start = n_block * BLOCK_N
    if (n_start >= tl.max(ke)) | (n_start + BLOCK_N <= tl.min(ks)):
        tl.store(
            logits_ptr + offs_m[:, None] * stride_l_m + offs_n[None, :] * stride_l_n,
            tl.full([BLOCK_M, BLOCK_N], float("-inf"), dtype=tl.float32),
            mask=mask_m[:, None] & mask_n[None, :],
        )
        return

    # Row r of the dot is query (r // H), head (r % H) -- the ordering the
    # reshape below relies on.
    BLOCK_R: tl.constexpr = BLOCK_M * num_heads
    offs_r = tl.arange(0, BLOCK_R)
    r_m = offs_r // num_heads
    r_h = offs_r % num_heads
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < head_dim
    mask_r = (m_block * BLOCK_M + r_m) < M

    q = tl.load(
        q_ptr
        + (m_block * BLOCK_M + r_m)[:, None] * stride_q_m
        + r_h[:, None] * stride_q_h
        + offs_d[None, :] * stride_q_d,
        mask=mask_r[:, None] & mask_d[None, :],
        other=0.0,
    )
    w = tl.load(
        weights_ptr
        + (m_block * BLOCK_M + r_m) * stride_w_m
        + r_h * stride_w_h,
        mask=mask_r,
        other=0.0,
    )

    k = tl.load(
        k_ptr + offs_n[:, None] * stride_k_n + offs_d[None, :] * stride_k_d,
        mask=mask_n[:, None] & mask_d[None, :],
        other=0.0,
    )
    k_scale = tl.load(k_scale_ptr + offs_n, mask=mask_n, other=0.0)

    # One MMA for BLOCK_M queries: [BLOCK_M*H, D] x [D, BLOCK_N].
    s = tl.dot(q, tl.trans(k)) * k_scale[None, :]
    s = tl.where(s > 0, s, 0.0) * w[:, None]

    s3 = tl.reshape(s, (BLOCK_M, num_heads, BLOCK_N))
    out = tl.sum(s3, axis=1)  # [BLOCK_M, BLOCK_N]

    valid = (offs_n[None, :] >= ks[:, None]) & (offs_n[None, :] < ke[:, None])
    out = tl.where(valid, out, float("-inf"))

    tl.store(
        logits_ptr + offs_m[:, None] * stride_l_m + offs_n[None, :] * stride_l_n,
        out,
        mask=mask_m[:, None] & mask_n[None, :],
    )


def fp8_mqa_logits_v2(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    clean_logits: bool = True,
) -> torch.Tensor:
    """Drop-in replacement for fp8_mqa_logits_triton.

    Args:
        q:            [M, H, D] fp8_e4m3fn
        kv:           (k_fp8 [N, D] fp8_e4m3fn, k_scales [N] float32)
        weights:      [M, H] float32
        cu_seqlen_ks: [M] int32, per-row first valid key
        cu_seqlen_ke: [M] int32, per-row end of valid keys
        clean_logits: pre-fill with -inf. The kernel writes every element of
            [M, N] either way, so this only matters for callers that read
            outside [ks, ke); kept for signature compatibility.
    Returns:
        logits: [M, N] float32
    """
    k_fp8, k_scales = kv
    k_scales = k_scales.reshape(-1).float()

    M, num_heads, head_dim = q.shape
    N = k_fp8.shape[0]

    logits = torch.empty((M, N), dtype=torch.float32, device=q.device)

    # Pre-decode FP8 -> bf16, same as v1. Doing this in-kernel instead costs
    # 2x (see docstring); this pass is ~2% of call time.
    q_bf16 = q.to(torch.bfloat16)
    k_bf16 = k_fp8.to(torch.bfloat16)

    grid = lambda meta: (  # noqa: E731
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )
    _fp8_mqa_logits_v2_kernel[grid](
        q_bf16,
        k_bf16,
        k_scales,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        logits,
        q_bf16.stride(0),
        q_bf16.stride(1),
        q_bf16.stride(2),
        k_bf16.stride(0),
        k_bf16.stride(1),
        weights.stride(0),
        weights.stride(1),
        logits.stride(0),
        logits.stride(1),
        M,
        N,
        num_heads=num_heads,
        head_dim=head_dim,
        BLOCK_D=triton.next_power_of_2(head_dim),
        M_BUCKET=min(_MAX_BLOCK_M, triton.next_power_of_2(M)),
    )
    return logits


def warmup_fp8_mqa_logits_v2(
    num_heads: int,
    head_dim: int,
    device,
) -> None:
    """Prime the autotune cache at init, once per M bucket.

    Without this the sweep runs inline on the first real chunk, at the real
    (huge) N -- 83 s on this box, which trips vLLM's `sample_tokens` RPC
    timeout and takes the engine down. v1 warms for the same reason; the
    difference here is that the key includes M_BUCKET, so every bucket that
    can occur at serving time has to be primed, not just one shape.

    N is a runtime scalar and per-program work is N-independent, so a short
    warmup N transfers to any chunk length -- the same assumption v1 makes.
    """
    max_block_n = max(c.kwargs["BLOCK_N"] for c in _AUTOTUNE_CONFIGS)
    n = max(4096, max_block_n)
    k = torch.empty(n, head_dim, dtype=torch.float8_e4m3fn, device=device)
    scales = torch.zeros(n, dtype=torch.float32, device=device)
    m = 1
    while m <= _MAX_BLOCK_M:
        q = torch.empty(
            m, num_heads, head_dim, dtype=torch.float8_e4m3fn, device=device
        )
        weights = torch.zeros(m, num_heads, dtype=torch.float32, device=device)
        ks = torch.zeros(m, dtype=torch.int32, device=device)
        ke = torch.full((m,), n, dtype=torch.int32, device=device)
        fp8_mqa_logits_v2(q, (k, scales), weights, ks, ke, clean_logits=False)
        m *= 2

    # Hand the blocks back. Triton's do_bench allocates a large L2-flush buffer
    # per sweep, and the caching allocator would hold it for the rest of the
    # process. At the 1M-context KV budget there is only ~270 MiB of slack, and
    # keeping it starves the indexer's own 512 MB peak-logits reserve during
    # cudagraph capture -- which OOMs the engine at startup.
    del k, scales
    torch.cuda.empty_cache()
