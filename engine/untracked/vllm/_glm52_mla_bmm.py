"""Per-head batched decode GEMM for MLA's absorbed up-projections on sm_80.

    out[b, h, :] = x[b, h, :K] @ W[h]        W: [H, K, N] bf16, N-contiguous

This is what MLA decode runs as ``torch.bmm`` twice per layer: ``W_UK_T``
(q_nope 192 -> latent 512) and ``W_UV`` (latent 512 -> v 256), 64 heads. At a
decode batch cuBLAS picks a 64x64 sliced tile and streams 12.6 / 16.8 MB of
weight at 0.8-0.9 TB/s. One program per (head, N-tile) with no split-K streams
it at 1.3-1.45 TB/s, measured on CMP 170HX under CUDA-graph replay
(2026-09-29, production strides):

    rows        W_UK_T (192x512)          W_UV (512x256)
       1       15.7 ->  9.1 us           13.8 -> 12.5 us
       8       15.5 ->  9.8 us           18.1 -> 11.5 us
      32       16.5 -> 11.3 us           20.5 -> 16.0 us

Error against an fp32 reference equals cuBLAS's at every size. Each output is
one fixed-order fp32 dot chain, so results are bitwise reproducible. Calls
outside the envelope (more than 32 rows, i.e. prefill; other dtypes, strides
or shapes) fall through to ``torch.bmm`` unchanged. ``GLM52_MLA_HEAD_BMM=0``
disables it.
"""

import os

import torch

from vllm.triton_utils import tl, triton

_ON = os.environ.get("GLM52_MLA_HEAD_BMM", "1") == "1"
_MAX_ROWS = 32

# (K, N) -> [(max_rows, BLOCK_N, BLOCK_K, num_warps, num_stages), ...], the
# best of a bn x bk x warps x stages sweep at 1/4/8/16/32 rows. Pinned, not
# autotuned: boots stay bit-reproducible.
_CONFIGS = {
    (192, 512): [(1, 64, 64, 2, 3), (8, 64, 32, 2, 1), (16, 128, 64, 8, 3), (32, 128, 64, 2, 3)],
    (512, 256): [(1, 64, 128, 2, 1), (4, 64, 128, 2, 2), (8, 128, 128, 4, 2), (16, 32, 32, 2, 1), (32, 128, 64, 2, 1)],
}


@triton.jit
def _head_bmm_kernel(x_ptr, w_ptr, o_ptr, M, sxb, sxh, sob, soh,
                     K: tl.constexpr, N: tl.constexpr, BLOCK_M: tl.constexpr,
                     BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    h = tl.program_id(0)
    pn = tl.program_id(1)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m[:, None] < M
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    w_base = w_ptr + h.to(tl.int64) * K * N
    for k0 in tl.static_range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(x_ptr + offs_m[:, None] * sxb + h * sxh + offs_k[None, :],
                    mask=mask_m, other=0.0)
        w = tl.load(w_base + offs_k[:, None] * N + offs_n[None, :])
        acc += tl.dot(a, w)
    tl.store(o_ptr + offs_m[:, None] * sob + h * soh + offs_n[None, :],
             acc.to(tl.bfloat16), mask=mask_m)


def _config(rows: int, k: int, n: int):
    table = _CONFIGS.get((k, n))
    if table is None:
        return None
    for max_rows, *cfg in table:
        if rows <= max_rows:
            return cfg
    return None


def head_bmm(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor) -> None:
    """``torch.bmm(x, w, out=out)`` for x [H, B, K], w [H, K, N], out [H, B, N].

    Same argument convention as the call sites it replaces: head-major views
    (usually transposes of token-major buffers).
    """
    heads, rows, k = x.shape
    n = w.shape[2]
    cfg = _config(rows, k, n) if _ON and 1 <= rows <= _MAX_ROWS else None
    if (
        cfg is None
        or x.dtype != torch.bfloat16
        or w.dtype != torch.bfloat16
        or out.dtype != torch.bfloat16
        or not w.is_contiguous()
        or tuple(w.shape) != (heads, k, n)
        or tuple(out.shape) != (heads, rows, n)
        or x.stride(2) != 1
        or out.stride(2) != 1
    ):
        torch.bmm(x, w, out=out)
        return
    block_n, block_k, num_warps, num_stages = cfg
    # Kernel indexes token-major: (row stride, head stride) of each view.
    _head_bmm_kernel[(heads, n // block_n)](
        x, w, out, rows, x.stride(1), x.stride(0), out.stride(1), out.stride(0),
        K=k, N=n, BLOCK_M=max(16, triton.next_power_of_2(rows)),
        BLOCK_N=block_n, BLOCK_K=block_k, num_warps=num_warps, num_stages=num_stages,
    )


def _head_bmm_new(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """``torch.bmm(x, w)``: a fresh contiguous [H, B, N] result."""
    out = torch.empty((x.shape[0], x.shape[1], w.shape[2]), dtype=x.dtype, device=x.device)
    head_bmm(x, w, out)
    return out


def _head_bmm_new_fake(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return torch.empty((x.shape[0], x.shape[1], w.shape[2]), dtype=x.dtype, device=x.device)


# Call sites inside a torch.compile-traced forward use the op, which the
# compiler treats as opaque: torch.ops.vllm.glm52_head_bmm_new(x, w).
from vllm.utils.torch_utils import direct_register_custom_op  # noqa: E402

direct_register_custom_op(
    op_name="glm52_head_bmm_new",
    op_func=_head_bmm_new,
    mutates_args=[],
    fake_impl=_head_bmm_new_fake,
)
