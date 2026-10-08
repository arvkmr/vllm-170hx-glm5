# SPDX-License-Identifier: Apache-2.0
"""Canonical tie-breaking for DSA decode topk (sm_80 production fix).

The decode topk kernels (persistent_topk, topKPerRowDecode) are value-exact
but break score ties nondeterministically: among boundary candidates with
exactly equal scores, the selected SET is a per-call lottery (measured:
~1,900/2,048 indices differ between back-to-back persistent_topk calls on
tied fp8-derived logits). Downstream sparse attention then wobbles run to
run, which under MTP + PP pipelining cascades into visible temp-0 token
flips (NOTES_longctx_copy_fidelity.md).

This kernel rebuilds the canonical selection from any value-exact topk
output, defining ground truth as: order by (score desc, index asc).

  v*        = min score among the kernel's selected entries (the boundary)
  keep      = selected entries with score > v*   (this SET is unique -- any
              value-exact kernel returns exactly these, in some order)
  canonical = keep  +  the lowest-index positions with score == v*, taken
              in index order until K entries total

One program per row: the 2048 selected entries are loaded once (8 KB), the
row is scanned in BLOCK_N chunks with an early exit once the tied slots are
filled. Ties come from repeated text, so the scan usually exits early. Pure
GPU, no host syncs -- safe inside CUDA graph capture, so FULL-graph replays
are deterministic too.

Cost: ~30-60 us/row worst case (one 1 MB scan at 262K row width), rows in
parallel CTAs. Per indexer layer call at decode shapes: ~0.1 ms.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _topk_canon_kernel(
    logits_ptr,      # [rows, L] fp32
    sel_ptr,         # [rows, K] int32  (kernel topk output, value-exact)
    out_ptr,         # [rows, K] int32  (canonical output)
    seq_lens_ptr,    # [rows] int32
    stride_l_row,
    stride_sel_row,
    stride_out_row,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,   # pow2 >= K
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    L = tl.load(seq_lens_ptr + row)

    offs_k = tl.arange(0, BLOCK_K)
    kmask = offs_k < K
    sel = tl.load(sel_ptr + row * stride_sel_row + offs_k, mask=kmask, other=-1)

    if L <= K:
        # Identity-shortcut rows: emit 0..L-1 then -1 explicitly. This is
        # persistent_topk's output; topKPerRowDecode returns the same set in
        # an unstable order.
        ident = tl.where(offs_k < L, offs_k, -1)
        tl.store(out_ptr + row * stride_out_row + offs_k, ident, mask=kmask)
        return

    # Only in-window entries count (see the prefill kernel).
    valid = kmask & (sel >= 0) & (sel < L)
    ssc = tl.load(
        logits_ptr + row * stride_l_row + tl.where(valid, sel, 0),
        mask=valid,
        other=float("inf"),
    )
    vstar = tl.min(tl.where(valid, ssc, float("inf")))

    # Entries strictly above the boundary: the unique part of the set.
    is_gt = valid & (ssc > vstar)
    n_gt = tl.sum(is_gt.to(tl.int32))
    pos = tl.cumsum(is_gt.to(tl.int32)) - is_gt.to(tl.int32)
    tl.store(out_ptr + row * stride_out_row + pos, sel, mask=is_gt)

    # Boundary-tied slots: refill with the lowest-index positions where
    # score == v*, scanning the row in index order. Early exit when full.
    need = K - n_gt
    filled = 0
    start = 0
    while (start < L) & (filled < need):
        offs = start + tl.arange(0, BLOCK_N)
        m = offs < L
        sc = tl.load(
            logits_ptr + row * stride_l_row + offs, mask=m, other=float("inf")
        )
        eq = m & (sc == vstar)
        c = tl.cumsum(eq.to(tl.int32)) - eq.to(tl.int32)
        take = eq & ((filled + c) < need)
        tl.store(
            out_ptr + row * stride_out_row + n_gt + filled + c,
            offs.to(tl.int32),
            mask=take,
        )
        filled += tl.sum(eq.to(tl.int32))
        start += BLOCK_N

    # Canonical ORDER too (index ascending). The score > v* entries above
    # keep the input kernel's order, which is not stable call to call for
    # either persistent_topk or topKPerRowDecode; sparse MLA accumulates in
    # index-list order, so an unstable order is ULP-level nondeterminism.
    tl.debug_barrier()
    row_out = out_ptr + row * stride_out_row + offs_k
    v = tl.load(row_out, mask=kmask, other=2147483647)
    # Unfilled slots (-1) sort last, keeping the -1 pad at the row end.
    v = tl.where(v < 0, 2147483647, v)
    v = tl.sort(v)
    tl.store(row_out, tl.where(v == 2147483647, -1, v), mask=kmask)


@triton.jit
def _topk_canon_prefill_kernel(
    logits_ptr,      # [rows, L] fp32
    sel_ptr,         # [rows, K] int32  (kernel topk output, value-exact)
    out_ptr,         # [rows, K] int32  (canonical SET, order not canonical)
    ks_ptr,          # [rows] int32  window start (inclusive)
    ke_ptr,          # [rows] int32  window end (exclusive)
    stride_l_row,
    stride_sel_row,
    stride_out_row,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,   # pow2 >= K
    BLOCK_N: tl.constexpr,
):
    """Prefill variant: valid positions are [ks, ke) per row.

    nv = ke - ks <= K rows: emit the whole window in index order, -1 pad
    (the set is trivially unique; no boundary exists).
    nv > K rows: keep the (unique) score > v* subset of the kernel's
    selection, refill boundary-tied slots with the lowest-index positions
    where score == v*, scanning [ks, ke) in index order.
    Output ORDER is canonicalized afterwards on the [rows, K] slice by
    canon_topk_indices_prefill (index-asc sort, then stable score-desc
    sort), reproducing torch.topk's (score desc, index asc) order without
    touching the full row width.
    """
    row = tl.program_id(0)
    ks = tl.load(ks_ptr + row)
    ke = tl.load(ke_ptr + row)
    nv = ke - ks

    offs_k = tl.arange(0, BLOCK_K)
    kmask = offs_k < K

    if nv <= K:
        # Whole window, index-ascending, -1 pad.
        v = ks + offs_k
        out = tl.where((offs_k < nv) & kmask, v, -1)
        tl.store(out_ptr + row * stride_out_row + offs_k, out, mask=kmask)
        return

    # sel holds the NATIVE kernel's output: indices RELATIVE to ks.
    # Work in absolute coordinates internally; the wrapper re-relativizes.
    sel = tl.load(sel_ptr + row * stride_sel_row + offs_k, mask=kmask, other=-1)
    # Only in-window native entries count; anything else (pad, or a stale /
    # out-of-window value) is ignored rather than trusted as an index.
    valid = kmask & (sel >= 0) & (sel < nv)
    sel_abs = tl.where(valid, sel + ks, 0)
    ssc = tl.load(
        logits_ptr + row * stride_l_row + sel_abs,
        mask=valid,
        other=float("inf"),
    )
    vstar = tl.min(tl.where(valid, ssc, float("inf")))

    is_gt = valid & (ssc > vstar)
    n_gt = tl.sum(is_gt.to(tl.int32))
    pos = tl.cumsum(is_gt.to(tl.int32)) - is_gt.to(tl.int32)
    tl.store(out_ptr + row * stride_out_row + pos, sel_abs, mask=is_gt)

    need = K - n_gt
    filled = 0
    start = ks
    while (start < ke) & (filled < need):
        offs = start + tl.arange(0, BLOCK_N)
        m = offs < ke
        sc = tl.load(
            logits_ptr + row * stride_l_row + offs, mask=m, other=float("inf")
        )
        eq = m & (sc == vstar)
        c = tl.cumsum(eq.to(tl.int32)) - eq.to(tl.int32)
        take = eq & ((filled + c) < need)
        tl.store(
            out_ptr + row * stride_out_row + n_gt + filled + c,
            offs.to(tl.int32),
            mask=take,
        )
        filled += tl.sum(eq.to(tl.int32))
        start += BLOCK_N


def canon_topk_indices_prefill(
    logits: torch.Tensor,       # [rows, L] fp32
    topk_indices: torch.Tensor, # [rows, >=K] int32, [:, :K] modified in place
    cu_seqlen_ks: torch.Tensor, # [rows] int32
    cu_seqlen_ke: torch.Tensor, # [rows] int32
    k: int,
) -> None:
    """Canonical prefill selection matching the NATIVE kernel's output
    conventions (empirically probed, NOTES SESSION 4 POSTSCRIPT-2):

      1. indices are RELATIVE to cu_seqlen_ks (the consumer,
         triton_convert_req_index_to_global_index, treats them as
         per-request positions into the block table);
      2. rows with window <= K emit IDENTITY order 0..w-1 then -1 pad.

    For window > K rows the set is canonicalized to the unique
    (score desc, index asc) selection and emitted in that order (the
    native kernel's own order is arbitrary/lottery there).
    """
    rows = topk_indices.shape[0]
    if rows == 0:
        return
    k = min(k, logits.shape[1])
    sel = topk_indices[:, :k]
    # -1 (pad), not uninitialized memory: a slot the rebuild does not fill must
    # read as "no token", never as a garbage index (vNext 2026-09-28: an
    # out-of-range garbage slot crashed the gather below on a 57K prompt).
    out = torch.full_like(sel, -1)
    _topk_canon_prefill_kernel[(rows,)](
        logits,
        sel,
        out,
        cu_seqlen_ks,
        cu_seqlen_ke,
        logits.stride(0),
        sel.stride(0),
        out.stride(0),
        K=k,
        BLOCK_K=triton.next_power_of_2(k),
        BLOCK_N=8192,
        num_warps=8,
    )
    # Canonical order on the K-wide slice (window > K rows only): index-
    # ascending first (-1 -> +inf sentinel so pads sink), then STABLE
    # score-descending -- equal scores keep index-ascending order.
    sentinel = torch.iinfo(torch.int32).max
    idx_sorted = torch.where(out < 0, out.new_full((), sentinel), out).sort(
        dim=-1
    ).values
    pad = idx_sorted == sentinel
    gath = torch.where(pad, idx_sorted.new_zeros(()), idx_sorted)
    scores = logits.gather(1, gath.long())
    scores = torch.where(pad, scores.new_full((), float("-inf")), scores)
    order = scores.sort(dim=-1, descending=True, stable=True).indices
    canon = idx_sorted.gather(1, order)
    canon = torch.where(pad.gather(1, order), canon.new_full((), -1), canon)
    # Short rows keep the kernel's identity order (native convention);
    # the kernel wrote them absolute (ks + 0..w-1), like everything else.
    ksc = cu_seqlen_ks[:rows].unsqueeze(1)
    kec = cu_seqlen_ke[:rows].unsqueeze(1)
    final = torch.where((kec - ksc) <= k, out, canon)
    # Native convention 1: emit indices RELATIVE to cu_seqlen_ks.
    sel.copy_(torch.where(final >= 0, final - ksc, final))


def canon_topk_indices(
    logits: torch.Tensor,       # [rows, L] fp32
    topk_indices: torch.Tensor, # [rows, K] int32, modified in place
    seq_lens: torch.Tensor,     # [rows] int32
) -> None:
    """Replace topk_indices with the canonical (score desc, index asc) set.

    Assumes topk_indices came from a value-exact topk over logits[:, :seq].
    Rows with seq_len <= K get the identity shortcut 0..L-1, -1 padded.
    Output is canonical in set AND order (index ascending), independent of
    which top-k kernel produced the input.
    """
    rows, k = topk_indices.shape
    if rows == 0:
        return
    out = torch.full_like(topk_indices, -1)  # never expose uninitialized slots
    _topk_canon_kernel[(rows,)](
        logits,
        topk_indices,
        out,
        seq_lens,
        logits.stride(0),
        topk_indices.stride(0),
        out.stride(0),
        K=k,
        BLOCK_K=triton.next_power_of_2(k),
        BLOCK_N=8192,
        num_warps=8,
    )
    topk_indices.copy_(out)
