#!/usr/bin/env python3
"""Deterministic moe_align_block_size (GLM52_MOE_ALIGN_DET, default on).

The CUDA moe_align_block_size op orders tokens within each expert segment by
atomic-increment arrival order -- nondeterministic run to run. Downstream,
the fused Marlin MoE's work geometry (stripe/tile boundaries) follows that
ordering, so expert outputs differ bitwise between identical calls
(measured: fused_marlin_moe on an identical batch twice -> different bits).
Through 92 layers this amplifies into the load-dependent verify-margin
wobble that survived the topk tie-break fix
(NOTES_longctx_copy_fidelity.md).

Fix: env-gated torch reimplementation ordered by (expert, token index) via
stable argsort -- fully deterministic, pure GPU, capture-safe, ~50-100 us at
mixed-batch sizes (16K flat ids). Semantics match the CUDA op's documented
contract: per-expert segments padded to block_size, pad value = numel,
expert_ids per block, num_tokens_post_padded.
"""

import os
import sys

import vllm

VLLM = os.path.dirname(vllm.__file__)
MAB = "model_executor/layers/fused_moe/moe_align_block_size.py"


def edit(path, old, new, label):
    full = os.path.join(VLLM, path)
    src = open(full).read()
    if new in src:
        print(f"  = {label} (already applied)")
        return
    if old not in src:
        print(f"  ! {label}: ANCHOR NOT FOUND", file=sys.stderr)
        sys.exit(1)
    if src.count(old) != 1:
        print(f"  ! {label}: anchor not unique", file=sys.stderr)
        sys.exit(1)
    open(full, "w").write(src.replace(old, new))
    print(f"  + {label}")


HELPER = '''

import os as _os

_GLM52_MOE_ALIGN_DET = _os.environ.get("GLM52_MOE_ALIGN_DET", "1") == "1"


def _moe_align_block_size_det(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
) -> None:
    """Deterministic (expert, token-index)-ordered fill of the align buffers.

    See patch_moe_align_det.py. Pure GPU ops, no host syncs; safe under
    CUDA graph capture. Matches the CUDA op's contract with pad = numel.
    """
    flat = topk_ids.flatten().to(torch.int64)
    T = flat.numel()
    order = torch.argsort(flat, stable=True)
    counts = torch.zeros(num_experts, dtype=torch.int64, device=flat.device)
    counts.scatter_add_(0, flat, torch.ones_like(flat))
    padded = ((counts + block_size - 1) // block_size) * block_size
    seg_start = torch.cumsum(padded, 0) - padded
    csum = torch.cumsum(counts, 0) - counts
    expert_of_sorted = flat[order]
    within = (
        torch.arange(T, device=flat.device, dtype=torch.int64)
        - csum[expert_of_sorted]
    )
    dest = seg_start[expert_of_sorted] + within
    sorted_ids.fill_(T)
    sorted_ids[dest] = order.to(torch.int32)
    blocks_per_e = padded // block_size
    cumb = torch.cumsum(blocks_per_e, 0)
    num_tokens_post_pad.copy_(
        (cumb[-1] * block_size).to(torch.int32).reshape(1)
    )
    bidx = torch.arange(expert_ids.numel(), device=flat.device)
    eb = torch.searchsorted(cumb, bidx, right=True).clamp(max=num_experts - 1)
    expert_ids.copy_(
        torch.where(bidx < cumb[-1], eb, torch.zeros_like(eb)).to(torch.int32)
    )
'''

edit(
    MAB,
    "def moe_align_block_size(",
    HELPER + "\n\ndef moe_align_block_size(",
    "moe_align: deterministic helper",
)

edit(
    MAB,
    "    ops.moe_align_block_size(\n"
    "        topk_ids,\n"
    "        num_experts,\n"
    "        block_size,\n"
    "        sorted_ids,\n"
    "        expert_ids,\n"
    "        num_tokens_post_pad,\n"
    "        expert_map if ignore_invalid_experts else None,\n"
    "    )",
    "    if _GLM52_MOE_ALIGN_DET and expert_map is None:\n"
    "        # (ignore_invalid_experts is a no-op without an expert_map: every\n"
    "        # topk id is valid, so both semantics coincide.)\n"
    "        # Deterministic path (patch_moe_align_det.py): the CUDA op's\n"
    "        # atomic arrival order makes downstream Marlin MoE outputs\n"
    "        # nondeterministic bitwise; stable (expert, index) order fixes it.\n"
    "        _moe_align_block_size_det(\n"
    "            topk_ids, block_size, num_experts,\n"
    "            sorted_ids, expert_ids, num_tokens_post_pad,\n"
    "        )\n"
    "    else:\n"
    "        ops.moe_align_block_size(\n"
    "            topk_ids,\n"
    "            num_experts,\n"
    "            block_size,\n"
    "            sorted_ids,\n"
    "            expert_ids,\n"
    "            num_tokens_post_pad,\n"
    "            expert_map if ignore_invalid_experts else None,\n"
    "        )",
    "moe_align: deterministic dispatch",
)

print("GLM52_MOE_ALIGN_DET=1 (default) uses the deterministic path; "
      "set 0 to restore the CUDA op.")
