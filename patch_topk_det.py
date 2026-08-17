#!/usr/bin/env python3
"""Deterministic decode topk (GLM52_TOPK_TORCH_DET=1) — interim fix.

ROOT CAUSE (NOTES_longctx_copy_fidelity.md): the DSA indexer logits contain
exact ties (fp8-quantized keys, repeated text), and both decode topk kernels
break ties nondeterministically — measured offline: persistent_topk returns
~1,900/2,048 different indices between back-to-back calls on tied scores;
topKPerRowDecode ~100. The selected score multiset is always exact, but the
SET of tied boundary candidates is a per-call lottery. Downstream this makes
sparse attention nondeterministic per step, which under MTP cascades into
accept-pattern lotteries and, under pipelined load, into margin collapse and
near-tie token flips (the long-context copy corruption).

torch.topk (CUB radix) is measured run-to-run stable on these shapes. This
patch, when GLM52_TOPK_TORCH_DET=1, overwrites the kernel-produced decode
topk rows (only rows with context > topk; shorter rows use the identity
shortcut and never read logits) with torch.topk results. ~0.5 ms per indexer
layer call at 262K width — acceptable for validation and as an interim fix;
the production fix is a canonical tie-breaking kernel.
"""

import os
import sys

import vllm

VLLM = os.path.dirname(vllm.__file__)
SAI = "model_executor/layers/sparse_attn_indexer.py"


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


edit(
    SAI,
    "        _glm52_idx_dump(k_cache_prefix, topk_indices, seq_lens, num_rows)",
    "        # GLM52_TOPK_TORCH_DET=1: replace the kernel's tie-lottery sets\n"
    "        # with run-to-run-stable torch.topk selections (see\n"
    "        # patch_topk_det.py). Rows with context <= topk keep the identity\n"
    "        # shortcut and are untouched.\n"
    '        if _os.environ.get("GLM52_TOPK_TORCH_DET", "0") == "1":\n'
    "            # Pure-GPU, sync-free, capture-safe: bakes into FULL cudagraphs\n"
    "            # so replays are deterministic too. Rows with ctx <= topk keep\n"
    "            # the kernel output via the where().\n"
    "            _sl = seq_lens.reshape(-1)[:num_rows].to(logits.device)\n"
    "            _long = (_sl > topk_tokens).unsqueeze(1)\n"
    "            _L = logits.shape[1]\n"
    "            _ar = torch.arange(_L, device=logits.device)\n"
    "            _lm = logits[:num_rows].masked_fill(\n"
    "                _ar.unsqueeze(0) >= _sl.unsqueeze(1), float(\"-inf\")\n"
    "            )\n"
    "            _cand = torch.topk(_lm, topk_tokens, dim=-1).indices.to(torch.int32)\n"
    "            topk_indices[:num_rows].copy_(\n"
    "                torch.where(_long, _cand, topk_indices[:num_rows])\n"
    "            )\n"
    "        _glm52_idx_dump(k_cache_prefix, topk_indices, seq_lens, num_rows)",
    "indexer: deterministic decode topk (env-gated)",
)

print("Enable with GLM52_TOPK_TORCH_DET=1.")
