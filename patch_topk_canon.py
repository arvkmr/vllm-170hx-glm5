#!/usr/bin/env python3
"""Production fix: canonical tie-break for decode topk (GLM52_TOPK_DET).

Installs glm52_topk_canon.py as vllm/_glm52_topk_canon.py and upgrades the
patch_topk_det.py hook into a mode dispatch:

  GLM52_TOPK_DET=canon   canonical tie-break kernel (score desc, index asc),
                         0.24 ms worst-case per layer call -- the fix.
  GLM52_TOPK_DET=torch   stable torch.topk overwrite (validation baseline).
  unset                  legacy: GLM52_TOPK_TORCH_DET=1 still means torch.

Requires patch_topk_det.py applied first (this edits its block).
See NOTES_longctx_copy_fidelity.md for the root cause.
"""

import os
import shutil
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


shutil.copyfile(
    "/home/user/vllm_install/glm52_topk_canon.py",
    os.path.join(VLLM, "_glm52_topk_canon.py"),
)
print("  + installed vllm/_glm52_topk_canon.py")

edit(
    SAI,
    '        if _os.environ.get("GLM52_TOPK_TORCH_DET", "0") == "1":\n'
    "            # Pure-GPU, sync-free, capture-safe: bakes into FULL cudagraphs\n"
    "            # so replays are deterministic too. Rows with ctx <= topk keep\n"
    "            # the kernel output via the where().\n"
    "            _sl = seq_lens.reshape(-1)[:num_rows].to(logits.device)\n"
    "            _long = (_sl > topk_tokens).unsqueeze(1)\n"
    "            _L = logits.shape[1]\n"
    "            _ar = torch.arange(_L, device=logits.device)\n"
    "            _lm = logits[:num_rows].masked_fill(\n"
    '                _ar.unsqueeze(0) >= _sl.unsqueeze(1), float("-inf")\n'
    "            )\n"
    "            _cand = torch.topk(_lm, topk_tokens, dim=-1).indices.to(torch.int32)\n"
    "            topk_indices[:num_rows].copy_(\n"
    "                torch.where(_long, _cand, topk_indices[:num_rows])\n"
    "            )",
    '        _det = _os.environ.get("GLM52_TOPK_DET", "")\n'
    '        if not _det and _os.environ.get("GLM52_TOPK_TORCH_DET", "0") == "1":\n'
    '            _det = "torch"\n'
    '        if _det == "canon":\n'
    "            # Canonical tie-break (score desc, index asc): rebuilds the\n"
    "            # unique deterministic set from any value-exact kernel output.\n"
    "            # Pure GPU, sync-free, capture-safe (bakes into FULL graphs).\n"
    "            from vllm import _glm52_topk_canon\n"
    "            _sl = (\n"
    "                seq_lens.reshape(-1)[:num_rows]\n"
    "                .to(device=logits.device, dtype=torch.int32)\n"
    "            )\n"
    "            _glm52_topk_canon.canon_topk_indices(\n"
    "                logits[:num_rows], topk_indices[:num_rows], _sl\n"
    "            )\n"
    '        elif _det == "torch":\n'
    "            # Stable torch.topk overwrite -- validation baseline.\n"
    "            _sl = seq_lens.reshape(-1)[:num_rows].to(logits.device)\n"
    "            _long = (_sl > topk_tokens).unsqueeze(1)\n"
    "            _L = logits.shape[1]\n"
    "            _ar = torch.arange(_L, device=logits.device)\n"
    "            _lm = logits[:num_rows].masked_fill(\n"
    '                _ar.unsqueeze(0) >= _sl.unsqueeze(1), float("-inf")\n'
    "            )\n"
    "            _cand = torch.topk(_lm, topk_tokens, dim=-1).indices.to(torch.int32)\n"
    "            topk_indices[:num_rows].copy_(\n"
    "                torch.where(_long, _cand, topk_indices[:num_rows])\n"
    "            )",
    "indexer: topk determinism mode dispatch (canon|torch)",
)

print("Modes: GLM52_TOPK_DET=canon (fix) | torch (baseline); legacy "
      "GLM52_TOPK_TORCH_DET=1 = torch.")
