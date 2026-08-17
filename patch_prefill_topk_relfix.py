#!/usr/bin/env python3
"""Fix the GLM52_PREFILL_TOPK_DET output convention (both modes).

ops.top_k_per_row_prefill's empirically-probed conventions (NOTES SESSION 4
POSTSCRIPT-2): (1) indices RELATIVE to cu_seqlen_ks -- the consumer
(triton_convert_req_index_to_global_index) treats them as per-request
positions; (2) rows with window <= topk emit IDENTITY order 0..w-1, -1 pad.

The original torch overwrite (and the first canon kernel) emitted ABSOLUTE
workspace indices: wrong for every co-chunked request (ks > 0), shifting
their selections past the true window into (self-padded) block-table tails.
This contaminated the session-4 build-quality numbers and broke the 8K
MTP-off probe.

This patch: reinstalls the fixed _glm52_topk_canon.py (relative emission +
identity short rows, validated against the native kernel by
test_prefill_topk_canon.py) and fixes the torch branch to subtract ks and
emit identity for short rows.
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
    if src.count(old) != 1:
        print(f"  ! {label}: anchor count {src.count(old)} != 1", file=sys.stderr)
        sys.exit(1)
    open(full, "w").write(src.replace(old, new))
    print(f"  + {label}")


shutil.copyfile(
    "/home/user/vllm_install/glm52_topk_canon.py",
    os.path.join(VLLM, "_glm52_topk_canon.py"),
)
print("  + reinstalled vllm/_glm52_topk_canon.py (relative-convention fix)")

edit(
    SAI,
    "                        _cand = torch.where(_keep, _cand,\n"
    "                                            torch.full_like(_cand, -1))\n"
    "                        topk_indices[_s:_e, :_n].copy_(_cand)\n",
    "                        # Native convention: indices relative to ks;\n"
    "                        # short rows (w <= topk) identity order + -1 pad\n"
    "                        # (NOTES SESSION 4 POSTSCRIPT-2).\n"
    "                        _cand = torch.where(_keep, _cand - _ks,\n"
    "                                            torch.full_like(_cand, -1))\n"
    "                        _iden = (torch.arange(_n, device=logits.device,\n"
    "                                              dtype=torch.int32)\n"
    "                                 .unsqueeze(0).expand_as(_cand))\n"
    "                        _idpad = torch.where(_iden < _nv, _iden,\n"
    "                                             torch.full_like(_cand, -1))\n"
    "                        _cand = torch.where(_nv <= _n, _idpad, _cand)\n"
    "                        topk_indices[_s:_e, :_n].copy_(_cand)\n",
    "torch mode: relative + identity short rows",
)

print("patch_prefill_topk_relfix: done")
