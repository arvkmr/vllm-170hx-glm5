#!/usr/bin/env python3
"""Production canon kernel for PREFILL topk determinism (session-4 item 4).

GLM52_PREFILL_TOPK_DET=torch (the validated fix for the KV-build lottery)
costs ~30s per 120K prefill: full-row-width torch.topk over every chunk x
layer. This adds mode `canon`: the windowed [ks,ke) canon-set kernel plus a
K-slice two-sort order pass -- output is the explicit (score desc, index
asc) canonical selection, deterministic across runs/layouts, measured
3.7 ms vs ~56 ms per 2048x131072 call (test_prefill_topk_canon.py ALL
PASS: determinism + parity with the stable-sort canonical reference).

NOTE canon's tie choice (lowest index) differs from torch.topk's
implementation-defined tie choice, so canon builds are deterministic but
not bit-identical to torch-mode builds; per-set failure propensities may
shift (build QUALITY is a separate axis -- session-4 item 3).

Requires patch_topk_canon.py applied (installs _glm52_topk_canon.py; this
re-installs the updated copy with the prefill kernel).
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
print("  + reinstalled vllm/_glm52_topk_canon.py (with prefill kernel)")

edit(
    SAI,
    '                if _os.environ.get("GLM52_PREFILL_TOPK_DET", "") == "torch":\n',
    '                _pdet = _os.environ.get("GLM52_PREFILL_TOPK_DET", "")\n'
    '                if _pdet == "canon":\n'
    "                    # Canonical prefill selection (score desc, index asc):\n"
    "                    # kernel canon-set + K-slice order pass, ~15x cheaper\n"
    "                    # than the torch overwrite. Kills the KV-build lottery\n"
    "                    # (patch_prefill_topk_canon.py).\n"
    "                    from vllm import _glm52_topk_canon as _gtc\n"
    "                    _gtc.canon_topk_indices_prefill(\n"
    "                        logits[:num_rows],\n"
    "                        topk_indices[:num_rows],\n"
    "                        cu_seqlen_ks[:num_rows],\n"
    "                        cu_seqlen_ke[:num_rows],\n"
    "                        topk_tokens,\n"
    "                    )\n"
    '                if _pdet == "torch":\n',
    "prefill det mode dispatch (canon)",
)

print("patch_prefill_topk_canon: done")
