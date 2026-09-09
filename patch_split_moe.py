#!/usr/bin/env python3
"""Install the batch-invariant decode-row MoE dispatch (split-MoE).

Copies _glm52_moe_split.py + the GEMV kernel sources into site-packages/vllm
and hooks MarlinExperts.apply via an install call appended to marlin_moe.py.
Pre-builds the CUDA extension so the 8 PP ranks load a cached .so instead of
racing the first compile at server start.

Env: GLM52_SPLIT_MOE=1 enables the dispatch (default 0 until validated).
See _glm52_moe_split.py and NOTES_longctx_copy_fidelity.md for the why.
"""

import os
import shutil
import subprocess
import sys

import vllm

VLLM = os.path.dirname(vllm.__file__)
HERE = os.path.dirname(os.path.abspath(__file__))
MARLIN = os.path.join(
    VLLM, "model_executor/layers/fused_moe/experts/marlin_moe.py")

# 1. copy the shim + kernel sources
for src, dst in [
    ("_glm52_moe_split.py", "_glm52_moe_split.py"),
    ("moe_gemv_marlin.cu", "_glm52_moe_gemv_marlin.cu"),
    ("moe_gemv_marlin_bind.cpp", "_glm52_moe_gemv_marlin_bind.cpp"),
]:
    shutil.copyfile(os.path.join(HERE, src), os.path.join(VLLM, dst))
    print(f"  + vllm/{dst}")

# 2. hook MarlinExperts.apply
HOOK = (
    "\n\n# GLM52: batch-invariant decode-row dispatch "
    "(env GLM52_SPLIT_MOE=1; see vllm/_glm52_moe_split.py).\n"
    "from vllm._glm52_moe_split import install_split_moe "
    "as _glm52_install_split_moe\n"
    "_glm52_install_split_moe(MarlinExperts)\n"
)
src = open(MARLIN).read()
if "_glm52_install_split_moe" in src:
    print("  = marlin_moe.py hook (already applied)")
else:
    open(MARLIN, "a").write(HOOK)
    print("  + marlin_moe.py hook")

# 3. pre-build the extension (cached under ~/.cache/torch_extensions)
print("  building glm52_moe_gemv_ext ...")
r = subprocess.run(
    [sys.executable, "-c",
     "import sys; sys.path.insert(0, %r); "
     "import _glm52_moe_split as m; m._get_ext(); print('build ok')" % VLLM],
    capture_output=True, text=True)
print(r.stdout.strip())
if r.returncode != 0:
    print(r.stderr[-3000:], file=sys.stderr)
    sys.exit(1)
print("GLM52_SPLIT_MOE=0 (default). Set 1 to enable the dispatch.")
