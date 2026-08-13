#!/usr/bin/env python3
"""Route the DSA indexer's prefill logits to the v2 kernel.

v2 (vllm/_glm52_idx_prefill_v2.py) blocks BLOCK_M query rows into a single
[BLOCK_M*H, D] x [D, BLOCK_N] MMA so they share each K tile, instead of v1's
one-program-per-query-row grid where the MMA's M dimension is just the 32
heads. Bit-exact vs v1 on every shape tested. Measured on CMP 170HX:
137 -> 105 ns per (query token x 1K keys) at M=128/N=262144, 124 -> 98 at
M=512, and 137 -> 105 at N=1048576. 1.27-1.31x.

Why it matters at long context: indexer logits is ~37% of a 1M-token prefill
(137 ns x L^2/2 pairs x 5 indexer layers on the binding rank = ~376 s of the
~17 min measured TTFT). This is the prefill twin of the fix patch_mqa_v2.py
already applied to the decode path.

GLM52_IDX_PREFILL_V2=0 falls back to v1.
"""

import os
import shutil
import sys

import vllm

VLLM = os.path.dirname(vllm.__file__)
BACKUP = os.path.join(os.path.dirname(VLLM), ".glm52-backup", "vllm")
SRC = "/home/user/vllm_install/glm52_idx_prefill_v2.py"
DST = os.path.join(VLLM, "_glm52_idx_prefill_v2.py")

SAI = "model_executor/layers/sparse_attn_indexer.py"

_EDITS: list[tuple[str, str, str, str]] = []


def edit(path, old, new, label):
    _EDITS.append((path, old, new, label))


def apply_all():
    files = {}
    todo = []
    for path, old, new, label in _EDITS:
        src = files.get(path) or open(os.path.join(VLLM, path)).read()
        if new in src:
            print(f"  = {label} (already applied)")
            files[path] = src
            continue
        if src.count(old) != 1:
            print(
                f"  ! {label}: anchor matched {src.count(old)} times in {path}, "
                f"expected 1 -- nothing written",
                file=sys.stderr,
            )
            sys.exit(1)
        files[path] = src.replace(old, new)
        todo.append(label)

    for path in files:
        dst = os.path.join(BACKUP, path)
        if not os.path.exists(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(os.path.join(VLLM, path), dst)
            print(f"  b backup -> {dst}")
    for path, src in files.items():
        open(os.path.join(VLLM, path), "w").write(src)
    for label in todo:
        print(f"  + {label}")


edit(
    SAI,
    "                else:\n"
    "                    # SM80/SM121 Triton fallback (PR #38476).\n"
    "                    _tl = _pt() if _PROF else 0.0\n"
    "                    logits = fp8_mqa_logits_triton(\n"
    "                        q_slice_cast,\n"
    "                        (k_quant_cast, k_scale_cast),\n"
    "                        weights[chunk.token_start : chunk.token_end],\n"
    "                        cu_seqlen_ks,\n"
    "                        cu_seqlen_ke,\n"
    "                        clean_logits=False,\n"
    "                    )",
    "                else:\n"
    "                    # SM80/SM121 Triton fallback (PR #38476).\n"
    "                    _tl = _pt() if _PROF else 0.0\n"
    '                    if _os.environ.get("GLM52_IDX_PREFILL_V2", "1") == "1":\n'
    "                        from vllm._glm52_idx_prefill_v2 import (\n"
    "                            fp8_mqa_logits_v2,\n"
    "                        )\n"
    "\n"
    "                        logits = fp8_mqa_logits_v2(\n"
    "                            q_slice_cast,\n"
    "                            (k_quant_cast, k_scale_cast),\n"
    "                            weights[chunk.token_start : chunk.token_end],\n"
    "                            cu_seqlen_ks,\n"
    "                            cu_seqlen_ke,\n"
    "                            clean_logits=False,\n"
    "                        )\n"
    "                    else:\n"
    "                        logits = fp8_mqa_logits_triton(\n"
    "                            q_slice_cast,\n"
    "                            (k_quant_cast, k_scale_cast),\n"
    "                            weights[chunk.token_start : chunk.token_end],\n"
    "                            cu_seqlen_ks,\n"
    "                            cu_seqlen_ke,\n"
    "                            clean_logits=False,\n"
    "                        )",
    "prefill logits -> v2 kernel",
)

# Prime the v2 autotune at init. Skipping this is not a slow first call, it
# is a dead engine: the sweep runs inline on the first real chunk at full N
# (83 s measured), which overruns vLLM's sample_tokens RPC timeout and takes
# EngineCore down. The MLA backend already has a warmup hook for v1's kernels;
# hang the v2 warmup off the same place.
edit(
    "v1/attention/backends/mla/triton_mla_sparse.py",
    "        warmup_fp8_mqa_logits_triton(\n"
    "            num_heads=indexer_num_heads, head_dim=indexer_head_dim, device=device\n"
    "        )",
    "        warmup_fp8_mqa_logits_triton(\n"
    "            num_heads=indexer_num_heads, head_dim=indexer_head_dim, device=device\n"
    "        )\n"
    '        if os.environ.get("GLM52_IDX_PREFILL_V2", "1") == "1":\n'
    "            from vllm._glm52_idx_prefill_v2 import warmup_fp8_mqa_logits_v2\n"
    "\n"
    "            warmup_fp8_mqa_logits_v2(\n"
    "                num_heads=indexer_num_heads,\n"
    "                head_dim=indexer_head_dim,\n"
    "                device=device,\n"
    "            )",
    "warm the v2 prefill autotune at init",
)

apply_all()
shutil.copyfile(SRC, DST)
print(f"  + kernel -> {DST}")
print("\nDSA prefill indexer now uses the v2 logits kernel "
      "(GLM52_IDX_PREFILL_V2=0 to revert).")
