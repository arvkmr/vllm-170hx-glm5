#!/usr/bin/env python3
"""Length-dispatch the decode top-k kernel (GLM52_TOPK_PERSISTENT_MAXLEN).

Measured on CMP 170HX at decode shapes (4 rows, topk 2048): persistent_topk
wins below ~300K row length (0.033-0.059ms vs fallback 0.075-0.079) but its
>131K tier scales worse; the vLLM fallback topKPerRowDecode wins beyond
(409K: 0.082 vs 0.112; 600K: 0.092 vs 0.142). Route by max_seq_len;
default crossover 300000, env-overridable. Both kernels are exact.
"""
import os, sys
import vllm

VLLM = os.path.dirname(vllm.__file__)
SAI = "model_executor/layers/sparse_attn_indexer.py"

def edit(path, old, new, label):
    full = os.path.join(VLLM, path)
    src = open(full).read()
    if new in src:
        print(f"  = {label} (already applied)"); return
    if old not in src or src.count(old) != 1:
        print(f"  ! {label}: anchor problem", file=sys.stderr); sys.exit(1)
    open(full, "w").write(src.replace(old, new))
    print(f"  + {label}")

edit(
    SAI,
    "        use_persistent_topk = current_platform.is_cuda() and topk_tokens in (\n"
    "            512,\n"
    "            1024,\n"
    "            2048,\n"
    "        )\n",
    "        use_persistent_topk = current_platform.is_cuda() and topk_tokens in (\n"
    "            512,\n"
    "            1024,\n"
    "            2048,\n"
    "        )\n"
    "        # persistent_topk's >131K tier scales worse than the fallback\n"
    "        # (measured crossover ~300K at decode shapes); dispatch by length.\n"
    "        _pmax = int(_os.environ.get(\"GLM52_TOPK_PERSISTENT_MAXLEN\",\n"
    "                                    \"300000\") or 0)\n"
    "        if _pmax and attn_metadata_narrowed.max_seq_len > _pmax:\n"
    "            use_persistent_topk = False\n",
    "indexer: length-dispatch decode topk",
)
