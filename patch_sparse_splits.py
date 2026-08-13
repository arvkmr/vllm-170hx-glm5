#!/usr/bin/env python3
"""Fix the sparse-MLA split heuristic (GLM52_MLA_SPLITS env overrides).

The old rule skipped splitting when baseline_grid * 4 >= sm_count -- i.e.
32 thin CTAs on 70 SMs counted as "busy" and decode-shaped calls
(4 tokens x 8 head-groups) ran the single-pass kernel at 0.347ms/layer.
Measured on CMP 170HX: splits=8 runs the same call at 0.189ms (1.8x).
New rule: split until the grid reaches ~2 waves of SMs (bounded by
per-split work >= _MIN_TOPK_PER_SPLIT and topk divisibility, as before).
Exact: split-K + merge is numerically the same reduction.
"""
import os, sys
import vllm

VLLM = os.path.dirname(vllm.__file__)
F = "v1/attention/ops/triton_mla_sparse_kernel.py"

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
    F,
    "    baseline = num_tokens * num_head_groups\n"
    "    if baseline == 0 or baseline * _SPLIT_MAX_OCCUPANCY >= sm_count:\n"
    "        return 1\n"
    "    ideal = triton.next_power_of_2(max(1, index_topk // _MIN_TOPK_PER_SPLIT))\n"
    "    max_splits = max(1, sm_count // baseline)\n",
    "    import os as _os\n"
    "    _force = int(_os.environ.get(\"GLM52_MLA_SPLITS\", \"0\") or 0)\n"
    "    if _force:\n"
    "        return _force\n"
    "    baseline = num_tokens * num_head_groups\n"
    "    # Old rule: return 1 when baseline*4 >= sm_count -- but 32 thin CTAs\n"
    "    # on 70 SMs is starvation, not saturation (measured 1.8x win from\n"
    "    # splitting at decode shapes). Split until ~2 waves of SMs.\n"
    "    if baseline == 0 or baseline >= sm_count * 2:\n"
    "        return 1\n"
    "    ideal = triton.next_power_of_2(max(1, index_topk // _MIN_TOPK_PER_SPLIT))\n"
    "    max_splits = max(1, (sm_count * 2) // baseline)\n",
    "sparse: split until ~2 SM waves",
)
