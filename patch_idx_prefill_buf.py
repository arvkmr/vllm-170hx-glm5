#!/usr/bin/env python3
"""Cap the indexer prefill K-gather workspace (GLM52_IDX_PREFILL_BUF_TOKENS).

Upstream sizes it max_model_len*40 tokens x 132B -- a "use as much as the
flashmla workspace would" heuristic, not a requirement (it only bounds
prefill chunk KV span; chunking handles the rest). At max_model_len=600000
that is 3.17GB per rank and OOMs capture. Cap at 12M tokens (1.58GB,
identical to the 262144-era size; chunk span still 20x model len)."""
import os, sys
import vllm

VLLM = os.path.dirname(vllm.__file__)
IDX = "v1/attention/backends/mla/indexer.py"

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
    IDX,
    "    return max_model_len * 40\n",
    "    import os as _os\n"
    "    _cap = int(_os.environ.get(\"GLM52_IDX_PREFILL_BUF_TOKENS\",\n"
    "                               \"12000000\") or 0)\n"
    "    if _cap:\n"
    "        # heuristic sizing, not a requirement: chunking bounds only the\n"
    "        # per-chunk KV span. 3.17GB at max_model_len=600000 OOMs capture.\n"
    "        return min(max_model_len * 40, _cap)\n"
    "    return max_model_len * 40\n",
    "indexer: cap prefill gather workspace",
)
