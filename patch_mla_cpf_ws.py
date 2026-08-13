#!/usr/bin/env python3
"""Cap the sparse-MLA chunked-prefill workspace rows (GLM52_MLA_CPF_WS).

The workspace (and its profile-run worst-case simulation in
mla_attention.py forward_impl) is 65536 rows x num_heads x (nope+v) x bf16
= 3.5GiB per rank. At max_model_len=600000 the accumulated per-rank
x-max-len buffers leave < 3.5GiB free at capture -> OOM. Capping rows
shrinks prefill context chunks (correct, marginally slower ultra-long
prefill) and the transient alike. Default 32768 (1.75GiB)."""
import os, sys
import vllm

VLLM = os.path.dirname(vllm.__file__)
F = "model_executor/layers/attention/sparse_mla_attention.py"

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
    "        workspace_size = min(\n"
    "            max(\n"
    "                8 * model_config.max_model_len,\n"
    "                4 * scheduler_config.max_num_seqs * cache_config.block_size,\n"
    "            ),\n"
    "            64 * 1024,\n",
    "        import os as _os\n"
    "        _ws_cap = int(_os.environ.get(\"GLM52_MLA_CPF_WS\", \"32768\") or 0)\n"
    "        workspace_size = min(\n"
    "            max(\n"
    "                8 * model_config.max_model_len,\n"
    "                4 * scheduler_config.max_num_seqs * cache_config.block_size,\n"
    "            ),\n"
    "            _ws_cap if _ws_cap else 64 * 1024,\n",
    "sparse mla: cap chunked-prefill workspace",
)
