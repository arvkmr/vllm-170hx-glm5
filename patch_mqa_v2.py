#!/usr/bin/env python3
"""Route decode MQA logits to the v2 kernel (GLM52_MQA_V2=0 to disable).

v2 (vllm/_glm52_mqa_v2.py): batch-major persistent tiles, q decoded once
per program, all next_n spec rows share each K tile via one
[next_n*H,D]x[D,64] MMA, runtime loop bounds (graph-safe). Exact-match vs
v1 on all tested shapes incl. -inf placement. Measured on CMP 170HX:
B=1@196K 0.353->0.194 ms/layer (flat to 196K); B=8@196K 2.74->0.97 (2.8x,
at the MMA compute floor ~50 TFLOPS for the irreducible relu-per-head
FLOPs)."""
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
    "            logits = fp8_paged_mqa_logits_triton(\n"
    "                padded_q_quant_cast,\n"
    "                kv_cache,\n"
    "                weights[:num_padded_tokens],\n"
    "                seq_lens,\n"
    "                decode_metadata.block_table,\n"
    "                max_model_len=active_max_model_len,\n"
    "                clean_logits=False,\n"
    "                skip_le=topk_tokens,\n"
    "            )",
    "            if _os.environ.get(\"GLM52_MQA_V2\", \"1\") == \"1\":\n"
    "                from vllm._glm52_mqa_v2 import fp8_paged_mqa_logits_v2\n"
    "                logits = fp8_paged_mqa_logits_v2(\n"
    "                    padded_q_quant_cast,\n"
    "                    ),\n"
    "                    kv_cache,\n"
    "                    weights[:num_padded_tokens],\n"
    "                    seq_lens,\n"
    "                    decode_metadata.block_table,\n"
    "                    max_model_len=active_max_model_len,\n"
    "                    skip_le=topk_tokens,\n"
    "                )\n"
    "            else:\n"
    "                logits = fp8_paged_mqa_logits_triton(\n"
    "                    padded_q_quant_cast,\n"
    "                    kv_cache,\n"
    "                    weights[:num_padded_tokens],\n"
    "                    seq_lens,\n"
    "                    decode_metadata.block_table,\n"
    "                    max_model_len=active_max_model_len,\n"
    "                    clean_logits=False,\n"
    "                    skip_le=topk_tokens,\n"
    "                )",
    "indexer: route decode logits to v2",
)
