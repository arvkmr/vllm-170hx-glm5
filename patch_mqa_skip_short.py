#!/usr/bin/env python3
"""Skip decode MQA-logits rows whose context fits inside top-k.

For context_len <= topk (2048) the downstream top-k takes the trivial
identity branch in BOTH consumers (persistent_topk.cuh `seq_len <= TopK`
and topKPerRowJob `rowLen <= topK`) and never reads the logits row -- the
entire paged MQA logits computation for such rows is dead work. The old
4-column capture-time buffers were accidentally skipping it via the L_MAX
clamp early-return (that is why short-ctx conc-16 measured ~198 before
full-width capture and ~165 after). Make the skip principled: a per-row
device-side early return when context_len <= SKIP_LE, passed as
topk_tokens by the indexer. Rows above topk compute fully.
"""

import os
import sys

import vllm

VLLM = os.path.dirname(vllm.__file__)
MQA = "v1/attention/ops/mqa_logits_triton.py"
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


def main():
    # host signature
    edit(
        MQA,
        "    block_tables: torch.Tensor,\n"
        "    max_model_len: int,\n"
        "    clean_logits: bool = True,\n"
        ") -> torch.Tensor:\n"
        "    \"\"\"Triton implementation of DeepGEMM's fp8_paged_mqa_logits.\n",
        "    block_tables: torch.Tensor,\n"
        "    max_model_len: int,\n"
        "    clean_logits: bool = True,\n"
        "    skip_le: int = 0,\n"
        ") -> torch.Tensor:\n"
        "    \"\"\"Triton implementation of DeepGEMM's fp8_paged_mqa_logits.\n",
        "host: skip_le kwarg",
    )
    # host launch: pass after L_MAX
    edit(
        MQA,
        "        logits.stride(0),\n"
        "        logits.stride(1),\n"
        "        logits.shape[1],\n"
        "        next_n=next_n,\n",
        "        logits.stride(0),\n"
        "        logits.stride(1),\n"
        "        logits.shape[1],\n"
        "        skip_le,\n"
        "        next_n=next_n,\n",
        "host: pass skip_le",
    )
    # kernel param
    edit(
        MQA,
        "    stride_l_t,\n"
        "    stride_l_n,\n"
        "    L_MAX,\n"
        "    next_n: tl.constexpr,\n",
        "    stride_l_t,\n"
        "    stride_l_n,\n"
        "    L_MAX,\n"
        "    SKIP_LE,\n"
        "    next_n: tl.constexpr,\n",
        "kernel: SKIP_LE param",
    )
    # kernel early return: rows fully inside top-k are never read by the
    # top-k consumers (identity shortcut) -- computing them is dead work.
    edit(
        MQA,
        "    if block_rk * block_size >= context_len:\n"
        "        return\n"
        "    if block_rk * block_size >= L_MAX:\n"
        "        return\n",
        "    if block_rk * block_size >= context_len:\n"
        "        return\n"
        "    if block_rk * block_size >= L_MAX:\n"
        "        return\n"
        "    if context_len <= SKIP_LE:\n"
        "        # Entire row fits inside top-k: the consumer takes the\n"
        "        # identity branch (persistent_topk trivial case /\n"
        "        # topKPerRowJob rowLen<=topK) and never reads logits.\n"
        "        return\n",
        "kernel: skip short rows",
    )
    # indexer decode call: enable the skip with the actual topk
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
        "            )",
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
        "indexer: enable skip at topk",
    )


if __name__ == "__main__":
    main()
