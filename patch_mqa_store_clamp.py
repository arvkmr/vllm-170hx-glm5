#!/usr/bin/env python3
"""Clamp _fp8_paged_mqa_logits_kernel stores to the logits buffer width.

ROOT CAUSE (2026-08-11): under FULL cudagraph capture, the decode-side
indexer logits buffer is allocated at capture-time max_seq_len -- which is
`max_query_len` (4 with MTP k=3) for every non-profiled capture size. The
kernel bounds its stores only by the RUNTIME `context_lens` device values,
so replays with real contexts write -inf tiles and logits floats far past
the captured buffer (the rank-1 IMA scribbler). Eager/PIECEWISE are safe
because the buffer is freshly sized >= context each call.

This patch passes the buffer width into the kernel and masks both the
early-exit -inf path and the main store with `k_offset < L_MAX`. It stops
the memory corruption; note that replays with context > captured width
still TRUNCATE the indexer candidate set (correctness follows in a second
step: realistic profile_seq_lens at capture and/or a seq-len dispatch
guard).
"""

import os
import sys

import vllm

VLLM = os.path.dirname(vllm.__file__)
MQA = "v1/attention/ops/mqa_logits_triton.py"


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
    # host: pass buffer width
    edit(
        MQA,
        "        logits.stride(0),\n"
        "        logits.stride(1),\n"
        "        next_n=next_n,\n",
        "        logits.stride(0),\n"
        "        logits.stride(1),\n"
        "        logits.shape[1],\n"
        "        next_n=next_n,\n",
        "host: pass logits width",
    )
    # kernel: accept width param (after the stride params, before constexprs)
    edit(
        MQA,
        "    stride_l_t,\n"
        "    stride_l_n,\n"
        "    next_n: tl.constexpr,\n",
        "    stride_l_t,\n"
        "    stride_l_n,\n"
        "    L_MAX,\n"
        "    next_n: tl.constexpr,\n",
        "kernel: accept L_MAX",
    )
    # kernel: early-exit for blocks entirely past the buffer
    edit(
        MQA,
        "    context_len = tl.load(context_lens_ptr + batch_id)\n"
        "    if block_rk * block_size >= context_len:\n"
        "        return\n",
        "    context_len = tl.load(context_lens_ptr + batch_id)\n"
        "    # Never touch memory past the logits buffer: under FULL cudagraph\n"
        "    # replay context_len can exceed the capture-time buffer width.\n"
        "    if block_rk * block_size >= context_len:\n"
        "        return\n"
        "    if block_rk * block_size >= L_MAX:\n"
        "        return\n",
        "kernel: early-exit past buffer",
    )
    # kernel: clamp the store mask
    edit(
        MQA,
        "    tl.store(\n"
        "        logits_ptr + token_id * stride_l_t + k_offset * stride_l_n,\n"
        "        out,\n"
        "        mask=mask_n,\n"
        "    )",
        "    tl.store(\n"
        "        logits_ptr + token_id * stride_l_t + k_offset * stride_l_n,\n"
        "        out,\n"
        "        mask=mask_n & (k_offset < L_MAX),\n"
        "    )",
        "kernel: clamp store to buffer width",
    )


if __name__ == "__main__":
    main()
