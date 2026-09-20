#!/usr/bin/env python3
"""Runtime lm_head quantization for GLM-5.2 (idempotent, anchored edits).

Why: at k=3 MTP every decode step projects hidden states through the FULL
154880x6144 bf16 lm_head FOUR times (target verify + 3 drafter iterations;
the drafter shares the target's lm_head object). On the original PP=8 layout,
that read 7.6 GB/step at 97% of HBM bandwidth on the last rank's critical path
-- 4.6 ms of the 65.9 ms step.
The checkpoint keeps lm_head unquantized (compressed-tensors ignore list).

Fix: quantize lm_head to Marlin W4A16 (or W8A16) AT LOAD TIME, in place, on
whichever rank owns it. Measured on the real weight: 1.171 -> 0.448 ms per
projection (INT4 g64), ~2.9 ms/step total, and frees ~1.4 GB on the last rank.
Swapping `lm_head.quant_method` covers all four uses because the drafter's
shared_head.head IS the same module object (_maybe_share_lm_head).

Env: GLM52_LMHEAD_BITS=4 (uint4b8 g64) or 8 (uint8b128 g64); unset/0 = off.

Run: .venv/bin/python patch_lmhead_quant.py
"""

import os
import sys

import vllm

VLLM = os.path.dirname(vllm.__file__)
RUNNER = "v1/worker/gpu_model_runner.py"
HELPER = "_glm52_lmhead_quant.py"

HELPER_SRC = '''"""Quantize lm_head to Marlin W4A16/W8A16 at load (GLM52_LMHEAD_BITS)."""
import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


def maybe_quantize_lm_head(model) -> None:
    bits = int(os.environ.get("GLM52_LMHEAD_BITS", "0") or 0)
    if bits not in (4, 8):
        return
    lm_head = getattr(model, "lm_head", None)
    weight = getattr(lm_head, "weight", None)
    if lm_head is None or weight is None or weight.numel() == 0:
        return

    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        apply_gptq_marlin_linear,
        marlin_make_workspace_new,
        marlin_permute_scales,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        gptq_quantize_weights,
    )
    from vllm.scalar_type import scalar_types

    qt = scalar_types.uint4b8 if bits == 4 else scalar_types.uint8b128
    dev = weight.device
    dt = weight.dtype
    V, H = weight.shape
    group = 64

    wt = weight.data.t().contiguous()               # [K=H, N=V], model dtype
    _, q_w, s, _, _ = gptq_quantize_weights(wt, qt, group, act_order=False)
    pf = 32 // bits
    mask = (1 << bits) - 1
    pack = torch.zeros(H // pf, V, dtype=torch.int32, device=dev)
    for i in range(pf):
        pack |= (q_w[i::pf, :].to(torch.int32) & mask) << (bits * i)
    empty_i = torch.empty(0, dtype=torch.int, device=dev)
    mq = ops.gptq_marlin_repack(pack, empty_i, H, V, bits)
    ms = marlin_permute_scales(s.to(dt), H, V, group)
    ws = marlin_make_workspace_new(dev)
    del wt, q_w, s, pack

    class _MarlinLMHeadMethod:
        def apply(self, layer, x, bias=None):
            return apply_gptq_marlin_linear(
                input=x,
                weight=mq,
                weight_scale=ms,
                weight_zp=empty_i,
                g_idx=empty_i,
                g_idx_sort_indices=empty_i,
                workspace=ws,
                wtype=qt,
                output_size_per_partition=V,
                input_size_per_partition=H,
                is_k_full=True,
                bias=bias,
            )

    # keep buffer refs alive on the module, swap the projection, free bf16
    lm_head._glm52_marlin_buffers = (mq, ms, ws)
    lm_head.quant_method = _MarlinLMHeadMethod()
    lm_head.weight.data = torch.empty(0, dtype=dt, device=dev)
    torch.cuda.empty_cache()
    logger.info(
        "GLM52: lm_head quantized to W%dA16 marlin (g%d): %.2f GB -> %.2f GB",
        bits, group, V * H * 2 / 1e9, mq.numel() * 4 / 1e9,
    )
'''


def main():
    helper_path = os.path.join(VLLM, HELPER)
    if open(helper_path).read() != HELPER_SRC if os.path.exists(helper_path) else True:
        open(helper_path, "w").write(HELPER_SRC)
        print(f"  + wrote {HELPER}")
    else:
        print(f"  = {HELPER} up to date")

    runner_path = os.path.join(VLLM, RUNNER)
    src = open(runner_path).read()
    anchor = (
        "                if hasattr(self, \"drafter\"):\n"
        "                    logger.info_once(\"Loading drafter model...\")\n"
        "                    if hasattr(self.drafter, \"load_model\"):\n"
        "                        self.drafter.load_model(self.model)\n"
    )
    insert = (
        "                # GLM52 (patch_lmhead_quant.py): optional runtime\n"
        "                # lm_head quantization. Placed BEFORE drafter load;\n"
        "                # _maybe_share_lm_head shares the module REFERENCE,\n"
        "                # so the drafter inherits the swapped quant_method.\n"
        "                from vllm._glm52_lmhead_quant import (\n"
        "                    maybe_quantize_lm_head,\n"
        "                )\n"
        "                maybe_quantize_lm_head(self.model)\n"
    )
    if insert in src:
        print("  = runner hook already applied")
    else:
        if anchor not in src:
            print("  ! runner anchor not found", file=sys.stderr)
            sys.exit(1)
        src = src.replace(anchor, insert + anchor)
        open(runner_path, "w").write(src)
        print("  + runner hook applied")


if __name__ == "__main__":
    main()
