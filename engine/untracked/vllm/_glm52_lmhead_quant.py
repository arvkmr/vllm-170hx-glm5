"""Quantize the target lm_head to Marlin W8A16/W4A16 at load (GLM52_LMHEAD_BITS).

Ported from the v0.26 stack (patch_lmhead_quant.py) to the fork's current
Marlin API. Every decode step projects hidden states through the full
154880x6144 BF16 lm_head twice on the last pipeline stage: the target's
verify rows and the DFlash2 drafter's candidate pass, which aliases the same
module (spec_decode/dflash/utils.py: ``dflash_model.lm_head = target_lm_head``)
and also goes through ``lm_head.quant_method.apply``. Swapping the module's
quant_method covers both. W8 (uint8b128, g64) is what v0.26 validated as
lossless on its canaries; unset/0 leaves the BF16 head untouched.
"""

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
        return  # not the last pipeline stage, or already quantized

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

    wt = weight.data.t().contiguous()  # [K=H, N=V]
    _, q_w, s = gptq_quantize_weights(wt, qt, group)
    pf = 32 // bits
    mask = (1 << bits) - 1
    pack = torch.zeros(H // pf, V, dtype=torch.int32, device=dev)
    for i in range(pf):
        pack |= (q_w[i::pf, :].to(torch.int32) & mask) << (bits * i)
    empty_i = torch.empty(0, dtype=torch.int, device=dev)
    mq = ops.gptq_marlin_repack(pack, H, V, bits)
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
                workspace=ws,
                wtype=qt,
                output_size_per_partition=V,
                input_size_per_partition=H,
                bias=bias,
            )

    lm_head._glm52_marlin_buffers = (mq, ms, ws)
    lm_head.quant_method = _MarlinLMHeadMethod()
    lm_head.weight.data = torch.empty(0, dtype=dt, device=dev)
    torch.cuda.empty_cache()
    logger.info(
        "GLM52: lm_head quantized to W%dA16 marlin (g%d): %.2f GB -> %.2f GB",
        bits,
        group,
        V * H * 2 / 1e9,
        mq.numel() * 4 / 1e9,
    )
