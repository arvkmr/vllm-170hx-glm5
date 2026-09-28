#!/usr/bin/env python3
"""GPU gate for the packed fp8_ds_mla reader on the pinned vNext engine.

Run only on an idle host after installation. It covers both the single-pass
and split-KV/merge paths against an FP32 reference made from the exact packed
bytes written by vLLM's production cache writer.
"""

from __future__ import annotations

import torch

from vllm import _custom_ops as ops
from vllm._glm52_mla_fp8 import triton_mla_sparse_attention_fp8

LORA = 512
ROPE = 64
DIM = LORA + ROPE
ENTRY = 656


def decode_cache(cache: torch.Tensor) -> torch.Tensor:
    rows = cache.shape[0]
    nope = cache[:, :LORA].view(torch.float8_e4m3fn).float()
    scales = cache[:, LORA : LORA + 16].view(torch.float32)
    nope = (nope.view(rows, 4, 128) * scales[:, :, None]).view(rows, LORA)
    rope = cache[:, LORA + 16 :].view(torch.bfloat16).float()
    return torch.cat((nope, rope), dim=-1)


def reference(q: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    flat = indices[:, 0].long()
    valid = (flat >= 0) & (flat < kv.shape[0])
    gathered = kv[flat.clamp(min=0)]
    logits = torch.einsum("thd,tkd->thk", q.float(), gathered) * (DIM**-0.5)
    logits.masked_fill_(~valid[:, None, :], float("-inf"))
    probs = torch.softmax(logits, dim=-1).nan_to_num(0.0)
    return torch.einsum("thk,tkd->thd", probs, gathered[:, :, :LORA])


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.cuda.set_device(0)
    generator = torch.Generator(device="cuda").manual_seed(20260927)
    rows, tokens, heads, topk = 4096, 2, 64, 2048
    kv_c = torch.randn(rows, LORA, dtype=torch.bfloat16, device="cuda", generator=generator)
    k_pe = torch.randn(rows, ROPE, dtype=torch.bfloat16, device="cuda", generator=generator)
    cache = torch.zeros(rows, ENTRY, dtype=torch.uint8, device="cuda")
    slots = torch.arange(rows, dtype=torch.int64, device="cuda")
    ops.concat_and_cache_mla(
        kv_c,
        k_pe,
        cache.view(rows, 1, ENTRY),
        slots,
        "fp8_ds_mla",
        torch.tensor(1.0, dtype=torch.float32, device="cuda"),
    )
    q = torch.randn(
        tokens, heads, DIM, dtype=torch.bfloat16, device="cuda", generator=generator
    )
    indices = torch.randint(
        0,
        rows,
        (tokens, 1, topk),
        dtype=torch.int32,
        device="cuda",
        generator=generator,
    )
    indices[1, :, ::3] = -1
    expected = reference(q, decode_cache(cache), indices)

    for splits in (1, 4):
        actual = triton_mla_sparse_attention_fp8(
            q,
            cache,
            indices,
            sm_scale=DIM**-0.5,
            num_kv_splits=splits,
            sm_count=torch.cuda.get_device_properties(0).multi_processor_count,
        )
        rel_l2 = ((actual.float() - expected).norm() / expected.norm()).item()
        if not torch.isfinite(actual).all() or rel_l2 > 0.01:
            raise SystemExit(f"packed-fp8 kernel failed: splits={splits} rel_l2={rel_l2:.3e}")
        print(f"packed-fp8 kernel: splits={splits} rel_l2={rel_l2:.3e} ok")


if __name__ == "__main__":
    main()
