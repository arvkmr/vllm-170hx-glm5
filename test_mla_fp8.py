#!/usr/bin/env python3
"""Correctness + speed check for the fp8_ds_mla sparse-MLA kernel on sm_80.

Everything is scored against `ref_fp32`: the same sparse attention done in
plain torch fp32 over the *exactly decoded* cache. That is the right yardstick
because the obvious alternative -- running the stock bf16 kernel on a
dequantized cache -- is itself lossy: it rounds `e4m3 * scale` into bf16's 8
mantissa bits, while the fp8 kernel keeps the 4-bit e4m3 value exact and
applies the scale in fp32. Scoring against the bf16 kernel would charge the
fp8 kernel for being more accurate than its reference.

So two numbers per case, both vs ref_fp32:

  fp8_err   the new kernel reading the packed cache directly.
  bf16_err  the stock kernel on a torch-dequantized copy -- the incumbent's
            error on identical inputs, i.e. the bar to clear.

Separately, `quant_err` is the bf16 kernel on the *original* KV vs the same
kernel on the decoded one: the cost of the format itself, which no kernel
can avoid. On random gaussian KV with flat attention over 2048 keys it is a
pessimistic figure -- real accuracy is the needle test on the live model.
"""

import sys

import torch

from vllm import _custom_ops as ops
from vllm.v1.attention.ops.triton_mla_sparse_kernel import triton_mla_sparse_attention

sys.path.insert(0, "/home/user/vllm_install")
from glm52_mla_fp8 import triton_mla_sparse_attention_fp8  # noqa: E402

DEV = "cuda:0"
LORA, ROPE, DIM = 512, 64, 576
ENTRY = 656


def decode_cache(cache_u8: torch.Tensor) -> torch.Tensor:
    """Decode the packed 656B layout to fp32 [N, 576] in plain torch.

    Kept in fp32 on purpose: this is what the fp8 kernel effectively sees, since
    it multiplies exact e4m3 values by an fp32 scale. Rounding it to bf16 here
    would bake an error into the reference that only the bf16 kernel commits.
    """
    n = cache_u8.shape[0]
    nope = cache_u8[:, :LORA].view(torch.float8_e4m3fn).to(torch.float32)
    scales = cache_u8[:, LORA : LORA + 16].view(torch.float32)  # [n, 4]
    nope = (nope.view(n, 4, 128) * scales[:, :, None]).view(n, LORA)
    rope = cache_u8[:, LORA + 16 :].view(torch.bfloat16).float()
    return torch.cat([nope, rope], dim=-1)


def build(num_slots: int, seed: int):
    """Random KV, its packed fp8 cache, and the exact torch-decoded version."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    kv_c = (torch.randn(num_slots, LORA, device=DEV, generator=g) * 1.5).to(torch.bfloat16)
    k_pe = torch.randn(num_slots, ROPE, device=DEV, generator=g).to(torch.bfloat16)
    kv_true = torch.cat([kv_c, k_pe], dim=-1)

    cache = torch.zeros(num_slots, ENTRY, dtype=torch.uint8, device=DEV)
    ops.concat_and_cache_mla(
        kv_c,
        k_pe,
        cache.view(num_slots, 1, ENTRY),
        torch.arange(num_slots, dtype=torch.int64, device=DEV),
        "fp8_ds_mla",
        torch.tensor(1.0, dtype=torch.float32, device=DEV),
    )
    return kv_true, cache, decode_cache(cache)


def test_decode_exhaustive() -> bool:
    """All 256 e4m3 byte patterns through the kernel's decode, exactly.

    This is the only zero-tolerance test available: everything downstream
    accumulates 2048 terms and lands in bf16, where a couple of ULP of
    disagreement says nothing. Here the answer is either bit-exact or wrong.
    """
    from vllm.triton_utils import tl, triton

    from glm52_mla_fp8 import _e4m3_bytes_to_bf16

    @triton.jit
    def _decode_all(src, out, N: tl.constexpr):
        off = tl.arange(0, N)
        tl.store(out + off, _e4m3_bytes_to_bf16(tl.load(src + off)))

    src = torch.arange(256, dtype=torch.uint8, device=DEV)
    got = torch.empty(256, dtype=torch.bfloat16, device=DEV)
    _decode_all[(1,)](src, got, N=256)
    # The kernel returns value/256; undo it in fp64 so the compare is exact.
    want = src.view(torch.float8_e4m3fn).double() / 256.0
    got = got.double()

    # 0x7F / 0xFF are e4m3 NaN, which this decode deliberately does not
    # reproduce (see module docstring) -- exclude them, then assert they are
    # at least finite rather than silently poisoning a sum.
    nan_slots = torch.tensor([0x7F, 0xFF], device=DEV)
    keep = torch.ones(256, dtype=torch.bool, device=DEV)
    keep[nan_slots] = False
    bad = (got != want) & keep
    nan_finite = bool(torch.isfinite(got[nan_slots]).all())

    ok = not bad.any() and nan_finite
    print(f"  all 256 e4m3 byte patterns decode exactly: {'OK' if ok else 'FAIL'}")
    if bad.any():
        for i in bad.nonzero().flatten()[:5].tolist():
            print(f"    byte 0x{i:02x}: got {got[i].item()!r} want {want[i].item()!r}")
    if not nan_finite:
        print("    NaN byte patterns did not decode to a finite value")
    return ok


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    """Max error relative to the magnitude of the reference tensor."""
    return ((a - b).abs().max() / b.abs().max().clamp(min=1e-6)).item()


def l2_err(a: torch.Tensor, b: torch.Tensor) -> float:
    """Relative L2 error -- the stable metric. Max-abs on a bf16 output is one
    unlucky element away from meaningless."""
    return ((a - b).norm() / b.norm().clamp(min=1e-6)).item()


def ref_fp32(q, kv, idx, sm_scale):
    """Sparse MLA in fp32 torch: the ground truth for a given (decoded) cache."""
    num_tokens, num_heads, _ = q.shape
    topk = idx.shape[-1]
    qf = q.float()
    kvf = kv.float()
    flat = idx.view(num_tokens, topk)
    valid = (flat >= 0) & (flat < kv.shape[0])
    gathered = kvf[flat.clamp(min=0)]  # [T, topk, 576]
    logits = torch.einsum("thd,tkd->thk", qf, gathered) * sm_scale
    logits = logits.masked_fill(~valid[:, None, :], float("-inf"))
    # A query with no valid KV has an all -inf row; softmax would give NaN, and
    # the kernels deliberately emit 0 there instead.
    any_valid = valid.any(dim=-1)
    probs = torch.softmax(logits, dim=-1).nan_to_num(0.0)
    out = torch.einsum("thk,tkd->thd", probs, gathered[:, :, :LORA])
    return out * any_valid[:, None, None]


def run_case(num_tokens, num_heads, topk, num_slots, splits, seed, invalid_frac=0.0):
    kv_true, cache, kv_deq = build(num_slots, seed)
    g = torch.Generator(device=DEV).manual_seed(seed + 1)
    q = (torch.randn(num_tokens, num_heads, DIM, device=DEV, generator=g)).to(
        torch.bfloat16
    )
    idx = torch.randint(
        0, num_slots, (num_tokens, 1, topk), dtype=torch.int32, device=DEV, generator=g
    )
    if invalid_frac:
        # -1 padding is how the real backend marks "fewer than topk hits".
        mask = torch.rand(idx.shape, device=DEV, generator=g) < invalid_frac
        idx = torch.where(mask, torch.full_like(idx, -1), idx)

    sm_scale = DIM**-0.5
    kw = dict(sm_scale=sm_scale, num_kv_splits=splits, sm_count=70)
    kv_deq_bf16 = kv_deq.to(torch.bfloat16)
    out_true = triton_mla_sparse_attention(q, kv_true.view(-1, 1, DIM), idx, **kw)
    out_deq = triton_mla_sparse_attention(q, kv_deq_bf16.view(-1, 1, DIM), idx, **kw)
    out_fp8 = triton_mla_sparse_attention_fp8(q, cache, idx, **kw)
    gold = ref_fp32(q, kv_deq, idx, sm_scale)

    fp8_err = l2_err(out_fp8.float(), gold)
    bf16_err = l2_err(out_deq.float(), gold)
    quant = l2_err(out_deq.float(), out_true.float())
    nan = bool(torch.isnan(out_fp8).any())
    # The bar is the incumbent kernel's own error on identical inputs, with
    # slack for a different summation order.
    ok = fp8_err <= max(1.5 * bf16_err, 1e-3) and not nan
    print(
        f"  tok={num_tokens:<4} heads={num_heads:<3} topk={topk:<5} "
        f"slots={num_slots:<6} splits={splits}  inval={invalid_frac:.0%}  "
        f"fp8_err={fp8_err:.2e}  bf16_err={bf16_err:.2e}  quant_err={quant:.2e}  "
        f"{'OK' if ok else 'FAIL' + (' (NaN)' if nan else '')}"
    )
    return ok


def main():
    torch.cuda.set_device(0)
    print(f"device capability {torch.cuda.get_device_capability(0)}\n")
    print("decode:")
    decode_ok = test_decode_exhaustive()

    print("\nattention (relative L2 vs fp32 torch on the decoded cache):")
    cases = [
        # (tokens, heads, topk, slots, splits, seed, invalid_frac)
        (1, 128, 2048, 8192, 1, 0, 0.0),
        (1, 128, 2048, 8192, 4, 1, 0.0),
        (4, 128, 2048, 8192, 1, 2, 0.0),
        (4, 128, 2048, 8192, 8, 3, 0.0),
        (32, 128, 2048, 16384, 1, 4, 0.0),
        (64, 128, 2048, 16384, 2, 5, 0.0),
        (8, 128, 512, 4096, 1, 6, 0.0),
        (8, 128, 2048, 8192, 1, 7, 0.30),  # partial-context padding
        (8, 128, 2048, 8192, 4, 8, 0.30),
        (2, 128, 2048, 8192, 1, 9, 1.00),  # every index invalid
        (16, 64, 2048, 8192, 1, 10, 0.0),  # smaller head count
        # Production slot counts: the live cache addresses 1,048,576 slots,
        # 64x beyond the cases above. Slot 1,048,575 * 656 B = 6.9e8 stays
        # under 2^31, but nothing below exercises the upper address range.
        (1, 128, 2048, 1048576, 1, 20, 0.0),
        (4, 128, 2048, 1048576, 4, 21, 0.0),
        (32, 128, 2048, 1048576, 2, 22, 0.0),
        (8, 128, 2048, 1048576, 1, 23, 0.30),  # partial-context padding
    ]
    ok = all([run_case(*c) for c in cases]) and decode_ok

    print("\nspeed (ms/call, decode-shaped; bf16 kernel for reference):")
    import triton

    for num_tokens, splits in [(1, 0), (4, 0), (32, 0), (64, 0)]:
        kv_true, cache, kv_deq = build(262144, 99)
        q = torch.randn(num_tokens, 128, DIM, device=DEV, dtype=torch.bfloat16)
        idx = torch.randint(
            0, 262144, (num_tokens, 1, 2048), dtype=torch.int32, device=DEV
        )
        kw = dict(sm_scale=DIM**-0.5, num_kv_splits=splits or None, sm_count=70)
        t_bf16 = triton.testing.do_bench(
            lambda: triton_mla_sparse_attention(q, kv_true.view(-1, 1, DIM), idx, **kw)
        )
        t_fp8 = triton.testing.do_bench(
            lambda: triton_mla_sparse_attention_fp8(q, cache, idx, **kw)
        )
        print(
            f"  tokens={num_tokens:<4} bf16 {t_bf16:7.3f}  fp8 {t_fp8:7.3f}  "
            f"({t_bf16 / t_fp8:.2f}x)"
        )
        del kv_true, cache, kv_deq
        torch.cuda.empty_cache()

    print("\n" + ("ALL PASS" if ok else "FAILURES"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
