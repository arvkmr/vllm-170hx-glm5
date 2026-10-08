#!/usr/bin/env python3
"""GPU gate for the sm_80 DSA ports (apply_engine_patch.py).

Run with the vNext venv on one idle CMP 170HX:

    CUDA_VISIBLE_DEVICES=0 ~/vllm_glm53_dflash2/venv/bin/python test_dsa_sm80.py

1. ``f32_to_e4m3_bits`` against torch's CUDA ``float8_e4m3fn`` cast over
   *every* float32 bit pattern (the software path must equal the hardware
   ``cvt.rn.satfinite`` semantics the native kernels use).
2. ``fused_q`` (index-Q RoPE + ue8m0 quant + weight folding; bf16 and fp8
   MQA query) against a float32 torch reference at GLM-5.3 shapes.
3. ``fused_norm_rope`` with both caches live: q/kv RMSNorm, the packed
   656-byte ``fp8_ds_mla`` MLA row and the (128 + 4)-byte indexer K row,
   decoded from bytes and compared with the reference.
4. The Triton MQA-logits kernels the indexer now uses, against the fork's
   torch references, including the byte-packed padded decode path.

Byte comparisons allow a rare one-ulp e4m3 difference: the reference RoPE is
computed by torch and may differ from the kernel's contracted FMA by one
float32 ulp, which can move a value across an e4m3 rounding boundary.
"""

import sys

import torch

from vllm.triton_utils import tl, triton

DEV = "cuda"
torch.manual_seed(0)
FAIL = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        FAIL.append(name)


def e4m3_ulp_diff(a_u8, b_u8):
    """Max |a-b| in e4m3 steps (monotonic key: sign-magnitude -> ordered)."""
    def key(x):
        x = x.to(torch.int32)
        mag = x & 0x7F
        return torch.where((x & 0x80) != 0, -mag, mag)
    return (key(a_u8) - key(b_u8)).abs()


def bytes_close(name, got_u8, ref_u8, max_frac=1e-3):
    d = e4m3_ulp_diff(got_u8, ref_u8)
    frac = (d != 0).float().mean().item()
    check(name, d.max().item() <= 1 and frac <= max_frac,
          f"(max ulp {d.max().item()}, mismatch frac {frac:.2e})")


def fp8(x):
    return x.to(torch.float8_e4m3fn).view(torch.uint8)


def ue8m0(x, dim=-1):
    amax = x.abs().amax(dim=dim, keepdim=True).clamp_min(1e-4)
    return torch.exp2(torch.ceil(torch.log2(amax / 448.0)))


def rope_interleaved(x, cos, sin):
    """x [..., 2h]; cos/sin broadcast [..., h]; pairs (2i, 2i+1)."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x2 * cos + x1 * sin
    return out


# --------------------------------------------------------------------------- 1
from vllm.v1.attention.ops.triton_e4m3 import f32_to_e4m3_bits  # noqa: E402


@triton.jit
def _sw_cvt_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(out_ptr + offs, f32_to_e4m3_bits(tl.load(x_ptr + offs, mask=m)), mask=m)


def test_exhaustive_cvt():
    kern = _sw_cvt_kernel
    chunk = 1 << 27
    bad = 0
    for c in range((1 << 32) // chunk):
        bits = torch.arange(c * chunk, (c + 1) * chunk, device=DEV, dtype=torch.int64)
        x = bits.to(torch.int32).view(torch.float32)  # wraps to all patterns
        out = torch.empty(chunk, dtype=torch.uint8, device=DEV)
        kern[(chunk // 1024,)](x, out, chunk, BLOCK=1024)
        ref = x.to(torch.float8_e4m3fn).view(torch.uint8)
        nan = torch.isnan(x)
        # Any NaN encoding is acceptable for NaN input.
        diff = (out != ref) & ~(nan & ((out & 0x7F) == 0x7F) & ((ref & 0x7F) == 0x7F))
        bad += int(diff.sum())
    check("f32_to_e4m3_bits == torch cast over all 2^32 float32", bad == 0,
          f"({bad} mismatches)")


# --------------------------------------------------------------------------- 2
def cos_sin_cache(max_pos, rot, theta=8e6):
    inv = 1.0 / theta ** (torch.arange(0, rot, 2, device=DEV).float() / rot)
    f = torch.outer(torch.arange(max_pos, device=DEV).float(), inv)
    return torch.cat([f.cos(), f.sin()], -1)  # fp32 [max_pos, rot]


def test_fused_q():
    from vllm.models.deepseek_v32.common.kernels import fused_q

    T, H, IH, ID = 37, 64, 32, 128
    pos = torch.randint(0, 100000, (T,), device=DEV, dtype=torch.int64)
    cs = cos_sin_cache(100000, 64)
    q_pe = torch.randn(T, H, 64, device=DEV, dtype=torch.bfloat16)
    ql_nope = torch.randn(T, H, 512, device=DEV, dtype=torch.bfloat16) * 3
    index_q = torch.randn(T, IH, ID, device=DEV, dtype=torch.bfloat16) * 2
    w = torch.randn(T, IH, device=DEV, dtype=torch.bfloat16)
    q_scale = torch.tensor([0.05], device=DEV, dtype=torch.float32)
    sm, hs = ID ** -0.5, IH ** -0.5

    cos, sin = cs[pos, :32], cs[pos, 32:]
    iq = index_q.float()
    iq_rot = iq.clone()
    iq_rot[..., :64] = rope_interleaved(iq[..., :64], cos[:, None], sin[:, None])
    s = ue8m0(iq_rot)
    ref_iq8 = fp8(iq_rot / s)
    ref_w = w.float() * s.squeeze(-1) * sm * hs
    ref_qpe = rope_interleaved(q_pe.float(), cos[:, None], sin[:, None])

    for quant in (False, True):
        iq8, wout, mqa = fused_q(pos, q_pe, cs, index_q, cs, ql_nope, q_scale, w,
                                 sm, hs, has_indexer=True,
                                 index_rope_interleave=True, quantize_mqa=quant)
        torch.accelerator.synchronize()
        tag = f"fused_q(quantize_mqa={quant})"
        check(f"{tag} returns float8 index_q", iq8.dtype == torch.float8_e4m3fn)
        bytes_close(f"{tag} index_q e4m3 bytes", iq8.view(torch.uint8), ref_iq8)
        check(f"{tag} index weights", torch.allclose(wout, ref_w, rtol=1e-5, atol=1e-6),
              f"(max abs {(wout - ref_w).abs().max().item():.2e})")
        if quant:
            ref = torch.cat([fp8(ql_nope.float() / q_scale), fp8(ref_qpe / q_scale)], -1)
            bytes_close(f"{tag} packed MQA query bytes", mqa.view(torch.uint8), ref)
        else:
            err = (mqa.float() - ref_qpe).abs().max().item()
            check(f"{tag} bf16 q_pe RoPE", err < 2e-2, f"(max abs {err:.2e})")


# --------------------------------------------------------------------------- 3
def test_fused_norm_rope():
    from vllm.models.deepseek_v32.common.kernels import fused_norm_rope

    T, NB, BS = 45, 8, 64
    pos = torch.randint(0, 50000, (T,), device=DEV, dtype=torch.int64)
    cs = cos_sin_cache(50000, 64)
    q_c = torch.randn(T, 2048, device=DEV, dtype=torch.bfloat16)
    kv_c = torch.randn(T, 512, device=DEV, dtype=torch.bfloat16) * 4
    k_pe = torch.randn(T, 64, device=DEV, dtype=torch.bfloat16)
    ik = torch.randn(T, 128, device=DEV, dtype=torch.bfloat16) * 3
    qw = torch.rand(2048, device=DEV, dtype=torch.bfloat16) + 0.5
    kw = torch.rand(512, device=DEV, dtype=torch.bfloat16) + 0.5
    ikw = torch.rand(128, device=DEV, dtype=torch.float32) + 0.5
    ikb = torch.randn(128, device=DEV, dtype=torch.float32) * 0.1
    topk = torch.zeros(T, 2048, device=DEV, dtype=torch.int32)
    slots = torch.randperm(NB * BS, device=DEV)[:T].to(torch.int64)
    slots[3] = -1  # a padding row: must write nothing
    idx_cache = torch.zeros(NB, BS, 132, device=DEV, dtype=torch.uint8)
    mla_cache = torch.zeros(NB, BS, 656, device=DEV, dtype=torch.uint8)
    kv_out = torch.empty_like(kv_c)
    kpe_out = torch.empty_like(k_pe)

    q_out = fused_norm_rope(
        pos, q_c, qw, 1e-5, kv_c, kw, 1e-5, k_pe, cs, ik, ikw, ikb, 1e-6, cs, topk,
        slot_mapping=slots, indexer_slot_mapping=slots, indexer_k_cache=idx_cache,
        indexer_cache_shuffled=False, mla_kv_cache=mla_cache,
        mla_kv_cache_dtype="fp8_ds_mla", mla_k_scale=None, has_indexer=True,
        index_rope_interleave=True, kv_c_out=kv_out, k_pe_out=kpe_out,
    )
    torch.accelerator.synchronize()

    def rms(x, w, eps):
        x = x.float()
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w.float()

    ref_q = rms(q_c, qw, 1e-5)
    check("q_c RMSNorm", (q_out.float() - ref_q).abs().max().item() < 3e-2)
    ref_kv = rms(kv_c, kw, 1e-5)  # the kernel quantizes the fp32 norm
    cos, sin = cs[pos, :32], cs[pos, 32:]
    ref_kpe = rope_interleaved(k_pe.float(), cos, sin)
    check("topk buffer cleared to -1", bool((topk == -1).all()))

    live = slots >= 0
    sl = slots[live]
    rows = mla_cache.view(-1, 656)[sl]
    kv_live = ref_kv[live]
    tiles = kv_live.view(-1, 4, 128)
    ref_scale = ue8m0(tiles)
    ref_nope = fp8((tiles / ref_scale).view(-1, 512))
    bytes_close("MLA fp8_ds_mla NoPE bytes", rows[:, :512].contiguous(), ref_nope)
    got_scale = rows[:, 512:528].contiguous().view(torch.float32)
    check("MLA fp8_ds_mla tile scales", torch.equal(got_scale, ref_scale.squeeze(-1)))
    got_rope = rows[:, 528:656].contiguous().view(torch.bfloat16).float()
    err = (got_rope - ref_kpe[live]).abs().max().item()
    check("MLA fp8_ds_mla bf16 RoPE tail", err < 2e-2, f"(max abs {err:.2e})")
    deq = rows[:, :512].contiguous().view(torch.float8_e4m3fn).float().view(-1, 4, 128)
    deq = (deq * got_scale[..., None]).view(-1, 512)
    rel = ((deq - kv_live).norm() / kv_live.norm()).item()
    check("MLA row dequantizes to normed kv_c", rel < 4e-2, f"(rel_l2 {rel:.2e})")

    # Indexer K: LayerNorm + interleaved RoPE on the first 64 dims + ue8m0.
    x = ik.float()
    xn = (x - x.mean(-1, keepdim=True)) * torch.rsqrt(
        x.var(-1, unbiased=False, keepdim=True) + 1e-6) * ikw + ikb
    xr = xn.clone()
    xr[:, :64] = rope_interleaved(xn[:, :64], cos, sin)
    s = ue8m0(xr)[live]
    ref_k8 = fp8(xr[live] / s)
    blk, off = sl // BS, sl % BS
    flat = idx_cache.view(NB, -1)
    vals = torch.stack([flat[b, o * 128:(o + 1) * 128] for b, o in zip(blk.tolist(), off.tolist())])
    scl = torch.stack([flat[b, BS * 128 + o * 4:BS * 128 + o * 4 + 4]
                       for b, o in zip(blk.tolist(), off.tolist())]).view(torch.float32)
    bytes_close("indexer K e4m3 bytes", vals, ref_k8)
    check("indexer K ue8m0 scales", torch.equal(scl, s))

    # Nothing else in either cache was touched (padding row skipped).
    mask = torch.zeros(NB * BS, dtype=torch.bool, device=DEV)
    mask[sl] = True
    check("MLA cache untouched outside live slots",
          bool((mla_cache.view(-1, 656)[~mask] == 0).all()))


# --------------------------------------------------------------------------- 4
def test_logits():
    from vllm.v1.attention.ops import triton_mqa_logits as L

    M, H, D, N = 40, 32, 128, 3000
    q = fp8(torch.randn(M, H, D, device=DEV) * 4).view(torch.float8_e4m3fn)
    k = fp8(torch.randn(N, D, device=DEV) * 4).view(torch.float8_e4m3fn)
    ks = torch.rand(N, device=DEV) * 0.01
    w = torch.randn(M, H, device=DEV)
    ke = torch.randint(1, N, (M,), device=DEV, dtype=torch.int32)
    kst = (ke - torch.randint(1, 1000, (M,), device=DEV, dtype=torch.int32)).clamp_min(0)
    got = L.fp8_mqa_logits_triton(q, (k, ks), w, kst, ke)
    ref = L.fp8_mqa_logits_torch(q, (k, ks), w, kst, ke)
    ar = torch.arange(N, device=DEV)
    m = (ar[None] >= kst[:, None]) & (ar[None] < ke[:, None])
    err = ((got - ref).abs()[m] / (ref.abs()[m] + 1)).max().item()
    check("prefill Triton MQA logits vs torch", err < 1e-3, f"(max rel {err:.2e})")

    # Paged decode, next_n=2, with the byte-packed padded layout the patch uses.
    from vllm.v1.attention.ops.common import pack_seq_triton

    B, nn_, BS, NB = 4, 2, 64, 64
    cache = torch.zeros(NB, BS, D + 4, device=DEV, dtype=torch.uint8)
    flat = cache.view(NB, -1)
    flat[:, : BS * D] = fp8(torch.randn(NB, BS * D, device=DEV) * 3)
    flat[:, BS * D:] = (torch.rand(NB, BS, device=DEV) * 0.01).view(torch.uint8).view(NB, -1)
    bt = torch.randperm(NB, device=DEV, dtype=torch.int32)[: B * 16].view(B, 16)
    lens = torch.tensor([2, 1, 2, 2], device=DEV, dtype=torch.int32)  # ragged
    ntok = int(lens.sum())
    qd = fp8(torch.randn(ntok, H, D, device=DEV) * 4)
    wd = torch.randn(ntok, H, device=DEV)
    ctx_last = torch.tensor([700, 64, 1000, 5], device=DEV, dtype=torch.int32)
    qp = pack_seq_triton(qd, lens, pad_value=0)  # [B, 2, H, D] uint8
    wp = pack_seq_triton(wd, lens, pad_value=0).reshape(-1, H)
    ctx = torch.stack([ctx_last - 1, ctx_last], 1).to(torch.int32)
    got = L.fp8_paged_mqa_logits(qp, cache.unsqueeze(-2), wp, ctx, bt, 1024, max_seq_len=1000)
    ref = L.fp8_paged_mqa_logits_torch(qp, cache.unsqueeze(-2), wp, ctx, bt, 1024)
    cm = torch.arange(1024, device=DEV)[None] < ctx.view(-1, 1)
    err = ((got - ref).abs()[cm] / (ref.abs()[cm] + 1)).max().item()
    check("paged Triton MQA logits vs torch (packed ragged)", err < 1e-3,
          f"(max rel {err:.2e})")
    # Row alignment: packed row (b, j) must equal the unpadded token's score.
    starts = torch.cumsum(lens, 0) - lens
    single = []
    for b in range(B):
        for j in range(int(lens[b])):
            t = int(starts[b]) + j
            r = L.fp8_paged_mqa_logits_torch(
                qd[t].view(1, 1, H, D), cache.unsqueeze(-2), wd[t:t + 1],
                ctx[b, j].view(1, 1), bt[b:b + 1], 1024)
            single.append(torch.allclose(got.view(B, nn_, -1)[b, j][: ctx[b, j]],
                                         r[0][: ctx[b, j]], rtol=1e-3, atol=1e-3))
    check("packed decode rows align with their tokens", all(single))


if __name__ == "__main__":
    cap = torch.cuda.get_device_capability()
    print(f"device {torch.cuda.get_device_name()} sm_{cap[0]}{cap[1]}")
    for t in (test_exhaustive_cvt, test_fused_q, test_fused_norm_rope, test_logits):
        print(t.__name__)
        t()
    print("FAILED: " + ", ".join(FAIL) if FAIL else "ALL PASS")
    sys.exit(1 if FAIL else 0)
