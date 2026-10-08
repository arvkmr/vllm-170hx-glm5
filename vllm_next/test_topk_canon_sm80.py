#!/usr/bin/env python3
"""GPU gate for GLM53_TOPK_CANON (v0.26 canonical top-k ported to vNext).

Checks the decode dispatcher (native top-k + canonical rebuild) and the
prefill path (top_k_per_row_prefill + canon_topk_indices_prefill) against
the exact canonical set: score desc, index asc on ties; relative indices and
identity short rows for prefill. Out-of-window logits are poisoned.
"""
import os
os.environ["GLM53_TOPK_CANON"] = "1"
import torch
from vllm.v1.worker.workspace import init_workspace_manager
init_workspace_manager(torch.device("cuda"))
from vllm import _custom_ops as ops
from vllm.model_executor.layers.indexer_topk import get_indexer_topk
from vllm._glm52_topk_canon import canon_topk_indices_prefill
K = 2048
FAIL = []

def expect(seg):
    w = seg.numel()
    if w <= K:
        return set(range(w))
    order = sorted(range(w), key=lambda i: (-float(seg[i]), i))
    return set(order[:K])

def check_rows(name, got, spec_segs):
    bad = 0
    for r, seg in enumerate(spec_segs):
        g = got[r][got[r] >= 0].tolist()
        if len(g) != len(set(g)) or set(g) != expect(seg.cpu()):
            bad += 1
    ok = bad == 0
    print(f"  {'PASS' if ok else 'FAIL'} {name}: bad rows {bad}/{len(spec_segs)}")
    if not ok: FAIL.append(name)

torch.manual_seed(0)
for poison in (float("nan"), float("inf"), 1e30):
    # decode: B requests x next_n rows, ties via coarse grid, width = max_model_len
    W = 300000
    for next_n, lens in ((1, [5000, 1500, 60000, 2048]), (8, [30000, 9000])):
        B = len(lens)
        seq = torch.tensor(lens, device="cuda", dtype=torch.int32).view(B, 1)
        if next_n > 1:
            seq = (seq - torch.arange(next_n - 1, -1, -1, device="cuda", dtype=torch.int32)[None])
        rows = B * next_n
        lg = ((torch.randn(rows, W, device="cuda") * 4).round() / 4)
        ends = seq.reshape(-1)
        lg = torch.where(torch.arange(W, device="cuda")[None] < ends[:, None], lg, torch.full_like(lg, poison))
        out = torch.full((rows, K), -1, device="cuda", dtype=torch.int32)
        get_indexer_topk("auto")(lg, seq, next_n, out, K, int(ends.max()))
        check_rows(f"decode next_n={next_n} poison={poison}", out.cpu(), [lg[r, :int(ends[r])] for r in range(rows)])
    # prefill: co-chunked requests, relative indices, short rows identity
    spec = [(0, 1500)] * 2 + [(1500, 40000)] * 3 + [(41500, 2048)] * 2 + [(43548, 9000)] * 2
    N = 60000; R = len(spec)
    lg = ((torch.randn(R, N, device="cuda") * 4).round() / 4)
    ks = torch.tensor([a for a, _ in spec], device="cuda", dtype=torch.int32)
    ke = torch.tensor([a + w for a, w in spec], device="cuda", dtype=torch.int32)
    ar = torch.arange(N, device="cuda")[None]
    lg = torch.where((ar >= ks[:, None]) & (ar < ke[:, None]), lg, torch.full_like(lg, poison))
    out = torch.full((R, K), -1, device="cuda", dtype=torch.int32)
    ops.top_k_per_row_prefill(lg, ks, ke, out, R, lg.stride(0), lg.stride(1), K)
    canon_topk_indices_prefill(lg, out, ks, ke, K)
    check_rows(f"prefill co-chunked poison={poison}", out.cpu(), [lg[r, a:a + w] for r, (a, w) in enumerate(spec)])
    short = [r for r, (a, w) in enumerate(spec) if w <= K]
    ident = all(out[r, :spec[r][1]].tolist() == list(range(spec[r][1])) for r in short)
    print(f"  {'PASS' if ident else 'FAIL'} prefill short rows identity order poison={poison}")
    if not ident: FAIL.append("identity")
# hostile native selections: out-of-window / huge entries must never
# surface as output indices (2026-09-28 OOB gather on a 57K prompt).
from vllm._glm52_topk_canon import canon_topk_indices
torch.manual_seed(1)
for trial in range(3):
    R, N = 16, 70000
    lg = ((torch.randn(R, N, device="cuda") * 4).round() / 4)
    ks = torch.randint(0, 5000, (R,), device="cuda", dtype=torch.int32)
    ke = (ks + torch.randint(3000, 60000, (R,), device="cuda", dtype=torch.int32)).clamp(max=N)
    sel = torch.randint(-5, 2**30, (R, K), device="cuda", dtype=torch.int32)
    canon_topk_indices_prefill(lg, sel, ks, ke, K)
    w = (ke - ks)[:, None]
    ok_p = bool(((sel == -1) | ((sel >= 0) & (sel < w))).all())
    seq = torch.randint(3000, N, (R,), device="cuda", dtype=torch.int32)
    sel2 = torch.randint(-5, 2**30, (R, K), device="cuda", dtype=torch.int32)
    canon_topk_indices(lg, sel2, seq)
    ok_d = bool(((sel2 == -1) | ((sel2 >= 0) & (sel2 < seq[:, None]))).all())
    pad_last = all((r[r >= 0].numel() == 0) or bool((r[: (r >= 0).sum()] >= 0).all()) for r in sel2.cpu())
    for name, ok in ((f"hostile prefill sel trial {trial}", ok_p), (f"hostile decode sel trial {trial}", ok_d and pad_last)):
        print(f"  {'PASS' if ok else 'FAIL'} {name}: all outputs -1 or in-window")
        if not ok: FAIL.append(name)
# timing at production width
W = 950000; rows = 8
lg = torch.randn(rows, W, device="cuda"); seq = torch.full((1, 8), 100000, device="cuda", dtype=torch.int32) - torch.arange(7, -1, -1, device="cuda", dtype=torch.int32)[None]
out = torch.empty(rows, K, device="cuda", dtype=torch.int32)
f = get_indexer_topk("auto")
for _ in range(3): f(lg, seq, 8, out, K, 100000)
torch.cuda.synchronize(); import time; t = time.time()
for _ in range(20): f(lg, seq, 8, out, K, 100000)
torch.cuda.synchronize(); print(f"  decode top-k + canon, 8 rows @100K ctx, 950K width: {(time.time()-t)/20*1000:.2f} ms/call")
print('ALL PASS' if not FAIL else 'FAILED: ' + ", ".join(FAIL))
