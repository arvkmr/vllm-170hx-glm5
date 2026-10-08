#!/usr/bin/env python3
"""Per-GPU silent-corruption check (no vLLM involved).

Each visible GPU, in its own process, for --seconds:
  * recomputes the same BF16 GEMM (decode-like shape, 16 x 6144 @ 6144 x 24576)
    and compares it bit-for-bit with its own first result, and
  * writes a pseudo-random pattern over a large buffer and reads it back.
A correct GPU never reports a mismatch. Counts are per GPU, with the PCI id.

    python3 gpu_consistency.py --seconds 300 --mem-gib 8
"""

import argparse
import multiprocessing as mp
import time


def worker(dev: int, seconds: float, mem_gib: float, q) -> None:
    import torch

    torch.cuda.set_device(dev)
    props = torch.cuda.get_device_properties(dev)
    pci = f"{props.pci_domain_id:04x}:{props.pci_bus_id:02x}:{props.pci_device_id:02x}"
    g = torch.Generator(device="cuda").manual_seed(1234)
    a = torch.randn(16, 6144, device="cuda", dtype=torch.bfloat16, generator=g)
    b = torch.randn(6144, 24576, device="cuda", dtype=torch.bfloat16, generator=g)
    ref = a @ b
    zero_ref_rows = int((ref.abs().amax(-1) == 0).sum())
    n_elems = int(mem_gib * (1 << 30) // 4)
    buf = torch.empty(n_elems, dtype=torch.int32, device="cuda")
    idx = torch.arange(n_elems, dtype=torch.int32, device="cuda")
    gemm_iters = gemm_bad = gemm_zero_rows = mem_passes = mem_bad = 0
    t_end = time.time() + seconds
    salt = 0
    while time.time() < t_end:
        for _ in range(200):
            out = a @ b
            diff = out != ref
            if bool(diff.any()):
                gemm_bad += 1
                gemm_zero_rows += int((out.abs().amax(-1) == 0).sum())
            gemm_iters += 1
        salt += 1
        pattern = (idx * 2654435761 + salt * 40503) & 0x7FFFFFFF
        buf.copy_(pattern)
        mem_bad += int((buf != pattern).sum())
        mem_passes += 1
    q.put(
        (dev, pci, gemm_iters, gemm_bad, gemm_zero_rows, zero_ref_rows, mem_passes, mem_bad)
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=300)
    ap.add_argument("--mem-gib", type=float, default=8)
    a = ap.parse_args()
    import torch

    n = torch.cuda.device_count()
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    ps = [ctx.Process(target=worker, args=(d, a.seconds, a.mem_gib, q)) for d in range(n)]
    for p in ps:
        p.start()
    rows = sorted(q.get() for _ in ps)
    for p in ps:
        p.join()
    print("dev pci           gemm_iters gemm_mismatch zero_rows mem_passes mem_bad")
    for dev, pci, gi, gb, gz, zr, mp_, mb in rows:
        flag = "  <-- FAULT" if gb or mb else ""
        print(f"{dev:3d} {pci}  {gi:10d} {gb:13d} {gz:9d} {mp_:10d} {mb:7d}{flag}")


if __name__ == "__main__":
    main()
