# SPDX-License-Identifier: Apache-2.0
"""Per-step batch-composition trace (GLM52_STEP_TRACE=<dir>).

CPU-only, buffered file appends from each worker process; no torch, no GPU
interaction, so it cannot close a race window. One line per execute_model
step AFTER the batch is reordered and the cudagraph mode is dispatched:

  S <n> <t> mode=<FULL|PIECEWISE|NONE> ntok=<unpadded> pad=<padded>
     uniform=<0/1> reqs=<rid>:<qlen>,<rid>:<qlen>,...   (runner batch order)
  M <n> ndec=<n_dec> M=<rows> on=<0/1>   (first MoE apply of the step that
                                          differs from the last logged one)

The request ids carry the client's `request_id`, so a probe's result can be
joined to the exact batch it ran in. See patch_step_trace.py.
"""
import os
import time

_DIR = os.environ.get("GLM52_STEP_TRACE", "")
_st = {"fh": None, "n": 0, "buf": [], "last_moe": None, "hdr": False}


def _write(line):
    st = _st
    st["buf"].append(line)
    if len(st["buf"]) >= 32:
        flush()


def flush():
    st = _st
    if not st["buf"]:
        return
    if st["fh"] is None:
        os.makedirs(_DIR, exist_ok=True)
        st["fh"] = open(os.path.join(_DIR, f"steptrace.pid{os.getpid()}.log"), "a")
    st["fh"].write("".join(st["buf"]))
    st["fh"].flush()
    st["buf"] = []


def step(runner, scheduler_output, cudagraph_mode, batch_desc, req_ids, tokens):
    _st["real"] = True          # a real scheduler batch is about to run (not a dummy/capture run)
    maybe_det_fill()
    if not _DIR:
        return
    try:
        st = _st
        st["n"] += 1
        st["last_moe"] = None
        if not st["hdr"]:
            st["hdr"] = True
            _write(f"H reorder_batch_threshold={getattr(runner, 'reorder_batch_threshold', '?')} "
                   f"uniform_decode_query_len={getattr(runner, 'uniform_decode_query_len', '?')} "
                   f"pid={os.getpid()}\n")
        reqs = ",".join(f"{r[-40:]}:{t}" for r, t in zip(req_ids, tokens))
        _write(f"S {st['n']} {time.monotonic():.3f} mode={cudagraph_mode.name} "
               f"ntok={scheduler_output.total_num_scheduled_tokens} "
               f"pad={batch_desc.num_tokens} uniform={int(bool(batch_desc.uniform))} "
               f"reqs={reqs}\n")
        # steps are ~tens of ms; flush lazily so the trace can't pace the loop
        if st["n"] % 16 == 0:
            flush()
    except Exception as e:  # never break the runner
        _write(f"E {e!r}\n")


def moe(n_dec, m, on):
    """Called from the split-MoE shim per apply; logs on change only."""
    if not _DIR:
        return
    key = (n_dec, m, bool(on))
    if _st["last_moe"] == key:
        return
    _st["last_moe"] = key
    _write(f"M {_st['n']} ndec={n_dec} M={m} on={int(bool(on))}\n")


import atexit  # noqa: E402

atexit.register(flush)


# ---------------------------------------------------------------------------
# Weight-integrity fingerprint (GLM52_WEIGHT_HASH=1; GLM52_WEIGHT_REHASH_EVERY=N)
# Per-layer digests of the raw parameter bytes on the GPU, printed to stderr
# at load and optionally every N steps. Identical checkpoints + deterministic
# loading must give identical digests on every boot; a differing digest on a
# layer means the bytes in VRAM differ (bad memory, load nondeterminism).
_WH = os.environ.get("GLM52_WEIGHT_HASH", "0") == "1"
_WH_EVERY = int(os.environ.get("GLM52_WEIGHT_REHASH_EVERY", "0") or 0)


def weight_hash(model, tag):
    if not _WH:
        return
    import re
    import sys
    import torch
    agg = {}
    with torch.no_grad():
        for name, p in list(model.named_parameters()) + [("B:" + n, b) for n, b in model.named_buffers()]:
            t = p.detach()
            raw = t.reshape(-1).view(torch.uint8)
            n4 = (raw.numel() // 4) * 4
            x = raw[:n4].view(torch.int32)
            h1 = 0
            h2 = 0
            S = 1 << 24
            for k, s in enumerate(range(0, x.numel(), S)):
                v = int(x[s:s + S].sum(dtype=torch.int64))
                h1 += v
                h2 += v * (k + 1)
            if n4 < raw.numel():
                h1 += int(raw[n4:].sum(dtype=torch.int64))
            m = re.match(r"model\.layers\.(\d+)\.", name)
            key = (("B" if name.startswith("B:") else "") + f"L{m.group(1)}") if m else name.split(".")[0]
            a = agg.setdefault(key, [0, 0, 0])
            a[0] += h1
            a[1] += h2
            a[2] += raw.numel()
    parts = " ".join(f"{k}:{(v[0] & 0xFFFFFFFFFFFF):012x}/{(v[1] & 0xFFFFFFFFFFFF):012x}/{v[2]}" for k, v in agg.items())
    print(f"[glm52-whash] tag={tag} pid={os.getpid()} {parts}", file=sys.stderr, flush=True)


def maybe_rehash(runner):
    if not _WH or not _WH_EVERY:
        return
    st = _st
    st["rehash_n"] = st.get("rehash_n", 0) + 1
    if st["rehash_n"] % _WH_EVERY == 0:
        try:
            weight_hash(runner.model, f"step{st['rehash_n']}")
        except Exception as e:
            print(f"[glm52-whash] rehash error {e!r}", flush=True)


# ---------------------------------------------------------------------------
# GLM52_ZERO_KV_ONCE=1: zero every KV / indexer cache tensor once, right before the
# first real execute_model step (after capture + warm-up). Boot-lottery bisect: if two
# boots become identical, the per-boot state lives in stale cache contents read past
# the written range.
_ZKV = os.environ.get("GLM52_ZERO_KV_ONCE", "0") == "1"


def maybe_zero_kv(runner):
    if not _ZKV or _st.get("zkv_done"):
        return
    _st["zkv_done"] = True
    import sys
    import torch
    n = 0
    tot = 0
    with torch.no_grad():
        for kv in getattr(runner, "kv_caches", []) or []:
            if isinstance(kv, torch.Tensor) and kv.numel():
                kv.zero_(); n += 1; tot += kv.numel() * kv.element_size()
        for name, buf in list(getattr(runner, "__dict__", {}).items()):
            pass
    torch.cuda.synchronize()
    print(f"[glm52-zerokv] zeroed {n} cache tensors ({tot/2**30:.1f} GiB) before the first real step",
          file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# GLM52_ZERO_RUNNER_BUFS=1: before the FIRST real execute_model (before its inputs are
# prepared), zero the runner's persistent per-step metadata/input tensors (positions,
# input_ids, query_start_loc, seq_lens, slot_mapping, block tables, attention-builder
# scratch, intermediate-tensor buffers) and the model's persistent indexer/relay buffers.
# Boot-lottery bisect for "stale tail" reads (an entry one past the valid range holding
# a value from a warm-up/capture batch).
_ZRB = os.environ.get("GLM52_ZERO_RUNNER_BUFS", "0") == "1"
_ZRB_NAMES = ("input_ids", "positions", "inputs_embeds", "query_start_loc", "seq_lens",
              "slot_mapping", "num_computed_tokens", "block_table", "mrope_positions",
              "topk_indices_buffer", "relay_buf", "intermediate", "decode_lens",
              "cu_seq", "req_id_per_token", "logits_indices", "query_lens")


def _zero_obj(obj, depth, seen, out, force=False):
    import torch
    if depth < 0 or id(obj) in seen:
        return
    seen.add(id(obj))
    for name, val in list(getattr(obj, "__dict__", {}).items()):
        want = force or any(k in name for k in _ZRB_NAMES)
        if isinstance(val, torch.Tensor):
            if want and val.is_cuda and val.numel():
                val.zero_(); out.append(f"{type(obj).__name__}.{name}")
        elif isinstance(val, (list, tuple)) and val and isinstance(val[0], torch.Tensor):
            if want:
                for i, t in enumerate(val):
                    if isinstance(t, torch.Tensor) and t.is_cuda and t.numel():
                        t.zero_(); out.append(f"{type(obj).__name__}.{name}[{i}]")
        elif hasattr(val, "gpu") and isinstance(getattr(val, "gpu"), torch.Tensor):
            if want:
                val.gpu.zero_(); out.append(f"{type(obj).__name__}.{name}.gpu")
                cpu = getattr(val, "cpu", None)
                if isinstance(cpu, torch.Tensor):
                    cpu.zero_()
        elif hasattr(val, "__dict__") and not isinstance(val, (type, torch.nn.Module)) and depth > 0 \
                and any(k in name for k in ("input_batch", "block_table", "builder", "attn_group", "metadata")):
            _zero_obj(val, depth - 1, seen, out, force=("builder" in name or "block_table" in name))
        elif isinstance(val, (list, tuple)) and depth > 0 and any(k in name for k in ("attn_groups", "builders")):
            for v in val:
                if isinstance(v, (list, tuple)):
                    for vv in v:
                        _zero_obj(vv, depth - 1, seen, out)
                else:
                    _zero_obj(v, depth - 1, seen, out)


def maybe_zero_runner_bufs(runner):
    if not _ZRB or _st.get("zrb_done"):
        return
    _st["zrb_done"] = True
    import sys
    import torch
    out = []
    with torch.no_grad():
        _zero_obj(runner, 3, set(), out)
        # model-side persistent buffers (indexer topk buffer, PP relay buffer)
        m = getattr(runner, "model", None)
        for mod in ([m] + [getattr(m, "model", None)] if m is not None else []):
            if mod is None:
                continue
            for name, val in list(vars(mod).items()):
                if isinstance(val, torch.Tensor) and val.is_cuda and any(k in name for k in _ZRB_NAMES):
                    val.zero_(); out.append(f"model.{name}")
    torch.cuda.synchronize()
    print(f"[glm52-zerobufs] zeroed {len(out)} tensors: {sorted(set(out))[:40]}", file=sys.stderr, flush=True)


# ---- allocator poison (GLM52_POISON_FLAG=<path>): before each step, if the flag file
# exists, grab every free block the caching allocator (and the driver) will give us,
# fill it with the byte the file names (hex, default ff), and release it back to the
# pool WITHOUT empty_cache. Any later read-before-write of an activation/Inductor/
# workspace buffer then sees the pattern instead of whatever the previous tenant left.
_POISON_FLAG = os.environ.get("GLM52_POISON_FLAG", "")


def maybe_poison(runner):
    if not _POISON_FLAG or not os.path.exists(_POISON_FLAG):
        return
    import sys
    import torch
    try:
        byte = int(open(_POISON_FLAG).read().strip() or "ff", 16) & 0xFF
    except Exception:
        byte = 0xFF
    torch.cuda.synchronize()
    held, total, sz = [], 0, 1 << 30
    while sz >= (1 << 20):
        try:
            t = torch.empty(sz, dtype=torch.uint8, device="cuda")
        except RuntimeError:
            sz >>= 1
            continue
        t.fill_(byte); held.append(t); total += sz
    torch.cuda.synchronize()
    del held
    key = ("poison", byte)
    if key not in _st:
        _st[key] = True
        print(f"[glm52-poison] byte=0x{byte:02x} filled {total/2**20:.0f} MiB of free blocks (first time)",
              file=sys.stderr, flush=True)


# ---- custom-op boundary hashes (GLM52_OPHASH=<layers csv, e.g. "0,1">): the Python-
# implemented custom ops (sparse_attn_indexer, unified_mla_attention_with_output) run as
# Python even inside the compiled graph, so their inputs/outputs can be fingerprinted in
# compiled mode. First GLM52_OPHASH_CALLS calls per (op, layer) are printed with shape,
# stride, 4K-page offset of data_ptr and sha256 of the bytes. One sync per hashed tensor.
_OPH = [s.strip() for s in os.environ.get("GLM52_OPHASH", "").split(",") if s.strip()]
_OPH_CALLS = int(os.environ.get("GLM52_OPHASH_CALLS", "6") or 6)


def _layer_idx(name):
    import re
    m = re.search(r"layers\.(\d+)\.", str(name))
    return m.group(1) if m else None


def ophash(op, layer_name, **tensors):
    if not _OPH:
        return
    li = _layer_idx(layer_name)
    if li not in _OPH or not _st.get("real"):
        return
    key = ("oph", op, li)
    n = _st.get(key, 0)
    if n >= _OPH_CALLS:
        return
    _st[key] = n + 1
    import hashlib
    import sys
    import torch
    parts = []
    for k, t in tensors.items():
        if not isinstance(t, torch.Tensor):
            parts.append(f"{k}={t!r}")
            continue
        try:
            b = t.detach().contiguous().view(torch.uint8) if t.dtype != torch.bool else t.detach().contiguous().to(torch.uint8)
            h = hashlib.sha256(b.cpu().numpy().tobytes()).hexdigest()[:12]
        except Exception as e:  # noqa: BLE001
            h = f"ERR:{e!r}"[:30]
        parts.append(f"{k}={h} {tuple(t.shape)}/{tuple(t.stride())}/{str(t.dtype).replace('torch.','')}/off{t.data_ptr() % 4096}")
    print(f"[glm52-ophash] L{li} {op} call{n} " + " ".join(parts), file=sys.stderr, flush=True)


# ---- deterministic fill of uninitialized memory (GLM52_DET_FILL=1): from the FIRST REAL
# step on (after compile/capture, which Inductor refuses to do in deterministic mode), every
# torch.empty / empty_strided (Inductor buffers included) is filled with NaN (floats) / max
# (ints) at allocation. Two boots then see identical "garbage"; NaN in the output pinpoints
# a real read-before-write. Needs CUBLAS_WORKSPACE_CONFIG=:4096:8 to keep cuBLAS quiet.
_DET_FILL = os.environ.get("GLM52_DET_FILL", "0") == "1"


def maybe_det_fill():
    if not _DET_FILL or _st.get("detfill_on"):
        return
    _st["detfill_on"] = True
    import sys
    import torch
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    print(f"[glm52-detfill] pid={os.getpid()} fill_uninitialized_memory=ON from step {_st.get('n')} "
          f"(deterministic algorithms, warn_only)", file=sys.stderr, flush=True)


# ---- load-time tracer for 1-byte device->device Tensor.to() copies (GLM52_TRACE_TO1=1):
# wraps torch.Tensor.to during load_model only (uninstalled before any compile), prints the
# Python call site of the first few CUDA 1-byte-element .to() calls and a per-site tally.
_TO1 = os.environ.get("GLM52_TRACE_TO1", "0") == "1"


def post_load(model):
    """Post-load fixups. GLM52_GATE_FP32=1: pre-cast every MoE gate weight to fp32 once
    (6 MB/layer) so the router GEMM does not re-cast the weight on every step."""
    import sys
    import torch
    if os.environ.get("GLM52_GATE_FP32", "0") != "1" or model is None:
        return
    n = 0
    for name, m in model.named_modules():
        if type(m).__name__ == "GateLinear" and getattr(m, "weight", None) is not None:
            m.weight_fp32 = m.weight.detach().to(torch.float32)
            n += 1
    print(f"[glm52-gate-fp32] pre-cast {n} gate weights to fp32", file=sys.stderr, flush=True)


def wrap_load_model(fn):
    import functools as _ft

    if not _TO1:
        @_ft.wraps(fn)
        def load_model_pl(self, *a, **k):
            r = fn(self, *a, **k)
            post_load(getattr(self, "model", None))
            return r
        return load_model_pl
    import collections
    import functools
    import sys
    import traceback
    import torch
    orig = torch.Tensor.to
    tally = collections.Counter()
    shown = [0]

    def to_wrapped(self, *a, **k):
        try:
            if self.is_cuda and self.element_size() == 1:
                frames = [fr for fr in traceback.extract_stack(limit=14)[:-1]
                          if "torch/" not in fr.filename and "_glm52_steptrace" not in fr.filename]
                site = f"{frames[-1].filename.split('site-packages/')[-1]}:{frames[-1].lineno} {frames[-1].name}" if frames else "?"
                tally[(site, str(self.dtype), str(a[:1]))] += 1
                if shown[0] < 6:
                    shown[0] += 1
                    print(f"[glm52-to1] {tuple(self.shape)} {self.dtype} contig={self.is_contiguous()} args={a} {k}\n"
                          + "".join(traceback.format_stack(limit=10)[:-1]), file=sys.stderr, flush=True)
        except Exception:
            pass
        return orig(self, *a, **k)

    @functools.wraps(fn)
    def load_model(self, *a, **k):
        torch.Tensor.to = to_wrapped
        try:
            return fn(self, *a, **k)
        finally:
            torch.Tensor.to = orig
            print(f"[glm52-to1] tally: {tally.most_common(12)}", file=sys.stderr, flush=True)
        post_load(getattr(self, "model", None))
    return load_model
