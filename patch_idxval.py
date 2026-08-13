#!/usr/bin/env python3
"""DSA indexer prefill topk bounds validator (env GLM52_IDXVAL=1; idempotent).

Coredump evidence (core_fat_587762.nvcudmp): vllm::topKPerRowPrefill MMU-
faults on rank 1 scanning column ~896 of a logits row that is only ~14 wide
-- cu_seqlen_ke contained another request's context length at kernel exec
time. This validator checks max(ke) <= logits.shape[1] and ks<=ke for every
prefill topk launch, DEFERRED by two calls (the async D2H has landed by
then; no added synchronization on the launch path -- race-preserving).
Device-side clones of ks/ke are kept so the corrupt values can be printed
in full when the check fires.
"""

import os
import sys

import vllm

VLLM = os.path.dirname(vllm.__file__)
SAI = "model_executor/layers/sparse_attn_indexer.py"
MPE = "v1/executor/multiproc_executor.py"


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


HELPER = '''
_IDXVAL_MODE = int(_os.environ.get("GLM52_IDXVAL", "0") or 0)
_IDXVAL = _IDXVAL_MODE > 0
_idxval_pending: list = []
_IDXVAL_RING_N = 8
_IDXVAL_ROWS = 64
_idxval_ring: list = []
_idxval_call = 0
_idxval_last_pre: list = []


def _idxval_post(ks, ke):
    """Mode 3: after the topk kernel is enqueued, enqueue a D2H re-read of
    ks/ke into the pinned ring.  It lands after the kernel executes; the
    next call compares it against the pre-launch snapshot."""
    global _idxval_call
    if _IDXVAL_MODE != 3 or not _idxval_last_pre:
        return
    if not _idxval_ring:
        for _ in range(_IDXVAL_RING_N):
            _idxval_ring.append({
                "ks": torch.zeros(_IDXVAL_ROWS, dtype=torch.int32,
                                  pin_memory=True),
                "ke": torch.zeros(_IDXVAL_ROWS, dtype=torch.int32,
                                  pin_memory=True),
                "meta": None,
            })
    pre_ks, pre_ke, width = _idxval_last_pre
    slot = _idxval_ring[(_idxval_call - 1) % _IDXVAL_RING_N]
    n = min(len(pre_ks), _IDXVAL_ROWS)
    slot["ks"][:n].copy_(ks[:n], non_blocking=True)
    slot["ke"][:n].copy_(ke[:n], non_blocking=True)
    slot["meta"] = (_idxval_call - 1, pre_ks[:n], pre_ke[:n], width)


def _idxval_check_and_arm(logits, ks, ke):
    """Bounds-check ks/ke vs logits width, deferred two calls (no sync).
    Also snapshot ks/ke into a pinned ring so a crash handler can print
    the last few calls' values post-mortem (async D2H, no sync here)."""
    global _idxval_pending, _idxval_call
    if _IDXVAL_MODE >= 2:
        # Synchronous pre-launch check: values are already baked by the
        # metadata-build kernel, so this sync cannot un-corrupt them --
        # it converts the would-be IMA into a clean report.  Result of the
        # =2 campaign: this check PASSED yet the kernel still faulted =>
        # the device values change between enqueue and execution (a
        # cross-stream writer).  Mode 3 re-reads them post-execution.
        ks_l = ks.tolist()
        ke_l = ke.tolist()
        width = logits.shape[1]
        if (max(ke_l) > width or min(ks_l) < 0
                or any(a > b for a, b in zip(ks_l, ke_l))):
            raise RuntimeError(
                f"GLM52 IDXVAL(sync): prefill topk bounds violation: "
                f"width={width} rows={logits.shape[0]}\\n"
                f"ks={ks_l}\\nke={ke_l}"
            )
        if _IDXVAL_MODE == 3:
            # verify post-exec snapshots armed >=2 calls ago
            for slot in _idxval_ring:
                if slot["meta"] is None:
                    continue
                call, pre_ks, pre_ke, w = slot["meta"]
                if _idxval_call - call < 2:
                    continue
                n = len(pre_ks)
                post_ks = slot["ks"][:n].tolist()
                post_ke = slot["ke"][:n].tolist()
                if post_ks != pre_ks or post_ke != pre_ke:
                    raise RuntimeError(
                        f"GLM52 IDXVAL(drift): ks/ke device memory was "
                        f"REWRITTEN after enqueue (call#{call}, width={w}):\\n"
                        f"pre_ks ={pre_ks}\\npost_ks={post_ks}\\n"
                        f"pre_ke ={pre_ke}\\npost_ke={post_ke}"
                    )
                slot["meta"] = None
            _idxval_last_pre.clear()
            _idxval_last_pre.extend([ks_l, ke_l, width])
        _idxval_call += 1
        return
    while len(_idxval_pending) > 2:
        stat_cpu, width, rows, ks_c, ke_c = _idxval_pending.pop(0)
        max_ke, min_ks, max_ks = (int(x) for x in stat_cpu.tolist())
        if max_ke > width or min_ks < 0 or max_ks > max_ke:
            raise RuntimeError(
                f"GLM52 IDXVAL: prefill topk bounds violation: "
                f"max_ke={max_ke} min_ks={min_ks} max_ks={max_ks} vs "
                f"logits width={width} rows={rows}\\n"
                f"ks={ks_c.tolist()}\\nke={ke_c.tolist()}"
            )
    stat = torch.stack([ke.max(), ks.min(), ks.max()]).to(
        "cpu", non_blocking=True
    )
    _idxval_pending.append(
        (stat, logits.shape[1], logits.shape[0], ks.clone(), ke.clone())
    )
    if not _idxval_ring:
        for _ in range(_IDXVAL_RING_N):
            _idxval_ring.append({
                "ks": torch.zeros(_IDXVAL_ROWS, dtype=torch.int32,
                                  pin_memory=True),
                "ke": torch.zeros(_IDXVAL_ROWS, dtype=torch.int32,
                                  pin_memory=True),
                "meta": None,
            })
    slot = _idxval_ring[_idxval_call % _IDXVAL_RING_N]
    n = min(ks.shape[0], _IDXVAL_ROWS)
    slot["ks"][:n].copy_(ks[:n], non_blocking=True)
    slot["ke"][:n].copy_(ke[:n], non_blocking=True)
    slot["meta"] = (_idxval_call, logits.shape[0], logits.shape[1], n)
    _idxval_call += 1


def _idxval_dump():
    """Called from the worker crash handler: print recent ks/ke snapshots.
    Entries >=2 calls old have certainly landed; newer ones may be torn."""
    out = []
    for slot in _idxval_ring:
        if slot["meta"] is None:
            continue
        call, rows, width, n = slot["meta"]
        age = _idxval_call - call
        out.append(
            f"IDXVAL call#{call} (age {age}) rows={rows} width={width} "
            f"ks={slot['ks'][:n].tolist()} ke={slot['ke'][:n].tolist()}"
        )
    return "\\n".join(sorted(out)) if out else "IDXVAL: no snapshots"
'''


def main():
    edit(
        SAI,
        "                num_rows = logits.shape[0]\n"
        "                _tk = _pt() if _PROF else 0.0\n"
        "                ops.top_k_per_row_prefill(\n",
        "                num_rows = logits.shape[0]\n"
        "                if _IDXVAL:\n"
        "                    _idxval_check_and_arm(\n"
        "                        logits, cu_seqlen_ks, cu_seqlen_ke\n"
        "                    )\n"
        "                if _IDXVAL_MODE == 3:\n"
        "                    # raw exec-time snapshot (lands where the kernel\n"
        "                    # would have read), then sanitize in place so the\n"
        "                    # corrupt instance SURVIVES instead of faulting.\n"
        "                    _idxval_post(cu_seqlen_ks, cu_seqlen_ke)\n"
        "                    cu_seqlen_ke.clamp_(0, logits.shape[1])\n"
        "                    cu_seqlen_ks.clamp_(0, logits.shape[1])\n"
        "                _tk = _pt() if _PROF else 0.0\n"
        "                ops.top_k_per_row_prefill(\n",
        "idxval: check before prefill topk",
    )
    # helper after the _PROF flag line (where _os is already imported)
    edit(
        SAI,
        '_PROF = _os.environ.get("GLM52_PROF") == "1"\n',
        '_PROF = _os.environ.get("GLM52_PROF") == "1"\n' + HELPER,
        "idxval: helper",
    )
    # crash handler: print the pinned ring when a worker dies
    edit(
        MPE,
        '                logger.exception("WorkerProc hit an exception.")\n',
        '                logger.exception("WorkerProc hit an exception.")\n'
        '                try:\n'
        '                    from vllm.model_executor.layers import (\n'
        '                        sparse_attn_indexer as _sai)\n'
        '                    if getattr(_sai, "_idxval_ring", None):\n'
        '                        logger.error("IDXVAL post-mortem:\\n%s",\n'
        '                                     _sai._idxval_dump())\n'
        '                except Exception:\n'
        '                    pass\n',
        "idxval: crash-time ring dump",
    )


if __name__ == "__main__":
    main()
