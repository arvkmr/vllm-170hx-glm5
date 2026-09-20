#!/usr/bin/env python3
"""Make the PP decode batch cap adapt to how many requests are decoding.

GLM52_PP_DECODE_BATCH_CAP splits ready decodes into groups so they pipeline
across the PP stages instead of forming one lockstep batch. The right cap is
not a constant: it trades pipeline depth (favours a small cap -- more groups)
against MoE batch efficiency (favours a large cap -- Marlin is barely cheaper
per token at 4 rows than at 8, so bigger batches amortise its fixed floor).

Which side wins depends on the number of decoding requests, not the context
length. Measured this box, fp8, gate off, aggregate decode tok/s:

    workload      cap=2    cap=1
    conc=4  (512 ctx)   78.8   101.3
    conc=8  (512 ctx)  163.6   135.5
    conc=16 (512 ctx)  189.7   142.7
    4 x 64K             90.8   100.3
    4 x 192K            45.8    67.7

On the original PP=8 measurements, below ~8 decoding requests there were too
few groups to fill the pipe, so cap=1 won (up to +48% at 4x192K). At 8+
requests the larger batch won instead. The serve scripts use the PP size as
the threshold, making the topology-derived default 12 on the twelve-card box.

So: cap 1 while fewer than GLM52_PP_DECODE_ADAPTIVE requests are decoding,
otherwise the configured GLM52_PP_DECODE_BATCH_CAP. Set
GLM52_PP_DECODE_ADAPTIVE=0 to disable and go back to a constant cap.
"""

import os
import shutil
import sys

import vllm

VLLM = os.path.dirname(vllm.__file__)
BACKUP = os.path.join(os.path.dirname(VLLM), ".glm52-backup", "vllm")
SCHED = "v1/core/sched/scheduler.py"

_EDITS: list[tuple[str, str, str, str]] = []


def edit(path, old, new, label):
    _EDITS.append((path, old, new, label))


def apply_all():
    files = {}
    todo = []
    for path, old, new, label in _EDITS:
        src = files.get(path) or open(os.path.join(VLLM, path)).read()
        if new in src:
            print(f"  = {label} (already applied)")
            files[path] = src
            continue
        if src.count(old) != 1:
            print(
                f"  ! {label}: anchor matched {src.count(old)} times in {path}, "
                f"expected 1 -- nothing written",
                file=sys.stderr,
            )
            sys.exit(1)
        files[path] = src.replace(old, new)
        todo.append(label)
    for path in files:
        dst = os.path.join(BACKUP, path)
        if not os.path.exists(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(os.path.join(VLLM, path), dst)
            print(f"  b backup -> {dst}")
    for path, src in files.items():
        open(os.path.join(VLLM, path), "w").write(src)
    for label in todo:
        print(f"  + {label}")


edit(
    SCHED,
    '_PP_DECODE_BATCH_CAP = int(os.environ.get("GLM52_PP_DECODE_BATCH_CAP", "0") or 0)',
    '_PP_DECODE_BATCH_CAP = int(os.environ.get("GLM52_PP_DECODE_BATCH_CAP", "0") or 0)\n'
    "# Below this many decoding requests, drop the cap to 1: with few requests\n"
    "# there are too few groups to fill the 8-stage pipe, and depth beats batch\n"
    "# size. 0 disables the adaptation and keeps the cap constant.\n"
    "_PP_DECODE_ADAPTIVE = int(\n"
    '    os.environ.get("GLM52_PP_DECODE_ADAPTIVE", "8") or 0\n'
    ")",
    "adaptive cap threshold",
)

edit(
    SCHED,
    "        # First, schedule the RUNNING requests.\n"
    "        req_index = 0\n"
    "        _decode_in_batch = 0",
    "        # First, schedule the RUNNING requests.\n"
    "        req_index = 0\n"
    "        _decode_in_batch = 0\n"
    "        _cap = _PP_DECODE_BATCH_CAP\n"
    "        if _PP_DECODE_ADAPTIVE > 0 and _cap > 0:\n"
    "            _n_decoding = sum(\n"
    "                1\n"
    "                for r in self.running\n"
    "                if r.num_computed_tokens >= r.num_prompt_tokens\n"
    "            )\n"
    "            if _n_decoding < _PP_DECODE_ADAPTIVE:\n"
    "                _cap = 1",
    "compute per-schedule effective cap",
)

edit(
    SCHED,
    "                and not getattr(request, \"_pp_spec_in_flight\", False)\n"
    "                and _decode_in_batch >= _PP_DECODE_BATCH_CAP\n"
    "            ):",
    "                and not getattr(request, \"_pp_spec_in_flight\", False)\n"
    "                and _decode_in_batch >= _cap\n"
    "            ):",
    "use the effective cap",
)

apply_all()
print("\nAdaptive PP decode cap enabled "
      "(GLM52_PP_DECODE_ADAPTIVE=0 to revert to a constant cap).")
