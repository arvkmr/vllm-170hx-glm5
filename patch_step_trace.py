#!/usr/bin/env python3
"""Install the per-step batch-composition trace (GLM52_STEP_TRACE=<dir>).

Copies _glm52_steptrace.py into site-packages/vllm, hooks the model runner
right after the cudagraph dispatch decision (post-reorder batch order, chosen
runtime mode, padding), and hooks the split-MoE shim so the (n_dec, M, on)
each step's MoE saw is recorded next to it. Inert unless the env is set.
"""
import os
import shutil
import sys

import vllm

VLLM = os.path.dirname(vllm.__file__)
HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = "v1/worker/gpu_model_runner.py"
SPLIT = "_glm52_moe_split.py"


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


shutil.copyfile(os.path.join(HERE, "_glm52_steptrace.py"),
                os.path.join(VLLM, "_glm52_steptrace.py"))
print("  + vllm/_glm52_steptrace.py")

edit(
    RUNNER,
    "from vllm import _glm52_shadow_verify, _glm52_spectrace\n",
    "from vllm import _glm52_shadow_verify, _glm52_spectrace\n"
    "from vllm import _glm52_steptrace\n",
    "runner: import",
)

edit(
    RUNNER,
    "                cudagraph_mode,\n"
    "                batch_desc,\n"
    "                should_ubatch,\n"
    "                num_tokens_across_dp,\n"
    "            )\n"
    "\n"
    "            num_tokens_padded = batch_desc.num_tokens\n",
    "                cudagraph_mode,\n"
    "                batch_desc,\n"
    "                should_ubatch,\n"
    "                num_tokens_across_dp,\n"
    "            )\n"
    "            _glm52_steptrace.step(self, scheduler_output, cudagraph_mode,\n"
    "                                  batch_desc, req_ids, tokens)\n"
    "\n"
    "            num_tokens_padded = batch_desc.num_tokens\n",
    "runner: step trace after dispatch",
)

# split shim: record (n_dec, M, on) per step; the installed shim carries the
# stats block from the 09-08 session, anchor on it.
edit(
    SPLIT,
    "        _glm52_ndec = num_decode_tokens_in_batch()\n",
    "        _glm52_ndec = num_decode_tokens_in_batch()\n"
    "        try:\n"
    "            from vllm import _glm52_steptrace as _glm52_st\n"
    "            _glm52_st.moe(_glm52_ndec, int(hidden_states.size(0)), _split_on())\n"
    "        except Exception:\n"
    "            pass\n",
    "split shim: per-step MoE record",
)

import py_compile  # noqa: E402

for p in (RUNNER, SPLIT, "_glm52_steptrace.py"):
    py_compile.compile(os.path.join(VLLM, p), doraise=True)
print("done (compiles)")
