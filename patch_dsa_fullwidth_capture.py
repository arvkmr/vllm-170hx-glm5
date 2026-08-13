#!/usr/bin/env python3
"""Capture FULL cudagraphs with full-width seq lens (DSA decode correctness).

The decode-side indexer logits buffer is allocated at capture-time
max_seq_len. Upstream passes realistic profile_seq_lens only to the FIRST
profiling capture per mode; every other size captures with
seq_lens = max_query_len (4 with MTP k=3), so replays truncate the sparse
top-k window beyond the captured width (and, before
patch_mqa_store_clamp.py, sprayed -inf out of bounds). Fix: pass
profile_seq_lens = max_model_len for EVERY FULL capture. Cost is one
num_tokens x max_model_len fp32 logits buffer in the shared graph pool
(~67 MB at size 64 / 262K ctx) -- the sparse attention itself is
context-independent (top-2048 KV per row), so nothing else scales.

Env: GLM52_DSA_CAPTURE_SEQLEN overrides the width (0 = stock behavior).
With this in place the ctx-gate (patch_dsa_ctx_gate.py) becomes
belt-and-braces: set GLM52_DSA_FULLCG_MAXLEN to the captured width.
"""

import os
import sys

import vllm

VLLM = os.path.dirname(vllm.__file__)
RUNNER = "v1/worker/gpu_model_runner.py"


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

def _glm52_capture_seqlen(self) -> int | None:
    """Full-width seq lens for FULL cudagraph capture (DSA decode buffers
    are capture-sized; undersized ones truncate the sparse top-k window).
    Returns None to use stock behavior when disabled via env=0."""
    v = _os.environ.get("GLM52_DSA_CAPTURE_SEQLEN")
    if v is not None and int(v) == 0:
        return None
    return int(v) if v else self.max_model_len
'''


def main():
    # helper next to the ctx-gate helper (which already imports _os)
    edit(
        RUNNER,
        "def glm52_disable_full_for_long_ctx(scheduler_output) -> bool:",
        HELPER.strip() + "\n\n\ndef glm52_disable_full_for_long_ctx(scheduler_output) -> bool:",
        "runner: capture-seqlen helper",
    )
    # profiling loop: full width for BOTH profile descs, not just i==0
    edit(
        RUNNER,
        "                            profile_seq_lens=(\n"
        "                                min(\n"
        "                                    self.max_model_len,\n"
        "                                    self.max_num_tokens // desc.num_tokens,\n"
        "                                )\n"
        "                                if mode == CUDAGraphMode.FULL and i == 0\n"
        "                                else None\n"
        "                            ),\n",
        "                            profile_seq_lens=(\n"
        "                                (\n"
        "                                    _glm52_capture_seqlen(self)\n"
        "                                    or min(\n"
        "                                        self.max_model_len,\n"
        "                                        self.max_num_tokens\n"
        "                                        // desc.num_tokens,\n"
        "                                    )\n"
        "                                )\n"
        "                                if mode == CUDAGraphMode.FULL\n"
        "                                else None\n"
        "                            ),\n",
        "runner: full-width profiling captures",
    )
    # main capture loop: pass full width for FULL captures (was absent ->
    # seq_lens fell back to max_query_len)
    edit(
        RUNNER,
        "            self._warmup_and_capture(\n"
        "                batch_desc,\n"
        "                cudagraph_runtime_mode=cudagraph_runtime_mode,\n"
        "                allow_microbatching=allow_microbatching,\n"
        "                profiler=profiler,\n"
        "            )",
        "            self._warmup_and_capture(\n"
        "                batch_desc,\n"
        "                cudagraph_runtime_mode=cudagraph_runtime_mode,\n"
        "                allow_microbatching=allow_microbatching,\n"
        "                profiler=profiler,\n"
        "                profile_seq_lens=(\n"
        "                    _glm52_capture_seqlen(self)\n"
        "                    if cudagraph_runtime_mode == CUDAGraphMode.FULL\n"
        "                    else None\n"
        "                ),\n"
        "            )",
        "runner: full-width main captures",
    )


if __name__ == "__main__":
    main()
