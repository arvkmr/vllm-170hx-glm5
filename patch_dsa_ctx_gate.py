#!/usr/bin/env python3
"""Hybrid DSA cudagraph gate: FULL graphs only for short-context batches.

With the store clamp (patch_mqa_store_clamp.py) FULL-graph DSA decode is
memory-safe, and for contexts <= topk (2048) it is also EXACT: the decode
top-k takes the identity shortcut (rowLen <= topK emits identity indices)
and never reads the logits buffer. Beyond 2048 the capture-sized buffer
silently truncates the sparse window, so those batches must run PIECEWISE.

This patch threads a deterministic long-context flag (computed from
scheduler_output, identical on every PP rank -- the batch descriptor feeds
PP recv shapes, so rank-local state must not influence it) into the
cudagraph dispatch as disable_full. The capture/dummy path is untouched.

Use with GLM52_DSA_FULLCG=1 (patch_dsa_no_fullcg.py escape hatch) to
restore UNIFORM_BATCH support; threshold env GLM52_DSA_FULLCG_MAXLEN
(default 2048, 0 disables the gate).
"""

import os
import sys

import vllm

VLLM = os.path.dirname(vllm.__file__)
RUNNER = "v1/worker/gpu_model_runner.py"
WORKER = "v1/worker/gpu_worker.py"


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

import os as _os

_GLM52_DSA_FULLCG_MAXLEN = int(
    _os.environ.get("GLM52_DSA_FULLCG_MAXLEN", "2048") or 0
)


def glm52_disable_full_for_long_ctx(scheduler_output) -> bool:
    """True if any request in this batch exceeds the exact-FULL-graph
    context bound. Computed from scheduler_output only, so every PP rank
    reaches the same verdict (the batch descriptor feeds recv shapes)."""
    if _GLM52_DSA_FULLCG_MAXLEN <= 0:
        return False
    n = scheduler_output.num_scheduled_tokens
    c = scheduler_output.scheduled_cached_reqs
    for i, rid in enumerate(c.req_ids):
        if c.num_computed_tokens[i] + n[rid] > _GLM52_DSA_FULLCG_MAXLEN:
            return True
    for r in scheduler_output.scheduled_new_reqs:
        if r.num_computed_tokens + n[r.req_id] > _GLM52_DSA_FULLCG_MAXLEN:
            return True
    return False
'''


def main():
    # helper + env constant at module level (after logger init)
    edit(
        RUNNER,
        "logger = init_logger(__name__)\n",
        "logger = init_logger(__name__)\n" + HELPER,
        "runner: long-ctx helper",
    )
    # new parameter on _determine_batch_execution_and_padding
    edit(
        RUNNER,
        "        num_scheduled_tokens_np: np.ndarray,\n"
        "        max_num_scheduled_tokens: int,\n"
        "        use_cascade_attn: bool,\n",
        "        num_scheduled_tokens_np: np.ndarray,\n"
        "        max_num_scheduled_tokens: int,\n"
        "        use_cascade_attn: bool,\n"
        "        disable_full_long_ctx: bool = False,\n",
        "runner: add disable_full_long_ctx param",
    )
    # feed it into dispatch
    edit(
        RUNNER,
        "        cudagraph_mode, batch_descriptor = dispatch_cudagraph(\n"
        "            num_tokens_padded, disable_full=use_cascade_attn or has_encoder_output\n"
        "        )",
        "        cudagraph_mode, batch_descriptor = dispatch_cudagraph(\n"
        "            num_tokens_padded,\n"
        "            disable_full=use_cascade_attn\n"
        "            or has_encoder_output\n"
        "            or disable_full_long_ctx,\n"
        "        )",
        "runner: gate dispatch",
    )
    # runner execute-path call site passes the flag (anchor on the cascade
    # kwarg unique to this site; the dummy-run site uses use_cascade_attn=False)
    edit(
        RUNNER,
        "                use_cascade_attn=cascade_attn_prefix_lens is not None,\n"
        "                num_encoder_reqs=len(scheduler_output.scheduled_encoder_inputs),\n",
        "                use_cascade_attn=cascade_attn_prefix_lens is not None,\n"
        "                disable_full_long_ctx=glm52_disable_full_for_long_ctx(\n"
        "                    scheduler_output\n"
        "                ),\n"
        "                num_encoder_reqs=len(scheduler_output.scheduled_encoder_inputs),\n",
        "runner: execute-path flag",
    )
    # worker recv-shape call site passes the same flag
    edit(
        WORKER,
        "                self.model_runner._determine_batch_execution_and_padding(\n"
        "                    num_tokens=num_scheduled_tokens,\n",
        "                self.model_runner._determine_batch_execution_and_padding(\n"
        "                    disable_full_long_ctx=(\n"
        "                        __import__(\n"
        "                            \"vllm.v1.worker.gpu_model_runner\",\n"
        "                            fromlist=[\"glm52_disable_full_for_long_ctx\"],\n"
        "                        ).glm52_disable_full_for_long_ctx(scheduler_output)\n"
        "                    ),\n"
        "                    num_tokens=num_scheduled_tokens,\n",
        "worker: recv-shape flag",
    )


if __name__ == "__main__":
    main()
