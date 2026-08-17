#!/usr/bin/env python3
"""Declare the DSA indexer's decode-first reorder requirement (BUG FIX).

`DeepseekV32IndexerMetadataBuilder` splits its batch with
`split_decodes_and_prefills`, whose documented precondition is a batch
reordered decode -> prefill. But the builder sets
`reorder_batch_threshold = None` (and no other builder for this model sets
one), so `GPUModelRunner._may_reorder_batch` never reorders, and the layout
is arrival-order luck. Whenever a prefill request sits ahead of a spec
decode in the persistent batch, the split classifies the decode's next_n
tokens into the PREFILL bucket; they are then processed by the indexer
prefill path -- causally valid but numerically different from the decode
path -- which perturbs the top-2048 selection tail and shifts verify logits
by a few points, flipping near-tie tokens at temperature 0.

Observed as sporadic long-context verbatim-copy corruption in mixed
decode+prefill batches (NOTES_longctx_copy_fidelity.md): margin on the
'_mar'/'_bind' near-tie collapses from ~+9 logits (decode-first) to
-1..+6 (prefill-first), flipping ~25% of generations at 120K. MTP-off and
serialized runs were clean only because their timing kept decodes in front.

Fix: set reorder_batch_threshold = decode_threshold so the runner restores
the ordering the split assumes. The reorder is a cheap CPU swap permutation
per step and the documented contract of split_decodes_and_prefills.
"""

import os
import sys

import vllm

VLLM = os.path.dirname(vllm.__file__)
IDX = "v1/attention/backends/mla/indexer.py"


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


edit(
    IDX,
    "        next_n = self.num_speculative_tokens + 1\n"
    "        self.decode_threshold = next_n\n"
    "        self.reorder_batch_threshold = None\n",
    "        next_n = self.num_speculative_tokens + 1\n"
    "        self.decode_threshold = next_n\n"
    "        # split_decodes_and_prefills (used in _get_decode_metadata) is\n"
    "        # documented to ASSUME a decode-first reordered batch. Without\n"
    "        # declaring a threshold the runner never reorders, and a decode\n"
    "        # request behind a prefill lands in the PREFILL bucket -- valid\n"
    "        # causally but numerically different (prefill-path indexer topk),\n"
    "        # which flips near-tie tokens at temp 0 in mixed batches. See\n"
    "        # patch_reorder_decodes.py / NOTES_longctx_copy_fidelity.md.\n"
    "        self.reorder_batch_threshold = self.decode_threshold\n",
    "indexer: declare decode-first reorder requirement",
)

print("done")
