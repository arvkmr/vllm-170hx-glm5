#!/usr/bin/env python3
"""Guard: DSA indexer opts out of FULL cudagraphs (silent-truncation bug).

FULL-graph capture sizes the decode-side indexer logits buffer at the
capture dummy's max_seq_len (as small as max_query_len for non-profiled
sizes), and the dispatcher replays graphs at any context length. With the
store-clamp patch this no longer corrupts memory, but contexts beyond the
captured width still get a silently TRUNCATED sparse top-k window -- an
insidious accuracy loss. Until capture uses realistic seq lens for every
size (or dispatch gains a seq-len guard), force the model to PIECEWISE by
declaring AttentionCGSupport.NEVER.

Escape hatch: GLM52_DSA_FULLCG=1 restores UNIFORM_BATCH (for testing;
correct only while every request stays below topk=2048 context, where the
decode top-k takes the identity shortcut and never reads the logits).
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


def main():
    edit(
        IDX,
        "    ) -> AttentionCGSupport:\n"
        "        return AttentionCGSupport.UNIFORM_BATCH\n",
        "    ) -> AttentionCGSupport:\n"
        "        import os as _os\n"
        "        if _os.environ.get(\"GLM52_DSA_FULLCG\", \"0\") == \"1\":\n"
        "            # Testing escape hatch; correct only below topk=2048\n"
        "            # context (identity shortcut) with the store clamp.\n"
        "            return AttentionCGSupport.UNIFORM_BATCH\n"
        "        # FULL-graph capture sizes the decode logits buffer at the\n"
        "        # capture dummy's max_seq_len; replays at longer contexts\n"
        "        # silently truncate the sparse top-k window. Force PIECEWISE.\n"
        "        return AttentionCGSupport.NEVER\n",
        "indexer: opt out of FULL cudagraphs",
    )


if __name__ == "__main__":
    main()
