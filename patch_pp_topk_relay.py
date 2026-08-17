#!/usr/bin/env python3
"""ROOT-CAUSE FIX: relay DSA topk selections across PP boundaries.

CRITICAL NAMING CONSTRAINT (learned the hard way, 08-16 evening): the
buffer reference MUST be stored under a PRIVATE name. Upstream's drafter
load path (llm_base_proposer.py:1579) attr-sniffs
`target_language_model.model.topk_indices_buffer` and, if present,
REBINDS the drafter's private buffer to the target's ("Sharing target
model topk_indices_buffer..." in the log) — entangling the drafter's
step0-write/steps1-2-read cycle with the target's rank-boundary shared
layers and collapsing draft acceptance 4.0 -> 1.5 on long-context copy
workloads while leaving outputs bit-correct (verify eats the rejected
drafts). Also ship a CLONE, never a live buffer view (async-send lap).

NOTE: after application, an env gate GLM52_PP_TOPK_RELAY (default 1) was
added in-tree for perf attribution; =0 disables send+seed (bug returns).

GLM-5.2 (and any DSA model with index_topk_freq>1): 'shared' indexer
layers never run their own indexer (mla.py skip_topk) — their sparse
attention consumes topk_indices_buffer as-is, which is only correct when
the sharing 'full' layer ran earlier IN THE SAME BATCH ON THE SAME RANK.
Under PP, every rank whose stage begins mid-sharing-group (partition
11,10,10,10,10,10,10,7 with full layers at 2+4k -> ranks 1..7 all start
with 1-3 shared layers) reads its per-rank buffer holding THE PREVIOUS
BATCH'S selections: the same request's prior chunk when quiet (benign),
another request's decode selections / drafter leftovers under co-flight
+ MTP (build-poisoning corruption — the session-5 root cause; cachemap
proof: quiet-rebuild divergence begins exactly at L12, the first KV
computed downstream of stale-buffer attention).

Fix: the rank's outgoing IntermediateTensors carry the current
topk_indices_buffer rows (the last full indexer's selections — exactly
the sharing group leader for the next rank's leading shared layers); the
receiving rank seeds its buffer from them before running its layers.
Cost: one [tokens, 2048] int32 tensor per PP hop (~16MB per 2048-token
chunk microbatch over PCIe ~ +10ms/hop; ~100KB at decode) — a few
percent of prefill, negligible at decode. Also cures the rank-7 drafter
staleness (each incoming batch re-seeds the buffer).
"""

import os
import sys

import vllm

VLLM = os.path.dirname(vllm.__file__)
DSV2 = "model_executor/models/deepseek_v2.py"


def edit(path, old, new, label):
    full = os.path.join(VLLM, path)
    src = open(full).read()
    if new in src:
        print(f"  = {label} (already applied)")
        return
    if src.count(old) != 1:
        print(f"  ! {label}: anchor count {src.count(old)} != 1", file=sys.stderr)
        sys.exit(1)
    open(full, "w").write(src.replace(old, new))
    print(f"  + {label}")


# 1. Keep a model-level reference to the shared buffer.
edit(
    DSV2,
    "        else:\n"
    "            topk_indices_buffer = None\n\n"
    "        if get_pp_group().is_first_rank:\n",
    "        else:\n"
    "            topk_indices_buffer = None\n"
    "        self._glm52_pp_topk_relay_buf = topk_indices_buffer\n\n"
    "        if get_pp_group().is_first_rank:\n",
    "model-level buffer reference",
)

# 2. Seed the buffer from the received intermediate tensors (non-first
#    ranks) before any layer runs.
edit(
    DSV2,
    "            assert intermediate_tensors is not None\n"
    "            hidden_states = intermediate_tensors[\"hidden_states\"]\n"
    "            residual = intermediate_tensors[\"residual\"]\n",
    "            assert intermediate_tensors is not None\n"
    "            hidden_states = intermediate_tensors[\"hidden_states\"]\n"
    "            residual = intermediate_tensors[\"residual\"]\n"
    "            # PP topk relay (patch_pp_topk_relay.py): seed this rank's\n"
    "            # per-rank buffer with the previous rank's last full-indexer\n"
    "            # selections so leading skip_topk (shared) layers attend the\n"
    "            # selections they are defined to share, not stale leftovers.\n"
    "            _tki = (\n"
    "                intermediate_tensors.tensors.get(\"topk_indices\")\n"
    "                if hasattr(intermediate_tensors, \"tensors\")\n"
    "                else None\n"
    "            )\n"
    "            if _tki is not None and self._glm52_pp_topk_relay_buf is not None:\n"
    "                _n = min(_tki.shape[0], self._glm52_pp_topk_relay_buf.shape[0])\n"
    "                self._glm52_pp_topk_relay_buf[:_n].copy_(_tki[:_n])\n",
    "seed buffer on receive",
)

# 3. Ship the buffer rows with the outgoing intermediate tensors.
edit(
    DSV2,
    "        if not get_pp_group().is_last_rank:\n"
    "            return IntermediateTensors(\n"
    "                {\"hidden_states\": hidden_states, \"residual\": residual}\n"
    "            )\n",
    "        if not get_pp_group().is_last_rank:\n"
    "            _out = {\"hidden_states\": hidden_states, \"residual\": residual}\n"
    "            if self._glm52_pp_topk_relay_buf is not None:\n"
    "                _nt = positions.shape[0]\n"
    "                # CLONE, never a live view: async PP send + mutable\n"
    "                # per-rank buffer = the patch-15 lap-race class; the\n"
    "                # view variant halved long-ctx draft acceptance.\n"
    "                _out[\"topk_indices\"] = self._glm52_pp_topk_relay_buf[:_nt].clone()\n"
    "            return IntermediateTensors(_out)\n",
    "ship buffer rows on send",
)

# 4. Extend the empty-intermediate-tensors factory (dummy/profile runs and
#    the runner's persistent PP staging allocate from this).
edit(
    DSV2,
    "        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(\n"
    "            [\"hidden_states\", \"residual\"], self.hidden_size\n"
    "        )\n",
    "        if self.is_v32:\n"
    "            _hs = self.hidden_size\n"
    "            _tk = config.index_topk\n\n"
    "            def _make_empty_it(batch_size, dtype, device):\n"
    "                return IntermediateTensors(\n"
    "                    {\n"
    "                        \"hidden_states\": torch.zeros(\n"
    "                            (batch_size, _hs), dtype=dtype, device=device\n"
    "                        ),\n"
    "                        \"residual\": torch.zeros(\n"
    "                            (batch_size, _hs), dtype=dtype, device=device\n"
    "                        ),\n"
    "                        \"topk_indices\": torch.full(\n"
    "                            (batch_size, _tk), -1,\n"
    "                            dtype=torch.int32, device=device,\n"
    "                        ),\n"
    "                    }\n"
    "                )\n\n"
    "            self.make_empty_intermediate_tensors = _make_empty_it\n"
    "        else:\n"
    "            self.make_empty_intermediate_tensors = (\n"
    "                make_empty_intermediate_tensors_factory(\n"
    "                    [\"hidden_states\", \"residual\"], self.hidden_size\n"
    "                )\n"
    "            )\n",
    "factory carries topk_indices",
)

import ast

ast.parse(open(os.path.join(VLLM, DSV2)).read())
print("patch_pp_topk_relay: done (syntax ok)")
