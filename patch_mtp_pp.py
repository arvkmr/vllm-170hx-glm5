#!/usr/bin/env python3
"""Apply MTP + pipeline-parallel support to vLLM 0.26.0 (idempotent).

vLLM 0.26.0 cannot run MTP speculative decoding together with PP>1: both PP
token-propagation paths assume one sampled token per request per step. The
async path hard-asserts it ("PP+async expects sampled_token_ids to have shape
[num_reqs, 1]") and is not fixable without a wire-format change, so this patch
set targets the sync-scheduling path (the default here).

THE CORE DESIGN (patches 4-6): cursor-based token shipping.

The stock sync path ships `all_token_ids[num_computed : num_computed + n]` to
non-last PP ranks. Under speculation that formula is wrong twice over:

  * The token sampled at the END OF PREFILL sits at index == num_computed and
    the first decode step computes n == 0, so it is NEVER shipped. Non-last
    ranks embed whatever occupies that slot -- the draft token that
    update_req_spec_token_ids just wrote there. This alone poisons the whole
    generation from output token 2 onward (observed: first token always
    correct, garbage after -- "1ames...", "9opro...").
  * Draft tokens ACCEPTED in step k advance num_computed without being
    shipped, and on rejection num_computed rolls BACK -- num_computed counts
    KV positions, not verified tokens, so it is the wrong shipping cursor.

Instead the scheduler now tracks, per request, how many verified tokens it has
shipped (`_pp_spec_num_shipped`, starting at the prompt length), and each step
ships exactly `all_token_ids[shipped:]`. The worker appends the payload at its
`num_tokens_no_spec` cursor. This is self-synchronising:

  * the prefill-sampled token ships on the first decode step;
  * an accepted draft re-ships onto the slot already holding the same value
    (idempotent);
  * a REJECTED draft's replacement lands exactly on top of the stale draft
    value, because the cursor never advances past unverified positions.

Run:  .venv/bin/python patch_mtp_pp.py [--revert]

Originals live in site-packages/.glm52-backup/; --revert restores runner and
scheduler from there (deepseek_mtp.py keeps its SupportsPP/embedding patches
unless you also restore it manually -- they are inert without a spec config).

Validate with mtp_verify.py: greedy output with SPEC_TOKENS=1 must be
token-identical to SPEC_TOKENS=0. Never judge this by "it produces text".
"""

import argparse
import os
import sys

import vllm

VLLM = os.path.dirname(vllm.__file__)
BACKUP = os.path.join(os.path.dirname(VLLM), ".glm52-backup", "vllm")

RUNNER = "v1/worker/gpu_model_runner.py"
MTP = "model_executor/models/deepseek_mtp.py"
SCHED = "v1/core/sched/scheduler.py"
CORE = "v1/engine/core.py"


def edit(path, old, new, label, superseded_by=None):
    full = os.path.join(VLLM, path)
    src = open(full).read()
    if new in src:
        print(f"  = {label} (already applied)")
        return
    if superseded_by and superseded_by in src:
        print(f"  = {label} (superseded by a later patch)")
        return
    if old not in src:
        print(f"  ! {label}: ANCHOR NOT FOUND", file=sys.stderr)
        sys.exit(1)
    if src.count(old) != 1:
        print(f"  ! {label}: anchor not unique ({src.count(old)}x)", file=sys.stderr)
        sys.exit(1)
    open(full, "w").write(src.replace(old, new))
    print(f"  + {label}")


# --------------------------------------------------------------------------
# 1. DeepSeekMTP must declare SupportsPP.
#
# The draft model config inherits the target's pipeline_parallel_size, so
# ModelConfig.verify_with_parallel_config rejects it with "Pipeline parallelism
# is not supported for this model" before anything loads. The declaration is
# nominal, exactly as NemotronHMTP does it: the MTP module is never split across
# stages -- gpu_model_runner builds the drafter only on the last PP rank.
# --------------------------------------------------------------------------
def patch_supports_pp():
    edit(
        MTP,
        "from .utils import (\n    get_pp_missing_layer_names,\n"
        "    get_spec_layer_idx_from_weight_name,\n    maybe_prefix,\n)",
        "from .interfaces import SupportsPP\nfrom .utils import (\n"
        "    get_pp_missing_layer_names,\n"
        "    get_spec_layer_idx_from_weight_name,\n"
        "    make_empty_intermediate_tensors_factory,\n    maybe_prefix,\n)",
        "deepseek_mtp: import SupportsPP",
    )
    edit(
        MTP,
        "class DeepSeekMTP(nn.Module, DeepseekV2MixtureOfExperts):",
        "class DeepSeekMTP(nn.Module, DeepseekV2MixtureOfExperts, SupportsPP):",
        "deepseek_mtp: declare SupportsPP",
    )
    edit(
        MTP,
        "        self.model = DeepSeekMultiTokenPredictor(\n"
        "            vllm_config=vllm_config, prefix=maybe_prefix(prefix, \"model\")\n"
        "        )\n"
        "        # Set MoE hyperparameters",
        "        self.model = DeepSeekMultiTokenPredictor(\n"
        "            vllm_config=vllm_config, prefix=maybe_prefix(prefix, \"model\")\n"
        "        )\n"
        "        # SupportsPP protocol requirement. The drafter lives wholly on the\n"
        "        # last PP rank and never joins the pipeline, so this is not expected\n"
        "        # to be called, but the interface must be satisfied.\n"
        "        self.make_empty_intermediate_tensors = (\n"
        "            make_empty_intermediate_tensors_factory(\n"
        "                [\"hidden_states\", \"residual\"], self.config.hidden_size\n"
        "            )\n"
        "        )\n"
        "        # Set MoE hyperparameters",
        "deepseek_mtp: make_empty_intermediate_tensors",
    )


# --------------------------------------------------------------------------
# 2. The draft model needs its own embed_tokens under PP.
#
# `_maybe_share_embeddings` only shares the target's embedding when the pp world
# size is 1; with PP>1 the target's embed_tokens lives on rank 0 while the
# drafter lives on the last rank ("The draft model's vocab embedding will be
# loaded separately from the target model"). But this loader drops every weight
# that is not part of an MTP layer, so the checkpoint's top-level embedding
# never arrives and the draft runs on uninitialised memory.
# --------------------------------------------------------------------------
def patch_draft_embeddings():
    edit(
        MTP,
        "from vllm.distributed import tensor_model_parallel_all_gather",
        "from vllm.distributed import tensor_model_parallel_all_gather\n"
        "from vllm.distributed.parallel_state import get_pp_group",
        "deepseek_mtp: import get_pp_group",
    )
    edit(
        MTP,
        "            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)\n"
        "            if spec_layer is None:\n"
        "                continue",
        "            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)\n"
        "            if spec_layer is None:\n"
        "                # Under PP the proposer cannot share the target's embedding\n"
        "                # (it lives on rank 0, the drafter on the last rank), so the\n"
        "                # draft must load its own -- but it is not a spec-layer\n"
        "                # weight, so it would otherwise be dropped here.\n"
        "                if (\n"
        "                    get_pp_group().world_size > 1\n"
        "                    and name.endswith(\"model.embed_tokens.weight\")\n"
        "                    and \"model.embed_tokens.weight\" in params_dict\n"
        "                ):\n"
        "                    param = params_dict[\"model.embed_tokens.weight\"]\n"
        "                    weight_loader = getattr(\n"
        "                        param, \"weight_loader\", default_weight_loader\n"
        "                    )\n"
        "                    weight_loader(param, loaded_weight)\n"
        "                    loaded_params.add(\"model.embed_tokens.weight\")\n"
        "                continue",
        "deepseek_mtp: load draft embed_tokens under PP",
    )


# --------------------------------------------------------------------------
# 3. Guard drafter dereferences on non-last PP ranks.
#
# gpu_model_runner creates `self.drafter` only on the last PP rank ("we put the
# entire draft model on the last PP rank"), but several init paths dereference
# it on every rank, gated only on `speculative_config` -> AttributeError on
# ranks 0..N-2. `load_model` already guards with hasattr; use the same idiom.
# The runtime drafting paths need nothing: execute_model returns early on
# non-last ranks before reaching them.
# --------------------------------------------------------------------------
def patch_drafter_guards():
    edit(
        RUNNER,
        "            if self.speculative_config and spec_decode_common_attn_metadata is None:\n"
        "                if isinstance(\n                    self.drafter,",
        "            if (\n                self.speculative_config\n"
        "                and hasattr(self, \"drafter\")  # last PP rank only\n"
        "                and spec_decode_common_attn_metadata is None\n            ):\n"
        "                if isinstance(\n                    self.drafter,",
        "runner: guard attn-metadata drafter deref",
    )
    edit(
        RUNNER,
        "            if self.speculative_config and isinstance(self.drafter, Step3p5MTPProposer):",
        "            if not hasattr(self, \"drafter\"):  # non-last PP rank\n"
        "                pass\n"
        "            elif self.speculative_config and isinstance(\n"
        "                self.drafter, Step3p5MTPProposer\n            ):",
        "runner: guard per-group proposer hooks",
    )
    edit(
        RUNNER,
        "            if self.speculative_config and (\n"
        "                self.speculative_config.use_eagle()\n"
        "                or self.speculative_config.uses_draft_model()\n"
        "                or self.speculative_config.uses_extract_hidden_states()\n"
        "            ):\n                assert isinstance(\n                    self.drafter,",
        "            if (\n                self.speculative_config\n"
        "                and hasattr(self, \"drafter\")  # last PP rank only\n"
        "                and (\n                    self.speculative_config.use_eagle()\n"
        "                    or self.speculative_config.uses_draft_model()\n"
        "                    or self.speculative_config.uses_extract_hidden_states()\n"
        "                )\n            ):\n"
        "                assert isinstance(\n                    self.drafter,",
        "runner: guard _dummy_run drafter run",
    )
    edit(
        RUNNER,
        "        if self.speculative_config and (\n"
        "            self.speculative_config.use_eagle()\n"
        "            or self.speculative_config.uses_draft_model()\n"
        "        ):\n            assert isinstance(\n                self.drafter,",
        "        # Last PP rank only. get_kv_cache_configs() already merges per-worker\n"
        "        # specs and takes min(num_blocks), so the drafter's extra KV layer on\n"
        "        # that rank is accounted for globally.\n"
        "        if (\n            self.speculative_config\n"
        "            and hasattr(self, \"drafter\")\n"
        "            and (\n                self.speculative_config.use_eagle()\n"
        "                or self.speculative_config.uses_draft_model()\n"
        "            )\n        ):\n"
        "            assert isinstance(\n                self.drafter,",
        "runner: guard drafter attn-backend init",
    )
    edit(
        RUNNER,
        "        if self.speculative_config and (\n"
        "            self.speculative_config.use_eagle()\n"
        "            or self.speculative_config.uses_draft_model()\n"
        "            or self.speculative_config.uses_extract_hidden_states()\n"
        "        ):\n            assert isinstance(\n                self.drafter,",
        "        if (\n            self.speculative_config\n"
        "            and hasattr(self, \"drafter\")  # last PP rank only\n"
        "            and (\n                self.speculative_config.use_eagle()\n"
        "                or self.speculative_config.uses_draft_model()\n"
        "                or self.speculative_config.uses_extract_hidden_states()\n"
        "            )\n        ):\n"
        "            assert isinstance(\n                self.drafter,",
        "runner: guard drafter cudagraph init",
    )


# --------------------------------------------------------------------------
# 4. Scheduler: cursor-based shipping (see module docstring).
# --------------------------------------------------------------------------
def patch_scheduler_cursor_shipping():
    edit(
        SCHED,
        "                num_tokens = num_scheduled_tokens[req_id] - len(\n"
        "                    spec_decode_tokens.get(req_id, ())\n"
        "                )\n"
        "                token_ids = req.all_token_ids[\n"
        "                    req.num_computed_tokens : req.num_computed_tokens + num_tokens\n"
        "                ]\n"
        "                new_token_ids.append(token_ids)",
        "                if self.vllm_config.speculative_config is not None:\n"
        "                    # Spec decode: ship every verified token the workers\n"
        "                    # have not been sent yet (cursor-based). Slicing from\n"
        "                    # num_computed_tokens is wrong under speculation: the\n"
        "                    # prefill-sampled token sits at an index the formula\n"
        "                    # never covers, accepted drafts advance num_computed\n"
        "                    # without being shipped, and rejection rolls it back.\n"
        "                    total_verified = len(req.all_token_ids)\n"
        "                    if idx >= num_running_reqs:\n"
        "                        # Resumed from preemption: ship nothing; instead\n"
        "                        # resync the full token list so the worker's\n"
        "                        # add_request() rebuilds its buffers exactly.\n"
        "                        shipped = total_verified\n"
        "                        all_token_ids[req_id] = req.all_token_ids.copy()\n"
        "                    else:\n"
        "                        shipped = getattr(req, \"_pp_spec_num_shipped\", None)\n"
        "                        if shipped is None:\n"
        "                            # First step as a cached request: the worker\n"
        "                            # has only the prompt (via NewRequestData).\n"
        "                            shipped = req.num_prompt_tokens\n"
        "                    token_ids = req.all_token_ids[shipped:total_verified]\n"
        "                    req._pp_spec_num_shipped = total_verified\n"
        "                else:\n"
        "                    num_tokens = num_scheduled_tokens[req_id] - len(\n"
        "                        spec_decode_tokens.get(req_id, ())\n"
        "                    )\n"
        "                    token_ids = req.all_token_ids[\n"
        "                        req.num_computed_tokens : req.num_computed_tokens\n"
        "                        + num_tokens\n"
        "                    ]\n"
        "                new_token_ids.append(token_ids)",
        "scheduler: cursor-based shipping under spec decode",
    )


# --------------------------------------------------------------------------
# 5. Worker: consume the cursor-based payload.
#
# Both the output_token_ids bookkeeping and the token_ids_cpu write anchor on
# the num_tokens_no_spec cursor instead of num_computed_tokens. The buffer
# write keeps upstream's exact-slice form; with num_new == len(payload) there
# is no silent numpy broadcast (the stock code assigns a 1-element list to a
# 2-slot slice after an accepted draft, replicating one token into both).
# --------------------------------------------------------------------------
def patch_worker_cursor_consume():
    edit(
        RUNNER,
        "                    new_token_ids = req_data.new_token_ids[i]\n"
        "                    # Add the sampled token(s) from the previous step (if any).\n"
        "                    # This doesn't include \"unverified\" tokens like spec tokens.\n"
        "                    num_new_tokens = (\n"
        "                        num_computed_tokens + len(new_token_ids) - req_state.num_tokens\n"
        "                    )\n"
        "                    if num_new_tokens == 1:\n"
        "                        # Avoid slicing list in most common case.\n"
        "                        req_state.output_token_ids.append(new_token_ids[-1])\n"
        "                    elif num_new_tokens > 0:\n"
        "                        req_state.output_token_ids.extend(\n"
        "                            new_token_ids[-num_new_tokens:]\n"
        "                        )",
        "                    new_token_ids = req_data.new_token_ids[i]\n"
        "                    if self.num_spec_tokens:\n"
        "                        # PP + spec decode: the scheduler ships exactly the\n"
        "                        # verified tokens this worker has not seen, in\n"
        "                        # order (cursor-based; see\n"
        "                        # _make_cached_request_data). num_computed-based\n"
        "                        # arithmetic is wrong here: it counts KV positions,\n"
        "                        # which roll back on rejection and run ahead of\n"
        "                        # shipping on acceptance.\n"
        "                        req_state.output_token_ids.extend(new_token_ids)\n"
        "                    else:\n"
        "                        # Add the sampled token(s) from the previous step\n"
        "                        # (if any). This doesn't include \"unverified\"\n"
        "                        # tokens like spec tokens.\n"
        "                        num_new_tokens = (\n"
        "                            num_computed_tokens\n"
        "                            + len(new_token_ids)\n"
        "                            - req_state.num_tokens\n"
        "                        )\n"
        "                        if num_new_tokens == 1:\n"
        "                            # Avoid slicing list in most common case.\n"
        "                            req_state.output_token_ids.append(new_token_ids[-1])\n"
        "                        elif num_new_tokens > 0:\n"
        "                            req_state.output_token_ids.extend(\n"
        "                                new_token_ids[-num_new_tokens:]\n"
        "                            )",
        "runner: cursor-based output_token_ids bookkeeping",
    )
    edit(
        RUNNER,
        "                end_token_index = max(\n"
        "                    start_token_index,\n"
        "                    num_computed_tokens + len(new_token_ids),\n"
        "                )",
        "                if self.num_spec_tokens:\n"
        "                    # Cursor-based shipping (PP + spec decode): append the\n"
        "                    # payload at the no-spec cursor. A rejected draft is\n"
        "                    # corrected naturally -- the cursor never advances past\n"
        "                    # unverified positions, so the replacement token lands\n"
        "                    # on top of the stale draft value.\n"
        "                    end_token_index = start_token_index + len(new_token_ids)\n"
        "                else:\n"
        "                    end_token_index = max(\n"
        "                        start_token_index,\n"
        "                        num_computed_tokens + len(new_token_ids),\n"
        "                    )",
        "runner: cursor-based token_ids_cpu write window",
    )


# --------------------------------------------------------------------------
# 6. Worker: recover output tokens for resumed requests under PP+spec.
#
# The cursor scheme ships nothing on resume (patch 4 resyncs the full list via
# `all_token_ids` instead), so extend the async-scheduling recovery branch to
# also fire for spec decode. Guarded on dict membership so a missing resync
# entry degrades to keeping the existing cached state.
# --------------------------------------------------------------------------
def patch_worker_resume_recovery():
    edit(
        RUNNER,
        "                if self.use_async_scheduling and num_output_tokens > 0:\n"
        "                    # We must recover the output token ids for resumed requests in the\n"
        "                    # async scheduling case, so that correct input_ids are obtained.\n"
        "                    resumed_token_ids = req_data.all_token_ids[req_id]\n"
        "                    req_state.output_token_ids = resumed_token_ids[-num_output_tokens:]",
        "                if (\n"
        "                    (self.use_async_scheduling or self.num_spec_tokens)\n"
        "                    and num_output_tokens > 0\n"
        "                    and req_id in req_data.all_token_ids\n"
        "                ):\n"
        "                    # We must recover the output token ids for resumed\n"
        "                    # requests (async scheduling; PP+spec cursor shipping,\n"
        "                    # which resyncs instead of re-shipping on resume), so\n"
        "                    # that correct input_ids are obtained via add_request().\n"
        "                    resumed_token_ids = req_data.all_token_ids[req_id]\n"
        "                    req_state.output_token_ids = resumed_token_ids[-num_output_tokens:]",
        "runner: resume recovery under PP+spec",
    )


# --------------------------------------------------------------------------
# 7. Serialize per-request steps under PP + sync + spec decode.
#
# Root cause found by probing actual model inputs (GLM52_DEBUG_INPUTS): with
# PP, the engine core queues multiple batches to keep pipeline stages busy.
# Without spec decode a request can never be scheduled while its previous step
# is in flight -- num_tokens_with_spec == num_computed_tokens makes
# num_new_tokens 0. Leftover spec_token_ids break that invariant: the request
# gets rescheduled on STALE drafts before its output is processed, and the
# freshly sampled token exists only on the last rank's GPU at that moment. The
# probe showed position 17 (the prefill-sampled token) computed exactly once,
# with the DRAFT embedded on rank 0 and the true token on the last rank, and
# never recomputed -- permanently divergent KV between ranks. This is exactly
# why the PP+async path uses a last-rank GPU broadcast instead; sync has no
# equivalent mechanism.
#
# Fix: do not schedule a running request under PP+sync+spec until the output
# of its previous step has been processed. This restores the vanilla
# serialization invariant; concurrency across DIFFERENT requests (what fills
# the pipeline under load) is untouched. With serialization, all_token_ids is
# current at scheduling time, so cursor-based shipping (patch 4) delivers the
# leading-position token before the step runs and drafts are always fresh.
# --------------------------------------------------------------------------
def patch_serialize_inflight():
    edit(
        SCHED,
        "        while req_index < len(self.running) and token_budget > 0:\n"
        "            request = self.running[req_index]\n"
        "\n"
        "            if (\n"
        "                request.num_output_placeholders > 0",
        "        while req_index < len(self.running) and token_budget > 0:\n"
        "            request = self.running[req_index]\n"
        "\n"
        "            if (\n"
        "                self.use_pp\n"
        "                and not self.scheduler_config.async_scheduling\n"
        "                and self.vllm_config.speculative_config is not None\n"
        "                and getattr(request, \"_pp_spec_in_flight\", False)\n"
        "            ):\n"
        "                # PP + sync + spec decode: this request's previous step has\n"
        "                # not been processed yet. Leftover spec_token_ids would make\n"
        "                # num_new_tokens > 0 and reschedule it on stale drafts,\n"
        "                # while the newly sampled token exists only on the last\n"
        "                # rank's GPU -- non-last ranks would embed the draft and\n"
        "                # their KV would diverge permanently from the last rank's.\n"
        "                # Mirrors the num_new_tokens == 0 serialization that\n"
        "                # non-spec decoding gets for free.\n"
        "                req_index += 1\n"
        "                continue\n"
        "\n"
        "            if (\n"
        "                request.num_output_placeholders > 0",
        "scheduler: skip in-flight requests under PP+sync+spec",
        superseded_by="sampling step only",
    )
    edit(
        SCHED,
        "        # Record the request ids that were scheduled in this step (MRV1-only).\n"
        "        if not self.use_v2_model_runner:",
        "        # PP + sync + spec decode: mark scheduled requests in flight; cleared\n"
        "        # in update_from_output once their output has been processed.\n"
        "        if (\n"
        "            self.use_pp\n"
        "            and not self.scheduler_config.async_scheduling\n"
        "            and self.vllm_config.speculative_config is not None\n"
        "        ):\n"
        "            for _rid in num_scheduled_tokens:\n"
        "                _r = self.requests.get(_rid)\n"
        "                if _r is not None:\n"
        "                    _r._pp_spec_in_flight = True\n"
        "\n"
        "        # Record the request ids that were scheduled in this step (MRV1-only).\n"
        "        if not self.use_v2_model_runner:",
        "scheduler: mark scheduled requests in flight",
        superseded_by="COUNTER, not boolean",
    )
    edit(
        SCHED,
        "            req_index = model_runner_output.req_id_to_index[req_id]\n"
        "            generated_token_ids = (\n"
        "                sampled_token_ids[req_index] if sampled_token_ids else []\n"
        "            )",
        "            req_index = model_runner_output.req_id_to_index[req_id]\n"
        "            # PP + sync + spec decode: this request's step output is now\n"
        "            # being processed; it may be scheduled again.\n"
        "            request._pp_spec_in_flight = False\n"
        "            generated_token_ids = (\n"
        "                sampled_token_ids[req_index] if sampled_token_ids else []\n"
        "            )",
        "scheduler: clear in-flight flag on output processing",
        superseded_by="counter reaches zero",
    )


# --------------------------------------------------------------------------
# 8. Global (not just per-request) serialization under PP + sync + spec.
#
# Per-request serialization (patch 7) still allows CONSECUTIVE batches with
# disjoint request sets to overlap in the pipeline. py-spy of a live 4-request
# hang showed all 8 workers idle in worker_busy_loop.dequeue while the engine
# core sat in post_step -> take_draft_token_ids -> collective_rpc ->
# get_response: issuing the drafter RPC while a second batch was in flight
# desyncs the executor's response stream, and the reply never routes back.
# (The NCCL "collective timeout" seen earlier was a downstream symptom of the
# stalled dispatch loop, not the cause.)
#
# Schedule NOTHING while any spec batch is in flight. Concurrent requests
# still batch together within a step (spec verification is batched across
# requests); what is lost is cross-batch pipelining while MTP is on. For
# throughput-oriented serving run SPEC_TOKENS=0.
# --------------------------------------------------------------------------
def patch_global_serialization():
    edit(
        SCHED,
        "        token_budget = self.max_num_scheduled_tokens\n"
        "        if self._pause_state == PauseState.PAUSED_ALL:\n"
        "            # Do not schedule any requests when paused.\n"
        "            token_budget = 0",
        "        token_budget = self.max_num_scheduled_tokens\n"
        "        if self._pause_state == PauseState.PAUSED_ALL:\n"
        "            # Do not schedule any requests when paused.\n"
        "            token_budget = 0\n"
        "        elif (\n"
        "            self.use_pp\n"
        "            and not self.scheduler_config.async_scheduling\n"
        "            and self.vllm_config.speculative_config is not None\n"
        "            and any(\n"
        "                getattr(r, \"_pp_spec_in_flight\", False) for r in self.running\n"
        "            )\n"
        "        ):\n"
        "            # PP + sync + spec decode: never dispatch a new batch (for ANY\n"
        "            # request, running or waiting) while one is in flight. The\n"
        "            # drafter RPC (take_draft_token_ids) issued between two\n"
        "            # in-flight batches desyncs the executor response stream and\n"
        "            # hangs the engine; see patch notes.\n"
        "            token_budget = 0",
        "scheduler: global batch serialization under PP+sync+spec",
        # Patch 12 rewrites this block into the env-gated form. Without this
        # marker a later patch run would RE-INSERT the ungated block (its
        # stock anchor -- the paused-state prefix -- survives patch 12),
        # silently restoring global serialization. That exact accident cost a
        # debugging session: budget=0 wavefront prefill despite patch 13.
        superseded_by="_PP_SPEC_GLOBAL_SER",
    )


# --------------------------------------------------------------------------
# 9. Do not issue the drafter RPC while a batch is being dispatched.
#
# step_with_batch_queue's dispatch path returns model_executed=True right
# after a NON-BLOCKING execute_model dispatch, and _process_engine_step then
# calls post_step -> take_draft_token_ids: a synchronous collective RPC racing
# the dispatch for the workers' message ring. Depending on timing this wedges
# the engine (observed twice: engine core stuck in get_response, all workers
# idle in dequeue; the 600 s NCCL watchdog abort was a downstream symptom).
#
# Only take drafts when the batch queue is drained. With global serialization
# (patch 8) every completion iteration drains the queue, so drafts are always
# collected immediately after the step that produced them -- the dispatch-time
# call was redundant. self.batch_queue is None on non-PP setups, which keeps
# stock behavior there.
# --------------------------------------------------------------------------
def patch_draft_rpc_gate():
    # NOTE: model_executed means "this iteration DISPATCHED a batch", not "a
    # batch completed" -- stock code therefore collected drafts only at
    # dispatch time, interleaved with the in-flight execute (the wedge), and
    # NEVER on completion iterations. Under the batch queue, collect drafts
    # exactly when the queue is drained (completion/idle iterations, workers
    # quiescent); keep stock behavior when there is no batch queue (non-PP).
    edit(
        CORE,
        "        if self.check_for_draft_tokens and not self.async_scheduling and model_executed:\n"
        "            draft_token_ids = self.model_executor.take_draft_token_ids()",
        "        take_drafts = model_executed\n"
        "        if (\n"
        "            self.check_for_draft_tokens\n"
        "            and not self.async_scheduling\n"
        "            and self.batch_queue is not None\n"
        "        ):\n"
        "            # PP batch queue: dispatch-time collection (model_executed\n"
        "            # True) races the in-flight execute for the worker message\n"
        "            # ring; completion iterations report model_executed False.\n"
        "            # Collect exactly when the queue is drained instead.\n"
        "            take_drafts = not self.batch_queue\n"
        "        if self.check_for_draft_tokens and not self.async_scheduling and take_drafts:\n"
        "            draft_token_ids = self.model_executor.take_draft_token_ids()",
        "core: collect drafts on drained queue, not at dispatch",
        superseded_by="the drafter RPC is fully retired",
    )


# --------------------------------------------------------------------------
# 10. Never dispatch empty batches under spec decode (batch-queue mode).
#
# The global serialization gate (patch 8) makes schedule() return empty while
# a batch is in flight, and step_with_batch_queue dispatches those empties as
# real execute_model RPCs. They pollute the batch queue -- which turned patch
# 9's drained-queue gate into "never", silently disabling speculation (9.5
# tok/s, spec counters frozen) -- and they interleave extra RPC/response
# traffic that the wedges correlate with. With DP=1 an empty dispatch does no
# useful work. Skipping them yields strict alternation: dispatch real batch ->
# complete -> collect drafts (queue drained) -> dispatch next. Every RPC
# sequence becomes identical to the single-request steady state that has been
# stable throughout.
# --------------------------------------------------------------------------
def patch_no_empty_dispatch():
    edit(
        CORE,
        "        model_executed = False\n"
        "        deferred_scheduler_output = None\n"
        "        if self.scheduler.has_requests():\n"
        "            scheduler_output = self.scheduler.schedule(self._should_throttle_prefills())\n"
        "            with self.log_error_detail(scheduler_output):\n"
        "                exec_future = self.model_executor.execute_model(\n"
        "                    scheduler_output, non_block=True\n"
        "                )",
        "        model_executed = False\n"
        "        deferred_scheduler_output = None\n"
        "        dispatch = True\n"
        "        if self.scheduler.has_requests():\n"
        "            scheduler_output = self.scheduler.schedule(self._should_throttle_prefills())\n"
        "            if (\n"
        "                self.vllm_config.speculative_config is not None\n"
        "                and scheduler_output.total_num_scheduled_tokens == 0\n"
        "            ):\n"
        "                # Spec decode + batch queue: do not dispatch empty batches\n"
        "                # (see patch_mtp_pp.py). If nothing is in flight either,\n"
        "                # this is an idle tick.\n"
        "                dispatch = False\n"
        "                if not batch_queue:\n"
        "                    return None, False\n"
        "            else:\n"
        "                with self.log_error_detail(scheduler_output):\n"
        "                    exec_future = self.model_executor.execute_model(\n"
        "                        scheduler_output, non_block=True\n"
        "                    )",
        "core: skip empty-batch dispatch under spec",
    )
    edit(
        CORE,
        "            if self.is_ec_consumer:\n"
        "                model_executed = scheduler_output.total_num_scheduled_tokens > 0",
        "            if dispatch and self.is_ec_consumer:\n"
        "                model_executed = scheduler_output.total_num_scheduled_tokens > 0",
        "core: guard ec_consumer flag on dispatch",
    )
    edit(
        CORE,
        "            if self.is_pooling_model or not model_executed:",
        "            if not dispatch:\n"
        "                # Empty batch was not dispatched; fall through to the\n"
        "                # blocking pop of the in-flight batch below.\n"
        "                pass\n"
        "            elif self.is_pooling_model or not model_executed:",
        "core: guard future construction on dispatch",
    )
    edit(
        CORE,
        "            if not deferred_scheduler_output:\n"
        "                # Add this step's future to the queue.",
        "            if dispatch and not deferred_scheduler_output:\n"
        "                # Add this step's future to the queue.",
        "core: guard batch-queue append on dispatch",
    )


# --------------------------------------------------------------------------
# 11. Piggyback drafts on the model-output reply; retire the drafter RPC.
#
# take_draft_token_ids is a full engine->worker7->engine round trip executed
# while every GPU is idle -- it sits on the critical path of each serialized
# step (a large part of the ~22 ms/step MTP overhead) and it is the second
# reply stream whose interleaving with in-flight batches correlates with every
# engine wedge observed. Instead: rank 7 attaches freshly proposed drafts to
# the ModelRunnerOutput it was going to send anyway (pickle preserves extra
# attributes), and the engine reads them right after processing the output.
# Same information, zero extra round trips, one reply stream.
# --------------------------------------------------------------------------
def patch_piggyback_drafts():
    edit(
        "v1/executor/multiproc_executor.py",
        "        if isinstance(output, Exception):\n"
        "            result = (WorkerProc.ResponseStatus.FAILURE, str(output))",
        "        # PP + spec decode: attach freshly proposed draft tokens to the\n"
        "        # model output instead of serving a separate take_draft RPC.\n"
        "        # Duck-typed on req_ids (ModelRunnerOutput); take_draft returns\n"
        "        # None when there is nothing new (no drafter on this rank, no\n"
        "        # spec config, or drafts already taken), which keeps this a\n"
        "        # no-op everywhere it does not apply.\n"
        "        if not isinstance(output, Exception) and hasattr(output, \"req_ids\"):\n"
        "            try:\n"
        "                _tdi = getattr(self.worker, \"take_draft_token_ids\", None)\n"
        "                _drafts = _tdi() if _tdi is not None else None\n"
        "                if _drafts is not None:\n"
        "                    output._pp_piggyback_drafts = _drafts\n"
        "            except Exception:\n"
        "                logger.exception(\"piggybacking draft tokens failed\")\n"
        "\n"
        "        if isinstance(output, Exception):\n"
        "            result = (WorkerProc.ResponseStatus.FAILURE, str(output))",
        "executor: piggyback drafts on model output",
    )
    edit(
        CORE,
        "        engine_core_outputs = self.scheduler.update_from_output(\n"
        "            scheduler_output, model_output\n"
        "        )\n"
        "        self._attach_iteration_details(engine_core_outputs, iteration_details)\n"
        "\n"
        "        # NOTE(nick): We can either handle the deferred tasks here or save",
        "        engine_core_outputs = self.scheduler.update_from_output(\n"
        "            scheduler_output, model_output\n"
        "        )\n"
        "        # PP + spec decode: drafts ride on the model output (see\n"
        "        # multiproc_executor.enqueue_output) instead of a separate RPC.\n"
        "        _drafts = getattr(model_output, \"_pp_piggyback_drafts\", None)\n"
        "        if _drafts is not None:\n"
        "            self.scheduler.update_draft_token_ids(_drafts)\n"
        "        self._attach_iteration_details(engine_core_outputs, iteration_details)\n"
        "\n"
        "        # NOTE(nick): We can either handle the deferred tasks here or save",
        "core: consume piggybacked drafts after output processing",
    )
    edit(
        CORE,
        "            # PP batch queue: dispatch-time collection (model_executed\n"
        "            # True) races the in-flight execute for the worker message\n"
        "            # ring; completion iterations report model_executed False.\n"
        "            # Collect exactly when the queue is drained instead.\n"
        "            take_drafts = not self.batch_queue",
        "            # PP batch queue: the drafter RPC is fully retired -- drafts\n"
        "            # arrive piggybacked on the model output (enqueue_output).\n"
        "            take_drafts = False",
        "core: retire drafter RPC under batch queue",
    )


# --------------------------------------------------------------------------
# 13. Serialize only steps that can SAMPLE; let pure prefill chunks pipeline.
#
# Patch 7's in-flight skip applied to every scheduled step, including chunked
# prefill chunks that do not reach the end of the prompt. Those steps sample
# nothing and draft nothing -- every token they need is already known to all
# ranks (the full prompt ships in NewRequestData; the cursor scheme ships []
# for them) -- so rescheduling the next chunk while the previous is in flight
# is exactly as safe as stock non-spec chunked prefill, which pipelines chunks
# across PP stages. Serializing them cost 8x on prefill: GPU-utilization
# sampling showed a single chunk wavefront walking rank 0->7 with 7 stages
# idle (~480 tok/s at ~0.5 s of compute per 2048-token chunk per stage).
#
# The gate now bites only when num_computed_tokens (already advanced by the
# in-flight step at schedule time) has reached the prompt end -- i.e. the
# in-flight step samples (last prefill chunk or any decode step). Decode
# serialization is unchanged. For preemption-resumed requests num_computed
# passes num_prompt_tokens while recompute chunks are still catching up on
# old OUTPUT tokens; those are also known-token steps, but serializing them
# too is merely conservative, and preemption is rare.
# --------------------------------------------------------------------------
def patch_prefill_chunk_pipelining():
    edit(
        SCHED,
        "            if (\n"
        "                self.use_pp\n"
        "                and not self.scheduler_config.async_scheduling\n"
        "                and self.vllm_config.speculative_config is not None\n"
        "                and getattr(request, \"_pp_spec_in_flight\", False)\n"
        "            ):\n"
        "                # PP + sync + spec decode: this request's previous step has\n"
        "                # not been processed yet. Leftover spec_token_ids would make\n"
        "                # num_new_tokens > 0 and reschedule it on stale drafts,\n"
        "                # while the newly sampled token exists only on the last\n"
        "                # rank's GPU -- non-last ranks would embed the draft and\n"
        "                # their KV would diverge permanently from the last rank's.\n"
        "                # Mirrors the num_new_tokens == 0 serialization that\n"
        "                # non-spec decoding gets for free.\n"
        "                req_index += 1\n"
        "                continue",
        "            if (\n"
        "                self.use_pp\n"
        "                and not self.scheduler_config.async_scheduling\n"
        "                and self.vllm_config.speculative_config is not None\n"
        "                and getattr(request, \"_pp_spec_in_flight\", False)\n"
        "                # ...sampling step only: pure prefill chunks (computed\n"
        "                # cursor still short of the prompt end) ship no sampled\n"
        "                # tokens and no drafts, so they may pipeline across PP\n"
        "                # stages exactly like stock non-spec chunked prefill.\n"
        "                and request.num_computed_tokens >= request.num_prompt_tokens\n"
        "            ):\n"
        "                # PP + sync + spec decode: this request's previous step has\n"
        "                # not been processed yet and may have sampled. Leftover\n"
        "                # spec_token_ids would make num_new_tokens > 0 and\n"
        "                # reschedule it on stale drafts, while the newly sampled\n"
        "                # token exists only on the last rank's GPU -- non-last\n"
        "                # ranks would embed the draft and their KV would diverge\n"
        "                # permanently from the last rank's. Mirrors the\n"
        "                # num_new_tokens == 0 serialization that non-spec decoding\n"
        "                # gets for free.\n"
        "                req_index += 1\n"
        "                continue",
        "scheduler: serialize sampling steps only (prefill chunks pipeline)",
    )


# --------------------------------------------------------------------------
# 15. Enable the pinned-input reuse guard under PP (UPSTREAM BUG).
#
# gpu_model_runner stages inputs in persistent pinned CPU tensors and
# H2D-copies them with non_blocking=True each step, guarded against reuse by
# prepare_inputs_event (synchronize_input_prep wraps every execute_model's
# input prep) -- but upstream only CREATES the event when async scheduling is
# on, assuming the sync path never has more than one batch in flight per
# worker. The PP batch queue violates that on the sync path: several batches
# are in flight, and each rank's H2D copies queue on the compute stream
# BEHIND pipeline send-waits (~0.5 s per stage here: PP=8, PCIe gen2 x4).
# The CPU then laps the still-pending copy and overwrites the pinned buffer,
# so the GPU receives the NEXT batch's input_ids: deterministic garbage KV
# for any prompt needing >= 3 prefill chunks (> 4096 tokens), which then
# poisons the prefix cache. Split prefills at high concurrency hit the same
# race (this -- not the in-flight counter -- was the "counter era"
# server-poisoning corruption; the counter was exonerated). <= 2 chunks
# stayed clean because the first copy always ran before the first send-wait.
#
# Fix: create prepare_inputs_event whenever PP > 1 as well. The existing
# synchronize_input_prep() then makes the CPU wait for the prior step's
# copies before touching the pinned buffers -- honest throttling instead of
# silent corruption, and full-depth pipelining remains whenever copies keep
# up. Upstream-worthy.
# --------------------------------------------------------------------------
def patch_pp_input_prep_guard():
    edit(
        RUNNER,
        "        self.prepare_inputs_event: torch.Event | None = None\n"
        "        if self.use_async_scheduling:\n"
        "            self.async_output_copy_stream = torch.cuda.Stream()\n"
        "            # Blocking (sleep) event to avoid busy-polling the CUDA driver lock;\n"
        "            # under TP contention that spin can balloon and make the rank a straggler.\n"
        "            self.prepare_inputs_event = torch.cuda.Event(blocking=True)",
        "        self.prepare_inputs_event: torch.Event | None = None\n"
        "        if self.use_async_scheduling:\n"
        "            self.async_output_copy_stream = torch.cuda.Stream()\n"
        "        if (\n"
        "            self.use_async_scheduling\n"
        "            or self.parallel_config.pipeline_parallel_size > 1\n"
        "        ):\n"
        "            # Blocking (sleep) event to avoid busy-polling the CUDA driver lock;\n"
        "            # under TP contention that spin can balloon and make the rank a straggler.\n"
        "            #\n"
        "            # PP also needs this guard, not just async scheduling: the PP batch\n"
        "            # queue keeps several batches in flight per worker on the SYNC path\n"
        "            # too, and the pinned CPU staging tensors (input_ids etc.) are\n"
        "            # reused across steps while their async H2D copies may still be\n"
        "            # queued behind a pipeline send-wait on the compute stream. With\n"
        "            # slow stages/interconnect the CPU laps the pending copy and\n"
        "            # overwrites the pinned buffer, so the GPU receives the NEXT\n"
        "            # batch's input_ids -- observed here as deterministic garbage KV\n"
        "            # for any prompt needing >= 3 prefill chunks (PP=8, PCIe gen2 x4).\n"
        "            self.prepare_inputs_event = torch.cuda.Event(blocking=True)",
        "runner: enable input-prep reuse guard under PP",
    )


# --------------------------------------------------------------------------
# 14. Drop STALE drafts at consumption (pipelined batches).
#
# With cross-batch pipelining (patches 12/13) a request can have several
# batches in flight (chunked prefill; split prefills at high concurrency),
# and EVERY popped output may carry piggybacked draft proposals -- including
# proposals made from a mid-prefill partial state, and proposals that lag a
# newer in-flight step. Consuming those lets leftover spec_token_ids make
# num_new_tokens > 0, so the boolean in-flight gate (cleared by the OLDEST
# pop) reschedules the request before its in-flight step's sampled token is
# processed: non-last ranks embed a stale draft at the leading position and
# their KV diverges from the last rank's (observed as "!!!!" = token-id-0
# outputs poisoning the whole server), or decode settles into a 2-deep
# pipeline running one-step-stale drafts at ~0 acceptance (observed after
# every fresh multi-chunk prefill: 1.06-1.38 tok/step at ~33 ms/step).
#
# Invariant: a request whose newest output was just processed by
# update_from_output has num_computed_tokens == num_tokens - 1. Drafts
# consumed in any other state are stale -- drop them. Stateless and
# idempotent, unlike the in-flight counter attempted first (which corrupted
# under concurrent load for reasons never fully traced -- do not revive it).
# --------------------------------------------------------------------------
def patch_stale_draft_guard():
    edit(
        SCHED,
        "            if request.is_prefill_chunk:\n"
        "                # Ignore draft tokens for prefill chunks.\n"
        "                if request.spec_token_ids:\n"
        "                    request.spec_token_ids = []\n"
        "                continue\n"
        "\n"
        "            # Add newly generated spec token ids to the request.",
        "            if request.is_prefill_chunk:\n"
        "                # Ignore draft tokens for prefill chunks.\n"
        "                if request.spec_token_ids:\n"
        "                    request.spec_token_ids = []\n"
        "                continue\n"
        "\n"
        "            if (\n"
        "                self.use_pp\n"
        "                and not self.scheduler_config.async_scheduling\n"
        "                and request.num_computed_tokens != request.num_tokens - 1\n"
        "            ):\n"
        "                # PP + sync + spec (pipelined batches): drafts are only valid\n"
        "                # if proposed right after this request's LATEST verified token.\n"
        "                # A freshly settled request (its newest output just processed\n"
        "                # by update_from_output) always has num_computed_tokens ==\n"
        "                # num_tokens - 1. Anything else means these drafts rode on an\n"
        "                # OLDER batch's output while a newer step of this request is\n"
        "                # still in flight (end of a pipelined chunked prefill, or an\n"
        "                # older decode step) -- accepting them would let leftover\n"
        "                # spec_token_ids make num_new_tokens > 0 and reschedule the\n"
        "                # request before its in-flight step's sampled token was\n"
        "                # processed: non-last ranks then embed a stale draft at the\n"
        "                # leading position and their KV diverges from the last rank's\n"
        "                # (\"!!!!\" outputs), or decode settles into a 2-deep pipeline\n"
        "                # running one-step-stale drafts at ~0 acceptance.\n"
        "                if request.spec_token_ids:\n"
        "                    request.spec_token_ids = []\n"
        "                continue\n"
        "\n"
        "            # Add newly generated spec token ids to the request.",
        "scheduler: drop stale drafts at consumption",
    )
    edit(
        SCHED,
        "        self._inflight_prefills.discard(request)\n"
        "        request.status = RequestStatus.PREEMPTED\n"
        "        request.num_computed_tokens = 0",
        "        self._inflight_prefills.discard(request)\n"
        "        request.status = RequestStatus.PREEMPTED\n"
        "        request.num_computed_tokens = 0\n"
        "        # PP+spec in-flight counter: outputs of a preempted request's in-flight\n"
        "        # batches are discarded, so their decrements never arrive. Reset here\n"
        "        # or the request is skipped forever at sampling steps after resume.\n"
        "        request._pp_spec_in_flight = 0",
        "scheduler: reset in-flight counter on preemption",
    )


# --------------------------------------------------------------------------
# 12. Relax the global serialization to opt-in (GLM52_PP_SPEC_GLOBAL_SER=1).
#
# Patch 8's global gate existed to keep the drafter RPC from interleaving with
# in-flight batches -- but patch 11 retired that RPC entirely (drafts ride on
# the model-output reply, one reply stream). With the root cause gone, global
# serialization only costs throughput: it collapses every runnable request
# into one lockstep batch per pipeline round, so PP stages idle 7/8 of the
# time at any concurrency. Per-request serialization (patch 7) is what
# correctness actually requires -- a request is never rescheduled until its
# previous output is processed, so cursor shipping and drafts stay fresh; it
# says nothing about DIFFERENT requests overlapping in the pipe, which is
# exactly the concurrency non-spec PP enjoys.
#
# Keep the old behavior reachable for A/B or as an escape hatch: set
# GLM52_PP_SPEC_GLOBAL_SER=1 in the server environment.
# --------------------------------------------------------------------------
def patch_relax_global_serialization():
    edit(
        SCHED,
        "import itertools\nimport time",
        "import itertools\nimport os\nimport time",
        "scheduler: import os for serialization gate",
    )
    edit(
        SCHED,
        "from vllm.compilation.cuda_graph import CUDAGraphStat",
        "# GLM-5.2 PP+spec escape hatch: force global batch serialization (the\n"
        "# pre-patch-11 behavior). Off by default; see patch_mtp_pp.py patch 12.\n"
        "_PP_SPEC_GLOBAL_SER = os.environ.get(\"GLM52_PP_SPEC_GLOBAL_SER\", \"0\") == \"1\"\n"
        "\n"
        "from vllm.compilation.cuda_graph import CUDAGraphStat",
        "scheduler: define global-serialization flag",
    )
    edit(
        SCHED,
        "        elif (\n"
        "            self.use_pp\n"
        "            and not self.scheduler_config.async_scheduling\n"
        "            and self.vllm_config.speculative_config is not None\n"
        "            and any(\n"
        "                getattr(r, \"_pp_spec_in_flight\", False) for r in self.running\n"
        "            )\n"
        "        ):\n"
        "            # PP + sync + spec decode: never dispatch a new batch (for ANY\n"
        "            # request, running or waiting) while one is in flight. The\n"
        "            # drafter RPC (take_draft_token_ids) issued between two\n"
        "            # in-flight batches desyncs the executor response stream and\n"
        "            # hangs the engine; see patch notes.\n"
        "            token_budget = 0",
        "        elif (\n"
        "            _PP_SPEC_GLOBAL_SER\n"
        "            and self.use_pp\n"
        "            and not self.scheduler_config.async_scheduling\n"
        "            and self.vllm_config.speculative_config is not None\n"
        "            and any(\n"
        "                getattr(r, \"_pp_spec_in_flight\", False) for r in self.running\n"
        "            )\n"
        "        ):\n"
        "            # PP + sync + spec decode, opt-in escape hatch: never dispatch\n"
        "            # a new batch while one is in flight. This was required while\n"
        "            # the drafter RPC (take_draft_token_ids) could interleave with\n"
        "            # in-flight batches and desync the executor response stream;\n"
        "            # that RPC is retired (drafts piggyback on the model output),\n"
        "            # so per-request serialization alone suffices for correctness\n"
        "            # and cross-batch pipelining is allowed again.\n"
        "            token_budget = 0",
        "scheduler: gate global serialization on env (default off)",
    )


def revert():
    import shutil

    for f in (RUNNER, SCHED, CORE):
        src = os.path.join(BACKUP, f)
        if not os.path.exists(src):
            print(f"  ! no backup for {f}", file=sys.stderr)
            sys.exit(1)
        shutil.copyfile(src, os.path.join(VLLM, f))
        print(f"  - reverted {f}")
    print("note: deepseek_mtp.py patches left in place (inert without spec config);")
    print(f"restore manually from {BACKUP} if needed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--revert", action="store_true")
    a = ap.parse_args()
    print(f"vllm at {VLLM}")
    if a.revert:
        revert()
    else:
        patch_supports_pp()
        patch_draft_embeddings()
        patch_drafter_guards()
        patch_scheduler_cursor_shipping()
        patch_worker_cursor_consume()
        patch_worker_resume_recovery()
        patch_serialize_inflight()
        patch_global_serialization()
        patch_draft_rpc_gate()
        patch_no_empty_dispatch()
        patch_piggyback_drafts()
        patch_relax_global_serialization()
        patch_prefill_chunk_pipelining()
        patch_stale_draft_guard()
        patch_pp_input_prep_guard()
        print("done - restart the server for changes to take effect")
