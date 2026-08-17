# Pipeline-parallel rank boundaries silently corrupt DSA sparse-attention output (stale `topk_indices_buffer`)

## TL;DR

For DeepSeek-V3.2-style **DSA (sparse attention indexer)** models, the per-rank
`topk_indices_buffer` is **not shipped across pipeline-parallel (PP) hops**. When a
PP partition starts a stage on a *shared* (skip-topk) layer whose *full* indexer
source layer lives on the **previous** rank, that layer — and every layer downstream
of it — attends using **stale top-k selections left in the buffer from a prior
batch/step**. Output corruption is silent (no crash, no assert), context- and
schedule-dependent, and only becomes visible under memory/scheduling pressure or
speculative decoding. Any DSA model with `index_topk_freq > 1` served with PP that
splits a full→shared indexer run across a rank boundary is affected.

## Affected configuration (reproduction context)

- Model: GLM-5.2 (753B DSA MoE), `index_topk_freq=4`, `index_skip_topk_offset=3`,
  `num_hidden_layers=78`. `config.json` `indexer_types` =
  `['full','full','full','shared','shared','shared','full', ...]`.
- vLLM 0.26.0, TRITON_MLA_SPARSE backend, PP=8, TP=1.
- Partition `VLLM_PP_LAYER_PARTITION=11,10,10,10,10,10,10,7`.
- Speculative decoding (MTP) amplifies it (see §Amplifiers); it is present without
  MTP but usually benign there.

Full-indexer layers under this config are `0,1,2,6,10,14,18,22,...` (i.e. L0–L2, then
every 4th layer). With the partition above, **every** PP stage `>=1` begins on a
*shared* layer whose nearest full-indexer source is on the previous rank:

| stage start | shared? | full source | source rank | start rank | cross-hop |
|---|---|---|---|---|---|
| L11 | yes | L10 | 0 | 1 | ✔ |
| L21 | yes | L18 | 1 | 2 | ✔ |
| L31 | yes | L30 | 2 | 3 | ✔ |
| L41 | yes | L38 | 3 | 4 | ✔ |
| L51 | yes | L50 | 4 | 5 | ✔ |
| L61 | yes | L58 | 5 | 6 | ✔ |
| L71 | yes | L70 | 6 | 7 | ✔ |

## Root cause

1. **Skip site** — `vllm/model_executor/layers/mla.py:169`:
   ```python
   if self.indexer and self.is_sparse and not self.skip_topk:
       self.indexer(...)   # writes self.topk_indices_buffer
   ```
   A `skip_topk` (shared) layer **never calls the indexer**. Its MLA attention
   consumes the buffer as-is via the backend
   (`triton_mla_sparse.py:223`, and identically `xpu_mla_sparse.py:256`):
   ```python
   topk_indices = self.topk_indices_buffer[:num_actual_tokens]
   ```
   The read is **backend-agnostic**: any backend that consumes the shared buffer
   (Triton and XPU sparse both do) is affected. The bug is a property of the
   buffer's PP lifetime, not of a specific attention kernel.

2. **Skip pattern** — `deepseek_v2.py:1100`:
   ```python
   skip = (max(layer_id - index_skip_topk_offset + 1, 0) % index_topk_freq) != 0
   ```
   A shared layer is *designed* to reuse the selections computed by the most recent
   full layer **earlier in the same forward pass**. This is correct within a single
   rank, because that full layer ran first and populated the buffer.

3. **The buffer is per-rank** — created in `DeepseekV2Model.__init__`
   (`deepseek_v2.py:~1388`), shape `max_num_batched_tokens x index_topk`, int32,
   and **never transmitted across PP hops**. When a stage *starts* on a shared layer,
   the full layer that should have seeded the buffer ran on the **previous** rank, in
   a **different process**, whose buffer this rank cannot see. This rank's buffer
   still holds whatever its *own* last full layer wrote — **on the previous batch or
   decode step**. (Buffer allocation: `deepseek_v2.py:1388`,
   `torch.empty(max_num_batched_tokens, index_topk, dtype=int32)`.)

So each rank-boundary shared layer attends with stale top-k indices:
- **Quiet / serial traffic:** a same-request, prior-chunk shift — usually benign,
  which is why it hid for so long.
- **Under co-flight (concurrent requests):** another request's decode-step
  selections, layered with older leftovers → **poisoned KV downstream**.

## Evidence (origin + cascade signature)

We captured per-block KV-cache byte-sums for a fixed 122K-token request across two
independent quiet builds (`quietA` vs `quietB`), per rank, per layer. A correct
instrument on a deterministic build must differ only in the final partial block (the
freshly generated tokens). Result:

```
rank 0: L0..L10  all diff=1   (final block only — SOUND)
rank 1: L11 diff=1 (SOUND) -> L12=1949 -> L13=1981 -> L14=1981 (all 1981 blocks)
rank 2: L21=1981   (diverges at the FIRST layer of the stage)
rank 3..7: first layer already =1981
```

This is exactly the mechanism's fingerprint:

- **Rank 0** has no rank-boundary staleness (its shared layers L3–5, L7–9 read the
  buffer its own full layers L2/L6 seeded earlier in the same pass) → fully sound.
- **Rank 1 is the origin.** Its first layer L11 is *shared* and reads a stale buffer,
  but **L11's KV is projected from the layer input (pre-attention)**, which is the
  still-sound hidden state received from rank 0 — so L11's KV is clean. L11's stale
  *attention* corrupts L11's **output** hidden, which is L12's input → **L12's KV is
  the first to diverge**, cascading through the rest of the rank.
- **Ranks 2–7** receive an already-corrupted hidden state, so they are poisoned from
  their **first** layer.

The single-layer sound→diverge boundary at the origin rank, plus full poisoning from
layer 1 on all downstream ranks, is uniquely explained by "stale per-rank buffer at
the PP boundary" and rules out numerics/hardware/allocator theories.

### Why it was elusive
Every determinism control we had trusted (fixed-position indexer-K bytes,
bit-identical across rebuilds) was measured on **rank 0 / early layers** — the one
region with no rank-boundary staleness. The corruption is present in quiet serial
builds too (`quietA != quietB` on rank 1+), but only benignly; the investigation was
structurally blind to it until the map covered all ranks.

## Amplifiers

- **Concurrency (co-flight):** the stale buffer holds a *different request's*
  selections, so the corruption tracks scheduling pressure.
- **Speculative decoding (MTP):** the drafter runs its own full indexer and stomps
  rank 7's buffer every spec iteration (`index_share_for_mtp_iteration`), so the
  target model's rank-7 shared layers read drafter-derived selections. With MTP off,
  only a benign leftover remains → clean.

## Fix (throughput-preserving)

Ship the most recent full-indexer selections across the PP hop inside the same
`IntermediateTensors` that already carry `hidden_states`, and **seed the receiving
rank's `topk_indices_buffer` at forward entry**, before any shared layer runs.

- Cost: ~16 MB per prefill chunk hop (≈ +4% on a 120K prefill), ~100 KB per decode
  hop.
- Because the top-k rows ride in the same `IntermediateTensors` as the hidden states,
  they inherit the identical batch/row layout — both ranks build metadata from the
  same broadcast `scheduler_output` with the same reorder, so row spaces align by the
  same mechanism that aligns the hiddens.
- Re-seeding at entry every batch also fixes the MTP rank-7 drafter-stomp for free:
  consumption always follows a fresh seed.
- A stage that starts on a full layer (not the case here) is a no-op by construction:
  the full layer overwrites the buffer before any shared consumer reads it.

### Implementation hazard: do NOT name the model-level reference `topk_indices_buffer`

The relay needs a model-level reference to the per-rank buffer. Naming that
attribute `topk_indices_buffer` on the model class silently triggers vLLM's
attribute-sniffed MTP buffer sharing (`llm_base_proposer.py:1579`:
`if hasattr(target_language_model.model, "topk_indices_buffer")` → the
drafter's private buffer is discarded and rebound to the target's), which
collapsed draft acceptance from ~3.9 to ~1.5 on long-context workloads while
leaving outputs correct — a ~2x decode-throughput loss that no correctness
test can see. Use a private name (e.g. `_pp_topk_relay_buf`). Full analysis
in the companion writeup on attr-sniffed MTP buffer sharing.

### Rejected alternatives
- **Repartition so every stage starts on a full layer:** rank-7 memory (drafter) is
  infeasible under this constraint.
- **Recompute the indexer at the boundary:** the indexer prefill logits are too
  expensive to redo.

## Acceptance criteria

The deliverable is measured by **task accuracy**, not bit-determinism (upstream vLLM
is itself non-deterministic at the ULP level; only accuracy degradation is a defect).

- **Primary (accuracy):** production-config copy-fidelity regression — the exact
  high-pressure protocol that reproduced the corruption (long-context verbatim copy,
  MTP k=3, decode-splitting, fp8_ds_mla KV, guard off) must go from its pre-fix
  failure rate to ~0 failures. **Status: MET in eager — 0/21 vs 14/21 pre-fix** at
  764/382/573, fresh-uid 120K. Must be re-confirmed under cudagraphs (see below).
- **Cudagraph re-confirmation:** the fix is a buffer write at forward entry whose
  value must be re-read on every graph replay — the cudagraph-aliasing failure class.
  **Status: MET — 0/21 under full-production hybrid cudagraphs**, same protocol/hot
  sets. The safe pattern was used: eager per-step staging into the *persistent*
  IntermediateTensors recv buffers, then a captured persistent-to-persistent seed
  copy that replays correctly (same mechanism as seq_lens/block_table updates); the
  extra topk key is registered for capture on both PP send and recv.
- **Non-regression (throughput):** the per-hop relay adds ~16 MB per prefill chunk
  hop and ~100 KB per decode hop. **Status: MET — relay cost ≈ 0.** A relay-on vs
  relay-off A/B on the identical stack read 19.4/118.3 vs 15.6/127.2 tok/s
  (conc1/conc16), i.e. within short-bench noise (conc1 was faster *with* the relay).
  The interconnect math (gen2 x4, ~2 GB/s, no P2P): ~56 ms added across 7 prefill
  hops of a multi-second 120K prefill (<1%), sub-ms per decode step — confirmed.

The per-rank KV cachemap gate was the **localization microscope** used to root-cause
and confirm the fix (it collapsed the rank-1-L12+/ranks-2–7 cascade to final-block-
only after the fix); it is a diagnostic, not a shipping gate.

### Deployment completeness: relay is necessary, not sufficient alone
The relay is *the* fix for the PP rank-boundary corruption documented here, but a
production deployment with bit-clean output on this workload also requires
**deterministic decode-time top-k** (tie-breaking). Tested directly: with the relay
ON but deterministic decode-topk OFF, the copy-fidelity protocol failed on the first
pass (uid 20764, digit transposition `8258584576`→`8258258576`) — a genuine temp-0
selection flip at a near-tie, an *independent* corruption channel from the PP buffer.
So the shipping config is **relay + deterministic decode-topk**. (Deterministic
`moe_align` is retained as free hygiene — an A/B showed it costs ≈0 at these shapes,
and its nondeterminism is only atomics-ULP, not a selection change.) Note the two
channels are independent: each was necessary, neither alone was sufficient for 0
failures.

### Note on residual ULP-scale variance (expected, benign)
After the fix, a quiet build's KV still shows a small, scattered, pre-state-keyed
variance that enters at the first MoE layer (traced to history-dependent kernel
autotune/algo-selection, not to the sparse-attention path). This is **consistent with
upstream vLLM's normal non-determinism**, does not change task output (victims pass),
and is **not** part of this bug. No fix is warranted.

## Scope

This is not GLM-5.2-specific. Any DSA / sparse-indexer model with
`index_topk_freq > 1`, served with pipeline parallelism whose partition places a
shared (skip-topk) layer at a stage boundary with its full source on the prior rank,
will silently corrupt output — **independent of attention backend** (Triton and XPU
sparse consumers both exhibit it). The V100 community fork worked around a related hazard
by pinning block size ("PP stages starting on a shared indexer layer"); the correct
general fix is to make the buffer travel with the hidden state across the hop.

## Appendix: performance epilogue

The fix itself is throughput-neutral (relay-on vs relay-off A/B within noise; §Acceptance).
A follow-up investigation of an apparent decode-throughput shortfall vs older reference
numbers found **no regression attributable to any correctness component**:

- **Relay, deterministic decode-topk, and deterministic `moe_align` each cost ≈0** at
  production shapes (per-component env-gated A/Bs; conc16 moved <10% across every
  toggle).
- **Quantization is acceptance-neutral.** W8 lm-head (shared by the MTP drafter) and
  fp8 KV were each toggled off and measured bit-exact-identical to baseline under
  greedy (spec acceptance length unchanged to two decimals per workload). These
  capacity/memory optimizations are therefore *free* on acceptance — a net positive.
- The apparent gap was mostly a **phantom reference** (the cited number came from a
  pre-patch favorable-batching regime; the honest same-era figure was ~10–20% higher
  than the "current" number, i.e. near-parity) plus **cross-workload / cross-context
  anchor mismatch** (historical acceptance anchors were measured on different prompts
  and context lengths with a since-rebuilt tooling path).
- A small residual acceptance difference on one synthetic workload remains **bounded,
  uncharacterized, and non-blocking** — it is config-invariant across every revertable
  knob, so it lives (if anywhere) in unrevertable patch evolution with no checkpoints
  to bisect, and it does not affect task accuracy.

Net: the shipping config (relay + deterministic decode-topk + deterministic moe_align,
fp8 KV, W8 head, guard off) is both correct and throughput-neutral versus the honest
pre-fix baseline.
