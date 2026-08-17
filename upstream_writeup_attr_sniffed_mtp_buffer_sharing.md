# MTP drafter buffer-sharing is keyed on `hasattr`, silently hijacking any model that exposes `topk_indices_buffer`

## TL;DR

`llm_base_proposer.py` decides whether to rebind the MTP drafter's
`topk_indices_buffer` to the **target model's** buffer by *attribute sniffing*:

```python
# llm_base_proposer.py:1579-1592 (also v1/worker/gpu/spec_decode/eagle/utils.py:103-112)
if hasattr(target_language_model.model, "topk_indices_buffer"):
    target_buffer = target_language_model.model.topk_indices_buffer
    if hasattr(self.model.model, "topk_indices_buffer"):
        del self.model.model.topk_indices_buffer
    self.model.model.topk_indices_buffer = target_buffer
    for module in self.model.modules():
        if hasattr(module, "topk_indices_buffer"):
            module.topk_indices_buffer = target_buffer
    logger.info(
        "Detected MTP model with topk_indices_buffer. "
        "Sharing target model topk_indices_buffer with the draft model."
    )
```

Any model class that exposes a **model-level attribute named
`topk_indices_buffer`** — for any reason — silently activates this path. The
drafter's private buffer (e.g. `deepseek_mtp.py:103`) is discarded and every
drafter module is rebound to the target's live buffer. There is no config
check: the intended gate for MTP index sharing
(`index_share_for_mtp_iteration`) is consulted elsewhere for skip-toggling,
but **the rebind itself keys purely on the attribute name**.

## How we hit it

While fixing a separate pipeline-parallel bug (see the companion writeup on
PP rank-boundary stale indexer buffers), our patch added one innocuous line to
`DeepseekV2Model.__init__` so the forward pass could reference the buffer:

```python
self.topk_indices_buffer = topk_indices_buffer   # model-level alias
```

Stock `DeepseekV2Model` never exposes this attr (the buffer is passed to
layers via constructor argument only), so the sharing path had always been
dormant for this model family. Our alias made `hasattr` true. From the very
next boot, the proposer printed "Sharing target model topk_indices_buffer…"
and rebound the drafter.

## Impact: silent quality collapse, no crash, no output corruption

With sharing active, the drafter's k-step draft loop and the target's
sparse-attention layers read/write the **same** buffer. In our deployment
(GLM-5.2, 753B DSA MoE, PP=8, MTP k=3) the observable effect was:

- **Draft acceptance collapsed from ~3.9-4.0 to ~1.3-1.7 tokens/step** on
  long-context workloads (verbatim copy at 8K-122K context) — a ~2x decode
  throughput loss.
- **Short-context workloads were unaffected** (at ctx < `index_topk` the
  sparse selection is trivially complete, so wrong/stale selections are
  indistinguishable from correct ones). Every short-context A/B we ran was
  structurally blind to the regression.
- **Task output stayed correct** — rejected drafts are simply eaten by the
  verifier — so no correctness test caught it. Only acceptance-rate telemetry
  showed it.

Forensics that pinned it: per-boot `Mean acceptance length` extracted from
~50 server logs across one day formed a perfect separator — every boot whose
log contains the "Sharing target model topk_indices_buffer" line is degraded,
every boot without it is healthy, with the cliff landing on the exact boot
(to the second) where the attribute first existed. Reverting the one alias
line (renaming our reference to a private name the probe cannot see) restored
acceptance to 4.00/100% on the same probes.

## Suggested fix

Key the rebind on explicit configuration, not attribute presence:

```python
share = getattr(draft_hf_config, "index_share_for_mtp_iteration", False)
if share and getattr(target_language_model.model, "topk_indices_buffer", None) is not None:
    ...
```

or require the target model to opt in via a well-known interface
(`SupportsIndexSharing` protocol / an explicit registry), so that an
attribute name collision on a 3rd-party or patched model cannot silently
change drafter data flow. At minimum, the log line should be a `warning`
rather than `info`, since it changes numerical behavior of speculation.

## Reproduction sketch

1. Take any DSA/MTP model whose drafter allocates its own
   `topk_indices_buffer` and whose target model does *not* expose the attr —
   acceptance healthy.
2. Add `self.topk_indices_buffer = <the buffer>` to the target model class.
   No other change.
3. Boot with MTP speculation; observe the "Sharing…" log line and a large
   drop in mean acceptance length on context-dependent workloads
   (ctx > `index_topk`), with unchanged task output.
