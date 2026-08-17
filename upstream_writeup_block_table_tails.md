# [Draft — vLLM issue] Stale block-table row tails can expose other requests' KV pages

**Component:** `vllm/v1/worker/block_table.py` (`BlockTable.append_row`,
`BlockTable.move_row`)
**Observed on:** v0.26.0, PP=8, DeepSeek-V3.2-style DSA model (GLM-5.2),
sm_80, chunked prefill + MTP spec decode. The mechanism is
model-independent; the *visibility* depends on consumers that read past a
row's true block count.

## Summary

`append_row` and `move_row` write only `[0:num_blocks]` of a block-table
row. The tail of the row retains whatever the previous tenant of that row
slot wrote — i.e., **block ids belonging to another request**. Any
consumer that walks even 1–2 blocks past the row's true boundary reads
another request's KV pages.

Consumers do this in practice: full-width row copies in the DSA indexer's
expand path, and any transiently over-extended effective length. In our
deployment the result was cross-request KV reads that manifested as
content splices at temperature 0: because all our test prompts shared a
common header, the foreign pages scored highly in the DSA indexer's
top-k and entered attention — producing "plausible reconstruction"
corruption (wrong-but-sensible substitutions, list jumps) rather than
garbage, which made it hard to attribute.

## Reproduction sketch

1. Serve any paged-KV model with several concurrent requests of unequal
   lengths so block-table row slots get recycled between requests of
   different block counts.
2. Instrument any consumer that copies block-table rows at full width
   (or dump rows directly): after a long-tenant row is recycled to a
   short-tenant request, `row[num_blocks:]` still holds the prior
   request's block ids.
3. With a DSA/sparse-attention model, corruption becomes user-visible as
   cross-request content splices at temp 0; with dense attention the
   stale ids are usually (not provably) unread.

## Fix that worked for us

Pad each row's tail with the row's **own last valid block id** on every
`append_row`/`move_row` (µs-scale numpy fill on row mutation only — not
on the hot path), and make `move_row` copy full padded rows so the
padding survives moves. Over-reads then return the request's own last
page: content-neutral. In our deployment this eliminated all
foreign-content corruption (failure rate 3× down overall; remaining
failures were a separate numerics issue).

An alternative is zero/sentinel-padding plus hard bounds validation in
consumers, but self-padding is strictly safer with zero consumer churn.

## Notes

- The bug is silent by construction: nothing validates that consumers
  stay within `num_blocks`, and the stale ids are *valid* block ids, so
  no bounds check fires.
- Security consideration: this is cross-request data exposure within a
  serving process (prompt-content-dependent), which may deserve more
  than a correctness label.

*Full investigation log available (multi-day root-cause across a
753B MoE deployment); happy to provide details, traces, or a PR.*
