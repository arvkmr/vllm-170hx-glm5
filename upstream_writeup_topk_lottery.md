# [Draft — vLLM issue] DSA top-k kernels: nondeterministic tie-breaking makes every KV build a lottery (and the prefill output convention is undocumented)

**Component:** `_C.persistent_topk`, `_C.cooperative_topk`,
`ops.top_k_per_row_decode`, `ops.top_k_per_row_prefill` (DeepSeek-V3.2 /
DSA sparse-attention index selection)
**Observed on:** v0.26.0, GLM-5.2 (753B DSA MoE), sm_80, fp8 indexer
K-cache, 120K+ contexts. Mechanism applies to any DSA model; severity
scales with context length and quantization.

## Part 1 — the tie lottery

DSA indexer logits contain **exact ties**: keys are fp8-quantized, and
long repetitive contexts produce many identical key vectors. Among
tied boundary candidates the top-k selection is a per-call lottery.
Measured offline (coarse-grid tied logits, 4×120K rows, topk 2048, quiet
GPU, back-to-back identical calls):

- `persistent_topk`: **~1,600–2,000 of 2,048 selected indices differ per
  call** (score multiset always exact — exact in values, lottery in sets)
- `topKPerRowDecode`: ~25–145 differ per call
- `torch.topk` (CUB): stable

### Why this matters more than "just tie order"

**Decode-side:** under MTP speculative decoding the selection lottery
cascades: per-step attention wobble → accept-pattern lottery →
batch-shape lottery → margin spread → temp-0 near-tie token flips.
(A canonical tie-break — keep the unique score>v* set, fill boundary
ties lowest-index-first — fixed this; ~0.24 ms for 24×262K rows.)

**Prefill-side (the severe one):** the prefill top-k decides which
positions each prefill token *attends to while the KV cache is being
built*. The lottery therefore makes **every request's KV build
nondeterministic**: per-chunk attention-set wobble → hidden-state wobble
compounding across chunks → the fp8 indexer K quantization **amplifies
ULP-level wobble into full quantization-step key changes** (we measured
1–3 bytes/132 differing at deep positions between two builds of an
identical prompt) → decode-time indexer logits shift O(0.1–1.0) at the
majority of positions → top-2048 boundary membership swaps 30–40
positions per layer-call → chaos amplification over 92 layers →
multi-logit swings at the LM head → temp-0 corruption at near-tie
tokens.

End-to-end verification: with a deterministic prefill selection
substituted, three from-scratch rebuilds of a 124K-token prompt produced
**bit-identical cache bytes, decode logits, and selections — across
different physical block layouts**. With the stock kernel, every rebuild
differs — quantified with an attended-position sampler over both the
indexer-K and MLA latent caches: **~57% of each row's top-2048 selected
positions differ between two builds of the identical prompt**
(|A∩B| ≈ 1178/2048 averaged over layers/ranks), and among positions
selected in both builds, 90–98% of the dequantized cache values differ.

Practical consequence: at temp 0, the same prompt gives different
long-context retrieval quality per submission — irreproducible evals,
and (with tight margins) verbatim-copy corruption. This is invisible on
H100-class deployments only to the extent margins are wider; the
mechanism is architectural.

## Part 2 — `top_k_per_row_prefill`'s output convention is undocumented
(and easy to get wrong)

Established empirically (probe: rows with `ks=[0,0,100,3000]`,
`ke=[50,2500,300,4096]`, untied logits):

1. Output indices are **relative to `cu_seqlen_ks`** (a row with
   ks=3000, window 1096 emits 0..1095). Consumers
   (`triton_convert_req_index_to_global_index`) require this.
2. Rows with `window <= topk` emit **identity order** `0..w-1` then
   `-1` padding — *not* value-sorted.

We initially wrote a replacement emitting absolute value-sorted indices;
it validated on single-chunk (ks=0) tests and corrupted everything else.
Suggest documenting the contract at the op definition and adding a
ks>0 parity test.

## Suggested upstream direction

- Deterministic tie-breaking (score desc, index asc) in the top-k
  kernels, or an opt-in canonicalization pass — decode *and* prefill.
- Document the prefill op's relative-index/identity-short-row contract.
- A KV-build reproducibility test: build the same long prompt twice,
  byte-compare the indexer cache.

## Canonicalization cost (measured, A100-class sm_80)

The windowed prefill canonicalization we run in production is a
two-stage pass over the native kernel's output: a per-row Triton kernel
rebuilds the canonical *set* (keep the unique score>v* entries, refill
boundary ties with lowest-index positions, [ks,ke)-windowed, identity
short rows), then two `torch.sort` calls on the [rows, K] slice produce
the canonical (score desc, index asc) *order*. At 2048 rows x 131072
width, K=2048: **3.8 ms/call**, vs ~53 ms for a full-row-width masked
`torch.topk` overwrite (the validation-mode alternative) — ~15x, and
~2s total added to a 120K-token prefill across chunks and layers (~2%).
Determinism and parity with a stable-sort canonical reference verified
across tie densities, plus set-parity with the native kernel on untied
logits at ks>0 (`test_prefill_topk_canon.py`).

Note for implementers: `torch.topk`'s tie choice is
implementation-defined and is NOT lowest-index under heavy ties, so it
cannot serve as the canonical reference (it is merely run-to-run
stable); use a stable descending sort as ground truth.

*Full investigation log, measurement harnesses, and a working
canonicalization kernel (windowed [ks,ke) variant included) available;
happy to upstream.*


## Addendum — in-place padded topk write clobbers prefill rows (native next_n path)

Separate from the tie lottery: in a MIXED batch under spec decode, the
indexer decode branch writes `topk_indices_buffer[:num_padded_tokens]`
(num_padded = num_decodes*next_n) IN PLACE, while the prefill branch has
already written its chunk selections at `[num_decode_tokens:]` (prefill is
decode-first: first prefill chunk token_start == num_decode_tokens). When
`requires_padding` makes num_padded > num_decode (variable decode lens,
which exist only under spec), the decode topk write overwrites the first
(num_padded - num_decode) prefill rows; the requires_padding unpack
restores only decode rows [0:num_decode], so those prefill positions keep
pad-row content and feed garbage into that chunk's sparse attention.

Structurally confirmed by code trace (batch_size==num_decodes; spill under
padding; prefill token_start==num_decode_tokens overlap; attention consumes
topk only after the op returns, so no intermediate read saves it). NOT
reproduced on sm_80 because `use_flattening=True` there makes decode
metadata per-token pseudo-requests with uniform lens, so requires_padding
never fires — but it is live on the native next_n>1 path (Hopper/deep_gemm).
Fix: run the padded topk into a scratch tensor and copy only the real
(unpacked) rows back to the shared buffer; also guard the num_decode==0
dummy path that writes buffer row 0.
