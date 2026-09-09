"""Batch-invariant decode-row MoE dispatch for GLM-5.2 W4A16 Marlin.

The last residual of the long-context copy corruption
(NOTES_longctx_copy_fidelity.md) is deterministic batch-composition
sensitivity of decode-row numerics in shared-batch MoE GEMMs: fused Marlin
picks moe_block_size 8 for decode-only batches but 64 once a prefill chunk
shares the batch, changing the K-reduction order for the SAME decode row.
At temp 0 with near-tie logits that flips greedy tokens.

Fix: route decode-token rows (rows [0:num_decode_tokens]; decode-first
order is guaranteed by the reorder_batch_threshold patch) through the
GEMV in moe_gemv_marlin.cu. Each (token, expert, chunk) block there reads
only its own activation row and the expert weights, and reduces in a fixed
order -- a token's output is bit-identical regardless of co-batched tokens.
Prefill rows keep fused Marlin. Also a perf win: the Marlin small-batch
floor (~0.45 ms/call) was the top decode bottleneck.

Env: GLM52_SPLIT_MOE=1 enables (default 0 until validated).

Capture notes: the dispatch runs inside the opaque torch.ops.vllm.moe_forward
custom op. FULL cudagraph capture of uniform-decode batches sees real attn
metadata (num_decode_tokens == num_tokens), so the gemv path is what gets
captured and replayed -- consistent, since FULL replays are always uniform
decode. Piecewise capture happens on dummy runs (metadata None) and bakes
the stock path; mixed batches small enough to replay piecewise (<= 64
tokens) therefore keep old behavior. KNOWN HOLE, judged rare (final prefill
chunks < ~56 tokens); revisit if the residual survives.
"""

import os

import torch

_ENABLED = os.environ.get("GLM52_SPLIT_MOE", "0") == "1"
# Shadow mode (eager-only diagnostic): run the stock path for the WHOLE
# batch as reference output, then the gemv on the decode slice into a
# scratch buffer; log per-call divergence and dump the first badly
# diverging call's inputs for offline repro. Production output = stock.
_SHADOW = os.environ.get("GLM52_SPLIT_MOE_SHADOW", "")
_shadow_dumped = False
_shadow_calls = 0
_ext = None
_meta_key = None


def _get_ext():
    global _ext
    if _ext is None:
        from torch.utils.cpp_extension import load
        here = os.path.dirname(os.path.abspath(__file__))
        # installed names in site-packages/vllm; repo names as fallback
        cu = os.path.join(here, "_glm52_moe_gemv_marlin.cu")
        cpp = os.path.join(here, "_glm52_moe_gemv_marlin_bind.cpp")
        if not os.path.exists(cu):
            cu = os.path.join(here, "moe_gemv_marlin.cu")
            cpp = os.path.join(here, "moe_gemv_marlin_bind.cpp")
        _ext = load(
            name="glm52_moe_gemv_ext",
            sources=[cu, cpp],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
    return _ext


def num_decode_tokens_in_batch():
    """num_decode_tokens of the current batch from the DSA indexer metadata,
    or None when unavailable (dummy/profile runs, DBO microbatching).

    Only the indexer metadata is trustworthy here: TritonMLASparseMetadata
    deliberately reports the whole batch as decode to force forward_mqa.
    """
    global _meta_key
    from vllm.forward_context import get_forward_context
    try:
        md = get_forward_context().attn_metadata
    except Exception:
        return None
    if not isinstance(md, dict):
        return None
    if _meta_key is not None:
        m = md.get(_meta_key)
        if m is not None:
            return int(m.num_decode_tokens)
    for k, m in md.items():
        if type(m).__name__ == "DeepseekV32IndexerMetadata":
            _meta_key = k
            return int(m.num_decode_tokens)
    return None


def moe_gemv_forward(x, w13, w13_scale, w13_zp, w2, w2_scale, w2_zp,
                     topk_weights, topk_ids, out, block_size=8):
    """Routed-experts MoE for a decode slice via the batch-invariant GEMV.

    x: [M, K] bf16/fp16, row-contiguous. out: [M, K], written in place.
    Weight/scale/zp tensors are the layer's Marlin-packed buffers, used
    as-is. All ops are capture-safe (pure GPU, no host syncs).
    """
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size)
    M, K = x.shape
    top_k = topk_ids.shape[1]
    E = w13.shape[0]
    N2 = w13_scale.shape[2]          # 2 * intermediate (gate+up)
    inter_sz = N2 // 2
    G = K // w13_scale.shape[1]
    nvt = M * top_k
    sorted_ids, expert_ids, _ = moe_align_block_size(topk_ids, block_size, E)
    ext = _get_ext()
    tw = topk_weights.reshape(-1).float()
    y1 = torch.empty(nvt, N2, dtype=torch.float32, device=x.device)
    ext.moe_gemv_marlin(x, w13, w13_scale, w13_zp, y1, sorted_ids,
                        expert_ids, tw, K, N2, G, block_size, nvt,
                        False, top_k)
    inter = (torch.nn.functional.silu(y1[:, :inter_sz])
             * y1[:, inter_sz:]).to(x.dtype)
    y2 = torch.empty(nvt, K, dtype=torch.float32, device=x.device)
    ext.moe_gemv_marlin(inter, w2, w2_scale, w2_zp, y2, sorted_ids,
                        expert_ids, tw, inter_sz, K, G, block_size, nvt,
                        True, 1)
    out.copy_(y2.view(M, top_k, K).sum(dim=1).to(out.dtype))


def _splittable(self, expert_map, apply_router_weight_on_input):
    """Config guard: the gemv covers exactly the GLM-5.2 serving shape --
    uint4 asymmetric (zp present), no EP, no LoRA, no bias, no input quant,
    router weights folded at GEMM 2."""
    return (expert_map is None
            and self._lora_context is None
            and not apply_router_weight_on_input
            and self.w1_zp is not None and self.w2_zp is not None
            and getattr(self, "w1_bias", None) is None
            and getattr(self, "w2_bias", None) is None
            and self.input_dtype is None
            and getattr(self, "gemm1_clamp_limit", None) is None
            # MarlinExperts.__init__ normalizes unset alpha/beta to the
            # identity values 1.0/0.0 -- never None
            and getattr(self, "gemm1_alpha", None) in (None, 1.0)
            and getattr(self, "gemm1_beta", None) in (None, 0.0)
            and self.is_k_full)


_dbg_seen = set()


def _debug_once(n, hidden_states):
    """One-shot per (n-is-None, M) shape: what the reader sees. stderr so it
    survives whatever stdout redirection the workers get."""
    import sys
    key = (n is None, int(hidden_states.size(0)))
    if key in _dbg_seen or len(_dbg_seen) > 16:
        return
    _dbg_seen.add(key)
    try:
        from vllm.forward_context import get_forward_context
        md = get_forward_context().attn_metadata
        desc = (type(md).__name__ if not isinstance(md, dict) else
                sorted({type(v).__name__ for v in md.values()}))
    except Exception as e:
        desc = f"ctx-err:{e!r}"
    print(f"[glm52-split-dbg] n_dec={n} M={hidden_states.size(0)} "
          f"meta={desc}", file=sys.stderr, flush=True)


def _shadow_compare(self, output, hidden_states, w1, w2, topk_weights,
                    topk_ids, n_dec):
    """Compare the gemv decode-slice output against the stock output
    already in `output[:n_dec]`. Log divergence; dump first bad call."""
    global _shadow_dumped, _shadow_calls
    _shadow_calls += 1
    scratch = torch.empty_like(output[:n_dec])
    moe_gemv_forward(hidden_states[:n_dec], w1, self.w1_scale, self.w1_zp,
                     w2, self.w2_scale, self.w2_zp,
                     topk_weights[:n_dec], topk_ids[:n_dec], scratch)
    ref = output[:n_dec].float()
    d = (scratch.float() - ref).norm() / ref.norm().clamp(min=1e-9)
    rel = d.item()   # sync: fine, shadow is eager-only diagnostic
    M = hidden_states.size(0)
    if rel > 2e-2 and not _shadow_dumped:
        _shadow_dumped = True
        path = os.path.join(_SHADOW, f"shadow_dump_rank_call{_shadow_calls}.pt")
        torch.save({
            "x": hidden_states[:n_dec].cpu(),
            "topk_weights": topk_weights[:n_dec].cpu(),
            "topk_ids": topk_ids[:n_dec].cpu(),
            "ref_out": output[:n_dec].cpu(),
            "gemv_out": scratch.cpu(),
            "n_dec": n_dec, "M": M,
            "w13_shape": tuple(w1.shape), "w2_shape": tuple(w2.shape),
        }, path)
        print(f"[glm52-shadow] DUMPED {path} rel={rel:.3e} "
              f"n_dec={n_dec} M={M}", flush=True)
    # offline-measured whole-layer gemv-vs-marlin delta is ~6e-3, so only
    # log clear outliers plus a periodic heartbeat
    if rel > 1.5e-2 or _shadow_calls % 500 == 1:
        print(f"[glm52-shadow] call={_shadow_calls} rel={rel:.3e} "
              f"n_dec={n_dec} M={M}", flush=True)


def install_split_moe(cls):
    """Wrap MarlinExperts.apply with the decode/prefill dispatch."""
    if getattr(cls, "_glm52_split_installed", False):
        return
    orig = cls.apply

    def apply(self, output, hidden_states, w1, w2, topk_weights, topk_ids,
              activation, global_num_experts, expert_map, a1q_scale,
              a2_scale, workspace13, workspace2, expert_tokens_meta,
              apply_router_weight_on_input):
        if "entry" not in _dbg_seen:
            _dbg_seen.add("entry")
            import sys
            conds = {
                "expert_map": expert_map is None,
                "lora": self._lora_context is None,
                "arwoi": not apply_router_weight_on_input,
                "zp": self.w1_zp is not None and self.w2_zp is not None,
                "bias": (getattr(self, "w1_bias", None) is None
                         and getattr(self, "w2_bias", None) is None),
                "in_dt": self.input_dtype is None,
                "clamp": getattr(self, "gemm1_clamp_limit", None) is None,
                "alpha": getattr(self, "gemm1_alpha", None) is None,
                "beta": getattr(self, "gemm1_beta", None) is None,
                "k_full": bool(self.is_k_full),
            }
            print(f"[glm52-split-dbg] apply ENTRY: enabled={_ENABLED} "
                  f"shadow={bool(_SHADOW)} act={getattr(activation, 'name', '?')} "
                  f"a1q={a1q_scale is not None} conds={conds} "
                  f"M={hidden_states.size(0)}",
                  file=sys.stderr, flush=True)
        n_dec = 0
        if ((_ENABLED or _SHADOW) and a1q_scale is None
                and getattr(activation, "name", "") == "SILU"
                and _splittable(self, expert_map,
                                apply_router_weight_on_input)):
            n = num_decode_tokens_in_batch()
            if n:
                n_dec = min(n, hidden_states.size(0))
            _debug_once(n, hidden_states)
        if _SHADOW:
            # reference: stock path for the whole batch
            r = orig(self, output, hidden_states, w1, w2, topk_weights,
                     topk_ids, activation, global_num_experts, expert_map,
                     a1q_scale, a2_scale, workspace13, workspace2,
                     expert_tokens_meta, apply_router_weight_on_input)
            if n_dec > 0:
                _shadow_compare(self, output, hidden_states, w1, w2,
                                topk_weights, topk_ids, n_dec)
            return r
        if n_dec <= 0:
            return orig(self, output, hidden_states, w1, w2, topk_weights,
                        topk_ids, activation, global_num_experts,
                        expert_map, a1q_scale, a2_scale, workspace13,
                        workspace2, expert_tokens_meta,
                        apply_router_weight_on_input)
        moe_gemv_forward(hidden_states[:n_dec], w1, self.w1_scale,
                         self.w1_zp, w2, self.w2_scale, self.w2_zp,
                         topk_weights[:n_dec], topk_ids[:n_dec],
                         output[:n_dec])
        if n_dec == hidden_states.size(0):
            return None
        return orig(self, output[n_dec:], hidden_states[n_dec:], w1, w2,
                    topk_weights[n_dec:], topk_ids[n_dec:], activation,
                    global_num_experts, expert_map, a1q_scale, a2_scale,
                    workspace13, workspace2, expert_tokens_meta,
                    apply_router_weight_on_input)

    cls.apply = apply
    cls._glm52_split_installed = True
    import sys
    print(f"[glm52-split-dbg] installed on {cls.__name__} "
          f"(enabled={_ENABLED} shadow={bool(_SHADOW)})",
          file=sys.stderr, flush=True)
