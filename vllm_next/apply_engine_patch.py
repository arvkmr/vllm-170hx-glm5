#!/usr/bin/env python3
"""Install the local source patches still required by the pinned vLLM fork.

The Morrowmake fork already carries the Ampere backend, DFlash2-over-PP,
deterministic sparse top-k/MoE, 64-bit KV addressing, and PP scheduling work.
It intentionally serves its reference model with a BF16 KV cache, however,
whereas this deployment needs the packed 656-byte ``fp8_ds_mla`` layout to
retain the existing million-token capacity. This fail-closed installer ports
the validated byte-decoding reader from ``glm52_mla_fp8.py``.

Its DFlash-over-PP relay is also only wired into the GLM-5.3-Flash
(``glm5next``) and a few dense models. The legacy 78-layer GLM-5.3 target
loads as ``DeepseekV32ForCausalLM``, whose backbone neither relays aux hidden
states between stages nor declares ``supports_aux_hidden_states_over_pp``, so
the engine refuses DFlash with PP=10. The second patch adopts the fork's own
``EagleModelMixin`` layout there, exactly as ``llama.py`` does.

Finally, the fork's Ampere work covers only glm5next; the DSA path this target
uses emits native e4m3 in its fused norm/RoPE and Q kernels and scores its
indexer with DeepGEMM (SM90+). Two patches port those to sm_80 with the fork's
own software-e4m3 helpers and Triton MQA-logits kernels.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import shutil
import subprocess
from pathlib import Path

EXPECTED_COMMIT = "378c37b0098a41a5cd25b3bf8b56d158e33a6cbf"
MARKER = "local-cmp170hx-fp8-ds-mla"
AUX_PP_MARKER = "local-cmp170hx-dsv32-aux-pp"
SM80_FP8_MARKER = "local-cmp170hx-dsv32-sm80-fp8"
SM80_INDEXER_MARKER = "local-cmp170hx-dsv32-sm80-indexer"
CANON_MARKER = "local-cmp170hx-canon-topk"
LMHEAD_MARKER = "local-cmp170hx-lmhead-quant"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        return text
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one source anchor, found {count}")
    return text.replace(old, new)


def replace_n(text: str, old: str, new: str, n: int, label: str) -> str:
    """Replace every occurrence of ``old``; require exactly ``n`` of them."""
    if new in text and old not in text:
        return text
    count = text.count(old)
    if count != n:
        raise RuntimeError(f"{label}: expected {n} source anchors, found {count}")
    return text.replace(old, new)


def patch_dsv32_sm80_fp8(kernels: str) -> str:
    """Make the DSA fused norm/RoPE and Q kernels compile on sm_80.

    Ampere Triton cannot name ``tl.float8e4nv``, neither as a cast target nor
    as a pointer element type. Below SM89 the kernels instead compute the e4m3
    bit pattern in software (the fork's ``f32_to_e4m3_bits``: round-to-nearest
    -even with saturation, the semantics of the hardware ``cvt.rn.satfinite``)
    and store it through a uint8 view of the same memory. Tensors handed back
    to callers keep their float8 dtype. Hopper+ and ROCm are unchanged.
    """
    kernels = replace_once(
        kernels,
        "_FP8_MAX = 224.0 if _USE_FNUZ else 448.0\n",
        "_FP8_MAX = 224.0 if _USE_FNUZ else 448.0\n"
        f"# {SM80_FP8_MARKER}: Triton names e4m3 natively only on SM89+; below\n"
        "# that the kernels write software-rounded e4m3 bits through byte views.\n"
        "_FP8_NATIVE = (\n"
        "    _USE_FNUZ\n"
        "    or not current_platform.is_cuda()\n"
        "    or current_platform.has_device_capability(89)\n"
        ")\n"
        "_FP8_NATIVE_TL = tl.constexpr(_FP8_NATIVE)\n"
        "\n"
        "\n"
        "def _as_fp8_bytes(t: torch.Tensor | None) -> torch.Tensor | None:\n"
        "    if _FP8_NATIVE or t is None:\n"
        "        return t\n"
        "    if t.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):\n"
        "        return t.view(torch.uint8)\n"
        "    return t\n",
        "sm80 fp8 capability switch",
    )
    kernels = replace_once(
        kernels,
        "from vllm.utils.torch_utils import is_quantized_kv_cache\n",
        "from vllm.utils.torch_utils import is_quantized_kv_cache\n"
        "from vllm.v1.attention.ops.triton_e4m3 import f32_to_e4m3_bits\n",
        "import software e4m3",
    )
    kernels = replace_once(
        kernels,
        "@triton.jit\ndef _fp8_ue8m0_quantize(",
        "@triton.jit\n"
        "def _to_fp8(x, USE_FNUZ: tl.constexpr):\n"
        "    if _FP8_NATIVE_TL:\n"
        "        y = x.to(tl.float8e4b8 if USE_FNUZ else tl.float8e4nv)\n"
        "    else:\n"
        "        y = f32_to_e4m3_bits(x.to(tl.float32))\n"
        "    return y\n"
        "\n\n"
        "@triton.jit\ndef _fp8_ue8m0_quantize(",
        "software-or-native fp8 cast helper",
    )
    kernels = replace_once(
        kernels,
        "    fp8_vals = tl.div_rn(vals, scale).to(fp8_dtype)\n",
        "    fp8_vals = _to_fp8(tl.div_rn(vals, scale), USE_FNUZ)\n",
        "indexer quantize cast",
    )
    for old, new, n in (
        (
            "tl.reshape((kv_2d / tile_scale).to(fp8_dtype), (KV_DIM,))",
            "tl.reshape(_to_fp8(kv_2d / tile_scale, USE_FNUZ), (KV_DIM,))",
            1,
        ),
        (
            "(kv_c.to(tl.float32) / scale).to(fp8_dtype)",
            "_to_fp8(kv_c.to(tl.float32) / scale, USE_FNUZ)",
            1,
        ),
        ("(r1 / scale).to(fp8_dtype)", "_to_fp8(r1 / scale, USE_FNUZ)", 2),
        ("(r2 / scale).to(fp8_dtype)", "_to_fp8(r2 / scale, USE_FNUZ)", 2),
        ("(ql_nope / scale).to(fp8_dtype)", "_to_fp8(ql_nope / scale, USE_FNUZ)", 1),
        # Kernel-launch pointer arguments: byte views below SM89.
        (
            "        indexer_k_cache,\n        idx_cache_scale_view,\n",
            "        _as_fp8_bytes(indexer_k_cache),\n        idx_cache_scale_view,\n",
            1,
        ),
        (
            "        mla_kv_cache,\n        mla_block_stride,\n",
            "        _as_fp8_bytes(mla_kv_cache),\n        mla_block_stride,\n",
            1,
        ),
        (
            "        index_q_fp8,\n        index_q_fp8.stride(0),\n",
            "        _as_fp8_bytes(index_q_fp8),\n        index_q_fp8.stride(0),\n",
            1,
        ),
        (
            "        mqa_q_fp8,\n        mqa_q_fp8.stride(0),\n",
            "        _as_fp8_bytes(mqa_q_fp8),\n        mqa_q_fp8.stride(0),\n",
            1,
        ),
        (
            "        q_pe_out,\n        q_pe_out.stride(0),\n",
            "        _as_fp8_bytes(q_pe_out),\n        q_pe_out.stride(0),\n",
            1,
        ),
    ):
        kernels = replace_n(kernels, old, new, n, f"sm80 fp8: {old.strip()}")
    # Below SM89 the kernels must mutate real uint8 buffers, never a uint8 view
    # of a float8 tensor made just for the launch: torch.compile tracks a
    # Triton kernel's writes through its argument tensors, and a dtype-view
    # temporary hid the write to the float8 base. Under compiled/graph mode
    # that let a reader see an unwritten buffer (sporadic token-0 outputs,
    # 2026-09-27). Caches stay in their uint8 allocation; outputs are
    # allocated as uint8 and viewed as float8 only after the kernel ran.
    for old, new, n in (
        (
            "        if indexer_k_cache.dtype == torch.uint8:\n"
            "            indexer_k_cache = indexer_k_cache.view(_FP8_DTYPE)\n",
            "        if indexer_k_cache.dtype == torch.uint8 and _FP8_NATIVE:\n"
            "            indexer_k_cache = indexer_k_cache.view(_FP8_DTYPE)\n",
            1,
        ),
        (
            "            mla_kv_cache = u8_cache.view(_FP8_DTYPE)\n",
            "            mla_kv_cache = u8_cache.view(_FP8_DTYPE) if _FP8_NATIVE else u8_cache\n",
            1,
        ),
        (
            "            if mla_cache_fp8 and mla_kv_cache.dtype == torch.uint8:\n",
            "            if mla_cache_fp8 and mla_kv_cache.dtype == torch.uint8 and _FP8_NATIVE:\n",
            1,
        ),
        (
            "            ql_nope.shape[2] + q_pe.shape[2],\n"
            "            dtype=_FP8_DTYPE,\n",
            "            ql_nope.shape[2] + q_pe.shape[2],\n"
            "            dtype=_FP8_DTYPE if _FP8_NATIVE else torch.uint8,\n",
            1,
        ),
        (
            "    index_q_fp8 = torch.empty_like(index_q, dtype=_FP8_DTYPE)\n",
            "    index_q_fp8 = torch.empty_like(\n"
            "        index_q, dtype=_FP8_DTYPE if _FP8_NATIVE else torch.uint8\n"
            "    )\n",
            1,
        ),
        (
            "        num_warps=1,\n    )\n    return index_q_fp8, index_weights_out, mqa_q\n",
            "        num_warps=1,\n    )\n"
            "    if not _FP8_NATIVE:\n"
            "        # Written as bytes above; callers get the float8 dtype.\n"
            "        index_q_fp8 = index_q_fp8.view(_FP8_DTYPE)\n"
            "        if quantize_mqa:\n"
            "            mqa_q = mqa_q.view(_FP8_DTYPE)\n"
            "    return index_q_fp8, index_weights_out, mqa_q\n",
            1,
        ),
    ):
        kernels = replace_n(kernels, old, new, n, f"sm80 fp8 buffers: {old.strip()[:50]}")
    if ".to(fp8_dtype)" in kernels:
        raise RuntimeError("sm80 fp8: an unported native fp8 cast remains")
    return kernels


def patch_indexer_sm80(indexer: str) -> str:
    """Give the generic DSA sparse indexer the fork's Ampere route.

    Without SM90+ the fork's DeepGEMM shim has no logits kernels, so the
    standard indexer (used by the 78-layer GLM-5.3 target) cannot score. The
    GLM-5.3-Flash indexer already falls back to the fork's validated Triton
    fp8 MQA-logits kernels; this applies the same fallback here, packs the
    padded decode Q as bytes (Ampere Triton cannot load an fp8 pointer) with
    the weights packed to the same rows, and gives prefill top-k the same
    canonical/tie-fix/sort handling the GLM-5.3-Flash path applies.
    """
    indexer = replace_once(
        indexer,
        "from vllm.utils.deep_gemm import (\n"
        "    fp8_fp4_mqa_logits,\n"
        "    fp8_fp4_paged_mqa_logits,\n"
        "    has_deep_gemm,\n"
        ")\n",
        "from vllm.utils.deep_gemm import (\n"
        "    fp8_fp4_mqa_logits,\n"
        "    fp8_fp4_paged_mqa_logits,\n"
        "    has_deep_gemm,\n"
        "    is_deep_gemm_supported,\n"
        ")\n"
        f"# {SM80_INDEXER_MARKER}: Ampere scores with the fork's Triton kernels.\n"
        "from vllm.model_executor.layers.indexer_topk import (\n"
        "    canonical_topk,\n"
        "    repair_topk_ties_,\n"
        "    sort_selected_topk_,\n"
        "    tiefix_topk_,\n"
        "    use_canonical_topk,\n"
        "    use_sorted_topk,\n"
        "    use_tiefix_topk,\n"
        "    use_topk_tie_repair,\n"
        ")\n"
        "from vllm.v1.attention.ops.triton_mqa_logits import (\n"
        "    fp8_mqa_logits as triton_fp8_mqa_logits,\n"
        "    fp8_paged_mqa_logits as triton_fp8_paged_mqa_logits,\n"
        ")\n"
        "import os as _os\n"
        "from vllm._glm52_topk_canon import canon_topk_indices_prefill\n"
        "_GLM53_TOPK_CANON = _os.environ.get(\"GLM53_TOPK_CANON\", \"0\") == \"1\"\n",
        "indexer sm80 imports",
    )
    indexer = replace_once(
        indexer,
        "                else:\n"
        "                    logits = fp8_fp4_mqa_logits(\n"
        "                        (q_slice_cast, q_scale_slice),\n",
        "                elif current_platform.is_cuda() and not is_deep_gemm_supported():\n"
        "                    # Ampere: Triton kernel dequantizing the e4m3 bytes\n"
        "                    # in-kernel (bf16 MMA, fp32 accumulate).\n"
        "                    assert not use_fp4_cache, \"MXFP4 indexer needs DeepGEMM\"\n"
        "                    logits = triton_fp8_mqa_logits(\n"
        "                        q_slice_cast,\n"
        "                        (k_quant_cast, k_scale_cast),\n"
        "                        weights[chunk.token_start : chunk.token_end],\n"
        "                        cu_seqlen_ks,\n"
        "                        cu_seqlen_ke,\n"
        "                    )\n"
        "                else:\n"
        "                    logits = fp8_fp4_mqa_logits(\n"
        "                        (q_slice_cast, q_scale_slice),\n",
        "indexer prefill logits fallback",
    )
    indexer = replace_once(
        indexer,
        "                ops.top_k_per_row_prefill(\n"
        "                    logits,\n"
        "                    cu_seqlen_ks,\n"
        "                    cu_seqlen_ke,\n"
        "                    topk_indices,\n"
        "                    num_rows,\n"
        "                    logits.stride(0),\n"
        "                    logits.stride(1),\n"
        "                    topk_tokens,\n"
        "                )\n",
        "                if _GLM53_TOPK_CANON:\n"
        f"                    # {CANON_MARKER}: native value-exact selection, then\n"
        "                    # v0.26's Triton canonical rebuild (relative indices,\n"
        "                    # identity short rows), written out of place.\n"
        "                    ops.top_k_per_row_prefill(\n"
        "                        logits,\n"
        "                        cu_seqlen_ks,\n"
        "                        cu_seqlen_ke,\n"
        "                        topk_indices,\n"
        "                        num_rows,\n"
        "                        logits.stride(0),\n"
        "                        logits.stride(1),\n"
        "                        topk_tokens,\n"
        "                    )\n"
        "                    canon_topk_indices_prefill(\n"
        "                        logits, topk_indices, cu_seqlen_ks, cu_seqlen_ke, topk_tokens\n"
        "                    )\n"
        "                elif use_canonical_topk():\n"
        "                    # The prefill selection decides what each token attends\n"
        "                    # while the KV cache is built; a tie lottery here makes\n"
        "                    # the cache itself irreproducible (as glm5next).\n"
        "                    canonical_topk(\n"
        "                        logits,\n"
        "                        topk_tokens,\n"
        "                        row_starts=cu_seqlen_ks,\n"
        "                        row_ends=cu_seqlen_ke,\n"
        "                        out=topk_indices,\n"
        "                        relative=True,\n"
        "                        identity_when_short=True,\n"
        "                    )\n"
        "                else:\n"
        "                    ops.top_k_per_row_prefill(\n"
        "                        logits,\n"
        "                        cu_seqlen_ks,\n"
        "                        cu_seqlen_ke,\n"
        "                        topk_indices,\n"
        "                        num_rows,\n"
        "                        logits.stride(0),\n"
        "                        logits.stride(1),\n"
        "                        topk_tokens,\n"
        "                    )\n"
        "                    if use_tiefix_topk():\n"
        "                        tiefix_topk_(\n"
        "                            logits,\n"
        "                            topk_indices,\n"
        "                            row_starts=cu_seqlen_ks,\n"
        "                            row_ends=cu_seqlen_ke,\n"
        "                            relative=True,\n"
        "                            sort=use_sorted_topk(),\n"
        "                        )\n"
        "                    elif use_topk_tie_repair():\n"
        "                        repair_topk_ties_(\n"
        "                            topk_indices,\n"
        "                            logits,\n"
        "                            topk_tokens,\n"
        "                            row_ends=cu_seqlen_ke,\n"
        "                            row_starts=cu_seqlen_ks,\n"
        "                            relative=True,\n"
        "                        )\n"
        "                    elif use_sorted_topk():\n"
        "                        sort_selected_topk_(topk_indices)\n",
        "indexer prefill deterministic top-k",
    )
    indexer = replace_once(
        indexer,
        "            else:\n"
        "                padded_q_quant_decode_tokens = pack_seq_triton(\n"
        "                    q_quant[:num_decode_tokens], decode_lens\n"
        "                )\n"
        "                padded_q_scale = None\n",
        "            elif current_platform.is_cuda() and not is_deep_gemm_supported():\n"
        "                # Ampere Triton cannot load an fp8 pointer: pack the e4m3\n"
        "                # bytes. Zero pads score zero and context_lens masks them.\n"
        "                padded_q_quant_decode_tokens = pack_seq_triton(\n"
        "                    q_quant[:num_decode_tokens].view(torch.uint8),\n"
        "                    decode_lens,\n"
        "                    pad_value=0,\n"
        "                )\n"
        "                padded_q_scale = None\n"
        "            else:\n"
        "                padded_q_quant_decode_tokens = pack_seq_triton(\n"
        "                    q_quant[:num_decode_tokens], decode_lens\n"
        "                )\n"
        "                padded_q_scale = None\n",
        "indexer decode byte packing",
    )
    indexer = replace_once(
        indexer,
        "        else:\n"
        "            logits = fp8_fp4_paged_mqa_logits(\n"
        "                (padded_q_quant_cast, padded_q_scale),\n",
        "        elif current_platform.is_cuda() and not is_deep_gemm_supported():\n"
        "            assert padded_q_scale is None, \"MXFP4 indexer needs DeepGEMM\"\n"
        "            # One weights row per padded Q row: pack them the same way.\n"
        "            if needs_padded_path and num_decode_tokens > 0:\n"
        "                decode_weights = pack_seq_triton(\n"
        "                    weights[:num_decode_tokens], decode_lens, pad_value=0\n"
        "                ).reshape(-1, weights.shape[-1])\n"
        "            else:\n"
        "                decode_weights = weights[:num_padded_tokens]\n"
        "            logits = triton_fp8_paged_mqa_logits(\n"
        "                padded_q_quant_cast,\n"
        "                kv_cache,\n"
        "                decode_weights,\n"
        "                seq_lens,\n"
        "                decode_metadata.block_table,\n"
        "                max_model_len,\n"
        "                max_seq_len=attn_metadata_narrowed.max_seq_len,\n"
        "            )\n"
        "        else:\n"
        "            logits = fp8_fp4_paged_mqa_logits(\n"
        "                (padded_q_quant_cast, padded_q_scale),\n",
        "indexer decode logits fallback",
    )
    # With graphs on, JIT warmup precompiles the Q packer for the dtype the
    # decode path packs; on Ampere that is the uint8 byte view (pad 0).
    indexer = replace_once(
        indexer,
        "            pack_dtype = torch.uint8 if use_fp4_cache else current_platform.fp8_dtype()\n"
        "            _PACK_SEQ_TRITON_KERNEL.register_warmup(\n"
        "                dtype=pack_dtype,\n"
        "                pad_value=0 if use_fp4_cache else -float(\"inf\"),\n"
        "            )\n",
        "            pack_bytes = use_fp4_cache or (\n"
        "                current_platform.is_cuda() and not is_deep_gemm_supported()\n"
        "            )\n"
        "            pack_dtype = torch.uint8 if pack_bytes else current_platform.fp8_dtype()\n"
        "            _PACK_SEQ_TRITON_KERNEL.register_warmup(\n"
        "                dtype=pack_dtype,\n"
        "                pad_value=0 if pack_bytes else -float(\"inf\"),\n"
        "            )\n",
        "indexer Q-pack warmup dtype",
    )
    return indexer


def patch_topk_canon_decode(topk: str) -> str:
    """Canonical decode top-k via v0.26's Triton rebuild (GLM53_TOPK_CANON=1).

    The fork's canonical path is a torch sort over the full max_model_len-wide
    logits row (~48 ms/step at 950K); its TIEFIX/SORTED post-processing
    corrupted decode on this target. v0.26's kernel rebuilds the canonical
    set from the native value-exact selection with one early-exit row scan.
    """
    topk = replace_once(
        topk,
        "@functools.cache\ndef get_indexer_topk(backend: str)",
        f"# {CANON_MARKER}\n"
        "_GLM53_TOPK_CANON = os.environ.get(\"GLM53_TOPK_CANON\", \"0\") == \"1\"\n"
        "\n\n"
        "@functools.cache\ndef get_indexer_topk(backend: str)",
        "decode canon switch",
    )
    if "\nimport os\n" not in topk:
        topk = replace_once(topk, "import functools\n", "import functools\nimport os\n", "import os")
    topk = replace_once(
        topk,
        "        self._run_backend(\n"
        "            logits, seq_lens, next_n, topk_indices, topk_tokens, max_seq_len\n"
        "        )\n",
        "        self._run_backend(\n"
        "            logits, seq_lens, next_n, topk_indices, topk_tokens, max_seq_len\n"
        "        )\n"
        "        if _GLM53_TOPK_CANON:\n"
        "            from vllm._glm52_topk_canon import canon_topk_indices\n"
        "\n"
        "            canon_topk_indices(\n"
        "                logits,\n"
        "                topk_indices,\n"
        "                self._row_ends(seq_lens, next_n, logits.shape[0]),\n"
        "            )\n"
        "            return\n",
        "decode canonical rebuild",
    )
    return topk


DSA_DRAFT_KV_MARKER = "local-cmp170hx-dsa-dflash-kv"

_DSA_DRAFT_KV_FUNCS = '''

# {marker}: DSA target (MLA + indexer, packed fp8_ds_mla 656-byte rows)
# plus a DFlash drafter's sliding-window layers. No mamba, so the glm5next
# layout does not apply, and its exact-fit geometry is impossible here: a
# 656-byte row gives MLA pages of 41 * 1024 * k bytes, which no 64-multiple
# BF16 drafter block can match. Instead each drafter layer co-owns one MLA
# tensor at disjoint block ids with its (smaller) page padded to the MLA
# page, so the pool's per-block cost is unchanged and the drafter's sliding
# window bounds it to a few block ids per request. The drafter block is used
# as the kernel block (a multiple of 16, never split), which keeps the
# padded strided view valid. Without this the generic fallback allocates
# every drafter layer full-length (6 x 4 KiB/token), capping KV at ~0.96M.
def _dsa_dflash_split(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> tuple[dict[str, KVCacheSpec], dict[str, KVCacheSpec]] | None:
    draft_specs = {{
        name: spec
        for name, spec in kv_cache_spec.items()
        if type(spec) is SlidingWindowSpec
    }}
    attn_specs = {{
        name: spec
        for name, spec in kv_cache_spec.items()
        if type(spec) is not SlidingWindowSpec
    }}
    if not draft_specs or not attn_specs:
        return None
    if not all(
        type(spec) is MLAAttentionSpec and spec.page_size_padded is None
        for spec in attn_specs.values()
    ):
        return None
    return attn_specs, draft_specs


def _get_kv_cache_groups_dsa_dflash(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec] | None:
    split = _dsa_dflash_split(kv_cache_spec)
    if split is None:
        return None
    attn_specs, draft_specs = split
    uniform_spec = UniformTypeKVCacheSpecs.from_specs(attn_specs)
    if uniform_spec is None:
        return None
    mla_page = max(spec.page_size_bytes for spec in attn_specs.values())
    mla_names = [
        name for name, spec in attn_specs.items() if spec.page_size_bytes == mla_page
    ]
    mla_block = attn_specs[mla_names[0]].block_size
    any_draft = next(iter(draft_specs.values()))
    if not all(spec == any_draft for spec in draft_specs.values()):
        logger.warning("DSA DFlash KV: drafter layers differ; shared layout not used")
        return None
    bytes_per_token = any_draft.page_size_bytes // any_draft.block_size
    draft_block = (mla_page // bytes_per_token) // 16 * 16
    while draft_block >= 16 and mla_block % draft_block:
        draft_block -= 16
    if draft_block < 16:
        logger.warning(
            "DSA DFlash KV: a %d B/token drafter block of >= 16 tokens does not "
            "fit the %d B MLA page (block %d); use --block-size 128. The "
            "drafter falls back to full-length KV (capacity drops).",
            bytes_per_token,
            mla_page,
            mla_block,
        )
        return None
    slots = _glm5_next_draft_mla_slots(vllm_config, mla_names)
    if slots == 0:
        logger.warning("DSA DFlash KV: no MLA layer on the drafter's stage")
        return None
    names = list(draft_specs)
    num_groups = cdiv(len(names), slots)
    per_group = cdiv(len(names), num_groups)
    chunks = [names[i : i + per_group] for i in range(0, len(names), per_group)]
    draft_groups: list[KVCacheGroupSpec] = []
    for chunk in chunks:
        fitted: dict[str, KVCacheSpec] = {{
            name: replace(
                draft_specs[name], block_size=draft_block, page_size_padded=mla_page
            )
            for name in chunk
        }}
        assert all(spec.page_size_bytes == mla_page for spec in fitted.values())
        draft_uniform = UniformTypeKVCacheSpecs.from_specs(fitted)
        if draft_uniform is None:
            return None
        draft_groups.append(KVCacheGroupSpec(list(fitted), draft_uniform))
    spec_config = vllm_config.speculative_config
    if spec_config is not None and spec_config.use_eagle_block_drop():
        for group in draft_groups:
            group.is_eagle_group = True
    logger.info(
        "DSA DFlash KV: %d drafter layers ride the MLA tensors in %d group(s) "
        "of %s (drafter block %d, %d of %d B used per padded page, %d MLA "
        "tensor(s) on the drafter's stage, no extra bytes per block)",
        len(names),
        len(chunks),
        [len(c) for c in chunks],
        draft_block,
        draft_block * bytes_per_token,
        mla_page,
        slots,
    )
    return [KVCacheGroupSpec(list(attn_specs), uniform_spec)] + draft_groups


def _dsa_dflash_tensor_layout(
    kv_cache_groups: list[KVCacheGroupSpec],
) -> tuple[KVCacheGroupSpec, list[KVCacheGroupSpec], list[str], list[str], int] | None:
    """Recognize the DSA+DFlash grouping above, after optional PP projection."""
    if len(kv_cache_groups) < 2 or not all(
        isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
        for group in kv_cache_groups
    ):
        return None
    attn_group = kv_cache_groups[0]
    attn_inner = cast(UniformTypeKVCacheSpecs, attn_group.kv_cache_spec).kv_cache_specs
    if not all(
        type(spec) is MLAAttentionSpec and spec.page_size_padded is None
        for spec in attn_inner.values()
    ):
        return None
    mla_page = max(spec.page_size_bytes for spec in attn_inner.values())
    mla_names = [
        name
        for name in attn_group.layer_names
        if attn_inner[name].page_size_bytes == mla_page
    ]
    idx_names = [name for name in attn_group.layer_names if name not in mla_names]
    draft_groups = kv_cache_groups[1:]
    for group in draft_groups:
        inner = cast(UniformTypeKVCacheSpecs, group.kv_cache_spec).kv_cache_specs
        if not inner or not all(
            type(spec) is SlidingWindowSpec and spec.page_size_padded == mla_page
            for spec in inner.values()
        ):
            return None
        if len(group.layer_names) > len(mla_names):
            return None
    return attn_group, draft_groups, mla_names, idx_names, mla_page
'''


def patch_kv_cache_dsa_dflash(kvu: str) -> str:
    """Window-bounded DFlash drafter KV for the DSA target (see the helper doc)."""
    kvu = replace_once(
        kvu,
        "\n\ndef _glm5_next_tensor_layout(\n",
        _DSA_DRAFT_KV_FUNCS.format(marker=DSA_DRAFT_KV_MARKER)
        + "\n\ndef _glm5_next_tensor_layout(\n",
        "dsa dflash kv helpers",
    )
    kvu = replace_once(
        kvu,
        "    elif glm5_groups := _get_kv_cache_groups_glm5_next(vllm_config, kv_cache_spec):\n"
        "        _warn_if_unannotated_eagle_mamba(vllm_config, glm5_groups)\n"
        "        return glm5_groups\n",
        "    elif glm5_groups := _get_kv_cache_groups_glm5_next(vllm_config, kv_cache_spec):\n"
        "        _warn_if_unannotated_eagle_mamba(vllm_config, glm5_groups)\n"
        "        return glm5_groups\n"
        "    elif dsa_groups := _get_kv_cache_groups_dsa_dflash(vllm_config, kv_cache_spec):\n"
        "        return dsa_groups\n",
        "dsa dflash grouping dispatch",
    )
    kvu = replace_once(
        kvu,
        '    """Return the largest cache group\'s bytes per block."""\n',
        '    """Return the largest cache group\'s bytes per block."""\n'
        "    if (dsa_layout := _dsa_dflash_tensor_layout(kv_cache_groups)) is not None:\n"
        "        # Drafter layers ride the MLA tensors: no extra bytes per block.\n"
        "        attn_group = dsa_layout[0]\n"
        "        return sum(\n"
        "            _get_per_layer_spec(attn_group, name).page_size_bytes\n"
        "            for name in attn_group.layer_names\n"
        "        )\n",
        "dsa dflash bytes per block",
    )
    kvu = replace_once(
        kvu,
        "    layout = vllm_config.cache_config.get_resolved_kv_cache_layout()\n"
        "    validate_kv_cache_layout(layout, kv_cache_groups)\n",
        "    if (dsa_layout := _dsa_dflash_tensor_layout(kv_cache_groups)) is not None:\n"
        "        attn_group, draft_groups, mla_names, idx_names, mla_page = dsa_layout\n"
        "        attn_specs = cast(\n"
        "            UniformTypeKVCacheSpecs, attn_group.kv_cache_spec\n"
        "        ).kv_cache_specs\n"
        "        bytes_per_block = _get_kv_cache_bytes_per_block(kv_cache_groups)\n"
        "        num_blocks = may_override_num_blocks(\n"
        "            vllm_config, available_memory // bytes_per_block\n"
        "        )\n"
        "        size = bytes_per_block * num_blocks\n"
        "        dsa_tensors: list[KVCacheTensor] = []\n"
        "\n"
        "        def _add(layer_name: str, page: int, offset: int) -> None:\n"
        "            dsa_tensors.append(\n"
        "                KVCacheTensor(\n"
        "                    size=size,\n"
        "                    layers=[layer_name],\n"
        "                    layer_stride=page * num_blocks,\n"
        "                    block_stride=page,\n"
        "                    offset=offset,\n"
        "                )\n"
        "            )\n"
        "\n"
        "        offset = 0\n"
        "        for index, mla_name in enumerate(mla_names):\n"
        "            _add(mla_name, mla_page, offset)\n"
        "            for draft_group in draft_groups:\n"
        "                if index < len(draft_group.layer_names):\n"
        "                    # Drafter layer i co-owns MLA tensor i at disjoint\n"
        "                    # block ids (its page is padded to the MLA page).\n"
        "                    _add(draft_group.layer_names[index], mla_page, offset)\n"
        "            offset += mla_page * num_blocks\n"
        "        for idx_name in idx_names:\n"
        "            page = attn_specs[idx_name].page_size_bytes\n"
        "            _add(idx_name, page, offset)\n"
        "            offset += page * num_blocks\n"
        "        return KVCacheConfig(\n"
        "            num_blocks=num_blocks,\n"
        "            kv_cache_tensors=dsa_tensors,\n"
        "            kv_cache_groups=kv_cache_groups,\n"
        "            prefix_cache_retention_interval=(\n"
        "                vllm_config.cache_config.prefix_cache_retention_interval\n"
        "            ),\n"
        "        )\n"
        "\n"
        "    layout = vllm_config.cache_config.get_resolved_kv_cache_layout()\n"
        "    validate_kv_cache_layout(layout, kv_cache_groups)\n",
        "dsa dflash tensor layout",
    )
    return kvu


def patch_runner_lmhead_quant(runner: str) -> str:
    """Quantize the lm_head after target + drafter load (GLM52_LMHEAD_BITS)."""
    return replace_once(
        runner,
        "            if self.draft_tail is not None:\n"
        "                self.draft_tail.load(self.model, self.dtype)\n"
        "        time_after_load = time.perf_counter()\n",
        "            if self.draft_tail is not None:\n"
        "                self.draft_tail.load(self.model, self.dtype)\n"
        f"            # {LMHEAD_MARKER}: the drafter aliases this lm_head.\n"
        "            from vllm._glm52_lmhead_quant import maybe_quantize_lm_head\n"
        "\n"
        "            maybe_quantize_lm_head(self.model)\n"
        "        time_after_load = time.perf_counter()\n",
        "runner lm_head quantization hook",
    )


MLA_HEAD_BMM_MARKER = "local-cmp170hx-mla-head-bmm"


def patch_mla_head_bmm(mla: str) -> str:
    """Route MLA decode's two absorbed up-projection bmms (W_UK_T, W_UV)
    through the sm_80 per-head kernel in glm52_mla_bmm.py; it falls back to
    torch.bmm itself outside its envelope (> 32 rows, other dtypes/shapes)."""
    mla = replace_once(
        mla,
        "                    mqa_ql_nope = mqa_q_nope.new_empty((B, N, L))\n"
        "                    torch.bmm(mqa_q_nope, W_UK_T, out=mqa_ql_nope.transpose(0, 1))\n",
        "                    mqa_ql_nope = mqa_q_nope.new_empty((B, N, L))\n"
        f"                    # {MLA_HEAD_BMM_MARKER}\n"
        "                    from vllm._glm52_mla_bmm import head_bmm as _glm52_head_bmm\n"
        "\n"
        "                    _glm52_head_bmm(mqa_q_nope, W_UK_T, mqa_ql_nope.transpose(0, 1))\n",
        "MLA decode W_UK_T bmm",
    )
    mla = replace_once(
        mla,
        "            # Multiply + Transpose (N, B, L) x (N, L, V)->(N, B, V)->(B, N, V)\n"
        "            torch.bmm(x, self.W_UV, out=out.transpose(0, 1))\n",
        "            # Multiply + Transpose (N, B, L) x (N, L, V)->(N, B, V)->(B, N, V)\n"
        f"            # {MLA_HEAD_BMM_MARKER}\n"
        "            from vllm._glm52_mla_bmm import head_bmm as _glm52_head_bmm\n"
        "\n"
        "            _glm52_head_bmm(x, self.W_UV, out.transpose(0, 1))\n",
        "MLA decode W_UV bmm",
    )
    # First load registers the permuted *view* as the parameter, so W_UK_T is
    # strided 12288 elements along its 512-wide output and W_UV has a
    # 16384-element K stride; the kernel above wants each head's [K, N] block
    # contiguous. Materialize them once at load (copies of a dequantized
    # temporary that is freed right after, so no memory is added).
    mla = replace_once(
        mla,
        '            replace_parameter(self, "W_UV", W_UV.transpose(0, 1), prefer_copy=True)\n'
        "            # Convert from (L, N, P) to (N, P, L)\n"
        '            replace_parameter(self, "W_UK_T", W_UK.permute(1, 2, 0), prefer_copy=True)\n',
        f"            # {MLA_HEAD_BMM_MARKER}: contiguous per-head blocks.\n"
        "            replace_parameter(\n"
        '                self, "W_UV", W_UV.transpose(0, 1).contiguous(), prefer_copy=True\n'
        "            )\n"
        "            # Convert from (L, N, P) to (N, P, L)\n"
        "            replace_parameter(\n"
        '                self, "W_UK_T", W_UK.permute(1, 2, 0).contiguous(), prefer_copy=True\n'
        "            )\n",
        "MLA contiguous W_UK_T / W_UV",
    )
    return mla


def patch_dsv32_head_bmm(attn: str) -> str:
    """The DSA model's own attention module runs its absorbed up-projections
    itself (it subclasses MLAAttention, so the contiguous weights above apply).
    W_UK_T sits in the torch.compile-traced forward, so it goes through the
    registered custom op; W_UV runs in the eager-break attention region and
    calls the kernel directly. Output layouts are those of the replaced bmms."""
    attn = replace_once(
        attn,
        "from vllm.models.deepseek_v32.common.kernels import fused_norm_rope, fused_q\n",
        "from vllm.models.deepseek_v32.common.kernels import fused_norm_rope, fused_q\n"
        f"# {MLA_HEAD_BMM_MARKER}: also registers torch.ops.vllm.glm52_head_bmm_new.\n"
        "from vllm._glm52_mla_bmm import head_bmm as _glm52_head_bmm\n",
        "DSA attention head bmm import",
    )
    attn = replace_once(
        attn,
        "        ql_nope = torch.bmm(q_nope.transpose(0, 1), self.W_UK_T).transpose(0, 1)\n",
        f"        # {MLA_HEAD_BMM_MARKER}\n"
        "        ql_nope = torch.ops.vllm.glm52_head_bmm_new(\n"
        "            q_nope.transpose(0, 1), self.W_UK_T\n"
        "        ).transpose(0, 1)\n",
        "DSA attention W_UK_T bmm",
    )
    attn = replace_once(
        attn,
        "        torch.bmm(x, self.W_UV, out=out)\n",
        f"        # {MLA_HEAD_BMM_MARKER}\n"
        "        _glm52_head_bmm(x, self.W_UV, out)\n",
        "DSA attention W_UV bmm",
    )
    return attn


THIN_GEMM_DSA_MARKER = "local-cmp170hx-thin-gemm-dsa"


def patch_thin_gemm_dsa(thin: str) -> str:
    """Keep two GLM-5.3 shapes on cuBLAS where the fork's thin-M BF16 GEMM
    loses (measured, graph replay, CMP 170HX): the dense MLP / drafter
    gate_up (24576 x 6144, 0.84-0.96x at M=8-16) and the drafter fc
    (6144 x 36864, 0.73x at M=32). Every other shape of this model wins."""
    return replace_once(
        thin,
        "_CUBLAS_FROM_M: dict[tuple[int, int], int] = {(4096, 8192): 17}\n",
        f"# {THIN_GEMM_DSA_MARKER}: GLM-5.3 shapes where cuBLAS wins.\n"
        "_CUBLAS_FROM_M: dict[tuple[int, int], int] = {\n"
        "    (4096, 8192): 17, (24576, 6144): 1, (6144, 36864): 17,\n"
        "}\n",
        "thin GEMM cuBLAS table",
    )


ROUTE_V2_DSA_MARKER = "local-cmp170hx-route-v2-dsa"


def patch_route_v2_dsa(init: str, warmup: str) -> tuple[str, str]:
    """Let the fork's fused sm_80 MoE router (VLLM_GLM5_DECODE_MOE_ROUTE_V2)
    serve GLM-5.3's 256-expert, hidden-6144 router, not only GLM-5.3-Flash's
    288 x 4096. The kernel and its tiling are shape-generic (E=256 splits as
    128+128, K=6144 fits the split-K tiling); only the host gates pin 288/4096.
    Checked against the production chain (cuBLAS fp32 logits ->
    fused_grouped_topk -> deterministic_moe_align_block_size) on real router
    weights: expert ids and the whole Marlin alignment identical, logits and
    weights differ only by fp32 rounding (<= 3.3e-6 / 9e-8)."""
    init = replace_once(
        init,
        '    bias = getattr(router, "e_score_correction_bias", None)\n'
        "    if bias is None or bias.dtype != torch.float32 or tuple(bias.shape) != (288,):\n"
        "        return False\n"
        '    w = getattr(gate, "weight", None)\n'
        "    if w is None or w.dtype != torch.bfloat16 or tuple(w.shape) != (288, 4096):\n"
        "        return False\n",
        f"    # {ROUTE_V2_DSA_MARKER}: also GLM-5.3's 256 x 6144 router.\n"
        '    w = getattr(gate, "weight", None)\n'
        "    if w is None or w.dtype != torch.bfloat16 or tuple(w.shape) not in (\n"
        "        (288, 4096), (256, 6144)\n"
        "    ):\n"
        "        return False\n"
        "    _route_e, _route_k = (int(d) for d in w.shape)\n"
        '    bias = getattr(router, "e_score_correction_bias", None)\n'
        "    if bias is None or bias.dtype != torch.float32 or tuple(bias.shape) != (_route_e,):\n"
        "        return False\n",
        "route v2 gate: expert/hidden shape",
    )
    init = replace_once(
        init,
        "    if x.dtype != torch.bfloat16 or x.dim() != 2 or x.shape[1] != 4096:\n",
        "    if x.dtype != torch.bfloat16 or x.dim() != 2 or x.shape[1] != _route_k:\n",
        "route v2 gate: activation width",
    )
    warmup = replace_once(
        warmup,
        "    if (num_experts, topk, hidden) != (288, 8, 4096):\n",
        f"    # {ROUTE_V2_DSA_MARKER}: also GLM-5.3's 256 x 6144 router.\n"
        "    if (num_experts, topk, hidden) not in ((288, 8, 4096), (256, 8, 6144)):\n",
        "route v2 warmup shape",
    )
    return init, warmup


def patch_dsv32_aux_over_pp(model: str) -> str:
    """Relay DFlash/EAGLE aux hidden states across PP stages for the DSA model."""
    model = replace_once(
        model,
        "from vllm.models.deepseek_v32.attention import DeepseekV32Attention\n",
        "from vllm.model_executor.models.interfaces import EagleModelMixin\n"
        "from vllm.models.deepseek_v32.attention import DeepseekV32Attention\n",
        "import EagleModelMixin",
    )
    model = replace_once(
        model,
        "class DeepseekV32Model(torch.nn.Module):\n"
        "    fall_back_to_pt_during_load = False\n",
        "class DeepseekV32Model(torch.nn.Module, EagleModelMixin):\n"
        "    fall_back_to_pt_during_load = False\n"
        f"    # {AUX_PP_MARKER}: DFlash aux states are relayed across PP stages\n"
        "    # with the fork's EagleModelMixin slot layout (as llama.py does).\n"
        "    supports_aux_hidden_states_over_pp = True\n",
        "declare aux-over-PP support",
    )
    model = replace_once(
        model,
        "        aux_hidden_states = []\n"
        "        for idx, layer in enumerate(\n"
        "            islice(self.layers, self.start_layer, self.end_layer),\n"
        "            start=self.start_layer,\n"
        "        ):\n"
        "            if idx in self.aux_hidden_state_layers:\n"
        "                aux_hidden_states.append(\n"
        "                    hidden_states if residual is None else hidden_states + residual\n"
        "                )\n"
        "            hidden_states, residual = layer(positions, hidden_states, residual, attn_in)\n"
        "            attn_in = None\n",
        "        # Aux layer L is the stream entering layer L. Use the mixin's PP\n"
        "        # layout: the first rank records its start layer and every rank\n"
        "        # records idx + 1 after layer idx, so a layer on a stage boundary is\n"
        "        # recorded once, upstream, and relayed from there.\n"
        "        remote_aux = self.collect_remote_aux_hidden_states(intermediate_tensors)\n"
        "        aux_hidden_states: list[torch.Tensor] = []\n"
        "        if get_pp_group().is_first_rank:\n"
        "            self._maybe_add_hidden_state(\n"
        "                aux_hidden_states, self.start_layer, hidden_states, residual\n"
        "            )\n"
        "        for idx, layer in enumerate(\n"
        "            islice(self.layers, self.start_layer, self.end_layer),\n"
        "            start=self.start_layer,\n"
        "        ):\n"
        "            hidden_states, residual = layer(positions, hidden_states, residual, attn_in)\n"
        "            attn_in = None\n"
        "            self._maybe_add_hidden_state(\n"
        "                aux_hidden_states, idx + 1, hidden_states, residual\n"
        "            )\n",
        "capture aux states in mixin layout",
    )
    model = replace_once(
        model,
        "            return IntermediateTensors(\n"
        "                {\"hidden_states\": hidden_states, \"residual\": residual}\n"
        "            )\n",
        "            return IntermediateTensors(\n"
        "                {\n"
        "                    \"hidden_states\": hidden_states,\n"
        "                    \"residual\": residual,\n"
        "                    **self.pack_local_aux_hidden_states(aux_hidden_states),\n"
        "                }\n"
        "            )\n",
        "send local aux states downstream",
    )
    model = replace_once(
        model,
        "        if len(aux_hidden_states) > 0:\n"
        "            return hidden_states, aux_hidden_states\n"
        "        return hidden_states\n\n"
        "    def load_weights(",
        "        aux_hidden_states = remote_aux + aux_hidden_states\n"
        "        if len(aux_hidden_states) > 0:\n"
        "            return hidden_states, aux_hidden_states\n"
        "        return hidden_states\n\n"
        "    def load_weights(",
        "prepend upstream aux states on the last rank",
    )
    model = replace_once(
        model,
        "            enable_glm52_low_latency_gemm(self, vllm_config.model_config.dtype)\n",
        "            enable_glm52_low_latency_gemm(self, vllm_config.model_config.dtype)\n\n"
        "    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:\n"
        "        # The inherited DeepseekV2 setter assigns the tuple directly and\n"
        "        # skips the per-rank PP slot layout the aux relay depends on.\n"
        "        self.model._set_aux_hidden_state_layers(tuple(layers))\n",
        "set aux layers through the mixin",
    )
    model = replace_once(
        model,
        "from vllm.sequence import IntermediateTensors\n",
        "from vllm.model_executor.models.utils import spec_decode_needs_target_embed\n"
        "from vllm.sequence import IntermediateTensors\n",
        "import spec_decode_needs_target_embed",
    )
    model = replace_once(
        model,
        "        if get_pp_group().is_first_rank:\n"
        "            self.embed_tokens = make_input_embedding(\n",
        "        # The last stage also holds the table when a DFlash/EAGLE drafter\n"
        "        # runs there: the drafter ships no embedding and aliases the\n"
        "        # target's, which a PPMissingLayer cannot provide (as glm5next).\n"
        "        if get_pp_group().is_first_rank or spec_decode_needs_target_embed(\n"
        "            vllm_config\n"
        "        ):\n"
        "            self.embed_tokens = make_input_embedding(\n",
        "keep the embedding on the drafter stage",
    )
    return model


def git_head(source: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()


def patch_source(source: Path, helper: Path, *, check: bool) -> None:
    if git_head(source) != EXPECTED_COMMIT:
        raise RuntimeError(
            f"engine HEAD must be {EXPECTED_COMMIT}; got {git_head(source)}"
        )

    backend_path = source / "vllm/v1/attention/backends/mla/triton_mla_sparse.py"
    mla_path = source / "vllm/model_executor/layers/attention/mla_attention.py"
    dsv32_path = source / "vllm/models/deepseek_v32/nvidia/model.py"
    kernels_path = source / "vllm/models/deepseek_v32/common/kernels.py"
    indexer_path = source / "vllm/model_executor/layers/sparse_attn_indexer.py"
    topk_path = source / "vllm/model_executor/layers/indexer_topk.py"
    canon_helper = helper.parent / "glm52_topk_canon.py"
    runner_path = source / "vllm/v1/worker/gpu/model_runner.py"
    kvu_path = source / "vllm/v1/core/kv_cache_utils.py"
    lmhead_helper = Path(__file__).resolve().parent / "glm52_lmhead_quant.py"
    mla_bmm_helper = Path(__file__).resolve().parent / "glm52_mla_bmm.py"
    thin_path = source / "vllm/ampere_thin_gemm/__init__.py"
    dsv32_attn_path = source / "vllm/models/deepseek_v32/attention.py"
    route_init_path = source / "vllm/ampere_decode/__init__.py"
    route_warmup_path = source / "vllm/ampere_decode/warmup.py"
    for path in (backend_path, mla_path, dsv32_path, kernels_path, indexer_path, topk_path, runner_path, kvu_path, helper, canon_helper, lmhead_helper, mla_bmm_helper, thin_path, dsv32_attn_path, route_init_path, route_warmup_path):
        if not path.is_file():
            raise RuntimeError(f"missing required file: {path}")

    backend = backend_path.read_text()
    mla = mla_path.read_text()

    backend = replace_once(
        backend,
        "from vllm.v1.attention.ops.triton_mla_sparse import triton_mla_sparse_fwd\n",
        "from vllm._glm52_mla_fp8 import triton_mla_sparse_attention_fp8\n"
        "from vllm.v1.attention.ops.triton_mla_sparse import triton_mla_sparse_fwd\n",
        "import packed-fp8 reader",
    )
    backend = replace_once(
        backend,
        '        "bfloat16",\n    ]',
        '        "bfloat16",\n'
        f'        "fp8_ds_mla",  # {MARKER}\n'
        '        "fp8",\n'
        '        "fp8_e4m3",\n'
        "    ]",
        "advertise packed-fp8 cache",
    )
    backend = replace_once(
        backend,
        '        if kv_cache_dtype not in (None, "auto", "float16", "bfloat16"):\n'
        '            return "Triton MLA Sparse currently supports only FP16/BF16 KV cache"',
        '        if kv_cache_dtype not in (\n'
        '            None, "auto", "float16", "bfloat16",\n'
        '            "fp8", "fp8_e4m3", "fp8_ds_mla",\n'
        '        ):\n'
        '            return "Triton MLA Sparse supports FP16/BF16 and packed FP8 KV cache"',
        "backend compatibility gate",
    )
    backend = replace_once(
        backend,
        '        if kv_cache_dtype not in ("auto", "float16", "bfloat16"):\n'
        "            raise NotImplementedError(\n"
        '                "TritonMLASparseImpl currently supports only FP16/BF16 KV cache."\n'
        "            )",
        '        if kv_cache_dtype not in ("auto", "float16", "bfloat16", "fp8_ds_mla"):\n'
        "            raise NotImplementedError(\n"
        '                "TritonMLASparseImpl supports FP16/BF16 and packed FP8 KV cache."\n'
        "            )",
        "implementation dtype gate",
    )
    backend = replace_once(
        backend,
        "        num_tokens = q.shape[0]\n"
        "        out, _, lse = triton_mla_sparse_fwd(\n",
        "        num_tokens = q.shape[0]\n"
        "        if self.kv_cache_dtype == \"fp8_ds_mla\":\n"
        "            # The cache is an opaque 656-byte row. The validated sm_80\n"
        "            # reader reconstructs e4m3 values from uint8 and never names\n"
        "            # an fp8 Triton type, which Ampere cannot compile. DCP is\n"
        "            # rejected by this backend, so no decode LSE is required.\n"
        "            assert kv_rows.dtype == torch.uint8 and kv_rows.shape[-1] == 656\n"
        "            out = triton_mla_sparse_attention_fp8(\n"
        "                q,\n"
        "                kv_rows,\n"
        "                topk_indices.view(num_tokens, 1, -1),\n"
        "                sm_scale=self.scale,\n"
        "            )\n"
        "            return out, None\n"
        "        out, _, lse = triton_mla_sparse_fwd(\n",
        "dispatch packed-fp8 reader",
    )

    mla = replace_once(
        mla,
        '    if backend_name == "FLASHMLA_SPARSE" and is_quantized_kv_cache(kv_cache_dtype):\n',
        '    if backend_name == "TRITON_MLA_SPARSE" and kv_cache_dtype in (\n'
        '        "fp8", "fp8_e4m3", "fp8_ds_mla"\n'
        '    ):\n'
        '        return "fp8_ds_mla"\n'
        '    if backend_name == "FLASHMLA_SPARSE" and is_quantized_kv_cache(kv_cache_dtype):\n',
        "canonicalize packed-fp8 cache dtype",
    )

    mla = patch_mla_head_bmm(mla)
    dsv32 = patch_dsv32_aux_over_pp(dsv32_path.read_text())
    kernels = patch_dsv32_sm80_fp8(kernels_path.read_text())
    indexer = patch_indexer_sm80(indexer_path.read_text())
    topk = patch_topk_canon_decode(topk_path.read_text())
    runner = patch_runner_lmhead_quant(runner_path.read_text())
    kvu = patch_kv_cache_dsa_dflash(kvu_path.read_text())
    thin = patch_thin_gemm_dsa(thin_path.read_text())
    dsv32_attn = patch_dsv32_head_bmm(dsv32_attn_path.read_text())
    route_init, route_warmup = patch_route_v2_dsa(
        route_init_path.read_text(), route_warmup_path.read_text()
    )

    ast.parse(backend, filename=str(backend_path))
    ast.parse(mla, filename=str(mla_path))
    ast.parse(dsv32, filename=str(dsv32_path))
    ast.parse(kernels, filename=str(kernels_path))
    ast.parse(indexer, filename=str(indexer_path))
    ast.parse(topk, filename=str(topk_path))
    ast.parse(runner, filename=str(runner_path))
    ast.parse(kvu, filename=str(kvu_path))
    ast.parse(thin, filename=str(thin_path))
    ast.parse(dsv32_attn, filename=str(dsv32_attn_path))
    ast.parse(route_init, filename=str(route_init_path))
    ast.parse(route_warmup, filename=str(route_warmup_path))
    ast.parse(lmhead_helper.read_text(), filename=str(lmhead_helper))
    ast.parse(mla_bmm_helper.read_text(), filename=str(mla_bmm_helper))
    ast.parse(canon_helper.read_text(), filename=str(canon_helper))
    ast.parse(helper.read_text(), filename=str(helper))

    if check:
        print("engine patch check: ok")
        return

    backend_path.write_text(backend)
    mla_path.write_text(mla)
    dsv32_path.write_text(dsv32)
    kernels_path.write_text(kernels)
    indexer_path.write_text(indexer)
    topk_path.write_text(topk)
    shutil.copyfile(canon_helper, source / "vllm/_glm52_topk_canon.py")
    runner_path.write_text(runner)
    kvu_path.write_text(kvu)
    thin_path.write_text(thin)
    dsv32_attn_path.write_text(dsv32_attn)
    route_init_path.write_text(route_init)
    route_warmup_path.write_text(route_warmup)
    shutil.copyfile(lmhead_helper, source / "vllm/_glm52_lmhead_quant.py")
    shutil.copyfile(mla_bmm_helper, source / "vllm/_glm52_mla_bmm.py")
    destination = source / "vllm/_glm52_mla_fp8.py"
    shutil.copyfile(helper, destination)
    digest = hashlib.sha256(helper.read_bytes()).hexdigest()
    (source / ".glm53-dflash2-local-patch").write_text(
        f"base {EXPECTED_COMMIT}\nhelper_sha256 {digest}\n"
    )
    print(f"engine patch applied; helper sha256={digest}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--helper", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    patch_source(args.source.resolve(), args.helper.resolve(), check=args.check)


if __name__ == "__main__":
    main()
