#!/usr/bin/env python3
"""Teach TRITON_MLA_SPARSE to serve from an fp8_ds_mla KV cache on sm_80.

Why this is mostly plumbing: the *write* side already works. vLLM's compiled
`concat_and_cache_mla` handles `fp8_ds_mla` on Ampere unchanged (verified with
a byte-level decode of what it produces), `fp8_ds_mla` already maps to uint8
storage at 656 B/token in kv_cache_interface, and the layer never routes a
sparse batch through the prefill upconvert path because
TritonMLASparseMetadata reports every token as a decode token. What was
missing was a *reader*: Triton cannot name `fp8e4nv` below sm_89, so the
attention kernel had to decode e4m3 from raw bytes. That lives in
glm52_mla_fp8.py, installed here as vllm/_glm52_mla_fp8.py.

Capacity on this box (11 MLA layers + 5 indexer caches on the binding rank):
13,332 -> 7,876 bytes/token, so the same 6.57 GiB/rank holds 529,024 ->
895,488 KV tokens, a 1.69x gain. The MLA cache itself drops 1.76x (1152 ->
656); the indexer caches are already fp8 and do not move.

Enable with `--kv-cache-dtype fp8_ds_mla` (or `fp8`, which canonicalizes to
it). GLM52_MLA_FP8=0 forces the bf16 kernel back even when the cache is fp8,
which is only useful for A/B timing -- it will produce garbage, since the
cache really is packed bytes.
"""

import os
import shutil
import sys

import vllm

VLLM = os.path.dirname(vllm.__file__)
BACKUP = os.path.join(os.path.dirname(VLLM), ".glm52-backup", "vllm")
SRC = "/home/user/vllm_install/glm52_mla_fp8.py"
DST = os.path.join(VLLM, "_glm52_mla_fp8.py")

TMS = "v1/attention/backends/mla/triton_mla_sparse.py"
MLA = "model_executor/layers/attention/mla_attention.py"

# Collected first, applied only once every anchor in the set has been checked:
# a half-patched site-packages is far worse than an unpatched one.
_EDITS: list[tuple[str, str, str, str]] = []


def edit(path, old, new, label):
    _EDITS.append((path, old, new, label))


def apply_all():
    files = {}
    todo = []
    for path, old, new, label in _EDITS:
        src = files.get(path) or open(os.path.join(VLLM, path)).read()
        if new in src:
            print(f"  = {label} (already applied)")
            files[path] = src
            continue
        if src.count(old) != 1:
            print(
                f"  ! {label}: anchor matched {src.count(old)} times in {path}, "
                f"expected 1 -- nothing written",
                file=sys.stderr,
            )
            sys.exit(1)
        files[path] = src.replace(old, new)
        todo.append(label)

    for path in files:
        # Keep a pristine copy next to the other GLM-5.2 patch backups. First
        # writer wins, so re-running never overwrites the original with a
        # already-patched one.
        dst = os.path.join(BACKUP, path)
        if not os.path.exists(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(os.path.join(VLLM, path), dst)
            print(f"  b backup -> {dst}")

    for path, src in files.items():
        open(os.path.join(VLLM, path), "w").write(src)
    for label in todo:
        print(f"  + {label}")

# ---------------------------------------------------------------- 1. dtype gate
edit(
    TMS,
    '    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [\n'
    '        "auto",\n'
    '        "float16",\n'
    '        "bfloat16",\n'
    '    ]',
    '    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [\n'
    '        "auto",\n'
    '        "float16",\n'
    '        "bfloat16",\n'
    '        # DeepSeek\'s packed 656 B/token layout: 512 e4m3 NoPE + 4 fp32\n'
    '        # group scales + 64 bf16 RoPE. Ampere has no fp8 hardware, but\n'
    '        # nothing here needs any -- the compiled writer already emits this\n'
    '        # format on sm_80 and _glm52_mla_fp8.py decodes it from bytes.\n'
    '        "fp8_ds_mla",\n'
    '        "fp8",  # alias, canonicalized in mla_attention.py\n'
    '    ]',
    "backend accepts fp8_ds_mla",
)

# ------------------------------------------------------------- 2. packed shape
edit(
    TMS,
    "    @staticmethod\n"
    "    def get_kv_cache_shape(\n"
    "        num_blocks: int,\n"
    "        block_size: int,\n"
    "        num_kv_heads: int,\n"
    "        head_size: int,\n"
    '        cache_dtype_str: str = "auto",\n'
    "    ) -> tuple[int, ...]:\n"
    "        return (num_blocks, block_size, head_size)",
    "    @staticmethod\n"
    "    def get_kv_cache_shape(\n"
    "        num_blocks: int,\n"
    "        block_size: int,\n"
    "        num_kv_heads: int,\n"
    "        head_size: int,\n"
    '        cache_dtype_str: str = "auto",\n'
    "    ) -> tuple[int, ...]:\n"
    '        if cache_dtype_str == "fp8_ds_mla":\n'
    "            # Bytes, not elements: the spec's dtype is uint8. Matches\n"
    "            # MLAAttentionSpec.real_page_size_bytes, which already knows\n"
    "            # this layout is 656 B/token.\n"
    "            return (num_blocks, block_size, 656)\n"
    "        return (num_blocks, block_size, head_size)",
    "packed 656B cache shape",
)

# ------------------------------------------------- 3. canonicalize fp8 aliases
edit(
    MLA,
    '    if backend_name == "FLASHMLA_SPARSE" and is_quantized_kv_cache(kv_cache_dtype):\n'
    '        return "fp8_ds_mla"',
    '    if (\n'
    '        backend_name in ("FLASHMLA_SPARSE", "TRITON_MLA_SPARSE")\n'
    "        and is_quantized_kv_cache(kv_cache_dtype)\n"
    "    ):\n"
    '        return "fp8_ds_mla"',
    "canonicalize fp8 -> fp8_ds_mla for TRITON_MLA_SPARSE",
)

# --------------------------------------------------------- 4. impl: fp8 branch
edit(
    TMS,
    "from vllm.v1.attention.ops.triton_mla_sparse_kernel import (\n"
    "    _DIM_QK,\n"
    "    KV_SPLITS_CANDIDATES,\n"
    "    triton_mla_sparse_attention,\n"
    ")",
    "from vllm._glm52_mla_fp8 import triton_mla_sparse_attention_fp8\n"
    "from vllm.v1.attention.backends.mla.sparse_utils import (\n"
    "    triton_convert_req_index_to_global_index,\n"
    ")\n"
    "from vllm.v1.attention.ops.triton_mla_sparse_kernel import (\n"
    "    _DIM_QK,\n"
    "    KV_SPLITS_CANDIDATES,\n"
    "    triton_mla_sparse_attention,\n"
    ")\n"
    "\n"
    "_FP8_DS_MLA_ENTRY_BYTES = 656",
    "impl imports",
)

edit(
    TMS,
    "    def __init__(self, *args, **kwargs) -> None:\n"
    "        super().__init__(*args, **kwargs)\n"
    "        self._sm_count: int | None = None\n"
    "        if self.topk_indices_buffer is not None:\n"
    "            self._sm_count = num_compute_units(self.topk_indices_buffer.device.index)\n"
    '        self._warmup_autotune(kwargs["indexer"])',
    "    def __init__(self, *args, **kwargs) -> None:\n"
    "        super().__init__(*args, **kwargs)\n"
    "        self._sm_count: int | None = None\n"
    "        if self.topk_indices_buffer is not None:\n"
    "            self._sm_count = num_compute_units(self.topk_indices_buffer.device.index)\n"
    "        # GLM52_MLA_FP8=0 is an A/B-timing escape hatch only: it points the\n"
    "        # bf16 kernel at a packed cache, which decodes to nonsense.\n"
    '        self._fp8_kv = self.kv_cache_dtype == "fp8_ds_mla" and (\n'
    '            os.environ.get("GLM52_MLA_FP8", "1") == "1"\n'
    "        )\n"
    '        self._warmup_autotune(kwargs["indexer"])',
    "impl: detect fp8 cache",
)

edit(
    TMS,
    "from dataclasses import dataclass, fields",
    "import os\nfrom dataclasses import dataclass, fields",
    "impl: import os",
)

# Autotune must be primed at init: it benchmarks configs on first call, and
# doing that inside a cudagraph capture (GLM52_DSA_FULLCG=1) is fatal.
edit(
    TMS,
    "        q = torch.zeros(1, self.num_heads, _DIM_QK, dtype=torch.bfloat16, device=device)\n"
    "        kv = torch.zeros(64, 1, _DIM_QK, dtype=torch.bfloat16, device=device)\n"
    "        indices = torch.zeros(1, 1, topk, dtype=torch.int32, device=device)\n"
    "        for splits in KV_SPLITS_CANDIDATES:\n"
    "            triton_mla_sparse_attention(\n"
    "                q,\n"
    "                kv,\n"
    "                indices,\n"
    "                sm_scale=self.softmax_scale,\n"
    "                num_kv_splits=splits,\n"
    "                sm_count=self._sm_count,\n"
    "            )",
    "        q = torch.zeros(1, self.num_heads, _DIM_QK, dtype=torch.bfloat16, device=device)\n"
    "        indices = torch.zeros(1, 1, topk, dtype=torch.int32, device=device)\n"
    "        if self._fp8_kv:\n"
    "            kv_fp8 = torch.zeros(\n"
    "                64, _FP8_DS_MLA_ENTRY_BYTES, dtype=torch.uint8, device=device\n"
    "            )\n"
    "            for splits in KV_SPLITS_CANDIDATES:\n"
    "                triton_mla_sparse_attention_fp8(\n"
    "                    q,\n"
    "                    kv_fp8,\n"
    "                    indices,\n"
    "                    sm_scale=self.softmax_scale,\n"
    "                    num_kv_splits=splits,\n"
    "                    sm_count=self._sm_count,\n"
    "                )\n"
    "        else:\n"
    "            kv = torch.zeros(64, 1, _DIM_QK, dtype=torch.bfloat16, device=device)\n"
    "            for splits in KV_SPLITS_CANDIDATES:\n"
    "                triton_mla_sparse_attention(\n"
    "                    q,\n"
    "                    kv,\n"
    "                    indices,\n"
    "                    sm_scale=self.softmax_scale,\n"
    "                    num_kv_splits=splits,\n"
    "                    sm_count=self._sm_count,\n"
    "                )",
    "impl: warm the fp8 autotune caches",
)

# The XPU base's forward_mqa hard-raises on a quantized cache, so override the
# whole dispatch rather than trying to slip past the check.
edit(
    TMS,
    "    def _forward_bf16_kv(\n"
    "        self,\n"
    "        q: torch.Tensor,  # [sq, heads, d_qk]\n"
    "        kv_c_and_k_pe_cache: torch.Tensor,  # [blocks, heads, d_qk]\n"
    "        topk_indices: torch.Tensor,  # [sq, topk]\n"
    "        attn_metadata: XPUMLASparseMetadata,\n"
    "    ) -> torch.Tensor:\n"
    "        num_tokens = q.shape[0]\n"
    "        kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.view(\n"
    "            -1, 1, kv_c_and_k_pe_cache.shape[-1]\n"
    "        )\n"
    "        topk_indices = topk_indices.view(num_tokens, 1, -1)\n"
    "        output = triton_mla_sparse_attention(\n"
    "            q,\n"
    "            kv_c_and_k_pe_cache,\n"
    "            topk_indices,\n"
    "            sm_scale=self.softmax_scale,\n"
    "            sm_count=self._sm_count,\n"
    "        )\n"
    "        return output[:, : self.num_heads, :]",
    "    def _forward_bf16_kv(\n"
    "        self,\n"
    "        q: torch.Tensor,  # [sq, heads, d_qk]\n"
    "        kv_c_and_k_pe_cache: torch.Tensor,  # [blocks, heads, d_qk]\n"
    "        topk_indices: torch.Tensor,  # [sq, topk]\n"
    "        attn_metadata: XPUMLASparseMetadata,\n"
    "    ) -> torch.Tensor:\n"
    "        num_tokens = q.shape[0]\n"
    "        kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.view(\n"
    "            -1, 1, kv_c_and_k_pe_cache.shape[-1]\n"
    "        )\n"
    "        topk_indices = topk_indices.view(num_tokens, 1, -1)\n"
    "        output = triton_mla_sparse_attention(\n"
    "            q,\n"
    "            kv_c_and_k_pe_cache,\n"
    "            topk_indices,\n"
    "            sm_scale=self.softmax_scale,\n"
    "            sm_count=self._sm_count,\n"
    "        )\n"
    "        return output[:, : self.num_heads, :]\n"
    "\n"
    "    def _forward_fp8_kv(\n"
    "        self,\n"
    "        q: torch.Tensor,  # [sq, heads, d_qk]\n"
    "        kv_c_and_k_pe_cache: torch.Tensor,  # [blocks, block_size, 656] uint8\n"
    "        topk_indices: torch.Tensor,  # [sq, topk]\n"
    "        attn_metadata: XPUMLASparseMetadata,\n"
    "    ) -> torch.Tensor:\n"
    "        num_tokens = q.shape[0]\n"
    "        output = triton_mla_sparse_attention_fp8(\n"
    "            q,\n"
    "            kv_c_and_k_pe_cache.view(-1, _FP8_DS_MLA_ENTRY_BYTES),\n"
    "            topk_indices.view(num_tokens, 1, -1),\n"
    "            sm_scale=self.softmax_scale,\n"
    "            sm_count=self._sm_count,\n"
    "        )\n"
    "        return output[:, : self.num_heads, :]\n"
    "\n"
    "    def forward_mqa(\n"
    "        self,\n"
    "        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],\n"
    "        kv_c_and_k_pe_cache: torch.Tensor,\n"
    "        attn_metadata: XPUMLASparseMetadata,\n"
    "        layer,\n"
    "    ) -> tuple[torch.Tensor, torch.Tensor | None]:\n"
    "        # Overrides XPUMLASparseImpl.forward_mqa, which raises outright on a\n"
    "        # quantized cache; otherwise identical.\n"
    "        if isinstance(q, tuple):\n"
    "            q = torch.cat(q, dim=-1)\n"
    "        num_actual_toks = q.shape[0]\n"
    "        assert self.topk_indices_buffer is not None\n"
    "        topk_indices = self.topk_indices_buffer[:num_actual_toks]\n"
    "        topk_indices_global = triton_convert_req_index_to_global_index(\n"
    "            attn_metadata.req_id_per_token,\n"
    "            attn_metadata.block_table,\n"
    "            topk_indices,\n"
    "            BLOCK_SIZE=attn_metadata.block_size,\n"
    "            NUM_TOPK_TOKENS=attn_metadata.topk_tokens,\n"
    "        )\n"
    "        forward = self._forward_fp8_kv if self._fp8_kv else self._forward_bf16_kv\n"
    "        attn_out = forward(\n"
    "            q, kv_c_and_k_pe_cache, topk_indices_global, attn_metadata\n"
    "        )\n"
    "        return attn_out, None",
    "impl: fp8 forward + dispatch",
)

# ------------------------------------ 5. drop an unreachable 3.5 GiB reserve
# Not an fp8 change, but it is what stands between this box and a 1M context.
# The dummy run allocates a worst-case buffer to model the up-projection inside
# `_compute_prefill_context`, sized workspace(64k) x heads x (qk_nope+v_head) =
# 3.5 GiB. TRITON_MLA_SPARSE can never reach that path: its metadata builder
# sets num_decode_tokens = num_actual_tokens, and forward_impl slices q to the
# same length, so `num_mha_tokens = q.size(0) - num_decode_tokens` is
# identically 0 and the dense-MHA branch is dead code. Reserving headroom for
# it costs exactly the margin the 1M KV cache needs. (Booting at 1M failed by
# ~20 MB with this allocation live.)
edit(
    MLA,
    "        if attn_metadata is None:\n"
    "            # During the profile run try to simulate to worse case output size\n"
    "            # for `self.kv_b_proj(kv_c_normed)` in `_compute_prefill_context`\n"
    "            # since this can be large\n"
    "            _ = torch.empty(",
    "        if attn_metadata is None:\n"
    "            # During the profile run try to simulate to worse case output size\n"
    "            # for `self.kv_b_proj(kv_c_normed)` in `_compute_prefill_context`\n"
    "            # since this can be large.\n"
    "            # Skipped for TRITON_MLA_SPARSE, whose TritonMLASparseMetadata\n"
    "            # reports every token as a decode token -- num_mha_tokens is\n"
    "            # then identically 0, so _compute_prefill_context is dead code\n"
    "            # and this 3.5 GiB reserve only shrinks the KV cache.\n"
    "            _sparse_mqa_only = (\n"
    '                self.attn_backend.get_name() == "TRITON_MLA_SPARSE"\n'
    "            )\n"
    "            _ = None if _sparse_mqa_only else torch.empty(",
    "skip unreachable 3.5 GiB prefill reserve (sparse MQA-only)",
)

apply_all()
shutil.copyfile(SRC, DST)
print(f"  + kernel -> {DST}")

print("\nfp8_ds_mla KV cache enabled for TRITON_MLA_SPARSE.")
print("Serve with: KV_CACHE_DTYPE=fp8_ds_mla ./start_glm52.sh")
