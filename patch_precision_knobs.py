#!/usr/bin/env python3
"""Two env-gated precision knobs (inert unless set):
  GLM52_GATE_FP32=1   MoE router logits via a true fp32 GEMM (x.float() @ W.float()).
                      On sm_80 GateLinear falls through to a bf16 F.linear and casts the
                      bf16-ROUNDED logits to fp32; GLM-5 configs require fp32 routing.
  GLM52_IDX_Q_BF16=1  indexer query scored in bf16 instead of fp8 (prefill kernels decode q to
                      bf16 anyway; the paged decode kernels quantize q on entry instead).
Idempotent edits of the installed vLLM."""
import os as _os
import vllm as _vllm
V = _os.path.dirname(_vllm.__file__) + "/"


def edit(path, old, new, label):
    s = open(path).read()
    if new in s:
        print(f"  [skip] {label}"); return
    assert s.count(old) == 1, f"{label}: {s.count(old)} matches"
    open(path, "w").write(s.replace(old, new)); print(f"  [ok] {label}")


# ---- gate fp32
p = V + "model_executor/layers/fused_moe/router/gate_linear.py"
edit(p, "@PluggableLayer.register(\"gate_linear\")\nclass GateLinear(ReplicatedLinear):\n",
     'import os as _glm52_os\n_GLM52_GATE_FP32 = _glm52_os.environ.get("GLM52_GATE_FP32", "0") == "1"\n\n\n@PluggableLayer.register("gate_linear")\nclass GateLinear(ReplicatedLinear):\n',
     "gate flag")
edit(p, """        if self.allow_ll_bf16_gemm and x.shape[0] <= 16 and x.dtype == torch.bfloat16:
""", """        if _GLM52_GATE_FP32:
            w32 = getattr(self, "weight_fp32", None)
            if w32 is None:
                w32 = self.weight.to(torch.float32)
            return torch.mm(x.to(torch.float32), w32.t()), None
        if self.allow_ll_bf16_gemm and x.shape[0] <= 16 and x.dtype == torch.bfloat16:
""", "gate fp32 branch")

# ---- indexer q bf16: skip the fp8 quant in Indexer.forward (unfused branch)
p = V + "model_executor/models/deepseek_v2.py"
edit(p, """        # we only quant q here since k quant is fused with cache insertion
        q = q.view(-1, self.head_dim)
        q_fp8, q_scale = per_token_group_quant_fp8(
            q,
            self.quant_block_size,
            column_major_scales=False,
            use_ue8m0=self.scale_fmt is not None,
        )
        q_fp8 = q_fp8.view(-1, self.n_head, self.head_dim)
        q_scale = q_scale.view(-1, self.n_head)
""", """        # we only quant q here since k quant is fused with cache insertion
        q = q.view(-1, self.head_dim)
        if _GLM52_IDX_Q_BF16:
            # GLM52: score the index in bf16 (no fp8 q); decode kernels quantize on entry.
            q_fp8 = q.to(torch.bfloat16).view(-1, self.n_head, self.head_dim)
            q_scale = torch.ones((q_fp8.shape[0], self.n_head), dtype=torch.float32, device=q.device)
        else:
            q_fp8, q_scale = per_token_group_quant_fp8(
                q,
                self.quant_block_size,
                column_major_scales=False,
                use_ue8m0=self.scale_fmt is not None,
            )
            q_fp8 = q_fp8.view(-1, self.n_head, self.head_dim)
            q_scale = q_scale.view(-1, self.n_head)
""", "indexer q bf16")
edit(p, "from vllm._glm52_tap import tap as _glm52_tap\n",
     'from vllm._glm52_tap import tap as _glm52_tap\nimport os as _glm52_os\n_GLM52_IDX_Q_BF16 = _glm52_os.environ.get("GLM52_IDX_Q_BF16", "0") == "1"\n', "indexer flag")

# ---- decode path: quantize a bf16 q on entry to the paged kernels
p = V + "model_executor/layers/sparse_attn_indexer.py"
edit(p, """        padded_q_quant_cast = (
            padded_q_quant_decode_tokens.view(torch.int8)
            if use_fp4_cache
            else padded_q_quant_decode_tokens
        )
""", """        padded_q_quant_cast = (
            padded_q_quant_decode_tokens.view(torch.int8)
            if use_fp4_cache
            else padded_q_quant_decode_tokens
        )
        if padded_q_quant_cast.dtype == torch.bfloat16:
            # GLM52_IDX_Q_BF16: the paged decode kernels read fp8 bytes; quantize here and
            # fold the per-group scale into the head weights (as the prefill path would).
            from vllm.model_executor.layers.quantization.utils.fp8_utils import (
                per_token_group_quant_fp8 as _glm52_q8,
            )
            _shp = padded_q_quant_cast.shape
            _qf, _qs = _glm52_q8(
                padded_q_quant_cast.reshape(-1, head_dim), quant_block_size,
                column_major_scales=False, use_ue8m0=scale_fmt is not None,
            )
            padded_q_quant_cast = _qf.view(_shp)
            weights = weights.clone()
            weights[:num_padded_tokens] = weights[:num_padded_tokens] * _qs.view(num_padded_tokens, -1)
""", "decode q requant")
print("done")
