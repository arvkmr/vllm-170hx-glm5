#!/usr/bin/env python3
"""Insert GLM52_TAP graph taps (see _glm52_tap.py) into mla.py, deepseek_v2.py (decoder layer
and Indexer forward). Idempotent; inert unless GLM52_TAP is set."""
import sys
import os as _os
import vllm as _vllm
V = _os.path.dirname(_vllm.__file__) + "/"


def edit(path, old, new, label, count=1):
    s = open(path).read()
    if new in s:
        print(f"  [skip] {label} already applied"); return
    assert s.count(old) == count, f"{label}: {s.count(old)} matches"
    open(path, "w").write(s.replace(old, new)); print(f"  [ok] {label}")


# ---- mla.py: MultiHeadLatentAttentionWrapper.forward
p = V + "model_executor/layers/mla.py"
edit(p, "import torch\n", "import torch\nfrom vllm._glm52_tap import tap as _glm52_tap\n", "mla import")
edit(p, """            qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
            q_c, kv_lora = qkv_lora.split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                dim=-1,
            )
            q_c = self.q_a_layernorm(q_c)
            q = self.q_b_proj(q_c)[0]
""", """            qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
            _glm52_tap(hidden_states, self.prefix, "attn_in"); _glm52_tap(qkv_lora, self.prefix, "qkv_a")
            q_c, kv_lora = qkv_lora.split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                dim=-1,
            )
            q_c = self.q_a_layernorm(q_c)
            _glm52_tap(q_c, self.prefix, "q_c")
            q = self.q_b_proj(q_c)[0]
            _glm52_tap(q, self.prefix, "q_b")
""", "mla q path")
edit(p, """        kv_c_normed = self.kv_a_layernorm(kv_c)
""", """        kv_c_normed = self.kv_a_layernorm(kv_c)
        _glm52_tap(kv_c_normed, self.prefix, "kv_c_normed")
""", "mla kv norm")
edit(p, """        if self.indexer and self.is_sparse and not self.skip_topk:
            self.indexer(hidden_states, q_c, positions, self.indexer_rope_emb)
""", """        _glm52_tap(q, self.prefix, "q_rope"); _glm52_tap(k_pe, self.prefix, "k_pe_rope")
        if self.indexer and self.is_sparse and not self.skip_topk:
            self.indexer(hidden_states, q_c, positions, self.indexer_rope_emb)
""", "mla post-rope")
edit(p, """        return self.o_proj(attn_out)[0]
""", """        _glm52_tap(attn_out, self.prefix, "attn_out")
        _o = self.o_proj(attn_out)[0]
        _glm52_tap(_o, self.prefix, "o_proj")
        return _o
""", "mla o_proj")

# ---- deepseek_v2.py: decoder layer + Indexer
p = V + "model_executor/models/deepseek_v2.py"
edit(p, "from vllm.utils.torch_utils import direct_register_custom_op\n",
     "from vllm.utils.torch_utils import direct_register_custom_op\nfrom vllm._glm52_tap import tap as _glm52_tap\n", "dsv2 import")
edit(p, """        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
""", """        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        _glm52_tap(hidden_states, self.layer_idx, "in_norm"); _glm52_tap(residual, self.layer_idx, "residual_in")
""", "dsv2 input norm")
edit(p, """            hidden_states = self.self_attn(positions, hidden_states, llama_4_scaling)

        if (
            not isinstance(self.self_attn, DeepseekAttention)
""", """            hidden_states = self.self_attn(positions, hidden_states, llama_4_scaling)
        _glm52_tap(hidden_states, self.layer_idx, "attn")

        if (
            not isinstance(self.self_attn, DeepseekAttention)
""", "dsv2 attn out")
edit(p, """        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        if self.use_sequence_parallel_moe:
""", """        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        _glm52_tap(hidden_states, self.layer_idx, "post_norm"); _glm52_tap(residual, self.layer_idx, "residual_mid")
        if self.use_sequence_parallel_moe:
""", "dsv2 post norm")
edit(p, """        else:
            hidden_states = self.mlp(hidden_states)

        if (
            isinstance(self.mlp, DeepseekV2MLP)
""", """        else:
            hidden_states = self.mlp(hidden_states)
        _glm52_tap(hidden_states, self.layer_idx, "mlp")

        if (
            isinstance(self.mlp, DeepseekV2MLP)
""", "dsv2 mlp out")
# Indexer.forward (the non-ROCm branches)
edit(p, """        q, _ = self.wq_b(qr)
        q = q.view(-1, self.n_head, self.head_dim)
""", """        q, _ = self.wq_b(qr)
        _glm52_tap(qr, self.prefix, "idx_qr"); _glm52_tap(q, self.prefix, "idx_q")
        q = q.view(-1, self.n_head, self.head_dim)
""", "indexer q")
edit(p, """            kw, _ = self.wk_weights_proj(hidden_states)
            k = kw[:, : self.head_dim]
            weights = kw[:, self.head_dim :]

            k = self.k_norm(k)
            k_pe, k_nope = torch.split(
                k, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
            )
""", """            kw, _ = self.wk_weights_proj(hidden_states)
            _glm52_tap(hidden_states, self.prefix, "idx_hidden"); _glm52_tap(kw, self.prefix, "idx_kw")
            k = kw[:, : self.head_dim]
            weights = kw[:, self.head_dim :]

            k = self.k_norm(k)
            _glm52_tap(k, self.prefix, "idx_knorm")
            k_pe, k_nope = torch.split(
                k, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
            )
""", "indexer k", count=2)
edit(p, """        return self.indexer_op(hidden_states, q_fp8, k, weights)
""", """        _glm52_tap(q_fp8, self.prefix, "idx_qfp8"); _glm52_tap(k, self.prefix, "idx_k"); _glm52_tap(weights, self.prefix, "idx_w")
        return self.indexer_op(hidden_states, q_fp8, k, weights)
""", "indexer op inputs", count=2)
print("done")

