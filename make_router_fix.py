#!/usr/bin/env python3
"""Build the GLM52_ROUTER_FIX scale file from an AWQ checkpoint and its bf16 original.

  make_router_fix.py --awq /path/GLM-5.3-AWQ-g64 --bf16 /path/GLM-5.3-BF16 --out router_fix_glm53_s.pt

The AWQ recipe folded per-channel smoothing scales s into three RMSNorm weights
(LN_awq = LN_bf16 / s) and multiplied the smoothed consumers by s, but two consumers of those
norms were left untouched (they were on the recipe's ignore list): the MoE router
`mlp.gate.weight` (fed by post_attention_layernorm) and the DSA indexer `indexer.wq_b.weight`
(fed by q_a_layernorm). They therefore see x/s instead of x. patch_router_fix.py multiplies
their input columns by s at load; this script recovers s = LN_bf16 / LN_awq per layer.

Output: {layer_idx: {"post_attention_layernorm": s, "q_a_layernorm": s_q, "kv_a_layernorm": s_kv}}
with s in the AWQ checkpoint's norm dtype. Layers whose norms are identical in both
checkpoints (nothing folded) are omitted, so the patch leaves them alone.
"""
import argparse
import json
import os

import torch
from safetensors import safe_open

NORMS = {
    "post_attention_layernorm": "model.layers.{L}.post_attention_layernorm.weight",
    "q_a_layernorm": "model.layers.{L}.self_attn.q_a_layernorm.weight",
    "kv_a_layernorm": "model.layers.{L}.self_attn.kv_a_layernorm.weight",
}


def load_index(d):
    return json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--awq", required=True)
    ap.add_argument("--bf16", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    wm_a, wm_b = load_index(a.awq), load_index(a.bf16)
    n_layers = 1 + max(int(k.split(".")[2]) for k in wm_a if k.startswith("model.layers."))
    handles = {}

    def get(d, wm, name):
        shard = wm[name]
        key = (d, shard)
        if key not in handles:
            handles[key] = safe_open(os.path.join(d, shard), "pt")
        return handles[key].get_tensor(name)

    out, folded = {}, 0
    for L in range(n_layers):
        rec = {}
        for norm, tmpl in NORMS.items():
            name = tmpl.format(L=L)
            if name not in wm_a or name not in wm_b:
                continue
            ln_a, ln_b = get(a.awq, wm_a, name), get(a.bf16, wm_b, name)
            s = ln_b.float() / ln_a.float()
            if torch.allclose(s, torch.ones_like(s), atol=1e-3):
                continue  # not folded on this layer
            rec[norm] = s.to(ln_a.dtype)
            folded += 1
        if rec:
            out[L] = rec
    torch.save(out, a.out)
    print(f"layers with folded norms: {len(out)} ({folded} norm tensors); wrote {a.out}")
    print("apply with: GLM52_ROUTER_FIX=" + os.path.abspath(a.out))


if __name__ == "__main__":
    main()
