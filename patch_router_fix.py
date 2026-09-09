#!/usr/bin/env python3
"""GLM52_ROUTER_FIX=<path.pt>: re-compensate the AWQ smoothing fold for the
two consumers the recipe missed.

The GLM-5.3-AWQ-g64 recipe folded per-channel AWQ smoothing scales s into
post_attention_layernorm (LN' = LN/s) and q_a_layernorm, and multiplied the
routed/shared expert up/gate weights and q_b_proj by s. The router
`mlp.gate.weight` and the indexer `wq_b.weight` were left IDENTICAL to the
original, so they consume x/s instead of x: expert routing and sparse
selection are systematically wrong. Fix at load: W_gate *= s (per input
column), W_wq_b *= s_q. The .pt maps layer -> {norm_name: s (bf16/f16)}
(coflight_probe/router_fix_s.pt, built from the two checkpoints)."""
import os, sys, vllm
F = os.path.join(os.path.dirname(vllm.__file__), "model_executor/models/deepseek_v2.py")
src = open(F).read()
HELPER = '''
_GLM52_ROUTER_FIX = _glm52_os.environ.get("GLM52_ROUTER_FIX", "")


def _glm52_apply_router_fix(model):
    """Multiply router gate / indexer wq_b columns by the folded smoothing scale."""
    if not _GLM52_ROUTER_FIX:
        return
    import re as _re
    S = torch.load(_GLM52_ROUTER_FIX)
    n_gate = n_wqb = 0
    with torch.no_grad():
        for name, p in model.named_parameters():
            m = _re.match(r"model\\.layers\\.(\\d+)\\.(mlp\\.gate\\.weight|self_attn\\.indexer\\.wq_b\\.weight)$", name)
            if not m:
                continue
            L = int(m.group(1)); rec = S.get(L, {})
            key = "post_attention_layernorm" if m.group(2).startswith("mlp") else "q_a_layernorm"
            if key not in rec:
                continue
            s = rec[key].to(device=p.device, dtype=torch.float32)
            assert p.shape[1] == s.numel(), (name, p.shape, s.shape)
            p.copy_((p.float() * s[None, :]).to(p.dtype))
            if key.startswith("post"): n_gate += 1
            else: n_wqb += 1
    print(f"[glm52-router-fix] applied: gate={n_gate} indexer_wq_b={n_wqb} from {_GLM52_ROUTER_FIX}",
          file=sys.stderr, flush=True)
'''
anchor = "_GLM52_FP16_EXACT = _glm52_os.environ.get(\"GLM52_FP16_EXACT_STREAM\", \"1\") == \"1\"\n"
if "_glm52_apply_router_fix" not in src:
    assert src.count(anchor) == 1
    src = src.replace(anchor, anchor + HELPER)
    if "\nimport sys\n" not in src:
        src = src.replace("import os as _glm52_os\n", "import os as _glm52_os\nimport sys\n", 1)
    print("  + helper")
# call site: end of DeepseekV2ForCausalLM.load_weights (the one returning loader.load_weights)
old = "        return loader.load_weights(weights)\n"
new = ("        _r = loader.load_weights(weights)\n"
       "        _glm52_apply_router_fix(self)\n"
       "        return _r\n")
if new in src:
    print("  = call")
else:
    assert src.count(old) >= 1, src.count(old)
    src = src.replace(old, new, 1)
    print("  + call (first return loader.load_weights site)")
open(F, "w").write(src)
import py_compile; py_compile.compile(F, doraise=True); print("done (compiles)")
