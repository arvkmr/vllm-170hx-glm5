#!/usr/bin/env python3
"""Build a hybrid checkpoint: the AWQ-g64 GLM-5.3 with selected 4-bit modules
replaced by the ORIGINAL bf16 tensors (precision-sensitivity bisect).

  graft_bf16.py --out DIR --modules indexer[,attn] [--layers 0-78]

modules:
  indexer  self_attn.indexer.{wq_b,wk,weights_proj}   (~0.2B params total)
  attn     self_attn.{q_b_proj,kv_b_proj,o_proj}      (~11.6B params, +2.1 GB/GPU)

Mechanics: DIR gets symlinks to every AWQ shard + one new shard
`model-bf16-graft.safetensors` holding the grafted `.weight` tensors (bf16);
the index drops the packed/scale/shape/zero_point entries of the grafted
modules; config.json's quantization_config.ignore gains the module names, so
the compressed-tensors loader builds them as UnquantizedLinearMethod exactly
like the already-unquantized layers 0-2. Modules that are already plain
`.weight` in the AWQ checkpoint are left alone.
"""
import argparse
import json
import os
import re
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

AWQ = "/home/user/models/GLM-5.3-AWQ-g64"
BF16 = "/home/user/srv/models/GLM-5.3-BF16"
MODS = {
    "indexer": ["self_attn.indexer.wq_b", "self_attn.indexer.wk", "self_attn.indexer.weights_proj"],
    "attn": ["self_attn.q_b_proj", "self_attn.kv_b_proj", "self_attn.o_proj"],
}

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--modules", required=True)
ap.add_argument("--layers", default="0-78")
a = ap.parse_args()
lo, hi = (int(x) for x in a.layers.split("-"))
mods = [m for g in a.modules.split(",") for m in MODS[g]]

aidx = json.load(open(f"{AWQ}/model.safetensors.index.json"))
bidx = json.load(open(f"{BF16}/model.safetensors.index.json"))["weight_map"]
wm = dict(aidx["weight_map"])
cfg = json.load(open(f"{AWQ}/config.json"))
ignore = list(cfg["quantization_config"]["ignore"])

os.makedirs(a.out, exist_ok=True)
for f in os.listdir(AWQ):
    if f.endswith(".safetensors"):
        dst = os.path.join(a.out, f)
        if not os.path.exists(dst):
            os.symlink(os.path.join(AWQ, f), dst)
    elif f not in ("config.json", "model.safetensors.index.json"):
        shutil.copyfile(os.path.join(AWQ, f), os.path.join(a.out, f))

grafted, skipped, missing = [], [], []
tensors = {}
handles = {}
for L in range(lo, hi + 1):
    for m in mods:
        name = f"model.layers.{L}.{m}"
        packed = f"{name}.weight_packed"
        plain = f"{name}.weight"
        if packed not in wm:
            if plain in wm:
                skipped.append(name)      # already unquantized in AWQ
            continue
        if plain not in bidx:
            missing.append(name)          # bf16 original has no such module
            continue
        shard = bidx[plain]
        if shard not in handles:
            handles[shard] = safe_open(f"{BF16}/{shard}", "pt")
        t = handles[shard].get_tensor(plain).to(torch.bfloat16).contiguous()
        tensors[plain] = t
        for suf in ("weight_packed", "weight_scale", "weight_shape", "weight_zero_point"):
            wm.pop(f"{name}.{suf}", None)
        wm[plain] = "model-bf16-graft.safetensors"
        ignore.append(name)
        grafted.append(name)

nbytes = sum(t.numel() * 2 for t in tensors.values())
print(f"grafted {len(grafted)} modules ({nbytes/1e9:.2f} GB bf16); already-plain skipped {len(skipped)}; "
      f"missing in bf16 original {len(missing)}: {missing[:5]}")
save_file(tensors, os.path.join(a.out, "model-bf16-graft.safetensors"), metadata={"format": "pt"})
aidx["weight_map"] = wm
json.dump(aidx, open(os.path.join(a.out, "model.safetensors.index.json"), "w"), indent=1)
cfg["quantization_config"]["ignore"] = ignore
json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=1)
open(os.path.join(a.out, "GRAFT.txt"), "w").write("\n".join(grafted) + "\n")
print("wrote", a.out)
