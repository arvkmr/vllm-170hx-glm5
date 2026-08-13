#!/usr/bin/env python3
"""Measure lm_head quantization error deterministically, via prompt_logprobs.

Exact-match canaries cannot answer this: with MTP + PP the greedy output of a
single config is not reproducible run-to-run (batch composition changes
reduction order), so text diffs measure scheduling noise, not the lm_head.

prompt_logprobs teacher-forces a fixed text through the model and reports the
logprob the model assigned to each real token. No sampling is involved, so the
result is a direct read of the logits after lm_head.

    GLM52_LMHEAD_BITS=0 ...   python lmhead_logprobs.py --out lp_bf16.json
    GLM52_LMHEAD_BITS=8 ...   python lmhead_logprobs.py --out lp_w8.json
                              python lmhead_logprobs.py --compare lp_bf16.json lp_w8.json

The comparison reports top-1 agreement (did the argmax token change?) and the
logprob deltas, which is what "lossless" has to mean for a quantized lm_head.
"""

import argparse
import json
import sys
import urllib.request

URL = "http://localhost:8000/v1/completions"

# >2048 tokens so the GLM52_DSA_FULLCG ctx gate drops to piecewise either way,
# keeping cudagraph mode from confounding an lm_head comparison.
TEXT = (
    "In a pipeline-parallel deployment the model is split by layer rather than by "
    "tensor, so each stage owns a contiguous block of transformer blocks and passes "
    "a single activation tensor to the next stage. On a PCIe gen2 x4 link this is "
    "decisive: an all-reduce per layer would saturate the bus, while one activation "
    "per boundary costs well under a millisecond. Speculative decoding complicates "
    "the picture, because the draft head lives on the final stage and its proposals "
    "must be verified against the target distribution before any token is emitted. "
    "The quantization scheme matters here too, since a weight-only kernel with group "
    "size 64 keeps the dequantization cost proportional to the number of active "
    "experts rather than the full parameter count. "
) * 22


def capture(gen_len=1):
    body = json.dumps({
        "model": "glm-5.2",
        "prompt": TEXT,
        "max_tokens": gen_len,
        "temperature": 0,
        "prompt_logprobs": 0,
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.loads(r.read().decode())
    pl = d["choices"][0]["prompt_logprobs"]
    out = []
    for entry in pl:
        if not entry:  # first token has no prediction
            continue
        # the taken token is the one flagged in the dict; keep its logprob and
        # the argmax token id, which is what a flipped logit would change.
        taken = min(entry.items(), key=lambda kv: kv[1]["rank"])
        best = max(entry.items(), key=lambda kv: kv[1]["logprob"])
        out.append({"tok": taken[0], "lp": taken[1]["logprob"], "argmax": best[0]})
    return out


def compare(fa, fb):
    a, b = json.load(open(fa)), json.load(open(fb))
    n = min(len(a), len(b))
    if len(a) != len(b):
        print(f"  WARNING: length mismatch {len(a)} vs {len(b)}, comparing first {n}")
    flips = sum(1 for i in range(n) if a[i]["argmax"] != b[i]["argmax"])
    deltas = [abs(a[i]["lp"] - b[i]["lp"]) for i in range(n)]
    deltas_sorted = sorted(deltas)
    mean = sum(deltas) / n
    p50 = deltas_sorted[n // 2]
    p99 = deltas_sorted[int(n * 0.99)]
    print(f"  tokens compared:      {n}")
    print(f"  argmax flips:         {flips}  ({flips / n * 100:.2f}%)")
    print(f"  |logprob| delta mean: {mean:.6f}")
    print(f"  |logprob| delta p50:  {p50:.6f}")
    print(f"  |logprob| delta p99:  {p99:.6f}")
    print(f"  |logprob| delta max:  {max(deltas):.6f}")
    return flips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = ap.parse_args()
    if args.compare:
        sys.exit(0 if compare(*args.compare) == 0 else 1)
    data = capture()
    print(f"captured {len(data)} token logprobs")
    if args.out:
        json.dump(data, open(args.out, "w"))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
