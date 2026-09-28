#!/usr/bin/env python3
"""Greedy repeat probe: the same arithmetic prompt N times, report every
token id 0 ('!') with its context and logprob, plus distinct outputs.

    python3 probe_bang.py PORT N
"""

import json
import sys
import urllib.request
from collections import Counter

port = sys.argv[1] if len(sys.argv) > 1 else "8000"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
url = f"http://127.0.0.1:{port}/v1/chat/completions"
prompt = "Compute 8472 * 391 and then subtract 17. Show each step."
outs = []
bang_runs = 0
for i in range(n):
    body = {
        "model": "glm-5.3",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1500,
        "temperature": 0,
        "seed": 0,
        "logprobs": True,
        "return_token_ids": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        url, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    d = json.load(urllib.request.urlopen(req, timeout=900))
    c = d["choices"][0]
    ids = c.get("token_ids") or []
    lps = [x["logprob"] for x in c["logprobs"]["content"]]
    toks = [x["token"] for x in c["logprobs"]["content"]]
    outs.append(tuple(ids))
    hits = [k for k, t in enumerate(ids) if t == 0]
    if hits:
        bang_runs += 1
        for k in hits:
            ctx = "".join(toks[max(0, k - 15) : k + 5])
            print(f"run {i}: token 0 at {k}, logprob {lps[k]:.3f}: {ctx!r}", flush=True)
print(f"runs={n} distinct={len(set(outs))} runs_with_token0={bang_runs}")
base = Counter(outs).most_common(1)[0][0]
for i, o in enumerate(outs):
    if o != base:
        k = next((j for j, (a, b) in enumerate(zip(base, o)) if a != b), None)
        print(f"  run {i} diverges from the majority at token {k}")
