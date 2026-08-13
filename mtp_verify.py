#!/usr/bin/env python3
"""Capture or compare greedy completions.

Speculative decoding is output-preserving: at temperature 0 the tokens produced
with MTP on must match those produced with MTP off, exactly. Any divergence
means the draft path is corrupting the target model's state (e.g. desynced KV),
which acceptance-rate numbers alone would not reveal.

  python mtp_verify.py save baseline.json
  python mtp_verify.py check baseline.json
"""

import json
import sys
import urllib.request

BASE = "http://localhost:8000"

PROMPTS = [
    "List the first 10 prime numbers, comma separated.",
    "Write a Python function that reverses a linked list. Code only.",
    "Explain in three sentences why the sky is blue.",
    "A farmer has 17 sheep. All but 9 run away. How many are left?",
    "Count from 1 to 30 separated by spaces.",
]


def gen(prompt, n=180):
    body = json.dumps({
        "model": "glm-5.2",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": n,
        "temperature": 0.0,
        "seed": 1234,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(f"{BASE}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=1800))
    m = r["choices"][0]["message"]
    return {
        "content": m.get("content"),
        "completion_tokens": r["usage"]["completion_tokens"],
    }


def main():
    mode, path = sys.argv[1], sys.argv[2]
    results = {p: gen(p) for p in PROMPTS}
    if mode == "save":
        json.dump(results, open(path, "w"), indent=1)
        print(f"saved {len(results)} baseline completions to {path}")
        for p, r in results.items():
            print(f"  [{r['completion_tokens']:4d} tok] {p[:50]}")
        return

    base = json.load(open(path))
    ok = True
    for p in PROMPTS:
        a, b = base[p]["content"], results[p]["content"]
        if a == b:
            print(f"  MATCH    ({results[p]['completion_tokens']:4d} tok) {p[:50]}")
        else:
            ok = False
            print(f"  MISMATCH {p[:50]}")
            print(f"    baseline: {(a or '')[:160]!r}")
            print(f"    now     : {(b or '')[:160]!r}")
    print("\nRESULT:", "identical - speculation is output-preserving" if ok
          else "DIVERGED - draft path is altering target output")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
