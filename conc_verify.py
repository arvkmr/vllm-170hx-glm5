#!/usr/bin/env python3
"""Concurrency-correctness check for PP+spec cross-batch pipelining.

Runs the same greedy prompts twice against the live server:
  1. sequentially (one at a time -- never more than one batch in flight, so
     this path is identical with or without cross-batch pipelining), then
  2. all concurrently (exercises multiple in-flight batches).

KV corruption from a scheduling bug shows up as garbage from token ~2 onward
(seen in earlier sessions: "1ames...", "9opro..."). Exact logprob ties may
legitimately flip under different batch shapes; those diverge at a single
plausible word, not into garbage. Report both, judge accordingly.
"""

import difflib
import json
import threading
import urllib.request

BASE = "http://localhost:8000/v1/chat/completions"

PROMPTS = [
    "List the first 10 prime numbers, comma separated.",
    "Write a Python function that reverses a linked list. Code only.",
    "Explain in three sentences why the sky is blue.",
    "A farmer has 17 sheep. All but 9 run away. How many are left?",
    "Count from 1 to 30 separated by spaces.",
    "Name the capitals of France, Japan, and Brazil.",
    "Write a haiku about winter.",
    "What is 17 * 23? Show the arithmetic.",
]


def gen(prompt, n=160):
    body = json.dumps({
        "model": "glm-5.2",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": n,
        "temperature": 0.0,
        "seed": 1234,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(BASE, data=body,
                                 headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=1800))
    return r["choices"][0]["message"].get("content") or ""


def main():
    print("sequential pass...")
    seq = [gen(p) for p in PROMPTS]

    print("concurrent pass...")
    conc = [None] * len(PROMPTS)

    def worker(i):
        conc[i] = gen(PROMPTS[i])

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(len(PROMPTS))]
    for t in ts:
        t.start()
    for t in ts:
        t.join()

    bad = 0
    for i, p in enumerate(PROMPTS):
        if seq[i] == conc[i]:
            print(f"  MATCH    {p[:45]}")
        else:
            sm = difflib.SequenceMatcher(None, seq[i], conc[i])
            ratio = sm.ratio()
            tag = "TIE-FLIP?" if ratio > 0.85 else "DIVERGED"
            if ratio <= 0.85:
                bad += 1
            print(f"  {tag} (sim={ratio:.2f}) {p[:45]}")
            print(f"    seq : {seq[i][:140]!r}")
            print(f"    conc: {conc[i][:140]!r}")
    print("\nRESULT:", "OK - no corruption under concurrency" if bad == 0
          else f"{bad} prompts look CORRUPTED")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
