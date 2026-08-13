#!/usr/bin/env python3
"""Greedy canaries: capture deterministic outputs so two server configs can be
diffed token-for-token.

Used to check that lm_head quantization (GLM52_LMHEAD_BITS) is lossless:

    GLM52_LMHEAD_BITS=8 ... ./start_glm52.sh
    python canary_glm52.py --out canary_w8.json
    GLM52_LMHEAD_BITS=0 ... ./start_glm52.sh
    python canary_glm52.py --out canary_bf16.json
    python canary_glm52.py --compare canary_w8.json canary_bf16.json

Prompts lean on code and arithmetic, where a single flipped logit is visible
rather than absorbed by paraphrase.
"""

import argparse
import json
import sys
import urllib.request

URL = "http://localhost:8000/v1/chat/completions"

PROMPTS = [
    ("code-quicksort", "Write a Python function `quicksort(xs)` using Lomuto partition. Code only."),
    ("code-binsearch", "Write a C function `int bsearch_lo(const int*a,int n,int x)` returning the "
                       "lowest index with a[i]>=x. Code only."),
    ("code-sql", "Write a SQL query returning the top 5 customers by total order value in 2024, "
                 "with ties broken by customer_id. Query only."),
    ("math-arith", "Compute 8472 * 391 and then subtract 17. Show each step."),
    ("math-word", "A tank holds 4200 L. It drains at 37 L/min for 45 min, then is refilled at "
                  "88 L/min for 12 min. How much is in it? Show your work."),
    ("recall-list", "List the first 15 prime numbers above 200, comma separated, nothing else."),
    ("json-strict", 'Return a JSON object with keys "alpha" (int 42), "beta" (list of the strings '
                    '"x","y"), "gamma" (float 3.5). JSON only.'),
    ("prose", "In exactly three sentences, explain why pipeline parallelism suits a PCIe-limited "
              "multi-GPU box better than tensor parallelism."),
]


def ask(prompt, gen=384):
    body = json.dumps({
        "model": "glm-5.2",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": gen,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.loads(r.read().decode())
    return d["choices"][0]["message"]["content"], d["usage"]["completion_tokens"]


def compare(fa, fb):
    a = json.load(open(fa))
    b = json.load(open(fb))
    same = 0
    for name in a:
        if name not in b:
            print(f"  {name:16s} MISSING in {fb}")
            continue
        if a[name]["text"] == b[name]["text"]:
            same += 1
            print(f"  {name:16s} identical ({a[name]['tokens']} tok)")
        else:
            ta, tb = a[name]["text"], b[name]["text"]
            i = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), min(len(ta), len(tb)))
            print(f"  {name:16s} DIVERGES at char {i}")
            print(f"      {fa}: ...{ta[max(0, i - 40):i + 60]!r}")
            print(f"      {fb}: ...{tb[max(0, i - 40):i + 60]!r}")
    print(f"=== {same}/{len(a)} canaries identical")
    return same == len(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    ap.add_argument("--gen", type=int, default=384)
    args = ap.parse_args()

    if args.compare:
        sys.exit(0 if compare(*args.compare) else 1)

    results = {}
    for name, prompt in PROMPTS:
        text, ntok = ask(prompt, args.gen)
        results[name] = {"prompt": prompt, "text": text, "tokens": ntok}
        print(f"  {name:16s} {ntok:4d} tok  {text.strip()[:60]!r}", flush=True)
    if args.out:
        json.dump(results, open(args.out, "w"), indent=1)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
