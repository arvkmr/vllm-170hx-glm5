#!/usr/bin/env python3
"""Decode and prefill throughput at a given concurrency against a live server.

usage: bench_conc.py PORT MODE C [TOKENS]
  MODE decode : C concurrent streams, TOKENS (default 512) generated each,
                ignore_eos, greedy. Reports per-stream decode tok/s (after the
                first token), aggregate output tok/s over the wall time, and
                spec-decode tokens/step from /metrics.
  MODE prefill: C concurrent requests of ~TOKENS (default 8192) prompt tokens,
                max_tokens=1, each with a unique prefix so prefix caching never
                hits. Reports TTFT and aggregate prompt tok/s.
Prints one JSON line.
"""
import json, random, re, sys, threading, time, urllib.request

PORT, MODE, C = sys.argv[1], sys.argv[2], int(sys.argv[3])
B = f"http://127.0.0.1:{PORT}"

DECODE_PROMPTS = [
    "Write a detailed essay about the history of the Roman Empire.",
    "Write a Python implementation of a red-black tree with insert and delete.",
    "Explain how TCP congestion control works, covering slow start, AIMD and fast recovery.",
    "Write a TypeScript Express server with JWT authentication and a Postgres user table.",
    "Describe the causes and consequences of the French Revolution in depth.",
    "Implement Dijkstra's algorithm in Rust with a binary heap and explain each step.",
    "Write a long short story about a lighthouse keeper who finds a message in a bottle.",
    "Explain the architecture of a modern CPU pipeline, including branch prediction and caches.",
]


def metrics():
    t = urllib.request.urlopen(B + "/metrics").read().decode()
    g = lambda n: float(re.search(rf"^{n}(?:{{[^}}]*}})?\s+([0-9.e+]+)$", t, re.M).group(1))
    return g("vllm:spec_decode_num_drafts_total"), g("vllm:spec_decode_num_accepted_tokens_total")


def post(body):
    return urllib.request.urlopen(urllib.request.Request(
        B + "/v1/chat/completions", json.dumps(body).encode(),
        {"Content-Type": "application/json"}), timeout=7200)


def decode_one(i, n, out):
    body = {"model": "glm-5.3", "messages": [{"role": "user", "content": DECODE_PROMPTS[i % 8]}],
            "max_tokens": n, "min_tokens": n, "ignore_eos": True, "temperature": 0,
            "stream": True, "stream_options": {"include_usage": True}}
    t0 = time.time(); first = None; toks = None
    with post(body) as r:
        for line in r:
            if not line.startswith(b"data: {"):
                continue
            if first is None:
                first = time.time()
            d = json.loads(line[6:])
            if d.get("usage"):
                toks = d["usage"]["completion_tokens"]
    end = time.time()
    out[i] = dict(start=t0, first=first, end=end, toks=toks)


def filler(seed, n_words):
    rnd = random.Random(seed)
    words = ["alpha", "river", "stone", "quantum", "ledger", "violet", "harbor", "matrix", "orbit",
             "canvas", "signal", "timber", "vector", "meadow", "cipher", "lantern", "falcon", "delta",
             "ember", "summit", "pixel", "glacier", "copper", "willow", "nebula", "anchor", "saffron"]
    return " ".join(rnd.choice(words) + str(rnd.randint(0, 999)) for _ in range(n_words))


def prefill_one(i, n, out, tag):
    # ~3.1 tokens per "wordNNN" item under the GLM tokenizer; usage gives the exact count.
    text = f"[{tag}-{i}-{random.random()}] " + filler(hash((tag, i)), int(n / 3.1))
    body = {"model": "glm-5.3", "messages": [{"role": "user", "content": text + "\nReply with one word."}],
            "max_tokens": 1, "temperature": 0}
    t0 = time.time()
    with post(body) as r:
        d = json.load(r)
    out[i] = dict(start=t0, end=time.time(), ptoks=d["usage"]["prompt_tokens"])


def run(target, args):
    out = [None] * C
    ts = [threading.Thread(target=target, args=(i, *args, out) if target is decode_one else (i, *args, out, time.time()))
          for i in range(C)]
    [t.start() for t in ts]; [t.join() for t in ts]
    return out


if MODE == "decode":
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 512
    d0, a0 = metrics()
    out = run(decode_one, (n,))
    d1, a1 = metrics()
    wall = max(o["end"] for o in out) - min(o["start"] for o in out)
    per = [(o["toks"] - 1) / (o["end"] - o["first"]) for o in out]
    ttft = [o["first"] - o["start"] for o in out]
    tot = sum(o["toks"] for o in out)
    print(json.dumps(dict(mode="decode", c=C, tokens_each=n, per_stream_toks=round(sum(per) / C, 1),
                          per_stream_min=round(min(per), 1), aggregate_toks=round(tot / wall, 1),
                          ttft_mean=round(sum(ttft) / C, 2),
                          tok_per_step=round((a1 - a0) / (d1 - d0) + 1, 2) if d1 > d0 else None)))
else:
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 8192
    out = run(prefill_one, (n,))
    wall = max(o["end"] for o in out) - min(o["start"] for o in out)
    lat = [o["end"] - o["start"] for o in out]
    tot = sum(o["ptoks"] for o in out)
    print(json.dumps(dict(mode="prefill", c=C, prompt_tokens_each=round(tot / C), ttft_mean=round(sum(lat) / C, 2),
                          ttft_max=round(max(lat), 2), aggregate_prefill_toks=round(tot / wall, 1))))
