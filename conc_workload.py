#!/usr/bin/env python3
"""Honest concurrency benchmark: diverse prompts, staggered starts.

Rebuilt 2026-08-12 (the original was lost off disk). A same-prompt harness like
bench_glm52.py overstates concurrent throughput badly: identical greedy prompts
decode in prefix-cache lockstep, so N streams share one KV path and route to one
expert set. Real traffic does not. Numbers from this script are the ones to
compare against the tuning ledger.

    python conc_workload.py 16 384              # conc 16, 384 generated tokens
    python conc_workload.py 16 384 --salt run2  # different prompt content

Prompts are unique per stream and per salt; starts are staggered 0.35 s so the
streams do not land in one lockstep batch; min_tokens is pinned so every stream
decodes exactly the same number of steps.
"""

import argparse
import json
import random
import threading
import time
import urllib.request

URL = "http://localhost:8000/v1/chat/completions"
METRICS = "http://localhost:8000/metrics"

TOPICS = (
    "a distributed log compaction strategy", "the failure modes of a hydraulic press",
    "cache coherence on a NUMA system", "tidal power generation tradeoffs",
    "designing a fault-tolerant message queue", "the metallurgy of turbine blades",
    "scheduling policies for a real-time kernel", "erosion control on steep farmland",
    "protocol design for satellite links", "the economics of grid-scale storage",
    "indexing strategies for time-series data", "thermal management in dense racks",
    "consensus under partial synchrony", "the chemistry of cement curing",
    "load balancing across heterogeneous nodes", "acoustic design of concert halls",
)


def build_prompt(i, salt, filler_words=700):
    rng = random.Random(f"{salt}:{i}")
    topic = TOPICS[i % len(TOPICS)]
    # Unique filler per stream so no two streams share a prefix-cache path.
    words = " ".join(
        f"{rng.choice(('note', 'item', 'entry', 'record'))}-{rng.randrange(10**6)}"
        for _ in range(filler_words)
    )
    return (
        f"[stream {i} salt {salt}] Reference material follows: {words}\n\n"
        f"Ignoring the reference material, write a detailed technical explanation of "
        f"{topic}. Be specific and concrete."
    )


def spec_counters():
    try:
        with urllib.request.urlopen(METRICS, timeout=10) as r:
            text = r.read().decode()
    except Exception:
        return None
    import re
    def grab(name):
        m = re.search(rf"^{re.escape(name)}(?:\{{[^}}]*\}})?\s+([0-9.e+]+)$", text, re.M)
        return float(m.group(1)) if m else None
    return grab("vllm:spec_decode_num_drafts_total"), grab("vllm:spec_decode_num_accepted_tokens_total")


def one(prompt, gen):
    body = json.dumps({
        "model": "glm-5.2",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": gen,
        "min_tokens": gen,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    d = json.loads(urllib.request.urlopen(req, timeout=3600).read().decode())
    return time.time() - t0, d["usage"]["completion_tokens"], d["usage"]["prompt_tokens"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("conc", type=int)
    ap.add_argument("gen", type=int, nargs="?", default=384)
    ap.add_argument("--salt", default="w1")
    ap.add_argument("--stagger", type=float, default=0.35)
    ap.add_argument("--filler", type=int, default=700,
                    help="filler words per prompt; ~2 tokens each. Below ~600 the "
                         "whole run stays under GLM52_DSA_FULLCG_MAXLEN (2048) and "
                         "keeps FULL cudagraphs, above it the gate drops to piecewise")
    a = ap.parse_args()

    prompts = [build_prompt(i, a.salt, a.filler) for i in range(a.conc)]
    results, lock = [], threading.Lock()

    def worker(p):
        try:
            r = one(p, a.gen)
        except Exception as e:  # noqa: BLE001
            r = ("ERR", repr(e)[:160], 0)
        with lock:
            results.append(r)

    before = spec_counters()
    threads = [threading.Thread(target=worker, args=(p,)) for p in prompts]
    t0 = time.time()
    for t in threads:
        t.start()
        time.sleep(a.stagger)
    for t in threads:
        t.join()
    wall = time.time() - t0
    after = spec_counters()

    ok = [r for r in results if r[0] != "ERR"]
    errs = [r for r in results if r[0] == "ERR"]
    if errs:
        print(f"  {len(errs)} errors, first: {errs[0][1]}")
    if not ok:
        return
    gen_tok = sum(r[1] for r in ok)
    # Discount the stagger ramp: the last stream starts conc*stagger in.
    effective = wall - a.stagger * (a.conc - 1) / 2
    line = (f"conc={a.conc:<3} gen={a.gen}  salt={a.salt}  wall={wall:6.1f}s  "
            f"decode={gen_tok / effective:7.1f} tok/s  per-stream={gen_tok / effective / len(ok):5.1f}")
    if before and after and before[0] is not None:
        steps = after[0] - before[0]
        if steps:
            line += f"  tok/step={(after[1] - before[1]) / steps + 1:.2f}"
    print(line)


if __name__ == "__main__":
    main()
