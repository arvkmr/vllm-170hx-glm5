#!/usr/bin/env python3
"""Measure MTP draft acceptance rate across a workload.

Reads vLLM's spec-decode counters before and after running some requests, so the
rate reflects only this run rather than the server's lifetime.
"""

import json
import re
import sys
import threading
import time
import urllib.request

BASE = "http://localhost:8000"
PROMPT = ("Summarize the following note. " + "The system records an event. " * 120)[:2600]


def counters():
    raw = urllib.request.urlopen(f"{BASE}/metrics", timeout=30).read().decode()
    out = {}
    for line in raw.splitlines():
        if line.startswith("#"):
            continue
        m = re.match(r"(vllm:spec_decode\w*)\{[^}]*\}\s+([0-9.eE+-]+)", line)
        if m:
            out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(2))
    return out


def one(gen):
    body = json.dumps({
        "model": "glm-5.2",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": gen, "min_tokens": gen, "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(f"{BASE}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=1800))
    return r["usage"]["completion_tokens"]


def main(conc, gen):
    before = counters()
    toks, lock = [], threading.Lock()

    def w():
        n = one(gen)
        with lock:
            toks.append(n)

    ts = [threading.Thread(target=w) for _ in range(conc)]
    t0 = time.time()
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    wall = time.time() - t0
    after = counters()

    delta = {k: after.get(k, 0) - before.get(k, 0) for k in set(after) | set(before)}
    print(f"concurrency={conc} gen={gen}  wall={wall:.1f}s  "
          f"decode={sum(toks) / wall:.1f} tok/s  per-stream={sum(toks) / wall / conc:.1f} tok/s")
    drafted = delta.get("vllm:spec_decode_num_draft_tokens", 0) or \
        delta.get("vllm:spec_decode_num_draft_tokens_total", 0)
    accepted = delta.get("vllm:spec_decode_num_accepted_tokens", 0) or \
        delta.get("vllm:spec_decode_num_accepted_tokens_total", 0)
    if drafted:
        print(f"  drafted={drafted:.0f} accepted={accepted:.0f} "
              f"acceptance={accepted / drafted * 100:.1f}%")
    else:
        print("  no spec-decode counters moved:",
              {k: v for k, v in delta.items() if v})


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1,
         int(sys.argv[2]) if len(sys.argv) > 2 else 128)
