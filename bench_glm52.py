#!/usr/bin/env python3
"""Throughput check for GLM-5.2, matching the gist's methodology.

512-token prompt / 128-token generation, greedy, thinking disabled so the token
budget is spent on measurable output rather than a reasoning trace.
"""

import argparse
import json
import threading
import time
import urllib.request

URL = "http://localhost:8000/v1/chat/completions"
MODEL = "glm-5.2"
# ~512 tokens of filler.
PROMPT = ("Summarize the following note. " + "The system records an event. " * 120)[:2600]


def one(gen_tokens):
    body = json.dumps(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": gen_tokens,
            "min_tokens": gen_tokens,  # force a fixed decode length
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    ).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=1800))
    dt = time.time() - t0
    u = r["usage"]
    return dt, u["completion_tokens"], u["prompt_tokens"]


def run(conc, gen_tokens):
    results, lock = [], threading.Lock()

    def worker():
        try:
            res = one(gen_tokens)
            with lock:
                results.append(res)
        except Exception as e:  # noqa: BLE001
            with lock:
                results.append(("ERR", repr(e)[:200], 0))

    threads = [threading.Thread(target=worker) for _ in range(conc)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t0

    errs = [r for r in results if r[0] == "ERR"]
    ok = [r for r in results if r[0] != "ERR"]
    if errs:
        raise RuntimeError(f"conc={conc}: {len(errs)} errors, first: {errs[0][1]}")
    gen = sum(r[1] for r in ok)
    pro = sum(r[2] for r in ok)
    print(
        f"  conc={conc:<3} wall={wall:6.1f}s  "
        f"decode={gen / wall:7.1f} tok/s  "
        f"total={(gen + pro) / wall:8.1f} tok/s  "
        f"per-stream={gen / wall / len(ok):5.1f} tok/s  (n={len(ok)})"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--model", default="glm-5.2")
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--conc", type=int, nargs="+", default=[1, 4, 8, 12, 16, 24, 32])
    a = ap.parse_args()
    if a.gen <= 0 or any(c <= 0 for c in a.conc):
        ap.error("generation and concurrency must be positive")
    MODEL, URL = a.model, a.base_url.rstrip("/") + "/v1/chat/completions"
    print(f"{MODEL} throughput  (prompt~512 tok, gen={a.gen} tok, greedy)")
    for c in a.conc:
        run(c, a.gen)
