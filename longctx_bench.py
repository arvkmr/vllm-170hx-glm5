#!/usr/bin/env python3
"""Long-context probe: prefill rate (TTFT) and decode rate at a given context.

Streams a completion for a ~N-token unique prompt and reports:
  - prompt_tokens as counted by the server
  - TTFT (= prefill time for an uncached prompt; near-zero on a prefix-cache hit)
  - decode tok/s measured from first token to last (excludes prefill)

The prompt is a seeded word salad with a unique salt so runs don't hit the
prefix cache unless --salt is reused deliberately (reuse = decode-only probe).

  python longctx_bench.py --tokens 28000 --gen 128 --salt run1
"""

import argparse
import json
import random
import time
import urllib.request

URL = "http://localhost:8000/v1/chat/completions"

WORDS = (
    "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima "
    "mike november oscar papa quebec romeo sierra tango uniform victor whiskey "
    "xray yankee zulu ledger vault beacon cipher lattice quorum relay "
).split()

# Measured with the model tokenizer via --calibrate (ratio 1.001 at 28K).
TOK_PER_WORD = 1.59


def build_prompt(n_tokens, salt):
    rng = random.Random(salt)
    n_words = int(n_tokens / TOK_PER_WORD)
    body = " ".join(rng.choice(WORDS) for _ in range(n_words))
    return (
        f"[session {salt}] Below is a log of radio code words. "
        f"Read it fully, then answer.\n\n{body}\n\n"
        "How many distinct code words appear in the log? Answer with a number "
        "and one short sentence."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=28000)
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--salt", default="probe")
    ap.add_argument("--calibrate", action="store_true",
                    help="count prompt tokens with the model tokenizer and exit")
    a = ap.parse_args()

    prompt = build_prompt(a.tokens, a.salt)

    if a.calibrate:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            "/home/user/srv/fast/models/GLM-5.2-AWQ-g64", trust_remote_code=True
        )
        n = len(tok.encode(prompt))
        print(f"target={a.tokens}  actual={n}  ratio={n / a.tokens:.3f}")
        return

    body = json.dumps({
        "model": "glm-5.2",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": a.gen,
        "min_tokens": a.gen,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(
        URL, data=body, headers={"Content-Type": "application/json"}
    )

    t0 = time.time()
    t_first = None
    n_chunks = 0
    usage = None
    with urllib.request.urlopen(req, timeout=7200) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            if d.get("usage"):
                usage = d["usage"]
            if d.get("choices") and (
                d["choices"][0]["delta"].get("content")
                or d["choices"][0]["delta"].get("reasoning")
            ):
                if t_first is None:
                    t_first = time.time()
                n_chunks += 1
    t_end = time.time()

    pt = usage["prompt_tokens"] if usage else -1
    ct = usage["completion_tokens"] if usage else n_chunks
    ttft = (t_first - t0) if t_first else float("nan")
    decode_dt = (t_end - t_first) if t_first else float("nan")
    print(
        f"prompt={pt} tok  gen={ct} tok\n"
        f"TTFT={ttft:.1f}s  (prefill {pt / ttft:.0f} tok/s if uncached)\n"
        f"decode: {ct - 1} tok in {decode_dt:.1f}s = {(ct - 1) / decode_dt:.1f} tok/s"
    )


if __name__ == "__main__":
    main()
