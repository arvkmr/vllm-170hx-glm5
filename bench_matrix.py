#!/usr/bin/env python3
"""Prefill + decode matrix across context lengths, plus a concurrent long-context
case. One driver so the numbers are comparable and land in one table.

    python bench_matrix.py                    # everything
    python bench_matrix.py --only single
    python bench_matrix.py --only conc

Decode is reported as **ms/step**, from the delta in
`vllm:spec_decode_num_drafts_total` (one draft == one model step). tok/s is
printed too but is not the number to compare: min_tokens forces generation past
the natural EOS, and the post-EOS filler is trivially predictable, so MTP
acceptance runs to ~100% and inflates tok/s at long context. Acceptance is
printed so that inflation is visible rather than silent.

Prompts are salted per (size, stream, run) so nothing hits the prefix cache;
the concurrent case therefore holds N *distinct* contexts, which is the case
that actually stresses KV capacity.
"""

import argparse
import json
import random
import re
import threading
import time
import urllib.request

BASE = "http://localhost:8000"
URL = f"{BASE}/v1/chat/completions"

WORDS = (
    "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima "
    "mike november oscar papa quebec romeo sierra tango uniform victor whiskey "
    "xray yankee zulu ledger vault beacon cipher lattice quorum relay "
).split()
TOK_PER_WORD = 1.59  # calibrated in longctx_bench.py


def build_prompt(n_tokens, salt):
    rng = random.Random(salt)
    body = " ".join(rng.choice(WORDS) for _ in range(int(n_tokens / TOK_PER_WORD)))
    return (
        f"[session {salt}] Below is a log of radio code words. "
        f"Read it fully, then answer.\n\n{body}\n\n"
        "How many distinct code words appear in the log? Answer with a number "
        "and one short sentence."
    )


def spec_counters():
    """(drafts, draft_tokens, accepted); drafts == model steps."""
    try:
        with urllib.request.urlopen(f"{BASE}/metrics", timeout=15) as r:
            text = r.read().decode()
    except Exception:
        return None

    def grab(name):
        m = re.search(rf"^{re.escape(name)}(?:\{{[^}}]*\}})?\s+([0-9.e+]+)$", text, re.M)
        return float(m.group(1)) if m else None

    return (
        grab("vllm:spec_decode_num_drafts_total"),
        grab("vllm:spec_decode_num_draft_tokens_total"),
        grab("vllm:spec_decode_num_accepted_tokens_total"),
    )


def stream_once(prompt, gen, out):
    """Stream one completion; append (ttft, decode_s, prompt_tok, gen_tok) to out."""
    body = json.dumps({
        "model": "glm-5.2",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": gen,
        "min_tokens": gen,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    t_first = None
    usage = None
    n = 0
    with urllib.request.urlopen(req, timeout=14400) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            if d.get("usage"):
                usage = d["usage"]
            ch = d.get("choices")
            if ch and (ch[0]["delta"].get("content") or ch[0]["delta"].get("reasoning")):
                if t_first is None:
                    t_first = time.time()
                n += 1
    t_end = time.time()
    out.append((
        (t_first - t0) if t_first else float("nan"),
        (t_end - t_first) if t_first else float("nan"),
        usage["prompt_tokens"] if usage else -1,
        usage["completion_tokens"] if usage else n,
    ))


def run_case(label, ctx, streams, gen, salt):
    """Prefill + decode for `streams` distinct contexts of `ctx` tokens.

    Prefill and decode are timed in SEPARATE passes. Doing them in one pass is
    wrong for multi-stream long context: prefills serialize against each other,
    so the first stream's decode window ends up containing every later stream's
    prefill. That inflated 4x64K to 1659 ms/step when the true figure is far
    lower -- the harness was measuring prefill contention and calling it decode.
    Pass 1 prefills every stream (gen=1). Pass 2 re-sends the same prompts, so
    prefill is a prefix-cache hit and the timed window is decode only.
    """
    prompts = [build_prompt(ctx, f"{salt}-{ctx}-{i}") for i in range(streams)]

    # -- pass 1: prefill only, concurrent, gen=1
    warm, threads = [], []
    t0 = time.time()
    for p in prompts:
        t = threading.Thread(target=stream_once, args=(p, 1, warm))
        t.start()
        threads.append(t)
        time.sleep(0.3)
    for t in threads:
        t.join()
    prefill_wall = time.time() - t0
    ptoks = sum(r[2] for r in warm) if warm else 0

    # -- pass 2: decode, prefill now cache-hot
    before = spec_counters()
    results, threads = [], []
    t0 = time.time()
    for p in prompts:
        t = threading.Thread(target=stream_once, args=(p, gen, results))
        t.start()
        threads.append(t)
        time.sleep(0.05)
    for t in threads:
        t.join()
    wall = time.time() - t0
    after = spec_counters()

    if not results:
        print(f"  {label:<26} FAILED (no results)")
        return None
    decodes = [r[1] for r in results]
    gtoks = sum(r[3] for r in results)

    steps = acc = None
    if before and after and before[0] is not None and after[0] is not None:
        steps = after[0] - before[0]
        dtok = after[1] - before[1]
        atok = after[2] - before[2]
        acc = (atok / dtok) if dtok else float("nan")

    # Per-stream step time: all streams decode concurrently, so wall-clock
    # decode divided by steps-per-stream.
    decode_s = max(decodes)
    ms_step = (decode_s * 1000 / (steps / streams)) if steps else float("nan")
    print(
        f"  {label:<26} prompt={ptoks:>9,}  prefill={prefill_wall:6.1f}s "
        f"({ptoks / prefill_wall:>6.0f} tok/s)   decode={decode_s:5.1f}s "
        f"{ms_step:6.1f} ms/step  {gtoks / decode_s:6.1f} tok/s agg"
        + (f"  accept={acc:.0%}" if acc is not None else "")
    )
    return dict(label=label, ctx=ctx, streams=streams, prompt_tokens=ptoks,
                prefill_s=prefill_wall, prefill_tps=ptoks / prefill_wall,
                decode_s=decode_s, ms_step=ms_step,
                decode_tps=gtoks / decode_s, accept=acc, wall=wall)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["single", "conc"], default=None)
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--salt", default="m1")
    ap.add_argument("--conc-ctx", type=int, default=196608)
    ap.add_argument("--conc-n", type=int, default=4)
    ap.add_argument("--single-ctx", default="32768,262144,1048576")
    a = ap.parse_args()

    rows = []
    if a.only != "conc":
        print("single stream (prefill uncached, then decode at that context):")
        for ctx in [int(x) for x in a.single_ctx.split(",")]:
            # Leave room for generation inside max_model_len.
            r = run_case(f"{ctx // 1024}K x1", min(ctx, 1048576 - a.gen - 64),
                         1, a.gen, a.salt)
            rows.append(r)

    if a.only != "single":
        n, c = a.conc_n, a.conc_ctx
        print(f"\n{n} concurrent distinct contexts of {c:,} "
              f"({n * c:,} KV tokens total):")
        rows.append(run_case(f"{c // 1024}K x{n}", c, n, a.gen, a.salt))

    print("\njson:", json.dumps([r for r in rows if r]))


if __name__ == "__main__":
    main()
