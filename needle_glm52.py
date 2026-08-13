#!/usr/bin/env python3
"""Needle-in-a-haystack retrieval + long-context decode probe for GLM-5.2.

Two modes over the same haystack builder:

  accuracy   sweep needle depths at a fixed context size, score exact recall
             python needle_glm52.py --tokens 262144 --depths 0,25,50,75,100

  decode     N concurrent streams over ONE shared haystack, measure decode rate
             python needle_glm52.py --tokens 262144 --conc 1,2,4,8 --gen 128

The decode mode warms the prefix cache first, so every stream starts decoding
immediately and all of them share the haystack's KV blocks -- 8 streams at 262K
would otherwise need ~2.1M KV tokens (~26 GiB/rank), far past what fits. It
therefore measures decode cost at depth, NOT KV capacity under independent
contexts.

Rate is reported as ms/step from the spec-decode draft counter, not tok/s:
with min_tokens forcing generation past a natural EOS the filler is trivially
predictable, MTP acceptance runs to ~100% and tok/s inflates (see the
glm52-vllm-serving notes). tok/s is printed too, but ms/step is the number to
compare across runs.
"""

import argparse
import json
import re
import threading
import time
import urllib.request

BASE = "http://localhost:8000"
URL = f"{BASE}/v1/chat/completions"
MODEL_DIR = "/home/user/srv/fast/models/GLM-5.2-AWQ-g64"

NEEDLE = "The maintenance passphrase for vault {vid} is {word}-{num}."
QUESTION = "What is the maintenance passphrase for vault {vid}? Reply with only the passphrase."

_FILLER_TOPICS = (
    "hydraulic pressure drifted within tolerance",
    "the relay logged a routine heartbeat",
    "coolant flow held steady on loop two",
    "the optical sensor reported nominal alignment",
    "battery bank charge cycled without fault",
    "the conveyor motor ran at reduced duty",
    "ambient humidity stayed inside the band",
    "the backup pump idled awaiting handoff",
)


_HAYSTACK_IDS: dict[tuple[int, int], list[int]] = {}


def _haystack_ids(tok, target_tokens, seed):
    """Encoded filler for a given size, built once and reused across depths.

    The token stream depends only on (target_tokens, seed) -- the needle is
    spliced in afterwards -- but this generates ~target/12*1.15 lines and
    encodes all of them, which at 1M tokens is minutes of CPU *per depth*. A
    5-depth sweep at 1M spent more time here than the server spent prefilling.
    """
    key = (target_tokens, seed)
    if key not in _HAYSTACK_IDS:
        import random

        rng = random.Random(seed)
        lines, i = [], 0
        # ~12 tokens per line; over-generate 15% so the token trim never runs short.
        while len(lines) < int(target_tokens / 12 * 1.15):
            i += 1
            lines.append(
                f"[{i:06d}] At station {rng.randint(10, 99)}, "
                f"{rng.choice(_FILLER_TOPICS)}."
            )
        _HAYSTACK_IDS[key] = tok.encode("\n".join(lines))[:target_tokens]
    return _HAYSTACK_IDS[key]


def build_haystack(tok, target_tokens, depth_pct, vault_id, word, num, seed=7):
    """Return a prompt of ~target_tokens with the needle at depth_pct."""
    ids = _haystack_ids(tok, target_tokens, seed)

    cut = int(len(ids) * depth_pct / 100)
    needle = NEEDLE.format(vid=vault_id, word=word, num=num)
    haystack = tok.decode(ids[:cut]) + "\n" + needle + "\n" + tok.decode(ids[cut:])
    return (
        "Below is a long maintenance log. Read it carefully, then answer the "
        "question at the end.\n\n"
        f"{haystack}\n\n" + QUESTION.format(vid=vault_id)
    )


def stream_once(prompt, gen, force_len=True, timeout=7200):
    """Stream one completion; return (text, usage, ttft, decode_seconds)."""
    body = {
        "model": "glm-5.2",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": gen,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if force_len:
        body["min_tokens"] = gen
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    t_first = None
    out, usage = [], None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            if d.get("usage"):
                usage = d["usage"]
            if d.get("choices"):
                delta = d["choices"][0]["delta"]
                piece = delta.get("content") or ""
                if piece or delta.get("reasoning"):
                    if t_first is None:
                        t_first = time.time()
                    out.append(piece)
    t_end = time.time()
    ttft = (t_first - t0) if t_first else float("nan")
    decode_s = (t_end - t_first) if t_first else float("nan")
    return "".join(out), usage, ttft, decode_s


def spec_counters():
    """Scrape the spec-decode counters; returns (drafts, draft_tokens, accepted)."""
    try:
        with urllib.request.urlopen(f"{BASE}/metrics", timeout=10) as r:
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


def run_accuracy(tok, args):
    print(f"=== needle accuracy: {args.tokens} tokens, depths {args.depths}")
    passed = 0
    for depth in args.depths:
        word, num, vid = "quartz", 4417 + depth, 39 + depth
        prompt = build_haystack(tok, args.tokens, depth, vid, word, num)
        secret = f"{word}-{num}"
        text, usage, ttft, _ = stream_once(prompt, args.gen, force_len=False)
        ok = secret in text
        passed += ok
        pt = usage["prompt_tokens"] if usage else -1
        print(
            f"  depth {depth:3d}%  prompt={pt:>7} tok  TTFT={ttft:6.1f}s  "
            f"{'PASS' if ok else 'FAIL'}  answer={text.strip()[:70]!r}"
        )
    print(f"=== {passed}/{len(args.depths)} passed")
    return passed == len(args.depths)


def run_decode(tok, args):
    word, num, vid = "quartz", 8801, 12
    prompt = build_haystack(tok, args.tokens, 50, vid, word, num)

    print(f"=== decode probe: {args.tokens} tokens shared prefix, gen={args.gen}")
    print("  warming prefix cache (one prefill) ...", flush=True)
    _, usage, ttft, _ = stream_once(prompt, 1, force_len=False)
    pt = usage["prompt_tokens"] if usage else -1
    print(f"  prefill {pt} tok in {ttft:.1f}s = {pt / ttft:.0f} tok/s")

    for conc in args.conc:
        before = spec_counters()
        results, lock = [], threading.Lock()

        def worker():
            r = stream_once(prompt, args.gen)
            with lock:
                results.append(r)

        threads = [threading.Thread(target=worker) for _ in range(conc)]
        t0 = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall = time.time() - t0
        after = spec_counters()

        gen_tokens = sum((u or {}).get("completion_tokens", 0) for _, u, _, _ in results)
        per_stream = [ (u or {}).get("completion_tokens", 0) / d
                       for _, u, _, d in results if d and d == d ]
        ttfts = [t for _, _, t, _ in results]
        line = (
            f"  conc {conc:2d}: aggregate {gen_tokens / wall:6.1f} tok/s  "
            f"per-stream {sum(per_stream) / len(per_stream):5.1f} tok/s  "
            f"TTFT {min(ttfts):.2f}-{max(ttfts):.2f}s"
        )
        if before and after and before[0] is not None and after[0] is not None:
            steps = after[0] - before[0]
            acc = ((after[2] - before[2]) / steps) if steps else float("nan")
            # steps counts drafts across all streams, so steps/conc is the step
            # count one stream saw. Time it against DECODE time, not wall --
            # wall includes TTFT, which is the queue wait for later streams and
            # would inflate ms/step by 2-4x at conc 8.
            decode_s = sum(d for _, _, _, d in results) / len(results)
            line += (f"  ms/step {decode_s / (steps / conc) * 1000:6.1f}"
                     f"  tok/step {acc + 1:.2f}")
        print(line, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=262144 - 512,
                    help="haystack size in tokens (leave room for template+question)")
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--depths", default="0,25,50,75,100")
    ap.add_argument("--conc", default="", help="comma list; enables decode mode")
    args = ap.parse_args()
    args.depths = [int(d) for d in args.depths.split(",") if d != ""]
    args.conc = [int(c) for c in args.conc.split(",") if c != ""]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

    if args.conc:
        run_decode(tok, args)
    else:
        run_accuracy(tok, args)


if __name__ == "__main__":
    main()
