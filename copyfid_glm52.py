#!/usr/bin/env python3
"""Verbatim-copy fidelity probe for GLM-5.2.

Reproducing an exact string from earlier in the context is the operation behind
tool-call arguments (paths, flags, env names), so it is the sharpest available
test for "the harness emits typos". Unlike a needle test -- which only asks
whether one fact is retrievable -- this asks whether ten exact strings survive
byte-for-byte, which is a much stricter read on the KV representation.

Each request buries a 10-item file manifest behind filler and asks for it back
verbatim. A response is "bad" if any item differs by a single character.

Filler is made unique per request (`Doc<uid> note <i>.`) so prefix caching
cannot serve one request's KV to another: every request genuinely re-reads its
own cache. This matters -- with identical prompts the prefix cache hit rate hits
95% and the probe stops measuring the KV path at all.

Modes:
  ctx        copy fidelity vs context length, temperature 0 (the sampler cannot
             contribute at temp 0, so every failure is a real retrieval failure)
  temps      sampling arms at a fixed context (isolates untruncated-tail draws)
  occupancy  fixed context, low vs high KV occupancy (catches address-dependent
             bugs: int32 overflow in block indexing, paging, eviction)
  logprobs   re-runs with logprobs and dumps the actual sampled token at each
             corruption site -- distinguishes a tail draw (healthy confident
             distribution, emitted token at p~1e-5) from NaN/degenerate logits

Examples:
    python copyfid_glm52.py ctx
    python copyfid_glm52.py ctx --contexts 120000 --n 12
    python copyfid_glm52.py temps --ctx 120000
    python copyfid_glm52.py occupancy
    python copyfid_glm52.py logprobs --ctx 120000 --n 64
"""

import argparse
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

URL = "http://localhost:8000/v1/completions"
MODEL = "glm-5.3"

# Deliberately awkward strings: underscores, digits, mixed case, and near-miss
# neighbours of each other. Easy strings are reproduced correctly even from a
# badly degraded cache, so they measure nothing.
NAMES = [
    "patch_dsa_fullwidth_capture.py",
    "glm52_idx_prefill_v2.py",
    "moe_gemv_marlin_bind.cpp",
    "patch_mqa_store_clamp.py",
    "KV_CACHE_MEM=21474836480",
    "GLM52_PP_DECODE_ADAPTIVE",
    "TRITON_MLA_SPARSE",
    "fp8_ds_mla",
    "sm_80",
    "AWQ-g64",
]
BLOCK = "\n".join(f"{i + 1}. {n}" for i, n in enumerate(NAMES))

SENT = ("The scheduler admits a batch, the pipeline stages advance in lockstep, "
        "and each rank forwards its activation slice to the next boundary. ")
TOKENS_PER_REP = 33  # measured against /v1/completions usage.prompt_tokens


def make_prompt(uid, ctx_tokens, placement="top"):
    reps = max(1, ctx_tokens // TOKENS_PER_REP)
    filler = [f"Doc{uid} note {i}. {SENT}" for i in range(reps)]
    manifest = "Here is a file manifest:\n\n" + BLOCK + "\n\n"
    # placement: where the manifest sits relative to the filler. fp8 KV error
    # is position-independent; indexer selection quality and the model's own
    # long-range retrieval are not -- so top vs end at equal total context
    # separates those hypotheses.
    if placement == "top":
        body = manifest + "".join(filler)
    elif placement == "mid":
        h = len(filler) // 2
        body = "".join(filler[:h]) + "\n\n" + manifest + "".join(filler[h:])
    elif placement == "end":
        # ~4K tokens of filler after the manifest so it is recent but not
        # inside the tail the model is currently generating from.
        k = max(0, len(filler) - 4000 // TOKENS_PER_REP)
        body = "".join(filler[:k]) + "\n\n" + manifest + "".join(filler[k:])
    else:
        raise ValueError(placement)
    where = "the top of" if placement == "top" else "earlier in"
    return (body +
            f"\n\nReproduce the file manifest from {where} this document "
            "exactly, character for character, as a numbered list.\n\n"
            "Here is a file manifest:\n\n")


def has_ctrl(s):
    return any(ord(c) < 9 or (13 < ord(c) < 32) for c in s)


def score(txt):
    """Return the list of (expected, got) mismatches for one response."""
    got = [ln.split(". ", 1)[-1].strip()
           for ln in txt.strip().splitlines() if ". " in ln][:len(NAMES)]
    errs = [(a, b) for a, b in zip(NAMES, got) if a != b]
    if len(got) < len(NAMES):
        errs.append((f"<only {len(got)}/{len(NAMES)} lines>", txt.strip()[:60]))
    return errs


def request(uid, ctx_tokens, params, logprobs=None, placement="top"):
    body = {"model": MODEL, "prompt": make_prompt(uid, ctx_tokens, placement),
            "max_tokens": 220, **params}
    if logprobs is not None:
        body["logprobs"] = logprobs
    req = urllib.request.Request(URL, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.loads(r.read())


def run_batch(uids, ctx_tokens, params, conc, placement="top"):
    def one(uid):
        d = request(uid, ctx_tokens, params, placement=placement)
        txt = d["choices"][0]["text"]
        return {"uid": uid, "ptok": d["usage"]["prompt_tokens"],
                "errs": score(txt), "ctrl": has_ctrl(txt)}
    # Stream a line per completed request so long passes are watchable
    # live (tail -f / Monitor on 'req uid=') instead of only reporting at
    # pass end. Results are re-ordered to match `uids` for report().
    with ThreadPoolExecutor(conc) as ex:
        futs = {ex.submit(one, uid): uid for uid in uids}
        by_uid = {}
        for f in as_completed(futs):
            r = f.result()
            by_uid[r["uid"]] = r
            if r["errs"]:
                a, b = r["errs"][0]
                print(f"  req uid={r['uid']} BAD  want {a!r} got {b!r}",
                      flush=True)
            else:
                print(f"  req uid={r['uid']} ok", flush=True)
        return [by_uid[u] for u in uids]


def report(tag, res):
    bad = sum(1 for r in res if r["errs"])
    ctrl = sum(1 for r in res if r["ctrl"])
    ptok = res[0]["ptok"] if res else 0
    pct = 100.0 * bad / len(res) if res else 0.0
    print(f"{tag}: {bad}/{len(res)} bad ({pct:.1f}%), {ctrl} control-byte, "
          f"ctx={ptok}", flush=True)
    seen = []
    for r in res:
        for a, b in r["errs"][:2]:
            if (a, b) not in seen:
                seen.append((a, b))
                print(f"        want {a!r}  got {b!r}", flush=True)
    bad_uids = [r["uid"] for r in res if r["errs"]]
    if bad_uids:
        # Failures are deterministic per-prompt at temp 0, so a failing uid is
        # a reusable single-request reproducer: `one --uid <uid> --ctx <ctx>`.
        print(f"        failing uids: {bad_uids}", flush=True)


def mode_ctx(a):
    # temp 0: no sampling contribution, so any failure is a real retrieval bug
    # NOT hash(): string hashing is randomized per process, and uids must be
    # reproducible so a failing uid works as a canary in a later invocation.
    pl_off = {"top": 0, "mid": 1, "end": 2}[a.placement]
    for ctx in a.contexts:
        res = run_batch([(ctx * 31 + pl_off * 104729 + i * 7919
                          + a.uid_offset) % 100000
                         for i in range(a.n)],
                        ctx, {"temperature": 0.0, "seed": 4242}, a.conc,
                        placement=a.placement)
        report(f"ctx~{ctx:<7d} {a.placement:3s} temp=0", res)


def mode_one(a):
    # Single-request canary: rerun one known-failing uid at temp 0. With the
    # deterministic failures this is the minimal reproducer -- ~1 min instead
    # of a 20-request batch, and prefix caching makes same-server reruns fast.
    d = request(a.uid, a.ctx, {"temperature": 0.0, "seed": 4242},
                placement=a.placement)
    txt = d["choices"][0]["text"]
    errs = score(txt)
    print(f"uid={a.uid} ctx={d['usage']['prompt_tokens']} "
          f"{'FAIL' if errs else 'PASS'}", flush=True)
    for want, got in errs:
        print(f"        want {want!r}  got {got!r}", flush=True)
    return 1 if errs else 0


def mode_temps(a):
    arms = [
        ("temp=0.0", {"temperature": 0.0}),
        ("temp=1.0 top_p=1.0 (untruncated)", {"temperature": 1.0, "top_p": 1.0}),
        ("temp=1.0 top_p=0.95", {"temperature": 1.0, "top_p": 0.95}),
        ("temp=0.7 top_p=0.95", {"temperature": 0.7, "top_p": 0.95}),
    ]
    for label, params in arms:
        res = run_batch([hash((label, i)) % 100000 for i in range(a.n)],
                        a.ctx, dict(params, seed=4242), a.conc)
        report(f"{label:34s}", res)


def mode_occupancy(a):
    # Context is held fixed; only the number of resident sequences changes, so
    # block addresses climb in the concurrent phase. An int32 overflow in block
    # indexing fires here and nowhere else.
    p = {"temperature": 0.0, "seed": 777}
    res = run_batch(range(4), a.ctx, p, 1)
    report("serial (low occupancy)   temp=0", res)
    res = run_batch(range(100, 100 + a.conc * 2), a.ctx, p, a.conc)
    report(f"{a.conc}-way (high occupancy) temp=0", res)
    print("cross-check occupancy in the server log:\n"
          "  grep -oE 'GPU KV cache usage: [0-9.]+%' logs/glm52.latest.log | tail",
          flush=True)


def mode_logprobs(a):
    """Dump the actual sampled token at each corruption site.

    tail draw  -> emitted token has logprob ~ -8..-20 while top-1 sits at ~-0.0
    NaN / bad  -> null/NaN logprobs, flat top-5, or token id 0
    """
    def one(i):
        d = request(90000 + i, a.ctx,
                    {"temperature": 1.0, "top_p": 1.0, "seed": 300000 + i},
                    logprobs=5)
        ch = d["choices"][0]
        txt = ch["text"]
        errs, ctrl = score(txt), has_ctrl(txt)
        if not errs and not ctrl:
            return None
        lp = ch.get("logprobs") or {}
        toks, tlp = lp.get("tokens", []), lp.get("token_logprobs", [])
        top = lp.get("top_logprobs") or []
        hits = []
        for j, (t, p) in enumerate(zip(toks, tlp)):
            raw = t if isinstance(t, str) else str(t)
            if has_ctrl(raw) or p is None or p != p or p < -7.0:
                alt = top[j] if j < len(top) and top[j] else {}
                hits.append((j, repr(raw), p,
                             {repr(k): v for k, v in list(alt.items())[:5]}))
        return {"seed": 300000 + i, "ctrl": ctrl, "errs": errs[:2], "hits": hits[:6]}

    with ThreadPoolExecutor(a.conc) as ex:
        res = [r for r in ex.map(one, range(a.n)) if r]
    print(f"{len(res)}/{a.n} flagged, "
          f"{sum(1 for r in res if r['ctrl'])} with control bytes", flush=True)
    for r in res:
        mark = "CONTROL-BYTE " if r["ctrl"] else ""
        print(f"\n=== {mark}seed={r['seed']} errs={r['errs']}")
        for pos, tok, p, top5 in r["hits"]:
            print(f"    pos={pos:3d} tok={tok:24s} logprob={p}")
            print(f"         top5={top5}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    p = sub.add_parser("ctx")
    p.add_argument("--contexts", type=int, nargs="+",
                   default=[8000, 40000, 80000, 120000])
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--conc", type=int, default=3)
    p.add_argument("--placement", choices=["top", "mid", "end"], default="top")
    # Distinct-but-deterministic uid sets for repeated passes on one server:
    # identical uids would prefix-cache-hit on pass 2+ (no prefill => the
    # decode+prefill co-flight trigger never fires). Failing uids stay
    # reproducible via `one --uid <uid>` regardless of offset.
    p.add_argument("--uid-offset", type=int, default=0)
    p.set_defaults(fn=mode_ctx)

    p = sub.add_parser("one")
    p.add_argument("--uid", type=int, required=True)
    p.add_argument("--ctx", type=int, default=120000)
    p.add_argument("--placement", choices=["top", "mid", "end"], default="top")
    p.set_defaults(fn=mode_one)

    p = sub.add_parser("temps")
    p.add_argument("--ctx", type=int, default=5000)
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--conc", type=int, default=16)
    p.set_defaults(fn=mode_temps)

    p = sub.add_parser("occupancy")
    p.add_argument("--ctx", type=int, default=120000)
    p.add_argument("--conc", type=int, default=8)
    p.set_defaults(fn=mode_occupancy)

    p = sub.add_parser("logprobs")
    p.add_argument("--ctx", type=int, default=5000)
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--conc", type=int, default=16)
    p.set_defaults(fn=mode_logprobs)

    args = ap.parse_args()
    sys.exit(args.fn(args))
