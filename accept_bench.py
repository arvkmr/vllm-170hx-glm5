"""Acceptance + step time over 8 prompts (C1, sequential, greedy, fixed length)."""
# usage: accept_bench.py PORT [TOKENS]   (default 384 tokens per prompt)
import json, re, sys, time, urllib.request
B = "http://127.0.0.1:" + sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 384
P = ["Write a detailed essay about the history of the Roman Empire.",
     "Write a Python implementation of a red-black tree with insert and delete.",
     "Explain how TCP congestion control works, covering slow start, AIMD and fast recovery.",
     "Write a TypeScript Express server with JWT authentication and a Postgres user table.",
     "Describe the causes and consequences of the French Revolution in depth.",
     "Implement Dijkstra's algorithm in Rust with a binary heap and explain each step.",
     "Write a long short story about a lighthouse keeper who finds a message in a bottle.",
     "Explain the architecture of a modern CPU pipeline, including branch prediction and caches."]
def m():
    t = urllib.request.urlopen(B + "/metrics").read().decode()
    g = lambda n: float(re.search(rf"^{n}(?:{{[^}}]*}})?\s+([0-9.e+]+)$", t, re.M).group(1))
    return g("vllm:spec_decode_num_drafts_total"), g("vllm:spec_decode_num_accepted_tokens_total")
tot_steps = tot_acc = tot_t = 0
for p in P:
    body = json.dumps({"model": "glm-5.3", "messages": [{"role": "user", "content": p}], "max_tokens": N,
                       "min_tokens": N, "ignore_eos": True, "temperature": 0, "stream": True}).encode()
    d0, a0 = m(); first = None
    with urllib.request.urlopen(urllib.request.Request(B + "/v1/chat/completions", body, {"Content-Type": "application/json"})) as r:
        for line in r:
            if line.startswith(b"data: {") and first is None: first = time.time()
    end = time.time(); d1, a1 = m()
    tot_steps += d1 - d0; tot_acc += a1 - a0; tot_t += end - first
print(json.dumps({"prompts": len(P), "steps": tot_steps, "tok_per_step": round(tot_acc / tot_steps + 1, 3),
                  "ms_per_step": round(tot_t * 1000 / tot_steps, 2), "tok_s": round((tot_acc + tot_steps) / tot_t, 2)}))
