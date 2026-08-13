#!/usr/bin/env bash
# Quick functional check for the GLM-5.2 server. Prints the reasoning trace and
# the answer separately so a broken --reasoning-parser is obvious.
set -euo pipefail
PORT=${PORT:-8000}
PROMPT=${1:-"What is 17 multiplied by 24? Explain briefly."}

curl -s "localhost:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "$(python3 -c '
import json,sys
print(json.dumps({
  "model": "glm-5.2",
  "messages": [{"role":"user","content": sys.argv[1]}],
  "max_tokens": 400,
  "temperature": 0.6,
}))' "$PROMPT")" \
| python3 -c '
import json,sys
r = json.load(sys.stdin)
if "choices" not in r:
    print("ERROR:", json.dumps(r)[:2000]); sys.exit(1)
m = r["choices"][0]["message"]
# vLLM 0.26.0 names this field "reasoning"; older builds used "reasoning_content".
rc = m.get("reasoning") or m.get("reasoning_content")
print("--- reasoning ---"); print((rc or "(none)")[:1200])
print("--- content ---");           print((m.get("content") or "(none)")[:1200])
print("--- finish_reason:", r["choices"][0]["finish_reason"], "| usage:", r.get("usage"))
'
