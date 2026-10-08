#!/usr/bin/env bash
set -euo pipefail
BASE=${BASE_URL:-http://127.0.0.1:${PORT:-8001}}
curl -fsS "$BASE/health" >/dev/null
BODY=$(curl -fsS "$BASE/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.3","messages":[{"role":"user","content":"Reply with exactly: vnext-ok"}],"temperature":0,"max_tokens":32}')
python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["choices"][0]["message"].get("content") or d["choices"][0]["message"])' <<<"$BODY"
echo "smoke: API healthy and chat request completed"
