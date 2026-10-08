#!/usr/bin/env bash
# Stop only the process whose PID this vNext checkout recorded.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# The proxy on :8000 stays up across restarts (it returns 502 while GLM is down).
# Free GPU 8 too: whatever runs next (GLM, or an eval server via serve.sh)
# refuses to launch while the Breeze TTS server holds it.
TTS_DIR=${TTS_DIR:-$HOME/breeze_tts}
if [ -x "$TTS_DIR/stop-tts.sh" ]; then "$TTS_DIR/stop-tts.sh"; fi
PIDFILE=${VLLM_NEXT_LOGDIR:-$HERE/logs}/vllm.pid
[ -s "$PIDFILE" ] || { echo "stop: no vNext pid file"; exit 0; }
PID=$(tr -dc '0-9' <"$PIDFILE")
[ -n "$PID" ] || { echo "stop: malformed pid file" >&2; exit 1; }
if ! kill -0 "$PID" 2>/dev/null; then rm -f "$PIDFILE"; echo "stop: stale pid file removed"; exit 0; fi
CMD=$(tr '\0' ' ' <"/proc/$PID/cmdline" 2>/dev/null || true)
case "$CMD" in
  *vllm*serve*GLM-5.3-Int4-Int8Mix-AWQ-g64*|*vllm*serve*GLM-5.3-UNCENSORED-Int4-Int8Mix-AWQ-g64*) ;;
  *) echo "stop: pid $PID is not the recorded GLM-5.3 vNext server; refusing" >&2; exit 1 ;;
esac
kill -TERM -- "-$PID" 2>/dev/null || kill -TERM "$PID"
# Wait for the whole process group, not just the API server: its PP workers
# can outlive it by seconds while still holding GPU memory, and a start in
# that window fails preflight --require-idle (seen 2026-09-30).
for _ in $(seq 1 60); do
  if ! kill -0 "$PID" 2>/dev/null && ! pgrep -g "$PID" >/dev/null 2>&1; then
    rm -f "$PIDFILE"; echo "stop: stopped $PID"; exit 0
  fi
  sleep 1
done
echo "stop: pid $PID did not exit after 60s; not sending SIGKILL" >&2
exit 1
