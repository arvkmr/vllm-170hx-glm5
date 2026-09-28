#!/usr/bin/env bash
# Stop only the process whose PID this vNext checkout recorded.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PIDFILE=${VLLM_NEXT_LOGDIR:-$HERE/logs}/vllm.pid
[ -s "$PIDFILE" ] || { echo "stop: no vNext pid file"; exit 0; }
PID=$(tr -dc '0-9' <"$PIDFILE")
[ -n "$PID" ] || { echo "stop: malformed pid file" >&2; exit 1; }
if ! kill -0 "$PID" 2>/dev/null; then rm -f "$PIDFILE"; echo "stop: stale pid file removed"; exit 0; fi
CMD=$(tr '\0' ' ' <"/proc/$PID/cmdline" 2>/dev/null || true)
case "$CMD" in
  *vllm*serve*GLM-5.3-Int4-Int8Mix-AWQ-g64*) ;;
  *) echo "stop: pid $PID is not the recorded GLM-5.3 vNext server; refusing" >&2; exit 1 ;;
esac
kill -TERM -- "-$PID" 2>/dev/null || kill -TERM "$PID"
for _ in $(seq 1 60); do kill -0 "$PID" 2>/dev/null || { rm -f "$PIDFILE"; echo "stop: stopped $PID"; exit 0; }; sleep 1; done
echo "stop: pid $PID did not exit after 60s; not sending SIGKILL" >&2
exit 1
