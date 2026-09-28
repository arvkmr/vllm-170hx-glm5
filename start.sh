#!/usr/bin/env bash
# Start only this checkout's vNext server; never match or stop the v0.26 one.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOGDIR=${VLLM_NEXT_LOGDIR:-$HERE/logs}
PIDFILE=$LOGDIR/vllm.pid
LOGFILE=$LOGDIR/serve.log
mkdir -p "$LOGDIR"

if [ -s "$PIDFILE" ]; then
  PID=$(tr -dc '0-9' <"$PIDFILE")
  if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
    echo "start: vNext server already running as pid $PID" >&2
    exit 1
  fi
fi
rm -f "$PIDFILE"
setsid "$HERE/serve.sh" >>"$LOGFILE" 2>&1 &
PID=$!
printf '%s\n' "$PID" >"$PIDFILE"
sleep 1
if ! kill -0 "$PID" 2>/dev/null; then
  echo "start: server exited during preflight; see $LOGFILE" >&2
  rm -f "$PIDFILE"
  exit 1
fi
echo "start: vNext pid $PID; log $LOGFILE; API ${HOST:-127.0.0.1}:${PORT:-8001}"
