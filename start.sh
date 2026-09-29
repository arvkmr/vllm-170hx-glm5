#!/usr/bin/env bash
# Start only this checkout's vNext server; never match or stop the v0.26 one.
#
# Defaults to the production server: PROFILE=agent (0.0.0.0:8000, 1M context,
# FULL decode graphs) with the original AWQ checkpoint via serve.sh. Override
# PROFILE=smoke for the 127.0.0.1:8001 validation run, or SERVE_SCRIPT for
# another checkpoint (e.g. SERVE_SCRIPT=serve-uncensored.sh).
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOGDIR=${VLLM_NEXT_LOGDIR:-$HERE/logs}
PIDFILE=$LOGDIR/vllm.pid
LOGFILE=$LOGDIR/serve.log
mkdir -p "$LOGDIR"

export PROFILE=${PROFILE:-agent}
# A cold-page-cache load of the 419 GB checkpoint takes ~16 min; the 600 s
# default killed a production restart once.
export VLLM_ENGINE_READY_TIMEOUT_S=${VLLM_ENGINE_READY_TIMEOUT_S:-1800}
SERVE_SCRIPT=${SERVE_SCRIPT:-serve.sh}

if [ -s "$PIDFILE" ]; then
  PID=$(tr -dc '0-9' <"$PIDFILE")
  if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
    echo "start: vNext server already running as pid $PID" >&2
    exit 1
  fi
fi
rm -f "$PIDFILE"
setsid "$HERE/$SERVE_SCRIPT" >>"$LOGFILE" 2>&1 &
PID=$!
printf '%s\n' "$PID" >"$PIDFILE"
sleep 1
if ! kill -0 "$PID" 2>/dev/null; then
  echo "start: server exited during preflight; see $LOGFILE" >&2
  rm -f "$PIDFILE"
  exit 1
fi
# serve.sh's per-profile bind defaults: agent listens on 0.0.0.0:8000.
case "$PROFILE" in
  agent) DEF_HOST=0.0.0.0 DEF_PORT=8000 ;;
  *) DEF_HOST=127.0.0.1 DEF_PORT=8001 ;;
esac
echo "start: vNext pid $PID; profile $PROFILE via $SERVE_SCRIPT; log $LOGFILE; API ${HOST:-$DEF_HOST}:${PORT:-$DEF_PORT}"
