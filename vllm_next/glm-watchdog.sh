#!/usr/bin/env bash
# Restart the production GLM server (127.0.0.1:8002) after a crash.
#
# Run from cron every minute. A crash leaves logs/vllm.pid pointing at a dead
# process; a deliberate ./stop.sh removes the pid file, so a clean stop is
# never undone. Only the agent profile on port 8002 is restarted (eval/smoke
# servers on 8001 are left alone), on the same checkpoint that crashed, at most
# MAX_RESTARTS times per WINDOW_S so a failing GPU cannot crash-loop.
set -uo pipefail
DIR=${VLLM_NEXT_DIR:-$HOME/vllm_install/vllm_next}
LOGDIR=$DIR/logs
PIDFILE=$LOGDIR/vllm.pid
SERVELOG=$LOGDIR/serve.log
WLOG=$LOGDIR/watchdog.log
STATE=$LOGDIR/watchdog.restarts
MAX_RESTARTS=${MAX_RESTARTS:-3}
WINDOW_S=${WINDOW_S:-21600}

exec 9>"$LOGDIR/watchdog.lock"
flock -n 9 || exit 0
log() { echo "$(date -u +%FT%TZ) $*" >>"$WLOG"; }

[ -s "$PIDFILE" ] || exit 0                       # stopped on purpose, or never started
PID=$(tr -dc '0-9' <"$PIDFILE")
[ -n "$PID" ] && kill -0 "$PID" 2>/dev/null && exit 0   # running (or still loading)
# stop.sh keeps the pid file until the whole process group is gone (up to
# 60 s after the API server exits); never mistake that window for a crash.
pgrep -f '(^|/)stop\.sh( |$)' >/dev/null && exit 0
[ -n "$PID" ] && pgrep -g "$PID" >/dev/null && exit 0

# Dead, but the pid file is still there: a crash. What was it running?
ARGS=$(grep "non-default args" "$SERVELOG" | tail -1)
case "$ARGS" in
  *"'port': 8002"*) ;;
  *) log "pid $PID dead but last launch was not the 8002 production server; not restarting"; rm -f "$PIDFILE"; exit 0 ;;
esac
case "$ARGS" in
  *GLM-5.3-UNCENSORED-Int4-Int8Mix-AWQ-g64*) SCRIPT=serve-uncensored.sh ;;
  *GLM-5.3-Int4-Int8Mix-AWQ-g64*) SCRIPT=serve.sh ;;
  *) log "pid $PID dead; unrecognised model in last launch; not restarting"; exit 0 ;;
esac

NOW=$(date +%s)
RECENT=$(awk -v t=$((NOW - WINDOW_S)) '$1 > t' "$STATE" 2>/dev/null | wc -l)
if [ "$RECENT" -ge "$MAX_RESTARTS" ]; then
  # Log once, then stay quiet until a human intervenes.
  [ -e "$LOGDIR/watchdog.gaveup" ] || { log "GLM crashed again; $RECENT restarts in the last $((WINDOW_S / 3600))h, giving up. Investigate, then ./start.sh"; touch "$LOGDIR/watchdog.gaveup"; }
  exit 0
fi
rm -f "$LOGDIR/watchdog.gaveup"

# Record why it died, for later diagnosis.
CAUSE=$(grep -E "illegal memory access|Xid|CUDA error|EngineDeadError|OutOfMemory|died unexpectedly" "$SERVELOG" | tail -1 | cut -c1-200)
XID=$(journalctl -k --since "-1h" --no-pager -q 2>/dev/null | grep -i "NVRM: Xid" | tail -1 | cut -c1-200)
log "GLM pid $PID crashed. last error: ${CAUSE:-none found} | kernel: ${XID:-none}"
echo "$NOW" >>"$STATE"
log "restarting with $SCRIPT ($((RECENT + 1))/$MAX_RESTARTS in window)"
# A live HBM quarantine holder (hbm_scan.py --hold) legitimately occupies a
# GPU; let preflight accept it instead of refusing to start.
HOLD=$HOME/hbm-scan/hold-ready.json
BUSY=0
if [ -s "$HOLD" ] && kill -0 "$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["pid"])' "$HOLD" 2>/dev/null)" 2>/dev/null; then
  BUSY=1; log "HBM quarantine holder alive; starting with ALLOW_BUSY_GPUS=1"
fi
cd "$DIR" && ALLOW_BUSY_GPUS=$BUSY SERVE_SCRIPT=$SCRIPT ./start.sh >>"$WLOG" 2>&1 || log "start.sh failed (exit $?); see $SERVELOG"
