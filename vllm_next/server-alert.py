#!/usr/bin/env python3
"""Push aiserver failures to a phone through ntfy.

Run from cron every minute as arveen (no root). Each check keeps state between
runs and notifies only on a transition: once when it goes bad, once when it
recovers. A message that cannot be delivered (network down) is queued and
retried on the next run.

This runs on the server, so it cannot report the server itself dying (power,
freeze, network). It covers what the host can still see:

  gpus     fewer than EXPECTED_GPUS cards in nvidia-smi, or nvidia-smi failing
  xid      any new NVRM Xid line in the kernel log (79 = fallen off the bus)
  watchdog new crash / restart / failure lines in glm-watchdog's log; this is
           the only signal for "restart failed", because a failed start.sh
           leaves no pid file and then looks exactly like a deliberate stop
  glm      :8002 unhealthy while the pid file says it should be running
  proxy    :8000 unreachable while GLM is meant to be running
  units    glm-gateway / glm-tailcat / gpu-fan-control / gpu-power-cap inactive
  temp     any core or HBM sensor at or above TEMP_ALERT
  replay   a card's PCIe replay counter increasing (marginal riser/link)

Config: ~/.config/server-alert/ntfy.env, mode 600:
  NTFY_URL=https://ntfy.sh/<secret topic>
  NTFY_TOKEN=            (optional, for an access-controlled topic)

  server-alert.py            one pass (what cron runs)
  server-alert.py --test     send a test notification
  server-alert.py --status   print current state without notifying
"""
from __future__ import annotations

import fcntl
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HOME = Path.home()
CONF = HOME / ".config/server-alert/ntfy.env"
STATE_DIR = HOME / ".local/state/server-alert"
STATE = STATE_DIR / "state.json"
OUTBOX = STATE_DIR / "outbox.jsonl"
LOG = STATE_DIR / "alert.log"
VLLM_DIR = Path(os.environ.get("VLLM_NEXT_DIR", HOME / "vllm_install/vllm_next"))
PIDFILE = VLLM_DIR / "logs/vllm.pid"
WATCHDOG_LOG = VLLM_DIR / "logs/watchdog.log"

HOST = "aiserver"
# 9 since 2026-10-03: serial 1322421041986 pulled (fell off the bus twice).
EXPECTED_GPUS = 9
TEMP_ALERT = 85            # same limit as thermal-stress.sh
TEMP_CLEAR = 80
UNITS = ["glm-gateway", "glm-tailcat", "gpu-fan-control", "gpu-power-cap"]
GLM_HEALTH = "http://127.0.0.1:8002/health"
PROXY_HEALTH = "http://127.0.0.1:8000/health"
# A cold load of the 419 GB checkpoint takes ~16 min (start.sh allows 30), so a
# GLM that has not been healthy yet under its current pid gets that long.
GLM_LOAD_GRACE_S = 35 * 60
GLM_DOWN_AFTER = 3         # consecutive failed minutes once it has been healthy
REPLAY_EVERY_S = 300       # nvidia-smi -q costs ~2 s of driver time
MAX_PER_HOUR = 30          # runaway guard; ntfy.sh limits anonymous senders

PRIO = {"urgent": "5", "high": "4", "default": "3", "low": "2"}


def log(msg: str) -> None:
    with LOG.open("a") as f:
        f.write(f"{time.strftime('%FT%T')} {msg}\n")


def run(cmd: list[str], timeout: float = 20) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout + p.stderr
    except (OSError, subprocess.TimeoutExpired) as e:
        return -1, str(e)


def http_ok(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def load_conf() -> dict[str, str]:
    conf = {}
    for line in CONF.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            conf[k.strip()] = v.strip()
    if not conf.get("NTFY_URL"):
        sys.exit(f"NTFY_URL missing in {CONF}")
    return conf


# ---------------------------------------------------------------- delivery

def post(conf: dict[str, str], msg: dict) -> bool:
    req = urllib.request.Request(conf["NTFY_URL"], data=msg["body"].encode(), method="POST")
    req.add_header("Title", msg["title"])
    req.add_header("Priority", PRIO[msg["prio"]])
    req.add_header("Tags", msg["tags"])
    if conf.get("NTFY_TOKEN"):
        req.add_header("Authorization", f"Bearer {conf['NTFY_TOKEN']}")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return 200 <= r.status < 300
    except Exception as e:
        log(f"send failed: {e}")
        return False


class Notifier:
    def __init__(self, conf: dict[str, str], state: dict, dry: bool = False):
        self.conf, self.state, self.dry = conf, state, dry
        self.pending: list[dict] = []

    def __call__(self, title: str, body: str, prio: str = "high", tags: str = "warning") -> None:
        self.pending.append({"title": f"{HOST}: {title}", "body": body, "prio": prio,
                             "tags": tags, "at": time.strftime("%H:%M")})

    def flush(self) -> None:
        queued = []
        if OUTBOX.exists():
            queued = [json.loads(l) for l in OUTBOX.read_text().splitlines() if l.strip()]
        msgs = queued + self.pending
        if self.dry:
            for m in msgs:
                print(f"[{m['prio']}] {m['title']}\n    {m['body']}")
            return
        now = time.time()
        sent = [t for t in self.state.get("sent", []) if t > now - 3600]
        left = []
        for i, m in enumerate(msgs):
            if len(sent) >= MAX_PER_HOUR:
                left = msgs[i:]
                if not self.state.get("throttled"):
                    post(self.conf, {"title": f"{HOST}: alerts throttled", "prio": "high", "tags": "mute",
                                     "body": f"Over {MAX_PER_HOUR}/h; holding the rest. See {LOG}"})
                    self.state["throttled"] = True
                break
            if m in queued and not m.get("late"):
                m["body"] += f" (delayed, raised {m['at']})"
                m["late"] = True
            if post(self.conf, m):
                sent.append(now)
                log(f"sent: {m['title']} | {m['body']}")
            else:
                left = msgs[i:]
                break
        else:
            self.state["throttled"] = False
        self.state["sent"] = sent
        OUTBOX.write_text("".join(json.dumps(m) + "\n" for m in left))


# ---------------------------------------------------------------- checks

def check_gpus(st: dict, notify: Notifier) -> list[dict]:
    rc, out = run(["nvidia-smi", "--query-gpu=index,pci.bus_id,serial,temperature.gpu,temperature.memory",
                   "--format=csv,noheader,nounits"])
    gpus, errors = [], []
    for line in out.splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) == 5 and f[2].isdigit():
            gpus.append({"idx": f[0], "bus": f[1][-12:].lower(), "serial": f[2],
                         "core": f[3], "mem": f[4]})
        elif line.strip():
            errors.append(line.strip())
    known = st.setdefault("serials", {})          # serial -> last bus seen
    present = {g["serial"] for g in gpus}
    bad = len(gpus) < EXPECTED_GPUS
    if bad and not st.get("gpus_bad"):
        missing = [f"{s} (was {b})" for s, b in known.items() if s not in present]
        body = f"{len(gpus)}/{EXPECTED_GPUS} GPUs visible."
        if missing:
            body += " Missing: " + ", ".join(missing) + "."
        if errors:
            body += " " + errors[0][:160]
        notify("GPU missing", body, "urgent", "rotating_light")
    elif not bad and st.get("gpus_bad"):
        notify("all GPUs back", f"{len(gpus)}/{EXPECTED_GPUS} GPUs visible.", "default", "white_check_mark")
    st["gpus_bad"] = bad
    if not bad:
        st["serials"] = {g["serial"]: g["bus"] for g in gpus}
    return gpus


def check_temps(st: dict, gpus: list[dict], notify: Notifier) -> None:
    hot = st.setdefault("hot", [])
    for g in gpus:
        temps = [int(t) for t in (g["core"], g["mem"]) if t.isdigit()]
        t = max(temps, default=0)
        if t >= TEMP_ALERT and g["serial"] not in hot:
            notify(f"GPU{g['idx']} at {t} C",
                   f"{g['bus']} serial {g['serial']}: core {g['core']} C, HBM {g['mem']} C (limit {TEMP_ALERT}).",
                   "urgent", "fire")
            hot.append(g["serial"])
        elif t and t < TEMP_CLEAR and g["serial"] in hot:
            notify(f"GPU{g['idx']} cooled", f"{g['bus']}: {t} C.", "default", "white_check_mark")
            hot.remove(g["serial"])


def check_replays(st: dict, gpus: list[dict], notify: Notifier) -> None:
    now = time.time()
    if now - st.get("replay_at", 0) < REPLAY_EVERY_S:
        return
    st["replay_at"] = now
    rc, out = run(["nvidia-smi", "-q"], timeout=60)
    if rc != 0:
        return
    counts, serial = {}, None
    for line in out.splitlines():
        if m := re.match(r"\s+Serial Number\s+:\s+(\d+)", line):
            serial = m.group(1)
        elif serial and (m := re.match(r"\s+Replays Since Reset\s+:\s+(\d+)", line)):
            counts[serial] = int(m.group(1))
        elif serial and (m := re.match(r"\s+Replay Number Rollovers\s+:\s+(\d+)", line)):
            counts[serial] += int(m.group(1)) << 32
    bus = {g["serial"]: g["bus"] for g in gpus}
    prev = st.get("replays", {})
    grew = [f"{bus.get(s, s)} +{n - prev[s]} (total {n})"
            for s, n in counts.items() if s in prev and n > prev[s]]
    if grew:
        notify("PCIe replays rising",
               "Link retransmissions on " + "; ".join(grew) + ". Marginal riser or adapter on that slot.",
               "high", "electric_plug")
    st["replays"] = counts


def check_xid(st: dict, notify: Notifier) -> None:
    cursor = STATE_DIR / "journal.cursor"
    if not cursor.exists():
        # Start from now, not the whole boot: seed the cursor with the newest entry.
        rc, out = run(["journalctl", "-k", "-n", "1", "--show-cursor", "-q", "-o", "cat", "--no-pager"])
        m = re.search(r"^-- cursor: (.+)$", out, re.M)
        if m:
            cursor.write_text(m.group(1))
        return
    rc, out = run(["journalctl", "-k", "--no-pager", "-q", "-o", "cat", f"--cursor-file={cursor}"])
    if rc != 0:
        return
    xids = [l for l in out.splitlines() if "NVRM: Xid" in l]
    if not xids:
        return
    codes = sorted({m.group(1) for l in xids if (m := re.search(r"\): (\d+),", l))})
    lead = next((l for l in xids if "): 79," in l), xids[0])
    prio = "urgent" if {"79", "48", "74", "95", "119"} & set(codes) else "high"
    notify(f"GPU error Xid {'/'.join(codes)}",
           f"{len(xids)} line(s). {lead.split('NVRM: ', 1)[-1][:220]}", prio, "boom")


def check_watchdog(st: dict, notify: Notifier) -> None:
    try:
        size = WATCHDOG_LOG.stat().st_size
    except FileNotFoundError:
        return
    off = st.get("wd_off", size)                  # first run: skip history
    if size < off:
        off = 0                                   # truncated or rotated
    if size > off:
        with WATCHDOG_LOG.open("rb") as f:
            f.seek(off)
            lines = f.read().decode(errors="replace").splitlines()
        for l in lines:
            text = l.split(" ", 1)[-1] if re.match(r"\d{4}-\d\d-\d\dT", l) else l
            if "crashed" in text:
                notify("GLM crashed", text[:300], "urgent", "rotating_light")
            elif "start.sh failed" in text or "exited during preflight" in text:
                notify("GLM restart FAILED", text[:200] + ". GLM is down and the watchdog will not retry.",
                       "urgent", "rotating_light")
            elif "giving up" in text:
                notify("watchdog gave up", text[:300], "urgent", "rotating_light")
    st["wd_off"] = size


def glm_pid() -> int | None:
    try:
        pid = int(re.sub(r"\D", "", PIDFILE.read_text()))
        os.kill(pid, 0)
        return pid
    except (FileNotFoundError, ValueError, ProcessLookupError, PermissionError):
        return None


def check_glm(st: dict, notify: Notifier) -> None:
    pid = glm_pid()
    healthy = http_ok(GLM_HEALTH)
    now = time.time()
    if healthy:
        if st.get("glm_down"):
            notify("GLM back up", "Healthy on :8002.", "default", "white_check_mark")
        st.update(glm_down=False, glm_fails=0, glm_healthy_pid=pid, glm_pid_since=None)
    elif pid is None:
        # No live server and no pid file: stopped on purpose (stop.sh), or a
        # failed restart, which check_watchdog reports. Nothing to add here.
        st.update(glm_fails=0, glm_pid_since=None)
        if st.get("glm_down"):
            st["glm_down"] = False
    else:
        if st.get("glm_pid_since_pid") != pid:
            st.update(glm_pid_since=now, glm_pid_since_pid=pid)
        st["glm_fails"] = st.get("glm_fails", 0) + 1
        was_up = st.get("glm_healthy_pid") == pid
        overdue = (st["glm_fails"] >= GLM_DOWN_AFTER) if was_up else \
                  (now - (st.get("glm_pid_since") or now) >= GLM_LOAD_GRACE_S)
        if overdue and not st.get("glm_down"):
            why = (f"unhealthy for {st['glm_fails']} min" if was_up
                   else f"still not healthy {GLM_LOAD_GRACE_S // 60} min after start")
            notify("GLM not responding", f"Process {pid} alive but :8002/health {why}.", "urgent", "rotating_light")
            st["glm_down"] = True
    # The owner's proxy on :8000 (text pass-through) is started by start.sh alongside GLM.
    proxy = http_ok(PROXY_HEALTH) if pid else True
    st["proxy_fails"] = 0 if proxy else st.get("proxy_fails", 0) + 1
    if st["proxy_fails"] == 3:
        notify("proxy :8000 down", "Proxy unreachable for 3 min while GLM is running.", "high", "warning")
    elif proxy and st.get("proxy_down"):
        notify("proxy :8000 back", "Proxy healthy.", "default", "white_check_mark")
    st["proxy_down"] = st["proxy_fails"] >= 3


def check_units(st: dict, notify: Notifier) -> None:
    rc, out = run(["systemctl", "is-active", *UNITS])
    states = dict(zip(UNITS, out.split()))
    fails = st.setdefault("unit_fails", {})
    for u in UNITS:
        ok = states.get(u) == "active"
        n = 0 if ok else fails.get(u, 0) + 1
        if n == 2:
            notify(f"{u} down", f"systemctl is-active: {states.get(u, 'unknown')}.", "urgent", "rotating_light")
        elif ok and fails.get(u, 0) >= 2:
            notify(f"{u} back", "active", "default", "white_check_mark")
        fails[u] = n


# ---------------------------------------------------------------- main

def main() -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    lock = open(STATE_DIR / "lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return
    conf = load_conf()
    if "--test" in sys.argv:
        ok = post(conf, {"title": f"{HOST}: test", "prio": "default", "tags": "bell",
                         "body": "server-alert can reach your phone."})
        sys.exit(0 if ok else 1)
    status = "--status" in sys.argv
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    notify = Notifier(conf, state, dry=status)
    gpus = check_gpus(state, notify)
    check_temps(state, gpus, notify)
    check_replays(state, gpus, notify)
    if not status:
        check_xid(state, notify)                  # consumes the journal cursor
        check_watchdog(state, notify)
    check_glm(state, notify)
    check_units(state, notify)
    notify.flush()
    if status:
        print(json.dumps({k: v for k, v in state.items() if k != "sent"}, indent=1))
        return
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state))
    tmp.replace(STATE)


if __name__ == "__main__":
    main()
