#!/usr/bin/env python3
"""GPU-aware fan controller for the ASRock Rack ROMED8-2T BMC.

The BMC's own Smart Fan curve is driven by CPU and motherboard sensors only --
it has no idea the GPUs exist. On this host that left every fan parked at 30%
duty while a CMP 170HX sat at 84 C, because the CPU was reading 37 C. The cards
are passively cooled, so chassis airflow is the *only* thing moving heat off
them.

This daemon closes that loop: it reads GPU core and HBM temperatures with
nvidia-smi and drives the GPU fan headers directly, via the ASRock Rack OEM IPMI
commands. Each header follows the hottest sensor (core or memory) of the cards
it cools.

    0x3a 0xd8 <16 bytes>   per-fan mode, 1 = manual, 0 = BMC auto
    0x3a 0xd6 <16 bytes>   per-fan duty, 0x00-0x64 (0-100%)
    0x3a 0xda              read back current duty

Byte index N addresses header FAN(N+1) -- index 3 is FAN4, index 6 is FAN7.
Verified empirically on this board; do not assume it holds on another model.
tools/gpu-fan-map.py measures which header cools which card.

Only the GPU headers are switched to manual. FAN1/2/3 stay on the BMC's own
curve, so CPU and chassis cooling keep working normally (and keep working if
this daemon is not running).

Failure is always toward more cooling: if nvidia-smi stops answering, or the
daemon hits an unexpected error, the managed fans go to 100% rather than
staying wherever they happened to be. On a clean shutdown they are handed back
to the BMC.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import subprocess
import sys
import time

# --- Fan topology ------------------------------------------------------------
# Byte index in the IPMI duty array -> board serial numbers (nvidia-smi
# `serial`) of the GPUs that header cools. Determined by a differential cooling
# test (tools/gpu-fan-map.py: drop one header, watch which cards warm), not by
# PCIe bus order: the slot layout does not follow bus enumeration on this
# chassis.
#
# Keyed by serial rather than nvidia-smi index or bus ID on purpose. If a card
# drops off the bus every later index shifts down by one, and a BIOS or riser
# change can renumber buses; either would silently steer each header by the
# wrong cards. A serial follows the physical card, and a missing serial is
# treated as a fault that forces its header to 100%. The airflow belongs to the
# slot, though: if cards are physically moved, rerun gpu-fan-map.py.
# Measured 2026-10-01 (idle, each header dropped to 20% for 180 s; rise is the
# mean of core and HBM). GPU8 rose only +1.5 C, under the tool's 2 C threshold,
# but FAN5 was the only header that moved it at all (every other header <= 0),
# so it is assigned there. FAN6 has a single card behind it.
FAN_GROUPS: dict[int, list[str]] = {
    3: ["1322621099439", "1322421002362", "1322921014351"],  # FAN4: GPU0 01:00.0 +2.0, GPU4 81:00.0 +2.3, GPU5 82:00.0 +2.0
    # 1322421041986 (was c2:00.0, then 84:00.0) pulled 2026-10-03 after two
    # Xid 79s; 1322321006589 now sits at 84:00.0. Both slots are on FAN5.
    4: ["1322821081015", "1322321006589"],  # FAN5: c1:00.0, 84:00.0
    5: ["1322321007949"],  # FAN6: GPU6 83:00.0 +2.5
    6: ["1322821044455", "1322421000158", "1322621129307"],  # FAN7: GPU1 02:00.0 +2.7, GPU2 45:00.0 +2.5, GPU3 46:00.0 +2.0
}


def fan_name(idx: int) -> str:
    return f"FAN{idx + 1}"

# --- Curve -------------------------------------------------------------------
# Duty is a straight line between these two points, clamped at both ends, and
# is driven by max(core, HBM) of the hottest card on the header.
# TEMP_MAX is deliberately well under the 85 C limit: reaching 100% duty only
# at 85 C would mean the fans are still ramping while the card is already
# throttling. Full speed by 70 C leaves headroom to actually stop the climb.
#
# HBM runs 13-15 C above the core at idle (45-49 C vs 30-34 C at 60% duty) but
# only ~4 C above it under load, so at idle it is the memory sensor that sets
# the duty. TEMP_IDLE is 45 C for that reason: a core-era 36 C start would have
# lifted every idle header to ~75%. 45 C still puts a typical idle card a few
# points onto the ramp, so drift is pushed on before it builds.
#
# The 60% floor (~1900 RPM, measured) stays: a 40% floor left idle cores at
# 40-49 C, because ~1400 RPM moved too little air through passive heatsinks.
TEMP_IDLE = 45.0  # at or below this, hold DUTY_IDLE
TEMP_MAX = 70.0  # at or above this, run DUTY_MAX
DUTY_IDLE = 60  # idle floor
DUTY_MAX = 100

# A card above this while its fan is already at 100% is not a control problem --
# there is simply not enough airflow reaching it, and no duty value will fix it.
# Logged loudly so the deficit is attributable to a specific card rather than
# silently absorbed. GPU2 (45:00.0) does this on the current chassis layout.
TEMP_CRITICAL = 80.0

# Ramping up is immediate; ramping down is rate limited. Without this the duty
# oscillates audibly, because dropping the fan raises the die temperature within
# one interval, which raises the duty again.
FALL_STEP = 2  # max duty points shed per interval
INTERVAL = 5.0  # seconds between samples

# Consecutive nvidia-smi failures tolerated before going to DUTY_MAX.
FAULT_TOLERANCE = 3

DUTY_ARRAY_LEN = 16

log = logging.getLogger("gpu-fan")


class IpmiError(RuntimeError):
    pass


def _ipmi(*args: str) -> str:
    """Run an ipmitool command, returning stdout."""
    cmd = ["ipmitool", *args]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=15, check=False
        )
    except subprocess.TimeoutExpired as exc:
        raise IpmiError(f"{' '.join(cmd)}: timed out") from exc
    if proc.returncode != 0:
        raise IpmiError(f"{' '.join(cmd)}: {proc.stderr.strip()}")
    return proc.stdout


def read_duty() -> list[int]:
    """Current per-fan duty as reported by the BMC."""
    out = _ipmi("raw", "0x3a", "0xda")
    vals = [int(tok, 16) for tok in out.split()]
    if len(vals) != DUTY_ARRAY_LEN:
        raise IpmiError(f"expected {DUTY_ARRAY_LEN} duty bytes, got {len(vals)}")
    return vals


def write_duty(duties: list[int]) -> None:
    _ipmi("raw", "0x3a", "0xd6", *(f"0x{d:02x}" for d in duties))


def set_modes(manual_indices: set[int]) -> None:
    """Put the named fans under manual control and leave every other fan on the
    BMC's own curve."""
    flags = [
        "0x01" if i in manual_indices else "0x00" for i in range(DUTY_ARRAY_LEN)
    ]
    _ipmi("raw", "0x3a", "0xd8", *flags)


class GpuTemp:
    """One card's reading. `hot` is what the curve sees: max(core, HBM), or the
    core alone if the HBM sensor reports [N/A]."""

    def __init__(self, index: int, core: float, mem: float | None,
                 bus: str = "", serial: str = ""):
        self.index = index
        self.bus = bus
        self.serial = serial
        self.core = core
        self.mem = mem
        self.hot = core if mem is None else max(core, mem)

    def __str__(self) -> str:
        which = "mem" if self.mem is not None and self.mem > self.core else "core"
        return f"GPU{self.index} {self.hot:.0f}C {which}"


def normalize_bus_id(bus_id: str) -> str:
    """'00000000:45:00.0' -> '45:00.0' (what lspci prints)."""
    bus_id = bus_id.strip().lower()
    if bus_id.count(":") == 2:
        bus_id = bus_id.split(":", 1)[1]
    return bus_id


def read_gpu_temps() -> dict[str, GpuTemp]:
    """Board serial -> current core/HBM temperatures."""
    proc = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,pci.bus_id,serial,temperature.gpu,temperature.memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {proc.stderr.strip()}")

    temps: dict[str, GpuTemp] = {}
    for line in proc.stdout.strip().splitlines():
        fields = [f.strip() for f in line.split(",")]
        try:
            idx_s, bus_s, serial, core_s, mem_s = fields
            try:
                mem: float | None = float(mem_s)
            except ValueError:
                mem = None  # HBM sensor [N/A]; fall back to the core alone
            if not serial.isdigit():
                raise ValueError(f"no usable serial: {serial!r}")
            temps[serial] = GpuTemp(int(idx_s), float(core_s), mem,
                                    normalize_bus_id(bus_s), serial)
        except ValueError:
            # A card that is falling off the bus reports [N/A]. Skip it here;
            # the caller treats a missing group member as a fault.
            log.warning("unparseable nvidia-smi row: %r", line)
    if not temps:
        raise RuntimeError("nvidia-smi returned no usable temperatures")
    return temps


def duty_for(temp: float) -> int:
    """Map the hottest die in a group to a duty percentage."""
    if temp <= TEMP_IDLE:
        return DUTY_IDLE
    if temp >= TEMP_MAX:
        return DUTY_MAX
    span = (temp - TEMP_IDLE) / (TEMP_MAX - TEMP_IDLE)
    return int(round(DUTY_IDLE + span * (DUTY_MAX - DUTY_IDLE)))


def clamp(duty: int) -> int:
    return max(0, min(DUTY_MAX, duty))


class Controller:
    def __init__(self, groups: dict[int, list[str]], dry_run: bool = False):
        self.groups = groups
        self.dry_run = dry_run
        self.applied: dict[int, int] = {i: DUTY_MAX for i in groups}
        self.faults = 0
        self._stop = False
        # start() parks the fans at 100% before the first reading lands. Once a
        # real temperature arrives we want to drop straight to the curve value
        # rather than crawl down at FALL_STEP per interval, which would spend
        # ~2.5 minutes at full noise on every restart.
        self._settled = False

    def request_stop(self, *_: object) -> None:
        self._stop = True

    # -- lifecycle -----------------------------------------------------------
    def start(self) -> None:
        if self.dry_run:
            log.info("dry run: not taking manual control")
            return
        set_modes(set(self.groups))
        log.info(
            "manual control taken for %s; other headers left on BMC auto",
            ", ".join(fan_name(i) for i in sorted(self.groups)),
        )
        # Start at full speed until the first good reading tells us otherwise.
        self.apply({i: DUTY_MAX for i in self.groups})

    def restore(self) -> None:
        """Hand the GPU headers back to the BMC."""
        if self.dry_run:
            return
        try:
            set_modes(set())
            log.info("restored all headers to BMC auto control")
        except IpmiError as exc:
            # Best effort: we are already on the way out. Leaving the fans at
            # whatever duty they hold is survivable; the BMC still enforces its
            # own failsafe if a fan stalls.
            log.error("could not restore BMC auto control: %s", exc)

    def panic(self) -> None:
        """Lost sight of the GPUs -- cool them as hard as we can."""
        if self.applied != {i: DUTY_MAX for i in self.groups}:
            log.error("fault threshold reached, forcing %d%%", DUTY_MAX)
        try:
            self.apply({i: DUTY_MAX for i in self.groups})
        except IpmiError as exc:
            # This is the failsafe path; it must not be what kills the daemon.
            # systemd restarts us, and start() re-asserts 100% on the way in.
            log.error("failsafe write failed: %s", exc)

    # -- control -------------------------------------------------------------
    def apply(self, targets: dict[int, int]) -> None:
        if self.dry_run:
            self.applied.update(targets)
            return
        duties = read_duty()
        changed = False
        for idx, duty in targets.items():
            duty = clamp(duty)
            if duties[idx] != duty:
                duties[idx] = duty
                changed = True
            self.applied[idx] = duty
        if changed:
            write_duty(duties)

    def step(self) -> None:
        try:
            temps = read_gpu_temps()
            self.faults = 0
        except Exception as exc:  # nvidia-smi gone, driver wedged, timeout
            self.faults += 1
            log.warning("temperature read failed (%d/%d): %s",
                        self.faults, FAULT_TOLERANCE, exc)
            if self.faults >= FAULT_TOLERANCE:
                self.panic()
            return

        targets: dict[int, int] = {}
        report = []
        for idx, gpus in self.groups.items():
            missing = [g for g in gpus if g not in temps]
            if missing:
                log.warning("GPU serial(s) %s missing from nvidia-smi; forcing %d%% on %s",
                            ", ".join(missing), DUTY_MAX, fan_name(idx))
                targets[idx] = DUTY_MAX
                report.append(f"{fan_name(idx)}=100%(fault)")
                continue

            # Track which card (and which sensor) is driving the group, not just
            # how hot it is -- on this chassis one card in a group can run
            # ~19 C above its neighbours, and that is invisible from the max.
            hot = max((temps[g] for g in gpus), key=lambda t: t.hot)
            hottest = hot.hot
            want = duty_for(hottest)
            current = self.applied.get(idx, DUTY_MAX)
            if not self._settled:
                # First good reading after start: adopt the curve directly.
                duty = want
            else:
                # Rise at once, fall gently.
                duty = want if want >= current else max(want, current - FALL_STEP)
            targets[idx] = duty
            report.append(f"{fan_name(idx)}: {hot} -> {duty}%")
            if duty >= DUTY_MAX and hottest >= TEMP_CRITICAL:
                log.error(
                    "%s is at %d%% and %s: cooling deficit, not a control "
                    "problem. Needs airflow or a lower power cap.",
                    fan_name(idx), DUTY_MAX, hot,
                )

        try:
            self.apply(targets)
        except IpmiError as exc:
            log.error("failed to apply duty: %s", exc)
            return

        self._settled = True
        log.info("  ".join(report))

    def run(self) -> int:
        self.start()
        try:
            while not self._stop:
                self.step()
                # Sleep in slices so a signal is acted on promptly.
                waited = 0.0
                while waited < INTERVAL and not self._stop:
                    time.sleep(0.25)
                    waited += 0.25
        finally:
            self.restore()
        return 0


def parse_groups(spec: str) -> dict[int, list[str]]:
    """Parse "3:1322621099439,1322421002362 6:1322421000158" into
    {3: ["1322621099439", "1322421002362"], 6: ["1322421000158"]}. The key is
    the fan byte index; GPUs are board serials as nvidia-smi reports them."""
    groups: dict[int, list[str]] = {}
    for chunk in spec.split():
        fan_s, _, gpus_s = chunk.partition(":")
        gpus = [g for g in gpus_s.split(",") if g != ""]
        for g in gpus:
            if not g.isdigit():
                raise ValueError(f"not a GPU serial: {g!r} (see nvidia-smi "
                                 "--query-gpu=index,serial --format=csv)")
        groups[int(fan_s)] = gpus
    return groups


def main() -> int:
    # Declared up front: the argparse defaults below read these names, and
    # Python forbids `global` after a name is used in the same scope.
    global DUTY_IDLE, TEMP_IDLE, TEMP_MAX, INTERVAL

    ap = argparse.ArgumentParser(
        description="Drive ROMED8-2T GPU fan headers from GPU core/HBM temperature."
    )
    ap.add_argument(
        "--groups",
        default=None,
        help='Override fan map, e.g. "3:1322621099439,1322421002362 '
        '6:1322421000158" (fan byte index:GPU serials)',
    )
    ap.add_argument("--idle-duty", type=int, default=DUTY_IDLE)
    ap.add_argument("--idle-temp", type=float, default=TEMP_IDLE)
    ap.add_argument("--max-temp", type=float, default=TEMP_MAX)
    ap.add_argument("--interval", type=float, default=INTERVAL)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Log the duty it would set without touching the BMC.",
    )
    ap.add_argument(
        "--once", action="store_true", help="Evaluate a single step and exit."
    )
    args = ap.parse_args()

    DUTY_IDLE = args.idle_duty
    TEMP_IDLE = args.idle_temp
    TEMP_MAX = args.max_temp
    INTERVAL = args.interval

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )

    if not args.dry_run and os.geteuid() != 0:
        log.error("needs root for /dev/ipmi0 (run under sudo, or use --dry-run)")
        return 1

    groups = parse_groups(args.groups) if args.groups else FAN_GROUPS
    if not groups:
        log.error("no fan groups configured (run tools/gpu-fan-map.py)")
        return 1
    ctl = Controller(groups, dry_run=args.dry_run)

    if args.once:
        # Evaluate one step and hand the fans straight back, so a manual probe
        # never leaves the BMC pinned in manual mode.
        ctl.start()
        try:
            ctl.step()
        finally:
            ctl.restore()
        return 0

    signal.signal(signal.SIGTERM, ctl.request_stop)
    signal.signal(signal.SIGINT, ctl.request_stop)
    return ctl.run()


if __name__ == "__main__":
    sys.exit(main())
