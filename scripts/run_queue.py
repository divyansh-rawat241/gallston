"""Run a queue of long training/evaluation jobs one after another, unattended.

Each non-empty, non-# line of the queue file is `<name> | <command>`. Jobs run
sequentially (one GPU), stdout+stderr go to runs/exp/<name>.log, and a job
that already finished (runs/exp/<name>.done exists) is skipped, so a queue
interrupted by a crash or a sleep can simply be started again.

While the queue runs, Windows is asked not to idle-sleep
(SetThreadExecutionState, the call media players use); this changes no
settings and ends with the process. Closing the lid still sleeps the laptop.

Usage: python scripts/run_queue.py runs/queue.txt
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001


def keep_awake(on: bool) -> None:
    if os.name == "nt":
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))


def main() -> None:
    queue = Path(sys.argv[1])
    logdir = ROOT / "runs" / "exp"
    logdir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    keep_awake(True)
    try:
        for line in queue.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name, cmd = (s.strip() for s in line.split("|", 1))
            done = logdir / f"{name}.done"
            if done.exists():
                print(f"[skip] {name}", flush=True)
                continue
            print(f"[start] {name} {time.strftime('%H:%M:%S')}", flush=True)
            t0 = time.time()
            with open(logdir / f"{name}.log", "w", encoding="utf-8") as log:
                rc = subprocess.run(cmd, shell=True, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
            mins = (time.time() - t0) / 60
            print(f"[{'done' if rc == 0 else 'FAILED'}] {name} rc={rc} {mins:.0f} min", flush=True)
            if rc == 0:
                done.write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
            time.sleep(30)  # let the driver release the GPU before the next job (a fold crashed at start without it)
    finally:
        keep_awake(False)


if __name__ == "__main__":
    main()
