"""Pause or resume a training run (and its data-loader workers) on Windows.

Suspends every thread of the process tree with NtSuspendProcess; resuming
continues exactly where it stopped. GPU memory stays allocated while paused,
and a reboot or sleep-to-shutdown loses the paused run.

Usage:
  python scripts/pause_training.py pause  <root_pid>
  python scripts/pause_training.py resume <root_pid>
"""
import ctypes
import subprocess
import sys


def tree(root: int) -> list[int]:
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "Get-CimInstance Win32_Process | ForEach-Object { \"$($_.ProcessId) $($_.ParentProcessId)\" }"],
                         capture_output=True, text=True).stdout.split("\n")
    parent = {}
    for line in out:
        parts = line.split()
        if len(parts) == 2:
            parent[int(parts[0])] = int(parts[1])
    pids, frontier = [root], [root]
    while frontier:
        nxt = [p for p, pp in parent.items() if pp in frontier and p not in pids]
        pids += nxt
        frontier = nxt
    return pids


def act(pid: int, resume: bool) -> bool:
    PROCESS_SUSPEND_RESUME = 0x0800
    h = ctypes.windll.kernel32.OpenProcess(PROCESS_SUSPEND_RESUME, False, pid)
    if not h:
        return False
    fn = ctypes.windll.ntdll.NtResumeProcess if resume else ctypes.windll.ntdll.NtSuspendProcess
    ok = fn(h) == 0
    ctypes.windll.kernel32.CloseHandle(h)
    return ok


if __name__ == "__main__":
    mode, root = sys.argv[1], int(sys.argv[2])
    pids = tree(root)
    done = [p for p in pids if act(p, resume=(mode == "resume"))]
    print(f"{mode}d {len(done)}/{len(pids)} processes: {done}")
