#!/usr/bin/env python3
"""Run a command while holding an exclusive flock for its entire lifetime.

Usage: flock_run.py <lockfile> <command> [args...]

Acquires a non-blocking exclusive flock(2) on <lockfile>. If another live process
already holds it, exits 0 (the caller's run is simply skipped, mirroring the previous
"skip (lock exists)" behaviour). Otherwise it replaces itself with <command> via
execvp, keeping the lock file descriptor open across the exec, so the kernel holds the
lock for the command's whole lifetime and releases it automatically on ANY exit --
normal, signal, SIGKILL, or reboot. A crashed run therefore can never leave a stale
lock, and there is no PID/age/gate heuristic (and no associated TOCTOU race) to get
wrong: mutual exclusion is enforced by the kernel.
"""

import errno
import fcntl
import os
import sys


def main() -> int:
    if len(sys.argv) < 3:
        sys.stderr.write("usage: flock_run.py <lockfile> <command> [args...]\n")
        return 2
    lockfile = sys.argv[1]
    cmd = sys.argv[2:]
    fd = os.open(lockfile, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        if exc.errno in (errno.EWOULDBLOCK, errno.EAGAIN):
            return 0  # held by a live run -> skip this invocation
        raise  # any other error is unexpected -> fail loudly rather than silently skip
    os.set_inheritable(fd, True)  # keep the lock-holding fd open across exec
    try:
        os.execvp(cmd[0], cmd)
    except OSError as exc:  # pragma: no cover - exec failure is unexpected
        sys.stderr.write(f"flock_run: exec failed: {exc}\n")
        return 127
    return 127  # unreachable


if __name__ == "__main__":
    raise SystemExit(main())
