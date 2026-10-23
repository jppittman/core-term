#!/usr/bin/env python3
"""Run a command once; print "<peak RSS KiB> <wall ms>" on stdout.

usage: runstat.py CMD [ARGS...]     (the command's stdout is discarded)

The peak RSS is the child's `ru_maxrss` from wait4(2), in KiB: the figure
`/usr/bin/time -f %M` prints, read the same way, for hosts that have no
/usr/bin/time (this one). Exit status 1 if the command failed.
"""
import os
import subprocess
import sys
import time

started = time.perf_counter()
child = subprocess.Popen(sys.argv[1:], stdout=subprocess.DEVNULL)
_, status, usage = os.wait4(child.pid, 0)
wall_ms = round((time.perf_counter() - started) * 1000)
print(usage.ru_maxrss, wall_ms)
sys.exit(0 if os.waitstatus_to_exitcode(status) == 0 else 1)
