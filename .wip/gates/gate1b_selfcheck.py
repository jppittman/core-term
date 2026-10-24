#!/usr/bin/env python3
"""Gate 1b's harness self-check: the probe's n=2 and n=40 digests at 16 px must
equal the values `the_optimizer_selects_what_it_selected` pins
(SELECTED_PROGRAM_DIGESTS in program/tests.rs, the (n, 16, digest) rows).

usage: gate1b_selfcheck.py PROBE_LINES TESTS_RS     exit 0 = equal, 2 = not
"""
import re
import sys

probe_path, tests_path = sys.argv[1:3]

source = open(tests_path).read()
block = re.search(r"const SELECTED_PROGRAM_DIGESTS[^=]*=\s*\[(.*?)\];", source, re.S)
if not block:
    sys.exit("selfcheck: SELECTED_PROGRAM_DIGESTS not found in " + tests_path)
pinned = {
    int(n): int(digest.replace("_", ""), 16)
    for n, side, digest in re.findall(r"\((\d+),\s*(\d+),\s*0x([0-9a-fA-F_]+)\)", block.group(1))
    if int(side) == 16
}
if sorted(pinned) != [2, 40]:
    sys.exit("selfcheck: expected pinned n=2 and n=40 at side 16, found %r" % sorted(pinned))

probed = {}
for line in open(probe_path):
    fields = dict(f.split("=", 1) for f in line.split()[1:])
    probed[int(fields["n"])] = int(fields["digest"], 16)

bad = False
for n in sorted(pinned):
    ok = probed.get(n) == pinned[n]
    print("selfcheck n=%d  probe=%s  pinned=%016x  %s" % (
        n, "%016x" % probed[n] if n in probed else "missing", pinned[n], "equal" if ok else "MISMATCH"))
    bad |= not ok
sys.exit(2 if bad else 0)
