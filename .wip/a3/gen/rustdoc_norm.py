#!/usr/bin/env -S python3 -I
"""Normalize `cargo doc` output to (file, message) with counts, for the A3 rustdoc gate (C4).

  cargo doc -p <crate> --no-deps --document-private-items 2>&1 | python3 -I rustdoc_norm.py
      prints one line per distinct warning, sorted:  <count> TAB <file> TAB <message>

  ... | python3 -I rustdoc_norm.py --compare BASE.txt
      prints NEW (a warning absent from BASE, or more often than BASE has it) and exits 1 if
      there is one; prints GONE for warnings that went away, which is information, not a
      failure; exits 0 otherwise.

A warning is a `warning: <message>` block of the output; its file is the path of the first
`--> path:line:col` line of the block with the line and column dropped, or `<no span>` when the
block has none. The trailing `warning: <crate> (lib doc) generated N warnings` summary is not a
warning of its own and is skipped. Nothing else about a block (line numbers, the quoted source,
the notes) is part of the key, so a warning that only moves down a file does not change it.
"""

import collections
import re
import sys

SUMMARY = re.compile(r"^`[^`]+` \((lib|bin|example|test|lib test|bench)[^)]*\) generated \d+ warnings?")
SPAN = re.compile(r"^\s*--> (.+?)(?::\d+)?(?::\d+)?\s*$")


def normalize(text):
    counts = collections.Counter()
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        i += 1
        if not line.startswith("warning: "):
            continue
        message = line[len("warning: ") :].strip()
        if SUMMARY.match(message):
            continue
        file = "<no span>"
        # The block runs to the next blank line.
        while i < len(lines) and lines[i].strip():
            m = SPAN.match(lines[i])
            if m and file == "<no span>":
                file = m.group(1)
            i += 1
        counts[(file, message)] += 1
    return counts


def render(counts):
    return "".join("%d\t%s\t%s\n" % (n, f, m) for (f, m), n in sorted(counts.items()))


def parse(text):
    counts = collections.Counter()
    for row in text.splitlines():
        if not row.strip():
            continue
        n, f, m = row.split("\t", 2)
        counts[(f, m)] += int(n)
    return counts


def main(argv):
    sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    head = normalize(sys.stdin.read())
    if "--compare" not in argv:
        sys.stdout.write(render(head))
        return 0
    with open(argv[argv.index("--compare") + 1], encoding="utf-8") as fh:
        base = parse(fh.read())
    new = {k: n for k, n in head.items() if n > base.get(k, 0)}
    gone = {k: n for k, n in base.items() if n > head.get(k, 0)}
    print("warnings: base %d, head %d" % (sum(base.values()), sum(head.values())))
    print("NEW %d" % len(new))
    for (f, m), n in sorted(new.items()):
        print("  NEW  %s  %s  (head %d, base %d)" % (f, m, head[(f, m)], base.get((f, m), 0)))
    print("GONE %d" % len(gone))
    for (f, m), n in sorted(gone.items()):
        print("  GONE %s  %s  (head %d, base %d)" % (f, m, head.get((f, m), 0), base[(f, m)]))
    return 1 if new else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
