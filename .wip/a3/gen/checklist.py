#!/usr/bin/env -S python3 -I
"""The A3 Reduce-site checklist (rulings B2, C10).

Every line that spells `Reduce {` in the workspace's Rust sources is a site, and every site
is classified in classes.json (next to this file). The script lists the sites with the very
grep the spec names, keys each by (file, enclosing item, trimmed hit text) with a per-key
count, so a line that moves does not unclassify a site, and prints

  UNCLASSIFIED  a hit no row covers (or more hits than the row's count)
  STALE         a row no hit covers (or fewer hits than the row's count)

It exits 1 when either list is non-empty, 2 when classes.json is malformed, 0 otherwise.

Run:  python3 -I .wip/a3/gen/checklist.py          (from anywhere; the root is found
                                                     from this file's location)
      python3 -I .wip/a3/gen/checklist.py --list   (also print every site and its class)
      --root DIR      scan DIR instead of the repository this file sits in
      --classes FILE  read FILE instead of classes.json beside this file
"""

import bisect
import collections
import glob
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
CLASSES = os.path.join(HERE, "classes.json")

# The spec's grep, verbatim: `grep -rn 'Reduce {' --include=*.rs <DIRS>`.
PATTERN = "Reduce {"
DIR_GLOBS = [
    "pixelflow-*/src",
    "pixelflow-*/examples",
    "pixelflow-*/tests",
    "pixelflow-pipeline/src/bin",
]

CLASS_ORDER = [
    "already-correct",
    "changed-by-A3",
    "left",
    "tests",
    "n/a",
    "A4",
    "A5",
]

TOP = "<top>"


# ---------------------------------------------------------------------------------------
# Lexing: blank out comments, string contents and char literals so braces can be matched.
# ---------------------------------------------------------------------------------------

_RAW_STRING = re.compile(r'b?r(#*)"')


def mask(src):
    """`src` with every comment, string and char literal replaced by spaces (newlines kept)."""
    out = list(src)
    n = len(src)

    def blank(a, b):
        for k in range(a, min(b, n)):
            if out[k] != "\n":
                out[k] = " "

    i = 0
    while i < n:
        c = src[i]
        if src.startswith("//", i):
            j = src.find("\n", i)
            j = n if j < 0 else j
            blank(i, j)
            i = j
        elif src.startswith("/*", i):
            depth, j = 1, i + 2
            while j < n and depth:
                if src.startswith("/*", j):
                    depth, j = depth + 1, j + 2
                elif src.startswith("*/", j):
                    depth, j = depth - 1, j + 2
                else:
                    j += 1
            blank(i, j)
            i = j
        elif c in "br" and (i == 0 or not (src[i - 1].isalnum() or src[i - 1] == "_")) and _RAW_STRING.match(src, i):
            m = _RAW_STRING.match(src, i)
            close = '"' + m.group(1)
            j = src.find(close, m.end())
            j = n if j < 0 else j + len(close)
            blank(i, j)
            i = j
        elif c == '"':
            j = i + 1
            while j < n and src[j] != '"':
                j += 2 if src[j] == "\\" else 1
            blank(i, j + 1)
            i = j + 1
        elif c == "'":
            if i + 1 < n and src[i + 1] == "\\":
                j = src.find("'", i + 3)
                j = n if j < 0 else j + 1
                blank(i, j)
                i = j
            elif i + 2 < n and src[i + 2] == "'":
                blank(i, i + 3)
                i += 3
            else:
                i += 1  # a lifetime or a label
        else:
            i += 1
    return "".join(out)


def match_brace(masked, open_at):
    """Offset of the `}` closing the `{` at `open_at` (or the end of the text)."""
    depth = 0
    for k in range(open_at, len(masked)):
        ch = masked[k]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return k
    return len(masked) - 1


def body_open(masked, start):
    """Offset of the `{` that opens the item whose header begins at `start`, or None when the
    item ends in `;` first (a bodiless declaration)."""
    depth = 0
    for k in range(start, len(masked)):
        ch = masked[k]
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif depth == 0 and ch == "{":
            return k
        elif depth == 0 and ch == ";":
            return None
    return None


_FN = re.compile(r"\bfn\s+([A-Za-z_][A-Za-z0-9_]*)")
_ITEM = re.compile(
    r"^[ \t]*(?:pub(?:\([^)]*\))?[ \t]+)?(?:unsafe[ \t]+)?"
    r"(?:(struct|enum|union|trait|mod)\s+([A-Za-z_][A-Za-z0-9_]*)|(impl)\b)",
    re.MULTILINE,
)


class Spans:
    """The fn and item bodies of one file, as inclusive line ranges."""

    def __init__(self, src):
        masked = mask(src)
        self.starts = [0]
        for m in re.finditer("\n", masked):
            self.starts.append(m.end())
        self.fns = []  # (first line, last line, name), 0-based lines
        self.items = []
        for m in _FN.finditer(masked):
            op = body_open(masked, m.end())
            if op is None:
                continue
            end = match_brace(masked, op)
            self.fns.append((self.line(m.start()), self.line(end), m.group(1)))
        for m in _ITEM.finditer(masked):
            op = body_open(masked, m.end())
            if op is None:
                continue
            end = match_brace(masked, op)
            if m.group(3):
                header = " ".join(masked[m.start(3) : op].split())
                header = header.split(" where ")[0]
                name = header
            else:
                name = "%s %s" % (m.group(1), m.group(2))
            self.items.append((self.line(m.start()), self.line(end), name))

    def line(self, offset):
        return bisect.bisect_right(self.starts, offset) - 1

    def enclosing(self, line0):
        """The innermost fn containing the line, else the innermost item, else TOP."""
        for spans in (self.fns, self.items):
            inside = [s for s in spans if s[0] <= line0 <= s[1]]
            if inside:
                return max(inside, key=lambda s: (s[0], -s[1]))[2]
        return TOP


# ---------------------------------------------------------------------------------------
# The hits.
# ---------------------------------------------------------------------------------------


def grep_hits():
    """[(file, line, trimmed text)] for the spec's grep, deduped by (file, line)."""
    dirs = []
    for pattern in DIR_GLOBS:
        dirs.extend(sorted(glob.glob(os.path.join(ROOT, pattern))))
    dirs = [os.path.relpath(d, ROOT) for d in dirs if os.path.isdir(d)]
    proc = subprocess.run(
        ["grep", "-rn", PATTERN, "--include=*.rs"] + dirs,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if proc.returncode not in (0, 1):
        sys.exit("grep failed (rc=%d): %s" % (proc.returncode, proc.stderr.strip()))
    raw = proc.stdout.splitlines()
    seen, hits = set(), []
    for row in raw:
        path, line, text = row.split(":", 2)
        if (path, int(line)) in seen:
            continue
        seen.add((path, int(line)))
        hits.append((path, int(line), text.strip()))
    return len(raw), hits


def keyed(hits):
    """[(key, file, line)] with key = (file, enclosing item, trimmed text)."""
    spans, out = {}, []
    for path, line, text in hits:
        if path not in spans:
            with open(os.path.join(ROOT, path), encoding="utf-8") as fh:
                spans[path] = Spans(fh.read())
        out.append(((path, spans[path].enclosing(line - 1), text), path, line))
    return out


# ---------------------------------------------------------------------------------------
# The rows.
# ---------------------------------------------------------------------------------------


def load_rows():
    with open(CLASSES, encoding="utf-8") as fh:
        rows = json.load(fh)
    problems, table = [], {}
    for n, row in enumerate(rows):
        where = "classes.json row %d" % n
        missing = [k for k in ("file", "fn", "text", "count", "class", "reason") if k not in row]
        if missing:
            problems.append("%s: missing %s" % (where, ", ".join(missing)))
            continue
        if row["class"] not in CLASS_ORDER:
            problems.append("%s: class %r is not one of %s" % (where, row["class"], CLASS_ORDER))
        if not isinstance(row["count"], int) or row["count"] < 1:
            problems.append("%s: count must be a positive integer" % where)
        if not str(row["reason"]).strip():
            problems.append("%s: reason is empty" % where)
        key = (row["file"], row["fn"], row["text"])
        if key in table:
            problems.append("%s: duplicate key %r" % (where, key))
        table[key] = row
    if problems:
        print("\n".join(problems), file=sys.stderr)
        sys.exit(2)
    return table


def stub(key, line):
    file, fn, text = key
    return json.dumps(
        {
            "file": file,
            "fn": fn,
            "text": text,
            "count": 1,
            "class": "?",
            "reason": "?",
            "lines": [line],
        },
        ensure_ascii=False,
    )


def option(argv, name):
    """The value following `name` in argv, or None."""
    if name not in argv:
        return None
    at = argv.index(name) + 1
    if at >= len(argv):
        sys.exit("%s needs a value" % name)
    return argv[at]


def main(argv):
    global ROOT, CLASSES
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ROOT = os.path.abspath(option(argv, "--root") or ROOT)
    CLASSES = os.path.abspath(option(argv, "--classes") or CLASSES)
    show_list = "--list" in argv
    table = load_rows()
    raw_lines, hits = grep_hits()
    sites = keyed(hits)

    by_key = collections.defaultdict(list)
    for key, path, line in sites:
        by_key[key].append((path, line))

    classified = collections.Counter()
    unclassified, stale = [], []
    for key, where in sorted(by_key.items()):
        row = table.get(key)
        covered = row["count"] if row else 0
        for path, line in sorted(where)[:covered]:
            classified[row["class"]] += 1
            if show_list:
                print("  %-15s %s:%d  %s" % (row["class"], path, line, key[1]))
        for path, line in sorted(where)[covered:]:
            unclassified.append((key, line))
    for key, row in sorted(table.items()):
        have = len(by_key.get(key, []))
        if have < row["count"]:
            stale.append((key, row["count"], have))

    try:
        head = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        head = "unknown"
    print("Reduce-site checklist at HEAD %s" % head)
    print("grep: %d lines, %d sites (deduped by (file, line))" % (raw_lines, len(hits)))
    print("%-16s %5s" % ("class", "sites"))
    for cls in CLASS_ORDER:
        print("%-16s %5d" % (cls, classified[cls]))
    print("%-16s %5d" % ("classified", sum(classified.values())))
    print("UNCLASSIFIED %d" % len(unclassified))
    for key, line in unclassified:
        print("  UNCLASSIFIED %s:%d  in %s  `%s`" % (key[0], line, key[1], key[2]))
        print("    row stub: %s" % stub(key, line))
    print("STALE %d" % len(stale))
    for key, want, have in stale:
        print("  STALE %s  in %s  `%s`  (row count %d, hits %d)" % (key[0], key[1], key[2], want, have))
    return 1 if unclassified or stale else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
