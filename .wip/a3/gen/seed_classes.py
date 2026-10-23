#!/usr/bin/env -S python3 -I
"""Seed classes.json from spec.md section 5's table, at the base the spec was written at.

This ran once, at 3bfdaded. It joins each table row's `file:line` to the grep's hit at that
line, takes the enclosing item and the trimmed text from the hit, and groups rows with the
same key into one row with a count. It refuses to overwrite an existing classes.json: after
the seed, classes.json is edited by hand (see README.md), and a re-seed would erase that.

  python3 -I seed_classes.py <spec.md> [<out.json>]    (out defaults to classes.json beside it)
"""

import collections
import importlib.util
import json
import os
import re
import sys

sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))

spec = importlib.util.spec_from_file_location("checklist", os.path.join(HERE, "checklist.py"))
checklist = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checklist)

ROW = re.compile(r"^\| `([^`]+):(\d+)` \| (.+?) \| (already-correct|changed-by-A3|left|tests|n/a|A4|A5) \| (.+) \|$")


def section_five(text):
    start = text.index("| Site (`3bfdaded`) |")
    end = text.index("\n---\n", start)
    return text[start:end].splitlines()


def main(argv):
    spec_path = argv[0]
    out = argv[1] if len(argv) > 1 else os.path.join(HERE, "classes.json")
    if os.path.exists(out):
        sys.exit("%s exists; classes.json is hand-edited after the seed, not regenerated" % out)
    with open(spec_path, encoding="utf-8") as fh:
        lines = section_five(fh.read())
    table, bad = {}, []
    for line in lines[2:]:
        m = ROW.match(line)
        if not m:
            bad.append(line)
            continue
        file, num, spec_in, cls, why = m.groups()
        if (file, int(num)) in table:
            bad.append("duplicate row " + line)
        table[(file, int(num))] = (spec_in, cls, why)
    _, hits = checklist.grep_hits()
    sites = checklist.keyed(hits)
    hit_at = {(path, line): key for key, path, line in sites}
    rows = collections.OrderedDict()
    for (file, num), (spec_in, cls, why) in sorted(table.items()):
        key = hit_at.get((file, num))
        if key is None:
            bad.append("table row %s:%d has no grep hit" % (file, num))
            continue
        row = rows.setdefault(
            key,
            {"file": key[0], "fn": key[1], "text": key[2], "count": 0, "class": cls, "reason": why, "lines": []},
        )
        if row["class"] != cls:
            bad.append("key %r has two classes: %s, %s" % (key, row["class"], cls))
        row["count"] += 1
        row["lines"].append(num)
        if why not in row["reason"]:
            row["reason"] += "; " + why
    for key in hit_at.values():
        if key not in rows:
            bad.append("grep hit with no table row: %r" % (key,))
    if bad:
        sys.exit("\n".join(bad))
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(list(rows.values()), fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    print("wrote %d rows covering %d sites to %s" % (len(rows), sum(r["count"] for r in rows.values()), out))


if __name__ == "__main__":
    main(sys.argv[1:])
