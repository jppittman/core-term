#!/usr/bin/env python3
"""Gate 2: pair the k-th saturation telemetry record with the k-th program.

usage: gate2_pair.py PROBE_LINES JSONL     (PROBE_LINES in the probe's call order)
Prints one line per program, sorted by n:
  n=.. applications=.. unions=.. classes_at_stop=.. live=.. iterations=.. stop=..
Exit 2 if the records cannot be paired one to one (a decline, a macro-tier
record, or a different count): the sink appends in call order, so a pairing
that is not exact is a pairing that is wrong.
"""
import json
import sys

probe_path, jsonl_path = sys.argv[1:3]

programs = []
for line in open(probe_path):
    fields = dict(f.split("=", 1) for f in line.split()[1:])
    programs.append(int(fields["n"]))

records = [json.loads(line) for line in open(jsonl_path) if line.strip()]
if len(records) != len(programs):
    sys.exit("gate2: %d telemetry records for %d programs (%s)" % (len(records), len(programs), jsonl_path))

rows = []
for k, (n, rec) in enumerate(zip(programs, records)):
    if "declined" in rec or rec.get("tier") != "runtime":
        sys.exit("gate2: record %d (n=%d) is not a runtime saturation: %s" % (k, n, json.dumps(rec)))
    rows.append((n, "n=%d applications=%s unions=%s classes_at_stop=%s live=%s iterations=%s stop=%s" % (
        n, rec["application_count"], rec["union_count"], rec["classes_at_stop"],
        rec["live_classes"], rec["iterations"], rec["stop_reason"])))
for _, text in sorted(rows):
    print(text)
