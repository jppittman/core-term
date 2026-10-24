#!/usr/bin/env python3
"""Gate 2x: the extraction half of the saturation telemetry records.

usage: gate2x_fields.py JSONL
One line per record, in file order (the k-th record is the k-th program's):
  record=k extraction_objective=.. extracted_latency_prior_cost=.. shared_pass_bytes=..
Numbers print as the sink wrote them (floats are not re-formatted). A record
without the fields (a decline) prints `declined=..` instead.
"""
import json
import sys

for k, line in enumerate(l for l in open(sys.argv[1]) if l.strip()):
    rec = json.loads(line, parse_float=str)
    if "declined" in rec:
        print("record=%d declined=%s" % (k, rec["declined"]))
        continue
    print("record=%d extraction_objective=%s extracted_latency_prior_cost=%s shared_pass_bytes=%s" % (
        k, rec["extraction_objective"], rec["extracted_latency_prior_cost"], rec["shared_pass_bytes"]))
