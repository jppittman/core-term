#!/usr/bin/env python3
"""Gate 5's report: callgrind totals plus the native runs' medians, as JSON.

usage: measure_report.py TAG CG_OUT RSS_KIB_LIST WALL_MS_LIST [BASE_JSON]
  RSS_KIB_LIST, WALL_MS_LIST: comma-separated, one entry per native run.

Reads `callgrind_annotate --inclusive=yes --threshold=100 CG_OUT` and reports
total Ir, `Saturation::saturate` inclusive Ir and extraction inclusive Ir
(SATURATE and EXTRACTION below name the functions; every matching line is listed
under "matched" so a rename shows up as an empty match instead of a wrong number).
Measured, never asserted: with BASE_JSON the report adds each figure's change
against it, and the exit status is 0 either way.
"""
import json
import re
import statistics
import subprocess
import sys

# Function names, as callgrind_annotate prints them (demangled, "file:function [object]").
SATURATE = re.compile(r"Saturation>::saturate$")
EXTRACTION = re.compile(r"::extract_dag_over(::<.*>)?$")

LINE = re.compile(r"^\s*([\d,]+)\s+\(\s*[\d.]+%\)\s+(.*?)\s*$")

tag, cg_out, rss_list, wall_list = sys.argv[1:5]
base_path = sys.argv[5] if len(sys.argv) > 5 else None

annotated = subprocess.run(
    ["callgrind_annotate", "--inclusive=yes", "--threshold=100", cg_out],
    check=True, capture_output=True, text=True,
).stdout

total = None
functions = []  # (inclusive Ir, "file:function [object]")
for text in annotated.splitlines():
    m = LINE.match(text)
    if not m:
        continue
    ir, name = int(m.group(1).replace(",", "")), m.group(2)
    if name == "PROGRAM TOTALS":
        total = ir
    else:
        functions.append((ir, name))
if total is None:
    sys.exit("measure_report: no PROGRAM TOTALS in callgrind_annotate output for " + cg_out)


def inclusive(pattern):
    """The largest matching function's inclusive Ir, and every match."""
    found = []
    for ir, name in functions:
        function = re.sub(r"\s*\[[^\]]*\]$", "", name.split(":", 1)[-1] if ":" in name else name)
        if pattern.search(function):
            found.append((ir, name))
    found.sort(reverse=True)
    return (found[0][0] if found else None), found


saturate_ir, saturate_matches = inclusive(SATURATE)
extraction_ir, extraction_matches = inclusive(EXTRACTION)

rss = [int(x) for x in rss_list.split(",")]
wall = [int(x) for x in wall_list.split(",")]
report = {
    "tag": tag,
    "total_ir": total,
    "saturate_inclusive_ir": saturate_ir,
    "extraction_inclusive_ir": extraction_ir,
    "median_peak_rss_kib": int(statistics.median(rss)),
    "median_wall_ms": int(statistics.median(wall)),
    "peak_rss_kib_runs": rss,
    "wall_ms_runs": wall,
    "matched": {
        "saturate": [{"ir": ir, "name": name} for ir, name in saturate_matches[:5]],
        "extraction": [{"ir": ir, "name": name} for ir, name in extraction_matches[:5]],
    },
}

if base_path:
    base = json.load(open(base_path))
    report["vs_base"] = {
        key: (None if base.get(key) in (None, 0) or report[key] is None
              else round(100.0 * (report[key] - base[key]) / base[key], 3))
        for key in ("total_ir", "saturate_inclusive_ir", "extraction_inclusive_ir",
                    "median_peak_rss_kib", "median_wall_ms")
    }
    report["vs_base"]["unit"] = "percent change from " + base_path.rsplit("/", 1)[-1]

print(json.dumps(report, indent=2))
