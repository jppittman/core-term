#!/usr/bin/env python3
"""Aggregate the beam-extraction sweep's rows into the results documents.

Deterministic columns only. `dag_cost` is a property of the term, so every
number here is exact; wall clock is reported per row but is only quoted when
the host's load allows it.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path


def rows(d):
    for p in sorted(Path(d).glob("*.jsonl")):
        for line in p.read_text().splitlines():
            if line.strip():
                yield json.loads(line)


def arm(mode):
    """(cap, width) from an `egraph_off_on` mode label."""
    cap = mode.split("-")[0].removeprefix("cap")
    width = 1 if "+beam" not in mode else int(mode.split("+beam")[1])
    return int(cap), (width if "+beam" in mode else 0)


def main(indir, outprefix):
    # (cap, width) -> name -> row
    cells = defaultdict(dict)
    for r in rows(indir):
        cap, w = arm(r["mode"])
        prev = cells[(cap, w)].get(r["name"])
        if prev is not None:
            assert prev["dag_cost"] == r["dag_cost"], f"{r['name']}: repeat run disagrees"
            continue
        cells[(cap, w)][r["name"]] = r

    caps = sorted({c for c, _ in cells})
    widths = sorted({w for _, w in cells})
    fams = {}
    for cell in cells.values():
        for n, r in cell.items():
            fams[n] = r["class"]

    lines = []
    csv = ["cap,width,family,kernels,sum_dag_cost,ratio_vs_greedy,sum_bytes,bytes_ratio,"
           "improved,unchanged,worse,fnv_identical,objective_shared,objective_tree,sum_optimize_ms"]
    js = {"schema": "beam-extraction-v1", "cells": []}

    for cap in caps:
        base = cells.get((cap, 0))
        if not base:
            continue
        for w in widths:
            cell = cells.get((cap, w))
            if not cell:
                continue
            per_fam = defaultdict(lambda: dict(
                n=0, dag=0, dag0=0, bytes=0, bytes0=0,
                better=0, same=0, worse=0, fnv=0, shared=0, tree=0, ms=0.0))
            for name, r in cell.items():
                b = base.get(name)
                if b is None:
                    continue
                f = per_fam[fams[name]]
                f["n"] += 1
                f["dag"] += r["dag_cost"]
                f["dag0"] += b["dag_cost"]
                f["bytes"] += r["bytes"]
                f["bytes0"] += b["bytes"]
                f["ms"] += r["optimize_ms"]
                f["fnv"] += int(r["code_fnv"] == b["code_fnv"])
                obj = (r.get("sat") or {}).get("extraction_objective")
                f["shared"] += int(obj == "shared")
                f["tree"] += int(obj in ("tree_cheaper", "tree_only"))
                if r["dag_cost"] < b["dag_cost"]:
                    f["better"] += 1
                elif r["dag_cost"] > b["dag_cost"]:
                    f["worse"] += 1
                else:
                    f["same"] += 1
            for fam, f in sorted(per_fam.items()):
                ratio = f["dag"] / f["dag0"] if f["dag0"] else 1.0
                bratio = f["bytes"] / f["bytes0"] if f["bytes0"] else 1.0
                label = "greedy" if w == 0 else f"beam{w}"
                csv.append(
                    f'{cap},{label},{fam},{f["n"]},{f["dag"]},{ratio:.6f},{f["bytes"]},'
                    f'{bratio:.6f},{f["better"]},{f["same"]},{f["worse"]},{f["fnv"]},'
                    f'{f["shared"]},{f["tree"]},{f["ms"]:.1f}')
                js["cells"].append(dict(cap=cap, width=label, family=fam, kernels=f["n"],
                                        dag_cost=f["dag"], ratio=ratio, bytes=f["bytes"],
                                        improved=f["better"], unchanged=f["same"],
                                        worse=f["worse"], fnv_identical=f["fnv"],
                                        optimize_ms=round(f["ms"], 1)))

    Path(outprefix + ".csv").write_text("\n".join(csv) + "\n")
    Path(outprefix + ".json").write_text(json.dumps(js, indent=2) + "\n")
    print("\n".join(csv))
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
