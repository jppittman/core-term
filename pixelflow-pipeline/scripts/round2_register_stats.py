#!/usr/bin/env python3
"""Round 2 Register statistics — UNGUIDED rows only.

Reads the per-expression, per-checkpoint curve CSVs written by
`phase3_round2_unguided_curves` (modes i/ii) and `phase3_round2_new_rules`
(mode iii) and computes every number the Register document
(`docs/plans/2026-09-01-phase3-round2-registration.md`) fixes:

  U(|R|)   median unguided regret at B against the unguided-only
           closure-aware reference AT THAT |R| (min over all checkpoints
           of that rule set's own curve; never pooled across |R|)
  L(|R|)   median truncation loss (cost@B vs cost@4B), Round 1's convention
  Y(|R|)   1 - (1 + L/2) / (1 + L)
  Delta1   95% bootstrap CI half-width of median U(62) (10,000 resamples,
           random.Random(42), order-statistic percentiles)
  Delta2   max(0.02, Y(|R|max) - Y(62)) per mode
  rho      Spearman rank correlation of U(|R|) against |R| (average ranks
           for ties; undefined when every U is tied)
  visible  share of classical expressions whose cost@B differs from the
           |R|=62 curve at the same B — "is the inflation reachable at B"
  fid      closure gain (ref(e,62) - ref(e,|R|)) / ref(e,62), the §4.2 column

Refuses to run on any row that is not an unguided curve (the schema has no
arm column: a guided run would be a different binary and a different file;
this script asserts the expected header exactly). Fails loud on any missing
grid point it was told to expect. No timing anywhere.

Usage:
  round2_register_stats.py --csv A.csv [--csv B.csv ...] \
      --expect base,dup:93,... --out-json docs/results/...json
"""

import argparse
import csv
import json
import math
import random
import statistics
import sys
from collections import defaultdict

HEADER = [
    "rule_set", "num_rules", "fingerprint", "expr_name", "origin", "tier",
    "node_count", "app_target", "app_actual", "sweeps_actual", "evals_actual",
    "apps_per_sweep", "cost", "ended", "ended_at_apps",
]
GRID = [25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200, 102400, 204800]
BUDGETS = [100, 200]
CYCLE_COST_THRESHOLD = 900_000
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 42
DELTA2_FLOOR = 0.02
EPSILON = 0.005


def die(msg):
    print(f"round2_register_stats: {msg}", file=sys.stderr)
    sys.exit(1)


def percentile(sorted_vals, p):
    if not sorted_vals:
        die("percentile of empty list")
    pos = p * (len(sorted_vals) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return sorted_vals[lo]
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def quartiles(vals):
    s = sorted(vals)
    return {
        "n": len(s),
        "p25": percentile(s, 0.25),
        "median": percentile(s, 0.5),
        "p75": percentile(s, 0.75),
        "p90": percentile(s, 0.9),
    }


def regret(cost, ref):
    if ref == 0:
        return 0.0 if cost == 0 else math.inf
    return (cost - ref) / ref


def loss_pct(cost_b, cost_4b):
    if cost_4b == 0:
        return 0.0 if cost_b == 0 else math.inf
    return (cost_b - cost_4b) / cost_4b * 100.0


def y_from_loss_pct(l_pct):
    l = l_pct / 100.0
    return 1.0 - (1.0 + l / 2.0) / (1.0 + l)


def spearman(xs, ys):
    """Average-rank Spearman; None when either side is fully tied."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    if len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    rx, ry = ranks(xs), ranks(ys)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den


def bootstrap_median_ci(vals, resamples, seed):
    rng = random.Random(seed)
    n = len(vals)
    meds = []
    for _ in range(resamples):
        sample = [vals[rng.randrange(n)] for _ in range(n)]
        meds.append(statistics.median(sample))
    meds.sort()
    lo = meds[int(0.025 * resamples)]
    hi = meds[int(0.975 * resamples) - 1]
    return lo, hi


def load(paths):
    curves = defaultdict(dict)  # (rule_set, expr) -> {target: row}
    meta = {}
    for path in paths:
        with open(path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            if header != HEADER:
                die(f"{path}: unexpected header {header} (expected {HEADER})")
            for row in reader:
                r = dict(zip(HEADER, row))
                rs = r["rule_set"]
                key = (rs, r["expr_name"])
                target = int(r["app_target"])
                if target in curves[key]:
                    # The |R|=62 arm is run once per binary; a second copy
                    # must be byte-identical (deterministic saturation) or
                    # the two files were not produced by the same code.
                    if curves[key][target] != r:
                        die(f"{path}: row for {key} at target {target} differs from an earlier file")
                    continue
                curves[key][target] = r
                m = meta.setdefault(rs, {"num_rules": int(r["num_rules"]), "fingerprint": r["fingerprint"]})
                if m["num_rules"] != int(r["num_rules"]) or m["fingerprint"] != r["fingerprint"]:
                    die(f"{path}: rule set {rs} has inconsistent identity across rows")
    for key, by_target in curves.items():
        if sorted(by_target) != GRID:
            die(f"curve {key} does not cover the full grid: {sorted(by_target)}")
    return curves, meta


def quiescence_cost(by_target):
    ended_at = int(next(iter(by_target.values()))["ended_at_apps"])
    q = None
    for t in GRID:
        r = by_target[t]
        if int(r["app_actual"]) <= ended_at:
            q = int(r["cost"])
    return q if q is not None else int(by_target[GRID[-1]]["cost"])


def per_rule_set(curves, rs, tier, base_rs="base"):
    exprs = sorted(e for (r, e) in curves if r == rs and curves[(r, e)][GRID[0]]["tier"] == tier)
    if not exprs:
        die(f"no {tier} curves for rule set {rs}")
    out = {"n": len(exprs)}
    # apps_per_sweep is fixed per (rule_set, expr) curve — one throwaway
    # probe value, identical at every grid checkpoint of that curve — so any
    # single checkpoint's row carries it; §0's binding rule ("every budget
    # reported in sweeps") reads off B in sweeps as B / this median.
    out["apps_per_sweep"] = quartiles(
        [int(curves[(rs, e)][GRID[0]]["apps_per_sweep"]) for e in exprs]
    )
    refs = {}
    for e in exprs:
        refs[e] = min(int(r["cost"]) for r in curves[(rs, e)].values())
    out["ref_zero_count"] = sum(1 for v in refs.values() if v == 0)
    for b in BUDGETS:
        cost_b = [int(curves[(rs, e)][b]["cost"]) for e in exprs]
        cost_4b = [int(curves[(rs, e)][4 * b]["cost"]) for e in exprs]
        app_b = [int(curves[(rs, e)][b]["app_actual"]) for e in exprs]
        sweeps_b = [int(curves[(rs, e)][b]["sweeps_actual"]) for e in exprs]
        evals_b = [int(curves[(rs, e)][b]["evals_actual"]) for e in exprs]
        # Matches enumerated per application, cumulative through checkpoint B
        # (the §7.1 overhead curve) — undefined (skipped) for a curve that
        # quiesced with zero applications recorded at B.
        evals_per_app_b = [ev / ap for ev, ap in zip(evals_b, app_b) if ap > 0]
        regs = [regret(c, refs[e]) for e, c in zip(exprs, cost_b)]
        losses = [loss_pct(c, c4) for c, c4 in zip(cost_b, cost_4b)]
        finite_losses = [l for l in losses if math.isfinite(l)]
        visible = None
        if (base_rs, exprs[0]) in curves and rs != base_rs:
            visible = sum(
                1 for e, c in zip(exprs, cost_b) if int(curves[(base_rs, e)][b]["cost"]) != c
            )
        out[f"B{b}"] = {
            "cost_at_B": quartiles(cost_b),
            "cost_at_4B": quartiles(cost_4b),
            "app_actual_at_B": quartiles(app_b),
            "sweeps_actual_at_B": quartiles(sweeps_b),
            "evals_per_app_at_B": quartiles(evals_per_app_b) if evals_per_app_b else None,
            "regret": quartiles(regs),
            "regret_infinite_count": sum(1 for r in regs if math.isinf(r)),
            "trunc_loss_pct": quartiles(finite_losses),
            "trunc_loss_infinite_count": len(losses) - len(finite_losses),
            "Y": y_from_loss_pct(percentile(sorted(finite_losses), 0.5)),
            "cost_at_B_differs_from_base_count": visible,
            "_regrets": regs,
        }
    q_all = [quiescence_cost(curves[(rs, e)]) for e in exprs]
    q_ok = [q for q in q_all if q < CYCLE_COST_THRESHOLD]
    ended = [curves[(rs, e)][GRID[0]]["ended"] for e in exprs]
    out["curve_end"] = {
        "quiescence_cost_all": quartiles(q_all),
        "quiescence_cost_excl_cycle": quartiles(q_ok) if q_ok else None,
        "cycle_sentinel_hits": len(q_all) - len(q_ok),
        "ended_at_apps": quartiles([int(curves[(rs, e)][GRID[0]]["ended_at_apps"]) for e in exprs]),
        "ended": {s: ended.count(s) for s in sorted(set(ended))},
    }
    # Per-checkpoint share of expressions whose cost differs from base — the
    # capacity picture: where along the curve does |R| become visible?
    if rs != base_rs:
        out["cost_differs_from_base_by_checkpoint"] = {
            str(t): sum(
                1 for e in exprs
                if int(curves[(rs, e)][t]["cost"]) != int(curves[(base_rs, e)][t]["cost"])
            )
            for t in GRID
        }
        # Closure gain against the base reference (§4.2): positive means the
        # inflated set reaches a strictly cheaper form somewhere on its curve.
        fids = []
        for e in exprs:
            ref_base = min(int(r["cost"]) for r in curves[(base_rs, e)].values())
            if ref_base == 0 or ref_base >= CYCLE_COST_THRESHOLD:
                continue
            fids.append((ref_base - refs[e]) / ref_base * 100.0)
        out["closure_gain_pct"] = {
            **quartiles(fids),
            "positive_count": sum(1 for f in fids if f > 1e-9),
            "negative_count": sum(1 for f in fids if f < -1e-9),
        }
    out["_refs"] = refs
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="append", required=True)
    ap.add_argument("--expect", required=True, help="comma-separated rule sets that must be present")
    ap.add_argument("--modes", required=True,
                    help="mode spec, e.g. 'i=base,dup:93,dup:124,dup:186,dup:248;ii=base,comp:93,comp:124;iii=base,new:95'")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", help="markdown tables for the Register document")
    args = ap.parse_args()

    curves, meta = load(args.csv)
    expected = args.expect.split(",")
    missing = [rs for rs in expected if rs not in meta]
    if missing:
        die(f"expected rule sets missing from the CSVs: {missing} (present: {sorted(meta)})")

    result = {"grid": GRID, "budgets": BUDGETS, "rule_sets": meta, "tiers": {}, "modes": {}}
    for tier in ["blitz", "rapid", "classical"]:
        result["tiers"][tier] = {}
        for rs in expected:
            stats = per_rule_set(curves, rs, tier)
            result["tiers"][tier][rs] = {k: v for k, v in stats.items() if not k.startswith("_")}
            for b in BUDGETS:
                result["tiers"][tier][rs][f"B{b}"].pop("_regrets")

    classical = {rs: per_rule_set(curves, rs, "classical") for rs in expected}
    # Delta1 from |R|=62 only.
    result["delta1"] = {}
    for b in BUDGETS:
        regs = classical["base"][f"B{b}"]["_regrets"]
        lo, hi = bootstrap_median_ci(regs, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED)
        result["delta1"][f"B{b}"] = {
            "median": statistics.median(regs), "ci_lo": lo, "ci_hi": hi,
            "half_width": (hi - lo) / 2.0,
            "resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED,
        }

    for spec in args.modes.split(";"):
        mode, sets = spec.split("=")
        sets = sets.split(",")
        if sets[0] != "base":
            die(f"mode {mode}: first grid point must be base, got {sets[0]}")
        rcount = [meta[rs]["num_rules"] for rs in sets]
        m = {"rule_sets": sets, "R": rcount, "B": {}}
        for b in BUDGETS:
            U = [classical[rs][f"B{b}"]["regret"]["median"] for rs in sets]
            L = [classical[rs][f"B{b}"]["trunc_loss_pct"]["median"] for rs in sets]
            Y = [classical[rs][f"B{b}"]["Y"] for rs in sets]
            d1 = result["delta1"][f"B{b}"]["half_width"]
            rho = spearman(rcount, U)
            # least-squares slope of U against |R|, per rule
            mx, my = statistics.mean(rcount), statistics.mean(U)
            sxx = sum((x - mx) ** 2 for x in rcount)
            slope_u = sum((x - mx) * (u - my) for x, u in zip(rcount, U)) / sxx if sxx else None
            m["B"][f"B{b}"] = {
                "U": U, "L": L, "Y": Y,
                "spearman_rho": rho,
                "U_max_minus_U_62": U[-1] - U[0],
                "delta1": d1,
                "H1_direction_holds": (rho is not None and rho >= 0.9),
                "H1_effect_holds": (U[-1] - U[0]) >= d1,
                "delta2": max(DELTA2_FLOOR, Y[-1] - Y[0]),
                "one_minus_Y": [1.0 - y for y in Y],
                "slope_U_per_rule": slope_u,
                "epsilon": EPSILON,
            }
        result["modes"][mode] = m

    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=1, sort_keys=True)
    print(f"wrote {args.out_json}")
    if args.out_md:
        with open(args.out_md, "w") as f:
            f.write(render_md(result, expected))
        print(f"wrote {args.out_md}")


def slope_str(slope):
    if slope == 0:
        return "0 (exact)"
    return f"{slope * 100:+.5f} pts/rule"


def fmt_pct(x):
    return "inf" if math.isinf(x) else f"{x * 100:.2f}"


def render_md(result, expected):
    out = []
    c = result["tiers"]["classical"]
    rs_R = result["rule_sets"]
    out.append("**Classical band (n=188), absolute cost and where the inflation is visible.** "
               "`visible@B` = expressions whose cost@B differs from the |R|=62 curve's; "
               "`first visible` = smallest grid checkpoint at which any expression's cost differs "
               "from |R|=62 (`never` = identical at all 14 checkpoints).\n\n")
    out.append("| rule set | \\|R\\| | fingerprint | cost@100 q1/med/q3 | cost@200 q1/med/q3 | cost@400 med | cost@800 med | "
               "curve-end med (excl. cycle) | cycle hits | app_actual@100 med | visible@100 | visible@200 | first visible | "
               "apps-to-end med | ended |\n|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|\n")
    for rs in expected:
        x = c[rs]
        b1, b2 = x["B100"], x["B200"]
        q = lambda d: f"{d['p25']:.0f} / {d['median']:.1f} / {d['p75']:.0f}"
        end = x["curve_end"]
        endq = end["quiescence_cost_excl_cycle"]
        first = "n/a"
        if "cost_differs_from_base_by_checkpoint" in x:
            diffs = x["cost_differs_from_base_by_checkpoint"]
            hits = [t for t in GRID if diffs[str(t)] > 0]
            first = str(hits[0]) if hits else "never"
        out.append(f"| `{rs}` | {rs_R[rs]['num_rules']} | `{rs_R[rs]['fingerprint']}` | {q(b1['cost_at_B'])} | {q(b2['cost_at_B'])} | "
                   f"{b1['cost_at_4B']['median']:.1f} | {b2['cost_at_4B']['median']:.1f} | "
                   f"{endq['median']:.1f} | {end['cycle_sentinel_hits']} | {b1['app_actual_at_B']['median']:.0f} | "
                   f"{b1['cost_at_B_differs_from_base_count'] if b1['cost_at_B_differs_from_base_count'] is not None else '—'} | "
                   f"{b2['cost_at_B_differs_from_base_count'] if b2['cost_at_B_differs_from_base_count'] is not None else '—'} | {first} | "
                   f"{end['ended_at_apps']['median']:.0f} | {', '.join(f'{k}={v}' for k, v in end['ended'].items())} |\n")
    out.append("\n**Classical band, unguided regret U against the unguided-only closure-aware reference at the same |R|, "
               "truncation loss L, and Y.** Percentages; regret quartiles are per-expression.\n\n")
    out.append("| rule set | \\|R\\| | U@100 med | p25 | p75 | p90 | U@200 med | p25 | p75 | p90 | L@100 med | Y@100 | L@200 med | Y@200 | closure gain vs 62: med / p90 / >0 / <0 (n) |\n"
               "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
    for rs in expected:
        x = c[rs]
        b1, b2 = x["B100"], x["B200"]
        r1, r2 = b1["regret"], b2["regret"]
        fid = x.get("closure_gain_pct")
        fid_s = (f"{fid['median']:.3f} / {fid['p90']:.3f} / {fid['positive_count']} / {fid['negative_count']} ({fid['n']})"
                 if fid else "— (reference)")
        out.append(f"| `{rs}` | {rs_R[rs]['num_rules']} | {fmt_pct(r1['median'])} | {fmt_pct(r1['p25'])} | {fmt_pct(r1['p75'])} | {fmt_pct(r1['p90'])} | "
                   f"{fmt_pct(r2['median'])} | {fmt_pct(r2['p25'])} | {fmt_pct(r2['p75'])} | {fmt_pct(r2['p90'])} | "
                   f"{b1['trunc_loss_pct']['median']:.3f} | {b1['Y'] * 100:.2f} | {b2['trunc_loss_pct']['median']:.3f} | {b2['Y'] * 100:.2f} | {fid_s} |\n")
    out.append("\n**Sweeps and match-enumeration overhead (classical, v2 §0.2/§7.1).** "
               "`apps_per_sweep` is one throwaway one-sweep probe per expression, median over the "
               "band; `B in sweeps` = B / that median (how much of one full rule-order pass a "
               "budget spends); `evals/app@B` = cumulative `EGraph::total_evals` / cumulative "
               "applications through checkpoint B — matches enumerated per application actually "
               "taken, the §7.1 flatness check.\n\n")
    out.append("| rule set | \\|R\\| | apps_per_sweep med | B=100 in sweeps | B=200 in sweeps | "
               "sweeps_actual@100 med | sweeps_actual@200 med | evals/app@100 med | evals/app@200 med |\n"
               "|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for rs in expected:
        x = c[rs]
        aps = x["apps_per_sweep"]["median"]
        b1, b2 = x["B100"], x["B200"]
        ev1 = b1["evals_per_app_at_B"]["median"] if b1["evals_per_app_at_B"] else float("nan")
        ev2 = b2["evals_per_app_at_B"]["median"] if b2["evals_per_app_at_B"] else float("nan")
        out.append(f"| `{rs}` | {rs_R[rs]['num_rules']} | {aps:.1f} | {100 / aps:.2f} | {200 / aps:.2f} | "
                   f"{b1['sweeps_actual_at_B']['median']:.2f} | {b2['sweeps_actual_at_B']['median']:.2f} | "
                   f"{ev1:.2f} | {ev2:.2f} |\n")
    for tier in ["blitz", "rapid"]:
        t = result["tiers"][tier]
        n = next(iter(t.values()))["n"]
        out.append(f"\n**{tier} (n={n}) — reported, no claim.**\n\n| rule set | \\|R\\| | cost@100 med | cost@200 med | U@100 med | L@100 med | apps-to-end med | ended |\n|---|---:|---:|---:|---:|---:|---:|---|\n")
        for rs in expected:
            x = t[rs]
            out.append(f"| `{rs}` | {rs_R[rs]['num_rules']} | {x['B100']['cost_at_B']['median']:.1f} | {x['B200']['cost_at_B']['median']:.1f} | "
                       f"{fmt_pct(x['B100']['regret']['median'])} | {x['B100']['trunc_loss_pct']['median']:.3f} | {x['curve_end']['ended_at_apps']['median']:.0f} | "
                       f"{', '.join(f'{k}={v}' for k, v in x['curve_end']['ended'].items())} |\n")
    out.append("\n**Per-mode H1 statistics (classical, from the tables above).**\n\n| mode | grid \\|R\\| | B | U(\\|R\\|) | Spearman rho | U(max) - U(62) | Delta1 | H1 direction | H1 effect | Y(\\|R\\|) | Delta2 | LS slope of U per rule |\n|---|---|---:|---|---:|---:|---:|---|---|---|---:|---:|\n")
    for mode, m in result["modes"].items():
        for b in BUDGETS:
            x = m["B"][f"B{b}"]
            rho = "undefined (all tied)" if x["spearman_rho"] is None else f"{x['spearman_rho']:.3f}"
            out.append(f"| ({mode}) | {m['R']} | {b} | {[round(u * 100, 3) for u in x['U']]} | {rho} | {x['U_max_minus_U_62'] * 100:+.3f} | {x['delta1'] * 100:.3f} | "
                       f"{'holds' if x['H1_direction_holds'] else 'FAILS'} | {'holds' if x['H1_effect_holds'] else 'FAILS'} | {[round(y * 100, 2) for y in x['Y']]} | {x['delta2']:.3f} | "
                       f"{slope_str(x['slope_U_per_rule'])} |\n")
    return "".join(out)


if __name__ == "__main__":
    main()
