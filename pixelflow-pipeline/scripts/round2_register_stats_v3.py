#!/usr/bin/env python3
"""Round 2 v3 statistics — the |R| effect with sweep order held fixed, and the
order effect on its own (`docs/plans/2026-09-01-phase3-round2-registration-v3.md`).

Reuses `round2_register_stats.py`'s loader and per-rule-set statistics
verbatim (same HEADER, same GRID, same closure-aware-reference convention,
same bootstrap protocol) rather than re-deriving them — a copy here would be
exactly the kind of second definition CLAUDE.md warns against. This script
adds only what v2's tool does not compute:

  Delta_U(p)  U(p) - U(OrderMatchedBase(seed, |p|))            (registration-v3 SS3)
  order table  U/L for base vs 3 Shuffled seeds vs StaticReorder(NumericFirst)
               (SS4), classical primary, rapid/blitz reported without claim
  seed spread  range and IQR of U(dup:124) / U(comp:93) across 3 interleave
               seeds (SS4's "seed sensitivity of an inflated point" addendum)
  differing counts  cost@B(p) != cost@B(reference) — computed against
               OrderMatchedBase for SS3's table, against `base` for SS4's
               (round2_register_stats.per_rule_set already gives the latter)

No timing anywhere. Fails loud (assertion) on any missing rule set — never
silently drops a row.

Usage:
  round2_register_stats_v3.py \
      --v2-csv docs/results/2026-09-01-round2-unguided-vs-rulecount-v2.csv \
      --matched-csv docs/results/2026-09-01-round2-order-matched-base-v3.csv \
      --order-csv docs/results/2026-09-01-round2-order-effect-v3.csv \
      --seed-csv docs/results/2026-09-01-round2-seed-sensitivity-v3.csv \
      --out-json docs/results/2026-09-01-round2-unguided-vs-rulecount-v3.json \
      --out-md docs/results/2026-09-01-round2-unguided-vs-rulecount-v3.md

(The three `--*-csv` inputs are the raw per-run outputs of
`phase3_round2_unguided_curves`; their row-level union is written separately
to `docs/results/2026-09-01-round2-unguided-vs-rulecount-v3.csv` by a plain
`cat` of the three files' bodies under one header, not by this script.)
"""

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from round2_register_stats import (  # noqa: E402
    HEADER, GRID, BUDGETS, per_rule_set, quartiles, fmt_pct,
    bootstrap_median_ci, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, DELTA2_FLOOR,
)

# Design SS7.1's flatness threshold for the unguided half of the overhead
# precondition (v2 SS5.6): median evals_actual/app_actual at B must stay
# within this multiple of its |R|=62 (production-order `base`) value.
OVERHEAD_FLAT_MULTIPLE = 2.0

# |R|max per mode on the realized grid (v2 SS4: comp:186/comp:248 never
# completed), where Delta2(v3) is read off.
LARGEST_INFLATED = {"i": "dup:248", "ii": "comp:124", "iii": "new:95"}

# The interleave seed every v2 inflated point (and thus v3's OrderMatchedBase
# reference) was built with — DEFAULT_INTERLEAVE_SEED, decimal form.
REGISTERED_SEED = 0x20260901

# Smallest inflated point per mode — where Delta1(v3) (SS5.2) is measured.
SMALLEST_INFLATED = {"i": "dup:93", "ii": "comp:93", "iii": "new:95"}

# Mode -> (v2 rule_set spec, its |R|, the matching OrderMatchedBase rule_set).
MODE_POINTS = {
    "i": [
        ("dup:93", 93, "base:matched:0x20260901:93"),
        ("dup:124", 124, "base:matched:0x20260901:124"),
        ("dup:186", 186, "base:matched:0x20260901:186"),
        ("dup:248", 248, "base:matched:0x20260901:248"),
    ],
    "ii": [
        ("comp:93", 93, "base:matched:0x20260901:93"),
        ("comp:124", 124, "base:matched:0x20260901:124"),
    ],
    "iii": [
        ("new:95", 95, "base:matched:0x20260901:95"),
    ],
}

ORDER_ROW_SETS = [
    "base",
    "base:shuffled:1",
    "base:shuffled:2",
    "base:shuffled:3",
    "base:static:numeric-first",
]

SEED_SETS = {
    "dup:124": ["dup:124", "dup:124:interleave:1", "dup:124:interleave:2"],
    "comp:93": ["comp:93", "comp:93:interleave:1", "comp:93:interleave:2"],
}


def die(msg):
    print(f"round2_register_stats_v3: {msg}", file=sys.stderr)
    sys.exit(1)


def load_rows(paths):
    """Same shape as round2_register_stats.load, duplicated only because that
    module's `load` is a free function over a fixed path list and we want to
    merge four files here; the per-row logic (dedup-check on overlapping
    (rule_set, expr, target) keys, e.g. `base` appearing in more than one
    file) is identical."""
    curves = defaultdict(dict)
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
                    if curves[key][target] != r:
                        die(f"{path}: row for {key} at target {target} differs from an earlier file "
                            f"(same rule_set/expr/checkpoint must be byte-identical — deterministic "
                            f"saturation)")
                    continue
                curves[key][target] = r
                m = meta.setdefault(rs, {"num_rules": int(r["num_rules"]), "fingerprint": r["fingerprint"]})
                if m["num_rules"] != int(r["num_rules"]) or m["fingerprint"] != r["fingerprint"]:
                    die(f"{path}: rule set {rs} has inconsistent identity across rows")
    for key, by_target in curves.items():
        if sorted(by_target) != GRID:
            die(f"curve {key} does not cover the full grid: {sorted(by_target)}")
    return curves, meta


def differing_count(curves, rs_a, rs_b, tier, b):
    exprs = sorted(e for (r, e) in curves if r == rs_a and curves[(r, e)][GRID[0]]["tier"] == tier)
    exprs_b = set(e for (r, e) in curves if r == rs_b and curves[(r, e)][GRID[0]]["tier"] == tier)
    missing = [e for e in exprs if e not in exprs_b]
    if missing:
        die(f"{rs_b}: missing {len(missing)} {tier} expressions present in {rs_a} "
            f"(first: {missing[0]!r}) — cannot compute a differing count")
    return sum(
        1 for e in exprs
        if int(curves[(rs_a, e)][b]["cost"]) != int(curves[(rs_b, e)][b]["cost"])
    )


def spearman_or_none(xs, ys):
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
    return num / den if den else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2-csv", required=True)
    ap.add_argument("--matched-csv", required=True)
    ap.add_argument("--order-csv", required=True)
    ap.add_argument("--seed-csv", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    curves, meta = load_rows([args.v2_csv, args.matched_csv, args.order_csv, args.seed_csv])

    all_needed = (
        ["base"]
        + [p[0] for m in MODE_POINTS.values() for p in m]
        + [p[2] for m in MODE_POINTS.values() for p in m]
        + ORDER_ROW_SETS
        + [rs for sets in SEED_SETS.values() for rs in sets]
    )
    missing = sorted(set(rs for rs in all_needed if rs not in meta))
    if missing:
        die(f"rule sets missing from the supplied CSVs: {missing}")

    result = {"grid": GRID, "budgets": BUDGETS, "registered_seed": REGISTERED_SEED,
              "modes": {}, "order_effect": {}, "seed_sensitivity": {}}

    # ---- classical/rapid/blitz per-rule-set stats for everything, once ----
    stats = {}
    for tier in ["classical", "rapid", "blitz"]:
        stats[tier] = {rs: per_rule_set(curves, rs, tier) for rs in set(all_needed)}

    # ================================================================
    # SS3 — Delta_U(p) = U(p) - U(OrderMatchedBase(seed, |p|)), order held
    # fixed. classical only (the pre-registered band).
    # ================================================================
    c = stats["classical"]
    for mode, points in MODE_POINTS.items():
        mode_out = {"points": []}
        rvals, delta_u = {100: [], 200: []}, {100: [], 200: []}
        for rs, R, ref_rs in points:
            row = {"rule_set": rs, "R": R, "order_matched_base": ref_rs}
            for b in BUDGETS:
                u_p = c[rs][f"B{b}"]["regret"]["median"]
                u_ref = c[ref_rs][f"B{b}"]["regret"]["median"]
                du = u_p - u_ref
                row[f"U_B{b}"] = u_p
                row[f"U_matched_B{b}"] = u_ref
                row[f"delta_U_B{b}"] = du
                row[f"differing_from_matched_B{b}"] = differing_count(curves, rs, ref_rs, "classical", b)
                rvals[b].append(R)
                delta_u[b].append(du)
            mode_out["points"].append(row)
        for b in BUDGETS:
            mode_out[f"spearman_rho_delta_U_B{b}"] = spearman_or_none(rvals[b], delta_u[b])
            mode_out[f"delta_U_at_max_R_B{b}"] = delta_u[b][-1] if delta_u[b] else None
            mode_out[f"delta_U_at_min_inflated_R_B{b}"] = delta_u[b][0] if delta_u[b] else None

        # Delta1(v3) (SS5.2): 95% bootstrap CI of the median of PAIRED
        # per-expression delta-regret (regret_p(e) - regret_matched(e)) at
        # the smallest inflated point this mode has. Both rule sets are
        # evaluated over the identical 188-expression classical universe, in
        # the same sorted order per_rule_set builds its `_regrets` list in
        # (`sorted(e for (r, e) in curves if r == rs and ...)`), so index-wise
        # pairing is exact — never re-sorted or re-matched by name here.
        smallest_rs = SMALLEST_INFLATED[mode]
        smallest_ref = next(ref for rs, _, ref in points if rs == smallest_rs)
        mode_out["delta1_v3"] = {}
        for b in BUDGETS:
            regs_p = c[smallest_rs][f"B{b}"]["_regrets"]
            regs_ref = c[smallest_ref][f"B{b}"]["_regrets"]
            assert len(regs_p) == len(regs_ref), (
                f"{mode}: {smallest_rs} has {len(regs_p)} classical regrets, "
                f"{smallest_ref} has {len(regs_ref)} — universes must match"
            )
            paired = [p - r for p, r in zip(regs_p, regs_ref)]
            lo, hi = bootstrap_median_ci(paired, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED)
            mode_out["delta1_v3"][f"B{b}"] = {
                "at_rule_set": smallest_rs, "median": statistics.median(paired),
                "ci_lo": lo, "ci_hi": hi, "half_width": (hi - lo) / 2.0,
                "resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED,
            }
            d1 = mode_out["delta1_v3"][f"B{b}"]["half_width"]
            du_max = mode_out[f"delta_U_at_max_R_B{b}"]
            mode_out[f"H1_v3_effect_holds_B{b}"] = du_max is not None and du_max >= d1
        rho100 = mode_out["spearman_rho_delta_U_B100"]
        mode_out["H1_v3_direction_holds"] = rho100 is not None and rho100 >= 0.9
        result["modes"][mode] = mode_out

    # ================================================================
    # SS4 — order effect in isolation, all |R|=62. classical primary,
    # rapid/blitz reported without claim.
    # ================================================================
    for tier in ["classical", "rapid", "blitz"]:
        t = stats[tier]
        rows = []
        for rs in ORDER_ROW_SETS:
            row = {"rule_set": rs}
            for b in BUDGETS:
                row[f"U_B{b}"] = t[rs][f"B{b}"]["regret"]["median"]
                row[f"L_B{b}"] = t[rs][f"B{b}"]["trunc_loss_pct"]["median"]
                row[f"differing_from_base_B{b}"] = (
                    0 if rs == "base" else differing_count(curves, rs, "base", tier, b)
                )
            rows.append(row)
        result["order_effect"][tier] = rows

    # ================================================================
    # Seed sensitivity: dup:124 and comp:93 under registered seed + 2 more.
    # classical only.
    # ================================================================
    for base_rs, sets in SEED_SETS.items():
        entry = {"rule_sets": sets, "B": {}}
        for b in BUDGETS:
            us = [c[rs][f"B{b}"]["regret"]["median"] for rs in sets]
            entry["B"][f"B{b}"] = {
                "U": dict(zip(sets, us)),
                "range": max(us) - min(us),
                "iqr_like_spread": max(us) - min(us),  # 3 points: range IS the spread
                "median": statistics.median(us),
            }
        result["seed_sensitivity"][base_rs] = entry

    # ================================================================
    # Registration extras (registration-v3 SS2/SS5.3-SS5.6): every rule
    # set's identity + B in sweeps + L/Y + the SS7.1 overhead ratio, Delta1
    # under v1's definition, paired Delta_Y and Delta2(v3). classical only.
    # ================================================================
    base_c = c["base"]
    threshold = {
        b: OVERHEAD_FLAT_MULTIPLE * base_c[f"B{b}"]["evals_per_app_at_B"]["median"] for b in BUDGETS
    }
    ordered_sets = (
        ["base"]
        + [p[2] for m in MODE_POINTS.values() for p in m]
        + ORDER_ROW_SETS[1:]
        + [p[0] for m in MODE_POINTS.values() for p in m]
        + [rs for sets in SEED_SETS.values() for rs in sets[1:]]
    )
    seen, per_set = set(), []
    for rs in ordered_sets:
        if rs in seen:
            continue
        seen.add(rs)
        s = c[rs]
        aps = s["apps_per_sweep"]["median"]
        row = {
            "rule_set": rs, "num_rules": meta[rs]["num_rules"],
            "fingerprint": meta[rs]["fingerprint"], "apps_per_sweep_median": aps,
        }
        for b in BUDGETS:
            sb = s[f"B{b}"]
            ev = sb["evals_per_app_at_B"]
            if ev is None:
                die(f"{rs}: evals_per_app undefined at B={b} — cannot evaluate SS7.1")
            row[f"B{b}"] = {
                "B_in_sweeps": b / aps,
                "U": sb["regret"]["median"],
                "L": sb["trunc_loss_pct"]["median"],
                "Y": sb["Y"],
                "evals_per_app": ev["median"],
                "evals_per_app_x_base": ev["median"] / base_c[f"B{b}"]["evals_per_app_at_B"]["median"],
                "flat": ev["median"] <= threshold[b],
                "differing_from_base": (
                    0 if rs == "base" else differing_count(curves, rs, "base", "classical", b)
                ),
            }
        per_set.append(row)

    delta1_v1 = {}
    for b in BUDGETS:
        regs = base_c[f"B{b}"]["_regrets"]
        lo, hi = bootstrap_median_ci(regs, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED)
        delta1_v1[f"B{b}"] = {
            "median_U_base": statistics.median(regs), "ci_lo": lo, "ci_hi": hi,
            "half_width": (hi - lo) / 2.0,
        }

    by_set = {r["rule_set"]: r for r in per_set}
    delta_y, delta2_v3, h1_v1_test = {}, {}, {}
    for mode, points in MODE_POINTS.items():
        delta_y[mode] = []
        for rs, R, ref_rs in points:
            row = {"rule_set": rs, "R": R, "order_matched_base": ref_rs}
            for b in BUDGETS:
                yp, ym = by_set[rs][f"B{b}"]["Y"], by_set[ref_rs][f"B{b}"]["Y"]
                row[f"Y_B{b}"], row[f"Y_matched_B{b}"], row[f"delta_Y_B{b}"] = yp, ym, yp - ym
            delta_y[mode].append(row)
        largest = next(r for r in delta_y[mode] if r["rule_set"] == LARGEST_INFLATED[mode])
        delta2_v3[mode] = {
            "at_rule_set": LARGEST_INFLATED[mode],
            **{f"B{b}": {"delta_Y": largest[f"delta_Y_B{b}"],
                         "delta2": max(DELTA2_FLOOR, largest[f"delta_Y_B{b}"]),
                         "floored": largest[f"delta_Y_B{b}"] < DELTA2_FLOOR}
               for b in BUDGETS},
        }
        h1_v1_test[mode] = {}
        for b in BUDGETS:
            du_max = result["modes"][mode][f"delta_U_at_max_R_B{b}"]
            d1 = delta1_v1[f"B{b}"]["half_width"]
            h1_v1_test[mode][f"B{b}"] = {
                "delta_U_max": du_max, "delta1_v1": d1, "ratio": du_max / d1,
                "clears": du_max >= d1,
            }

    result["registration_extras"] = {
        "overhead_flat_multiple": OVERHEAD_FLAT_MULTIPLE,
        "overhead_threshold_evals_per_app": {f"B{b}": threshold[b] for b in BUDGETS},
        "per_rule_set": per_set,
        "delta1_v1_definition": delta1_v1,
        "delta_Y": delta_y,
        "delta2_v3": delta2_v3,
        "h1_delta_U_vs_delta1_v1": h1_v1_test,
    }

    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=1, sort_keys=True)
    print(f"wrote {args.out_json}")

    with open(args.out_md, "w") as f:
        f.write(render_md(result))
    print(f"wrote {args.out_md}")


def render_md(result):
    out = []
    out.append("## SS3 - the |R| effect, order held fixed\n\n")
    out.append("`Delta_U(p) = U(p) - U(OrderMatchedBase(seed, |p|))`, seed = "
                f"`0x{result['registered_seed']:08x}` (DEFAULT_INTERLEAVE_SEED). "
                "Classical band (n=188).\n\n")
    for mode, m in result["modes"].items():
        out.append(f"**Mode ({mode})**\n\n")
        out.append("| rule set | \\|R\\| | matched-base ref | U(p)@100 | U(matched)@100 | "
                    "Delta_U@100 | differing@100 | U(p)@200 | U(matched)@200 | Delta_U@200 | "
                    "differing@200 |\n|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in m["points"]:
            out.append(
                f"| `{row['rule_set']}` | {row['R']} | `{row['order_matched_base']}` | "
                f"{fmt_pct(row['U_B100'])} | {fmt_pct(row['U_matched_B100'])} | "
                f"{fmt_pct(row['delta_U_B100'])} | {row['differing_from_matched_B100']} | "
                f"{fmt_pct(row['U_B200'])} | {fmt_pct(row['U_matched_B200'])} | "
                f"{fmt_pct(row['delta_U_B200'])} | {row['differing_from_matched_B200']} |\n"
            )
        rho100 = m["spearman_rho_delta_U_B100"]
        rho200 = m["spearman_rho_delta_U_B200"]
        d1_100 = m["delta1_v3"]["B100"]
        d1_200 = m["delta1_v3"]["B200"]
        out.append(
            f"\nSpearman rho(Delta_U, |R|): B=100 = "
            f"{'undefined' if rho100 is None else f'{rho100:.3f}'}, B=200 = "
            f"{'undefined' if rho200 is None else f'{rho200:.3f}'}. "
            f"Delta_U at max |R|: {fmt_pct(m['delta_U_at_max_R_B100'])}% (B=100), "
            f"{fmt_pct(m['delta_U_at_max_R_B200'])}% (B=200).\n\n"
            f"Delta1(v3) at `{d1_100['at_rule_set']}` (95% bootstrap CI half-width of paired "
            f"median Delta_U): B=100 = {fmt_pct(d1_100['half_width'])} pts "
            f"(median {fmt_pct(d1_100['median'])}, CI [{fmt_pct(d1_100['ci_lo'])}, "
            f"{fmt_pct(d1_100['ci_hi'])}]), B=200 = {fmt_pct(d1_200['half_width'])} pts "
            f"(median {fmt_pct(d1_200['median'])}).\n\n"
            f"**H1(v3) verdict ({mode}):** direction "
            f"{'HOLDS' if m['H1_v3_direction_holds'] else 'FAILS'} (rho >= 0.9), effect @100 "
            f"{'HOLDS' if m['H1_v3_effect_holds_B100'] else 'FAILS'} (|Delta_U(max)| >= Delta1), "
            f"effect @200 {'HOLDS' if m['H1_v3_effect_holds_B200'] else 'FAILS'}.\n\n"
        )

    out.append("## SS4 - the order effect on its own (all rule sets |R|=62)\n\n")
    for tier, rows in result["order_effect"].items():
        claim = "" if tier == "classical" else " (reported, no claim)"
        out.append(f"**{tier}{claim}**\n\n")
        out.append("| rule set | U@100 | L@100 | differing-from-base@100 | U@200 | L@200 | "
                    "differing-from-base@200 |\n|---|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            out.append(
                f"| `{row['rule_set']}` | {fmt_pct(row['U_B100'])} | {row['L_B100']:.3f} | "
                f"{row['differing_from_base_B100']} | {fmt_pct(row['U_B200'])} | "
                f"{row['L_B200']:.3f} | {row['differing_from_base_B200']} |\n"
            )
        out.append("\n")

    out.append("## Seed sensitivity of an inflated point\n\n")
    out.append("Registered seed (`0x20260901`) plus two additional interleave seeds "
                "(`1`, `2`), classical band.\n\n")
    for base_rs, entry in result["seed_sensitivity"].items():
        out.append(f"**{base_rs}**\n\n| seed variant | U@100 | U@200 |\n|---|---:|---:|\n")
        for rs in entry["rule_sets"]:
            out.append(
                f"| `{rs}` | {fmt_pct(entry['B']['B100']['U'][rs])} | "
                f"{fmt_pct(entry['B']['B200']['U'][rs])} |\n"
            )
        out.append(
            f"\nSpread (range) across the 3 seeds: B=100 = "
            f"{fmt_pct(entry['B']['B100']['range'])} pts, B=200 = "
            f"{fmt_pct(entry['B']['B200']['range'])} pts.\n\n"
        )

    x = result["registration_extras"]
    thr = x["overhead_threshold_evals_per_app"]
    out.append("## Registration extras (SS2 / SS5.3-SS5.6 of the v3 registration)\n\n")
    out.append("Every rule set this registration's numbers touch, classical band (n=188). "
                "`B in sweeps` = B / median apps_per_sweep (one-sweep probe per expression). "
                f"SS7.1 flat <=> evals/app at B <= {x['overhead_flat_multiple']:.0f}x `base`'s "
                f"({thr['B100']:.2f} @100, {thr['B200']:.2f} @200).\n\n")
    out.append("| rule set | \\|R\\| | fingerprint | aps med | B100 sweeps | B200 sweeps | U@100 | L@100 | "
                "Y@100 | U@200 | L@200 | Y@200 | ev/app@100 | x base | flat@100 | ev/app@200 | x base | "
                "flat@200 | differing@100/@200 |\n"
                "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|---|\n")
    for r in x["per_rule_set"]:
        b1, b2 = r["B100"], r["B200"]
        out.append(
            f"| `{r['rule_set']}` | {r['num_rules']} | `{r['fingerprint']}` | {r['apps_per_sweep_median']:.0f} | "
            f"{b1['B_in_sweeps']:.2f} | {b2['B_in_sweeps']:.2f} | {fmt_pct(b1['U'])} | {b1['L']:.3f} | "
            f"{fmt_pct(b1['Y'])} | {fmt_pct(b2['U'])} | {b2['L']:.3f} | {fmt_pct(b2['Y'])} | "
            f"{b1['evals_per_app']:.2f} | {b1['evals_per_app_x_base']:.2f} | {'yes' if b1['flat'] else 'no'} | "
            f"{b2['evals_per_app']:.2f} | {b2['evals_per_app_x_base']:.2f} | {'yes' if b2['flat'] else 'no'} | "
            f"{b1['differing_from_base']} / {b2['differing_from_base']} |\n"
        )
    d1 = x["delta1_v1_definition"]
    out.append("\nDelta1 under v1's definition (95% bootstrap CI half-width of median U at `base`, "
                f"{BOOTSTRAP_RESAMPLES} resamples, seed {BOOTSTRAP_SEED}): B=100 = "
                f"{fmt_pct(d1['B100']['half_width'])} pts (median {fmt_pct(d1['B100']['median_U_base'])}, "
                f"CI [{fmt_pct(d1['B100']['ci_lo'])}, {fmt_pct(d1['B100']['ci_hi'])}]); B=200 = "
                f"{fmt_pct(d1['B200']['half_width'])} pts (median {fmt_pct(d1['B200']['median_U_base'])}, "
                f"CI [{fmt_pct(d1['B200']['ci_lo'])}, {fmt_pct(d1['B200']['ci_hi'])}]).\n\n")
    out.append("**Delta_U(max) vs Delta1 (v1's definition)**\n\n| mode | B | Delta_U(max) | Delta1(v1) | "
                "ratio | clears? |\n|---|---:|---:|---:|---:|---|\n")
    for mode, t in x["h1_delta_U_vs_delta1_v1"].items():
        for b in BUDGETS:
            e = t[f"B{b}"]
            out.append(f"| ({mode}) | {b} | {fmt_pct(e['delta_U_max'])} | {fmt_pct(e['delta1_v1'])} | "
                       f"{e['ratio']:+.2f} | {'yes' if e['clears'] else 'no'} |\n")
    out.append("\n**Paired Delta_Y(p) = Y(p) - Y(OrderMatchedBase) and Delta2(v3) = "
                f"max({DELTA2_FLOOR}, Delta_Y at |R|max)**\n\n| mode | rule set | \\|R\\| | Y(p)@100 | "
                "Y(matched)@100 | Delta_Y@100 | Y(p)@200 | Y(matched)@200 | Delta_Y@200 |\n"
                "|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for mode, rows in x["delta_Y"].items():
        for r in rows:
            out.append(f"| ({mode}) | `{r['rule_set']}` | {r['R']} | {fmt_pct(r['Y_B100'])} | "
                       f"{fmt_pct(r['Y_matched_B100'])} | {fmt_pct(r['delta_Y_B100'])} | "
                       f"{fmt_pct(r['Y_B200'])} | {fmt_pct(r['Y_matched_B200'])} | "
                       f"{fmt_pct(r['delta_Y_B200'])} |\n")
    out.append("\n| mode | at | Delta2(v3)@100 | Delta2(v3)@200 |\n|---|---|---:|---:|\n")
    for mode, d in x["delta2_v3"].items():
        out.append(f"| ({mode}) | `{d['at_rule_set']}` | {d['B100']['delta2']:.4f}"
                   f"{' (floor)' if d['B100']['floored'] else ''} | {d['B200']['delta2']:.4f}"
                   f"{' (floor)' if d['B200']['floored'] else ''} |\n")
    out.append("\n")
    return "".join(out)


if __name__ == "__main__":
    main()
