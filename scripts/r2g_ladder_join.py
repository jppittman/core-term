#!/usr/bin/env python3
"""Join two `phase3_at_budget_eval` runs per set into the four-arm R2G ladder.

docs/plans/2026-09-01-guide-return-to-go.md §7: arms `unguided`, `control`
(PerRuleRateGuide), `linear` (frozen Round-1 strict-bit LinearCandidateGuide)
and `r2g` (LinearReturnGuide). The harness evaluates one claim arm per run
(its `linear` slot; `context.claim_guide` says which Guide ran), so the
four-arm table is a per-expression join of a strict-bit run and an R2G run
on the same corpus. The join is checked, not assumed: `unguided` and
`control` are deterministic and must be cost-identical between the two runs
for every expression, and every expression must appear in both — either
failure is a hard error.

Statistics follow the harness verbatim: `percentile` is the same linear
interpolation, ratio = arm cost@B / unguided cost@B with the zero-reference
convention, regret = % over the empirical best of ALL FOUR arms at ANY
checkpoint (one more arm than the Round-1b reference, stated in the doc).
Nothing here is timed.

Usage:
  python3 scripts/r2g_ladder_join.py \
      --strict-prefix docs/results/2026-09-01-r2g-ladder-strict \
      --r2g-prefix    docs/results/2026-09-01-r2g-ladder-r2g \
      --out-json docs/results/2026-09-01-guide-return-to-go.json \
      --out-csv  docs/results/2026-09-01-guide-return-to-go.csv \
      --out-md   /dev/stdout
"""
import argparse
import json
import math
import sys

SETS = ["dev", "sh", "bezier"]
TIERS = [100, 200]
MARGIN = {100: 0.06, 200: 0.07}
# §7 reference medians (PR #1091) and the pre-registered R2G targets.
REFERENCE = {
    ("sh", 100): {"control": 0.9028, "linear": 0.9039, "r2g_must": ("<", 0.9039 - 0.06)},
    ("sh", 200): {"control": 0.8940, "linear": 0.8959, "r2g_must": ("<", 0.8959 - 0.07)},
    ("dev", 100): {"control": 0.5655, "linear": 0.5366, "r2g_must": ("<=", 0.5366 + 0.06)},
    ("dev", 200): {"control": 0.6991, "linear": 0.6959, "r2g_must": ("<=", 0.6959 + 0.07)},
    ("bezier", 100): {"control": 0.9098, "linear": 0.9098, "r2g_must": None},
    ("bezier", 200): {"control": 0.9098, "linear": 0.8855, "r2g_must": None},
}
TRIG_RULE_IDX = [20, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40]
ARMS = ["control", "linear", "r2g"]


def percentile(sorted_vals, p):
    assert sorted_vals, "percentile of empty list"
    pos = p * (len(sorted_vals) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return sorted_vals[lo]
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def dist(vals):
    v = sorted(vals)
    return {
        "n": len(v),
        "q1": percentile(v, 0.25),
        "median": percentile(v, 0.5),
        "q3": percentile(v, 0.75),
        "p90": percentile(v, 0.9),
        "max": v[-1],
        "inf_count": sum(1 for x in v if math.isinf(x)),
    }


def ratio(a, b):
    if b == 0:
        return 1.0 if a == 0 else math.inf
    return a / b


def pct_over(a, r):
    if r == 0:
        return 0.0 if a == 0 else math.inf
    return (a - r) / r * 100.0


def cost_at(arm, b):
    grid = arm["grid"]
    assert b in grid, f"budget {b} not on grid {grid}"
    return arm["cost"][grid.index(b)]


def read_rows(path):
    rows = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            assert r["name"] not in rows, f"{path}: duplicate row {r['name']}"
            rows[r["name"]] = r
    return rows


def read_context(path):
    with open(path) as f:
        return json.load(f)["context"]


def join_set(strict_prefix, r2g_prefix, set_name):
    s_rows = read_rows(f"{strict_prefix}-{set_name}.jsonl")
    r_rows = read_rows(f"{r2g_prefix}-{set_name}.jsonl")
    s_ctx = read_context(f"{strict_prefix}-{set_name}.json")
    r_ctx = read_context(f"{r2g_prefix}-{set_name}.json")
    assert s_ctx["claim_guide"] == "LinearCandidateGuide", s_ctx["claim_guide"]
    assert r_ctx["claim_guide"].startswith("LinearReturnGuide["), r_ctx["claim_guide"]
    assert s_ctx["corpus"] == r_ctx["corpus"], (s_ctx["corpus"], r_ctx["corpus"])
    missing = set(s_rows) ^ set(r_rows)
    assert not missing, f"{set_name}: expressions not in both runs: {sorted(missing)[:10]}"
    joined = {}
    for name, s in s_rows.items():
        r = r_rows[name]
        assert s["tier"] == "classical" and r["tier"] == "classical", name
        for shared in ("unguided", "control"):
            assert s["arms"][shared]["cost"] == r["arms"][shared]["cost"], (
                f"{set_name}/{name}: {shared} arm differs between runs — the harness is not "
                f"deterministic across these two invocations; refusing to join"
            )
            assert s["arms"][shared]["app_actual"] == r["arms"][shared]["app_actual"], name
        joined[name] = {
            "unguided": s["arms"]["unguided"],
            "control": s["arms"]["control"],
            "linear": s["arms"]["linear"],
            "r2g": r["arms"]["linear"],
            "at_budget_r2g": r["at_budget"]["linear"],
            "full_run_r2g": r["full_run"]["linear"] if r.get("full_run") else None,
            "at_budget_linear": s["at_budget"]["linear"],
            "full_run_linear": s["full_run"]["linear"] if s.get("full_run") else None,
            "at_budget_unguided": s["at_budget"]["unguided"],
            "full_run_unguided": s["full_run"]["unguided"] if s.get("full_run") else None,
        }
    return joined, {"strict": s_ctx, "r2g": r_ctx}


def tier_table(joined, set_name, b):
    rows = list(joined.values())
    n = len(rows)
    best = {k: min(min(v[a]["cost"]) for a in ["unguided"] + ARMS) for k, v in joined.items()}
    out = {"set": set_name, "B": b, "n": n, "arms": {}}
    out["unguided_regret_at_b_pct"] = dist(
        [pct_over(cost_at(v["unguided"], b), best[k]) for k, v in joined.items()]
    )
    out["unguided_regret_at_4b_pct"] = dist(
        [pct_over(cost_at(v["unguided"], 4 * b), best[k]) for k, v in joined.items()]
    )
    for arm in ARMS:
        ratios, regrets, gaps = [], [], []
        imp = unch = worse = 0
        for k, v in joined.items():
            c = cost_at(v[arm], b)
            u = cost_at(v["unguided"], b)
            ratios.append(ratio(c, u))
            regrets.append(pct_over(c, best[k]))
            gaps.append(pct_over(c, cost_at(v["unguided"], 4 * b)))
            if c < u:
                imp += 1
            elif c == u:
                unch += 1
            else:
                worse += 1
        out["arms"][arm] = {
            "ratio_vs_unguided_at_b": dist(ratios),
            "regret_pct": dist(regrets),
            "gap_vs_unguided_at_4b_pct": dist(gaps),
            "improved": imp,
            "unchanged": unch,
            "worse": worse,
        }
    # head-to-head r2g vs strict-bit linear, and r2g vs control
    for other in ["linear", "control"]:
        lt = eq = gt = 0
        for v in rows:
            a, o = cost_at(v["r2g"], b), cost_at(v[other], b)
            if a < o:
                lt += 1
            elif a == o:
                eq += 1
            else:
                gt += 1
        out[f"head_to_head_r2g_vs_{other}"] = {"r2g_lower": lt, "equal": eq, "r2g_higher": gt}
    # §7 verdicts
    m = {arm: out["arms"][arm]["ratio_vs_unguided_at_b"]["median"] for arm in ARMS}
    ref = REFERENCE[(set_name, b)]
    out["reference_medians_pr1091"] = {"control": ref["control"], "linear": ref["linear"]}
    out["reproduces_reference"] = {
        arm: abs(m[arm] - ref[arm]) < 5e-4 for arm in ["control", "linear"]
    }
    must = ref["r2g_must"]
    if must is None:
        out["registered_claim"] = None
    else:
        op, thr = must
        holds = (m["r2g"] < thr) if op == "<" else (m["r2g"] <= thr)
        out["registered_claim"] = {
            "statement": f"m_r2g {op} {thr:.4f}",
            "m_r2g": m["r2g"],
            "holds": holds,
        }
    out["r2g_minus_linear_median"] = m["r2g"] - m["linear"]
    out["within_margin_of_linear"] = abs(m["r2g"] - m["linear"]) <= MARGIN[b]
    out["margin_M_B"] = MARGIN[b]
    return out


def rule_firing(joined, set_name):
    """Trig-rule firings (harness `rule_firing_summary` semantics) for the r2g
    arm, the strict-bit arm and unguided: fired (strict-positive) at B=100,
    B=200, full run, and expressions with any firing."""
    table = {}
    for arm_key, ab_key, full_key in [
        ("unguided", "at_budget_unguided", "full_run_unguided"),
        ("linear", "at_budget_linear", "full_run_linear"),
        ("r2g", "at_budget_r2g", "full_run_r2g"),
    ]:
        per_rule = {idx: {"fired_100": 0, "strict_100": 0, "fired_200": 0, "strict_200": 0,
                          "fired_full": 0, "strict_full": 0, "exprs": 0} for idx in TRIG_RULE_IDX}
        for v in joined.values():
            full = v[full_key]
            assert full is not None and full.get("by_rule_idx") is not None, (
                f"{set_name}: row lacks the per-rule-index histogram")
            for idx in TRIG_RULE_IDX:
                k = str(idx)
                for b in (100, 200):
                    h = v[ab_key][str(b)]["by_rule_idx"].get(k, {"fired": 0, "strict_positive": 0})
                    per_rule[idx][f"fired_{b}"] += h["fired"]
                    per_rule[idx][f"strict_{b}"] += h["strict_positive"]
                hf = full["by_rule_idx"].get(k, {"fired": 0, "strict_positive": 0})
                per_rule[idx]["fired_full"] += hf["fired"]
                per_rule[idx]["strict_full"] += hf["strict_positive"]
                if hf["fired"] > 0:
                    per_rule[idx]["exprs"] += 1
        table[arm_key] = per_rule
    return table


def fmt_d(d):
    return f"{d['q1']:.3f} / {d['median']:.3f} / {d['q3']:.3f} (p90 {d['p90']:.3f})"


def fmt_pct(d):
    return f"{d['q1']:.2f}% / {d['median']:.2f}% / {d['q3']:.2f}% (p90 {d['p90']:.1f}%)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict-prefix", required=True)
    ap.add_argument("--r2g-prefix", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    # Rule names in all_rules() order, from the harness's own report (it
    # names every registered trig index from `all_rules()`; a trainer
    # checkpoint's `rule_names` covers only rules seen in TRAIN).
    with open(f"{args.strict_prefix}-sh.json") as f:
        firing = json.load(f)["rule_firing"]
    rule_names = {}
    for entry in firing:
        for per_rule in entry["arms"].values():
            for idx, agg in per_rule.items():
                rule_names[int(idx)] = agg["rule_name"]
    for idx in TRIG_RULE_IDX:
        assert rule_names.get(idx), f"harness report names no rule for registered trig idx {idx}"

    report = {"sets": {}, "contexts": {}, "rule_firing_sh": None}
    csv_lines = ["set,B,n,arm,ratio_q1,ratio_median,ratio_q3,ratio_p90,improved,unchanged,worse,"
                 "regret_median_pct,gap_vs_4b_median_pct,reference_median_pr1091,registered_claim,claim_holds,"
                 "r2g_vs_linear_lower,r2g_vs_linear_equal,r2g_vs_linear_higher,M_B,within_margin_of_linear"]
    md = []
    for set_name in SETS:
        joined, ctx = join_set(args.strict_prefix, args.r2g_prefix, set_name)
        report["contexts"][set_name] = ctx
        report["sets"][set_name] = {}
        for b in TIERS:
            t = tier_table(joined, set_name, b)
            report["sets"][set_name][str(b)] = t
            md.append(f"### {set_name}, B = {b} (n = {t['n']})\n")
            md.append("| arm | ratio vs unguided@B (q1/med/q3, p90) | improved / unch / worse | regret vs best (4-arm) | gap vs unguided@4B | PR #1091 reference median |")
            md.append("|---|---|---|---|---|---:|")
            md.append(f"| (a) unguided @B | 1.000 (by definition) | — | {fmt_pct(t['unguided_regret_at_b_pct'])} | — | — |")
            labels = {"control": "(c) PerRuleRateGuide [control]", "linear": "(d) strict-bit LinearCandidateGuide (frozen v1)", "r2g": "(e) LinearReturnGuide (R2G, this round)"}
            for arm in ARMS:
                a = t["arms"][arm]
                ref = t["reference_medians_pr1091"].get(arm)
                ref_s = f"{ref:.4f}" + (" ✓" if t["reproduces_reference"].get(arm) else "") if ref is not None else "—"
                md.append(f"| {labels[arm]} | {fmt_d(a['ratio_vs_unguided_at_b'])} | {a['improved']} / {a['unchanged']} / {a['worse']} | {fmt_pct(a['regret_pct'])} | {fmt_pct(a['gap_vs_unguided_at_4b_pct'])} | {ref_s} |")
                claim = t["registered_claim"]
                csv_lines.append(",".join(str(x) for x in [
                    set_name, b, t["n"], arm,
                    f"{a['ratio_vs_unguided_at_b']['q1']:.4f}", f"{a['ratio_vs_unguided_at_b']['median']:.4f}",
                    f"{a['ratio_vs_unguided_at_b']['q3']:.4f}", f"{a['ratio_vs_unguided_at_b']['p90']:.4f}",
                    a["improved"], a["unchanged"], a["worse"],
                    f"{a['regret_pct']['median']:.3f}", f"{a['gap_vs_unguided_at_4b_pct']['median']:.3f}",
                    "" if ref is None else f"{ref:.4f}",
                    "" if (arm != "r2g" or claim is None) else claim["statement"],
                    "" if (arm != "r2g" or claim is None) else claim["holds"],
                    t["head_to_head_r2g_vs_linear"]["r2g_lower"] if arm == "r2g" else "",
                    t["head_to_head_r2g_vs_linear"]["equal"] if arm == "r2g" else "",
                    t["head_to_head_r2g_vs_linear"]["r2g_higher"] if arm == "r2g" else "",
                    MARGIN[b], t["within_margin_of_linear"] if arm == "r2g" else "",
                ]))
            h = t["head_to_head_r2g_vs_linear"]
            hc = t["head_to_head_r2g_vs_control"]
            claim = t["registered_claim"]
            claim_s = ("no claim registered (reported only)" if claim is None else
                       f"registered: `{claim['statement']}` — m_r2g = **{claim['m_r2g']:.4f}** → **{'HOLDS' if claim['holds'] else 'FAILS'}**")
            md.append("")
            md.append(f"Head-to-head at B={b}: r2g < strict-bit on {h['r2g_lower']}, equal on {h['equal']}, r2g > strict-bit on {h['r2g_higher']}; "
                      f"r2g vs control: {hc['r2g_lower']} / {hc['equal']} / {hc['r2g_higher']}. "
                      f"m_r2g − m_linear = {t['r2g_minus_linear_median']:+.4f} (M_{b} = {MARGIN[b]}; within margin: {t['within_margin_of_linear']}). {claim_s}\n")
        if set_name == "sh":
            rf = rule_firing(joined, set_name)
            report["rule_firing_sh"] = rf
            md.append("### sh trig-rule firings — cells `fired (strict-positive)` pooled over 95 expressions; `exprs` = expressions with any firing\n")
            md.append("| idx | rule | arm | @100 | @200 | full run | exprs |")
            md.append("|---:|---|---|---|---|---|---:|")
            for idx in TRIG_RULE_IDX:
                for arm in ["unguided", "linear", "r2g"]:
                    p = rf[arm][idx]
                    md.append(f"| {idx} | {rule_names[idx]} | {arm} | {p['fired_100']} ({p['strict_100']}) | {p['fired_200']} ({p['strict_200']}) | {p['fired_full']} ({p['strict_full']}) | {p['exprs']} |")
            md.append("")

    with open(args.out_json, "w") as f:
        json.dump(report, f, indent=1)
    with open(args.out_csv, "w") as f:
        f.write("\n".join(csv_lines) + "\n")
    with open(args.out_md, "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"wrote {args.out_json}, {args.out_csv}, {args.out_md}", file=sys.stderr)


if __name__ == "__main__":
    main()
