//! Round-3 "measure spread where the budget binds" analysis
//! (`docs/plans/2026-09-01-guide-return-to-go.md` §2b).
//!
//! Reads the `r2g_trajectories_{tier}.jsonl` files `gen_r2g_trajectories`
//! mints (consuming its round-3 `TrajectoryRow::checkpoints` field — every
//! [`pixelflow_pipeline::training::r2g::BUDGET_LADDER`] rung, not just the
//! original B=100/200) and reports, per `(tier, budget, node-count band)`:
//!
//! - the share of expressions with ZERO return spread across all 12
//!   orderings (`zero_spread_all_share`) — round 2's dataset-gate statistic,
//!   generalized across the whole budget ladder instead of just B=100;
//! - the share with zero spread among the 11 GUIDED policies ONLY
//!   (`zero_spread_guided_share`, excludes `"unguided"`) — the CREDIT
//!   signal: unguided-vs-guided spread is ordering+dedup (the production
//!   sweep visits candidates in a fixed rule-major order guided orderings
//!   never do), guided-vs-guided spread is what a LEARNED Guide could
//!   actually improve on;
//! - return-spread quartiles (all-policy);
//! - the share of expressions where `unguided`'s return differs from at
//!   least one guided policy's return at that budget
//!   (`unguided_differs_share`);
//! - a stop-reason histogram, read from each checkpoint's typed
//!   `SaturationStop` (`CheckpointRow::stop`, not inferred);
//! - `qualifies`: `true` iff `zero_spread_guided_share < 0.5` — the
//!   selection-rule test §2b's task defines ("cells where guided-among-
//!   themselves spread is non-zero for ≥50% of expressions").
//!
//! Usage:
//! ```bash
//! cargo run --release -p pixelflow-pipeline --features training --bin r2g_spread_vs_budget -- \
//!     --data-dir pixelflow-pipeline/data \
//!     --tiers train,dev,sh,bezier \
//!     --out-json docs/results/2026-09-01-r2g-spread-vs-budget.json \
//!     --out-csv docs/results/2026-09-01-r2g-spread-vs-budget.csv \
//!     --out-md docs/results/2026-09-01-r2g-spread-vs-budget.md
//! ```

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;

use clap::Parser;

use pixelflow_pipeline::training::r2g::{BUDGET_LADDER, TrajectoryRow};

#[derive(Parser)]
#[command(name = "r2g_spread_vs_budget")]
#[command(about = "Round-3 spread-vs-budget analysis over gen_r2g_trajectories' mint output")]
struct Args {
    #[arg(long, default_value = "pixelflow-pipeline/data")]
    data_dir: String,
    #[arg(long, default_value = "train,dev,sh,bezier")]
    tiers: String,
    #[arg(long)]
    out_json: Option<String>,
    #[arg(long)]
    out_csv: Option<String>,
    #[arg(long)]
    out_md: Option<String>,
}

/// Node-count bands, per the round-3 task's own table. `<=50` is out of
/// scope (this report is specifically about the regime `gen_strict_labels`'
/// inherited `--max-expr-nodes 250` cut was gating out — round 2's finding)
/// — such an expression contributes to the `"all"` pooled row but no named
/// band.
const BANDS: [(&str, usize, usize); 4] = [
    ("51-100", 51, 100),
    ("101-250", 101, 250),
    ("251-1000", 251, 1000),
    (">1000", 1001, usize::MAX),
];

fn band_of(node_count: usize) -> Option<&'static str> {
    BANDS
        .iter()
        .find(|&&(_, lo, hi)| node_count >= lo && node_count <= hi)
        .map(|&(name, _, _)| name)
}

fn read_rows(path: &std::path::Path) -> Vec<TrajectoryRow> {
    let f = File::open(path)
        .unwrap_or_else(|e| panic!("r2g_spread_vs_budget: cannot open {}: {e}", path.display()));
    BufReader::new(f)
        .lines()
        .map(|l| {
            let l =
                l.unwrap_or_else(|e| panic!("r2g_spread_vs_budget: read {}: {e}", path.display()));
            serde_json::from_str::<TrajectoryRow>(&l).unwrap_or_else(|e| {
                panic!(
                    "r2g_spread_vs_budget: parse {}: {e}\nline: {l}",
                    path.display()
                )
            })
        })
        .collect()
}

#[derive(Debug, Default, Clone)]
struct Cell {
    n_expr: usize,
    zero_spread_all: usize,
    zero_spread_guided: usize,
    spreads_all: Vec<f32>,
    unguided_eligible: usize,
    unguided_differs: usize,
    stop_hist: BTreeMap<String, usize>,
}

/// (Q1, median, Q3) via linear interpolation — same convention as
/// `training::r2g::spread_report`'s own `quantiles_f32`, restated here so
/// this standalone analysis binary has no dependency on that (private)
/// helper.
fn quantiles(mut xs: Vec<f32>) -> (f32, f32, f32) {
    if xs.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    xs.sort_by(|a, b| a.partial_cmp(b).expect("quantiles: NaN spread value"));
    let q = |p: f32| -> f32 {
        let n = xs.len();
        if n == 1 {
            return xs[0];
        }
        let pos = p * (n as f32 - 1.0);
        let lo = pos.floor() as usize;
        let hi = pos.ceil() as usize;
        if lo == hi {
            xs[lo]
        } else {
            let frac = pos - lo as f32;
            xs[lo] * (1.0 - frac) + xs[hi] * frac
        }
    };
    (q(0.25), q(0.5), q(0.75))
}

fn main() {
    let args = Args::parse();
    let data_dir = PathBuf::from(&args.data_dir);
    let tiers: Vec<String> = args
        .tiers
        .split(',')
        .map(|s| s.trim().to_string())
        .collect();

    // (tier, budget, band) -> Cell. `band == "all"` is the pooled row.
    let mut cells: BTreeMap<(String, usize, String), Cell> = BTreeMap::new();
    let mut tier_expr_counts: BTreeMap<String, usize> = BTreeMap::new();

    for tier in &tiers {
        let path = data_dir.join(format!("r2g_trajectories_{tier}.jsonl"));
        if !path.exists() {
            eprintln!(
                "r2g_spread_vs_budget: WARNING — {} does not exist, skipping tier {tier:?}",
                path.display()
            );
            continue;
        }
        let rows = read_rows(&path);
        eprintln!(
            "r2g_spread_vs_budget: {tier}: {} trajectory rows loaded",
            rows.len()
        );

        let mut by_expr: BTreeMap<&str, Vec<&TrajectoryRow>> = BTreeMap::new();
        for r in &rows {
            by_expr.entry(r.expr_name.as_str()).or_default().push(r);
        }
        tier_expr_counts.insert(tier.clone(), by_expr.len());

        for group in by_expr.values() {
            let node_count = group.first().map_or(0, |r| r.expr_node_count);
            let mut credited_bands: Vec<String> = vec!["all".to_string()];
            if let Some(b) = band_of(node_count) {
                credited_bands.push(b.to_string());
            }

            for &budget in &BUDGET_LADDER {
                let mut all_returns: Vec<f32> = Vec::new();
                let mut guided_returns: Vec<f32> = Vec::new();
                let mut unguided_return: Option<f32> = None;
                let mut stops: Vec<&str> = Vec::new();

                for row in group {
                    let Some(cp) = row.checkpoints.iter().find(|c| c.budget == budget) else {
                        continue;
                    };
                    stops.push(cp.stop.as_str());
                    if let Some(r) = cp.return_val {
                        all_returns.push(r);
                        if row.policy == "unguided" {
                            unguided_return = Some(r);
                        } else {
                            guided_returns.push(r);
                        }
                    }
                }

                // No trajectory carried a label at this budget for this
                // expression (the zero-best-exclusion convention, or the
                // budget wasn't reached by any policy) — no signal to
                // credit at this (expr, budget) cell.
                if all_returns.is_empty() {
                    continue;
                }

                let spread_all = all_returns
                    .iter()
                    .copied()
                    .fold(f32::NEG_INFINITY, f32::max)
                    - all_returns.iter().copied().fold(f32::INFINITY, f32::min);
                let zero_all = spread_all <= 0.0;

                // A single guided sample (or none) has no spread to speak
                // of — degenerate, counted as zero-spread rather than
                // excluded, matching `spread_report`'s own single-
                // trajectory convention.
                let zero_guided = if guided_returns.len() < 2 {
                    true
                } else {
                    let s = guided_returns
                        .iter()
                        .copied()
                        .fold(f32::NEG_INFINITY, f32::max)
                        - guided_returns.iter().copied().fold(f32::INFINITY, f32::min);
                    s <= 0.0
                };

                let (unguided_eligible, unguided_differs) = match unguided_return {
                    Some(u) if !guided_returns.is_empty() => {
                        let differs = guided_returns.iter().any(|&g| (g - u).abs() > 1e-6);
                        (true, differs)
                    }
                    _ => (false, false),
                };

                for band_key in &credited_bands {
                    let cell = cells
                        .entry((tier.clone(), budget, band_key.clone()))
                        .or_default();
                    cell.n_expr += 1;
                    if zero_all {
                        cell.zero_spread_all += 1;
                    }
                    if zero_guided {
                        cell.zero_spread_guided += 1;
                    }
                    cell.spreads_all.push(spread_all.max(0.0));
                    if unguided_eligible {
                        cell.unguided_eligible += 1;
                        if unguided_differs {
                            cell.unguided_differs += 1;
                        }
                    }
                    for &s in &stops {
                        *cell.stop_hist.entry(s.to_string()).or_insert(0) += 1;
                    }
                }
            }
        }
    }

    // ── CSV ─────────────────────────────────────────────────────────────
    if let Some(path) = &args.out_csv {
        let mut out = File::create(path)
            .unwrap_or_else(|e| panic!("r2g_spread_vs_budget: cannot create {path}: {e}"));
        writeln!(
            out,
            "tier,budget,band,n_expr,zero_spread_all_share,zero_spread_guided_share,\
             qualifies,spread_q1,spread_median,spread_q3,unguided_eligible,\
             unguided_differs_share,stop_hist"
        )
        .unwrap();
        for ((tier, budget, band), cell) in &cells {
            let (q1, med, q3) = quantiles(cell.spreads_all.clone());
            let zero_all_share = cell.zero_spread_all as f64 / cell.n_expr as f64;
            let zero_guided_share = cell.zero_spread_guided as f64 / cell.n_expr as f64;
            let unguided_differs_share = if cell.unguided_eligible == 0 {
                0.0
            } else {
                cell.unguided_differs as f64 / cell.unguided_eligible as f64
            };
            let stop_hist_str = cell
                .stop_hist
                .iter()
                .map(|(k, v)| format!("{k}:{v}"))
                .collect::<Vec<_>>()
                .join("|");
            writeln!(
                out,
                "{tier},{budget},{band},{},{zero_all_share:.4},{zero_guided_share:.4},\
                 {},{q1:.4},{med:.4},{q3:.4},{},{unguided_differs_share:.4},\"{stop_hist_str}\"",
                cell.n_expr,
                zero_guided_share < 0.5,
                cell.unguided_eligible,
            )
            .unwrap();
        }
        eprintln!("r2g_spread_vs_budget: wrote {path}");
    }

    // ── JSON ────────────────────────────────────────────────────────────
    if let Some(path) = &args.out_json {
        let mut cell_entries = Vec::new();
        for ((tier, budget, band), cell) in &cells {
            let (q1, med, q3) = quantiles(cell.spreads_all.clone());
            let zero_all_share = cell.zero_spread_all as f64 / cell.n_expr as f64;
            let zero_guided_share = cell.zero_spread_guided as f64 / cell.n_expr as f64;
            let unguided_differs_share = if cell.unguided_eligible == 0 {
                0.0
            } else {
                cell.unguided_differs as f64 / cell.unguided_eligible as f64
            };
            let stop_hist_json = cell
                .stop_hist
                .iter()
                .map(|(k, v)| format!("{k:?}:{v}"))
                .collect::<Vec<_>>()
                .join(",");
            cell_entries.push(format!(
                "    {{\"tier\":{tier:?},\"budget\":{budget},\"band\":{band:?},\
                 \"n_expr\":{},\"zero_spread_all_share\":{zero_all_share:.4},\
                 \"zero_spread_guided_share\":{zero_guided_share:.4},\
                 \"qualifies\":{},\"spread_q1\":{q1:.4},\"spread_median\":{med:.4},\
                 \"spread_q3\":{q3:.4},\"unguided_eligible\":{},\
                 \"unguided_differs_share\":{unguided_differs_share:.4},\
                 \"stop_hist\":{{{stop_hist_json}}}}}",
                cell.n_expr,
                zero_guided_share < 0.5,
                cell.unguided_eligible,
            ));
        }
        let tier_counts_json = tier_expr_counts
            .iter()
            .map(|(t, n)| format!("{t:?}:{n}"))
            .collect::<Vec<_>>()
            .join(",");
        let json = format!(
            "{{\n  \"budget_ladder\": {BUDGET_LADDER:?},\n  \"bands\": {:?},\n  \
             \"tier_expr_counts\": {{{tier_counts_json}}},\n  \"cells\": [\n{}\n  ]\n}}\n",
            BANDS.iter().map(|&(n, _, _)| n).collect::<Vec<_>>(),
            cell_entries.join(",\n"),
        );
        std::fs::write(path, json)
            .unwrap_or_else(|e| panic!("r2g_spread_vs_budget: cannot write {path}: {e}"));
        eprintln!("r2g_spread_vs_budget: wrote {path}");
    }

    // ── Markdown ────────────────────────────────────────────────────────
    if let Some(path) = &args.out_md {
        let mut md = String::new();
        md.push_str("# R2G spread vs. budget (round 3)\n\n");
        md.push_str(&format!("Budget ladder: `{BUDGET_LADDER:?}`.\n\n"));
        md.push_str("Per-tier expression counts (trajectory rows grouped by `expr_name`):\n\n");
        for (t, n) in &tier_expr_counts {
            md.push_str(&format!("- **{t}**: {n}\n"));
        }
        md.push('\n');

        for tier in &tiers {
            if !tier_expr_counts.contains_key(tier) {
                continue;
            }
            md.push_str(&format!("## {tier}\n\n"));
            for band in ["all", "51-100", "101-250", "251-1000", ">1000"] {
                let rows: Vec<_> = BUDGET_LADDER
                    .iter()
                    .filter_map(|&b| {
                        cells
                            .get(&(tier.clone(), b, band.to_string()))
                            .map(|c| (b, c))
                    })
                    .collect();
                if rows.is_empty() {
                    continue;
                }
                md.push_str(&format!("### band {band}\n\n"));
                md.push_str(
                    "| B | n_expr | zero-spread (all 12) | zero-spread (guided-only) | \
                     qualifies | spread Q1/median/Q3 | unguided differs | top stop reasons |\n\
                     |---:|---:|---:|---:|:---:|---|---:|---|\n",
                );
                for (b, c) in rows {
                    let (q1, med, q3) = quantiles(c.spreads_all.clone());
                    let zero_all_share = c.zero_spread_all as f64 / c.n_expr as f64;
                    let zero_guided_share = c.zero_spread_guided as f64 / c.n_expr as f64;
                    let unguided_differs_share = if c.unguided_eligible == 0 {
                        0.0
                    } else {
                        c.unguided_differs as f64 / c.unguided_eligible as f64
                    };
                    let mut stop_sorted: Vec<(&String, &usize)> = c.stop_hist.iter().collect();
                    stop_sorted.sort_by(|a, b| b.1.cmp(a.1));
                    let stop_str = stop_sorted
                        .iter()
                        .take(3)
                        .map(|(k, v)| format!("{k}:{v}"))
                        .collect::<Vec<_>>()
                        .join(", ");
                    md.push_str(&format!(
                        "| {b} | {} | {:.1}% | {:.1}% | {} | {q1:.3}/{med:.3}/{q3:.3} | {:.1}% (n={}) | {stop_str} |\n",
                        c.n_expr,
                        100.0 * zero_all_share,
                        100.0 * zero_guided_share,
                        if zero_guided_share < 0.5 { "**YES**" } else { "no" },
                        100.0 * unguided_differs_share,
                        c.unguided_eligible,
                    ));
                }
                md.push('\n');
            }
        }
        std::fs::write(path, md)
            .unwrap_or_else(|e| panic!("r2g_spread_vs_budget: cannot write {path}: {e}"));
        eprintln!("r2g_spread_vs_budget: wrote {path}");
    }

    println!("r2g_spread_vs_budget: {} cells computed", cells.len());
}
