//! Tightened-labeler re-measurement for Phase 3 (the Guide), design doc §3
//! option 3 ("two-stage: strict-label cold start, tightened-labeler
//! refinement" — `docs/plans/2026-08-31-guide-design-revision.md`).
//!
//! `guide_headroom` (2026-08-30) measured two points on the
//! over-approximation spectrum — the loose labeler (`derivation_ancestors`)
//! and the strict lower bound (node literally on the extracted path) — and
//! found they correlate only moderately (Spearman ρ ≈ 0.35, per-rule) with a
//! clean split by rule class: structural/congruence rules (commutative,
//! fma-fusion, distribute, reverse-associative, associative, identity) score
//! 63-84% under the labeler bound and ~0% under the strict one.
//!
//! This binary adds the third point: `derivation_ancestors_tight`
//! (`pixelflow-search/src/egraph/provenance.rs`) — a real, additive
//! narrowing of the loose labeler's three named over-approximation axes,
//! provably a subset of the loose bound and a superset of the strict one on
//! every episode (enforced below as a running assertion, not just a design
//! claim). It answers the design doc's open question directly: does
//! tightening move the structural/congruence rule class toward strict (they
//! really are mostly waste) or does substantial enabling-credit survive
//! (strict under-credits them, and stage-2 label refinement will matter)?
//!
//! # Same corpus, same sample, as `guide_headroom`
//!
//! To make this round's correlations directly comparable to the published
//! ρ≈0.35 loose-vs-strict figure (not confounded by a different sample),
//! this binary reuses `guide_headroom`'s corpus loading and stride-sampling
//! logic verbatim (same `corpus_train.bin`/`corpus_dev.bin`, same
//! deterministic `(i as f64) * stride` selection, same default
//! `--limit 800 --min-expressions 500`) — with the same corpus files
//! untouched since 2026-08-30, `--limit 800` reproduces the identical
//! 800-expression sample `guide_headroom` measured.
//!
//! Every number this binary reports is a deterministic count or a
//! `CostModel::latency_prior()` cost — no wall-clock timing gates
//! correctness. See `guide_headroom.rs`'s module doc for the same
//! `SATURATE_TIMEOUT`-as-safety-ceiling rationale, reused verbatim here.
//!
//! Usage:
//! ```bash
//! cargo run --release -p pixelflow-pipeline --features training --bin guide_headroom_tight -- \
//!     --corpus-dir pixelflow-pipeline/data --limit 800 \
//!     --out docs/results/2026-09-01-tightened-labeler.json
//! ```

use std::collections::BTreeMap;
use std::io::Write as _;
use std::path::PathBuf;

use clap::Parser;

use pixelflow_ir::ExprArena;
use pixelflow_pipeline::training::corpus::read_corpus;
use pixelflow_search::egraph::{CostModel, EClassId, EGraph, ENodeId, Origin, extract_dag};
use pixelflow_search::math::all_rules;

#[derive(Parser)]
#[command(name = "guide_headroom_tight")]
#[command(
    about = "Phase 3 tightened-labeler re-measurement: loose vs tight vs strict load-bearing bounds, at corpus scale"
)]
struct Args {
    /// Directory holding `corpus_train.bin` / `corpus_dev.bin`.
    #[arg(long, default_value = "pixelflow-pipeline/data")]
    corpus_dir: String,

    /// Minimum number of expressions to measure (task floor: >= 300).
    #[arg(long, default_value_t = 500)]
    min_expressions: usize,

    /// Cap on the number of expressions measured (0 = no cap), stride-sampled
    /// deterministically. Kept at 800 by default to match `guide_headroom`'s
    /// published sample exactly (see module docs).
    #[arg(long, default_value_t = 800)]
    limit: usize,

    /// Write the full structured result as JSON to this path.
    #[arg(long)]
    out: Option<String>,
}

const SATURATE_MAX_ITERS: usize = 100;
const SATURATE_MAX_CLASSES: usize = 10_000;
const SATURATE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);

/// Rule names this round classifies as "structural/congruence" per
/// `docs/plans/2026-08-31-guide-design-revision.md` §2.1 — the class the
/// design doc says scores 63-84% labeler / ~0% strict, and whose fate under
/// tightening is this measurement's central question.
const STRUCTURAL_CONGRUENCE_RULES: &[&str] = &[
    "commutative",
    "fma-fusion",
    "distribute",
    "reverse-associative",
    "associative",
    "identity",
];

struct ExprMeasurement {
    name: String,
    node_count: usize,
    total_applications: usize,
    loose_lb: usize,
    tight_lb: usize,
    strict_lb: usize,
    extracted_cost: usize,
    quiesced_before_cap: bool,
}

impl ExprMeasurement {
    fn loose_ratio(&self) -> f64 {
        ratio(self.loose_lb, self.total_applications)
    }
    fn tight_ratio(&self) -> f64 {
        ratio(self.tight_lb, self.total_applications)
    }
    fn strict_ratio(&self) -> f64 {
        ratio(self.strict_lb, self.total_applications)
    }
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

#[derive(Default, Clone, Copy)]
struct RuleAgg {
    fired: usize,
    loose_lb: usize,
    tight_lb: usize,
    strict_lb: usize,
}

/// Walk the extraction's chosen nodes, root-down, recursively through each
/// chosen node's chosen children — identical walk to `labeler.rs`'s private
/// `chosen_tagged_nodes`, reimplemented against the public API (that
/// function isn't `pub`, and this round's task is additive measurement only,
/// not touching `labeler.rs`'s existing surface beyond what the tightened
/// track already adds). Returns `(EClassId, ENodeId)` pairs — the complete
/// per-class chosen-node map both `derivation_ancestors` and
/// `derivation_ancestors_tight` expect, and specifically what
/// `derivation_ancestors_tight`'s axis-1 short-circuit needs (unlike
/// `guide_headroom.rs`'s `chosen_node_tags`, which drops the class id since
/// the loose walk doesn't need it back).
fn chosen_class_node_pairs(
    egraph: &EGraph,
    root: EClassId,
    choices: &[Option<usize>],
) -> Vec<(EClassId, ENodeId)> {
    use std::collections::BTreeSet;

    let mut visited: BTreeSet<EClassId> = BTreeSet::new();
    let mut stack = vec![root];
    let mut result = Vec::new();

    while let Some(class) = stack.pop() {
        let canonical = egraph.find(class);
        if !visited.insert(canonical) {
            continue;
        }
        let idx = canonical.index();
        let node_idx = choices.get(idx).and_then(|o| *o).unwrap_or_else(|| {
            panic!(
                "guide_headroom_tight: e-class {idx} reachable from root {} via the chosen \
                 extraction has no recorded choice — extractor invariant violated",
                root.index()
            )
        });
        let nodes = egraph.nodes(canonical);
        assert!(
            node_idx < nodes.len(),
            "guide_headroom_tight: node_idx {node_idx} out of bounds ({}) for e-class {idx}",
            nodes.len()
        );
        let tags = egraph.tags(canonical);
        result.push((canonical, tags[node_idx]));
        for child in nodes[node_idx].children() {
            stack.push(child);
        }
    }
    result
}

fn quantiles(mut xs: Vec<f64>) -> (f64, f64, f64) {
    if xs.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    xs.sort_by(|a, b| a.partial_cmp(b).expect("guide_headroom_tight: NaN ratio"));
    let q = |p: f64| -> f64 {
        let n = xs.len();
        if n == 1 {
            return xs[0];
        }
        let pos = p * (n as f64 - 1.0);
        let lo = pos.floor() as usize;
        let hi = pos.ceil() as usize;
        if lo == hi {
            xs[lo]
        } else {
            let frac = pos - lo as f64;
            xs[lo] * (1.0 - frac) + xs[hi] * frac
        }
    };
    (q(0.25), q(0.5), q(0.75))
}

/// Spearman rank correlation, average-rank tie handling. `None` if either
/// series has zero variance (undefined, not zero). O(n log n) — fine at the
/// per-rule scale (tens of rows) this is used for.
fn spearman(xs: &[f64], ys: &[f64]) -> Option<f64> {
    assert_eq!(xs.len(), ys.len());
    let rx = average_ranks(xs);
    let ry = average_ranks(ys);
    pearson(&rx, &ry)
}

fn average_ranks(xs: &[f64]) -> Vec<f64> {
    let n = xs.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        xs[a]
            .partial_cmp(&xs[b])
            .expect("guide_headroom_tight: NaN in spearman input")
    });
    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && xs[idx[j + 1]] == xs[idx[i]] {
            j += 1;
        }
        let avg_rank_1indexed = ((i + 1) + (j + 1)) as f64 / 2.0;
        for &k in idx.iter().take(j + 1).skip(i) {
            ranks[k] = avg_rank_1indexed;
        }
        i = j + 1;
    }
    ranks
}

fn pearson(xs: &[f64], ys: &[f64]) -> Option<f64> {
    let n = xs.len() as f64;
    if n == 0.0 {
        return None;
    }
    let mx = xs.iter().sum::<f64>() / n;
    let my = ys.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut vx = 0.0;
    let mut vy = 0.0;
    for i in 0..xs.len() {
        let dx = xs[i] - mx;
        let dy = ys[i] - my;
        cov += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    if vx == 0.0 || vy == 0.0 {
        return None;
    }
    Some(cov / (vx.sqrt() * vy.sqrt()))
}

/// Online (single-pass, O(1) memory) accumulator for the Pearson
/// correlation of two 0/1 series. Used for the "per-application" pooled
/// correlation across (potentially) millions of recorded applications,
/// where materializing full rank vectors would be wasteful.
///
/// This is mathematically exact for Spearman rank correlation too, not an
/// approximation: any two-valued (binary) variable's average-rank transform
/// is an affine function of the raw 0/1 value (there are only two distinct
/// ranks, one per value, so *some* affine map always fits exactly), and
/// Pearson correlation is invariant under affine transforms of either input
/// (with positive slope, which average-rank assignment always has here). So
/// Spearman-with-average-rank-ties on binary data equals Pearson on the raw
/// 0/1 values, exactly — computing the online Pearson sum below is not a
/// shortcut that trades accuracy for speed, it is the same number by a
/// closed-form identity.
#[derive(Default, Clone, Copy)]
struct BinaryCorrAccum {
    n: u64,
    sum_x: u64,
    sum_y: u64,
    sum_xy: u64,
}

impl BinaryCorrAccum {
    fn add(&mut self, x: bool, y: bool) {
        self.n += 1;
        self.sum_x += x as u64;
        self.sum_y += y as u64;
        self.sum_xy += (x && y) as u64;
    }

    fn phi(&self) -> Option<f64> {
        let n = self.n as f64;
        if n == 0.0 {
            return None;
        }
        let (sx, sy, sxy) = (self.sum_x as f64, self.sum_y as f64, self.sum_xy as f64);
        let cov = sxy - sx * sy / n;
        // For a 0/1 variable, sum(x^2) == sum(x), so var*n = sum_x - sum_x^2/n.
        let vx = sx - sx * sx / n;
        let vy = sy - sy * sy / n;
        if vx == 0.0 || vy == 0.0 {
            return None;
        }
        Some(cov / (vx.sqrt() * vy.sqrt()))
    }
}

fn main() {
    let args = Args::parse();

    let corpus_dir = PathBuf::from(&args.corpus_dir);
    let train_path = corpus_dir.join("corpus_train.bin");
    let dev_path = corpus_dir.join("corpus_dev.bin");

    let mut entries: Vec<(String, ExprArena, pixelflow_ir::ExprId)> = read_corpus(&train_path)
        .unwrap_or_else(|e| {
            panic!(
                "guide_headroom_tight: failed to read {}: {e}",
                train_path.display()
            )
        });
    let dev_entries = read_corpus(&dev_path).unwrap_or_else(|e| {
        panic!(
            "guide_headroom_tight: failed to read {}: {e}",
            dev_path.display()
        )
    });
    let train_count = entries.len();
    entries.extend(dev_entries);
    let total_available = entries.len();

    assert!(
        total_available >= args.min_expressions,
        "guide_headroom_tight: corpus has only {total_available} expressions \
         (train {train_count} + dev {}), need >= {} — regenerate a larger corpus, \
         do not silently measure on too few expressions",
        total_available - train_count,
        args.min_expressions
    );

    // Identical stride-sampling to `guide_headroom.rs`, so the same
    // `--limit` reproduces the exact same 800-expression sample.
    if args.limit > 0 && args.limit < entries.len() {
        let stride = entries.len() as f64 / args.limit as f64;
        let mut sampled = Vec::with_capacity(args.limit);
        for i in 0..args.limit {
            let idx = ((i as f64) * stride) as usize;
            sampled.push(entries[idx.min(entries.len() - 1)].clone());
        }
        entries = sampled;
    }
    let n = entries.len();
    eprintln!(
        "guide_headroom_tight: measuring {n} expressions (train {train_count}, dev {}, corpus total {total_available})",
        total_available - train_count
    );

    let costs = CostModel::latency_prior();
    let rule_names: Vec<String> = {
        let probe = EGraph::with_rules(all_rules());
        (0..all_rules().len())
            .map(|i| {
                probe
                    .rule(i)
                    .map(|r| r.name().to_string())
                    .unwrap_or_else(|| format!("<rule {i}>"))
            })
            .collect()
    };

    let mut measurements: Vec<ExprMeasurement> = Vec::with_capacity(n);
    let mut rule_agg: BTreeMap<usize, RuleAgg> = BTreeMap::new();
    let mut total_apps_all: u64 = 0;
    let mut total_loose_lb_all: u64 = 0;
    let mut total_tight_lb_all: u64 = 0;
    let mut total_strict_lb_all: u64 = 0;

    // Pooled per-application correlation accumulators.
    let mut corr_loose_strict = BinaryCorrAccum::default();
    let mut corr_tight_strict = BinaryCorrAccum::default();
    let mut corr_tight_loose = BinaryCorrAccum::default();

    for (i, (name, arena, root)) in entries.iter().enumerate() {
        let mut egraph = EGraph::with_rules(all_rules());
        let root_class = egraph.add_arena(arena, *root);
        let saturate_started = std::time::Instant::now();
        let sat_stats =
            egraph.saturate_with_limits(SATURATE_MAX_ITERS, SATURATE_MAX_CLASSES, SATURATE_TIMEOUT);
        let saturate_elapsed = saturate_started.elapsed();
        let hit_iteration_cap = sat_stats.iterations >= SATURATE_MAX_ITERS;
        let hit_class_cap = egraph.num_classes() > SATURATE_MAX_CLASSES;
        assert!(
            saturate_elapsed < SATURATE_TIMEOUT,
            "guide_headroom_tight: expression '{name}' ran {saturate_elapsed:?}, hitting the \
             {SATURATE_TIMEOUT:?} safety ceiling — fail loud rather than silently report a \
             truncated sample as converged"
        );
        let quiesced_before_cap = !hit_iteration_cap && !hit_class_cap;

        let extraction = extract_dag(&egraph, root_class, &costs);
        let chosen = chosen_class_node_pairs(&egraph, extraction.root, &extraction.choices);

        let loose_set = egraph.derivation_ancestors(&chosen);
        let tight_set = egraph.derivation_ancestors_tight(&chosen);
        let mut strict_set: std::collections::BTreeSet<pixelflow_search::egraph::ApplicationId> =
            std::collections::BTreeSet::new();
        for &(_, tag) in &chosen {
            if let Some(Origin::Rule(app_id)) = egraph.provenance().origin(tag) {
                strict_set.insert(app_id);
            }
        }

        let total_applications = egraph.provenance().application_count();
        let loose_lb = loose_set.len();
        let tight_lb = tight_set.len();
        let strict_lb = strict_set.len();

        assert!(
            strict_lb <= tight_lb && tight_lb <= loose_lb,
            "guide_headroom_tight: containment invariant violated for expression '{name}' — \
             strict ({strict_lb}) <= tight ({tight_lb}) <= loose ({loose_lb}) must always hold; \
             a violation means derivation_ancestors_tight has drifted from being a genuine \
             narrowing of derivation_ancestors and a superset of the strict walk — fail loud, \
             this is the core safety property the whole tightened-labeler track relies on"
        );

        for (app_id, record) in egraph.provenance().applications() {
            let agg = rule_agg.entry(record.rule_idx).or_default();
            agg.fired += 1;
            let in_loose = loose_set.contains(&app_id);
            let in_tight = tight_set.contains(&app_id);
            let in_strict = strict_set.contains(&app_id);
            if in_loose {
                agg.loose_lb += 1;
            }
            if in_tight {
                agg.tight_lb += 1;
            }
            if in_strict {
                agg.strict_lb += 1;
            }
            corr_loose_strict.add(in_loose, in_strict);
            corr_tight_strict.add(in_tight, in_strict);
            corr_tight_loose.add(in_tight, in_loose);
        }

        total_apps_all += total_applications as u64;
        total_loose_lb_all += loose_lb as u64;
        total_tight_lb_all += tight_lb as u64;
        total_strict_lb_all += strict_lb as u64;

        measurements.push(ExprMeasurement {
            name: name.clone(),
            node_count: arena.nodes_raw().len(),
            total_applications,
            loose_lb,
            tight_lb,
            strict_lb,
            extracted_cost: extraction.total_cost,
            quiesced_before_cap,
        });

        if (i + 1) % 50 == 0 || i + 1 == n {
            eprintln!("guide_headroom_tight: {}/{n} expressions measured", i + 1);
        }
    }

    let loose_ratios: Vec<f64> = measurements.iter().map(|m| m.loose_ratio()).collect();
    let tight_ratios: Vec<f64> = measurements.iter().map(|m| m.tight_ratio()).collect();
    let strict_ratios: Vec<f64> = measurements.iter().map(|m| m.strict_ratio()).collect();
    let (l_q1, l_med, l_q3) = quantiles(loose_ratios.clone());
    let (t_q1, t_med, t_q3) = quantiles(tight_ratios.clone());
    let (s_q1, s_med, s_q3) = quantiles(strict_ratios.clone());

    let pooled_loose = ratio(total_loose_lb_all as usize, total_apps_all as usize);
    let pooled_tight = ratio(total_tight_lb_all as usize, total_apps_all as usize);
    let pooled_strict = ratio(total_strict_lb_all as usize, total_apps_all as usize);

    println!("=== Phase 3 tightened-labeler re-measurement: {n} expressions ===");
    println!(
        "total rule applications: {total_apps_all}  \
         (loose LB: {total_loose_lb_all}, tight LB: {total_tight_lb_all}, strict LB: {total_strict_lb_all})"
    );
    println!();
    println!("--- pooled ratios (sum LB / sum applications) ---");
    println!("  loose (derivation_ancestors):        {pooled_loose:.4}");
    println!("  tight (derivation_ancestors_tight):  {pooled_tight:.4}");
    println!("  strict (node literally on path):     {pooled_strict:.4}");
    println!();
    println!("--- per-expression ratio quartiles ---");
    println!("  loose:  median {l_med:.4}  Q1 {l_q1:.4}  Q3 {l_q3:.4}  (implied savings 1/median: {:.2}x)", if l_med > 0.0 { 1.0/l_med } else { f64::INFINITY });
    println!("  tight:  median {t_med:.4}  Q1 {t_q1:.4}  Q3 {t_q3:.4}  (implied savings 1/median: {:.2}x)", if t_med > 0.0 { 1.0/t_med } else { f64::INFINITY });
    println!("  strict: median {s_med:.4}  Q1 {s_q1:.4}  Q3 {s_q3:.4}  (implied savings 1/median: {:.2}x)", if s_med > 0.0 { 1.0/s_med } else { f64::INFINITY });

    // --- Per-rule table + per-rule Spearman correlations ---
    let mut rows: Vec<(usize, RuleAgg)> = rule_agg.iter().map(|(&k, &v)| (k, v)).collect();
    rows.sort_by(|a, b| {
        let ra = ratio(a.1.loose_lb, a.1.fired);
        let rb = ratio(b.1.loose_lb, b.1.fired);
        rb.partial_cmp(&ra)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    let fired_rows: Vec<&(usize, RuleAgg)> = rows.iter().filter(|(_, a)| a.fired > 0).collect();
    let loose_pcts: Vec<f64> = fired_rows
        .iter()
        .map(|(_, a)| ratio(a.loose_lb, a.fired))
        .collect();
    let tight_pcts: Vec<f64> = fired_rows
        .iter()
        .map(|(_, a)| ratio(a.tight_lb, a.fired))
        .collect();
    let strict_pcts: Vec<f64> = fired_rows
        .iter()
        .map(|(_, a)| ratio(a.strict_lb, a.fired))
        .collect();
    let rho_loose_strict_per_rule = spearman(&loose_pcts, &strict_pcts);
    let rho_tight_strict_per_rule = spearman(&tight_pcts, &strict_pcts);
    let rho_tight_loose_per_rule = spearman(&tight_pcts, &loose_pcts);

    println!();
    println!(
        "--- per-rule rank correlation (Spearman ρ, n={} rule instances that fired) ---",
        fired_rows.len()
    );
    println!(
        "  loose vs strict:  ρ = {}",
        rho_loose_strict_per_rule
            .map(|r| format!("{r:.4}"))
            .unwrap_or_else(|| "undefined".to_string())
    );
    println!(
        "  tight vs strict:  ρ = {}",
        rho_tight_strict_per_rule
            .map(|r| format!("{r:.4}"))
            .unwrap_or_else(|| "undefined".to_string())
    );
    println!(
        "  tight vs loose:   ρ = {}",
        rho_tight_loose_per_rule
            .map(|r| format!("{r:.4}"))
            .unwrap_or_else(|| "undefined".to_string())
    );

    println!();
    println!(
        "--- per-application pooled correlation (binary load-bearing labels, n={total_apps_all}) ---"
    );
    println!(
        "  loose vs strict:  φ = {}",
        corr_loose_strict
            .phi()
            .map(|r| format!("{r:.4}"))
            .unwrap_or_else(|| "undefined".to_string())
    );
    println!(
        "  tight vs strict:  φ = {}",
        corr_tight_strict
            .phi()
            .map(|r| format!("{r:.4}"))
            .unwrap_or_else(|| "undefined".to_string())
    );
    println!(
        "  tight vs loose:   φ = {}",
        corr_tight_loose
            .phi()
            .map(|r| format!("{r:.4}"))
            .unwrap_or_else(|| "undefined".to_string())
    );
    println!(
        "  (φ = Pearson correlation on raw 0/1 labels; exactly equal to Spearman with \
         average-rank ties on binary data — see BinaryCorrAccum's doc comment)"
    );

    // --- The structural/congruence class table: the design-doc deliverable ---
    println!();
    println!("--- structural/congruence rule class (merged across operator instances) ---");
    println!(
        "{:<24} {:>9} {:>9} {:>7} {:>9} {:>7} {:>9} {:>7}",
        "rule", "fired", "loose-LB", "loose%", "tight-LB", "tight%", "strict-LB", "strict%"
    );
    let mut structural_by_name: BTreeMap<&str, RuleAgg> = BTreeMap::new();
    for (idx, agg) in &rows {
        let name = rule_names.get(*idx).map(|s| s.as_str()).unwrap_or("");
        if STRUCTURAL_CONGRUENCE_RULES.contains(&name) {
            let entry = structural_by_name.entry(name).or_default();
            entry.fired += agg.fired;
            entry.loose_lb += agg.loose_lb;
            entry.tight_lb += agg.tight_lb;
            entry.strict_lb += agg.strict_lb;
        }
    }
    for &name in STRUCTURAL_CONGRUENCE_RULES {
        let agg = structural_by_name.get(name).copied().unwrap_or_default();
        println!(
            "{:<24} {:>9} {:>9} {:>6.1}% {:>9} {:>6.1}% {:>9} {:>6.1}%",
            name,
            agg.fired,
            agg.loose_lb,
            ratio(agg.loose_lb, agg.fired) * 100.0,
            agg.tight_lb,
            ratio(agg.tight_lb, agg.fired) * 100.0,
            agg.strict_lb,
            ratio(agg.strict_lb, agg.fired) * 100.0,
        );
    }
    let structural_total = structural_by_name
        .values()
        .fold(RuleAgg::default(), |mut acc, a| {
            acc.fired += a.fired;
            acc.loose_lb += a.loose_lb;
            acc.tight_lb += a.tight_lb;
            acc.strict_lb += a.strict_lb;
            acc
        });
    println!(
        "{:<24} {:>9} {:>9} {:>6.1}% {:>9} {:>6.1}% {:>9} {:>6.1}%",
        "TOTAL (pooled)",
        structural_total.fired,
        structural_total.loose_lb,
        ratio(structural_total.loose_lb, structural_total.fired) * 100.0,
        structural_total.tight_lb,
        ratio(structural_total.tight_lb, structural_total.fired) * 100.0,
        structural_total.strict_lb,
        ratio(structural_total.strict_lb, structural_total.fired) * 100.0,
    );

    // --- Full per-rule-name table (merged across operator instances) ---
    println!();
    println!("--- full per-rule table (merged by name) ---");
    println!(
        "{:<24} {:>9} {:>9} {:>7} {:>9} {:>7} {:>9} {:>7}",
        "rule", "fired", "loose-LB", "loose%", "tight-LB", "tight%", "strict-LB", "strict%"
    );
    let mut by_name: BTreeMap<&str, RuleAgg> = BTreeMap::new();
    for (idx, agg) in &rows {
        let name = rule_names.get(*idx).map(|s| s.as_str()).unwrap_or("<unknown>");
        let entry = by_name.entry(name).or_default();
        entry.fired += agg.fired;
        entry.loose_lb += agg.loose_lb;
        entry.tight_lb += agg.tight_lb;
        entry.strict_lb += agg.strict_lb;
    }
    let mut name_rows: Vec<(&str, RuleAgg)> = by_name.into_iter().collect();
    name_rows.sort_by(|a, b| {
        ratio(b.1.loose_lb, b.1.fired)
            .partial_cmp(&ratio(a.1.loose_lb, a.1.fired))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for (name, agg) in &name_rows {
        println!(
            "{:<24} {:>9} {:>9} {:>6.1}% {:>9} {:>6.1}% {:>9} {:>6.1}%",
            name,
            agg.fired,
            agg.loose_lb,
            ratio(agg.loose_lb, agg.fired) * 100.0,
            agg.tight_lb,
            ratio(agg.tight_lb, agg.fired) * 100.0,
            agg.strict_lb,
            ratio(agg.strict_lb, agg.fired) * 100.0,
        );
    }

    if let Some(out_path) = &args.out {
        let mut json = String::new();
        json.push_str("{\n");
        json.push_str(&format!("  \"num_expressions\": {n},\n"));
        json.push_str(&format!("  \"total_applications\": {total_apps_all},\n"));
        json.push_str(&format!("  \"pooled_loose_ratio\": {pooled_loose:.6},\n"));
        json.push_str(&format!("  \"pooled_tight_ratio\": {pooled_tight:.6},\n"));
        json.push_str(&format!("  \"pooled_strict_ratio\": {pooled_strict:.6},\n"));
        json.push_str(&format!(
            "  \"loose_ratio_quartiles\": [{l_q1:.6}, {l_med:.6}, {l_q3:.6}],\n"
        ));
        json.push_str(&format!(
            "  \"tight_ratio_quartiles\": [{t_q1:.6}, {t_med:.6}, {t_q3:.6}],\n"
        ));
        json.push_str(&format!(
            "  \"strict_ratio_quartiles\": [{s_q1:.6}, {s_med:.6}, {s_q3:.6}],\n"
        ));
        json.push_str(&format!(
            "  \"per_rule_spearman\": {{\"loose_vs_strict\": {}, \"tight_vs_strict\": {}, \"tight_vs_loose\": {}, \"n\": {}}},\n",
            rho_loose_strict_per_rule
                .map(|v| format!("{v:.6}"))
                .unwrap_or_else(|| "null".to_string()),
            rho_tight_strict_per_rule
                .map(|v| format!("{v:.6}"))
                .unwrap_or_else(|| "null".to_string()),
            rho_tight_loose_per_rule
                .map(|v| format!("{v:.6}"))
                .unwrap_or_else(|| "null".to_string()),
            fired_rows.len(),
        ));
        json.push_str(&format!(
            "  \"per_application_phi\": {{\"loose_vs_strict\": {}, \"tight_vs_strict\": {}, \"tight_vs_loose\": {}, \"n\": {}}},\n",
            corr_loose_strict
                .phi()
                .map(|v| format!("{v:.6}"))
                .unwrap_or_else(|| "null".to_string()),
            corr_tight_strict
                .phi()
                .map(|v| format!("{v:.6}"))
                .unwrap_or_else(|| "null".to_string()),
            corr_tight_loose
                .phi()
                .map(|v| format!("{v:.6}"))
                .unwrap_or_else(|| "null".to_string()),
            total_apps_all,
        ));
        json.push_str("  \"structural_congruence_rules\": [\n");
        for (i, &name) in STRUCTURAL_CONGRUENCE_RULES.iter().enumerate() {
            let agg = structural_by_name.get(name).copied().unwrap_or_default();
            json.push_str(&format!(
                "    {{\"rule\": \"{name}\", \"fired\": {}, \"loose_lb\": {}, \"tight_lb\": {}, \"strict_lb\": {}, \"loose_pct\": {:.6}, \"tight_pct\": {:.6}, \"strict_pct\": {:.6}}}{}\n",
                agg.fired,
                agg.loose_lb,
                agg.tight_lb,
                agg.strict_lb,
                ratio(agg.loose_lb, agg.fired),
                ratio(agg.tight_lb, agg.fired),
                ratio(agg.strict_lb, agg.fired),
                if i + 1 < STRUCTURAL_CONGRUENCE_RULES.len() { "," } else { "" }
            ));
        }
        json.push_str("  ],\n");
        json.push_str("  \"per_rule\": [\n");
        for (i, (name, agg)) in name_rows.iter().enumerate() {
            json.push_str(&format!(
                "    {{\"rule\": \"{name}\", \"fired\": {}, \"loose_lb\": {}, \"tight_lb\": {}, \"strict_lb\": {}}}{}\n",
                agg.fired,
                agg.loose_lb,
                agg.tight_lb,
                agg.strict_lb,
                if i + 1 < name_rows.len() { "," } else { "" }
            ));
        }
        json.push_str("  ],\n");
        json.push_str("  \"per_expression\": [\n");
        for (i, m) in measurements.iter().enumerate() {
            json.push_str(&format!(
                "    {{\"name\": {:?}, \"node_count\": {}, \"total_applications\": {}, \"loose_lb\": {}, \"tight_lb\": {}, \"strict_lb\": {}, \"loose_ratio\": {:.6}, \"tight_ratio\": {:.6}, \"strict_ratio\": {:.6}, \"extracted_cost\": {}, \"quiesced_before_cap\": {}}}{}\n",
                m.name,
                m.node_count,
                m.total_applications,
                m.loose_lb,
                m.tight_lb,
                m.strict_lb,
                m.loose_ratio(),
                m.tight_ratio(),
                m.strict_ratio(),
                m.extracted_cost,
                m.quiesced_before_cap,
                if i + 1 < measurements.len() { "," } else { "" }
            ));
        }
        json.push_str("  ]\n");
        json.push_str("}\n");

        let mut f = std::fs::File::create(out_path)
            .unwrap_or_else(|e| panic!("guide_headroom_tight: cannot create {out_path}: {e}"));
        f.write_all(json.as_bytes())
            .unwrap_or_else(|e| panic!("guide_headroom_tight: cannot write {out_path}: {e}"));
        eprintln!("guide_headroom_tight: wrote {out_path}");
    }
}
