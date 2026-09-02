//! Counterfactual replay: does removing one application actually help or
//! hurt the extraction at budget B, and do the hand-drawn hindsight bounds
//! (loose / tight / strict / strict-by-output-class) agree with that ground
//! truth? (docs/plans/2026-09-01-guide-return-to-go.md §4, the validation
//! half of the R2G redirect — this binary does not touch §1-§3's training
//! target or model; it only scores the bounds against a measured
//! counterfactual.)
//!
//! # The counterfactual
//!
//! For an expression's UNGUIDED trajectory to budget `B`, and a sampled
//! recorded application `a` at ordinal `t_a`, `τ \ a` is the deterministic
//! masked replay of the identical trajectory with `a` skipped — not applied,
//! not recorded, so the budget slot it would have consumed is spent on
//! whatever candidate the same rule-major scan finds next
//! ([`EGraph::saturate_until_applications_observed`],
//! [`pixelflow_search::egraph::ApplicationMask`]; see that method's doc and
//! `saturate_until_applications_observed_without_mask_matches_original_bit_for_bit`
//! in `pixelflow-search/src/egraph/graph.rs` for the bit-identity pin this
//! relies on).
//!
//! `Δ_a = ln(cost(τ\a, B)) − ln(cost(τ, B))`. This is exactly the plan's
//! `R(τ\a,B) − R(τ,B)` log-regret difference
//! (`R(τ,B) = ln(cost(τ,B) / c*_e)`): the per-expression reference cost
//! `c*_e` cancels in the subtraction, so no cross-trajectory "empirical
//! best" needs to be minted here — the K=12-trajectory diversity dataset
//! (plan §2) is a separate, larger piece of work this binary does not build.
//! `Δ_a > 0` — masked cost exceeds original — means `a` was genuinely
//! helpful; `Δ_a < 0` means `a` was actively wasteful (its slot was worth
//! more to whatever fired instead); `Δ_a = 0` means it didn't matter.
//!
//! # The four bounds
//!
//! `loose`/`tight`/`strict` are [`EpisodeLabels::compute`] /
//! `compute_tight` / `compute_strict` against the ORIGINAL (unmasked)
//! trajectory's extraction at B — the labeler's existing hindsight-ancestry
//! walks, unmodified. `strict_by_output_class` is NEW and computed inline,
//! per the task spec: positive iff the canonical class of `a`'s output — any
//! e-node whose origin is `Origin::Rule(a)`, or either side of a
//! `UnionEvent` `a` caused — is (after `find`) a canonical class the chosen
//! extraction actually visits. This is what credits `pythagorean`-shaped
//! rules whose output unions into an already-existing node (the literal `1`
//! constant): `compute_strict` misses that firing entirely because the
//! chosen node it unioned into was never *created* by `a`, but the class it
//! touched is exactly the one the extraction walks.
//!
//! # Sampling and the wall-clock ceiling
//!
//! Applications are sampled uniformly (seeded, without replacement) from
//! each expression's STATE-CHANGING recorded applications (those that
//! caused at least one `UnionEvent` — the no-op re-fires that make up most
//! of a sweep have `Δ = 0` by construction and would only inflate a tie
//! count, so they are not sampled; their share is reported). This binary is
//! CPU-heavy (one full masked replay per sampled application), so — per this
//! task's own instruction, an explicit exception to this codebase's usual
//! "safety ceilings panic" convention for the SAMPLING loop specifically —
//! `--wall-clock-ceiling-secs` bounds it: hitting the ceiling stops sampling
//! loudly (an `eprintln!` and a `wall_clock_ceiling_hit: true` field in the
//! report) with whatever count was achieved, never a silent partial. Each
//! individual saturation call still has its own generous per-run timeout
//! that PANICS if it binds (`SATURATE_TIMEOUT`) — that is a correctness
//! ceiling on one measurement, unrelated to the outer sampling budget.
//!
//! Usage:
//! ```bash
//! cargo run --release -p pixelflow-pipeline --features training --bin counterfactual_credit -- \
//!     --corpus-dir pixelflow-pipeline/data \
//!     --n-expr-sh 30 --n-expr-dev 30 --n-apps 20 --budget 100 \
//!     --out-jsonl docs/results/2026-09-01-counterfactual-credit.jsonl \
//!     --out-json docs/results/2026-09-01-counterfactual-credit.json \
//!     --out-md docs/results/2026-09-01-counterfactual-credit.md
//! ```

use std::collections::{BTreeSet, HashMap};
use std::io::Write as _;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use clap::Parser;

use pixelflow_ir::ExprArena;
use pixelflow_pipeline::training::corpus::read_corpus;
use pixelflow_search::egraph::provenance::{ApplicationId, ENodeId, Origin};
use pixelflow_search::egraph::{
    CostModel, EClassId, EGraph, EpisodeLabels, Label, config_for_node_count, extract_dag,
};
use pixelflow_search::math::all_rules;

/// Per-run safety ceiling for ONE `saturate_until_applications[_observed]`
/// call. This is a correctness guard (offline measurement must fail loud
/// rather than silently truncate one replay), not the outer sampling
/// ceiling below — see the module doc.
const SATURATE_TIMEOUT: Duration = Duration::from_secs(30);
/// Sweep-count ceiling for one run to a B ≤ a few hundred applications —
/// generous headroom over the handful of sweeps that actually takes.
const SATURATE_MAX_ITERS: usize = 500;

#[derive(Parser)]
#[command(name = "counterfactual_credit")]
#[command(about = "Score the hindsight bounds against a measured leave-one-out counterfactual")]
struct Args {
    /// Directory holding `corpus_dev.bin` / `corpus_dev_ood.bin`.
    #[arg(long, default_value = "pixelflow-pipeline/data")]
    corpus_dir: String,

    /// Minimum `sh` (OOD) expressions to sample (`corpus_dev_ood.bin`,
    /// `dev_sh_*` entries).
    #[arg(long, default_value_t = 30)]
    n_expr_sh: usize,

    /// Minimum DEV classical expressions to sample (`corpus_dev.bin`).
    #[arg(long, default_value_t = 30)]
    n_expr_dev: usize,

    /// Target sampled (state-changing) applications per expression.
    #[arg(long, default_value_t = 20)]
    n_apps: usize,

    /// Budget B (recorded applications) both the original and every masked
    /// replay run to.
    #[arg(long, default_value_t = 100)]
    budget: usize,

    /// Seed for expression selection and per-expression application
    /// sampling — deterministic and reproducible from this one value.
    #[arg(long, default_value_t = 0x5EED_C0DE_C0FF_EE01)]
    seed: u64,

    /// Wall-clock ceiling on the WHOLE sampling loop (across every
    /// expression and application) — see the module doc. Not a per-run
    /// correctness ceiling.
    #[arg(long, default_value_t = 1_200)]
    wall_clock_ceiling_secs: u64,

    #[arg(
        long,
        default_value = "docs/results/2026-09-01-counterfactual-credit.jsonl"
    )]
    out_jsonl: String,

    #[arg(
        long,
        default_value = "docs/results/2026-09-01-counterfactual-credit.json"
    )]
    out_json: String,

    #[arg(
        long,
        default_value = "docs/results/2026-09-01-counterfactual-credit.md"
    )]
    out_md: String,
}

// ---------------------------------------------------------------------------
// Seeded RNG — same xorshift64* idiom this crate's other binaries use
// locally (e.g. `bench_extraction_3way.rs`'s `SeededRng`); determinism only,
// not cryptographic quality, and not worth a shared dependency.
// ---------------------------------------------------------------------------

struct SeededRng(u64);

impl SeededRng {
    fn new(seed: u64) -> Self {
        // SplitMix64-style finalizer so a low-entropy seed (including 0)
        // doesn't produce a degenerate short cycle.
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        Self(z ^ (z >> 31))
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Fisher-Yates shuffle.
    fn shuffle<T>(&mut self, s: &mut [T]) {
        for i in (1..s.len()).rev() {
            let j = (self.next_u64() % (i as u64 + 1)) as usize;
            s.swap(i, j);
        }
    }
}

/// Seeded selection of at least `min_n` items out of `entries` (all of them
/// if fewer are available): shuffle then take a prefix, so the same seed
/// reproduces the same sample and every entry has equal a-priori chance of
/// selection.
fn seeded_select<T: Clone>(mut entries: Vec<T>, min_n: usize, rng: &mut SeededRng) -> Vec<T> {
    rng.shuffle(&mut entries);
    if min_n >= entries.len() {
        entries
    } else {
        entries.truncate(min_n);
        entries
    }
}

fn uptime() -> String {
    std::process::Command::new("uptime")
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|e| format!("uptime unavailable: {e}"))
}

fn git_rev() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|e| format!("git unavailable: {e}"))
}

// ---------------------------------------------------------------------------
// Per-expression replay context: everything computed ONCE against the
// original (unmasked) trajectory, reused across every sampled application.
// ---------------------------------------------------------------------------

struct ExprContext {
    name: String,
    arena: ExprArena,
    root: pixelflow_ir::ExprId,
    max_classes: usize,
    egraph: EGraph,
    cost_original: usize,
    loose: EpisodeLabels,
    tight: EpisodeLabels,
    strict: EpisodeLabels,
    /// `Origin::Rule(a)`-tagged nodes this application created, keyed by
    /// `a` — built once per expression from `provenance().origins()`
    /// instead of re-scanning it per sampled application.
    created_by: HashMap<ApplicationId, Vec<ENodeId>>,
    /// Class pairs unioned as a direct result of application `a`.
    unions_by: HashMap<ApplicationId, Vec<(EClassId, EClassId)>>,
    /// `ENodeId -> its current canonical EClassId`.
    node_to_class: HashMap<ENodeId, EClassId>,
    /// Canonical classes the chosen extraction actually visits.
    reachable: BTreeSet<EClassId>,
    /// Recorded, state-changing application ids with ordinal `< budget`, in
    /// ordinal order — the pool `--n-apps` samples from.
    candidates: Vec<(ApplicationId, usize)>, // (id, rule_idx)
}

/// Walk the chosen extraction from `root`, collecting every canonical class
/// it visits. Mirrors `labeler::chosen_tagged_nodes`'s class-level walk
/// (classes only — this binary needs no node-level tag tracking), rebuilt
/// here against the public API per this task's "computed inline"
/// instruction rather than reaching into a private helper.
fn reachable_canonical_classes(
    egraph: &EGraph,
    root: EClassId,
    choices: &[Option<usize>],
) -> BTreeSet<EClassId> {
    let mut visited = BTreeSet::new();
    let mut stack = vec![root];
    while let Some(class) = stack.pop() {
        let canonical = egraph.find(class);
        if !visited.insert(canonical) {
            continue;
        }
        let idx = canonical.index();
        let node_idx = choices.get(idx).and_then(|o| *o).unwrap_or_else(|| {
            panic!(
                "counterfactual_credit: e-class {idx} is reachable from the chosen extraction \
                 but has no recorded choice — extractor/labeler invariant violated"
            )
        });
        let nodes = egraph.nodes(canonical);
        assert!(
            node_idx < nodes.len(),
            "counterfactual_credit: node_idx {node_idx} out of bounds ({}) for class {idx}",
            nodes.len()
        );
        for child in nodes[node_idx].children() {
            stack.push(child);
        }
    }
    visited
}

/// Build the replay context for one expression: run the UNGUIDED trajectory
/// to `budget`, extract, compute the three existing bounds, and index
/// everything the per-application scoring loop needs.
///
/// # Panics
///
/// Panics if the saturation call hits its own wall-clock safety ceiling
/// (`SATURATE_TIMEOUT`) — a correctness guard, not the outer sampling
/// ceiling (see module doc).
fn build_expr_context(
    name: String,
    arena: ExprArena,
    root: pixelflow_ir::ExprId,
    budget: usize,
) -> ExprContext {
    let max_classes = config_for_node_count(arena.nodes_raw().len()).max_classes;
    let mut egraph = EGraph::with_rules(all_rules());
    let root_class = egraph.add_arena(&arena, root);
    let stats = egraph.saturate_until_applications(
        budget,
        SATURATE_MAX_ITERS,
        max_classes,
        SATURATE_TIMEOUT,
    );
    assert!(
        stats.stop != pixelflow_search::egraph::SaturationStop::Timeout,
        "counterfactual_credit: '{name}' hit the {SATURATE_TIMEOUT:?} per-run safety ceiling \
         while replaying to B={budget} — fail loud rather than score a truncated trajectory"
    );

    let costs = CostModel::latency_prior();
    let extraction = extract_dag(&egraph, root_class, &costs);
    let cost_original = extraction.total_cost;

    let loose = EpisodeLabels::compute(&egraph, extraction.root, &extraction.choices);
    let tight = EpisodeLabels::compute_tight(&egraph, extraction.root, &extraction.choices);
    let strict = EpisodeLabels::compute_strict(&egraph, extraction.root, &extraction.choices);

    let mut created_by: HashMap<ApplicationId, Vec<ENodeId>> = HashMap::new();
    for (node, origin) in egraph.provenance().origins() {
        if let Origin::Rule(app_id) = origin {
            created_by.entry(app_id).or_default().push(node);
        }
    }
    let mut unions_by: HashMap<ApplicationId, Vec<(EClassId, EClassId)>> = HashMap::new();
    for event in egraph.provenance().union_events() {
        if let Some(app_id) = event.application_id {
            unions_by
                .entry(app_id)
                .or_default()
                .push((event.class_a, event.class_b));
        }
    }

    let mut node_to_class: HashMap<ENodeId, EClassId> = HashMap::new();
    for class in egraph.class_ids() {
        for &tag in egraph.tags(class) {
            node_to_class.insert(tag, class);
        }
    }

    let reachable = reachable_canonical_classes(&egraph, extraction.root, &extraction.choices);

    // State-changing recorded applications with ordinal < budget: the
    // no-op re-fires (idempotent matches) that dominate a sweep have
    // Δ = 0 by construction and are excluded from the sample per the
    // module doc.
    let mut candidates: Vec<(ApplicationId, usize)> = Vec::new();
    let bound = budget.min(stats.applications) as u64;
    for (app_id, record) in egraph.provenance().applications() {
        if app_id.as_u64() >= bound {
            continue;
        }
        if unions_by.contains_key(&app_id) {
            candidates.push((app_id, record.rule_idx));
        }
    }

    ExprContext {
        name,
        arena,
        root,
        max_classes,
        egraph,
        cost_original,
        loose,
        tight,
        strict,
        created_by,
        unions_by,
        node_to_class,
        reachable,
        candidates,
    }
}

/// The strict-by-output-class bound (task spec, verbatim): positive iff the
/// canonical class of `a`'s output — any node `a` created, or either side
/// of a union `a` caused — is a class the chosen extraction actually
/// visits. Credits e.g. `pythagorean`'s union into an already-existing `1`,
/// which `EpisodeLabels::compute_strict` cannot see (that node was never
/// *created* by `a`).
fn strict_by_output_class(ctx: &ExprContext, app_id: ApplicationId) -> bool {
    if let Some(nodes) = ctx.created_by.get(&app_id) {
        for &node in nodes {
            if let Some(&class) = ctx.node_to_class.get(&node)
                && ctx.reachable.contains(&ctx.egraph.find(class))
            {
                return true;
            }
        }
    }
    if let Some(pairs) = ctx.unions_by.get(&app_id) {
        for &(a, b) in pairs {
            if ctx.reachable.contains(&ctx.egraph.find(a))
                || ctx.reachable.contains(&ctx.egraph.find(b))
            {
                return true;
            }
        }
    }
    false
}

/// Re-run the identical trajectory with `app_id` masked and return its
/// extraction cost at `budget` — `cost(τ\a, B)`.
///
/// # Panics
///
/// Same per-run safety-ceiling contract as [`build_expr_context`].
fn masked_replay_cost(ctx: &ExprContext, app_id: ApplicationId, budget: usize) -> usize {
    let mask = pixelflow_search::egraph::ApplicationMask {
        skip_ordinal: app_id.as_u64(),
    };
    let mut egraph = EGraph::with_rules(all_rules());
    let root_class = egraph.add_arena(&ctx.arena, ctx.root);
    let stats = egraph.saturate_until_applications_observed(
        budget,
        SATURATE_MAX_ITERS,
        ctx.max_classes,
        SATURATE_TIMEOUT,
        Some(mask),
    );
    assert!(
        stats.stop != pixelflow_search::egraph::SaturationStop::Timeout,
        "counterfactual_credit: masked replay of '{}' (skip ordinal {}) hit the \
         {SATURATE_TIMEOUT:?} per-run safety ceiling — fail loud rather than score a \
         truncated replay",
        ctx.name,
        app_id.as_u64()
    );
    let costs = CostModel::latency_prior();
    extract_dag(&egraph, root_class, &costs).total_cost
}

// ---------------------------------------------------------------------------
// One sampled application's row.
// ---------------------------------------------------------------------------

struct Row {
    expr_name: String,
    set: &'static str,
    application_ordinal: u64,
    rule_idx: usize,
    rule_name: String,
    cost_original: usize,
    cost_masked: usize,
    delta: f64,
    bit_loose: bool,
    bit_tight: bool,
    bit_strict: bool,
    bit_strict_class: bool,
}

// ---------------------------------------------------------------------------
// Correlation statistics.
// ---------------------------------------------------------------------------

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}

/// Pearson correlation coefficient. For a binary `xs` (0.0/1.0, our four
/// bounds), this IS the point-biserial correlation with `ys` by
/// construction — no separate formula needed. `NaN` when either series has
/// zero variance (reported as such, never silently coerced to 0).
fn pearson(xs: &[f64], ys: &[f64]) -> f64 {
    assert_eq!(xs.len(), ys.len());
    let n = xs.len();
    if n < 2 {
        return f64::NAN;
    }
    let mx = mean(xs);
    let my = mean(ys);
    let mut cov = 0.0;
    let mut vx = 0.0;
    let mut vy = 0.0;
    for i in 0..n {
        let dx = xs[i] - mx;
        let dy = ys[i] - my;
        cov += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    if vx == 0.0 || vy == 0.0 {
        return f64::NAN;
    }
    cov / (vx.sqrt() * vy.sqrt())
}

/// Average-rank transform (ties share the mean of their would-be ranks) —
/// the standard Spearman tie convention.
fn average_ranks(xs: &[f64]) -> Vec<f64> {
    let n = xs.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| xs[a].partial_cmp(&xs[b]).expect("counterfactual_credit: NaN in rank"));
    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && xs[order[j + 1]] == xs[order[i]] {
            j += 1;
        }
        // Ranks are 1-based; average the tied block's would-be ranks.
        let avg_rank = ((i + 1) + (j + 1)) as f64 / 2.0;
        for &idx in &order[i..=j] {
            ranks[idx] = avg_rank;
        }
        i = j + 1;
    }
    ranks
}

fn spearman(xs: &[f64], ys: &[f64]) -> f64 {
    pearson(&average_ranks(xs), &average_ranks(ys))
}

#[derive(Default)]
struct BoundStats {
    pearson_pooled: f64,
    spearman_pooled: f64,
    pearson_sh: f64,
    spearman_sh: f64,
    pearson_dev: f64,
    spearman_dev: f64,
    n_pooled: usize,
    n_sh: usize,
    n_dev: usize,
}

fn score_bound(rows: &[&Row], bit_of: impl Fn(&Row) -> bool) -> BoundStats {
    let bits: Vec<f64> = rows.iter().map(|r| if bit_of(r) { 1.0 } else { 0.0 }).collect();
    let deltas: Vec<f64> = rows.iter().map(|r| r.delta).collect();
    let sh_idx: Vec<usize> = rows
        .iter()
        .enumerate()
        .filter(|(_, r)| r.set == "sh")
        .map(|(i, _)| i)
        .collect();
    let dev_idx: Vec<usize> = rows
        .iter()
        .enumerate()
        .filter(|(_, r)| r.set == "dev")
        .map(|(i, _)| i)
        .collect();
    let subset = |idx: &[usize], v: &[f64]| -> Vec<f64> { idx.iter().map(|&i| v[i]).collect() };

    let bits_sh = subset(&sh_idx, &bits);
    let deltas_sh = subset(&sh_idx, &deltas);
    let bits_dev = subset(&dev_idx, &bits);
    let deltas_dev = subset(&dev_idx, &deltas);

    BoundStats {
        pearson_pooled: pearson(&bits, &deltas),
        spearman_pooled: spearman(&bits, &deltas),
        pearson_sh: pearson(&bits_sh, &deltas_sh),
        spearman_sh: spearman(&bits_sh, &deltas_sh),
        pearson_dev: pearson(&bits_dev, &deltas_dev),
        spearman_dev: spearman(&bits_dev, &deltas_dev),
        n_pooled: rows.len(),
        n_sh: sh_idx.len(),
        n_dev: dev_idx.len(),
    }
}

fn json_num(x: f64) -> String {
    if x.is_nan() {
        "null".to_string()
    } else {
        format!("{x:.6}")
    }
}

fn write_bound_json(out: &mut String, name: &str, s: &BoundStats, trailing_comma: bool) {
    out.push_str(&format!(
        "    {name:?}: {{\"pearson_pooled\": {}, \"spearman_pooled\": {}, \
         \"pearson_sh\": {}, \"spearman_sh\": {}, \"pearson_dev\": {}, \"spearman_dev\": {}, \
         \"n_pooled\": {}, \"n_sh\": {}, \"n_dev\": {}}}{}\n",
        json_num(s.pearson_pooled),
        json_num(s.spearman_pooled),
        json_num(s.pearson_sh),
        json_num(s.spearman_sh),
        json_num(s.pearson_dev),
        json_num(s.spearman_dev),
        s.n_pooled,
        s.n_sh,
        s.n_dev,
        if trailing_comma { "," } else { "" },
    ));
}

fn main() {
    let args = Args::parse();
    let start = Instant::now();
    let ceiling = Duration::from_secs(args.wall_clock_ceiling_secs);
    let load_start = uptime();

    let corpus_dir = PathBuf::from(&args.corpus_dir);
    let dev_path = corpus_dir.join("corpus_dev.bin");
    let ood_path = corpus_dir.join("corpus_dev_ood.bin");

    let dev_entries = read_corpus(&dev_path)
        .unwrap_or_else(|e| panic!("counterfactual_credit: failed to read {}: {e}", dev_path.display()));
    let ood_entries = read_corpus(&ood_path)
        .unwrap_or_else(|e| panic!("counterfactual_credit: failed to read {}: {e}", ood_path.display()));
    let sh_entries: Vec<(String, ExprArena, pixelflow_ir::ExprId)> = ood_entries
        .into_iter()
        .filter(|(name, _, _)| name.starts_with("dev_sh_"))
        .collect();

    let mut select_rng = SeededRng::new(args.seed);
    let dev_sample = seeded_select(dev_entries, args.n_expr_dev, &mut select_rng);
    let sh_sample = seeded_select(sh_entries, args.n_expr_sh, &mut select_rng);

    assert!(
        dev_sample.len() >= args.n_expr_dev.min(334),
        "counterfactual_credit: corpus_dev.bin has fewer classical entries than requested"
    );
    assert!(
        !sh_sample.is_empty(),
        "counterfactual_credit: no dev_sh_* entries found in corpus_dev_ood.bin"
    );

    eprintln!(
        "counterfactual_credit: sampling {} sh + {} DEV classical expressions, \
         target {} applications each, budget B={}, wall-clock ceiling {:?}",
        sh_sample.len(),
        dev_sample.len(),
        args.n_apps,
        args.budget,
        ceiling
    );

    let out_jsonl_path = PathBuf::from(&args.out_jsonl);
    if let Some(parent) = out_jsonl_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let mut out = std::io::BufWriter::new(
        std::fs::File::create(&out_jsonl_path)
            .unwrap_or_else(|e| panic!("counterfactual_credit: cannot create {}: {e}", out_jsonl_path.display())),
    );

    let mut apply_rng = SeededRng::new(args.seed ^ 0xA5A5_A5A5_A5A5_A5A5);
    let mut rows: Vec<Row> = Vec::new();
    let mut expressions_processed = 0usize;
    let mut insufficient_candidates: Vec<String> = Vec::new();
    let mut ceiling_hit = false;

    'expressions: for (set, sample) in [("sh", &sh_sample), ("dev", &dev_sample)] {
        for (name, arena, root) in sample.iter() {
            if start.elapsed() >= ceiling {
                ceiling_hit = true;
                eprintln!(
                    "counterfactual_credit: WALL-CLOCK CEILING ({ceiling:?}) HIT after {} \
                     expressions / {} applications — stopping the sampling loop now, loudly, \
                     with the count achieved (not a silent partial)",
                    expressions_processed,
                    rows.len()
                );
                break 'expressions;
            }

            let ctx = build_expr_context(name.clone(), arena.clone(), *root, args.budget);
            expressions_processed += 1;

            if ctx.candidates.len() < args.n_apps {
                insufficient_candidates.push(format!(
                    "{name} ({} state-changing applications < {} target)",
                    ctx.candidates.len(),
                    args.n_apps
                ));
            }

            let mut pool = ctx.candidates.clone();
            let sampled: Vec<(ApplicationId, usize)> = if pool.len() <= args.n_apps {
                pool
            } else {
                apply_rng.shuffle(&mut pool);
                pool.truncate(args.n_apps);
                pool
            };

            let rule_names: Vec<String> = (0..all_rules().len())
                .map(|i| {
                    ctx.egraph
                        .rule(i)
                        .map(|r| r.name().to_string())
                        .unwrap_or_else(|| format!("<rule {i}>"))
                })
                .collect();

            for (app_id, rule_idx) in sampled {
                if start.elapsed() >= ceiling {
                    ceiling_hit = true;
                    eprintln!(
                        "counterfactual_credit: WALL-CLOCK CEILING ({ceiling:?}) HIT mid-expression \
                         '{name}' after {} applications total — stopping now, loudly, with the \
                         count achieved (not a silent partial)",
                        rows.len()
                    );
                    break 'expressions;
                }

                let cost_masked = masked_replay_cost(&ctx, app_id, args.budget);
                let delta = if ctx.cost_original == 0 || cost_masked == 0 {
                    eprintln!(
                        "counterfactual_credit: WARNING zero extraction cost for '{name}' \
                         (original={}, masked={}) at ordinal {} — recording delta=0.0 rather \
                         than an undefined ln(0)",
                        ctx.cost_original,
                        cost_masked,
                        app_id.as_u64()
                    );
                    0.0
                } else {
                    (cost_masked as f64).ln() - (ctx.cost_original as f64).ln()
                };

                let bit_loose = ctx.loose.labels.get(&app_id) == Some(&Label::LoadBearing);
                let bit_tight = ctx.tight.labels.get(&app_id) == Some(&Label::LoadBearing);
                let bit_strict = ctx.strict.labels.get(&app_id) == Some(&Label::LoadBearing);
                let bit_strict_class = strict_by_output_class(&ctx, app_id);

                let row = Row {
                    expr_name: name.clone(),
                    set,
                    application_ordinal: app_id.as_u64(),
                    rule_idx,
                    rule_name: rule_names
                        .get(rule_idx)
                        .cloned()
                        .unwrap_or_else(|| format!("<rule {rule_idx}>")),
                    cost_original: ctx.cost_original,
                    cost_masked,
                    delta,
                    bit_loose,
                    bit_tight,
                    bit_strict,
                    bit_strict_class,
                };

                writeln!(
                    out,
                    "{{\"expr_name\":{:?},\"set\":{:?},\"application_ordinal\":{},\
                     \"rule_idx\":{},\"rule_name\":{:?},\"cost_original\":{},\
                     \"cost_masked\":{},\"delta\":{:.6},\"bit_loose\":{},\"bit_tight\":{},\
                     \"bit_strict\":{},\"bit_strict_by_output_class\":{}}}",
                    row.expr_name,
                    row.set,
                    row.application_ordinal,
                    row.rule_idx,
                    row.rule_name,
                    row.cost_original,
                    row.cost_masked,
                    row.delta,
                    row.bit_loose,
                    row.bit_tight,
                    row.bit_strict,
                    row.bit_strict_class,
                )
                .unwrap_or_else(|e| panic!("counterfactual_credit: write to {}: {e}", out_jsonl_path.display()));

                rows.push(row);
            }

            if expressions_processed.is_multiple_of(10) {
                eprintln!(
                    "counterfactual_credit: {expressions_processed} expressions processed, \
                     {} applications sampled so far ({:?} elapsed)",
                    rows.len(),
                    start.elapsed()
                );
            }
        }
    }
    out.flush()
        .unwrap_or_else(|e| panic!("counterfactual_credit: flushing {}: {e}", out_jsonl_path.display()));

    assert!(
        !rows.is_empty(),
        "counterfactual_credit: sampled zero applications — nothing to score \
         (wall-clock ceiling hit immediately, or every expression had zero \
         state-changing applications)"
    );

    // ---- Δ distribution ----
    let n_zero = rows.iter().filter(|r| r.delta == 0.0).count();
    let n_positive = rows.iter().filter(|r| r.delta > 0.0).count();
    let n_negative = rows.iter().filter(|r| r.delta < 0.0).count();
    let n = rows.len();

    // ---- per-bound correlations ----
    let row_refs: Vec<&Row> = rows.iter().collect();
    let loose_stats = score_bound(&row_refs, |r| r.bit_loose);
    let tight_stats = score_bound(&row_refs, |r| r.bit_tight);
    let strict_stats = score_bound(&row_refs, |r| r.bit_strict);
    let strict_class_stats = score_bound(&row_refs, |r| r.bit_strict_class);

    println!(
        "=== counterfactual_credit: {n} applications sampled ({expressions_processed} \
         expressions, wall_clock_ceiling_hit={ceiling_hit}) ==="
    );
    println!(
        "  delta distribution: zero {n_zero}/{n} ({:.1}%), positive {n_positive}/{n} ({:.1}%), \
         negative {n_negative}/{n} ({:.1}%)",
        100.0 * n_zero as f64 / n as f64,
        100.0 * n_positive as f64 / n as f64,
        100.0 * n_negative as f64 / n as f64,
    );
    println!(
        "  {:<24} {:>10} {:>10} {:>8} {:>8} {:>8}",
        "bound", "pearson", "spearman", "n_pool", "n_sh", "n_dev"
    );
    for (name, s) in [
        ("loose", &loose_stats),
        ("tight", &tight_stats),
        ("strict", &strict_stats),
        ("strict_by_output_class", &strict_class_stats),
    ] {
        println!(
            "  {name:<24} {:>10.4} {:>10.4} {:>8} {:>8} {:>8}",
            s.pearson_pooled, s.spearman_pooled, s.n_pooled, s.n_sh, s.n_dev
        );
    }
    if !insufficient_candidates.is_empty() {
        println!(
            "  {} expressions had fewer than {} state-changing applications (used all available):",
            insufficient_candidates.len(),
            args.n_apps
        );
        for line in &insufficient_candidates {
            println!("    {line}");
        }
    }

    // ---- JSON report ----
    let mut json = String::new();
    json.push_str("{\n");
    json.push_str(&format!("  \"seed\": {},\n", args.seed));
    json.push_str(&format!("  \"budget\": {},\n", args.budget));
    json.push_str(&format!("  \"n_apps_target\": {},\n", args.n_apps));
    json.push_str(&format!("  \"n_expr_sh\": {},\n", sh_sample.len()));
    json.push_str(&format!("  \"n_expr_dev\": {},\n", dev_sample.len()));
    json.push_str(&format!(
        "  \"expressions_processed\": {expressions_processed},\n"
    ));
    json.push_str(&format!(
        "  \"applications_sampled\": {n},\n"
    ));
    json.push_str(&format!(
        "  \"wall_clock_ceiling_secs\": {},\n",
        args.wall_clock_ceiling_secs
    ));
    json.push_str(&format!("  \"wall_clock_ceiling_hit\": {ceiling_hit},\n"));
    json.push_str(&format!(
        "  \"elapsed_secs\": {:.1},\n",
        start.elapsed().as_secs_f64()
    ));
    json.push_str(&format!(
        "  \"insufficient_candidates_count\": {},\n",
        insufficient_candidates.len()
    ));
    json.push_str("  \"insufficient_candidates\": [\n");
    for (i, line) in insufficient_candidates.iter().enumerate() {
        json.push_str(&format!(
            "    {line:?}{}\n",
            if i + 1 < insufficient_candidates.len() { "," } else { "" }
        ));
    }
    json.push_str("  ],\n");
    json.push_str("  \"delta_distribution\": {\n");
    json.push_str(&format!("    \"n\": {n},\n"));
    json.push_str(&format!(
        "    \"zero\": {n_zero}, \"positive\": {n_positive}, \"negative\": {n_negative},\n"
    ));
    json.push_str(&format!(
        "    \"share_zero\": {:.6}, \"share_positive\": {:.6}, \"share_negative\": {:.6}\n",
        n_zero as f64 / n as f64,
        n_positive as f64 / n as f64,
        n_negative as f64 / n as f64,
    ));
    json.push_str("  },\n");
    json.push_str("  \"bounds\": {\n");
    write_bound_json(&mut json, "loose", &loose_stats, true);
    write_bound_json(&mut json, "tight", &tight_stats, true);
    write_bound_json(&mut json, "strict", &strict_stats, true);
    write_bound_json(&mut json, "strict_by_output_class", &strict_class_stats, false);
    json.push_str("  },\n");
    json.push_str(&format!("  \"git_rev\": {:?},\n", git_rev()));
    json.push_str(&format!("  \"load_at_start\": {:?},\n", load_start));
    json.push_str(&format!("  \"load_at_end\": {:?}\n", uptime()));
    json.push_str("}\n");

    let out_json_path = PathBuf::from(&args.out_json);
    if let Some(parent) = out_json_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    std::fs::write(&out_json_path, &json)
        .unwrap_or_else(|e| panic!("counterfactual_credit: cannot write {}: {e}", out_json_path.display()));
    eprintln!("counterfactual_credit: wrote {}", out_json_path.display());

    // ---- Markdown summary ----
    let mut md = String::new();
    md.push_str("# Counterfactual credit: hindsight bounds vs measured leave-one-out Δ\n\n");
    md.push_str(&format!(
        "Sample: {} `sh` + {} DEV classical expressions, {} applications sampled \
         ({} at ordinal < B={}, seed {:#x}). wall_clock_ceiling_hit = {ceiling_hit}. \
         git {}.\n\n",
        sh_sample.len(),
        dev_sample.len(),
        n,
        n,
        args.budget,
        args.seed,
        git_rev(),
    ));
    md.push_str(&format!(
        "Δ distribution: zero {n_zero}/{n} ({:.1}%), positive {n_positive}/{n} ({:.1}%), \
         negative {n_negative}/{n} ({:.1}%).\n\n",
        100.0 * n_zero as f64 / n as f64,
        100.0 * n_positive as f64 / n as f64,
        100.0 * n_negative as f64 / n as f64,
    ));
    md.push_str("| bound | Pearson (pooled) | Spearman (pooled) | Pearson (sh) | Spearman (sh) | Pearson (dev) | Spearman (dev) | n |\n");
    md.push_str("|---|---:|---:|---:|---:|---:|---:|---:|\n");
    for (name, s) in [
        ("loose", &loose_stats),
        ("tight", &tight_stats),
        ("strict", &strict_stats),
        ("strict_by_output_class", &strict_class_stats),
    ] {
        md.push_str(&format!(
            "| {name} | {} | {} | {} | {} | {} | {} | {} |\n",
            json_num(s.pearson_pooled),
            json_num(s.spearman_pooled),
            json_num(s.pearson_sh),
            json_num(s.spearman_sh),
            json_num(s.pearson_dev),
            json_num(s.spearman_dev),
            s.n_pooled,
        ));
    }
    if !insufficient_candidates.is_empty() {
        md.push_str(&format!(
            "\n{} expressions had fewer than {} state-changing applications and contributed \
             all available instead:\n\n",
            insufficient_candidates.len(),
            args.n_apps
        ));
        for line in &insufficient_candidates {
            md.push_str(&format!("- {line}\n"));
        }
    }

    let out_md_path = PathBuf::from(&args.out_md);
    if let Some(parent) = out_md_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    std::fs::write(&out_md_path, &md)
        .unwrap_or_else(|e| panic!("counterfactual_credit: cannot write {}: {e}", out_md_path.display()));
    eprintln!("counterfactual_credit: wrote {}", out_md_path.display());
    eprintln!("counterfactual_credit: wrote {}", out_jsonl_path.display());
}
