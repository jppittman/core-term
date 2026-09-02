//! Anytime budget curves for e-graph saturation: unguided vs oracle-filtered.
//!
//! Guided-saturation SCOPING measurement (Phase 3 prep,
//! docs/plans/2026-07-07-guided-saturation-redesign.md /
//! docs/plans/2026-08-05-egraph-nnue-research-workflow.md). Answers: **if a
//! perfect Guide replayed only the load-bearing rewrite applications, what
//! happens to e-graph size, saturation work, and extraction quality?** This
//! bounds what ANY learned Guide can deliver.
//!
//! # Why "anytime curves", not "full vs truncated vs oracle" regimes
//!
//! The kernel language is strongly normalizing (bounded `Reduce`, no
//! recursion) and the optimizer inherits that stance: saturation runs under a
//! budget tier and spends it, full stop — it does not detect or rely on
//! reaching a fixed point, and a fixed point is not a certified state
//! anywhere in this pipeline (`EGraph::saturate_with_limits` stops on
//! iteration count, class count, timeout, OR convergence — whichever comes
//! first — and production callers never distinguish which one fired). So
//! there is no privileged "C_full" to call the reference and everything else
//! "regret against". This harness instead runs ONE thing per rule set: a
//! single incremental saturation from empty to a generous ceiling (3x the
//! expression's own production tier budget — see [`FRACS`]), sampling
//! extraction cost at standardized work fractions along the way. That is the
//! anytime curve. Two curves are run per expression — **unguided** (the full
//! 62-rule library, `pixelflow_search::math::all_rules()`) and
//! **oracle-filtered** (only the rules a hindsight pass over the
//! unguided run's best-found extraction marked load-bearing) — and the
//! reference for regret at any work level is the lowest cost either curve
//! ever reaches, empirically, never a claimed optimum.
//!
//! # Oracle replay: the approximation actually taken
//!
//! Exact per-application replay is structurally hard here and the task this
//! harness answers anticipates that: "if exact replay is hard, approximate
//! honestly and say how." `EClassId`s are allocated sequentially as
//! `EGraph::add` is called; a recorded `ApplicationRecord::match_root` names
//! a class in the ORIGINAL run's id space, and skipping even one
//! non-load-bearing application during a replay shifts every later
//! `EClassId` by one, so "replay only these `ApplicationId`s on a fresh
//! graph" has no stable target to aim `apply_single_rule` at without a
//! content-addressed match key this architecture does not have (adding one
//! would be a `provenance.rs` redesign, out of scope this round).
//!
//! So this harness filters at RULE granularity instead of match granularity:
//! `R* = { rule_idx : some load-bearing application in the unguided run's
//! best-found state fired this rule }`, then a fresh e-graph's rule LIST is
//! restricted to `R*` for the whole oracle curve. This sidesteps id stability
//! entirely (no cross-run `EClassId`/`ApplicationId` needs to survive), and
//! it makes the "transitive closure of match dependencies" concern the task
//! flags moot: filtering keeps or drops a whole rule, so a chain "rule A's
//! product enables rule B's match" survives automatically whenever both A and
//! B are in `R*` — no explicit dependency-closure computation needed. It is
//! coarser than the task's literal ask in exactly one, safe direction: it
//! retains EVERY firing of an oracle-approved rule, including firings that
//! individually would have been wasted, so oracle's work numbers are a
//! pessimistic (upper-bound) estimate of what a perfect application-level
//! oracle would need. This also matches how the Guide is motivated in the
//! redesign doc ("Can rule filtering keep the e-graph within budget...") —
//! global rule masking is the mechanism the thesis names, not a
//! simplification of it.
//!
//! # Non-direct-creator share (redesign.md lines 88-90 follow-up)
//!
//! Additive, read-only measurement, no change to `derivation_ancestors`'s
//! logic: for every application the labeler marks `LoadBearing` (against the
//! unguided run's best-found extraction), this harness checks whether the
//! e-node(s) that application actually CREATED are among the literal
//! chosen-extraction tags (walked the same way `labeler::chosen_tagged_nodes`
//! does, via the public `find`/`nodes`/`tags` API). The fraction that are
//! NOT is reported as `credited_non_direct_ratio`.
//!
//! **This does not isolate any single over-approximation axis** (a PR
//! review caught an earlier version of this doc/code claiming it did).
//! `derivation_ancestors` documents three (`provenance.rs`): (1) crediting
//! every node in a class, not just the chosen one, (2) pulling in union
//! events by class membership, (3) no fixed-point pruning. A load-bearing
//! application whose created node isn't on the literal chosen path could be
//! there via any of the three, OR because it is a genuine transitive
//! enabler the labeler is *correctly* crediting (real "A enables B" credit
//! chess-style provenance can't observe any other way — see the redesign
//! plan's "observed credit" section). This measurement cannot tell those
//! apart; it answers "what fraction of load-bearing applications are not the
//! literal creator of a chosen node," not "what fraction is over-crediting
//! via class membership specifically." Distinguishing genuine transitive
//! credit from each named over-approximation axis would need per-application
//! provenance of *why* it entered the ancestry closure, which
//! `derivation_ancestors` does not currently record and this round does not
//! add. The one new API surface, `Provenance::origins()`, is a read-only
//! iterator mirroring the existing `applications()`/`union_events()`
//! accessors — no semantic change.
//!
//! # "Quiesced before cap" — a diagnostic, not an organizing axis
//!
//! For each curve this harness also records whether `saturate_with_limits`
//! stopped producing new nodes (`total_unions == 0`) strictly before the
//! expression's own nominal tier budget was spent. This is reported purely
//! as an emergent fact (how often does a production-sized budget have slack
//! left over, and does that correlate with expression size) — it never
//! partitions the corpus or gates which numbers get analyzed.
//!
//! # Shaders in-sample
//!
//! Five named, hand-written realistic kernels (`swirl`, `circle_sdf`, `poly`,
//! `redundant`, `normalize` — the same five `pixelflow-search/examples/
//! rule_report.rs` uses) are run through the identical pipeline alongside the
//! synthetic corpus, and reported against the synthetic corpus's per-tier
//! distribution, so the synthetic-corpus conclusions aren't only validated on
//! synthetic shapes.
//!
//! All measurements are deterministic counts and static-latency-prior cost
//! units (`CostModel::latency_prior()`). The only `Duration` ever passed to
//! `saturate_with_limits` is a 300s safety ceiling no expression here comes
//! close to — wall-clock plays no role in any reported number.
//!
//! Run: `cargo run --release -p pixelflow-search --example oracle_filtered_budget_curves`
//! Output: prints a summary to stdout and writes a full per-expression,
//! per-curve, per-work-fraction CSV to `--out` (default
//! `docs/results/2026-08-30-oracle-filtered-budget-curves.csv`).

use std::collections::{BTreeSet, HashMap};
use std::io::Write as _;
use std::path::PathBuf;
use std::time::Duration;

use pixelflow_ir::arena::ExprNode;
use pixelflow_ir::{ExprArena, ExprId, OpKind};
use pixelflow_search::egraph::{
    CostModel, EClassId, EGraph, ENodeId, EpisodeLabels, ExtractedDAG, Origin, Rewrite, all_rules,
    collect_rule_templates, config_for_node_count, extract_dag,
};
use pixelflow_search::nnue::{BwdGenConfig, BwdGenerator};

/// Safety ceiling only — see module docs. Never approached by anything here.
/// Applies to one expression's WHOLE curve (`run_anytime_curve`'s single-
/// iteration `saturate_with_limits` calls each pass the *remaining* duration
/// against one deadline computed once, not this constant re-passed fresh —
/// `saturate_with_limits` starts its own `Instant::now()` clock on every
/// call, so re-passing the constant would let a `ceiling_iters`-long curve
/// run for up to `ceiling_iters * SAFETY_TIMEOUT`, not `SAFETY_TIMEOUT`).
const SAFETY_TIMEOUT: Duration = Duration::from_secs(300);

/// Standardized work-fraction grid, expressed as a multiple of the
/// expression's own production tier budget (`config_for_node_count`).
/// 0.25/0.5 stand in for the task's "truncated budget" regimes; 1.0 is the
/// tier's nominal budget; 2.0/3.0 probe beyond it for the
/// better-forms-at-max-budget observation.
const FRACS: &[f64] = &[0.25, 0.5, 1.0, 2.0, 3.0];

// ============================================================================
// Corpus generation
// ============================================================================

struct Band {
    max_depth: usize,
    leaf_prob: f32,
    num_vars: usize,
}

/// Mirrors the shape (not the exact list) of `pixelflow-pipeline/src/bin/
/// gen_bench_corpus.rs`'s `BANDS` — depth/leaf/var knobs spanning tiny to
/// very-deep expressions, so the corpus naturally spans all three production
/// time-control tiers (blitz/rapid/classical).
const BANDS: &[Band] = &[
    Band {
        max_depth: 2,
        leaf_prob: 0.55,
        num_vars: 2,
    },
    Band {
        max_depth: 3,
        leaf_prob: 0.50,
        num_vars: 2,
    },
    Band {
        max_depth: 4,
        leaf_prob: 0.45,
        num_vars: 3,
    },
    Band {
        max_depth: 5,
        leaf_prob: 0.40,
        num_vars: 4,
    },
    Band {
        max_depth: 6,
        leaf_prob: 0.35,
        num_vars: 4,
    },
    Band {
        max_depth: 8,
        leaf_prob: 0.30,
        num_vars: 4,
    },
    Band {
        max_depth: 10,
        leaf_prob: 0.25,
        num_vars: 4,
    },
    Band {
        max_depth: 12,
        leaf_prob: 0.20,
        num_vars: 4,
    },
    Band {
        max_depth: 15,
        leaf_prob: 0.18,
        num_vars: 4,
    },
    Band {
        max_depth: 18,
        leaf_prob: 0.15,
        num_vars: 4,
    },
];

/// 10 bands x 22 = 220 >= the task's 200 floor.
const SAMPLES_PER_BAND: usize = 22;

/// Fixed seed: the whole synthetic corpus (and therefore every measurement
/// below) is reproducible byte-for-byte from this one constant.
const BASE_SEED: u64 = 20260830;

/// Compact the reachable subtree of `root` in `src` into a fresh, minimal
/// arena, remapping `ExprId`s along the way.
///
/// `EGraph::add_arena` walks `0..arena.len()` unconditionally (it requires
/// topological order from 0, not just reachability from `root`).
/// `BwdGenerator::generate_arena` packs BOTH the optimized and unoptimized
/// subtree of one `BwdTrainingPairArena` side by side in a single arena, so
/// passing that raw arena straight to `add_arena` would silently add the
/// disconnected sibling subtree's nodes into the e-graph too, polluting every
/// work/cost measurement with rule matches against an expression nobody
/// asked to saturate (the same B3 bug `pixelflow-pipeline`'s corpus writer
/// documents closing).
fn compact_subtree(
    src: &ExprArena,
    root: ExprId,
    dst: &mut ExprArena,
    memo: &mut HashMap<u32, ExprId>,
) -> ExprId {
    if let Some(&id) = memo.get(&root.0) {
        return id;
    }
    let node = src.node(root).clone();
    let new_id = match node {
        ExprNode::Var(v) => dst.push_var(v),
        ExprNode::Const(v) => dst.push_const(v),
        ExprNode::Unary(op, c) => {
            let nc = compact_subtree(src, c, dst, memo);
            dst.push_unary(op, nc)
        }
        ExprNode::Binary(op, a, b) => {
            let na = compact_subtree(src, a, dst, memo);
            let nb = compact_subtree(src, b, dst, memo);
            dst.push_binary(op, na, nb)
        }
        ExprNode::Ternary(op, a, b, c) => {
            let na = compact_subtree(src, a, dst, memo);
            let nb = compact_subtree(src, b, dst, memo);
            let nc = compact_subtree(src, c, dst, memo);
            dst.push_ternary(op, na, nb, nc)
        }
        ExprNode::Nary(op, _, _) => {
            let kids: Vec<ExprId> = src
                .children(root)
                .map(|c| compact_subtree(src, c, dst, memo))
                .collect();
            dst.push_nary(op, &kids)
        }
        ExprNode::Param(i) => panic!(
            "compact_subtree: unexpected Param({i}) in a BwdGenerator expression \
             (the generator never produces Param nodes)"
        ),
        ExprNode::Buffer(_) => panic!(
            "compact_subtree: unexpected Buffer node in a BwdGenerator expression \
             (the generator never produces memory ops)"
        ),
    };
    memo.insert(root.0, new_id);
    new_id
}

struct CorpusItem {
    name: String,
    arena: ExprArena,
    root: ExprId,
    node_count: usize,
    /// `true` for the five named realistic kernels (the "shaders in-sample"
    /// check); `false` for synthetic band-generated expressions.
    is_named_shader: bool,
}

fn build_synthetic_corpus() -> Vec<CorpusItem> {
    let templates = collect_rule_templates();
    let mut generator = BwdGenerator::new(BASE_SEED, BwdGenConfig::default(), templates);
    let mut corpus = Vec::with_capacity(BANDS.len() * SAMPLES_PER_BAND);

    for (band_idx, band) in BANDS.iter().enumerate() {
        generator.config = BwdGenConfig {
            max_depth: band.max_depth,
            leaf_prob: band.leaf_prob,
            num_vars: band.num_vars,
            ..BwdGenConfig::default()
        };
        for sample in 0..SAMPLES_PER_BAND {
            let pair = generator.generate_arena();
            let mut dst = ExprArena::with_capacity(pair.arena.node_count_subtree(pair.unoptimized));
            let mut memo = HashMap::new();
            let new_root = compact_subtree(&pair.arena, pair.unoptimized, &mut dst, &mut memo);
            let node_count = dst.len();
            corpus.push(CorpusItem {
                name: format!("band{band_idx}_depth{}_s{sample}", band.max_depth),
                arena: dst,
                root: new_root,
                node_count,
                is_named_shader: false,
            });
        }
    }
    corpus
}

/// The five named realistic kernels from `rule_report.rs`, for the
/// shaders-in-sample check.
fn named_shader_corpus() -> Vec<CorpusItem> {
    fn swirl() -> (ExprArena, ExprId) {
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let y = a.push_var(1);
        let xx = a.push_binary(OpKind::Mul, x, x);
        let yy = a.push_binary(OpKind::Mul, y, y);
        let d = a.push_binary(OpKind::Add, xx, yy);
        let s = a.push_unary(OpKind::Sqrt, d);
        let kf = a.push_const(3.0);
        let sf = a.push_binary(OpKind::Mul, s, kf);
        let sn = a.push_unary(OpKind::Sin, sf);
        let ka = a.push_const(0.5);
        let prod = a.push_binary(OpKind::Mul, sn, ka);
        let kb = a.push_const(0.5);
        let out = a.push_binary(OpKind::Add, prod, kb);
        (a, out)
    }
    fn circle_sdf() -> (ExprArena, ExprId) {
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let y = a.push_var(1);
        let cx = a.push_const(0.3);
        let cy = a.push_const(-0.2);
        let dx = a.push_binary(OpKind::Sub, x, cx);
        let dy = a.push_binary(OpKind::Sub, y, cy);
        let dx2 = a.push_binary(OpKind::Mul, dx, dx);
        let dy2 = a.push_binary(OpKind::Mul, dy, dy);
        let sum = a.push_binary(OpKind::Add, dx2, dy2);
        let dist = a.push_unary(OpKind::Sqrt, sum);
        let r = a.push_const(0.5);
        let out = a.push_binary(OpKind::Sub, dist, r);
        (a, out)
    }
    fn poly() -> (ExprArena, ExprId) {
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let ka = a.push_const(2.0);
        let kb = a.push_const(-3.0);
        let kc = a.push_const(1.0);
        let xx = a.push_binary(OpKind::Mul, x, x);
        let ax2 = a.push_binary(OpKind::Mul, ka, xx);
        let bx = a.push_binary(OpKind::Mul, kb, x);
        let s1 = a.push_binary(OpKind::Add, ax2, bx);
        let out = a.push_binary(OpKind::Add, s1, kc);
        (a, out)
    }
    fn redundant() -> (ExprArena, ExprId) {
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let y = a.push_var(1);
        let s = a.push_binary(OpKind::Add, x, y);
        let s2 = a.push_binary(OpKind::Mul, s, s);
        let two = a.push_const(2.0);
        let ts = a.push_binary(OpKind::Mul, two, s);
        let out = a.push_binary(OpKind::Add, s2, ts);
        (a, out)
    }
    fn normalize() -> (ExprArena, ExprId) {
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let y = a.push_var(1);
        let xx = a.push_binary(OpKind::Mul, x, x);
        let yy = a.push_binary(OpKind::Mul, y, y);
        let d = a.push_binary(OpKind::Add, xx, yy);
        let s = a.push_unary(OpKind::Sqrt, d);
        let out = a.push_binary(OpKind::Div, x, s);
        (a, out)
    }

    type ShaderBuilder = fn() -> (ExprArena, ExprId);
    let cases: Vec<(&str, ShaderBuilder)> = vec![
        ("shader_swirl", swirl),
        ("shader_circle_sdf", circle_sdf),
        ("shader_poly", poly),
        ("shader_redundant", redundant),
        ("shader_normalize", normalize),
    ];
    cases
        .into_iter()
        .map(|(name, build)| {
            let (arena, root) = build();
            let node_count = arena.node_count_subtree(root);
            CorpusItem {
                name: name.to_string(),
                arena,
                root,
                node_count,
                is_named_shader: true,
            }
        })
        .collect()
}

fn tier_name(node_count: usize) -> &'static str {
    match node_count {
        0..=10 => "blitz",
        11..=50 => "rapid",
        _ => "classical",
    }
}

// ============================================================================
// Chosen-tag walk (mirrors `labeler::chosen_tagged_nodes`, public API only)
// ============================================================================

fn chosen_tags(egraph: &EGraph, root: EClassId, choices: &[Option<usize>]) -> BTreeSet<ENodeId> {
    let mut visited: BTreeSet<EClassId> = BTreeSet::new();
    let mut stack = vec![root];
    let mut tags = BTreeSet::new();
    while let Some(class) = stack.pop() {
        let canonical = egraph.find(class);
        if !visited.insert(canonical) {
            continue;
        }
        let idx = canonical.index();
        let node_idx = choices.get(idx).and_then(|o| *o).unwrap_or_else(|| {
            panic!("chosen_tags: e-class {idx} reachable from root has no recorded choice")
        });
        let nodes = egraph.nodes(canonical);
        let node_tags = egraph.tags(canonical);
        tags.insert(node_tags[node_idx]);
        for child in nodes[node_idx].children() {
            stack.push(child);
        }
    }
    tags
}

// ============================================================================
// Anytime curve
// ============================================================================

#[derive(Clone, Debug)]
struct Checkpoint {
    frac: f64,
    iteration: usize,
    applications: usize,
    classes: usize,
    nodes: usize,
    cost: usize,
}

fn make_checkpoint(
    egraph: &EGraph,
    extraction: &ExtractedDAG,
    iteration: usize,
    frac: f64,
) -> Checkpoint {
    Checkpoint {
        frac,
        iteration,
        applications: egraph.provenance().recorded_count(),
        classes: egraph.num_classes(),
        nodes: egraph.node_count(),
        // The cost of the kernel this checkpoint would EMIT: each distinct
        // chosen e-class once. `total_cost` is the DP's tree cost, which
        // pays a shared subterm once per use and so tracks how much sharing
        // saturation happened to expose rather than how good the extraction
        // is — not the quantity a regret curve is asking about (#1111).
        cost: extraction.dag_cost,
    }
}

/// Run one incremental saturation from empty to `FRACS.last() * n_iters_nominal`
/// iterations, sampling extraction cost at each fraction in `FRACS`. Returns
/// the checkpoints, the iteration at which the run stopped producing new
/// nodes (`None` if it never did, within the tested ceiling), and the final
/// e-graph + extraction (whichever iteration the loop actually stopped at) —
/// the latter two feed hindsight labeling for the caller that runs the
/// unguided curve.
fn run_anytime_curve(
    arena: &ExprArena,
    root: ExprId,
    rules: Vec<Box<dyn Rewrite>>,
    n_iters_nominal: usize,
    n_classes_nominal: usize,
) -> (Vec<Checkpoint>, Option<usize>, EGraph, ExtractedDAG) {
    let costs = CostModel::latency_prior();
    let target_iters: Vec<usize> = FRACS
        .iter()
        .map(|f| ((*f * n_iters_nominal as f64).round() as usize).max(1))
        .collect();
    assert!(
        target_iters.windows(2).all(|w| w[0] < w[1]),
        "FRACS must produce strictly increasing iteration targets for n_iters_nominal={n_iters_nominal}, got {target_iters:?}"
    );
    let ceiling_iters = *target_iters.last().expect("FRACS must be non-empty");
    // Per-checkpoint class caps (parallel to `target_iters`), not one fixed
    // `ceiling_classes` reused for the whole curve. A PR review caught this:
    // reusing the final (3x) class cap throughout meant a checkpoint labeled
    // "frac=1.0" could actually reflect an e-graph that grew to the 3x class
    // allowance -- iteration count alone doesn't bound class growth, since a
    // rule sweep can pack far more class growth into few iterations than the
    // nominal tier budget intends. Verified against the committed data: 59
    // of 225 unguided rows exceeded their tier's nominal class cap at
    // frac=1.0, some reaching close to 3x it.
    let target_classes: Vec<usize> = FRACS
        .iter()
        .map(|f| ((*f * n_classes_nominal as f64).round() as usize).max(1))
        .collect();

    let mut egraph = EGraph::with_rules(rules);
    let root_class = egraph.add_arena(arena, root);

    let mut checkpoints: Vec<Checkpoint> = Vec::with_capacity(FRACS.len());
    let mut quiesced_at: Option<usize> = None;
    let mut target_idx = 0;
    // One deadline for the whole curve, not one per single-iteration call
    // (see `SAFETY_TIMEOUT`'s doc comment for why re-passing the constant
    // would let this loop run far longer than the documented ceiling).
    let curve_deadline = std::time::Instant::now() + SAFETY_TIMEOUT;

    for iter in 1..=ceiling_iters {
        let remaining = curve_deadline.saturating_duration_since(std::time::Instant::now());
        assert!(
            !remaining.is_zero(),
            "oracle_filtered_budget_curves: one expression's curve ran past the \
             {SAFETY_TIMEOUT:?} safety ceiling (iteration {iter}/{ceiling_iters}) -- this \
             was expected to never bind at this corpus's scale; fail loud rather than \
             silently truncate and report a partial curve"
        );
        // Cap growth at the class budget of the NEXT checkpoint not yet
        // reached -- once that checkpoint is sampled below, `target_idx`
        // advances and the cap grows for the next one.
        let active_class_cap = target_classes[target_idx.min(target_classes.len() - 1)];
        let stats = egraph.saturate_with_limits(1, active_class_cap, remaining);
        if target_idx < target_iters.len() && iter == target_iters[target_idx] {
            let extraction = extract_dag(&egraph, root_class, &costs);
            checkpoints.push(make_checkpoint(
                &egraph,
                &extraction,
                iter,
                FRACS[target_idx],
            ));
            target_idx += 1;
        }
        // `stats.iterations == 0` means the ACTIVE checkpoint's class cap
        // was already exceeded before this round could run at all -- not
        // genuine quiescence (the next checkpoint's larger cap may still
        // allow growth once `target_idx` advances past it). Only a round
        // that actually ran and found nothing (`iterations > 0 &&
        // total_unions == 0`) is real quiescence -- the same distinction
        // `guide_scope_saturation_delta.rs` makes for its own class-cap
        // guard, for the same underlying reason (allocated vs. requested
        // class count are different quantities `saturate_with_limits`
        // enforces internally).
        if stats.iterations > 0 && stats.total_unions == 0 {
            quiesced_at = Some(iter);
            break;
        }
    }

    let final_extraction = extract_dag(&egraph, root_class, &costs);
    let final_iter = quiesced_at.unwrap_or(ceiling_iters);
    while target_idx < target_iters.len() {
        checkpoints.push(make_checkpoint(
            &egraph,
            &final_extraction,
            final_iter,
            FRACS[target_idx],
        ));
        target_idx += 1;
    }

    (checkpoints, quiesced_at, egraph, final_extraction)
}

// ============================================================================
// Per-expression measurement
// ============================================================================

#[derive(Clone, Debug)]
struct CurveRow {
    expr_name: String,
    tier: &'static str,
    node_count: usize,
    curve: &'static str, // "unguided" | "oracle"
    frac: f64,
    iteration: usize,
    rules_allowed: usize,
    applications: usize,
    classes: usize,
    nodes: usize,
    cost: usize,
    regret_pct: f64,
}

struct ExprMeasurement {
    rows: Vec<CurveRow>,
    unguided_quiesced_before_nominal: bool,
    oracle_quiesced_before_nominal: bool,
    /// unguided cost at frac=1.0 minus unguided cost at frac=3.0 (>=0 if a
    /// strictly cheaper form existed beyond the nominal budget).
    better_form_gap_pct: f64,
    load_bearing: usize,
    total_applications: usize,
    credited_non_direct: usize,
}

fn measure_expression(item: &CorpusItem) -> ExprMeasurement {
    let tier = tier_name(item.node_count);
    let config = config_for_node_count(item.node_count);
    let n_iters = config.max_iterations;
    let n_classes = config.max_classes;

    let full_rules = all_rules();
    let full_rules_count = full_rules.len();
    let (uc_checkpoints, uc_quiesced, uc_egraph, uc_extraction) =
        run_anytime_curve(&item.arena, item.root, full_rules, n_iters, n_classes);

    let labels = EpisodeLabels::compute(&uc_egraph, uc_extraction.root, &uc_extraction.choices);
    let oracle_rule_idxs: BTreeSet<usize> = labels
        .load_bearing
        .iter()
        .filter_map(|app_id| uc_egraph.provenance().application(*app_id))
        .map(|record| record.rule_idx)
        .collect();
    let oracle_rules: Vec<Box<dyn Rewrite>> = all_rules()
        .into_iter()
        .enumerate()
        .filter(|(idx, _)| oracle_rule_idxs.contains(idx))
        .map(|(_, r)| r)
        .collect();
    let oracle_rules_len = oracle_rules.len();

    let (oc_checkpoints, oc_quiesced, _oc_egraph, _oc_extraction) =
        run_anytime_curve(&item.arena, item.root, oracle_rules, n_iters, n_classes);

    // Over-approximation looseness.
    let chosen = chosen_tags(&uc_egraph, uc_extraction.root, &uc_extraction.choices);
    let mut created_by: HashMap<u64, Vec<ENodeId>> = HashMap::new();
    for (enode_id, origin) in uc_egraph.provenance().origins() {
        if let Origin::Rule(app_id) = origin {
            created_by
                .entry(app_id.as_u64())
                .or_default()
                .push(enode_id);
        }
    }
    let mut credited_non_direct = 0usize;
    for app_id in &labels.load_bearing {
        let created = created_by.get(&app_id.as_u64());
        let on_chosen_path = created
            .map(|nodes| nodes.iter().any(|n| chosen.contains(n)))
            .unwrap_or(false);
        if !on_chosen_path {
            credited_non_direct += 1;
        }
    }

    let global_best = uc_checkpoints
        .iter()
        .chain(oc_checkpoints.iter())
        .map(|c| c.cost)
        .min()
        .expect("FRACS non-empty => at least one checkpoint per curve");

    // `global_best == 0` means SOME checkpoint (either curve, any frac)
    // extracted to a free `Const`/`Var`. A checkpoint that matches that
    // (cost == 0 too) is exactly as good -- 0% regret, correct. A checkpoint
    // with positive cost is not "equally good relative to a free reference"
    // -- (cost - 0) / 0 is not 0, it is unboundedly worse, since ANY nonzero
    // cost is infinitely far from free. Reporting 0.0 there (the previous
    // behavior) silently erased a real regret signal for any expression
    // whose curve simplifies to zero-cost only at a LATER checkpoint than
    // an earlier positive-cost one -- not observed in this corpus's current
    // checkpoint grid (see the report's "checkpoint-grid miscalibration"
    // finding: costs are flat across checkpoints for every sampled
    // expression here), but exactly the failure mode a recalibrated,
    // finer-grained rerun would be vulnerable to.
    let regret = |cost: usize| -> f64 {
        if global_best == 0 {
            if cost == 0 { 0.0 } else { f64::INFINITY }
        } else {
            (cost as f64 - global_best as f64) / global_best as f64 * 100.0
        }
    };

    let mut rows = Vec::with_capacity(uc_checkpoints.len() + oc_checkpoints.len());
    let uc_nominal_cost = uc_checkpoints
        .iter()
        .find(|c| (c.frac - 1.0).abs() < 1e-9)
        .expect("frac=1.0 checkpoint always recorded")
        .cost;
    let uc_max_cost = uc_checkpoints
        .iter()
        .find(|c| (c.frac - *FRACS.last().unwrap()).abs() < 1e-9)
        .expect("max-frac checkpoint always recorded")
        .cost;
    let better_form_gap_pct = if uc_nominal_cost == 0 {
        0.0
    } else {
        (uc_nominal_cost as f64 - uc_max_cost as f64) / uc_nominal_cost as f64 * 100.0
    };

    for c in &uc_checkpoints {
        rows.push(CurveRow {
            expr_name: item.name.clone(),
            tier,
            node_count: item.node_count,
            curve: "unguided",
            frac: c.frac,
            iteration: c.iteration,
            rules_allowed: full_rules_count,
            applications: c.applications,
            classes: c.classes,
            nodes: c.nodes,
            cost: c.cost,
            regret_pct: regret(c.cost),
        });
    }
    for c in &oc_checkpoints {
        rows.push(CurveRow {
            expr_name: item.name.clone(),
            tier,
            node_count: item.node_count,
            curve: "oracle",
            frac: c.frac,
            iteration: c.iteration,
            rules_allowed: oracle_rules_len,
            applications: c.applications,
            classes: c.classes,
            nodes: c.nodes,
            cost: c.cost,
            regret_pct: regret(c.cost),
        });
    }

    ExprMeasurement {
        rows,
        unguided_quiesced_before_nominal: uc_quiesced.map(|q| q < n_iters).unwrap_or(false),
        oracle_quiesced_before_nominal: oc_quiesced.map(|q| q < n_iters).unwrap_or(false),
        better_form_gap_pct,
        load_bearing: labels.load_bearing.len(),
        total_applications: uc_egraph.provenance().recorded_count(),
        credited_non_direct,
    }
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out_path = args
        .iter()
        .position(|a| a == "--out")
        .and_then(|i| args.get(i + 1))
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from("docs/results/2026-08-30-oracle-filtered-budget-curves.csv")
        });

    let rule_count = all_rules().len();
    println!("rule library size: {rule_count} rules");

    let mut corpus = build_synthetic_corpus();
    let synthetic_count = corpus.len();
    corpus.extend(named_shader_corpus());
    eprintln!(
        "corpus: {} synthetic + {} named shaders = {} expressions across {} bands",
        synthetic_count,
        corpus.len() - synthetic_count,
        corpus.len(),
        BANDS.len()
    );
    assert!(
        synthetic_count >= 200,
        "task requires >=200 synthetic corpus expressions, got {synthetic_count}"
    );

    let mut all_rows: Vec<CurveRow> = Vec::new();
    let mut quiesce_diag: Vec<(String, &'static str, usize, bool, bool)> = Vec::new();
    let mut better_form_diag: Vec<(String, &'static str, f64)> = Vec::new();
    let mut looseness_diag: Vec<(String, &'static str, usize, usize, usize)> = Vec::new();

    for (i, item) in corpus.iter().enumerate() {
        let m = measure_expression(item);
        all_rows.extend(m.rows);
        let tier = tier_name(item.node_count);
        quiesce_diag.push((
            item.name.clone(),
            tier,
            item.node_count,
            m.unguided_quiesced_before_nominal,
            m.oracle_quiesced_before_nominal,
        ));
        better_form_diag.push((item.name.clone(), tier, m.better_form_gap_pct));
        looseness_diag.push((
            item.name.clone(),
            tier,
            m.load_bearing,
            m.total_applications,
            m.credited_non_direct,
        ));
        if (i + 1) % 40 == 0 {
            eprintln!("... {}/{} expressions done", i + 1, corpus.len());
        }
    }

    // ------------------------------------------------------------------
    // CSV
    // ------------------------------------------------------------------
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent).expect("create output directory");
    }
    let mut f = std::fs::File::create(&out_path).expect("create output CSV");
    writeln!(
        f,
        "expr_name,tier,node_count,curve,frac,iteration,rules_allowed,applications,classes,nodes,cost,regret_pct"
    )
    .unwrap();
    for r in &all_rows {
        writeln!(
            f,
            "{},{},{},{},{},{},{},{},{},{},{},{:.4}",
            r.expr_name,
            r.tier,
            r.node_count,
            r.curve,
            r.frac,
            r.iteration,
            r.rules_allowed,
            r.applications,
            r.classes,
            r.nodes,
            r.cost,
            r.regret_pct,
        )
        .unwrap();
    }
    println!("wrote {} rows to {}", all_rows.len(), out_path.display());

    // ------------------------------------------------------------------
    // Summary: regret% and work (applications) by curve x frac, overall and
    // per tier. Synthetic corpus only (named shaders reported separately).
    // ------------------------------------------------------------------
    let synthetic_names: BTreeSet<&str> = corpus
        .iter()
        .filter(|c| !c.is_named_shader)
        .map(|c| c.name.as_str())
        .collect();
    let tiers = ["blitz", "rapid", "classical"];

    println!("\n=== anytime curves: regret% vs work, synthetic corpus (n={synthetic_count}) ===");
    for scope in ["ALL"].iter().chain(tiers.iter()) {
        println!("--- scope: {scope} ---");
        for curve in ["unguided", "oracle"] {
            for &frac in FRACS {
                let mut regrets: Vec<f64> = Vec::new();
                let mut apps: Vec<f64> = Vec::new();
                for r in all_rows.iter().filter(|r| {
                    r.curve == curve
                        && (r.frac - frac).abs() < 1e-9
                        && synthetic_names.contains(r.expr_name.as_str())
                        && (*scope == "ALL" || r.tier == *scope)
                }) {
                    regrets.push(r.regret_pct);
                    apps.push(r.applications as f64);
                }
                if regrets.is_empty() {
                    continue;
                }
                regrets.sort_by(|a, b| a.partial_cmp(b).unwrap());
                apps.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
                let median = |v: &[f64]| v[v.len() / 2];
                println!(
                    "  {curve:<9} frac={frac:<4.2} n={:<4} regret%: mean={:>7.2} median={:>7.2} | \
                     applications: mean={:>8.1} median={:>8.1}",
                    regrets.len(),
                    mean(&regrets),
                    median(&regrets),
                    mean(&apps),
                    median(&apps),
                );
            }
        }
    }

    // ------------------------------------------------------------------
    // Diagnostic: quiesced before nominal cap (NOT an organizing axis).
    // ------------------------------------------------------------------
    println!("\n=== diagnostic: quiesced before nominal (frac=1.0) budget ===");
    for scope in ["ALL"].iter().chain(tiers.iter()) {
        let rows: Vec<_> = quiesce_diag
            .iter()
            .filter(|(name, t, ..)| {
                synthetic_names.contains(name.as_str()) && (*scope == "ALL" || t == scope)
            })
            .collect();
        if rows.is_empty() {
            continue;
        }
        let n = rows.len();
        let uc = rows.iter().filter(|(_, _, _, u, _)| *u).count();
        let oc = rows.iter().filter(|(_, _, _, _, o)| *o).count();
        println!(
            "  {scope:<10} n={n:<4} unguided quiesced early: {uc}/{n} ({:.1}%)  \
             oracle quiesced early: {oc}/{n} ({:.1}%)",
            uc as f64 / n as f64 * 100.0,
            oc as f64 / n as f64 * 100.0,
        );
    }

    // ------------------------------------------------------------------
    // Diagnostic: better forms beyond nominal budget (unguided curve).
    // ------------------------------------------------------------------
    println!(
        "\n=== diagnostic: better forms beyond nominal budget (unguided, frac 1.0 -> {:.1}) ===",
        FRACS.last().unwrap()
    );
    for scope in ["ALL"].iter().chain(tiers.iter()) {
        let gaps: Vec<f64> = better_form_diag
            .iter()
            .filter(|(name, t, _)| {
                synthetic_names.contains(name.as_str()) && (*scope == "ALL" || t == scope)
            })
            .map(|(_, _, g)| *g)
            .collect();
        if gaps.is_empty() {
            continue;
        }
        let n = gaps.len();
        let improved = gaps.iter().filter(|g| **g > 1e-9).count();
        let mean = gaps.iter().sum::<f64>() / n as f64;
        let mut sorted = gaps.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "  {scope:<10} n={n:<4} strictly-better-at-max-budget: {improved}/{n} ({:.1}%)  \
             mean gap={:.2}%  median gap={:.2}%",
            improved as f64 / n as f64 * 100.0,
            mean,
            sorted[n / 2],
        );
    }

    // ------------------------------------------------------------------
    // Over-approximation looseness.
    // ------------------------------------------------------------------
    let synth_loose: Vec<_> = looseness_diag
        .iter()
        .filter(|(name, ..)| synthetic_names.contains(name.as_str()))
        .collect();
    let total_lb: usize = synth_loose.iter().map(|(_, _, lb, _, _)| lb).sum();
    let total_credited: usize = synth_loose.iter().map(|(_, _, _, _, c)| c).sum();
    println!("\n=== non-direct-creator share of load-bearing applications (redesign.md 88-90) ===");
    println!(
        "  overall: {total_credited}/{total_lb} ({:.2}%) of load-bearing applications did not \
         literally create a node on the chosen-extraction path (does NOT isolate class-membership \
         over-approximation specifically -- see module doc's \"Non-direct-creator share\" section; \
         genuine transitive-enabler credit and all three named over-approximation axes are lumped \
         together here)",
        total_credited as f64 / total_lb.max(1) as f64 * 100.0,
    );
    for t in tiers {
        let lb: usize = synth_loose
            .iter()
            .filter(|(_, tt, ..)| *tt == t)
            .map(|(_, _, lb, _, _)| lb)
            .sum();
        let c: usize = synth_loose
            .iter()
            .filter(|(_, tt, ..)| *tt == t)
            .map(|(_, _, _, _, c)| c)
            .sum();
        if lb > 0 {
            println!("  {t:<10}: {c}/{lb} ({:.2}%)", c as f64 / lb as f64 * 100.0);
        }
    }

    // ------------------------------------------------------------------
    // Shaders in-sample.
    // ------------------------------------------------------------------
    println!("\n=== shaders in-sample check (named realistic kernels vs synthetic corpus) ===");
    for item in corpus.iter().filter(|c| c.is_named_shader) {
        let tier = tier_name(item.node_count);
        let synth_regret_at_1: Vec<f64> = all_rows
            .iter()
            .filter(|r| {
                r.tier == tier
                    && r.curve == "unguided"
                    && (r.frac - 1.0).abs() < 1e-9
                    && synthetic_names.contains(r.expr_name.as_str())
            })
            .map(|r| r.regret_pct)
            .collect();
        let (lo, hi) = if synth_regret_at_1.is_empty() {
            (0.0, 0.0)
        } else {
            let mut s = synth_regret_at_1.clone();
            s.sort_by(|a, b| a.partial_cmp(b).unwrap());
            (s[0], s[s.len() - 1])
        };
        let uc_regret = all_rows
            .iter()
            .find(|r| {
                r.expr_name == item.name && r.curve == "unguided" && (r.frac - 1.0).abs() < 1e-9
            })
            .map(|r| r.regret_pct)
            .unwrap_or(f64::NAN);
        let oc_regret = all_rows
            .iter()
            .find(|r| {
                r.expr_name == item.name && r.curve == "oracle" && (r.frac - 1.0).abs() < 1e-9
            })
            .map(|r| r.regret_pct)
            .unwrap_or(f64::NAN);
        let in_sample = uc_regret >= lo && uc_regret <= hi;
        println!(
            "  {:<20} nodes={:<4} tier={:<10} unguided_regret@1.0={:>7.2}%  oracle_regret@1.0={:>7.2}%  \
             synthetic-{tier}-tier range=[{lo:.2}%,{hi:.2}%] in-sample={}",
            item.name, item.node_count, tier, uc_regret, oc_regret, in_sample,
        );
    }
}
