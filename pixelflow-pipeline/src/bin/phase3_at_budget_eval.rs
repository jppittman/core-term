//! The at-budget evaluation of the Phase 3 registered claim, on DEV.
//!
//! Implements `docs/plans/2026-08-31-guide-design-revision.md` §5 against the
//! committed registration `docs/plans/2026-09-01-phase3-registration.md`
//! (nothing in that document is revised here). Every arm of the ablation
//! ladder goes through the ONE anytime-curve definition
//! ([`pixelflow_search::egraph::anytime`]); the arms differ only in how the
//! e-graph is advanced between checkpoints:
//!
//! | arm | stepper |
//! |---|---|
//! | `unguided` | [`run_anytime_curve`] (fixed rule-then-class sweeps) — supplies both unguided-at-B and unguided-at-4B |
//! | `control` | [`GuidedSaturation`] over [`PerRuleRateGuide`] (per-rule TRAIN strict-positive rate, no candidate-local information) |
//! | `linear` | [`GuidedSaturation`] over [`LinearCandidateGuide`] (the trained cold-start Guide — the claim) |
//!
//! A strict-oracle arm (order candidates by their true strict label) is NOT
//! run: the strict label is defined on the `ApplicationId`s of one specific
//! run, and transporting it onto a differently-ordered guided run's candidates
//! needs a `(rule, class content at firing time) -> label` map the provenance
//! log does not record (`ApplicationRecord::match_root` is a class id that
//! union/rebuild renumbers). Building that map means instrumenting the
//! unguided sweep itself — not a cheap replay, so per the task spec it is
//! skipped and said so.
//!
//! # Metric (registration §2/§6, verbatim semantics)
//!
//! Costs are `CostModel::latency_prior()` `extract_dag` costs — deterministic,
//! no wall-clock anywhere in any reported number. Cost at budget B is read at
//! the grid checkpoint for B (the first between-rounds point with cumulative
//! recorded applications ≥ B; both curves report `app_actual`). Regret uses
//! the empirical best any arm reaches at any checkpoint of that expression.
//! Zero-cost references follow the established convention: a positive cost
//! against a zero-cost reference is infinite loss, never 0%.
//!
//! # Production-units context (integration audit, PR #1079)
//!
//! Production saturates with `config_for_node_count` (classical: 100 rounds /
//! 5,000 classes / 200 ms) and has no application counter or cap, so the
//! registered application budget is a proxy for a machine production does not
//! literally run. For every evaluated expression this binary ALSO runs the
//! exact production saturation step
//! ([`pixelflow_search::runtime::production_saturation_probe`] — the same
//! function body `optimize_runtime_arena` runs, not a copy) and records its
//! stop reason (read, not inferred), provenance application count at stop, and
//! latency-prior cost, then locates that cost on the unguided anytime curve.
//! Wall-clock is a stop condition of that call only; `Timeout` stops are
//! flagged machine-dependent and the host load average is recorded as
//! context, never as a metric.
//!
//! # Corpus discipline
//!
//! Reads `corpus_dev.bin` ONLY. TRAIN is not needed (the Guide was trained on
//! it); `corpus_final.bin` is never opened — FINAL stays untouched until a
//! publication run.
//!
//! # Resumability
//!
//! Per-expression results are appended to `--out-jsonl` as each expression
//! finishes; a re-run skips names already present. The aggregate report is
//! computed from the JSONL (`--aggregate-only` recomputes it without running
//! anything), so a crash mid-run loses one expression, not the run.
//!
//! Usage:
//! ```bash
//! cargo run --release -p pixelflow-pipeline --features training --bin phase3_at_budget_eval -- \
//!     --out-jsonl docs/results/2026-09-01-phase3-at-budget-eval.jsonl \
//!     --out-json docs/results/2026-09-01-phase3-at-budget-eval.json \
//!     --out-md docs/results/2026-09-01-phase3-at-budget-eval-report.md
//! ```

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::io::{BufRead, BufReader, Write as _};
use std::path::{Path, PathBuf};
use std::time::Duration;

use clap::Parser;
use serde::{Deserialize, Serialize};

use pixelflow_ir::{ExprArena, ExprId};
use pixelflow_pipeline::journal::append_record;
use pixelflow_pipeline::training::corpus::read_corpus;
use pixelflow_search::egraph::{
    APP_CHECKPOINT_GRID, AnytimeCurveOutput, ApplicationId, CostModel, EClassId, EGraph, ENode,
    ENodeId, EpisodeLabels, GuidedSaturation, Origin, SaturationStop, all_rules,
    config_for_node_count, run_anytime_curve, run_anytime_curve_with,
};
use pixelflow_search::nnue::factored::EMBED_DIM;
use pixelflow_search::nnue::guide::SaturationGuide;
use pixelflow_search::nnue::guide::linear::{LinearCandidateGuide, PerRuleRateGuide};
use pixelflow_search::runtime::production_saturation_probe;

#[derive(Parser)]
#[command(name = "phase3_at_budget_eval")]
#[command(about = "Phase 3 at-budget ablation ladder on DEV against the registered claim")]
struct Args {
    /// Directory holding `corpus_dev.bin` (only file read).
    #[arg(long, default_value = "pixelflow-pipeline/data")]
    corpus_dir: String,

    /// `train_guide` checkpoint for the linear Guide (the claim arm).
    #[arg(
        long,
        default_value = "pixelflow-pipeline/data/guide_checkpoint_strict_v1.json"
    )]
    checkpoint: String,

    /// `train_guide` report JSON — supplies the per-rule TRAIN rates the
    /// control arm is built from.
    #[arg(
        long,
        default_value = "docs/results/2026-09-01-train-guide-report.json"
    )]
    train_guide_report: String,

    /// DEV classical expressions to evaluate, size-stratified over all DEV
    /// classical expressions. 0 = all of them.
    #[arg(long, default_value_t = 0)]
    classical_samples: usize,

    /// DEV blitz and rapid expressions per band, reported for completeness
    /// only (no claim is registered on them). 0 = none.
    #[arg(long, default_value_t = 30)]
    other_samples: usize,

    /// Per-expression results, appended as produced (resume skips names
    /// already present).
    #[arg(
        long,
        default_value = "docs/results/2026-09-01-phase3-at-budget-eval.jsonl"
    )]
    out_jsonl: String,

    #[arg(
        long,
        default_value = "docs/results/2026-09-01-phase3-at-budget-eval.json"
    )]
    out_json: String,

    #[arg(
        long,
        default_value = "docs/results/2026-09-01-phase3-at-budget-eval-report.md"
    )]
    out_md: String,

    /// Recompute the aggregate report from the JSONL without evaluating.
    #[arg(long, default_value_t = false)]
    aggregate_only: bool,

    /// Comma-separated expression names to skip (documented in the report).
    #[arg(long, default_value = "")]
    skip_names: String,

    /// Journal to append the run record to.
    #[arg(long, default_value = "docs/results/journal.jsonl")]
    journal: String,

    /// Override the guided arms' checkpoint grid (comma-separated, strictly
    /// increasing, must contain every registered B and 4B). Sensitivity
    /// check only: a guided round's remaining scored survivors are dropped
    /// when a checkpoint's application target is reached, so a denser grid
    /// can only cost a guided arm, never help it.
    #[arg(long, default_value = "")]
    guided_grid: String,
}

// ---------------------------------------------------------------------------
// Registered constants (docs/plans/2026-09-01-phase3-registration.md §4–§6).
// Not adjustable after the fact; restated here so the report is self-checking.
// ---------------------------------------------------------------------------

/// Registered classical tiers: (B, Y as a fraction, median unguided
/// truncation loss L in percent).
const REGISTERED_TIERS: [(usize, f64, f64); 2] = [(100, 0.163, 48.47), (200, 0.090, 21.92)];

/// Guided arms are sampled through 4B of the secondary tier — every
/// registered quantity (B, 4B for both tiers) is on this grid, and running a
/// guided arm to quiescence would answer no registered question.
const GUIDED_GRID: &[usize] = &[25, 50, 100, 200, 400, 800];

/// Per-curve wall-clock safety ceiling — PANICS inside the shared curve runner
/// if it binds, never truncates, never appears in a number.
const SAFETY_TIMEOUT: Duration = Duration::from_secs(1800);
const SWEEP_SAFETY_CEILING: usize = 10_000;

/// The structural/congruence rule class the design revision (§2.1) names:
/// rules whose job is enabling congruence closure, scored 63–84% load-bearing
/// by the labeler bound and ~0% by the strict bound.
const STRUCTURAL_RULES: [&str; 6] = [
    "commutative",
    "associative",
    "reverse-associative",
    "distribute",
    "fma-fusion",
    "identity",
];

const ARM_NAMES: [&str; 3] = ["unguided", "control", "linear"];

fn tier_name(node_count: usize) -> &'static str {
    match node_count {
        0..=10 => "blitz",
        11..=50 => "rapid",
        _ => "classical",
    }
}

fn stop_name(stop: SaturationStop) -> &'static str {
    match stop {
        SaturationStop::Quiesced => "quiesced",
        SaturationStop::ApplicationBudget => "app_budget",
        SaturationStop::ClassCap => "class_cap",
        SaturationStop::IterationCeiling => "iteration_ceiling",
        SaturationStop::Timeout => "timeout",
    }
}

fn is_structural(name: &str) -> bool {
    STRUCTURAL_RULES.contains(&name)
}

// ---------------------------------------------------------------------------
// Per-expression record (one JSONL line).
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Clone)]
struct ArmCurve {
    grid: Vec<usize>,
    app_actual: Vec<usize>,
    cost: Vec<usize>,
    rounds: Vec<usize>,
    classes: Vec<usize>,
    clamped: Vec<bool>,
    ended: String,
    ended_at_apps: usize,
    /// Guided arms only: distinct candidate keys scored over the whole run
    /// (dedup coverage diagnostic).
    seen_keys: Option<usize>,
}

impl ArmCurve {
    fn idx(&self, b: usize) -> usize {
        self.grid
            .iter()
            .position(|&g| g == b)
            .unwrap_or_else(|| panic!("budget {b} is not on this arm's grid {:?}", self.grid))
    }
    fn cost_at(&self, b: usize) -> usize {
        self.cost[self.idx(b)]
    }
    fn apps_at(&self, b: usize) -> usize {
        self.app_actual[self.idx(b)]
    }
    fn rounds_at(&self, b: usize) -> usize {
        self.rounds[self.idx(b)]
    }
    fn best_cost(&self) -> usize {
        *self.cost.iter().min().expect("non-empty curve")
    }
}

/// What the first `applications` recorded applications of one arm's run
/// looked like — read off the arm's final provenance log, which is
/// append-only and ordered by `ApplicationId`.
#[derive(Serialize, Deserialize, Clone)]
struct AtBudgetDiag {
    applications: usize,
    structural: usize,
    /// Applications among the first `applications` whose output node is on
    /// the arm's OWN final extracted path (strict hindsight label).
    strict_positive: usize,
    rounds: usize,
    rule_hist: BTreeMap<String, usize>,
}

#[derive(Serialize, Deserialize, Clone)]
struct ProductionRow {
    node_count_reachable: usize,
    config_tier: String,
    max_iterations: usize,
    max_classes: usize,
    hard_timeout_ms: u64,
    stop: String,
    iterations: usize,
    total_unions: usize,
    applications: usize,
    classes_after: usize,
    cost: usize,
}

/// Enabler-starvation measurement (task 2): did the guided arms ever build the
/// classes that unguided's numeric strict-positives needed structural rules
/// to reach?
#[derive(Serialize, Deserialize, Clone, Default)]
struct EnablerDiag {
    unguided_strict_positive: usize,
    unguided_numeric_strict_positive: usize,
    /// Numeric strict-positives whose output node's tight derivation
    /// ancestry (excluding itself) contains a structural application.
    numeric_structurally_enabled: usize,
    /// Numeric strict-positives with a direct child class whose chosen node
    /// was created by a structural application.
    numeric_direct_child_structural: usize,
    /// Of `unguided_numeric_strict_positive`, how many output terms exist in
    /// each guided arm's final e-graph.
    numeric_present_in: BTreeMap<String, usize>,
    /// Of `numeric_structurally_enabled`, how many output terms exist in
    /// each guided arm's final e-graph.
    enabled_present_in: BTreeMap<String, usize>,
}

#[derive(Serialize, Deserialize, Clone)]
struct ExprRow {
    name: String,
    tier: String,
    node_count: usize,
    class_cap: usize,
    arms: BTreeMap<String, ArmCurve>,
    /// arm -> B -> diagnostics at B.
    at_budget: BTreeMap<String, BTreeMap<usize, AtBudgetDiag>>,
    enabler: EnablerDiag,
    production: Option<ProductionRow>,
}

// ---------------------------------------------------------------------------
// Running one expression.
// ---------------------------------------------------------------------------

fn curve_to_arm(out: &AnytimeCurveOutput, seen_keys: Option<usize>) -> ArmCurve {
    let c = &out.curve;
    ArmCurve {
        grid: c.checkpoints.iter().map(|k| k.app_target).collect(),
        app_actual: c.checkpoints.iter().map(|k| k.app_actual).collect(),
        cost: c.checkpoints.iter().map(|k| k.cost).collect(),
        rounds: c.checkpoints.iter().map(|k| k.sweeps).collect(),
        classes: c.checkpoints.iter().map(|k| k.classes).collect(),
        clamped: c.checkpoints.iter().map(|k| k.clamped).collect(),
        ended: stop_name(c.ended).to_string(),
        ended_at_apps: c.ended_at_apps,
        seen_keys,
    }
}

fn rule_name(egraph: &EGraph, idx: usize) -> String {
    egraph
        .rule(idx)
        .unwrap_or_else(|| panic!("rule_idx {idx} has no rule in this e-graph"))
        .name()
        .to_string()
}

/// Strict hindsight labels for a finished curve (its own final extraction).
fn strict_positives(out: &AnytimeCurveOutput) -> BTreeSet<ApplicationId> {
    let root = out.extraction.root;
    EpisodeLabels::compute_strict(&out.egraph, root, &out.extraction.choices).load_bearing
}

fn at_budget_diag(
    out: &AnytimeCurveOutput,
    arm: &ArmCurve,
    positives: &BTreeSet<ApplicationId>,
    b: usize,
) -> AtBudgetDiag {
    let applications = arm.apps_at(b);
    let mut structural = 0usize;
    let mut strict_positive = 0usize;
    let mut rule_hist: BTreeMap<String, usize> = BTreeMap::new();
    for (id, rec) in out.egraph.provenance().applications() {
        if id.as_u64() as usize >= applications {
            continue;
        }
        let name = rule_name(&out.egraph, rec.rule_idx);
        if is_structural(&name) {
            structural += 1;
        }
        if positives.contains(&id) {
            strict_positive += 1;
        }
        *rule_hist.entry(name).or_default() += 1;
    }
    AtBudgetDiag {
        applications,
        structural,
        strict_positive,
        rounds: arm.rounds_at(b),
        rule_hist,
    }
}

/// A concrete term read off one e-graph's chosen extraction, so it can be
/// looked up in ANOTHER e-graph (class ids do not transfer; structure does).
enum Term {
    Leaf(ENode),
    Op {
        op: &'static dyn pixelflow_search::egraph::Op,
        children: Vec<Term>,
    },
}

fn term_of_chosen(
    egraph: &EGraph,
    choices: &[Option<usize>],
    class: EClassId,
    node_idx: usize,
) -> Term {
    let node = &egraph.nodes(class)[node_idx];
    match node {
        ENode::Op { op, children } => Term::Op {
            op: *op,
            children: children
                .iter()
                .map(|&child| {
                    let c = egraph.find(child);
                    let idx = choices.get(c.index()).and_then(|o| *o).unwrap_or_else(|| {
                        panic!(
                            "term_of_chosen: child class {} of a chosen node has no extraction choice",
                            c.index()
                        )
                    });
                    term_of_chosen(egraph, choices, c, idx)
                })
                .collect(),
        },
        leaf => Term::Leaf(leaf.clone()),
    }
}

/// All `(class, node)` pairs of an e-graph, canonical classes only.
struct NodeIndex<'a> {
    egraph: &'a EGraph,
    entries: Vec<(EClassId, &'a ENode)>,
}

impl<'a> NodeIndex<'a> {
    fn new(egraph: &'a EGraph) -> Self {
        let mut entries = Vec::new();
        for c in egraph.class_ids() {
            for n in egraph.nodes(c) {
                entries.push((c, n));
            }
        }
        Self { egraph, entries }
    }

    /// The canonical class containing `term`, if this e-graph has built it.
    fn lookup(&self, term: &Term) -> Option<EClassId> {
        match term {
            Term::Leaf(leaf) => self
                .entries
                .iter()
                .find(|(_, n)| *n == leaf)
                .map(|(c, _)| *c),
            Term::Op { op, children } => {
                let mut child_classes = Vec::with_capacity(children.len());
                for ch in children {
                    child_classes.push(self.lookup(ch)?);
                }
                let kind = op.kind();
                self.entries
                    .iter()
                    .find(|(_, n)| match n {
                        ENode::Op {
                            op: o,
                            children: cs,
                        } => {
                            o.kind() == kind
                                && cs.len() == child_classes.len()
                                && cs
                                    .iter()
                                    .zip(&child_classes)
                                    .all(|(a, b)| self.egraph.find(*a) == *b)
                        }
                        _ => false,
                    })
                    .map(|(c, _)| *c)
            }
        }
    }
}

fn enabler_diag(
    unguided: &AnytimeCurveOutput,
    unguided_positives: &BTreeSet<ApplicationId>,
    guided: &[(&str, &AnytimeCurveOutput)],
) -> EnablerDiag {
    let eg = &unguided.egraph;
    let prov = eg.provenance();
    let choices = &unguided.extraction.choices;

    // Output nodes of every application (a rule RHS may create several),
    // each tag's class, and the set of tags the extraction chose. A strict
    // positive is an application at least one of whose outputs is chosen;
    // the enabler analysis looks at exactly those chosen outputs.
    let mut outputs_of: HashMap<ApplicationId, Vec<ENodeId>> = HashMap::new();
    for (tag, origin) in prov.origins() {
        if let Origin::Rule(app) = origin {
            outputs_of.entry(app).or_default().push(tag);
        }
    }
    let mut class_of_tag: HashMap<ENodeId, (EClassId, usize)> = HashMap::new();
    for c in eg.class_ids() {
        for (i, tag) in eg.tags(c).iter().enumerate() {
            class_of_tag.insert(*tag, (c, i));
        }
    }
    let chosen: HashSet<ENodeId> = {
        let mut seen: HashSet<EClassId> = HashSet::new();
        let mut stack = vec![unguided.extraction.root];
        let mut out = HashSet::new();
        while let Some(c) = stack.pop() {
            let c = eg.find(c);
            if !seen.insert(c) {
                continue;
            }
            let idx = choices[c.index()].unwrap_or_else(|| {
                panic!(
                    "class {} reachable via the chosen extraction has no choice",
                    c.index()
                )
            });
            out.insert(eg.tags(c)[idx]);
            stack.extend(eg.nodes(c)[idx].children());
        }
        out
    };
    let structural_app = |app: ApplicationId| -> bool {
        let rec = prov
            .application(app)
            .unwrap_or_else(|| panic!("no record for {app:?}"));
        is_structural(&rule_name(eg, rec.rule_idx))
    };

    let indexes: Vec<(&str, NodeIndex<'_>)> = guided
        .iter()
        .map(|(name, out)| (*name, NodeIndex::new(&out.egraph)))
        .collect();

    let mut d = EnablerDiag::default();
    for (name, _) in guided {
        d.numeric_present_in.insert(name.to_string(), 0);
        d.enabled_present_in.insert(name.to_string(), 0);
    }
    d.unguided_strict_positive = unguided_positives.len();

    for &app in unguided_positives {
        if structural_app(app) {
            continue;
        }
        d.unguided_numeric_strict_positive += 1;
        let chosen_outputs: Vec<ENodeId> = outputs_of
            .get(&app)
            .unwrap_or_else(|| panic!("strict-positive {app:?} has no output node"))
            .iter()
            .copied()
            .filter(|t| chosen.contains(t))
            .collect();
        assert!(
            !chosen_outputs.is_empty(),
            "strict-positive {app:?} has no chosen output node — label/extraction disagree"
        );
        // Credit the application once; "enabled" / "present" if ANY of its
        // chosen outputs is.
        let mut enabled = false;
        let mut direct = false;
        let mut terms = Vec::new();
        for tag in chosen_outputs {
            let (class, node_idx) = *class_of_tag.get(&tag).unwrap_or_else(|| {
                panic!("output node {tag:?} of {app:?} is in no canonical class")
            });
            let mut ancestry = eg.derivation_ancestors_tight(&[(class, tag)]);
            ancestry.remove(&app);
            enabled |= ancestry.iter().any(|&a| structural_app(a));
            let node = &eg.nodes(class)[node_idx];
            direct |= node.children().iter().any(|&child| {
                let c = eg.find(child);
                let idx = choices[c.index()].expect("chosen child has a choice");
                matches!(prov.origin(eg.tags(c)[idx]), Some(Origin::Rule(a)) if structural_app(a))
            });
            terms.push(term_of_chosen(eg, choices, class, node_idx));
        }
        if enabled {
            d.numeric_structurally_enabled += 1;
        }
        if direct {
            d.numeric_direct_child_structural += 1;
        }
        for (name, index) in &indexes {
            if terms.iter().any(|t| index.lookup(t).is_some()) {
                *d.numeric_present_in.get_mut(*name).expect("arm registered") += 1;
                if enabled {
                    *d.enabled_present_in.get_mut(*name).expect("arm registered") += 1;
                }
            }
        }
    }
    d
}

struct Guides {
    control: PerRuleRateGuide,
    linear: LinearCandidateGuide,
    embeds: Vec<[f32; EMBED_DIM]>,
}

/// One expression's fixed curve environment, shared by every arm.
struct CurveInput<'a> {
    arena: &'a ExprArena,
    root: ExprId,
    class_cap: usize,
    costs: &'a CostModel,
    guided_grid: &'a [usize],
}

fn run_guided<G: SaturationGuide>(
    guide: &G,
    embeds: &[[f32; EMBED_DIM]],
    input: &CurveInput<'_>,
) -> (AnytimeCurveOutput, usize) {
    let mut stepper = GuidedSaturation::new(guide, embeds);
    let out = run_anytime_curve_with(
        input.arena,
        input.root,
        all_rules(),
        input.guided_grid,
        input.class_cap,
        SWEEP_SAFETY_CEILING,
        SAFETY_TIMEOUT,
        input.costs,
        &mut stepper,
    );
    let seen = stepper.seen_key_count();
    (out, seen)
}

fn evaluate_expression(
    name: &str,
    arena: &ExprArena,
    root: ExprId,
    guides: &Guides,
    costs: &CostModel,
    guided_grid: &[usize],
) -> ExprRow {
    let node_count = arena.nodes_raw().len();
    let class_cap = config_for_node_count(node_count).max_classes;

    let unguided = run_anytime_curve(
        arena,
        root,
        all_rules(),
        APP_CHECKPOINT_GRID,
        class_cap,
        SWEEP_SAFETY_CEILING,
        SAFETY_TIMEOUT,
        costs,
    );
    let input = CurveInput {
        arena,
        root,
        class_cap,
        costs,
        guided_grid,
    };
    let (control, control_seen) = run_guided(&guides.control, &guides.embeds, &input);
    let (linear, linear_seen) = run_guided(&guides.linear, &guides.embeds, &input);

    let outs: [(&str, &AnytimeCurveOutput, Option<usize>); 3] = [
        ("unguided", &unguided, None),
        ("control", &control, Some(control_seen)),
        ("linear", &linear, Some(linear_seen)),
    ];

    let mut arms = BTreeMap::new();
    let mut at_budget = BTreeMap::new();
    let mut unguided_positives = BTreeSet::new();
    for (arm_name, out, seen) in outs {
        let arm = curve_to_arm(out, seen);
        let positives = strict_positives(out);
        let mut per_b = BTreeMap::new();
        for (b, _, _) in REGISTERED_TIERS {
            per_b.insert(b, at_budget_diag(out, &arm, &positives, b));
        }
        if arm_name == "unguided" {
            unguided_positives = positives;
        }
        at_budget.insert(arm_name.to_string(), per_b);
        arms.insert(arm_name.to_string(), arm);
    }

    let enabler = enabler_diag(
        &unguided,
        &unguided_positives,
        &[("control", &control), ("linear", &linear)],
    );

    let production = production_saturation_probe(arena, root).map(|p| ProductionRow {
        node_count_reachable: p.node_count,
        config_tier: match p.config.max_iterations {
            20 => "blitz",
            50 => "rapid",
            100 => "classical",
            other => panic!("unknown production config tier: {other} iterations"),
        }
        .to_string(),
        max_iterations: p.config.max_iterations,
        max_classes: p.config.max_classes,
        hard_timeout_ms: p.config.hard_timeout.as_millis() as u64,
        stop: stop_name(p.stop).to_string(),
        iterations: p.iterations,
        total_unions: p.total_unions,
        applications: p.applications,
        classes_after: p.classes_after,
        cost: p.cost,
    });

    ExprRow {
        name: name.to_string(),
        tier: tier_name(node_count).to_string(),
        node_count,
        class_cap,
        arms,
        at_budget,
        enabler,
        production,
    }
}

// ---------------------------------------------------------------------------
// Aggregation.
// ---------------------------------------------------------------------------

fn percentile(sorted: &[f64], p: f64) -> f64 {
    assert!(!sorted.is_empty(), "percentile of empty slice");
    let pos = p * (sorted.len() as f64 - 1.0);
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let frac = pos - lo as f64;
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

#[derive(Serialize, Clone)]
struct Dist {
    n: usize,
    q1: f64,
    median: f64,
    q3: f64,
    p90: f64,
    max: f64,
    inf_count: usize,
}

fn dist(values: &[f64]) -> Dist {
    let mut v: Vec<f64> = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).expect("NaN in distribution"));
    if v.is_empty() {
        return Dist {
            n: 0,
            q1: f64::NAN,
            median: f64::NAN,
            q3: f64::NAN,
            p90: f64::NAN,
            max: f64::NAN,
            inf_count: 0,
        };
    }
    Dist {
        n: v.len(),
        q1: percentile(&v, 0.25),
        median: percentile(&v, 0.5),
        q3: percentile(&v, 0.75),
        p90: percentile(&v, 0.9),
        max: *v.last().expect("non-empty"),
        inf_count: v.iter().filter(|x| x.is_infinite()).count(),
    }
}

/// `a / b` with the zero-reference convention.
fn ratio(a: usize, b: usize) -> f64 {
    if b == 0 {
        if a == 0 { 1.0 } else { f64::INFINITY }
    } else {
        a as f64 / b as f64
    }
}

/// `(a - r) / r * 100` with the zero-reference convention.
fn pct_over(a: usize, r: usize) -> f64 {
    if r == 0 {
        if a == 0 { 0.0 } else { f64::INFINITY }
    } else {
        (a as f64 - r as f64) / r as f64 * 100.0
    }
}

#[derive(Serialize, Clone)]
struct ArmAtB {
    arm: String,
    /// `arm_cost@B / unguided_cost@B`, per expression.
    ratio_vs_unguided_at_b: Dist,
    improved: usize,
    unchanged: usize,
    worse: usize,
    /// `(arm_cost@B - unguided_cost@4B) / unguided_cost@4B` in percent.
    gap_vs_unguided_at_4b_pct: Dist,
    /// `(arm_cost@B - best) / best` in percent, best = empirical best of all
    /// arms at any checkpoint.
    regret_pct: Dist,
    /// Share of the unguided truncation gap `(u@B - u@4B)` the arm closed, on
    /// expressions with a positive gap.
    gap_closed_frac: Dist,
    /// Diagnostics at B: pooled structural share of applications, pooled
    /// strict-positive share (precision), median rounds to reach B.
    structural_share_pooled: f64,
    strict_precision_pooled: f64,
    rounds_to_b: Dist,
    app_actual_at_b: Dist,
    /// Guided runs that had already quiesced (dedup exhaustion) before
    /// reaching B applications — for them "at B" is "at quiescence".
    ended_before_b: usize,
    /// Expressions where this arm's curve reaches the empirical best cost
    /// somewhere along it.
    reaches_empirical_best: usize,
    /// This arm's cost@B against the production call's cost.
    vs_production_better: usize,
    vs_production_equal: usize,
    vs_production_worse: usize,
    /// Applications when the guided run ended (quiesced or grid end).
    ended_at_apps: Dist,
    /// `(seen_keys - recorded applications) / seen_keys` on quiesced runs:
    /// candidate keys the loop marked seen but whose `apply_single_rule`
    /// recorded nothing (stale `node_idx` after an earlier application in
    /// the same round rebuilt the class) — burned, never retried.
    burned_key_share: Dist,
}

#[derive(Serialize, Clone)]
struct TierResult {
    band: String,
    b: usize,
    n: usize,
    y_registered: f64,
    l_registered_pct: f64,
    /// Registered Y-clause threshold on the median ratio.
    ratio_threshold: f64,
    /// Registered 4B-approach threshold on the median gap (L/2, percent).
    gap_threshold_pct: f64,
    unguided_regret_at_b_pct: Dist,
    unguided_regret_at_4b_pct: Dist,
    unguided_truncation_loss_pct: Dist,
    arms: Vec<ArmAtB>,
    /// Per-arm verdicts against the registered claim (classical only is
    /// binding; other bands carry the same fields with no claim).
    y_clause_holds: BTreeMap<String, bool>,
    approach_clause_holds: BTreeMap<String, bool>,
    beats_unguided_at_all: BTreeMap<String, bool>,
    linear_beats_control: bool,
    unguided_structural_share_pooled: f64,
    unguided_strict_precision_pooled: f64,
    /// Head-to-head at B: expressions where linear's cost is below / equal
    /// to / above control's.
    linear_lt_control: usize,
    linear_eq_control: usize,
    linear_gt_control: usize,
}

#[derive(Serialize, Clone)]
struct ProductionSummary {
    band: String,
    n: usize,
    n_probe_none: usize,
    stops: BTreeMap<String, usize>,
    effective_b: Dist,
    share_apps_ge: BTreeMap<usize, f64>,
    /// Production cost relative to the unguided curve.
    ratio_vs_unguided_at_100: Dist,
    ratio_vs_unguided_at_200: Dist,
    ratio_vs_unguided_at_4b_800: Dist,
    regret_vs_best_pct: Dist,
    /// Smallest unguided grid checkpoint whose cost <= production's cost
    /// (`0` = production is worse than every unguided checkpoint; recorded
    /// as the grid target).
    equivalent_checkpoint_hist: BTreeMap<usize, usize>,
    /// Unguided grid checkpoint whose app_actual first reaches production's
    /// application count.
    app_bracket_hist: BTreeMap<usize, usize>,
    /// Rounds production ran vs its own cap.
    iterations: Dist,
    classes_after: Dist,
}

#[derive(Serialize, Clone)]
struct EnablerSummary {
    band: String,
    n: usize,
    unguided_strict_positive_total: usize,
    numeric_total: usize,
    numeric_structurally_enabled: usize,
    numeric_direct_child_structural: usize,
    numeric_present_in: BTreeMap<String, usize>,
    enabled_present_in: BTreeMap<String, usize>,
    /// Per-arm pooled structural share at B=100 and B=200.
    structural_share_at_b: BTreeMap<String, BTreeMap<usize, f64>>,
    /// Per-arm pooled top rules at B=100.
    top_rules_at_100: BTreeMap<String, Vec<(String, usize)>>,
    /// Guided arms: seen candidate keys / applications at end of run.
    seen_keys_per_application: BTreeMap<String, Dist>,
}

#[derive(Serialize)]
struct Report {
    n_rows: usize,
    n_by_band: BTreeMap<String, usize>,
    skipped_names: Vec<String>,
    tiers: Vec<TierResult>,
    production: Vec<ProductionSummary>,
    enabler: Vec<EnablerSummary>,
    context: BTreeMap<String, String>,
}

fn tier_result(rows: &[&ExprRow], band: &str, b: usize, y: f64, l: f64) -> TierResult {
    let b4 = 4 * b;
    fn ung(r: &ExprRow) -> &ArmCurve {
        &r.arms["unguided"]
    }
    let best = |r: &ExprRow| {
        r.arms
            .values()
            .map(ArmCurve::best_cost)
            .min()
            .expect("arms")
    };

    let unguided_regret_at_b: Vec<f64> = rows
        .iter()
        .map(|r| pct_over(ung(r).cost_at(b), best(r)))
        .collect();
    let unguided_regret_at_4b: Vec<f64> = rows
        .iter()
        .map(|r| pct_over(ung(r).cost_at(b4), best(r)))
        .collect();
    let trunc: Vec<f64> = rows
        .iter()
        .map(|r| pct_over(ung(r).cost_at(b), ung(r).cost_at(b4)))
        .collect();

    let mut arms = Vec::new();
    for arm_name in ["control", "linear"] {
        let mut ratios = Vec::new();
        let mut gaps = Vec::new();
        let mut regrets = Vec::new();
        let mut closed = Vec::new();
        let (mut imp, mut unch, mut worse) = (0, 0, 0);
        let (mut apps, mut structural, mut positive) = (0usize, 0usize, 0usize);
        let mut rounds = Vec::new();
        let mut app_actual = Vec::new();
        let mut ended_before_b = 0usize;
        let mut reaches_best = 0usize;
        let (mut pb, mut pe, mut pw) = (0usize, 0usize, 0usize);
        let mut ended_at = Vec::new();
        let mut burned = Vec::new();
        for r in rows {
            let a = &r.arms[arm_name];
            if a.ended != "app_budget" && a.ended_at_apps < b {
                ended_before_b += 1;
            }
            if a.best_cost() == best(r) {
                reaches_best += 1;
            }
            if let Some(p) = &r.production {
                match a.cost_at(b).cmp(&p.cost) {
                    std::cmp::Ordering::Less => pb += 1,
                    std::cmp::Ordering::Equal => pe += 1,
                    std::cmp::Ordering::Greater => pw += 1,
                }
            }
            ended_at.push(a.ended_at_apps as f64);
            if a.ended == "quiesced" {
                let seen = a.seen_keys.expect("guided arm records seen_keys");
                assert!(
                    seen >= a.ended_at_apps,
                    "{}: {} recorded more applications ({}) than candidate keys scored ({seen})",
                    r.name,
                    arm_name,
                    a.ended_at_apps
                );
                burned.push((seen - a.ended_at_apps) as f64 / seen.max(1) as f64);
            }
            let ca = a.cost_at(b);
            let cu = ung(r).cost_at(b);
            let cu4 = ung(r).cost_at(b4);
            let rt = ratio(ca, cu);
            ratios.push(rt);
            match ca.cmp(&cu) {
                std::cmp::Ordering::Less => imp += 1,
                std::cmp::Ordering::Equal => unch += 1,
                std::cmp::Ordering::Greater => worse += 1,
            }
            gaps.push(pct_over(ca, cu4));
            regrets.push(pct_over(ca, best(r)));
            if cu > cu4 {
                closed.push((cu as f64 - ca as f64) / (cu as f64 - cu4 as f64));
            }
            let d = &r.at_budget[arm_name][&b];
            apps += d.applications;
            structural += d.structural;
            positive += d.strict_positive;
            rounds.push(d.rounds as f64);
            app_actual.push(a.apps_at(b) as f64);
        }
        arms.push(ArmAtB {
            arm: arm_name.to_string(),
            ratio_vs_unguided_at_b: dist(&ratios),
            improved: imp,
            unchanged: unch,
            worse,
            gap_vs_unguided_at_4b_pct: dist(&gaps),
            regret_pct: dist(&regrets),
            gap_closed_frac: dist(&closed),
            structural_share_pooled: if apps == 0 {
                0.0
            } else {
                structural as f64 / apps as f64
            },
            strict_precision_pooled: if apps == 0 {
                0.0
            } else {
                positive as f64 / apps as f64
            },
            rounds_to_b: dist(&rounds),
            app_actual_at_b: dist(&app_actual),
            ended_before_b,
            reaches_empirical_best: reaches_best,
            vs_production_better: pb,
            vs_production_equal: pe,
            vs_production_worse: pw,
            ended_at_apps: dist(&ended_at),
            burned_key_share: dist(&burned),
        });
    }

    let ratio_threshold = 1.0 - y;
    let gap_threshold_pct = l / 2.0;
    let mut y_clause = BTreeMap::new();
    let mut approach = BTreeMap::new();
    let mut beats = BTreeMap::new();
    for a in &arms {
        y_clause.insert(
            a.arm.clone(),
            a.ratio_vs_unguided_at_b.median <= ratio_threshold,
        );
        approach.insert(
            a.arm.clone(),
            a.gap_vs_unguided_at_4b_pct.median <= gap_threshold_pct,
        );
        beats.insert(a.arm.clone(), a.ratio_vs_unguided_at_b.median < 1.0);
    }
    let linear_beats_control = arms[1].regret_pct.median < arms[0].regret_pct.median;
    let (mut u_apps, mut u_struct, mut u_pos) = (0usize, 0usize, 0usize);
    let (mut lt, mut eq, mut gt) = (0usize, 0usize, 0usize);
    for r in rows {
        let d = &r.at_budget["unguided"][&b];
        u_apps += d.applications;
        u_struct += d.structural;
        u_pos += d.strict_positive;
        match r.arms["linear"]
            .cost_at(b)
            .cmp(&r.arms["control"].cost_at(b))
        {
            std::cmp::Ordering::Less => lt += 1,
            std::cmp::Ordering::Equal => eq += 1,
            std::cmp::Ordering::Greater => gt += 1,
        }
    }

    TierResult {
        band: band.to_string(),
        b,
        n: rows.len(),
        y_registered: y,
        l_registered_pct: l,
        ratio_threshold,
        gap_threshold_pct,
        unguided_regret_at_b_pct: dist(&unguided_regret_at_b),
        unguided_regret_at_4b_pct: dist(&unguided_regret_at_4b),
        unguided_truncation_loss_pct: dist(&trunc),
        arms,
        y_clause_holds: y_clause,
        approach_clause_holds: approach,
        beats_unguided_at_all: beats,
        linear_beats_control,
        unguided_structural_share_pooled: if u_apps == 0 {
            0.0
        } else {
            u_struct as f64 / u_apps as f64
        },
        unguided_strict_precision_pooled: if u_apps == 0 {
            0.0
        } else {
            u_pos as f64 / u_apps as f64
        },
        linear_lt_control: lt,
        linear_eq_control: eq,
        linear_gt_control: gt,
    }
}

fn production_summary(rows: &[&ExprRow], band: &str) -> ProductionSummary {
    let with: Vec<(&ExprRow, &ProductionRow)> = rows
        .iter()
        .filter_map(|r| r.production.as_ref().map(|p| (*r, p)))
        .collect();
    let mut stops: BTreeMap<String, usize> = BTreeMap::new();
    let mut eff = Vec::new();
    let mut r100 = Vec::new();
    let mut r200 = Vec::new();
    let mut r800 = Vec::new();
    let mut regret = Vec::new();
    let mut eq_hist: BTreeMap<usize, usize> = BTreeMap::new();
    let mut br_hist: BTreeMap<usize, usize> = BTreeMap::new();
    let mut iters = Vec::new();
    let mut classes = Vec::new();
    for (r, p) in &with {
        *stops.entry(p.stop.clone()).or_default() += 1;
        eff.push(p.applications as f64);
        let u = &r.arms["unguided"];
        r100.push(ratio(p.cost, u.cost_at(100)));
        r200.push(ratio(p.cost, u.cost_at(200)));
        r800.push(ratio(p.cost, u.cost_at(800)));
        let best = r
            .arms
            .values()
            .map(ArmCurve::best_cost)
            .min()
            .expect("arms");
        regret.push(pct_over(p.cost, best.min(p.cost)));
        let eq = u
            .grid
            .iter()
            .zip(&u.cost)
            .find(|(_, c)| **c <= p.cost)
            .map(|(&g, _)| g)
            .unwrap_or(0);
        *eq_hist.entry(eq).or_default() += 1;
        let br = u
            .grid
            .iter()
            .zip(&u.app_actual)
            .find(|(_, a)| **a >= p.applications)
            .map(|(&g, _)| g)
            .unwrap_or(usize::MAX);
        *br_hist.entry(br).or_default() += 1;
        iters.push(p.iterations as f64);
        classes.push(p.classes_after as f64);
    }
    let mut share = BTreeMap::new();
    for t in [100usize, 200, 400, 800, 1600] {
        let c = with.iter().filter(|(_, p)| p.applications >= t).count();
        share.insert(
            t,
            if with.is_empty() {
                0.0
            } else {
                c as f64 / with.len() as f64
            },
        );
    }
    ProductionSummary {
        band: band.to_string(),
        n: with.len(),
        n_probe_none: rows.len() - with.len(),
        stops,
        effective_b: dist(&eff),
        share_apps_ge: share,
        ratio_vs_unguided_at_100: dist(&r100),
        ratio_vs_unguided_at_200: dist(&r200),
        ratio_vs_unguided_at_4b_800: dist(&r800),
        regret_vs_best_pct: dist(&regret),
        equivalent_checkpoint_hist: eq_hist,
        app_bracket_hist: br_hist,
        iterations: dist(&iters),
        classes_after: dist(&classes),
    }
}

fn enabler_summary(rows: &[&ExprRow], band: &str) -> EnablerSummary {
    let mut s = EnablerSummary {
        band: band.to_string(),
        n: rows.len(),
        unguided_strict_positive_total: 0,
        numeric_total: 0,
        numeric_structurally_enabled: 0,
        numeric_direct_child_structural: 0,
        numeric_present_in: BTreeMap::new(),
        enabled_present_in: BTreeMap::new(),
        structural_share_at_b: BTreeMap::new(),
        top_rules_at_100: BTreeMap::new(),
        seen_keys_per_application: BTreeMap::new(),
    };
    let mut hist: BTreeMap<String, BTreeMap<String, usize>> = BTreeMap::new();
    let mut pooled: BTreeMap<String, BTreeMap<usize, (usize, usize)>> = BTreeMap::new();
    let mut seen: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for r in rows {
        s.unguided_strict_positive_total += r.enabler.unguided_strict_positive;
        s.numeric_total += r.enabler.unguided_numeric_strict_positive;
        s.numeric_structurally_enabled += r.enabler.numeric_structurally_enabled;
        s.numeric_direct_child_structural += r.enabler.numeric_direct_child_structural;
        for (k, v) in &r.enabler.numeric_present_in {
            *s.numeric_present_in.entry(k.clone()).or_default() += v;
        }
        for (k, v) in &r.enabler.enabled_present_in {
            *s.enabled_present_in.entry(k.clone()).or_default() += v;
        }
        for (arm, per_b) in &r.at_budget {
            for (b, d) in per_b {
                let e = pooled
                    .entry(arm.clone())
                    .or_default()
                    .entry(*b)
                    .or_default();
                e.0 += d.structural;
                e.1 += d.applications;
                if *b == 100 {
                    let h = hist.entry(arm.clone()).or_default();
                    for (rule, n) in &d.rule_hist {
                        *h.entry(rule.clone()).or_default() += n;
                    }
                }
            }
        }
        for (arm, a) in &r.arms {
            if let Some(k) = a.seen_keys {
                seen.entry(arm.clone())
                    .or_default()
                    .push(k as f64 / a.ended_at_apps.max(1) as f64);
            }
        }
    }
    for (arm, per_b) in pooled {
        let m: BTreeMap<usize, f64> = per_b
            .into_iter()
            .map(|(b, (st, ap))| (b, if ap == 0 { 0.0 } else { st as f64 / ap as f64 }))
            .collect();
        s.structural_share_at_b.insert(arm, m);
    }
    for (arm, h) in hist {
        let mut v: Vec<(String, usize)> = h.into_iter().collect();
        v.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        v.truncate(8);
        s.top_rules_at_100.insert(arm, v);
    }
    for (arm, v) in seen {
        s.seen_keys_per_application.insert(arm, dist(&v));
    }
    s
}

fn fmt_d(d: &Dist) -> String {
    if d.n == 0 {
        return "n=0".to_string();
    }
    format!(
        "{:.3} / {:.3} / {:.3} (p90 {:.3}{})",
        d.q1,
        d.median,
        d.q3,
        d.p90,
        if d.inf_count > 0 {
            format!(", {} inf", d.inf_count)
        } else {
            String::new()
        }
    )
}

fn fmt_pct(d: &Dist) -> String {
    if d.n == 0 {
        return "n=0".to_string();
    }
    format!(
        "{:.2}% / {:.2}% / {:.2}% (p90 {:.1}%{})",
        d.q1,
        d.median,
        d.q3,
        d.p90,
        if d.inf_count > 0 {
            format!(", {} inf", d.inf_count)
        } else {
            String::new()
        }
    )
}

fn write_markdown(report: &Report, path: &Path, jsonl: &str) {
    let mut md = String::new();
    md.push_str(
        "# Phase 3 at-budget evaluation on DEV (ablation ladder against the registered claim)\n\n",
    );
    md.push_str("**Date:** 2026-09-01 · **Registration:** `docs/plans/2026-09-01-phase3-registration.md` (unrevised) · **Authority:** `docs/plans/2026-08-31-guide-design-revision.md` §5\n\n");
    md.push_str(&format!(
        "Per-expression rows: `{jsonl}` ({} expressions: {}). Skipped: {}.\n\n",
        report.n_rows,
        report
            .n_by_band
            .iter()
            .map(|(k, v)| format!("{k} {v}"))
            .collect::<Vec<_>>()
            .join(", "),
        if report.skipped_names.is_empty() {
            "none".to_string()
        } else {
            report.skipped_names.join(", ")
        }
    ));
    md.push_str("Cost = `CostModel::latency_prior()` `extract_dag` cost, deterministic; no wall-clock in any number below. \
All arms go through `egraph::anytime::run_anytime_curve_with`; guided arms via `GuidedSaturation` (dedup set carried across checkpoints). \
Distributions are q1 / median / q3 (p90). Regret reference = empirical best of ALL arms at ANY checkpoint for that expression. \
Strict-oracle arm: not run (see the harness module doc — no cheap provenance replay exists for it).\n\n");

    for t in &report.tiers {
        let binding = t.band == "classical";
        md.push_str(&format!(
            "## {} band, B = {} ({})\n\n",
            t.band,
            t.b,
            if binding {
                format!(
                    "REGISTERED: Y = {:.1}% → median ratio ≤ {:.3}; 4B-approach: median gap ≤ {:.1}%",
                    t.y_registered * 100.0,
                    t.ratio_threshold,
                    t.gap_threshold_pct
                )
            } else {
                "reported for completeness, NO claim registered".to_string()
            }
        ));
        md.push_str(&format!("n = {}.\n\n", t.n));
        md.push_str("| arm | ratio vs unguided@B (q1/med/q3, p90) | improved / unchanged / worse | gap vs unguided@4B | regret vs best | gap closed (n with gap) | structural share @B | strict precision @B | rounds to B (med) |\n|---|---|---|---|---|---|---|---|---|\n");
        md.push_str(&format!(
            "| (a) unguided @B | 1.000 (by definition) | — | {} | {} | 0 | {:.3} | {:.4} | — |\n",
            fmt_pct(&t.unguided_truncation_loss_pct),
            fmt_pct(&t.unguided_regret_at_b_pct),
            t.unguided_structural_share_pooled,
            t.unguided_strict_precision_pooled,
        ));
        md.push_str(&format!(
            "| (b) unguided @4B | — | — | 0 | {} | 1 | — | — | — |\n",
            fmt_pct(&t.unguided_regret_at_4b_pct)
        ));
        for a in &t.arms {
            let label = match a.arm.as_str() {
                "control" => "(c) PerRuleRateGuide @B [control]",
                "linear" => "(d) LinearCandidateGuide @B [claim]",
                other => other,
            };
            md.push_str(&format!(
                "| {label} | {} | {} / {} / {} | {} | {} | {} (n={}) | {:.3} | {:.4} | {:.0} |\n",
                fmt_d(&a.ratio_vs_unguided_at_b),
                a.improved,
                a.unchanged,
                a.worse,
                fmt_pct(&a.gap_vs_unguided_at_4b_pct),
                fmt_pct(&a.regret_pct),
                fmt_d(&a.gap_closed_frac),
                a.gap_closed_frac.n,
                a.structural_share_pooled,
                a.strict_precision_pooled,
                a.rounds_to_b.median,
            ));
        }
        md.push('\n');
        md.push_str(&format!(
            "Ladder diagnostics: linear < control on {} expressions, equal on {}, linear > control on {}. ",
            t.linear_lt_control, t.linear_eq_control, t.linear_gt_control
        ));
        for a in &t.arms {
            md.push_str(&format!(
                "{}: quiesced before B on {} / {} (ended at {} applications; burned-key share {}); reaches the empirical best on {}; vs production cost better / equal / worse = {} / {} / {}. ",
                a.arm,
                a.ended_before_b,
                t.n,
                fmt_d(&a.ended_at_apps),
                fmt_d(&a.burned_key_share),
                a.reaches_empirical_best,
                a.vs_production_better,
                a.vs_production_equal,
                a.vs_production_worse,
            ));
        }
        md.push_str("\n\n");
        if binding {
            let lin = &t.arms[1];
            let ctl = &t.arms[0];
            md.push_str(&format!(
                "**Verdict (B={}):** (d) LinearCandidateGuide median ratio vs unguided-at-B = **{:.3}** against the registered threshold ≤ {:.3} (Y = {:.1}%) — the Y-clause **{}**; median gap vs unguided-at-4B = {:.2}% against ≤ {:.1}% — the 4B-approach clause **{}**; (d) {} unguided-at-B at all (median ratio {} 1.0, kill-gate view). Ladder: (c) control median ratio {:.3}, regret {:.2}% vs (d) regret {:.2}% — (d) **{}** (c).\n\n",
                t.b,
                lin.ratio_vs_unguided_at_b.median,
                t.ratio_threshold,
                t.y_registered * 100.0,
                if t.y_clause_holds["linear"] { "HOLDS" } else { "FAILS" },
                lin.gap_vs_unguided_at_4b_pct.median,
                t.gap_threshold_pct,
                if t.approach_clause_holds["linear"] { "HOLDS" } else { "FAILS" },
                if t.beats_unguided_at_all["linear"] { "beats" } else { "does not beat" },
                if t.beats_unguided_at_all["linear"] { "<" } else { "≥" },
                ctl.ratio_vs_unguided_at_b.median,
                ctl.regret_pct.median,
                lin.regret_pct.median,
                if t.linear_beats_control { "beats" } else { "does not beat" },
            ));
        }
    }

    md.push_str("## Enabler-starvation diagnostics\n\n");
    for e in &report.enabler {
        md.push_str(&format!("### {} (n = {})\n\n", e.band, e.n));
        md.push_str(&format!(
            "Unguided strict-positive applications: {} total, {} numeric (non-structural). Of the numeric ones, **{}** ({:.1}%) have a structural application in their tight derivation ancestry (\"structurally enabled\"), {} have a direct child whose chosen node a structural rule created.\n\n",
            e.unguided_strict_positive_total,
            e.numeric_total,
            e.numeric_structurally_enabled,
            if e.numeric_total == 0 { 0.0 } else { 100.0 * e.numeric_structurally_enabled as f64 / e.numeric_total as f64 },
            e.numeric_direct_child_structural,
        ));
        md.push_str("| guided arm | numeric strict-positive terms unguided reached that this arm's final e-graph contains | of the structurally-enabled ones |\n|---|---|---|\n");
        for (arm, n) in &e.numeric_present_in {
            let en = e.enabled_present_in.get(arm).copied().unwrap_or(0);
            md.push_str(&format!(
                "| {arm} | {n} / {} ({:.1}%) | {en} / {} ({:.1}%) |\n",
                e.numeric_total,
                if e.numeric_total == 0 {
                    0.0
                } else {
                    100.0 * *n as f64 / e.numeric_total as f64
                },
                e.numeric_structurally_enabled,
                if e.numeric_structurally_enabled == 0 {
                    0.0
                } else {
                    100.0 * en as f64 / e.numeric_structurally_enabled as f64
                },
            ));
        }
        md.push_str("\n| arm | structural share of applications @100 | @200 | top rules @100 |\n|---|---|---|---|\n");
        for (arm, m) in &e.structural_share_at_b {
            md.push_str(&format!(
                "| {arm} | {:.3} | {:.3} | {} |\n",
                m.get(&100).copied().unwrap_or(f64::NAN),
                m.get(&200).copied().unwrap_or(f64::NAN),
                e.top_rules_at_100
                    .get(arm)
                    .map(|v| v
                        .iter()
                        .map(|(r, n)| format!("{r} {n}"))
                        .collect::<Vec<_>>()
                        .join(", "))
                    .unwrap_or_default()
            ));
        }
        md.push('\n');
        for (arm, d) in &e.seen_keys_per_application {
            md.push_str(&format!(
                "- {arm}: distinct candidate keys scored per recorded application over the run (dedup coverage): {}\n",
                fmt_d(d)
            ));
        }
        md.push('\n');
    }

    md.push_str(
        "## Production-units context (exact production saturation call per expression)\n\n",
    );
    md.push_str("`production_saturation_probe` = the same function body `optimize_runtime_arena` runs (`config_for_node_count` → `saturate_with_full_budget`), stop reason READ from the loop. Wall-clock is a stop condition of that call only; `timeout` stops are machine-dependent.\n\n");
    for p in &report.production {
        md.push_str(&format!(
            "### {} (n = {}, probe returned None for {})\n\n",
            p.band, p.n, p.n_probe_none
        ));
        md.push_str(&format!(
            "- stop reasons: {}\n- effective B (applications at stop): {} (max {:.0})\n- share with applications ≥ 100 / 200 / 400 / 800 / 1600: {}\n- rounds run: {}; classes at stop: {}\n- production cost / unguided cost@100: {}\n- production cost / unguided cost@200: {}\n- production cost / unguided cost@800: {}\n- production regret vs empirical best: {}\n- equivalent unguided checkpoint (smallest grid B whose unguided cost ≤ production's; 0 = worse than every checkpoint): {}\n- unguided checkpoint whose app_actual first reaches production's application count: {}\n\n",
            p.stops.iter().map(|(k, v)| format!("{k} {v}")).collect::<Vec<_>>().join(", "),
            fmt_d(&p.effective_b),
            p.effective_b.max,
            p.share_apps_ge.iter().map(|(k, v)| format!("≥{k}: {:.1}%", v * 100.0)).collect::<Vec<_>>().join(", "),
            fmt_d(&p.iterations),
            fmt_d(&p.classes_after),
            fmt_d(&p.ratio_vs_unguided_at_100),
            fmt_d(&p.ratio_vs_unguided_at_200),
            fmt_d(&p.ratio_vs_unguided_at_4b_800),
            fmt_pct(&p.regret_vs_best_pct),
            p.equivalent_checkpoint_hist.iter().map(|(k, v)| format!("{k}: {v}")).collect::<Vec<_>>().join(", "),
            p.app_bracket_hist.iter().map(|(k, v)| format!("{}: {v}", if *k == usize::MAX { "beyond grid".to_string() } else { k.to_string() })).collect::<Vec<_>>().join(", "),
        ));
    }

    md.push_str("## Context (not metrics)\n\n");
    for (k, v) in &report.context {
        md.push_str(&format!("- {k}: {v}\n"));
    }

    std::fs::write(path, md).unwrap_or_else(|e| panic!("cannot write {}: {e}", path.display()));
}

fn read_rows(path: &Path) -> Vec<ExprRow> {
    let Ok(file) = std::fs::File::open(path) else {
        return Vec::new();
    };
    let mut rows = Vec::new();
    for (i, line) in BufReader::new(file).lines().enumerate() {
        let line = line.unwrap_or_else(|e| panic!("{}: I/O error: {e}", path.display()));
        if line.trim().is_empty() {
            continue;
        }
        let row: ExprRow = serde_json::from_str(&line).unwrap_or_else(|e| {
            panic!(
                "{}:{}: malformed row (delete the line to re-run that expression): {e}",
                path.display(),
                i + 1
            )
        });
        rows.push(row);
    }
    rows
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

/// Size-stratified stride sample of `n` items (`items` sorted by size);
/// `None` = every item.
fn stride_sample<T>(mut items: Vec<T>, n: Option<usize>) -> Vec<T> {
    let Some(n) = n else {
        return items;
    };
    if n >= items.len() {
        return items;
    }
    let stride = items.len() as f64 / n as f64;
    let mut keep: Vec<usize> = (0..n).map(|i| ((i as f64) * stride) as usize).collect();
    keep.dedup();
    let keep: HashSet<usize> = keep.into_iter().collect();
    let mut out = Vec::with_capacity(keep.len());
    for (i, item) in items.drain(..).enumerate() {
        if keep.contains(&i) {
            out.push(item);
        }
    }
    out
}

fn main() {
    let args = Args::parse();
    let jsonl_path = PathBuf::from(&args.out_jsonl);
    let skipped: Vec<String> = args
        .skip_names
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect();
    let load_start = uptime();

    if !args.aggregate_only {
        let costs = CostModel::latency_prior();
        let guided_grid: Vec<usize> = if args.guided_grid.trim().is_empty() {
            GUIDED_GRID.to_vec()
        } else {
            args.guided_grid
                .split(',')
                .map(|t| {
                    t.trim()
                        .parse::<usize>()
                        .unwrap_or_else(|e| panic!("--guided-grid: bad entry {t:?}: {e}"))
                })
                .collect()
        };
        for (b, _, _) in REGISTERED_TIERS {
            assert!(
                guided_grid.contains(&b) && guided_grid.contains(&(4 * b)),
                "--guided-grid {guided_grid:?} must contain registered B={b} and 4B={}",
                4 * b
            );
        }
        let dev_path = PathBuf::from(&args.corpus_dir).join("corpus_dev.bin");
        let mut entries = read_corpus(&dev_path)
            .unwrap_or_else(|e| panic!("failed to read {}: {e}", dev_path.display()));
        entries.sort_by(|a, b| {
            a.1.nodes_raw()
                .len()
                .cmp(&b.1.nodes_raw().len())
                .then_with(|| a.0.cmp(&b.0))
        });
        let mut by_band: BTreeMap<&str, Vec<(String, ExprArena, ExprId)>> = BTreeMap::new();
        for (name, arena, root) in entries {
            by_band
                .entry(tier_name(arena.nodes_raw().len()))
                .or_default()
                .push((name, arena, root));
        }
        let counts: BTreeMap<&str, usize> = by_band.iter().map(|(k, v)| (*k, v.len())).collect();
        eprintln!("phase3_at_budget_eval: DEV population by band: {counts:?}");

        let mut selected: Vec<(String, ExprArena, ExprId)> = Vec::new();
        selected.extend(stride_sample(
            by_band.remove("classical").unwrap_or_default(),
            (args.classical_samples > 0).then_some(args.classical_samples),
        ));
        for band in ["rapid", "blitz"] {
            selected.extend(stride_sample(
                by_band.remove(band).unwrap_or_default(),
                Some(args.other_samples),
            ));
        }

        let existing: HashSet<String> =
            read_rows(&jsonl_path).into_iter().map(|r| r.name).collect();
        eprintln!(
            "phase3_at_budget_eval: {} selected, {} already done, {} skipped by flag",
            selected.len(),
            existing.len(),
            skipped.len()
        );

        let guides = Guides {
            control: PerRuleRateGuide::from_train_guide_report(Path::new(&args.train_guide_report))
                .unwrap_or_else(|e| panic!("control guide: {e}")),
            linear: LinearCandidateGuide::load(Path::new(&args.checkpoint))
                .unwrap_or_else(|e| panic!("linear guide: {e}")),
            embeds: vec![[0.0f32; EMBED_DIM]; all_rules().len()],
        };

        if let Some(parent) = jsonl_path.parent() {
            std::fs::create_dir_all(parent).expect("create output directory");
        }
        let mut out = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&jsonl_path)
            .unwrap_or_else(|e| panic!("cannot open {}: {e}", jsonl_path.display()));

        let total = selected.len();
        for (i, (name, arena, root)) in selected.iter().enumerate() {
            if existing.contains(name) || skipped.contains(name) {
                continue;
            }
            eprintln!(
                "phase3_at_budget_eval: [{}/{}] {} ({} nodes, {})",
                i + 1,
                total,
                name,
                arena.nodes_raw().len(),
                tier_name(arena.nodes_raw().len())
            );
            let row = evaluate_expression(name, arena, *root, &guides, &costs, &guided_grid);
            let line = serde_json::to_string(&row).expect("serialize row");
            writeln!(out, "{line}")
                .unwrap_or_else(|e| panic!("write {}: {e}", jsonl_path.display()));
            out.flush().expect("flush");
        }
    }

    // ------------------------------------------------------------------
    // Aggregate.
    // ------------------------------------------------------------------
    let rows = read_rows(&jsonl_path);
    assert!(
        !rows.is_empty(),
        "no rows in {} — nothing to aggregate",
        jsonl_path.display()
    );
    let mut n_by_band: BTreeMap<String, usize> = BTreeMap::new();
    for r in &rows {
        *n_by_band.entry(r.tier.clone()).or_default() += 1;
    }
    let mut tiers = Vec::new();
    let mut production = Vec::new();
    let mut enabler = Vec::new();
    for band in ["classical", "rapid", "blitz"] {
        let sub: Vec<&ExprRow> = rows.iter().filter(|r| r.tier == band).collect();
        if sub.is_empty() {
            continue;
        }
        for (b, y, l) in REGISTERED_TIERS {
            tiers.push(tier_result(&sub, band, b, y, l));
        }
        production.push(production_summary(&sub, band));
        enabler.push(enabler_summary(&sub, band));
    }
    let mut context = BTreeMap::new();
    context.insert("source_rev".to_string(), git_rev());
    context.insert("load_at_start".to_string(), load_start);
    context.insert("load_at_end".to_string(), uptime());
    context.insert("checkpoint".to_string(), args.checkpoint.clone());
    context.insert(
        "train_guide_report".to_string(),
        args.train_guide_report.clone(),
    );
    context.insert(
        "guided_grid".to_string(),
        format!(
            "{} (unguided: {APP_CHECKPOINT_GRID:?})",
            if args.guided_grid.trim().is_empty() {
                format!("{GUIDED_GRID:?}")
            } else {
                args.guided_grid.clone()
            }
        ),
    );
    context.insert(
        "structural_rules".to_string(),
        format!("{STRUCTURAL_RULES:?}"),
    );
    context.insert("arms".to_string(), format!("{ARM_NAMES:?}"));

    let report = Report {
        n_rows: rows.len(),
        n_by_band,
        skipped_names: skipped,
        tiers,
        production,
        enabler,
        context,
    };
    let json = serde_json::to_string_pretty(&report).expect("serialize report");
    std::fs::write(&args.out_json, json)
        .unwrap_or_else(|e| panic!("cannot write {}: {e}", args.out_json));
    write_markdown(&report, Path::new(&args.out_md), &args.out_jsonl);

    // Console summary: the registered-claim sentences.
    for t in &report.tiers {
        if t.band != "classical" {
            continue;
        }
        let lin = &t.arms[1];
        let ctl = &t.arms[0];
        println!(
            "classical B={}: n={} | linear median ratio {:.3} (threshold {:.3}) Y-clause {} | gap vs 4B {:.2}% (threshold {:.1}%) {} | control median ratio {:.3} | regret unguided {:.2}% control {:.2}% linear {:.2}% unguided@4B {:.2}%",
            t.b,
            t.n,
            lin.ratio_vs_unguided_at_b.median,
            t.ratio_threshold,
            if t.y_clause_holds["linear"] {
                "HOLDS"
            } else {
                "FAILS"
            },
            lin.gap_vs_unguided_at_4b_pct.median,
            t.gap_threshold_pct,
            if t.approach_clause_holds["linear"] {
                "HOLDS"
            } else {
                "FAILS"
            },
            ctl.ratio_vs_unguided_at_b.median,
            t.unguided_regret_at_b_pct.median,
            ctl.regret_pct.median,
            lin.regret_pct.median,
            t.unguided_regret_at_4b_pct.median,
        );
    }
    for p in &report.production {
        println!(
            "production {}: n={} stops {:?} effective-B median {:.0} (q1 {:.0}, q3 {:.0}) share>=100 {:.1}% >=200 {:.1}%",
            p.band,
            p.n,
            p.stops,
            p.effective_b.median,
            p.effective_b.q1,
            p.effective_b.q3,
            p.share_apps_ge[&100] * 100.0,
            p.share_apps_ge[&200] * 100.0
        );
    }

    // Journal (deterministic-metric record; no timing).
    let journal_record = serde_json::json!({
        "record": "phase3_at_budget_eval",
        "ts_unix": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).expect("clock").as_secs(),
        "config": {
            "source_rev": report.context["source_rev"],
            "corpus": "corpus_dev.bin only (FINAL untouched)",
            "protocol": format!("arms=unguided,control,linear;guided_grid={GUIDED_GRID:?};cost=latency_prior;class_cap=production_tier;registered_tiers={REGISTERED_TIERS:?}"),
            "checkpoint": args.checkpoint,
        },
        "n_by_band": report.n_by_band,
        "classical": report.tiers.iter().filter(|t| t.band == "classical").map(|t| serde_json::json!({
            "B": t.b,
            "n": t.n,
            "linear_median_ratio": t.arms[1].ratio_vs_unguided_at_b.median,
            "control_median_ratio": t.arms[0].ratio_vs_unguided_at_b.median,
            "ratio_threshold": t.ratio_threshold,
            "y_clause_linear": t.y_clause_holds["linear"],
            "approach_clause_linear": t.approach_clause_holds["linear"],
            "linear_gap_vs_4b_median_pct": t.arms[1].gap_vs_unguided_at_4b_pct.median,
            "regret_median_pct": {"unguided_at_b": t.unguided_regret_at_b_pct.median, "control": t.arms[0].regret_pct.median, "linear": t.arms[1].regret_pct.median, "unguided_at_4b": t.unguided_regret_at_4b_pct.median},
        })).collect::<Vec<_>>(),
        "production_classical": report.production.iter().find(|p| p.band == "classical").map(|p| serde_json::json!({
            "stops": p.stops, "effective_b_median": p.effective_b.median, "effective_b_q1": p.effective_b.q1, "effective_b_q3": p.effective_b.q3,
        })),
        "outputs": [args.out_jsonl, args.out_json, args.out_md],
    });
    append_record(Path::new(&args.journal), &journal_record);
    eprintln!(
        "phase3_at_budget_eval: wrote {} and {}",
        args.out_json, args.out_md
    );
}
