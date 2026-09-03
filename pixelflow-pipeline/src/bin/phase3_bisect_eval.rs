//! Guided-regression bisect harness (retired-branch era).
//!
//! One purpose: measure the three Phase 3 arms — `unguided`, `control`
//! (per-rule TRAIN strict-positive rate, mapped by `rule_idx` exactly as
//! Round 1b's `PerRuleRateGuide::from_train_guide_report` does), `linear`
//! (the strict-label additive Guide) — on the guided grid, and report at
//! every checkpoint BOTH cost shapes:
//!
//! - `tree_dp`: `ExtractedDAG::total_cost`, the extraction DP's objective
//!   (what every Round-1 number was read from), and
//! - `dag`: the materialized cost — each distinct e-class reachable through
//!   the chosen nodes priced ONCE (what #1117's `ChoiceCost::dag` reports on
//!   main). `tree_walk` re-derives the tree total from the same walk so a
//!   DP/walk disagreement is visible rather than assumed away.
//!
//! Nothing else: no production probe, no strict labels, no journal. The
//! loop is the branch's own `run_anytime_curve_with` restated line for line
//! (it has to be, because that function prices only the tree and does not
//! hand back the e-graph at each checkpoint), so the arms are advanced by
//! exactly the branch's `UnguidedStepper` / `GuidedSaturation`.
//!
//! Every metric is deterministic; wall clock is only the panicking ceiling.

use std::collections::{BTreeMap, HashSet};
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use clap::Parser;
use serde::Serialize;

use pixelflow_ir::{ExprArena, ExprId};
use pixelflow_pipeline::schema::fnv1a64_hex;
use pixelflow_pipeline::training::corpus::read_corpus;
use pixelflow_pipeline::training::structural::FenceKey;
use pixelflow_search::egraph::{
    AnytimeStep, AnytimeStepper, CostFunction, CostModel, EClassId, EGraph, ENode,
    GuidedSaturation, SaturationStop, UnguidedStepper, all_rules, config_for_node_count,
    extract_dag,
};
use pixelflow_search::nnue::factored::EMBED_DIM;
use pixelflow_search::nnue::guide::SaturationGuide;
use pixelflow_search::nnue::guide::linear::{LinearCandidateGuide, PerRuleRateGuide};

/// The registered guided grid (docs/plans/2026-09-01-phase3-registration.md).
const GRID: &[usize] = &[25, 50, 100, 200, 400, 800];
const SWEEP_SAFETY_CEILING: usize = 10_000;
const SAFETY_TIMEOUT: Duration = Duration::from_secs(1800);

#[derive(Parser)]
#[command(name = "phase3_bisect_eval")]
struct Args {
    /// Corpus file to evaluate.
    #[arg(long)]
    corpus: String,
    /// Directory holding `corpus_train.bin` (the TRAIN fence source).
    #[arg(long)]
    corpus_dir: String,
    /// Only entries whose name starts with this prefix. Empty = every entry.
    #[arg(long, default_value = "")]
    name_prefix: String,
    /// `train_guide` checkpoint for the linear arm.
    #[arg(long)]
    checkpoint: String,
    /// `train_guide` report JSON (per-rule TRAIN rates for the control arm).
    #[arg(long)]
    train_guide_report: String,
    /// Per-expression rows, appended; names already present are skipped.
    #[arg(long)]
    out_jsonl: String,
    /// Free-text label for the source revision / lever configuration.
    #[arg(long)]
    era: String,
    /// Evaluate at most this many selected expressions (0 = all).
    #[arg(long, default_value_t = 0)]
    limit: usize,
}

#[derive(Serialize)]
struct Checkpoint {
    app_target: usize,
    app_actual: usize,
    sweeps: usize,
    classes: usize,
    nodes: usize,
    tree_dp: usize,
    tree_walk: usize,
    dag: usize,
    stop: String,
    clamped: bool,
}

#[derive(Serialize)]
struct Arm {
    checkpoints: Vec<Checkpoint>,
    ended: String,
    ended_at_apps: usize,
    seen_keys: Option<usize>,
}

#[derive(Serialize)]
struct Row {
    name: String,
    era: String,
    corpus_fnv64: String,
    node_count: usize,
    class_cap: usize,
    arms: BTreeMap<String, Arm>,
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

fn tier_is_classical(node_count: usize) -> bool {
    node_count > 50
}

/// Price a settled choice map: `tree_walk` sums every child at every use,
/// `dag` prices each distinct reachable e-class once. This is the walk
/// `cost_of_choices` performs on main (#1117) at `LatticeShape::POINT`,
/// restated here because this era has no such function.
fn price_choices(
    egraph: &EGraph,
    root: EClassId,
    choices: &[Option<usize>],
    costs: &CostModel,
) -> (usize, usize) {
    let chosen = |canonical: EClassId| -> &ENode {
        let idx = canonical.index();
        let node_idx = choices.get(idx).and_then(|o| *o).unwrap_or_else(|| {
            panic!(
                "price_choices: e-class {idx} is reachable from root {} but has no choice",
                root.index()
            )
        });
        let nodes = egraph.nodes(canonical);
        assert!(
            node_idx < nodes.len(),
            "price_choices: node_idx {node_idx} out of bounds ({}) for e-class {idx}",
            nodes.len()
        );
        &nodes[node_idx]
    };

    let n = egraph.num_classes();
    let mut tree: Vec<Option<usize>> = vec![None; n];
    let mut color: Vec<u8> = vec![0u8; n];
    let mut dag = 0usize;
    let root_canonical = egraph.find(root);
    let mut stack: Vec<(EClassId, bool)> = vec![(root_canonical, false)];

    while let Some((class, children_done)) = stack.pop() {
        let canonical = egraph.find(class);
        let idx = canonical.index();
        if !children_done {
            if color[idx] == 2 {
                continue;
            }
            assert!(
                color[idx] != 1,
                "price_choices: the choice graph is cyclic at e-class {idx} (root {})",
                root.index()
            );
            color[idx] = 1;
            stack.push((canonical, true));
            if let ENode::Op { children, .. } = chosen(canonical) {
                for &child in children {
                    stack.push((child, false));
                }
            }
            continue;
        }
        color[idx] = 2;
        let node = chosen(canonical);
        let own = costs.node_cost(node, None);
        let children_cost = match node {
            ENode::Op { children, .. } => children
                .iter()
                .map(|&c| {
                    tree[egraph.find(c).index()].expect("post-order visits children first")
                })
                .fold(0usize, usize::saturating_add),
            ENode::Var(_) | ENode::Const(_) | ENode::Buffer(_) => 0,
        };
        tree[idx] = Some(own.saturating_add(children_cost));
        dag = dag.saturating_add(own);
    }
    (
        tree[root_canonical.index()].expect("root is costed by the walk"),
        dag,
    )
}

/// The branch's `run_anytime_curve_with`, restated with dag pricing at each
/// checkpoint. Identical stop / clamp / sweep-accounting semantics.
fn run_curve(
    arena: &ExprArena,
    root: ExprId,
    class_cap: usize,
    costs: &CostModel,
    stepper: &mut impl AnytimeStepper,
) -> (Vec<Checkpoint>, String, usize) {
    let mut egraph = EGraph::with_rules(all_rules());
    let root_class = egraph.add_arena(arena, root);
    let deadline = Instant::now() + SAFETY_TIMEOUT;

    let mut checkpoints: Vec<Checkpoint> = Vec::with_capacity(GRID.len());
    let mut sweeps_total = 0usize;
    let mut ended: Option<SaturationStop> = None;
    let mut last_apps = 0usize;

    for &target in GRID {
        if let Some(prev) = checkpoints.last() {
            if ended.is_some() {
                checkpoints.push(Checkpoint {
                    app_target: target,
                    app_actual: prev.app_actual,
                    sweeps: prev.sweeps,
                    classes: prev.classes,
                    nodes: prev.nodes,
                    tree_dp: prev.tree_dp,
                    tree_walk: prev.tree_walk,
                    dag: prev.dag,
                    stop: prev.stop.clone(),
                    clamped: true,
                });
                continue;
            }
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        assert!(
            !remaining.is_zero(),
            "bisect: curve exceeded the {SAFETY_TIMEOUT:?} safety ceiling before target {target}"
        );
        let sweeps_left = SWEEP_SAFETY_CEILING
            .checked_sub(sweeps_total)
            .unwrap_or_else(|| panic!("sweep accounting underflow ({sweeps_total})"));
        let stats = stepper.advance(
            &mut egraph,
            AnytimeStep {
                app_target: target,
                sweeps_left,
                max_classes: class_cap,
                remaining,
            },
        );
        assert!(
            stats.stop != SaturationStop::Timeout,
            "bisect: saturation hit the wall-clock ceiling at target {target}"
        );
        sweeps_total += stats.iterations;
        let extraction = extract_dag(&egraph, root_class, costs);
        let (tree_walk, dag) =
            price_choices(&egraph, extraction.root, &extraction.choices, costs);
        checkpoints.push(Checkpoint {
            app_target: target,
            app_actual: stats.applications,
            sweeps: sweeps_total,
            classes: egraph.num_classes(),
            nodes: egraph.node_count(),
            tree_dp: extraction.total_cost,
            tree_walk,
            dag,
            stop: stop_name(stats.stop).to_string(),
            clamped: false,
        });
        last_apps = stats.applications;
        if stats.stop != SaturationStop::ApplicationBudget {
            ended = Some(stats.stop);
        }
    }
    (
        checkpoints,
        stop_name(ended.unwrap_or(SaturationStop::ApplicationBudget)).to_string(),
        last_apps,
    )
}

fn run_guided<G: SaturationGuide>(
    guide: &G,
    embeds: &[[f32; EMBED_DIM]],
    arena: &ExprArena,
    root: ExprId,
    class_cap: usize,
    costs: &CostModel,
) -> Arm {
    let mut stepper = GuidedSaturation::new(guide, embeds);
    let (checkpoints, ended, ended_at_apps) = run_curve(arena, root, class_cap, costs, &mut stepper);
    Arm {
        checkpoints,
        ended,
        ended_at_apps,
        seen_keys: Some(stepper.seen_key_count()),
    }
}

fn enforce_train_fence(corpus_dir: &Path, entries: &[(String, ExprArena, ExprId)]) {
    let path = corpus_dir.join("corpus_train.bin");
    assert!(path.exists(), "TRAIN corpus not found at {}", path.display());
    let train: HashSet<FenceKey> = read_corpus(&path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()))
        .iter()
        .map(|(_, a, r)| FenceKey::of(a, *r))
        .collect();
    let collisions: Vec<&str> = entries
        .iter()
        .filter(|(_, a, r)| train.contains(&FenceKey::of(a, *r)))
        .map(|(n, _, _)| n.as_str())
        .collect();
    assert!(
        collisions.is_empty(),
        "TRAIN-fence violation: {} entries collide: {:?}",
        collisions.len(),
        &collisions[..collisions.len().min(10)]
    );
    eprintln!(
        "bisect: TRAIN fence OK — {} entries probed against {} TRAIN structures",
        entries.len(),
        train.len()
    );
}

fn existing_names(path: &Path) -> HashSet<String> {
    let Ok(f) = std::fs::File::open(path) else {
        return HashSet::new();
    };
    std::io::BufReader::new(f)
        .lines()
        .map(|l| l.expect("read jsonl line"))
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let v: serde_json::Value = serde_json::from_str(&l).expect("parse jsonl row");
            v["name"].as_str().expect("row has a name").to_string()
        })
        .collect()
}

fn main() {
    let args = Args::parse();
    let corpus_path = PathBuf::from(&args.corpus);
    let corpus_bytes = std::fs::read(&corpus_path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", corpus_path.display()));
    let corpus_fnv64 = fnv1a64_hex(&corpus_bytes);
    let entries = read_corpus(&corpus_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", corpus_path.display()));
    enforce_train_fence(Path::new(&args.corpus_dir), &entries);

    let selected: Vec<&(String, ExprArena, ExprId)> = entries
        .iter()
        .filter(|(name, arena, _)| {
            name.starts_with(&args.name_prefix) && tier_is_classical(arena.nodes_raw().len())
        })
        .collect();
    let selected = if args.limit > 0 {
        selected.into_iter().take(args.limit).collect()
    } else {
        selected
    };

    let costs = CostModel::latency_prior();
    let rules = all_rules();
    let control = PerRuleRateGuide::from_train_guide_report(Path::new(&args.train_guide_report))
        .unwrap_or_else(|e| panic!("control guide: {e}"));
    let linear = LinearCandidateGuide::load(Path::new(&args.checkpoint))
        .unwrap_or_else(|e| panic!("linear guide: {e}"));
    let embeds = vec![[0.0f32; EMBED_DIM]; rules.len()];

    let out_path = PathBuf::from(&args.out_jsonl);
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent).expect("create output directory");
    }
    let done = existing_names(&out_path);
    let mut out = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&out_path)
        .unwrap_or_else(|e| panic!("cannot open {}: {e}", out_path.display()));

    eprintln!(
        "bisect[{}]: {} selected from {} ({} already done), corpus fnv64 {corpus_fnv64}",
        args.era,
        selected.len(),
        corpus_path.display(),
        done.len()
    );
    let total = selected.len();
    for (i, (name, arena, root)) in selected.into_iter().enumerate() {
        if done.contains(name) {
            continue;
        }
        let node_count = arena.nodes_raw().len();
        let class_cap = config_for_node_count(node_count).max_classes;
        let started = Instant::now();

        let mut arms = BTreeMap::new();
        let (checkpoints, ended, ended_at_apps) =
            run_curve(arena, *root, class_cap, &costs, &mut UnguidedStepper);
        arms.insert(
            "unguided".to_string(),
            Arm {
                checkpoints,
                ended,
                ended_at_apps,
                seen_keys: None,
            },
        );
        arms.insert(
            "control".to_string(),
            run_guided(&control, &embeds, arena, *root, class_cap, &costs),
        );
        arms.insert(
            "linear".to_string(),
            run_guided(&linear, &embeds, arena, *root, class_cap, &costs),
        );

        let row = Row {
            name: name.clone(),
            era: args.era.clone(),
            corpus_fnv64: corpus_fnv64.clone(),
            node_count,
            class_cap,
            arms,
        };
        let line = serde_json::to_string(&row).expect("serialize row");
        writeln!(out, "{line}").expect("write row");
        out.flush().expect("flush row");
        eprintln!(
            "bisect[{}]: [{}/{}] {} ({} nodes) {:.1}s",
            args.era,
            i + 1,
            total,
            name,
            node_count,
            started.elapsed().as_secs_f64()
        );
    }
}
