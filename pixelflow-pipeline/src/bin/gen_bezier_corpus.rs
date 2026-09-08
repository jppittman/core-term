//! `bezier` — the polynomial-only OOD family
//! (docs/plans/2026-09-01-phase3-round1b-domain-shift-registration.md §3b).
//!
//! Round 1b's question is whether the linear Guide's advantage survives
//! domain shift toward trig-dominant kernels (`sh`, this family's sibling
//! generator). `bezier` is the control case: the global per-rule prior is
//! *right* here — there is no trig to suppress — so both the Guide and its
//! per-rule control arm are predicted to do at or better than DEV-overall
//! on it (§1.2). Every expression this binary emits is built ONLY from
//! `Add`/`Sub`/`Mul`/`MulAdd`/`Neg` (plus `Var`/`Const` leaves) — no `Div`,
//! no `Sqrt`, no trig/transcendental op anywhere — so the family lands in
//! the registration's `polynomial-only` op-composition stratum by
//! construction, not by luck. `assert_polynomial_only` below checks that
//! mechanically, on every admitted entry, rather than trusting the
//! construction code to have gotten it right.
//!
//! # Node counts (measured, not the registration's own estimate)
//!
//! `arena.len()` (== the harness's `node_count`, since these arenas carry
//! no dead nodes) is fixed by (form, degree) alone — literal control-point
//! values change none of it: `bezier-bernstein` 62 (degree 3) / 78 (degree
//! 4), `bezier-casteljau` 40 (degree 3, BELOW the 51-node floor — always
//! discarded) / 58 (degree 4), `bezier-patch` 127. Four of the five (form,
//! degree) combinations clear the band; `bezier-casteljau` degree 3 never
//! does, which is why the generation loop's attrition rate is a
//! deterministic ~20%, not a tail risk.
//!
//! # Forms (registration §3b, each draw picks one uniformly)
//!
//! - `bezier-bernstein`: degree n in {3, 4}, B(t) = Σ C(n,i)(1−t)^(n−i)tⁱPᵢ,
//!   evaluated as the squared distance from point (Y, c₀) to the curve
//!   point at parameter X — the coordinate variables ARE the curve
//!   parameter and the query point's second coordinate, per the fixed
//!   parameterisation (t = X).
//! - `bezier-casteljau`: the same squared-distance form, but B_x/B_y are
//!   each computed by nested `lerp(a, b, t) = a + (b−a)·t` (de Casteljau),
//!   built as `MulAdd(Sub(b, a), t, a)` — one rounding, matching the
//!   registration's note that this is the form `fma-fusion` has the most
//!   to do with.
//! - `bezier-patch`: a fixed bicubic tensor-product patch,
//!   z(X, Y) = Σᵢ Σⱼ Pᵢⱼ·Bᵢ(X)·Bⱼ(Y), 16 independent heights.
//!
//! # Two different notions of "the same expression" (see `training::structural`)
//!
//! [`FenceKey`] is literal-blind by design (`X*2.0` and `X*3.0` are the
//! SAME key — that is exactly what makes it the right tool for "has TRAIN
//! already seen this shape"). It is used here for exactly one purpose: the
//! registration §3 TRAIN-fence hard-error check. It is deliberately NOT
//! used to decide whether two `bezier` draws are "distinct" — every draw of
//! one (form, degree) combination shares the SAME literal-blind structure
//! (same op tree, different control-point constants), so a literal-blind
//! notion of "distinct" would cap this family at one entry per (form,
//! degree) pair, defeating the registered 60-100-distinct target the
//! coefficient variation is supposed to produce (§3: "seeded
//! coefficient/degree/form variation"). In-family dedup here instead uses
//! [`ExactKey`], a local literal-INCLUDING structural hash — the ordinary
//! meaning of "distinct expression" — which exists only to catch a
//! degenerate repeated draw (RNG bug), not to enforce novelty of shape.
//!
//! # Oracle validation (two independent checks, not one)
//!
//! [`Quarantine`] (reused unchanged, registration §3: "the numeric
//! quarantine applies unchanged") cross-checks the JIT lane against
//! `pixelflow_ir`'s tree-walking `eval_scalar` on the SAME arena — it
//! catches an emitter/lowering bug, but it cannot catch the generator
//! building the WRONG expression. The generator itself, and the
//! from-scratch scalar reference that catches THAT, live in
//! [`pixelflow_pipeline::training::bezier_family`] (`oracle_tests` there).

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use clap::Parser;
use pixelflow_ir::{ExprArena, ExprId, ExprNode, OpKind};
use pixelflow_pipeline::training::bezier_family::{FORMS, Lcg};
use pixelflow_pipeline::training::corpus::{read_corpus, write_corpus};
use pixelflow_pipeline::training::quarantine::Quarantine;
use pixelflow_pipeline::training::split::SplitManifest;
use pixelflow_pipeline::training::structural::FenceKey;

#[derive(Parser)]
#[command(name = "gen_bezier_corpus")]
#[command(about = "Generate the `bezier` DEV-only OOD family (round 1b, §3b)")]
struct Args {
    /// Directory holding `corpus_train.bin` (fence source, read-only) and
    /// `corpus_dev_ood.bin` (write target — merged, not overwritten: any
    /// non-`dev_bezier_*` entries already there, e.g. the `sh` family, are
    /// preserved verbatim).
    #[arg(long, default_value = "pixelflow-pipeline/data")]
    output: String,

    /// Split manifest path — must register `bezier` under `[dev] families`
    /// or this binary refuses to run (the same discipline `gen_bench_corpus`
    /// applies to `[final] kernels`: a generator producing a family the
    /// manifest does not know about is a manifest bug, not a corpus bug).
    #[arg(long, default_value = "pixelflow-pipeline/corpus_split.toml")]
    manifest: String,

    /// Random seed.
    #[arg(long, default_value_t = 20_260_901)]
    seed: u64,

    /// Target number of admitted (node-count-in-band, deduped, quarantine-passed)
    /// expressions. Registration §3: 60-100 distinct per family.
    #[arg(long, default_value_t = 80)]
    target: usize,

    /// Quarantine exclusion sidecar (JSONL). Defaults to
    /// `<output>/bezier_quarantine.jsonl`.
    #[arg(long)]
    quarantine_log: Option<String>,
}

/// Registration §3 size band: node count > 50, aim 51-400.
const MIN_NODES: usize = 51;
const MAX_NODES: usize = 400;

/// Registration §3: "fewer than 30 survivors is a failed generation".
const MIN_SURVIVORS: usize = 30;

/// Attempts-per-target multiplier before giving up. Generous: node-count
/// filtering alone discards ~1/5 of draws (`bezier-casteljau` at degree 3
/// falls under the floor — see `tests::node_counts_span_the_band_across_forms`),
/// so reaching `target` well inside this budget is the expected case, not
/// the edge case.
const ATTEMPTS_PER_TARGET: usize = 40;

// ============================================================================
// In-family dedup: a literal-INCLUDING structural key. See the module doc
// for why this is deliberately not `FenceKey`.
// ============================================================================

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ExactNode {
    tag: u8,
    payload: u32,
    children: Box<[u32]>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ExactKey(Vec<ExactNode>);

impl ExactKey {
    /// Same post-order DAG walk as `FenceKey::of`, but keeping the literal
    /// payload (`Const` bit pattern, `Var`/`Param` index) instead of
    /// dropping it — two `bezier` draws with different control points are
    /// meant to compare unequal here.
    fn of(arena: &ExprArena, root: ExprId) -> Self {
        enum Task {
            Visit(ExprId),
            Emit(ExprId),
        }
        let mut work = vec![Task::Visit(root)];
        let mut visited = HashSet::new();
        let mut nodes = Vec::new();
        let mut ids = std::collections::HashMap::<ExprId, u32>::new();

        while let Some(task) = work.pop() {
            match task {
                Task::Visit(id) => {
                    if !visited.insert(id) {
                        continue;
                    }
                    work.push(Task::Emit(id));
                    for child in arena.children(id) {
                        work.push(Task::Visit(child));
                    }
                }
                Task::Emit(id) => {
                    let children: Box<[u32]> = arena
                        .children(id)
                        .map(|c| *ids.get(&c).expect("child visited before parent emits"))
                        .collect();
                    let (tag, payload) = match arena.node(id) {
                        ExprNode::Var(i) => (0u8, u32::from(*i)),
                        ExprNode::Const(v) => (1u8, v.to_bits()),
                        ExprNode::Param(i) => (2u8, u32::from(*i)),
                        ExprNode::Buffer(b) => (3u8, u32::from(b.0)),
                        ExprNode::Uniform(u) => (5u8, u32::from(u.0)),
                        _ => (4u8, arena.kind(id) as u32),
                    };
                    let key_id = nodes.len() as u32;
                    nodes.push(ExactNode {
                        tag,
                        payload,
                        children,
                    });
                    ids.insert(id, key_id);
                }
            }
        }
        ExactKey(nodes)
    }
}

// ============================================================================
// Op-composition self-check (registration §3b: "no trig or transcendental
// rule can match anything in it" — checked, not assumed).
// ============================================================================

const ALLOWED_NON_LEAF_OPS: [OpKind; 5] = [
    OpKind::Add,
    OpKind::Sub,
    OpKind::Mul,
    OpKind::MulAdd,
    OpKind::Neg,
];

fn assert_polynomial_only(name: &str, arena: &ExprArena) {
    for node in arena.nodes_raw() {
        let op = match node {
            ExprNode::Var(_)
            | ExprNode::Const(_)
            | ExprNode::Param(_)
            | ExprNode::Buffer(_)
            | ExprNode::Uniform(_) => {
                continue;
            }
            ExprNode::Ref(k) => panic!(
                "bezier entry {name} contains Ref({k:?}) — the family is built from \
                 arithmetic alone and never composes by reference"
            ),
            ExprNode::Unary(op, _)
            | ExprNode::Binary(op, _, _)
            | ExprNode::Ternary(op, _, _, _)
            | ExprNode::Nary(op, _, _) => *op,
        };
        assert!(
            ALLOWED_NON_LEAF_OPS.contains(&op),
            "bezier entry {name} contains {op:?}, which is outside \
             {{Add,Sub,Mul,MulAdd,Neg}} — the family must be polynomial-only \
             by construction (registration §3b)"
        );
    }
}

// ============================================================================
// TRAIN fence (registration §3: "every OOD FenceKey probed against
// corpus_train.bin, collision = hard error").
// ============================================================================

fn train_fence_keys(corpus_dir: &Path) -> HashSet<FenceKey> {
    let path = corpus_dir.join("corpus_train.bin");
    assert!(
        path.exists(),
        "TRAIN corpus not found at {} — the fence check cannot run without it \
         (regenerate with gen_bench_corpus --output {})",
        path.display(),
        corpus_dir.display()
    );
    let entries =
        read_corpus(&path).unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
    assert!(
        !entries.is_empty(),
        "TRAIN corpus at {} is empty — a fence built from it would block nothing",
        path.display()
    );
    entries
        .iter()
        .map(|(_, arena, root)| FenceKey::of(arena, *root))
        .collect()
}

// ============================================================================
// Report
// ============================================================================

fn percentile(sorted: &[usize], p: f64) -> usize {
    assert!(!sorted.is_empty());
    let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[idx]
}

fn main() {
    let args = Args::parse();

    let manifest = SplitManifest::load(Path::new(&args.manifest))
        .unwrap_or_else(|e| panic!("failed to load split manifest {}: {e}", args.manifest));
    assert!(
        manifest.dev_families.iter().any(|f| f == "bezier"),
        "split manifest {} does not register \"bezier\" under [dev] families — add it before \
         generating (a family this manifest cannot back with a corpus must not exist)",
        args.manifest
    );

    std::fs::create_dir_all(&args.output)
        .unwrap_or_else(|e| panic!("failed to create output directory {}: {e}", args.output));
    let output_dir = PathBuf::from(&args.output);
    let ood_path = output_dir.join("corpus_dev_ood.bin");
    let quarantine_log_path = args
        .quarantine_log
        .clone()
        .unwrap_or_else(|| format!("{}/bezier_quarantine.jsonl", args.output));

    let train_keys = train_fence_keys(&output_dir);
    let mut quarantine = Quarantine::new(&quarantine_log_path);
    let mut rng = Lcg::new(args.seed);

    let mut seen_within: HashSet<ExactKey> = HashSet::new();
    let mut admitted: Vec<(String, ExprArena, ExprId)> = Vec::new();
    let mut form_counts: std::collections::BTreeMap<&'static str, usize> =
        std::collections::BTreeMap::new();
    let mut node_counts: Vec<usize> = Vec::new();
    let mut op_universe: HashSet<OpKind> = HashSet::new();

    let mut attempts = 0usize;
    let mut too_small = 0usize;
    let mut too_large = 0usize;
    let mut dup_within = 0usize;
    let mut quarantined = 0usize;
    let max_attempts = args.target * ATTEMPTS_PER_TARGET;

    while admitted.len() < args.target && attempts < max_attempts {
        attempts += 1;
        let form = FORMS[rng.choice(FORMS.len())];
        let (arena, root) = form.build(&mut rng);
        // `arena.len()`, not `node_count_subtree(root)`: these arenas are
        // built fresh per draw with zero dead nodes (nothing pushed is ever
        // abandoned), so `len()` already equals the post-`reachable_subtree`
        // compacted node count `write_corpus` stores and the harness reads
        // back as `node_count` (`phase3_at_budget_eval.rs`:
        // `arena.nodes_raw().len()`). `node_count_subtree` instead counts
        // per-*reference* (a shared node — e.g. `t` used by every power —
        // counted once per parent), which is a different, larger number and
        // not the one the registered size band is measured against.
        let n = arena.len();
        if n < MIN_NODES {
            too_small += 1;
            continue;
        }
        if n > MAX_NODES {
            too_large += 1;
            continue;
        }

        let exact_key = ExactKey::of(&arena, root);
        if !seen_within.insert(exact_key) {
            dup_within += 1;
            continue;
        }

        let train_key = FenceKey::of(&arena, root);
        assert!(
            !train_keys.contains(&train_key),
            "TRAIN-fence violation: a bezier draw (form {}, degree {}, attempt {attempts}) \
             shares a feature-quotient structure with a TRAIN expression — registration §3 \
             makes this a hard, named error (a structural duplicate of TRAIN is a leak, not a \
             hygiene event)",
            form.label(),
            form.degree()
        );

        assert_polynomial_only(&format!("attempt {attempts}"), &arena);

        let name = format!("dev_bezier_{:05}", admitted.len());
        if !quarantine.check(&name, &arena, root) {
            quarantined += 1;
            continue;
        }

        for node in arena.nodes_raw() {
            let op = match node {
                ExprNode::Var(_)
                | ExprNode::Const(_)
                | ExprNode::Param(_)
                | ExprNode::Buffer(_)
                | ExprNode::Uniform(_) => {
                    continue;
                }
                ExprNode::Ref(k) => panic!(
                    "bezier entry {name} contains Ref({k:?}) — a name for a kernel \
                     interned in this process, which a corpus file outlives"
                ),
                ExprNode::Unary(op, _)
                | ExprNode::Binary(op, _, _)
                | ExprNode::Ternary(op, _, _, _)
                | ExprNode::Nary(op, _, _) => *op,
            };
            op_universe.insert(op);
        }
        *form_counts.entry(form.label()).or_insert(0) += 1;
        node_counts.push(n);
        admitted.push((name, arena, root));
    }

    assert!(
        admitted.len() >= MIN_SURVIVORS,
        "bezier generation FAILED: only {} survived (< {MIN_SURVIVORS} — registration §3) after \
         {attempts} attempts (too_small={too_small}, too_large={too_large}, \
         dup_within={dup_within}, quarantined={quarantined}); see {quarantine_log_path}",
        admitted.len()
    );

    let (checked, excluded, mismatched) = quarantine.tallies();
    quarantine.finish();

    // Merge into corpus_dev_ood.bin: preserve any entry that is not a
    // `dev_bezier_*` name (e.g. the `sh` family) untouched, replace the
    // bezier entries wholesale with this run's admissions.
    let existing = if ood_path.exists() {
        read_corpus(&ood_path)
            .unwrap_or_else(|e| panic!("failed to read existing {}: {e}", ood_path.display()))
    } else {
        Vec::new()
    };
    let preserved: Vec<(String, ExprArena, ExprId)> = existing
        .into_iter()
        .filter(|(name, _, _)| !name.starts_with("dev_bezier_"))
        .collect();
    let preserved_count = preserved.len();
    let mut out_entries = preserved;
    out_entries.extend(admitted.iter().map(|(n, a, r)| (n.clone(), a.clone(), *r)));
    write_corpus(&ood_path, &out_entries)
        .unwrap_or_else(|e| panic!("failed to write {}: {e}", ood_path.display()));

    // ── Report ───────────────────────────────────────────────────────────
    let mut sorted_nodes = node_counts.clone();
    sorted_nodes.sort_unstable();
    let min_n = sorted_nodes[0];
    let max_n = *sorted_nodes.last().expect("nonempty");
    let mean_n = node_counts.iter().sum::<usize>() as f64 / node_counts.len() as f64;
    let p50 = percentile(&sorted_nodes, 0.50);
    let p90 = percentile(&sorted_nodes, 0.90);

    println!("\n=== bezier family: generation report ===");
    println!(
        "Attempts: {attempts}  admitted: {}  too_small(<{MIN_NODES}): {too_small}  \
         too_large(>{MAX_NODES}): {too_large}  dup_within: {dup_within}  quarantined: \
         {quarantined}",
        admitted.len()
    );
    println!("Quarantine: checked={checked} excluded={excluded} mismatched={mismatched}");
    println!(
        "Node count: min={min_n} max={max_n} mean={mean_n:.1} p50={p50} p90={p90} \
         (band {MIN_NODES}-{MAX_NODES})"
    );
    println!("By form/degree: {form_counts:?}");
    let mut ops_sorted: Vec<String> = op_universe.iter().map(|o| format!("{o:?}")).collect();
    ops_sorted.sort();
    println!("Op composition (union over all admitted): {ops_sorted:?}");
    let trig_present = op_universe.iter().any(|o| {
        matches!(
            o,
            OpKind::Sin
                | OpKind::Cos
                | OpKind::Tan
                | OpKind::Asin
                | OpKind::Acos
                | OpKind::Atan
                | OpKind::Atan2
        )
    });
    assert!(
        !trig_present,
        "bezier corpus contains a trig op — must be trig-free by construction"
    );
    println!(
        "Trig-free: confirmed (0 trig ops across {} entries)",
        admitted.len()
    );
    println!(
        "TRAIN overlap: 0 (every admitted FenceKey was probed against {} TRAIN structures with \
         no collision — a collision panics generation before this line is reached)",
        train_keys.len()
    );
    println!(
        "Wrote {} bezier entries + {preserved_count} preserved entries = {} total to {}",
        admitted.len(),
        admitted.len() + preserved_count,
        ood_path.display()
    );
}

// ============================================================================
// The writer's own checks: the admission policy over the family. The
// family's numeric oracle tests are in `training::bezier_family`.
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use pixelflow_pipeline::training::bezier_family::Form;

    #[test]
    fn every_admitted_form_is_polynomial_only() {
        let mut rng = Lcg::new(42);
        for _ in 0..50 {
            let form = FORMS[rng.choice(FORMS.len())];
            let (arena, _root) = form.build(&mut rng);
            assert_polynomial_only(&format!("{:?}", form), &arena);
        }
    }

    #[test]
    fn node_counts_span_the_band_across_forms() {
        // Descriptive, not a hard gate on any one form: reports which forms
        // clear MIN_NODES so the binary's own generation loop's attrition
        // is not a surprise.
        let mut rng = Lcg::new(123);
        let mut by_form: std::collections::BTreeMap<String, Vec<usize>> =
            std::collections::BTreeMap::new();
        for form in FORMS {
            for _ in 0..20 {
                let (arena, _root) = form.build(&mut rng);
                by_form
                    .entry(format!("{} (degree {})", form.label(), form.degree()))
                    .or_default()
                    .push(arena.len());
            }
        }
        for (label, counts) in &by_form {
            let min = counts.iter().min().unwrap();
            let max = counts.iter().max().unwrap();
            println!(
                "{label}: node counts range {min}..{max} (n={})",
                counts.len()
            );
        }
        // At least SOME draws must clear the classical floor, or the
        // generation loop in main() could never admit anything.
        assert!(
            by_form.values().flatten().any(|&n| n >= MIN_NODES),
            "not one of 100 sampled draws cleared the {MIN_NODES}-node floor — the family as \
             constructed cannot populate the classical band at all"
        );
    }

    #[test]
    fn exact_key_distinguishes_different_control_points_same_topology() {
        let mut rng_a = Lcg::new(1);
        let mut rng_b = Lcg::new(2);
        let (arena_a, root_a) = Form::BernsteinCubic.build(&mut rng_a);
        let (arena_b, root_b) = Form::BernsteinCubic.build(&mut rng_b);
        assert_ne!(
            ExactKey::of(&arena_a, root_a),
            ExactKey::of(&arena_b, root_b),
            "two bernstein-cubic draws with different (seeded) control points must be distinct \
             under the literal-including ExactKey, even though they share one FenceKey"
        );
        assert_eq!(
            FenceKey::of(&arena_a, root_a),
            FenceKey::of(&arena_b, root_b),
            "…and DO share one FenceKey — same op tree, same shape, only the constants differ; \
             this is exactly why FenceKey is not used for in-family dedup here"
        );
    }

    #[test]
    fn exact_key_is_deterministic_and_repeat_draws_collide() {
        let mut rng1 = Lcg::new(99);
        let mut rng2 = Lcg::new(99);
        let (a1, r1) = Form::Patch.build(&mut rng1);
        let (a2, r2) = Form::Patch.build(&mut rng2);
        assert_eq!(
            ExactKey::of(&a1, r1),
            ExactKey::of(&a2, r2),
            "same seed must reproduce the identical draw (determinism), and ExactKey must \
             recognize the resulting duplicate"
        );
    }
}
