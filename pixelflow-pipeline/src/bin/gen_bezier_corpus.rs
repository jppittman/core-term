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
//! catches an emitter/lowering bug, but it cannot catch this file building
//! the WRONG expression (a transposed binomial coefficient, an off-by-one
//! power), because both lanes would agree on whatever tree got built. The
//! `oracle_tests` module below is the check that catches THAT: a
//! from-scratch scalar-Rust reference for each form, cross-checked against
//! `eval_scalar` on genuinely independent control points and sample points.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use clap::Parser;
use pixelflow_ir::{ExprArena, ExprId, ExprNode, OpKind};
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
/// falls under the floor — see `oracle_tests::node_counts_span_the_band`),
/// so reaching `target` well inside this budget is the expected case, not
/// the edge case.
const ATTEMPTS_PER_TARGET: usize = 40;

// ============================================================================
// RNG — the same PCG-style LCG `pixelflow_search::nnue::BwdGenerator` uses,
// so a `bezier` reader recognizes the recipe instead of learning a new one.
// ============================================================================

struct Lcg(u64);

impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        (self.0 >> 33) as f32 / (1u64 << 31) as f32
    }

    /// Uniform in `[lo, hi)`.
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + self.next_f32() * (hi - lo)
    }

    /// Uniform index in `0..n`. `n` must be nonzero.
    fn choice(&mut self, n: usize) -> usize {
        let v = (self.next_f32() * n as f32) as usize;
        v.min(n - 1)
    }
}

// ============================================================================
// Bernstein / de Casteljau construction — POLY-only (Add/Sub/Mul/MulAdd/Neg)
// ============================================================================

const BINOM3: [f32; 4] = [1.0, 3.0, 3.0, 1.0];
const BINOM4: [f32; 5] = [1.0, 4.0, 6.0, 4.0, 1.0];

fn binom(n: usize) -> &'static [f32] {
    match n {
        3 => &BINOM3,
        4 => &BINOM4,
        other => panic!("bezier corpus only defines degree 3/4, got {other}"),
    }
}

fn fold_add(a: &mut ExprArena, xs: &[ExprId]) -> ExprId {
    assert!(!xs.is_empty(), "fold_add of an empty term list");
    let mut acc = xs[0];
    for &x in &xs[1..] {
        acc = a.push_binary(OpKind::Add, acc, x);
    }
    acc
}

/// `[base^0 (represented as None, i.e. "no factor"), base^1, .., base^n]`.
fn powers(a: &mut ExprArena, base: ExprId, n: usize) -> Vec<Option<ExprId>> {
    let mut v = vec![None; n + 1];
    if n >= 1 {
        v[1] = Some(base);
    }
    for k in 2..=n {
        let prev = v[k - 1].expect("power k-1 already computed");
        v[k] = Some(a.push_binary(OpKind::Mul, prev, base));
    }
    v
}

/// One 1-D Bernstein-basis-weighted sum, degree `n`, evaluated at `t`, self-
/// contained (recomputes its own power table each call — deliberately not
/// shared across the two coordinate calls a curve needs, both because it
/// keeps this function simple and because the registration's own node-count
/// estimates (~60 cubic / ~85 quartic for `bezier-bernstein`) assume the
/// un-shared construction).
fn bernstein_1d(a: &mut ExprArena, t: ExprId, n: usize, values: &[f32]) -> ExprId {
    assert_eq!(values.len(), n + 1, "bernstein_1d: need n+1 control values");
    let one = a.push_const(1.0);
    let omt = a.push_binary(OpKind::Sub, one, t);
    let t_pows = powers(a, t, n);
    let omt_pows = powers(a, omt, n);
    let b = binom(n);
    let terms: Vec<ExprId> = (0..=n)
        .map(|i| {
            let c = a.push_const(b[i]);
            let mut acc = c;
            if let Some(tp) = t_pows[i] {
                acc = a.push_binary(OpKind::Mul, acc, tp);
            }
            if let Some(op) = omt_pows[n - i] {
                acc = a.push_binary(OpKind::Mul, acc, op);
            }
            let pc = a.push_const(values[i]);
            a.push_binary(OpKind::Mul, acc, pc)
        })
        .collect();
    fold_add(a, &terms)
}

/// `lerp(a, b, t) = a + (b - a)*t`, one `Sub` + one `MulAdd` (one rounding).
fn lerp(a: &mut ExprArena, p0: ExprId, p1: ExprId, t: ExprId) -> ExprId {
    let diff = a.push_binary(OpKind::Sub, p1, p0);
    a.push_ternary(OpKind::MulAdd, diff, t, p0)
}

/// de Casteljau reduction of `points` (already-pushed leaf `ExprId`s) at `t`:
/// repeatedly lerp adjacent pairs until one point remains. `n` points give
/// `n-1 + n-2 + .. + 1` lerps (6 for 4 points / degree 3, 10 for 5 points /
/// degree 4 — registration §3b).
fn de_casteljau(a: &mut ExprArena, points: &[ExprId], t: ExprId) -> ExprId {
    assert!(points.len() >= 2, "de Casteljau needs at least 2 points");
    let mut level = points.to_vec();
    while level.len() > 1 {
        let next: Vec<ExprId> = level.windows(2).map(|w| lerp(a, w[0], w[1], t)).collect();
        level = next;
    }
    level[0]
}

/// The squared-distance tail shared by `bezier-bernstein` and
/// `bezier-casteljau`: `(Bx(X) - Y)^2 + (By(X) - c0)^2` (registration §3b).
/// Squared, not `Sqrt`ed, so the family stays `Div`/`Sqrt`-free.
fn squared_distance(a: &mut ExprArena, bx: ExprId, by: ExprId, y: ExprId, c0: f32) -> ExprId {
    let c0_id = a.push_const(c0);
    let dx = a.push_binary(OpKind::Sub, bx, y);
    let dx2 = a.push_binary(OpKind::Mul, dx, dx);
    let dy = a.push_binary(OpKind::Sub, by, c0_id);
    let dy2 = a.push_binary(OpKind::Mul, dy, dy);
    a.push_binary(OpKind::Add, dx2, dy2)
}

/// Draw `n+1` control values ~ U(lo, hi).
fn draw_controls(rng: &mut Lcg, n: usize, lo: f32, hi: f32) -> Vec<f32> {
    (0..=n).map(|_| rng.range(lo, hi)).collect()
}

fn build_bernstein(rng: &mut Lcg, n: usize) -> (ExprArena, ExprId) {
    let mut a = ExprArena::new();
    let t = a.push_var(0);
    let y = a.push_var(1);
    let cx = draw_controls(rng, n, -2.0, 2.0);
    let cy = draw_controls(rng, n, -2.0, 2.0);
    let c0 = rng.range(-2.0, 2.0);
    let bx = bernstein_1d(&mut a, t, n, &cx);
    let by = bernstein_1d(&mut a, t, n, &cy);
    let root = squared_distance(&mut a, bx, by, y, c0);
    (a, root)
}

fn build_casteljau(rng: &mut Lcg, n: usize) -> (ExprArena, ExprId) {
    let mut a = ExprArena::new();
    let t = a.push_var(0);
    let y = a.push_var(1);
    let cx = draw_controls(rng, n, -2.0, 2.0);
    let cy = draw_controls(rng, n, -2.0, 2.0);
    let c0 = rng.range(-2.0, 2.0);
    let px: Vec<ExprId> = cx.iter().map(|&v| a.push_const(v)).collect();
    let py: Vec<ExprId> = cy.iter().map(|&v| a.push_const(v)).collect();
    let bx = de_casteljau(&mut a, &px, t);
    let by = de_casteljau(&mut a, &py, t);
    let root = squared_distance(&mut a, bx, by, y, c0);
    (a, root)
}

/// Fixed bicubic tensor-product patch: z(X,Y) = Σᵢ Σⱼ Pᵢⱼ·Bᵢ(X)·Bⱼ(Y), 16
/// independent heights ~ U(-2,2) (registration §3b). `Bᵢ`/`Bⱼ` are built
/// per (i,j) term rather than shared across terms — same "self-contained,
/// not hand-optimized for sharing" choice as `bernstein_1d`.
fn build_patch(rng: &mut Lcg) -> (ExprArena, ExprId) {
    let mut a = ExprArena::new();
    let x = a.push_var(0);
    let y = a.push_var(1);
    let n = 3usize;
    let b = binom(n);
    let one_x = a.push_const(1.0);
    let omx = a.push_binary(OpKind::Sub, one_x, x);
    let x_pows = powers(&mut a, x, n);
    let omx_pows = powers(&mut a, omx, n);
    let one_y = a.push_const(1.0);
    let omy = a.push_binary(OpKind::Sub, one_y, y);
    let y_pows = powers(&mut a, y, n);
    let omy_pows = powers(&mut a, omy, n);

    let basis =
        |a: &mut ExprArena, pows: &[Option<ExprId>], ompows: &[Option<ExprId>], i: usize| {
            let c = a.push_const(b[i]);
            let mut acc = c;
            if let Some(p) = pows[i] {
                acc = a.push_binary(OpKind::Mul, acc, p);
            }
            if let Some(op) = ompows[n - i] {
                acc = a.push_binary(OpKind::Mul, acc, op);
            }
            acc
        };

    let mut terms = Vec::with_capacity((n + 1) * (n + 1));
    for i in 0..=n {
        let bi = basis(&mut a, &x_pows, &omx_pows, i);
        for j in 0..=n {
            let bj = basis(&mut a, &y_pows, &omy_pows, j);
            let height = rng.range(-2.0, 2.0);
            let p = a.push_const(height);
            let bij = a.push_binary(OpKind::Mul, bi, bj);
            terms.push(a.push_binary(OpKind::Mul, bij, p));
        }
    }
    let root = fold_add(&mut a, &terms);
    (a, root)
}

#[derive(Clone, Copy, Debug)]
enum Form {
    BernsteinCubic,
    BernsteinQuartic,
    CasteljauCubic,
    CasteljauQuartic,
    Patch,
}

const FORMS: [Form; 5] = [
    Form::BernsteinCubic,
    Form::BernsteinQuartic,
    Form::CasteljauCubic,
    Form::CasteljauQuartic,
    Form::Patch,
];

impl Form {
    fn label(self) -> &'static str {
        match self {
            Form::BernsteinCubic | Form::BernsteinQuartic => "bezier-bernstein",
            Form::CasteljauCubic | Form::CasteljauQuartic => "bezier-casteljau",
            Form::Patch => "bezier-patch",
        }
    }

    fn degree(self) -> usize {
        match self {
            Form::BernsteinCubic | Form::CasteljauCubic | Form::Patch => 3,
            Form::BernsteinQuartic | Form::CasteljauQuartic => 4,
        }
    }

    fn build(self, rng: &mut Lcg) -> (ExprArena, ExprId) {
        match self {
            Form::BernsteinCubic => build_bernstein(rng, 3),
            Form::BernsteinQuartic => build_bernstein(rng, 4),
            Form::CasteljauCubic => build_casteljau(rng, 3),
            Form::CasteljauQuartic => build_casteljau(rng, 4),
            Form::Patch => build_patch(rng),
        }
    }
}

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
    let mut rng = Lcg(args.seed);

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
// Oracle validation: an independent scalar-Rust reference for each form,
// cross-checked against `eval_scalar` on the SAME construction code this
// binary uses — this is the check that would catch a wrong formula, which
// `Quarantine` (JIT vs. `eval_scalar` on the SAME, possibly-wrong, arena)
// structurally cannot.
// ============================================================================

#[cfg(test)]
mod oracle_tests {
    use super::*;
    use pixelflow_ir::{BindingTable, eval_scalar};

    fn eval(a: &ExprArena, root: ExprId, x: f32, y: f32) -> f32 {
        eval_scalar(a, root, &[x, y, 0.0, 0.0], &BindingTable::empty())
    }

    fn ref_bernstein_1d(t: f32, b: &[f32], values: &[f32]) -> f32 {
        let n = values.len() - 1;
        (0..=n)
            .map(|i| b[i] * (1.0 - t).powi((n - i) as i32) * t.powi(i as i32) * values[i])
            .sum()
    }

    fn ref_lerp(a: f32, b: f32, t: f32) -> f32 {
        a + (b - a) * t
    }

    fn ref_de_casteljau(points: &[f32], t: f32) -> f32 {
        let mut level = points.to_vec();
        while level.len() > 1 {
            level = level.windows(2).map(|w| ref_lerp(w[0], w[1], t)).collect();
        }
        level[0]
    }

    const SAMPLE_T: [f32; 6] = [-1.5, -0.3, 0.0, 0.37, 1.0, 1.8];
    const SAMPLE_Y: f32 = 0.42;

    fn assert_close(got: f32, want: f32, ctx: &str) {
        let tol = 1e-3 * want.abs().max(1.0);
        assert!(
            (got - want).abs() <= tol,
            "{ctx}: got {got}, want {want} (tol {tol})"
        );
    }

    #[test]
    fn bernstein_cubic_matches_the_scalar_reference() {
        let mut rng = Lcg(1);
        let (a, root) = build_bernstein(&mut rng, 3);
        // Re-derive the same draws to build the reference independently:
        // build_bernstein consumes rng in a fixed, known order (cx, cy, c0).
        let mut rng2 = Lcg(1);
        let cx = draw_controls(&mut rng2, 3, -2.0, 2.0);
        let cy = draw_controls(&mut rng2, 3, -2.0, 2.0);
        let c0 = rng2.range(-2.0, 2.0);
        for &t in &SAMPLE_T {
            let bx = ref_bernstein_1d(t, &BINOM3, &cx);
            let by = ref_bernstein_1d(t, &BINOM3, &cy);
            let want = (bx - SAMPLE_Y).powi(2) + (by - c0).powi(2);
            let got = eval(&a, root, t, SAMPLE_Y);
            assert_close(got, want, &format!("bernstein cubic t={t}"));
        }
    }

    #[test]
    fn bernstein_quartic_matches_the_scalar_reference() {
        let mut rng = Lcg(7);
        let (a, root) = build_bernstein(&mut rng, 4);
        let mut rng2 = Lcg(7);
        let cx = draw_controls(&mut rng2, 4, -2.0, 2.0);
        let cy = draw_controls(&mut rng2, 4, -2.0, 2.0);
        let c0 = rng2.range(-2.0, 2.0);
        for &t in &SAMPLE_T {
            let bx = ref_bernstein_1d(t, &BINOM4, &cx);
            let by = ref_bernstein_1d(t, &BINOM4, &cy);
            let want = (bx - SAMPLE_Y).powi(2) + (by - c0).powi(2);
            let got = eval(&a, root, t, SAMPLE_Y);
            assert_close(got, want, &format!("bernstein quartic t={t}"));
        }
    }

    #[test]
    fn casteljau_cubic_matches_de_casteljau_and_bernstein_agree() {
        let mut rng = Lcg(3);
        let (a, root) = build_casteljau(&mut rng, 3);
        let mut rng2 = Lcg(3);
        let cx = draw_controls(&mut rng2, 3, -2.0, 2.0);
        let cy = draw_controls(&mut rng2, 3, -2.0, 2.0);
        let c0 = rng2.range(-2.0, 2.0);
        for &t in &SAMPLE_T {
            // Two independent scalar references: de Casteljau's own
            // recursion, AND the closed-form Bernstein sum (a mathematical
            // identity for the same control points) — both must agree with
            // the arena.
            let bx_dc = ref_de_casteljau(&cx, t);
            let by_dc = ref_de_casteljau(&cy, t);
            let bx_bern = ref_bernstein_1d(t, &BINOM3, &cx);
            let by_bern = ref_bernstein_1d(t, &BINOM3, &cy);
            assert_close(
                bx_dc,
                bx_bern,
                &format!("de Casteljau/Bernstein identity x t={t}"),
            );
            assert_close(
                by_dc,
                by_bern,
                &format!("de Casteljau/Bernstein identity y t={t}"),
            );
            let want = (bx_dc - SAMPLE_Y).powi(2) + (by_dc - c0).powi(2);
            let got = eval(&a, root, t, SAMPLE_Y);
            assert_close(got, want, &format!("casteljau cubic t={t}"));
        }
    }

    #[test]
    fn casteljau_quartic_matches_the_scalar_reference() {
        let mut rng = Lcg(9);
        let (a, root) = build_casteljau(&mut rng, 4);
        let mut rng2 = Lcg(9);
        let cx = draw_controls(&mut rng2, 4, -2.0, 2.0);
        let cy = draw_controls(&mut rng2, 4, -2.0, 2.0);
        let c0 = rng2.range(-2.0, 2.0);
        for &t in &SAMPLE_T {
            let bx = ref_de_casteljau(&cx, t);
            let by = ref_de_casteljau(&cy, t);
            let want = (bx - SAMPLE_Y).powi(2) + (by - c0).powi(2);
            let got = eval(&a, root, t, SAMPLE_Y);
            assert_close(got, want, &format!("casteljau quartic t={t}"));
        }
    }

    #[test]
    fn patch_matches_the_scalar_reference() {
        let mut rng = Lcg(5);
        let (a, root) = build_patch(&mut rng);
        let mut rng2 = Lcg(5);
        // build_patch draws heights in (i, j) row-major order, 16 total.
        let mut heights = [[0.0f32; 4]; 4];
        for row in &mut heights {
            for h in row.iter_mut() {
                *h = rng2.range(-2.0, 2.0);
            }
        }
        for &tx in &SAMPLE_T {
            for &ty in &[0.1f32, 0.6, 1.2] {
                let want: f32 = (0..=3)
                    .flat_map(|i| (0..=3).map(move |j| (i, j)))
                    .map(|(i, j)| {
                        let bi = BINOM3[i] * (1.0 - tx).powi((3 - i) as i32) * tx.powi(i as i32);
                        let bj = BINOM3[j] * (1.0 - ty).powi((3 - j) as i32) * ty.powi(j as i32);
                        bi * bj * heights[i][j]
                    })
                    .sum();
                let got = eval(&a, root, tx, ty);
                assert_close(got, want, &format!("patch t=({tx},{ty})"));
            }
        }
    }

    #[test]
    fn de_casteljau_reduces_correct_lerp_count() {
        // Registration §3b: 6 lerps for cubic, 10 for quartic, per
        // coordinate. Spot-check the reduction count itself, independent of
        // arena construction.
        assert_eq!(
            {
                let mut count = 0;
                let mut level = vec![0.0f32; 4];
                while level.len() > 1 {
                    level = level
                        .windows(2)
                        .map(|_| {
                            count += 1;
                            0.0
                        })
                        .collect();
                }
                count
            },
            6
        );
        assert_eq!(
            {
                let mut count = 0;
                let mut level = vec![0.0f32; 5];
                while level.len() > 1 {
                    level = level
                        .windows(2)
                        .map(|_| {
                            count += 1;
                            0.0
                        })
                        .collect();
                }
                count
            },
            10
        );
    }

    #[test]
    fn every_admitted_form_is_polynomial_only() {
        let mut rng = Lcg(42);
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
        let mut rng = Lcg(123);
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
        let mut rng_a = Lcg(1);
        let mut rng_b = Lcg(2);
        let (arena_a, root_a) = build_bernstein(&mut rng_a, 3);
        let (arena_b, root_b) = build_bernstein(&mut rng_b, 3);
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
        let mut rng1 = Lcg(99);
        let mut rng2 = Lcg(99);
        let (a1, r1) = build_patch(&mut rng1);
        let (a2, r2) = build_patch(&mut rng2);
        assert_eq!(
            ExactKey::of(&a1, r1),
            ExactKey::of(&a2, r2),
            "same seed must reproduce the identical draw (determinism), and ExactKey must \
             recognize the resulting duplicate"
        );
    }
}
