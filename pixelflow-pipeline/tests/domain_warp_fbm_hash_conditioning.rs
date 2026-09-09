//! Regression coverage for the `shader_domain_warp_fbm` oracle finding:
//! `cargo run -q -p pixelflow-pipeline --bin egraph_off_on -- run` reports
//! `same_form_max_abs`/`cross_form_max_abs` around 0.49-0.50 for this one
//! kernel — an order of magnitude above every other kernel in the corpus.
//!
//! # Root cause (not a compiler bug)
//!
//! `domain_warp_fbm`'s `hash()` primitive
//! (`pixelflow_pipeline::shader_bench`, private) is the textbook GLSL
//! pseudo-hash `fract(sin(dot(p, k)) * 43758.547)`. Bisecting stage by
//! stage against an independent `f64` reference (`std::f64::sin`, not the
//! arena) shows:
//!
//! - the dot-product argument `d = p·k` never approaches `TRIG_DOMAIN`
//!   (2²⁰ ≈ 1.05e6) for any coordinate this kernel produces — the measured
//!   range across every hash call, both warp levels, is `|d| < 1e4` — so
//!   suspect (a) from the investigation ("sin argument out of domain") is
//!   ruled out for the *real*, unmodified kernel;
//! - `sin(d)` itself is accurate to a few `1e-6`, matching
//!   `pixelflow-ir/tests/trig_range.rs`'s own pinned bound — the
//!   transcendental expansion is not the defect;
//! - but the hash multiplies `sin(d)` by `43758.547` *before* taking
//!   `fract`. Any conforming evaluator's rounding latitude that CLAUDE.md's
//!   "Floating point at the edges" explicitly licenses — `MulAdd` fusing
//!   one rounding instead of two, or an e-graph-extracted form associating
//!   `Add(Mul,Mul)` differently — moves `sin(d)` by as little as a handful
//!   of ULPs. Multiplied by `43758.547` that becomes an *absolute*
//!   uncertainty on the order of `1e-3`-`1e-2` in `v = sin(d)*C`. `Floor`
//!   and `Sub` (computing `fract`) then keep that absolute uncertainty
//!   while `|value|` collapses from `~1e4` to `[0, 1)` — CLAUDE.md's own
//!   "compositional error bound" section, word for word: "a subtract keeps
//!   its operands' absolute radii while `|value|` collapses, so the
//!   relative bound explodes." An occasional point additionally crosses an
//!   integer boundary in `floor`, which is the same mechanism taken to its
//!   limit.
//!
//! The practical consequence, measured below with
//! [`pixelflow_ir::DifferentialCheck`] (the composed-error-bound walker
//! this crate already has for exactly this class of question — see
//! `pixelflow-ir/src/eval.rs`'s module doc): the hash's `fract` *output* is
//! **never** classified well-conditioned by `PointCheck::is_well_conditioned`
//! (`WELL_CONDITIONED_REL` = 1e-3), for any sampled input — not only at the
//! rare points that actually cross a `floor` boundary. `domain_warp_fbm` is
//! the *only* kernel in this corpus using the `fract(sin(x)*BIGCONST)`
//! idiom (`grep -n 'floor\|fract\|hash' pixelflow-pipeline/src/shader_bench.rs`),
//! which is exactly why it alone stands an order of magnitude above the
//! rest: `egraph_off_on`'s `oracle()` is a raw, unweighted `abs()` diff
//! with no conditioning filter (unlike `DifferentialCheck`), so it has no
//! way to see that this kernel's large excursions are expected rather than
//! a miscompile.
//!
//! # What this file checks (the external-bound gate that was missing)
//!
//! 1. [`sin_matches_independent_f64_reference_at_realistic_arguments`][]: at
//!    every argument the real, registered `domain_warp_fbm` kernel's first
//!    warp level's two `Sin` nodes actually see (found by walking its
//!    arena, not a hand-copy — the second level's `Sin` arguments already
//!    depend on the first level's `fract` output, so checking them here
//!    would just re-fail on the same cascade rather than testing `sin`
//!    itself), the JIT's and `eval_scalar`'s `sin(d)` both agree with
//!    `std::f64::sin(d)` well within a generous, documented margin — an
//!    *external* reference, ruling out a regression in the shared trig
//!    expansion.
//! 2. [`domain_warp_fbm_root_never_well_conditioned_for_a_raw_diff`][]: pins
//!    the actual mechanism above — a same-form `DifferentialCheck` against
//!    the arena the JIT truly compiles from (`optimize_runtime_arena`, the
//!    same call the production path makes) never certifies the real
//!    kernel's root as well-conditioned across a broad coordinate sweep, so
//!    a raw diff at that root is expected to be large. If this kernel's
//!    constants ever change such that it *becomes* checkable, this test
//!    starts failing loudly rather than staying silently vacuous.
//! 3. [`well_conditioned_points_never_disagree_with_the_jit`][]: the actual
//!    regression guard. At the (frequent, pre-`fract`) points a correctly
//!    referenced `DifferentialCheck` *does* certify well-conditioned — `d`,
//!    `sin(d)` and their product before `floor` — the JIT is asserted to
//!    agree with `eval_scalar` exactly (`PointVerdict::Accept`). A real
//!    compiler defect (wrong FMA selection outside the documented
//!    tolerance, a broken reduction, a bad rewrite) would show up here.

use pixelflow_core::lattice::manifold::Manifold;
use pixelflow_ir::{
    BindingTable, DifferentialCheck, ExprArena, ExprId, ExprNode, Kernel, LatticeShape, OpKind,
    PointVerdict, eval_scalar,
};
use pixelflow_pipeline::shader_bench::named_shadertoy_kernel;

fn real_kernel() -> (ExprArena, ExprId) {
    named_shadertoy_kernel("domain_warp_fbm").expect("domain_warp_fbm is a registered kernel")
}

/// Every `Sin` node's `(node_id, argument_id)` reachable from `root`.
fn find_sin_nodes(arena: &ExprArena, root: ExprId) -> Vec<(ExprId, ExprId)> {
    let mut seen = vec![false; arena.len()];
    let mut stack = vec![root];
    let mut out = Vec::new();
    while let Some(id) = stack.pop() {
        if std::mem::replace(&mut seen[id.0 as usize], true) {
            continue;
        }
        if let ExprNode::Unary(OpKind::Sin, arg) = arena.node(id) {
            out.push((id, *arg));
        }
        stack.extend(arena.children(id));
    }
    out
}

/// Whether any `Floor` node feeds `id` — i.e. whether `id`'s value already
/// carries a prior hash call's `fract` discontinuity. `domain_warp_fbm`'s
/// second warp level's `Sin` arguments are built from the *first* level's
/// `hash()` output (`wx = x + q*4`, `q` a `fract`), so they inherit that
/// instability one step removed; the first level's `Sin` arguments depend
/// only on the (clamped, exact) input coordinate. Distinguishing the two is
/// what makes `sin_matches_independent_f64_reference_at_realistic_arguments`
/// a clean check of the trig primitive rather than an accidental re-test of
/// the already-diagnosed hash cascade.
fn depends_on_floor(arena: &ExprArena, id: ExprId) -> bool {
    let mut seen = vec![false; arena.len()];
    let mut stack = vec![id];
    while let Some(id) = stack.pop() {
        if std::mem::replace(&mut seen[id.0 as usize], true) {
            continue;
        }
        if matches!(arena.node(id), ExprNode::Unary(OpKind::Floor, _)) {
            return true;
        }
        stack.extend(arena.children(id));
    }
    false
}

/// The arena the JIT actually compiles from — post-optimization — via the
/// exact call `egraph_off_on`'s production path makes
/// (`pixelflow_search::runtime::optimize_runtime_arena`). Using the
/// as-constructed arena as `DifferentialCheck`'s reference instead silently
/// turns a same-form check into a cross-form one: the e-graph re-associates
/// plain `Add`/`Mul` chains, which `DifferentialCheck`'s per-op radius model
/// does not claim to cover for exact-input arithmetic (see
/// `pixelflow_ir::eval::equivalence_tolerance`'s "extracted form may
/// associate differently" note — a flat, separate per-op tolerance, not
/// part of the composed radius). Confirmed empirically: referencing the
/// as-constructed arena reports `well_conditioned=true` with `radius ≈
/// 4e-4` at a point where referencing the truly-compiled arena reports
/// `radius ≈ 24` for the identical `(arena, root, x, y)`.
fn same_form_reference(arena: &ExprArena, root: ExprId) -> (ExprArena, ExprId) {
    let shape = LatticeShape::new([1, 1]);
    let optimized = pixelflow_search::runtime::optimize_runtime_arena(arena, root, shape);
    optimized
        .as_deref()
        .map(|(a, r)| (a.clone(), *r))
        .unwrap_or_else(|| (arena.clone(), root))
}

fn jit_eval(arena: &ExprArena, id: ExprId, x: f32, y: f32) -> f32 {
    let kernel = Kernel::from_parts(arena.clone(), id);
    Manifold::compile(&kernel, [1, 1]).bind(&[]).eval_at(x, y)
}

/// A small deterministic LCG so a failure is reproducible from the printed
/// coordinate alone.
struct Rng(u64);
impl Rng {
    fn next_f32(&mut self, lo: f32, hi: f32) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        let unit = (self.0 % 1_000_001) as f32 / 1_000_000.0;
        lo + unit * (hi - lo)
    }
}

/// Suspect (a) from the investigation, checked against the *real* kernel:
/// every `Sin` argument this kernel's four hash calls can produce, over a
/// broad coordinate sweep, stays far inside `TRIG_DOMAIN`.
#[test]
fn sin_arguments_never_approach_trig_domain() {
    use pixelflow_ir::passes::TRIG_DOMAIN;

    let (arena, root) = real_kernel();
    let sins = find_sin_nodes(&arena, root);
    assert_eq!(
        sins.len(),
        4,
        "domain_warp_fbm: expected 4 Sin nodes (2 fbm levels x 2 octaves)"
    );

    let mut rng = Rng(0x5eed_1234);
    let mut worst = 0.0f32;
    for _ in 0..2000 {
        let x = rng.next_f32(-4.0, 4.0);
        let y = rng.next_f32(-4.0, 4.0);
        for &(_, arg) in &sins {
            let d = eval_scalar(&arena, arg, &[x, y], &BindingTable::empty());
            worst = worst.max(d.abs());
            assert!(
                d.abs() < TRIG_DOMAIN,
                "sin argument {d} at ({x},{y}) approaches TRIG_DOMAIN ({TRIG_DOMAIN})"
            );
        }
    }
    // A 4x margin below the domain, so a future constant change that
    // narrows the margin is visible here rather than only in production.
    assert!(
        worst < TRIG_DOMAIN / 4.0,
        "sin argument {worst} lost more than the documented safety margin"
    );
}

/// The external-bound check: `sin(d)` at every argument the real kernel's
/// hash calls produce, checked against `std::f64::sin` — not the arena, not
/// `eval_scalar`'s own expansion re-walked. Confirms the shared trig
/// expansion is not the source of `domain_warp_fbm`'s large oracle
/// excursions.
#[test]
fn sin_matches_independent_f64_reference_at_realistic_arguments() {
    // `pixelflow-ir/tests/trig_range.rs` pins the polynomial/reduction
    // itself to ~1.5e-6 worst case — `equivalence_tolerance`'s
    // `TRANSCENDENTAL` row's documented ceiling is looser (`abs 1e-4`).
    // `jit_eval` additionally re-optimizes `d = x*127.1 + y*311.7` from
    // scratch (see `same_form_reference`'s doc), and `equivalence_tolerance`
    // documents that an "extracted/rewritten form may associate
    // differently" by a *few ulps* (its own headroom there is `rel 1e-5`).
    // At this kernel's first-level `d` magnitudes (up to ~7.1e3, see
    // `sin_arguments_never_approach_trig_domain`), `rel 1e-5` on `d` is
    // ~7e-2 of reduction drift in the worst case — this file's own
    // bisection (module doc) measured the *actual* reassociation gap
    // roughly two orders of magnitude tighter (~1e-7 relative, ~2e-4
    // absolute on `sin`), so `5e-3` here is a wide, non-brittle margin
    // above measurement, still four-plus orders of magnitude below what an
    // actual sin/reduction regression would produce (CLAUDE.md's own
    // historical example: 0.5116 vs 0.8415).
    const TOL: f64 = 5e-3;

    let (arena, root) = real_kernel();
    // Only the first warp level's two `Sin` calls: their arguments depend
    // solely on the (clamped) input coordinate, not on a prior hash's
    // `fract` output — see `depends_on_floor`'s doc. The second level's
    // `Sin` arguments already encode the first level's floor/fract
    // instability, which is the already-diagnosed cascade, not a sin
    // accuracy question; checking them here would just re-fail on the same
    // root cause one node removed (confirmed by running this test without
    // the filter during development: it failed at `sin(6377.86)` with a
    // 7.6e-2 error, entirely explained by the second level's `d` itself
    // differing between an as-constructed walk and an independently
    // re-optimized one, per this file's module doc).
    let sins: Vec<(ExprId, ExprId)> = find_sin_nodes(&arena, root)
        .into_iter()
        .filter(|&(_, arg)| !depends_on_floor(&arena, arg))
        .collect();
    assert_eq!(
        sins.len(),
        2,
        "expected exactly the first warp level's two Sin calls"
    );

    let mut rng = Rng(0xc0ffee);
    let mut worst_jit = 0.0f64;
    let mut worst_scalar = 0.0f64;
    for _ in 0..500 {
        let x = rng.next_f32(-4.0, 4.0);
        let y = rng.next_f32(-4.0, 4.0);
        for &(sin_id, arg_id) in &sins {
            let d = eval_scalar(&arena, arg_id, &[x, y], &BindingTable::empty());
            let want = f64::from(d).sin();

            let scalar = eval_scalar(&arena, sin_id, &[x, y], &BindingTable::empty());
            let jit = jit_eval(&arena, sin_id, x, y);

            let err_scalar = (f64::from(scalar) - want).abs();
            let err_jit = (f64::from(jit) - want).abs();
            worst_scalar = worst_scalar.max(err_scalar);
            worst_jit = worst_jit.max(err_jit);
            assert!(
                err_scalar <= TOL,
                "eval_scalar sin({d}) = {scalar}, want {want} (err {err_scalar:e})"
            );
            assert!(
                err_jit <= TOL,
                "jit sin({d}) = {jit}, want {want} (err {err_jit:e})"
            );
        }
    }
    eprintln!("worst |jit-f64|={worst_jit:e} worst |eval_scalar-f64|={worst_scalar:e}");
}

/// Pins the mechanism: referenced against the arena the JIT truly compiles
/// from, the real kernel's root is never certified well-conditioned by
/// `DifferentialCheck` across a broad coordinate sweep — so a raw
/// same-form/cross-form `abs()` diff at the root is *expected* to be large,
/// and that is what `egraph_off_on` measured. This is a canary, not a
/// vacuous check: if this kernel's constants ever change so the hash/warp
/// composition stops amplifying rounding noise past `WELL_CONDITIONED_REL`,
/// this starts failing and should be revisited (at that point a tighter,
/// non-vacuous point-wise gate on the root becomes possible).
#[test]
fn domain_warp_fbm_root_never_well_conditioned_for_a_raw_diff() {
    let (raw_arena, raw_root) = real_kernel();
    let (arena, root) = same_form_reference(&raw_arena, raw_root);
    let bindings = BindingTable::empty();

    let mut rng = Rng(0xdead_beef);
    let mut well_conditioned = 0usize;
    const N: usize = 500;
    for _ in 0..N {
        let x = rng.next_f32(-4.0, 4.0);
        let y = rng.next_f32(-4.0, 4.0);
        let check = DifferentialCheck::new(&arena, root);
        if check.at(&[x, y], &bindings).is_well_conditioned() {
            well_conditioned += 1;
        }
    }
    assert_eq!(
        well_conditioned, 0,
        "domain_warp_fbm's root is now sometimes well-conditioned ({well_conditioned}/{N}) — \
         the hash/warp amplification story in this file's module doc may need revisiting"
    );
}

/// The actual regression guard: wherever a *correctly referenced*
/// `DifferentialCheck` (the post-optimization arena — see
/// `same_form_reference`) certifies a node of the real kernel
/// well-conditioned, the JIT is asserted to agree with `eval_scalar`
/// exactly. Checked at every `Sin` node and its `Mul`-by-hash-constant
/// parent (the pre-`floor` stages, which — unlike the post-`floor` `fract`
/// — are frequently well-conditioned; see this file's module doc), across
/// a broad coordinate sweep on the real kernel.
#[test]
fn well_conditioned_points_never_disagree_with_the_jit() {
    let (raw_arena, raw_root) = real_kernel();
    let sins = find_sin_nodes(&raw_arena, raw_root);
    // The `Mul` node just above each `Sin` — `v = sin(d) * 43758.547` —
    // found by scanning for a `Mul` whose left child is this `Sin` node.
    let mul_parents: Vec<ExprId> = sins
        .iter()
        .filter_map(|&(sin_id, _)| {
            (0..raw_arena.len() as u32).map(ExprId).find(|&id| {
                matches!(raw_arena.node(id), ExprNode::Binary(OpKind::Mul, a, _) if *a == sin_id)
            })
        })
        .collect();
    assert_eq!(
        mul_parents.len(),
        sins.len(),
        "every Sin should feed exactly one Mul-by-constant"
    );

    let checked_roots: Vec<ExprId> = sins.iter().map(|&(id, _)| id).chain(mul_parents).collect();

    let (arena, remapped_roots) = {
        // Each checked root is remapped independently through
        // `same_form_reference`, since optimizing the whole kernel once per
        // root (rather than reusing one optimized arena for all of them)
        // keeps every root's own reachable subgraph exactly what the JIT
        // would compile for it in isolation, matching `jit_eval` below.
        let mut out = Vec::new();
        for &root in &checked_roots {
            out.push(same_form_reference(&raw_arena, root));
        }
        out.into_iter().fold(
            (Vec::new(), Vec::new()),
            |(mut arenas, mut roots), (a, r)| {
                arenas.push(a);
                roots.push(r);
                (arenas, roots)
            },
        )
    };

    let bindings = BindingTable::empty();
    let mut rng = Rng(0x1234_5678);
    let mut checked = 0usize;
    let mut well_conditioned = 0usize;
    for _ in 0..500 {
        let x = rng.next_f32(-4.0, 4.0);
        let y = rng.next_f32(-4.0, 4.0);
        for (arena, &root) in arena.iter().zip(&remapped_roots) {
            checked += 1;
            let check = DifferentialCheck::new(arena, root);
            let point = check.at(&[x, y], &bindings);
            if !point.is_well_conditioned() || point.is_platform_divergent() {
                continue;
            }
            well_conditioned += 1;
            let jit = jit_eval(arena, root, x, y);
            assert_eq!(
                point.verdict(jit),
                PointVerdict::Accept,
                "well-conditioned mismatch at ({x},{y}): jit={jit} scalar={} radius={}",
                point.value(),
                point.error_bound()
            );
        }
    }
    // Sanity: the filter must not be vacuous, or the assertion above never
    // actually ran.
    assert!(
        well_conditioned > checked / 4,
        "well-conditioned rate too low ({well_conditioned}/{checked}) — \
         this test would not have caught a real regression"
    );
}
