//! Optimization-quality guards for the glyph winding kernels.
//!
//! The winding kernels are the hottest runtime-composed arenas in the system
//! (every glyph bake evaluates every reachable node per pixel), and their
//! cost is dominated by derivative-bearing subtrees:
//!
//! - `AnalyticalLine`: `d = X − ((Y−y0)·k + x0)` with baked constants, so
//!   `DX(d) = 1` and `DY(d) = −k` — the whole gradient `√(DX²+DY²)` is a
//!   compile-time constant. If the optimizer doesn't fold it, every pixel
//!   pays a sqrt for a number known at bake time.
//! - `AnalyticalQuad`: the value path and both `Dwrt` gradient paths share
//!   the discriminant / `sqrt(disc)` subtrees; losing that sharing multiplies
//!   the per-pixel sqrt count.
//!
//! These tests count surviving operations through the runtime pipeline
//! (`optimize_runtime_arena` → `lower_dwrt`) — the exact stages
//! `Lattice::bake` runs — so a regression in derivative folding or CSE shows
//! up as a hard number, not a benchmark whisper.

use pixelflow_graphics::fonts::ttf_curve_analytical::{AnalyticalLine, AnalyticalQuad};
use pixelflow_ir::arena::{ExprArena, ExprId, ExprNode};
use pixelflow_ir::passes::lower_dwrt_owned;
use pixelflow_ir::OpKind;

/// Count reachable nodes matching `pred` from `root`.
fn count_reachable(arena: &ExprArena, root: ExprId, pred: impl Fn(&ExprNode) -> bool) -> usize {
    let len = arena.nodes_raw().len();
    let mut seen = vec![false; len];
    let mut stack = vec![root];
    let mut count = 0;
    while let Some(id) = stack.pop() {
        if std::mem::replace(&mut seen[id.0 as usize], true) {
            continue;
        }
        if pred(arena.node(id)) {
            count += 1;
        }
        stack.extend(arena.children(id));
    }
    count
}

fn count_op(arena: &ExprArena, root: ExprId, op: OpKind) -> usize {
    count_reachable(arena, root, |n| match n {
        ExprNode::Unary(k, _) => *k == op,
        ExprNode::Binary(k, _, _) => *k == op,
        ExprNode::Ternary(k, _, _, _) => *k == op,
        _ => false,
    })
}

fn total_reachable(arena: &ExprArena, root: ExprId) -> usize {
    count_reachable(arena, root, |_| true)
}

/// Run the same optimization stages `Lattice::bake` runs, then lower any
/// residual `Dwrt` exactly as the compile entries do, and report the final
/// (arena, root) the emitter would actually schedule. Prints per-stage
/// counts so a failure localizes to the stage that dropped the ball.
fn bake_pipeline(arena: &ExprArena, root: ExprId) -> (ExprArena, ExprId) {
    let optimized = pixelflow_search::runtime::optimize_runtime_arena(
        arena,
        root,
        pixelflow_ir::LatticeShape::POINT,
    );
    let (a, r) = optimized
        .as_deref()
        .map(|(a, r)| (a.clone(), *r))
        .unwrap_or_else(|| (arena.clone(), root));
    eprintln!(
        "  post-egraph: total={} sqrt={} dwrt={}",
        total_reachable(&a, r),
        count_op(&a, r, OpKind::Sqrt),
        count_op(&a, r, OpKind::Dwrt),
    );
    let (dl, dr) =
        lower_dwrt_owned(arena, root).expect("dwrt lowering must succeed on winding kernels");
    eprintln!(
        "  lower_dwrt-only baseline: total={} sqrt={}",
        total_reachable(&dl, dr),
        count_op(&dl, dr, OpKind::Sqrt),
    );
    lower_dwrt_owned(&a, r).expect("dwrt lowering must succeed on winding kernels")
}

#[test]
fn line_gradient_folds_to_a_constant() {
    let line = AnalyticalLine::from_points([2.0, 1.0], [10.0, 30.0]).expect("non-degenerate");
    let kernel = line.kernel();
    let (arena, root) = kernel.parts();

    let raw_sqrt = count_op(arena, root, OpKind::Sqrt);
    let raw_dwrt = count_op(arena, root, OpKind::Dwrt);
    let raw_total = total_reachable(arena, root);

    let (opt, opt_root) = bake_pipeline(arena, root);
    let opt_sqrt = count_op(&opt, opt_root, OpKind::Sqrt);
    let opt_dwrt = count_op(&opt, opt_root, OpKind::Dwrt);
    let opt_total = total_reachable(&opt, opt_root);

    eprintln!(
        "line: raw total={raw_total} sqrt={raw_sqrt} dwrt={raw_dwrt} -> \
         optimized total={opt_total} sqrt={opt_sqrt} dwrt={opt_dwrt}"
    );

    assert_eq!(opt_dwrt, 0, "Dwrt must be fully resolved by bake time");
    // DX(d) = 1 and DY(d) = -dx_over_dy (both constants), so
    // sqrt(DX² + DY²) is a compile-time constant: no sqrt may survive.
    assert_eq!(
        opt_sqrt, 0,
        "the line kernel's gradient is a compile-time constant; a surviving \
         sqrt means every pixel of every straight glyph edge pays for a \
         number known at bake time"
    );
}

#[test]
fn quad_shares_the_discriminant_between_value_and_gradient() {
    // A genuinely quadratic segment (control point off the chord).
    let quad = AnalyticalQuad::new([0.0, 0.0], [8.0, 20.0], [16.0, 0.0]);
    let kernel = quad.kernel();
    let (arena, root) = kernel.parts();

    let raw_sqrt = count_op(arena, root, OpKind::Sqrt);
    let raw_dwrt = count_op(arena, root, OpKind::Dwrt);
    let raw_total = total_reachable(arena, root);

    let (opt, opt_root) = bake_pipeline(arena, root);
    let opt_sqrt = count_op(&opt, opt_root, OpKind::Sqrt);
    let opt_dwrt = count_op(&opt, opt_root, OpKind::Dwrt);
    let opt_total = total_reachable(&opt, opt_root);

    eprintln!(
        "quad: raw total={raw_total} sqrt={raw_sqrt} dwrt={raw_dwrt} -> \
         optimized total={opt_total} sqrt={opt_sqrt} dwrt={opt_dwrt}"
    );

    assert_eq!(opt_dwrt, 0, "Dwrt must be fully resolved by bake time");
    // The raw kernel holds ONE sqrt(disc) (shared by both roots) plus the
    // two gradient sqrts (one per root's ramp). Differentiating d(Y) chains
    // through sqrt(disc), whose derivative reuses the SAME sqrt node
    // (d/dY √u = u′/(2√u)) — so a fully-shared result still has exactly
    // three sqrts: sqrt(disc), and the two gradient-magnitude sqrts.
    // Every sqrt beyond that is lost sharing, paid per pixel per curve.
    assert!(
        opt_sqrt <= 3,
        "expected ≤3 surviving sqrts (shared disc + two gradient ramps), \
         got {opt_sqrt}: the value/gradient paths have stopped sharing the \
         discriminant"
    );
}

/// Every op the lowered winding kernels contain must be representable in
/// the e-graph — a gap here silently turns the whole runtime optimization
/// tier into a no-op for glyphs (exactly what happened when `BitAnd`, the
/// Y-range mask combinator, was missing from `op_from_kind`).
#[test]
fn lowered_winding_ops_are_all_egraph_representable() {
    let line = AnalyticalLine::from_points([2.0, 1.0], [10.0, 30.0]).expect("non-degenerate");
    let kernel = line.kernel();
    let (arena, root) = kernel.parts();
    let (lowered, lroot) = lower_dwrt_owned(arena, root).expect("lower");
    let mut missing = std::collections::BTreeSet::new();
    let len = lowered.nodes_raw().len();
    let mut seen = vec![false; len];
    let mut stack = vec![lroot];
    while let Some(id) = stack.pop() {
        if std::mem::replace(&mut seen[id.0 as usize], true) {
            continue;
        }
        let kind = match lowered.node(id) {
            ExprNode::Unary(k, _) => Some(*k),
            ExprNode::Binary(k, _, _) => Some(*k),
            ExprNode::Ternary(k, _, _, _) => Some(*k),
            ExprNode::Param(i) => {
                missing.insert(format!("Param({i})"));
                None
            }
            _ => None,
        };
        if let Some(k) = kind {
            if !pixelflow_search::runtime::is_egraph_representable(k) {
                missing.insert(format!("{k:?}"));
            }
        }
        stack.extend(lowered.children(id));
    }
    assert!(
        missing.is_empty(),
        "ops unconvertible to the e-graph: {missing:?} — the runtime \
         optimizer bails out entirely on any kernel containing them"
    );
}

/// Optimization must preserve a real glyph's coverage within float-
/// reassociation noise. The JIT-vs-interpreter goldens deliberately compare
/// the compiler against the interpreter on the SAME (optimized) arena, so
/// this is the guard that pins optimized-vs-raw — without it, an unsound
/// rewrite would slip past the goldens by corrupting both sides equally.
///
/// Tolerance: reassociation/FMA-fusion re-rounds a long winding sum at the
/// 1e-4 scale (observed); an unsound rule (wrong branch of a root, a lost
/// mask) shifts coverage by O(1).
#[test]
fn optimized_glyph_matches_raw_within_reassociation_noise() {
    use pixelflow_graphics::fonts::Font;
    use pixelflow_ir::binding::BindingTable;
    use pixelflow_ir::eval_scalar;

    const FONT_DATA: &[u8] = include_bytes!("../assets/DejaVuSansMono-Fallback.ttf");
    let font = Font::parse(FONT_DATA).unwrap();

    for ch in ['A', 'O', 'g', '8'] {
        let kernel = font.glyph_kernel_scaled(ch, 32.0).expect("glyph kernel");
        let (arena, root) = kernel.parts();
        let (raw, raw_root) = lower_dwrt_owned(arena, root).expect("lower raw");
        let optimized = pixelflow_search::runtime::optimize_runtime_arena(
            arena,
            root,
            pixelflow_ir::LatticeShape::POINT,
        )
        .expect("glyph arenas must optimize (pure arithmetic + Dwrt + masks)");
        let (opt, opt_root) = (&optimized.0, optimized.1);

        for j in 0..32usize {
            for i in 0..32usize {
                let (x, y) = (i as f32 + 0.5, j as f32 + 0.5);
                let want = eval_scalar(&raw, raw_root, &[x, y], &BindingTable::empty());
                let got = eval_scalar(opt, opt_root, &[x, y], &BindingTable::empty());
                assert!(
                    (want - got).abs() < 1e-3,
                    "{ch}@32: optimized {got} != raw {want} at texel ({i},{j}) — \
                     an optimization changed glyph coverage beyond rounding noise"
                );
            }
        }
    }
}
