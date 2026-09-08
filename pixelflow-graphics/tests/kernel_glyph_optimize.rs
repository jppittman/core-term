//! Optimization-quality guards for the glyph coverage kernels.
//!
//! A glyph kernel is the hottest runtime-composed arena in the system (every
//! bake evaluates every reachable node per pixel), and its cost is dominated
//! by the antialiasing ramps: every edge function `d` is normalised by
//! `‖∇d‖ = √(DX(d)² + DY(d)²)`, and for the affine edge functions —
//! a chord's perpendicular and along-chord distances — both derivatives are
//! constants, so the whole normalisation is a compile-time number. If the
//! optimizer does not fold it, every pixel pays a `sqrt` for a value known
//! at bake time, once per edge.
//!
//! These tests count surviving operations through the runtime pipeline
//! (`optimize_runtime_arena` → `lower_dwrt`) — the exact stages
//! `Lattice::bake` runs — so a regression in derivative folding or CSE shows
//! up as a hard number, not a benchmark whisper.

use pixelflow_graphics::fonts::{loop_blinn, Contour, Font, Outline, Segment};
use pixelflow_ir::arena::{ExprArena, ExprId, ExprNode};
use pixelflow_ir::passes::{expand_refs_owned, lower_dwrt_owned};
use pixelflow_ir::{Kernel, OpKind};

const FONT_DATA: &[u8] = include_bytes!("../assets/DejaVuSansMono-Fallback.ttf");

/// A glyph kernel's arena with its references linked. The winding sum is
/// composed by reference, and a name has no derivative, declares no buffer,
/// and counts as one node — so every count and every lowering below starts
/// from the linked arena, as the pipeline's own first step does.
fn linked(kernel: &Kernel) -> (ExprArena, ExprId) {
    let (arena, root) = kernel.parts();
    expand_refs_owned(arena, root)
}

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
fn bake_pipeline(arena: &ExprArena, root: ExprId, shape: [u32; 2]) -> (ExprArena, ExprId) {
    let optimized = pixelflow_search::runtime::optimize_runtime_arena(
        arena,
        root,
        pixelflow_ir::LatticeShape::new(shape),
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
        lower_dwrt_owned(arena, root).expect("dwrt lowering must succeed on glyph kernels");
    eprintln!(
        "  lower_dwrt-only baseline: total={} sqrt={}",
        total_reachable(&dl, dr),
        count_op(&dl, dr, OpKind::Sqrt),
    );
    lower_dwrt_owned(&a, r).expect("dwrt lowering must succeed on glyph kernels")
}

/// A closed polygon of `n` straight edges: no curves, so every edge function
/// is affine and every gradient a constant.
fn polygon(points: &[[f32; 2]]) -> Outline {
    let segments = (0..points.len())
        .map(|i| Segment::Line {
            from: points[i],
            to: points[(i + 1) % points.len()],
        })
        .collect();
    Outline {
        contours: vec![Contour::new(segments).expect("a polygon's own vertices close the loop")],
    }
}

/// Every affine edge function's gradient folds: a polygon's kernel keeps
/// one `sqrt` per edge — the capsule distance `√(d² + t²)`, which really is
/// per pixel — and not one more. The three `√(DX² + DY²)` per edge (two
/// affine normalisations and, for a curve, its implicit's) are the ones that
/// must vanish.
#[test]
fn affine_edge_gradients_fold_to_constants() {
    let outline = polygon(&[
        [2.0, 1.0],
        [30.0, 4.0],
        [26.0, 28.0],
        [3.0, 20.0],
        [11.0, 9.0],
    ]);
    let glyph = loop_blinn::glyph(&outline);
    let (arena, root) = linked(&glyph.kernel);

    let raw_sqrt = count_op(&arena, root, OpKind::Sqrt);
    let raw_dwrt = count_op(&arena, root, OpKind::Dwrt);
    let raw_total = total_reachable(&arena, root);

    let (opt, opt_root) = bake_pipeline(&arena, root, [32, 32]);
    let opt_sqrt = count_op(&opt, opt_root, OpKind::Sqrt);
    let opt_dwrt = count_op(&opt, opt_root, OpKind::Dwrt);
    let opt_total = total_reachable(&opt, opt_root);

    eprintln!(
        "pentagon: raw total={raw_total} sqrt={raw_sqrt} dwrt={raw_dwrt} -> \
         optimized total={opt_total} sqrt={opt_sqrt} dwrt={opt_dwrt}"
    );

    assert_eq!(opt_dwrt, 0, "Dwrt must be fully resolved by bake time");
    assert!(
        opt_sqrt <= 5,
        "a pentagon's kernel needs one sqrt per edge (the capsule distance); \
         {opt_sqrt} survived, so an affine edge function's gradient — a \
         compile-time constant — is being computed per pixel"
    );
}

/// Every op a glyph kernel contains must be representable in the e-graph —
/// a gap here silently turns the whole runtime optimization tier into a
/// no-op for glyphs (exactly what happened when `BitAnd`, the mask
/// combinator, was missing from `op_from_kind`).
#[test]
fn lowered_glyph_ops_are_all_egraph_representable() {
    let font = Font::parse(FONT_DATA).unwrap();
    let glyph = font.glyph_kernel_scaled('g', 16.0).expect("glyph");
    let (arena, root) = linked(&glyph.kernel);
    let (lowered, lroot) = lower_dwrt_owned(&arena, root).expect("lower");
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
/// 1e-4 scale (observed); an unsound rule (a lost mask, a flipped cone)
/// shifts coverage by O(1).
///
/// **Sizes, not one size.** A rounding difference only *decides* something
/// where a comparison sits on a knife edge, and which rows land on one is a
/// function of the size, so a single size is not a sample of this failure
/// mode. The scanline kernel this replaced had such an edge — the
/// quadratic solver's `disc >= 0` at a curve's vertex row — and 29 texels
/// of `'8'` diverged across four of these sizes for exactly that reason.
/// This kernel has no discriminant, and the set is asserted empty.
const SIZES: [u32; 10] = [7, 9, 11, 13, 15, 17, 19, 21, 23, 32];

#[test]
fn optimized_glyph_matches_raw_within_reassociation_noise() {
    use pixelflow_ir::binding::BindingTable;
    use pixelflow_ir::Evaluator;

    /// Reassociation and FMA fusion re-round a long winding sum at the 1e-4
    /// scale; an unsound rewrite moves coverage by O(1).
    const TOLERANCE: f32 = 1e-3;

    let font = Font::parse(FONT_DATA).unwrap();
    let mut divergences: Vec<String> = Vec::new();

    for size in SIZES {
        for ch in ['A', 'O', 'g', '8'] {
            let glyph = font
                .glyph_kernel_scaled(ch, size as f32)
                .expect("glyph kernel");
            let (arena, root) = linked(&glyph.kernel);
            let (raw, raw_root) = lower_dwrt_owned(&arena, root).expect("lower raw");
            // `glyph.kernel`'s winding sum reads a piece table that travels
            // with the kernel itself; both `eval_scalar` oracles below need
            // it bound, not empty — `lower_dwrt`/`optimize_runtime_arena`
            // restructure the Dwrt and arithmetic subtrees only, never the
            // buffer declarations, so the one slot survives unchanged in
            // both `raw` and `opt`.
            let data: Vec<&[f32]> = raw
                .buffers()
                .iter()
                .map(|decl| {
                    glyph
                        .kernel
                        .buffer_data()
                        .find(|(id, _)| *id == decl.id)
                        .map(|(_, d)| d.as_ref())
                        .expect("glyph kernel carries data for every slot it declares")
                })
                .collect();
            let raw_table = BindingTable::bind(&raw, &data).expect("bind winding table (raw)");
            // The lattice a bake of this glyph would compile at — the
            // extraction is a function of the shape, and the bake's is the one
            // that reaches pixels.
            let extent = size + size / 2;
            let optimized = pixelflow_search::runtime::optimize_runtime_arena(
                &arena,
                root,
                pixelflow_ir::LatticeShape::new([extent, extent]),
            )
            .expect("glyph arenas must optimize (pure arithmetic + Dwrt + masks)");
            let (opt, opt_root) = (&optimized.0, optimized.1);
            let opt_table = BindingTable::bind(opt, &data).expect("bind winding table (opt)");

            // Prepared once per arena: the raw arena is large and un-unrolled,
            // and preparing it per texel was this test's whole cost.
            let raw_eval = Evaluator::new(&raw, raw_root);
            let opt_eval = Evaluator::new(opt, opt_root);
            for j in 0..extent as usize {
                for i in 0..extent as usize {
                    let (x, y) = (i as f32 + 0.5, j as f32 + 0.5);
                    let want = raw_eval.eval(&[x, y], &raw_table);
                    let got = opt_eval.eval(&[x, y], &opt_table);
                    // Before the comparison, not folded into it: `NaN >= x` is
                    // false, so a threshold test *accepts* a non-finite
                    // coverage silently. A collector has to say so itself.
                    assert!(
                        want.is_finite() && got.is_finite(),
                        "{ch}@{size} texel ({i},{j}): non-finite coverage \
                         (raw {want}, optimized {got})"
                    );
                    if (want - got).abs() >= TOLERANCE {
                        divergences.push(format!(
                            "{ch}@{size} texel ({i},{j}): raw {want} vs optimized {got} \
                             (delta {})",
                            got - want
                        ));
                    }
                }
            }
        }
    }

    // Reported together rather than at the first hit: which texels sit on a
    // knife edge is the diagnostic, and one texel does not show it.
    assert!(
        divergences.is_empty(),
        "optimization changed coverage at {} texels:\n{}",
        divergences.len(),
        divergences.join("\n")
    );
}
