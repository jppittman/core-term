//! The two schedules for a polynomial, as arena builders.
//!
//! A polynomial is a *value*: `p(x) = Σ aᵢ xⁱ`. Horner and Estrin are two
//! *schedules* for that value — same coefficients, same function, different
//! dependency graph. Horner is a chain of `n` fused multiply-adds, each
//! waiting on the last; Estrin is a balanced tree of the same `n` FMAs over a
//! `log₂ n` chain of squarings, so its critical path is `O(log n)` where
//! Horner's is `O(n)`, at the cost of `log₂ n` extra multiplies and a wider
//! live set.
//!
//! Everything in the compiler that prices an expression prices its *ops*, not
//! its *shape* (`CostModel::latency_prior` sums node costs), so the two forms
//! are indistinguishable to extraction except that Estrin looks strictly
//! worse. Whether the machine agrees is an empirical question, which is what
//! `examples/horner_vs_estrin.rs` measures.
//!
//! Both builders emit `OpKind::MulAdd` for every coefficient step, matching
//! what `pixelflow_ir::passes`'s `horner_step` emits for the transcendental
//! expansions — so the Horner arm here IS production's shape, not a
//! restatement of it.

use pixelflow_ir::{ExprArena, ExprId, OpKind};

/// Which schedule to emit for the same coefficient list.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolyForm {
    /// `a₀ + x(a₁ + x(a₂ + …))` — one `MulAdd` per coefficient, all serial.
    Horner,
    /// Balanced tree over the powers `x², x⁴, x⁸, …` — the same count of
    /// `MulAdd`s plus one `Mul` per power level, in a `O(log n)`-deep DAG.
    Estrin,
}

impl PolyForm {
    /// Short lowercase name, for table headers and JSONL records.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            PolyForm::Horner => "horner",
            PolyForm::Estrin => "estrin",
        }
    }
}

/// Emit `Σ coeffs[i]·xⁱ` into `arena` under `form`, returning its root.
///
/// `coeffs` is **ascending** in degree (`coeffs[0]` is the constant term),
/// matching `pixelflow_ir::passes::EXP2_POLY` and friends.
///
/// # Panics
///
/// Panics on an empty coefficient list: the degree-0 polynomial is a constant,
/// which has no schedule to compare and is never what a caller meant.
#[must_use]
pub fn build(arena: &mut ExprArena, form: PolyForm, coeffs: &[f32], x: ExprId) -> ExprId {
    assert!(!coeffs.is_empty(), "poly::build: empty coefficient list");
    match form {
        PolyForm::Horner => horner(arena, coeffs, x),
        PolyForm::Estrin => estrin(arena, coeffs, x),
    }
}

/// `a₀ + x(a₁ + x(a₂ + …))`, highest degree down — one `MulAdd` per step.
fn horner(arena: &mut ExprArena, coeffs: &[f32], x: ExprId) -> ExprId {
    let mut acc = arena.push_const(coeffs[coeffs.len() - 1]);
    for &c in coeffs.iter().rev().skip(1) {
        let c = arena.push_const(c);
        acc = arena.push_ternary(OpKind::MulAdd, acc, x, c);
    }
    acc
}

/// Estrin's scheme: pair adjacent coefficients into `aᵢ + x·aᵢ₊₁`, then fold
/// the pairs together against `x²`, `x⁴`, `x⁸`, … until one node is left.
///
/// An odd element at the end of a level is carried up unchanged rather than
/// padded with a zero coefficient — a `MulAdd` against a zero addend is still
/// a real instruction, and the folder cannot remove it (`x·0` is not `0` for
/// non-finite `x`).
fn estrin(arena: &mut ExprArena, coeffs: &[f32], x: ExprId) -> ExprId {
    // Level 0: the linear pairs. `MulAdd(a, b, c)` is `a·b + c`.
    let mut level: Vec<ExprId> = coeffs
        .chunks(2)
        .map(|pair| match pair {
            [lo, hi] => {
                let lo = arena.push_const(*lo);
                let hi = arena.push_const(*hi);
                arena.push_ternary(OpKind::MulAdd, hi, x, lo)
            }
            [lo] => arena.push_const(*lo),
            _ => unreachable!("chunks(2) yields 1 or 2 elements"),
        })
        .collect();

    let mut power = x;
    while level.len() > 1 {
        power = arena.push_binary(OpKind::Mul, power, power);
        level = level
            .chunks(2)
            .map(|pair| match pair {
                [lo, hi] => arena.push_ternary(OpKind::MulAdd, *hi, power, *lo),
                [lo] => *lo,
                _ => unreachable!("chunks(2) yields 1 or 2 elements"),
            })
            .collect();
    }
    level[0]
}

/// Longest path from `root` to a leaf, weighted by `cost` — the DAG's
/// critical path, which is what the machine's out-of-order engine is bounded
/// by when consecutive evaluations cannot overlap.
///
/// The sum of the same weights over the same nodes is what
/// `CostModel::latency_prior` charges. Printing both is the point: they are
/// the same table read two ways, and they disagree about Estrin.
#[must_use]
pub fn critical_path(arena: &ExprArena, root: ExprId, cost: impl Fn(OpKind) -> f64) -> f64 {
    let mut memo: Vec<Option<f64>> = vec![None; arena.len()];
    // Explicit stack: a polynomial DAG is O(degree) deep, and recursion here
    // would blow the stack for the same reason the degree sweep exists.
    let mut stack = vec![(root, false)];
    while let Some((id, expanded)) = stack.pop() {
        let idx = id.0 as usize;
        if memo[idx].is_some() {
            continue;
        }
        if !expanded {
            stack.push((id, true));
            for child in arena.children(id) {
                if memo[child.0 as usize].is_none() {
                    stack.push((child, false));
                }
            }
            continue;
        }
        let kind = arena.kind(id);
        let own = match kind {
            OpKind::Var | OpKind::Const | OpKind::Buffer => 0.0,
            k => cost(k),
        };
        let deepest_child = arena
            .children(id)
            .map(|c| memo[c.0 as usize].expect("children resolved before parent"))
            .fold(0.0f64, f64::max);
        memo[idx] = Some(own + deepest_child);
    }
    memo[root.0 as usize].expect("root resolved")
}

#[cfg(all(test, feature = "training"))]
mod tests {
    use super::*;
    use pixelflow_ir::binding::BindingTable;
    use pixelflow_ir::eval_scalar;

    fn coeffs(n: usize) -> Vec<f32> {
        // Alternating, decaying — a well-conditioned polynomial on [0, 1],
        // so a disagreement between the forms is a scheduling bug and not
        // catastrophic cancellation.
        (0..n)
            .map(|i| {
                let sign = if i.is_multiple_of(2) { 1.0 } else { -1.0 };
                sign * 0.75f32.powi(i as i32) / (i as f32 + 1.0)
            })
            .collect()
    }

    /// The two schedules must compute the same function. Not bit-identical:
    /// they round in a different order, which is exactly the property being
    /// traded, so the check is a relative band and not an equality.
    #[test]
    fn estrin_agrees_with_horner() {
        for n in 1..=33 {
            let cs = coeffs(n);
            let mut arena = ExprArena::new();
            let x = arena.push_var(0);
            let h = build(&mut arena, PolyForm::Horner, &cs, x);
            let e = build(&mut arena, PolyForm::Estrin, &cs, x);
            let bindings = BindingTable::empty();
            for step in 0..=20 {
                let xv = step as f32 / 20.0;
                let vars = [xv, 0.0, 0.0, 0.0];
                let hv = eval_scalar(&arena, h, &vars, &bindings);
                let ev = eval_scalar(&arena, e, &vars, &bindings);
                let tol = 1e-5 * hv.abs().max(1e-3);
                assert!(
                    (hv - ev).abs() <= tol,
                    "degree {n} at x={xv}: horner {hv} vs estrin {ev} (tol {tol})"
                );
            }
        }
    }

    /// Estrin buys depth with width: strictly more nodes, strictly shorter
    /// critical path, for every degree past the trivial ones. This is the
    /// structural claim the benchmark then prices — if it ever stops holding,
    /// the builder is wrong and the timings mean nothing.
    #[test]
    fn estrin_is_shallower_and_wider_than_horner() {
        let unit = |_k: OpKind| 1.0;
        for n in 6..=33 {
            let cs = coeffs(n);
            let mut ha = ExprArena::new();
            let hx = ha.push_var(0);
            let h = build(&mut ha, PolyForm::Horner, &cs, hx);
            let mut ea = ExprArena::new();
            let ex = ea.push_var(0);
            let e = build(&mut ea, PolyForm::Estrin, &cs, ex);

            let hd = critical_path(&ha, h, unit);
            let ed = critical_path(&ea, e, unit);
            assert!(
                ed < hd,
                "degree {n}: estrin depth {ed} not below horner {hd}"
            );

            assert!(
                op_nodes(&ea, e) > op_nodes(&ha, h),
                "degree {n}: estrin should cost extra multiplies for its powers"
            );
        }
    }

    fn op_nodes(arena: &ExprArena, root: ExprId) -> usize {
        let mut visited = vec![false; arena.len()];
        let mut stack = vec![root];
        let mut n = 0;
        while let Some(id) = stack.pop() {
            if visited[id.0 as usize] {
                continue;
            }
            visited[id.0 as usize] = true;
            if !matches!(arena.kind(id), OpKind::Var | OpKind::Const | OpKind::Buffer) {
                n += 1;
            }
            stack.extend(arena.children(id));
        }
        n
    }
}
