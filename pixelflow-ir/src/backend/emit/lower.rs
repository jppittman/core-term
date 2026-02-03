//! IR lowering pass: expand compound ops into primitives.
//!
//! Compound operations (sin, cos, exp, log) are implemented as polynomial
//! approximations using only primitive ops (add, mul, mul_add, etc.).
//! This pass transforms compound IR into primitive IR, so the JIT emitter
//! only needs to handle primitives.
//!
//! The polynomial implementations mirror `compounds.rs` but at the IR level.

#[cfg(feature = "alloc")]
use alloc::boxed::Box;
use crate::expr::Expr;
use crate::kind::OpKind;
use core::f32::consts::{FRAC_PI_2, PI, TAU};

/// Lower an expression tree, expanding compound ops into primitives.
///
/// After lowering, the expression will only contain:
/// - Var, Const
/// - Add, Sub, Mul, Div, Neg
/// - MulAdd (FMA)
/// - Min, Max, Abs
/// - Sqrt, Rsqrt, Recip
/// - Floor, Ceil
/// - Select, comparisons
#[cfg(feature = "alloc")]
pub fn lower(expr: &Expr) -> Expr {
    match expr {
        Expr::Var(idx) => Expr::Var(*idx),
        Expr::Const(val) => Expr::Const(*val),

        Expr::Unary(op, child) => {
            let child = lower(child);
            match op {
                // Compound ops - expand to primitives
                OpKind::Sin => lower_sin(child),
                OpKind::Cos => lower_cos(child),
                OpKind::Exp => lower_exp(child),
                OpKind::Ln => lower_ln(child),

                // Primitives - pass through
                _ => Expr::Unary(*op, Box::new(child)),
            }
        }

        Expr::Binary(op, lhs, rhs) => {
            let lhs = lower(lhs);
            let rhs = lower(rhs);
            // All binary ops are primitives
            Expr::Binary(*op, Box::new(lhs), Box::new(rhs))
        }

        Expr::Ternary(op, a, b, c) => {
            let a = lower(a);
            let b = lower(b);
            let c = lower(c);
            Expr::Ternary(*op, Box::new(a), Box::new(b), Box::new(c))
        }

        Expr::Nary(op, children) => {
            let children = children.iter().map(lower).collect();
            Expr::Nary(*op, children)
        }
    }
}

// =============================================================================
// Compound Op Expansions
// =============================================================================

/// Helper: create constant
#[cfg(feature = "alloc")]
fn c(val: f32) -> Expr {
    Expr::Const(val)
}

/// Helper: binary op
#[cfg(feature = "alloc")]
fn bin(op: OpKind, lhs: Expr, rhs: Expr) -> Expr {
    Expr::Binary(op, Box::new(lhs), Box::new(rhs))
}

/// Helper: a + b
#[cfg(feature = "alloc")]
fn add(a: Expr, b: Expr) -> Expr {
    bin(OpKind::Add, a, b)
}

/// Helper: a - b
#[cfg(feature = "alloc")]
fn sub(a: Expr, b: Expr) -> Expr {
    bin(OpKind::Sub, a, b)
}

/// Helper: a * b
#[cfg(feature = "alloc")]
fn mul(a: Expr, b: Expr) -> Expr {
    bin(OpKind::Mul, a, b)
}

/// Helper: a * b + c (FMA)
#[cfg(feature = "alloc")]
fn mul_add(a: Expr, b: Expr, c: Expr) -> Expr {
    Expr::Ternary(OpKind::MulAdd, Box::new(a), Box::new(b), Box::new(c))
}

/// Helper: floor(x)
#[cfg(feature = "alloc")]
fn floor(x: Expr) -> Expr {
    Expr::Unary(OpKind::Floor, Box::new(x))
}

/// Helper: min(a, b)
#[cfg(feature = "alloc")]
fn min(a: Expr, b: Expr) -> Expr {
    bin(OpKind::Min, a, b)
}

/// Helper: max(a, b)
#[cfg(feature = "alloc")]
fn max(a: Expr, b: Expr) -> Expr {
    bin(OpKind::Max, a, b)
}

/// Sin via Chebyshev polynomial.
///
/// Mirrors `Compounds::sin` from compounds.rs.
#[cfg(feature = "alloc")]
fn lower_sin(x: Expr) -> Expr {
    // Range reduction to [-π, π]
    // k = floor(x * (1/TAU) + 0.5)
    let two_pi_inv = 1.0 / TAU;
    let k = floor(add(mul(x.clone(), c(two_pi_inv)), c(0.5)));

    // x_reduced = x - k * TAU
    let x_reduced = sub(x, mul(k, c(TAU)));

    // t = x_reduced * (1/PI), normalized to [-1, 1]
    let t = mul(x_reduced, c(1.0 / PI));

    // t² for polynomial
    let t2 = mul(t.clone(), t.clone());

    // Chebyshev coefficients for sin(π*t)
    let c1 = 3.14159265358979_f32;  // π
    let c3 = -5.16771278004997_f32;
    let c5 = 2.55016403987734_f32;
    let c7 = -0.599264528932149_f32;

    // Horner's method: p = c7*t² + c5, then p*t² + c3, then p*t² + c1
    let p = mul_add(c(c7), t2.clone(), c(c5));
    let p = mul_add(p, t2.clone(), c(c3));
    let p = mul_add(p, t2, c(c1));

    // result = t * p
    mul(t, p)
}

/// Cos via sin(x + π/2).
#[cfg(feature = "alloc")]
fn lower_cos(x: Expr) -> Expr {
    // cos(x) = sin(x + π/2)
    lower_sin(add(x, c(FRAC_PI_2)))
}

/// Exp via exp2: e^x = 2^(x * log2(e))
#[cfg(feature = "alloc")]
fn lower_exp(x: Expr) -> Expr {
    const LOG2_E: f32 = 1.4426950408889634;
    lower_exp2(mul(x, c(LOG2_E)))
}

/// Exp2: 2^x via polynomial.
///
/// Mirrors `Compounds::exp2`.
#[cfg(feature = "alloc")]
fn lower_exp2(x: Expr) -> Expr {
    // Clamp to avoid overflow
    let x = max(min(x, c(126.0)), c(-126.0));

    // Split into integer and fractional parts
    let xi = floor(x.clone());
    let xf = sub(x, xi.clone());

    // Polynomial coefficients for 2^xf on [0, 1)
    let c0 = 1.0_f32;
    let c1 = 0.6931471805599453_f32;  // ln(2)
    let c2 = 0.24022650695910071_f32;
    let c3 = 0.05550410866482157_f32;
    let c4 = 0.009618129107628477_f32;
    let c5 = 0.0013333558146428443_f32;

    // Horner's method
    let p = mul_add(c(c5), xf.clone(), c(c4));
    let p = mul_add(p, xf.clone(), c(c3));
    let p = mul_add(p, xf.clone(), c(c2));
    let p = mul_add(p, xf.clone(), c(c1));
    let p = mul_add(p, xf, c(c0));

    // For the integer exponent, we'd need bit manipulation.
    // Simplified: use pow approximation. For now, just return polynomial part.
    // TODO: proper 2^xi via ldexp/bit manipulation
    //
    // The polynomial p approximates 2^xf for xf in [0,1).
    // Full result needs p * 2^xi, which requires integer exponent handling.
    // For JIT, we could add an ldexp op, or emit the bit manipulation.
    //
    // For now, this is the polynomial part only - works for small x.
    p
}

/// Natural log via log2: ln(x) = log2(x) * ln(2)
#[cfg(feature = "alloc")]
fn lower_ln(x: Expr) -> Expr {
    const LN_2: f32 = 0.6931471805599453;
    mul(lower_log2(x), c(LN_2))
}

/// Log2 via polynomial.
///
/// Simplified version - full impl needs exponent extraction via bit ops.
#[cfg(feature = "alloc")]
fn lower_log2(x: Expr) -> Expr {
    // Simplified: polynomial for log2(1+t) where t = x - 1
    // This only works well for x near 1. Full impl needs range reduction.
    let t = sub(x, c(1.0));

    // Polynomial coefficients for log2(1+t)
    let c1 = 1.4426950408889634_f32;  // 1/ln(2)
    let c2 = -0.7213475204444817_f32;
    let c3 = 0.4808983469629878_f32;
    let c4 = -0.3606737602222408_f32;

    // Horner's method
    let p = mul_add(c(c4), t.clone(), c(c3));
    let p = mul_add(p, t.clone(), c(c2));
    let p = mul_add(p, t.clone(), c(c1));

    mul(p, t)
}

#[cfg(test)]
#[cfg(feature = "alloc")]
mod tests {
    use super::*;
    use crate::backend::emit::needs;

    #[test]
    fn test_lower_primitives_unchanged() {
        // Primitives should pass through unchanged
        let expr = Expr::Binary(
            OpKind::Add,
            Box::new(Expr::Var(0)),
            Box::new(Expr::Var(1)),
        );
        let lowered = lower(&expr);
        assert!(matches!(lowered, Expr::Binary(OpKind::Add, _, _)));
    }

    #[test]
    fn test_lower_sin_expands() {
        // Sin should expand to a larger expression
        let expr = Expr::Unary(OpKind::Sin, Box::new(Expr::Var(0)));
        let lowered = lower(&expr);

        // Should NOT be a Sin anymore
        assert!(!matches!(lowered, Expr::Unary(OpKind::Sin, _)));

        // Should have grown significantly (polynomial expansion)
        let nodes = lowered.node_count();
        assert!(nodes > 10, "lowered sin has {} nodes", nodes);

        // Check register requirement doesn't exceed limit
        let regs = needs(&lowered);
        assert!(regs < 20, "lowered sin needs {} registers (too many!)", regs);
    }

    #[test]
    fn test_lower_cos_expands() {
        let expr = Expr::Unary(OpKind::Cos, Box::new(Expr::Var(0)));
        let lowered = lower(&expr);
        assert!(!matches!(lowered, Expr::Unary(OpKind::Cos, _)));
    }

    #[test]
    fn test_lower_sin_register_count() {
        // Debug: print the lowered sin structure
        let expr = Expr::Unary(OpKind::Sin, Box::new(Expr::Var(0)));
        let lowered = lower(&expr);

        eprintln!("Lowered sin node count: {}", lowered.node_count());
        eprintln!("Lowered sin depth: {}", lowered.depth());
        eprintln!("Lowered sin register label: {}", needs(&lowered));
    }
}
