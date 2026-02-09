//! # Chebyshev-Based Trigonometric Functions
//!
//! SIMD-vectorized sin, cos, atan2 using polynomial approximations.
//! All operations vectorize across all SIMD lanes simultaneously, replacing
//! per-lane libm scalar calls with parallel polynomial evaluation.
//!
//! **Accuracy**: ~1e-6 (sufficient for graphics/harmonics).
//! **Speed**: 5-10x faster than per-lane libm on SIMD backends.
//!
//! # Implementation Note
//!
//! This module builds AST graphs using operators, enabling automatic FMA fusion
//! and other optimizations. The graph is evaluated at the return boundary.

use crate::Field;
use crate::{Manifold, ManifoldExt};

/// The standard 4D Field domain.
type Field4 = (Field, Field, Field, Field);

/// Evaluate a manifold graph to Field.
/// Since Field is a constant manifold, coordinates don't matter.
#[inline(always)]
fn eval<M: Manifold<Field4, Output = Field>>(m: M) -> Field {
    let zero = Field::from(0.0);
    m.eval((zero, zero, zero, zero))
}

// ============================================================================
// Constants (Computed at Compile Time)
// ============================================================================

const PI: f32 = core::f32::consts::PI;
const PI_2: f32 = core::f32::consts::FRAC_PI_2;
const PI_4: f32 = core::f32::consts::FRAC_PI_4;

/// Precomputed: 1 / π (computed at compile time)
const fn inv_pi() -> f32 {
    1.0 / PI
}

const PI_INV: f32 = inv_pi();

/// Range reduction: Map angle x to [-π/2, π/2].
///
/// Returns (reduced_x, sign_flip).
/// x = k*π + reduced_x
/// sign_flip is 1.0 if k is even, -1.0 if k is odd.
#[inline(always)]
fn range_reduce_half_pi(x: Field) -> (Field, Field) {
    // k = round(x / π) = floor(x/π + 0.5)
    let k = eval((x * Field::from(PI_INV) + Field::from(0.5)).floor());

    // reduced_x = x - k * π
    let reduced_x = eval(x - k * Field::from(PI));

    // Determine if k is odd.
    // k is integer-valued float.
    // is_odd = abs((k/2 - floor(k/2)) * 2)
    // if k is even, k/2 is integer, frac is 0. is_odd = 0.
    // if k is odd, k/2 has .5, frac is 0.5. is_odd = 1.
    // Evaluate half_k to Field to avoid move issues with AST nodes
    let half_k = eval(k * Field::from(0.5));
    let is_odd = eval((half_k - half_k.floor()) * Field::from(2.0)).abs();

    // sign = 1 - 2 * is_odd (1 if even, -1 if odd)
    let sign = eval(Field::from(1.0) - Field::from(2.0) * is_odd);

    (reduced_x, sign)
}

/// Taylor approximation for sin(x) on [-π/2, π/2].
///
/// Uses 6-term Taylor series (degree 11).
/// Accuracy: ~1e-6.
#[inline(always)]
pub(crate) fn cheby_sin(x: Field) -> Field {
    let (x, sign) = range_reduce_half_pi(x);

    // Taylor coefficients for sin(x):
    // x - x^3/6 + x^5/120 - x^7/5040 + x^9/362880 - x^11/39916800
    const C1: f32 = 1.0;
    const C3: f32 = -1.0 / 6.0;
    const C5: f32 = 1.0 / 120.0;
    const C7: f32 = -1.0 / 5040.0;
    const C9: f32 = 1.0 / 362880.0;
    const C11: f32 = -1.0 / 39916800.0;

    let x2 = eval(x * x);

    // Horner's method: x * (C1 + x2 * (C3 + x2 * (C5 + x2 * (C7 + x2 * (C9 + x2 * C11)))))
    let poly = eval(
        ((((Field::from(C11) * x2 + Field::from(C9)) * x2 + Field::from(C7)) * x2 + Field::from(C5)) * x2 + Field::from(C3)) * x2 + Field::from(C1)
    );

    eval(poly * x * sign)
}

/// Taylor approximation for cos(x) on [-π/2, π/2].
///
/// Uses 6-term Taylor series (degree 10).
/// 1 - x^2/2 + x^4/24 - x^6/720 + x^8/40320 - x^10/3628800
#[inline(always)]
pub(crate) fn cheby_cos(x: Field) -> Field {
    let (x, sign) = range_reduce_half_pi(x);

    // Taylor coefficients for cos(x):
    const C0: f32 = 1.0;
    const C2: f32 = -1.0 / 2.0;
    const C4: f32 = 1.0 / 24.0;
    const C6: f32 = -1.0 / 720.0;
    const C8: f32 = 1.0 / 40320.0;
    const C10: f32 = -1.0 / 3628800.0;

    let x2 = eval(x * x);

    // Horner's: ((((C10 * x2 + C8) * x2 + C6) * x2 + C4) * x2 + C2) * x2 + C0
    let poly = eval(
        ((((Field::from(C10) * x2 + Field::from(C8)) * x2 + Field::from(C6)) * x2 + Field::from(C4)) * x2 + Field::from(C2)) * x2 + Field::from(C0)
    );

    eval(poly * sign)
}

/// Approximation for atan2(y, x).
///
/// Uses polynomial approximation on [0, 0.414] with range reduction.
#[inline(always)]
pub(crate) fn cheby_atan2(y: Field, x: Field) -> Field {
    let ax = x.abs();
    let ay = y.abs();

    // u = min(ax, ay), v = max(ax, ay)
    let u = eval(ax.min(ay));
    let v = eval(ax.max(ay));
    let v_safe = eval(v.max(Field::from(1e-20)));

    // Range reduction:
    // If u > v * tan(π/8) (approx 0.414), use r = (v-u)/(v+u) and result is π/4 - atan(r).
    // Else use r = u/v and result is atan(r).
    // This maps argument to [0, 0.414] where Taylor series converges rapidly.

    let threshold = eval(v_safe * Field::from(0.41421356));
    let use_reduction = u.gt(threshold);

    // Compute r based on reduction choice
    let num_reduced = eval(v - u);
    let den_reduced = eval(v + u);

    // Select numerator and denominator
    let num = use_reduction.select(num_reduced, u);
    let den = use_reduction.select(den_reduced, v_safe);
    let r = eval(num / den);
    let r2 = eval(r * r);

    // Taylor series for atan(r) on [0, 0.414] (degree 11)
    // r - r^3/3 + r^5/5 - r^7/7 + r^9/9 - r^11/11
    const C1: f32 = 1.0;
    const C3: f32 = -1.0 / 3.0;
    const C5: f32 = 1.0 / 5.0;
    const C7: f32 = -1.0 / 7.0;
    const C9: f32 = 1.0 / 9.0;
    const C11: f32 = -1.0 / 11.0;

    let poly = eval(
        ((((Field::from(C11) * r2 + Field::from(C9)) * r2 + Field::from(C7)) * r2 + Field::from(C5)) * r2 + Field::from(C3)) * r2 + Field::from(C1)
    );
    let atan_r = eval(poly * r);

    // Adjust result based on reduction: π/4 - atan(r) or atan(r)
    // bias = reduction ? π/4 : 0
    // sign = reduction ? -1 : 1
    // result = bias + sign * atan_r
    let bias = use_reduction.select(Field::from(PI_4), Field::from(0.0));
    let sign = use_reduction.select(Field::from(-1.0), Field::from(1.0));
    let theta_0 = eval(bias + sign * atan_r);

    // Now map back to full circle based on x, y quadrants
    // If |y| > |x| (v=ay, u=ax), we computed atan(ax/ay) or reduced version.
    // Result should be π/2 - theta_0.
    // If |x| >= |y|, result is theta_0.
    let y_gt_x = ay.gt(ax);
    let theta = y_gt_x.select(Field::from(PI_2) - theta_0.clone(), theta_0);

    // Reconstruct full angle based on signs of x, y.
    // If x < 0: theta = π - theta
    let x_neg = x.lt(Field::from(0.0));
    let theta_x = x_neg.select(Field::from(PI) - theta.clone(), theta);

    // If y < 0: theta = -theta
    let y_neg = y.lt(Field::from(0.0));
    let result = y_neg.select(-theta_x.clone(), theta_x);

    eval(result)
}

// Tests are integrated via spherical harmonics in combinators/spherical.rs
// The Chebyshev approximations are verified through their correctness
// in computing surface normals and spherical harmonic coefficients.

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_approx_eq(a: f32, b: f32, tol: f32) {
        let diff = (a - b).abs();
        if diff > tol {
            panic!("assertion failed: `(left approx right)`\n  left: `{:?}`,\n right: `{:?}`,\n  diff: `{:?}` > `{:?}`", a, b, diff, tol);
        }
    }

    #[test]
    fn cheby_sin_should_match_std_precision_within_tolerance() {
        // Test range [-2PI, 2PI]
        let steps = 1000;
        for i in 0..steps {
            let t = (i as f32 / steps as f32) * 4.0 * PI - 2.0 * PI;
            let val = Field::from(t);
            let res = cheby_sin(val);
            let mut buf = [0.0f32; crate::PARALLELISM];
            res.store(&mut buf);

            let expected = t.sin();
            // Tolerance: 1e-6
            assert_approx_eq(buf[0], expected, 1e-6);
        }
    }

    #[test]
    fn cheby_cos_should_match_std_precision_within_tolerance() {
        // Test range [-2PI, 2PI]
        let steps = 1000;
        for i in 0..steps {
            let t = (i as f32 / steps as f32) * 4.0 * PI - 2.0 * PI;
            let val = Field::from(t);
            let res = cheby_cos(val);
            let mut buf = [0.0f32; crate::PARALLELISM];
            res.store(&mut buf);

            let expected = t.cos();
            assert_approx_eq(buf[0], expected, 1e-6);
        }
    }

    #[test]
    fn cheby_atan2_should_match_std_precision_within_tolerance() {
        // Test grid covering all quadrants
        let steps = 50;
        for y_i in 0..steps {
            for x_i in 0..steps {
                let y = (y_i as f32 / steps as f32) * 20.0 - 10.0;
                let x = (x_i as f32 / steps as f32) * 20.0 - 10.0;

                // Skip near origin where atan2 is unstable
                if x.abs() < 1e-5 && y.abs() < 1e-5 { continue; }

                let y_f = Field::from(y);
                let x_f = Field::from(x);
                let res = cheby_atan2(y_f, x_f);
                let mut buf = [0.0f32; crate::PARALLELISM];
                res.store(&mut buf);

                let expected = y.atan2(x);
                // 1e-5 tolerance for atan2 approximation (accumulated errors)
                assert_approx_eq(buf[0], expected, 1e-5);
            }
        }
    }
}
