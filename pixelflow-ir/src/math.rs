//! Generic mathematical implementations for SIMD backends.
//!
//! This module provides high-level implementations of transcendental functions
//! (sin, cos, atan2, etc.) using basic SIMD operations. These serve as
//! default implementations for the `SimdOps` trait, ensuring 1-to-1 mapping
//! between IR operations and backend execution.

use crate::backend::SimdOps;
use core::f32::consts::{PI, FRAC_PI_2, TAU};

/// 1 / π
const PI_INV: f32 = 1.0 / PI;
/// 1 / 2π
const TWO_PI_INV: f32 = 1.0 / TAU;

/// Range reduction: Map angle x to [-π, π].
/// x' = x - 2π * round(x / 2π)
#[inline(always)]
fn range_reduce_pi<T: SimdOps>(x: T) -> T {
    // k = round(x / 2π)
    let scaled = x * T::splat(TWO_PI_INV);
    let rounded = (scaled + T::splat(0.5)).simd_floor();
    
    // x' = x - 2π * k
    x - rounded * T::splat(TAU)
}

/// Chebyshev approximation for sin(x) on [-π, π].
#[inline(always)]
pub fn sin<T: SimdOps>(x: T) -> T {
    let x = range_reduce_pi(x);
    let t = x * T::splat(PI_INV); // Normalize to [-1, 1]
    
    // Chebyshev coefficients
    let c1 = T::splat(1.6719970703125);
    let c3 = T::splat(-0.645963541666667);
    let c5 = T::splat(0.079689450);
    let c7 = T::splat(-0.0046817541);

    let t2 = t * t;
    
    // Horner's method: (((C7*t2 + C5)*t2 + C3)*t2 + C1)*t
    // Using mul_add for FMA
    let p = c7.mul_add(t2, c5);
    let p = p.mul_add(t2, c3);
    let p = p.mul_add(t2, c1);
    
    p * t
}

/// Chebyshev approximation for cos(x) on [-π, π].
#[inline(always)]
pub fn cos<T: SimdOps>(x: T) -> T {
    let x = range_reduce_pi(x);
    let t = x * T::splat(PI_INV);
    
    let c0 = T::splat(1.5707963267948966);
    let c2 = T::splat(-2.467401341);
    let c4 = T::splat(0.609469381);
    let c6 = T::splat(-0.038854038);

    let t2 = t * t;
    
    // ((C6*t2 + C4)*t2 + C2)*t2 + C0
    let p = c6.mul_add(t2, c4);
    let p = p.mul_add(t2, c2);
    
    p.mul_add(t2, c0)
}

/// Tangent via sin/cos.
#[inline(always)]
pub fn tan<T: SimdOps>(x: T) -> T {
    sin(x) / cos(x)
}

/// Chebyshev approximation for atan2(y, x).
#[inline(always)]
pub fn atan2<T: SimdOps>(y: T, x: T) -> T {
    let r = y / x;
    let r_abs = r.simd_abs();
    
    let c1 = T::splat(0.999999999);
    let c3 = T::splat(-0.333333333);
    let c5 = T::splat(0.2);
    let c7 = T::splat(-0.142857143);
    
    let t = r_abs;
    let t2 = t * t;
    
    let p = c7.mul_add(t2, c5);
    let p = p.mul_add(t2, c3);
    let p = p.mul_add(t2, c1);
    let atan_approx = p * t;
    
    // Handle |r| > 1: atan(r) = pi/2 - atan(1/r)
    let mask_large = r_abs.cmp_gt(T::splat(1.0));
    let atan_large = T::splat(FRAC_PI_2) - T::splat(1.0).div(r_abs) * atan_approx; // Re-approx? No, uses same approx logic on inverted input implies re-eval?
    // Wait, the original code reused atan_approx? No.
    // Original code: `let atan_large = Field::from(PI_2) - (Field::from(1.0) / r_abs) * atan_approx.clone();`
    // This looks wrong if atan_approx was computed on r. 
    // Usually for |r|>1 you recompute poly on 1/r.
    // But let's stick to porting what was there for now, or fix it?
    // Actually, `atan_approx` computed on `r` when `r > 1` is garbage (diverges).
    // So we effectively select `atan_large` which should be computed correctly.
    // The original code was likely relying on `Manifold` lazy eval?
    // "Use ManifoldExt's select which builds AST" -> Yes.
    // Here we are executing immediately.
    // We must compute both branches or use select on inputs?
    // To avoid divergence, we should select inputs first.
    // But `r` is valid.
    
    // Correct logic for SIMD execution:
    // 1. input = select(mask_large, 1/r_abs, r_abs)
    // 2. compute poly(input)
    // 3. result = select(mask_large, pi/2 - poly, poly)
    
    let input = T::simd_select(mask_large, T::splat(1.0) / r_abs, r_abs);
    let i2 = input * input;
    let p2 = c7.mul_add(i2, c5);
    let p2 = p2.mul_add(i2, c3);
    let p2 = p2.mul_add(i2, c1);
    let poly = p2 * input;
    
    let atan_val = T::simd_select(mask_large, T::splat(FRAC_PI_2) - poly, poly);
    
    // Sign corrections
    let sign_y = y.simd_abs().div(y); // NaN if y=0. Should use copysign logic.
    // T::splat(1.0).copysign(y)? SimdOps doesn't have copysign.
    // fallback: select(y < 0, -1, 1).
    let mask_neg_y = y.cmp_lt(T::splat(0.0));
    let sign_y = T::simd_select(mask_neg_y, T::splat(-1.0), T::splat(1.0));
    
    let atan_signed = atan_val * sign_y;
    
    let mask_neg_x = x.cmp_lt(T::splat(0.0));
    let correction = T::splat(PI) * sign_y;
    
    T::simd_select(mask_neg_x, atan_signed - correction, atan_signed)
}

/// Chebyshev approximation for asin(x) on [-1, 1].
///
/// Uses the identity: for |x| > 0.5, asin(x) = π/2 - 2*asin(sqrt((1-x)/2))
/// This improves accuracy near the boundaries where the derivative is steep.
#[inline(always)]
pub fn asin<T: SimdOps>(x: T) -> T {
    let abs_x = x.simd_abs();

    // For |x| <= 0.5, use direct polynomial approximation
    // Coefficients for asin(x) ≈ x + c3*x³ + c5*x⁵ + c7*x⁷
    let c3 = T::splat(0.166666666666667); // 1/6
    let c5 = T::splat(0.075);             // 3/40
    let c7 = T::splat(0.044642857);       // 15/336
    let c9 = T::splat(0.030381944);       // 35/1152

    let x2 = abs_x * abs_x;

    // Horner's method for small |x|
    let p_small = c9.mul_add(x2, c7);
    let p_small = p_small.mul_add(x2, c5);
    let p_small = p_small.mul_add(x2, c3);
    let asin_small = abs_x.mul_add(p_small * x2, abs_x);

    // For |x| > 0.5, use identity: asin(x) = π/2 - 2*asin(sqrt((1-x)/2))
    let half = T::splat(0.5);
    let one = T::splat(1.0);
    let t = ((one - abs_x) * half).simd_sqrt();
    let t2 = t * t;

    let p_large = c9.mul_add(t2, c7);
    let p_large = p_large.mul_add(t2, c5);
    let p_large = p_large.mul_add(t2, c3);
    let asin_t = t.mul_add(p_large * t2, t);
    let asin_large = T::splat(FRAC_PI_2) - asin_t - asin_t;

    // Select based on |x| > 0.5
    let mask_large = abs_x.cmp_gt(half);
    let result = T::simd_select(mask_large, asin_large, asin_small);

    // Restore sign
    let mask_neg = x.cmp_lt(T::splat(0.0));
    T::simd_select(mask_neg, T::splat(0.0) - result, result)
}

/// Chebyshev approximation for acos(x) on [-1, 1].
///
/// Uses identity: acos(x) = π/2 - asin(x)
#[inline(always)]
pub fn acos<T: SimdOps>(x: T) -> T {
    T::splat(FRAC_PI_2) - asin(x)
}

/// Power function: base^exp.
///
/// Computed as exp2(exp * log2(base)).
/// For negative bases, returns NaN (consistent with IEEE 754).
#[inline(always)]
pub fn pow<T: SimdOps>(base: T, exp: T) -> T {
    // pow(base, exp) = 2^(exp * log2(base))
    // This handles positive bases correctly.
    // For base <= 0, log2 returns NaN/undefined, which propagates.
    (exp * base.log2()).exp2()
}

/// Hypotenuse: sqrt(x² + y²).
///
/// Computes the Euclidean distance from origin to (x, y).
#[inline(always)]
pub fn hypot<T: SimdOps>(x: T, y: T) -> T {
    (x * x + y * y).simd_sqrt()
}
