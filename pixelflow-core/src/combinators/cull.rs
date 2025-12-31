//! # Cull Utilities
//!
//! Gradient-based culling utilities using Taylor bounds with Hessian.
//!
//! ## The Math
//!
//! For a function f(x,y) evaluated at center c with half-tile size h:
//!
//! ```text
//! f(c + δ) ≈ f(c) + ∇f·δ + ½δᵀHδ
//!
//! Error bound: |f - f(c)| ≤ |∇f|·h + ½||H||·h²
//! ```
//!
//! For quadratic curves like u² - v, the Hessian is constant, so the bound is exact!
//!
//! ## Usage Patterns
//!
//! These utilities help with:
//!
//! 1. **Tile-based culling** (before per-pixel evaluation):
//!    Evaluate at tile center, use bounds to skip entire tile.
//!
//! 2. **Newton-Raphson refinement**:
//!    Use gradient for NR iteration toward zero crossing.
//!
//! The actual culling decision happens at the rasterizer level, not per-pixel.

/// For quadratic Bézier u² - v, compute the exact Hessian bound.
///
/// The Hessian of f = u² - v where u = ua*x + ub*y + uc is:
/// H = [[2ua², 2ua*ub], [2ua*ub, 2ub²]]
///
/// This is constant (doesn't depend on x, y), so R₂ = (ua*hx + ub*hy)² is exact!
#[inline(always)]
pub fn quadratic_hessian_bound(ua: f32, ub: f32, hx: f32, hy: f32) -> f32 {
    let u_dev = ua.abs() * hx + ub.abs() * hy;
    u_dev * u_dev
}
