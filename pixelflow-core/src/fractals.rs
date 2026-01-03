//! Fractal manifolds: Mandelbrot, Julia, and friends.
//!
//! Fractals are the quintessential demonstration of iterative computation.
//! This module implements them as manifolds, using SIMD-parallel iteration
//! with per-lane retirement for efficiency.
//!
//! # Design Philosophy
//!
//! These manifolds follow the "Compute" eigenshader pattern - they implement
//! `Manifold` directly rather than composing from simpler combinators because:
//!
//! 1. Complex number arithmetic requires tracking two values (real, imaginary)
//! 2. We need to track iteration count alongside the complex state
//! 3. Per-lane retirement requires careful mask management
//!
//! The output is a normalized iteration count (0.0 = inside set, 1.0 = escaped immediately).
//! Use `Map` or color combinators to turn this into colors.
//!
//! # Example
//!
//! ```ignore
//! use pixelflow_core::fractals::Mandelbrot;
//! use pixelflow_graphics::ColorCube;
//! use pixelflow_core::combinators::At;
//!
//! // Create Mandelbrot with 256 max iterations
//! let mandelbrot = Mandelbrot::new(256);
//!
//! // The output is iteration count / max_iterations (0.0 to 1.0)
//! // Map to a color gradient using the iteration value as red channel
//! let colored = At {
//!     inner: ColorCube,
//!     x: mandelbrot,  // Red = iteration fraction
//!     y: 0.2,         // Green
//!     z: 0.1,         // Blue
//!     w: 1.0,         // Alpha
//! };
//! ```

use crate::numeric::Numeric;
use crate::{Field, Manifold};

/// Mandelbrot set: z_{n+1} = z_n² + c, where c = (x, y).
///
/// For each coordinate (x, y), iterates until |z| > 2 (escape) or max_iterations.
/// Returns normalized iteration count: `iterations / max_iterations`.
///
/// - 0.0 means the point escaped on the first iteration
/// - 1.0 means the point never escaped (inside the set)
///
/// For the classic Mandelbrot view, use coordinates roughly in [-2.5, 1.0] × [-1.5, 1.5].
#[derive(Clone, Copy, Debug)]
pub struct Mandelbrot {
    /// Maximum number of iterations before declaring a point "inside" the set.
    pub max_iterations: u32,
}

impl Mandelbrot {
    /// Create a new Mandelbrot manifold with the given iteration limit.
    ///
    /// Higher values give more detail at deep zooms but cost more computation.
    /// Common values: 64 (fast), 256 (balanced), 1024 (detailed).
    #[inline]
    pub const fn new(max_iterations: u32) -> Self {
        Self { max_iterations }
    }
}

impl Manifold for Mandelbrot {
    type Output = Field;

    #[inline(always)]
    fn eval_raw(&self, x: Field, y: Field, _z: Field, _w: Field) -> Field {
        // c = x + iy (the coordinate is the complex parameter)
        let cr = x;
        let ci = y;

        // z starts at 0
        let mut zr = Field::from(0.0f32);
        let mut zi = Field::from(0.0f32);

        // Track iteration count per lane
        let mut iterations = Field::from(0.0f32);
        let one = Field::from(1.0f32);
        let four = Field::from(4.0f32);
        let two = Field::from(2.0f32);

        // Track which lanes are still iterating (haven't escaped)
        // active = all-1s mask initially
        let zero = Field::from(0.0f32);
        let mut active = zero.ge(zero); // 0 >= 0 is always true -> all-1s mask

        for _ in 0..self.max_iterations {
            // |z|² = zr² + zi²
            let zr_sq = zr.raw_mul(zr);
            let zi_sq = zi.raw_mul(zi);
            let magnitude_sq = zr_sq.raw_add(zi_sq);

            // Check escape: |z|² > 4 means |z| > 2
            let escaped = magnitude_sq.gt(four);

            // Retire escaped lanes: active = active AND NOT escaped
            active = Field::select_raw(escaped, zero, active);

            // Early exit if all lanes have escaped
            if !active.any() {
                break;
            }

            // z = z² + c
            // z² = (zr² - zi²) + i(2·zr·zi)
            let new_zr = zr_sq.raw_sub(zi_sq).raw_add(cr);
            let new_zi = two.raw_mul(zr).raw_mul(zi).raw_add(ci);

            // Only update active lanes
            zr = Field::select_raw(active, new_zr, zr);
            zi = Field::select_raw(active, new_zi, zi);

            // Increment iteration count for active lanes
            iterations = Field::select_raw(active, iterations.raw_add(one), iterations);
        }

        // Normalize: points inside the set have iterations = max_iterations
        // Return 1.0 - (iterations / max) so inside=0 (black) and edge=bright
        let max = Field::from(self.max_iterations as f32);
        one.raw_sub(iterations.raw_div(max))
    }
}

/// Julia set: z_{n+1} = z_n² + c, where z₀ = (x, y) and c is fixed.
///
/// Unlike Mandelbrot where c varies and z₀ = 0, Julia uses a fixed c
/// and z₀ = (x, y). Each c value produces a different fractal.
///
/// Returns normalized iteration count: `iterations / max_iterations`.
///
/// # Famous c values:
/// - (-0.7, 0.27015) - Classic "rabbit" shape
/// - (-0.8, 0.156) - Dendrite pattern
/// - (-0.4, 0.6) - Spirals
/// - (0.285, 0.01) - Seahorse valley
#[derive(Clone, Copy, Debug)]
pub struct Julia {
    /// Real part of the c parameter.
    pub c_real: f32,
    /// Imaginary part of the c parameter.
    pub c_imag: f32,
    /// Maximum iterations before declaring a point "inside".
    pub max_iterations: u32,
}

impl Julia {
    /// Create a new Julia set with the given c parameter and iteration limit.
    #[inline]
    pub const fn new(c_real: f32, c_imag: f32, max_iterations: u32) -> Self {
        Self {
            c_real,
            c_imag,
            max_iterations,
        }
    }

    /// Classic "rabbit" Julia set.
    #[inline]
    pub const fn rabbit(max_iterations: u32) -> Self {
        Self::new(-0.7, 0.27015, max_iterations)
    }

    /// Dendrite Julia set.
    #[inline]
    pub const fn dendrite(max_iterations: u32) -> Self {
        Self::new(-0.8, 0.156, max_iterations)
    }

    /// Spiral Julia set.
    #[inline]
    pub const fn spiral(max_iterations: u32) -> Self {
        Self::new(-0.4, 0.6, max_iterations)
    }

    /// Seahorse valley Julia set.
    #[inline]
    pub const fn seahorse(max_iterations: u32) -> Self {
        Self::new(0.285, 0.01, max_iterations)
    }
}

impl Manifold for Julia {
    type Output = Field;

    #[inline(always)]
    fn eval_raw(&self, x: Field, y: Field, _z: Field, _w: Field) -> Field {
        // c is fixed
        let cr = Field::from(self.c_real);
        let ci = Field::from(self.c_imag);

        // z₀ = (x, y) - the coordinate is the initial complex value
        let mut zr = x;
        let mut zi = y;

        // Track iteration count per lane
        let mut iterations = Field::from(0.0f32);
        let one = Field::from(1.0f32);
        let four = Field::from(4.0f32);
        let two = Field::from(2.0f32);

        // Track which lanes are still iterating
        let zero = Field::from(0.0f32);
        let mut active = zero.ge(zero);

        for _ in 0..self.max_iterations {
            // |z|² = zr² + zi²
            let zr_sq = zr.raw_mul(zr);
            let zi_sq = zi.raw_mul(zi);
            let magnitude_sq = zr_sq.raw_add(zi_sq);

            // Check escape
            let escaped = magnitude_sq.gt(four);
            active = Field::select_raw(escaped, zero, active);

            if !active.any() {
                break;
            }

            // z = z² + c
            let new_zr = zr_sq.raw_sub(zi_sq).raw_add(cr);
            let new_zi = two.raw_mul(zr).raw_mul(zi).raw_add(ci);

            zr = Field::select_raw(active, new_zr, zr);
            zi = Field::select_raw(active, new_zi, zi);

            iterations = Field::select_raw(active, iterations.raw_add(one), iterations);
        }

        // Normalize
        let max = Field::from(self.max_iterations as f32);
        one.raw_sub(iterations.raw_div(max))
    }
}

/// Burning Ship fractal: z_{n+1} = (|Re(z)| + i|Im(z)|)² + c
///
/// A variant of Mandelbrot that takes absolute values before squaring,
/// creating a distinctive "burning ship" shape.
#[derive(Clone, Copy, Debug)]
pub struct BurningShip {
    /// Maximum iterations.
    pub max_iterations: u32,
}

impl BurningShip {
    /// Create a new Burning Ship fractal with the given iteration limit.
    #[inline]
    pub const fn new(max_iterations: u32) -> Self {
        Self { max_iterations }
    }
}

impl Manifold for BurningShip {
    type Output = Field;

    #[inline(always)]
    fn eval_raw(&self, x: Field, y: Field, _z: Field, _w: Field) -> Field {
        let cr = x;
        // Flip Y so the ship is right-side up
        let ci = Field::from(0.0f32).raw_sub(y);

        let mut zr = Field::from(0.0f32);
        let mut zi = Field::from(0.0f32);

        let mut iterations = Field::from(0.0f32);
        let one = Field::from(1.0f32);
        let four = Field::from(4.0f32);
        let two = Field::from(2.0f32);

        let zero = Field::from(0.0f32);
        let mut active = zero.ge(zero);

        for _ in 0..self.max_iterations {
            let zr_sq = zr.raw_mul(zr);
            let zi_sq = zi.raw_mul(zi);
            let magnitude_sq = zr_sq.raw_add(zi_sq);

            let escaped = magnitude_sq.gt(four);
            active = Field::select_raw(escaped, zero, active);

            if !active.any() {
                break;
            }

            // Take absolute values before squaring (the key difference)
            let abs_zr = zr.abs();
            let abs_zi = zi.abs();

            // z = (|zr| + i|zi|)² + c
            let new_zr = abs_zr.raw_mul(abs_zr).raw_sub(abs_zi.raw_mul(abs_zi)).raw_add(cr);
            let new_zi = two.raw_mul(abs_zr).raw_mul(abs_zi).raw_add(ci);

            zr = Field::select_raw(active, new_zr, zr);
            zi = Field::select_raw(active, new_zi, zi);

            iterations = Field::select_raw(active, iterations.raw_add(one), iterations);
        }

        let max = Field::from(self.max_iterations as f32);
        one.raw_sub(iterations.raw_div(max))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PARALLELISM;

    #[test]
    fn mandelbrot_origin_is_inside() {
        let m = Mandelbrot::new(100);
        // Origin (0, 0) is inside the Mandelbrot set
        let result = m.eval_raw(
            Field::from(0.0),
            Field::from(0.0),
            Field::from(0.0),
            Field::from(0.0),
        );
        let mut buf = [0.0f32; PARALLELISM];
        result.store(&mut buf);
        // Inside points return 0.0 (escaped after 0 iterations normalized = 1 - 1 = 0)
        assert!(buf[0] < 0.1, "Origin should be inside the set");
    }

    #[test]
    fn mandelbrot_far_point_escapes() {
        let m = Mandelbrot::new(100);
        // Point (10, 10) is far outside, should escape immediately
        let result = m.eval_raw(
            Field::from(10.0),
            Field::from(10.0),
            Field::from(0.0),
            Field::from(0.0),
        );
        let mut buf = [0.0f32; PARALLELISM];
        result.store(&mut buf);
        // Points that escape early return values close to 1.0
        assert!(buf[0] > 0.9, "Far point should escape quickly");
    }

    #[test]
    fn julia_evaluates() {
        let j = Julia::rabbit(100);
        let result = j.eval_raw(
            Field::from(0.0),
            Field::from(0.0),
            Field::from(0.0),
            Field::from(0.0),
        );
        let mut buf = [0.0f32; PARALLELISM];
        result.store(&mut buf);
        // Just verify it produces a value in [0, 1]
        assert!(buf[0] >= 0.0 && buf[0] <= 1.0);
    }

    #[test]
    fn burning_ship_evaluates() {
        let bs = BurningShip::new(100);
        let result = bs.eval_raw(
            Field::from(-1.75),
            Field::from(-0.04),
            Field::from(0.0),
            Field::from(0.0),
        );
        let mut buf = [0.0f32; PARALLELISM];
        result.store(&mut buf);
        assert!(buf[0] >= 0.0 && buf[0] <= 1.0);
    }
}
