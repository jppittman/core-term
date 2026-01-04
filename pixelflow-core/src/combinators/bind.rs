//! # Named Variable Bindings via ZST Markers
//!
//! This module provides semantic aliases for coordinate slots using zero-sized
//! types (ZSTs). This gives meaningful names without any runtime cost.
//!
//! # Design
//!
//! Instead of using X, Y, Z, W directly, you can use named bindings like:
//! - `ZR` (real part of z) → maps to Z
//! - `ZI` (imaginary part of z) → maps to W
//! - `CR` (real part of c) → maps to X
//! - `CI` (imaginary part of c) → maps to Y
//!
//! # Example
//!
//! ```ignore
//! use pixelflow_core::combinators::bind::{ZR, ZI, CR, CI};
//!
//! // Mandelbrot iteration step using semantic names
//! let step_real = ZR * ZR - ZI * ZI + CR;  // zr² - zi² + cr
//! let step_imag = ZR * ZI * 2.0 + CI;      // 2·zr·zi + ci
//! ```
//!
//! # With Let for Local Bindings
//!
//! Use `Let` to introduce a computed value that can be reused:
//!
//! ```ignore
//! use pixelflow_core::combinators::bind::Let;
//! use pixelflow_core::{X, Y, W};
//!
//! // Let radius = sqrt(x² + y²) in radius * 2
//! let with_radius = Let {
//!     value: (X * X + Y * Y).sqrt(),
//!     body: W * 2.0,  // W carries the bound value
//! };
//! ```

use core::marker::PhantomData;

use crate::ops::{Add, Div, Max, Min, Mul, Sub, Sqrt, MulRsqrt};
use crate::{Computational, Field, Manifold, X, Y, Z, W};

// ============================================================================
// Named Marker Types (ZSTs)
// ============================================================================

/// Marker for real part of complex z.
#[derive(Clone, Copy, Debug, Default)]
pub struct ZrName;

/// Marker for imaginary part of complex z.
#[derive(Clone, Copy, Debug, Default)]
pub struct ZiName;

/// Marker for real part of complex c.
#[derive(Clone, Copy, Debug, Default)]
pub struct CrName;

/// Marker for imaginary part of complex c.
#[derive(Clone, Copy, Debug, Default)]
pub struct CiName;

// ============================================================================
// Bind: Named Reference to a Coordinate Slot
// ============================================================================

/// A named variable binding.
///
/// Wraps a coordinate type with a semantic name for clarity.
/// `Bind<ZrName, Z>` behaves exactly like `Z` but documents its purpose.
#[derive(Debug)]
pub struct Bind<Name, Coord = Z>(PhantomData<(Name, Coord)>);

impl<Name, Coord> Bind<Name, Coord> {
    /// Create a new named binding.
    #[inline]
    pub const fn new() -> Self {
        Bind(PhantomData)
    }
}

impl<Name, Coord> Copy for Bind<Name, Coord> {}

impl<Name, Coord> Clone for Bind<Name, Coord> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<Name, Coord> Default for Bind<Name, Coord> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<Name, Coord, I> Manifold<I> for Bind<Name, Coord>
where
    I: Computational,
    Name: Send + Sync,
    Coord: Manifold<I> + Default + Send + Sync,
{
    type Output = Coord::Output;

    #[inline(always)]
    fn eval_raw(&self, x: I, y: I, z: I, w: I) -> Self::Output {
        Coord::default().eval_raw(x, y, z, w)
    }
}

// ============================================================================
// Operator Overloads for Bind
// ============================================================================

impl<Name, Coord, Rhs: Manifold> core::ops::Add<Rhs> for Bind<Name, Coord> {
    type Output = Add<Self, Rhs>;
    #[inline]
    fn add(self, rhs: Rhs) -> Self::Output {
        Add(self, rhs)
    }
}

impl<Name, Coord, Rhs: Manifold> core::ops::Sub<Rhs> for Bind<Name, Coord> {
    type Output = Sub<Self, Rhs>;
    #[inline]
    fn sub(self, rhs: Rhs) -> Self::Output {
        Sub(self, rhs)
    }
}

impl<Name, Coord, Rhs: Manifold> core::ops::Mul<Rhs> for Bind<Name, Coord> {
    type Output = Mul<Self, Rhs>;
    #[inline]
    fn mul(self, rhs: Rhs) -> Self::Output {
        Mul(self, rhs)
    }
}

// Rsqrt fusion: Bind / Sqrt<R> → MulRsqrt<Bind, R>
impl<Name, Coord, R: Manifold> core::ops::Div<Sqrt<R>> for Bind<Name, Coord> {
    type Output = MulRsqrt<Self, R>;
    #[inline(always)]
    fn div(self, rhs: Sqrt<R>) -> Self::Output {
        MulRsqrt(self, rhs.0)
    }
}

// Generic div for other types
impl<Name, Coord, Rhs: Manifold> core::ops::Div<Rhs> for Bind<Name, Coord>
where
    Rhs: NotSqrt,
{
    type Output = Div<Self, Rhs>;
    #[inline]
    fn div(self, rhs: Rhs) -> Self::Output {
        Div(self, rhs)
    }
}

/// Marker trait to exclude Sqrt from the generic Div impl.
pub trait NotSqrt {}

// Implement NotSqrt for common types
impl NotSqrt for f32 {}
impl NotSqrt for i32 {}
impl NotSqrt for Field {}
impl NotSqrt for X {}
impl NotSqrt for Y {}
impl NotSqrt for Z {}
impl NotSqrt for W {}
impl<Name, Coord> NotSqrt for Bind<Name, Coord> {}
impl<L, R> NotSqrt for Add<L, R> {}
impl<L, R> NotSqrt for Sub<L, R> {}
impl<L, R> NotSqrt for Mul<L, R> {}
impl<L, R> NotSqrt for Div<L, R> {}
impl<L, R> NotSqrt for Max<L, R> {}
impl<L, R> NotSqrt for Min<L, R> {}

// ============================================================================
// Pre-defined Bindings for Complex Number Iteration
// ============================================================================

/// Real part of complex z (uses Z slot).
pub const ZR: Bind<ZrName, Z> = Bind::new();

/// Imaginary part of complex z (uses W slot).
pub const ZI: Bind<ZiName, W> = Bind::new();

/// Real part of complex c (uses X slot).
pub const CR: Bind<CrName, X> = Bind::new();

/// Imaginary part of complex c (uses Y slot).
pub const CI: Bind<CiName, Y> = Bind::new();

// ============================================================================
// Let: Introduce a Local Binding
// ============================================================================

/// Introduce a local binding.
///
/// `Let` evaluates `value` and makes it available via the W coordinate
/// when evaluating `body`.
///
/// # Example
///
/// ```ignore
/// use pixelflow_core::combinators::bind::Let;
/// use pixelflow_core::{X, Y, W};
///
/// // Compute radius once, use it via W
/// let with_radius = Let {
///     value: (X * X + Y * Y).sqrt(),
///     body: W * 2.0,  // W = radius
/// };
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Let<Value, Body> {
    /// The value to bind (will be available as W in body).
    pub value: Value,
    /// The body expression that uses the bound value via W.
    pub body: Body,
}

impl<Value, Body> Let<Value, Body> {
    /// Create a new let binding.
    #[inline]
    pub fn new(value: Value, body: Body) -> Self {
        Self { value, body }
    }
}

impl<Value, Body> Manifold<Field> for Let<Value, Body>
where
    Value: Manifold<Field, Output = Field>,
    Body: Manifold<Field>,
{
    type Output = Body::Output;

    #[inline(always)]
    fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> Self::Output {
        // Evaluate the value expression
        let bound_value = self.value.eval_raw(x, y, z, w);

        // Thread the bound value through W when evaluating body
        self.body.eval_raw(x, y, z, bound_value)
    }
}

// ============================================================================
// Let2: Two-Variable Binding for Complex Numbers
// ============================================================================

/// Introduce two local bindings (e.g., for complex numbers).
///
/// `Let2` binds two values: one to Z, one to W. This is useful for
/// complex number operations where you need both real and imaginary parts.
///
/// # Example
///
/// ```ignore
/// use pixelflow_core::combinators::bind::{Let2, ZR, ZI};
///
/// // Bind initial complex value, compute magnitude squared
/// let mag_sq = Let2 {
///     value_z: X,  // zr = X
///     value_w: Y,  // zi = Y
///     body: ZR * ZR + ZI * ZI,  // zr² + zi²
/// };
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Let2<ValueZ, ValueW, Body> {
    /// The value to bind to Z (zr in complex iteration).
    pub value_z: ValueZ,
    /// The value to bind to W (zi in complex iteration).
    pub value_w: ValueW,
    /// The body expression that uses both bindings.
    pub body: Body,
}

impl<ValueZ, ValueW, Body> Let2<ValueZ, ValueW, Body> {
    /// Create a new two-variable let binding.
    #[inline]
    pub fn new(value_z: ValueZ, value_w: ValueW, body: Body) -> Self {
        Self { value_z, value_w, body }
    }
}

impl<ValueZ, ValueW, Body> Manifold<Field> for Let2<ValueZ, ValueW, Body>
where
    ValueZ: Manifold<Field, Output = Field>,
    ValueW: Manifold<Field, Output = Field>,
    Body: Manifold<Field>,
{
    type Output = Body::Output;

    #[inline(always)]
    fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> Self::Output {
        // Evaluate both value expressions
        let bound_z = self.value_z.eval_raw(x, y, z, w);
        let bound_w = self.value_w.eval_raw(x, y, z, w);

        // Thread both bound values when evaluating body
        self.body.eval_raw(x, y, bound_z, bound_w)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PARALLELISM;

    #[test]
    fn bind_evaluates_as_coordinate() {
        let x_val = Field::from(3.0);
        let y_val = Field::from(4.0);
        let z_val = Field::from(5.0);
        let w_val = Field::from(6.0);

        // ZR maps to Z, ZI maps to W
        let result_zr = ZR.eval_raw(x_val, y_val, z_val, w_val);
        let result_zi = ZI.eval_raw(x_val, y_val, z_val, w_val);

        let mut buf = [0.0f32; PARALLELISM];
        result_zr.store(&mut buf);
        assert_eq!(buf[0], 5.0);  // Z value

        result_zi.store(&mut buf);
        assert_eq!(buf[0], 6.0);  // W value
    }

    #[test]
    fn bind_in_expressions() {
        // zr² - zi² + cr
        let step_real = ZR * ZR - ZI * ZI + CR;

        // Evaluate at cr=0.5, ci=0.25, zr=1, zi=2
        let result = step_real.eval_raw(
            Field::from(0.5),   // cr (X)
            Field::from(0.25),  // ci (Y)
            Field::from(1.0),   // zr (Z)
            Field::from(2.0),   // zi (W)
        );

        let mut buf = [0.0f32; PARALLELISM];
        result.store(&mut buf);
        // 1² - 2² + 0.5 = 1 - 4 + 0.5 = -2.5
        assert_eq!(buf[0], -2.5);
    }

    #[test]
    fn let_binds_value() {
        // Let w = 10 in w + w
        let expr = Let {
            value: 10.0f32,
            body: W + W,
        };

        let result = expr.eval_raw(
            Field::from(0.0),
            Field::from(0.0),
            Field::from(0.0),
            Field::from(0.0),
        );

        let mut buf = [0.0f32; PARALLELISM];
        result.store(&mut buf);
        assert_eq!(buf[0], 20.0);
    }

    #[test]
    fn let_with_computed_value() {
        // Let w = x + y in w * 2
        let expr = Let {
            value: X + Y,
            body: W * 2.0,
        };

        let result = expr.eval_raw(
            Field::from(3.0),
            Field::from(4.0),
            Field::from(0.0),
            Field::from(0.0),
        );

        let mut buf = [0.0f32; PARALLELISM];
        result.store(&mut buf);
        // (3 + 4) * 2 = 14
        assert_eq!(buf[0], 14.0);
    }

    #[test]
    fn let2_binds_two_values() {
        // Let2 zr = X, zi = Y in zr² + zi² (magnitude squared)
        let expr = Let2 {
            value_z: X,
            value_w: Y,
            body: ZR * ZR + ZI * ZI,
        };

        let result = expr.eval_raw(
            Field::from(3.0),
            Field::from(4.0),
            Field::from(0.0),
            Field::from(0.0),
        );

        let mut buf = [0.0f32; PARALLELISM];
        result.store(&mut buf);
        // 3² + 4² = 9 + 16 = 25
        assert_eq!(buf[0], 25.0);
    }
}
