//! # PixelFlow Core
//!
//! A minimal lambda calculus EDSL over SIMD fields.
//!
//! The type system IS the AST. `Field` is the computational substrate.
//! `Manifold` is the core abstraction: a function from coordinates to values.

#![no_std]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

extern crate alloc;

// ============================================================================
// Backend (internal SIMD implementation)
// ============================================================================

/// SIMD backend abstractions (internal).
pub(crate) mod backend;

// ============================================================================
// Field: The User-Facing Abstraction
// ============================================================================

use backend::{Backend, FloatBatchOps};

// Select native backend based on architecture
#[cfg(target_arch = "x86_64")]
use backend::x86::Sse2 as NativeBackend;

#[cfg(target_arch = "aarch64")]
use backend::arm::Neon as NativeBackend;

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
use backend::scalar::Scalar as NativeBackend;

/// The computational substrate.
///
/// A `Field` represents a value—or many values computed in parallel.
/// Users treat it as a mathematical object; the SIMD nature is hidden.
///
/// # Creating Fields
/// Use `From`/`Into` for ergonomic construction:
/// ```ignore
/// let f: Field = 1.0.into();
/// let f = Field::from(42);
/// ```
pub type Field = <NativeBackend as Backend>::Batch<f32>;

impl From<f32> for Field {
    #[inline(always)]
    fn from(val: f32) -> Self {
        backend::SimdBatch::splat(val)
    }
}

impl From<i32> for Field {
    #[inline(always)]
    fn from(val: i32) -> Self {
        backend::SimdBatch::splat(val as f32)
    }
}

/// Number of values computed in parallel (implementation detail).
pub const PARALLELISM: usize = NativeBackend::LANES;

// ============================================================================
// Manifold: The Core Abstraction
// ============================================================================

/// A manifold maps coordinates to values.
///
/// This is the lambda calculus over fields.
/// Types encode program structure; `eval` is interpretation.
pub trait Manifold: Send + Sync {
    /// Evaluate at the given coordinates.
    fn eval(&self, x: Field, y: Field, z: Field, w: Field) -> Field;
}

// ============================================================================
// Variables
// ============================================================================

/// The X coordinate.
#[derive(Clone, Copy, Debug, Default)]
pub struct X;

/// The Y coordinate.
#[derive(Clone, Copy, Debug, Default)]
pub struct Y;

/// The Z coordinate.
#[derive(Clone, Copy, Debug, Default)]
pub struct Z;

/// The W coordinate (time).
#[derive(Clone, Copy, Debug, Default)]
pub struct W;

impl Manifold for X {
    #[inline(always)]
    fn eval(&self, x: Field, _y: Field, _z: Field, _w: Field) -> Field {
        x
    }
}

impl Manifold for Y {
    #[inline(always)]
    fn eval(&self, _x: Field, y: Field, _z: Field, _w: Field) -> Field {
        y
    }
}

impl Manifold for Z {
    #[inline(always)]
    fn eval(&self, _x: Field, _y: Field, z: Field, _w: Field) -> Field {
        z
    }
}

impl Manifold for W {
    #[inline(always)]
    fn eval(&self, _x: Field, _y: Field, _z: Field, w: Field) -> Field {
        w
    }
}

// ============================================================================
// Constants (auto-promoting)
// ============================================================================

impl Manifold for f32 {
    #[inline(always)]
    fn eval(&self, _x: Field, _y: Field, _z: Field, _w: Field) -> Field {
        Field::splat(*self)
    }
}

impl Manifold for i32 {
    #[inline(always)]
    fn eval(&self, _x: Field, _y: Field, _z: Field, _w: Field) -> Field {
        Field::splat(*self as f32)
    }
}

// ============================================================================
// Binary Operations
// ============================================================================

/// Addition: L + R
#[derive(Clone, Copy, Debug)]
pub struct Add<L, R>(pub L, pub R);

/// Subtraction: L - R
#[derive(Clone, Copy, Debug)]
pub struct Sub<L, R>(pub L, pub R);

/// Multiplication: L * R
#[derive(Clone, Copy, Debug)]
pub struct Mul<L, R>(pub L, pub R);

/// Division: L / R
#[derive(Clone, Copy, Debug)]
pub struct Div<L, R>(pub L, pub R);

impl<L: Manifold, R: Manifold> Manifold for Add<L, R> {
    #[inline(always)]
    fn eval(&self, x: Field, y: Field, z: Field, w: Field) -> Field {
        self.0.eval(x, y, z, w) + self.1.eval(x, y, z, w)
    }
}

impl<L: Manifold, R: Manifold> Manifold for Sub<L, R> {
    #[inline(always)]
    fn eval(&self, x: Field, y: Field, z: Field, w: Field) -> Field {
        self.0.eval(x, y, z, w) - self.1.eval(x, y, z, w)
    }
}

impl<L: Manifold, R: Manifold> Manifold for Mul<L, R> {
    #[inline(always)]
    fn eval(&self, x: Field, y: Field, z: Field, w: Field) -> Field {
        self.0.eval(x, y, z, w) * self.1.eval(x, y, z, w)
    }
}

impl<L: Manifold, R: Manifold> Manifold for Div<L, R> {
    #[inline(always)]
    fn eval(&self, x: Field, y: Field, z: Field, w: Field) -> Field {
        self.0.eval(x, y, z, w) / self.1.eval(x, y, z, w)
    }
}

// ============================================================================
// Operator Overloads (core::ops)
// ============================================================================

use core::ops;

// Macro to implement binary operators for all manifold combinations
macro_rules! impl_binop {
    ($trait:ident, $method:ident, $struct:ident) => {
        // M op M
        impl<L: Manifold, R: Manifold> ops::$trait<R> for $struct<L, R>
        where
            $struct<L, R>: Manifold,
        {
            type Output = $struct<Self, R>;
            fn $method(self, rhs: R) -> Self::Output {
                $struct(self, rhs)
            }
        }
    };
}

// Implement Add for our coordinate types
impl<R: Manifold> ops::Add<R> for X {
    type Output = Add<X, R>;
    fn add(self, rhs: R) -> Self::Output {
        Add(self, rhs)
    }
}
impl<R: Manifold> ops::Add<R> for Y {
    type Output = Add<Y, R>;
    fn add(self, rhs: R) -> Self::Output {
        Add(self, rhs)
    }
}
impl<R: Manifold> ops::Add<R> for Z {
    type Output = Add<Z, R>;
    fn add(self, rhs: R) -> Self::Output {
        Add(self, rhs)
    }
}
impl<R: Manifold> ops::Add<R> for W {
    type Output = Add<W, R>;
    fn add(self, rhs: R) -> Self::Output {
        Add(self, rhs)
    }
}

// Implement Mul for our coordinate types
impl<R: Manifold> ops::Mul<R> for X {
    type Output = Mul<X, R>;
    fn mul(self, rhs: R) -> Self::Output {
        Mul(self, rhs)
    }
}
impl<R: Manifold> ops::Mul<R> for Y {
    type Output = Mul<Y, R>;
    fn mul(self, rhs: R) -> Self::Output {
        Mul(self, rhs)
    }
}
impl<R: Manifold> ops::Mul<R> for Z {
    type Output = Mul<Z, R>;
    fn mul(self, rhs: R) -> Self::Output {
        Mul(self, rhs)
    }
}
impl<R: Manifold> ops::Mul<R> for W {
    type Output = Mul<W, R>;
    fn mul(self, rhs: R) -> Self::Output {
        Mul(self, rhs)
    }
}

// Implement Sub for coordinate types
impl<R: Manifold> ops::Sub<R> for X {
    type Output = Sub<X, R>;
    fn sub(self, rhs: R) -> Self::Output {
        Sub(self, rhs)
    }
}
impl<R: Manifold> ops::Sub<R> for Y {
    type Output = Sub<Y, R>;
    fn sub(self, rhs: R) -> Self::Output {
        Sub(self, rhs)
    }
}
impl<R: Manifold> ops::Sub<R> for Z {
    type Output = Sub<Z, R>;
    fn sub(self, rhs: R) -> Self::Output {
        Sub(self, rhs)
    }
}
impl<R: Manifold> ops::Sub<R> for W {
    type Output = Sub<W, R>;
    fn sub(self, rhs: R) -> Self::Output {
        Sub(self, rhs)
    }
}

// Implement Div for coordinate types
impl<R: Manifold> ops::Div<R> for X {
    type Output = Div<X, R>;
    fn div(self, rhs: R) -> Self::Output {
        Div(self, rhs)
    }
}
impl<R: Manifold> ops::Div<R> for Y {
    type Output = Div<Y, R>;
    fn div(self, rhs: R) -> Self::Output {
        Div(self, rhs)
    }
}
impl<R: Manifold> ops::Div<R> for Z {
    type Output = Div<Z, R>;
    fn div(self, rhs: R) -> Self::Output {
        Div(self, rhs)
    }
}
impl<R: Manifold> ops::Div<R> for W {
    type Output = Div<W, R>;
    fn div(self, rhs: R) -> Self::Output {
        Div(self, rhs)
    }
}

// Implement operators for Add<L,R> so we can chain: (X + Y) + Z
impl<L: Manifold, R1: Manifold, R2: Manifold> ops::Add<R2> for Add<L, R1> {
    type Output = Add<Add<L, R1>, R2>;
    fn add(self, rhs: R2) -> Self::Output {
        Add(self, rhs)
    }
}
impl<L: Manifold, R1: Manifold, R2: Manifold> ops::Mul<R2> for Add<L, R1> {
    type Output = Mul<Add<L, R1>, R2>;
    fn mul(self, rhs: R2) -> Self::Output {
        Mul(self, rhs)
    }
}

// Implement operators for Mul<L,R> so we can chain: (X * X) + (Y * Y)
impl<L: Manifold, R1: Manifold, R2: Manifold> ops::Add<R2> for Mul<L, R1> {
    type Output = Add<Mul<L, R1>, R2>;
    fn add(self, rhs: R2) -> Self::Output {
        Add(self, rhs)
    }
}
impl<L: Manifold, R1: Manifold, R2: Manifold> ops::Mul<R2> for Mul<L, R1> {
    type Output = Mul<Mul<L, R1>, R2>;
    fn mul(self, rhs: R2) -> Self::Output {
        Mul(self, rhs)
    }
}

// ============================================================================
// Unary Operations
// ============================================================================

/// Square root.
#[derive(Clone, Copy, Debug)]
pub struct Sqrt<T>(pub T);

/// Absolute value.
#[derive(Clone, Copy, Debug)]
pub struct Abs<T>(pub T);

impl<T: Manifold> Manifold for Sqrt<T> {
    #[inline(always)]
    fn eval(&self, x: Field, y: Field, z: Field, w: Field) -> Field {
        self.0.eval(x, y, z, w).sqrt()
    }
}

impl<T: Manifold> Manifold for Abs<T> {
    #[inline(always)]
    fn eval(&self, x: Field, y: Field, z: Field, w: Field) -> Field {
        self.0.eval(x, y, z, w).abs()
    }
}

/// Maximum of two values.
#[derive(Clone, Copy, Debug)]
pub struct Max<L, R>(pub L, pub R);

/// Minimum of two values.
#[derive(Clone, Copy, Debug)]
pub struct Min<L, R>(pub L, pub R);

impl<L: Manifold, R: Manifold> Manifold for Max<L, R> {
    #[inline(always)]
    fn eval(&self, x: Field, y: Field, z: Field, w: Field) -> Field {
        self.0.eval(x, y, z, w).max(self.1.eval(x, y, z, w))
    }
}

impl<L: Manifold, R: Manifold> Manifold for Min<L, R> {
    #[inline(always)]
    fn eval(&self, x: Field, y: Field, z: Field, w: Field) -> Field {
        self.0.eval(x, y, z, w).min(self.1.eval(x, y, z, w))
    }
}

// ============================================================================
// Comparison Operations (produce masks)
// ============================================================================

use backend::BatchArithmetic;

/// Less than: L < R
#[derive(Clone, Copy, Debug)]
pub struct Lt<L, R>(pub L, pub R);

/// Greater than: L > R
#[derive(Clone, Copy, Debug)]
pub struct Gt<L, R>(pub L, pub R);

/// Less than or equal: L <= R
#[derive(Clone, Copy, Debug)]
pub struct Le<L, R>(pub L, pub R);

/// Greater than or equal: L >= R
#[derive(Clone, Copy, Debug)]
pub struct Ge<L, R>(pub L, pub R);

impl<L: Manifold, R: Manifold> Manifold for Lt<L, R> {
    #[inline(always)]
    fn eval(&self, x: Field, y: Field, z: Field, w: Field) -> Field {
        BatchArithmetic::cmp_lt(self.0.eval(x, y, z, w), self.1.eval(x, y, z, w))
    }
}

impl<L: Manifold, R: Manifold> Manifold for Gt<L, R> {
    #[inline(always)]
    fn eval(&self, x: Field, y: Field, z: Field, w: Field) -> Field {
        BatchArithmetic::cmp_gt(self.0.eval(x, y, z, w), self.1.eval(x, y, z, w))
    }
}

impl<L: Manifold, R: Manifold> Manifold for Le<L, R> {
    #[inline(always)]
    fn eval(&self, x: Field, y: Field, z: Field, w: Field) -> Field {
        BatchArithmetic::cmp_le(self.0.eval(x, y, z, w), self.1.eval(x, y, z, w))
    }
}

impl<L: Manifold, R: Manifold> Manifold for Ge<L, R> {
    #[inline(always)]
    fn eval(&self, x: Field, y: Field, z: Field, w: Field) -> Field {
        BatchArithmetic::cmp_ge(self.0.eval(x, y, z, w), self.1.eval(x, y, z, w))
    }
}

// ============================================================================
// Select: Branchless Conditional with Short-Circuit
// ============================================================================

use backend::SimdBatch;

/// Select: if cond { if_true } else { if_false }
///
/// Short-circuits: if all lanes pass, only evaluates if_true.
/// If no lanes pass, only evaluates if_false.
/// Otherwise, evaluates both and blends.
#[derive(Clone, Copy, Debug)]
pub struct Select<C, T, F> {
    /// The condition (mask).
    pub cond: C,
    /// Value when condition is true.
    pub if_true: T,
    /// Value when condition is false.
    pub if_false: F,
}

impl<C: Manifold, T: Manifold, F: Manifold> Manifold for Select<C, T, F> {
    #[inline(always)]
    fn eval(&self, x: Field, y: Field, z: Field, w: Field) -> Field {
        let mask = self.cond.eval(x, y, z, w);

        // Short-circuit: skip unused branches
        if mask.all() {
            return self.if_true.eval(x, y, z, w);
        }
        if !mask.any() {
            return self.if_false.eval(x, y, z, w);
        }

        // Straddle: must evaluate both and blend
        let t = self.if_true.eval(x, y, z, w);
        let f = self.if_false.eval(x, y, z, w);
        BatchArithmetic::select(mask, t, f)
    }
}

// ============================================================================
// Fix: Iteration to Fixed Point
// ============================================================================

/// Maximum iterations before giving up.
const FIX_MAX_ITERS: u32 = 1024;

/// Fix: iterate until convergence.
///
/// Iterates step until done returns true (per-lane).
/// State is passed via W coordinate to both step and done.
///
/// ```ignore
/// let mandelbrot = Fix {
///     seed: 0.0,  // z₀ = 0
///     step: W.mul(W).add(X),  // z² + c
///     done: W.mul(W).gt(4.0), // escaped when |z|² > 4
/// };
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Fix<Seed, Step, Done> {
    /// Initial state.
    pub seed: Seed,
    /// Step function: current state (w) → next state.
    pub step: Step,
    /// Termination condition: state (w) → mask (true = converged).
    pub done: Done,
}

impl<Seed: Manifold, Step: Manifold, Done: Manifold> Manifold for Fix<Seed, Step, Done> {
    #[inline(always)]
    fn eval(&self, x: Field, y: Field, z: Field, w: Field) -> Field {
        // Initial state
        let mut state = self.seed.eval(x, y, z, w);

        // Track which lanes are still active (all 1s = active)
        let mut active = 1.0f32.eval(x, y, z, w);

        for _ in 0..FIX_MAX_ITERS {
            // Check termination
            let done_mask = self.done.eval(x, y, z, state);

            // Retire converged lanes: active = active AND NOT done
            active = BatchArithmetic::select(done_mask, 0.0f32.eval(x, y, z, w), active);

            // Early exit if all lanes converged
            if !active.any() {
                break;
            }

            // Step only active lanes
            let next = self.step.eval(x, y, z, state);
            state = BatchArithmetic::select(active, next, state);
        }

        state
    }
}

// ============================================================================
// DSL Extension Trait
// ============================================================================

/// Extension methods for composing manifolds.
pub trait ManifoldExt: Manifold + Sized {
    /// Add two manifolds.
    fn add<R: Manifold>(self, rhs: R) -> Add<Self, R> {
        Add(self, rhs)
    }
    /// Subtract two manifolds.
    fn sub<R: Manifold>(self, rhs: R) -> Sub<Self, R> {
        Sub(self, rhs)
    }
    /// Multiply two manifolds.
    fn mul<R: Manifold>(self, rhs: R) -> Mul<Self, R> {
        Mul(self, rhs)
    }
    /// Divide two manifolds.
    fn div<R: Manifold>(self, rhs: R) -> Div<Self, R> {
        Div(self, rhs)
    }
    /// Square root.
    fn sqrt(self) -> Sqrt<Self> {
        Sqrt(self)
    }
    /// Absolute value.
    fn abs(self) -> Abs<Self> {
        Abs(self)
    }

    // Comparisons
    /// Less than.
    fn lt<R: Manifold>(self, rhs: R) -> Lt<Self, R> {
        Lt(self, rhs)
    }
    /// Greater than.
    fn gt<R: Manifold>(self, rhs: R) -> Gt<Self, R> {
        Gt(self, rhs)
    }
    /// Less than or equal.
    fn le<R: Manifold>(self, rhs: R) -> Le<Self, R> {
        Le(self, rhs)
    }
    /// Greater than or equal.
    fn ge<R: Manifold>(self, rhs: R) -> Ge<Self, R> {
        Ge(self, rhs)
    }

    /// Conditional select. If self (as mask), use if_true; else if_false.
    fn select<T: Manifold, F: Manifold>(self, if_true: T, if_false: F) -> Select<Self, T, F> {
        Select {
            cond: self,
            if_true,
            if_false,
        }
    }

    /// Maximum of two manifolds.
    fn max<R: Manifold>(self, rhs: R) -> Max<Self, R> {
        Max(self, rhs)
    }

    /// Minimum of two manifolds.
    fn min<R: Manifold>(self, rhs: R) -> Min<Self, R> {
        Min(self, rhs)
    }

    /// Type-erase this manifold into a boxed trait object.
    ///
    /// This stops the nested type madness and produces a storable,
    /// passable value with a simple type.
    fn boxed(self) -> BoxedManifold
    where
        Self: 'static,
    {
        alloc::boxed::Box::new(self)
    }
}

impl<T: Manifold + Sized> ManifoldExt for T {}

/// A type-erased manifold.
///
/// Use `manifold.boxed()` to create one.
pub type BoxedManifold = alloc::boxed::Box<dyn Manifold>;

impl Manifold for BoxedManifold {
    #[inline(always)]
    fn eval(&self, x: Field, y: Field, z: Field, w: Field) -> Field {
        (**self).eval(x, y, z, w)
    }
}

// ============================================================================
// Materialize: The Observation Boundary
// ============================================================================

/// Sample a manifold at 2D coordinates and write to buffer.
///
/// This is the observation boundary: manifold algebra → bytes.
/// Evaluates at (x, y, 0, 0) for PARALLELISM consecutive x values.
///
/// # Arguments
/// * `m` - The manifold to sample
/// * `x_start` - Starting x coordinate (samples x_start, x_start+1, ..., x_start+PARALLELISM-1)
/// * `y` - Y coordinate (constant across the batch)
/// * `out` - Output buffer, must be at least PARALLELISM bytes
///
/// # Panics
/// Panics if `out.len() < PARALLELISM`
pub fn materialize<M: Manifold>(m: &M, x_start: f32, y: f32, out: &mut [u8]) {
    assert!(out.len() >= PARALLELISM, "buffer too small");

    // Create coordinate fields (internal - not exposed)
    let x = Field::sequential_from(x_start);
    let yf = Field::splat(y);
    let zero = Field::splat(0.0);

    // Evaluate manifold at 2D coordinates (z=0, w=0)
    let result = m.eval(x, yf, zero, zero);

    // Clamp to [0, 1] and convert to u8
    let clamped = result.max(Field::splat(0.0)).min(Field::splat(1.0));
    let scaled = clamped * Field::splat(255.0);

    // Store to buffer
    for i in 0..PARALLELISM {
        out[i] = scaled.extract_lane(i) as u8;
    }
}

/// Sample a manifold and write f32 values directly.
///
/// Like `materialize` but outputs f32 instead of u8.
pub fn materialize_f32<M: Manifold>(m: &M, x_start: f32, y: f32, out: &mut [f32]) {
    assert!(out.len() >= PARALLELISM, "buffer too small");

    let x = Field::sequential_from(x_start);
    let yf = Field::splat(y);
    let zero = Field::splat(0.0);

    let result = m.eval(x, yf, zero, zero);

    for i in 0..PARALLELISM {
        out[i] = result.extract_lane(i);
    }
}
