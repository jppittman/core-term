//! # Compile-Time Bounded Iteration via Type-Level Recursion
//!
//! This module provides `FixAt`, a compile-time bounded iteration combinator
//! that expands to nested `At` types at the type level.
//!
//! Unlike runtime `Fix` which uses a loop, `FixAt` creates a type that IS
//! the unrolled computation graph. The compiler sees the full structure.
//!
//! # Type-Level Natural Numbers
//!
//! We use Peano encoding for compile-time naturals:
//! - `N0` = zero iterations (just return seed)
//! - `N1` = `Succ<N0>` = one iteration
//! - `N2` = `Succ<N1>` = two iterations
//! - etc.
//!
//! # How It Works
//!
//! `FixAt<N2, Seed, Step>` expands at the type level to:
//! ```text
//! At { inner: Step, w: At { inner: Step, w: Seed } }
//! ```
//!
//! The step manifold receives the current state via W and produces the next state.
//! X, Y, Z are passed through unchanged.
//!
//! # Example
//!
//! ```ignore
//! use pixelflow_core::combinators::fix_at::{FixAt, N3};
//! use pixelflow_core::{W, Manifold};
//!
//! // Define a step function: state = state * 2
//! let double = W * 2.0;
//!
//! // Apply it 3 times starting from 1.0
//! // Result type: At<Step, At<Step, At<Step, Seed>>>
//! let iterated = FixAt::<N3, _, _>::new(1.0f32, double);
//!
//! // Evaluates: 1.0 * 2 * 2 * 2 = 8.0
//! ```

use core::marker::PhantomData;

use crate::combinators::At;
use crate::{Computational, Field, Manifold, W, X, Y, Z};

// ============================================================================
// Type-Level Natural Numbers (Peano Encoding)
// ============================================================================

/// Zero - base case for type-level naturals.
#[derive(Clone, Copy, Debug, Default)]
pub struct Zero;

/// Successor - represents N + 1 at the type level.
#[derive(Clone, Copy, Debug, Default)]
pub struct Succ<N>(PhantomData<N>);

// Convenient type aliases
/// Type-level 0
pub type N0 = Zero;
/// Type-level 1
pub type N1 = Succ<N0>;
/// Type-level 2
pub type N2 = Succ<N1>;
/// Type-level 3
pub type N3 = Succ<N2>;
/// Type-level 4
pub type N4 = Succ<N3>;
/// Type-level 5
pub type N5 = Succ<N4>;
/// Type-level 6
pub type N6 = Succ<N5>;
/// Type-level 7
pub type N7 = Succ<N6>;
/// Type-level 8
pub type N8 = Succ<N7>;
/// Type-level 16
pub type N16 = Succ<Succ<Succ<Succ<Succ<Succ<Succ<Succ<N8>>>>>>>>;
/// Type-level 32
pub type N32 = Succ<Succ<Succ<Succ<Succ<Succ<Succ<Succ<Succ<Succ<Succ<Succ<Succ<Succ<Succ<Succ<N16>>>>>>>>>>>>>>>>;

// ============================================================================
// Type-Level Iteration Trait
// ============================================================================

/// Trait that computes the result type of N iterations.
///
/// This is the core mechanism: each impl produces an associated type
/// that represents the iterated computation.
pub trait IterateAt<Seed, Step> {
    /// The resulting manifold type after N iterations.
    type Output: Manifold<Field>;

    /// Build the iterated manifold.
    fn build(seed: Seed, step: Step) -> Self::Output;
}

// Base case: Zero iterations returns the seed unchanged
impl<Seed, Step> IterateAt<Seed, Step> for Zero
where
    Seed: Manifold<Field, Output = Field> + Clone + Send + Sync,
    Step: Clone + Send + Sync,
{
    type Output = Seed;

    #[inline(always)]
    fn build(seed: Seed, _step: Step) -> Self::Output {
        seed
    }
}

// Recursive case: Succ<N> applies Step to the result of N iterations
impl<N, Seed, Step> IterateAt<Seed, Step> for Succ<N>
where
    N: IterateAt<Seed, Step>,
    N::Output: Manifold<Field, Output = Field> + Send + Sync,
    Seed: Manifold<Field, Output = Field> + Clone + Send + Sync,
    Step: Manifold<Field, Output = Field> + Clone + Send + Sync,
{
    // The type expands: At { inner: Step, x: X, y: Y, z: Z, w: <N iterations> }
    type Output = At<X, Y, Z, N::Output, Step>;

    #[inline(always)]
    fn build(seed: Seed, step: Step) -> Self::Output {
        // First build the inner N iterations
        let inner_iterations = N::build(seed, step.clone());

        // Then wrap in one more At application
        At {
            inner: step,
            x: X,
            y: Y,
            z: Z,
            w: inner_iterations,
        }
    }
}

// ============================================================================
// User-Facing Wrapper
// ============================================================================

/// Compile-time bounded iteration via type-level recursion.
///
/// Applies `step` to `seed` exactly N times, where N is a type-level natural.
/// The resulting type is a nested chain of `At` combinators that the compiler
/// can fully inline and optimize.
///
/// # Type Parameters
///
/// - `N`: Type-level iteration count (e.g., `N3` for 3 iterations)
/// - `Seed`: Initial state manifold
/// - `Step`: Step function manifold (receives state via W, outputs new state)
///
/// # Example
///
/// ```ignore
/// use pixelflow_core::combinators::fix_at::{FixAt, N4};
/// use pixelflow_core::W;
///
/// // Newton-Raphson: x_{n+1} = x - f(x)/f'(x)
/// // For sqrt(2): f(x) = x² - 2, f'(x) = 2x
/// // Step: x - (x² - 2) / (2x) = (x + 2/x) / 2
/// let step = (W + 2.0 / W) / 2.0;
///
/// // 4 iterations starting from 1.5
/// let sqrt2 = FixAt::<N4, _, _>::new(1.5f32, step);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct FixAt<N, Seed, Step> {
    /// Marker for the iteration count type.
    _count: PhantomData<N>,
    /// The initial state.
    pub seed: Seed,
    /// The step function.
    pub step: Step,
}

impl<N, Seed, Step> FixAt<N, Seed, Step>
where
    N: IterateAt<Seed, Step>,
    Seed: Clone,
    Step: Clone,
{
    /// Create a new bounded iteration.
    #[inline]
    pub fn new(seed: Seed, step: Step) -> Self {
        Self {
            _count: PhantomData,
            seed,
            step,
        }
    }

    /// Expand to the iterated manifold type.
    ///
    /// This consumes self and returns the expanded type, which is a
    /// nested chain of At combinators.
    #[inline(always)]
    pub fn expand(self) -> N::Output {
        N::build(self.seed, self.step)
    }
}

// FixAt itself is a Manifold that delegates to the expanded form
impl<N, Seed, Step> Manifold<Field> for FixAt<N, Seed, Step>
where
    N: IterateAt<Seed, Step> + Send + Sync,
    N::Output: Manifold<Field>,
    Seed: Manifold<Field, Output = Field> + Clone + Send + Sync,
    Step: Manifold<Field, Output = Field> + Clone + Send + Sync,
{
    type Output = <N::Output as Manifold<Field>>::Output;

    #[inline(always)]
    fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> Self::Output {
        let expanded = N::build(self.seed.clone(), self.step.clone());
        expanded.eval_raw(x, y, z, w)
    }
}

// ============================================================================
// Two-State Variable Iteration (FixAt2)
// ============================================================================

/// Trait for type-level two-variable iteration.
///
/// This is the core mechanism for iterating two state variables (e.g., complex z = zr + i*zi)
/// where X, Y pass through unchanged (carry c values) and Z, W carry state.
pub trait IterateAt2<SeedZ, SeedW, StepZ, StepW> {
    /// The resulting type for the Z output after N iterations.
    type OutputZ: Manifold<Field>;
    /// The resulting type for the W output after N iterations.
    type OutputW: Manifold<Field>;

    /// Build the Z output (e.g., real part).
    fn build_z(seed_z: SeedZ, seed_w: SeedW, step_z: StepZ, step_w: StepW) -> Self::OutputZ;
    /// Build the W output (e.g., imaginary part).
    fn build_w(seed_z: SeedZ, seed_w: SeedW, step_z: StepZ, step_w: StepW) -> Self::OutputW;
}

// Base case: Zero iterations returns the seeds unchanged
impl<SeedZ, SeedW, StepZ, StepW> IterateAt2<SeedZ, SeedW, StepZ, StepW> for Zero
where
    SeedZ: Manifold<Field, Output = Field> + Clone + Send + Sync,
    SeedW: Manifold<Field, Output = Field> + Clone + Send + Sync,
    StepZ: Clone + Send + Sync,
    StepW: Clone + Send + Sync,
{
    type OutputZ = SeedZ;
    type OutputW = SeedW;

    #[inline(always)]
    fn build_z(seed_z: SeedZ, _seed_w: SeedW, _step_z: StepZ, _step_w: StepW) -> Self::OutputZ {
        seed_z
    }

    #[inline(always)]
    fn build_w(_seed_z: SeedZ, seed_w: SeedW, _step_z: StepZ, _step_w: StepW) -> Self::OutputW {
        seed_w
    }
}

// Recursive case: Succ<N> applies step to the result of N iterations
impl<N, SeedZ, SeedW, StepZ, StepW> IterateAt2<SeedZ, SeedW, StepZ, StepW> for Succ<N>
where
    N: IterateAt2<SeedZ, SeedW, StepZ, StepW>,
    N::OutputZ: Manifold<Field, Output = Field> + Clone + Send + Sync,
    N::OutputW: Manifold<Field, Output = Field> + Clone + Send + Sync,
    SeedZ: Manifold<Field, Output = Field> + Clone + Send + Sync,
    SeedW: Manifold<Field, Output = Field> + Clone + Send + Sync,
    StepZ: Manifold<Field, Output = Field> + Clone + Send + Sync,
    StepW: Manifold<Field, Output = Field> + Clone + Send + Sync,
{
    // Type expands: At { inner: StepZ, x: X, y: Y, z: N::OutputZ, w: N::OutputW }
    type OutputZ = At<X, Y, N::OutputZ, N::OutputW, StepZ>;
    type OutputW = At<X, Y, N::OutputZ, N::OutputW, StepW>;

    #[inline(always)]
    fn build_z(seed_z: SeedZ, seed_w: SeedW, step_z: StepZ, step_w: StepW) -> Self::OutputZ {
        let prev_z = N::build_z(seed_z.clone(), seed_w.clone(), step_z.clone(), step_w.clone());
        let prev_w = N::build_w(seed_z, seed_w, step_z.clone(), step_w);
        At {
            inner: step_z,
            x: X,
            y: Y,
            z: prev_z,
            w: prev_w,
        }
    }

    #[inline(always)]
    fn build_w(seed_z: SeedZ, seed_w: SeedW, step_z: StepZ, step_w: StepW) -> Self::OutputW {
        let prev_z = N::build_z(seed_z.clone(), seed_w.clone(), step_z.clone(), step_w.clone());
        let prev_w = N::build_w(seed_z, seed_w, step_z, step_w.clone());
        At {
            inner: step_w,
            x: X,
            y: Y,
            z: prev_z,
            w: prev_w,
        }
    }
}

/// Compile-time bounded two-variable iteration.
///
/// Applies `step_z` and `step_w` to `(seed_z, seed_w)` exactly N times.
/// This is designed for complex number iteration where:
/// - X, Y carry the constant c (cr, ci)
/// - Z, W carry the state z (zr, zi)
///
/// # Example: Mandelbrot Step
///
/// ```ignore
/// use pixelflow_core::combinators::fix_at::{FixAt2, N4};
/// use pixelflow_core::combinators::bind::{ZR, ZI, CR, CI};
///
/// // z' = z² + c
/// // zr' = zr² - zi² + cr
/// // zi' = 2·zr·zi + ci
/// let step_real = ZR * ZR - ZI * ZI + CR;
/// let step_imag = ZR * ZI * 2.0 + CI;
///
/// // 4 iterations starting from (0, 0)
/// let mandelbrot = FixAt2::<N4, _, _, _, _>::new(0.0f32, 0.0f32, step_real, step_imag);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct FixAt2<N, SeedZ, SeedW, StepZ, StepW> {
    _count: PhantomData<N>,
    /// Initial real part (Z state).
    pub seed_z: SeedZ,
    /// Initial imaginary part (W state).
    pub seed_w: SeedW,
    /// Step function for real part.
    pub step_z: StepZ,
    /// Step function for imaginary part.
    pub step_w: StepW,
}

impl<N, SeedZ, SeedW, StepZ, StepW> FixAt2<N, SeedZ, SeedW, StepZ, StepW>
where
    N: IterateAt2<SeedZ, SeedW, StepZ, StepW>,
    SeedZ: Clone,
    SeedW: Clone,
    StepZ: Clone,
    StepW: Clone,
{
    /// Create a new bounded two-variable iteration.
    #[inline]
    pub fn new(seed_z: SeedZ, seed_w: SeedW, step_z: StepZ, step_w: StepW) -> Self {
        Self {
            _count: PhantomData,
            seed_z,
            seed_w,
            step_z,
            step_w,
        }
    }

    /// Expand to the Z output manifold.
    #[inline(always)]
    pub fn expand_z(self) -> N::OutputZ {
        N::build_z(self.seed_z, self.seed_w, self.step_z, self.step_w)
    }

    /// Expand to the W output manifold.
    #[inline(always)]
    pub fn expand_w(self) -> N::OutputW
    where
        SeedZ: Clone,
        SeedW: Clone,
        StepZ: Clone,
        StepW: Clone,
    {
        N::build_w(self.seed_z, self.seed_w, self.step_z, self.step_w)
    }
}

// ============================================================================
// Helper: Type-level addition for composing iterations
// ============================================================================

/// Add two type-level naturals: `Add<N, M>::Output = N + M`
pub trait Add<M> {
    /// The sum N + M as a type.
    type Output;
}

impl<M> Add<M> for Zero {
    type Output = M; // 0 + M = M
}

impl<N, M> Add<M> for Succ<N>
where
    N: Add<M>,
{
    type Output = Succ<N::Output>; // (N+1) + M = (N + M) + 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PARALLELISM;

    #[test]
    fn zero_iterations_returns_seed() {
        // N0 iterations of anything should just return the seed
        let seed = 42.0f32;
        let step = W * 2.0; // Double the state
        let result = FixAt::<N0, _, _>::new(seed, step);

        let out = result.eval_raw(
            Field::from(0.0),
            Field::from(0.0),
            Field::from(0.0),
            Field::from(0.0),
        );

        let mut buf = [0.0f32; PARALLELISM];
        out.store(&mut buf);
        assert_eq!(buf[0], 42.0);
    }

    #[test]
    fn one_iteration_applies_step() {
        // N1: seed -> step(seed)
        // step = W * 2, seed = 3
        // Result: 3 * 2 = 6
        let seed = 3.0f32;
        let step = W * 2.0;
        let result = FixAt::<N1, _, _>::new(seed, step);

        let out = result.eval_raw(
            Field::from(0.0),
            Field::from(0.0),
            Field::from(0.0),
            Field::from(0.0),
        );

        let mut buf = [0.0f32; PARALLELISM];
        out.store(&mut buf);
        assert_eq!(buf[0], 6.0);
    }

    #[test]
    fn three_iterations() {
        // N3: seed -> step³(seed)
        // step = W * 2, seed = 1
        // Result: 1 * 2 * 2 * 2 = 8
        let seed = 1.0f32;
        let step = W * 2.0;
        let result = FixAt::<N3, _, _>::new(seed, step);

        let out = result.eval_raw(
            Field::from(0.0),
            Field::from(0.0),
            Field::from(0.0),
            Field::from(0.0),
        );

        let mut buf = [0.0f32; PARALLELISM];
        out.store(&mut buf);
        assert_eq!(buf[0], 8.0);
    }

    #[test]
    fn geometric_series() {
        // Geometric series: sum = 1 + 0.5 + 0.25 + 0.125 + ...
        // Using iteration: state = state + 0.5^n
        // We can compute this as: state_{n+1} = state + current_term
        //                         term_{n+1} = term * 0.5
        // But since we only have one state variable (W), let's do a simpler test:
        //
        // Halving iteration: x_{n+1} = x * 0.5
        // After 4 iterations: 16 -> 8 -> 4 -> 2 -> 1

        let seed = 16.0f32;
        let step = W * 0.5;

        let result = FixAt::<N4, _, _>::new(seed, step);

        let out = result.eval_raw(
            Field::from(0.0),
            Field::from(0.0),
            Field::from(0.0),
            Field::from(0.0),
        );

        let mut buf = [0.0f32; PARALLELISM];
        out.store(&mut buf);

        assert_eq!(buf[0], 1.0);
    }

    #[test]
    fn fibonacci_like() {
        // A simpler recurrence: x_{n+1} = x + n (where we use X as n)
        // This shows that FixAt properly threads X through
        // step(x, y, z, w) = w + x
        // With x=1: seed=0 -> 0+1=1 -> 1+1=2 -> 2+1=3

        let seed = 0.0f32;
        let step = W + X;

        let result = FixAt::<N3, _, _>::new(seed, step);

        // Evaluate at x=1
        let out = result.eval_raw(
            Field::from(1.0),
            Field::from(0.0),
            Field::from(0.0),
            Field::from(0.0),
        );

        let mut buf = [0.0f32; PARALLELISM];
        out.store(&mut buf);

        // 0 + 1 + 1 + 1 = 3
        assert_eq!(buf[0], 3.0);
    }

    #[test]
    fn type_expansion_is_correct() {
        // Verify that expand() produces the same result as direct eval
        let seed = 2.0f32;
        let step = W + 1.0;
        let fix = FixAt::<N3, _, _>::new(seed, step);

        // Via FixAt::eval_raw
        let out1 = fix.eval_raw(
            Field::from(0.0),
            Field::from(0.0),
            Field::from(0.0),
            Field::from(0.0),
        );

        // Via expand().eval_raw
        let expanded = FixAt::<N3, _, _>::new(2.0f32, W + 1.0).expand();
        let out2 = expanded.eval_raw(
            Field::from(0.0),
            Field::from(0.0),
            Field::from(0.0),
            Field::from(0.0),
        );

        let mut buf1 = [0.0f32; PARALLELISM];
        let mut buf2 = [0.0f32; PARALLELISM];
        out1.store(&mut buf1);
        out2.store(&mut buf2);

        // 2 + 1 + 1 + 1 = 5
        assert_eq!(buf1[0], 5.0);
        assert_eq!(buf2[0], 5.0);
    }

    // ========================================================================
    // FixAt2 Tests
    // ========================================================================

    #[test]
    fn fix_at2_zero_iterations() {
        // N0 iterations should return the seeds unchanged
        let fix = FixAt2::<N0, _, _, _, _>::new(
            3.0f32,  // seed_z
            4.0f32,  // seed_w
            Z + 1.0, // step_z (unused)
            W + 1.0, // step_w (unused)
        );

        let z_out = fix.clone().expand_z();
        let w_out = fix.expand_w();

        let z_result = z_out.eval_raw(
            Field::from(0.0),
            Field::from(0.0),
            Field::from(0.0),
            Field::from(0.0),
        );
        let w_result = w_out.eval_raw(
            Field::from(0.0),
            Field::from(0.0),
            Field::from(0.0),
            Field::from(0.0),
        );

        let mut buf = [0.0f32; PARALLELISM];
        z_result.store(&mut buf);
        assert_eq!(buf[0], 3.0);

        w_result.store(&mut buf);
        assert_eq!(buf[0], 4.0);
    }

    #[test]
    fn fix_at2_one_iteration() {
        // N1 iteration:
        // z' = z + 1, w' = w + 2
        // (3, 4) -> (4, 6)
        let fix = FixAt2::<N1, _, _, _, _>::new(
            3.0f32,  // seed_z
            4.0f32,  // seed_w
            Z + 1.0, // step_z
            W + 2.0, // step_w
        );

        let z_out = fix.clone().expand_z();
        let w_out = fix.expand_w();

        let z_result = z_out.eval_raw(
            Field::from(0.0),
            Field::from(0.0),
            Field::from(0.0),
            Field::from(0.0),
        );
        let w_result = w_out.eval_raw(
            Field::from(0.0),
            Field::from(0.0),
            Field::from(0.0),
            Field::from(0.0),
        );

        let mut buf = [0.0f32; PARALLELISM];
        z_result.store(&mut buf);
        assert_eq!(buf[0], 4.0);

        w_result.store(&mut buf);
        assert_eq!(buf[0], 6.0);
    }

    #[test]
    fn fix_at2_three_iterations() {
        // N3 iterations of doubling:
        // z' = z * 2, w' = w * 2
        // (1, 2) -> (2, 4) -> (4, 8) -> (8, 16)
        let fix = FixAt2::<N3, _, _, _, _>::new(
            1.0f32,  // seed_z
            2.0f32,  // seed_w
            Z * 2.0, // step_z
            W * 2.0, // step_w
        );

        let z_out = fix.clone().expand_z();
        let w_out = fix.expand_w();

        let z_result = z_out.eval_raw(
            Field::from(0.0),
            Field::from(0.0),
            Field::from(0.0),
            Field::from(0.0),
        );
        let w_result = w_out.eval_raw(
            Field::from(0.0),
            Field::from(0.0),
            Field::from(0.0),
            Field::from(0.0),
        );

        let mut buf = [0.0f32; PARALLELISM];
        z_result.store(&mut buf);
        assert_eq!(buf[0], 8.0);

        w_result.store(&mut buf);
        assert_eq!(buf[0], 16.0);
    }

    #[test]
    fn fix_at2_complex_squaring() {
        // Test complex multiplication: z² where z = zr + i*zi
        // (zr + i*zi)² = zr² - zi² + 2i·zr·zi
        // Real: zr² - zi²
        // Imag: 2·zr·zi
        //
        // Starting with z = (1, 1) = 1 + i
        // z² = (1-1) + 2i = 2i = (0, 2)
        // z⁴ = (2i)² = -4 = (-4, 0)
        // But we're squaring twice, so:
        // (1,1) -> (0, 2) -> (-4, 0)

        let step_real = Z * Z - W * W;
        let step_imag = Z * W * 2.0;

        let fix = FixAt2::<N2, _, _, _, _>::new(
            1.0f32, // seed_z = zr = 1
            1.0f32, // seed_w = zi = 1
            step_real,
            step_imag,
        );

        let z_out = fix.clone().expand_z();
        let w_out = fix.expand_w();

        let z_result = z_out.eval_raw(
            Field::from(0.0),
            Field::from(0.0),
            Field::from(0.0),
            Field::from(0.0),
        );
        let w_result = w_out.eval_raw(
            Field::from(0.0),
            Field::from(0.0),
            Field::from(0.0),
            Field::from(0.0),
        );

        let mut buf = [0.0f32; PARALLELISM];
        z_result.store(&mut buf);
        assert_eq!(buf[0], -4.0);

        w_result.store(&mut buf);
        assert_eq!(buf[0], 0.0);
    }

    #[test]
    fn fix_at2_with_constants_from_xy() {
        // Test that X and Y pass through correctly (for Mandelbrot's c constant)
        // z' = z + c where c comes from (X, Y)
        // Starting at z = (0, 0), c = (1, 2)
        // After 3 iterations: (0,0) -> (1,2) -> (2,4) -> (3,6)

        let step_real = Z + X;
        let step_imag = W + Y;

        let fix = FixAt2::<N3, _, _, _, _>::new(
            0.0f32, // seed_z
            0.0f32, // seed_w
            step_real,
            step_imag,
        );

        let z_out = fix.clone().expand_z();
        let w_out = fix.expand_w();

        // Evaluate at c = (1, 2)
        let z_result = z_out.eval_raw(
            Field::from(1.0),
            Field::from(2.0),
            Field::from(0.0),
            Field::from(0.0),
        );
        let w_result = w_out.eval_raw(
            Field::from(1.0),
            Field::from(2.0),
            Field::from(0.0),
            Field::from(0.0),
        );

        let mut buf = [0.0f32; PARALLELISM];
        z_result.store(&mut buf);
        assert_eq!(buf[0], 3.0);

        w_result.store(&mut buf);
        assert_eq!(buf[0], 6.0);
    }
}
