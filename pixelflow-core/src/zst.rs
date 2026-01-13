//! Zero-sized type marker trait.
//!
//! This module provides a marker trait [`Zst`] for identifying types that have
//! no runtime representation (zero-sized types). This is useful for:
//!
//! - Static assertion that a type compiles to no data
//! - Enabling optimizations based on ZST knowledge
//! - Documenting which types are purely compile-time constructs
//!
//! # What is a ZST?
//!
//! A zero-sized type (ZST) is a type that occupies zero bytes of memory.
//! In Rust, these types exist only at compile time and have no runtime cost.
//!
//! Examples in PixelFlow:
//! - Coordinate variables (`X`, `Y`, `Z`, `W`) - no data, just identity
//! - Spherical harmonic basis functions - only const generic parameters
//! - Operators where all operands are ZSTs - no data to store
//!
//! # Example
//!
//! ```ignore
//! use pixelflow_core::{Zst, X, Y};
//! use pixelflow_core::ops::Add;
//!
//! fn assert_zst<T: Zst>() {}
//!
//! assert_zst::<X>();           // ✓ X is a ZST
//! assert_zst::<Y>();           // ✓ Y is a ZST
//! assert_zst::<Add<X, Y>>();   // ✓ Add<X, Y> is a ZST (both operands are ZSTs)
//! ```

/// Marker trait for zero-sized types.
///
/// Types implementing this trait have `size_of::<T>() == 0` and are purely
/// compile-time constructs with no runtime representation.
///
/// Note: All ZST types in pixelflow-core also implement `Copy` (via derives),
/// but this trait does not require it as a supertrait. The trait is purely
/// for identifying zero-sized types at compile time.
pub trait Zst {}

// ============================================================================
// Base ZST Implementations
// ============================================================================

// Coordinate variables
impl crate::Zst for crate::variables::X {}
impl crate::Zst for crate::variables::Y {}
impl crate::Zst for crate::variables::Z {}
impl crate::Zst for crate::variables::W {}

// Spherical harmonics
impl<const L: usize, const M: i32> crate::Zst for crate::combinators::SphericalHarmonic<L, M> {}
impl<const L: usize> crate::Zst for crate::combinators::ZonalHarmonic<L> {}
impl<const NUM_COEFFS: usize> crate::Zst for crate::combinators::ShProject<NUM_COEFFS> {}

// Backend markers (internal, but technically ZSTs)
impl crate::Zst for crate::backend::scalar::Scalar {}

#[cfg(target_arch = "x86_64")]
impl crate::Zst for crate::backend::x86::Sse2 {}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
impl crate::Zst for crate::backend::x86::Avx2 {}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
impl crate::Zst for crate::backend::x86::Avx512 {}

#[cfg(target_arch = "aarch64")]
impl crate::Zst for crate::backend::arm::Neon {}

// ============================================================================
// Blanket Implementations for Operators
// ============================================================================

// Binary operators: ZST + ZST → ZST
impl<L: crate::Zst, R: crate::Zst> crate::Zst for crate::ops::Add<L, R> {}
impl<L: crate::Zst, R: crate::Zst> crate::Zst for crate::ops::Sub<L, R> {}
impl<L: crate::Zst, R: crate::Zst> crate::Zst for crate::ops::Mul<L, R> {}
impl<L: crate::Zst, R: crate::Zst> crate::Zst for crate::ops::Div<L, R> {}
impl<L: crate::Zst, R: crate::Zst> crate::Zst for crate::ops::Max<L, R> {}
impl<L: crate::Zst, R: crate::Zst> crate::Zst for crate::ops::Min<L, R> {}
impl<L: crate::Zst, R: crate::Zst> crate::Zst for crate::ops::MulRsqrt<L, R> {}

// Ternary operators: ZST + ZST + ZST → ZST
impl<A: crate::Zst, B: crate::Zst, C: crate::Zst> crate::Zst for crate::ops::MulAdd<A, B, C> {}
impl<Acc: crate::Zst, Val: crate::Zst, Mask: crate::Zst> crate::Zst
    for crate::ops::AddMasked<Acc, Val, Mask>
{
}

// Unary operators: ZST → ZST
impl<M: crate::Zst> crate::Zst for crate::ops::Sqrt<M> {}
impl<M: crate::Zst> crate::Zst for crate::ops::Abs<M> {}
impl<M: crate::Zst> crate::Zst for crate::ops::Floor<M> {}
impl<M: crate::Zst> crate::Zst for crate::ops::Rsqrt<M> {}
impl<M: crate::Zst> crate::Zst for crate::ops::Sin<M> {}
impl<M: crate::Zst> crate::Zst for crate::ops::Cos<M> {}
impl<M: crate::Zst> crate::Zst for crate::ops::Log2<M> {}
impl<M: crate::Zst> crate::Zst for crate::ops::Exp2<M> {}
impl<M: crate::Zst> crate::Zst for crate::ops::Neg<M> {}

// Comparison operators: ZST + ZST → ZST
impl<L: crate::Zst, R: crate::Zst> crate::Zst for crate::ops::Lt<L, R> {}
impl<L: crate::Zst, R: crate::Zst> crate::Zst for crate::ops::Gt<L, R> {}
impl<L: crate::Zst, R: crate::Zst> crate::Zst for crate::ops::Le<L, R> {}
impl<L: crate::Zst, R: crate::Zst> crate::Zst for crate::ops::Ge<L, R> {}
impl<L: crate::Zst, R: crate::Zst> crate::Zst for crate::ops::SoftGt<L, R> {}
impl<L: crate::Zst, R: crate::Zst> crate::Zst for crate::ops::SoftLt<L, R> {}
impl<Mask: crate::Zst, IfTrue: crate::Zst, IfFalse: crate::Zst> crate::Zst
    for crate::ops::SoftSelect<Mask, IfTrue, IfFalse>
{
}

// Logic operators: ZST + ZST → ZST
impl<L: crate::Zst, R: crate::Zst> crate::Zst for crate::ops::And<L, R> {}
impl<L: crate::Zst, R: crate::Zst> crate::Zst for crate::ops::Or<L, R> {}
impl<M: crate::Zst> crate::Zst for crate::ops::BNot<M> {}

// ============================================================================
// Blanket Implementations for Combinators
// ============================================================================

// Select combinator: ZST + ZST + ZST → ZST
impl<C: crate::Zst, T: crate::Zst, F: crate::Zst> crate::Zst
    for crate::combinators::Select<C, T, F>
{
}

// Map combinator: ZST → ZST (but only if T is also ZST)
impl<M: crate::Zst, T: crate::Zst> crate::Zst for crate::combinators::Map<M, T> {}

// Project combinator: ZST → ZST (D is a dimension marker, always ZST)
impl<M: crate::Zst, D: crate::variables::Dimension + Copy> crate::Zst
    for crate::combinators::Project<M, D>
{
}

// At combinator: ZST + ZST + ZST + ZST + ZST → ZST
impl<Cx: crate::Zst, Cy: crate::Zst, Cz: crate::Zst, Cw: crate::Zst, M: crate::Zst> crate::Zst
    for crate::combinators::At<Cx, Cy, Cz, Cw, M>
{
}

// Fix combinator: ZST + ZST + ZST → ZST
impl<Seed: crate::Zst, Step: crate::Zst, Done: crate::Zst> crate::Zst
    for crate::combinators::Fix<Seed, Step, Done>
{
}

// ============================================================================
// Copy Implementations for ZST Types (Step 3)
// ============================================================================
//
// This is the key insight: we only want Copy on zero-sized types.
// Large expression trees should not be implicitly copied.
//
// Strategy:
// 1. Remove #[derive(Copy)] from all operators/combinators
// 2. Add blanket impl Copy only when all type parameters are Zst
// 3. Result: Add<X, Y> is Copy, but Add<Field, Field> is not

// Binary operators: Copy when both operands are ZST
impl<L: crate::Zst + Copy, R: crate::Zst + Copy> Copy for crate::ops::Add<L, R> {}
impl<L: crate::Zst + Copy, R: crate::Zst + Copy> Copy for crate::ops::Sub<L, R> {}
impl<L: crate::Zst + Copy, R: crate::Zst + Copy> Copy for crate::ops::Mul<L, R> {}
impl<L: crate::Zst + Copy, R: crate::Zst + Copy> Copy for crate::ops::Div<L, R> {}
impl<L: crate::Zst + Copy, R: crate::Zst + Copy> Copy for crate::ops::Max<L, R> {}
impl<L: crate::Zst + Copy, R: crate::Zst + Copy> Copy for crate::ops::Min<L, R> {}
impl<L: crate::Zst + Copy, R: crate::Zst + Copy> Copy for crate::ops::MulRsqrt<L, R> {}

// Ternary operators: Copy when all operands are ZST
impl<A: crate::Zst + Copy, B: crate::Zst + Copy, C: crate::Zst + Copy> Copy
    for crate::ops::MulAdd<A, B, C>
{
}
impl<Acc: crate::Zst + Copy, Val: crate::Zst + Copy, Mask: crate::Zst + Copy> Copy
    for crate::ops::AddMasked<Acc, Val, Mask>
{
}

// Unary operators: Copy when operand is ZST
impl<M: crate::Zst + Copy> Copy for crate::ops::Sqrt<M> {}
impl<M: crate::Zst + Copy> Copy for crate::ops::Abs<M> {}
impl<M: crate::Zst + Copy> Copy for crate::ops::Floor<M> {}
impl<M: crate::Zst + Copy> Copy for crate::ops::Rsqrt<M> {}
impl<M: crate::Zst + Copy> Copy for crate::ops::Sin<M> {}
impl<M: crate::Zst + Copy> Copy for crate::ops::Cos<M> {}
impl<M: crate::Zst + Copy> Copy for crate::ops::Log2<M> {}
impl<M: crate::Zst + Copy> Copy for crate::ops::Exp2<M> {}
impl<M: crate::Zst + Copy> Copy for crate::ops::Neg<M> {}

// Comparison operators: Copy when both operands are ZST
impl<L: crate::Zst + Copy, R: crate::Zst + Copy> Copy for crate::ops::Lt<L, R> {}
impl<L: crate::Zst + Copy, R: crate::Zst + Copy> Copy for crate::ops::Gt<L, R> {}
impl<L: crate::Zst + Copy, R: crate::Zst + Copy> Copy for crate::ops::Le<L, R> {}
impl<L: crate::Zst + Copy, R: crate::Zst + Copy> Copy for crate::ops::Ge<L, R> {}
impl<L: crate::Zst + Copy, R: crate::Zst + Copy> Copy for crate::ops::SoftGt<L, R> {}
impl<L: crate::Zst + Copy, R: crate::Zst + Copy> Copy for crate::ops::SoftLt<L, R> {}
impl<Mask: crate::Zst + Copy, IfTrue: crate::Zst + Copy, IfFalse: crate::Zst + Copy> Copy
    for crate::ops::SoftSelect<Mask, IfTrue, IfFalse>
{
}

// Logic operators: Copy when operands are ZST
impl<L: crate::Zst + Copy, R: crate::Zst + Copy> Copy for crate::ops::And<L, R> {}
impl<L: crate::Zst + Copy, R: crate::Zst + Copy> Copy for crate::ops::Or<L, R> {}
impl<M: crate::Zst + Copy> Copy for crate::ops::BNot<M> {}

// Combinators: Copy when all parameters are ZST
impl<C: crate::Zst + Copy, T: crate::Zst + Copy, F: crate::Zst + Copy> Copy
    for crate::combinators::Select<C, T, F>
{
}
impl<M: crate::Zst + Copy, T: crate::Zst + Copy> Copy for crate::combinators::Map<M, T> {}
impl<M: crate::Zst + Copy, D: crate::variables::Dimension + Copy> Copy
    for crate::combinators::Project<M, D>
{
}
impl<
        Cx: crate::Zst + Copy,
        Cy: crate::Zst + Copy,
        Cz: crate::Zst + Copy,
        Cw: crate::Zst + Copy,
        M: crate::Zst + Copy,
    > Copy for crate::combinators::At<Cx, Cy, Cz, Cw, M>
{
}
impl<Seed: crate::Zst + Copy, Step: crate::Zst + Copy, Done: crate::Zst + Copy> Copy
    for crate::combinators::Fix<Seed, Step, Done>
{
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Helper function to assert a type is a ZST
    fn assert_zst<T: crate::Zst>() {
        assert_eq!(core::mem::size_of::<T>(), 0);
    }

    #[test]
    fn test_coordinate_variables_are_zst() {
        assert_zst::<crate::variables::X>();
        assert_zst::<crate::variables::Y>();
        assert_zst::<crate::variables::Z>();
        assert_zst::<crate::variables::W>();
    }

    #[test]
    fn test_spherical_harmonics_are_zst() {
        assert_zst::<crate::combinators::SphericalHarmonic<0, 0>>();
        assert_zst::<crate::combinators::SphericalHarmonic<1, -1>>();
        assert_zst::<crate::combinators::ZonalHarmonic<0>>();
        assert_zst::<crate::combinators::ZonalHarmonic<2>>();
        assert_zst::<crate::combinators::ShProject<9>>();
    }

    #[test]
    fn test_operators_with_zst_operands_are_zst() {
        use crate::variables::{X, Y};

        // Binary operators
        assert_zst::<crate::ops::Add<X, Y>>();
        assert_zst::<crate::ops::Sub<X, Y>>();
        assert_zst::<crate::ops::Mul<X, Y>>();
        assert_zst::<crate::ops::Div<X, Y>>();

        // Unary operators
        assert_zst::<crate::ops::Sqrt<X>>();
        assert_zst::<crate::ops::Abs<Y>>();
        assert_zst::<crate::ops::Sin<X>>();
        assert_zst::<crate::ops::Cos<Y>>();

        // Comparison operators
        assert_zst::<crate::ops::Lt<X, Y>>();
        assert_zst::<crate::ops::Gt<X, Y>>();
    }

    #[test]
    fn test_complex_expression_is_zst() {
        use crate::variables::{X, Y};

        // Test that (X * X + Y * Y).sqrt() is a ZST
        type XSquared = crate::ops::Mul<X, X>;
        type YSquared = crate::ops::Mul<Y, Y>;
        type Sum = crate::ops::Add<XSquared, YSquared>;
        type Distance = crate::ops::Sqrt<Sum>;

        assert_zst::<Distance>();
    }

    #[test]
    fn test_combinators_with_zst_operands_are_zst() {
        use crate::variables::{X, Y, Z};

        // Select combinator
        type Cond = crate::ops::Gt<X, Y>;
        type Select = crate::combinators::Select<Cond, X, Z>;
        assert_zst::<Select>();

        // At combinator
        type At = crate::combinators::At<X, Y, Z, X, Y>;
        assert_zst::<At>();
    }
}
