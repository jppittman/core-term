//! BFloat16 scalar type for use with `Field<Bf16>`.
//!
//! `Bf16` is the top 16 bits of an IEEE 754 single-precision float:
//! - 1 sign bit
//! - 8 exponent bits (same range as f32)
//! - 7 mantissa bits (vs 23 for f32)
//!
//! The primary purpose of `Field<Bf16>` is **memory bandwidth reduction**: store
//! data in bf16 format, load it into a SIMD vector, upcast to f32 for computation,
//! then optionally downcast back to bf16 for storage. This halves memory traffic
//! compared to f32 at the cost of ~3 decimal digits of precision.
//!
//! # Layout
//!
//! bf16 is literally the upper 16 bits of f32:
//! ```text
//! f32:  SEEEEEEE EMMMMMMM MMMMMMMM MMMMMMMM
//! bf16: SEEEEEEE EMMMMMMM
//! ```
//! where S=sign, E=exponent, M=mantissa.
//!
//! This makes conversion trivial:
//! - f32 → bf16: `(f32.to_bits() >> 16) as u16` (truncation, round-toward-zero)
//! - bf16 → f32: `f32::from_bits((bf16 as u32) << 16)` (exact, zero-extends mantissa)

use crate::algebra::Algebra;

/// BFloat16 scalar type.
///
/// A 16-bit floating-point format that shares the f32 exponent range.
/// Used as the element type for `Field<Bf16>` to enable bf16 SIMD backends.
///
/// # Storage
///
/// The raw bits are stored as a `u16`. Values are interpreted as the upper
/// 16 bits of the IEEE 754 f32 representation.
///
/// # Note on arithmetic
///
/// Scalar `Bf16` arithmetic upcasts to f32, computes, then truncates back.
/// For vectorized computation, use `Field<Bf16>` which converts entire SIMD
/// vectors at once via `to_f32_lo` / `to_f32_hi`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
#[repr(transparent)]
pub struct Bf16(pub u16);

impl Bf16 {
    /// Construct from raw bf16 bits (the upper 16 bits of an f32).
    #[inline(always)]
    pub const fn from_bits(bits: u16) -> Self {
        Self(bits)
    }

    /// Return the raw bf16 bits.
    #[inline(always)]
    pub const fn to_bits(self) -> u16 {
        self.0
    }

    /// Convert f32 to bf16 by truncating the lower 16 mantissa bits (round toward zero).
    ///
    /// For round-to-nearest-even, use the SIMD conversion paths in `Field<Bf16>::from_f32`
    /// when hardware bf16 instructions are available.
    #[inline(always)]
    pub fn from_f32_truncate(val: f32) -> Self {
        Self((val.to_bits() >> 16) as u16)
    }

    /// Convert bf16 to f32 by zero-extending the mantissa.
    ///
    /// This is an exact conversion (no rounding occurs).
    #[inline(always)]
    pub fn to_f32(self) -> f32 {
        f32::from_bits((self.0 as u32) << 16)
    }

    /// bf16 representation of 0.0
    pub const ZERO: Self = Self(0x0000);

    /// bf16 representation of 1.0
    pub const ONE: Self = Self(0x3F80);

    /// bf16 representation of -1.0
    pub const NEG_ONE: Self = Self(0xBF80);

    /// bf16 positive infinity
    pub const INFINITY: Self = Self(0x7F80);

    /// bf16 NaN (quiet)
    pub const NAN: Self = Self(0x7FC0);
}

// ============================================================================
// Algebra implementation for Bf16
//
// All arithmetic routes through f32 (upcast → compute → truncate back).
// This is the scalar path; the vectorized path goes through Field<Bf16>.
// ============================================================================

impl Algebra for Bf16 {
    type Mask = bool;

    #[inline(always)]
    fn zero() -> Self {
        Self::ZERO
    }

    #[inline(always)]
    fn one() -> Self {
        Self::ONE
    }

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self::from_f32_truncate(self.to_f32() + rhs.to_f32())
    }

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self::from_f32_truncate(self.to_f32() - rhs.to_f32())
    }

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self::from_f32_truncate(self.to_f32() * rhs.to_f32())
    }

    #[inline(always)]
    fn neg(self) -> Self {
        // Toggle the sign bit — no upcast needed
        Self(self.0 ^ 0x8000)
    }

    #[inline(always)]
    fn lt(self, rhs: Self) -> bool {
        self.to_f32() < rhs.to_f32()
    }

    #[inline(always)]
    fn le(self, rhs: Self) -> bool {
        self.to_f32() <= rhs.to_f32()
    }

    #[inline(always)]
    fn gt(self, rhs: Self) -> bool {
        self.to_f32() > rhs.to_f32()
    }

    #[inline(always)]
    fn ge(self, rhs: Self) -> bool {
        self.to_f32() >= rhs.to_f32()
    }

    #[inline(always)]
    fn eq(self, rhs: Self) -> bool {
        // Compare bit patterns directly (handles +0/-0 unification if needed)
        // For IEEE semantics, NaN != NaN, which is preserved here since to_f32 roundtrips.
        self.to_f32() == rhs.to_f32()
    }

    #[inline(always)]
    fn ne(self, rhs: Self) -> bool {
        self.to_f32() != rhs.to_f32()
    }

    #[inline(always)]
    fn select(mask: bool, if_true: Self, if_false: Self) -> Self {
        if mask { if_true } else { if_false }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bf16_roundtrip() {
        // 1.0 in f32 = 0x3F800000, upper 16 bits = 0x3F80
        let one = Bf16::from_f32_truncate(1.0);
        assert_eq!(one.to_bits(), 0x3F80);
        assert_eq!(one.to_f32(), 1.0);
    }

    #[test]
    fn test_bf16_zero() {
        let z = Bf16::zero();
        assert_eq!(z.to_bits(), 0x0000);
        assert_eq!(z.to_f32(), 0.0);
    }

    #[test]
    fn test_bf16_neg() {
        let one = Bf16::ONE;
        let neg_one = one.neg();
        assert_eq!(neg_one.to_bits(), 0xBF80);
        assert_eq!(neg_one.to_f32(), -1.0);
    }

    #[test]
    fn test_bf16_add() {
        let a = Bf16::from_f32_truncate(1.0);
        let b = Bf16::from_f32_truncate(2.0);
        let c = a.add(b);
        // 3.0 in bf16
        assert_eq!(c.to_f32(), 3.0);
    }

    #[test]
    fn test_bf16_algebra_constants() {
        assert_eq!(Bf16::ZERO, Bf16::from_f32_truncate(0.0));
        assert_eq!(Bf16::ONE, Bf16::from_f32_truncate(1.0));
    }
}
