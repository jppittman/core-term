use core::ops::{Add, BitAnd, BitOr, Mul, Not, Shl, Shr, Sub};

/// Trait for types supported by SIMD operations.
pub trait SimdElement: Copy {}
impl SimdElement for u32 {}
impl SimdElement for u16 {}
impl SimdElement for u8 {}

/// Abstract interface for SIMD vector types.
///
/// This trait allows pipelines to be written generically over the SIMD width.
pub trait Simd:
    Copy
    + Clone
    + Sized
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + BitAnd<Output = Self>
    + BitOr<Output = Self>
    + Not<Output = Self>
    + Shl<i32, Output = Self>
    + Shr<i32, Output = Self>
{
    /// The element type (e.g., u32, u16, u8).
    type Element: SimdElement;

    /// The number of lanes in this vector.
    const LANES: usize;

    /// Creates a vector with all lanes set to `val`.
    fn splat(val: Self::Element) -> Self;

    /// Selects elements from `self` or `other` based on `mask`.
    ///
    /// Typically: `(self & mask) | (other & !mask)`.
    fn select(self, other: Self, mask: Self) -> Self;

    /// Computes the minimum of two vectors.
    fn min(self, other: Self) -> Self;

    /// Computes the maximum of two vectors.
    fn max(self, other: Self) -> Self;

    /// Computes saturating addition.
    fn saturating_add(self, other: Self) -> Self;

    /// Computes saturating subtraction.
    fn saturating_sub(self, other: Self) -> Self;

    /// The type resulting from a cast to element type `U`.
    type Cast<U: SimdElement>: Simd<Element = U>;

    /// Casts to another SIMD vector with element type `U`.
    fn cast<U: SimdElement>(self) -> Self::Cast<U>;

    /// Returns a vector containing the lane indices (0, 1, 2, ...).
    fn iota() -> Self;

    /// Loads a vector from a pointer.
    unsafe fn load(ptr: *const Self::Element) -> Self;

    /// Stores the vector to a pointer.
    unsafe fn store(self, ptr: *mut Self::Element);
}
