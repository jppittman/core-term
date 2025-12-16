//! SIMD batch types and backend selection.

use crate::backend::{Backend, SimdBatch};
use core::fmt::Debug;
use core::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Not, Rem, Shl, Shr, Sub};

// Select NativeBackend based on architecture
#[cfg(target_arch = "x86_64")]
pub use crate::backend::x86::Sse2 as NativeBackend;

#[cfg(target_arch = "aarch64")]
pub use crate::backend::arm::Neon as NativeBackend;

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub use crate::backend::scalar::Scalar as NativeBackend;

/// The platform-native SIMD batch type.
pub type Batch<T> = <NativeBackend as Backend>::Batch<T>;

/// Number of lanes in the native batch.
pub const LANES: usize = NativeBackend::LANES;

// Re-export common traits
pub use crate::backend::BatchOps;

/// Helper to calculate aligned stride.
pub const fn calculate_aligned_stride(width: usize) -> usize {
    (width + LANES - 1) & !(LANES - 1)
}

/// Shuffle mask for RGBA↔BGRA conversion.
pub const SHUFFLE_RGBA_BGRA: [u8; 16] = [2, 1, 0, 3, 6, 5, 4, 7, 10, 9, 8, 11, 14, 13, 12, 15];

// ============================================================================
// Scalar <-> Batch Arithmetic
// ============================================================================

macro_rules! impl_scalar_ops {
    ($($t:ty),*) => {
        $(
            // Batch + Scalar
            impl Add<$t> for Batch<$t> {
                type Output = Batch<$t>;
                #[inline(always)]
                fn add(self, rhs: $t) -> Self::Output {
                    self + Batch::<$t>::splat(rhs)
                }
            }
            // Scalar + Batch
            impl Add<Batch<$t>> for $t {
                type Output = Batch<$t>;
                #[inline(always)]
                fn add(self, rhs: Batch<$t>) -> Self::Output {
                    Batch::<$t>::splat(self) + rhs
                }
            }

            // Batch - Scalar
            impl Sub<$t> for Batch<$t> {
                type Output = Batch<$t>;
                #[inline(always)]
                fn sub(self, rhs: $t) -> Self::Output {
                    self - Batch::<$t>::splat(rhs)
                }
            }
            // Scalar - Batch
            impl Sub<Batch<$t>> for $t {
                type Output = Batch<$t>;
                #[inline(always)]
                fn sub(self, rhs: Batch<$t>) -> Self::Output {
                    Batch::<$t>::splat(self) - rhs
                }
            }

            // Batch * Scalar
            impl Mul<$t> for Batch<$t> {
                type Output = Batch<$t>;
                #[inline(always)]
                fn mul(self, rhs: $t) -> Self::Output {
                    self * Batch::<$t>::splat(rhs)
                }
            }
            // Scalar * Batch
            impl Mul<Batch<$t>> for $t {
                type Output = Batch<$t>;
                #[inline(always)]
                fn mul(self, rhs: Batch<$t>) -> Self::Output {
                    Batch::<$t>::splat(self) * rhs
                }
            }

            // Batch / Scalar
            impl Div<$t> for Batch<$t> {
                type Output = Batch<$t>;
                #[inline(always)]
                fn div(self, rhs: $t) -> Self::Output {
                    self / Batch::<$t>::splat(rhs)
                }
            }
            // Scalar / Batch
            impl Div<Batch<$t>> for $t {
                type Output = Batch<$t>;
                #[inline(always)]
                fn div(self, rhs: Batch<$t>) -> Self::Output {
                    Batch::<$t>::splat(self) / rhs
                }
            }

            // Batch % Scalar
            impl Rem<$t> for Batch<$t> {
                type Output = Batch<$t>;
                #[inline(always)]
                fn rem(self, rhs: $t) -> Self::Output {
                    self % Batch::<$t>::splat(rhs)
                }
            }
            // Scalar % Batch
            impl Rem<Batch<$t>> for $t {
                type Output = Batch<$t>;
                #[inline(always)]
                fn rem(self, rhs: Batch<$t>) -> Self::Output {
                    Batch::<$t>::splat(self) % rhs
                }
            }
        )*
    }
}

macro_rules! impl_scalar_bitwise {
    ($($t:ty),*) => {
        $(
            // Batch & Scalar
            impl BitAnd<$t> for Batch<$t> {
                type Output = Batch<$t>;
                #[inline(always)]
                fn bitand(self, rhs: $t) -> Self::Output {
                    self & Batch::<$t>::splat(rhs)
                }
            }
            // Scalar & Batch
            impl BitAnd<Batch<$t>> for $t {
                type Output = Batch<$t>;
                #[inline(always)]
                fn bitand(self, rhs: Batch<$t>) -> Self::Output {
                    Batch::<$t>::splat(self) & rhs
                }
            }

            // Batch | Scalar
            impl BitOr<$t> for Batch<$t> {
                type Output = Batch<$t>;
                #[inline(always)]
                fn bitor(self, rhs: $t) -> Self::Output {
                    self | Batch::<$t>::splat(rhs)
                }
            }
            // Scalar | Batch
            impl BitOr<Batch<$t>> for $t {
                type Output = Batch<$t>;
                #[inline(always)]
                fn bitor(self, rhs: Batch<$t>) -> Self::Output {
                    Batch::<$t>::splat(self) | rhs
                }
            }

            // Batch ^ Scalar
            impl BitXor<$t> for Batch<$t> {
                type Output = Batch<$t>;
                #[inline(always)]
                fn bitxor(self, rhs: $t) -> Self::Output {
                    self ^ Batch::<$t>::splat(rhs)
                }
            }
            // Scalar ^ Batch
            impl BitXor<Batch<$t>> for $t {
                type Output = Batch<$t>;
                #[inline(always)]
                fn bitxor(self, rhs: Batch<$t>) -> Self::Output {
                    Batch::<$t>::splat(self) ^ rhs
                }
            }
        )*
    }
}

// Implement for standard types
impl_scalar_ops!(f32, f64, i8, i16, i32, u8, u16, u32);
impl_scalar_bitwise!(i8, i16, i32, u8, u16, u32);
