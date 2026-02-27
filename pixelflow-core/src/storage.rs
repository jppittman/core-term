//! # Storage Mapping for Field<A>
//!
//! This module provides the type-level mapping from Algebra types to their
//! SIMD storage representations. This is the core of the `Field<A>` unification.
//!
//! ## Mapping
//!
//! | Algebra Type | Storage Type |
//! |--------------|--------------|
//! | `f32` | `NativeSimd` (F32x16, F32x8, etc.) |
//! | `bool` | `NativeMask` (Mask16, etc.) |
//! | `u32` | `NativeU32Simd` (U32x16, etc.) |
//! | `Dual<N, A>` | Compound: (Storage<A>, [Storage<A>; N]) |

use crate::algebra::Algebra;
use crate::backend::{SimdBf16Ops, SimdOps, SimdU32Ops};

/// Trait that maps an Algebra type to its SIMD storage representation.
///
/// This trait enables `Field<A>` to work with different algebra types by
/// providing the appropriate SIMD storage for each.
pub trait FieldStorage: Algebra {
    /// The SIMD storage type for this algebra.
    type Storage: Copy + Clone + Send + Sync;

    /// Create storage by splatting a scalar value.
    fn splat_storage(val: Self) -> Self::Storage;

    /// Create zero storage.
    fn zero_storage() -> Self::Storage;

    /// Create one storage.
    fn one_storage() -> Self::Storage;
}

// ============================================================================
// Storage Implementation for f32
// ============================================================================

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f", pixelflow_avx512f))]
type NativeF32Storage = crate::backend::x86::F32x16;

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(all(target_feature = "avx512f", pixelflow_avx512f)),
    pixelflow_avx2
))]
type NativeF32Storage = crate::backend::x86::F32x8;

#[cfg(all(
    target_arch = "x86_64",
    not(all(target_feature = "avx512f", pixelflow_avx512f)),
    not(all(target_feature = "avx2", pixelflow_avx2))
))]
type NativeF32Storage = crate::backend::x86::F32x4;

#[cfg(target_arch = "aarch64")]
type NativeF32Storage = crate::backend::arm::F32x4;

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
type NativeF32Storage = crate::backend::scalar::ScalarF32;

impl FieldStorage for f32 {
    type Storage = NativeF32Storage;

    #[inline(always)]
    fn splat_storage(val: Self) -> Self::Storage {
        <Self::Storage as SimdOps>::splat(val)
    }

    #[inline(always)]
    fn zero_storage() -> Self::Storage {
        <Self::Storage as SimdOps>::splat(0.0)
    }

    #[inline(always)]
    fn one_storage() -> Self::Storage {
        <Self::Storage as SimdOps>::splat(1.0)
    }
}

// ============================================================================
// Storage Implementation for bool (Mask)
// ============================================================================

/// Native SIMD mask storage type for the current platform.
///
/// This is the native mask type from the IR backend:
/// - AVX-512: `Mask16` (k-register, runs on separate execution unit)
/// - AVX2: `Mask8` (float-encoded in YMM registers)
/// - SSE2: `Mask4` (float-encoded in XMM registers)
/// - NEON: `Mask4`
/// - Scalar: `MaskScalar`
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f", pixelflow_avx512f))]
/// Native mask storage for AVX-512.
pub type NativeMaskStorage = crate::backend::x86::Mask16;

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(all(target_feature = "avx512f", pixelflow_avx512f)),
    pixelflow_avx2
))]
/// Native mask storage for AVX2.
pub type NativeMaskStorage = crate::backend::x86::Mask8;

#[cfg(all(
    target_arch = "x86_64",
    not(all(target_feature = "avx512f", pixelflow_avx512f)),
    not(all(target_feature = "avx2", pixelflow_avx2))
))]
/// Native mask storage for SSE2 (fallback).
pub type NativeMaskStorage = crate::backend::x86::Mask4;

#[cfg(target_arch = "aarch64")]
/// Native mask storage for ARM NEON.
pub type NativeMaskStorage = crate::backend::arm::Mask4;

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
/// Native mask storage for scalar fallback.
pub type NativeMaskStorage = crate::backend::scalar::MaskScalar;

impl FieldStorage for bool {
    type Storage = NativeMaskStorage;

    #[inline(always)]
    fn splat_storage(val: Self) -> Self::Storage {
        // MaskOps doesn't have splat, so we use bitwise ops
        if val {
            // All bits set
            !<Self::Storage as Default>::default() | <Self::Storage as Default>::default()
        } else {
            <Self::Storage as Default>::default()
        }
    }

    #[inline(always)]
    fn zero_storage() -> Self::Storage {
        <Self::Storage as Default>::default()
    }

    #[inline(always)]
    fn one_storage() -> Self::Storage {
        !<Self::Storage as Default>::default()
    }
}

// ============================================================================
// Storage Implementation for u32
// ============================================================================

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f", pixelflow_avx512f))]
type NativeU32Storage = crate::backend::x86::U32x16;

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(all(target_feature = "avx512f", pixelflow_avx512f)),
    pixelflow_avx2
))]
type NativeU32Storage = crate::backend::x86::U32x8;

#[cfg(all(
    target_arch = "x86_64",
    not(all(target_feature = "avx512f", pixelflow_avx512f)),
    not(all(target_feature = "avx2", pixelflow_avx2))
))]
type NativeU32Storage = crate::backend::x86::U32x4;

#[cfg(target_arch = "aarch64")]
type NativeU32Storage = crate::backend::arm::U32x4;

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
type NativeU32Storage = crate::backend::scalar::ScalarU32;

impl FieldStorage for u32 {
    type Storage = NativeU32Storage;

    #[inline(always)]
    fn splat_storage(val: Self) -> Self::Storage {
        <Self::Storage as SimdU32Ops>::splat(val)
    }

    #[inline(always)]
    fn zero_storage() -> Self::Storage {
        <Self::Storage as SimdU32Ops>::splat(0)
    }

    #[inline(always)]
    fn one_storage() -> Self::Storage {
        <Self::Storage as SimdU32Ops>::splat(1)
    }
}

// ============================================================================
// Storage Implementation for Bf16
// ============================================================================

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f", pixelflow_avx512f))]
type NativeBf16Storage = crate::backend::x86::BF16x32;

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(all(target_feature = "avx512f", pixelflow_avx512f)),
    pixelflow_avx2
))]
type NativeBf16Storage = crate::backend::x86::BF16x16;

#[cfg(all(
    target_arch = "x86_64",
    not(all(target_feature = "avx512f", pixelflow_avx512f)),
    not(all(target_feature = "avx2", pixelflow_avx2))
))]
type NativeBf16Storage = crate::backend::x86::BF16x8;

#[cfg(target_arch = "aarch64")]
type NativeBf16Storage = crate::backend::arm::BF16x8;

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
type NativeBf16Storage = crate::backend::scalar::ScalarBf16;

impl FieldStorage for crate::bf16::Bf16 {
    type Storage = NativeBf16Storage;

    #[inline(always)]
    fn splat_storage(val: Self) -> Self::Storage {
        <Self::Storage as SimdBf16Ops>::splat(val.to_bits())
    }

    #[inline(always)]
    fn zero_storage() -> Self::Storage {
        <Self::Storage as SimdBf16Ops>::splat(crate::bf16::Bf16::ZERO.to_bits())
    }

    #[inline(always)]
    fn one_storage() -> Self::Storage {
        <Self::Storage as SimdBf16Ops>::splat(crate::bf16::Bf16::ONE.to_bits())
    }
}

// ============================================================================
// Storage Implementation for Dual<N, A> (Automatic Differentiation)
// ============================================================================

use crate::dual::Dual;

/// SIMD storage for dual numbers.
///
/// Stores a SIMD batch of dual numbers in Structure-of-Arrays layout:
/// - `val`: SIMD vector of function values
/// - `partials`: Array of SIMD vectors for partial derivatives
///
/// This is more efficient than Array-of-Structures because SIMD operations
/// can be applied uniformly to all values or all derivatives.
#[derive(Copy, Clone, Debug)]
pub struct DualStorage<const N: usize, A: FieldStorage> {
    /// SIMD storage for the function values
    pub val: A::Storage,
    /// SIMD storage for each partial derivative
    pub partials: [A::Storage; N],
}

impl<const N: usize, A: FieldStorage> Default for DualStorage<N, A>
where
    A::Storage: Default,
{
    fn default() -> Self {
        Self {
            val: A::Storage::default(),
            partials: core::array::from_fn(|_| A::Storage::default()),
        }
    }
}

impl<const N: usize, A: FieldStorage> FieldStorage for Dual<N, A>
where
    A::Storage: Default,
{
    type Storage = DualStorage<N, A>;

    #[inline(always)]
    fn splat_storage(val: Self) -> Self::Storage {
        DualStorage {
            val: A::splat_storage(val.val),
            partials: core::array::from_fn(|i| A::splat_storage(val.partials[i])),
        }
    }

    #[inline(always)]
    fn zero_storage() -> Self::Storage {
        DualStorage {
            val: A::zero_storage(),
            partials: core::array::from_fn(|_| A::zero_storage()),
        }
    }

    #[inline(always)]
    fn one_storage() -> Self::Storage {
        // One has value 1 and zero derivatives (constant function)
        DualStorage {
            val: A::one_storage(),
            partials: core::array::from_fn(|_| A::zero_storage()),
        }
    }
}
