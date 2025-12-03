//! SIMD batch types and backend selection.

use crate::backend::Backend;

// Select NativeBackend based on architecture
#[cfg(target_arch = "x86_64")]
pub use crate::backends::x86::Avx512 as NativeBackend;

#[cfg(target_arch = "aarch64")]
pub use crate::backends::arm::Neon as NativeBackend;

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub use crate::backends::scalar::Scalar as NativeBackend;

/// The platform-native SIMD batch type.
pub type Batch<T> = <NativeBackend as Backend>::Batch<T>;

/// Number of lanes in the native batch.
pub const LANES: usize = NativeBackend::LANES;

// Re-export common traits
pub use crate::backend::BatchOps as SimdOps;

/// Helper to calculate aligned stride.
pub const fn calculate_aligned_stride(width: usize) -> usize {
    (width + LANES - 1) & !(LANES - 1)
}

/// Shuffle mask for RGBA↔BGRA conversion.
pub const SHUFFLE_RGBA_BGRA: [u8; 16] = [2, 1, 0, 3, 6, 5, 4, 7, 10, 9, 8, 11, 14, 13, 12, 15];
