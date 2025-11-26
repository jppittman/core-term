//! Platform-specific SIMD backends.
//!
//! Each backend implements the `SimdOps` trait using platform intrinsics.

#[cfg(target_arch = "x86_64")]
pub mod x86;

#[cfg(target_arch = "aarch64")]
pub mod arm;

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub mod scalar;
