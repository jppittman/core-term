//! Platform-specific SIMD backends.

// Scalar backend is always available for testing/reference
pub mod scalar;

#[cfg(target_arch = "x86_64")]
pub mod x86;

#[cfg(target_arch = "aarch64")]
pub mod arm;

// Legacy scalar fallback (simulating SIMD)
// pub mod scalar_simd;
