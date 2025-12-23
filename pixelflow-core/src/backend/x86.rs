//! x86_64 backend.
//!
//! SSE2 is deprecated. AVX512 coming soon.
//! Fallback to Scalar for now since SSE2 is deprecated and AVX512 is not ready/supported in this env.

pub use super::scalar::Scalar as Sse2;
pub use super::scalar::Scalar as Avx512;
