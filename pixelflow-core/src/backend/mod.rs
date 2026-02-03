//! Backend trait and SIMD operations.
//!
//! This module re-exports the SIMD backend from pixelflow-ir.
//! All backend types and traits are owned by pixelflow-ir, while
//! pixelflow-core owns the Field wrapper and user-facing API.

// Re-export all backend types from IR
pub use pixelflow_ir::backend::*;
