//! NNUE structures and utilities.
//!
//! Re-exports from `pixelflow_search::nnue` for convenience.

// Re-export everything from pixelflow_search::nnue
pub use pixelflow_search::nnue::*;

/// Type alias for backward compatibility.
pub type OpType = pixelflow_search::nnue::OpKind;
