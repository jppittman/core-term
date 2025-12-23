//! Surface-based rendering for PixelFlow v11.0 architecture.
//!
//! This module provides terminal rendering using the Manifold abstraction
//! from pixelflow-core, enabling functional composition and efficient
//! SIMD evaluation.
//!
//! # Architecture
//!
//! The terminal grid is built as a binary search tree of Select combinators:
//!
//! ```text
//! ColorManifold::new(
//!   Select { cond: Lt(X, mid), if_true: left_r, if_false: right_r },
//!   Select { cond: Lt(X, mid), if_true: left_g, if_false: right_g },
//!   ...
//! )
//! ```
//!
//! Each color channel is a separate Select tree, enabling use of the
//! standard pixelflow-core Select combinator (which works on Field values).
//!
//! See [`manifold`] module for the elegant type-level composition.

pub mod manifold;
pub mod terminal;

pub use manifold::{
    build_grid, Cell, CellA, CellB, CellChannel, CellFactory, CellG, CellR, Color, ConstCoverage,
    LocalCoords,
};
pub use terminal::TerminalSurface;
