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
//! PackRGBA {
//!   r: Select(x < mid, left_r, right_r),
//!   g: Select(x < mid, left_g, right_g),
//!   ...
//! }
//! ```
//!
//! Each color channel is a separate Select tree, enabling use of the
//! standard pixelflow-core Select combinator (which works on Field values).
//!
//! See [`manifold`] module for the elegant type-level composition.

pub mod grid;
pub mod manifold;
pub mod terminal;

pub use grid::GridBuffer;
pub use manifold::{
    build_grid, Cell, CellA, CellB, CellChannel, CellFactory, CellG, CellR, ConstCoverage,
    LocalCoords, SolidColor,
};
pub use terminal::TerminalSurface;
