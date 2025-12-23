//! Surface-based rendering for PixelFlow v11.0 architecture.
//!
//! This module provides terminal rendering using the Manifold abstraction
//! from pixelflow-core, enabling functional composition and efficient
//! SIMD evaluation.
//!
//! # Architecture
//!
//! The terminal is rendered as a tiled manifold where each tile is a cell.
//! The type hierarchy encodes the rendering algorithm:
//!
//! ```text
//! Tile<GridLookup<G, A>>
//!   where G: CellGrid     -- provides cell data (char, fg, bg)
//!         A: GlyphAtlas   -- provides coverage manifolds for characters
//! ```
//!
//! See [`manifold`] module for the elegant type-level composition.

pub mod grid;
pub mod manifold;
pub mod terminal;

pub use grid::GridBuffer;
pub use manifold::{
    Blend, CellData, CellGrid, CellRenderer, GlyphAtlas, GridLookup, SolidColor, Terminal, Tile,
    terminal,
};
pub use terminal::TerminalSurface;
