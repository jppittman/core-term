//! Surface-based rendering for PixelFlow v11.0 architecture.
//!
//! This module provides the `TerminalSurface` which implements the
//! `Surface<u32>` trait from pixelflow-core, enabling functional
//! composition and zero-copy rendering.

pub mod grid;
pub mod terminal;

pub use grid::GridBuffer;
pub use terminal::TerminalSurface;
