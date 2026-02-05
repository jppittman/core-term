//! Terminal surface implementation.
//!
//! NOTE: This module is temporarily stubbed during the pixelflow-core color system refactoring.
//! The previous implementation depended on APIs (Batch, surfaces::*, traits::Manifold)
//! that have been replaced with the new Field-based Manifold system.
//!
//! TODO: Reimplement using the new pixelflow-core API:
//! - Field instead of Batch<C>
//! - Manifold<I> with Output associated type instead of Manifold<P, C>
//! - New combinator patterns
//! - Skip graph construction for clean lines (is_dirty == false)

use crate::term::snapshot::TerminalSnapshot;
use std::sync::Arc;

/// A terminal rendered as a functional surface.
///
/// Placeholder implementation - needs to be updated for new pixelflow-core API.
pub struct TerminalSurface {
    _placeholder: (),
}

impl TerminalSurface {
    /// Creates a new terminal surface from a snapshot.
    ///
    /// When building the manifold graph, lines where `is_dirty == false`
    /// should be skipped - their pixels haven't changed since last frame.
    pub fn from_snapshot(
        _snapshot: &TerminalSnapshot,
        _glyph_factory: Arc<dyn Fn(char) + Send + Sync>,
        _cell_width: u32,
        _cell_height: u32,
    ) -> Self {
        // TODO: Build manifold graph here, skipping clean lines:
        // for line in snapshot.lines.iter() {
        //     if !line.is_dirty { continue; }
        //     // ... build glyph manifolds for this row
        // }
        Self { _placeholder: () }
    }

    /// Creates a terminal surface from a pre-computed flat list of cells.
    #[must_use]
    pub fn from_cells(_cells: Vec<()>, _cols: usize, _cell_width: u32, _cell_height: u32) -> Self {
        Self { _placeholder: () }
    }

    /// Creates a new terminal surface.
    #[must_use]
    pub fn new(_cols: usize, _rows: usize, _cell_width: u32, _cell_height: u32) -> Self {
        Self { _placeholder: () }
    }
}
