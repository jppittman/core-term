// src/pixels.rs
//! Type aliases to distinguish between logical and physical pixels.
//!
//! ## Coordinate Systems
//!
//! This terminal emulator uses three coordinate systems:
//!
//! 1. **Cell Coordinates** (usize): Grid positions (column, row)
//!    - Example: (0, 0) is top-left cell
//!    - Used by: Terminal emulator, Renderer (input/output)
//!
//! 2. **Logical Pixels** (LogicalPx): Platform-independent points
//!    - On macOS: "points" (pts)
//!    - On Retina displays: 1 point = 2 physical pixels (scale_factor = 2.0)
//!    - Used by: Configuration, window sizing
//!
//! 3. **Physical Pixels** (PhysicalPx): Actual framebuffer pixels
//!    - The real pixels in the framebuffer
//!    - physical_px = logical_px * scale_factor
//!    - Used by: Rasterizer, framebuffer operations
//!
//! ## Conversion Rules
//!
//! - Cell → Physical Pixels: `cell * cell_size_physical_px`
//! - Logical → Physical: `logical_px * scale_factor`
//! - Physical → Logical: `physical_px / scale_factor`

/// Logical pixels (platform-independent points).
/// On Retina displays, 1 LogicalPx = scale_factor PhysicalPx.
pub type LogicalPx = usize;

/// Physical pixels (actual framebuffer pixels).
/// These are the real pixels that get drawn to the screen.
pub type PhysicalPx = usize;

/// Scale factor converting logical to physical pixels.
pub type ScaleFactor = f64;
