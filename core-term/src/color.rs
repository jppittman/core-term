// src/color.rs

//! Re-exports color types from pixelflow-render.
//!
//! The Color type and all conversion logic now lives in pixelflow-render,
//! as it's part of the renderer's semantic API.

// Re-export color types from pixelflow-render
pub use pixelflow_graphics::render::{Color, NamedColor};
