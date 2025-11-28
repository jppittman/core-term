// src/renderer/mod.rs
//! Renderer module - converts terminal state to drawing commands and rasterizes to pixels.

mod renderer;
pub mod types;

pub use renderer::Renderer;
pub use types::{PlatformState, RenderCommand};

#[cfg(test)]
mod renderer_tests;
