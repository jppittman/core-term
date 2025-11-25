// src/renderer/mod.rs
//! Renderer module - converts terminal state to drawing commands and rasterizes to pixels.

pub mod actor;
pub mod rasterizer;
mod renderer;

pub use actor::{spawn_render_thread, RenderChannels, RenderResult, RenderWork};
pub use rasterizer::{FontBackend, FramebufferConfig, Rasterizer};
pub use renderer::Renderer;

#[cfg(test)]
mod renderer_tests;
