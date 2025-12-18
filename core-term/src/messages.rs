//! Message types for App thread <-> Engine proxy communication.

use pixelflow_runtime::EngineEvent;

/// Messages sent from proxy (engine thread) to worker (app thread).
#[derive(Debug)]
pub enum AppEvent {
    /// Display event from engine (keyboard, mouse, resize, focus, etc.)
    Engine(EngineEvent),
    /// Shutdown signal.
    Shutdown,
}

/// Render request from proxy to worker.
#[derive(Debug, Clone, Copy)]
pub struct RenderRequest {
    pub width_px: u32,
    pub height_px: u32,
}
