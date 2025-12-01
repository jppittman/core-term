// src/display/messages.rs
//! Message types for display system.
//!
//! - DisplayEvent: Platform events sent from driver to engine
//! - RenderSnapshot: Framebuffer data for presentation
//! - DriverConfig: Window configuration

use crate::input::{KeySymbol, Modifiers};
use serde::{Deserialize, Serialize};

/// Configuration passed to the driver via Configure command.
/// Contains parameters needed to set up the display window.
#[derive(Debug, Clone)]
pub struct DriverConfig {
    pub initial_window_x: f64,
    pub initial_window_y: f64,
    pub initial_cols: usize,
    pub initial_rows: usize,
    pub cell_width_px: usize,
    pub cell_height_px: usize,
    pub bytes_per_pixel: usize,
    pub bits_per_component: usize,
    pub bits_per_pixel: usize,
    pub max_draw_latency_seconds: f64,
    /// Target frames per second for the vsync loop.
    /// Engine will call render() at this rate.
    pub target_fps: u32,
}

/// Platform-agnostic display events.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DisplayEvent {
    /// Key press event.
    Key {
        symbol: KeySymbol,
        modifiers: Modifiers,
        text: Option<String>,
    },

    /// Window/framebuffer resize (physical pixels).
    Resize {
        width_px: u32,
        height_px: u32,
        scale_factor: f64,
    },

    /// User requested window close.
    CloseRequested,

    /// Window gained focus.
    FocusGained,

    /// Window lost focus.
    FocusLost,

    /// Mouse button press.
    MouseButtonPress {
        button: u8,
        x: i32,
        y: i32,
        scale_factor: f64,
        modifiers: Modifiers,
    },

    /// Mouse button release.
    MouseButtonRelease {
        button: u8,
        x: i32,
        y: i32,
        scale_factor: f64,
        modifiers: Modifiers,
    },

    /// Mouse movement.
    MouseMove {
        x: i32,
        y: i32,
        scale_factor: f64,
        modifiers: Modifiers,
    },

    /// Paste data from clipboard.
    PasteData { text: String },

    /// Clipboard data requested by another application (X11 SelectionRequest).
    ClipboardDataRequested,
}
