// src/display/messages.rs
//! Message types for display system.
//!
//! - WindowId: Opaque window identifier for multi-window support
//! - DisplayEvent: Platform events sent from driver to engine

use crate::input::{KeySymbol, Modifiers};
use serde::{Deserialize, Serialize};

/// Opaque window identifier for multi-window support.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WindowId(pub u32);

impl WindowId {
    /// The default/primary window ID.
    pub const PRIMARY: WindowId = WindowId(0);
}

/// Platform-agnostic display events sent from driver to engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DisplayEvent {
    // ========================================================================
    // Lifecycle
    // ========================================================================
    /// Window was successfully created.
    WindowCreated {
        id: WindowId,
        width_px: u32,
        height_px: u32,
        scale: f64,
    },

    /// Window was destroyed.
    WindowDestroyed { id: WindowId },

    /// User requested window close (e.g., clicked X button).
    CloseRequested { id: WindowId },

    // ========================================================================
    // Resize / Scale
    // ========================================================================
    /// Window/framebuffer was resized (physical pixels).
    Resized {
        id: WindowId,
        width_px: u32,
        height_px: u32,
    },

    /// Display scale factor changed (e.g., moved to different DPI monitor).
    ScaleChanged { id: WindowId, scale: f64 },

    // ========================================================================
    // Input
    // ========================================================================
    /// Key press/release event.
    Key {
        id: WindowId,
        symbol: KeySymbol,
        modifiers: Modifiers,
        text: Option<String>,
    },

    // ========================================================================
    // Mouse
    // ========================================================================
    /// Mouse button press.
    MouseButtonPress {
        id: WindowId,
        button: u8,
        x: i32,
        y: i32,
        modifiers: Modifiers,
    },

    /// Mouse button release.
    MouseButtonRelease {
        id: WindowId,
        button: u8,
        x: i32,
        y: i32,
        modifiers: Modifiers,
    },

    /// Mouse movement.
    MouseMove {
        id: WindowId,
        x: i32,
        y: i32,
        modifiers: Modifiers,
    },

    /// Mouse scroll wheel.
    MouseScroll {
        id: WindowId,
        dx: f32,
        dy: f32,
        x: i32,
        y: i32,
        modifiers: Modifiers,
    },

    // ========================================================================
    // Focus
    // ========================================================================
    /// Window gained focus.
    FocusGained { id: WindowId },

    /// Window lost focus.
    FocusLost { id: WindowId },

    // ========================================================================
    // Clipboard (global, no window ID)
    // ========================================================================
    /// Paste data from clipboard.
    PasteData { text: String },

    /// Clipboard data requested by another application (X11 SelectionRequest).
    ClipboardDataRequested,
}

// ============================================================================
// Legacy types (deprecated, will be removed)
// ============================================================================

/// Configuration passed to the driver via Configure command.
///
/// **DEPRECATED**: Use `DriverCommand::CreateWindow` instead.
/// This struct will be removed in a future version.
#[deprecated(note = "Use DriverCommand::CreateWindow instead")]
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
    pub target_fps: u32,
}
