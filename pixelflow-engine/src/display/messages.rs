// src/display/messages.rs
//! Message types for display system.

use crate::input::{KeySymbol, Modifiers};
use serde::{Deserialize, Serialize};

// ============================================================================
// Display Events (Driver -> Engine)
// ============================================================================

/// Unique identifier for a window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct WindowId(pub u64);

impl WindowId {
    /// Primary window ID (0)
    pub const PRIMARY: WindowId = WindowId(0);
}

/// Events from the display driver.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DisplayEvent {
    /// Window created
    WindowCreated {
        id: WindowId,
        width_px: u32,
        height_px: u32,
        scale: f64,
        refresh_rate: f64,
    },
    /// Window resized
    Resized {
        id: WindowId,
        width_px: u32,
        height_px: u32,
    },
    /// Window close requested by user
    CloseRequested { id: WindowId },
    /// Window destroyed
    WindowDestroyed { id: WindowId },
    /// DPI scale factor changed
    ScaleChanged { id: WindowId, scale: f64 },
    /// Keyboard key press/release
    Key {
        id: WindowId,
        symbol: KeySymbol,
        state: ElementState,
        modifiers: Modifiers,
        text: Option<String>,
    },
    /// Mouse button press
    MouseButtonPress {
        id: WindowId,
        button: u8,
        x: u32,
        y: u32,
        modifiers: Modifiers,
    },
    /// Mouse button release
    MouseButtonRelease {
        id: WindowId,
        button: u8,
        x: u32,
        y: u32,
        modifiers: Modifiers,
    },
    /// Mouse move
    MouseMove {
        id: WindowId,
        x: u32,
        y: u32,
        modifiers: Modifiers,
    },
    /// Mouse scroll
    MouseScroll {
        id: WindowId,
        dx: f64,
        dy: f64,
        x: u32,
        y: u32,
        modifiers: Modifiers,
    },
    /// Focus gained
    FocusGained { id: WindowId },
    /// Focus lost
    FocusLost { id: WindowId },
    /// Paste data received
    PasteData { text: String },
    /// Clipboard data requested (for copy)
    ClipboardDataRequested,
}

/// Element state (Pressed/Released)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ElementState {
    Pressed,
    Released,
}
