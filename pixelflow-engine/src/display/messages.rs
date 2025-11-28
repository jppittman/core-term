// src/display/messages.rs
//! Message types for communication between DisplayManager and DisplayDriver.
//!
//! This module defines the message-based protocol for display operations.
//! All communication happens via ownership transfer - no shared state.

use crate::input::{KeySymbol, Modifiers};
use std::fmt;

/// Custom error type for DisplayDriver operations.
///
/// This allows us to handle specific recovery scenarios (like PresentationFailed)
/// while still supporting generic errors.
#[derive(Debug)]
pub enum DisplayError {
    /// The presentation failed, but the RenderSnapshot is returned safely
    /// so the render loop does not starve.
    PresentationFailed(RenderSnapshot, String),

    /// A generic, unrecoverable error (wrapper for anyhow).
    Generic(anyhow::Error),
}

// Allow standard anyhow errors to bubble up as Generic
impl From<anyhow::Error> for DisplayError {
    fn from(e: anyhow::Error) -> Self {
        DisplayError::Generic(e)
    }
}

// Boilerplate Error trait implementation
impl std::error::Error for DisplayError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            DisplayError::Generic(e) => Some(e.root_cause()),
            _ => None,
        }
    }
}

impl fmt::Display for DisplayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DisplayError::PresentationFailed(_, reason) => {
                write!(f, "Presentation failed: {}", reason)
            }
            DisplayError::Generic(e) => write!(f, "Display error: {}", e),
        }
    }
}

/// A snapshot containing all data needed to present a frame to the display.
/// This pulls rendering state out of the driver, making it stateless.
#[derive(Debug, Clone)]
pub struct RenderSnapshot {
    /// The framebuffer pixel data (RGBA or BGRA format depending on platform).
    pub framebuffer: Box<[u8]>,
    /// Width of the framebuffer in pixels.
    pub width_px: u32,
    /// Height of the framebuffer in pixels.
    pub height_px: u32,
}

/// Configuration passed to the DisplayDriver during initialization.
/// Contains static parameters needed to set up the display window.
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
}

/// Requests sent from DisplayManager to DisplayDriver.
#[derive(Debug, Clone)]
pub enum DriverRequest {
    /// Complete initialization: Show window and return metrics.
    /// Driver responds with InitComplete containing dimensions.
    Init,

    /// Request pending native events from the platform.
    /// Driver responds with Events containing any queued events.
    PollEvents,

    /// Request ownership of the framebuffer for rendering.
    /// Driver responds with Framebuffer containing Box<[u8]>.
    RequestFramebuffer,

    /// Display the framebuffer. Manager sends this with RenderSnapshot containing
    /// framebuffer + dimensions. Driver takes ownership, displays it, and responds
    /// with PresentComplete.
    Present(RenderSnapshot),

    /// Set the window title.
    SetTitle(String),

    /// Ring the terminal bell.
    Bell,

    /// Set cursor visibility.
    SetCursorVisibility(bool),

    /// Copy text to clipboard (deprecated: use SubmitClipboardData).
    CopyToClipboard(String),

    /// Request paste from clipboard.
    RequestPaste,

    /// Submit clipboard data in response to ClipboardDataRequested event.
    /// Used by X11 selection protocol where clipboard owner must provide data on demand.
    SubmitClipboardData(String),
}

/// Responses sent from DisplayDriver to DisplayManager.
#[derive(Debug)]
pub enum DriverResponse {
    /// Initialization complete with discovered metrics.
    InitComplete {
        width_px: u32,
        height_px: u32,
        scale_factor: f64,
    },

    /// Native events that occurred.
    Events(Vec<DisplayEvent>),

    /// Framebuffer ownership transferred to manager for rendering.
    Framebuffer(Box<[u8]>),

    /// Frame presentation complete, RenderSnapshot ownership returned to manager for reuse.
    PresentComplete(RenderSnapshot),

    /// Window title was set.
    TitleSet,

    /// Bell was rung.
    BellRung,

    /// Cursor visibility was set.
    CursorVisibilitySet,

    /// Text was copied to clipboard.
    ClipboardCopied,

    /// Paste was requested (data will arrive via PasteData event).
    PasteRequested,

    /// Clipboard data was submitted successfully.
    ClipboardDataSubmitted,
}

/// Platform-agnostic display events.
#[derive(Debug, Clone)]
pub enum DisplayEvent {
    /// Key press event.
    Key {
        symbol: KeySymbol,
        modifiers: Modifiers,
        text: Option<String>,
    },

    /// Window/framebuffer resize.
    Resize { width_px: u32, height_px: u32 },

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
    /// Driver should respond with SubmitClipboardData containing the current clipboard text.
    ClipboardDataRequested,
}
