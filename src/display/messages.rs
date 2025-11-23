// src/display/messages.rs
//! Message types for communication between DisplayManager and DisplayDriver.
//!
//! This module defines the message-based protocol for display operations.
//! All communication happens via ownership transfer - no shared state.

use crate::platform::backends::{KeySymbol, Modifiers};

/// Requests sent from DisplayManager to DisplayDriver.
#[derive(Debug, Clone)]
pub enum DriverRequest {
    /// Phase 1 initialization: Driver creates window and discovers metrics.
    /// Driver responds with InitComplete containing dimensions.
    Init,

    /// Request pending native events from the platform.
    /// Driver responds with Events containing any queued events.
    PollEvents,

    /// Request ownership of the framebuffer for rendering.
    /// Driver responds with Framebuffer containing Box<[u8]>.
    RequestFramebuffer,

    /// Display the framebuffer. Manager sends this with framebuffer ownership.
    /// Driver takes ownership, displays it, and responds with PresentComplete.
    Present(Box<[u8]>),

    /// Set the window title.
    SetTitle(String),

    /// Ring the terminal bell.
    Bell,

    /// Set cursor visibility.
    SetCursorVisibility(bool),

    /// Copy text to clipboard.
    CopyToClipboard(String),

    /// Request paste from clipboard.
    RequestPaste,
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

    /// Frame presentation complete, framebuffer ownership returned to manager for reuse.
    PresentComplete(Box<[u8]>),

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
        modifiers: Modifiers,
    },

    /// Mouse button release.
    MouseButtonRelease {
        button: u8,
        x: i32,
        y: i32,
        modifiers: Modifiers,
    },

    /// Mouse movement.
    MouseMove {
        x: i32,
        y: i32,
        modifiers: Modifiers,
    },

    /// Paste data from clipboard.
    PasteData { text: String },
}
