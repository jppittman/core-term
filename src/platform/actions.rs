// src/platform/actions.rs
//
// Defines actions that can be sent to the PTY or UI services.

/// Commands that can be sent to the PTY.
#[derive(Debug, Clone)]
pub enum PlatformAction {
    /// Write a byte sequence to the PTY.
    Write(Vec<u8>),
    /// Resize the PTY.
    ResizePty { cols: u16, rows: u16 },
    /// Render a list of commands to the UI.
    Render(Vec<crate::platform::backends::RenderCommand>),
    /// Set the title of the window.
    SetTitle(String),
    /// Ring the terminal bell.
    RingBell,
    /// Copy text to the clipboard.
    CopyToClipboard(String),
    /// Set the visibility of the cursor.
    SetCursorVisibility(bool),
    /// Request Paste Data from the clipboard.
    RequestPaste,
}
