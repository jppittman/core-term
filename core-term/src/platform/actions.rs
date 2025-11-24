// src/platform/actions.rs
//
// Defines actions that can be sent to the PTY or UI services.

/// Commands that can be sent to the PTY.
#[derive(Debug)]
pub enum PlatformAction {
    /// Write a byte sequence to the PTY.
    Write(Vec<u8>),
    /// Resize the PTY.
    ResizePty { cols: u16, rows: u16 },
    /// Request the platform to render the provided snapshot.
    /// The orchestrator sends this in response to RequestSnapshot events.
    /// The platform should render it and return via ControlEvent::FrameRendered.
    RequestRedraw(Box<crate::term::snapshot::TerminalSnapshot>),
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
    /// Orchestrator has completed shutdown and it's safe to cleanup platform resources.
    ShutdownComplete,
}
