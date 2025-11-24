// src/display/driver.rs
//! DisplayDriver trait - minimal RISC interface for platform-specific display primitives.
//!
//! This trait defines the minimal set of platform-specific operations required
//! to drive a display. All complexity lives in DisplayManager (Strategy Pattern).
//!
//! ## Threading Model
//! - DisplayDriver runs on the main thread (required by macOS/Cocoa)
//! - DisplayManager runs on a background thread
//! - Communication is message-based via channels (no shared state)
//!
//! ## Lifecycle
//! 1. `new()` - Pure initialization (register classes, minimal setup)
//! 2. `handle_request(Init)` - Create window, discover metrics
//! 3. Request/response loop - All operations via messages
//! 4. `Drop` - Cleanup (no explicit shutdown message)

use crate::display::messages::{DisplayError, DriverRequest, DriverResponse};
use anyhow::Result;

/// Minimal platform-specific display driver interface.
///
/// Implementations should be RISC-style: provide only the primitives needed
/// for the platform. All common logic lives in DisplayManager.
pub trait DisplayDriver {
    /// Pure initialization only - no window creation, no resource allocation.
    ///
    /// On macOS: Register NSView class, initialize NSApp if needed.
    /// On X11: Connect to X server, verify extensions.
    ///
    /// Window creation happens in `handle_request(Init)`.
    fn new() -> Result<Self>
    where
        Self: Sized;

    /// Handle a request from DisplayManager, returning a response.
    ///
    /// This is the only method called after `new()`. All operations are
    /// message-based to enable clean threading and ownership transfer.
    ///
    /// ## Request/Response Pairs
    /// - `Init` → `InitComplete` (discover window metrics)
    /// - `PollEvents` → `Events` (fetch pending native events)
    /// - `RequestFramebuffer` → `Framebuffer` (transfer ownership to manager)
    /// - `Present(buf)` → `PresentComplete` (display and return ownership)
    /// - `SetTitle(s)` → `TitleSet`
    /// - `Bell` → `BellRung`
    /// - `SetCursorVisibility(b)` → `CursorVisibilitySet`
    /// - `CopyToClipboard(s)` → `ClipboardCopied`
    /// - `RequestPaste` → `PasteRequested` (data arrives via `PasteData` event)
    ///
    /// ## Error Handling
    /// Returns `DisplayError` instead of `anyhow::Result` to enable safe buffer recovery.
    /// When a `Present` request fails, the buffer is returned via `DisplayError::PresentationFailed`
    /// to prevent starvation of the framebuffer ping-pong pattern.
    fn handle_request(&mut self, request: DriverRequest) -> Result<DriverResponse, DisplayError>;

    // Drop trait handles cleanup - no explicit shutdown method needed
}
