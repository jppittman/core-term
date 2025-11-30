use crate::input::{CursorIcon, KeySymbol, Modifiers, MouseButton};
use pixelflow_core::pipe::Surface;

#[derive(Debug, Clone)]
pub enum EngineEvent {
    /// Window was resized by the user or OS.
    Resize(u32, u32),
    /// User pressed a key.
    KeyDown {
        key: KeySymbol,
        mods: Modifiers,
        text: Option<String>,
    },
    /// User moved/clicked mouse.
    MouseClick { x: u32, y: u32, button: MouseButton },
    /// Mouse move
    MouseMove { x: u32, y: u32, mods: Modifiers },
    /// Mouse release
    MouseRelease { x: u32, y: u32, button: MouseButton },
    /// Paste text.
    Paste(String),
    /// Focus gained.
    FocusGained,
    /// Focus lost.
    FocusLost,
    /// The application explicitly woke the loop (e.g. from PTY thread).
    Wake,
    /// OS requested app close.
    CloseRequested,
}

#[derive(Debug, Clone)]
pub enum AppAction {
    /// Do nothing, continue waiting for events.
    Continue,
    /// State changed, schedule a render for this frame.
    Redraw,
    /// Update the window title.
    SetTitle(String),
    /// Request a window resize.
    ResizeRequest(u32, u32),
    /// Change cursor.
    SetCursorIcon(CursorIcon),
    /// Copy text to clipboard.
    CopyToClipboard(String),
    /// Request paste.
    RequestPaste,
    /// Gracefully terminate the process.
    Quit,
}

#[derive(Debug, Clone)]
pub struct AppState {
    /// Logical width in pixels (already scaled by engine)
    pub width_px: u32,
    /// Logical height in pixels (already scaled by engine)
    pub height_px: u32,
}

pub trait Application {
    /// THE DATA PLANE
    /// The Hot Path: Produce a frame based on current state.
    /// Returns a composed surface that will be materialized into the framebuffer.
    /// One vtable call per surface, then pure SIMD execution.
    fn render(&mut self, state: &AppState) -> Option<Box<dyn Surface<u32> + Send + Sync>>;

    /// THE CONTROL PLANE
    /// The Control Path: Process input or wake signals.
    /// Returns an Action to command the Host (e.g., "Set Title", "Quit").
    fn on_event(&mut self, event: EngineEvent) -> AppAction;
}
