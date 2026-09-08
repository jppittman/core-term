//! Engine/app event and action vocabulary.
//!
//! The `Application<P>` trait that used to live here — one `render` method
//! returning `Option<Box<dyn Manifold<Output = Discrete>>>` — had no
//! implementor in the tree and was shadowed by [`api::public::Application`]
//! (`crate::api::public`), which is the one the engine actually calls. It went
//! with the per-batch lane in S4a.

use crate::input::{CursorIcon, KeySymbol, Modifiers, MouseButton};

#[derive(Debug, Clone)]
pub enum EngineEvent {
    /// Window was resized by the user or OS.
    Resize(u32, u32),
    /// Display scale factor changed (e.g., moved to different DPI monitor).
    ScaleChanged(f64),
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
    /// Mouse scroll wheel.
    MouseScroll {
        x: u32,
        y: u32,
        dx: f32,
        dy: f32,
        mods: Modifiers,
    },
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
