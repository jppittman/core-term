use crate::input::{KeySymbol, Modifiers, MouseButton};
use pixelflow_render::Frame;

/// Events sent from the Engine to the Application.
#[derive(Debug, Clone)]
pub enum EngineEvent {
    /// Control events (window resize, close, etc.)
    Control(EngineEventControl),
    /// Data events (input, file drops, etc.)
    Management(EngineEventManagement),
    /// Frame request (VSync)
    Data(EngineEventData),
}

#[derive(Debug, Clone)]
pub enum EngineEventControl {
    Resize(u32, u32),
    CloseRequested,
    ScaleChanged(f64),
}

#[derive(Debug, Clone)]
pub enum EngineEventManagement {
    KeyDown {
        key: KeySymbol,
        mods: Modifiers,
        text: Option<String>,
    },
    MouseClick {
        x: u32,
        y: u32,
        button: MouseButton,
    },
    MouseRelease {
        x: u32,
        y: u32,
        button: MouseButton,
    },
    MouseMove {
        x: u32,
        y: u32,
        mods: Modifiers,
    },
    MouseScroll {
        x: u32,
        y: u32,
        dx: f32,
        dy: f32,
        mods: Modifiers,
    },
    FocusGained,
    FocusLost,
    Paste(String),
}

#[derive(Debug, Clone)]
pub enum EngineEventData {
    RequestFrame {
        timestamp: std::time::Instant,
        target_timestamp: std::time::Instant,
        refresh_interval: std::time::Duration,
    },
}

/// Commands sent from the Application to the Engine.
pub enum AppData<P: pixelflow_core::Pixel> {
    RenderSurface(std::sync::Arc<dyn pixelflow_core::Surface<P, f32> + Send + Sync>),
}

impl<P: pixelflow_core::Pixel> std::fmt::Debug for AppData<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RenderSurface(_) => f.debug_tuple("RenderSurface").finish(),
        }
    }
}

/// Application trait that defines the logic.
pub trait Application {
    fn send(&self, event: EngineEvent) -> anyhow::Result<()>;
}

/// Application management commands (change title, etc.)
#[derive(Debug, Clone)]
pub enum AppManagement {
    SetTitle(String),
    ResizeRequest(u32, u32),
    CopyToClipboard(String),
    RequestPaste,
    SetCursorIcon(CursorIcon),
}

#[derive(Debug, Clone, Copy)]
pub enum CursorIcon {
    Default,
    Pointer,
    Text,
}
