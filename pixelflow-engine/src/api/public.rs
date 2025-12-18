use crate::input::{KeySymbol, Modifiers, MouseButton};
// use pixelflow_render::Frame;

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
    RenderSurfaceU32(std::sync::Arc<dyn pixelflow_core::Surface<P, u32> + Send + Sync>),
    Skipped,
}

impl<P: pixelflow_core::Pixel> std::fmt::Debug for AppData<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RenderSurface(_) => f.debug_tuple("RenderSurface").finish(),
            Self::RenderSurfaceU32(_) => f.debug_tuple("RenderSurfaceU32").finish(),
            Self::Skipped => f.debug_tuple("Skipped").finish(),
        }
    }
}

/// Application trait that defines the logic.
pub trait Application {
    fn send(&self, event: EngineEvent) -> anyhow::Result<()>;
}

impl Application
    for actor_scheduler::ActorHandle<EngineEventData, EngineEventControl, EngineEventManagement>
{
    fn send(&self, event: EngineEvent) -> anyhow::Result<()> {
        let msg = match event {
            EngineEvent::Control(ctrl) => actor_scheduler::Message::Control(ctrl),
            EngineEvent::Management(mgmt) => actor_scheduler::Message::Management(mgmt),
            EngineEvent::Data(data) => actor_scheduler::Message::Data(data),
        };
        self.send(msg)
            .map_err(|e| anyhow::anyhow!("Failed to send event to application: {}", e))
    }
}

/// Application management commands (change title, etc.)
#[derive(Debug, Clone)]
pub enum AppManagement {
    SetTitle(String),
    ResizeRequest(u32, u32),
    CopyToClipboard(String),
    RequestPaste,
    SetCursorIcon(CursorIcon),
    Quit,
}

#[derive(Debug, Clone, Copy)]
pub enum CursorIcon {
    Default,
    Pointer,
    Text,
}

/// Descriptor for creating a new window.
#[derive(Debug, Clone)]
pub struct WindowDescriptor {
    pub width: u32,
    pub height: u32,
    pub title: String,
    pub resizable: bool,
}

impl Default for WindowDescriptor {
    fn default() -> Self {
        Self {
            width: 800,
            height: 600,
            title: "PixelFlow".into(),
            resizable: true,
        }
    }
}
