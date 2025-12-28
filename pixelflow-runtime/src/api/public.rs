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
///
/// Note: The pixel type `P` is kept for compatibility but the actual rendering
/// is done using Manifolds that produce Discrete values.
pub enum AppData<P: pixelflow_graphics::Pixel> {
    /// A continuous surface (manifold) for rendering.
    /// The manifold should produce Discrete values.
    RenderSurface(
        std::sync::Arc<
            dyn pixelflow_core::Manifold<Output = pixelflow_graphics::Discrete> + Send + Sync,
        >,
    ),
    /// A discrete surface rendered at u32 coordinates.
    /// Uses the same manifold interface but intended for pixel-aligned rendering.
    RenderSurfaceU32(
        std::sync::Arc<
            dyn pixelflow_core::Manifold<Output = pixelflow_graphics::Discrete> + Send + Sync,
        >,
    ),
    /// Frame was skipped (no rendering needed).
    Skipped,
    #[doc(hidden)]
    _Phantom(std::marker::PhantomData<P>),
}

impl<P: pixelflow_graphics::Pixel> std::fmt::Debug for AppData<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RenderSurface(_) => f.debug_tuple("RenderSurface").finish(),
            Self::RenderSurfaceU32(_) => f.debug_tuple("RenderSurfaceU32").finish(),
            Self::Skipped => f.debug_tuple("Skipped").finish(),
            Self::_Phantom(_) => f.debug_tuple("_Phantom").finish(),
        }
    }
}

/// Application trait that defines the logic.
pub trait Application {
    fn send(&self, event: EngineEvent) -> Result<(), crate::error::RuntimeError>;
}

impl Application
    for actor_scheduler::ActorHandle<EngineEventData, EngineEventControl, EngineEventManagement>
{
    fn send(&self, event: EngineEvent) -> Result<(), crate::error::RuntimeError> {
        let msg = match event {
            EngineEvent::Control(ctrl) => actor_scheduler::Message::Control(ctrl),
            EngineEvent::Management(mgmt) => actor_scheduler::Message::Management(mgmt),
            EngineEvent::Data(data) => actor_scheduler::Message::Data(data),
        };
        self.send(msg)
            .map_err(|e| crate::error::RuntimeError::EventSendError(e.to_string()))
    }
}

/// Application management commands (change title, etc.)
#[derive(Debug, Clone)]
pub enum AppManagement {
    /// Configure the engine with initial settings (sent on startup).
    Configure(crate::config::EngineConfig),
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
