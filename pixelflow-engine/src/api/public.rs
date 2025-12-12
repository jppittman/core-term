use crate::input::{Key, Modifiers, MouseButton};
use pixelflow_core::{Pixel, Surface};
use std::fmt::Debug;

// ============================================================================
// Application Trait
// ============================================================================

/// Application trait that receives engine events.
pub trait Application: Send + Sync {
    /// Associated pixel type (generic)
    type Pixel: Pixel;

    /// Handle engine event (input, control, etc)
    fn handle_event(
        &mut self,
        event: EngineEvent,
        response_channel: &mut actor_scheduler::ActorHandle<
            crate::api::private::EngineData<Self::Pixel>,
            crate::api::private::EngineControl<Self::Pixel>,
            AppManagement,
        >,
    );
}

// ============================================================================
// Engine -> App Events
// ============================================================================

#[derive(Debug, Clone)]
pub enum EngineEvent {
    Data(EngineEventData),
    Control(EngineEventControl),
    Management(EngineEventManagement),
}

#[derive(Debug, Clone)]
pub enum EngineEventData {
    RequestFrame {
        timestamp: std::time::Instant,
        target_timestamp: std::time::Instant,
        refresh_interval: std::time::Duration,
    },
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
        key: Key,
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
        dx: f64,
        dy: f64,
        mods: Modifiers,
    },
    FocusGained,
    FocusLost,
    Paste(String),
}

// ============================================================================
// App -> Engine Response Data
// ============================================================================

// RenderSurface holds a surface in continuous coordinates (f32).
// Engine handles rasterization to u32.
pub enum AppData<P: Pixel> {
    RenderSurface(Box<dyn Surface<P, f32> + Send + Sync>),
}

// ============================================================================
// App -> Engine Response Management
// ============================================================================

/// Management messages sent from App to Engine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AppManagement {
    /// Request app to exit.
    Quit,
    /// Notify app of resize.
    Resize { width: u32, height: u32 },
    /// Set window title.
    SetTitle(String),
    /// Copy text to clipboard.
    CopyToClipboard(String),
    /// Request paste from clipboard.
    RequestPaste,
    /// Request resize.
    ResizeRequest(u32, u32),
    /// Set cursor icon (placeholder).
    SetCursorIcon(u8),
}

// Implement From for ActorScheduler compatibility
impl From<AppManagement>
    for actor_scheduler::Message<
        crate::api::private::EngineData<pixelflow_render::color::Rgba>,
        crate::api::private::EngineControl<pixelflow_render::color::Rgba>,
        AppManagement,
    >
{
    fn from(msg: AppManagement) -> Self {
        actor_scheduler::Message::Management(msg)
    }
}
