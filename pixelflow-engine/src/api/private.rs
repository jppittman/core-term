use crate::api::public::AppData;
use crate::display::messages::{DisplayEvent, WindowId};
use crate::vsync_actor::{RenderedResponse, VsyncCommand, VsyncManagement};
use actor_scheduler::{ActorHandle, ActorScheduler, Message, WakeHandler};
use pixelflow_core::Pixel;
use pixelflow_render::Frame;
use std::fmt::Debug;
use std::sync::Arc;

// ============================================================================
// Constants
// ============================================================================

// Priority channel tuning for UI responsiveness
pub const DISPLAY_EVENT_BUFFER_SIZE: usize = 128; // Large buffer for bursts (mouse move)
pub const DISPLAY_EVENT_BURST_LIMIT: usize = 16; // Process 16 events before yielding

// ============================================================================
// Message Types
// ============================================================================

/// Data messages (High Priority)
///
/// Use for:
/// - User input (latency sensitive)
/// - Window resizing (relayout urgency)
/// - OS signals
#[derive(Debug)]
pub enum EngineData<P: Pixel> {
    /// Input event from window system (Driver -> Engine)
    FromDriver(DisplayEvent),
    /// Data from application (App -> Engine)
    // Note: AppData contains a Box<dyn Surface>, so it's not Debug
    FromApp(AppData<P>),
    /// Recycled frame (Driver -> Engine)
    RecycleFrame(Frame<P>),
}

// Ignore Debug for AppData since Surface isn't Debug
impl<P: Pixel> Debug for AppData<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AppData::RenderSurface(_) => write!(f, "RenderSurface(...)"),
        }
    }
}

/// Control messages (Low Priority)
///
/// Use for:
/// - Lifecycle management
/// - Render completed notifications
/// - Driver acknowledgements
#[derive(Debug)]
pub enum EngineControl<P: Pixel> {
    /// Render completed, ready to present
    PresentComplete(Frame<P>),
    /// VSync actor is ready
    VsyncActorReady(ActorHandle<RenderedResponse, VsyncCommand, VsyncManagement>),
    /// VSync pulse
    VSync {
        timestamp: std::time::Instant,
        target_timestamp: std::time::Instant,
        refresh_interval: std::time::Duration,
    },
    /// Update refresh rate
    UpdateRefreshRate(f64),
    /// Driver acknowledgement
    DriverAck,
    /// Request exit
    Quit,
}

/// Driver Command (Legacy Wrapper)
///
/// This enum wraps the underlying Message type for compatibility with
/// existing driver code. New code should use ActorHandle directly.
#[derive(Debug)]
pub enum DriverCommand<P: Pixel> {
    /// Configure window (Deprecated, handled via config)
    Configure,
    /// Create window (Deprecated, handled via config)
    CreateWindow {
        id: WindowId,
        width: u32,
        height: u32,
        title: String,
    },
    /// Destroy window
    DestroyWindow { id: WindowId },
    /// Present frame
    Present { id: WindowId, frame: Frame<P> },
    /// Set window title
    SetTitle { id: WindowId, title: String },
    /// Set window size
    SetSize {
        id: WindowId,
        width: u32,
        height: u32,
    },
    /// Copy to clipboard
    CopyToClipboard(String),
    /// Request paste
    RequestPaste,
    /// Ring bell
    Bell,
    /// Shutdown driver
    Shutdown,
}

// ============================================================================
// Actor Types
// ============================================================================

/// Type alias for the engine actor handle
pub type EngineActorHandle<P> =
    ActorHandle<EngineData<P>, EngineControl<P>, crate::api::public::AppManagement>;

/// Type alias for the engine scheduler
pub type EngineActorScheduler<P> =
    ActorScheduler<EngineData<P>, EngineControl<P>, crate::api::public::AppManagement>;

/// Create a new engine actor with priority configuration.
pub fn create_engine_actor<P: Pixel>(
    wake_handler: Option<Arc<dyn WakeHandler>>,
) -> (EngineActorHandle<P>, EngineActorScheduler<P>) {
    // Configure priority channel for UI events
    let config = actor_scheduler::PriorityConfig {
        high_priority_buffer: DISPLAY_EVENT_BUFFER_SIZE,
        burst_limit: DISPLAY_EVENT_BURST_LIMIT,
        ..Default::default()
    };

    ActorScheduler::new_with_config(config, wake_handler)
}

// ============================================================================
// Trait Implementations
// ============================================================================

// Convert DisplayEvent to EngineData (High Priority)
impl<P: Pixel> From<DisplayEvent>
    for Message<EngineData<P>, EngineControl<P>, crate::api::public::AppManagement>
{
    fn from(event: DisplayEvent) -> Self {
        Message::Data(EngineData::FromDriver(event))
    }
}

// Helper for Frame conversion to avoid orphan rules
pub fn frame_to_message<P: Pixel>(
    frame: Frame<P>,
) -> Message<EngineData<P>, EngineControl<P>, crate::api::public::AppManagement> {
    Message::Data(EngineData::RecycleFrame(frame))
}

// Convert EngineControl to Message (Low Priority)
impl<P: Pixel> From<EngineControl<P>>
    for Message<EngineData<P>, EngineControl<P>, crate::api::public::AppManagement>
{
    fn from(ctrl: EngineControl<P>) -> Self {
        Message::Control(ctrl)
    }
}
