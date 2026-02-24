use actor_scheduler::{ActorHandle, ActorScheduler};
use pixelflow_graphics::render::frame::Frame;
use pixelflow_graphics::render::rasterizer::RenderResponse;
use std::sync::Arc;

use crate::api::public::CursorIcon;
use crate::pixel::PlatformPixel;
// use crate::input::MouseButton;

// Re-export WindowId from public API for backward compatibility
pub use crate::api::public::WindowId;

use crate::display::messages::{DisplayEvent, Window};

/// Commands sent to the Display Driver.
#[derive(Debug)]
pub enum DriverCommand {
    CreateWindow {
        id: WindowId,
        width: u32,
        height: u32,
        title: String,
    },
    DestroyWindow {
        id: WindowId,
    },
    Shutdown,
    Present {
        id: WindowId,
        frame: Frame<PlatformPixel>,
    },
    SetTitle {
        id: WindowId,
        title: String,
    },
    SetSize {
        id: WindowId,
        width: u32,
        height: u32,
    },
    CopyToClipboard(String),
    RequestPaste,
    Bell,
    SetCursorIcon {
        icon: CursorIcon,
    },
}

// Engine data message (high throughput, frame timing)
pub enum EngineData {
    FromDriver(DisplayEvent),
    FromApp(crate::api::public::AppData),
    VSync {
        timestamp: std::time::Instant,
        target_timestamp: std::time::Instant,
        refresh_interval: std::time::Duration,
    },
    PresentComplete(Window),
    /// Render complete - carries the render response (cooked frame + timing).
    /// Window is reconstructed from pending_render metadata in the engine.
    RenderComplete(RenderResponse<PlatformPixel>),
}

impl std::fmt::Debug for EngineData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FromDriver(event) => f.debug_tuple("FromDriver").field(event).finish(),
            Self::FromApp(data) => f.debug_tuple("FromApp").field(data).finish(),
            Self::VSync {
                timestamp,
                target_timestamp,
                refresh_interval,
            } => f
                .debug_struct("VSync")
                .field("timestamp", timestamp)
                .field("target_timestamp", target_timestamp)
                .field("refresh_interval", refresh_interval)
                .finish(),
            Self::PresentComplete(window) => {
                f.debug_tuple("PresentComplete").field(window).finish()
            }
            Self::RenderComplete(response) => f
                .debug_struct("RenderComplete")
                .field("render_time", &response.render_time)
                .field("frame_size", &(response.frame.width, response.frame.height))
                .finish(),
        }
    }
}

impl From<crate::api::public::AppData> for EngineData {
    fn from(data: crate::api::public::AppData) -> Self {
        EngineData::FromApp(data)
    }
}

impl From<DisplayEvent>
    for actor_scheduler::Message<EngineData, EngineControl, crate::api::public::AppManagement>
{
    fn from(evt: DisplayEvent) -> Self {
        actor_scheduler::Message::Data(EngineData::FromDriver(evt))
    }
}

// Engine control message (low frequency, configuration/lifecycle)
#[derive(Debug, Default)]
pub enum EngineControl {
    UpdateRefreshRate(f64),
    VsyncActorReady(
        actor_scheduler::ActorHandle<
            crate::vsync_actor::RenderedResponse,
            crate::vsync_actor::VsyncCommand,
            crate::vsync_actor::VsyncManagement,
        >,
    ),
    #[default]
    Quit,
    DriverAck,
}

// For now, let's assume channel.rs was supposed to define them but the refactor moved them here.
// But if I define them here, I need to make sure channel.rs imports them.
// The error was "file not found for module api".
// So I am restoring the file.

pub const DISPLAY_EVENT_BUFFER_SIZE: usize = 256;
pub const DISPLAY_EVENT_BURST_LIMIT: usize = 32;

pub type EngineActorHandle =
    ActorHandle<EngineData, EngineControl, crate::api::public::AppManagement>;
pub type EngineActorScheduler =
    ActorScheduler<EngineData, EngineControl, crate::api::public::AppManagement>;

#[must_use]
pub fn create_engine_actor(
    wake_handler: Option<Arc<dyn actor_scheduler::WakeHandler>>,
) -> (EngineActorHandle, EngineActorScheduler) {
    actor_scheduler::ActorScheduler::new_with_wake_handler(
        DISPLAY_EVENT_BURST_LIMIT,
        DISPLAY_EVENT_BUFFER_SIZE,
        wake_handler,
    )
}
