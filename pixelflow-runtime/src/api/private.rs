use actor_scheduler::{ActorHandle, ActorScheduler};
use pixelflow_graphics::render::Frame;
use pixelflow_graphics::Pixel;
use std::sync::Arc;

use crate::api::public::CursorIcon;
// use crate::input::MouseButton;

/// Window ID wrapper
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WindowId(pub u64);

impl WindowId {
    pub const PRIMARY: Self = Self(0);
}

use crate::display::messages::DisplayEvent;

/// Commands sent to the Display Driver.
#[derive(Debug)]
pub enum DriverCommand<P: Pixel> {
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
        frame: Frame<P>,
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

// Engine data message (high priority)
#[derive(Debug)]
pub enum EngineData<P: Pixel> {
    FromDriver(DisplayEvent),
    FromApp(crate::api::public::AppData<P>),
}

impl<P: Pixel> From<crate::api::public::AppData<P>> for EngineData<P> {
    fn from(data: crate::api::public::AppData<P>) -> Self {
        EngineData::FromApp(data)
    }
}

impl<P: Pixel> From<DisplayEvent>
    for actor_scheduler::Message<EngineData<P>, EngineControl<P>, crate::api::public::AppManagement>
{
    fn from(evt: DisplayEvent) -> Self {
        actor_scheduler::Message::Data(EngineData::FromDriver(evt))
    }
}

// Engine control message (low priority)
#[derive(Debug)]
pub enum EngineControl<P: Pixel> {
    PresentComplete(Frame<P>),
    VSync {
        timestamp: std::time::Instant,
        target_timestamp: std::time::Instant,
        refresh_interval: std::time::Duration,
    },
    UpdateRefreshRate(f64),
    VsyncActorReady(
        actor_scheduler::ActorHandle<
            crate::vsync_actor::RenderedResponse,
            crate::vsync_actor::VsyncCommand,
            crate::vsync_actor::VsyncManagement,
        >,
    ),
    Quit,
    DriverAck,
}

impl<P: Pixel> From<EngineControl<P>>
    for actor_scheduler::Message<EngineData<P>, EngineControl<P>, crate::api::public::AppManagement>
{
    fn from(ctrl: EngineControl<P>) -> Self {
        actor_scheduler::Message::Control(ctrl)
    }
}

// For now, let's assume channel.rs was supposed to define them but the refactor moved them here.
// But if I define them here, I need to make sure channel.rs imports them.
// The error was "file not found for module api".
// So I am restoring the file.

pub const DISPLAY_EVENT_BUFFER_SIZE: usize = 256;
pub const DISPLAY_EVENT_BURST_LIMIT: usize = 32;

// We wrap ActorHandle in Arc to allow sharing, as ActorHandle is no longer Clone.
pub type EngineActorHandle<P> =
    Arc<ActorHandle<EngineData<P>, EngineControl<P>, crate::api::public::AppManagement>>;
pub type EngineActorScheduler<P> =
    ActorScheduler<EngineData<P>, EngineControl<P>, crate::api::public::AppManagement>;

pub fn create_engine_actor<P: Pixel>(
    wake_handler: Option<Arc<dyn actor_scheduler::WakeHandler>>,
) -> (EngineActorHandle<P>, EngineActorScheduler<P>) {
    // Note: wake_handler is no longer used by ActorScheduler::new
    // We should probably remove it from the signature in a future refactor,
    // but for now we keep it for backward compatibility and ignore it.
    let _ = wake_handler;
    let (handle, scheduler) = actor_scheduler::ActorScheduler::new(DISPLAY_EVENT_BUFFER_SIZE, DISPLAY_EVENT_BURST_LIMIT);
    (Arc::new(handle), scheduler)
}
