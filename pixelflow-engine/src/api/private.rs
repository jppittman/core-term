use actor_scheduler::{ActorHandle, ActorScheduler};
use pixelflow_core::Pixel;
use pixelflow_render::Frame;
use std::sync::Arc;

use crate::input::{KeySymbol, Modifiers};

/// Window ID wrapper
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WindowId(pub u64);

impl WindowId {
    pub const PRIMARY: Self = Self(0);
}

/// Events from the Display Driver.
#[derive(Debug, Clone)]
pub enum DisplayEvent {
    WindowCreated {
        id: WindowId,
        width_px: u32,
        height_px: u32,
        scale: f64,
    },
    Resized {
        id: WindowId,
        width_px: u32,
        height_px: u32,
    },
    WindowDestroyed {
        id: WindowId,
    },
    CloseRequested {
        id: WindowId,
    },
    ScaleChanged {
        id: WindowId,
        scale: f64,
    },
    Key {
        id: WindowId,
        symbol: KeySymbol,
        modifiers: Modifiers,
        text: Option<String>,
    },
    MouseButtonPress {
        id: WindowId,
        button: u8,
        x: i32,
        y: i32,
        modifiers: Modifiers,
    },
    MouseButtonRelease {
        id: WindowId,
        button: u8,
        x: i32,
        y: i32,
        modifiers: Modifiers,
    },
    MouseMove {
        id: WindowId,
        x: i32,
        y: i32,
        modifiers: Modifiers,
    },
    MouseScroll {
        id: WindowId,
        dx: f32,
        dy: f32,
        x: i32,
        y: i32,
        modifiers: Modifiers,
    },
    FocusGained {
        id: WindowId,
    },
    FocusLost {
        id: WindowId,
    },
    PasteData {
        text: String,
    },
    ClipboardDataRequested,
}

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
}

// Engine data message (high priority)
#[derive(Debug)]
pub enum EngineData<P: Pixel> {
    FromDriver(DisplayEvent),
    FromApp(crate::api::public::AppData<P>),
}

impl<P: Pixel> From<DisplayEvent> for actor_scheduler::Message<EngineData<P>, EngineControl<P>, crate::api::public::AppManagement> {
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
    VsyncActorReady(actor_scheduler::ActorHandle<
        crate::vsync_actor::RenderedResponse,
        crate::vsync_actor::VsyncCommand,
        crate::vsync_actor::VsyncManagement,
    >),
    Quit,
    DriverAck,
}

impl<P: Pixel> From<EngineControl<P>> for actor_scheduler::Message<EngineData<P>, EngineControl<P>, crate::api::public::AppManagement> {
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

pub type EngineActorHandle<P> = ActorHandle<EngineData<P>, EngineControl<P>, crate::api::public::AppManagement>;
pub type EngineActorScheduler<P> = ActorScheduler<EngineData<P>, EngineControl<P>, crate::api::public::AppManagement>;

pub fn create_engine_actor<P: Pixel>(
    wake_handler: Option<Arc<dyn actor_scheduler::WakeHandler>>,
) -> (EngineActorHandle<P>, EngineActorScheduler<P>) {
    // Note: wake_handler is no longer used by ActorScheduler::new
    // We should probably remove it from the signature in a future refactor,
    // but for now we keep it for backward compatibility and ignore it.
    let _ = wake_handler;
    actor_scheduler::ActorScheduler::new(
        DISPLAY_EVENT_BUFFER_SIZE,
        DISPLAY_EVENT_BURST_LIMIT,
    )
}
