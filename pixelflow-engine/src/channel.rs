// src/channel.rs
//!
//! DEPRECATED: Legacy channel module for backward compatibility.
//!
//! All message types have been moved to crate::api::private.
//! Use those types directly in new code.

// Re-export all types from api::private for backward compatibility
pub use crate::api::private::{
    create_engine_actor, DriverCommand, EngineActorHandle,
    EngineActorScheduler, EngineControl, EngineData, DISPLAY_EVENT_BUFFER_SIZE,
    DISPLAY_EVENT_BURST_LIMIT,
};

// Re-export AppManagement from public API
pub use crate::api::public::AppManagement;

use crate::display::DisplayEvent;
use pixelflow_core::Pixel;
use actor_scheduler::WakeHandler;
use std::sync::Arc;

// ============================================================================
// Legacy Wrapper Types (for backward compatibility with old driver code)
// ============================================================================

/// Channel bundle for engine-side communication.
///
/// DEPRECATED: Use EngineActorHandle and EngineActorScheduler directly.
pub struct EngineChannels<P: Pixel> {
    /// ActorHandle for sending messages to engine
    pub handle: EngineActorHandle<P>,
    /// ActorScheduler for receiving messages in engine
    pub scheduler: EngineActorScheduler<P>,
}

/// Create channels for driver -> engine communication.
///
/// DEPRECATED: Use create_engine_actor directly.
pub fn create_engine_channels<P: Pixel>(
    wake_handler: Option<Arc<dyn WakeHandler>>,
) -> EngineChannels<P> {
    let (handle, scheduler) = create_engine_actor(wake_handler);

    EngineChannels { handle, scheduler }
}

// Type alias for the engine handle (makes migration easier)
pub type EngineSender<P> = EngineActorHandle<P>;

// Backwards-compatible EngineCommand enum
// DEPRECATED: Use Message::Data/Control directly with .into()
#[derive(Debug)]
pub enum EngineCommand<P: Pixel> {
    /// Display event from driver (converts to Message::Data)
    DisplayEvent(DisplayEvent),
    /// VSync actor ready (converts to Message::Control)
    VsyncActorReady(actor_scheduler::ActorHandle<
        crate::vsync_actor::RenderedResponse,
        crate::vsync_actor::VsyncCommand,
        crate::vsync_actor::VsyncManagement,
    >),
    /// Present complete (converts to Message::Control)
    PresentComplete(pixelflow_render::Frame<P>),
    /// Driver ack (converts to Message::Control)
    DriverAck,
}

impl<P: Pixel> From<EngineCommand<P>> for actor_scheduler::Message<EngineData<P>, EngineControl<P>, AppManagement> {
    fn from(cmd: EngineCommand<P>) -> Self {
        match cmd {
            EngineCommand::DisplayEvent(evt) => evt.into(),
            EngineCommand::VsyncActorReady(actor) => EngineControl::VsyncActorReady(actor).into(),
            EngineCommand::PresentComplete(frame) => EngineControl::PresentComplete(frame).into(),
            EngineCommand::DriverAck => EngineControl::DriverAck.into(),
        }
    }
}
