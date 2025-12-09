//! Public API traits for applications.
//!
//! Applications must implement the `Application` trait which receives engine events.
//! The simplest implementation is to use an ActorHandle which gets a blanket impl.
//!
//! # Example
//!
//! ```ignore
//! use actor_scheduler::{ActorScheduler, Actor, Message};
//! use pixelflow_engine::{EngineEvent, EngineEventData, EngineEventControl, EngineEventManagement, Application};
//!
//! // Create app actor with lane-specific types
//! let (app_handle, app_scheduler) = ActorScheduler::<
//!     EngineEventData,
//!     EngineEventControl,
//!     EngineEventManagement
//! >::new(10, 128);
//!
//! // app_handle automatically implements Application via blanket impl
//! pixelflow_engine::run(app_handle, config)?;
//! ```

use super::{EngineEvent, EngineEventControl, EngineEventData, EngineEventManagement};
use actor_scheduler::{ActorHandle, Message, SendError};

/// Trait for applications that receive engine events.
///
/// Apps must implement this trait to be used with the engine.
/// The simplest implementation is to use ActorHandle which gets a blanket impl for free.
pub trait Application {
    /// Send an event to the application.
    ///
    /// Events are pre-categorized by priority lane (Data/Control/Management)
    /// so implementations can route appropriately.
    fn send(&self, event: EngineEvent) -> Result<(), SendError>;
}

/// Blanket impl for ActorHandle - apps using actor-scheduler get it for free.
///
/// Automatically unwraps EngineEvent and routes to the correct priority lane.
impl Application
    for ActorHandle<EngineEventData, EngineEventControl, EngineEventManagement>
{
    fn send(&self, event: EngineEvent) -> Result<(), SendError> {
        match event {
            EngineEvent::Data(data) => self.send(Message::Data(data)),
            EngineEvent::Control(ctrl) => self.send(Message::Control(ctrl)),
            EngineEvent::Management(mgmt) => self.send(Message::Management(mgmt)),
        }
    }
}
