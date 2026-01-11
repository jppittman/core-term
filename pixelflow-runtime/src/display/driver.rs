use super::messages::{DisplayControl, DisplayData, DisplayMgmt};
use super::platform::Platform;
use crate::api::private::{DriverCommand, EngineActorHandle as EngineSender};
use crate::error::RuntimeError;
use actor_scheduler::{Actor, ActorStatus, SystemStatus};
use pixelflow_graphics::Pixel;

/// Legacy DisplayDriver trait for backward compatibility with old X11 driver code.
///
/// DEPRECATED: New drivers should implement PlatformOps instead.
pub trait DisplayDriver: Clone + Send {
    /// The pixel type for this driver.
    type Pixel: Pixel;

    /// Create a new driver with the given engine sender.
    fn new(engine_tx: EngineSender) -> Result<Self, RuntimeError>;

    /// Send a command to the driver.
    fn send(&self, cmd: DriverCommand) -> Result<(), RuntimeError>;

    /// Run the driver event loop (blocking).
    fn run(&self) -> Result<(), RuntimeError>;
}

/// The Driver Actor - wraps a Platform implementation as an Actor.
///
/// The troupe! macro owns the scheduler. This actor just delegates to the Platform.
/// Marked [main] in the troupe - runs on the calling thread (GUI/main thread).
pub struct DriverActor<P: Platform> {
    platform: P,
}

impl<P: Platform> DriverActor<P> {
    /// Create a new DriverActor wrapping the given platform.
    pub fn new(platform: P) -> Self {
        Self { platform }
    }
}

impl<P: Platform> Actor<DisplayData, DisplayControl, DisplayMgmt> for DriverActor<P> {
    fn handle_data(&mut self, data: DisplayData) {
        self.platform.handle_data(data);
    }

    fn handle_control(&mut self, ctrl: DisplayControl) {
        self.platform.handle_control(ctrl);
    }

    fn handle_management(&mut self, mgmt: DisplayMgmt) {
        self.platform.handle_management(mgmt);
    }

    fn park(&mut self, status: SystemStatus) -> ActorStatus {
        // Delegate to platform's park - this is where OS event loop integration happens
        self.platform.park(status)
    }
}
