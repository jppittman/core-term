use super::messages::{DisplayControl, DisplayData, DisplayMgmt};
use super::platform::Platform;
use crate::channel::{DriverCommand, EngineSender};
use actor_scheduler::ActorScheduler;
use anyhow::Result;
use pixelflow_graphics::Pixel;

/// Legacy DisplayDriver trait for backward compatibility with old X11 driver code.
///
/// DEPRECATED: New drivers should implement PlatformOps instead.
pub trait DisplayDriver: Clone + Send {
    /// The pixel type for this driver.
    type Pixel: Pixel;

    /// Create a new driver with the given engine sender.
    fn new(engine_tx: EngineSender<Self::Pixel>) -> Result<Self>;

    /// Send a command to the driver.
    fn send(&self, cmd: DriverCommand<Self::Pixel>) -> Result<()>;

    /// Run the driver event loop (blocking).
    fn run(&self) -> Result<()>;
}

/// The Generic Driver Actor.
/// It drives the `Platform` using the `ActorScheduler`.
pub struct DriverActor<P: Platform> {
    scheduler: ActorScheduler<DisplayData<P::Pixel>, DisplayControl, DisplayMgmt>,
    platform: P,
}

impl<P: Platform> DriverActor<P> {
    /// Create a new DriverActor.
    pub fn new(
        scheduler: ActorScheduler<DisplayData<P::Pixel>, DisplayControl, DisplayMgmt>,
        platform: P,
    ) -> Self {
        Self {
            scheduler,
            platform,
        }
    }

    /// Run the driver loop.
    pub fn run(self) -> Result<()> {
        // 1. Drain Scheduler (Priority Logic)
        // We delegate to scheduler.run which loops forever and calls actor.park(hint).
        self.scheduler.run(self.platform);

        // If run returns, it means we are shutting down (channels closed).
        Ok(())
    }
}
