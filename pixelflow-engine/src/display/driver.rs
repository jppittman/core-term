use super::messages::{DisplayControl, DisplayData, DisplayMgmt};
use super::platform::Platform;
use actor_scheduler::ActorScheduler;
use anyhow::Result;

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
    pub fn run(&mut self) -> Result<()> {
        loop {
            // 1. Drain Scheduler (Priority Logic)
            // We implement the drain loop manually here to support the ParkHint
            // and because we want to drive the platform's `park` method.

            // This is effectively `ActorScheduler::run` but unrolled to allow `park` with specific hint.
            // Actually, we can use `scheduler.run(&mut self.platform)` IF `ActorScheduler` supported `ParkHint`.
            // Which I updated it to do!

            // So I can just delegate to scheduler.run?
            // `scheduler.run` loops forever.
            // But `Platform::park` needs to be called.
            // Yes, `scheduler.run` calls `actor.park(hint)`.
            // So this `run` method is just a wrapper.

            self.scheduler.run(&mut self.platform);

            // If run returns, it means we are shutting down (channels closed).
            return Ok(());
        }
    }
}
