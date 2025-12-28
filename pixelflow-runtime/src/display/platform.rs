use crate::display::messages::{DisplayControl, DisplayData, DisplayMgmt};
use crate::display::ops::PlatformOps;
use crate::error::RuntimeError;
use actor_scheduler::{Actor, ParkHint};
use pixelflow_graphics::Pixel;

/// The Platform Trait.
/// Implementers must be an Actor that handles display messages.
pub trait Platform:
    Actor<DisplayData<Self::Pixel>, DisplayControl, DisplayMgmt, Error = RuntimeError> + Send + 'static
{
    /// The pixel format required by this platform.
    type Pixel: Pixel;
}

/// A generic wrapper that turns any `PlatformOps` implementation into a `Platform` Actor.
pub struct PlatformActor<Ops: PlatformOps> {
    ops: Ops,
}

impl<Ops: PlatformOps> PlatformActor<Ops> {
    pub fn new(ops: Ops) -> Self {
        Self { ops }
    }
}

impl<Ops: PlatformOps> Actor<DisplayData<Ops::Pixel>, DisplayControl, DisplayMgmt>
    for PlatformActor<Ops>
{
    type Error = RuntimeError;

    fn handle_data(&mut self, msg: DisplayData<Ops::Pixel>) -> Result<(), RuntimeError> {
        self.ops.handle_data(msg);
        Ok(())
    }

    fn handle_control(&mut self, msg: DisplayControl) -> Result<(), RuntimeError> {
        self.ops.handle_control(msg);
        Ok(())
    }

    fn handle_management(&mut self, msg: DisplayMgmt) -> Result<(), RuntimeError> {
        self.ops.handle_management(msg);
        Ok(())
    }

    fn park(&mut self, hint: ParkHint) -> Result<ParkHint, RuntimeError> {
        Ok(self.ops.park(hint))
    }
}

impl<Ops: PlatformOps> Platform for PlatformActor<Ops> {
    type Pixel = Ops::Pixel;
}
