use crate::display::messages::{DisplayControl, DisplayData, DisplayMgmt};
use crate::display::ops::PlatformOps;
use actor_scheduler::{Actor, ActorStatus};
use pixelflow_graphics::Pixel;

/// The Platform Trait.
/// Implementers must be an Actor that handles display messages.
pub trait Platform:
    Actor<DisplayData<Self::Pixel>, DisplayControl, DisplayMgmt> + Send + 'static
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
    fn handle_data(&mut self, msg: DisplayData<Ops::Pixel>) -> Result<(), actor_scheduler::ActorError> {
        self.ops.handle_data(msg)
    }

    fn handle_control(&mut self, msg: DisplayControl) -> Result<(), actor_scheduler::ActorError> {
        self.ops.handle_control(msg)
    }

    fn handle_management(&mut self, msg: DisplayMgmt) -> Result<(), actor_scheduler::ActorError> {
        self.ops.handle_management(msg)
    }

    fn park(&mut self, hint: ActorStatus) -> ActorStatus {
        self.ops.park(hint)
    }
}

impl<Ops: PlatformOps> Platform for PlatformActor<Ops> {
    type Pixel = Ops::Pixel;
}
