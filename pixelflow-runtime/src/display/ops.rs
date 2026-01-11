use super::messages::{DisplayControl, DisplayData, DisplayMgmt};
use actor_scheduler::ActorStatus;
use pixelflow_graphics::Pixel;

/// Backend-specific operations for the display platform.
pub trait PlatformOps: Send + 'static {
    type Pixel: Pixel;

    fn handle_data(&mut self, data: DisplayData<Self::Pixel>) -> Result<(), actor_scheduler::ActorError>;
    fn handle_control(&mut self, ctrl: DisplayControl) -> Result<(), actor_scheduler::ActorError>;
    fn handle_management(&mut self, mgmt: DisplayMgmt) -> Result<(), actor_scheduler::ActorError>;
    fn park(&mut self, hint: ActorStatus) -> ActorStatus;
}
