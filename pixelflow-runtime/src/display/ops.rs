use super::messages::{DisplayControl, DisplayData, DisplayMgmt};
use actor_scheduler::{ActorStatus, SystemStatus};

/// Backend-specific operations for the display platform.
pub trait PlatformOps: Send + 'static {
    fn handle_data(&mut self, data: DisplayData);
    fn handle_control(&mut self, ctrl: DisplayControl);
    fn handle_management(&mut self, mgmt: DisplayMgmt);
    fn park(&mut self, status: SystemStatus) -> ActorStatus;
}
