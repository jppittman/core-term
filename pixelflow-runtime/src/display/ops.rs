use super::messages::{DisplayControl, DisplayData, DisplayMgmt};
use actor_scheduler::{ActorStatus, HandlerError, HandlerResult, SystemStatus};

/// Backend-specific operations for the display platform.
pub trait PlatformOps: Send + 'static {
    fn handle_data(&mut self, data: DisplayData) -> HandlerResult;
    fn handle_control(&mut self, ctrl: DisplayControl) -> HandlerResult;
    fn handle_management(&mut self, mgmt: DisplayMgmt) -> HandlerResult;
    fn park(&mut self, status: SystemStatus) -> Result<ActorStatus, HandlerError>;
}
