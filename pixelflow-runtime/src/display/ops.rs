use super::messages::{DisplayControl, DisplayData, DisplayMgmt};
use actor_scheduler::ParkHint;
use pixelflow_graphics::Pixel;

/// Backend-specific operations for the display platform.
pub trait PlatformOps: Send + 'static {
    type Pixel: Pixel;

    fn handle_data(&mut self, data: DisplayData<Self::Pixel>);
    fn handle_control(&mut self, ctrl: DisplayControl);
    fn handle_management(&mut self, mgmt: DisplayMgmt);
    fn park(&mut self, hint: ParkHint);
}
