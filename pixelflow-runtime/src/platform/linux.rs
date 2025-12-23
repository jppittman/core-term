//! Linux platform implementation.
//!
//! This module provides a stub implementation for Linux.
//! The actual X11 driver uses the older DisplayDriver architecture.

use crate::api::private::EngineActorHandle;
use crate::display::messages::{DisplayControl, DisplayData, DisplayMgmt};
use crate::display::ops::PlatformOps;
use actor_scheduler::ParkHint;
use anyhow::Result;
use pixelflow_graphics::render::color::Bgra8;

/// Linux platform pixel type (BGRA for X11).
pub type LinuxPixel = Bgra8;

/// Linux platform operations (X11-based).
pub struct LinuxOps {
    #[allow(dead_code)]
    engine_handle: EngineActorHandle<LinuxPixel>,
}

impl LinuxOps {
    /// Create new Linux platform ops.
    pub fn new(engine_handle: EngineActorHandle<LinuxPixel>) -> Result<Self> {
        Ok(Self { engine_handle })
    }
}

impl PlatformOps for LinuxOps {
    type Pixel = LinuxPixel;

    fn handle_data(&mut self, _data: DisplayData<Self::Pixel>) {
        // TODO: Implement X11 data handling
    }

    fn handle_control(&mut self, _ctrl: DisplayControl) {
        // TODO: Implement X11 control handling
    }

    fn handle_management(&mut self, _mgmt: DisplayMgmt) {
        // TODO: Implement X11 management handling
    }

    fn park(&mut self, hint: ParkHint) {
        // Simple sleep-based parking for now
        match hint {
            ParkHint::Poll => {
                // Poll mode - return immediately
            }
            ParkHint::Wait => {
                // Wait mode - sleep a bit to avoid busy-looping
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
    }
}
