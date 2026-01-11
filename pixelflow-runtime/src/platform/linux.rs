//! Linux platform implementation.
//!
//! Bridge to X11DisplayDriver using the new PlatformOps trait.

use crate::api::private::EngineActorHandle;
use crate::api::private::{DriverCommand, EngineActorHandle as EngineSender};
use crate::display::driver::DisplayDriver as OldDisplayDriver;
use crate::display::drivers::X11DisplayDriver;
use crate::display::messages::{DisplayControl, DisplayData, DisplayMgmt, WindowId};
use crate::display::ops::PlatformOps;
use crate::error::RuntimeError;
use actor_scheduler::ActorStatus;
use pixelflow_graphics::render::color::Bgra8;

/// Linux platform pixel type (BGRA for X11).
pub type LinuxPixel = Bgra8;

/// Linux platform operations - bridges to X11DisplayDriver.
pub struct LinuxOps {
    engine_handle: EngineActorHandle,
    driver: Option<X11DisplayDriver>,
    engine_sender: Option<EngineSender>,
}

impl LinuxOps {
    /// Create new Linux platform ops.
    pub fn new(engine_handle: EngineActorHandle) -> Result<Self, RuntimeError> {
        Ok(Self {
            engine_handle,
            driver: None,
            engine_sender: None,
        })
    }
}

impl PlatformOps for LinuxOps {
    type Pixel = LinuxPixel;

    fn handle_data(&mut self, data: DisplayData<Self::Pixel>) -> Result<(), actor_scheduler::ActorError> {
        match data {
            DisplayData::Present { frame, .. } => {
                if let Some(driver) = &self.driver {
                    driver
                        .send(DriverCommand::Present {
                            id: WindowId::PRIMARY,
                            frame,
                        })
                        .expect("Failed to send Present to X11 driver");
                }
            }
        }
        Ok(())
    }

    fn handle_control(&mut self, ctrl: DisplayControl) -> Result<(), actor_scheduler::ActorError> {
        if let Some(driver) = &self.driver {
            match ctrl {
                DisplayControl::Shutdown => {
                    driver
                        .send(DriverCommand::Shutdown)
                        .expect("Failed to send Shutdown to X11 driver");
                }
                DisplayControl::SetTitle { title, .. } => {
                    driver
                        .send(DriverCommand::SetTitle {
                            id: WindowId::PRIMARY,
                            title,
                        })
                        .expect("Failed to send SetTitle to X11 driver");
                }
                DisplayControl::SetSize { width, height, .. } => {
                    driver
                        .send(DriverCommand::SetSize {
                            id: WindowId::PRIMARY,
                            width,
                            height,
                        })
                        .expect("Failed to send SetSize to X11 driver");
                }
                DisplayControl::Copy { text } => {
                    driver
                        .send(DriverCommand::CopyToClipboard(text))
                        .expect("Failed to send CopyToClipboard to X11 driver");
                }
                DisplayControl::RequestPaste => {
                    driver
                        .send(DriverCommand::RequestPaste)
                        .expect("Failed to send RequestPaste to X11 driver");
                }
                DisplayControl::SetCursor { cursor, .. } => {
                    driver
                        .send(DriverCommand::SetCursorIcon { icon: cursor })
                        .expect("Failed to send SetCursorIcon to X11 driver");
                }
                DisplayControl::Bell => {
                    driver
                        .send(DriverCommand::Bell)
                        .expect("Failed to send Bell to X11 driver");
                }
                DisplayControl::SetVisible { .. } | DisplayControl::RequestRedraw { .. } => {
                    // Not implemented for Linux yet
                }
            }
        }
        Ok(())
    }

    fn handle_management(&mut self, mgmt: DisplayMgmt) -> Result<(), actor_scheduler::ActorError> {
        match mgmt {
            DisplayMgmt::Create { id, settings } => {
                let engine_handle = self.engine_handle.clone();
                self.engine_sender = Some(engine_handle.clone());

                // Create X11 driver directly with the engine handle (ActorHandle)
                let driver =
                    X11DisplayDriver::new(engine_handle).expect("Failed to create X11 driver");

                // Send CreateWindow command
                driver
                    .send(DriverCommand::CreateWindow {
                        id,
                        width: settings.width,
                        height: settings.height,
                        title: settings.title.clone(),
                    })
                    .expect("Failed to send CreateWindow");

                // Spawn X11 event loop on separate thread
                // IMPORTANT: The original driver must be moved to the thread, as clones cannot run()
                let driver_for_ops = driver.clone();
                std::thread::Builder::new()
                    .name("x11-event-loop".to_string())
                    .spawn(move || {
                        driver.run().expect("X11 driver failed");
                    })
                    .expect("Failed to spawn X11 event loop");

                self.driver = Some(driver_for_ops);
            }
            DisplayMgmt::Destroy { .. } => {
                if let Some(driver) = &self.driver {
                    driver
                        .send(DriverCommand::Shutdown)
                        .expect("Failed to send Shutdown to X11 driver on Destroy");
                }
                self.driver = None;
            }
        }
        Ok(())
    }

    fn park(&mut self, hint: ActorStatus) -> ActorStatus {
        // Simple sleep-based parking
        match hint {
            ActorStatus::Busy => {
                // Poll mode - return immediately
            }
            ActorStatus::Idle => {
                // Wait mode - sleep a bit to avoid busy-looping
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
        hint
    }
}
