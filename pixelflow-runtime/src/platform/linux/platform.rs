//! Linux platform implementation.
//!
//! Bridge to X11DisplayDriver using the new PlatformOps trait.

use crate::api::private::EngineActorHandle;
use crate::api::private::EngineData;
use crate::display::messages::{DisplayControl, DisplayData, DisplayEvent, DisplayMgmt, WindowId};
use crate::display::ops::PlatformOps;
use crate::error::RuntimeError;
use crate::platform::linux::window::X11Window;
use crate::platform::waker::X11Waker;
use actor_scheduler::{ActorStatus, Message, SystemStatus};
use log::{error, info};
use pixelflow_graphics::render::color::Bgra8;
use pixelflow_graphics::render::Frame;
use std::mem;
use x11::xlib;

use super::events;

/// Linux platform pixel type (BGRA for X11).
pub type LinuxPixel = Bgra8;

/// Linux platform operations - direct X11 implementation.
pub struct LinuxOps {
    engine_handle: EngineActorHandle,
    waker: X11Waker,
    window: Option<X11Window>,
}

impl LinuxOps {
    /// Create new Linux platform ops.
    pub fn new(engine_handle: EngineActorHandle) -> Result<Self, RuntimeError> {
        Ok(Self {
            engine_handle,
            waker: X11Waker::new(),
            window: None,
        })
    }
}

impl PlatformOps for LinuxOps {
    fn handle_data(&mut self, data: DisplayData) {
        if let Some(window) = &mut self.window {
            match data {
                DisplayData::Present { frame, .. } => {
                    let (returned_frame, result) = window.present(frame);
                    if let Err(e) = result {
                        error!("X11: Present failed: {:?}", e);
                    }
                    // Return buffer to engine
                    let _ = self
                        .engine_handle
                        .send(Message::Data(EngineData::PresentComplete(returned_frame)));
                }
            }
        }
    }

    fn handle_control(&mut self, ctrl: DisplayControl) {
        if let Some(window) = &mut self.window {
            match ctrl {
                DisplayControl::Shutdown => {
                    // Handled by scheduler drop/exit, but we can close early if needed
                }
                DisplayControl::SetTitle { title, .. } => {
                    window.set_title(&title);
                }
                DisplayControl::SetSize { width, height, .. } => {
                    window.set_size(width, height);
                }
                DisplayControl::Copy { text } => {
                    window.copy_to_clipboard(&text);
                }
                DisplayControl::RequestPaste => {
                    window.request_paste();
                }
                DisplayControl::SetCursor { cursor, .. } => {
                    window.set_cursor(cursor);
                }
                DisplayControl::Bell => {
                    window.bell();
                }
                DisplayControl::SetVisible { .. } | DisplayControl::RequestRedraw { .. } => {
                    // Not implemented for Linux yet
                }
            }
        }
    }

    fn handle_management(&mut self, mgmt: DisplayMgmt) {
        match mgmt {
            DisplayMgmt::Create { settings } => {
                info!(
                    "X11: Creating window '{}' {}x{}",
                    settings.title, settings.width, settings.height
                );
                match X11Window::new(&settings, &self.waker) {
                    Ok(window) => {
                        let id = WindowId(window.window as u64);
                        // Allocate initial frame buffer
                        let frame = Frame::<LinuxPixel>::new(window.width, window.height);
                        
                        // Send WindowCreated event
                        let _ = self.engine_handle.send(Message::Data(EngineData::FromDriver(
                            DisplayEvent::WindowCreated {
                                id,
                                width_px: window.width,
                                height_px: window.height,
                                scale: window.scale_factor,
                                frame,
                            },
                        )));
                        self.window = Some(window);
                    }
                    Err(e) => {
                        error!("Failed to create X11 window: {}", e);
                    }
                }
            }
            DisplayMgmt::Destroy { .. } => {
                // Drop window to close it
                self.window = None;
            }
        }
    }

    fn park(&mut self, status: SystemStatus) -> ActorStatus {
        if let Some(window) = &mut self.window {
            let window_id = WindowId(window.window as u64);

            // Poll for X11 events
            // If Busy, check pending without blocking.
            // If Idle, block on XNextEvent (waker will interrupt).
            let block = matches!(status, SystemStatus::Idle);

            unsafe {
                let has_event = if block {
                    true // XNextEvent blocks
                } else {
                    xlib::XPending(window.display) > 0
                };

                if has_event {
                    let mut event: xlib::XEvent = mem::zeroed();
                    xlib::XNextEvent(window.display, &mut event);

                    if let Some(display_event) = events::map_event(&event, window, window_id) {
                        if matches!(display_event, DisplayEvent::CloseRequested { .. }) {
                            info!("X11: CloseRequested");
                        }
                        self
                            .engine_handle
                            .send(Message::Data(EngineData::FromDriver(display_event)))
                            .expect("failed to send engine event");
                    }

                    // Drain remaining pending events non-blocking
                    while xlib::XPending(window.display) > 0 {
                        xlib::XNextEvent(window.display, &mut event);
                        if let Some(display_event) = events::map_event(&event, window, window_id) {
                            let _ = self
                                .engine_handle
                                .send(Message::Data(EngineData::FromDriver(display_event)));
                        }
                    }
                    return ActorStatus::Busy;
                }
            }
        }
        ActorStatus::Idle
    }
}
