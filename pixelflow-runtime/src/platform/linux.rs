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
use std::mem;

#[cfg(use_x11_display)]
use x11::xlib;

/// Linux platform pixel type (BGRA for X11).
pub type LinuxPixel = Bgra8;

/// Linux platform operations (X11-based).
#[allow(dead_code)]
pub struct LinuxOps {
    engine_handle: EngineActorHandle<LinuxPixel>,
    #[cfg(use_x11_display)]
    display: *mut xlib::Display,
    #[cfg(use_x11_display)]
    waker_window: xlib::Window,
    #[cfg(use_x11_display)]
    wake_atom: xlib::Atom,
}

// SAFETY: We own the display connection and only access it from the main thread.
// XSendEvent is thread-safe after XInitThreads.
unsafe impl Send for LinuxOps {}

impl LinuxOps {
    /// Create new Linux platform ops.
    pub fn new(engine_handle: EngineActorHandle<LinuxPixel>) -> Result<Self> {
        // We assume X11 is available on Linux for this implementation.
        // If use_x11_display is not set, this will fail to compile (X11Waker missing),
        // which is preferable to silent fallback.
        #[cfg(use_x11_display)]
        unsafe {
            // Ensure X11 multithreading support is initialized
            if xlib::XInitThreads() == 0 {
                return Err(anyhow::anyhow!("Failed to initialize X11 threads"));
            }

            let display = xlib::XOpenDisplay(std::ptr::null());
            if display.is_null() {
                return Err(anyhow::anyhow!("Failed to open X11 display"));
            }

            let screen = xlib::XDefaultScreen(display);
            let root = xlib::XRootWindow(display, screen);

            // Create a 1x1 unmapped window for receiving wake events
            let waker_window =
                xlib::XCreateSimpleWindow(display, root, 0, 0, 1, 1, 0, 0, 0);

            // Configure the global X11Waker to use this display/window.
            // Note: We use a singleton Waker because the actor instantiation (via troupe! macro)
            // does not allow easy injection of a shared Waker instance.
            let waker = crate::platform::waker::X11Waker::new();
            waker.set_target(display, waker_window);
            let wake_atom = waker
                .wake_atom()
                .ok_or_else(|| anyhow::anyhow!("Failed to get wake atom"))?;

            Ok(Self {
                engine_handle,
                display,
                waker_window,
                wake_atom,
            })
        }

        #[cfg(not(use_x11_display))]
        {
            // If built without X11 support on Linux (e.g. explicit headless), error out
            // or return a stub that panics on park.
            // However, since build.rs forces x11 on linux, this path should be unreachable
            // unless manually overridden.
            Err(anyhow::anyhow!("LinuxOps requires X11 support (use_x11_display)"))
        }
    }

    #[cfg(use_x11_display)]
    unsafe fn process_event(&mut self, event: &xlib::XEvent) {
        // Handle wake event
        if event.type_ == xlib::ClientMessage {
            let msg = event.client_message;
            if msg.window == self.waker_window && msg.message_type == self.wake_atom {
                // Wake up event - intentional no-op to return from blocking
                return;
            }
        }

        // TODO: Handle other X11 events
    }
}

impl Drop for LinuxOps {
    fn drop(&mut self) {
        #[cfg(use_x11_display)]
        unsafe {
            if !self.display.is_null() {
                // IMPORTANT: clear waker first to avoid use-after-free race
                crate::platform::waker::X11Waker::clear();

                xlib::XDestroyWindow(self.display, self.waker_window);
                xlib::XCloseDisplay(self.display);
            }
        }
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

    fn park(&mut self, hint: ParkHint) -> ParkHint {
        #[cfg(use_x11_display)]
        unsafe {
            xlib::XFlush(self.display);

            match hint {
                ParkHint::Poll => {
                    // Process pending events without blocking
                    while xlib::XPending(self.display) > 0 {
                        let mut event = mem::zeroed();
                        xlib::XNextEvent(self.display, &mut event);
                        self.process_event(&event);
                    }
                }
                ParkHint::Wait => {
                    // Block until an event arrives
                    let mut event = mem::zeroed();
                    xlib::XNextEvent(self.display, &mut event);
                    self.process_event(&event);

                    // Process any remaining pending events
                    while xlib::XPending(self.display) > 0 {
                        let mut event = mem::zeroed();
                        xlib::XNextEvent(self.display, &mut event);
                        self.process_event(&event);
                    }
                }
            }
        }

        #[cfg(not(use_x11_display))]
        {
            // Fallback to sleep if no X11
            if matches!(hint, ParkHint::Wait) {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }

        hint
    }
}
