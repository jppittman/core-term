use crate::api::private::{EngineActorHandle, EngineData, WindowId};
use crate::display::messages::{DisplayControl, DisplayData, DisplayEvent, DisplayMgmt};
use crate::display::ops::PlatformOps;
use crate::platform::macos::cocoa::{self, event_type, NSApplication, NSPasteboard};
use crate::platform::macos::events;
use crate::platform::macos::sys;
use crate::platform::macos::window::MacWindow;
use crate::platform::PlatformPixel;
use actor_scheduler::{Message, ParkHint};
use anyhow::Result;

use std::collections::HashMap;

const NS_APPLICATION_ACTIVATION_POLICY_REGULAR: isize = 0;

/// The macOS Platform Actor.
/// Manages NSApplication, NSWindows, and Event Loop.
pub struct MetalOps {
    app: NSApplication,
    windows: HashMap<WindowId, MacWindow>,
    // Mapping from NSWindow pointer to WindowId for event routing
    // Note: NSWindow is a wrapper around Id, so we cast Id to usize or wrap generic
    window_map: HashMap<usize, WindowId>,
    // Handle to send events back to the engine
    event_tx: EngineActorHandle<PlatformPixel>,
}

unsafe impl Send for MetalOps {}

impl MetalOps {
    pub fn new(event_tx: EngineActorHandle<PlatformPixel>) -> Result<Self> {
        // Initialize Cocoa Application
        let app = unsafe {
            // Pool:
            let cls_pool = sys::class(b"NSAutoreleasePool\0");
            let _pool: sys::Id = sys::send(
                sys::send(cls_pool, sys::sel(b"alloc\0")),
                sys::sel(b"init\0"),
            );

            let app = NSApplication::shared();

            // Use wrapper methods
            app.set_activation_policy(NS_APPLICATION_ACTIVATION_POLICY_REGULAR);

            app.finish_launching();
            app.activate_ignoring_other_apps(true);

            app
        };

        Ok(Self {
            app,
            windows: HashMap::new(),
            window_map: HashMap::new(),
            event_tx,
        })
    }
}

impl PlatformOps for MetalOps {
    type Pixel = PlatformPixel;

    fn handle_data(&mut self, msg: DisplayData<Self::Pixel>) {
        match msg {
            DisplayData::Present { id, frame } => {
                log::trace!("MetalOps: Presenting frame for window {:?}", id);
                if let Some(win) = self.windows.get_mut(&id) {
                    win.present(frame);
                }
            }
        }
    }

    fn handle_control(&mut self, msg: DisplayControl) {
        match msg {
            DisplayControl::SetTitle { id, title } => {
                if let Some(win) = self.windows.get_mut(&id) {
                    win.set_title(&title);
                }
            }
            DisplayControl::SetSize { id, width, height } => {
                if let Some(win) = self.windows.get_mut(&id) {
                    win.set_size(width, height);
                }
            }
            DisplayControl::SetCursor { id, cursor } => {
                if let Some(win) = self.windows.get_mut(&id) {
                    win.set_cursor(cursor);
                }
            }
            DisplayControl::SetVisible { id, visible } => {
                if let Some(win) = self.windows.get_mut(&id) {
                    win.set_visible(visible);
                }
            }
            DisplayControl::RequestRedraw { id } => {
                if let Some(win) = self.windows.get_mut(&id) {
                    win.request_redraw();
                }
            }
            DisplayControl::Bell => {
                // NSBeep()
            }
            DisplayControl::Copy { text } => {
                let pb = NSPasteboard::general();
                pb.clear_contents();
                pb.set_string(&text);
            }
            DisplayControl::RequestPaste => {
                // Implementation pending
            }
            DisplayControl::Shutdown => {
                unsafe {
                    // [NSApp terminate:nil]
                    // Wrapper doesn't have terminate yet, stick to sys::send
                    sys::send_1::<(), sys::Id>(
                        self.app.0,
                        sys::sel(b"terminate:\0"),
                        std::ptr::null_mut(),
                    );
                }
            }
        }
    }

    fn handle_management(&mut self, msg: DisplayMgmt) {
        match msg {
            DisplayMgmt::Create { id, settings } => {
                match MacWindow::new(settings) {
                    Ok(win) => {
                        let ptr = win.window.0;
                        let width = win.current_width;
                        let height = win.current_height;
                        let scale = win.scale_factor();

                        self.windows.insert(id, win);
                        self.window_map.insert(ptr as usize, id);

                        // Emit WindowCreated event so Engine knows initial size
                        let _ = self.event_tx.send(Message::Data(EngineData::FromDriver(
                            DisplayEvent::WindowCreated {
                                id,
                                width_px: width,
                                height_px: height,
                                scale,
                            },
                        )));
                    }
                    Err(e) => {
                        // Log error?
                        eprintln!("Failed to create window: {}", e);
                    }
                }
            }
            DisplayMgmt::Destroy { id } => {
                if let Some(mut win) = self.windows.remove(&id) {
                    win.set_visible(false);
                    // Drop closes it implicitly or we call close
                    // win.window.close(); // If we expose it
                    self.window_map.remove(&(win.window.0 as usize));
                }
            }
        }
    }

    fn park(&mut self, hint: ParkHint) -> ParkHint {
        // Logic for event loop interaction
        unsafe {
            let until_date = match hint {
                ParkHint::Wait => {
                    // Distant Future
                    let cls = sys::class(b"NSDate\0");
                    sys::send(cls, sys::sel(b"distantFuture\0"))
                }
                ParkHint::Poll => {
                    // Distant Past (return immediately)
                    let cls = sys::class(b"NSDate\0");
                    sys::send(cls, sys::sel(b"distantPast\0"))
                }
            };

            let mode = cocoa::make_nsstring("kCFRunLoopDefaultMode");

            // Poll for window resize
            for (id, window) in self.windows.iter_mut() {
                if let Some((width, height)) = window.poll_resize() {
                    let _ = self.event_tx.send(Message::Data(EngineData::FromDriver(
                        DisplayEvent::Resized {
                            id: *id,
                            width_px: width,
                            height_px: height,
                        },
                    )));
                }
            }

            // Use wrapper for next_event
            let event = self.app.next_event(
                u64::MAX,
                until_date,
                mode,
                true, // dequeue
            );

            // Release mode string
            sys::send::<()>(mode, sys::sel(b"release\0"));

            if !event.is_null() {
                let ty = event.type_();
                match ty {
                    event_type::APP_KIT_DEFINED
                    | event_type::SYSTEM_DEFINED
                    | event_type::APPLICATION_DEFINED => {
                        log::trace!("MetalOps: Received Internal/WakeUp event");
                    }
                    _ => {
                        let ns_win = event.window();
                        if !ns_win.0.is_null() {
                            if let Some(wid) = self.window_map.get(&(ns_win.0 as usize)) {
                                // We have the window ID.
                                // Get window height from self.windows if needed for coordinate flip.
                                let height = if let Some(w) = self.windows.get(wid) {
                                    w.size().1 as f64
                                } else {
                                    0.0
                                };

                                if let Some(ev) = events::map_event(event, height) {
                                    // Dispatch ev!
                                    // We send it to the Engine via event_tx
                                    let _ = self
                                        .event_tx
                                        .send(Message::Data(EngineData::FromDriver(ev)));
                                }
                            }
                        }
                    }
                }

                self.app.send_event(event);
            }
        }
        hint
    }
}
