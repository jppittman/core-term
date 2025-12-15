use crate::api::private::{EngineActorHandle, EngineData, WindowId};
use crate::display::messages::{DisplayControl, DisplayData, DisplayEvent, DisplayMgmt};
use crate::display::platform::Platform;
use crate::input::KeySymbol;
use crate::platform::macos::cocoa::{self, event_type, NSApplication};
use crate::platform::macos::events;
use crate::platform::macos::sys;
use crate::platform::macos::window::MacWindow;
use actor_scheduler::{Actor, Message, ParkHint};
use anyhow::Result;
use pixelflow_render::color::Rgba;
use std::collections::HashMap;

/// The macOS Platform Actor.
/// Manages NSApplication, NSWindows, and Event Loop.
pub struct MetalPlatform {
    app: NSApplication,
    windows: HashMap<WindowId, MacWindow>,
    // Mapping from NSWindow pointer to WindowId for event routing
    // Note: NSWindow is a wrapper around Id, so we cast Id to usize or wrap generic
    window_map: HashMap<usize, WindowId>,
    event_tx: EngineActorHandle<Rgba>,
}

unsafe impl Send for MetalPlatform {}

impl MetalPlatform {
    pub fn new(event_tx: EngineActorHandle<Rgba>) -> Result<Self> {
        let app = NSApplication::shared();
        app.set_activation_policy(0); // Regular app
        app.finish_launching();
        app.activate_ignoring_other_apps(true);

        Ok(Self {
            app,
            windows: HashMap::new(),
            window_map: HashMap::new(),
            event_tx,
        })
    }
}

impl Platform for MetalPlatform {
    type Pixel = Rgba;
}

impl Actor<DisplayData<Rgba>, DisplayControl, DisplayMgmt> for MetalPlatform {
    fn handle_data(&mut self, msg: DisplayData<Rgba>) {
        match msg {
            DisplayData::Present { id, frame } => {
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
                let pb = cocoa::NSPasteboard::general();
                pb.clear_contents();
                pb.set_string(&text);
            }
            DisplayControl::RequestPaste => {
                // Logic to get pasteboard string and send back event?
                // But we are the platform, to whom do we send?
                // The Actor trait doesn't have a "Sender" back to Engine.
                // The DriverActor has the scheduler.
                // Does Platform have a reference to EngineSender?
                // Ah, Platform trait usually doesn't enforce it, but we might need it.
                // Wait, we generate DisplayEvents. Where do they go?
                // `Actor::park` returns events? No.
                // The `DriverActor` holds the scheduler.
                // The `DriverActor` used to have `cmd_tx`... no.

                // `DriverActor` needs to forward events from `Platform` to `Engine`.
                // `MetalPlatform` needs `EngineSender`.
                // We should add `event_tx: Sender<DisplayEvent>` to MetalPlatform.
                // But `Platform` trait doesn't mandate `new`. `DriverActor` accepts `P: Platform`.
                // So `MetalPlatform::new(event_tx)` is how we construct it.
            }
            DisplayControl::Shutdown => {
                unsafe {
                    // [NSApp terminate:nil]
                    let app = NSApplication::shared();
                    sys::send_1::<(), sys::Id>(
                        app.0,
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
                        self.windows.insert(id, win);
                        self.window_map.insert(ptr as usize, id);
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

    fn park(&mut self, hint: ParkHint) {
        // Logic for event loop interaction
        unsafe {
            let until_date = match hint {
                ParkHint::Wait => {
                    // Distant Future
                    let cls = sys::class(b"NSDate\0");
                    sys::send(cls, sys::sel(b"distantFuture\0"))
                }
                ParkHint::Poll => {
                    // Distant Past
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

            let event = self.app.next_event(u64::MAX, until_date, mode, true);

            // Release mode string if make_nsstring implies ownership or autorelease?
            // make_nsstring uses alloc/init so it returns +1 retained object.
            // We should release it.
            sys::send::<()>(mode, sys::sel(b"release\0"));

            if !event.is_null() {
                let ty = event.type_();
                match ty {
                    event_type::APP_KIT_DEFINED
                    | event_type::SYSTEM_DEFINED
                    | event_type::APPLICATION_DEFINED => {
                        // Internal or WakeUp
                    }
                    _ => {
                        // Map to DisplayEvent
                        // We need key symbol, etc.
                        // And we need to route to window ID.
                        let ns_win = event.window();
                        if !ns_win.0.is_null() {
                            if let Some(wid) = self.window_map.get(&(ns_win.0 as usize)) {
                                // We have the window ID.
                                // Get window height from self.windows if needed for coordinate flip.
                                // But map_event needs window_height.
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
    }
}
