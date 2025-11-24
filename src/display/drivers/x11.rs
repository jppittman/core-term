#![cfg(use_x11_display)]

//! Minimal X11 DisplayDriver implementation using x11-rb
//!
//! This driver provides basic windowing and event handling for X11.
//! Features:
//! - Window creation with fixed size
//! - Basic event handling (keyboard, mouse, window close)
//! - Framebuffer presentation via XPutImage

use crate::display::driver::DisplayDriver;
use crate::display::messages::{
    DisplayError, DisplayEvent, DriverConfig, DriverRequest, DriverResponse, RenderSnapshot,
};
use crate::platform::backends::{KeySymbol, Modifiers};
use anyhow::{Context, Result};
use log::{debug, info, trace, warn};
use std::ptr;
use x11::xlib::*;

const BYTES_PER_PIXEL: usize = 4;

pub struct X11DisplayDriver {
    display: *mut Display,
    window: Window,
    gc: GC,
    visual: *mut Visual,
    width_px: u32,
    height_px: u32,
    scale_factor: f64,
    framebuffer: Option<Box<[u8]>>,
}

impl DisplayDriver for X11DisplayDriver {
    fn new() -> Result<Self> {
        info!("X11DisplayDriver::new() - Initializing X11 display");

        unsafe {
            // Open connection to X server
            let display = XOpenDisplay(ptr::null());
            if display.is_null() {
                return Err(anyhow::anyhow!("Failed to open X11 display. Is DISPLAY set?"));
            }

            let screen = XDefaultScreen(display);
            let root = XRootWindow(display, screen);
            let visual = XDefaultVisual(display, screen);

            // Create window with fixed size (will be updated in init)
            let width = 800;
            let height = 600;

            let window = XCreateSimpleWindow(
                display,
                root,
                0,
                0,
                width,
                height,
                1,
                XBlackPixel(display, screen),
                XWhitePixel(display, screen),
            );

            if window == 0 {
                XCloseDisplay(display);
                return Err(anyhow::anyhow!("Failed to create X11 window"));
            }

            // Set window title
            let title = b"core-term\0";
            XStoreName(display, window, title.as_ptr() as *const i8);

            // Subscribe to events
            XSelectInput(
                display,
                window,
                ExposureMask
                    | KeyPressMask
                    | KeyReleaseMask
                    | ButtonPressMask
                    | ButtonReleaseMask
                    | PointerMotionMask
                    | StructureNotifyMask,
            );

            // Create graphics context
            let gc = XCreateGC(display, window, 0, ptr::null_mut());

            // Set WM_DELETE_WINDOW protocol for clean shutdown
            let wm_delete_window = XInternAtom(display, b"WM_DELETE_WINDOW\0".as_ptr() as *const i8, 0);
            XSetWMProtocols(display, window, &wm_delete_window as *const u64 as *mut u64, 1);

            info!("X11DisplayDriver: Created window {}x{}", width, height);

            Ok(Self {
                display,
                window,
                gc,
                visual,
                width_px: width,
                height_px: height,
                scale_factor: 1.0, // X11 typically doesn't auto-scale like macOS
                framebuffer: None,
            })
        }
    }

    fn handle_request(
        &mut self,
        request: DriverRequest,
    ) -> std::result::Result<DriverResponse, DisplayError> {
        match request {
            DriverRequest::Init(config) => Ok(self.handle_init(config)?),
            DriverRequest::PollEvents => Ok(self.handle_poll_events()?),
            DriverRequest::RequestFramebuffer => Ok(self.handle_request_framebuffer()?),
            DriverRequest::Present(snapshot) => self.handle_present(snapshot),
            DriverRequest::SetTitle(title) => Ok(self.handle_set_title(&title)?),
            DriverRequest::Bell => Ok(self.handle_bell()?),
            DriverRequest::SetCursorVisibility(_visible) => Ok(DriverResponse::CursorVisibilitySet),
            DriverRequest::CopyToClipboard(_text) => Ok(DriverResponse::ClipboardCopied),
            DriverRequest::RequestPaste => Ok(DriverResponse::PasteRequested),
        }
    }
}

impl X11DisplayDriver {
    fn handle_init(&mut self, config: DriverConfig) -> Result<DriverResponse> {
        info!("X11DisplayDriver: Initializing with config");

        // Calculate window size based on terminal dimensions
        let cols = config.initial_cols as u32;
        let rows = config.initial_rows as u32;
        let cell_width = config.cell_width_px as u32;
        let cell_height = config.cell_height_px as u32;

        self.width_px = cols * cell_width;
        self.height_px = rows * cell_height;

        unsafe {
            // Resize window
            XResizeWindow(self.display, self.window, self.width_px, self.height_px);

            // Map (show) window
            XMapWindow(self.display, self.window);
            XFlush(self.display);
        }

        info!(
            "X11DisplayDriver: Initialized {}x{} px",
            self.width_px, self.height_px
        );

        Ok(DriverResponse::InitComplete {
            width_px: self.width_px,
            height_px: self.height_px,
            scale_factor: self.scale_factor,
        })
    }

    fn handle_poll_events(&mut self) -> Result<DriverResponse> {
        let mut events = Vec::new();

        unsafe {
            // Process all pending events
            while XPending(self.display) > 0 {
                let mut event: XEvent = std::mem::zeroed();
                XNextEvent(self.display, &mut event);

                if let Some(display_event) = self.convert_event(&event) {
                    events.push(display_event);
                }
            }
        }

        Ok(DriverResponse::Events(events))
    }

    fn convert_event(&self, event: &XEvent) -> Option<DisplayEvent> {
        unsafe {
            match event.get_type() {
                KeyPress => {
                    let key_event = XKeyEvent::from(event.key);
                    let keysym = XLookupKeysym(&mut XKeyEvent::from(event.key) as *mut XKeyEvent, 0);

                    // Map keysym to KeySymbol
                    let symbol = self.map_keysym_to_symbol(keysym);
                    let modifiers = self.extract_modifiers(key_event.state);

                    // Get text representation
                    let mut buffer = [0u8; 32];
                    let count = XLookupString(
                        &mut XKeyEvent::from(event.key) as *mut XKeyEvent,
                        buffer.as_mut_ptr() as *mut i8,
                        buffer.len() as i32,
                        ptr::null_mut(),
                        ptr::null_mut(),
                    );

                    let text = if count > 0 {
                        Some(String::from_utf8_lossy(&buffer[..count as usize]).to_string())
                    } else {
                        None
                    };

                    Some(DisplayEvent::Key {
                        symbol,
                        modifiers,
                        text,
                    })
                }
                ButtonPress => {
                    let button_event = XButtonEvent::from(event.button);
                    Some(DisplayEvent::MouseButtonPress {
                        button: (button_event.button - 1) as u8,
                        x: button_event.x,
                        y: button_event.y,
                        scale_factor: self.scale_factor,
                        modifiers: self.extract_modifiers(button_event.state),
                    })
                }
                ButtonRelease => {
                    let button_event = XButtonEvent::from(event.button);
                    Some(DisplayEvent::MouseButtonRelease {
                        button: (button_event.button - 1) as u8,
                        x: button_event.x,
                        y: button_event.y,
                        scale_factor: self.scale_factor,
                        modifiers: self.extract_modifiers(button_event.state),
                    })
                }
                MotionNotify => {
                    let motion_event = XMotionEvent::from(event.motion);
                    Some(DisplayEvent::MouseMove {
                        x: motion_event.x,
                        y: motion_event.y,
                        scale_factor: self.scale_factor,
                        modifiers: self.extract_modifiers(motion_event.state),
                    })
                }
                ClientMessage => {
                    // WM_DELETE_WINDOW
                    Some(DisplayEvent::CloseRequested)
                }
                _ => None,
            }
        }
    }

    fn map_keysym_to_symbol(&self, keysym: u64) -> KeySymbol {
        // Map X11 keysyms to KeySymbol enum
        match keysym {
            XK_BackSpace => KeySymbol::Backspace,
            XK_Tab => KeySymbol::Tab,
            XK_Return => KeySymbol::Enter,
            XK_Escape => KeySymbol::Escape,
            XK_Delete => KeySymbol::Delete,
            XK_Home => KeySymbol::Home,
            XK_Left => KeySymbol::Left,
            XK_Up => KeySymbol::Up,
            XK_Right => KeySymbol::Right,
            XK_Down => KeySymbol::Down,
            XK_Page_Up => KeySymbol::PageUp,
            XK_Page_Down => KeySymbol::PageDown,
            XK_End => KeySymbol::End,
            XK_Insert => KeySymbol::Insert,
            XK_F1 => KeySymbol::F1,
            XK_F2 => KeySymbol::F2,
            XK_F3 => KeySymbol::F3,
            XK_F4 => KeySymbol::F4,
            XK_F5 => KeySymbol::F5,
            XK_F6 => KeySymbol::F6,
            XK_F7 => KeySymbol::F7,
            XK_F8 => KeySymbol::F8,
            XK_F9 => KeySymbol::F9,
            XK_F10 => KeySymbol::F10,
            XK_F11 => KeySymbol::F11,
            XK_F12 => KeySymbol::F12,
            _ => {
                // Try to convert to char
                if keysym < 0x100 {
                    KeySymbol::Char(keysym as u8 as char)
                } else {
                    KeySymbol::Char('\0')
                }
            }
        }
    }

    fn extract_modifiers(&self, state: u32) -> Modifiers {
        let mut modifiers = Modifiers::empty();

        if state & ShiftMask != 0 {
            modifiers |= Modifiers::SHIFT;
        }
        if state & ControlMask != 0 {
            modifiers |= Modifiers::CONTROL;
        }
        if state & Mod1Mask != 0 {
            modifiers |= Modifiers::ALT;
        }
        if state & Mod4Mask != 0 {
            modifiers |= Modifiers::SUPER;
        }

        modifiers
    }

    fn handle_request_framebuffer(&mut self) -> Result<DriverResponse> {
        let size = (self.width_px * self.height_px * BYTES_PER_PIXEL as u32) as usize;
        let framebuffer = vec![0u8; size].into_boxed_slice();
        Ok(DriverResponse::Framebuffer(framebuffer))
    }

    fn handle_present(&mut self, snapshot: RenderSnapshot) -> std::result::Result<DriverResponse, DisplayError> {
        trace!("X11DisplayDriver: Presenting frame");

        let RenderSnapshot {
            framebuffer,
            width_px,
            height_px,
        } = snapshot;

        unsafe {
            // Create XImage from framebuffer
            let image = XCreateImage(
                self.display,
                self.visual,
                24, // depth
                ZPixmap,
                0,
                framebuffer.as_ptr() as *mut i8,
                self.width_px,
                self.height_px,
                32, // bitmap_pad
                0,  // bytes_per_line (auto-calculate)
            );

            if image.is_null() {
                let recovered_snapshot = RenderSnapshot {
                    framebuffer,
                    width_px,
                    height_px,
                };
                return Err(DisplayError::PresentationFailed(
                    recovered_snapshot,
                    "Failed to create XImage".to_string()
                ));
            }

            // Draw image to window
            XPutImage(
                self.display,
                self.window,
                self.gc,
                image,
                0,
                0,
                0,
                0,
                self.width_px,
                self.height_px,
            );

            // Clean up (don't free data, we still own framebuffer)
            (*image).data = ptr::null_mut();
            XDestroyImage(image);

            XFlush(self.display);
        }

        let recovered_snapshot = RenderSnapshot {
            framebuffer,
            width_px,
            height_px,
        };
        Ok(DriverResponse::PresentComplete(recovered_snapshot))
    }

    fn handle_set_title(&mut self, title: &str) -> Result<DriverResponse> {
        unsafe {
            let c_title = std::ffi::CString::new(title)?;
            XStoreName(self.display, self.window, c_title.as_ptr());
            XFlush(self.display);
        }
        Ok(DriverResponse::TitleSet)
    }

    fn handle_bell(&mut self) -> Result<DriverResponse> {
        unsafe {
            XBell(self.display, 0);
            XFlush(self.display);
        }
        Ok(DriverResponse::BellRung)
    }
}

impl Drop for X11DisplayDriver {
    fn drop(&mut self) {
        info!("X11DisplayDriver::drop() - Cleaning up");
        unsafe {
            if !self.gc.is_null() {
                XFreeGC(self.display, self.gc);
            }
            if self.window != 0 {
                XDestroyWindow(self.display, self.window);
            }
            if !self.display.is_null() {
                XCloseDisplay(self.display);
            }
        }
        info!("X11DisplayDriver::drop() - Cleanup complete");
    }
}
