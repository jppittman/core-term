#![cfg(use_x11_display)]

//! X11 DisplayDriver implementation using xlib.

use crate::display::driver::DisplayDriver;
use crate::display::messages::{
    DisplayError, DisplayEvent, DriverConfig, DriverRequest, DriverResponse, RenderSnapshot,
};
use crate::input::{KeySymbol, Modifiers};
use crate::platform::waker::EventLoopWaker;
use anyhow::{anyhow, Result};
use log::{debug, info};
use std::ffi::CString;
use std::mem;
use std::os::raw::c_int;
use std::ptr;
use x11::{xlib, xlib::KeySym, keysym};

// --- Atoms ---
#[derive(Debug, Clone, Copy)]
struct SelectionAtoms {
    clipboard: xlib::Atom,
    targets: xlib::Atom,
    utf8_string: xlib::Atom,
    text: xlib::Atom,
    xa_string: xlib::Atom,
}

impl SelectionAtoms {
    fn new(display: *mut xlib::Display) -> Self {
        unsafe {
            let intern =
                |name: &[u8]| xlib::XInternAtom(display, name.as_ptr() as *const i8, xlib::False);
            Self {
                clipboard: intern(b"CLIPBOARD\0"),
                targets: intern(b"TARGETS\0"),
                utf8_string: intern(b"UTF8_STRING\0"),
                text: intern(b"TEXT\0"),
                xa_string: xlib::XA_STRING,
            }
        }
    }
}

// --- Waker Implementation ---

pub struct X11Waker {
    window: xlib::Window,
}

impl X11Waker {
    pub fn new(window: xlib::Window) -> Self {
        Self { window }
    }
}

impl EventLoopWaker for X11Waker {
    fn wake(&self) -> Result<()> {
        unsafe {
            let display = xlib::XOpenDisplay(ptr::null());
            if display.is_null() {
                return Err(anyhow!("Failed to open X display for Waker"));
            }

            let mut event: xlib::XClientMessageEvent = mem::zeroed();
            event.type_ = xlib::ClientMessage;
            event.window = self.window;
            event.format = 32;

            xlib::XSendEvent(
                display,
                self.window,
                xlib::False,
                xlib::NoEventMask,
                &mut xlib::XEvent {
                    client_message: event,
                },
            );

            xlib::XFlush(display);
            xlib::XCloseDisplay(display);
        }
        Ok(())
    }
}

// --- Display Driver ---

pub struct X11DisplayDriver {
    display: *mut xlib::Display,
    screen: c_int,
    _root: xlib::Window,

    window: xlib::Window,
    gc: xlib::GC,
    wm_delete_window: xlib::Atom,
    atoms: SelectionAtoms,

    width_px: u32,
    height_px: u32,
    scale_factor: f64,

    // State to track pending clipboard requests
    pending_selection_request: Option<xlib::XSelectionRequestEvent>,
}

impl DisplayDriver for X11DisplayDriver {
    fn new(config: &DriverConfig) -> Result<Self> {
        unsafe {
            if xlib::XInitThreads() == 0 {
                return Err(anyhow!("XInitThreads failed"));
            }

            let display = xlib::XOpenDisplay(ptr::null());
            if display.is_null() {
                return Err(anyhow!("Failed to open X display"));
            }

            let screen = xlib::XDefaultScreen(display);
            let root = xlib::XRootWindow(display, screen);
            let _visual = xlib::XDefaultVisual(display, screen);

            let wm_delete_name = CString::new("WM_DELETE_WINDOW").unwrap();
            let wm_delete_window = xlib::XInternAtom(display, wm_delete_name.as_ptr(), xlib::False);

            let atoms = SelectionAtoms::new(display);

            // Calculate window size from config
            let width = (config.initial_cols * config.cell_width_px) as u32;
            let height = (config.initial_rows * config.cell_height_px) as u32;

            let black = xlib::XBlackPixel(display, screen);
            let white = xlib::XWhitePixel(display, screen);

            let window =
                xlib::XCreateSimpleWindow(display, root, 0, 0, width, height, 0, white, black);

            // Select Input Events
            xlib::XSelectInput(
                display,
                window,
                xlib::KeyPressMask
                    | xlib::KeyReleaseMask
                    | xlib::ButtonPressMask
                    | xlib::ButtonReleaseMask
                    | xlib::PointerMotionMask
                    | xlib::StructureNotifyMask
                    | xlib::FocusChangeMask
                    | xlib::ExposureMask
                    | xlib::PropertyChangeMask,
            );

            // Set WM Protocols
            xlib::XSetWMProtocols(display, window, &wm_delete_window as *const _ as *mut _, 1);

            // Initialize GC
            let gc = xlib::XCreateGC(display, window, 0, ptr::null_mut());
            xlib::XSetForeground(display, gc, white);
            xlib::XSetBackground(display, gc, black);

            // Map Window
            xlib::XMapWindow(display, window);
            xlib::XFlush(display);

            info!("X11DisplayDriver created: {}x{} px", width, height);

            Ok(Self {
                display,
                screen,
                _root: root,
                window,
                gc,
                wm_delete_window,
                atoms,
                width_px: width,
                height_px: height,
                scale_factor: 1.0,
                pending_selection_request: None,
            })
        }
    }

    fn create_waker(&self) -> Box<dyn EventLoopWaker> {
        Box::new(X11Waker::new(self.window))
    }

    fn handle_request(
        &mut self,
        request: DriverRequest,
    ) -> std::result::Result<DriverResponse, DisplayError> {
        match request {
            DriverRequest::Init => Ok(self.handle_init()?),
            DriverRequest::PollEvents => Ok(self.handle_poll_events()?),
            DriverRequest::RequestFramebuffer => Ok(self.handle_request_framebuffer()?),
            DriverRequest::Present(snapshot) => self.handle_present(snapshot),
            DriverRequest::SetTitle(title) => Ok(self.handle_set_title(&title)?),
            DriverRequest::Bell => Ok(self.handle_bell()?),
            DriverRequest::SetCursorVisibility(_visible) => Ok(DriverResponse::CursorVisibilitySet),
            DriverRequest::CopyToClipboard(text) => Ok(self.handle_copy_to_clipboard(&text)?),
            DriverRequest::SubmitClipboardData(text) => {
                Ok(self.handle_submit_clipboard_data(&text)?)
            }
            DriverRequest::RequestPaste => Ok(self.handle_request_paste()?),
        }
    }
}

impl X11DisplayDriver {
    fn handle_init(&mut self) -> Result<DriverResponse> {
        info!("X11: Init - returning metrics");
        Ok(DriverResponse::InitComplete {
            width_px: self.width_px,
            height_px: self.height_px,
            scale_factor: self.scale_factor,
        })
    }

    fn handle_poll_events(&mut self) -> Result<DriverResponse> {
        let mut events = Vec::new();

        unsafe {
            while xlib::XPending(self.display) > 0 {
                let mut event: xlib::XEvent = mem::zeroed();
                xlib::XNextEvent(self.display, &mut event);

                match event.type_ {
                    xlib::ClientMessage => {
                        let client = event.client_message;
                        // Check if this is WM_DELETE_WINDOW
                        let data0 = client.data.as_longs()[0];
                        if data0 as xlib::Atom != self.wm_delete_window {
                            debug!("X11: Received ClientMessage (Wake)");
                            continue;
                        }
                        events.push(DisplayEvent::CloseRequested);
                    }
                    xlib::SelectionRequest => {
                        self.pending_selection_request = Some(event.selection_request);
                        events.push(DisplayEvent::ClipboardDataRequested);
                    }
                    xlib::SelectionNotify => {
                        if let Some(paste_event) = self.handle_selection_notify(&event.selection) {
                            events.push(paste_event);
                        }
                    }
                    _ => {
                        if let Some(de) = self.convert_event(&event) {
                            events.push(de);
                        }
                    }
                }
            }
        }

        Ok(DriverResponse::Events(events))
    }

    fn convert_event(&self, event: &xlib::XEvent) -> Option<DisplayEvent> {
        unsafe {
            match event.type_ {
                xlib::KeyPress => {
                    let key_event = event.key;
                    let mut keysym = 0;
                    let mut buffer = [0u8; 32];
                    let count = xlib::XLookupString(
                        &key_event as *const _ as *mut _,
                        buffer.as_mut_ptr() as *mut i8,
                        buffer.len() as c_int,
                        &mut keysym,
                        ptr::null_mut(),
                    );

                    let text = if count > 0 {
                        Some(String::from_utf8_lossy(&buffer[..count as usize]).to_string())
                    } else {
                        None
                    };

                    let modifiers = self.extract_modifiers(key_event.state);
                    let symbol = self.xkeysym_to_keysymbol(keysym, text.as_deref().unwrap_or(""));

                    Some(DisplayEvent::Key {
                        symbol,
                        modifiers,
                        text,
                    })
                }
                xlib::ConfigureNotify => {
                    let conf = event.configure;
                    Some(DisplayEvent::Resize {
                        width_px: conf.width as u32,
                        height_px: conf.height as u32,
                    })
                }
                xlib::FocusIn => Some(DisplayEvent::FocusGained),
                xlib::FocusOut => Some(DisplayEvent::FocusLost),

                xlib::ButtonPress | xlib::ButtonRelease | xlib::MotionNotify => {
                    self.convert_mouse_event(event)
                }

                _ => None,
            }
        }
    }

    unsafe fn convert_mouse_event(&self, event: &xlib::XEvent) -> Option<DisplayEvent> {
        match event.type_ {
            xlib::ButtonPress => {
                let e = event.button;
                let modifiers = self.extract_modifiers(e.state);
                Some(DisplayEvent::MouseButtonPress {
                    button: e.button as u8,
                    x: e.x,
                    y: e.y,
                    scale_factor: self.scale_factor,
                    modifiers,
                })
            }
            xlib::ButtonRelease => {
                let e = event.button;
                let modifiers = self.extract_modifiers(e.state);
                Some(DisplayEvent::MouseButtonRelease {
                    button: e.button as u8,
                    x: e.x,
                    y: e.y,
                    scale_factor: self.scale_factor,
                    modifiers,
                })
            }
            xlib::MotionNotify => {
                let e = event.motion;
                let modifiers = self.extract_modifiers(e.state);
                Some(DisplayEvent::MouseMove {
                    x: e.x,
                    y: e.y,
                    scale_factor: self.scale_factor,
                    modifiers,
                })
            }
            _ => None,
        }
    }

    fn extract_modifiers(&self, state: u32) -> Modifiers {
        let mut modifiers = Modifiers::empty();
        if (state & xlib::ShiftMask) != 0 {
            modifiers.insert(Modifiers::SHIFT);
        }
        if (state & xlib::ControlMask) != 0 {
            modifiers.insert(Modifiers::CONTROL);
        }
        if (state & xlib::Mod1Mask) != 0 {
            modifiers.insert(Modifiers::ALT);
        }
        if (state & xlib::Mod4Mask) != 0 {
            modifiers.insert(Modifiers::SUPER);
        }
        modifiers
    }

    fn xkeysym_to_keysymbol(&self, keysym_val: xlib::KeySym, text: &str) -> KeySymbol {
        if !text.is_empty() {
            let chars: Vec<char> = text.chars().collect();
            if chars.len() == 1 && chars[0] != '\u{FFFD}' {
                return KeySymbol::Char(chars[0]);
            }
        }

        match keysym_val as u32 {
            keysym::XK_Shift_L | keysym::XK_Shift_R => KeySymbol::Shift,
            keysym::XK_Control_L | keysym::XK_Control_R => KeySymbol::Control,
            keysym::XK_Alt_L | keysym::XK_Alt_R | keysym::XK_Meta_L | keysym::XK_Meta_R => {
                KeySymbol::Alt
            }
            keysym::XK_Super_L | keysym::XK_Super_R => KeySymbol::Super,
            keysym::XK_Return => KeySymbol::Enter,
            keysym::XK_BackSpace => KeySymbol::Backspace,
            keysym::XK_Tab => KeySymbol::Tab,
            keysym::XK_Escape => KeySymbol::Escape,
            keysym::XK_Home => KeySymbol::Home,
            keysym::XK_Left => KeySymbol::Left,
            keysym::XK_Up => KeySymbol::Up,
            keysym::XK_Right => KeySymbol::Right,
            keysym::XK_Down => KeySymbol::Down,
            keysym::XK_Page_Up => KeySymbol::PageUp,
            keysym::XK_Page_Down => KeySymbol::PageDown,
            keysym::XK_End => KeySymbol::End,
            keysym::XK_Insert => KeySymbol::Insert,
            keysym::XK_Delete => KeySymbol::Delete,
            keysym::XK_F1 => KeySymbol::F1,
            keysym::XK_F2 => KeySymbol::F2,
            keysym::XK_F3 => KeySymbol::F3,
            keysym::XK_F4 => KeySymbol::F4,
            keysym::XK_F5 => KeySymbol::F5,
            keysym::XK_F6 => KeySymbol::F6,
            keysym::XK_F7 => KeySymbol::F7,
            keysym::XK_F8 => KeySymbol::F8,
            keysym::XK_F9 => KeySymbol::F9,
            keysym::XK_F10 => KeySymbol::F10,
            keysym::XK_F11 => KeySymbol::F11,
            keysym::XK_F12 => KeySymbol::F12,
            _ => KeySymbol::Unknown,
        }
    }

    // --- Clipboard Handling ---

    fn handle_submit_clipboard_data(&mut self, text: &str) -> Result<DriverResponse> {
        // If we have a pending request, fulfill it
        if let Some(req) = self.pending_selection_request.take() {
            unsafe {
                self.fulfill_selection_request(&req, text);
            }
        }

        // Re-assert ownership
        unsafe {
            xlib::XSetSelectionOwner(
                self.display,
                self.atoms.clipboard,
                self.window,
                xlib::CurrentTime,
            );
            xlib::XFlush(self.display);
        }
        Ok(DriverResponse::ClipboardDataSubmitted)
    }

    unsafe fn fulfill_selection_request(&self, req: &xlib::XSelectionRequestEvent, text: &str) {
        let mut ev: xlib::XSelectionEvent = mem::zeroed();
        ev.type_ = xlib::SelectionNotify;
        ev.display = req.display;
        ev.requestor = req.requestor;
        ev.selection = req.selection;
        ev.target = req.target;
        ev.time = req.time;
        ev.property = 0; // Reject by default

        if req.target == self.atoms.utf8_string
            || req.target == self.atoms.text
            || req.target == self.atoms.xa_string
        {
            // Send data
            xlib::XChangeProperty(
                self.display,
                req.requestor,
                req.property,
                req.target,
                8,
                xlib::PropModeReplace,
                text.as_ptr(),
                text.len() as c_int,
            );
            ev.property = req.property;
        } else if req.target == self.atoms.targets {
            // Send supported targets
            let targets = [
                self.atoms.utf8_string,
                self.atoms.text,
                self.atoms.xa_string,
                self.atoms.targets,
            ];
            xlib::XChangeProperty(
                self.display,
                req.requestor,
                req.property,
                xlib::XA_ATOM,
                32,
                xlib::PropModeReplace,
                targets.as_ptr() as *const u8,
                targets.len() as c_int,
            );
            ev.property = req.property;
        }

        xlib::XSendEvent(
            self.display,
            req.requestor,
            xlib::False,
            xlib::NoEventMask,
            &mut xlib::XEvent { selection: ev },
        );
        xlib::XFlush(self.display);
    }

    unsafe fn handle_selection_notify(
        &self,
        event: &xlib::XSelectionEvent,
    ) -> Option<DisplayEvent> {
        if event.property == 0 {
            return None;
        } // Paste failed

        let mut type_ret = 0;
        let mut format_ret = 0;
        let mut nitems = 0;
        let mut bytes_after = 0;
        let mut prop_ret = ptr::null_mut();

        xlib::XGetWindowProperty(
            self.display,
            event.requestor,
            event.property,
            0,
            i64::MAX / 4,
            xlib::True,
            xlib::AnyPropertyType as u64,
            &mut type_ret,
            &mut format_ret,
            &mut nitems,
            &mut bytes_after,
            &mut prop_ret,
        );

        if !prop_ret.is_null() {
            let data = std::slice::from_raw_parts(prop_ret, nitems as usize);
            let text = String::from_utf8_lossy(data).to_string();
            xlib::XFree(prop_ret as *mut std::ffi::c_void);
            return Some(DisplayEvent::PasteData { text });
        }
        None
    }

    fn handle_copy_to_clipboard(&mut self, _text: &str) -> Result<DriverResponse> {
        // Just assert ownership - data will be provided when requested
        unsafe {
            xlib::XSetSelectionOwner(
                self.display,
                self.atoms.clipboard,
                self.window,
                xlib::CurrentTime,
            );
            xlib::XFlush(self.display);
        }
        Ok(DriverResponse::ClipboardCopied)
    }

    fn handle_request_paste(&mut self) -> Result<DriverResponse> {
        unsafe {
            xlib::XConvertSelection(
                self.display,
                self.atoms.clipboard,
                self.atoms.utf8_string,
                self.atoms.clipboard,
                self.window,
                xlib::CurrentTime,
            );
            xlib::XFlush(self.display);
        }
        Ok(DriverResponse::PasteRequested)
    }

    fn handle_request_framebuffer(&mut self) -> Result<DriverResponse> {
        let size = (self.width_px * self.height_px * 4) as usize;
        let buffer = vec![0u8; size].into_boxed_slice();
        Ok(DriverResponse::Framebuffer(buffer))
    }

    fn handle_present(
        &mut self,
        snapshot: RenderSnapshot,
    ) -> std::result::Result<DriverResponse, DisplayError> {
        unsafe {
            let depth = xlib::XDefaultDepth(self.display, self.screen);
            let visual = xlib::XDefaultVisual(self.display, self.screen);
            let data_ptr = snapshot.framebuffer.as_ptr() as *mut i8;

            let image = xlib::XCreateImage(
                self.display,
                visual,
                depth as u32,
                xlib::ZPixmap,
                0,
                data_ptr,
                snapshot.width_px,
                snapshot.height_px,
                32,
                0,
            );

            if image.is_null() {
                return Err(DisplayError::PresentationFailed(
                    snapshot,
                    "XCreateImage failed".into(),
                ));
            }

            xlib::XPutImage(
                self.display,
                self.window,
                self.gc,
                image,
                0,
                0,
                0,
                0,
                snapshot.width_px,
                snapshot.height_px,
            );

            (*image).data = ptr::null_mut(); // Don't let XDestroyImage free our Rust buffer
            xlib::XDestroyImage(image);
            xlib::XFlush(self.display);
        }

        Ok(DriverResponse::PresentComplete(snapshot))
    }

    fn handle_set_title(&mut self, title: &str) -> Result<DriverResponse> {
        unsafe {
            let c_title = CString::new(title)?;
            xlib::XStoreName(self.display, self.window, c_title.as_ptr());
            xlib::XFlush(self.display);
        }
        Ok(DriverResponse::TitleSet)
    }

    fn handle_bell(&mut self) -> Result<DriverResponse> {
        unsafe {
            xlib::XBell(self.display, 0);
            xlib::XFlush(self.display);
        }
        Ok(DriverResponse::BellRung)
    }
}

impl Drop for X11DisplayDriver {
    fn drop(&mut self) {
        unsafe {
            xlib::XFreeGC(self.display, self.gc);
            xlib::XDestroyWindow(self.display, self.window);
            xlib::XCloseDisplay(self.display);
        }
        info!("X11DisplayDriver dropped");
    }
}
