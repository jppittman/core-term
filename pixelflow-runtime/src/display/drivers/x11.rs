//! X11 DisplayDriver implementation using xlib.
//!
//! Driver struct is just cmd_tx - trivially Clone.
//! run() reads Configure, creates X11 resources, runs event loop.

use crate::channel::{DriverCommand, EngineCommand, EngineSender};
use crate::display::driver::DisplayDriver;
use crate::display::messages::{DisplayEvent, WindowId};
use crate::error::RuntimeError;
use crate::input::{KeySymbol, Modifiers};
use crate::platform::waker::{EventLoopWaker, X11Waker};
use log::{info, trace};
use pixelflow_graphics::render::color::Bgra8;

// Type alias for backward compatibility
type Bgra = Bgra8;
use pixelflow_graphics::render::Frame;
use std::ffi::{CStr, CString};
use std::mem;
use std::os::raw::{c_char, c_int};
use std::ptr;
use std::sync::mpsc::{sync_channel, Receiver, SyncSender, TrySendError};
use x11::{keysym, xlib};

// Standard X11 cursor font constants (from X11/cursorfont.h)
const XC_ARROW: u32 = 2;
const XC_HAND2: u32 = 60;
const XC_XTERM: u32 = 152;

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

// --- Run State (only original driver has this) ---
struct RunState {
    cmd_rx: Receiver<DriverCommand<Bgra>>,
    engine_tx: EngineSender<Bgra>,
}

// --- Display Driver ---

/// X11 display driver.
///
/// Clone to get a handle for sending commands. Only the original can run().
pub struct X11DisplayDriver {
    cmd_tx: SyncSender<DriverCommand<Bgra>>,
    waker: X11Waker,
    /// Only present on original, None on clones
    run_state: Option<RunState>,
}

impl Clone for X11DisplayDriver {
    fn clone(&self) -> Self {
        Self {
            cmd_tx: self.cmd_tx.clone(),
            waker: self.waker.clone(),
            run_state: None, // Clones can't run
        }
    }
}

impl DisplayDriver for X11DisplayDriver {
    type Pixel = Bgra;

    fn new(engine_tx: EngineSender<Bgra>) -> Result<Self, RuntimeError> {
        let (cmd_tx, cmd_rx) = sync_channel(16);

        Ok(Self {
            cmd_tx,
            waker: X11Waker::new(),
            run_state: Some(RunState { cmd_rx, engine_tx }),
        })
    }

    fn send(&self, cmd: DriverCommand<Bgra>) -> Result<(), RuntimeError> {
        let mut cmd = cmd;
        loop {
            match self.cmd_tx.try_send(cmd) {
                Ok(()) => break,
                Err(TrySendError::Full(returned)) => {
                    // Buffer full - wake event loop to drain, then retry
                    self.waker.wake()?;
                    cmd = returned;
                    std::thread::yield_now();
                }
                Err(TrySendError::Disconnected(_)) => {
                    return Err(RuntimeError::DriverChannelDisconnected);
                }
            }
        }
        // Command queued, wake event loop to process it
        self.waker.wake()?;
        Ok(())
    }

    fn run(&self) -> Result<(), RuntimeError> {
        let run_state = self
            .run_state
            .as_ref()
            .ok_or_else(|| RuntimeError::DriverCloneError)?;

        run_event_loop(&run_state.cmd_rx, &run_state.engine_tx, &self.waker)
    }
}

// --- Event Loop ---

fn run_event_loop(
    cmd_rx: &Receiver<DriverCommand<Bgra>>,
    engine_tx: &EngineSender<Bgra>,
    waker: &X11Waker,
) -> Result<(), RuntimeError> {
    // 1. Read CreateWindow command first
    let (window_id, width, height, title) = match cmd_rx
        .recv()
        .map_err(|_| RuntimeError::DriverChannelDisconnected)?
    {
        DriverCommand::CreateWindow {
            id,
            width,
            height,
            title,
        } => (id, width, height, title),
        other => return Err(RuntimeError::UnexpectedCommand(format!("{:?}", other))),
    };

    info!("X11: Creating window '{}' {}x{}", title, width, height);

    // 2. Create X11 resources
    unsafe {
        if xlib::XInitThreads() == 0 {
            return Err(RuntimeError::XInitThreadsFailed);
        }

        let display = xlib::XOpenDisplay(ptr::null());
        if display.is_null() {
            return Err(RuntimeError::XOpenDisplayFailed);
        }

        let screen = xlib::XDefaultScreen(display);
        let root = xlib::XRootWindow(display, screen);

        let wm_delete_name = CString::new("WM_DELETE_WINDOW").unwrap();
        let wm_delete_window = xlib::XInternAtom(display, wm_delete_name.as_ptr(), xlib::False);

        let atoms = SelectionAtoms::new(display);

        let black = xlib::XBlackPixel(display, screen);
        let white = xlib::XWhitePixel(display, screen);

        let x11_window =
            xlib::XCreateSimpleWindow(display, root, 0, 0, width, height, 0, white, black);

        // Set window title
        if let Ok(c_title) = CString::new(title.as_str()) {
            xlib::XStoreName(display, x11_window, c_title.as_ptr());
        }

        // Initialize waker now that we have display and window
        waker.set_target(display, x11_window);
        let wake_atom = waker.wake_atom().unwrap();

        // Select Input Events
        xlib::XSelectInput(
            display,
            x11_window,
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
        xlib::XSetWMProtocols(
            display,
            x11_window,
            &wm_delete_window as *const _ as *mut _,
            1,
        );

        // Initialize GC
        let gc = xlib::XCreateGC(display, x11_window, 0, ptr::null_mut());
        xlib::XSetForeground(display, gc, white);
        xlib::XSetBackground(display, gc, black);

        // Map Window
        xlib::XMapWindow(display, x11_window);
        xlib::XFlush(display);

        // Initialize XRM database for querying Xft.dpi
        xlib::XrmInitialize();
        let resource_string = xlib::XResourceManagerString(display);
        let xrm_db = if resource_string.is_null() {
            None
        } else {
            Some(xlib::XrmGetStringDatabase(resource_string))
        };

        // 3. Run event loop
        let mut state = X11State {
            window_id,
            display,
            screen,
            x11_window,
            gc,
            wm_delete_window,
            atoms,
            wake_atom,
            xrm_db,
            width_px: width,
            height_px: height,
            scale_factor: 1.0,
            clipboard_data: String::new(),
        };

        // Query scale factor from Xft.dpi using initialized XRM database
        state.scale_factor = state.query_scale_factor();

        info!(
            "X11: Window created {}x{} px, scale {:.2}",
            width, height, state.scale_factor
        );

        // Send WindowCreated event
        let _ = engine_tx.send(EngineCommand::DisplayEvent(DisplayEvent::WindowCreated {
            id: window_id,
            width_px: width,
            height_px: height,
            scale: state.scale_factor,
        }));

        state.event_loop(cmd_rx, engine_tx)
        // X11State::drop() handles cleanup
    }
}

// --- X11 State (only exists during run) ---

struct X11State {
    window_id: WindowId,
    display: *mut xlib::Display,
    screen: c_int,
    x11_window: xlib::Window,
    gc: xlib::GC,
    wm_delete_window: xlib::Atom,
    atoms: SelectionAtoms,
    wake_atom: xlib::Atom,
    xrm_db: Option<xlib::XrmDatabase>,
    width_px: u32,
    height_px: u32,
    scale_factor: f64,
    /// Data we own for the CLIPBOARD selection
    clipboard_data: String,
}

impl X11State {
    /// Query Xft.dpi from X resources and calculate scale factor.
    /// Standard DPI is 96, so scale_factor = dpi / 96.0
    fn query_scale_factor(&self) -> f64 {
        let Some(xrm_db) = self.xrm_db else {
            info!("X11: No XRM database, using scale 1.0");
            return 1.0;
        };

        unsafe {
            let name = CString::new("Xft.dpi").unwrap();
            let class = CString::new("Xft.Dpi").unwrap();
            let mut type_return: *mut c_char = ptr::null_mut();
            let mut value_return: xlib::XrmValue = mem::zeroed();

            let found = xlib::XrmGetResource(
                xrm_db,
                name.as_ptr(),
                class.as_ptr(),
                &mut type_return,
                &mut value_return,
            );

            if found == xlib::True && !value_return.addr.is_null() {
                let dpi_str = CStr::from_ptr(value_return.addr as *const c_char);
                if let Ok(dpi_str) = dpi_str.to_str() {
                    if let Ok(dpi) = dpi_str.parse::<f64>() {
                        let scale = dpi / 96.0;
                        info!("X11: Xft.dpi = {}, scale_factor = {:.2}", dpi, scale);
                        return scale;
                    }
                }
            }

            info!("X11: Xft.dpi not found, using scale 1.0");
            1.0
        }
    }
}

impl X11State {
    fn event_loop(
        &mut self,
        cmd_rx: &Receiver<DriverCommand<Bgra>>,
        engine_tx: &EngineSender<Bgra>,
    ) -> Result<(), RuntimeError> {
        loop {
            // 1. Process all pending commands first
            while let Ok(cmd) = cmd_rx.try_recv() {
                match cmd {
                    DriverCommand::CreateWindow { .. } => {
                        // Already created, ignore
                    }
                    DriverCommand::DestroyWindow { .. } => {
                        info!("X11: DestroyWindow received");
                        return Ok(());
                    }
                    DriverCommand::Shutdown => {
                        info!("X11: Shutdown command received");
                        return Ok(());
                    }
                    DriverCommand::Present { frame, .. } => {
                        let result = self.handle_present(frame);
                        if let Ok(frame) = result {
                            let _ = engine_tx.send(EngineCommand::PresentComplete(frame));
                        }
                    }
                    DriverCommand::SetTitle { title, .. } => {
                        self.handle_set_title(&title);
                    }
                    DriverCommand::SetSize { width, height, .. } => {
                        self.handle_set_size(width, height);
                    }
                    DriverCommand::CopyToClipboard(text) => {
                        self.handle_copy_to_clipboard(&text);
                    }
                    DriverCommand::RequestPaste => {
                        self.handle_request_paste();
                    }
                    DriverCommand::Bell => {
                        self.handle_bell();
                    }
                    DriverCommand::SetCursorIcon { icon } => {
                        self.handle_set_cursor_icon(icon);
                    }
                }
            }

            // 2. Process all pending X11 events
            unsafe {
                while xlib::XPending(self.display) > 0 {
                    let mut event: xlib::XEvent = mem::zeroed();
                    xlib::XNextEvent(self.display, &mut event);

                    if let Some(display_event) = self.convert_xevent(&event) {
                        if matches!(display_event, DisplayEvent::CloseRequested { .. }) {
                            info!("X11: CloseRequested, exiting event loop");
                            return Ok(());
                        }
                        let _ = engine_tx.send(EngineCommand::DisplayEvent(display_event));
                    }
                }
            }

            // 3. Block on next X11 event (waker will post event when cmd arrives)
            unsafe {
                let mut event: xlib::XEvent = mem::zeroed();
                xlib::XNextEvent(self.display, &mut event);

                if let Some(display_event) = self.convert_xevent(&event) {
                    if matches!(display_event, DisplayEvent::CloseRequested { .. }) {
                        info!("X11: CloseRequested, exiting event loop");
                        return Ok(());
                    }
                    let _ = engine_tx.send(EngineCommand::DisplayEvent(display_event));
                }
            }
        }
    }

    fn convert_xevent(&mut self, event: &xlib::XEvent) -> Option<DisplayEvent> {
        let id = self.window_id;
        unsafe {
            match event.type_ {
                xlib::ClientMessage => {
                    let client = event.client_message;
                    // Check for window manager close request
                    let data0 = client.data.as_longs()[0];
                    if data0 as xlib::Atom == self.wm_delete_window {
                        return Some(DisplayEvent::CloseRequested { id });
                    }
                    // Wake events are ignored - they just break us out of XNextEvent
                    if client.message_type == self.wake_atom {
                        trace!("X11: Wake event received");
                        return None;
                    }
                    trace!("X11: Unknown ClientMessage type={}", client.message_type);
                    None
                }
                xlib::SelectionRequest => {
                    // Respond to selection requests directly - we have the data locally
                    self.handle_selection_request(&event.selection_request);
                    None
                }
                xlib::SelectionNotify => self.handle_selection_notify(&event.selection),
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
                        id,
                        symbol,
                        modifiers,
                        text,
                    })
                }
                xlib::ConfigureNotify => {
                    let conf = event.configure;
                    self.width_px = conf.width as u32;
                    self.height_px = conf.height as u32;
                    Some(DisplayEvent::Resized {
                        id,
                        width_px: self.width_px,
                        height_px: self.height_px,
                    })
                }
                xlib::FocusIn => Some(DisplayEvent::FocusGained { id }),
                xlib::FocusOut => Some(DisplayEvent::FocusLost { id }),
                xlib::ButtonPress => {
                    let e = event.button;
                    let modifiers = self.extract_modifiers(e.state);
                    // X11 uses buttons 4/5 for scroll wheel
                    match e.button {
                        4 => Some(DisplayEvent::MouseScroll {
                            id,
                            dx: 0.0,
                            dy: 1.0, // Scroll up
                            x: e.x,
                            y: e.y,
                            modifiers,
                        }),
                        5 => Some(DisplayEvent::MouseScroll {
                            id,
                            dx: 0.0,
                            dy: -1.0, // Scroll down
                            x: e.x,
                            y: e.y,
                            modifiers,
                        }),
                        6 => Some(DisplayEvent::MouseScroll {
                            id,
                            dx: -1.0, // Scroll left
                            dy: 0.0,
                            x: e.x,
                            y: e.y,
                            modifiers,
                        }),
                        7 => Some(DisplayEvent::MouseScroll {
                            id,
                            dx: 1.0, // Scroll right
                            dy: 0.0,
                            x: e.x,
                            y: e.y,
                            modifiers,
                        }),
                        _ => Some(DisplayEvent::MouseButtonPress {
                            id,
                            button: e.button as u8,
                            x: e.x,
                            y: e.y,
                            modifiers,
                        }),
                    }
                }
                xlib::ButtonRelease => {
                    let e = event.button;
                    // Skip release events for scroll buttons
                    if e.button >= 4 && e.button <= 7 {
                        return None;
                    }
                    let modifiers = self.extract_modifiers(e.state);
                    Some(DisplayEvent::MouseButtonRelease {
                        id,
                        button: e.button as u8,
                        x: e.x,
                        y: e.y,
                        modifiers,
                    })
                }
                xlib::MotionNotify => {
                    let e = event.motion;
                    let modifiers = self.extract_modifiers(e.state);
                    Some(DisplayEvent::MouseMove {
                        id,
                        x: e.x,
                        y: e.y,
                        modifiers,
                    })
                }
                _ => None,
            }
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

    // --- Command handlers ---

    fn handle_present(&mut self, frame: Frame<Bgra>) -> Result<Frame<Bgra>, RuntimeError> {
        unsafe {
            let depth = xlib::XDefaultDepth(self.display, self.screen);
            let visual = xlib::XDefaultVisual(self.display, self.screen);
            let data_ptr = frame.data.as_ptr() as *mut i8;

            let image = xlib::XCreateImage(
                self.display,
                visual,
                depth as u32,
                xlib::ZPixmap,
                0,
                data_ptr,
                frame.width as u32,
                frame.height as u32,
                32,
                0,
            );

            if image.is_null() {
                return Err(RuntimeError::XCreateImageFailed);
            }

            xlib::XPutImage(
                self.display,
                self.x11_window,
                self.gc,
                image,
                0,
                0,
                0,
                0,
                frame.width as u32,
                frame.height as u32,
            );

            (*image).data = ptr::null_mut();
            xlib::XDestroyImage(image);
            xlib::XFlush(self.display);
        }

        Ok(frame)
    }

    fn handle_set_size(&mut self, width: u32, height: u32) {
        unsafe {
            xlib::XResizeWindow(self.display, self.x11_window, width, height);
            xlib::XFlush(self.display);
        }
    }

    fn handle_set_cursor_icon(&mut self, icon: crate::api::public::CursorIcon) {
        unsafe {
            let cursor_shape = match icon {
                crate::api::public::CursorIcon::Default => XC_ARROW,
                crate::api::public::CursorIcon::Pointer => XC_HAND2,
                crate::api::public::CursorIcon::Text => XC_XTERM,
            };

            let cursor = xlib::XCreateFontCursor(self.display, cursor_shape);
            xlib::XDefineCursor(self.display, self.x11_window, cursor);
            xlib::XFreeCursor(self.display, cursor);
            xlib::XFlush(self.display);
        }
    }

    fn handle_set_title(&mut self, title: &str) {
        unsafe {
            if let Ok(c_title) = CString::new(title) {
                xlib::XStoreName(self.display, self.x11_window, c_title.as_ptr());
                xlib::XFlush(self.display);
            }
        }
    }

    fn handle_bell(&mut self) {
        unsafe {
            xlib::XBell(self.display, 0);
            xlib::XFlush(self.display);
        }
    }

    fn handle_copy_to_clipboard(&mut self, text: &str) {
        // Store the text so we can respond to SelectionRequest events
        self.clipboard_data = text.to_string();

        unsafe {
            xlib::XSetSelectionOwner(
                self.display,
                self.atoms.clipboard,
                self.x11_window,
                xlib::CurrentTime,
            );
            xlib::XFlush(self.display);
        }
    }

    /// Respond to a SelectionRequest from another app wanting to paste our data.
    fn handle_selection_request(&mut self, event: &xlib::XSelectionRequestEvent) {
        unsafe {
            let mut response: xlib::XSelectionEvent = mem::zeroed();
            response.type_ = xlib::SelectionNotify;
            response.requestor = event.requestor;
            response.selection = event.selection;
            response.target = event.target;
            response.time = event.time;

            // Check what format was requested
            if event.target == self.atoms.targets {
                // TARGETS request: tell them what formats we support
                let targets = [
                    self.atoms.targets,
                    self.atoms.utf8_string,
                    self.atoms.text,
                    self.atoms.xa_string,
                ];
                xlib::XChangeProperty(
                    self.display,
                    event.requestor,
                    event.property,
                    xlib::XA_ATOM,
                    32,
                    xlib::PropModeReplace,
                    targets.as_ptr() as *const u8,
                    targets.len() as i32,
                );
                response.property = event.property;
            } else if event.target == self.atoms.utf8_string
                || event.target == self.atoms.text
                || event.target == self.atoms.xa_string
            {
                // Text data request: provide our clipboard data
                let data = self.clipboard_data.as_bytes();
                let target_type = if event.target == self.atoms.utf8_string {
                    self.atoms.utf8_string
                } else {
                    self.atoms.xa_string
                };

                xlib::XChangeProperty(
                    self.display,
                    event.requestor,
                    event.property,
                    target_type,
                    8,
                    xlib::PropModeReplace,
                    data.as_ptr(),
                    data.len() as i32,
                );
                response.property = event.property;
            } else {
                // Unknown format - reject by setting property to None
                response.property = 0;
                trace!("X11: Rejecting selection request for unknown target");
            }

            // Send SelectionNotify back to requestor
            xlib::XSendEvent(
                self.display,
                event.requestor,
                xlib::False,
                0,
                &mut response as *mut _ as *mut xlib::XEvent,
            );
            xlib::XFlush(self.display);
        }
    }

    fn handle_request_paste(&mut self) {
        unsafe {
            xlib::XConvertSelection(
                self.display,
                self.atoms.clipboard,
                self.atoms.utf8_string,
                self.atoms.clipboard,
                self.x11_window,
                xlib::CurrentTime,
            );
            xlib::XFlush(self.display);
        }
    }

    unsafe fn handle_selection_notify(
        &self,
        event: &xlib::XSelectionEvent,
    ) -> Option<DisplayEvent> {
        if event.property == 0 {
            return None;
        }

        let mut type_ret = 0;
        let mut format_ret = 0;
        let mut nitems = 0;
        let mut bytes_after = 0;
        let mut prop_ret: *mut u8 = ptr::null_mut();

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
}

impl Drop for X11State {
    fn drop(&mut self) {
        unsafe {
            if let Some(xrm_db) = self.xrm_db {
                xlib::XrmDestroyDatabase(xrm_db);
            }
            xlib::XFreeGC(self.display, self.gc);
            xlib::XDestroyWindow(self.display, self.x11_window);
            xlib::XCloseDisplay(self.display);
        }
        info!("X11State dropped - resources cleaned up");
    }
}
