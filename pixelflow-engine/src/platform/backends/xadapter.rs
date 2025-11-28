// src/backends/x11/xadapter.rs

//! Defines the `XAdapter` trait and its mock implementation for testing the X11 backend.
//!
//! The `XAdapter` trait provides an abstraction over direct Xlib, Xft, XIM, and Xcursor calls,
//! allowing the `XDriver` to be tested without a live X server.

use std::collections::{HashMap, VecDeque};
use std::ffi::{CStr, CString};
use std::fmt;
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Result};
use x11::{keysym, xcursor, xft, xlib};

// Re-export common X11 types for convenience within the adapter and its users.
// These are opaque types or type aliases from the x11-rs crates.
pub use x11::xft::{XftColor, XftDraw, XftFont};
pub use x11::xlib::{
    Atom, Bool, Display, KeyCode, KeySym, Screen, Visual, Window, XClassHint, XEvent, XPoint,
    XRectangle, XSetWindowAttributes, XSizeHints, XStandardColormap, XTextProperty, XWMHints, XIC,
    XIM,
};

/// Represents an error that can occur within the XAdapter system.
#[derive(Debug, thiserror::Error)]
pub enum XAdapterError {
    #[error("X11 operation failed: {0}")]
    OperationFailed(String),
    #[error("Simulated X11 error: {0}")]
    SimulatedError(String),
    #[error("Feature not implemented in mock: {0}")]
    MockNotImplemented(String),
    #[error("Invalid parameters for mock operation: {0}")]
    MockInvalidParams(String),
}

/// Opaque handle for a mocked display connection, primarily for type safety in tests.
/// In a live adapter, this would correspond to `*mut xlib::Display`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MockDisplay(usize); // Using usize to represent a pointer-like ID

/// Represents a generic XID for mocked resources like Windows, Pixmaps, GCs, etc.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct MockXid(u64);

impl From<xlib::Window> for MockXid {
    fn from(id: xlib::Window) -> Self {
        MockXid(id)
    }
}
impl From<MockXid> for xlib::Window {
    fn from(id: MockXid) -> Self {
        id.0
    }
}
// Similar From/Into could be added for other XID types like Pixmap, Gc, Cursor if needed
// to improve type safety in mock expectations, though the trait will use raw xlib types.

/// A simplified representation of an X event for the mock adapter.
#[derive(Debug, Clone)]
pub enum MockXConcreteEvent {
    KeyPress {
        window: Window,
        keycode: KeyCode,
        state: u32, // Modifiers
        time: u64,
    },
    KeyRelease {
        window: Window,
        keycode: KeyCode,
        state: u32,
        time: u64,
    },
    ButtonPress {
        window: Window,
        button: u32,
        state: u32,
        x_root: i32,
        y_root: i32,
        time: u64,
    },
    ButtonRelease {
        window: Window,
        button: u32,
        state: u32,
        x_root: i32,
        y_root: i32,
        time: u64,
    },
    MotionNotify {
        window: Window,
        state: u32,
        x_root: i32,
        y_root: i32,
        time: u64,
    },
    Expose {
        window: Window,
        x: i32,
        y: i32,
        width: i32,
        height: i32,
        count: i32,
    },
    ConfigureNotify {
        window: Window,
        x: i32,
        y: i32,
        width: i32,
        height: i32,
        border_width: i32,
        above: Window,
        override_redirect: Bool,
    },
    ClientMessage {
        window: Window,
        message_type: Atom,
        format: i32,
        data: [libc::c_long; 5], // ClientMessageData
    },
    FocusIn {
        window: Window,
        mode: i32,
        detail: i32,
    },
    FocusOut {
        window: Window,
        mode: i32,
        detail: i32,
    },
    PropertyNotify {
        window: Window,
        atom: Atom,
        time: u64,
        state: i32, // PropertyNewValue or PropertyDelete
    },
    SelectionClear {
        window: Window,
        selection: Atom,
        time: u64,
    },
    SelectionNotify {
        requestor: Window,
        selection: Atom,
        target: Atom,
        property: Atom, // Atom or None
        time: u64,
    },
    SelectionRequest {
        owner: Window,
        requestor: Window,
        selection: Atom,
        target: Atom,
        property: Atom, // Atom or None
        time: u64,
    },
    // Add other event types as needed
}

/// Records a call made to the `XAdapter` for later inspection in tests.
/// Parameters are stored in a way that facilitates assertion.
#[derive(Debug, Clone)]
pub enum RecordedCall {
    // Display and Connection
    OpenDisplay {
        name: Option<String>,
    },
    CloseDisplay {
        dpy: *mut Display,
    },
    Flush {
        dpy: *mut Display,
    },
    Sync {
        dpy: *mut Display,
        discard: Bool,
    },
    Pending {
        dpy: *mut Display,
    },
    NextEvent {
        dpy: *mut Display,
    },
    ConnectionNumber {
        dpy: *mut Display,
    },
    DefaultScreen {
        dpy: *mut Display,
    },
    RootWindow {
        dpy: *mut Display,
        screen_number: i32,
    },
    DefaultDepth {
        dpy: *mut Display,
        screen_number: i32,
    },
    DefaultVisual {
        dpy: *mut Display,
        screen_number: i32,
    },
    DefaultColormap {
        dpy: *mut Display,
        screen_number: i32,
    },
    InternAtom {
        dpy: *mut Display,
        name: String,
        only_if_exists: Bool,
    },
    GetAtomName {
        dpy: *mut Display,
        atom: Atom,
    },

    // Window Management
    CreateSimpleWindow {
        dpy: *mut Display,
        parent: Window,
        x: i32,
        y: i32,
        width: u32,
        height: u32,
        border_width: u32,
        border: u64,     // unsigned long
        background: u64, // unsigned long
    },
    DestroyWindow {
        dpy: *mut Display,
        w: Window,
    },
    MapWindow {
        dpy: *mut Display,
        w: Window,
    },
    SelectInput {
        dpy: *mut Display,
        w: Window,
        event_mask: libc::c_long,
    },
    StoreName {
        dpy: *mut Display,
        w: Window,
        name: String,
    },
    ChangeProperty {
        dpy: *mut Display,
        w: Window,
        property: Atom,
        target_type: Atom, // Note: xlib uses 'type' which is a keyword
        format: i32,
        mode: i32,
        data: Vec<u8>, // Store a copy of the data
        nelements: i32,
    },
    SetWMProtocols {
        dpy: *mut Display,
        w: Window,
        protocols: Vec<Atom>,
        count: i32,
    },

    // Hints
    AllocSizeHints, // No params for XAllocSizeHints itself
    SetWMNormalHints {
        dpy: *mut Display,
        w: Window,
        hints: Box<XSizeHints>,
    }, // Box to own it

    // Xft Font & Drawing
    XftFontOpenName {
        dpy: *mut Display,
        screen: i32,
        name: String,
    },
    XftFontClose {
        dpy: *mut Display,
        font: *mut XftFont,
    },
    XftDrawCreate {
        dpy: *mut Display,
        drawable: Window,
        visual: *mut Visual,
        colormap: Atom,
    },
    XftDrawDestroy {
        draw: *mut XftDraw,
    },
    XftDrawStringUtf8 {
        draw: *mut XftDraw,
        color: Box<XftColor>, // Box to own it
        font: *mut XftFont,
        x: i32,
        y: i32,
        text: String,
        len: i32,
    },
    XftDrawRect {
        draw: *mut XftDraw,
        color: Box<XftColor>,
        x: i32,
        y: i32,
        width: u32,
        height: u32,
    },
    XftColorAllocName {
        dpy: *mut Display,
        visual: *mut Visual,
        colormap: Atom,
        name: String,
    },
    XftColorFree {
        dpy: *mut Display,
        visual: *mut Visual,
        colormap: Atom,
        color: Box<XftColor>,
    },

    // XIM
    OpenIM {
        dpy: *mut Display, /* ... other args ... */
    },
    CloseIM {
        im: XIM,
    },
    CreateIC {
        im: XIM, /* ... varargs ... */
    },
    DestroyIC {
        ic: XIC,
    },
    SetICFocus {
        ic: XIC,
    },
    UnsetICFocus {
        ic: XIC,
    },
    LookupString {
        ic: XIC,
        event: Box<XEvent>,
        buffer: Vec<u8>,
        nbytes: i32,
    },

    // Cursor
    CursorLibraryLoadCursor {
        dpy: *mut Display,
        name: String,
    },
    DefineCursor {
        dpy: *mut Display,
        w: Window,
        cursor: xlib::Cursor,
    },
    FreeCursor {
        dpy: *mut Display,
        cursor: xlib::Cursor,
    },

    // Selections (Clipboard/DND)
    SetSelectionOwner {
        dpy: *mut Display,
        selection: Atom,
        owner: Window,
        time: u64,
    },
    GetSelectionOwner {
        dpy: *mut Display,
        selection: Atom,
    },
    ConvertSelection {
        dpy: *mut Display,
        selection: Atom,
        target: Atom,
        property: Atom,
        requestor: Window,
        time: u64,
    },
    // Add more calls as needed
}

/// The `XAdapter` trait abstracts X11 library calls for testability.
///
/// Methods generally mirror their Xlib/Xft/XIM counterparts, using raw X11 types
/// in their signatures. This simplifies the `LiveXAdapter` and integration with
/// existing `XDriver` code. The `MockXAdapter` will work with these types, often
/// treating pointers and IDs as opaque handles or numerical values.
///
/// All methods that can logically fail should return `Result<T, anyhow::Error>`.
/// `Display` pointers are typically the first argument, akin to C Xlib calls.
#[allow(clippy::too_many_arguments, clippy::type_complexity)] // Xlib functions often have many args
pub trait XAdapter: Send + Sync + fmt::Debug + 'static {
    // Display and Connection
    fn open_display(&self, name: Option<&CStr>) -> Result<*mut Display>;
    fn close_display(&self, dpy: *mut Display) -> Result<()>;
    fn flush(&self, dpy: *mut Display) -> Result<()>;
    fn sync(&self, dpy: *mut Display, discard: Bool) -> Result<()>;
    fn pending(&self, dpy: *mut Display) -> Result<i32>; // Returns number of events
    fn next_event(&self, dpy: *mut Display, event_return: *mut XEvent) -> Result<()>;
    fn connection_number(&self, dpy: *mut Display) -> Result<i32>;
    fn default_screen(&self, dpy: *mut Display) -> Result<i32>;
    fn root_window(&self, dpy: *mut Display, screen_number: i32) -> Result<Window>;
    fn default_depth(&self, dpy: *mut Display, screen_number: i32) -> Result<i32>;
    fn default_visual(&self, dpy: *mut Display, screen_number: i32) -> Result<*mut Visual>;
    fn default_colormap(&self, dpy: *mut Display, screen_number: i32) -> Result<Atom>; // Colormap is Atom
    fn intern_atom(&self, dpy: *mut Display, name: &CStr, only_if_exists: Bool) -> Result<Atom>;
    fn get_atom_name(&self, dpy: *mut Display, atom: Atom) -> Result<Option<String>>; // Returns name or None

    // Window Management
    fn create_simple_window(
        &self,
        dpy: *mut Display,
        parent: Window,
        x: i32,
        y: i32,
        width: u32,
        height: u32,
        border_width: u32,
        border: u64,     // unsigned long
        background: u64, // unsigned long
    ) -> Result<Window>;
    fn destroy_window(&self, dpy: *mut Display, w: Window) -> Result<()>;
    fn map_window(&self, dpy: *mut Display, w: Window) -> Result<()>;
    fn select_input(&self, dpy: *mut Display, w: Window, event_mask: libc::c_long) -> Result<()>;
    fn store_name(&self, dpy: *mut Display, w: Window, name: &CStr) -> Result<()>;
    fn change_property(
        &self,
        dpy: *mut Display,
        w: Window,
        property: Atom,
        property_type: Atom, // Renamed from 'type'
        format: i32,
        mode: i32,
        data: *const u8,
        nelements: i32,
    ) -> Result<()>;
    fn set_wm_protocols(
        &self,
        dpy: *mut Display,
        w: Window,
        protocols: *mut Atom,
        count: i32,
    ) -> Result<Bool>; // Returns Status

    // Hints
    fn alloc_size_hints(&self) -> Result<*mut XSizeHints>;
    fn set_wm_normal_hints(
        &self,
        dpy: *mut Display,
        w: Window,
        hints: *mut XSizeHints,
    ) -> Result<()>;
    // Add XAllocWMHints, XSetWMHints, XAllocClassHint, XSetClassHint similarly

    // Xft Font & Drawing
    fn xft_font_open_name(
        &self,
        dpy: *mut Display,
        screen: i32,
        name: &CStr,
    ) -> Result<*mut XftFont>;
    fn xft_font_close(&self, dpy: *mut Display, font: *mut XftFont) -> Result<()>;
    fn xft_draw_create(
        &self,
        dpy: *mut Display,
        drawable: Window, // Actually xlib::Drawable, which is Window or Pixmap
        visual: *mut Visual,
        colormap: Atom, // Colormap, not u64
    ) -> Result<*mut XftDraw>;
    fn xft_draw_destroy(&self, draw: *mut XftDraw) -> Result<()>;
    fn xft_draw_string_utf8(
        &self,
        draw: *mut XftDraw,
        color: *const XftColor,
        font: *mut XftFont,
        x: i32,
        y: i32,
        text: *const u8, // *const libc::c_char in C, but u8 is fine for UTF-8 bytes
        len: i32,
    ) -> Result<()>;
    fn xft_draw_rect(
        &self,
        draw: *mut XftDraw,
        color: *const XftColor,
        x: i32,
        y: i32,
        width: u32,
        height: u32,
    ) -> Result<()>;

    fn xft_color_alloc_name(
        &self,
        dpy: *mut Display,
        visual: *mut Visual,
        colormap: Atom, // Colormap
        name: &CStr,
        result: *mut XftColor,
    ) -> Result<Bool>; // Returns Bool (0 or 1 for success/failure)

    fn xft_color_free(
        &self,
        dpy: *mut Display,
        visual: *mut Visual,
        colormap: Atom, // Colormap
        color: *mut XftColor,
    ) -> Result<()>;

    // XIM - Abridged example
    fn open_im(
        &self,
        dpy: *mut Display,
        rdb: *mut xlib::XrmDatabase,
        res_name: *mut libc::c_char,
        res_class: *mut libc::c_char,
    ) -> Result<XIM>; // Can return null
    fn close_im(&self, im: XIM) -> Result<()>; // Returns Status
    fn create_ic(&self, im: XIM /*, ... va_list ... */) -> Result<XIC>; // Variadic, tricky to mock perfectly.
                                                                        // For mocking, might simplify signature or expect specific va_list patterns.
    fn destroy_ic(&self, ic: XIC) -> Result<()>;
    fn set_ic_focus(&self, ic: XIC) -> Result<()>;
    fn unset_ic_focus(&self, ic: XIC) -> Result<()>;
    fn lookup_string(
        &self,
        ic: XIC,
        event: *mut XEvent,
        buffer_return: *mut libc::c_char,
        bytes_buffer: i32,
        keysym_return: *mut KeySym,
        status_return: *mut i32, // Status for XLookupChars, XLookupKeySym, XLookupBoth
    ) -> Result<i32>; // Returns number of bytes

    // Cursor
    fn cursor_library_load_cursor(&self, dpy: *mut Display, name: &CStr) -> Result<xlib::Cursor>; // Can be 0
    fn define_cursor(&self, dpy: *mut Display, w: Window, cursor: xlib::Cursor) -> Result<()>;
    fn free_cursor(&self, dpy: *mut Display, cursor: xlib::Cursor) -> Result<()>;

    // Selections (Clipboard/DND)
    fn set_selection_owner(
        &self,
        dpy: *mut Display,
        selection: Atom,
        owner: Window,
        time: u64, // Time
    ) -> Result<()>;
    fn get_selection_owner(&self, dpy: *mut Display, selection: Atom) -> Result<Window>; // Returns Window or None
    fn convert_selection(
        &self,
        dpy: *mut Display,
        selection: Atom,
        target: Atom,
        property: Atom, // Atom or None
        requestor: Window,
        time: u64, // Time
    ) -> Result<()>;
    // Add more methods as needed, mirroring Xlib functions used in x11.rs
}

/// Mock implementation of the `XAdapter` trait for testing.
#[derive(Debug, Clone)]
pub struct MockXAdapter {
    calls: Arc<Mutex<Vec<RecordedCall>>>,
    event_queue: Arc<Mutex<VecDeque<XEvent>>>, // Stores raw XEvent for simplicity
    // Mocked server state:
    next_xid: Arc<Mutex<u64>>,
    atoms: Arc<Mutex<HashMap<String, Atom>>>,
    next_atom_id: Arc<Mutex<Atom>>,
    // Add other mock state as needed, e.g., window properties, font details
    // For functions that return pointers, the mock might need to manage some dummy memory
    // or return dangling pointers if the content isn't read.
    // For simplicity, many "created" resources might just return pre-defined mock handles.
}

impl MockXAdapter {
    /// Creates a new `MockXAdapter`.
    pub fn new() -> Self {
        Self {
            calls: Arc::new(Mutex::new(Vec::new())),
            event_queue: Arc::new(Mutex::new(VecDeque::new())),
            next_xid: Arc::new(Mutex::new(1000)), // Start XIDs from a high number
            atoms: Arc::new(Mutex::new(HashMap::new())),
            next_atom_id: Arc::new(Mutex::new(100)), // Start custom atoms from a high number
        }
    }

    fn record_call(&self, call: RecordedCall) {
        self.calls.lock().unwrap().push(call);
    }

    /// Retrieves a copy of the recorded calls.
    pub fn get_calls(&self) -> Vec<RecordedCall> {
        self.calls.lock().unwrap().clone()
    }

    /// Clears the log of recorded calls.
    pub fn clear_calls(&self) {
        self.calls.lock().unwrap().clear();
    }

    /// Pushes an XEvent onto the mock event queue.
    /// The event should be a fully formed `xlib::XEvent`.
    pub fn push_event(&self, event: XEvent) {
        self.event_queue.lock().unwrap().push_back(event);
    }

    fn new_mock_xid(&self) -> u64 {
        let mut id = self.next_xid.lock().unwrap();
        *id += 1;
        *id
    }

    fn get_or_create_atom(&self, name_str: &str, only_if_exists: bool) -> Atom {
        let mut atoms = self.atoms.lock().unwrap();
        if let Some(atom) = atoms.get(name_str) {
            return *atom;
        }
        if only_if_exists {
            return xlib::None as Atom;
        }
        let mut next_atom = self.next_atom_id.lock().unwrap();
        let new_atom_id = *next_atom;
        *next_atom += 1;
        atoms.insert(name_str.to_string(), new_atom_id);
        new_atom_id
    }

    // Helper to simulate XftColor allocation for mocking.
    // In a real scenario, XftColor contains pixel values, etc.
    // For mock, we can just use the name as part of its identity if needed, or a simple u32.
    fn mock_xft_color_from_name(&self, _name: &CStr) -> XftColor {
        // For simplicity, return a zeroed XftColor. Tests might need to configure specific
        // return values if the XftColor content is inspected by XDriver.
        unsafe { std::mem::zeroed() }
    }
}

impl Default for MockXAdapter {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(unused_variables, clippy::fn_to_numeric_cast)] // For mock implementations
impl XAdapter for MockXAdapter {
    // --- Display and Connection ---
    fn open_display(&self, name: Option<&CStr>) -> Result<*mut Display> {
        let name_str = name.map(|s| s.to_string_lossy().into_owned());
        self.record_call(RecordedCall::OpenDisplay { name: name_str });
        // Return a non-null, unique-ish pointer for the mock display.
        // The actual value doesn't matter as long as it's consistent for this "connection".
        Ok(self.new_mock_xid() as *mut Display)
    }

    fn close_display(&self, dpy: *mut Display) -> Result<()> {
        self.record_call(RecordedCall::CloseDisplay { dpy });
        Ok(())
    }

    fn flush(&self, dpy: *mut Display) -> Result<()> {
        self.record_call(RecordedCall::Flush { dpy });
        Ok(())
    }

    fn sync(&self, dpy: *mut Display, discard: Bool) -> Result<()> {
        self.record_call(RecordedCall::Sync { dpy, discard });
        Ok(())
    }

    fn pending(&self, dpy: *mut Display) -> Result<i32> {
        self.record_call(RecordedCall::Pending { dpy });
        Ok(self.event_queue.lock().unwrap().len() as i32)
    }

    fn next_event(&self, dpy: *mut Display, event_return: *mut XEvent) -> Result<()> {
        self.record_call(RecordedCall::NextEvent { dpy });
        if let Some(event) = self.event_queue.lock().unwrap().pop_front() {
            unsafe {
                *event_return = event;
            }
            Ok(())
        } else {
            // Block indefinitely if no events? Or return error?
            // Real XNextEvent blocks. For tests, this might hang if not managed.
            // Consider adding a timeout mechanism or specific error for empty queue.
            Err(anyhow!(XAdapterError::MockNotImplemented(
                "MockXAdapter::next_event blocking on empty queue".to_string()
            )))
        }
    }

    fn connection_number(&self, dpy: *mut Display) -> Result<i32> {
        self.record_call(RecordedCall::ConnectionNumber { dpy });
        Ok(3) // Arbitrary fd
    }

    fn default_screen(&self, dpy: *mut Display) -> Result<i32> {
        // self.record_call(...)
        Ok(0)
    }

    fn root_window(&self, dpy: *mut Display, screen_number: i32) -> Result<Window> {
        // self.record_call(...)
        Ok(self.new_mock_xid()) // Mock root window ID
    }

    fn default_depth(&self, dpy: *mut Display, screen_number: i32) -> Result<i32> {
        Ok(24)
    }
    fn default_visual(&self, dpy: *mut Display, screen_number: i32) -> Result<*mut Visual> {
        Ok(self.new_mock_xid() as *mut Visual) // Dummy visual
    }
    fn default_colormap(&self, dpy: *mut Display, screen_number: i32) -> Result<Atom> {
        Ok(self.new_mock_xid()) // Dummy colormap
    }

    fn intern_atom(&self, dpy: *mut Display, name: &CStr, only_if_exists: Bool) -> Result<Atom> {
        let name_str = name.to_string_lossy().into_owned();
        self.record_call(RecordedCall::InternAtom {
            dpy,
            name: name_str.clone(),
            only_if_exists,
        });
        Ok(self.get_or_create_atom(&name_str, only_if_exists != 0))
    }
    fn get_atom_name(&self, dpy: *mut Display, atom: Atom) -> Result<Option<String>> {
        self.record_call(RecordedCall::GetAtomName { dpy, atom });
        let atoms_map = self.atoms.lock().unwrap();
        for (name, &id) in atoms_map.iter() {
            if id == atom {
                return Ok(Some(name.clone()));
            }
        }
        Ok(None)
    }

    // --- Window Management (Subset) ---
    fn create_simple_window(
        &self,
        dpy: *mut Display,
        parent: Window,
        x: i32,
        y: i32,
        width: u32,
        height: u32,
        border_width: u32,
        border: u64,
        background: u64,
    ) -> Result<Window> {
        self.record_call(RecordedCall::CreateSimpleWindow {
            dpy,
            parent,
            x,
            y,
            width,
            height,
            border_width,
            border,
            background,
        });
        Ok(self.new_mock_xid()) // Return a new mock window ID
    }

    fn destroy_window(&self, dpy: *mut Display, w: Window) -> Result<()> {
        self.record_call(RecordedCall::DestroyWindow { dpy, w });
        Ok(())
    }

    fn map_window(&self, dpy: *mut Display, w: Window) -> Result<()> {
        self.record_call(RecordedCall::MapWindow { dpy, w });
        Ok(())
    }

    fn select_input(&self, dpy: *mut Display, w: Window, event_mask: libc::c_long) -> Result<()> {
        self.record_call(RecordedCall::SelectInput { dpy, w, event_mask });
        Ok(())
    }

    fn store_name(&self, dpy: *mut Display, w: Window, name: &CStr) -> Result<()> {
        self.record_call(RecordedCall::StoreName {
            dpy,
            w,
            name: name.to_string_lossy().into_owned(),
        });
        Ok(())
    }

    fn change_property(
        &self,
        dpy: *mut Display,
        w: Window,
        property: Atom,
        property_type: Atom,
        format: i32,
        mode: i32,
        data: *const u8,
        nelements: i32,
    ) -> Result<()> {
        let data_len = match format {
            8 => nelements as usize,
            16 => nelements as usize * 2,
            32 => nelements as usize * 4,
            _ => 0,
        };
        let data_slice = if !data.is_null() && data_len > 0 {
            unsafe { std::slice::from_raw_parts(data, data_len).to_vec() }
        } else {
            Vec::new()
        };
        self.record_call(RecordedCall::ChangeProperty {
            dpy,
            w,
            property,
            target_type: property_type,
            format,
            mode,
            data: data_slice,
            nelements,
        });
        Ok(())
    }

    fn set_wm_protocols(
        &self,
        dpy: *mut Display,
        w: Window,
        protocols: *mut Atom,
        count: i32,
    ) -> Result<Bool> {
        let protocols_slice = if !protocols.is_null() && count > 0 {
            unsafe { std::slice::from_raw_parts(protocols, count as usize).to_vec() }
        } else {
            Vec::new()
        };
        self.record_call(RecordedCall::SetWMProtocols {
            dpy,
            w,
            protocols: protocols_slice,
            count,
        });
        Ok(1) // Success
    }

    // --- Hints (Subset) ---
    fn alloc_size_hints(&self) -> Result<*mut XSizeHints> {
        self.record_call(RecordedCall::AllocSizeHints);
        // This is tricky. The caller will write to this.
        // For a mock, we can return a pointer to a Box'ed, zeroed structure.
        // The Box needs to live long enough for the caller to use it.
        // This usually means the mock or test needs to hold onto the Box.
        // A simpler mock might just return a dangling pointer if XDriver is known not to misuse it,
        // or if tests can verify calls without needing to inspect the written values.
        // For now, let's assume tests will verify the XSetWMNormalHints call instead.
        Ok(Box::into_raw(Box::new(unsafe {
            std::mem::zeroed::<XSizeHints>()
        })))
    }

    fn set_wm_normal_hints(
        &self,
        dpy: *mut Display,
        w: Window,
        hints: *mut XSizeHints,
    ) -> Result<()> {
        // To properly record, we'd need to dereference hints and clone it.
        // Ensure hints is not null before dereferencing.
        let hints_box = if !hints.is_null() {
            unsafe { Box::new(*hints) } // Clone the content
        } else {
            // Or handle as an error/panic depending on expected XDriver behavior
            return Err(anyhow!(XAdapterError::MockInvalidParams(
                "NULL hints pointer".to_string()
            )));
        };
        self.record_call(RecordedCall::SetWMNormalHints {
            dpy,
            w,
            hints: hints_box,
        });
        Ok(())
    }

    // --- Xft Font & Drawing (Subset) ---
    fn xft_font_open_name(
        &self,
        dpy: *mut Display,
        screen: i32,
        name: &CStr,
    ) -> Result<*mut XftFont> {
        let name_str = name.to_string_lossy().into_owned();
        self.record_call(RecordedCall::XftFontOpenName {
            dpy,
            screen,
            name: name_str,
        });
        Ok(self.new_mock_xid() as *mut XftFont) // Dummy font pointer
    }

    fn xft_font_close(&self, dpy: *mut Display, font: *mut XftFont) -> Result<()> {
        self.record_call(RecordedCall::XftFontClose { dpy, font });
        Ok(())
    }

    fn xft_draw_create(
        &self,
        dpy: *mut Display,
        drawable: Window,
        visual: *mut Visual,
        colormap: Atom,
    ) -> Result<*mut XftDraw> {
        self.record_call(RecordedCall::XftDrawCreate {
            dpy,
            drawable,
            visual,
            colormap,
        });
        Ok(self.new_mock_xid() as *mut XftDraw) // Dummy draw pointer
    }

    fn xft_draw_destroy(&self, draw: *mut XftDraw) -> Result<()> {
        self.record_call(RecordedCall::XftDrawDestroy { draw });
        Ok(())
    }

    fn xft_draw_string_utf8(
        &self,
        draw: *mut XftDraw,
        color: *const XftColor,
        font: *mut XftFont,
        x: i32,
        y: i32,
        text: *const u8,
        len: i32,
    ) -> Result<()> {
        let text_str = if !text.is_null() && len > 0 {
            unsafe {
                let slice = std::slice::from_raw_parts(text, len as usize);
                String::from_utf8_lossy(slice).into_owned()
            }
        } else {
            String::new()
        };
        let color_box = if !color.is_null() {
            unsafe { Box::new(*color) }
        } else {
            Box::new(unsafe { std::mem::zeroed() })
        };
        self.record_call(RecordedCall::XftDrawStringUtf8 {
            draw,
            color: color_box,
            font,
            x,
            y,
            text: text_str,
            len,
        });
        Ok(())
    }

    fn xft_draw_rect(
        &self,
        draw: *mut XftDraw,
        color: *const XftColor,
        x: i32,
        y: i32,
        width: u32,
        height: u32,
    ) -> Result<()> {
        let color_box = if !color.is_null() {
            unsafe { Box::new(*color) }
        } else {
            Box::new(unsafe { std::mem::zeroed() })
        };
        self.record_call(RecordedCall::XftDrawRect {
            draw,
            color: color_box,
            x,
            y,
            width,
            height,
        });
        Ok(())
    }

    fn xft_color_alloc_name(
        &self,
        dpy: *mut Display,
        visual: *mut Visual,
        colormap: Atom,
        name: &CStr,
        result: *mut XftColor,
    ) -> Result<Bool> {
        let name_str = name.to_string_lossy().into_owned();
        self.record_call(RecordedCall::XftColorAllocName {
            dpy,
            visual,
            colormap,
            name: name_str.clone(),
        });
        if result.is_null() {
            return Err(anyhow!(XAdapterError::MockInvalidParams(
                "NULL result pointer for XftColorAllocName".to_string()
            )));
        }
        unsafe {
            *result = self.mock_xft_color_from_name(name);
        }
        Ok(1) // Success
    }

    fn xft_color_free(
        &self,
        dpy: *mut Display,
        visual: *mut Visual,
        colormap: Atom,
        color: *mut XftColor,
    ) -> Result<()> {
        let color_box = if !color.is_null() {
            unsafe { Box::new(*color) }
        } else {
            Box::new(unsafe { std::mem::zeroed() })
        };
        self.record_call(RecordedCall::XftColorFree {
            dpy,
            visual,
            colormap,
            color: color_box,
        });
        Ok(())
    }

    // --- XIM (Subset) ---
    fn open_im(
        &self,
        dpy: *mut Display,
        _rdb: *mut xlib::XrmDatabase,
        _res_name: *mut libc::c_char,
        _res_class: *mut libc::c_char,
    ) -> Result<XIM> {
        self.record_call(RecordedCall::OpenIM { dpy });
        Ok(self.new_mock_xid() as XIM) // Dummy XIM
    }
    fn close_im(&self, im: XIM) -> Result<()> {
        self.record_call(RecordedCall::CloseIM { im });
        Ok(())
    }
    fn create_ic(&self, im: XIM /*, ... */) -> Result<XIC> {
        self.record_call(RecordedCall::CreateIC { im });
        Ok(self.new_mock_xid() as XIC) // Dummy XIC
    }
    fn destroy_ic(&self, ic: XIC) -> Result<()> {
        self.record_call(RecordedCall::DestroyIC { ic });
        Ok(())
    }
    fn set_ic_focus(&self, ic: XIC) -> Result<()> {
        self.record_call(RecordedCall::SetICFocus { ic });
        Ok(())
    }
    fn unset_ic_focus(&self, ic: XIC) -> Result<()> {
        self.record_call(RecordedCall::UnsetICFocus { ic });
        Ok(())
    }
    fn lookup_string(
        &self,
        ic: XIC,
        event: *mut XEvent,
        buffer_return: *mut libc::c_char,
        bytes_buffer: i32,
        keysym_return: *mut KeySym,
        status_return: *mut i32,
    ) -> Result<i32> {
        // This is a complex one to mock accurately.
        // A simple mock might return 0 (no chars) and set status to XLookupNone.
        // Tests would need to configure specific lookup results.
        let event_box = if !event.is_null() {
            unsafe { Box::new(*event) }
        } else {
            Box::new(unsafe { std::mem::zeroed() })
        };
        self.record_call(RecordedCall::LookupString {
            ic,
            event: event_box,
            buffer: Vec::new(), // Simplified for recording
            nbytes: bytes_buffer,
        });
        if !status_return.is_null() {
            unsafe {
                *status_return = xlib::XLookupNone;
            }
        }
        if !keysym_return.is_null() && !event.is_null() {
            // Simplistic: try to extract keysym from the event if it's a KeyPress
            unsafe {
                if (*event).type_ == xlib::KeyPress {
                    let key_event = &(*event).key;
                    // This is a simplification; real XLookupString does more complex processing
                    // to get the KeySym. XmbLookupString is also different.
                    // For now, if testing needs specific keysyms from lookup_string,
                    // the mock needs to be enhanced or tests need to provide XKeyEvent with pre-looked-up keysyms.
                    // This mock currently doesn't handle the actual KeySym remapping logic that XLookupString might do.
                    // It's assumed XFilterEvent and XLookupString on XIM provides the primary text and possibly keysym.
                    // If raw XLookupString on KeyEvents is used, it would need a keysym map.
                    // For XIM, the keysym is often less important than the composed string.
                    *keysym_return = key_event.keycode as KeySym; // Very naive, just to have a value
                } else {
                    *keysym_return = keysym::XK_VoidSymbol as KeySym;
                }
            }
        }
        Ok(0) // No characters composed
    }

    // --- Cursor (Subset) ---
    fn cursor_library_load_cursor(&self, dpy: *mut Display, name: &CStr) -> Result<xlib::Cursor> {
        let name_str = name.to_string_lossy().into_owned();
        self.record_call(RecordedCall::CursorLibraryLoadCursor {
            dpy,
            name: name_str,
        });
        Ok(self.new_mock_xid()) // Dummy cursor ID
    }
    fn define_cursor(&self, dpy: *mut Display, w: Window, cursor: xlib::Cursor) -> Result<()> {
        self.record_call(RecordedCall::DefineCursor { dpy, w, cursor });
        Ok(())
    }
    fn free_cursor(&self, dpy: *mut Display, cursor: xlib::Cursor) -> Result<()> {
        self.record_call(RecordedCall::FreeCursor { dpy, cursor });
        Ok(())
    }

    // --- Selections (Subset) ---
    fn set_selection_owner(
        &self,
        dpy: *mut Display,
        selection: Atom,
        owner: Window,
        time: u64,
    ) -> Result<()> {
        self.record_call(RecordedCall::SetSelectionOwner {
            dpy,
            selection,
            owner,
            time,
        });
        Ok(())
    }
    fn get_selection_owner(&self, dpy: *mut Display, selection: Atom) -> Result<Window> {
        self.record_call(RecordedCall::GetSelectionOwner { dpy, selection });
        Ok(xlib::None as Window) // Default: no owner
    }
    fn convert_selection(
        &self,
        dpy: *mut Display,
        selection: Atom,
        target: Atom,
        property: Atom,
        requestor: Window,
        time: u64,
    ) -> Result<()> {
        self.record_call(RecordedCall::ConvertSelection {
            dpy,
            selection,
            target,
            property,
            requestor,
            time,
        });
        Ok(())
    }
}

/// A live implementation of `XAdapter` that calls actual X11 functions.
/// This is what would be used in production.
#[derive(Debug, Clone)]
pub struct LiveXAdapter;

impl LiveXAdapter {
    pub fn new() -> Self {
        // Check if Xlib is thread-safe if this adapter is to be used across threads directly.
        // XInitThreads() should be called by the application early if threaded Xlib access is needed.
        // For core-term, XDriver is typically owned by a single thread.
        Self
    }
}

impl XAdapter for LiveXAdapter {
    // --- Display and Connection ---
    fn open_display(&self, name: Option<&CStr>) -> Result<*mut Display> {
        let name_ptr = name.map_or(std::ptr::null(), |s| s.as_ptr());
        let dpy = unsafe { xlib::XOpenDisplay(name_ptr) };
        if dpy.is_null() {
            Err(anyhow!("XOpenDisplay failed for name: {:?}", name))
        } else {
            Ok(dpy)
        }
    }

    fn close_display(&self, dpy: *mut Display) -> Result<()> {
        let status = unsafe { xlib::XCloseDisplay(dpy) };
        if status == 0 {
            Ok(())
        } else {
            Err(anyhow!("XCloseDisplay error: {}", status))
        }
    }

    fn flush(&self, dpy: *mut Display) -> Result<()> {
        unsafe { xlib::XFlush(dpy) };
        Ok(())
    }
    fn sync(&self, dpy: *mut Display, discard: Bool) -> Result<()> {
        unsafe { xlib::XSync(dpy, discard) };
        Ok(())
    }
    fn pending(&self, dpy: *mut Display) -> Result<i32> {
        Ok(unsafe { xlib::XPending(dpy) })
    }
    fn next_event(&self, dpy: *mut Display, event_return: *mut XEvent) -> Result<()> {
        unsafe { xlib::XNextEvent(dpy, event_return) };
        Ok(())
    }
    fn connection_number(&self, dpy: *mut Display) -> Result<i32> {
        Ok(unsafe { xlib::XConnectionNumber(dpy) })
    }
    fn default_screen(&self, dpy: *mut Display) -> Result<i32> {
        Ok(unsafe { xlib::XDefaultScreen(dpy) })
    }
    fn root_window(&self, dpy: *mut Display, screen_number: i32) -> Result<Window> {
        Ok(unsafe { xlib::XRootWindow(dpy, screen_number) })
    }
    fn default_depth(&self, dpy: *mut Display, screen_number: i32) -> Result<i32> {
        Ok(unsafe { xlib::XDefaultDepth(dpy, screen_number) })
    }
    fn default_visual(&self, dpy: *mut Display, screen_number: i32) -> Result<*mut Visual> {
        Ok(unsafe { xlib::XDefaultVisual(dpy, screen_number) })
    }
    fn default_colormap(&self, dpy: *mut Display, screen_number: i32) -> Result<Atom> {
        Ok(unsafe { xlib::XDefaultColormap(dpy, screen_number) })
    }
    fn intern_atom(&self, dpy: *mut Display, name: &CStr, only_if_exists: Bool) -> Result<Atom> {
        Ok(unsafe { xlib::XInternAtom(dpy, name.as_ptr(), only_if_exists) })
    }
    fn get_atom_name(&self, dpy: *mut Display, atom: Atom) -> Result<Option<String>> {
        let name_ptr = unsafe { xlib::XGetAtomName(dpy, atom) };
        if name_ptr.is_null() {
            Ok(None)
        } else {
            let name = unsafe { CStr::from_ptr(name_ptr).to_string_lossy().into_owned() };
            unsafe { xlib::XFree(name_ptr as *mut libc::c_void) };
            Ok(Some(name))
        }
    }

    // --- Window Management (Subset) ---
    fn create_simple_window(
        &self,
        dpy: *mut Display,
        parent: Window,
        x: i32,
        y: i32,
        width: u32,
        height: u32,
        border_width: u32,
        border: u64,
        background: u64,
    ) -> Result<Window> {
        Ok(unsafe {
            xlib::XCreateSimpleWindow(
                dpy,
                parent,
                x,
                y,
                width,
                height,
                border_width,
                border,
                background,
            )
        })
    }
    fn destroy_window(&self, dpy: *mut Display, w: Window) -> Result<()> {
        unsafe { xlib::XDestroyWindow(dpy, w) };
        Ok(())
    }
    fn map_window(&self, dpy: *mut Display, w: Window) -> Result<()> {
        unsafe { xlib::XMapWindow(dpy, w) };
        Ok(())
    }
    fn select_input(&self, dpy: *mut Display, w: Window, event_mask: libc::c_long) -> Result<()> {
        unsafe { xlib::XSelectInput(dpy, w, event_mask) };
        Ok(())
    }
    fn store_name(&self, dpy: *mut Display, w: Window, name: &CStr) -> Result<()> {
        unsafe { xlib::XStoreName(dpy, w, name.as_ptr() as *mut libc::c_char) };
        Ok(())
    }
    fn change_property(
        &self,
        dpy: *mut Display,
        w: Window,
        property: Atom,
        property_type: Atom,
        format: i32,
        mode: i32,
        data: *const u8,
        nelements: i32,
    ) -> Result<()> {
        unsafe {
            xlib::XChangeProperty(
                dpy,
                w,
                property,
                property_type,
                format,
                mode,
                data,
                nelements,
            );
        }
        Ok(())
    }
    fn set_wm_protocols(
        &self,
        dpy: *mut Display,
        w: Window,
        protocols: *mut Atom,
        count: i32,
    ) -> Result<Bool> {
        Ok(unsafe { xlib::XSetWMProtocols(dpy, w, protocols, count) })
    }

    // --- Hints (Subset) ---
    fn alloc_size_hints(&self) -> Result<*mut XSizeHints> {
        Ok(unsafe { xlib::XAllocSizeHints() })
    }
    fn set_wm_normal_hints(
        &self,
        dpy: *mut Display,
        w: Window,
        hints: *mut XSizeHints,
    ) -> Result<()> {
        unsafe { xlib::XSetWMNormalHints(dpy, w, hints) };
        Ok(())
    }

    // --- Xft Font & Drawing (Subset) ---
    fn xft_font_open_name(
        &self,
        dpy: *mut Display,
        screen: i32,
        name: &CStr,
    ) -> Result<*mut XftFont> {
        let font = unsafe { xft::XftFontOpenName(dpy, screen, name.as_ptr()) };
        if font.is_null() {
            Err(anyhow!(
                "XftFontOpenName failed for: {}",
                name.to_string_lossy()
            ))
        } else {
            Ok(font)
        }
    }
    fn xft_font_close(&self, dpy: *mut Display, font: *mut XftFont) -> Result<()> {
        unsafe { xft::XftFontClose(dpy, font) };
        Ok(())
    }
    fn xft_draw_create(
        &self,
        dpy: *mut Display,
        drawable: Window,
        visual: *mut Visual,
        colormap: Atom,
    ) -> Result<*mut XftDraw> {
        let draw = unsafe { xft::XftDrawCreate(dpy, drawable, visual, colormap) };
        if draw.is_null() {
            Err(anyhow!("XftDrawCreate failed"))
        } else {
            Ok(draw)
        }
    }
    fn xft_draw_destroy(&self, draw: *mut XftDraw) -> Result<()> {
        unsafe { xft::XftDrawDestroy(draw) };
        Ok(())
    }
    fn xft_draw_string_utf8(
        &self,
        draw: *mut XftDraw,
        color: *const XftColor,
        font: *mut XftFont,
        x: i32,
        y: i32,
        text: *const u8,
        len: i32,
    ) -> Result<()> {
        unsafe { xft::XftDrawStringUtf8(draw, color, font, x, y, text, len) };
        Ok(())
    }
    fn xft_draw_rect(
        &self,
        draw: *mut XftDraw,
        color: *const XftColor,
        x: i32,
        y: i32,
        width: u32,
        height: u32,
    ) -> Result<()> {
        unsafe { xft::XftDrawRect(draw, color, x, y, width, height) };
        Ok(())
    }
    fn xft_color_alloc_name(
        &self,
        dpy: *mut Display,
        visual: *mut Visual,
        colormap: Atom,
        name: &CStr,
        result: *mut XftColor,
    ) -> Result<Bool> {
        Ok(unsafe { xft::XftColorAllocName(dpy, visual, colormap, name.as_ptr(), result) })
    }
    fn xft_color_free(
        &self,
        dpy: *mut Display,
        visual: *mut Visual,
        colormap: Atom,
        color: *mut XftColor,
    ) -> Result<()> {
        unsafe { xft::XftColorFree(dpy, visual, colormap, color) };
        Ok(())
    }

    // --- XIM (Subset) ---
    fn open_im(
        &self,
        dpy: *mut Display,
        rdb: *mut xlib::XrmDatabase,
        res_name: *mut libc::c_char,
        res_class: *mut libc::c_char,
    ) -> Result<XIM> {
        let im = unsafe { xlib::XOpenIM(dpy, rdb, res_name, res_class) };
        if im.is_null() {
            Err(anyhow!("XOpenIM failed"))
        } else {
            Ok(im)
        }
    }
    fn close_im(&self, im: XIM) -> Result<()> {
        let status = unsafe { xlib::XCloseIM(im) };
        if status == 0 {
            Ok(())
        } else {
            Err(anyhow!("XCloseIM error: {}", status))
        }
    }
    fn create_ic(&self, im: XIM /*, ... */) -> Result<XIC> {
        // Live version needs to handle varargs. This requires careful FFI or limiting what's supported.
        // Example: if XNInputStyle is always the primary thing set:
        // let ic = unsafe { xlib::XCreateIC(im, xlib::XNInputStyle_DESTROY.as_ptr(), xlib::XIMPreeditNothing | xlib::XIMStatusNothing, std::ptr::null_mut() as *const libc::c_char) };
        // This is highly simplified and likely incorrect for general use.
        // Proper variadic FFI is complex. The actual x11.rs uses a helper for this.
        // For now, this is a placeholder.
        Err(anyhow!(XAdapterError::MockNotImplemented(
            "LiveXAdapter::create_ic with varargs".to_string()
        )))
    }
    fn destroy_ic(&self, ic: XIC) -> Result<()> {
        unsafe { xlib::XDestroyIC(ic) };
        Ok(())
    }
    fn set_ic_focus(&self, ic: XIC) -> Result<()> {
        unsafe { xlib::XSetICFocus(ic) };
        Ok(())
    }
    fn unset_ic_focus(&self, ic: XIC) -> Result<()> {
        unsafe { xlib::XUnsetICFocus(ic) };
        Ok(())
    }
    fn lookup_string(
        &self,
        ic: XIC,
        event: *mut XEvent,
        buffer_return: *mut libc::c_char,
        bytes_buffer: i32,
        keysym_return: *mut KeySym,
        status_return: *mut i32,
    ) -> Result<i32> {
        Ok(unsafe {
            xlib::XmbLookupString(
                ic,
                event as *mut xlib::XKeyPressedEvent,
                buffer_return,
                bytes_buffer,
                keysym_return,
                status_return,
            )
        })
    }

    // --- Cursor (Subset) ---
    fn cursor_library_load_cursor(&self, dpy: *mut Display, name: &CStr) -> Result<xlib::Cursor> {
        Ok(unsafe { xcursor::XcursorLibraryLoadCursor(dpy, name.as_ptr()) })
    }
    fn define_cursor(&self, dpy: *mut Display, w: Window, cursor: xlib::Cursor) -> Result<()> {
        unsafe { xlib::XDefineCursor(dpy, w, cursor) };
        Ok(())
    }
    fn free_cursor(&self, dpy: *mut Display, cursor: xlib::Cursor) -> Result<()> {
        unsafe { xlib::XFreeCursor(dpy, cursor) };
        Ok(())
    }

    // --- Selections (Subset) ---
    fn set_selection_owner(
        &self,
        dpy: *mut Display,
        selection: Atom,
        owner: Window,
        time: u64,
    ) -> Result<()> {
        unsafe { xlib::XSetSelectionOwner(dpy, selection, owner, time) };
        Ok(())
    }
    fn get_selection_owner(&self, dpy: *mut Display, selection: Atom) -> Result<Window> {
        Ok(unsafe { xlib::XGetSelectionOwner(dpy, selection) })
    }
    fn convert_selection(
        &self,
        dpy: *mut Display,
        selection: Atom,
        target: Atom,
        property: Atom,
        requestor: Window,
        time: u64,
    ) -> Result<()> {
        unsafe { xlib::XConvertSelection(dpy, selection, target, property, requestor, time) };
        Ok(())
    }
}
