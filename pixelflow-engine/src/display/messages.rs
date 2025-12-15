use crate::api::private::WindowId;
use crate::api::public::WindowDescriptor;
use crate::input::{KeySymbol, Modifiers};
use pixelflow_core::Pixel;
use pixelflow_render::Frame;

/// High-throughput data messages (Lowest Priority)
#[derive(Debug)]
pub enum DisplayData<P: Pixel> {
    Present { id: WindowId, frame: Frame<P> },
}

/// Time-critical control messages (Highest Priority)
#[derive(Debug, Clone)]
pub enum DisplayControl {
    SetTitle {
        id: WindowId,
        title: String,
    },
    SetSize {
        id: WindowId,
        width: u32,
        height: u32,
    },
    SetCursor {
        id: WindowId,
        cursor: crate::api::public::CursorIcon,
    },
    SetVisible {
        id: WindowId,
        visible: bool,
    },
    RequestRedraw {
        id: WindowId,
    },
    Bell,
    Copy {
        text: String,
    },
    RequestPaste,
    Shutdown,
}

/// Lifecycle and management messages (Medium Priority)
#[derive(Debug, Clone)]
pub enum DisplayMgmt {
    Create {
        id: WindowId,
        settings: WindowDescriptor,
    },
    Destroy {
        id: WindowId,
    },
}

/// Events from the Display Driver.
#[derive(Debug, Clone)]
pub enum DisplayEvent {
    WindowCreated {
        id: WindowId,
        width_px: u32,
        height_px: u32,
        scale: f64,
    },
    Resized {
        id: WindowId,
        width_px: u32,
        height_px: u32,
    },
    WindowDestroyed {
        id: WindowId,
    },
    CloseRequested {
        id: WindowId,
    },
    ScaleChanged {
        id: WindowId,
        scale: f64,
    },
    Key {
        id: WindowId,
        symbol: KeySymbol,
        modifiers: Modifiers,
        text: Option<String>,
    },
    MouseButtonPress {
        id: WindowId,
        button: u8,
        x: i32,
        y: i32,
        modifiers: Modifiers,
    },
    MouseButtonRelease {
        id: WindowId,
        button: u8,
        x: i32,
        y: i32,
        modifiers: Modifiers,
    },
    MouseMove {
        id: WindowId,
        x: i32,
        y: i32,
        modifiers: Modifiers,
    },
    MouseScroll {
        id: WindowId,
        dx: f32,
        dy: f32,
        x: i32,
        y: i32,
        modifiers: Modifiers,
    },
    FocusGained {
        id: WindowId,
    },
    FocusLost {
        id: WindowId,
    },
    PasteData {
        text: String,
    },
    ClipboardDataRequested,
}
