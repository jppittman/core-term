pub use crate::api::private::WindowId;
use crate::api::public::{CursorIcon, WindowDescriptor};
use crate::input::{KeySymbol, Modifiers};
use pixelflow_graphics::render::Frame;
use pixelflow_graphics::Pixel;

/// Data messages for the display driver (high priority)
#[derive(Debug)]
pub enum DisplayData<P: Pixel> {
    Present { id: WindowId, frame: Frame<P> },
}

/// Control messages for the display driver (low priority)
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
        cursor: CursorIcon,
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

/// Management messages for the display driver (handled separately)
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

/// Events emitted by the display driver
#[derive(Debug, Clone)]
pub enum DisplayEvent {
    WindowCreated {
        id: WindowId,
        width_px: u32,
        height_px: u32,
        scale: f64,
    },
    WindowDestroyed {
        id: WindowId,
    },
    Resized {
        id: WindowId,
        width_px: u32,
        height_px: u32,
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
    CloseRequested {
        id: WindowId,
    },
}
