use crate::api::private::WindowId;
use crate::api::public::{CursorIcon, WindowDescriptor};
use crate::display::messages::DisplayEvent;
use anyhow::Result;
use pixelflow_core::Pixel;
use pixelflow_render::Frame;
use std::time::Duration;

/// Interface for handling events from the application/driver.
/// This matches the `GenericDriver`'s internal logic.
pub trait EventHandler {
    fn handle_event(&mut self, event: DisplayEvent);
}

/// A platform window.
pub trait Window<P: Pixel>: Sized {
    /// Update the window (e.g. check for resize).
    /// Returns any immediate events (optional).
    // Or maybe just pure state management?
    fn set_title(&mut self, title: &str);
    fn set_size(&mut self, width: u32, height: u32);
    fn size(&self) -> (u32, u32);
    fn set_cursor(&mut self, cursor: CursorIcon);
    fn set_visible(&mut self, visible: bool);
    fn request_redraw(&mut self);
    fn present(&mut self, frame: Frame<P>);
}

/// A platform application.
pub trait Application: Sized {
    type Pixel: Pixel;
    type Window: Window<Self::Pixel>;

    /// Initialize the application.
    fn new() -> Result<Self>;

    /// Create a new window.
    fn create_window(&self, id: WindowId, desc: WindowDescriptor) -> Result<Self::Window>;

    /// Run the event loop.
    /// - `handler`: Callback for events.
    /// - `timeout`: If some, return after timeout (for integration tests or polling).
    fn run(&self, handler: &mut dyn EventHandler) -> Result<()>;

    /// Clipboard
    fn clipboard_copy(&self, text: &str);
    fn clipboard_paste(&self);

    /// Bell
    fn bell(&self);
}
