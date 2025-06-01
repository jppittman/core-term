// src/backends/mod.rs

//! Defines the `Driver` trait for backend implementations (e.g., X11, console)
//! and common types used by backends and the renderer, such as `BackendEvent`,
//! `CellCoords`, `TextRunStyle`, and `CellRect`.

// Import Color directly from the color module, not via glyph.
use crate::color::Color;
use crate::glyph::AttrFlags; // AttrFlags is correctly in glyph.
pub use crate::keys::{KeySymbol, Modifiers}; // Re-export KeySymbol and Modifiers
use anyhow::Result;
use std::os::unix::io::RawFd;

// Re-export driver implementations so they can be accessed via `crate::platform::backends::console::ConsoleDriver`, etc.
pub mod console;
pub mod x11;

// Import enums for Driver trait method signatures
pub use x11::window::CursorVisibility; // For set_cursor_visibility - Made pub
pub use x11::FocusState; // For set_focus - Made pub


// It can be useful to re-export concrete driver types if they are frequently
// used directly by `main.rs` or other high-level modules, though this is optional.
// Example:
// pub use console::ConsoleDriver;
// pub use x11::XDriver;

// --- Public Constants ---
// Default character dimensions for the terminal window.
// These are used if a backend cannot determine them from the environment
// or for initial setup before full font metrics are available.
pub const DEFAULT_WINDOW_WIDTH_CHARS: usize = 80;
pub const DEFAULT_WINDOW_HEIGHT_CHARS: usize = 24;

/// Represents events originating from the backend (platform-specific UI/input).
/// These events are processed by the `AppOrchestrator`, which may then update the
/// `TerminalEmulator` or perform other actions like writing to the PTY.
#[derive(Debug, Clone, PartialEq, Eq)] // Added Eq for full PartialEq comparison if needed by event handlers
pub enum BackendEvent {
    /// A keyboard key was pressed.
    Key {
        symbol: KeySymbol,    // Our new abstract key representation
        modifiers: Modifiers, // Our new abstract modifier flags
        text: String,         // Text from XLookupString or similar, preserved
    },
    /// The window or display area was resized by the platform.
    /// Provides new dimensions in pixels. The orchestrator uses these
    /// along with font metrics to calculate new cell dimensions for the terminal.
    Resize {
        /// New width of the display area in pixels.
        width_px: u16,
        /// New height of the display area in pixels.
        height_px: u16,
    },
    /// The application received a request to close from the platform
    /// (e.g., user clicked the window's close button).
    CloseRequested,
    /// The terminal window gained input focus.
    FocusGained,
    /// The terminal window lost input focus.
    FocusLost,
    /// A mouse button was pressed.
    MouseButtonPress {
        button: MouseButton,
        x: u16,
        y: u16,
        modifiers: Modifiers,
    },
    /// A mouse button was released.
    MouseButtonRelease {
        button: MouseButton,
        x: u16,
        y: u16,
        modifiers: Modifiers,
    },
    /// The mouse was moved.
    MouseMove {
        x: u16,
        y: u16,
        modifiers: Modifiers,
    },
    /// Paste data received from clipboard or primary selection.
    PasteData { text: String },
}

/// Represents mouse buttons.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MouseButton {
    Left,
    Middle,
    Right,
    ScrollUp,
    ScrollDown,
    Other(u8),
}

/// Commands for the renderer to execute.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RenderCommand {
    /// Clears the entire display area with the specified background color.
    ClearAll { bg: Color },
    /// Draws a run of text characters at a given cell coordinate with a specified style.
    DrawTextRun {
        x: usize,
        y: usize,
        text: String,
        fg: Color,
        bg: Color,
        flags: AttrFlags,
        is_selected: bool,
    },
    /// Fills a rectangular area of cells with a specified concrete color.
    FillRect {
        x: usize,
        y: usize,
        width: usize,
        height: usize,
        color: Color,
        is_selection_bg: bool,
    },
    /// Sets the visibility of the cursor.
    SetCursorVisibility { visible: bool },
    /// Sets the window title.
    SetWindowTitle { title: String },
    /// Rings the terminal bell.
    RingBell,
    /// Presents the composed frame to the display.
    PresentFrame,
}

/// Holds the current state of the platform, including display metrics and event sources.
/// This struct is typically owned and managed by a `Driver` implementation.
#[derive(Debug, Clone, PartialEq)]
pub struct PlatformState {
    /// Optional raw file descriptor that the orchestrator can monitor for platform events.
    pub event_fd: Option<RawFd>,
    /// Width of a single character cell in pixels.
    pub font_cell_width_px: usize,
    /// Height of a single character cell in pixels.
    pub font_cell_height_px: usize,
    /// The display scale factor (e.g., for HiDPI displays).
    pub scale_factor: f64,
    /// Current width of the display area in pixels.
    pub display_width_px: u16,
    /// Current height of the display area in pixels.
    pub display_height_px: u16,
}

/// Defines coordinates for a single character cell on the terminal grid (0-based).
#[derive(Debug, Clone, Copy, PartialEq, Eq)] // Added Eq
pub struct CellCoords {
    /// 0-based column index (x-coordinate).
    pub x: usize,
    /// 0-based row index (y-coordinate).
    pub y: usize,
}

/// Defines the visual style (colors and attribute flags) for a run of text.
/// This struct is used by the `Renderer` to instruct the `Driver` on how to draw text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)] // Added Eq
pub struct TextRunStyle {
    /// Foreground color for the text. Expected to be a concrete color by the driver.
    pub fg: Color,
    /// Background color for the text. Expected to be a concrete color by the driver.
    pub bg: Color,
    /// Attribute flags (e.g., bold, underline) for the text.
    pub flags: AttrFlags,
}

/// Defines a rectangular area of cells on the terminal grid.
/// Uses 0-based coordinates, and width/height are in number of cells.
#[derive(Debug, Clone, Copy, PartialEq, Eq)] // Added Eq
pub struct CellRect {
    /// 0-based column index of the top-left corner of the rectangle.
    pub x: usize,
    /// 0-based row index of the top-left corner of the rectangle.
    pub y: usize,
    /// Width of the rectangle in character cells.
    pub width: usize,
    /// Height of the rectangle in character cells.
    pub height: usize,
}

/// Defines the interface for a rendering and platform interaction driver.
///
/// A `Driver` is responsible for:
/// 1.  Window and display setup and management (e.g., creating an X11 window or setting up the console).
/// 2.  Handling platform-specific events (input, resize, close requests) and
///     translating them into generic `BackendEvent`s.
/// 3.  Providing an event source (e.g., a file descriptor) that the main event loop can monitor.
/// 4.  Implementing abstract drawing primitives that the `Renderer` uses to draw the
///     terminal state without needing to know backend-specific details.
/// 5.  Providing font and display metrics (character cell size, total display size in pixels).
/// 6.  Handling other platform-specific operations like setting the window title or ringing the bell.
pub trait Driver {
    /// Creates and initializes a new driver instance.
    ///
    /// This method should perform all necessary setup for the backend, such as
    /// opening display connections, creating windows, or configuring the console.
    /// Initial dimensions might be hints or determined by the driver itself.
    ///
    /// # Returns
    /// * `Result<Self>` where `Self` is `Sized`: The initialized driver instance or an error if setup fails.
    fn new() -> Result<Self>
    where
        Self: Sized;

    /// Returns an optional raw file descriptor that the orchestrator can monitor
    /// (e.g., using `epoll` or `select`) for platform events.
    ///
    /// If the driver uses a different event notification mechanism or polls internally,
    /// it can return `None`.
    fn get_event_fd(&self) -> Option<RawFd>;

    /// Processes any pending platform events.
    ///
    /// This method should be called when `get_event_fd()` indicates activity,
    /// or periodically if no FD is provided (polling). It translates native
    /// platform events (like X11 events or console input) into a list of
    /// generic `BackendEvent`s for the orchestrator to handle.
    ///
    /// # Returns
    /// * `Result<Vec<BackendEvent>>`: A vector of `BackendEvent`s that occurred,
    ///   or an error if event processing failed critically.
    fn process_events(&mut self) -> Result<Vec<BackendEvent>>;

    /// Retrieves the current platform state, including display metrics and event file descriptor.
    ///
    /// # Returns
    /// The `PlatformState` struct containing the current state.
    fn get_platform_state(&self) -> PlatformState;

    // --- Render Command Execution ---

    /// Executes a list of render commands.
    ///
    /// This method is called by the `Renderer` to draw the terminal state.
    /// The driver should iterate through the commands and apply them to its display surface.
    ///
    /// # Arguments
    /// * `commands`: A `Vec<RenderCommand>` to be executed by the driver.
    ///
    /// # Returns
    /// * `Result<()>`: Ok if all commands were processed successfully, or an error if one failed.
    fn execute_render_commands(&mut self, commands: Vec<RenderCommand>) -> Result<()>;

    /// Presents the composed frame to the display.
    /// For double-buffered systems, this would typically swap the buffers.
    /// For other systems (like a console driver), it might flush an output stream.
    fn present(&mut self) -> Result<()>;

    // --- Additional Driver Capabilities ---

    /// Sets the window title for graphical backends.
    /// For console backends, this might be a no-op or use an OSC sequence.
    fn set_title(&mut self, title: &str);

    /// Rings the terminal bell (audible or visual).
    fn bell(&mut self);

    /// Sets the visibility of the cursor for the backend.
    fn set_cursor_visibility(&mut self, visibility: CursorVisibility);

    /// Informs the driver about focus changes.
    /// This can be used by the driver to change cursor appearance (e.g., solid vs. hollow).
    fn set_focus(&mut self, focus_state: FocusState);

    /// Performs any necessary cleanup before the driver is dropped.
    /// This includes releasing platform resources (e.g., closing display connections,
    /// restoring terminal modes). This method should be idempotent.
    fn cleanup(&mut self) -> Result<()>;

    // --- Selection Handling (optional for backends) ---

    /// Takes ownership of the specified selection (e.g., PRIMARY, CLIPBOARD).
    /// Atom types are u64 for trait compatibility; X11 backend will cast to xlib::Atom.
    #[allow(unused_variables)]
    fn own_selection(&mut self, selection_name_atom: u64, text: String) {
        // Default implementation: no-op for backends that don't support selection ownership.
        log::trace!("Driver::own_selection called but not implemented for this backend.");
    }

    /// Requests data from the specified selection in the given target format.
    /// Atom types are u64 for trait compatibility.
    #[allow(unused_variables)]
    fn request_selection_data(&mut self, selection_name_atom: u64, target_atom: u64) {
        // Default implementation: no-op for backends that don't support requesting selection data.
        log::trace!("Driver::request_selection_data called but not implemented for this backend.");
    }
}
