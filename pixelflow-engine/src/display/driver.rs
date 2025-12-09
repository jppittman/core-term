// src/display/driver.rs
//! DisplayDriver trait - unified driver/sender interface.
//!
//! The driver IS the sender. One type, one interface:
//! - `new()` creates the driver (just channels, no platform resources yet)
//! - `send()` sends commands (always non-blocking)
//! - `run()` blocks, runs the event loop
//!
//! ## Threading Model
//! - Driver is Clone - just cmd_tx, trivially cloneable
//! - `send(Configure)` queues config for event loop
//! - `run()` blocks, reads Configure, creates platform resources, runs loop
//! - Other commands are sent via channel to the running event loop
//!
//! ## Lifecycle
//! 1. `new(engine_tx)` - Create driver (just channels)
//! 2. `driver.clone()` - Engine gets a clone
//! 3. `driver.send(Configure(config))` - Queue config
//! 4. Main thread calls `driver.run()` - blocks until shutdown
//! 5. Engine spawns on thread, calls `driver.send(Present(...))` etc.

use crate::api::private::{DriverCommand, EngineActorHandle};
use anyhow::Result;
use pixelflow_core::Pixel;

/// Platform-specific display driver.
///
/// The driver IS the sender. Just one field: `cmd_tx`.
/// Clone it to give the engine a handle.
/// - `send()` queues commands (non-blocking)
/// - `run()` blocks, reads Configure, creates resources, runs event loop
///
/// ## Pixel Type
/// Each platform declares its required pixel format via the `Pixel` associated type.
/// This ensures type safety: the engine renders to the correct format for the platform.
/// - Cocoa: `Rgba` (CGImage with kCGImageAlphaPremultipliedLast)
/// - X11: `Bgra` (XImage with ZPixmap on little-endian)
/// - Web: `Rgba` (ImageData)
pub trait DisplayDriver: Clone + Send {
    /// The pixel format required by this display driver.
    type Pixel: Pixel;

    /// Create a new driver without engine channel.
    ///
    /// This only creates channels. The engine handle is injected later via
    /// SetEngineHandle command to avoid circular references and memory leaks.
    /// Platform resources (window, etc.) are created when `run()` is called.
    fn new() -> Result<Self>;

    /// Send a command to the driver (non-blocking).
    ///
    /// - `Configure(config)`: Queue config, must be sent before run()
    /// - `Present(snapshot)`: Queue frame for display
    /// - `SetTitle(s)`: Set window title
    /// - `CopyToClipboard(s)`: Copy text to clipboard
    /// - `RequestPaste`: Request paste, data arrives via DisplayEvent::PasteData
    /// - `Bell`: Ring the terminal bell
    /// - `Shutdown`: Stop the event loop
    fn send(&self, cmd: DriverCommand<Self::Pixel>) -> Result<()>;

    /// Run the driver event loop (blocking).
    ///
    /// Reads Configure from channel first, creates platform resources,
    /// then runs the event loop until Shutdown or close.
    fn run(&self) -> Result<()>;
}
