//! Defines the `Platform` trait, the hardware abstraction layer (HAL) for the terminal.

use crate::platform::actions::PlatformAction;
use crate::platform::backends::PlatformState;
use crate::platform::{PlatformChannels, PlatformEvent};
use anyhow::Result;

/// A trait that abstracts over platform-specific operations.
///
/// This trait acts as a hardware abstraction layer (HAL) for the terminal,
/// providing a consistent interface for interacting with the windowing system,
/// input devices, and the pseudo-terminal (PTY). Implementors of this trait
/// handle the specifics of a given platform, such as Linux/X11 or macOS/Cocoa.
pub trait Platform {
    /// Creates a new platform-specific instance.
    ///
    /// This method initializes the connection to the windowing system and creates a
    /// window. The platform receives channels to communicate with the orchestrator
    /// and other actors that were spawned in main.
    ///
    /// This should be pure initialization only - no thread spawning, no event sending.
    ///
    /// # Arguments
    ///
    /// * `channels` - Grouped channels for Platform ↔ Orchestrator communication
    ///
    /// # Returns
    ///
    /// On success, returns the platform driver instance.
    fn new(channels: PlatformChannels) -> Result<Self>
    where
        Self: Sized;

    /// Polls for new events from the platform.
    ///
    /// This method checks for events from both the UI backend (e.g., window
    /// resize, keyboard input) and the PTY's output stream. It may block
    /// briefly to wait for events.
    ///
    /// # Returns
    ///
    /// A `Result` containing a vector of `PlatformEvent`s that have occurred
    /// since the last poll.
    fn poll_events(&mut self) -> Result<Vec<PlatformEvent>>;

    /// Dispatches a vector of actions to be executed by the platform.
    ///
    /// These actions can include writing to the PTY, changing the cursor,
    /// or managing the window state.
    ///
    /// # Arguments
    ///
    /// * `action` - A vector of `PlatformAction`s to be executed.
    ///
    /// # Returns
    ///
    /// A `Result` indicating whether the actions were dispatched successfully.
    fn dispatch_actions(&mut self, action: Vec<PlatformAction>) -> Result<()>;

    /// Gets a snapshot of the current state of the platform.
    ///
    /// # Returns
    ///
    /// The current `PlatformState`, including window dimensions and font metrics.
    fn get_current_platform_state(&self) -> PlatformState;

    /// Runs the platform's native event loop, blocking until shutdown.
    ///
    /// This method consumes `self` and runs the platform-specific event loop:
    /// - macOS: `NSApp.run()` - blocks in Cocoa event loop
    /// - Linux: X11/Wayland event loop
    ///
    /// Events are forwarded to the Orchestrator actor via channels.
    /// Returns when the application shuts down (window closed, quit requested, etc.).
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on clean shutdown, or an error if the event loop fails.
    fn run(self) -> Result<()>;

    /// Performs any necessary cleanup before the platform is dropped.
    ///
    /// This includes releasing platform resources (e.g., closing display
    /// connections, restoring terminal modes). This method should be idempotent.
    fn cleanup(&mut self) -> Result<()>;
}
