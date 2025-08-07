//! Defines the `Platform` trait, the hardware abstraction layer (HAL) for the terminal.

use crate::platform::backends::PlatformState;
use anyhow::Result;

use super::actions::PlatformAction;
use super::PlatformEvent;

/// A trait that abstracts over platform-specific operations.
///
/// This trait acts as a hardware abstraction layer (HAL) for the terminal,
/// providing a consistent interface for interacting with the windowing system,
/// input devices, and the pseudo-terminal (PTY). Implementors of this trait
/// handle the specifics of a given platform, such as Linux/X11 or macOS/Cocoa.
pub trait Platform {
    /// Creates a new platform-specific instance and its initial state.
    ///
    /// This method initializes the connection to the windowing system, creates a
    /// window, sets up the PTY, and spawns the shell process.
    ///
    /// # Arguments
    ///
    /// * `initial_pty_cols` - The initial number of columns for the PTY.
    /// * `initial_pty_rows` - The initial number of rows for the PTY.
    /// * `shell_command` - The command to run in the PTY (e.g., `/bin/bash`).
    /// * `shell_args` - The arguments to pass to the shell command.
    ///
    /// # Returns
    ///
    /// On success, returns a tuple containing the platform driver instance and
    /// the initial `PlatformState`.
    fn new(
        initial_pty_cols: u16,
        initial_pty_rows: u16,
        shell_command: String,
        shell_args: Vec<String>,
    ) -> Result<(Self, PlatformState)>
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

    /// Performs any necessary cleanup before the platform is dropped.
    ///
    /// This includes releasing platform resources (e.g., closing display
    /// connections, restoring terminal modes). This method should be idempotent.
    fn cleanup(&mut self) -> Result<()>;
}
