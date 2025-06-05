// src/platform/platform_trait.rs
//
// Defines the `Platform` trait, which abstracts over platform-specific
// functionality for PTY and UI interaction.

use crate::platform::backends::PlatformState; // BackendEvent is not directly used by the trait's method signatures
use anyhow::Result;

use super::actions::PlatformAction;
use super::PlatformEvent;

/// A trait that defines the interface for a platform implementation.
///
/// This trait abstracts over the specific details of how a platform interacts
/// with the PTY (pseudo-terminal) and the UI. It provides methods for
/// creating a new platform instance, polling for PTY and UI events,
/// dispatching actions to the PTY and UI, getting the current platform state,
/// and shutting down the platform.
pub trait Platform {
    /// Creates a new platform instance.
    ///
    /// # Arguments
    ///
    /// * `initial_pty_cols` - The initial number of columns for the PTY.
    /// * `initial_pty_rows` - The initial number of rows for the PTY.
    /// * `shell_command` - The command to run in the PTY.
    /// * `shell_args` - The arguments to pass to the shell command.
    ///
    /// # Returns
    ///
    /// A `Result` containing a tuple of the platform instance and the initial
    /// `PlatformState`, or an error if initialization fails.
    fn new(
        initial_pty_cols: u16,
        initial_pty_rows: u16,
        shell_command: String,
        shell_args: Vec<String>,
    ) -> Result<(Self, PlatformState)>
    where
        Self: Sized;

    /// Polls for an event from the UI.
    ///
    /// # Returns
    ///
    /// A `Result` containing an `Option` with the `BackendEvent` if available,
    /// or `None` if no event is available. Returns an error if polling fails.
    fn poll_events(&mut self) -> Result<Vec<PlatformEvent>>;

    /// Dispatches an action to the PTY.
    ///
    /// # Arguments
    ///
    /// * `action` - The `PtyActionCommand` to dispatch.
    ///
    /// # Returns
    ///
    /// A `Result` indicating success or failure.
    fn dispatch_actions(&mut self, action: Vec<PlatformAction>) -> Result<()>;

    /// Gets the current state of the platform.
    ///
    /// # Returns
    ///
    /// The current `PlatformState`.
    fn get_current_platform_state(&self) -> PlatformState;
}
