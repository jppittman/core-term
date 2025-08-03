// src/platform/platform_trait.rs
//
// Defines the `Platform` trait, which abstracts over platform-specific
// functionality for PTY and UI interaction.

use crate::platform::backends::PlatformState;
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
    fn new(
        initial_pty_cols: u16,
        initial_pty_rows: u16,
        shell_command: String,
        shell_args: Vec<String>,
    ) -> Result<(Self, PlatformState)>
    where
        Self: Sized;

    /// Polls for events from the PTY and UI.
    fn poll_events(&mut self) -> Result<Vec<PlatformEvent>>;

    /// Dispatches actions to the PTY or UI.
    fn dispatch_actions(&mut self, action: Vec<PlatformAction>) -> Result<()>;

    /// Gets the current state of the platform.
    fn get_current_platform_state(&self) -> PlatformState;

    /// Performs any necessary cleanup before the platform is dropped.
    fn cleanup(&mut self) -> Result<()>;
}
