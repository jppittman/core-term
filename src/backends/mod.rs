// src/backends/mod.rs
// Declares backend modules and defines the common trait.

// Declare the sub-modules within this directory
pub mod console;
pub mod x11;

// --- Re-export specific backend implementations ---
// This allows `use backends::ConsoleBackend;` instead of `use backends::console::ConsoleBackend;`
pub use console::ConsoleBackend;
pub use x11::XBackend;


// --- Define the TerminalBackend Trait Here ---

// Use items from the crate root (where Term is defined)
use crate::Term;

// Necessary imports for the trait signature
use anyhow::Result;
use std::os::fd::RawFd;

// Defines the interface for different terminal display backends
pub trait TerminalBackend {
    // Perform any backend-specific initialization
    fn init(&mut self) -> Result<()>;

    // Get a list of file descriptors the backend wants to be polled
    fn get_event_fds(&self) -> Vec<RawFd>;

    // Handle an event for a specific file descriptor managed by the backend.
    // Returns true if the main loop should exit.
    fn handle_event(&mut self, term: &mut Term, event_fd: RawFd, event_kind: u32) -> Result<bool>;

    // Draw the current state of the Term to the backend's display
    // Needs &mut self if drawing might change internal state (like color cache)
    fn draw(&mut self, term: &Term) -> Result<()>;

    // Get the current dimensions (cols, rows) from the backend
    fn get_dimensions(&self) -> (usize, usize);
}

