// src/io/traits.rs

/// A source of IO events that can be polled and read.
/// On Unix, this maps to AsRawFd.
/// (Future: On Windows, this will map to AsRawHandle).
#[cfg(unix)]
pub trait EventSource: std::io::Read + std::os::unix::io::AsRawFd + Send {}

// Auto-implement for any type that satisfies the bounds
#[cfg(unix)]
impl<T> EventSource for T where T: std::io::Read + std::os::unix::io::AsRawFd + Send {}

use crate::ansi::AnsiCommand;

/// Trait for sending parsed ANSI commands to the application.
pub trait PtySender: Send {
    fn send(&self, cmds: Vec<AnsiCommand>) -> Result<(), anyhow::Error>;
}
