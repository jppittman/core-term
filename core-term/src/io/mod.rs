// src/io/mod.rs
//
// I/O module - handles PTY, event monitoring, and timing.
// Separated from platform module which handles display/windowing.

pub mod event_monitor_actor;
pub mod pty;
pub mod vsync_actor;

#[cfg(test)]
mod pty_tests;

// Platform-specific event monitoring implementations
#[cfg(target_os = "macos")]
pub mod kqueue;

#[cfg(target_os = "linux")]
pub mod epoll;

// Platform-agnostic re-exports
#[cfg(target_os = "macos")]
pub mod event {
    pub use super::kqueue::*;
}

#[cfg(target_os = "linux")]
pub mod event {
    pub use super::epoll::*;
}
