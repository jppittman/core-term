// src/os/epoll.rs

//! This module provides a wrapper around `epoll` functionality for managing
//! and polling file descriptors for I/O events.

use anyhow::{Context, Result};
use nix::sys::epoll::{
    epoll_create1, epoll_ctl, epoll_wait, EpollCreateFlags, EpollEvent, EpollFlags, EpollOp,
};
use std::os::unix::io::{AsRawFd, RawFd};
use log::{debug, trace, warn};

// The maximum number of events to retrieve in a single call to epoll_wait.
// This value can be tuned based on expected workload and desired responsiveness.
const MAX_EVENTS_BUFFER_SIZE: usize = 16;

/// A wrapper for managing an `epoll` instance and monitoring file descriptor events.
/// It encapsulates the epoll file descriptor and an internal buffer for events.
#[derive(Debug)]
pub struct EventMonitor {
    /// The epoll file descriptor.
    epoll_fd: RawFd,
    /// Internal buffer to store events retrieved from `epoll_wait`.
    /// This buffer is reused across calls to `events()`.
    event_buffer: [EpollEvent; MAX_EVENTS_BUFFER_SIZE],
}

impl EventMonitor {
    /// Creates a new `EventMonitor` instance.
    ///
    /// This involves creating a new `epoll` file descriptor. The internal event
    /// buffer is also initialized.
    ///
    /// # Returns
    /// * `Result<Self>`: An instance of `EventMonitor` or an error if creation fails.
    pub fn new() -> Result<Self> {
        let epoll_fd = epoll_create1(EpollCreateFlags::EPOLL_CLOEXEC)
            .context("Failed to create epoll instance")?;
        debug!("EventMonitor created with epoll_fd: {}", epoll_fd);
        Ok(Self {
            epoll_fd,
            // Initialize the buffer with empty events.
            event_buffer: [EpollEvent::empty(); MAX_EVENTS_BUFFER_SIZE],
        })
    }

    /// Adds a file descriptor to the `EventMonitor` for monitoring.
    ///
    /// # Arguments
    /// * `fd`: The file descriptor to add.
    /// * `token`: A u64 token associated with this fd, returned by `events()`.
    /// * `flags`: The `EpollFlags` specifying the events to monitor (e.g., `EPOLLIN`).
    ///
    /// # Returns
    /// * `Result<()>`: Ok if successful, or an error.
    pub fn add(&self, fd: RawFd, token: u64, flags: EpollFlags) -> Result<()> {
        let mut event = EpollEvent::new(flags, token);
        epoll_ctl(self.epoll_fd, EpollOp::EpollCtlAdd, fd, &mut event)
            .with_context(|| format!("Failed to add fd {} to epoll (token: {})", fd, token))?;
        trace!("Added fd {} to epoll_fd {} with token {} and flags {:?}", fd, self.epoll_fd, token, flags);
        Ok(())
    }

    /// Modifies the events for an already monitored file descriptor.
    ///
    /// # Arguments
    /// * `fd`: The file descriptor to modify.
    /// * `token`: The u64 token associated with this fd.
    /// * `flags`: The new `EpollFlags`.
    ///
    /// # Returns
    /// * `Result<()>`: Ok if successful, or an error.
    pub fn modify(&self, fd: RawFd, token: u64, flags: EpollFlags) -> Result<()> {
        let mut event = EpollEvent::new(flags, token);
        epoll_ctl(self.epoll_fd, EpollOp::EpollCtlMod, fd, &mut event)
            .with_context(|| format!("Failed to modify fd {} in epoll (token: {})", fd, token))?;
        trace!("Modified fd {} in epoll_fd {} to token {} and flags {:?}", fd, self.epoll_fd, token, flags);
        Ok(())
    }

    /// Removes a file descriptor from the `EventMonitor`.
    ///
    /// # Arguments
    /// * `fd`: The file descriptor to remove.
    ///
    /// # Returns
    /// * `Result<()>`: Ok if successful, or an error.
    pub fn delete(&self, fd: RawFd) -> Result<()> {
        // An event is not strictly needed for delete prior to Linux 2.6.9, but providing null is problematic
        // with some bindings, and providing an empty event is safe.
        let mut event = EpollEvent::empty();
        epoll_ctl(self.epoll_fd, EpollOp::EpollCtlDel, fd, &mut event)
            .with_context(|| format!("Failed to delete fd {} from epoll", fd))?;
        trace!("Deleted fd {} from epoll_fd {}", fd, self.epoll_fd);
        Ok(())
    }

    /// Waits for I/O events on the monitored file descriptors and returns an iterator over them.
    ///
    /// This method blocks until events occur or the timeout expires.
    /// The events are stored in an internal buffer, and a slice referring to these events is returned.
    ///
    /// **Note:** The returned slice `&[EpollEvent]` is valid only until the next mutable call
    /// to `events()` on this `EventMonitor` instance, as that call may overwrite the internal buffer.
    ///
    /// # Arguments
    /// * `timeout_ms`: The timeout in milliseconds. A value of -1 means block indefinitely.
    ///
    /// # Returns
    /// * `Result<&[EpollEvent]>`: A slice containing the `EpollEvent`s that occurred.
    ///   Returns an empty slice if the timeout expires before any events.
    ///   Returns an error if `epoll_wait` itself fails.
    pub fn events(&mut self, timeout_ms: isize) -> Result<&[EpollEvent]> {
        trace!("EventMonitor: polling for events with timeout {}ms on epoll_fd {}", timeout_ms, self.epoll_fd);

        let num_events = epoll_wait(self.epoll_fd, &mut self.event_buffer, timeout_ms)
            .context("epoll_wait failed in EventMonitor")?;

        trace!("EventMonitor: epoll_wait on fd {} returned {} events", self.epoll_fd, num_events);
        // The slice `&self.event_buffer[0..num_events]` is safe because epoll_wait guarantees
        // that it has initialized `num_events` elements of the buffer.
        Ok(&self.event_buffer[0..num_events])
    }
}

// Re-export `EpollEvent` and `EpollFlags` from `nix` for convenience if `main.rs`
// or other modules need to directly work with these types from the `os::epoll` module.
pub use nix::sys::epoll::{EpollEvent, EpollFlags};

impl Drop for EventMonitor {
    fn drop(&mut self) {
        // The epoll_fd is a system resource that should be closed.
        // nix::unistd::close handles negative fd values appropriately (returns EBADF).
        if let Err(e) = nix::unistd::close(self.epoll_fd) {
            warn!("Failed to close epoll_fd {} in EventMonitor::drop: {}", self.epoll_fd, e);
        } else {
            debug!("Closed epoll_fd {} in EventMonitor::drop", self.epoll_fd);
        }
    }
}

// SAFETY: EventMonitor owns the epoll_fd (a RawFd, which is i32).
// Epoll operations on different epoll instances are thread-safe.
// If EventMonitor were to be used across threads, ensure that operations
// on the *same* EventMonitor instance (same epoll_fd) are properly synchronized
// if methods other than `events()` (like `add`/`delete`) can be called concurrently.
// For typical single-threaded event loop usage, this is fine.
// RawFd itself is Send + Sync. The primary concern would be mutable access patterns.
// Given its typical use in a single event loop, marking it Send + Sync is generally acceptable,
// assuming the main application ensures safe usage patterns if threads are involved.
unsafe impl Send for EventMonitor {}
unsafe impl Sync for EventMonitor {}
