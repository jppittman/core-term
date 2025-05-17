// src/os/epoll.rs

//! This module provides a wrapper around `epoll` functionality using raw `libc`
//! FFI calls for managing and polling file descriptors for I/O events.
//! It defines type-safe enums and bitflags for epoll operations and events.

use anyhow::{Context, Result};
use bitflags::bitflags; // For creating flag enums
use log::{debug, trace, warn};
use std::io;
use std::os::unix::io::RawFd;
// Removed: `use std::ptr;` as it's no longer directly used.

// --- libc epoll constants and structures ---

/// Flag for `epoll_create1` to close the epoll file descriptor upon exec.
const EPOLL_CREATE_CLOEXEC: libc::c_int = libc::O_CLOEXEC; // Or libc::EPOLL_CLOEXEC if available and preferred

/// Defines the operations for `epoll_ctl`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)] // Matches libc::c_int type for epoll_ctl operations
pub enum EpollCtlOp {
    /// Add a file descriptor to the interest list.
    Add = libc::EPOLL_CTL_ADD,
    /// Modify event settings for a file descriptor.
    Mod = libc::EPOLL_CTL_MOD,
    /// Remove a file descriptor from the interest list.
    Del = libc::EPOLL_CTL_DEL,
}

bitflags! {
    /// Represents the event flags for `epoll_ctl` and `epoll_wait`.
    /// These flags specify the types of events to monitor on a file descriptor.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
    pub struct EpollFlags: u32 {
        /// Available for read operations.
        const EPOLLIN = libc::EPOLLIN as u32;
        /// Available for write operations.
        const EPOLLOUT = libc::EPOLLOUT as u32;
        /// Urgent data available for read operations.
        const EPOLLPRI = libc::EPOLLPRI as u32;
        /// Error condition happened on the associated file descriptor.
        /// `epoll_wait` will always wait for this event; it is not necessary to set it in events.
        const EPOLLERR = libc::EPOLLERR as u32;
        /// Hang up happened on the associated file descriptor.
        /// `epoll_wait` will always wait for this event; it is not necessary to set it in events.
        const EPOLLHUP = libc::EPOLLHUP as u32;
        /// Stream socket peer closed connection, or shut down writing half of connection.
        /// (This flag is especially useful for writing simple code to detect peer shutdown
        /// when using Edge Triggered monitoring.)
        const EPOLLRDHUP = libc::EPOLLRDHUP as u32;
        /// Sets the Edge Triggered behavior for the associated file descriptor.
        const EPOLLET = libc::EPOLLET as u32;
        /// Sets the one-shot behavior. After an event is pulled with `epoll_wait`,
        /// the associated file descriptor is internally disabled and no other events
        /// will be reported. The user must use `EPOLL_CTL_MOD` to rearm the file descriptor.
        const EPOLLONESHOT = libc::EPOLLONESHOT as u32;
    }
}

// The EpollEvent struct is now directly using libc::epoll_event by type alias
// for clarity when dealing with FFI.
// Our helper methods will operate on this libc type.
//
// pub type EpollEvent = libc::epoll_event; // This would be an option
//
// However, to add helper methods like `token()` and `flags()`, it's often cleaner
// to wrap it or provide free functions. Given the direct usage in `event_buffer`
// and FFI calls, we will use `libc::epoll_event` directly in the buffer
// and for FFI, and provide accessor functions for the data if needed by `main.rs`.

/// Helper function to create a `libc::epoll_event`.
///
/// # Arguments
/// * `flags`: An `EpollFlags` bitmask specifying the events to monitor.
/// * `token`: A `u64` token to associate with this event.
fn new_libc_epoll_event(flags: EpollFlags, token: u64) -> libc::epoll_event {
    libc::epoll_event {
        events: flags.bits(), // Get the raw u32 value from bitflags
        u64: token,           // Directly assign to the u64 field of the union
    }
}

/// Helper to get the token from a `libc::epoll_event`.
#[allow(dead_code)] // Potentially used by main.rs or other modules
pub fn epoll_event_token(event: &libc::epoll_event) -> u64 {
    // SAFETY: Accessing union field. We store tokens in u64.
    unsafe { event.u64 }
}

/// Helper to get `EpollFlags` from a `libc::epoll_event`.
#[allow(dead_code)] // Potentially used by main.rs or other modules
pub fn epoll_event_flags(event: &libc::epoll_event) -> EpollFlags {
    EpollFlags::from_bits_truncate(event.events)
}

/// The maximum number of events to retrieve in a single call to `epoll_wait`.
const MAX_EVENTS_BUFFER_SIZE: usize = 16;

/// A wrapper for managing an `epoll` instance and monitoring file descriptor events.
/// It encapsulates the epoll file descriptor and an internal buffer for events.
#[derive(Debug)]
pub struct EventMonitor {
    /// The epoll file descriptor.
    epoll_fd: RawFd,
    /// Internal buffer to store events retrieved from `epoll_wait`.
    /// This buffer now directly uses `libc::epoll_event`.
    event_buffer: [libc::epoll_event; MAX_EVENTS_BUFFER_SIZE],
}

impl EventMonitor {
    /// Creates a new `EventMonitor` instance.
    pub fn new() -> Result<Self> {
        // SAFETY: Calling libc::epoll_create1. EPOLL_CREATE_CLOEXEC is a standard flag.
        let epoll_fd = unsafe { libc::epoll_create1(EPOLL_CREATE_CLOEXEC) };
        if epoll_fd == -1 {
            return Err(io::Error::last_os_error())
                .context("Failed to create epoll instance (epoll_create1)");
        }
        debug!("EventMonitor created with epoll_fd: {}", epoll_fd);
        Ok(Self {
            epoll_fd,
            // Initialize with default (zeroed) libc::epoll_event structs.
            event_buffer: [unsafe { std::mem::zeroed() }; MAX_EVENTS_BUFFER_SIZE],
        })
    }

    /// Adds a file descriptor to the `EventMonitor` for monitoring.
    pub fn add(&self, fd: RawFd, token: u64, flags: EpollFlags) -> Result<()> {
        let mut event = new_libc_epoll_event(flags, token);
        // SAFETY: Calling libc::epoll_ctl. `event` is now `libc::epoll_event`.
        if unsafe {
            libc::epoll_ctl(
                self.epoll_fd,
                EpollCtlOp::Add as libc::c_int,
                fd,
                &mut event,
            )
        } == -1
        {
            return Err(io::Error::last_os_error())
                .with_context(|| format!("Failed to add fd {} to epoll (token: {})", fd, token));
        }
        trace!(
            "Added fd {} to epoll_fd {} with token {} and flags {:?}",
            fd, self.epoll_fd, token, flags
        );
        Ok(())
    }

    /// Modifies the events for an already monitored file descriptor.
    pub fn modify(&self, fd: RawFd, token: u64, flags: EpollFlags) -> Result<()> {
        let mut event = new_libc_epoll_event(flags, token);
        // SAFETY: Calling libc::epoll_ctl. `event` is now `libc::epoll_event`.
        if unsafe {
            libc::epoll_ctl(
                self.epoll_fd,
                EpollCtlOp::Mod as libc::c_int,
                fd,
                &mut event,
            )
        } == -1
        {
            return Err(io::Error::last_os_error()).with_context(|| {
                format!("Failed to modify fd {} in epoll (token: {})", fd, token)
            });
        }
        trace!(
            "Modified fd {} in epoll_fd {} to token {} and flags {:?}",
            fd, self.epoll_fd, token, flags
        );
        Ok(())
    }

    /// Removes a file descriptor from the `EventMonitor`.
    pub fn delete(&self, fd: RawFd) -> Result<()> {
        let mut event: libc::epoll_event = unsafe { std::mem::zeroed() }; // Argument is ignored for DEL.
        // SAFETY: Calling libc::epoll_ctl.
        if unsafe {
            libc::epoll_ctl(
                self.epoll_fd,
                EpollCtlOp::Del as libc::c_int,
                fd,
                &mut event,
            )
        } == -1
        {
            return Err(io::Error::last_os_error())
                .with_context(|| format!("Failed to delete fd {} from epoll", fd));
        }
        trace!("Deleted fd {} from epoll_fd {}", fd, self.epoll_fd);
        Ok(())
    }

    /// Waits for I/O events on the monitored file descriptors.
    /// Returns a slice of `libc::epoll_event`.
    pub fn events(&mut self, timeout_ms: isize) -> Result<&[libc::epoll_event]> {
        trace!(
            "EventMonitor: polling for events with timeout {}ms on epoll_fd {}",
            timeout_ms, self.epoll_fd
        );

        // SAFETY: Calling libc::epoll_wait.
        let num_events = unsafe {
            libc::epoll_wait(
                self.epoll_fd,
                self.event_buffer.as_mut_ptr(), // This is now *mut libc::epoll_event
                MAX_EVENTS_BUFFER_SIZE as libc::c_int,
                timeout_ms as libc::c_int,
            )
        };

        if num_events == -1 {
            let err = io::Error::last_os_error();
            if err.kind() == io::ErrorKind::Interrupted {
                trace!("EventMonitor: epoll_wait interrupted (EINTR), returning empty slice.");
                return Ok(&self.event_buffer[0..0]);
            }
            return Err(err).context("epoll_wait failed in EventMonitor");
        }

        trace!(
            "EventMonitor: epoll_wait on fd {} returned {} events",
            self.epoll_fd, num_events
        );
        Ok(&self.event_buffer[0..num_events as usize])
    }
}

impl Drop for EventMonitor {
    fn drop(&mut self) {
        // SAFETY: Calling libc::close on a valid file descriptor.
        if unsafe { libc::close(self.epoll_fd) } == -1 {
            warn!(
                "Failed to close epoll_fd {} in EventMonitor::drop: {}",
                self.epoll_fd,
                io::Error::last_os_error()
            );
        } else {
            debug!("Closed epoll_fd {} in EventMonitor::drop", self.epoll_fd);
        }
    }
}

// SAFETY: EventMonitor owns the epoll_fd. Standard FDs are Send/Sync.
unsafe impl Send for EventMonitor {}
unsafe impl Sync for EventMonitor {}
