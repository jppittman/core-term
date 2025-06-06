// src/os/epoll.rs

//! This module provides a wrapper around `epoll` functionality using raw `libc`
//! FFI calls for managing and polling file descriptors for I/O events.
//! It defines type-safe enums and bitflags for epoll operations and events.

use anyhow::{Context, Result};
use bitflags::bitflags;
pub use libc::epoll_event;
use log::{debug, trace, warn};
use std::io;
use std::os::unix::io::RawFd;

const EPOLL_CREATE_CLOEXEC: libc::c_int = libc::O_CLOEXEC;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum EpollCtlOp {
    Add = libc::EPOLL_CTL_ADD,
    Mod = libc::EPOLL_CTL_MOD,
    Del = libc::EPOLL_CTL_DEL,
}

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
    pub struct EpollFlags: u32 {
        const EPOLLIN = libc::EPOLLIN as u32;
        const EPOLLOUT = libc::EPOLLOUT as u32;
        const EPOLLPRI = libc::EPOLLPRI as u32;
        const EPOLLERR = libc::EPOLLERR as u32;
        const EPOLLHUP = libc::EPOLLHUP as u32;
        const EPOLLRDHUP = libc::EPOLLRDHUP as u32;
        const EPOLLET = libc::EPOLLET as u32;
        const EPOLLONESHOT = libc::EPOLLONESHOT as u32;
    }
}

fn new_libc_epoll_event(flags: EpollFlags, token: u64) -> libc::epoll_event {
    libc::epoll_event {
        events: flags.bits(),
        u64: token,
    }
}

#[allow(dead_code)]
pub fn epoll_event_token(event: &libc::epoll_event) -> u64 {
    // Removed unnecessary unsafe block as field access to Copy types in a union is safe.
    event.u64
}

#[allow(dead_code)]
pub fn epoll_event_flags(event: &libc::epoll_event) -> EpollFlags {
    EpollFlags::from_bits_truncate(event.events)
}

#[derive(Debug)]
pub struct EventMonitor {
    epoll_fd: RawFd,
}

impl EventMonitor {
    pub fn new() -> Result<Self> {
        let epoll_fd = unsafe { libc::epoll_create1(EPOLL_CREATE_CLOEXEC) };
        if epoll_fd == -1 {
            return Err(io::Error::last_os_error())
                .context("Failed to create epoll instance (epoll_create1)");
        }
        debug!("EventMonitor created with epoll_fd: {}", epoll_fd);
        Ok(Self { epoll_fd })
    }

    pub fn add(&self, fd: RawFd, token: u64, flags: EpollFlags) -> Result<()> {
        let mut event = new_libc_epoll_event(flags, token);
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
            fd,
            self.epoll_fd,
            token,
            flags
        );
        Ok(())
    }

    pub fn modify(&self, fd: RawFd, token: u64, flags: EpollFlags) -> Result<()> {
        let mut event = new_libc_epoll_event(flags, token);
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
            fd,
            self.epoll_fd,
            token,
            flags
        );
        Ok(())
    }

    pub fn delete(&self, fd: RawFd) -> Result<()> {
        let mut event: libc::epoll_event = unsafe { std::mem::zeroed() };
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

    pub fn events(
        &self, // No longer needs &mut self
        events_out: &mut Vec<libc::epoll_event>,
        timeout_ms: isize,
    ) -> Result<()> {
        // Returns () on success
        trace!(
            "EventMonitor: polling for events with timeout {}ms on epoll_fd {}",
            timeout_ms,
            self.epoll_fd
        );

        let num_events = unsafe {
            // 2. Call epoll_wait directly on the vector's buffer.
            //    We can write up to `capacity()` events into it.
            libc::epoll_wait(
                self.epoll_fd,
                events_out.as_mut_ptr(),
                events_out.capacity() as libc::c_int,
                timeout_ms as libc::c_int,
            )
        };

        if num_events == -1 {
            let err = io::Error::last_os_error();
            if err.kind() == io::ErrorKind::Interrupted {
                trace!("EventMonitor: epoll_wait interrupted (EINTR), returning.");
                return Ok(());
            }
            return Err(err).context("epoll_wait failed in EventMonitor");
        }

        // 3. UNSAFE: This is the critical part. We are telling Rust that
        //    the kernel has initialized `num_events` elements in the vector,
        //    so we can safely set the vector's length to this new value.
        //    This is safe because epoll_wait guarantees initialization.
        unsafe {
            events_out.set_len(num_events as usize);
        }

        trace!(
            "EventMonitor: epoll_wait on fd {} returned {} events",
            self.epoll_fd,
            num_events
        );
        Ok(())
    }
}

impl Drop for EventMonitor {
    fn drop(&mut self) {
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

unsafe impl Send for EventMonitor {}
unsafe impl Sync for EventMonitor {}
