// src/io/event.rs
//! Event monitoring abstraction (kqueue/epoll wrapper).

use std::os::unix::io::RawFd;

#[cfg(target_os = "macos")]
pub use crate::io::kqueue::{EventMonitor, KqueueFlags};

#[cfg(target_os = "linux")]
pub use crate::io::epoll::{EpollFlags as KqueueFlags, EventMonitor};

#[derive(Debug, Clone, Copy)]
pub struct Event {
    pub fd: RawFd,
    pub token: u64,
    pub flags: KqueueFlags,
}
