// src/platform/os/mod.rs
pub mod event_monitor_actor;
pub mod pty;

#[cfg(target_os = "linux")]
pub mod epoll;
#[cfg(target_os = "linux")]
pub use epoll as event;

#[cfg(target_os = "macos")]
pub mod kqueue;
#[cfg(target_os = "macos")]
pub use kqueue as event;

// Platform-agnostic event flags
use bitflags::bitflags;

bitflags! {
    /// Platform-agnostic flags for monitoring file descriptors.
    /// These map to kqueue/epoll flags depending on the platform.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
    pub struct EventFlags: u16 {
        /// Monitor for read readiness
        const READ = 0x0001;
        /// Monitor for write readiness
        const WRITE = 0x0002;
    }
}

#[cfg(target_os = "macos")]
impl From<EventFlags> for kqueue::KqueueFlags {
    fn from(flags: EventFlags) -> Self {
        let mut kq_flags = kqueue::KqueueFlags::empty();
        if flags.contains(EventFlags::READ) {
            kq_flags |= kqueue::KqueueFlags::EPOLLIN;
        }
        if flags.contains(EventFlags::WRITE) {
            kq_flags |= kqueue::KqueueFlags::EPOLLOUT;
        }
        kq_flags
    }
}

#[cfg(target_os = "linux")]
impl From<EventFlags> for epoll::EpollFlags {
    fn from(flags: EventFlags) -> Self {
        let mut ep_flags = epoll::EpollFlags::empty();
        if flags.contains(EventFlags::READ) {
            ep_flags |= epoll::EpollFlags::EPOLLIN;
        }
        if flags.contains(EventFlags::WRITE) {
            ep_flags |= epoll::EpollFlags::EPOLLOUT;
        }
        ep_flags
    }
}

#[derive(Debug, Clone)]
pub enum PtyActionCommand {
    Write(Vec<u8>),
    Resize { cols: u16, rows: u16 },
}
