// src/platform/os/mod.rs
pub mod pty;

#[cfg(target_os = "linux")]
pub mod epoll;
#[cfg(target_os = "linux")]
pub use epoll as event;

#[cfg(target_os = "macos")]
pub mod kqueue;
#[cfg(target_os = "macos")]
pub use kqueue as event;

#[derive(Debug, Clone)]
pub enum PtyActionCommand {
    Write(Vec<u8>),
    Resize { cols: u16, rows: u16 },
}
