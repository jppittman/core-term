// src/platform/mod.rs
//
// This module re-exports the platform-specific functionalities.

pub use backends::BackendEvent;

pub mod actions;
pub mod backends;
pub mod console_platform;
pub mod linux_x11;
pub mod os;
pub mod platform_trait;

pub enum PlatformEvent {
    BackendEvent(BackendEvent),
    IOEvent { data: Vec<u8> },
}

impl From<BackendEvent> for PlatformEvent {
    fn from(b: BackendEvent) -> Self {
        PlatformEvent::BackendEvent(b)
    }
}
