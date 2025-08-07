// src/platform/mod.rs
//
// This module re-exports the platform-specific functionalities.

pub use backends::BackendEvent;

pub mod actions;
pub mod backends;
#[cfg(target_os = "linux")]
pub mod console_platform;
pub mod font_manager;
#[cfg(target_os = "linux")]
pub mod linux_x11;
pub mod macos; // Add this line
pub mod os;
pub mod platform_trait;
pub use macos::MacosPlatform; // Add this line

#[cfg(test)]
pub mod mock;

pub enum PlatformEvent {
    BackendEvent(BackendEvent),
    IOEvent { data: Vec<u8> },
}

impl From<BackendEvent> for PlatformEvent {
    fn from(b: BackendEvent) -> Self {
        PlatformEvent::BackendEvent(b)
    }
}
