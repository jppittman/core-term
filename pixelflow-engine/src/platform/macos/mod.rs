//! macOS platform support.

pub mod cocoa;
pub mod events;
pub mod platform;
pub mod sys;
pub mod window;

pub use platform::MetalPlatform;
