// src/display/drivers/mod.rs
//! Platform-specific display driver implementations.

#[cfg(target_os = "macos")]
pub mod cocoa;

#[cfg(target_os = "macos")]
pub use cocoa::CocoaDisplayDriver;
