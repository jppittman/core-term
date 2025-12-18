// src/display/mod.rs
//! Channel-based display system.
//!
//! - DisplayDriver: Platform-specific driver (Cocoa, X11, etc.)
//! - Messages: Display events and render snapshots

// pub mod generic;
pub mod ops;
pub mod platform;

pub mod driver;
pub mod drivers;
pub mod messages;

pub use driver::DriverActor; // Changed from DisplayDriver
pub use messages::DisplayEvent;
pub use platform::Platform;

// Re-export the active display driver for convenience
// #[cfg(use_cocoa_display)]
// pub use drivers::MetalDisplayDriver;

#[cfg(use_x11_display)]
pub use drivers::X11DisplayDriver;

#[cfg(use_headless_display)]
pub use drivers::HeadlessDisplayDriver;

#[cfg(use_web_display)]
pub use drivers::WebDisplayDriver;
