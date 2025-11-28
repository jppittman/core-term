// src/display/mod.rs
//! Message-based display system with Strategy Pattern.
//!
//! - DisplayDriver: Platform-specific primitives (Cocoa, X11, etc.)
//! - DisplayManager: Common logic and state management
//! - Messages: Request/Response protocol for communication

pub mod driver;
pub mod drivers;
pub mod manager;
pub mod messages;

pub use driver::DisplayDriver;
pub use manager::{DisplayManager, DisplayMetrics};
pub use messages::{DisplayError, DisplayEvent, DriverRequest, DriverResponse};

// Re-export the active display driver for convenience
#[cfg(use_cocoa_display)]
pub use drivers::CocoaDisplayDriver;

#[cfg(use_x11_display)]
pub use drivers::X11DisplayDriver;

#[cfg(use_headless_display)]
pub use drivers::HeadlessDisplayDriver;
