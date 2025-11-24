//! Display driver implementations.
//!
//! Display drivers are selected at build time based on:
//! - DISPLAY_DRIVER environment variable
//! - Cargo features (display_cocoa, display_x11, display_headless)
//! - Target OS defaults

#[cfg(use_cocoa_display)]
pub mod cocoa;
#[cfg(use_cocoa_display)]
pub use cocoa::CocoaDisplayDriver;

#[cfg(use_x11_display)]
pub mod x11;
#[cfg(use_x11_display)]
pub use x11::X11DisplayDriver;

#[cfg(use_headless_display)]
pub mod headless;
#[cfg(use_headless_display)]
pub use headless::HeadlessDisplayDriver;
