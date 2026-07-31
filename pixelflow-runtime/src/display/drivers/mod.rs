//! Display driver implementations.
//!
//! Display drivers are selected at build time based on:
//! - DISPLAY_DRIVER environment variable
//! - Cargo features (display_cocoa, display_x11)
//! - Target OS defaults

#[cfg(use_cocoa_display)]
pub mod metal;
#[cfg(use_cocoa_display)]
pub use metal::MetalDisplayDriver;

// #[cfg(use_x11_display)]
// pub mod x11;
// #[cfg(use_x11_display)]
// pub use x11::X11DisplayDriver;

#[cfg(use_web_display)]
pub mod web;
#[cfg(use_web_display)]
pub use web::WebDisplayDriver;
