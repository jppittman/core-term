pub mod waker;

#[cfg(target_os = "macos")]
pub mod macos;

#[cfg(target_os = "linux")]
pub mod linux;

#[cfg(target_os = "macos")]
pub use macos::*;

#[cfg(target_os = "linux")]
pub use linux::*;

// Platform-appropriate pixel format
#[cfg(target_os = "macos")]
pub type PlatformPixel = pixelflow_graphics::render::color::Rgba8;

#[cfg(target_os = "linux")]
pub type PlatformPixel = pixelflow_graphics::render::color::Bgra8;

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
pub type PlatformPixel = pixelflow_graphics::render::color::Rgba8;

pub type PlatformDriver = crate::display::driver::DriverActor<ActivePlatform>;

// Re-export platform-specific types
#[cfg(target_os = "macos")]
pub type ActivePlatform = crate::display::platform::PlatformActor<macos::MetalOps>;

#[cfg(target_os = "linux")]
pub type ActivePlatform = crate::display::platform::PlatformActor<linux::LinuxOps>;
