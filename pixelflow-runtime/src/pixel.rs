//! Platform-specific pixel type definition.

// Platform-appropriate pixel format
#[cfg(target_os = "macos")]
pub type PlatformPixel = pixelflow_graphics::render::color::Rgba8;

#[cfg(target_os = "linux")]
pub type PlatformPixel = pixelflow_graphics::render::color::Bgra8;

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
pub type PlatformPixel = pixelflow_graphics::render::color::Rgba8;

// Platform-appropriate ColorCube (re-export from pixelflow-graphics)
pub type ColorCube = pixelflow_graphics::PlatformColorCube;
