//! Scenes for tests that care about which port fired, not what is drawn.

use pixelflow_core::Kernel;
use pixelflow_graphics::render::scene::{compile_platform_packed, Scene};

/// A solid opaque colour as a packed program over a `frame[0] × frame[1]`
/// lattice: four constant channel kernels, nothing bound.
///
/// The simplest scene there is, and the shape every scene has — four channel
/// kernels and a frame — so a fixture built from it exercises the path
/// production takes rather than the lane on its way out.
///
/// # Panics
///
/// Panics on a degenerate frame extent, or if the kernel fails to compile.
#[must_use]
pub fn constant_scene(rgba: [f32; 4], frame: [u32; 2]) -> Scene {
    let channels = rgba.map(Kernel::constant);
    Scene::Packed(compile_platform_packed(&channels, frame).bind(&[]))
}

/// [`constant_scene`] in opaque black — what a test that only cares about the
/// protocol should hand the engine.
///
/// # Panics
///
/// Panics on a degenerate frame extent, or if the kernel fails to compile.
#[must_use]
pub fn black_scene(frame: [u32; 2]) -> Scene {
    constant_scene([0.0, 0.0, 0.0, 1.0], frame)
}
