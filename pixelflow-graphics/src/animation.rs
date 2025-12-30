//! Compositional animation combinators.
//!
//! Build animations by composing manifolds with time-varying transformations.
//! Time flows through the W dimension - wrap any manifold to modify how it evolves.
//!
//! All combinators are purely compositional - they build AST nodes that are
//! evaluated lazily, enabling full optimization by LLVM.

use pixelflow_core::jet::Jet3;
use pixelflow_core::{Add, At, Field, Manifold, Mul, Sub, W, X, Y, Z};

// ============================================================================
// Time Shifting
// ============================================================================

/// Translate the W (time) dimension.
///
/// Type alias for `At<X, Y, Z, Add<W, f32>, M>`.
/// Purely compositional - the time offset becomes part of the AST.
pub type TimeShift<M> = At<X, Y, Z, Add<W, f32>, M>;

/// Create a time-shifted manifold.
///
/// Shifts the W coordinate by `offset` before evaluating the inner manifold.
/// Used to set the "current time" in the animation system.
///
/// # Example
/// ```ignore
/// let scene = time_shift(scene, current_time);
/// ```
pub fn time_shift<M>(inner: M, offset: f32) -> TimeShift<M> {
    At {
        inner,
        x: X,
        y: Y,
        z: Z,
        w: W + offset,
    }
}

// ============================================================================
// Screen Coordinate Remapping
// ============================================================================

/// Normalize screen coordinates to device space.
///
/// Type alias using At combinator for coordinate transformation.
/// Maps pixel coordinates (0..width, 0..height) to normalized device coordinates
/// suitable for camera systems (e.g., for perspective projection, ~60° FOV).
///
/// Maps the screen to [−aspect*scale, aspect*scale] × [−scale, scale] where
/// scale = 2.0/height.
pub type ScreenRemap<M> = At<
    Mul<Sub<X, Field>, Field>, // (x - width/2) * scale
    Mul<Sub<Field, Y>, Field>, // (height/2 - y) * scale
    Z,
    W,
    M,
>;

/// Create a screen-remapped manifold.
///
/// # Example
/// ```ignore
/// let scene = screen_remap(scene, 1920.0, 1080.0);
/// ```
pub fn screen_remap<M>(inner: M, width: f32, height: f32) -> ScreenRemap<M> {
    let scale = Field::from(2.0 / height);
    let half_width = Field::from(width * 0.5);
    let half_height = Field::from(height * 0.5);

    At {
        inner,
        x: (X - half_width) * scale,
        y: (half_height - Y) * scale,
        z: Z,
        w: W,
    }
}

// ============================================================================
// Oscillation
// ============================================================================

/// Oscillate a Jet3-based geometry with sinusoidal motion.
///
/// Produces time-varying ray intersection by modulating the geometry based on
/// sin(W * frequency). Works with any Jet3 input geometry.
///
/// The oscillation is expressed compositionally: the geometry receives
/// an offset computed from W, allowing animation via pure algebraic composition.
///
/// # Design
/// Uses `pixelflow_core::ops::Sin` to compose the sinusoid algebraically
/// rather than imperative computation.
///
/// # Example
/// For a full example showing an oscillating sphere, see `examples/animated_sphere.rs`.
/// The pattern is: wrap a geometry inside Surface, and use Sin(W * frequency)
/// to build time-varying transforms.
#[derive(Clone, Copy, Debug)]
pub struct Oscillate<G> {
    pub geometry: G,
    pub amplitude: f32,
    pub frequency: f32,
}

impl<G: Manifold<Jet3, Output = Jet3>> Oscillate<G> {
    /// Create an oscillating geometry wrapper.
    ///
    /// The oscillation function `sin(W * frequency) * amplitude` modulates
    /// the geometry based on the time dimension (W).
    pub fn new(geometry: G, amplitude: f32, frequency: f32) -> Self {
        Self {
            geometry,
            amplitude,
            frequency,
        }
    }
}
