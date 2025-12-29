//! Compositional animation combinators.
//!
//! Build animations by composing manifolds with time-varying transformations.
//! Time flows through the W dimension - wrap any manifold to modify how it evolves.

use pixelflow_core::jet::Jet3;
use pixelflow_core::{Field, Manifold};

// ============================================================================
// Time Shifting
// ============================================================================

/// Translate the W (time) dimension.
///
/// Wraps a time-dependent manifold and shifts the time parameter before evaluation.
/// Used to set the "current time" in the animation system.
///
/// # Example
/// ```ignore
/// let scene = TimeShift {
///     inner: scene,
///     offset: current_time,
/// };
/// ```
#[derive(Clone, Copy, Debug)]
pub struct TimeShift<M> {
    pub inner: M,
    pub offset: f32,
}

impl<M: Manifold<Output = O>, O> Manifold for TimeShift<M>
where
    O: Manifold<Output = O>,
{
    type Output = O;

    #[inline(always)]
    fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> O {
        self.inner.eval_raw(x, y, z, w + Field::from(self.offset))
    }
}

// ============================================================================
// Screen Coordinate Remapping
// ============================================================================

/// Normalize screen coordinates to device space.
///
/// Remaps pixel coordinates (0..width, 0..height) to normalized device coordinates
/// suitable for camera systems (e.g., for perspective projection, ~60° FOV).
///
/// Maps the screen to [−aspect*scale, aspect*scale] × [−scale, scale] where
/// scale = 2.0/height.
///
/// # Example
/// ```ignore
/// let scene = ScreenRemap {
///     inner: scene,
///     width: 1920.0,
///     height: 1080.0,
/// };
/// ```
#[derive(Clone, Copy, Debug)]
pub struct ScreenRemap<M> {
    pub inner: M,
    pub width: f32,
    pub height: f32,
}

impl<M: Manifold<Output = O>, O: Manifold<Output = O>> Manifold for ScreenRemap<M> {
    type Output = O;

    #[inline(always)]
    fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> O {
        let scale = 2.0 / self.height;
        let sx = (x - Field::from(self.width * 0.5)) * Field::from(scale);
        let sy = (Field::from(self.height * 0.5) - y) * Field::from(scale);
        self.inner.eval_raw(sx, sy, z, w)
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
