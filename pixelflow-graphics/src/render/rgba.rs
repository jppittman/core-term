//! # Color Manifold
//!
//! Colors as functors over (R, G, B, A) channel manifolds.
//!
//! A ColorManifold holds 4 scalar manifolds (one per channel) and outputs
//! `Discrete` (packed u32 RGBA pixels ready for framebuffer).
//!
//! Use `.red()`, `.green()`, `.blue()`, `.alpha()` to access channel manifolds for mapping.

use pixelflow_core::{Discrete, Field, Manifold};

/// Color manifold - holds 4 channel manifolds (R, G, B, A).
///
/// When evaluated, this packs the 4 channel values into `Discrete` (u32 RGBA pixels).
///
/// # Example
///
/// ```ignore
/// // Create a solid red color
/// let red = Color::new(1.0, 0.0, 0.0, 1.0);
///
/// // Create a gradient
/// let gradient = Color::new(X / 100.0, 0.5, 0.5, 1.0);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Color<R, G, B, A> {
    r: R,
    g: G,
    b: B,
    a: A,
}

impl<R, G, B, A> Color<R, G, B, A> {
    /// Create a new color manifold from 4 channel manifolds.
    pub fn new(r: R, g: G, b: B, a: A) -> Self {
        Self { r, g, b, a }
    }

    /// Access the red channel manifold.
    pub fn red(&self) -> &R {
        &self.r
    }

    /// Access the green channel manifold.
    pub fn green(&self) -> &G {
        &self.g
    }

    /// Access the blue channel manifold.
    pub fn blue(&self) -> &B {
        &self.b
    }

    /// Access the alpha channel manifold.
    pub fn alpha(&self) -> &A {
        &self.a
    }
}

impl<R, G, B, A> Manifold for Color<R, G, B, A>
where
    R: Manifold<Output = Field>,
    G: Manifold<Output = Field>,
    B: Manifold<Output = Field>,
    A: Manifold<Output = Field>,
{
    type Output = Discrete;

    #[inline(always)]
    fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> Discrete {
        let r = self.r.eval_raw(x, y, z, w);
        let g = self.g.eval_raw(x, y, z, w);
        let b = self.b.eval_raw(x, y, z, w);
        let a = self.a.eval_raw(x, y, z, w);
        Discrete::pack(r, g, b, a)
    }
}

/// Lift a scalar manifold to a grayscale color (same value in R, G, B, A=1).
#[derive(Clone, Copy, Debug)]
pub struct Lift<M>(pub M);

impl<M: Manifold<Output = Field> + Clone> Manifold for Lift<M> {
    type Output = Discrete;

    #[inline(always)]
    fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> Discrete {
        let v = self.0.eval_raw(x, y, z, w);
        Discrete::pack(v, v, v, Field::from(1.0))
    }
}

/// Map a function over all channels before packing.
#[derive(Clone, Copy, Debug)]
pub struct ColorMap<C, F> {
    color: C,
    func: F,
}

impl<R, G, B, A, F> ColorMap<Color<R, G, B, A>, F>
where
    R: Manifold<Output = Field>,
    G: Manifold<Output = Field>,
    B: Manifold<Output = Field>,
    A: Manifold<Output = Field>,
    F: Fn(Field) -> Field + Send + Sync + Copy,
{
    /// Create a new ColorMap.
    pub fn new(color: Color<R, G, B, A>, func: F) -> Self {
        Self { color, func }
    }
}

impl<R, G, B, A, F> Manifold for ColorMap<Color<R, G, B, A>, F>
where
    R: Manifold<Output = Field>,
    G: Manifold<Output = Field>,
    B: Manifold<Output = Field>,
    A: Manifold<Output = Field>,
    F: Fn(Field) -> Field + Send + Sync + Copy,
{
    type Output = Discrete;

    #[inline(always)]
    fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> Discrete {
        let r = (self.func)(self.color.r.eval_raw(x, y, z, w));
        let g = (self.func)(self.color.g.eval_raw(x, y, z, w));
        let b = (self.func)(self.color.b.eval_raw(x, y, z, w));
        let a = (self.func)(self.color.a.eval_raw(x, y, z, w));
        Discrete::pack(r, g, b, a)
    }
}
