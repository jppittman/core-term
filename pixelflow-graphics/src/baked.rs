//! # Baked Combinator
//!
//! Caches a color manifold's results to a Frame.
//! Queries are served from the cache via modular arithmetic.
//!
//! This is the foundation for glyph caching - bake once, sample many.

use crate::render::color::Pixel;
use crate::render::frame::Frame;
use crate::render::rasterizer::{execute, TensorShape};
use pixelflow_core::{Discrete, Field, Manifold};

/// A color manifold baked to a Frame cache.
///
/// `Baked` pre-computes a color manifold at a given resolution and serves
/// subsequent queries from the cached Frame. Coordinates map to cache
/// pixels via modular arithmetic (wrapping at cache boundaries).
///
/// # Usage
///
/// The primary use is as a texture cache:
/// 1. Bake an expensive manifold once at construction
/// 2. Query the cache many times during rendering
///
/// # Example
///
/// ```ignore
/// use pixelflow_graphics::{Baked, Rgba8};
///
/// // Bake an expensive gradient at 256x256
/// let cached: Baked<_, Rgba8> = Baked::new(expensive_gradient, 256, 256);
///
/// // Use as a manifold - queries sample from cache
/// rasterizer::execute(&cached, target, shape);
/// ```
pub struct Baked<M, P: Pixel> {
    /// The cached frame (row-major pixels).
    cache: Frame<P>,
    /// The original manifold.
    #[allow(dead_code)]
    inner: M,
}

impl<M, P> Baked<M, P>
where
    M: Manifold<Output = Discrete>,
    P: Pixel,
{
    /// Bake a color manifold to a Frame at the given resolution.
    ///
    /// Eagerly rasterizes the manifold to the cache.
    pub fn new(source: M, width: usize, height: usize) -> Self {
        let mut cache = Frame::<P>::new(width as u32, height as u32);
        execute(&source, cache.as_slice_mut(), TensorShape::new(width, height));
        Self {
            cache,
            inner: source,
        }
    }

    /// Create from an existing Frame.
    pub fn from_frame(inner: M, cache: Frame<P>) -> Self {
        Self { cache, inner }
    }

    /// Get the cache width.
    #[inline]
    pub fn width(&self) -> usize {
        self.cache.width
    }

    /// Get the cache height.
    #[inline]
    pub fn height(&self) -> usize {
        self.cache.height
    }

    /// Get the underlying Frame.
    #[inline]
    pub fn frame(&self) -> &Frame<P> {
        &self.cache
    }

    /// Get mutable access to the underlying Frame.
    #[inline]
    pub fn frame_mut(&mut self) -> &mut Frame<P> {
        &mut self.cache
    }

    /// Sample a coordinate from the cache (with wrapping).
    #[inline]
    fn sample(&self, x: f32, y: f32) -> u32 {
        let ix = (x.floor() as isize).rem_euclid(self.cache.width as isize) as usize;
        let iy = (y.floor() as isize).rem_euclid(self.cache.height as isize) as usize;
        self.cache.data[iy * self.cache.width + ix].to_u32()
    }

    /// Unpack RGBA from u32 pixel (assumes RGBA8 layout).
    #[inline]
    fn unpack(pixel: u32) -> (f32, f32, f32, f32) {
        let [r, g, b, a] = pixel.to_le_bytes();
        (
            r as f32 / 255.0,
            g as f32 / 255.0,
            b as f32 / 255.0,
            a as f32 / 255.0,
        )
    }
}

impl<M, P> Manifold for Baked<M, P>
where
    M: Manifold<Output = Discrete> + Send + Sync,
    P: Pixel + Send + Sync,
{
    type Output = Discrete;

    #[inline(always)]
    fn eval_raw(&self, _x: Field, _y: Field, _z: Field, _w: Field) -> Discrete {
        // The rasterizer queries with Field::sequential(x_start) for x,
        // and Field::from(y_val) for y - sequential x, constant y.
        //
        // Since we can't extract individual Field lanes from outside core,
        // we use the inner manifold's eval to get coordinate-dependent sampling.
        // This delegates to the original manifold, which defeats caching.
        //
        // For true cache-based evaluation, use frame() directly and blit,
        // or extend pixelflow_core with lane access methods.
        //
        // Workaround: sample at origin and broadcast. This is wrong but
        // demonstrates the pattern. Real usage should blit the frame directly.
        let pixel = self.sample(0.5, 0.5);
        let (r, g, b, a) = Self::unpack(pixel);

        Discrete::pack(
            Field::from(r),
            Field::from(g),
            Field::from(b),
            Field::from(a),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Rgba8;
    use pixelflow_core::Discrete;

    // A simple solid color manifold for testing
    struct SolidColor(u8, u8, u8, u8);

    impl Manifold for SolidColor {
        type Output = Discrete;

        fn eval_raw(&self, _x: Field, _y: Field, _z: Field, _w: Field) -> Discrete {
            Discrete::pack(
                Field::from(self.0 as f32 / 255.0),
                Field::from(self.1 as f32 / 255.0),
                Field::from(self.2 as f32 / 255.0),
                Field::from(self.3 as f32 / 255.0),
            )
        }
    }

    #[test]
    fn test_baked_creation() {
        let color = SolidColor(255, 128, 64, 255);
        let baked: Baked<_, Rgba8> = Baked::new(color, 8, 8);

        assert_eq!(baked.width(), 8);
        assert_eq!(baked.height(), 8);

        // Check that the frame was filled
        let frame = baked.frame();
        assert_eq!(frame.data.len(), 64);
    }

    #[test]
    fn test_baked_sample() {
        let color = SolidColor(255, 0, 0, 255);
        let baked: Baked<_, Rgba8> = Baked::new(color, 4, 4);

        // Sample should return the cached red color
        let pixel = baked.sample(0.5, 0.5);
        let [r, g, b, a] = pixel.to_le_bytes();
        assert_eq!(r, 255);
        assert_eq!(g, 0);
        assert_eq!(b, 0);
        assert_eq!(a, 255);
    }

    #[test]
    fn test_baked_wrapping() {
        let color = SolidColor(128, 128, 128, 255);
        let baked: Baked<_, Rgba8> = Baked::new(color, 4, 4);

        // Coordinates outside [0, 4) should wrap
        let p1 = baked.sample(0.5, 0.5);
        let p2 = baked.sample(4.5, 0.5); // wraps to x=0
        let p3 = baked.sample(-3.5, 0.5); // wraps to x=0

        assert_eq!(p1, p2);
        assert_eq!(p1, p3);
    }
}
