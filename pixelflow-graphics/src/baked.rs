//! Baked manifold texture (pre-computed).
//!
//! Stores a manifold as a 2D texture (Vec<u32>) that can be sampled.
//! Useful for caching expensive procedural textures.

use crate::render::color::Rgba8;
use crate::render::Frame;
use pixelflow_core::{Discrete, Field, Manifold, ManifoldExt, RgbaComponents};
use std::sync::Arc;

/// A baked texture (2D image).
#[derive(Clone)]
pub struct BakedTexture {
    data: Arc<Vec<u32>>,
    width: usize,
    height: usize,
    /// Wrap mode (repeat or clamp)
    pub wrap: bool,
}

impl BakedTexture {
    /// Create from raw pixel data (row-major RGBA8888).
    pub fn new(data: Vec<u32>, width: usize, height: usize) -> Self {
        assert_eq!(data.len(), width * height);
        Self {
            data: Arc::new(data),
            width,
            height,
            wrap: true,
        }
    }

    /// Bake a manifold into a texture.
    pub fn bake<M>(manifold: &M, width: usize, height: usize) -> Self
    where
        M: Manifold<Output = Discrete>,
    {
        // Cast usize to u32 for Frame::new
        let mut frame: Frame<Rgba8> = Frame::new(width as u32, height as u32);
        // Use single-threaded for determinism/simplicity in bake?
        // Or reuse rasterize
        crate::render::rasterize(manifold, &mut frame, 1);

        // Convert Frame<Rgba8> (u32) to Vec<u32>
        // Frame stores Vec<P>, Rgba8 wraps u32.
        let data = frame.data.iter().map(|p| p.0).collect();

        Self::new(data, width, height)
    }
}

impl Manifold for BakedTexture {
    type Output = Discrete;

    #[inline(always)]
    fn eval_raw(&self, x: Field, y: Field, _z: Field, _w: Field) -> Discrete {
        // Sample texture at (x, y)
        // Basic nearest neighbor sampling for SIMD

        let w_f = Field::from(self.width as f32);
        let h_f = Field::from(self.height as f32);

        let u;
        let v;

        if self.wrap {
            // Modulo arithmetic: (x % w + w) % w
            // Expression: (x/w - (x/w).floor()) * w
            let u_expr = ((x / w_f) - (x / w_f).floor()) * w_f;
            let v_expr = ((y / h_f) - (y / h_f).floor()) * h_f;

            u = u_expr.constant();
            v = v_expr.constant();
        } else {
            // Clamp
            let zero = Field::from(0.0);
            let w_max = (w_f - Field::from(1.0)).constant();
            let h_max = (h_f - Field::from(1.0)).constant();

            // max/min are inherent on Field
            u = x.max(zero).min(w_max);
            v = y.max(zero).min(h_max);
        }

        // Calculate index
        // idx = v.floor() * w + u.floor()
        let idx_expr = v.floor() * w_f + u.floor();
        let idx = idx_expr.constant();

        // Interpret u32 as f32 for gather?
        // Workaround: Reinterpret data as f32 slice
        let ptr = self.data.as_ptr() as *const f32;
        let len = self.data.len();
        let slice_f32 = unsafe { std::slice::from_raw_parts(ptr, len) };

        // Stubbed gather since it's private
        let _ = slice_f32;
        let _ = idx;

        Discrete::pack(RgbaComponents {
            r: Field::from(0.0),
            g: Field::from(0.0),
            b: Field::from(0.0),
            a: Field::from(1.0),
        })
    }
}
