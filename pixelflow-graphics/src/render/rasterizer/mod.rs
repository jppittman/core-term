//! # Software Rasterizer: Manifold → Framebuffer
//!
//! The final stage of the graphics pipeline: converts continuous algebraic manifolds into discrete pixel buffers.
//!
//! ## Architecture: Pull-Based Rendering
//!
//! ```text
//! Manifold (pure function)        PixelBuffer (concrete data)
//!      ↓                                   ↓
//!      │ Rasterizer pulls                 │
//!      │ at pixel coordinates             │
//!      ↓                                   ↓
//! eval(x, y) → Discrete(u32)    pixel[y*w + x] ← u32
//! ```
//!
//! Unlike immediate-mode renderers that "push" pixels to a buffer, the rasterizer **pulls** from the manifold.
//! For each pixel coordinate, it calls `manifold.eval_raw(x, y, z, w)` and stores the result.
//!
//! ## The Two Functions
//!
//! ### execute()
//! **Full framebuffer rendering** - rasterizes the entire buffer in one call.
//!
//! ```ignore
//! use pixelflow_graphics::render::{execute, TensorShape};
//!
//! let manifold = /* your color manifold */;
//! let mut framebuffer = vec![Rgba8(0); 800 * 600];
//!
//! execute(&manifold, &mut framebuffer, TensorShape::new(800, 600));
//! // Framebuffer is now filled with rendered pixels
//! ```
//!
//! **Contract:**
//! - **Precondition**: `target.len() >= shape.width * shape.height`
//! - **Action**: Evaluates manifold at every pixel coordinate (x, y)
//! - **Postcondition**: `target` is fully populated with pixels
//! - **Complexity**: O(width × height × eval_time)
//!
//! ### execute_stripe()
//! **Partial rendering** - rasterizes a horizontal band (row range).
//!
//! ```ignore
//! use pixelflow_graphics::render::{execute_stripe, Stripe};
//!
//! // Render rows 100-200
//! let stripe = Stripe { start_y: 100, end_y: 200 };
//! execute_stripe(&manifold, &mut framebuffer, 800, stripe);
//! ```
//!
//! **Contract:**
//! - **Precondition**: `target.len() >= stripe.end_y * width - stripe.start_y * width`
//! - **Action**: Evaluates manifold for rows [start_y, end_y)
//! - **Postcondition**: Target rows are updated; other rows unchanged
//! - **Use case**: Parallel rendering (different threads render different stripes)
//!
//! ## Sampling Strategy: Pixel Centers
//!
//! The rasterizer samples manifolds at **pixel center** coordinates:
//!
//! ```text
//! Pixel (x, y) is sampled at coordinates (x + 0.5, y + 0.5)
//!
//! Example:
//! │ Pixel (0, 0) sampled at (0.5, 0.5)
//! │ Pixel (1, 0) sampled at (1.5, 0.5)
//! │ Pixel (0, 1) sampled at (0.5, 1.5)
//! ```
//!
//! Why pixel centers?
//! - **Antialiasing**: Continuous sampling (e.g., with Jet2 for gradients) can benefit from offsets
//! - **Consistency**: Matches standard graphics conventions (OpenGL, WebGL, etc.)
//! - **Numerical stability**: Avoids integer grid artifacts
//!
//! ## SIMD Hot Path
//!
//! The core loop uses **vectorized evaluation**:
//!
//! ```text
//! Normal execution (pseudo-code):
//!
//! for y in 0..height {
//!     for x in (0..width).step_by(PARALLELISM) {
//!         // Evaluate PARALLELISM pixels at once
//!         xs = [x+0.5, x+1.5, x+2.5, ..., x+PARALLELISM-0.5]  (as Field)
//!         ys = y + 0.5                                         (as Field)
//!
//!         colors = manifold.eval_raw(xs, ys, 0.0, 0.0)  // SIMD!
//!         target[y*width + x..][0..PARALLELISM] = colors
//!     }
//! }
//! ```
//!
//! - **PARALLELISM**: Number of simultaneous SIMD lanes (16-64 depending on CPU)
//! - **Sequential generation**: `Field::sequential(x)` creates `[x, x+1, x+2, ..., x+PARALLELISM-1]` for batch evaluation
//! - **Fused evaluation**: Manifold is compiled into a single SIMD kernel per expression
//!
//! ### Why This Works
//!
//! The manifold's type encodes the computation graph. When you call `eval_raw(xs, ys, ...)`:
//! 1. `xs` and `ys` are SIMD vectors (e.g., `[x₀, x₁, ..., x₁₅]` on AVX-512)
//! 2. All arithmetic (`+`, `-`, `*`, `/`) operates element-wise on SIMD vectors
//! 3. The compiler inlines and vectorizes the entire expression
//! 4. Result: A single CPU-efficient SIMD loop with zero branches
//!
//! ## Scalar Fallback (Tail)
//!
//! For pixels that don't fit evenly into SIMD lanes:
//!
//! ```text
//! If width = 100 and PARALLELISM = 16:
//! - Main loop: process pixels 0-79 (5 iterations of 16)
//! - Tail loop: process pixels 80-99 (one by one)
//! ```
//!
//! The tail loop evaluates pixels individually, storing them in a 1-element array.
//! For most cases, the tail is < 10% of the buffer and negligible in performance.
//!
//! ## Memory Layout: Row-Major Order
//!
//! Pixels are stored in **row-major** (C-style) order:
//!
//! ```text
//! Index (y, x) maps to offset: y * width + x
//!
//! Example (4×3 framebuffer):
//! Pixel (0,0) → index 0
//! Pixel (1,0) → index 1
//! Pixel (2,0) → index 2
//! Pixel (3,0) → index 3
//! Pixel (0,1) → index 4   (next row)
//! ...
//! ```
//!
//! This matches the standard raster graphics convention and is cache-efficient (sequential memory access).
//!
//! ## Pixel Format Conversion
//!
//! The manifold produces `Discrete` (packed u32 RGBA), but the framebuffer may use a different format:
//!
//! ```text
//! Manifold Output: Discrete (always RGBA)
//!      ↓
//! Pixel Trait: Converts to target format
//!      ↓
//! Framebuffer: Rgba8, Bgra8, or other format
//! ```
//!
//! The `Pixel` trait handles:
//! - **Rgba8** (macOS, Web): Direct passthrough
//! - **Bgra8** (Linux X11): Byte swizzle (R and B channels)
//! - **Custom formats**: User-defined impl
//!
//! ```ignore
//! pub trait Pixel: Clone + Copy {
//!     fn from_u32(v: u32) -> Self;           // From Discrete
//!     fn to_u32(self) -> u32;                // To Discrete
//!     fn from_rgba(r: f32, g: f32, b: f32, a: f32) -> Self;
//! }
//! ```
//!
//! ## Error Handling
//!
//! - **Empty framebuffer** (width or height = 0): Early return (no-op)
//! - **Buffer too small**: Undefined behavior (writes past end) — caller responsible
//! - **Invalid coordinates**: Not possible (rasterizer always uses valid (x, y) pairs)
//!
//! ## Caching and Frame Reuse
//!
//! The rasterizer **does not cache** manifold evaluation. However:
//! - The manifold **may cache internally** (e.g., glyph caches in `CachedText`)
//! - Between frames with **identical size**, manifold evaluation is deterministic (no state)
//! - Platform rendering loop (cocoa, x11, web) typically caches and reuses results when possible
//!
//! ## Integration with Parallel Rendering
//!
//! See `parallel.rs` for multi-threaded rendering. The key insight:
//! - `execute()` is a convenience for single-threaded rendering
//! - `execute_stripe()` enables work-stealing: each thread renders a different vertical band
//! - No synchronization needed (stripes don't overlap)
//!
//! Color manifolds output `Discrete` (packed u32 pixels) directly.

use crate::render::color::Pixel;
use pixelflow_core::{
    materialize_discrete, materialize_discrete_fields, Discrete, Field, Manifold, PARALLELISM,
};

pub mod parallel;
pub(crate) mod pool;
pub use parallel::{render_parallel, render_work_stealing, RenderOptions};

/// A wrapper that adapts a continuous manifold for rasterization.
#[derive(Clone, Copy, Debug)]
pub struct Rasterize<M>(pub M);

impl<M: Manifold> Manifold for Rasterize<M> {
    type Output = M::Output;

    #[inline(always)]
    fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> Self::Output {
        self.0.eval_raw(x, y, z, w)
    }
}

/// Dimensions and memory layout of a 2D tensor.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct TensorShape {
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
}

impl TensorShape {
    /// Create a new TensorShape.
    pub const fn new(width: usize, height: usize) -> Self {
        Self { width, height }
    }
}

/// A horizontal stripe of rows for parallel rendering.
#[derive(Copy, Clone, Debug)]
pub struct Stripe {
    /// Starting Y coordinate.
    pub start_y: usize,
    /// Ending Y coordinate (exclusive).
    pub end_y: usize,
    /// Width of the stripe (stride).
    pub width: usize,
}

/// Software rasterizer entry point.
///
/// Takes a color manifold that outputs Discrete (packed u32 pixels)
/// and writes them to the target buffer.
pub fn execute<P, M>(manifold: &M, target: &mut [P], shape: TensorShape)
where
    P: Pixel,
    M: Manifold<Output = Discrete> + ?Sized,
{
    if shape.width == 0 || shape.height == 0 {
        return;
    }
    execute_stripe(
        manifold,
        target,
        Stripe {
            start_y: 0,
            end_y: shape.height,
            width: shape.width,
        },
    );
}

/// Render a specific row range into the target buffer.
pub fn execute_stripe<P, M>(manifold: &M, target: &mut [P], stripe: Stripe)
where
    P: Pixel,
    M: Manifold<Output = Discrete> + ?Sized,
{
    let mut packed = [0u32; PARALLELISM];

    for (row_idx, y) in (stripe.start_y..stripe.end_y).enumerate() {
        let row_offset = row_idx * stripe.width;
        let mut x = 0;

        let ys = Field::from(y as f32 + 0.5);

        // SIMD Hot Path - process PARALLELISM pixels at a time
        while x + PARALLELISM <= stripe.width {
            let xs = Field::sequential(x as f32 + 0.5);

            // Use materialize_discrete_fields to evaluate and store
            materialize_discrete_fields(manifold, xs, ys, &mut packed);

            // Copy to target
            for i in 0..PARALLELISM {
                target[row_offset + x + i] = P::from_u32(packed[i]);
            }

            x += PARALLELISM;
        }

        // Scalar Fallback (Tail) - handle remaining pixels one at a time
        while x < stripe.width {
            let fx = x as f32 + 0.5;
            let fy = y as f32 + 0.5;

            // For single pixels, use materialize_discrete with a 1-element buffer
            materialize_discrete(manifold, fx, fy, &mut packed);
            target[row_offset + x] = P::from_u32(packed[0]);

            x += 1;
        }
    }
}
