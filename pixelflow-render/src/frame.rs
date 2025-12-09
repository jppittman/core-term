// pixelflow-render/src/frame.rs
//! Framebuffer type for rendered output.
//!
//! Frame is both a target (write into via execute) AND a Surface (read from).
//! This enables Frame-to-Frame compositing operations.

use crate::color::Pixel;
use pixelflow_core::batch::Batch;
use pixelflow_core::traits::Surface;
use pixelflow_core::SimdBatch;

/// A framebuffer of pixels in a specific format.
///
/// Generic over pixel type for compile-time format safety.
/// Use `Frame<Rgba>` for web/standard APIs, `Frame<Bgra>` for X11.
#[derive(Debug, Clone)]
pub struct Frame<P: Pixel> {
    /// Pixel data.
    pub data: Box<[P]>,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
}

impl<P: Pixel> Frame<P> {
    /// Create a new frame filled with the default pixel (typically black/transparent).
    pub fn new(width: u32, height: u32) -> Self {
        let size = (width as usize) * (height as usize);
        let data = vec![P::default(); size].into_boxed_slice();
        Self {
            data,
            width,
            height,
        }
    }

    /// Create a frame from existing pixel data.
    ///
    /// # Panics
    /// Panics if data length doesn't match width * height.
    pub fn from_data(data: Box<[P]>, width: u32, height: u32) -> Self {
        assert_eq!(data.len(), (width as usize) * (height as usize));
        Self {
            data,
            width,
            height,
        }
    }

    /// Create a frame from raw RGBA bytes (reinterpreted as pixels).
    ///
    /// # Panics
    /// Panics if byte length doesn't match width * height * 4.
    ///
    /// # Safety
    /// Assumes the bytes represent valid RGBA pixels in memory order.
    pub fn from_bytes(bytes: Vec<u8>, width: u32, height: u32) -> Self
    where
        P: Copy,
    {
        let expected_len = (width as usize) * (height as usize) * 4;
        assert_eq!(bytes.len(), expected_len, "Byte length must match width * height * 4");

        // Reinterpret bytes as pixels (u32)
        let pixels: Vec<P> = unsafe {
            let ptr = bytes.as_ptr() as *const P;
            let len = bytes.len() / 4;
            std::slice::from_raw_parts(ptr, len).to_vec()
        };
        std::mem::forget(bytes); // Don't drop the original Vec

        Self::from_data(pixels.into_boxed_slice(), width, height)
    }

    /// Convert to a different pixel format.
    pub fn convert<D: Pixel + From<P>>(self) -> Frame<D> {
        let data: Box<[D]> = self.data.iter().map(|&p| D::from(p)).collect();
        Frame {
            data,
            width: self.width,
            height: self.height,
        }
    }

    /// Get raw bytes (for passing to platform APIs).
    ///
    /// # Safety
    /// The returned slice aliases self.data. Don't use both simultaneously.
    pub fn as_bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.data.as_ptr() as *const u8, self.data.len() * 4) }
    }

    /// Get mutable raw bytes.
    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        unsafe {
            std::slice::from_raw_parts_mut(self.data.as_mut_ptr() as *mut u8, self.data.len() * 4)
        }
    }

    /// Get pixels as u32 slice (for rasterizer).
    pub fn as_u32_slice(&self) -> &[u32] {
        unsafe { std::slice::from_raw_parts(self.data.as_ptr() as *const u32, self.data.len()) }
    }

    /// Get mutable pixels as u32 slice (for rasterizer).
    pub fn as_u32_slice_mut(&mut self) -> &mut [u32] {
        unsafe {
            std::slice::from_raw_parts_mut(self.data.as_mut_ptr() as *mut u32, self.data.len())
        }
    }

    /// Get mutable pixel slice (for execute).
    pub fn as_slice_mut(&mut self) -> &mut [P] {
        &mut self.data
    }

    /// Get immutable pixel slice.
    pub fn as_slice(&self) -> &[P] {
        &self.data
    }
}

// =============================================================================
// Frame as Surface (wrap-around sampling)
// =============================================================================

/// Frame implements Surface with wrap-around out-of-bounds behavior.
///
/// This enables using a Frame as a compositing source:
/// ```ignore
/// let bg = Frame::<Rgba>::load("background.png");
/// let composed = mask.over::<Rgba>(foreground, &bg);
/// ```
impl<P: Pixel> Surface<P> for Frame<P> {
    #[inline(always)]
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<P> {
        use pixelflow_core::backend::BatchArithmetic;

        let w_batch = Batch::<u32>::splat(self.width);
        let h_batch = Batch::<u32>::splat(self.height);

        // Fast path: check if all coordinates are in bounds
        // This avoids expensive division for the common case
        let x_in_bounds = x.cmp_lt(w_batch);
        let y_in_bounds = y.cmp_lt(h_batch);
        let all_in_bounds = x_in_bounds.all() && y_in_bounds.all();

        let (x_final, y_final) = if all_in_bounds {
            // Fast path: no wrapping needed
            (x, y)
        } else {
            // Slow path: wrap coordinates (x % w, y % h)
            // Division is expensive on ARM NEON (falls back to scalar)
            let x_mod = x - (x / w_batch) * w_batch;
            let y_mod = y - (y / h_batch) * h_batch;
            (x_mod, y_mod)
        };

        // Linear index: y * w + x
        let idx = y_final * w_batch + x_final;

        // Gather pixels
        P::batch_gather(&self.data, idx)
    }
}

// Note: Arc<Frame<P>> automatically implements Surface via blanket impl in pixelflow-core
