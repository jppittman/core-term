// pixelflow-render/src/frame.rs
//! Framebuffer type for rendered output.
//!
//! Frame is both a target (write into via execute) AND a Surface (read from).
//! This enables Frame-to-Frame compositing operations.

use crate::color::Pixel;
use pixelflow_core::batch::Batch;
use pixelflow_core::traits::Manifold;
use pixelflow_core::SimdBatch;

/// A framebuffer of pixels in a specific format.
///
/// Generic over pixel type for compile-time format safety.
/// Use `Frame<Rgba>` for web/standard APIs, `Frame<Bgra>` for X11.
#[derive(Debug)]
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
// Frame as Manifold (wrap-around sampling)
// =============================================================================

/// Frame implements Manifold with wrap-around out-of-bounds behavior.
///
/// This enables using a Frame as a compositing source:
/// ```ignore
/// let bg = Frame::<Rgba>::load("background.png");
/// let composed = mask.over::<Rgba>(foreground, &bg);
/// ```
impl<P: Pixel> Manifold<P> for Frame<P> {
    #[inline(always)]
    fn eval(&self, x: Batch<u32>, y: Batch<u32>, _z: Batch<u32>, _w: Batch<u32>) -> Batch<P> {
        let w_batch = Batch::<u32>::splat(self.width);
        let h_batch = Batch::<u32>::splat(self.height);

        // Wrap coordinates (x % w, y % h)
        let x_mod = x - (x / w_batch) * w_batch;
        let y_mod = y - (y / h_batch) * h_batch;

        // Linear index: y * w + x
        let idx = y_mod * w_batch + x_mod;

        // Gather pixels
        P::batch_gather(&self.data, idx)
    }
}
