// pixelflow-render/src/frame.rs
//! Framebuffer type for rendered output.
//!
//! Frame is both a target (write into via execute) AND a Surface (read from).
//! This enables Frame-to-Frame compositing operations.

use crate::color::Pixel;
use pixelflow_core::pipe::Surface;
use pixelflow_core::Batch;

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
        let w = self.width as usize;
        let h = self.height as usize;

        // Extract coordinates and compute wrap-around element-wise
        let x_arr = x.to_array_usize();
        let y_arr = y.to_array_usize();

        // Calculate wrapped indices
        let idx0 = (y_arr[0] % h) * w + (x_arr[0] % w);
        let idx1 = (y_arr[1] % h) * w + (x_arr[1] % w);
        let idx2 = (y_arr[2] % h) * w + (x_arr[2] % w);
        let idx3 = (y_arr[3] % h) * w + (x_arr[3] % w);

        // Gather pixels from buffer as u32
        let p0 = self.data[idx0].to_u32();
        let p1 = self.data[idx1].to_u32();
        let p2 = self.data[idx2].to_u32();
        let p3 = self.data[idx3].to_u32();

        // Create u32 batch and transmute to P (P is repr(transparent) over u32)
        let result_u32: Batch<u32> = Batch::new(p0, p1, p2, p3);
        result_u32.transmute()
    }
}
