// pixelflow-render/src/frame.rs
//! Framebuffer type for rendered output.

use crate::color::Pixel;

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
        Self { data, width, height }
    }

    /// Create a frame from existing pixel data.
    ///
    /// # Panics
    /// Panics if data length doesn't match width * height.
    pub fn from_data(data: Box<[P]>, width: u32, height: u32) -> Self {
        assert_eq!(data.len(), (width as usize) * (height as usize));
        Self { data, width, height }
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
        unsafe {
            std::slice::from_raw_parts(
                self.data.as_ptr() as *const u8,
                self.data.len() * 4,
            )
        }
    }

    /// Get mutable raw bytes.
    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        unsafe {
            std::slice::from_raw_parts_mut(
                self.data.as_mut_ptr() as *mut u8,
                self.data.len() * 4,
            )
        }
    }

    /// Get pixels as u32 slice (for rasterizer).
    pub fn as_u32_slice(&self) -> &[u32] {
        unsafe {
            std::slice::from_raw_parts(
                self.data.as_ptr() as *const u32,
                self.data.len(),
            )
        }
    }

    /// Get mutable pixels as u32 slice (for rasterizer).
    pub fn as_u32_slice_mut(&mut self) -> &mut [u32] {
        unsafe {
            std::slice::from_raw_parts_mut(
                self.data.as_mut_ptr() as *mut u32,
                self.data.len(),
            )
        }
    }
}
