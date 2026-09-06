// pixelflow-render/src/frame.rs
//! Framebuffer type for rendered output.
//!
//! Frame is both a target (write into via execute) AND a Surface (read from).
//! This enables Frame-to-Frame compositing operations.

use super::color::Pixel;

/// A framebuffer of pixels in a specific format.
///
/// Frames are the result of `materialize` operations.
#[derive(Clone, Debug)]
pub struct Frame<P: Pixel> {
    /// Width in pixels
    pub width: usize,
    /// Height in pixels
    pub height: usize,
    /// Pixel data (row-major)
    pub data: Vec<P>,
}

impl<P: Pixel> Frame<P> {
    /// Create a new frame filled with the default pixel (typically black/transparent).
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        let size = (width as usize) * (height as usize);
        let data = vec![P::default(); size];
        Self {
            data,
            width: width as usize,
            height: height as usize,
        }
    }

    /// Create a frame from existing pixel data.
    ///
    /// # Panics
    /// Panics if data length doesn't match width * height.
    #[must_use]
    pub fn from_data(data: Vec<P>, width: u32, height: u32) -> Self {
        assert_eq!(data.len(), (width as usize) * (height as usize));
        Self {
            data,
            width: width as usize,
            height: height as usize,
        }
    }

    /// Convert to a different pixel format.
    pub fn convert<D: Pixel + From<P>>(self) -> Frame<D> {
        let data: Vec<D> = self.data.into_iter().map(D::from).collect();
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
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                self.data.as_ptr() as *const u8,
                self.data.len() * std::mem::size_of::<P>(),
            )
        }
    }

    /// Get mutable raw bytes.
    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        unsafe {
            std::slice::from_raw_parts_mut(
                self.data.as_mut_ptr() as *mut u8,
                self.data.len() * std::mem::size_of::<P>(),
            )
        }
    }

    /// This frame's pixels as the packed `u32` words they are.
    ///
    /// # Panics
    ///
    /// Panics unless `P` has `u32`'s layout — `Rgba8` and `Bgra8` are
    /// `#[repr(transparent)]` wrappers of one. Both size and alignment are
    /// checked: a `P` of four `u8`s would be the right size and the wrong
    /// alignment, and the cast below assumes both.
    #[must_use]
    pub fn as_u32_slice(&self) -> &[u32] {
        Self::assert_u32_layout();
        unsafe { std::slice::from_raw_parts(self.data.as_ptr() as *const u32, self.data.len()) }
    }

    /// [`Frame::as_u32_slice`], mutably — how a renderer that produces packed
    /// words writes them straight into the frame.
    ///
    /// # Panics
    ///
    /// Panics unless `P` has `u32`'s layout.
    pub fn as_u32_slice_mut(&mut self) -> &mut [u32] {
        Self::assert_u32_layout();
        unsafe {
            std::slice::from_raw_parts_mut(self.data.as_mut_ptr() as *mut u32, self.data.len())
        }
    }

    /// # Panics
    ///
    /// Panics unless `P` has `u32`'s size and alignment.
    fn assert_u32_layout() {
        assert_eq!(
            (std::mem::size_of::<P>(), std::mem::align_of::<P>(),),
            (std::mem::size_of::<u32>(), std::mem::align_of::<u32>()),
            "{} is not laid out as a packed u32 word",
            std::any::type_name::<P>()
        );
    }

    /// Get mutable pixel slice (for execute).
    pub fn as_slice_mut(&mut self) -> &mut [P] {
        &mut self.data
    }

    /// Get immutable pixel slice.
    #[must_use]
    pub fn as_slice(&self) -> &[P] {
        &self.data
    }
}
