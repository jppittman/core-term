// pixelflow-render/src/frame.rs
//! Framebuffer type for rendered output.
//!
//! Frame is both a target (write into via execute) AND a Surface (read from).
//! This enables Frame-to-Frame compositing operations.

use super::color::Pixel;

/// A framebuffer of pixels in a specific format.
///
/// Frames are the result of `materialize` operations.
/// Shared memory segment metadata for zero-copy rendering.
#[derive(Debug)]
pub struct ShmInfo {
    /// Segment ID (shmid from libc)
    pub shmid: i32,
    /// Attached address in client process
    pub shmaddr: usize,
    /// Size of the segment in bytes
    pub size: usize,
}

#[cfg(unix)]
impl Drop for ShmInfo {
    fn drop(&mut self) {
        unsafe {
            libc::shmdt(self.shmaddr as *mut std::ffi::c_void);
        }
    }
}

/// A safe, non-allocating wrapper around either heap or shared memory buffers.
#[derive(Debug)]
pub enum PixelBuffer<P: Pixel> {
    Heap(Vec<P>),
    Shared {
        shmaddr: *mut P,
        len: usize,
    },
}

impl<P: Pixel> Clone for PixelBuffer<P> {
    fn clone(&self) -> Self {
        Self::Heap(self.to_vec())
    }
}

impl<P: Pixel> std::ops::Deref for PixelBuffer<P> {
    type Target = [P];

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Heap(v) => v,
            Self::Shared { shmaddr, len } => unsafe {
                std::slice::from_raw_parts(*shmaddr, *len)
            },
        }
    }
}

impl<P: Pixel> std::ops::DerefMut for PixelBuffer<P> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            Self::Heap(v) => v,
            Self::Shared { shmaddr, len } => unsafe {
                std::slice::from_raw_parts_mut(*shmaddr, *len)
            },
        }
    }
}

impl<P: Pixel> PixelBuffer<P> {
    pub fn iter(&self) -> std::slice::Iter<'_, P> {
        use std::ops::Deref;
        self.deref().iter()
    }

    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, P> {
        use std::ops::DerefMut;
        self.deref_mut().iter_mut()
    }
}

impl<'a, P: Pixel> IntoIterator for &'a PixelBuffer<P> {
    type Item = &'a P;
    type IntoIter = std::slice::Iter<'a, P>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, P: Pixel> IntoIterator for &'a mut PixelBuffer<P> {
    type Item = &'a mut P;
    type IntoIter = std::slice::IterMut<'a, P>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

unsafe impl<P: Pixel + Send> Send for PixelBuffer<P> {}
unsafe impl<P: Pixel + Sync> Sync for PixelBuffer<P> {}

impl<P: Pixel + PartialEq> PartialEq for PixelBuffer<P> {
    fn eq(&self, other: &Self) -> bool {
        use std::ops::Deref;
        self.deref() == other.deref()
    }
}

#[derive(Debug)]
pub struct Frame<P: Pixel> {
    /// Width in pixels
    pub width: usize,
    /// Height in pixels
    pub height: usize,
    /// Pixel data (row-major)
    pub data: PixelBuffer<P>,
    /// Optional shared memory information for zero-copy rendering.
    pub shm_info: Option<ShmInfo>,
}

impl<P: Pixel> Frame<P> {
    /// Create a new frame filled with the default pixel (typically black/transparent).
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        let size = (width as usize) * (height as usize);
        let data = vec![P::default(); size];
        Self {
            data: PixelBuffer::Heap(data),
            width: width as usize,
            height: height as usize,
            shm_info: None,
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
            data: PixelBuffer::Heap(data),
            width: width as usize,
            height: height as usize,
            shm_info: None,
        }
    }

    /// Convert to a different pixel format.
    pub fn convert<D: Pixel + From<P>>(self) -> Frame<D> {
        let data: Vec<D> = self.data.iter().copied().map(D::from).collect();
        Frame {
            data: PixelBuffer::Heap(data),
            width: self.width,
            height: self.height,
            shm_info: None,
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

    /// Get pixels as u32 slice (for rasterizer).
    ///
    /// # Safety
    /// This function is safe only if `P` has the same memory layout as `u32`.
    /// For example, `Rgba` and `Bgra` are typically `u32` aliases.
    #[must_use]
    pub fn as_u32_slice(&self) -> &[u32] {
        assert_eq!(std::mem::size_of::<P>(), std::mem::size_of::<u32>());
        unsafe { std::slice::from_raw_parts(self.data.as_ptr() as *const u32, self.data.len()) }
    }

    /// Get mutable pixels as u32 slice (for rasterizer).
    ///
    /// # Safety
    /// This function is safe only if `P` has the same memory layout as `u32`.
    /// For example, `Rgba` and `Bgra` are typically `u32` aliases.
    pub fn as_u32_slice_mut(&mut self) -> &mut [u32] {
        assert_eq!(std::mem::size_of::<P>(), std::mem::size_of::<u32>());
        unsafe {
            std::slice::from_raw_parts_mut(self.data.as_mut_ptr() as *mut u32, self.data.len())
        }
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


impl<P: Pixel> Clone for Frame<P> {
    fn clone(&self) -> Self {
        Frame {
            width: self.width,
            height: self.height,
            data: self.data.clone(),
            shm_info: None,
        }
    }
}
