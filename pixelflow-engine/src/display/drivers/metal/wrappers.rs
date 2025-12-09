//! Type-safe Rust wrappers around Metal FFI.
//!
//! This module provides clean, readable Rust types that wrap the raw Objective-C
//! Metal API. All unsafe FFI calls are contained here.

use objc2::msg_send;
use objc2::runtime::{AnyObject, Sel};
use objc2::sel;
use std::ffi::c_void;

// ============================================================================
// Metal Types
// ============================================================================

/// Metal GPU device
pub struct MetalDevice {
    ptr: *mut AnyObject,
}

/// Metal buffer (shared CPU/GPU memory)
pub struct MetalBuffer {
    ptr: *mut AnyObject,
}

/// Metal command queue
pub struct MetalCommandQueue {
    ptr: *mut AnyObject,
}

/// Metal command buffer
pub struct MetalCommandBuffer {
    ptr: *mut AnyObject,
}

/// Metal texture
pub struct MetalTexture {
    ptr: *mut AnyObject,
}

/// Metal layer (CAMetalLayer)
pub struct MetalLayer {
    ptr: *mut AnyObject,
}

/// Metal drawable (current frame from layer)
pub struct MetalDrawable {
    ptr: *mut AnyObject,
}

// ============================================================================
// Metal Structs (for replaceRegion)
// ============================================================================

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct MTLOrigin {
    pub x: u64,
    pub y: u64,
    pub z: u64,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct MTLSize {
    pub width: u64,
    pub height: u64,
    pub depth: u64,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct MTLRegion {
    pub origin: MTLOrigin,
    pub size: MTLSize,
}

// Implement Encode for Metal structs
unsafe impl objc2::Encode for MTLOrigin {
    const ENCODING: objc2::Encoding = objc2::Encoding::Struct(
        "?",
        &[
            objc2::Encoding::ULongLong,
            objc2::Encoding::ULongLong,
            objc2::Encoding::ULongLong,
        ],
    );
}

unsafe impl objc2::Encode for MTLSize {
    const ENCODING: objc2::Encoding = objc2::Encoding::Struct(
        "?",
        &[
            objc2::Encoding::ULongLong,
            objc2::Encoding::ULongLong,
            objc2::Encoding::ULongLong,
        ],
    );
}

unsafe impl objc2::Encode for MTLRegion {
    const ENCODING: objc2::Encoding =
        objc2::Encoding::Struct("?", &[MTLOrigin::ENCODING, MTLSize::ENCODING]);
}

// ============================================================================
// FFI Bindings
// ============================================================================

#[link(name = "Metal", kind = "framework")]
#[link(name = "QuartzCore", kind = "framework")]
extern "C" {
    fn MTLCreateSystemDefaultDevice() -> *mut AnyObject;
}

// Storage modes for Metal buffers
pub const MTL_RESOURCE_STORAGE_MODE_SHARED: u64 = 0;

// Pixel formats
pub const MTL_PIXEL_FORMAT_BGRA8_UNORM: u64 = 80;

// ============================================================================
// MetalDevice Implementation
// ============================================================================

impl MetalDevice {
    /// Create default Metal device (GPU)
    pub fn create_system_default() -> Option<Self> {
        let ptr = unsafe { MTLCreateSystemDefaultDevice() };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    /// Create a new command queue
    pub fn new_command_queue(&self) -> MetalCommandQueue {
        let ptr: *mut AnyObject = unsafe { msg_send![self.ptr, newCommandQueue] };
        MetalCommandQueue { ptr }
    }

    /// Create a shared buffer (CPU and GPU can both access)
    pub fn new_buffer_with_length(&self, length: u64) -> Option<MetalBuffer> {
        let ptr: *mut AnyObject = unsafe {
            msg_send![
                self.ptr,
                newBufferWithLength: length,
                options: MTL_RESOURCE_STORAGE_MODE_SHARED
            ]
        };
        if ptr.is_null() {
            None
        } else {
            Some(MetalBuffer { ptr })
        }
    }

    pub fn as_ptr(&self) -> *mut AnyObject {
        self.ptr
    }
}

// ============================================================================
// MetalBuffer Implementation
// ============================================================================

impl MetalBuffer {
    /// Get CPU-accessible pointer to buffer contents
    pub fn contents(&self) -> *mut u8 {
        let ptr: *mut c_void = unsafe { msg_send![self.ptr, contents] };
        ptr as *mut u8
    }

    /// Release the buffer (manual memory management)
    pub unsafe fn release(self) {
        let _: () = msg_send![self.ptr, release];
    }

    pub fn as_ptr(&self) -> *mut AnyObject {
        self.ptr
    }
}

// ============================================================================
// MetalCommandQueue Implementation
// ============================================================================

impl MetalCommandQueue {
    /// Create a new command buffer
    pub fn command_buffer(&self) -> MetalCommandBuffer {
        let ptr: *mut AnyObject = unsafe { msg_send![self.ptr, commandBuffer] };
        MetalCommandBuffer { ptr }
    }

    pub fn as_ptr(&self) -> *mut AnyObject {
        self.ptr
    }
}

// ============================================================================
// MetalCommandBuffer Implementation
// ============================================================================

impl MetalCommandBuffer {
    /// Present the drawable to screen
    pub fn present_drawable(&self, drawable: &MetalDrawable) {
        unsafe {
            let _: () = msg_send![self.ptr, presentDrawable: drawable.ptr];
        }
    }

    /// Commit the command buffer for execution
    pub fn commit(&self) {
        unsafe {
            let _: () = msg_send![self.ptr, commit];
        }
    }

    pub fn as_ptr(&self) -> *mut AnyObject {
        self.ptr
    }
}

// ============================================================================
// MetalTexture Implementation
// ============================================================================

impl MetalTexture {
    /// Replace a region of the texture with CPU data
    pub fn replace_region(
        &self,
        region: MTLRegion,
        mipmap_level: u64,
        bytes: *const u8,
        bytes_per_row: u64,
    ) {
        unsafe {
            let _: () = msg_send![
                self.ptr,
                replaceRegion: region,
                mipmapLevel: mipmap_level,
                withBytes: bytes as *const std::ffi::c_void,
                bytesPerRow: bytes_per_row
            ];
        }
    }

    pub fn as_ptr(&self) -> *mut AnyObject {
        self.ptr
    }
}

// ============================================================================
// MetalLayer Implementation
// ============================================================================

impl MetalLayer {
    /// Set the pixel format for the layer
    pub fn set_pixel_format(&self, format: u64) {
        unsafe {
            let _: () = msg_send![self.ptr, setPixelFormat: format];
        }
    }

    /// Set the device this layer uses
    pub fn set_device(&self, device: &MetalDevice) {
        unsafe {
            let _: () = msg_send![self.ptr, setDevice: device.ptr];
        }
    }

    /// Set layer to be opaque (optimization)
    pub fn set_opaque(&self, opaque: bool) {
        unsafe {
            let _: () = msg_send![self.ptr, setOpaque: opaque];
        }
    }

    /// Get the next drawable to render into
    pub fn next_drawable(&self) -> Option<MetalDrawable> {
        let ptr: *mut AnyObject = unsafe { msg_send![self.ptr, nextDrawable] };
        if ptr.is_null() {
            None
        } else {
            Some(MetalDrawable { ptr })
        }
    }

    pub fn as_ptr(&self) -> *mut AnyObject {
        self.ptr
    }
}

impl From<*mut AnyObject> for MetalLayer {
    fn from(ptr: *mut AnyObject) -> Self {
        Self { ptr }
    }
}

// ============================================================================
// MetalDrawable Implementation
// ============================================================================

impl MetalDrawable {
    /// Get the texture to render into
    pub fn texture(&self) -> MetalTexture {
        let ptr: *mut AnyObject = unsafe { msg_send![self.ptr, texture] };
        MetalTexture { ptr }
    }

    pub fn as_ptr(&self) -> *mut AnyObject {
        self.ptr
    }
}
