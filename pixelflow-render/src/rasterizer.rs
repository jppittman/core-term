// pixelflow-render/src/rasterizer.rs
//! Rendering utilities for pixelflow-render.
//!
//! The primary rendering function is `pixelflow_core::execute()`.
//! This module provides convenience wrappers for common use cases.

use crate::color::Pixel;
use crate::frame::Frame;
use pixelflow_core::traits::Surface;

/// Render a surface into a Frame.
///
/// This is a convenience wrapper around `pixelflow_core::execute()`.
///
/// # Example
/// ```ignore
/// let mut frame = Frame::<Rgba>::new(800, 600);
/// let surface = mask.over::<Rgba>(fg, bg);
/// render(surface, &mut frame);
/// ```
pub fn render<P, S>(surface: &S, frame: &mut Frame<P>)
where
    P: Pixel,
    S: Surface<P> + ?Sized,
{
    let width = frame.width as usize;
    let height = frame.height as usize;
    pixelflow_core::execute(surface, frame.as_slice_mut(), width, height);
}

/// Render a surface into a typed pixel buffer.
///
/// The buffer must have at least `width * height` elements.
pub fn render_to_buffer<P, S>(surface: &S, buffer: &mut [P], width: usize, height: usize)
where
    P: Pixel,
    S: Surface<P> + ?Sized,
{
    pixelflow_core::execute(surface, buffer, width, height);
}

/// Render a Surface<P> into a u32 buffer.
///
/// This is the main render function for platform integration.
/// Since Pixel types are `repr(transparent)` wrappers around u32,
/// we can safely write them to a u32 buffer.
pub fn render_pixel<P, S>(surface: &S, buffer: &mut [u32], width: usize, height: usize)
where
    P: Pixel,
    S: Surface<P> + ?Sized,
{
    // SAFETY: P is repr(transparent) over u32
    let typed_buffer: &mut [P] =
        unsafe { core::slice::from_raw_parts_mut(buffer.as_mut_ptr() as *mut P, buffer.len()) };
    pixelflow_core::execute(surface, typed_buffer, width, height);
}

/// Render a Surface<u32> directly into a u32 buffer.
///
/// This is for backward compatibility with code that uses raw u32 surfaces.
/// For new code, prefer the typed `render_pixel()` or `render_to_buffer()`.
pub fn render_u32<S>(surface: &S, buffer: &mut [u32], width: usize, height: usize)
where
    S: Surface<u32> + ?Sized,
{
    pixelflow_core::execute(surface, buffer, width, height);
}

// Re-export execute for convenience
pub use pixelflow_core::execute;
