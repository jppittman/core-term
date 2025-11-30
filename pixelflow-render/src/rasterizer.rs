// pixelflow-render/src/rasterizer.rs
//! Rendering utilities for pixelflow-render.
//!
//! The primary rendering function is `pixelflow_core::execute()`.
//! This module provides convenience wrappers for common use cases.

use crate::color::Pixel;
use crate::frame::Frame;
use pixelflow_core::pipe::Surface;
use pixelflow_core::Batch;

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
pub fn render<P, S>(surface: S, frame: &mut Frame<P>)
where
    P: Pixel,
    S: Surface<P>,
{
    let width = frame.width as usize;
    let height = frame.height as usize;
    pixelflow_core::execute(surface, frame.as_slice_mut(), width, height);
}

/// Render a surface into a typed pixel buffer.
///
/// The buffer must have at least `width * height` elements.
pub fn render_to_buffer<P, S>(surface: S, buffer: &mut [P], width: usize, height: usize)
where
    P: Pixel,
    S: Surface<P>,
{
    pixelflow_core::execute(surface, buffer, width, height);
}

/// Render a Surface<u32> directly into a u32 buffer.
///
/// This is for backward compatibility with code that uses raw u32 surfaces.
/// For new code, prefer the typed `render()` or `render_to_buffer()`.
pub fn render_u32<S>(surface: &S, buffer: &mut [u32], width: usize, height: usize)
where
    S: Surface<u32> + ?Sized,
{
    const LANES: usize = 4;

    for y in 0..height {
        let row_start = y * width;
        let y_batch = Batch::splat(y as u32);

        // Hot path: process 4 pixels at a time (SIMD)
        let mut x = 0;
        while x + LANES <= width {
            let x_batch = Batch::new(x as u32, (x + 1) as u32, (x + 2) as u32, (x + 3) as u32);

            let result = surface.eval(x_batch, y_batch);

            // Store 4 pixels
            unsafe {
                result.store(buffer.as_mut_ptr().add(row_start + x));
            }

            x += LANES;
        }

        // Cold path: handle remaining pixels (< 4)
        while x < width {
            let x_batch = Batch::splat(x as u32);
            let result = surface.eval(x_batch, y_batch);
            buffer[row_start + x] = result.to_array_usize()[0] as u32;
            x += 1;
        }
    }
}

// Re-export execute for convenience
pub use pixelflow_core::execute;
