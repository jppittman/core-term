//! Stateless frame processor for pixel-based rendering.
//!
//! This module provides a single pure function: `process_frame()`.
//! It transforms a framebuffer based on a list of operations, with no internal state.

use crate::commands::Op;
use crate::glyph;
use pixelflow_core::Batch;

/// Safe wrapper for framebuffer access with clipping.
struct ScreenView<'a> {
    fb: &'a mut [u32],
    width: usize,
    height: usize,
}

impl<'a> ScreenView<'a> {
    fn new(fb: &'a mut [u32], width: usize, height: usize) -> Self {
        Self { fb, width, height }
    }

    /// Clear entire screen to a single color.
    fn clear(&mut self, color: u32) {
        let color_batch = Batch::splat(color);
        let num_pixels = self.fb.len();
        let num_batches = num_pixels / 4;
        let remainder = num_pixels % 4;

        // Process full batches of 4 pixels
        for i in 0..num_batches {
            unsafe {
                color_batch.store(self.fb.as_mut_ptr().add(i * 4));
            }
        }

        // Handle remainder
        for i in 0..remainder {
            self.fb[num_batches * 4 + i] = color;
        }
    }

    /// Blit raw RGBA data to framebuffer with clipping.
    fn blit(&mut self, data: &[u8], data_width: usize, x: usize, y: usize) {
        if data.len() % 4 != 0 {
            return; // Invalid data, not RGBA
        }

        let data_pixels = data.len() / 4;
        let data_height = data_pixels / data_width;

        for row in 0..data_height {
            let fb_y = y + row;
            if fb_y >= self.height {
                break;
            }

            for col in 0..data_width {
                let fb_x = x + col;
                if fb_x >= self.width {
                    break;
                }

                let data_idx = (row * data_width + col) * 4;
                let fb_idx = fb_y * self.width + fb_x;

                // Convert RGBA bytes to u32
                let r = data[data_idx];
                let g = data[data_idx + 1];
                let b = data[data_idx + 2];
                let a = data[data_idx + 3];
                self.fb[fb_idx] = u32::from_le_bytes([r, g, b, a]);
            }
        }
    }

    /// Render a glyph at the specified position with SIMD alpha blending.
    fn draw_glyph(&mut self, ch: char, x: usize, y: usize, fg: u32, bg: u32, cell_width: usize, cell_height: usize) {
        // CRITICAL: Fill the entire cell with background color FIRST
        // This ensures that spaces and partial glyphs properly clear previous content
        let bg_batch = Batch::splat(bg);

        for row in 0..cell_height {
            let fb_y = y + row;
            if fb_y >= self.height {
                break;
            }

            // Fill entire row of the cell with background
            for col_batch in (0..cell_width).step_by(4) {
                let fb_x = x + col_batch;
                if fb_x >= self.width {
                    break;
                }

                let batch_size = (cell_width - col_batch).min(4).min(self.width - fb_x);
                if batch_size == 4 {
                    let fb_idx = fb_y * self.width + fb_x;
                    unsafe {
                        bg_batch.store(self.fb.as_mut_ptr().add(fb_idx));
                    }
                } else {
                    for i in 0..batch_size {
                        let fb_idx = fb_y * self.width + fb_x + i;
                        self.fb[fb_idx] = bg;
                    }
                }
            }
        }

        // Now decompress and render the glyph on top of the background
        let rendered = glyph::render_glyph_natural(ch, cell_height, false, false);
        let alpha_mask = &rendered.data;
        let glyph_width = rendered.width;
        let glyph_height = rendered.height;

        // Position glyph using bearing information for proper baseline alignment
        // bearing_x is horizontal offset from left edge
        // bearing_y is vertical offset from baseline (negative = below baseline)
        // The baseline is at a fixed position in the cell (typically ~80% down)
        let baseline_y = (cell_height as f32 * 0.8) as i32;

        let offset_x = rendered.bearing_x.max(0) as usize;
        let offset_y = (baseline_y - glyph_height as i32 - rendered.bearing_y).max(0) as usize;

        // Prepare SIMD batches for fg and bg colors
        let fg_batch = Batch::splat(fg);
        let bg_batch = Batch::splat(bg);

        // Process each row of the glyph
        for row in 0..glyph_height {
            let fb_y = y + offset_y + row;
            if fb_y >= self.height {
                break;
            }

            for col_batch in (0..glyph_width).step_by(4) {
                let fb_x = x + offset_x + col_batch;
                if fb_x >= self.width {
                    break;
                }

                // How many pixels in this batch? (might be < 4 at edge)
                let batch_size = (glyph_width - col_batch).min(4).min(self.width - fb_x);

                if batch_size == 4 {
                    // Full batch: use SIMD
                    let alpha_idx = row * glyph_width + col_batch;

                    // Build alpha batch from grayscale values
                    let a0 = alpha_mask[alpha_idx];
                    let a1 = alpha_mask[alpha_idx + 1];
                    let a2 = alpha_mask[alpha_idx + 2];
                    let a3 = alpha_mask[alpha_idx + 3];

                    let alpha_pixels = [
                        u32::from_le_bytes([255, 255, 255, a0]),
                        u32::from_le_bytes([255, 255, 255, a1]),
                        u32::from_le_bytes([255, 255, 255, a2]),
                        u32::from_le_bytes([255, 255, 255, a3]),
                    ];
                    let alpha_batch = unsafe { Batch::load(alpha_pixels.as_ptr()) };

                    // Blend
                    let blended = fg_batch.blend_alpha(bg_batch, alpha_batch);

                    // Store to framebuffer
                    let fb_idx = fb_y * self.width + fb_x;
                    unsafe {
                        blended.store(self.fb.as_mut_ptr().add(fb_idx));
                    }
                } else {
                    // Partial batch at edge: use scalar
                    for i in 0..batch_size {
                        let alpha_idx = row * glyph_width + col_batch + i;
                        let alpha = alpha_mask[alpha_idx] as f32 / 255.0;

                        let fg_bytes = fg.to_le_bytes();
                        let bg_bytes = bg.to_le_bytes();

                        let r = (fg_bytes[0] as f32 * alpha + bg_bytes[0] as f32 * (1.0 - alpha)) as u8;
                        let g = (fg_bytes[1] as f32 * alpha + bg_bytes[1] as f32 * (1.0 - alpha)) as u8;
                        let b = (fg_bytes[2] as f32 * alpha + bg_bytes[2] as f32 * (1.0 - alpha)) as u8;

                        let fb_idx = fb_y * self.width + fb_x + i;
                        self.fb[fb_idx] = u32::from_le_bytes([r, g, b, 255]);
                    }
                }
            }
        }
    }
}

/// Process a frame by executing a list of rendering operations.
///
/// This is the only public API for pixelflow-render. It is completely stateless:
/// - No caching (glyphs decompressed on-the-fly from ROM)
/// - No font management (only pre-baked Noto Sans Mono)
/// - No configuration state
///
/// # Parameters
/// - `framebuffer`: Raw pixel buffer (RGBA as u32, row-major)
/// - `width`: Framebuffer width in pixels
/// - `height`: Framebuffer height in pixels
/// - `cell_width`: Cell width in pixels (for text rendering)
/// - `cell_height`: Cell height in pixels (for text rendering)
/// - `commands`: List of operations to execute
///
/// # Example
/// ```ignore
/// let mut fb = vec![0u32; 1920 * 1080];
/// let ops = [
///     Op::Clear { color: 0xFF000000 },
///     Op::Text { ch: 'A', x: 0, y: 0, fg: 0xFFFFFFFF, bg: 0xFF000000 },
/// ];
/// process_frame(&mut fb, 1920, 1080, 12, 24, &ops);
/// ```
pub fn process_frame<T: AsRef<[u8]>>(
    framebuffer: &mut [u32],
    width: usize,
    height: usize,
    cell_width: usize,
    cell_height: usize,
    commands: &[Op<T>],
) {
    let mut screen = ScreenView::new(framebuffer, width, height);

    for op in commands {
        match op {
            Op::Clear { color } => {
                // Convert Color to u32 once, right before execution
                let color_u32: u32 = (*color).into();
                screen.clear(color_u32);
            }

            Op::Blit { data, w, x, y } => {
                screen.blit(data.as_ref(), *w, *x, *y);
            }

            Op::Text { ch, x, y, fg, bg } => {
                // Convert Color to u32 once, right before execution
                let fg_u32: u32 = (*fg).into();
                let bg_u32: u32 = (*bg).into();
                screen.draw_glyph(*ch, *x, *y, fg_u32, bg_u32, cell_width, cell_height);
            }
        }
    }
}
