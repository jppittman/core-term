//! Stateless frame processor for pixel-based rendering.

use crate::commands::Op;
use crate::glyph::{self, GlyphRenderCoords, GlyphStyleOverrides, RenderTarget};
use pixelflow_core::pipe::Surface;
use pixelflow_core::{Batch, TensorView};

/// Mutable view of the screen for rendering.
///
/// This struct wraps the framebuffer and layout information, providing
/// methods for drawing primitives.
pub struct ScreenViewMut<'a> {
    /// The framebuffer slice.
    pub fb: &'a mut [u32],
    /// Screen width in pixels.
    pub width: usize,
    /// Screen height in pixels.
    pub height: usize,
    /// Cell width for text layout.
    pub cell_width: usize,
    /// Cell height for text layout.
    pub cell_height: usize,
}

impl<'a> ScreenViewMut<'a> {
    /// Creates a new `ScreenViewMut`.
    pub fn new(fb: &'a mut [u32], width: usize, height: usize, cw: usize, ch: usize) -> Self {
        Self {
            fb,
            width,
            height,
            cell_width: cw,
            cell_height: ch,
        }
    }

    fn clear(&mut self, color: u32) {
        // Simple fill loop
        for px in self.fb.iter_mut() {
            *px = color;
        }
    }

    fn blit(&mut self, data: &[u8], data_width: usize, x: usize, y: usize) {
        if data.len() % 4 != 0 {
            panic!("bad");
        }
        let data_height = (data.len() / 4) / data_width;

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

                let idx = (row * data_width + col) * 4;
                let r = data[idx];
                let g = data[idx + 1];
                let b = data[idx + 2];
                let a = data[idx + 3];

                let fb_idx = fb_y * self.width + fb_x;
                self.fb[fb_idx] = u32::from_le_bytes([r, g, b, a]);
            }
        }
    }

    fn fill_rect(&mut self, x: usize, y: usize, w: usize, h: usize, color: u32) {
        for row in 0..h {
            let fb_y = y + row;
            if fb_y >= self.height {
                break;
            }
            let row_start = fb_y * self.width;
            for col in 0..w {
                let fb_x = x + col;
                if fb_x >= self.width {
                    break;
                }
                self.fb[row_start + fb_x] = color;
            }
        }
    }

    /// Draw a glyph at the given pixel position.
    fn draw_glyph(&mut self, ch: char, pos: (usize, usize), style: GlyphStyleOverrides) {
        let (px, py) = pos; // Position in pixels

        // 1. Clear the cell background at pixel position
        for y in 0..self.cell_height {
            let fb_y = py + y;
            if fb_y >= self.height {
                break;
            }

            let row_start = fb_y * self.width;
            for x in 0..self.cell_width {
                let fb_x = px + x;
                if fb_x >= self.width {
                    break;
                }
                self.fb[row_start + fb_x] = style.bg;
            }
        }

        // 2. Calculate Layout (Pixel Coords)
        let metrics = glyph::get_glyph_metrics(ch, self.cell_height);

        let baseline = (self.cell_height as f32 * 0.8) as i32;

        let x_px = (px as i32 + metrics.bearing_x.max(0)) as usize;
        let y_px =
            (py as i32 + (baseline - metrics.height as i32 - metrics.bearing_y).max(0)) as usize;

        // 3. Render Direct (Zero Copy)
        // Bounds check before calling unsafe slice
        if x_px + metrics.width <= self.width && y_px + metrics.height <= self.height {
            glyph::render_glyph_direct(
                ch,
                RenderTarget {
                    dest: self.fb,
                    stride: self.width,
                },
                GlyphRenderCoords {
                    x_px,
                    y_px,
                    cell_height: self.cell_height,
                },
                style,
            );
        }
    }
}

/// Immutable view of the screen, implementing Surface.
#[derive(Copy, Clone)]
pub struct ScreenView<'a> {
    /// The tensor view of the screen.
    pub tensor: TensorView<'a, u32>,
    /// Cell width for text layout.
    pub cell_width: usize,
    /// Cell height for text layout.
    pub cell_height: usize,
}

impl<'a> ScreenView<'a> {
    /// Creates a new `ScreenView`.
    pub fn new(tensor: TensorView<'a, u32>, cw: usize, ch: usize) -> Self {
        Self {
            tensor,
            cell_width: cw,
            cell_height: ch,
        }
    }
}

impl<'a> Surface<u32> for ScreenView<'a> {
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<u32> {
        // Safe wrapper around unsafe gather_2d (which clamps internally)
        unsafe { self.tensor.gather_2d(x, y) }
    }
}

/// Materialize a Surface into the framebuffer.
///
/// This is the core rendering function. It evaluates the surface at every
/// pixel coordinate and writes the result to the framebuffer.
///
/// Processing is done in batches of 4 pixels (SIMD lanes) for efficiency.
///
/// # Parameters
/// * `screen` - The mutable screen view to render into.
/// * `surface` - The composed surface to evaluate.
pub fn materialize_into(screen: &mut ScreenViewMut, surface: &dyn Surface<u32>) {
    for py in 0..screen.height {
        let row_start = py * screen.width;

        // Process in batches of 4 pixels
        let mut px = 0;
        while px + 4 <= screen.width {
            let batch_x = Batch::new(
                px as u32,
                (px + 1) as u32,
                (px + 2) as u32,
                (px + 3) as u32,
            );
            let batch_y = Batch::splat(py as u32);

            let colors = surface.eval(batch_x, batch_y);

            // Write 4 pixels to framebuffer
            unsafe {
                colors.store(screen.fb.as_mut_ptr().add(row_start + px));
            }

            px += 4;
        }

        // Handle remaining pixels (< 4)
        while px < screen.width {
            let batch_x = Batch::splat(px as u32);
            let batch_y = Batch::splat(py as u32);
            let colors = surface.eval(batch_x, batch_y);
            // Extract first lane only
            screen.fb[row_start + px] = unsafe { *(&colors as *const _ as *const u32) };
            px += 1;
        }
    }
}

/// Process a list of rendering commands and update the framebuffer.
///
/// DEPRECATED: Use `materialize_into` with Surface-based rendering instead.
/// This function is kept for backwards compatibility during migration.
#[deprecated(note = "Use materialize_into with Surface-based rendering")]
pub fn process_frame<T: AsRef<[u8]>>(
    screen: &mut ScreenViewMut,
    commands: &[Op<T>],
) {
    for op in commands {
        match op {
            Op::Clear { color } => {
                screen.clear((*color).into());
            }

            Op::Blit { data, w, x, y } => {
                screen.blit(data.as_ref(), *w, *x, *y);
            }

            Op::Text {
                ch,
                x,
                y,
                fg,
                bg,
                bold,
                italic,
            } => {
                let style = GlyphStyleOverrides {
                    fg: (*fg).into(),
                    bg: (*bg).into(),
                    bold: *bold,
                    italic: *italic,
                };

                screen.draw_glyph(*ch, (*x, *y), style);
            }

            Op::Rect { x, y, w, h, color } => {
                screen.fill_rect(*x, *y, *w, *h, (*color).into());
            }
        }
    }
}
