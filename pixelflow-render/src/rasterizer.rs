//! Stateless frame processor for pixel-based rendering.

use crate::color::Rgba;
use crate::commands::Op;
use crate::glyph::font;
use pixelflow_core::dsl::{MaskExt, SurfaceExt};
use pixelflow_core::ops::Max;
use pixelflow_core::pipe::Surface;
use pixelflow_core::{execute_typed, Batch, TensorView, TensorViewMut};

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
    fn draw_glyph(&mut self, ch: char, pos: (usize, usize), fg: u32, bg: u32, bold: bool, italic: bool) {
        const ITALIC_SHEAR: i32 = 50;

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
                self.fb[row_start + fb_x] = bg;
            }
        }

        // 2. Get glyph Surface from pixelflow-fonts
        let f = font();
        let glyph = match f.glyph(ch, self.cell_height as f32) {
            Some(g) => g,
            None => return,
        };

        let bounds = glyph.bounds();
        if bounds.width == 0 || bounds.height == 0 {
            return;
        }

        // 3. Calculate Layout (Pixel Coords)
        let baseline = (self.cell_height as f32 * 0.8) as i32;
        let x_px = (px as i32 + bounds.bearing_x.max(0)) as usize;
        let y_px = (py as i32 + (baseline - bounds.height as i32 - bounds.bearing_y).max(0)) as usize;

        let width = bounds.width as usize;
        let height = bounds.height as usize;

        // Bounds check
        if x_px + width > self.width || y_px + height > self.height {
            return;
        }

        // 4. Create view and render using pixelflow
        let mut screen_view = TensorViewMut::new(
            self.fb,
            self.width,
            self.height,
            self.width,
        );

        let mut window = unsafe { screen_view.sub_view(x_px, y_px, width, height) };

        // Rgba IS a Surface<Rgba> - no wrapper needed
        let fg_color = Rgba(fg);
        let bg_color = Rgba(bg);

        // The pixelflow way: Glyph IS the Surface<u8>, compose directly
        // Rgba IS a Surface<Rgba> - colors can be used directly without wrappers
        match (bold, italic) {
            (false, false) => {
                execute_typed(glyph.over::<Rgba, _, _>(fg_color, bg_color), &mut window);
            }
            (true, false) => {
                let bold_glyph = Max(&glyph, (&glyph).offset(1, 0));
                execute_typed(bold_glyph.over::<Rgba, _, _>(fg_color, bg_color), &mut window);
            }
            (false, true) => {
                let italic_glyph = (&glyph).skew(ITALIC_SHEAR);
                execute_typed(italic_glyph.over::<Rgba, _, _>(fg_color, bg_color), &mut window);
            }
            (true, true) => {
                let italic_glyph = (&glyph).skew(ITALIC_SHEAR);
                let bold_italic = Max(italic_glyph, italic_glyph.offset(1, 0));
                execute_typed(bold_italic.over::<Rgba, _, _>(fg_color, bg_color), &mut window);
            }
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
                screen.draw_glyph(*ch, (*x, *y), (*fg).into(), (*bg).into(), *bold, *italic);
            }

            Op::Rect { x, y, w, h, color } => {
                screen.fill_rect(*x, *y, *w, *h, (*color).into());
            }
        }
    }
}
