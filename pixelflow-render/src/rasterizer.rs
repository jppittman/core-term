//! Stateless frame processor for pixel-based rendering.

use crate::commands::Op;
use crate::glyph::{self, RenderTarget, GlyphRenderCoords, GlyphStyleOverrides};

/// Safe wrapper for framebuffer access and layout context.
struct ScreenView<'a> {
    fb: &'a mut [u32],
    width: usize,
    height: usize,
    cell_width: usize,
    cell_height: usize,
}

impl<'a> ScreenView<'a> {
    fn new(fb: &'a mut [u32], width: usize, height: usize, cw: usize, ch: usize) -> Self {
        Self { fb, width, height, cell_width: cw, cell_height: ch }
    }

    fn clear(&mut self, color: u32) {
        // Simple fill loop
        for px in self.fb.iter_mut() {
            *px = color;
        }
    }

    fn blit(&mut self, data: &[u8], data_width: usize, x: usize, y: usize) {
        if data.len() % 4 != 0 { panic!("bad"); }
        let data_height = (data.len() / 4) / data_width;
        
        for row in 0..data_height {
            let fb_y = y + row;
            if fb_y >= self.height { break; }
            
            for col in 0..data_width {
                let fb_x = x + col;
                if fb_x >= self.width { break; }
                
                let idx = (row * data_width + col) * 4;
                let r = data[idx];
                let g = data[idx+1];
                let b = data[idx+2];
                let a = data[idx+3];
                
                let fb_idx = fb_y * self.width + fb_x;
                self.fb[fb_idx] = u32::from_le_bytes([r, g, b, a]);
            }
        }
    }

    /// Draw a glyph using the new Zero-Copy Pipeline.
    fn draw_glyph(&mut self, ch: char, pos: (usize, usize), style: GlyphStyleOverrides) {
        let (col, row) = pos;
        
        // 1. Clear the cell background
        // We do this procedurally first to ensure the cell is clean.
        let cx = col * self.cell_width;
        let cy = row * self.cell_height;
        
        for y in 0..self.cell_height {
            let fb_y = cy + y;
            if fb_y >= self.height { break; }
            
            let row_start = fb_y * self.width;
            for x in 0..self.cell_width {
                let fb_x = cx + x;
                if fb_x >= self.width { break; }
                self.fb[row_start + fb_x] = style.bg;
            }
        }

        // 2. Calculate Layout (Pixel Coords)
        let metrics = glyph::get_glyph_metrics(ch, self.cell_height);

        let baseline = (self.cell_height as f32 * 0.8) as i32;
        
        let x_px = (cx as i32 + metrics.bearing_x.max(0)) as usize;
        let y_px = (cy as i32 + (baseline - metrics.height as i32 - metrics.bearing_y).max(0)) as usize;

        // 3. Render Direct (Zero Copy)
        // Bounds check before calling unsafe slice
        if x_px + metrics.width <= self.width && y_px + metrics.height <= self.height {
            glyph::render_glyph_direct(
                ch,
                RenderTarget { dest: self.fb, stride: self.width },
                GlyphRenderCoords { x_px, y_px, cell_height: self.cell_height },
                style
            );
        }
    }
}

pub fn process_frame<T: AsRef<[u8]>>(
    framebuffer: &mut [u32],
    width: usize,
    height: usize,
    cell_width: usize,
    cell_height: usize,
    commands: &[Op<T>],
) {
    let mut screen = ScreenView::new(framebuffer, width, height, cell_width, cell_height);

    for op in commands {
        match op {
            Op::Clear { color } => {
                screen.clear((*color).into());
            }

            Op::Blit { data, w, x, y } => {
                screen.blit(data.as_ref(), *w, *x, *y);
            }

            Op::Text { ch, x, y, fg, bg } => {
                let style = GlyphStyleOverrides {
                    fg: (*fg).into(),
                    bg: (*bg).into(),
                    bold: false, // Default, could be exposed in Op::Text later
                    italic: false,
                };
                
                screen.draw_glyph(*ch, (*x, *y), style);
            }
        }
    }
}
