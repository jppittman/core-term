use crate::{Font, GlyphBounds};
use pixelflow_core::batch::Batch;
use pixelflow_core::pipe::Surface;
use pixelflow_core::{MapPixels, TensorView, TensorViewMut};
use std::collections::HashMap;

/// A simple buffer to hold pixel data.
pub struct Buffer<T> {
    pub data: Vec<T>,
    pub width: usize,
    pub height: usize,
    pub stride: usize,
}

impl<T: Copy + Default> Buffer<T> {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            data: vec![T::default(); width * height],
            width,
            height,
            stride: width,
        }
    }

    pub fn view(&self) -> TensorView<'_, T> {
        TensorView::new(&self.data, self.width, self.height, self.stride)
    }

    pub fn view_mut(&mut self) -> TensorViewMut<'_, T> {
        TensorViewMut::new(&mut self.data, self.width, self.height, self.stride)
    }
}

// Implement Surface for Buffer<u8>
impl Surface<u8> for Buffer<u8> {
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<u8> {
        let view = self.view();
        unsafe {
             let val = view.gather_2d(x, y);
             val.cast::<u8>()
        }
    }
}

pub struct AtlasEntry {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    pub bounds: GlyphBounds,
}

pub struct Atlas {
    pub buffer: Buffer<u8>,
    pub entries: HashMap<char, AtlasEntry>,
}

#[derive(Clone, Copy)]
pub struct AtlasConfig {
    pub size: f32,
    pub max_width: u32,
    pub max_height: u32,
    pub padding: u32,
}

impl Atlas {
    pub fn build(font: &Font, chars: impl Iterator<Item = char>, config: AtlasConfig) -> Self {
        let mut buffer = Buffer::<u8>::new(config.max_width as usize, config.max_height as usize);
        let mut entries = HashMap::new();

        let mut current_x = config.padding;
        let mut current_y = config.padding;
        let mut row_height = 0u32;

        for ch in chars {
            if let Some(glyph) = font.glyph(ch, config.size) {
                let bounds = glyph.bounds();
                let w = bounds.width;
                let h = bounds.height;

                // Check if we need to wrap to next row
                if current_x + w + config.padding > config.max_width {
                    current_x = config.padding;
                    current_y += row_height + config.padding;
                    row_height = 0;
                }

                if current_y + h + config.padding > config.max_height {
                     // Atlas full
                     continue;
                }

                row_height = row_height.max(h);

                // Materialize glyph into buffer at (current_x, current_y)
                {
                    let mut view_mut = buffer.view_mut();
                    unsafe {
                        let mut sub = view_mut.sub_view(current_x as usize, current_y as usize, w as usize, h as usize);

                        let h_batch = Batch::splat(h);
                        let one = Batch::splat(1);

                        sub.map_pixels(|x_batch, y_batch| {
                            // Flip Y for texture storage
                            let glyph_y = h_batch.saturating_sub(one).saturating_sub(y_batch);
                            glyph.eval(x_batch, glyph_y)
                        });
                    }
                }

                entries.insert(ch, AtlasEntry {
                    x: current_x,
                    y: current_y,
                    width: w,
                    height: h,
                    bounds,
                });

                current_x += w + config.padding;
            }
        }

        Self {
            buffer,
            entries,
        }
    }

    pub fn sampler(&self, ch: char) -> Option<AtlasSampler<'_>> {
        self.entries.get(&ch).map(|entry| AtlasSampler {
            atlas: self,
            entry,
        })
    }
}

#[derive(Clone, Copy)]
pub struct AtlasSampler<'a> {
    pub atlas: &'a Atlas,
    pub entry: &'a AtlasEntry,
}

impl<'a> Surface<u8> for AtlasSampler<'a> {
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<u8> {
        let atlas_x = x + Batch::splat(self.entry.x);
        let atlas_y = y + Batch::splat(self.entry.y);
        self.atlas.buffer.eval(atlas_x, atlas_y)
    }
}
