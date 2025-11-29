use crate::rasterizer::{Rasterizer, GlyphBounds};
use crate::GlyphId;
use pixelflow_core::{Batch, TensorView, TensorViewMut, execute};
use pixelflow_core::pipe::Surface;
use std::collections::HashMap;

pub struct Buffer<T> {
    pub data: Vec<T>,
    pub width: usize,
    pub height: usize,
    pub stride: usize,
}

impl<T: Copy + Default> Buffer<T> {
    pub fn new(width: usize, height: usize) -> Self {
        // Align stride to 16 bytes for SIMD friendliness
        let stride = (width + 15) & !15;
        let data = vec![T::default(); stride * height];
        Self {
            data,
            width,
            height,
            stride,
        }
    }

    pub fn view_mut(&mut self) -> TensorViewMut<'_, T> {
        TensorViewMut::new(&mut self.data, self.width, self.height, self.stride)
    }

    pub fn view(&self) -> TensorView<'_, T> {
        TensorView::new(&self.data, self.width, self.height, self.stride)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct AtlasConfig {
    pub max_width: u32,
    pub max_height: u32,
    pub padding: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct AtlasEntry {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    pub bounds: GlyphBounds,
}

pub struct Atlas {
    pub buffer: Buffer<u8>,
    pub entries: HashMap<GlyphId, AtlasEntry>,
}

impl Atlas {
    pub fn build(
        rasterizer: &Rasterizer,
        glyphs: impl Iterator<Item = GlyphId>,
        config: AtlasConfig,
    ) -> Self {
        let mut entries = HashMap::new();

        // 1. Measure all glyphs
        let mut glyph_list = Vec::new();
        for id in glyphs {
            let (glyph, bounds) = rasterizer.glyph_with_bounds(id);
            if bounds.width == 0 || bounds.height == 0 {
                 entries.insert(id, AtlasEntry { x:0, y:0, width:0, height:0, bounds });
                 continue;
            }
            glyph_list.push((id, glyph, bounds));
        }

        // 2. Shelf packing (sort by height)
        glyph_list.sort_by(|a, b| b.2.height.cmp(&a.2.height));

        let mut atlas_entries = Vec::new();
        let mut current_x = 0;
        let mut current_y = 0;
        let mut row_height = 0;
        let pad = config.padding;

        for (id, glyph, bounds) in glyph_list {
            if current_x + bounds.width + pad > config.max_width {
                current_x = 0;
                current_y += row_height + pad;
                row_height = 0;
            }

            if current_y + bounds.height > config.max_height {
                eprintln!("Atlas full! Skipping glyph {:?}", id);
                continue;
            }

            let entry = AtlasEntry {
                x: current_x,
                y: current_y,
                width: bounds.width,
                height: bounds.height,
                bounds,
            };

            atlas_entries.push((id, glyph, entry));
            entries.insert(id, entry);

            current_x += bounds.width + pad;
            row_height = row_height.max(bounds.height);
        }

        let final_height = (current_y + row_height).max(1);

        // 3. Render
        let mut buffer = Buffer::new(config.max_width as usize, final_height as usize);
        let mut view = buffer.view_mut();

        for (_id, glyph, entry) in atlas_entries {
            if (entry.x + entry.width) as usize <= view.width && (entry.y + entry.height) as usize <= view.height {
                unsafe {
                    let mut sub = view.sub_view(entry.x as usize, entry.y as usize, entry.width as usize, entry.height as usize);
                    execute(glyph.surface(), &mut sub);
                }
            }
        }

        Self {
            buffer,
            entries,
        }
    }

    pub fn sampler(&self, glyph: GlyphId) -> Option<AtlasSampler<'_>> {
        self.entries.get(&glyph).map(|entry| AtlasSampler {
            atlas: self,
            entry,
        })
    }
}

#[derive(Clone, Copy)]
pub struct AtlasSampler<'a> {
    atlas: &'a Atlas,
    entry: &'a AtlasEntry,
}

impl<'a> Surface<u8> for AtlasSampler<'a> {
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<u8> {
        let off_x = Batch::splat(self.entry.x);
        let off_y = Batch::splat(self.entry.y);

        let atlas_x = x + off_x;
        let atlas_y = y + off_y;

        unsafe { self.atlas.buffer.view().gather_2d(atlas_x, atlas_y).cast() }
    }
}
