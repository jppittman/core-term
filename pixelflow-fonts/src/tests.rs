#[cfg(test)]
mod tests {
    use crate::{Font, Rasterizer, RasterConfig, Hinting, Atlas, AtlasConfig};
    use std::fs;
    use pixelflow_core::Batch;
    use pixelflow_core::pipe::Surface;

    #[test]
    fn test_rasterize_glyph() {
        let path = "assets/font/Noto_Sans_Mono/NotoSansMono-VariableFont_wdth,wght.ttf";
        let data = fs::read(path).or_else(|_| fs::read(format!("../{}", path))).expect("Failed to read font file");

        let font = Font::from_bytes(&data).unwrap();
        let config = RasterConfig { size: 32.0, hinting: Hinting::None };
        let rasterizer = Rasterizer::new(&font, config);

        let glyph_id = font.glyph_id('A').expect("Glyph A not found");
        let glyph = rasterizer.glyph(glyph_id);
        let surface = glyph.surface();

        let mut hit = false;
        for y in 0..40 {
            for x in 0..40 {
                let bx = Batch::splat(x);
                let by = Batch::splat(y);
                let coverage = surface.eval(bx, by);

                let val = coverage.cast::<u32>().to_bytes_packed()[0];
                if val > 0 {
                    hit = true;
                }
            }
        }

        assert!(hit, "Glyph A should have some coverage");
    }

    #[test]
    fn test_atlas_build() {
        let path = "assets/font/Noto_Sans_Mono/NotoSansMono-VariableFont_wdth,wght.ttf";
        let data = fs::read(path).or_else(|_| fs::read(format!("../{}", path))).expect("Failed to read font file");

        let font = Font::from_bytes(&data).unwrap();
        let config = RasterConfig { size: 16.0, hinting: Hinting::None };
        let rasterizer = Rasterizer::new(&font, config);

        let glyph_a = font.glyph_id('A').unwrap();
        let glyph_b = font.glyph_id('B').unwrap();

        let atlas = Atlas::build(&rasterizer, vec![glyph_a, glyph_b].into_iter(), AtlasConfig {
            max_width: 128,
            max_height: 128,
            padding: 2,
        });

        assert!(atlas.entries.contains_key(&glyph_a));
        assert!(atlas.entries.contains_key(&glyph_b));

        let sampler = atlas.sampler(glyph_a).unwrap();
        // Sample somewhat inside the atlas
        // 'A' should be at 0,0 (shelf packing).
        // We evaluate relative to the glyph origin (0,0).
        // Sampler adds atlas offset.

        let mut hit = false;
        for y in 0..20 {
            for x in 0..20 {
                 let bx = Batch::splat(x);
                 let by = Batch::splat(y);
                 let val = sampler.eval(bx, by).cast::<u32>().to_bytes_packed()[0];
                 if val > 0 { hit = true; }
            }
        }
        assert!(hit, "Atlas sampler for A should return coverage");
    }
}
