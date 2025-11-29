#[cfg(test)]
mod tests {
    use crate::{Font, Rasterizer, RasterConfig, Hinting};
    use std::fs;
    use pixelflow_core::Batch;
    use pixelflow_core::pipe::Surface;

    #[test]
    fn test_rasterize_glyph() {
        let path = "assets/font/Noto_Sans_Mono/NotoSansMono-VariableFont_wdth,wght.ttf";
        // Try fallback path if running from crate dir
        let data = fs::read(path).or_else(|_| fs::read(format!("../{}", path))).expect("Failed to read font file");

        let font = Font::from_bytes(&data).unwrap();
        let config = RasterConfig { size: 32.0, hinting: Hinting::None };
        let rasterizer = Rasterizer::new(&font, config);

        let glyph_id = font.glyph_id('A').expect("Glyph A not found");
        let glyph = rasterizer.glyph(glyph_id);

        let surface = glyph.surface();

        let mut hit = false;
        // Search a larger area to be safe.
        for y in 0..40 {
            for x in 0..40 {
                let bx = Batch::splat(x);
                let by = Batch::splat(y);
                let coverage = surface.eval(bx, by);

                let val = coverage.cast::<u32>().to_bytes_packed()[0];
                if val > 0 {
                    hit = true;
                    // println!("Hit at {},{} : {}", x, y, val);
                }
            }
        }

        assert!(hit, "Glyph A should have some coverage");
    }
}
