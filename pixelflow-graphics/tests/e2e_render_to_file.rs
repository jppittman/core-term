#[cfg(test)]
mod tests {
    use pixelflow_core::{Discrete, Field, Manifold, RgbaComponents};
    use pixelflow_graphics::render::{rasterize, Frame};
    use pixelflow_graphics::render::color::Rgba8;

    #[test]
    fn test_e2e_render_to_file() {
        struct Gradient;
        impl Manifold for Gradient {
            type Output = Discrete;
            fn eval_raw(&self, x: Field, y: Field, _z: Field, _w: Field) -> Discrete {
                // x, y are coordinates.
                // Normalize to 0-1 for 256x256 image
                // To divide by field, we need expression evaluation
                // But x is Field. 256.0 is const.
                // x/256.0 is AST.
                // We need values for packing.
                // Use .constant() (which evaluates at 0, but x IS the value here)
                // Wait, x passed to eval_raw IS the value.
                // But operators return AST.
                // (x / Field::from(256.0)).constant() works.

                // Note: I need ManifoldExt for constant().
                use pixelflow_core::ManifoldExt;

                let u = (x / Field::from(256.0)).constant();
                let v = (y / Field::from(256.0)).constant();

                let r = u;
                let g = v;
                let b = Field::from(0.0);

                Discrete::pack(RgbaComponents {
                    r,
                    g,
                    b,
                    a: Field::from(1.0),
                })
            }
        }

        let manifold = Gradient;
        let mut frame: Frame<Rgba8> = Frame::new(256, 256);
        rasterize(&manifold, &mut frame, 1);

        // No assertions on file output, just checking it runs.
    }
}
