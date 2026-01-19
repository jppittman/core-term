#[cfg(test)]
mod tests {
    use pixelflow_core::{Discrete, Field, Manifold, ManifoldExt, RgbaComponents};
    use pixelflow_graphics::render::{rasterize, Frame};
    use pixelflow_graphics::render::color::Rgba8;

    #[test]
    fn test_sphere_raymarch() {
        // A simple sphere SDF
        struct SphereSdf;
        impl Manifold for SphereSdf {
            type Output = Discrete;
            fn eval_raw(&self, x: Field, y: Field, z: Field, _w: Field) -> Discrete {
                // Sphere at origin, r=100
                let dist = (x * x + y * y + z * z).sqrt().constant() - Field::from(100.0);

                // If dist < 0, inside.
                // Simple hard edge
                let mask = dist.lt(Field::from(0.0));

                // Color based on mask
                let one = Field::from(1.0);
                let zero = Field::from(0.0);

                // Inside = Gray, Outside = Black (transparent)
                // Actually let's make it white vs black
                let gray = mask.select(one, zero).constant();

                Discrete::pack(RgbaComponents {
                    r: gray,
                    g: gray,
                    b: gray,
                    a: one, // Always opaque for this test to see it
                })
            }
        }

        let manifold = SphereSdf;
        let mut frame: Frame<Rgba8> = Frame::new(256, 256);

        rasterize(&manifold, &mut frame, 1);
    }
}
