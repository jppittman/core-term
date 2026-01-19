#[cfg(test)]
mod tests {
    use pixelflow_core::{Discrete, Field, Manifold, RgbaComponents};
    use pixelflow_graphics::render::{rasterize, Frame};
    use pixelflow_graphics::render::color::Rgba8;

    #[test]
    fn test_scene3d_composition() {
        struct Scene;
        impl Manifold for Scene {
            type Output = Discrete;
            fn eval_raw(&self, x: Field, y: Field, _z: Field, _w: Field) -> Discrete {
                // Dummy scene
                let gray = Field::from(0.5);
                Discrete::pack(RgbaComponents {
                    r: gray,
                    g: gray,
                    b: gray,
                    a: Field::from(1.0),
                })
            }
        }

        struct Overlay;
        impl Manifold for Overlay {
            type Output = Discrete;
            fn eval_raw(&self, x: Field, y: Field, _z: Field, _w: Field) -> Discrete {
                let r = Field::from(1.0);
                let g = Field::from(0.0);
                let b = Field::from(0.0);
                Discrete::pack(RgbaComponents {
                    r,
                    g,
                    b,
                    a: Field::from(1.0),
                })
            }
        }

        let scene = Scene;
        let mut frame: Frame<Rgba8> = Frame::new(64, 64);
        rasterize(&scene, &mut frame, 1);

        let overlay = Overlay;
        let mut frame2: Frame<Rgba8> = Frame::new(64, 64);
        rasterize(&overlay, &mut frame2, 1);
    }
}
