use pixelflow_core::{materialize_discrete, Discrete, Field, Manifold, ManifoldExt};
use pixelflow_graphics::scene3d::{
    ColorChecker, ColorReflect, ColorScreenToDir, ColorSky, ColorSurface, PlaneGeometry,
};
use pixelflow_graphics::render::color::RgbaColorCube;

// SphereAt: Translates UnitSphere
#[derive(Clone, Copy)]
struct SphereAt {
    cx: f32,
    cy: f32,
    cz: f32,
    r: f32,
}

use pixelflow_core::jet::Jet3;

impl Manifold<Jet3> for SphereAt {
    type Output = Jet3;
    fn eval_raw(&self, rx: Jet3, ry: Jet3, rz: Jet3, _w: Jet3) -> Jet3 {
        let cx = Field::from(self.cx);
        let cy = Field::from(self.cy);
        let cz = Field::from(self.cz);
        let r = Field::from(self.r);

        let doc = rx * Jet3::constant(cx) + ry * Jet3::constant(cy) + rz * Jet3::constant(cz);
        let c_sq = cx * cx + cy * cy + cz * cz;
        let r_sq = r * r;

        let b = doc * Jet3::constant(Field::from(-2.0));
        // Evaluate AST to constant for Jet3 constructor
        let c_val = (c_sq - r_sq).constant();
        let c = Jet3::constant(c_val);

        let four = Jet3::constant(Field::from(4.0));
        let disc = b * b - c * four;

        // Simplified intersection logic for asm check
        // We just return discriminant to ensure code gen happens
        disc
    }
}

struct ColorScreenRemap<M> {
    inner: M,
}

impl<M: Manifold<Output = Discrete>> Manifold for ColorScreenRemap<M> {
    type Output = Discrete;
    #[inline(always)]
    fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> Discrete {
        self.inner.eval_raw(x, y, z, w)
    }
}

#[inline(never)]
fn render_chrome_trace(x_start: f32, y: f32, out: &mut [u32]) {
    // 1. Scene Composition
    let sphere = ColorSurface {
        geometry: SphereAt { cx: 0.0, cy: 0.0, cz: 5.0, r: 1.0 },
        material: ColorReflect {
            inner: ColorChecker::<RgbaColorCube>::default(),
        },
        background: ColorSky::<RgbaColorCube>::default(),
    };

    // 2. Camera Transform (Screen -> Direction Jets)
    let camera = ColorScreenToDir { inner: sphere };

    // 3. Render
    let renderable = ColorScreenRemap { inner: camera };

    // Use materialize_discrete to evaluate and store (avoids private store method)
    materialize_discrete(&renderable, x_start, y, out);
}

fn main() {
    let mut buf = [0u32; 16];
    render_chrome_trace(0.0, 0.0, &mut buf);
    println!("Output: {:?}", buf);
}
