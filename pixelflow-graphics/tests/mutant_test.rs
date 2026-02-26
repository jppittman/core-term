//! Mutant Killing Tests
//!
//! These tests are designed to fail if specific logic bugs (mutants) are introduced.

use pixelflow_core::jet::Jet3;
use pixelflow_core::{Discrete, Field, Manifold, ManifoldCompat, ManifoldExt};
use pixelflow_macros::{kernel, ManifoldExpr};

type Field4 = (Field, Field, Field, Field);
type Jet3_4 = (Jet3, Jet3, Jet3, Jet3);

use pixelflow_graphics::render::color::Rgba8;
use pixelflow_graphics::render::frame::Frame;
use pixelflow_graphics::render::rasterizer::rasterize;
use pixelflow_graphics::scene3d::{ScreenToDir, Surface};

/// Sphere at given center with radius.
#[derive(Clone, Copy, ManifoldExpr)]
struct SphereAt {
    center: (f32, f32, f32),
    radius: f32,
}

impl Manifold<Jet3_4> for SphereAt {
    type Output = Jet3;

    #[inline]
    fn eval(&self, p: Jet3_4) -> Jet3 {
        let (rx, ry, rz, _w) = p;
        let cx = Jet3::constant(Field::from(self.center.0));
        let cy = Jet3::constant(Field::from(self.center.1));
        let cz = Jet3::constant(Field::from(self.center.2));

        let d_dot_c = rx * cx + ry * cy + rz * cz;
        let c_sq = cx * cx + cy * cy + cz * cz;
        let r_sq = Jet3::constant(Field::from(self.radius * self.radius));
        let discriminant = d_dot_c * d_dot_c - (c_sq - r_sq);

        let epsilon_sq = Jet3::constant(Field::from(0.0001));
        d_dot_c - (discriminant + epsilon_sq).sqrt()
    }
}

/// Convert grayscale Field to Discrete RGBA
struct GrayToRgba<M> {
    inner: M,
}

impl<M: ManifoldCompat<Field, Output = Field>> Manifold<Field4> for GrayToRgba<M> {
    type Output = Discrete;

    fn eval(&self, p: Field4) -> Discrete {
        let (x, y, z, w) = p;
        let gray = self.inner.eval_raw(x, y, z, w);
        Discrete::pack(gray, gray, gray, Field::from(1.0))
    }
}

/// Remap pixel coordinates to normalized screen coordinates.
struct ScreenRemap<M> {
    inner: M,
    width: f32,
    height: f32,
}

impl<M: ManifoldCompat<Field, Output = Field> + ManifoldExt> Manifold<Field4> for ScreenRemap<M> {
    type Output = Field;

    fn eval(&self, p: Field4) -> Field {
        let (px, py, z, w) = p;
        let width = Field::from(self.width);
        let height = Field::from(self.height);

        let scale = Field::from(2.0) / height;
        let x = (px - width * Field::from(0.5)) * scale.clone();
        let y = (height * Field::from(0.5) - py) * scale;

        self.inner.eval_at(x, y, z, w)
    }
}

/// Test that objects behind the camera are NOT rendered.
///
/// If the `valid_t` check (`t > 0`) in `Surface` is mutated to allow negative values,
/// this test should fail because the sphere behind the camera would become visible.
#[test]
fn object_behind_camera_should_not_be_visible() {
    const W: usize = 100;
    const H: usize = 100;

    // A white sphere at z = -4 (behind the camera).
    // Camera looks towards +Z.
    let white_mat = kernel!(|| 1.0)();
    let black_bg = kernel!(|| 0.0)();

    let scene = Surface {
        geometry: SphereAt {
            center: (0.0, 0.0, -4.0),
            radius: 1.0,
        },
        // White material
        material: white_mat,
        // Black background
        background: black_bg,
    };

    let screen = ScreenRemap {
        inner: ScreenToDir { inner: scene },
        width: W as f32,
        height: H as f32,
    };

    let renderable = GrayToRgba { inner: screen };

    let mut frame = Frame::<Rgba8>::new(W as u32, H as u32);
    rasterize(&renderable, &mut frame, 1);

    // Check the center pixel.
    let center = &frame.data[(H / 2) * W + (W / 2)];

    // Should be black (background), not white (sphere).
    assert_eq!(
        center.r(), 0,
        "Center pixel should be black (background). Found r={}, meaning object behind camera was rendered.",
        center.r()
    );
}
