//! ASM inspection - mirrors the exact benchmark pattern
//!
//! Run: cargo-asm -p pixelflow-graphics --example chrome_asm eval_one_pixel --release

use pixelflow_core::combinators::At;
use pixelflow_core::{Discrete, Field, Manifold};
use pixelflow_graphics::scene3d::{
    ColorChecker, ColorReflect, ColorScreenToDir, ColorSky, ColorSurface, PlaneGeometry, SphereAt,
};
use std::hint::black_box;

// This is the EXACT pattern from the benchmark
#[derive(Copy, Clone)]
struct ColorScreenRemap<M> {
    inner: M,
    width: f32,
    height: f32,
}

impl<M: Manifold<Output = Discrete>> Manifold for ColorScreenRemap<M> {
    type Output = Discrete;

    #[inline(always)]
    fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> Discrete {
        let width = Field::from(self.width);
        let height = Field::from(self.height);
        let scale = Field::from(2.0) / height;
        let sx = (x - width * Field::from(0.5)) * scale;
        let sy = (height * Field::from(0.5) - y) * scale;
        // Use At combinator to evaluate at transformed coordinates
        At {
            inner: &self.inner,
            x: sx,
            y: sy,
            z,
            w,
        }
        .eval()
    }
}

/// Evaluate one pixel - this is what we want ASM for
#[inline(never)]
pub fn eval_one_pixel(x: Field, y: Field) -> Discrete {
    let world = ColorSurface {
        geometry: PlaneGeometry { height: -1.0 },
        material: ColorChecker,
        background: ColorSky,
    };

    let scene = ColorSurface {
        geometry: SphereAt {
            center: (0.0, 0.0, 4.0),
            radius: 1.0,
        },
        material: ColorReflect { inner: world },
        background: world,
    };

    let renderable = ColorScreenRemap {
        inner: ColorScreenToDir { inner: scene },
        width: 1920.0,
        height: 1080.0,
    };

    renderable.eval_raw(x, y, Field::from(0.0), Field::from(0.0))
}

fn main() {
    let x = Field::sequential(960.0);
    let y = Field::from(540.0);
    let result = eval_one_pixel(black_box(x), black_box(y));
    black_box(result);
}
