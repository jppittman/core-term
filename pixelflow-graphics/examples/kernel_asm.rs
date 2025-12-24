use pixelflow_core::{Field, Manifold};
use pixelflow_graphics::fonts::ttf::{Curve, Geometry};
use std::hint::black_box;

// Prevent LTO from removing the function we want to inspect
#[inline(never)]
fn run_kernel(geo: &Geometry, x: Field, y: Field) -> Field {
    geo.eval_raw(x, y, Field::from(0.0), Field::from(0.0))
}

fn main() {
    let p0 = [10.0, 10.0];
    let p1 = [20.0, 30.0];
    let line = Curve([p0, p1]);
    
    let geo = Geometry {
        lines: vec![line].into(),
        quads: vec![].into(),
    };

    let x = Field::sequential(0.0);
    let y = Field::from(15.0);

    // Run it once to make sure it's linked
    let result = run_kernel(black_box(&geo), black_box(x), black_box(y));
    black_box(result);
}
