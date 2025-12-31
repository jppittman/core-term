use pixelflow_core::{Field, Manifold};
use pixelflow_graphics::fonts::ttf::{LoopBlinnTriangle, TriangulatedGlyph};
use std::hint::black_box;

// Prevent LTO from removing the function we want to inspect
#[inline(never)]
fn run_kernel(glyph: &TriangulatedGlyph, x: Field, y: Field) -> Field {
    glyph.eval_raw(x, y, Field::from(0.0), Field::from(0.0))
}

fn main() {
    // Create a simple triangulated glyph (square made of 2 triangles)
    let tri1 = LoopBlinnTriangle::solid([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]);
    let tri2 = LoopBlinnTriangle::solid([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);

    let glyph = TriangulatedGlyph {
        triangles: vec![tri1, tri2].into(),
    };

    let x = Field::sequential(0.0);
    let y = Field::from(0.5);

    // Run it once to make sure it's linked
    let result = run_kernel(black_box(&glyph), black_box(x), black_box(y));
    black_box(result);
}
