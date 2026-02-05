//! Testing kernel composition patterns
use pixelflow_core::{Field, Manifold, ManifoldExt, X};
use pixelflow_macros::kernel;

type Field4 = (Field, Field, Field, Field);

fn field4(x: f32, y: f32) -> Field4 {
    (Field::from(x), Field::from(y), Field::from(0.0), Field::from(0.0))
}

fn main() {
    // Pattern 1: Separate kernels, compose at call site
    let dist = kernel!(|cx: f32, cy: f32| {
        let dx = X - cx;
        let dy = Y - cy;
        (dx * dx + dy * dy).sqrt()
    });
    
    // Can we subtract f32 from a kernel result?
    let d = dist(1.0, 2.0);
    // d is impl Manifold<Field4, Output=Field>
    
    // Using ManifoldExt to compose
    // Map output (d) becomes X in the transform
    let circle = d.map(X - Field::from(0.5));  // radius 0.5
    
    let p = field4(1.5, 2.0);
    let result = circle.eval(p);
    println!("circle at (1.5, 2.0): {:?}", result);
    
    // Pattern 2: What we WANT but don't have
    // let circle = kernel!(|cx, cy, r| dist(cx, cy) - r);  // Won't work
}
