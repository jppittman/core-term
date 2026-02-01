//! Testing kernel composition patterns
use pixelflow_core::{Field, Manifold, ManifoldExt, ManifoldExpr};
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
    // map() requires the closure to implement Manifold, which is tricky for closures with inference
    // Instead, let's use explicit subtraction if possible, or explicit Map usage

    // Explicit composition: d - 0.5
    // Note: WithContext doesn't implement arithmetic ops directly unless ManifoldExpr is derived/impl'd
    // AND the trait bounds for ops are satisfied.
    // Instead of fighting the type system for this test example, let's just evaluate d and subtract.
    
    let p = field4(1.5, 2.0);
    let d_val = d.eval(p);
    let circle_val = d_val - Field::from(0.5);

    println!("d at (1.5, 2.0): {:?}", d_val);
    println!("circle at (1.5, 2.0): {:?}", circle_val);
    
    // Pattern 2: What we WANT but don't have
    // let circle = kernel!(|cx, cy, r| dist(cx, cy) - r);  // Won't work
}
