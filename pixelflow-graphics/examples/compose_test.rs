//! Testing kernel composition patterns
use pixelflow_core::{Field, Manifold, ManifoldExt};
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

    let d = dist(1.0, 2.0);
    let p = field4(1.5, 2.0);
    println!("dist at (1.5, 2.0): {:?}", d.eval(p));

    // Pattern 2: Compose via a second kernel using 'inner: kernel' parameter.
    // The kernel macro's `inner: kernel` type accepts any ManifoldExpr as an
    // operand, enabling algebraic composition at the expression level.
    let circle = kernel!(|inner: kernel, r: f32| inner - r);

    let c = circle(dist(1.0, 2.0), 0.5);
    let result = c.eval(p);
    println!("circle SDF at (1.5, 2.0): {:?}", result);

    // Pattern 3: Direct scalar evaluation via eval4
    let val = d.eval4(1.5, 2.0, 0.0, 0.0);
    println!("eval4 result: {:?}", val);
}
