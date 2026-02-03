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
    
    // Can we subtract f32 from a kernel result?
    let d = dist(1.0, 2.0);
    // d is impl Manifold<Field4, Output=Field>
    
    // Using ManifoldExt to compose
    // d is WithContext<...>, we need to evaluate it to use it in map's closure
    // but Manifold::map operates on the Manifold itself, not the Field result.
    // The previous code `d.map(|f| ...)` tries to use `f` as `Field` but `d` is not a Field.
    // We should use the arithmetic operator overload on the Manifold directly if possible,
    // or correct the usage of `map`. Since `Manifold` implements Sub<f32> (via AST nodes usually),
    // let's try direct subtraction if the trait is implemented, otherwise use the `Map` combinator properly.
    
    // Actually, `ManifoldExt::map` takes a manifold as the second argument (the transformation).
    // The transformation manifold receives the output of the first as its input (X).
    // So we should construct a manifold that subtracts 0.5 from X.
    use pixelflow_core::X;
    let circle = d.map(X - Field::from(0.5));

    let p = field4(1.5, 2.0);
    let result = circle.eval(p);
    println!("circle at (1.5, 2.0): {:?}", result);
    
    // Pattern 2: What we WANT but don't have
    // let circle = kernel!(|cx, cy, r| dist(cx, cy) - r);  // Won't work
}
