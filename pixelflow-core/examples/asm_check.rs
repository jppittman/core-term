// examples/asm_check.rs

use pixelflow_core::{
    materialize_discrete, Discrete, Field, Manifold, ManifoldExt, RgbaComponents,
};

// Removed render_kernel (F32) as Vector trait for (Field, Field, Field, Field) is not readily available
// and we are interested in Discrete optimization.

#[inline(never)]
fn render_discrete_kernel(x_start: f32, y: f32, out: &mut [u32]) {
    // A discrete color manifold (manually constructed to inspect asm)
    struct RedCircle;
    impl pixelflow_core::Manifold for RedCircle {
        type Output = Discrete;
        fn eval_raw(&self, x: Field, y: Field, _z: Field, _w: Field) -> Discrete {
            let dist = ((x * x + y * y).sqrt() - Field::from(1.0)).constant();
            let mask = dist.lt(Field::from(0.0));
            let one = Field::from(1.0);
            let zero = Field::from(0.0);
            // Red inside, black outside
            let clamped = mask.select(one, zero).constant();
            let discrete = Discrete::pack(RgbaComponents {
                r: clamped,
                g: clamped,
                b: clamped,
                a: one,
            });
            discrete
        }
    }

    materialize_discrete(&RedCircle, x_start, y, out);
}

fn main() {
    let mut buf_u32 = [0u32; 16];
    render_discrete_kernel(0.0, 0.0, &mut buf_u32);
    println!("U32 Output: {:?}", buf_u32);
}
