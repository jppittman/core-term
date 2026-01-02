//! ASM inspection for Loop-Blinn triangle evaluation hot path
//!
//! Compile with: cargo build --release -p pixelflow-graphics --example quad_asm
//! Inspect with: objdump -d target/release/examples/quad_asm | grep -A200 eval_triangle_kernel

use pixelflow_core::{Field, Manifold};
use pixelflow_graphics::fonts::ttf::{curved_triangle, solid_triangle, LoopBlinnTriangle};
use std::hint::black_box;

/// Prevent inlining so we can find this function in the ASM
#[inline(never)]
pub fn eval_triangle_kernel(tri: &LoopBlinnTriangle, x: Field, y: Field) -> Field {
    tri.eval_raw(x, y, Field::from(0.0), Field::from(0.0))
}

fn main() {
    // Create a solid triangle
    let solid_tri = solid_triangle([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]);

    // Create a curved triangle (quadratic Bezier)
    let curved_tri = curved_triangle(
        [0.0, 0.0],  // P0 on-curve
        [0.5, 0.5],  // P1 control
        [1.0, 0.0],  // P2 on-curve
        1.0,         // CCW winding
    );

    // Evaluate at test point
    let x = Field::from(0.5);
    let y = Field::from(0.25);

    // Run kernels (black_box prevents optimization)
    let solid_result = eval_triangle_kernel(black_box(&solid_tri), black_box(x), black_box(y));
    let curved_result = eval_triangle_kernel(black_box(&curved_tri), black_box(x), black_box(y));

    // Use results to prevent dead code elimination
    println!("Solid triangle result: {:?}", black_box(solid_result));
    println!("Curved triangle result: {:?}", black_box(curved_result));
}
