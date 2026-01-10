//! ASM inspection for Quad evaluation hot path
//!
//! Compile with: cargo build --release -p pixelflow-graphics --example quad_asm
//! Inspect with: objdump -d target/release/examples/quad_asm | grep -A200 eval_quad_kernel

use pixelflow_core::{Field, Manifold};
use pixelflow_graphics::fonts::ttf::{loop_blinn_quad, make_line, LineKernel, Quad, QuadKernel};
use std::hint::black_box;

/// Prevent inlining so we can find this function in the ASM
#[inline(never)]
pub fn eval_quad_kernel(quad: &Quad<QuadKernel, LineKernel>, x: Field, y: Field) -> Field {
    quad.eval_raw(x, y, Field::from(0.0), Field::from(0.0))
}

/// Also test Line for comparison
#[inline(never)]
pub fn eval_line_kernel(
    line: &pixelflow_graphics::fonts::ttf::Line<LineKernel>,
    x: Field,
    y: Field,
) -> Field {
    line.eval_raw(x, y, Field::from(0.0), Field::from(0.0))
}

fn main() {
    // Create a simple quadratic Bezier curve using Loop-Blinn implicit representation
    let quad = loop_blinn_quad([[0.0, 0.0], [0.5, 1.0], [1.0, 0.0]]);

    // Create a simple line
    let line = make_line([[0.0, 0.0], [1.0, 1.0]]);

    // Evaluate at test point
    let x = Field::from(0.5);
    let y = Field::from(0.5);

    // Run kernels (black_box prevents optimization)
    let quad_result = eval_quad_kernel(black_box(&quad), black_box(x), black_box(y));
    let line_result = eval_line_kernel(black_box(&line), black_box(x), black_box(y));

    // Use results to prevent dead code elimination
    println!("Quad result: {:?}", black_box(quad_result));
    println!("Line result: {:?}", black_box(line_result));
}
