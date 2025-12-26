//! Test that FMA fusion works at the type level.
//!
//! This example verifies that `X * Y + Z` produces `MulAdd<X, Y, Z>`,
//! enabling automatic compile-time fusion to FMA instructions.

use pixelflow_core::*;
use std::hint::black_box;

/// Evaluate FMA with runtime values (prevents constant folding)
#[inline(never)]
fn eval_fma_runtime(x: Field, y: Field, z: Field) -> Field {
    // X * Y + Z uses the FMA intrinsic via MulAdd
    let expr = X * Y + Z;
    expr.eval_raw(x, y, z, Field::from(0.0))
}

fn main() {
    // This should compile to MulAdd<X, Y, Z> - type-level FMA fusion!
    let fma_expr = X * Y + Z;

    // The type checker proves the fusion happened at compile time
    let _: MulAdd<X, Y, Z> = fma_expr;

    // Use runtime values to prevent constant folding
    let x = black_box(Field::sequential(1.0));
    let y = black_box(Field::from(2.0));
    let z = black_box(Field::from(3.0));

    let result = eval_fma_runtime(x, y, z);
    black_box(result);

    println!("FMA fusion confirmed at compile time!");
    println!("Type of `X * Y + Z` is MulAdd<X, Y, Z>");
    println!();
    println!("Check assembly with:");
    println!("  objdump -d target/release/examples/fma_test | grep -B2 -A15 eval_fma_runtime");
}
