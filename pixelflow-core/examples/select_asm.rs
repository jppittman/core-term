//! ASM comparison for Select with different condition types
//!
//! Run: cargo-asm -p pixelflow-core --example select_asm select_gt --release
//! Run: cargo-asm -p pixelflow-core --example select_asm select_field --release

use pixelflow_core::{Field, ManifoldCompat, ManifoldExt, X};
use std::hint::black_box;

/// Select with Gt<X, f32> condition - goes through FieldCondition::eval_mask
#[inline(never)]
#[unsafe(no_mangle)]
pub fn select_gt(x: Field, y: Field, z: Field, w: Field) -> Field {
    let m = X.gt(2.0f32).select(1.0f32, 0.0f32);
    m.eval_raw(x, y, z, w)
}

/// Select with Field condition - uses impl FieldCondition for Field
#[inline(never)]
#[unsafe(no_mangle)]
pub fn select_field(x: Field, y: Field, z: Field, w: Field) -> Field {
    let mask = x.gt(Field::from(2.0)); // Field::gt returns Field
    let m = mask.select(Field::from(1.0), Field::from(0.0));
    m.eval_raw(x, y, z, w)
}

fn main() {
    let x = Field::sequential(0.0);
    let y = Field::from(0.0);
    let z = Field::from(0.0);
    let w = Field::from(0.0);

    black_box(select_gt(
        black_box(x),
        black_box(y),
        black_box(z),
        black_box(w),
    ));
    black_box(select_field(
        black_box(x),
        black_box(y),
        black_box(z),
        black_box(w),
    ));
}
