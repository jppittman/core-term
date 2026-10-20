//! `jit_cache::compile` documents that canonically identical kernels at one
//! shape share one compiled region, and that a different shape does not. A
//! key that stopped hitting would recompile every glyph and nothing would
//! compute a wrong value, so the promise is observed here, on the handle
//! `compile` returns.

#![cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]

use std::sync::Arc;

use pixelflow_codegen::jit_cache::compile;
use pixelflow_ir::{ExprArena, Kernel, LatticeShape, OpKind};

const SHAPE: LatticeShape = LatticeShape::new([64, 64]);

/// `sqrt(x*x + y*y)`, built into `a`.
fn circle_in(mut a: ExprArena) -> Kernel {
    let x = a.push_var(0);
    let y = a.push_var(1);
    let x2 = a.push_binary(OpKind::Mul, x, x);
    let y2 = a.push_binary(OpKind::Mul, y, y);
    let s = a.push_binary(OpKind::Add, x2, y2);
    let root = a.push_unary(OpKind::Sqrt, s);
    Kernel::from_parts(a, root)
}

#[test]
fn identical_kernels_share_code() {
    let m1 = compile(&circle_in(ExprArena::new()), SHAPE)
        .expect("compile")
        .kernel;
    // Construction garbage: unreachable nodes must not perturb the key.
    let mut garbage = ExprArena::new();
    let g = garbage.push_const(123.0);
    garbage.push_unary(OpKind::Sqrt, g);
    let m2 = compile(&circle_in(garbage), SHAPE).expect("compile").kernel;
    assert!(
        Arc::ptr_eq(&m1, &m2),
        "canonically identical kernels must share one compiled region"
    );
}

/// `X + Y` with its leaves pushed in either order is one program, so it is
/// one compiled region: the key is a function of what the root denotes, not
/// of where a builder happened to push the leaves.
#[test]
fn x_plus_y_pushed_in_either_order_is_one_entry() {
    const ONLY_HERE: f32 = 0.6180339;
    let x_first = {
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let y = a.push_var(1);
        let k = a.push_const(ONLY_HERE);
        let sum = a.push_binary(OpKind::Add, x, y);
        let root = a.push_binary(OpKind::Mul, sum, k);
        Kernel::from_parts(a, root)
    };
    let y_first = {
        let mut a = ExprArena::new();
        let k = a.push_const(ONLY_HERE);
        let y = a.push_var(1);
        let x = a.push_var(0);
        let sum = a.push_binary(OpKind::Add, x, y);
        let root = a.push_binary(OpKind::Mul, sum, k);
        Kernel::from_parts(a, root)
    };
    let m1 = compile(&x_first, SHAPE).expect("compile").kernel;
    let m2 = compile(&y_first, SHAPE).expect("compile").kernel;
    assert!(
        Arc::ptr_eq(&m1, &m2),
        "one program pushed in two orders must share one compiled region"
    );
}
