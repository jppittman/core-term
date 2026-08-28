//! Exercises the two proc-macro entry points that unit tests inside
//! `pixelflow-compiler` cannot reach directly: `proc_macro::TokenStream`
//! only converts from a real macro invocation, never from a function call in
//! a `#[test]` in the same crate. Integration tests are compiled as a
//! separate crate, so the macros here go through genuine expansion.

use pixelflow_compiler::{ManifoldExpr, kernel_value};
use pixelflow_core::Kernel;

#[test]
fn kernel_value_macro_expands_to_a_usable_kernel_builder() {
    // Under a mutated (empty) macro expansion this would fail to compile:
    // `let builder = /* nothing */;` is a syntax error, and `Kernel::sum`
    // requires a real `Kernel` value, not whatever an empty expansion left
    // behind. Reaching the assertion at all proves real code was generated.
    let builder = kernel_value!(|r: f32| X + r);
    let a: Kernel = builder(1.0);
    let b: Kernel = builder(2.0);
    let _combined: Kernel = Kernel::sum(&[a, b]);
}

#[derive(ManifoldExpr)]
#[allow(dead_code)]
struct DummyCombinator<M>(M);

fn assert_manifold_expr<T: pixelflow_core::ManifoldExpr>() {}

#[test]
fn derive_manifold_expr_implements_the_marker_trait_for_the_annotated_type() {
    assert_manifold_expr::<DummyCombinator<f32>>();
}
