//! A fold over nothing is its monoid's identity, and costs nothing.
//!
//! A lattice narrower than one SIMD batch has no full batch to run: `pack`
//! builds no main column fold, and only the remainder fold has terms. The
//! kernel at width 1 has two scopes fewer (the main column fold and the sum
//! inside it) than the same kernel at a width that has a main batch, and
//! still computes the same samples. A fold the kernel itself writes over
//! nothing is not compiled either, and lowers to its identity;
//! `a_fold_the_kernel_writes_over_nothing_is_its_identity` fails when
//! `arena_to_schedule`'s arm for it is reverted.
//!
//! The identity is also where the fold is *placed*, how a read it addresses
//! is lowered, and whether a derivative may pass through that read: all three
//! read the arena's variance, in which a fold over nothing is `CONST`
//! (`compute_arena_variance`).
//! `a_fold_over_nothing_is_placed_where_its_identity_is`,
//! `a_read_addressed_by_a_fold_over_nothing_is_a_broadcast` and
//! `the_derivative_of_a_read_addressed_by_a_fold_over_nothing_is_zero` fail
//! when `compute_arena_variance`'s empty-fold arm
//! (`pixelflow-ir/src/variance.rs`) is reverted.

#![cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]

use pixelflow_codegen::CompileError;
use pixelflow_codegen::emit::{ScopeTraffic, compile};
use pixelflow_ir::arena::{BufferDecl, BufferIdentity};
use pixelflow_ir::fold::{Binder, Fold, Monoid};
use pixelflow_ir::{ExprArena, ExprId, LatticeShape, OpKind};

const ROWS: u32 = 2;
const ORIGIN: [f32; 2] = [3.0, 7.0];
/// More than one batch on every tier (4, 8 and 16 lanes would leave
/// remainders of 1, 5 and 5), so the column fold has a main range.
const WIDE: usize = 37;
const PIECES: u32 = 5;
/// What the main column fold is: its own loop, and the sum's loop inside it.
const MAIN_SCOPES: usize = 2;

/// `Σ_{i<5} (x/2 + y/4 + i/8)`: a surviving fold whose body varies with the
/// column, so the lattice's column fold is not hoisted out from under it.
fn kernel() -> (ExprArena, ExprId) {
    let mut a = ExprArena::new();
    let binder = Binder::from_slot(0).expect("slot 0 exists");
    let [x, y, i] = [0, 1, binder.var()].map(|v| a.push_var(v));
    let [half, quarter, eighth] = [0.5, 0.25, 0.125].map(|v| a.push_const(v));
    let sx = a.push_binary(OpKind::Mul, x, half);
    let sy = a.push_binary(OpKind::Mul, y, quarter);
    let si = a.push_binary(OpKind::Mul, i, eighth);
    let column = a.push_binary(OpKind::Add, sx, sy);
    let body = a.push_binary(OpKind::Add, column, si);
    let root = a.push_reduce(Fold::new(Monoid::SUM, binder, 0..PIECES), body);
    (a, root)
}

/// `Σ_{i<5} (x/2 + y/4 + i/8)` in closed form. Every term is a multiple of
/// `1/8` and every partial sum is small, so `f32` sums them exactly in any
/// order.
fn exact(x: f32, y: f32) -> f32 {
    let i_sum: f32 = (0..PIECES).map(|i| i as f32 / 8.0).sum();
    PIECES as f32 * (x / 2.0 + y / 4.0) + i_sum
}

/// The samples of a `width`-wide, two-row lattice, and the number of scopes it
/// compiled.
fn run(width: usize) -> (Vec<f32>, usize) {
    let (arena, root) = kernel();
    let shape = LatticeShape::new([width as u32, ROWS]);
    let compiled = compile(&arena, root, shape).expect("the kernel compiles");
    let mut out = vec![f32::NAN; ROWS as usize * width];
    let uniforms: [f32; 0] = [];
    let ctx = [uniforms.as_ptr(), ORIGIN.as_ptr()];
    // SAFETY: the kernel declares no buffer and no uniform, so the uniform
    // block is unread; the origin block is `ORIGIN`, and `out` holds
    // `ROWS * width` samples at pitch `width`.
    unsafe {
        compiled.code.call(ctx.as_ptr(), out.as_mut_ptr(), width);
    }
    (out, compiled.traffic.scopes.len())
}

#[test]
fn a_lattice_with_no_full_batch_compiles_no_loop_for_it() {
    let (narrow, narrow_scopes) = run(1);
    let (wide, wide_scopes) = run(WIDE);
    assert_eq!(
        narrow_scopes + MAIN_SCOPES,
        wide_scopes,
        "the narrow lattice still has a main column fold, or a sum inside one"
    );
    for (width, samples) in [(1, &narrow), (WIDE, &wide)] {
        for row in 0..ROWS as usize {
            for col in 0..width {
                let (x, y) = (ORIGIN[0] + col as f32, ORIGIN[1] + row as f32);
                assert_eq!(
                    samples[row * width + col],
                    exact(x, y),
                    "width {width}, row {row}, col {col}"
                );
            }
        }
    }
}

/// `(x + Σ_{i<0} body) · (y + Π_{i<0} body)`: two folds over nothing, whose
/// bodies are never compiled, and which read as `0` and `1`.
#[test]
fn a_fold_the_kernel_writes_over_nothing_is_its_identity() {
    let mut a = ExprArena::new();
    let binder = Binder::from_slot(0).expect("slot 0 exists");
    let [x, y, i] = [0, 1, binder.var()].map(|v| a.push_var(v));
    let body = a.push_binary(OpKind::Mul, i, x);
    let none = |monoid| Fold::new(monoid, binder, 0..0);
    let sum = a.push_reduce(none(Monoid::SUM), body);
    let product = a.push_reduce(none(Monoid::PRODUCT), body);
    let left = a.push_binary(OpKind::Add, x, sum);
    let right = a.push_binary(OpKind::Add, y, product);
    let root = a.push_binary(OpKind::Mul, left, right);

    let width = 3;
    let compiled =
        compile(&a, root, LatticeShape::new([width as u32, ROWS])).expect("the kernel compiles");
    let mut out = vec![f32::NAN; ROWS as usize * width];
    let uniforms: [f32; 0] = [];
    let ctx = [uniforms.as_ptr(), ORIGIN.as_ptr()];
    // SAFETY: as in `run`.
    unsafe {
        compiled.code.call(ctx.as_ptr(), out.as_mut_ptr(), width);
    }
    for row in 0..ROWS as usize {
        for col in 0..width {
            let (x, y) = (ORIGIN[0] + col as f32, ORIGIN[1] + row as f32);
            assert_eq!(
                out[row * width + col],
                x * (y + 1.0),
                "row {row}, col {col}"
            );
        }
    }
}

/// What a compile of `root` at `WIDE` columns places where: every scope's
/// traffic and every scope's trip count, in nest order.
fn nest(a: &ExprArena, root: ExprId) -> (Vec<ScopeTraffic>, Vec<u64>) {
    let compiled =
        compile(a, root, LatticeShape::new([WIDE as u32, ROWS])).expect("the kernel compiles");
    (compiled.traffic.scopes, compiled.traffic.trips)
}

/// A fold over nothing is placed where its identity would be:
/// `y + Π_{i<0}(i·x)` compiles to the nest `y + 1` does, scope for scope.
///
/// The fold's body reads the column and the lanes (`x` is
/// `x0 + col + lane` after `collapse`). A variance read off the body,
/// `deps(body) \ {i}`, would make the sum `{row, col, lane}`, and scoping
/// would compute it per batch in the column fold rather than once per row.
/// Scoping places by the arena's variance, in which an empty fold is
/// `CONST` (`compute_arena_variance`). The two kernels take different
/// binder slots from `collapse` (the fold holds slot 0), which changes no
/// count: every scope's whole traffic is compared, not a pin.
#[test]
fn a_fold_over_nothing_is_placed_where_its_identity_is() {
    let mut folded = ExprArena::new();
    let binder = Binder::from_slot(0).expect("slot 0 exists");
    let [x, y, i] = [0, 1, binder.var()].map(|v| folded.push_var(v));
    let body = folded.push_binary(OpKind::Mul, i, x);
    let none = folded.push_reduce(Fold::new(Monoid::PRODUCT, binder, 0..0), body);
    let sum = folded.push_binary(OpKind::Add, y, none);

    let mut identity = ExprArena::new();
    let y = identity.push_var(1);
    let one = identity.push_const(Monoid::PRODUCT.identity());
    let plain = identity.push_binary(OpKind::Add, y, one);

    assert_eq!(nest(&folded, sum), nest(&identity, plain));
}

/// Rows of the table `read_at` declares. Not `ROWS`, or the last row index
/// (1.0) would be the product identity's own constant, which the arena
/// interns into one value where the folded kernel keeps two.
const TABLE_ROWS: u32 = ROWS + 2;

/// `buffer[address, y]`, the address built in the arena it is read from.
fn read_at(address: impl FnOnce(&mut ExprArena) -> ExprId) -> (ExprArena, ExprId) {
    let mut a = ExprArena::new();
    let buffer = a.declare_buffer(BufferDecl {
        id: BufferIdentity::mint(),
        width: WIDE as u32,
        height: TABLE_ROWS,
    });
    let y = a.push_var(1);
    let address = address(&mut a);
    let root = a.push_gather(buffer, address, y);
    (a, root)
}

/// A read addressed by a fold over nothing is addressed by its identity: a
/// constant, the same at every lane, so one load broadcast and not a gather.
///
/// The fold's body reads the column and the lanes, so an address read off the
/// body would carry the lane bit and lowering would split the read the other
/// way. The plain read's address is the identity itself; the two denote one
/// function and must compile to one nest, scope for scope.
#[test]
fn a_read_addressed_by_a_fold_over_nothing_is_a_broadcast() {
    let (folded, read) = read_at(|a| {
        let binder = Binder::from_slot(0).expect("slot 0 exists");
        let [x, i] = [0, binder.var()].map(|v| a.push_var(v));
        let body = a.push_binary(OpKind::Mul, i, x);
        a.push_reduce(Fold::new(Monoid::PRODUCT, binder, 0..0), body)
    });
    let (identity, plain) = read_at(|a| a.push_const(Monoid::PRODUCT.identity()));

    assert_eq!(nest(&folded, read), nest(&identity, plain));
}

/// `∂/∂x` of a read addressed by `Π_{i<trips}(i·x)`: a table read whose
/// address names the differentiated axis through a fold.
fn dx_of_a_read_addressed_by_a_fold(trips: u32) -> (ExprArena, ExprId) {
    let (mut a, read) = read_at(|a| {
        let binder = Binder::from_slot(0).expect("slot 0 exists");
        let [x, i] = [0, binder.var()].map(|v| a.push_var(v));
        let body = a.push_binary(OpKind::Mul, i, x);
        a.push_reduce(Fold::new(Monoid::PRODUCT, binder, 0..trips), body)
    });
    let axis = a.push_const(0.0);
    let root = a.push_binary(OpKind::Dwrt, read, axis);
    (a, root)
}

/// A read does not vary with a coordinate; only its address does
/// (`passes::lower_dwrt`'s tabulation rule), so the derivative of a read is
/// `0` exactly where the address is constant along the axis. An address that
/// is a fold over nothing is the monoid's identity, a constant, whatever its
/// never-evaluated body reads: the derivative compiles, and is `0` at every
/// sample. Read off the body it would name `x`, and the compile would be
/// refused as a derivative through a bound-memory read.
#[test]
fn the_derivative_of_a_read_addressed_by_a_fold_over_nothing_is_zero() {
    let (a, root) = dx_of_a_read_addressed_by_a_fold(0);
    let table: Vec<f32> = (0..WIDE * TABLE_ROWS as usize)
        .map(|i| 1.0 + i as f32)
        .collect();
    let compiled = compile(&a, root, LatticeShape::new([WIDE as u32, ROWS]))
        .expect("the derivative of a constant address compiles");
    let mut out = vec![f32::NAN; ROWS as usize * WIDE];
    let uniforms: [f32; 0] = [];
    let ctx = [table.as_ptr(), uniforms.as_ptr(), ORIGIN.as_ptr()];
    // SAFETY: the kernel declares one buffer (`table`, the context's first
    // slot) and no uniform; the origin block is `ORIGIN`, and `out` holds
    // `ROWS * WIDE` samples at pitch `WIDE`.
    unsafe {
        compiled.code.call(ctx.as_ptr(), out.as_mut_ptr(), WIDE);
    }
    for (sample, got) in out.iter().enumerate() {
        assert_eq!(*got, 0.0, "sample {sample}");
    }
}

/// The other side of the same rule: a fold over something still names its
/// body's axis, and the derivative through that address is refused.
#[test]
fn the_derivative_of_a_read_addressed_by_a_fold_over_something_is_refused() {
    let (a, root) = dx_of_a_read_addressed_by_a_fold(PIECES);
    let refused = compile(&a, root, LatticeShape::new([WIDE as u32, ROWS])).err();
    assert_eq!(
        refused,
        Some(CompileError::Legalize(
            "lower_dwrt: cannot differentiate a bound-memory read"
        ))
    );
}

#[test]
#[should_panic(expected = "degenerate extent")]
fn a_lattice_with_no_column_is_refused() {
    let (arena, root) = kernel();
    drop(compile(&arena, root, LatticeShape::new([0, ROWS])));
}
