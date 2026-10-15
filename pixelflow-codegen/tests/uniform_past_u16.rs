//! A uniform at a slot past what 16 bits index is read at its own slot.
//!
//! A kernel declaring more than 65,536 uniforms reads the last one, not slot 3:
//! the control plane is 64-bit (CLAUDE.md), and a narrowed offset would compute
//! a plausible value from the wrong argument on every backend.

#![cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]

use pixelflow_codegen::emit::compile;
use pixelflow_ir::arena::{ExprArena, UniformDecl, UniformIdentity};
use pixelflow_ir::{LatticeShape, OpKind};

/// Slot 3 shifted up by a full 16-bit range: the slot a 16-bit offset would
/// wrap to 3.
const PAST_U16: usize = 3 + (u16::MAX as usize + 1);
const ARGUMENTS: usize = PAST_U16 + 1;
const WIDTH: usize = 11;
const ROWS: usize = 2;
const ORIGIN: [f32; 2] = [2.0, 5.0];

#[test]
fn a_uniform_past_u16_is_read_at_its_value() {
    let mut a = ExprArena::new();
    let mut last = None;
    for i in 0..ARGUMENTS {
        last = Some(a.declare_uniform(UniformDecl {
            id: UniformIdentity::mint(),
            default: i as f32,
        }));
    }
    let last = last.expect("declared");
    let x = a.push_var(0);
    let y = a.push_var(1);
    let xy = a.push_binary(OpKind::Add, x, y);
    let u = a.push_uniform(last);
    let root = a.push_binary(OpKind::Add, xy, u);

    let compiled =
        compile(&a, root, LatticeShape::new([WIDTH as u32, ROWS as u32])).expect("compiles");
    // Slot i holds -(i as f32): slot 3 (what a wrapped offset reads) holds
    // -3.0, slot PAST_U16 holds -65539.0.
    let block: Vec<f32> = (0..ARGUMENTS).map(|i| -(i as f32)).collect();
    let mut out = vec![f32::NAN; WIDTH * ROWS];
    let ctx = [block.as_ptr(), ORIGIN.as_ptr()];
    unsafe { compiled.code.call(ctx.as_ptr(), out.as_mut_ptr(), WIDTH) };
    for row in 0..ROWS {
        for col in 0..WIDTH {
            let want = ORIGIN[0] + col as f32 + ORIGIN[1] + row as f32 - PAST_U16 as f32;
            assert_eq!(out[row * WIDTH + col], want, "row {row} col {col}");
        }
    }
}
