//! A context slot past what `ldr x`'s scaled 12-bit immediate reaches, read
//! and the values checked.
//!
//! The context is the array of buffer base pointers, then the uniform and
//! origin blocks. A kernel that declares 4,097 buffers reads the last through
//! slot 4,096, one past the 4,095 that `ldr x, [ctx, #slot·8]` encodes on
//! NEON: an address too far for the instruction is computed first, and the
//! pointer it loads must still be the last buffer's, not the first's. The
//! legacy NEON pipeline panics on it, so this runs on the selection pipeline.

#![cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]

use pixelflow_codegen::emit::compile;
use pixelflow_ir::arena::{BufferDecl, BufferIdentity, ExprArena};
use pixelflow_ir::{LatticeShape, OpKind};

include!("support/knob.rs");

/// One slot past the scaled immediate's last: 4,096.
const BUFFERS: usize = 4097;
const WIDTH: usize = 11;
const ROWS: usize = 3;
const ORIGIN: [f32; 2] = [2.0, 5.0];

#[test]
fn a_context_slot_past_imm12_loads() {
    if !selection() {
        return;
    }
    let mut a = ExprArena::new();
    let buffers: Vec<_> = (0..BUFFERS)
        .map(|_| {
            a.declare_buffer(BufferDecl {
                id: BufferIdentity::mint(),
                width: 64,
                height: 1,
            })
        })
        .collect();
    let x = a.push_var(0);
    let y = a.push_var(1);
    let last = a.push_buffer(buffers[BUFFERS - 1]);
    let gathered = a.push_binary(OpKind::RawGather, last, x);
    let root = a.push_binary(OpKind::Add, gathered, y);
    let shape = LatticeShape::new([WIDTH as u32, ROWS as u32]);
    let code = compile(&a, root, shape).expect("a kernel reading slot 4,096 compiles");

    // The buffers a wrapped slot would read are all zero, so a kernel that
    // loads the wrong pointer gets `y` alone.
    let table: Vec<f32> = (0..64).map(|i| 100.0 + i as f32).collect();
    let zeros = [0.0f32; 64];
    let uniforms: [f32; 0] = [];
    let mut ctx: Vec<*const f32> = (0..BUFFERS - 1).map(|_| zeros.as_ptr()).collect();
    ctx.extend([table.as_ptr(), uniforms.as_ptr(), ORIGIN.as_ptr()]);
    let mut out = vec![f32::NAN; ROWS * WIDTH];
    unsafe { code.code.call(ctx.as_ptr(), out.as_mut_ptr(), WIDTH) };
    for row in 0..ROWS {
        for col in 0..WIDTH {
            let (x, y) = (ORIGIN[0] + col as f32, ORIGIN[1] + row as f32);
            assert_eq!(
                out[row * WIDTH + col],
                table[x as usize] + y,
                "row {row} col {col}"
            );
        }
    }
}
