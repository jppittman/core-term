//! x86-64 SSE/AVX instruction encoding.
//!
//! Each function emits raw machine code bytes for one instruction (or a small fixed sequence).

use super::Reg;
use crate::kind::OpKind;

// =============================================================================
// Encoding Helpers
// =============================================================================

/// Emit a VEX-encoded 3-operand instruction (AVX style).
/// VEX.128.0F: dst = op(src1, src2)
fn emit_vex_128_0f(code: &mut Vec<u8>, opcode: u8, dst: Reg, src1: Reg, src2: Reg) {
    // 3-byte VEX prefix for xmm0-xmm15
    // VEX.128.0F: C4 RXB.01111 W.vvvv.L.pp
    let r = if dst.0 >= 8 { 0 } else { 0x80 };
    let x = 0x40; // X not used for register-register
    let b = if src2.0 >= 8 { 0 } else { 0x20 };
    let vvvv = (!src1.0 & 0xF) << 3;

    code.push(0xC4);
    code.push(r | x | b | 0x01); // map = 0F
    code.push(vvvv | 0x00);      // W=0, L=0 (128-bit), pp=00
    code.push(opcode);
    code.push(0xC0 | ((dst.0 & 7) << 3) | (src2.0 & 7)); // ModRM
}

/// Emit SSE instruction (legacy encoding, 2-operand: dst op= src)
fn emit_sse_rr(code: &mut Vec<u8>, prefix: Option<u8>, opcode: &[u8], dst: Reg, src: Reg) {
    if let Some(p) = prefix {
        code.push(p);
    }

    // REX prefix if needed (for xmm8-xmm15)
    let rex = 0x40
        | (if dst.0 >= 8 { 0x04 } else { 0 })
        | (if src.0 >= 8 { 0x01 } else { 0 });
    if rex != 0x40 {
        code.push(rex);
    }

    code.extend_from_slice(opcode);
    code.push(0xC0 | ((dst.0 & 7) << 3) | (src.0 & 7));
}

// =============================================================================
// Load / Store
// =============================================================================

/// MOVAPS xmm, [rdi + offset] - Load 128-bit aligned
pub fn emit_movaps_load(code: &mut Vec<u8>, dst: Reg, offset: u16) {
    // REX if needed
    if dst.0 >= 8 {
        code.push(0x44); // REX.R
    }
    code.push(0x0F);
    code.push(0x28);

    if offset == 0 {
        code.push(0x07 | ((dst.0 & 7) << 3)); // [rdi]
    } else if offset < 128 {
        code.push(0x47 | ((dst.0 & 7) << 3)); // [rdi + disp8]
        code.push(offset as u8);
    } else {
        code.push(0x87 | ((dst.0 & 7) << 3)); // [rdi + disp32]
        code.extend_from_slice(&(offset as u32).to_le_bytes());
    }
}

/// MOVAPS [rdi + offset], xmm - Store 128-bit aligned
pub fn emit_movaps_store(code: &mut Vec<u8>, src: Reg, offset: u16) {
    if src.0 >= 8 {
        code.push(0x44);
    }
    code.push(0x0F);
    code.push(0x29);

    if offset == 0 {
        code.push(0x07 | ((src.0 & 7) << 3));
    } else if offset < 128 {
        code.push(0x47 | ((src.0 & 7) << 3));
        code.push(offset as u8);
    } else {
        code.push(0x87 | ((src.0 & 7) << 3));
        code.extend_from_slice(&(offset as u32).to_le_bytes());
    }
}

// =============================================================================
// Arithmetic (SSE)
// =============================================================================

/// ADDPS xmm, xmm
pub fn emit_addps(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    emit_sse_rr(code, None, &[0x0F, 0x58], dst, src);
}

/// SUBPS xmm, xmm
pub fn emit_subps(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    emit_sse_rr(code, None, &[0x0F, 0x5C], dst, src);
}

/// MULPS xmm, xmm
pub fn emit_mulps(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    emit_sse_rr(code, None, &[0x0F, 0x59], dst, src);
}

/// DIVPS xmm, xmm
pub fn emit_divps(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    emit_sse_rr(code, None, &[0x0F, 0x5E], dst, src);
}

/// SQRTPS xmm, xmm
pub fn emit_sqrtps(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    emit_sse_rr(code, None, &[0x0F, 0x51], dst, src);
}

/// RSQRTPS xmm, xmm (approximate)
pub fn emit_rsqrtps(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    emit_sse_rr(code, None, &[0x0F, 0x52], dst, src);
}

/// RCPPS xmm, xmm (approximate reciprocal)
pub fn emit_rcpps(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    emit_sse_rr(code, None, &[0x0F, 0x53], dst, src);
}

/// MINPS xmm, xmm
pub fn emit_minps(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    emit_sse_rr(code, None, &[0x0F, 0x5D], dst, src);
}

/// MAXPS xmm, xmm
pub fn emit_maxps(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    emit_sse_rr(code, None, &[0x0F, 0x5F], dst, src);
}

// =============================================================================
// Bitwise
// =============================================================================

/// XORPS xmm, xmm (also used for negation via sign bit flip)
pub fn emit_xorps(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    emit_sse_rr(code, None, &[0x0F, 0x57], dst, src);
}

/// ANDPS xmm, xmm
pub fn emit_andps(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    emit_sse_rr(code, None, &[0x0F, 0x54], dst, src);
}

// =============================================================================
// Constants
// =============================================================================

/// Load constant into register (placeholder - needs constant pool)
pub fn emit_const(code: &mut Vec<u8>, dst: Reg, _val: f32, _scratch: [Reg; 4]) {
    // XORPS dst, dst (zero for now)
    emit_xorps(code, dst, dst);
    // TODO: proper constant pool
}

// =============================================================================
// High-level dispatch
// =============================================================================

/// Emit unary operation
pub fn emit_unary(code: &mut Vec<u8>, op: OpKind, dst: Reg, src: Reg, scratch: [Reg; 4]) {
    match op {
        OpKind::Sqrt => emit_sqrtps(code, dst, src),
        OpKind::Rsqrt => {
            emit_rsqrtps(code, dst, src);
            // TODO: Newton-Raphson refinement
        }
        OpKind::Recip => emit_rcpps(code, dst, src),

        // TODO: transcendentals
        _ => panic!("unary emit not implemented for {:?}", op),
    }
}

/// Emit binary operation
pub fn emit_binary(code: &mut Vec<u8>, op: OpKind, dst: Reg, src1: Reg, src2: Reg) {
    // SSE is 2-operand, so we may need to move first
    if dst.0 != src1.0 {
        emit_sse_rr(code, None, &[0x0F, 0x28], dst, src1); // MOVAPS dst, src1
    }

    match op {
        OpKind::Add => emit_addps(code, dst, src2),
        OpKind::Sub => emit_subps(code, dst, src2),
        OpKind::Mul => emit_mulps(code, dst, src2),
        OpKind::Div => emit_divps(code, dst, src2),
        OpKind::Min => emit_minps(code, dst, src2),
        OpKind::Max => emit_maxps(code, dst, src2),

        _ => panic!("binary emit not implemented for {:?}", op),
    }
}

/// Emit ternary operation
pub fn emit_ternary(code: &mut Vec<u8>, op: OpKind, dst: Reg, a: Reg, b: Reg, c: Reg) {
    match op {
        OpKind::MulAdd => {
            // Without FMA: dst = a * b; dst = dst + c
            if dst.0 != a.0 {
                emit_sse_rr(code, None, &[0x0F, 0x28], dst, a);
            }
            emit_mulps(code, dst, b);
            emit_addps(code, dst, c);
        }

        OpKind::Clamp => {
            // max(min(a, c), b)
            if dst.0 != a.0 {
                emit_sse_rr(code, None, &[0x0F, 0x28], dst, a);
            }
            emit_minps(code, dst, c);
            emit_maxps(code, dst, b);
        }

        _ => panic!("ternary emit not implemented for {:?}", op),
    }
}

// =============================================================================
// Prologue / Epilogue
// =============================================================================

/// Emit function prologue
pub fn emit_prologue(_code: &mut Vec<u8>) {
    // Input pointer in rdi (System V) or rcx (Windows)
    // For now, assume System V
}

/// Emit function epilogue
pub fn emit_epilogue(code: &mut Vec<u8>, result: Reg) {
    // Move result to xmm0 if not already there
    if result.0 != 0 {
        emit_sse_rr(code, None, &[0x0F, 0x28], Reg(0), result);
    }
    // RET
    code.push(0xC3);
}
