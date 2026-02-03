//! ARM64/NEON instruction encoding.
//!
//! Each function emits raw machine code bytes for one instruction (or a small fixed sequence).
//! These are the "atoms" that compound operations are built from.

use super::Reg;
use crate::kind::OpKind;

// =============================================================================
// Instruction Encoding Helpers
// =============================================================================

/// Encode a NEON 3-same instruction (binary vector ops).
/// Format: Vd.4S, Vn.4S, Vm.4S
#[inline]
fn encode_3same(opcode: u32, dst: Reg, src1: Reg, src2: Reg) -> u32 {
    opcode
        | (dst.0 as u32 & 0x1F)
        | ((src1.0 as u32 & 0x1F) << 5)
        | ((src2.0 as u32 & 0x1F) << 16)
}

/// Encode a NEON 2-reg misc instruction (unary vector ops).
/// Format: Vd.4S, Vn.4S
#[inline]
fn encode_2misc(opcode: u32, dst: Reg, src: Reg) -> u32 {
    opcode | (dst.0 as u32 & 0x1F) | ((src.0 as u32 & 0x1F) << 5)
}

/// Write a 32-bit instruction to the code buffer.
#[inline]
fn emit32(code: &mut Vec<u8>, inst: u32) {
    code.extend_from_slice(&inst.to_le_bytes());
}

// =============================================================================
// Load / Store
// =============================================================================

/// LDR Vd, [X0, #offset] - Load 128-bit vector from base + offset
pub fn emit_ldr_voff(code: &mut Vec<u8>, dst: Reg, offset: u16) {
    // LDR Qt, [Xn, #imm] - 128-bit load
    // Encoding: 0x3DC00000 | (imm12 << 10) | (Rn << 5) | Rt
    // imm12 is offset/16 for 128-bit loads
    let imm12 = (offset / 16) as u32;
    let inst = 0x3DC00000 | (imm12 << 10) | (0 << 5) | (dst.0 as u32); // X0 as base
    emit32(code, inst);
}

/// STR Vt, [X0, #offset] - Store 128-bit vector to base + offset
pub fn emit_str_voff(code: &mut Vec<u8>, src: Reg, offset: u16) {
    let imm12 = (offset / 16) as u32;
    let inst = 0x3D800000 | (imm12 << 10) | (0 << 5) | (src.0 as u32);
    emit32(code, inst);
}

/// LDR Vd, [SP, #offset] - Load 128-bit vector from stack
pub fn emit_ldr_sp(code: &mut Vec<u8>, dst: Reg, offset: u16) {
    // LDR Qt, [SP, #imm] - 128-bit load, SP = X31
    // Encoding: 0x3DC00000 | (imm12 << 10) | (Rn << 5) | Rt
    // imm12 is offset/16 for 128-bit loads
    let imm12 = (offset / 16) as u32;
    let inst = 0x3DC00000 | (imm12 << 10) | (31 << 5) | (dst.0 as u32); // SP = X31
    emit32(code, inst);
}

/// STR Vt, [SP, #offset] - Store 128-bit vector to stack
pub fn emit_str_sp(code: &mut Vec<u8>, src: Reg, offset: u16) {
    // STR Qt, [SP, #imm] - 128-bit store, SP = X31
    let imm12 = (offset / 16) as u32;
    let inst = 0x3D800000 | (imm12 << 10) | (31 << 5) | (src.0 as u32); // SP = X31
    emit32(code, inst);
}

/// SUB SP, SP, #imm - Allocate stack frame
pub fn emit_sub_sp(code: &mut Vec<u8>, size: u16) {
    // SUB SP, SP, #imm12
    // Encoding: 0xD10003FF | (imm12 << 10)
    let inst = 0xD10003FF | ((size as u32) << 10);
    emit32(code, inst);
}

/// ADD SP, SP, #imm - Deallocate stack frame
pub fn emit_add_sp(code: &mut Vec<u8>, size: u16) {
    // ADD SP, SP, #imm12
    // Encoding: 0x910003FF | (imm12 << 10)
    let inst = 0x910003FF | ((size as u32) << 10);
    emit32(code, inst);
}

// =============================================================================
// Arithmetic - Single Instructions
// =============================================================================

/// FADD Vd.4S, Vn.4S, Vm.4S
pub fn emit_fadd(code: &mut Vec<u8>, dst: Reg, src1: Reg, src2: Reg) {
    emit32(code, encode_3same(0x4E20D400, dst, src1, src2));
}

/// FSUB Vd.4S, Vn.4S, Vm.4S
pub fn emit_fsub(code: &mut Vec<u8>, dst: Reg, src1: Reg, src2: Reg) {
    emit32(code, encode_3same(0x4EA0D400, dst, src1, src2));
}

/// FMUL Vd.4S, Vn.4S, Vm.4S
pub fn emit_fmul(code: &mut Vec<u8>, dst: Reg, src1: Reg, src2: Reg) {
    emit32(code, encode_3same(0x6E20DC00, dst, src1, src2));
}

/// FDIV Vd.4S, Vn.4S, Vm.4S
pub fn emit_fdiv(code: &mut Vec<u8>, dst: Reg, src1: Reg, src2: Reg) {
    emit32(code, encode_3same(0x6E20FC00, dst, src1, src2));
}

/// FMLA Vd.4S, Vn.4S, Vm.4S (fused multiply-add: Vd += Vn * Vm)
pub fn emit_fmla(code: &mut Vec<u8>, dst: Reg, src1: Reg, src2: Reg) {
    emit32(code, encode_3same(0x4E20CC00, dst, src1, src2));
}

/// FMIN Vd.4S, Vn.4S, Vm.4S
pub fn emit_fmin(code: &mut Vec<u8>, dst: Reg, src1: Reg, src2: Reg) {
    emit32(code, encode_3same(0x4EA0F400, dst, src1, src2));
}

/// FMAX Vd.4S, Vn.4S, Vm.4S
pub fn emit_fmax(code: &mut Vec<u8>, dst: Reg, src1: Reg, src2: Reg) {
    emit32(code, encode_3same(0x4E20F400, dst, src1, src2));
}

/// FSQRT Vd.4S, Vn.4S
pub fn emit_fsqrt(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    emit32(code, encode_2misc(0x6EA1F800, dst, src));
}

/// FABS Vd.4S, Vn.4S
pub fn emit_fabs(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    emit32(code, encode_2misc(0x4EA0F800, dst, src));
}

/// FNEG Vd.4S, Vn.4S
pub fn emit_fneg(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    emit32(code, encode_2misc(0x6EA0F800, dst, src));
}

/// FRINTM Vd.4S, Vn.4S (floor)
pub fn emit_frintm(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    emit32(code, encode_2misc(0x4E219800, dst, src));
}

/// FRINTP Vd.4S, Vn.4S (ceil)
pub fn emit_frintp(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    emit32(code, encode_2misc(0x4EA18800, dst, src));
}

// =============================================================================
// Approximate operations (estimate + refinement)
// =============================================================================

/// FRSQRTE + FRSQRTS refinement (~3 instructions for rsqrt)
pub fn emit_frsqrt(code: &mut Vec<u8>, dst: Reg, src: Reg, scratch: Reg) {
    // est = frsqrte(src)
    emit32(code, encode_2misc(0x6EA1D800, dst, src));
    // scratch = est * est
    emit32(code, encode_3same(0x6E20DC00, scratch, dst, dst));
    // scratch = frsqrts(src, scratch) = (3 - src * scratch) / 2
    emit32(code, encode_3same(0x4EA0FC00, scratch, src, scratch));
    // dst = est * scratch (refined)
    emit32(code, encode_3same(0x6E20DC00, dst, dst, scratch));
}

/// FRECPE + FRECPS refinement (~3 instructions for recip)
pub fn emit_frecip(code: &mut Vec<u8>, dst: Reg, src: Reg, scratch: Reg) {
    // est = frecpe(src)
    emit32(code, encode_2misc(0x4EA1D800, dst, src));
    // scratch = frecps(src, est) = 2 - src * est
    emit32(code, encode_3same(0x4E20FC00, scratch, src, dst));
    // dst = est * scratch (refined)
    emit32(code, encode_3same(0x6E20DC00, dst, dst, scratch));
}

// =============================================================================
// Comparisons
// =============================================================================

/// FCMGT Vd.4S, Vn.4S, Vm.4S (greater than)
pub fn emit_fcmgt(code: &mut Vec<u8>, dst: Reg, src1: Reg, src2: Reg) {
    emit32(code, encode_3same(0x6EA0E400, dst, src1, src2));
}

/// FCMGE Vd.4S, Vn.4S, Vm.4S (greater or equal)
pub fn emit_fcmge(code: &mut Vec<u8>, dst: Reg, src1: Reg, src2: Reg) {
    emit32(code, encode_3same(0x6E20E400, dst, src1, src2));
}

/// FCMEQ Vd.4S, Vn.4S, Vm.4S (equal)
pub fn emit_fcmeq(code: &mut Vec<u8>, dst: Reg, src1: Reg, src2: Reg) {
    emit32(code, encode_3same(0x4E20E400, dst, src1, src2));
}

// =============================================================================
// Selection / Blending
// =============================================================================

/// BSL Vd.16B, Vn.16B, Vm.16B (bitwise select: Vd = (Vd & Vn) | (~Vd & Vm))
pub fn emit_bsl(code: &mut Vec<u8>, mask: Reg, if_true: Reg, if_false: Reg) {
    emit32(code, encode_3same(0x6E601C00, mask, if_true, if_false));
}

// =============================================================================
// Constants
// =============================================================================

/// Load a floating-point constant into a vector register.
///
/// Strategy: Load 32-bit immediate into GP register (W16) via MOVZ + MOVK,
/// then DUP to broadcast to all lanes.
///
/// TODO: Use a constant pool with LDR for better performance (fewer instructions,
/// better I-cache utilization).
pub fn emit_fmov_imm(code: &mut Vec<u8>, dst: Reg, val: f32, _scratch: [Reg; 4]) {
    let bits = val.to_bits();

    if bits == 0 {
        // MOVI Vd.4S, #0 - single instruction for zero
        emit32(code, 0x4F000400 | (dst.0 as u32));
        return;
    }

    // General case: load via GP register (W16)
    // This is 3 instructions but works for any f32 value.
    // Use W16 (IP0) as scratch - it's caller-saved and not used for arguments
    let lo16 = (bits & 0xFFFF) as u32;
    let hi16 = (bits >> 16) as u32;

    // MOVZ W16, #lo16
    emit32(code, 0x52800010 | (lo16 << 5));

    // MOVK W16, #hi16, LSL #16
    emit32(code, 0x72A00010 | (hi16 << 5));

    // DUP Vd.4S, W16
    emit32(code, 0x4E040C00 | (dst.0 as u32) | (16 << 5));
}

/// Duplicate scalar to all lanes: DUP Vd.4S, Vn.S[0]
pub fn emit_dup_s0(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    emit32(code, 0x4E040400 | (dst.0 as u32) | ((src.0 as u32) << 5));
}

// =============================================================================
// Compound Operations (emit full instruction sequences)
// =============================================================================

/// Emit unary operation - dispatches to appropriate instruction(s)
pub fn emit_unary(code: &mut Vec<u8>, op: OpKind, dst: Reg, src: Reg, scratch: [Reg; 4]) {
    match op {
        OpKind::Neg => emit_fneg(code, dst, src),
        OpKind::Abs => emit_fabs(code, dst, src),
        OpKind::Sqrt => emit_fsqrt(code, dst, src),
        OpKind::Rsqrt => emit_frsqrt(code, dst, src, scratch[0]),
        OpKind::Recip => emit_frecip(code, dst, src, scratch[0]),
        OpKind::Floor => emit_frintm(code, dst, src),
        OpKind::Ceil => emit_frintp(code, dst, src),

        // Transcendentals - emit full polynomial sequence
        OpKind::Sin => emit_sin_poly(code, dst, src, scratch),
        OpKind::Cos => emit_cos_poly(code, dst, src, scratch),
        OpKind::Exp => emit_exp_poly(code, dst, src, scratch),
        OpKind::Exp2 => emit_exp2_poly(code, dst, src, scratch),
        OpKind::Ln => emit_ln_poly(code, dst, src, scratch),
        OpKind::Log2 => emit_log2_poly(code, dst, src, scratch),

        _ => panic!("unary emit not implemented for {:?}", op),
    }
}

/// Emit binary operation
pub fn emit_binary(code: &mut Vec<u8>, op: OpKind, dst: Reg, src1: Reg, src2: Reg) {
    match op {
        OpKind::Add => emit_fadd(code, dst, src1, src2),
        OpKind::Sub => emit_fsub(code, dst, src1, src2),
        OpKind::Mul => emit_fmul(code, dst, src1, src2),
        OpKind::Div => emit_fdiv(code, dst, src1, src2),
        OpKind::Min => emit_fmin(code, dst, src1, src2),
        OpKind::Max => emit_fmax(code, dst, src1, src2),

        // Comparisons (result is mask in dst)
        OpKind::Gt => emit_fcmgt(code, dst, src1, src2),
        OpKind::Ge => emit_fcmge(code, dst, src1, src2),
        OpKind::Lt => emit_fcmgt(code, dst, src2, src1), // swap args
        OpKind::Le => emit_fcmge(code, dst, src2, src1),
        OpKind::Eq => emit_fcmeq(code, dst, src1, src2),
        OpKind::Ne => {
            // Ne = not Eq
            emit_fcmeq(code, dst, src1, src2);
            // MVN (bitwise NOT) - placeholder, need to add this
        }

        _ => panic!("binary emit not implemented for {:?}", op),
    }
}

/// Emit ternary operation
pub fn emit_ternary(code: &mut Vec<u8>, op: OpKind, dst: Reg, a: Reg, b: Reg, c: Reg) {
    match op {
        OpKind::MulAdd => {
            // dst = a * b + c
            // FMLA does: dst = dst + src1 * src2
            //
            // Problem: if dst == a and dst != c, copying c to dst would clobber a
            // before we can use it. In that case, use FMUL + FADD instead.
            if (dst.0 == a.0 || dst.0 == b.0) && dst.0 != c.0 {
                // dst overlaps with a or b, can't use FMLA safely
                // Use FMUL + FADD: dst = a * b, then dst = dst + c
                emit_fmul(code, dst, a, b);
                emit_fadd(code, dst, dst, c);
            } else {
                // Safe to use FMLA
                if dst.0 != c.0 {
                    // MOV dst, c first
                    emit32(code, 0x4EA01C00 | (dst.0 as u32) | ((c.0 as u32) << 5) | ((c.0 as u32) << 16));
                }
                emit_fmla(code, dst, a, b);
            }
        }

        OpKind::Select => {
            // dst = a ? b : c (a is mask)
            // Need to move mask to dst first for BSL
            if dst.0 != a.0 {
                emit32(code, 0x4EA01C00 | (dst.0 as u32) | ((a.0 as u32) << 5) | ((a.0 as u32) << 16));
            }
            emit_bsl(code, dst, b, c);
        }

        OpKind::Clamp => {
            // dst = clamp(a, b, c) = max(min(a, c), b)
            emit_fmin(code, dst, a, c);  // dst = min(a, hi)
            emit_fmax(code, dst, dst, b); // dst = max(dst, lo)
        }

        _ => panic!("ternary emit not implemented for {:?}", op),
    }
}

// =============================================================================
// Transcendental Polynomials
// =============================================================================

/// Sin via Chebyshev polynomial - emits ~15 instructions
pub fn emit_sin_poly(code: &mut Vec<u8>, dst: Reg, src: Reg, scratch: [Reg; 4]) {
    // TODO: Implement full polynomial sequence
    // For now, placeholder
    emit_fmul(code, dst, src, src); // x^2 as placeholder
}

/// Cos via Chebyshev polynomial
pub fn emit_cos_poly(code: &mut Vec<u8>, dst: Reg, src: Reg, scratch: [Reg; 4]) {
    // TODO: Implement
    emit_fmul(code, dst, src, src);
}

/// Exp via polynomial
pub fn emit_exp_poly(code: &mut Vec<u8>, dst: Reg, src: Reg, scratch: [Reg; 4]) {
    // TODO: Implement
    emit_fmul(code, dst, src, src);
}

/// Exp2 via polynomial
pub fn emit_exp2_poly(code: &mut Vec<u8>, dst: Reg, src: Reg, scratch: [Reg; 4]) {
    // TODO: Implement
    emit_fmul(code, dst, src, src);
}

/// Natural log via polynomial
pub fn emit_ln_poly(code: &mut Vec<u8>, dst: Reg, src: Reg, scratch: [Reg; 4]) {
    // TODO: Implement
    emit_fmul(code, dst, src, src);
}

/// Log2 via polynomial
pub fn emit_log2_poly(code: &mut Vec<u8>, dst: Reg, src: Reg, scratch: [Reg; 4]) {
    // TODO: Implement
    emit_fmul(code, dst, src, src);
}

// =============================================================================
// Prologue / Epilogue
// =============================================================================

/// Emit function prologue
pub fn emit_prologue(code: &mut Vec<u8>) {
    // For a simple JIT kernel, we might not need much
    // Input pointer already in X0
    // Just ensure we're aligned
}

/// Emit function epilogue (return)
pub fn emit_epilogue(code: &mut Vec<u8>, result: Reg) {
    // Move result to V0 if not already there
    if result.0 != 0 {
        // MOV V0, Vresult (ORR Vd.16B, Vn.16B, Vn.16B)
        emit32(code, 0x4EA01C00 | ((result.0 as u32) << 5) | ((result.0 as u32) << 16));
    }
    // RET
    emit32(code, 0xD65F03C0);
}
