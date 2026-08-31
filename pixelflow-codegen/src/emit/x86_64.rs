//! x86-64 SSE/AVX instruction encoding.
//!
//! Each function emits raw machine code bytes for one instruction (or a small fixed sequence).
//!
//! Two encoding strategies:
//! - **Legacy SSE** (2-operand, destructive): `dst op= src`
//! - **VEX** (3-operand, non-destructive): `dst = op(src1, src2)`
//!
//! Transcendental builtins (atan2, atan, asin, acos) use VEX encoding for the
//! 3-operand form which avoids extra MOV instructions in multi-step sequences.

use super::{Reg, unimplemented_op};
use alloc::vec::Vec;
use pixelflow_ir::kind::OpKind;

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
    code.push(vvvv); // W=0, L=0 (128-bit), pp=00
    code.push(opcode);
    code.push(0xC0 | ((dst.0 & 7) << 3) | (src2.0 & 7)); // ModRM
}

/// Emit SSE instruction (legacy encoding, 2-operand: dst op= src)
fn emit_sse_rr(code: &mut Vec<u8>, prefix: Option<u8>, opcode: &[u8], dst: Reg, src: Reg) {
    if let Some(p) = prefix {
        code.push(p);
    }

    // REX prefix if needed (for xmm8-xmm15)
    let rex = 0x40 | (if dst.0 >= 8 { 0x04 } else { 0 }) | (if src.0 >= 8 { 0x01 } else { 0 });
    if rex != 0x40 {
        code.push(rex);
    }

    code.extend_from_slice(opcode);
    code.push(0xC0 | ((dst.0 & 7) << 3) | (src.0 & 7));
}

// =============================================================================
// Load / Store
// =============================================================================

/// MOVAPS xmm, xmm - Register-to-register copy
pub fn emit_movaps(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    emit_sse_rr(code, None, &[0x0F, 0x28], dst, src);
}

// =============================================================================
// Arithmetic (SSE legacy 2-operand)
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
// Arithmetic (VEX 3-operand)
// =============================================================================

// =============================================================================
// Bitwise (VEX 3-operand)
// =============================================================================

/// VANDPS dst, src1, src2 — bitwise AND
fn emit_vandps(code: &mut Vec<u8>, dst: Reg, src1: Reg, src2: Reg) {
    emit_vex_128_0f(code, 0x54, dst, src1, src2);
}

/// VANDNPS dst, src1, src2 — bitwise NOT(src1) AND src2
fn emit_vandnps(code: &mut Vec<u8>, dst: Reg, src1: Reg, src2: Reg) {
    emit_vex_128_0f(code, 0x55, dst, src1, src2);
}

/// VORPS dst, src1, src2 — bitwise OR
fn emit_vorps(code: &mut Vec<u8>, dst: Reg, src1: Reg, src2: Reg) {
    emit_vex_128_0f(code, 0x56, dst, src1, src2);
}

/// VXORPS dst, src1, src2 — bitwise XOR
fn emit_vxorps(code: &mut Vec<u8>, dst: Reg, src1: Reg, src2: Reg) {
    emit_vex_128_0f(code, 0x57, dst, src1, src2);
}

// =============================================================================
// Comparisons (VEX)
// =============================================================================

/// VCMPPS predicates
const CMP_LT: u8 = 1; // Less than (ordered, non-signaling)
const CMP_NLE: u8 = 6; // Not less-or-equal, i.e. greater than (unordered)

// =============================================================================
// Constants
// =============================================================================

/// Load a splat f32 constant into an XMM register via RIP-relative load.
///
/// Strategy: emit a JMP over 16 bytes of inline constant data, then load
/// with MOVAPS [RIP + disp]. This avoids needing GP scratch registers.
///
/// Layout in code stream:
/// ```text
///   JMP +16          ; 2 bytes (EB 10)
///   <16 bytes data>  ; 4x f32 splatted
///   MOVAPS dst, [RIP + disp32]  ; RIP-relative load
/// ```
fn emit_f32_const(code: &mut Vec<u8>, dst: Reg, val: f32) {
    let bits = val.to_bits();

    // Fast path: zero constant
    if bits == 0 {
        emit_vxorps(code, dst, dst, dst);
        return;
    }

    // JMP rel8 over 16 bytes of constant data
    code.push(0xEB);
    code.push(0x10); // jump +16

    // Emit 16 bytes: 4 copies of the f32
    for _ in 0..4 {
        code.extend_from_slice(&bits.to_le_bytes());
    }

    // MOVUPS dst, [RIP + disp32]
    // The displacement is relative to the end of this instruction.
    // MOVUPS (unaligned load) is required here: the constant is embedded inline
    // in the code stream at an arbitrary byte offset, so its address is not
    // guaranteed 16-byte aligned. MOVAPS would #GP-fault on a misaligned load.
    // Opcode 0F 10, ModRM = 0x05 | (dst.0 << 3), then disp32.
    // Total instruction length = (optional REX) + 2(opcode) + 1(ModRM) + 4(disp32) = 7 or 8 bytes.
    // RIP points to end of instruction, so disp32 = -(16 + instruction_length).

    let needs_rex = dst.0 >= 8;
    let inst_len: i32 = if needs_rex { 8 } else { 7 };
    let disp: i32 = -(16 + inst_len);

    if needs_rex {
        code.push(0x44); // REX.R
    }
    code.push(0x0F);
    code.push(0x10);
    code.push(0x05 | ((dst.0 & 7) << 3)); // ModRM: mod=00, rm=101 (RIP-relative)
    code.extend_from_slice(&disp.to_le_bytes());
}

/// Load constant into register (placeholder for the high-level emit dispatch).
///
/// Uses RIP-relative constant embedding for non-zero values, VXORPS for zero.
pub fn emit_const(code: &mut Vec<u8>, dst: Reg, val: f32, _scratch: [Reg; 4]) {
    emit_f32_const(code, dst, val);
}

// =============================================================================
// VEX integer / convert / round primitives
// =============================================================================
//
// The transcendental builtins below are faithful ports of the aarch64 (NEON)
// implementations in `aarch64.rs` — same algorithms and coefficients — emitted
// with AVX (VEX.128) encodings. AVX gives us `vroundps` plus 128-bit integer
// ops (`vcvttps2dq`, `vpslld`, `vpaddd`, ...) needed for exp/log bit twiddling.

/// General 3-byte VEX encoder for 128-bit ops.
///
/// `pp`: 0=none, 1=0x66, 2=0xF3, 3=0xF2. `mmmmm`: 1=0F, 2=0F38, 3=0F3A.
/// `reg` is the ModRM.reg operand (a register, or a `/digit` opcode extension
/// passed as `Reg(digit)`); `vvvv` is the inverted extra source (pass `Reg(0)`
/// when unused — that encodes the required `1111`); `rm` is the ModRM.rm reg.
#[allow(clippy::too_many_arguments)]
fn emit_vex(
    code: &mut Vec<u8>,
    pp: u8,
    mmmmm: u8,
    w: u8,
    reg: Reg,
    vvvv: Reg,
    rm: Reg,
    opcode: u8,
) {
    let rbit = if reg.0 >= 8 { 0x00 } else { 0x80 };
    let xbit = 0x40;
    let bbit = if rm.0 >= 8 { 0x00 } else { 0x20 };
    code.push(0xC4);
    code.push(rbit | xbit | bbit | mmmmm);
    code.push((w << 7) | ((!vvvv.0 & 0xF) << 3) | pp);
    code.push(opcode);
    code.push(0xC0 | ((reg.0 & 7) << 3) | (rm.0 & 7));
}

/// VROUNDPS dst, src, imm8 — round packed f32 (imm: 0=nearest, 1=floor, 2=ceil, 3=trunc).
fn emit_vroundps(code: &mut Vec<u8>, dst: Reg, src: Reg, imm: u8) {
    emit_vex(code, 1, 3, 0, dst, Reg(0), src, 0x08); // VEX.128.66.0F3A.WIG 08 /r ib
    code.push(imm);
}

/// VCVTTPS2DQ dst, src — convert packed f32 → i32 with truncation.
fn emit_vcvttps2dq(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    emit_vex(code, 2, 1, 0, dst, Reg(0), src, 0x5B); // VEX.128.F3.0F.WIG 5B /r
}

/// VCVTDQ2PS dst, src — convert packed i32 → f32.
fn emit_vcvtdq2ps(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    emit_vex(code, 0, 1, 0, dst, Reg(0), src, 0x5B); // VEX.128.0F.WIG 5B /r
}

// ─────────────────────────── bound-memory gather ────────────────────────────
//
// This build has no AVX2, so there is no `vgatherdps` at 128 bits. A gather is
// therefore four independent scalar loads assembled into a lane vector:
// extract each lane's integer index to a GPR, load that element, and insert it
// into the destination lane. All four instructions below are plain AVX (the
// VEX encodings of SSE4.1 ops), the same tier the rest of this backend already
// emits (`vroundps`, `vandnps`).

/// Third VEX byte with W=0 and `vvvv` unused (encoded inverted, so all ones);
/// OR in the `pp` field for the operand-size prefix.
const VEX_W0_NO_VVVV: u8 = 0xF << 3;
/// `pp = 01` — the `66` prefix.
const PP_66: u8 = 1;
/// `pp = 10` — the `F3` prefix.
const PP_F3: u8 = 2;

/// `mov dstGPR, [ctxGPR + disp32]` — load a buffer base pointer out of the
/// context struct. Mirrors the AVX-512 backend's loader.
pub fn emit_load_ptr_from_ctx(code: &mut Vec<u8>, dst_gpr: u8, ctx_gpr: u8, disp: i32) {
    debug_assert!(
        dst_gpr < 8 && ctx_gpr < 8,
        "emit_load_ptr_from_ctx: GPR8 only"
    );
    // REX.W ; 8B ; mod=10 reg=dst r/m=ctx ; disp32
    code.push(0x48);
    code.push(0x8B);
    code.push(0x80 | ((dst_gpr & 7) << 3) | (ctx_gpr & 7));
    code.extend_from_slice(&disp.to_le_bytes());
}

/// `vpextrd r32, xmmSRC, lane` — move one 32-bit lane into a GP register.
fn emit_vpextrd_to_gpr(code: &mut Vec<u8>, dst_gpr: u8, src: Reg, lane: u8) {
    debug_assert!(dst_gpr < 8, "emit_vpextrd_to_gpr: GPR8 only");
    debug_assert!(lane < 4, "vpextrd lane must be 0..4");
    // VEX.128.66.0F3A.W0 16 /r ib — note the *xmm* is the ModRM.reg operand and
    // the GPR is r/m, the reverse of the usual direction.
    let rbit = if src.0 >= 8 { 0x00 } else { 0x80 };
    code.push(0xC4);
    code.push(rbit | 0x40 | 0x20 | 3); // R X B mmmmm=0F3A (GPR8 => B set)
    code.push(VEX_W0_NO_VVVV | PP_66);
    code.push(0x16);
    code.push(0xC0 | ((src.0 & 7) << 3) | (dst_gpr & 7));
    code.push(lane);
}

/// `vmovss xmmDST, [baseGPR + indexGPR*4]` — load one f32 element, zeroing the
/// upper lanes.
fn emit_vmovss_load_scaled(code: &mut Vec<u8>, dst: Reg, base_gpr: u8, index_gpr: u8) {
    debug_assert!(base_gpr < 8 && index_gpr < 8, "GPR8 only");
    debug_assert!(base_gpr != 5, "base rbp/r13 would force a disp form");
    // VEX.LIG.F3.0F.WIG 10 /r, mod=00 rm=100 (SIB), SIB scale=4.
    let rbit = if dst.0 >= 8 { 0x00 } else { 0x80 };
    code.push(0xC4);
    code.push(rbit | 0x40 | 0x20 | 1); // mmmmm=0F
    code.push(VEX_W0_NO_VVVV | PP_F3);
    code.push(0x10);
    code.push(((dst.0 & 7) << 3) | 0b100); // mod=00, rm=SIB
    code.push((0b10 << 6) | ((index_gpr & 7) << 3) | (base_gpr & 7)); // scale=4
}

/// `vinsertps xmmDST, xmmSRC1, xmmSRC2, imm8` — place lane 0 of `src2` into
/// lane `dst_lane` of the result, keeping `src1`'s other lanes.
fn emit_vinsertps(code: &mut Vec<u8>, dst: Reg, src1: Reg, src2: Reg, dst_lane: u8) {
    debug_assert!(dst_lane < 4, "vinsertps lane must be 0..4");
    // VEX.128.66.0F3A.WIG 21 /r ib. imm8: [7:6] source lane, [5:4] dest lane,
    // [3:0] zero mask (none).
    emit_vex(code, 1, 3, 0, dst, src1, src2, 0x21);
    code.push(dst_lane << 4);
}

/// Scratch the scalar gather sequence clobbers. The vector pair must be
/// distinct from each other and from the gather's index operand; the GPRs must
/// be free across the sequence.
#[derive(Clone, Copy)]
pub struct GatherScratch {
    /// GPR receiving the buffer base pointer.
    pub base_gpr: u8,
    /// GPR receiving each lane's element index.
    pub index_gpr: u8,
    /// GPR holding the caller's context pointer (read-only).
    pub ctx_gpr: u8,
    /// Vector register for the truncated integer indices.
    pub idx_lanes: Reg,
    /// Vector register for one loaded element.
    pub value: Reg,
}

/// `dst = base[idx_lane]` for each lane — the whole gather sequence.
///
/// `idx` holds the *float* indices (the lowering already clamped them in
/// range). `dst` may alias `idx`: the indices are converted into scratch before
/// the first write to `dst`.
pub fn emit_gather_scalar(code: &mut Vec<u8>, dst: Reg, idx: Reg, slot: u16, s: GatherScratch) {
    debug_assert!(s.idx_lanes.0 != s.value.0 && s.idx_lanes.0 != idx.0);
    emit_vcvttps2dq(code, s.idx_lanes, idx);
    emit_load_ptr_from_ctx(code, s.base_gpr, s.ctx_gpr, i32::from(slot) * 8);
    for lane in 0..4u8 {
        emit_vpextrd_to_gpr(code, s.index_gpr, s.idx_lanes, lane);
        if lane == 0 {
            // Lane 0 seeds the vector (vmovss zeroes lanes 1..4).
            emit_vmovss_load_scaled(code, dst, s.base_gpr, s.index_gpr);
        } else {
            emit_vmovss_load_scaled(code, s.value, s.base_gpr, s.index_gpr);
            emit_vinsertps(code, dst, dst, s.value, lane);
        }
    }
}

/// VPADDD dst, src1, src2 — packed i32 add.
fn emit_vpaddd(code: &mut Vec<u8>, dst: Reg, src1: Reg, src2: Reg) {
    emit_vex(code, 1, 1, 0, dst, src1, src2, 0xFE); // VEX.128.66.0F.WIG FE /r
}

/// VPSLLD dst, src, imm8 — packed i32 shift-left-logical by immediate.
fn emit_vpslld_imm(code: &mut Vec<u8>, dst: Reg, src: Reg, imm: u8) {
    // VEX.128.66.0F.WIG 72 /6 ib ; dst = vvvv, src = rm, /6 in ModRM.reg.
    emit_vex(code, 1, 1, 0, Reg(6), dst, src, 0x72);
    code.push(imm);
}

/// VPSRLD dst, src, imm8 — packed i32 shift-right-logical by immediate.
fn emit_vpsrld_imm(code: &mut Vec<u8>, dst: Reg, src: Reg, imm: u8) {
    // VEX.128.66.0F.WIG 72 /2 ib.
    emit_vex(code, 1, 1, 0, Reg(2), dst, src, 0x72);
    code.push(imm);
}

/// Bit-select (NEON BSL analogue): `dst = (mask & if_true) | (~mask & if_false)`.
///
/// `tmp` must differ from `dst`, `mask`, `if_true`, and `if_false`.
#[allow(clippy::too_many_arguments)]
fn emit_blend(code: &mut Vec<u8>, dst: Reg, mask: Reg, if_true: Reg, if_false: Reg, tmp: Reg) {
    emit_vandps(code, tmp, mask, if_true); // tmp = mask & if_true
    emit_vandnps(code, dst, mask, if_false); // dst = ~mask & if_false
    emit_vorps(code, dst, tmp, dst); // dst = blended
}

// VCMPPS ordered predicates (subset).
const CMP_EQ: u8 = 0; // EQ_OQ
const CMP_LE: u8 = 2; // LE_OS
const CMP_NEQ: u8 = 4; // NEQ_UQ
const CMP_GE: u8 = 5; // NLT_US (>=)

// =============================================================================
// Transcendental Builtins — inline polynomial sequences
// =============================================================================
//
// Faithful ports of the aarch64 builtins (same algorithms / coefficients).
//
// Register contract:
//   dst  — output register
//   src  — input register (read-only; never clobbered)
//   scratch[0..4] — clobbered scratch (4 distinct registers)

/// MOVUPS [rsp+disp8], xmm — red-zone spill store (unaligned, leaf-safe).
pub fn emit_movups_store_rsp(code: &mut Vec<u8>, src: Reg, disp: i8) {
    if src.0 >= 8 {
        code.push(0x44); // REX.R
    }
    code.push(0x0F);
    code.push(0x11);
    code.push(0x44 | ((src.0 & 7) << 3)); // mod=01, reg=src, rm=100 (SIB)
    code.push(0x24); // SIB: base=rsp, no index
    code.push(disp as u8);
}

/// MOVUPS [rsp+disp32], xmm — spill store into an allocated frame.
pub fn emit_movups_store_rsp32(code: &mut Vec<u8>, src: Reg, disp: i32) {
    if src.0 >= 8 {
        code.push(0x44); // REX.R
    }
    code.push(0x0F);
    code.push(0x11);
    code.push(0x84 | ((src.0 & 7) << 3)); // mod=10, reg=src, rm=100 (SIB)
    code.push(0x24); // SIB: base=rsp, no index
    code.extend_from_slice(&disp.to_le_bytes());
}

/// MOVUPS xmm, [rsp+disp32] — reload from an allocated frame.
pub fn emit_movups_load_rsp32(code: &mut Vec<u8>, dst: Reg, disp: i32) {
    if dst.0 >= 8 {
        code.push(0x44); // REX.R
    }
    code.push(0x0F);
    code.push(0x10);
    code.push(0x84 | ((dst.0 & 7) << 3)); // mod=10, reg=dst, rm=100 (SIB)
    code.push(0x24); // SIB: base=rsp, no index
    code.extend_from_slice(&disp.to_le_bytes());
}

/// `sub rsp, imm32` — allocate a spill frame (kernels stay leaf functions;
/// no base pointer, offsets are rsp-relative).
pub fn emit_sub_rsp(code: &mut Vec<u8>, size: u32) {
    code.extend_from_slice(&[0x48, 0x81, 0xEC]);
    code.extend_from_slice(&size.to_le_bytes());
}

/// `add rsp, imm32` — release the spill frame before `ret`.
pub fn emit_add_rsp(code: &mut Vec<u8>, size: u32) {
    code.extend_from_slice(&[0x48, 0x81, 0xC4]);
    code.extend_from_slice(&size.to_le_bytes());
}

/// MOVUPS xmm, [rsp+disp8] — red-zone reload (unaligned, leaf-safe).
pub fn emit_movups_load_rsp(code: &mut Vec<u8>, dst: Reg, disp: i8) {
    if dst.0 >= 8 {
        code.push(0x44);
    }
    code.push(0x0F);
    code.push(0x10);
    code.push(0x44 | ((dst.0 & 7) << 3));
    code.push(0x24);
    code.push(disp as u8);
}

/// MOVUPS [base64], xmm — unaligned store to a GP-register base (mod=00).
///
/// `base_gpr` must not be rsp/rbp/r12/r13 (those encodings require SIB or a
/// displacement form).
pub fn emit_movups_store_base(code: &mut Vec<u8>, src: Reg, base_gpr: u8) {
    debug_assert!(
        base_gpr & 7 != 4 && base_gpr & 7 != 5,
        "emit_movups_store_base: base must not be rsp/rbp/r12/r13"
    );
    if src.0 >= 8 || base_gpr >= 8 {
        code.push(0x40 | (((src.0 >> 3) & 1) << 2) | ((base_gpr >> 3) & 1)); // REX.R/B
    }
    code.push(0x0F);
    code.push(0x11);
    code.push(((src.0 & 7) << 3) | (base_gpr & 7)); // mod=00
}

// ---------------------------------------------------------------------------
// Public unary builtin entry points
// ---------------------------------------------------------------------------

pub fn emit_floor_builtin(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    emit_vroundps(code, dst, src, 1);
}

pub fn emit_ceil_builtin(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    emit_vroundps(code, dst, src, 2);
}

pub fn emit_round_builtin(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    emit_vroundps(code, dst, src, 0); // round to nearest (even)
}

/// select(cond, if_true, if_false) — bit blend (`cond` is an all-ones/zeros mask).
///
/// `tmp` must differ from `cond`, `if_true`, and `if_false`.
#[allow(clippy::too_many_arguments)]
pub fn emit_select(code: &mut Vec<u8>, dst: Reg, cond: Reg, if_true: Reg, if_false: Reg, tmp: Reg) {
    emit_blend(code, dst, cond, if_true, if_false, tmp);
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

        // Negation: flip the sign bit (dst = src XOR 0x80000000).
        OpKind::Neg => {
            let mask = scratch[0];
            emit_f32_const(code, mask, f32::from_bits(0x8000_0000));
            emit_vxorps(code, dst, src, mask);
        }

        // Absolute value: clear the sign bit (dst = src AND 0x7FFFFFFF).
        OpKind::Abs => {
            let mask = scratch[0];
            emit_f32_const(code, mask, f32::from_bits(0x7FFF_FFFF));
            emit_vandps(code, dst, src, mask);
        }

        // Rounding (AVX vroundps)
        OpKind::Floor => emit_floor_builtin(code, dst, src),
        OpKind::Ceil => emit_ceil_builtin(code, dst, src),
        OpKind::Round => emit_round_builtin(code, dst, src),

        // Bit-manip primitives (integer-domain). Single instructions.
        OpKind::TruncToInt => emit_vcvttps2dq(code, dst, src),
        OpKind::IntToFloat => emit_vcvtdq2ps(code, dst, src),

        _ => unimplemented_op("x86-64", op),
    }
}

/// Emit a logical shift of i32 lanes by a compile-time immediate.
/// `Shl` -> `vpslld`, `Shr` -> `vpsrld` (logical). VEX form is 3-operand
/// (`dst = src << imm`), so there is no two-operand hazard.
pub fn emit_shift_imm(code: &mut Vec<u8>, op: OpKind, dst: Reg, src: Reg, amount: u8) {
    match op {
        OpKind::Shl => emit_vpslld_imm(code, dst, src, amount),
        OpKind::Shr => emit_vpsrld_imm(code, dst, src, amount),
        _ => unimplemented_op("x86-64", op),
    }
}

// =============================================================================
// Instruction selection: OpKind -> a closed set of binary-op mnemonics
// =============================================================================
//
// This used to be one flat `match OpKind { Add => emit_addps(...), ...,
// _ => panic!() }`: "which ops exist" and "how do I encode this op" were the
// same match, so a missing arm only showed up as a runtime panic on whatever
// input happened to exercise it — exactly how AVX-512's binary-op match sat
// at 6-of-15 ops for a release (see avx512.rs `emit_binary`, and
// docs/designs/2026-07-25-two-level-ir-and-backend-completeness.md).
// Splitting selection from encoding makes "does this backend support op Y"
// a question with one authoritative answer, not an emergent property of a
// match arm nobody re-checked:
//
//   1. **Selection** (`X86BinaryInsn::select`): OpKind -> Option<X86BinaryInsn>.
//      Still partial over the full `OpKind` (most of its variants are unary,
//      ternary, or eliminated by `lowering` before they ever reach a backend
//      — see `lowering.rs`), so this keeps a `_ => None`. But every op this
//      backend claims to encode is named exactly once, here, as data — a
//      completeness test enumerates against this function directly instead
//      of poking the flat dispatch and hoping to hit every case.
//   2. **Encoding** (`X86BinaryInsn::encode`): X86BinaryInsn -> bytes. This
//      match has NO wildcard: it is exhaustive over the closed mnemonic
//      enum, so adding a variant without teaching `encode` how to emit it is
//      a compile error, not a silently-missing arm.
//
// An instruction is a value, not a function call: constructing
// `X86BinaryInsn::AddPs` and encoding it are two separate steps, so
// selection (which op maps to which mnemonic) is testable independently of
// encoding (which mnemonic maps to which bytes).

/// A binary SSE mnemonic this backend knows how to encode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum X86BinaryInsn {
    AddPs,
    SubPs,
    MulPs,
    DivPs,
    MinPs,
    MaxPs,
    /// `vcmpps` ordered predicate immediate (`CMP_EQ`/`CMP_LT`/...).
    CmpPs(u8),
    /// Packed i32 lane add (`IAdd`).
    PAddD,
    AndPs,
    OrPs,
}

impl X86BinaryInsn {
    /// Select the mnemonic for `op`, or `None` if this backend has no binary
    /// encoding for it. `None` covers every non-binary `OpKind` (unary,
    /// ternary, structural) as well as ops eliminated by `lowering` before
    /// scheduling (transcendentals, `Dwrt`, `Reduce`, `Gather`) — none of
    /// those can reach `emit_binary` from a correctly-lowered arena, so
    /// `None` reaching a caller is an upstream invariant violation, not a
    /// missing feature.
    pub(crate) const fn select(op: OpKind) -> Option<Self> {
        match op {
            OpKind::Add => Some(Self::AddPs),
            OpKind::Sub => Some(Self::SubPs),
            OpKind::Mul => Some(Self::MulPs),
            OpKind::Div => Some(Self::DivPs),
            OpKind::Min => Some(Self::MinPs),
            OpKind::Max => Some(Self::MaxPs),
            OpKind::Eq => Some(Self::CmpPs(CMP_EQ)),
            OpKind::Ne => Some(Self::CmpPs(CMP_NEQ)),
            OpKind::Lt => Some(Self::CmpPs(CMP_LT)),
            OpKind::Le => Some(Self::CmpPs(CMP_LE)),
            OpKind::Gt => Some(Self::CmpPs(CMP_NLE)),
            OpKind::Ge => Some(Self::CmpPs(CMP_GE)),
            OpKind::IAdd => Some(Self::PAddD),
            OpKind::BitAnd => Some(Self::AndPs),
            OpKind::BitOr => Some(Self::OrPs),
            _ => None,
        }
    }

    /// Encode `dst = dst <mnemonic> src2` (the two-operand SSE setup —
    /// `dst` already holding `src1` — runs in the caller, `emit_binary`).
    /// Exhaustive over the closed `X86BinaryInsn` set: no wildcard arm.
    fn encode(self, code: &mut Vec<u8>, dst: Reg, src2: Reg) {
        match self {
            Self::AddPs => emit_addps(code, dst, src2),
            Self::SubPs => emit_subps(code, dst, src2),
            Self::MulPs => emit_mulps(code, dst, src2),
            Self::DivPs => emit_divps(code, dst, src2),
            Self::MinPs => emit_minps(code, dst, src2),
            Self::MaxPs => emit_maxps(code, dst, src2),
            // Comparisons -> all-ones / all-zeros mask (ordered predicates).
            Self::CmpPs(pred) => emit_cmp_tail(code, dst, src2, pred),
            // Bit-manip primitives: the VEX 3-operand encoders take `dst` as
            // the vvvv source, so this is in-place (dst already holds src1).
            Self::PAddD => emit_vpaddd(code, dst, dst, src2),
            Self::AndPs => emit_vandps(code, dst, dst, src2),
            Self::OrPs => emit_vorps(code, dst, dst, src2),
        }
    }
}

/// Emit binary operation
pub fn emit_binary(code: &mut Vec<u8>, op: OpKind, dst: Reg, src1: Reg, src2: Reg) {
    // SSE is 2-operand, so we may need to move first
    if dst.0 != src1.0 {
        emit_sse_rr(code, None, &[0x0F, 0x28], dst, src1); // MOVAPS dst, src1
    }

    match X86BinaryInsn::select(op) {
        Some(insn) => insn.encode(code, dst, src2),
        None => unimplemented_op("x86-64", op),
    }
}

/// Emit the trailing `CMPPS dst, src2, imm8` of an in-place compare (dst already
/// holds src1). Produces an all-ones / all-zeros mask.
fn emit_cmp_tail(code: &mut Vec<u8>, dst: Reg, src2: Reg, pred: u8) {
    let rex = 0x40 | (if dst.0 >= 8 { 0x04 } else { 0 }) | (if src2.0 >= 8 { 0x01 } else { 0 });
    if rex != 0x40 {
        code.push(rex);
    }
    code.push(0x0F);
    code.push(0xC2);
    code.push(0xC0 | ((dst.0 & 7) << 3) | (src2.0 & 7));
    code.push(pred);
}

/// Emit ternary operation
#[allow(clippy::too_many_arguments)]
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

        _ => unimplemented_op("x86-64", op),
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

// =============================================================================
// Branches — for the shared driver's Select short-circuit guards.
// =============================================================================

/// MOVMSKPS eax, xmm — gather the 4 lane sign bits into eax (0b0000..0b1111).
/// For a select mask (lanes all-ones or all-zeros), eax == 0 means all-false
/// and eax == 0xF means all-true.
pub fn emit_movmskps_eax(code: &mut Vec<u8>, src: Reg) {
    if src.0 >= 8 {
        code.push(0x41); // REX.B
    }
    code.push(0x0F);
    code.push(0x50);
    code.push(0xC0 | (src.0 & 7)); // mod=11, reg=eax(0), rm=src
}

/// Emit `jcc rel32` with a zero placeholder; returns the offset of the rel32
/// field (pass to [`patch_rel32`]). `cc` is the 0x8_ condition byte (0x84 = je/jz,
/// 0x85 = jne/jnz).
pub fn emit_jcc_rel32(code: &mut Vec<u8>, cc: u8) -> usize {
    code.push(0x0F);
    code.push(cc);
    let pos = code.len();
    code.extend_from_slice(&[0, 0, 0, 0]);
    pos
}

/// Emit `jmp rel32` with a zero placeholder; returns the rel32 field offset.
pub fn emit_jmp_rel32(code: &mut Vec<u8>) -> usize {
    code.push(0xE9);
    let pos = code.len();
    code.extend_from_slice(&[0, 0, 0, 0]);
    pos
}

/// Patch a rel32 branch displacement (emitted by [`emit_jcc_rel32`] /
/// [`emit_jmp_rel32`]) so it lands at `target`.
pub fn patch_rel32(code: &mut [u8], pos: usize, target: usize) {
    let rel = (target as i64) - (pos as i64 + 4);
    code[pos..pos + 4].copy_from_slice(&(rel as i32).to_le_bytes());
}

/// TEST eax, eax (sets ZF iff eax == 0).
pub fn emit_test_eax(code: &mut Vec<u8>) {
    code.extend_from_slice(&[0x85, 0xC0]);
}

/// CMP eax, imm8 (sign-extended).
pub fn emit_cmp_eax_imm8(code: &mut Vec<u8>, imm: u8) {
    code.extend_from_slice(&[0x83, 0xF8, imm]);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `X86BinaryInsn::select` is `emit_binary`'s completeness contract made
    /// checkable: every op `crate::emit::coverage::
    /// REQUIRED_BINARY_OPS` lists must select `Some`, or this backend would
    /// panic the moment the scheduler handed it that op. Cheaper and more
    /// precise than the emit-and-catch-panic sweep in `emit/mod.rs`'s
    /// `backend_op_coverage` tests, because it checks the *selection* step
    /// directly instead of inferring "not supported" from a caught panic.
    #[test]
    fn selects_every_required_binary_op() {
        use crate::emit::coverage::REQUIRED_BINARY_OPS;

        let unselected: alloc::vec::Vec<OpKind> = REQUIRED_BINARY_OPS
            .iter()
            .copied()
            .filter(|&op| X86BinaryInsn::select(op).is_none())
            .collect();

        assert!(
            unselected.is_empty(),
            "X86BinaryInsn::select has no mnemonic for: {unselected:?}"
        );
    }
}

// =============================================================================
// The SSE2 `IsaBackend` driver
// =============================================================================

/// The SSE2 half of code generation.
///
/// **This file is where SSE2-specific bugs live, and the only place they
/// can.** Emission is a pure function into `Vec<u8>`, so everything here
/// compiles, typechecks and is swept for op coverage on every host, whatever
/// CPU it has. Only [`Native`](super::super::Native) decides which backend a
/// build instantiates, and only [`executable`](super::super::executable) needs
/// the matching hardware.
///
/// The consequence worth stating: a change that does not touch an ISA file
/// cannot introduce a platform-specific bug. That is the bargain `unsafe`
/// makes — confine what cannot be checked, so the rest is checked by
/// construction.
///
/// Dead in a build that selected a different `Native`; that is the intended
/// shape, not an oversight, which is why the allow sits here once rather than
/// on each item.
#[allow(dead_code)]
pub(crate) mod driver {
    use super::super::*;
    use alloc::vec::Vec;
    use pixelflow_ir::kind::OpKind;

    /// The SSE2 register file (xmm, 128-bit).
    ///
    /// SysV has no callee-saved XMM registers, so every register past the inputs
    /// is fair game; the pool stops at xmm9 to leave xmm10 as the two-operand
    /// hazard/select temp, xmm11-12 for reloads, and xmm13-15 for builtins.
    pub(crate) const SSE2_FILE: regalloc::RegisterFile = regalloc::RegisterFile {
        inputs: INPUT_REGS,
        scratch_base: 4,
        scratch_count: 6,
        reload: [Reg(11), Reg(12)],
        // xmm13: builtin scratch, unused by `emit_select` (whose internal temp is
        // X86_SCRATCH/xmm10) and by the reload path (movups / RIP-relative consts
        // touch no scratch).
        select_reload: Reg(13),
        vector_bytes: 16,
    }
    .checked();

    /// Fixed scratch outside the allocatable range / reload regs: used for the
    /// binary two-operand hazard and as the select blend temp.
    ///
    /// SSE2-only (like the rest of this section down to `X86Backend`): dead under
    /// a build that selects `Avx2Backend`/`Avx512Backend` instead, so gated off
    /// those widths — otherwise it (and `X86Backend`) would be unreachable dead
    /// code under `+avx2`/`+avx512f`, which the test-matrix's per-ISA clippy pass
    /// would then flag as a *new* failure on those levels only.
    const X86_SCRATCH: Reg = Reg(10);

    /// Scratch quad for builtins (sin/cos/exp/atan2/...), which need FOUR distinct
    /// scratch registers. Clear of the allocatable range (4-9) and the reload regs
    /// (11,12). Includes `X86_SCRATCH` (xmm10): builtins don't use it for the
    /// hazard/select roles, so it is free as a fourth scratch here.
    const X86_BUILTIN_SCRATCH: [Reg; 4] = [Reg(10), Reg(13), Reg(14), Reg(15)];

    /// Map a `FrameLayout` spill offset to a red-zone `[rsp+disp8]` displacement.
    fn x86_redzone_disp(offset: u32) -> Result<i8, &'static str> {
        // Slots live below rsp: offset 0 -> [rsp-16], 16 -> [rsp-32], ...
        // Only called in red-zone mode (the prologue switches to an allocated
        // frame when the layout exceeds the zone), so overflow here is an
        // internal invariant violation, not a kernel-size limit.
        let disp = -(offset as i64 + 16);
        if disp < -128 {
            return Err("x86 spill: red-zone displacement out of range (prologue mode bug)");
        }
        Ok(disp as i8)
    }

    /// `dst = left op right` honoring SSE's two-operand form for *any* register
    /// assignment from the allocator.
    ///
    /// `emit_binary` computes `dst <- left; dst op= right`, which corrupts `right`
    /// when `dst == right` and `dst != left`. The allocator may assign `dst ==
    /// right`, so handle it: swap for commutative ops, otherwise stash `right` in
    /// the fixed scratch.
    fn emit_binary_safe(code: &mut Vec<u8>, op: OpKind, dst: Reg, left: Reg, right: Reg) {
        use super::*;
        let commutative = matches!(
            op,
            OpKind::Add | OpKind::Mul | OpKind::Min | OpKind::Max | OpKind::Eq | OpKind::Ne
        );
        if dst == left || dst != right {
            emit_binary(code, op, dst, left, right);
        } else if commutative {
            emit_binary(code, op, dst, right, left);
        } else {
            emit_movaps(code, X86_SCRATCH, right);
            emit_movaps(code, dst, left);
            emit_binary(code, op, dst, dst, X86_SCRATCH);
        }
    }

    /// x86-64 implementation of the shared driver's leaf operations.
    ///
    /// `frame_bytes` is set by `prologue`: 0 = spills fit the 128-byte red zone
    /// (no frame is allocated), otherwise the size of the allocated frame and
    /// spill slots are `[rsp + offset]`.
    pub(crate) struct X86Backend {
        frame_bytes: u32,
        file: regalloc::RegisterFile,
    }

    impl X86Backend {
        pub(crate) fn new(ctx: EmitCtx) -> Self {
            Self {
                frame_bytes: 0,
                file: SSE2_FILE.capped(ctx.max_regs),
            }
        }

        fn spill_store(&self, code: &mut Vec<u8>, src: Reg, offset: u32) {
            if self.frame_bytes == 0 {
                let disp = x86_redzone_disp(offset).expect("red-zone mode implies fitting offsets");
                super::emit_movups_store_rsp(code, src, disp);
            } else {
                super::emit_movups_store_rsp32(code, src, offset as i32);
            }
        }

        fn spill_load(&self, code: &mut Vec<u8>, dst: Reg, offset: u32) {
            if self.frame_bytes == 0 {
                let disp = x86_redzone_disp(offset).expect("red-zone mode implies fitting offsets");
                super::emit_movups_load_rsp(code, dst, disp);
            } else {
                super::emit_movups_load_rsp32(code, dst, offset as i32);
            }
        }
    }

    impl IsaBackend for X86Backend {
        /// rel32 field offset of the branch (uniform for jcc/jmp on x86).
        type Branch = usize;

        fn register_file(&self) -> regalloc::RegisterFile {
            self.file
        }

        fn begin(&mut self, _schedule: &[regalloc::Def]) -> Result<(), &'static str> {
            Ok(()) // x86 const loads are self-contained; no pool.
        }

        fn frame_ready(&mut self, frame_size: u32) {
            // Red zone when it fits (max slot offset frame_size-16 maps to disp
            // -(frame_size) >= -128); otherwise an allocated frame, with slots at
            // [rsp + offset]. Latched here so the body's spill addressing agrees
            // with the prologue emitted afterwards.
            self.frame_bytes = if frame_size <= 128 { 0 } else { frame_size };
        }

        fn prologue(&mut self, code: &mut Vec<u8>, _frame_size: u32) {
            // movups is alignment-agnostic, so the frame only needs its
            // 16-multiple size; red-zone mode needs no setup for a leaf.
            if self.frame_bytes > 0 {
                super::emit_sub_rsp(code, self.frame_bytes);
            }
        }

        fn emit_plan(
            &mut self,
            code: &mut Vec<u8>,
            plan: &InstructionPlan,
        ) -> Result<(), &'static str> {
            use super::*;
            for reload in &plan.reloads {
                match reload {
                    Reload::FromStack { target, offset } => {
                        self.spill_load(code, *target, *offset);
                    }
                    Reload::Const { target, val_bits } => {
                        emit_const(
                            code,
                            *target,
                            f32::from_bits(*val_bits),
                            X86_BUILTIN_SCRATCH,
                        );
                    }
                }
            }
            if let Some((dst, src)) = plan.setup_mov {
                emit_movaps(code, dst, src);
            }
            match &plan.op {
                ResolvedOp::Nop => {}
                ResolvedOp::LoadConst { dst, val_bits } => {
                    emit_const(code, *dst, f32::from_bits(*val_bits), X86_BUILTIN_SCRATCH);
                }
                ResolvedOp::Unary { op, dst, src } => {
                    emit_unary(code, *op, *dst, *src, X86_BUILTIN_SCRATCH);
                }
                ResolvedOp::ShiftImm {
                    op,
                    dst,
                    src,
                    amount,
                } => {
                    emit_shift_imm(code, *op, *dst, *src, *amount);
                }
                ResolvedOp::Gather { dst, idx, slot } => {
                    // No AVX2 here, so no 128-bit vgatherdps: assemble the lanes
                    // from four scalar loads. The context pointer (array of buffer
                    // base pointers) is caller-provided in rdi and never touched by
                    // arithmetic/const emission, so it survives to here; rax/rcx
                    // are caller-saved and unused by the rest of the body.
                    // Both vector scratch registers are outside the allocatable
                    // range and the reload pair (see X86_BUILTIN_SCRATCH).
                    super::emit_gather_scalar(
                        code,
                        *dst,
                        *idx,
                        *slot,
                        super::GatherScratch {
                            base_gpr: 0,  // rax
                            index_gpr: 1, // rcx
                            ctx_gpr: 7,   // rdi
                            idx_lanes: Reg(13),
                            value: Reg(14),
                        },
                    );
                }
                ResolvedOp::Binary {
                    op,
                    dst,
                    left,
                    right,
                } => emit_binary_safe(code, *op, *dst, *left, *right),
                ResolvedOp::Select {
                    dst,
                    if_true,
                    if_false,
                } => {
                    // setup_mov already placed the mask in `dst`; blend in place.
                    emit_select(code, *dst, *dst, *if_true, *if_false, X86_SCRATCH);
                }
                ResolvedOp::FusedMulAdd { dst, a, b } => {
                    // No hardware FMA assumed: `dst` already holds c (setup_mov);
                    // compute a*b in the fixed scratch, then add. a,b are never
                    // X86_SCRATCH (allocator/reload regs), and `a` is copied out
                    // before any write, so c==a / c==b are handled.
                    emit_movaps(code, X86_SCRATCH, *a);
                    emit_binary(code, OpKind::Mul, X86_SCRATCH, X86_SCRATCH, *b);
                    emit_binary(code, OpKind::Add, *dst, *dst, X86_SCRATCH);
                }
                ResolvedOp::DecomposedMulAdd {
                    dst,
                    a,
                    b,
                    c,
                    c_deferred,
                } => {
                    // dst = a*b, reload c (after the multiply, if deferred), dst += c.
                    emit_binary_safe(code, OpKind::Mul, *dst, *a, *b);
                    match c_deferred {
                        Some(DeferredReload::FromStack(off)) => {
                            self.spill_load(code, *c, *off);
                        }
                        Some(DeferredReload::Const(bits)) => {
                            emit_const(code, *c, f32::from_bits(*bits), X86_BUILTIN_SCRATCH);
                        }
                        None => {}
                    }
                    emit_binary_safe(code, OpKind::Add, *dst, *dst, *c);
                }
            }
            if let Some(store) = &plan.store {
                self.spill_store(code, store.src, store.offset);
            }
            Ok(())
        }

        fn emit_mov(&mut self, code: &mut Vec<u8>, dst: Reg, src: Reg) {
            super::emit_movaps(code, dst, src);
        }

        fn emit_store(
            &mut self,
            code: &mut Vec<u8>,
            src: Reg,
            offset: u32,
        ) -> Result<(), &'static str> {
            self.spill_store(code, src, offset);
            Ok(())
        }

        fn emit_resolve(
            &mut self,
            code: &mut Vec<u8>,
            vid: regalloc::ValueId,
            target: Reg,
            locs: &[Option<Loc>],
        ) -> Reg {
            match location_of(locs, vid) {
                Loc::Reg(reg) => reg,
                Loc::Remat(bits) => {
                    super::emit_const(code, target, f32::from_bits(bits), X86_BUILTIN_SCRATCH);
                    target
                }
                Loc::Spill(offset) => {
                    self.spill_load(code, target, offset);
                    target
                }
            }
        }

        fn emit_skip_if_all_false(&mut self, code: &mut Vec<u8>, mask_reg: Reg) -> usize {
            super::emit_movmskps_eax(code, mask_reg);
            super::emit_test_eax(code);
            super::emit_jcc_rel32(code, 0x84) // jz: taken when eax == 0 (all lanes false)
        }

        fn emit_skip_if_all_true(&mut self, code: &mut Vec<u8>, mask_reg: Reg) -> usize {
            super::emit_movmskps_eax(code, mask_reg);
            super::emit_cmp_eax_imm8(code, 0x0F);
            super::emit_jcc_rel32(code, 0x84) // je: taken when eax == 0xF (all lanes true)
        }

        fn emit_jump(&mut self, code: &mut Vec<u8>) -> usize {
            super::emit_jmp_rel32(code)
        }

        fn patch_branch(&mut self, code: &mut Vec<u8>, branch: usize, target: usize) {
            super::patch_rel32(code, branch, target);
        }

        fn epilogue(&mut self, code: &mut Vec<u8>, result_reg: Reg, _frame_size: u32) {
            if result_reg.0 != 0 {
                super::emit_movaps(code, Reg(0), result_reg);
            }
            if self.frame_bytes > 0 {
                super::emit_add_rsp(code, self.frame_bytes);
            }
            code.push(0xC3); // RET
        }

        // SysV: rdi = ctx (read-only in the body's gathers), rsi = out,
        // rdx = groups, rcx = rows, r8 = row-skip bytes, xmm0..3 = x0/y0/z/w.
        // Loop registers: r9 = inner counter, r10 = preserved row count, r11 =
        // outer counter; the body's scratch GPRs are rax/rcx (gather, movmskps) — disjoint. In
        // red-zone mode (`frame_bytes == 0`) the body spills below rsp, so the
        // coordinate slots allocated here at [rsp..) never collide with it.
        fn emit_collapse_loop(
            &mut self,
            frame_hoist: &[u8],
            row_hoist: &[u8],
            body: &[u8],
            result_reg: Reg,
            _frame_size: u32,
            hoist_slots: u32,
        ) -> Result<Vec<u8>, &'static str> {
            let vw = self.file.vector_bytes;
            let f = self.frame_bytes;
            let total = f + (5 + hoist_slots) * vw;
            let slot = |k: u32| (f + k * vw) as i32;
            let mut code: Vec<u8> =
                Vec::with_capacity(frame_hoist.len() + row_hoist.len() + body.len() + 160);

            super::emit_sub_rsp(&mut code, total);
            for k in 0..4u32 {
                super::emit_movups_store_rsp32(&mut code, Reg(k as u8), slot(k));
            }
            super::emit_movups_store_rsp32(&mut code, Reg(0), slot(4));
            code.extend_from_slice(frame_hoist);
            // Preserve rows away from rcx, which gather/select guards may clobber.
            code.extend_from_slice(&[0x49, 0x89, 0xCA]); // mov r10, rcx
            code.extend_from_slice(&[0x4D, 0x31, 0xDB]); // xor r11, r11

            let outer_top = code.len();
            code.extend_from_slice(&[0x4D, 0x39, 0xD3]); // cmp r11, r10
            code.extend_from_slice(&[0x0F, 0x83]);
            let outer_jae_at = code.len();
            code.extend_from_slice(&[0, 0, 0, 0]);

            // LICM prologue: X-invariant values, recomputed once per row. Reload
            // coordinates first because the previous body/Y-step clobbered xmm0..3.
            for k in 0..4u32 {
                super::emit_movups_load_rsp32(&mut code, Reg(k as u8), slot(k));
            }
            code.extend_from_slice(row_hoist);
            code.extend_from_slice(&[0x4D, 0x31, 0xC9]); // xor r9, r9

            let inner_top = code.len();
            code.extend_from_slice(&[0x49, 0x39, 0xD1]); // cmp r9, rdx
            code.extend_from_slice(&[0x0F, 0x83]);
            let inner_jae_at = code.len();
            code.extend_from_slice(&[0, 0, 0, 0]);

            for k in 0..4u32 {
                super::emit_movups_load_rsp32(&mut code, Reg(k as u8), slot(k));
            }
            code.extend_from_slice(body);

            // out[i] = result ; add rsi, 16.
            super::emit_movups_store_base(&mut code, result_reg, 6);
            code.extend_from_slice(&[0x48, 0x83, 0xC6, vw as u8]);

            // X += lane count (xmm0/1 are about to be reloaded next iteration).
            super::emit_movups_load_rsp32(&mut code, Reg(0), slot(0));
            super::emit_const(&mut code, Reg(1), (vw / 4) as f32, X86_BUILTIN_SCRATCH);
            super::emit_binary(&mut code, OpKind::Add, Reg(0), Reg(0), Reg(1));
            super::emit_movups_store_rsp32(&mut code, Reg(0), slot(0));

            // inc r9 ; jmp inner_top (backward).
            code.extend_from_slice(&[0x49, 0xFF, 0xC1]);
            code.push(0xE9);
            let inner_jmp_at = code.len();
            code.extend_from_slice(&[0, 0, 0, 0]);

            let row_end = code.len();
            let inner_jae_rel = (row_end as i32) - (inner_jae_at as i32 + 4);
            code[inner_jae_at..inner_jae_at + 4].copy_from_slice(&inner_jae_rel.to_le_bytes());
            let inner_jmp_rel = (inner_top as i32) - (inner_jmp_at as i32 + 4);
            code[inner_jmp_at..inner_jmp_at + 4].copy_from_slice(&inner_jmp_rel.to_le_bytes());

            // Reset X, advance Y, and skip any scalar tail in the output row.
            super::emit_movups_load_rsp32(&mut code, Reg(0), slot(4));
            super::emit_movups_store_rsp32(&mut code, Reg(0), slot(0));
            super::emit_movups_load_rsp32(&mut code, Reg(0), slot(1));
            super::emit_const(&mut code, Reg(1), 1.0, X86_BUILTIN_SCRATCH);
            super::emit_binary(&mut code, OpKind::Add, Reg(0), Reg(0), Reg(1));
            super::emit_movups_store_rsp32(&mut code, Reg(0), slot(1));
            code.extend_from_slice(&[0x4C, 0x01, 0xC6]); // add rsi, r8

            code.extend_from_slice(&[0x49, 0xFF, 0xC3]); // inc r11
            code.push(0xE9);
            let outer_jmp_at = code.len();
            code.extend_from_slice(&[0, 0, 0, 0]);

            let end = code.len();
            super::emit_add_rsp(&mut code, total);
            code.push(0xC3); // RET

            let outer_jae_rel = (end as i32) - (outer_jae_at as i32 + 4);
            code[outer_jae_at..outer_jae_at + 4].copy_from_slice(&outer_jae_rel.to_le_bytes());
            let outer_jmp_rel = (outer_top as i32) - (outer_jmp_at as i32 + 4);
            code[outer_jmp_at..outer_jmp_at + 4].copy_from_slice(&outer_jmp_rel.to_le_bytes());
            Ok(code)
        }
    }
}
