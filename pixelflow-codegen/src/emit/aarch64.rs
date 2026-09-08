//! ARM64/NEON instruction encoding.
//!
//! Each function emits raw machine code bytes for one instruction (or a small fixed sequence).
//! These are the "atoms" that compound operations are built from.

use super::{Reg, unimplemented_op};
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use pixelflow_ir::kind::OpKind;

// =============================================================================
// Instruction Encoding Helpers
// =============================================================================

/// Encode a NEON 3-same instruction (binary vector ops).
/// Format: Vd.4S, Vn.4S, Vm.4S
#[inline]
fn encode_3same(opcode: u32, dst: Reg, src1: Reg, src2: Reg) -> u32 {
    opcode | (dst.0 as u32 & 0x1F) | ((src1.0 as u32 & 0x1F) << 5) | ((src2.0 as u32 & 0x1F) << 16)
}

/// Encode a NEON 2-reg misc instruction (unary vector ops).
/// Format: Vd.4S, Vn.4S
#[inline]
fn encode_2misc(opcode: u32, dst: Reg, src: Reg) -> u32 {
    opcode | (dst.0 as u32 & 0x1F) | ((src.0 as u32 & 0x1F) << 5)
}

/// Write a 32-bit instruction to the code buffer.
#[inline]
pub fn emit32(code: &mut Vec<u8>, inst: u32) {
    code.extend_from_slice(&inst.to_le_bytes());
}

// =============================================================================
// Load / Store
// =============================================================================

/// An address spelled `[base, #offset]` — the scaled-immediate addressing mode.
///
/// `offset` is in BYTES. aarch64 encodes it divided by the *access size*, so
/// that divisor belongs to the instruction ([`emit_ldr_q`] moves 16 bytes,
/// [`emit_ldr_x`] moves 8) rather than to the address: one `Mem` names a
/// different immediate field in each, and an offset that is not a multiple of
/// the access size is not encodable at all.
///
/// The base being an [`Xr`] is the point. `sp`, `x17` and `x0` are values here;
/// they used to be the `_sp`, `_x17` and `_voff` suffixes of three separate
/// functions that encoded the same instruction.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Mem {
    /// The register the displacement is measured from.
    pub base: Xr,
    /// Displacement in bytes; must be a multiple of the access size.
    pub offset: u32,
}

/// An address spelled `[base, w<index>, uxtw #n]` — a 32-bit index register,
/// zero-extended to 64 bits and scaled by the access size.
///
/// A different addressing *mode* from [`Mem`], not a different kind of offset,
/// so it is a different type. Only the instruction that encodes it accepts one,
/// which is what spares every other encoder from rejecting at runtime an
/// address it cannot spell.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct MemIndexed {
    /// The register holding the buffer base.
    pub base: Xr,
    /// The element index, read as the 32-bit `w<index>`.
    pub index: Xr,
}

/// Bytes moved by a `q` (128-bit vector) access — also the scale of its offset.
const Q_BYTES: u32 = 16;
/// Bytes moved by an `x` (64-bit general) access.
const X_BYTES: u32 = 8;
/// The largest value a 12-bit scaled immediate holds.
const MAX_IMM12: u32 = 4095;
/// The largest 16-byte-aligned displacement `add`'s own 12-bit immediate holds.
const MAX_ADD_IMM: u32 = 4080;

/// Rewrite `addr` as `[x16]`, computing `base + offset` into IP0 first.
///
/// The fallback for a displacement past the 12-bit scaled immediate — a spill
/// frame deeper than 64 KiB. `add`'s immediate is 12 bits too, so a large
/// displacement takes several of them.
fn address_in_ip0(code: &mut Vec<u8>, Mem { base, offset }: Mem) -> Mem {
    let mut remaining = offset;
    let first = remaining.min(MAX_ADD_IMM);
    add(code, xr::X16, base, Imm12(first as u16));
    remaining -= first;
    while remaining > 0 {
        let chunk = remaining.min(MAX_ADD_IMM);
        add(code, xr::X16, xr::X16, Imm12(chunk as u16));
        remaining -= chunk;
    }
    Mem {
        base: xr::X16,
        offset: 0,
    }
}

/// The 128-bit vector transfers, which are one encoding —
/// `opcode | (imm12 << 10) | (Rn << 5) | Rt` — differing only in `opcode`.
fn ldst_q(code: &mut Vec<u8>, opcode: u32, vec: Reg, addr: Mem) {
    assert!(
        addr.offset.is_multiple_of(Q_BYTES),
        "128-bit access offset {} is not 16-byte aligned",
        addr.offset
    );
    let addr = if addr.offset / Q_BYTES > MAX_IMM12 {
        address_in_ip0(code, addr)
    } else {
        addr
    };
    emit32(
        code,
        opcode | ((addr.offset / Q_BYTES) << 10) | ((addr.base.0 as u32) << 5) | (vec.0 as u32),
    );
}

/// `ldr q<dst>, [addr]` — load a 128-bit vector.
pub fn emit_ldr_q(code: &mut Vec<u8>, dst: Reg, addr: Mem) {
    ldst_q(code, 0x3DC0_0000, dst, addr);
}

/// `str q<src>, [addr]` — store a 128-bit vector.
pub fn emit_str_q(code: &mut Vec<u8>, src: Reg, addr: Mem) {
    ldst_q(code, 0x3D80_0000, src, addr);
}

/// `ldr x<dst>, [addr]` — load a 64-bit pointer into a *general* register,
/// e.g. a buffer base out of the context array.
///
/// A different instruction on a different register file from [`emit_ldr_q`],
/// and the signature says so: the destination is an [`Xr`], not a [`Reg`], and
/// the offset scales by 8 rather than 16. No IP0 fallback — every caller
/// indexes the small context array, so an offset past the immediate is a bug
/// and not a deep frame.
pub fn emit_ldr_x(code: &mut Vec<u8>, dst: Xr, addr: Mem) {
    assert!(
        addr.offset.is_multiple_of(X_BYTES),
        "pointer load offset {} not 8-byte aligned",
        addr.offset
    );
    let imm12 = addr.offset / X_BYTES;
    assert!(
        imm12 <= MAX_IMM12,
        "pointer load offset {} exceeds LDR imm12 range",
        addr.offset
    );
    emit32(
        code,
        0xF940_0000 | (imm12 << 10) | ((addr.base.0 as u32) << 5) | (dst.0 as u32),
    );
}

/// Bytes moved by an `s` (32-bit scalar SIMD&FP) access.
const S_BYTES: u32 = 4;

/// `ldr s<dst>, [addr]` — load one `f32` into lane 0 of a vector register,
/// zeroing the other lanes. The offset scales by 4; past the 12-bit
/// immediate it goes through IP0 like a deep spill slot.
pub fn emit_ldr_s(code: &mut Vec<u8>, dst: Reg, addr: Mem) {
    assert!(
        addr.offset.is_multiple_of(S_BYTES),
        "32-bit access offset {} is not 4-byte aligned",
        addr.offset
    );
    let addr = if addr.offset / S_BYTES > MAX_IMM12 {
        address_in_ip0(code, addr)
    } else {
        addr
    };
    emit32(
        code,
        0xBD40_0000
            | ((addr.offset / S_BYTES) << 10)
            | ((addr.base.0 as u32) << 5)
            | (dst.0 as u32),
    );
}

/// `dup v<dst>.4s, v<src>.s[0]` — broadcast lane 0 to every lane.
pub fn emit_dup_lane0(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    emit32(code, 0x4E04_0400 | ((src.0 as u32) << 5) | (dst.0 as u32));
}

/// `dst = splat(block[offset])`: `ldr x9, [x0, #ctx_slot*8]` fetches the
/// block's base out of the context, `ldr s<dst>, [x9, #offset*4]` the value,
/// and `dup` spreads it. `x9` is the base scratch the gather already claims.
pub fn emit_uniform_load(code: &mut Vec<u8>, dst: Reg, load: super::UniformLoad) {
    const BASE_GPR: Xr = Xr(9);
    emit_ldr_x(
        code,
        BASE_GPR,
        Mem {
            base: xr::X0,
            offset: u32::from(load.ctx_slot) * X_BYTES,
        },
    );
    emit_ldr_s(
        code,
        dst,
        Mem {
            base: BASE_GPR,
            offset: u32::from(load.offset) * S_BYTES,
        },
    );
    emit_dup_lane0(code, dst, dst);
}

/// `ldr w<dst>, [base, w<index>, uxtw #2]` — load one 32-bit element at
/// `base + index * 4` into the 32-bit view of a general register.
///
/// Takes a [`MemIndexed`] because that is the mode it encodes; `#2` is the
/// shift a 4-byte access implies, which is why the scale is not an operand.
pub fn emit_ldr_w(code: &mut Vec<u8>, dst: Xr, addr: MemIndexed) {
    emit32(
        code,
        0xB860_5800 | ((addr.index.0 as u32) << 16) | ((addr.base.0 as u32) << 5) | (dst.0 as u32),
    );
}

// =============================================================================
// Stack frame
// =============================================================================

/// SUB SP, SP, #imm - Allocate stack frame.
///
/// ARM64 ADD/SUB immediate has a 12-bit field (max 4095). For larger frames,
/// we emit multiple instructions, each subtracting up to 4080 (largest
/// 16-byte-aligned value in 12 bits).
pub fn emit_sub_sp(code: &mut Vec<u8>, size: u32) {
    let mut remaining = size;
    while remaining > 0 {
        let chunk = remaining.min(4080);
        assert!(chunk <= 4095, "ARM64 immediate overflow in emit_sub_sp");
        let inst = 0xD10003FF | (chunk << 10);
        emit32(code, inst);
        remaining -= chunk;
    }
}

/// ADD SP, SP, #imm - Deallocate stack frame.
///
/// See `emit_sub_sp` for why we emit multiple instructions.
pub fn emit_add_sp(code: &mut Vec<u8>, size: u32) {
    let mut remaining = size;
    while remaining > 0 {
        let chunk = remaining.min(4080);
        assert!(chunk <= 4095, "ARM64 immediate overflow in emit_add_sp");
        let inst = 0x910003FF | (chunk << 10);
        emit32(code, inst);
        remaining -= chunk;
    }
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

/// NOT Vd.16B, Vn.16B (bitwise NOT, 2-register miscellaneous)
pub fn emit_not(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    emit32(code, encode_2misc(0x2E205800, dst, src));
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
/// Strategy (in priority order):
/// 1. Zero: MOVI Vd.4S, #0 (1 instruction)
/// 2. FMOV-encodable: FMOV Vd.4S, #imm8 (1 instruction)
/// 3. General: MOVZ W16 + MOVK W16 + DUP Vd.4S, W16 (3 instructions)
///
/// TODO: Use a constant pool with LDR for better performance on general case.
pub fn emit_fmov_imm(code: &mut Vec<u8>, dst: Reg, val: f32) {
    let bits = val.to_bits();

    if bits == 0 {
        // MOVI Vd.4S, #0 - single instruction for zero
        emit32(code, 0x4F000400 | (dst.0 as u32));
        return;
    }

    // Try FMOV Vd.4S, #imm8 for common float constants (1 instruction)
    if let Some(imm8) = try_encode_fmov_imm8(val) {
        let abc = ((imm8 as u32) >> 5) & 0x7;
        let defgh = (imm8 as u32) & 0x1F;
        // FMOV Vd.4S, #imm8: 0x4F00F400 | abc<<16 | defgh<<5 | Rd
        emit32(
            code,
            0x4F00_F400 | (abc << 16) | (defgh << 5) | (dst.0 as u32),
        );
        return;
    }

    // General case: load via GP register (W16)
    // This is 3 instructions but works for any f32 value.
    // Use W16 (IP0) as scratch - it's caller-saved and not used for arguments
    let lo16 = bits & 0xFFFF;
    let hi16 = bits >> 16;

    // MOVZ W16, #lo16
    emit32(code, 0x52800010 | (lo16 << 5));

    // MOVK W16, #hi16, LSL #16
    emit32(code, 0x72A00010 | (hi16 << 5));

    // DUP Vd.4S, W16
    emit32(code, 0x4E040C00 | (dst.0 as u32) | (16 << 5));
}

/// Try to encode an f32 as an ARM64 FMOV (vector, immediate) 8-bit value.
///
/// An f32 is FMOV-encodable when its bit pattern matches:
///   `[a] [NOT(b)] [bbbbb] [cdefgh] [19 zeros]`
/// producing imm8 = `abcdefgh`.
///
/// This covers values of the form `(-1)^a * 2^n * (1.0 + frac/64)`
/// where n is in [-3, +4] and frac is in [0, 63].
/// Common examples: 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 0.25, 1.5, etc.
///
/// Returns `None` for non-encodable values (including ±0.0, denormals, NaN, Inf).
#[must_use]
pub fn try_encode_fmov_imm8(val: f32) -> Option<u8> {
    let bits = val.to_bits();

    // Low 19 bits must be zero
    if bits & 0x7_FFFF != 0 {
        return None;
    }

    // ±0.0 is not FMOV-encodable (would require b=0 giving exp=0 which is denormal)
    if bits & 0x7FFF_FFFF == 0 {
        return None;
    }

    // bits[29:25] must all equal b, where NOT(b) = bit[30]
    let not_b = (bits >> 30) & 1;
    let b = not_b ^ 1;
    let rep5 = if b == 1 { 0x1F } else { 0x00 };
    let actual = (bits >> 25) & 0x1F;
    if actual != rep5 {
        return None;
    }

    // Extract imm8 = a:b:c:d:e:f:g:h
    let a = (bits >> 31) & 1;
    let c = (bits >> 24) & 1;
    let d = (bits >> 23) & 1;
    let e = (bits >> 22) & 1;
    let f = (bits >> 21) & 1;
    let g = (bits >> 20) & 1;
    let h = (bits >> 19) & 1;
    let imm8 = (a << 7) | (b << 6) | (c << 5) | (d << 4) | (e << 3) | (f << 2) | (g << 1) | h;
    Some(imm8 as u8)
}

// =============================================================================
// Constant Pool Support
// =============================================================================

/// Returns true if the given f32 needs a constant pool entry (not zero, not FMOV-encodable).
#[must_use]
pub fn needs_const_pool(val: f32) -> bool {
    val.to_bits() != 0 && try_encode_fmov_imm8(val).is_none()
}

/// Emit `ADR X17, #0` as a placeholder. Returns the code offset for later patching.
///
/// ADR encodes a PC-relative offset into X17 (IP1, platform scratch register).
/// The offset is patched after the constant pool position is known.
pub fn emit_adr_x17_placeholder(code: &mut Vec<u8>) -> usize {
    let pos = code.len();
    // ADR X17, #0 — will be patched. Encoding: 0x10000011 (Rd=X17=17, imm=0)
    emit32(code, 0x10000011);
    pos
}

/// Patch a previously emitted `ADR X17` placeholder at `adr_pos` to point to `target_pos`.
/// If `is_adrp` is true, assumes 8 bytes are reserved and patches `ADRP X17` + `ADD X17`.
pub fn patch_adr_or_adrp(code: &mut [u8], adr_pos: usize, target_pos: usize, is_adrp: bool) {
    if is_adrp {
        assert!(
            adr_pos + 8 <= code.len(),
            "patch_adr_or_adrp: adr_pos {} + 8 exceeds code length {}",
            adr_pos,
            code.len()
        );

        let pc_page = (adr_pos as i64) & !0xFFF;
        let target_page = (target_pos as i64) & !0xFFF;
        let page_offset = (target_page - pc_page) >> 12;

        assert!(
            (-(1 << 20)..(1 << 20)).contains(&page_offset),
            "ADRP page offset {} out of range (±4GB)",
            page_offset
        );

        // 1. Patch ADRP
        let imm_bits = (page_offset as u32) & 0x1F_FFFF;
        let immlo = imm_bits & 0x3;
        let immhi = (imm_bits >> 2) & 0x7FFFF;
        let adrp_inst = 0x90000011 | (immlo << 29) | (immhi << 5);
        code[adr_pos..adr_pos + 4].copy_from_slice(&adrp_inst.to_le_bytes());

        // 2. Patch ADD (immediate)
        // ADD X17, X17, #target_pos_within_page
        let page_inner_offset = (target_pos as u32) & 0xFFF;
        let add_inst = 0x91000231 | (page_inner_offset << 10);
        code[adr_pos + 4..adr_pos + 8].copy_from_slice(&add_inst.to_le_bytes());
    } else {
        assert!(
            adr_pos + 4 <= code.len(),
            "patch_adr_or_adrp: adr_pos {} + 4 exceeds code length {}",
            adr_pos,
            code.len()
        );
        let offset = (target_pos as i64) - (adr_pos as i64);
        assert!(
            (-(1 << 20)..(1 << 20)).contains(&offset),
            "ADR offset {} out of range (±1MB)",
            offset
        );
        let offset_bits = (offset as u32) & 0x1F_FFFF;
        let immlo = offset_bits & 0x3;
        let immhi = (offset_bits >> 2) & 0x7FFFF;
        let inst = 0x10000011 | (immlo << 29) | (immhi << 5);
        code[adr_pos..adr_pos + 4].copy_from_slice(&inst.to_le_bytes());
    }
}

/// Emit a constant pool entry: 16 bytes = f32 value splatted 4x (fills a 128-bit NEON register).
pub fn emit_pool_entry(code: &mut Vec<u8>, val_bits: u32) {
    let bytes = val_bits.to_le_bytes();
    for _ in 0..4 {
        code.extend_from_slice(&bytes);
    }
}

// =============================================================================
// Bound-Memory Gather (scalar-load lowering — NEON has no native gather)
// =============================================================================

/// UMOV Wd, Vn.S[lane] — extract a 32-bit vector lane into a GP register.
pub fn emit_umov_w(code: &mut Vec<u8>, dst: Xr, src: Reg, lane: u8) {
    debug_assert!(lane < 4);
    let imm5 = ((lane as u32) << 3) | 0b100; // S-lane element size
    emit32(
        code,
        0x0E003C00 | (imm5 << 16) | ((src.0 as u32) << 5) | (dst.0 as u32),
    );
}

/// INS Vd.S[lane], Wn — insert a GP register into a 32-bit vector lane.
pub fn emit_ins_w(code: &mut Vec<u8>, dst: Reg, lane: u8, src: Xr) {
    debug_assert!(lane < 4);
    let imm5 = ((lane as u32) << 3) | 0b100;
    emit32(
        code,
        0x4E001C00 | (imm5 << 16) | ((src.0 as u32) << 5) | (dst.0 as u32),
    );
}

/// GP registers used by the scalar-load gather sequence.
pub struct GatherGprs {
    /// Holds the buffer base pointer (survives the whole sequence).
    pub base: Xr,
    /// Scratch: one extracted lane index at a time. Clobbered.
    pub idx: Xr,
    /// Scratch: one loaded value at a time. Clobbered.
    pub val: Xr,
}

/// dst.4S = base[idx_int.S[lane]] for each lane — the NEON gather: four scalar
/// loads through GP scratch. `gprs.base` holds the buffer base pointer;
/// `idx_int` holds int32 lane indices (already converted and in-bounds by the
/// `expand_gather` lowering). Clobbers `gprs.idx` and `gprs.val`.
pub fn emit_gather(code: &mut Vec<u8>, dst: Reg, idx_int: Reg, gprs: GatherGprs) {
    for lane in 0..4 {
        emit_umov_w(code, gprs.idx, idx_int, lane);
        emit_ldr_w(
            code,
            gprs.val,
            MemIndexed {
                base: gprs.base,
                index: gprs.idx,
            },
        );
        emit_ins_w(code, dst, lane, gprs.val);
    }
}

// =============================================================================
// Integer Vector Operations (for bit manipulation in transcendentals)
// =============================================================================

/// USHR Vd.4S, Vn.4S, #shift (unsigned shift right by immediate)
fn emit_ushr(code: &mut Vec<u8>, dst: Reg, src: Reg, shift: u8) {
    // Encoding: 0x6F200400 | ((32 - shift) << 16) as immh:immb
    // For .4S: immh = 001x, so (32-shift) in bits [19:16]
    // A shift by zero is the identity, and USHR cannot encode it: `64 - 0` is
    // 64, which does not fit the 6-bit immediate field. Emit the move instead
    // of refusing a perfectly portable operation — `fold_is_platform_specific`
    // classifies a count of 0 as agreeing on every target, so the encoder has
    // to honour it.
    if shift == 0 {
        emit_mov(code, dst, src);
        return;
    }
    // `immh` selects the element size: 01xx is .4S, 001x is .8H, 1xxx is .2D.
    // Only shifts in 1..=32 keep `64 - shift` inside 01xx, so anything else
    // silently encodes a DIFFERENT element size and crosses lane boundaries.
    assert!(
        shift <= 32,
        "aarch64 USHR .4S: shift {shift} exceeds 32 — the immediate would \
         encode a different element size, not a 32-bit lane shift"
    );
    let immhb = (64 - shift as u32) & 0x3F; // USHR uses (immh:immb) = (size*2 - shift)
    let inst = 0x6F200400 | (dst.0 as u32) | ((src.0 as u32) << 5) | (immhb << 16);
    emit32(code, inst);
}

/// SHL Vd.4S, Vn.4S, #shift (shift left by immediate)
fn emit_shl(code: &mut Vec<u8>, dst: Reg, src: Reg, shift: u8) {
    // For .4S: immh:immb = shift + 32. At shift >= 32 that carries into
    // `immh`, making it 1xxx — which the ARM ARM decodes as .2D, a 64-bit
    // element shift that leaks bits across the 32-bit lane boundary. Refuse
    // loudly rather than emit a silently different instruction.
    assert!(
        shift < 32,
        "aarch64 SHL .4S: shift {shift} is out of range for a 32-bit lane — \
         the immediate would encode .2D and cross lane boundaries"
    );
    let immhb = (shift as u32) + 32;
    let inst = 0x4F005400 | (dst.0 as u32) | ((src.0 as u32) << 5) | (immhb << 16);
    emit32(code, inst);
}

/// ADD Vd.4S, Vn.4S, Vm.4S (integer add)
fn emit_add_i32(code: &mut Vec<u8>, dst: Reg, src1: Reg, src2: Reg) {
    emit32(code, encode_3same(0x4EA08400, dst, src1, src2));
}

/// AND Vd.16B, Vn.16B, Vm.16B (bitwise AND)
fn emit_and(code: &mut Vec<u8>, dst: Reg, src1: Reg, src2: Reg) {
    emit32(code, encode_3same(0x4E201C00, dst, src1, src2));
}

/// ORR Vd.16B, Vn.16B, Vm.16B (bitwise OR)
fn emit_orr(code: &mut Vec<u8>, dst: Reg, src1: Reg, src2: Reg) {
    emit32(code, encode_3same(0x4EA01C00, dst, src1, src2));
}

/// FCVTZS Vd.4S, Vn.4S (float to signed int, round toward zero)
pub fn emit_fcvtzs(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    emit32(code, encode_2misc(0x4EA1B800, dst, src));
}

/// SCVTF Vd.4S, Vn.4S (signed int to float)
#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
fn emit_scvtf(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    emit32(code, encode_2misc(0x4E21D800, dst, src));
}

/// FRINTA Vd.4S, Vn.4S (round to nearest, ties away from zero)
fn emit_frinta(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    emit32(code, encode_2misc(0x6E218800, dst, src));
}

/// MOV Vd.16B, Vn.16B (register copy via ORR)
fn emit_mov(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    if dst.0 != src.0 {
        emit_orr(code, dst, src, src);
    }
}

/// round(x) — round to nearest, ties away from zero. ARM64 FRINTA instruction.
pub fn emit_round_builtin(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    emit_frinta(code, dst, src);
}

// =============================================================================
// Binary Transcendental Builtins
// =============================================================================

// =============================================================================
// Compound Operations (emit full instruction sequences)
// =============================================================================

/// How many registers this backend's encodings need beyond their operands.
///
/// Only the reciprocal estimates: `FRECPE`/`FRSQRTE` are estimates, and the
/// Newton-Raphson step that refines them needs somewhere to hold the
/// correction. `Neg` and `Abs` are single instructions here (`FNEG`, `FABS`),
/// unlike the x86 backends where they materialize a sign mask, and `BSL`
/// blends a select from its three operands.
pub(crate) fn temps_for(op: &super::ScheduledOp) -> u8 {
    use super::ScheduledOp;
    match op {
        ScheduledOp::Unary(OpKind::Rsqrt | OpKind::Recip, _) => 1,
        // The gather's truncated-index lanes.
        ScheduledOp::Gather(..) => 1,
        _ => 0,
    }
}

#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
/// `dst = op(src)`.
///
/// `temp` is the allocator's temp for this instruction; only the reciprocal
/// estimates use it, to hold the Newton-Raphson correction.
pub(crate) fn emit_unary(code: &mut Vec<u8>, op: OpKind, dst: Reg, src: Reg, temp: Option<Reg>) {
    match op {
        OpKind::Neg => emit_fneg(code, dst, src),
        OpKind::Abs => emit_fabs(code, dst, src),
        OpKind::Sqrt => emit_fsqrt(code, dst, src),
        OpKind::Rsqrt => emit_frsqrt(code, dst, src, super::declared_temp(temp)),
        OpKind::Recip => emit_frecip(code, dst, src, super::declared_temp(temp)),
        OpKind::Floor => emit_frintm(code, dst, src),
        OpKind::Ceil => emit_frintp(code, dst, src),
        OpKind::Round => emit_round_builtin(code, dst, src),

        // Bit-manip primitives (integer-domain conversions).
        OpKind::TruncToInt => emit_fcvtzs(code, dst, src), // f32 -> i32 (truncate)
        OpKind::IntToFloat => emit_scvtf(code, dst, src),  // i32 -> f32

        // Transcendentals (sin/cos/tan/exp/exp2/ln/log2/log10/atan/asin/acos) are
        // expanded to primitive arithmetic by `lowering` before codegen, so they
        // never reach a backend. Reaching here means lowering was skipped.
        _ => unimplemented_op("aarch64", op),
    }
}

/// Emit a logical shift of i32 lanes by a compile-time immediate.
/// `Shl` -> `SHL`, `Shr` -> `USHR` (logical right). NEON shifts are imm-form.
pub fn emit_shift_imm(code: &mut Vec<u8>, op: OpKind, dst: Reg, src: Reg, amount: u8) {
    match op {
        OpKind::Shl => emit_shl(code, dst, src, amount),
        OpKind::Shr => emit_ushr(code, dst, src, amount),
        _ => unimplemented_op("aarch64", op),
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
            // Ne = not Eq: FCMEQ then bitwise NOT
            emit_fcmeq(code, dst, src1, src2);
            emit_not(code, dst, dst);
        }

        // Bit-manip primitives (integer-domain).
        OpKind::IAdd => emit_add_i32(code, dst, src1, src2),
        OpKind::BitAnd => emit_and(code, dst, src1, src2),
        OpKind::BitOr => emit_orr(code, dst, src1, src2),

        _ => unimplemented_op("aarch64", op),
    }
}

// =============================================================================
// Select Short-Circuit Helpers
// =============================================================================

/// UMINV Sd, Vn.4S — horizontal unsigned minimum across all 4 lanes.
/// Result is in lane 0 of dst (scalar Sd).
/// If mask is all-ones (0xFFFFFFFF per lane), result = 0xFFFFFFFF.
pub fn emit_uminv(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    // UMINV Vd.4S: 0x6EB1A800 | Rd | (Rn << 5)
    // Encoding: 0 1 1 0 1 1 1 0 1 0 1 1 0 0 0 1  1 0 1 0 1 0 0 0  Rn:5 Rd:5
    emit32(code, 0x6EB1A800 | (dst.0 as u32) | ((src.0 as u32) << 5));
}

/// UMAXV Sd, Vn.4S — horizontal unsigned maximum across all 4 lanes.
/// If mask is all-zeros, result = 0x00000000.
pub fn emit_umaxv(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    // UMAXV Vd.4S: 0x6E30A800 | Rd | (Rn << 5)
    emit32(code, 0x6E30A800 | (dst.0 as u32) | ((src.0 as u32) << 5));
}

/// FMOV Wd, Sn — move lane 0 of SIMD register to GP register W16.
/// We always use W16 as the GP scratch for Select branching.
pub fn emit_fmov_to_gp(code: &mut Vec<u8>, src: Reg) {
    // FMOV W16, Sn: 0x1E260000 | (Rn << 5) | Rd
    // where Rd=16 (W16), Rn is the SIMD register number
    emit32(code, 0x1E260000 | ((src.0 as u32) << 5) | 16);
}

/// CBZ W16, #offset — branch if W16 == 0 (mask all-false).
/// `offset` is in bytes, must be aligned to 4, range ±1MB.
/// Returns the index in `code` where the offset is encoded (for patching).
pub fn emit_cbz_w16(code: &mut Vec<u8>) -> usize {
    let patch_pos = code.len();
    // CBZ W16, #0 (placeholder offset)
    // Encoding: 0 0110100 imm19 Rt
    // Rt = 16 (W16)
    emit32(code, 0x34000010); // imm19 = 0, will be patched
    patch_pos
}

/// B #offset — unconditional branch (for skipping past else-arm).
/// Returns the index in `code` where the offset is encoded (for patching).
pub fn emit_b(code: &mut Vec<u8>) -> usize {
    let patch_pos = code.len();
    // B #0 (placeholder)
    emit32(code, 0x14000000); // imm26 = 0, will be patched
    patch_pos
}

/// Patch a CBZ/CBNZ instruction at `patch_pos` to branch to `target_pos`.
/// Both positions are byte offsets into the code buffer.
pub fn patch_cbz_cbnz(code: &mut [u8], patch_pos: usize, target_pos: usize) {
    let offset = (target_pos as i64 - patch_pos as i64) / 4;
    assert!(
        (-(1 << 18)..(1 << 18)).contains(&offset),
        "CBZ/CBNZ branch offset {} out of range (±1MB)",
        offset
    );
    let imm19 = (offset as u32) & 0x7FFFF;
    let existing = u32::from_le_bytes([
        code[patch_pos],
        code[patch_pos + 1],
        code[patch_pos + 2],
        code[patch_pos + 3],
    ]);
    let patched = (existing & 0xFF00001F) | (imm19 << 5);
    code[patch_pos..patch_pos + 4].copy_from_slice(&patched.to_le_bytes());
}

/// Patch an unconditional B instruction at `patch_pos` to branch to `target_pos`.
pub fn patch_b(code: &mut [u8], patch_pos: usize, target_pos: usize) {
    let offset = (target_pos as i64 - patch_pos as i64) / 4;
    assert!(
        (-(1 << 25)..(1 << 25)).contains(&offset),
        "B branch offset {} out of range (±128MB)",
        offset
    );
    let imm26 = (offset as u32) & 0x3FFFFFF;
    let existing = u32::from_le_bytes([
        code[patch_pos],
        code[patch_pos + 1],
        code[patch_pos + 2],
        code[patch_pos + 3],
    ]);
    let patched = (existing & 0xFC000000) | imm26;
    code[patch_pos..patch_pos + 4].copy_from_slice(&patched.to_le_bytes());
}

// =============================================================================
// Prologue / Epilogue
// =============================================================================

// =============================================================================
// Disassembly support
// =============================================================================

/// Disassemble a raw code buffer into a human-readable string.
///
#[must_use]
pub fn disassemble_code(code: &[u8]) -> String {
    let mut out = String::new();
    for (i, chunk) in code.chunks(4).enumerate() {
        if chunk.len() < 4 {
            break;
        }
        let word = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let offset = i * 4;
        let mnemonic = decode_aarch64_mnemonic(word);
        out.push_str(&format!("{:4x}: {:08x}  {}\n", offset, word, mnemonic));
    }
    out
}

/// Decode a 32-bit AArch64 instruction into a mnemonic string.
///
/// Covers the instructions actually emitted by this JIT:
/// - NEON floating-point (fadd, fmul, fsub, fdiv, fabs, fneg, fsqrt, fmin, fmax)
/// - NEON integer (add, sub, shl, ushr, and, orr, not)
/// - Comparisons (fcmgt, fcmge, fcmeq)
/// - Selection (bsl)
/// - Move/dup (mov v, dup)
/// - Memory (ldr, str, ldp, stp)
/// - Control (ret, cbz, cbnz, b, b.cond, subs)
/// - Constants (movi, fmov imm, movz, movk)
///
/// Everything else returns "unknown".
fn decode_aarch64_mnemonic(word: u32) -> String {
    let rd = word & 0x1F;
    let rn = (word >> 5) & 0x1F;
    let rm = (word >> 16) & 0x1F;

    // RET
    if word == 0xD65F03C0 {
        return "ret".into();
    }

    // MOVI Vd.4S, #0 (common zero-fill)
    if word & 0xFFFFFC00 == 0x4F000400 {
        return format!("movi v{}.4s, #0", rd);
    }

    // FMOV Vd.4S, #imm8
    if word & 0xFFC0FC00 == 0x4F00F400 {
        return format!("fmov v{}.4s, #imm8", rd);
    }

    // DUP Vd.4S, W16 (from GP)
    if word & 0xFFFFFC00 == 0x4E040C00 {
        return format!("dup v{}.4s, w{}", rd, rn);
    }

    // DUP Vd.4S, Vn.S[0] (scalar dup)
    if word & 0xFFFFFC00 == 0x4E040400 {
        return format!("dup v{}.4s, v{}.s[0]", rd, rn);
    }

    // MOVZ Wd, #imm16
    if word & 0xFFE0001F == 0x52800010 {
        let imm16 = (word >> 5) & 0xFFFF;
        return format!("movz w16, #0x{:x}", imm16);
    }
    if word & 0xFFE00000 == 0x52800000 {
        let imm16 = (word >> 5) & 0xFFFF;
        return format!("movz w{}, #0x{:x}", rd, imm16);
    }
    // MOVZ Xd, #imm16 (64-bit)
    if word & 0xFFE00000 == 0xD2800000 {
        let imm16 = (word >> 5) & 0xFFFF;
        return format!("movz x{}, #0x{:x}", rd, imm16);
    }

    // MOVK Wd, #imm16, LSL #16
    if word & 0xFFE00000 == 0x72A00000 {
        let imm16 = (word >> 5) & 0xFFFF;
        return format!("movk w{}, #0x{:x}, lsl #16", rd, imm16);
    }

    // MOV Xd, Xm (ORR Xd, XZR, Xm)
    if word & 0xFFE0FFE0 == 0xAA0003E0 {
        return format!("mov x{}, x{}", rd, rm);
    }

    // NEON 3-same (binary vector ops) — top bits determine the operation
    // Extract opcode bits for classification
    let top11 = word >> 21;

    // ORR Vd.16B, Vn.16B, Vm.16B (also MOV when Vn==Vm)
    if word & 0xFFE0FC00 == 0x4EA01C00 {
        if rn == rm {
            return format!("mov v{}.16b, v{}.16b", rd, rn);
        }
        return format!("orr v{}.16b, v{}.16b, v{}.16b", rd, rn, rm);
    }

    // FADD Vd.4S, Vn.4S, Vm.4S
    if word & 0xFFE0FC00 == 0x4E20D400 {
        return format!("fadd v{}.4s, v{}.4s, v{}.4s", rd, rn, rm);
    }
    // FSUB Vd.4S, Vn.4S, Vm.4S
    if word & 0xFFE0FC00 == 0x4EA0D400 {
        return format!("fsub v{}.4s, v{}.4s, v{}.4s", rd, rn, rm);
    }
    // FMUL Vd.4S, Vn.4S, Vm.4S
    if word & 0xFFE0FC00 == 0x6E20DC00 {
        return format!("fmul v{}.4s, v{}.4s, v{}.4s", rd, rn, rm);
    }
    // FDIV Vd.4S, Vn.4S, Vm.4S
    if word & 0xFFE0FC00 == 0x6E20FC00 {
        return format!("fdiv v{}.4s, v{}.4s, v{}.4s", rd, rn, rm);
    }
    // FMLA Vd.4S, Vn.4S, Vm.4S (fused multiply-add)
    if word & 0xFFE0FC00 == 0x4E20CC00 {
        return format!("fmla v{}.4s, v{}.4s, v{}.4s", rd, rn, rm);
    }
    // FMIN Vd.4S, Vn.4S, Vm.4S
    if word & 0xFFE0FC00 == 0x4EA0F400 {
        return format!("fmin v{}.4s, v{}.4s, v{}.4s", rd, rn, rm);
    }
    // FMAX Vd.4S, Vn.4S, Vm.4S
    if word & 0xFFE0FC00 == 0x4E20F400 {
        return format!("fmax v{}.4s, v{}.4s, v{}.4s", rd, rn, rm);
    }

    // FCMGT Vd.4S, Vn.4S, Vm.4S
    if word & 0xFFE0FC00 == 0x6EA0E400 {
        return format!("fcmgt v{}.4s, v{}.4s, v{}.4s", rd, rn, rm);
    }
    // FCMGE Vd.4S, Vn.4S, Vm.4S
    if word & 0xFFE0FC00 == 0x6E20E400 {
        return format!("fcmge v{}.4s, v{}.4s, v{}.4s", rd, rn, rm);
    }
    // FCMEQ Vd.4S, Vn.4S, Vm.4S
    if word & 0xFFE0FC00 == 0x4E20E400 {
        return format!("fcmeq v{}.4s, v{}.4s, v{}.4s", rd, rn, rm);
    }

    // BSL Vd.16B, Vn.16B, Vm.16B
    if word & 0xFFE0FC00 == 0x6E601C00 {
        return format!("bsl v{}.16b, v{}.16b, v{}.16b", rd, rn, rm);
    }

    // AND Vd.16B, Vn.16B, Vm.16B
    if word & 0xFFE0FC00 == 0x4E201C00 {
        return format!("and v{}.16b, v{}.16b, v{}.16b", rd, rn, rm);
    }

    // ADD Vd.4S, Vn.4S, Vm.4S (integer)
    if word & 0xFFE0FC00 == 0x4EA08400 {
        return format!("add v{}.4s, v{}.4s, v{}.4s", rd, rn, rm);
    }
    // SUB Vd.4S, Vn.4S, Vm.4S (integer)
    if word & 0xFFE0FC00 == 0x6EA08400 {
        return format!("sub v{}.4s, v{}.4s, v{}.4s", rd, rn, rm);
    }

    // FRSQRTS Vd.4S, Vn.4S, Vm.4S
    if word & 0xFFE0FC00 == 0x4EA0FC00 {
        return format!("frsqrts v{}.4s, v{}.4s, v{}.4s", rd, rn, rm);
    }
    // FRECPS Vd.4S, Vn.4S, Vm.4S
    if word & 0xFFE0FC00 == 0x4E20FC00 {
        return format!("frecps v{}.4s, v{}.4s, v{}.4s", rd, rn, rm);
    }

    // 2-reg misc (unary vector ops)
    // FSQRT Vd.4S, Vn.4S
    if word & 0xFFFFFC00 == 0x6EA1F800 {
        return format!("fsqrt v{}.4s, v{}.4s", rd, rn);
    }
    // FABS Vd.4S, Vn.4S
    if word & 0xFFFFFC00 == 0x4EA0F800 {
        return format!("fabs v{}.4s, v{}.4s", rd, rn);
    }
    // FNEG Vd.4S, Vn.4S
    if word & 0xFFFFFC00 == 0x6EA0F800 {
        return format!("fneg v{}.4s, v{}.4s", rd, rn);
    }
    // NOT Vd.16B, Vn.16B
    if word & 0xFFFFFC00 == 0x2E205800 {
        return format!("not v{}.16b, v{}.16b", rd, rn);
    }
    // FRINTM Vd.4S, Vn.4S (floor)
    if word & 0xFFFFFC00 == 0x4E219800 {
        return format!("frintm v{}.4s, v{}.4s", rd, rn);
    }
    // FRINTP Vd.4S, Vn.4S (ceil)
    if word & 0xFFFFFC00 == 0x4EA18800 {
        return format!("frintp v{}.4s, v{}.4s", rd, rn);
    }
    // FRINTA Vd.4S, Vn.4S (round)
    if word & 0xFFFFFC00 == 0x6E218800 {
        return format!("frinta v{}.4s, v{}.4s", rd, rn);
    }
    // FRSQRTE Vd.4S, Vn.4S
    if word & 0xFFFFFC00 == 0x6EA1D800 {
        return format!("frsqrte v{}.4s, v{}.4s", rd, rn);
    }
    // FRECPE Vd.4S, Vn.4S
    if word & 0xFFFFFC00 == 0x4EA1D800 {
        return format!("frecpe v{}.4s, v{}.4s", rd, rn);
    }
    // FCVTZS Vd.4S, Vn.4S
    if word & 0xFFFFFC00 == 0x4EA1B800 {
        return format!("fcvtzs v{}.4s, v{}.4s", rd, rn);
    }
    // SCVTF Vd.4S, Vn.4S
    if word & 0xFFFFFC00 == 0x4E21D800 {
        return format!("scvtf v{}.4s, v{}.4s", rd, rn);
    }
    // UMINV Sd, Vn.4S
    if word & 0xFFFFFC00 == 0x6EB1A800 {
        return format!("uminv s{}, v{}.4s", rd, rn);
    }
    // UMAXV Sd, Vn.4S
    if word & 0xFFFFFC00 == 0x6E30A800 {
        return format!("umaxv s{}, v{}.4s", rd, rn);
    }

    // FMOV Wd, Sn (SIMD to GP)
    if word & 0xFFFFFC00 == 0x1E260000 {
        return format!("fmov w{}, s{}", rd, rn);
    }

    // LDR Qt, [Xn, #imm] (128-bit unsigned offset)
    if word & 0xFFC00000 == 0x3DC00000 {
        let imm12 = (word >> 10) & 0xFFF;
        let byte_offset = imm12 * 16;
        return format!("ldr q{}, [x{}, #{}]", rd, rn, byte_offset);
    }
    // STR Qt, [Xn, #imm] (128-bit unsigned offset)
    if word & 0xFFC00000 == 0x3D800000 {
        let imm12 = (word >> 10) & 0xFFF;
        let byte_offset = imm12 * 16;
        return format!("str q{}, [x{}, #{}]", rd, rn, byte_offset);
    }

    // LDR Qt, [Xn], #16 (post-index)
    if word & 0xFFFFFC00 == 0x3CC10400 {
        return format!("ldr q{}, [x{}], #16", rd, rn);
    }
    // STR Qt, [Xn], #16 (post-index)
    if word & 0xFFFFFC00 == 0x3C810400 {
        return format!("str q{}, [x{}], #16", rd, rn);
    }

    // STP Xt1, Xt2, [Xn, #imm]! (pre-index GP pair)
    if word & 0xFFC00000 == 0xA9800000 | (0b11 << 23) {
        let rt2 = (word >> 10) & 0x1F;
        let imm7 = ((word >> 15) & 0x7F) as i32;
        let offset = (if imm7 >= 64 { imm7 - 128 } else { imm7 }) * 8;
        return format!("stp x{}, x{}, [x{}, #{}]!", rd, rt2, rn, offset);
    }

    // STP Xt1, Xt2, [Xn, #imm] (signed offset GP pair, no writeback)
    if word & 0xFFC00000 == 0xA9000000 {
        let rt2 = (word >> 10) & 0x1F;
        let imm7 = ((word >> 15) & 0x7F) as i32;
        let offset = (if imm7 >= 64 { imm7 - 128 } else { imm7 }) * 8;
        return format!("stp x{}, x{}, [x{}, #{}]", rd, rt2, rn, offset);
    }

    // LDP Xt1, Xt2, [Xn], #imm (post-index GP pair)
    if word & 0xFFC00000 == 0xA8C00000 {
        let rt2 = (word >> 10) & 0x1F;
        let imm7 = ((word >> 15) & 0x7F) as i32;
        let offset = (if imm7 >= 64 { imm7 - 128 } else { imm7 }) * 8;
        return format!("ldp x{}, x{}, [x{}], #{}", rd, rt2, rn, offset);
    }

    // LDP Xt1, Xt2, [Xn, #imm] (signed offset GP pair)
    if word & 0xFFC00000 == 0xA9400000 {
        let rt2 = (word >> 10) & 0x1F;
        let imm7 = ((word >> 15) & 0x7F) as i32;
        let offset = (if imm7 >= 64 { imm7 - 128 } else { imm7 }) * 8;
        return format!("ldp x{}, x{}, [x{}, #{}]", rd, rt2, rn, offset);
    }

    // STP Qt1, Qt2, [Xn, #imm] (NEON pair, signed offset)
    if word & 0xFFC00000 == 0xAD000000 {
        let rt2 = (word >> 10) & 0x1F;
        let imm7 = ((word >> 15) & 0x7F) as i32;
        let offset = (if imm7 >= 64 { imm7 - 128 } else { imm7 }) * 16;
        return format!("stp q{}, q{}, [x{}, #{}]", rd, rt2, rn, offset);
    }

    // LDP Qt1, Qt2, [Xn, #imm] (NEON pair, signed offset)
    if word & 0xFFC00000 == 0xAD400000 {
        let rt2 = (word >> 10) & 0x1F;
        let imm7 = ((word >> 15) & 0x7F) as i32;
        let offset = (if imm7 >= 64 { imm7 - 128 } else { imm7 }) * 16;
        return format!("ldp q{}, q{}, [x{}, #{}]", rd, rt2, rn, offset);
    }

    // ADD Xd, Xn, #imm12 (GP immediate)
    if word & 0xFF000000 == 0x91000000 {
        let imm12 = (word >> 10) & 0xFFF;
        return format!("add x{}, x{}, #{}", rd, rn, imm12);
    }

    // SUB Xd, Xn, #imm12 (GP immediate) -- includes SUBS via 0xF1
    if word & 0xFF000000 == 0xD1000000 {
        let imm12 = (word >> 10) & 0xFFF;
        return format!("sub x{}, x{}, #{}", rd, rn, imm12);
    }
    if word & 0xFF000000 == 0xF1000000 {
        let imm12 = (word >> 10) & 0xFFF;
        return format!("subs x{}, x{}, #{}", rd, rn, imm12);
    }

    // USHR Vd.4S, Vn.4S, #shift
    // Mask includes immh[3:2] (bits 22:21) so the .4S arrangement selector
    // (immh = 01xx) is matched; bits 20:16 carry the shift amount and stay free.
    if word & 0xFFE0FC00 == 0x6F200400 {
        let immhb = (word >> 16) & 0x3F;
        let shift = 64u32.wrapping_sub(immhb) & 0x3F;
        return format!("ushr v{}.4s, v{}.4s, #{}", rd, rn, shift);
    }

    // SHL Vd.4S, Vn.4S, #shift
    // Mask includes immh[3:2] (bits 22:21) so only the .4S arrangement
    // (immh = 01xx) matches; bits 20:16 carry the shift amount and stay free.
    // Without this a 64-bit `SHL .2D` (immh = 1xxx) would mis-decode as `.4s`.
    if word & 0xFFE0FC00 == 0x4F205400 {
        let immhb = (word >> 16) & 0x3F;
        let shift = immhb.wrapping_sub(32);
        return format!("shl v{}.4s, v{}.4s, #{}", rd, rn, shift);
    }

    // ADR Xd, #imm
    if word & 0x9F000000 == 0x10000000 {
        return format!("adr x{}, <imm>", rd);
    }
    // ADRP Xd, #imm
    if word & 0x9F000000 == 0x90000000 {
        return format!("adrp x{}, <imm>", rd);
    }

    // CBZ Wt (32-bit)
    if word & 0xFF000000 == 0x34000000 {
        return format!("cbz w{}, <imm>", rd);
    }
    // CBNZ Wt (32-bit)
    if word & 0xFF000000 == 0x35000000 {
        return format!("cbnz w{}, <imm>", rd);
    }
    // CBZ Xt (64-bit)
    if word & 0xFF000000 == 0xB4000000 {
        return format!("cbz x{}, <imm>", rd);
    }
    // CBNZ Xt (64-bit)
    if word & 0xFF000000 == 0xB5000000 {
        return format!("cbnz x{}, <imm>", rd);
    }

    // B.cond
    if word & 0xFF000010 == 0x54000000 {
        let cond = word & 0xF;
        let cond_name = match cond {
            0x0 => "eq",
            0x1 => "ne",
            0x2 => "cs",
            0x3 => "cc",
            0x4 => "mi",
            0x5 => "pl",
            0x8 => "hi",
            0x9 => "ls",
            0xA => "ge",
            0xB => "lt",
            0xC => "gt",
            0xD => "le",
            _ => "??",
        };
        return format!("b.{} <imm>", cond_name);
    }

    // B (unconditional)
    if word & 0xFC000000 == 0x14000000 {
        return "b <imm>".to_string();
    }

    // SUBS Xd, Xn, Xm (register)
    if word & 0xFFE00000 == 0xEB000000 {
        return format!("subs x{}, x{}, x{}", rd, rn, rm);
    }

    let _ = top11; // suppress unused warning
    "unknown".into()
}

// =============================================================================
// dump_jit_asm — compile expression and return disassembly
// =============================================================================

/// Compile an expression from an [`ExprArena`] and return its disassembly.
///
/// This is a diagnostic entry point: it compiles the expression through the
/// normal JIT pipeline, then disassembles the resulting machine code instead
/// of executing it. Useful for inspecting what the JIT generates.
///
/// # Errors
///
/// Returns [`crate::error::CompileError`] if compilation fails (same errors as
/// [`compile`](super::compile)).
#[cfg(target_arch = "aarch64")]
pub fn dump_jit_asm(
    arena: &pixelflow_ir::arena::ExprArena,
    root: pixelflow_ir::arena::ExprId,
) -> Result<String, crate::error::CompileError> {
    let result = super::compile(arena, root)?;
    Ok(disassemble_code(result.code.as_bytes()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fmov_imm8_common_values() {
        // Encodable values — imm8 derived from ARM ARM bit layout:
        //   f32 = [a][NOT(b)][bbbbb][cdefgh][19 zeros]
        //   imm8 = a:b:c:d:e:f:g:h
        assert_eq!(try_encode_fmov_imm8(2.0), Some(0x00)); // 0x40000000
        assert_eq!(try_encode_fmov_imm8(0.5), Some(0x60)); // 0x3F000000
        assert_eq!(try_encode_fmov_imm8(1.0), Some(0x70)); // 0x3F800000
        assert_eq!(try_encode_fmov_imm8(1.5), Some(0x78)); // 0x3FC00000
        assert_eq!(try_encode_fmov_imm8(-1.0), Some(0xF0)); // 0xBF800000
        assert_eq!(try_encode_fmov_imm8(-0.5), Some(0xE0)); // 0xBF000000
        assert_eq!(try_encode_fmov_imm8(-2.0), Some(0x80)); // 0xC0000000
        assert_eq!(try_encode_fmov_imm8(4.0), Some(0x10)); // 0x40800000

        // More encodable values
        assert_eq!(try_encode_fmov_imm8(3.0), Some(0x08)); // 0x40400000
        assert_eq!(try_encode_fmov_imm8(0.25), Some(0x50)); // 0x3E800000
        assert_eq!(try_encode_fmov_imm8(0.125), Some(0x40)); // 0x3E000000

        // Non-encodable values
        assert_eq!(try_encode_fmov_imm8(0.0), None);
        assert_eq!(try_encode_fmov_imm8(-0.0), None);
        assert_eq!(try_encode_fmov_imm8(0.1), None);
        assert_eq!(try_encode_fmov_imm8(f32::NAN), None);
        assert_eq!(try_encode_fmov_imm8(f32::INFINITY), None);
        assert_eq!(try_encode_fmov_imm8(100.0), None);
    }

    #[test]
    fn fmov_imm8_roundtrip() {
        // Every valid imm8 should encode a value that round-trips
        for imm8 in 0..=255u8 {
            let a = (imm8 >> 7) & 1;
            let b = (imm8 >> 6) & 1;
            let not_b = b ^ 1;
            let cdefgh = imm8 & 0x3F;

            let mut bits: u32 = 0;
            bits |= (a as u32) << 31;
            bits |= (not_b as u32) << 30;
            // bits[29:25] = bbbbb
            let rep5 = if b == 1 { 0x1F_u32 } else { 0x00 };
            bits |= rep5 << 25;
            bits |= (cdefgh as u32) << 19;
            // bits[18:0] = 0

            let val = f32::from_bits(bits);
            let result = try_encode_fmov_imm8(val);
            assert_eq!(
                result,
                Some(imm8),
                "imm8={imm8:#04x} -> f32={val} ({bits:#010x}) did not roundtrip"
            );
        }
    }

    #[test]
    fn emit_fmov_imm_uses_single_instruction_for_encodable() {
        let mut code = Vec::new();
        let dst = Reg(0);

        // 1.0 is FMOV-encodable → should emit exactly 1 instruction (4 bytes)
        emit_fmov_imm(&mut code, dst, 1.0);
        assert_eq!(
            code.len(),
            4,
            "FMOV-encodable value should emit 1 instruction"
        );

        // Verify the encoding: 0x4F00F400 | (abc<<16) | (defgh<<5) | Rd
        // imm8=0x70=0b01110000, abc=011=3, defgh=10000=16
        let inst = u32::from_le_bytes(code[..4].try_into().unwrap());
        assert_eq!(inst, 0x4F03_F600, "FMOV V0.4S, #1.0 encoding");
    }

    #[test]
    fn emit_fmov_imm_zero_is_movi() {
        let mut code = Vec::new();
        emit_fmov_imm(&mut code, Reg(0), 0.0);
        assert_eq!(code.len(), 4, "zero should emit 1 instruction (MOVI)");
    }

    #[test]
    fn emit_fmov_imm_fallback_for_non_encodable() {
        let mut code = Vec::new();
        emit_fmov_imm(&mut code, Reg(0), core::f32::consts::PI);
        assert_eq!(
            code.len(),
            12,
            "non-encodable should emit 3 instructions (MOVZ+MOVK+DUP)"
        );
    }

    // =====================================================================
    // Disassembler tests
    // =====================================================================

    #[test]
    fn disassemble_ret() {
        let mut code = Vec::new();
        ret(&mut code);
        let dis = disassemble_code(&code);
        assert!(
            dis.contains("ret"),
            "disassembly should contain 'ret', got: {dis}"
        );
    }

    #[test]
    fn disassemble_fadd() {
        let mut code = Vec::new();
        emit_fadd(&mut code, Reg(0), Reg(1), Reg(2));
        let dis = disassemble_code(&code);
        assert!(
            dis.contains("fadd v0.4s, v1.4s, v2.4s"),
            "expected fadd decode, got: {dis}"
        );
    }

    // Round-trip the NEON shift-by-immediate encoders through the disassembler.
    // These guard the `immh` arrangement bits in the decoder masks: the `.4S`
    // form sets immh = 01xx, so emitted words have bits[23:21] = 001. A mask
    // that ignores those bits either never matches (the USHR bug fixed here) or
    // mis-decodes the 64-bit `.2D` form as `.4s` (the SHL case).
    #[test]
    fn disassemble_ushr() {
        let mut code = Vec::new();
        emit_ushr(&mut code, Reg(0), Reg(0), 23); // used by the log2 lowering
        let dis = disassemble_code(&code);
        assert!(
            dis.contains("ushr v0.4s, v0.4s, #23"),
            "expected ushr decode, got: {dis}"
        );
    }

    #[test]
    fn disassemble_shl() {
        let mut code = Vec::new();
        emit_shl(&mut code, Reg(1), Reg(2), 8);
        let dis = disassemble_code(&code);
        assert!(
            dis.contains("shl v1.4s, v2.4s, #8"),
            "expected shl decode, got: {dis}"
        );
    }

    #[test]
    fn disassemble_mov_vec() {
        let mut code = Vec::new();
        emit_mov(&mut code, Reg(5), Reg(3));
        let dis = disassemble_code(&code);
        assert!(
            dis.contains("mov v5.16b, v3.16b"),
            "expected mov decode, got: {dis}"
        );
    }

    #[test]
    fn disassemble_sequence() {
        let mut code = Vec::new();
        emit_fmul(&mut code, Reg(4), Reg(0), Reg(0));
        emit_fsqrt(&mut code, Reg(4), Reg(4));
        ret(&mut code);
        let dis = disassemble_code(&code);
        assert!(dis.contains("fmul"), "missing fmul in: {dis}");
        assert!(dis.contains("fsqrt"), "missing fsqrt in: {dis}");
        assert!(dis.contains("ret"), "missing ret in: {dis}");
    }

    #[test]
    fn disassemble_zero_const() {
        let mut code = Vec::new();
        emit_fmov_imm(&mut code, Reg(0), 0.0);
        let dis = disassemble_code(&code);
        assert!(
            dis.contains("movi"),
            "zero should decode as movi, got: {dis}"
        );
    }

    #[test]
    fn disassemble_ldr_str() {
        let mut code = Vec::new();
        emit_ldr_q(
            &mut code,
            Reg(0),
            Mem {
                base: xr::X0,
                offset: 32,
            },
        );
        emit_str_q(
            &mut code,
            Reg(1),
            Mem {
                base: xr::X0,
                offset: 48,
            },
        );
        let dis = disassemble_code(&code);
        assert!(dis.contains("ldr"), "missing ldr in: {dis}");
        assert!(dis.contains("str"), "missing str in: {dis}");
    }

    #[test]
    fn disassemble_code_empty() {
        let dis = disassemble_code(&[]);
        assert!(
            dis.is_empty(),
            "empty code should produce empty disassembly"
        );
    }

    #[test]
    fn disassemble_code_short_chunk() {
        // Less than 4 bytes should produce nothing
        let dis = disassemble_code(&[0x00, 0x01]);
        assert!(
            dis.is_empty(),
            "short chunk should produce empty disassembly"
        );
    }

    #[test]
    fn disassemble_offsets_are_sequential() {
        let mut code = Vec::new();
        emit_fadd(&mut code, Reg(0), Reg(1), Reg(2));
        emit_fsub(&mut code, Reg(0), Reg(1), Reg(2));
        ret(&mut code);
        let dis = disassemble_code(&code);
        // Lines should start with offsets 0, 4, 8
        assert!(
            dis.starts_with("   0:"),
            "first line should start at offset 0, got: {dis}"
        );
        let lines: Vec<&str> = dis.lines().collect();
        assert_eq!(lines.len(), 3);
        assert!(
            lines[1].starts_with("   4:"),
            "second line should start at offset 4"
        );
        assert!(
            lines[2].starts_with("   8:"),
            "third line should start at offset 8"
        );
    }

    /// Encodings cross-checked against clang: `fcvtzs v28.4s, v5.4s` etc.,
    /// assembled with `clang -c -arch arm64` and dumped with objdump.
    #[test]
    fn gather_primitive_encodings() {
        fn one(f: impl FnOnce(&mut Vec<u8>)) -> u32 {
            let mut code = Vec::new();
            f(&mut code);
            assert_eq!(code.len(), 4);
            u32::from_le_bytes(code[..4].try_into().unwrap())
        }

        // fcvtzs v28.4s, v5.4s
        assert_eq!(one(|c| emit_fcvtzs(c, Reg(28), Reg(5))), 0x4EA1B8BC);
        // ldr x9, [x0, #8]
        assert_eq!(
            one(|c| emit_ldr_x(
                c,
                Xr(9),
                Mem {
                    base: xr::X0,
                    offset: 8
                }
            )),
            0xF9400409
        );
        // ldr x9, [x0]
        assert_eq!(
            one(|c| emit_ldr_x(
                c,
                Xr(9),
                Mem {
                    base: xr::X0,
                    offset: 0
                }
            )),
            0xF9400009
        );
        // umov w10, v28.s[0..3]
        assert_eq!(one(|c| emit_umov_w(c, Xr(10), Reg(28), 0)), 0x0E043F8A);
        assert_eq!(one(|c| emit_umov_w(c, Xr(10), Reg(28), 1)), 0x0E0C3F8A);
        assert_eq!(one(|c| emit_umov_w(c, Xr(10), Reg(28), 2)), 0x0E143F8A);
        assert_eq!(one(|c| emit_umov_w(c, Xr(10), Reg(28), 3)), 0x0E1C3F8A);
        // ldr w11, [x9, w10, uxtw #2]
        assert_eq!(
            one(|c| emit_ldr_w(
                c,
                Xr(11),
                MemIndexed {
                    base: Xr(9),
                    index: Xr(10)
                }
            )),
            0xB86A592B
        );
        // ins v6.s[0..3], w11
        assert_eq!(one(|c| emit_ins_w(c, Reg(6), 0, Xr(11))), 0x4E041D66);
        assert_eq!(one(|c| emit_ins_w(c, Reg(6), 1, Xr(11))), 0x4E0C1D66);
        assert_eq!(one(|c| emit_ins_w(c, Reg(6), 2, Xr(11))), 0x4E141D66);
        assert_eq!(one(|c| emit_ins_w(c, Reg(6), 3, Xr(11))), 0x4E1C1D66);
    }

    #[test]
    fn gather_compound_is_four_scalar_loads() {
        let mut code = Vec::new();
        emit_gather(
            &mut code,
            Reg(6),
            Reg(28),
            GatherGprs {
                base: Xr(9),
                idx: Xr(10),
                val: Xr(11),
            },
        );
        // 4 lanes x (umov + ldr + ins) = 12 instructions.
        assert_eq!(code.len(), 12 * 4);
    }

    /// The base register and the addressing mode are operands, so one `ldr q`
    /// covers what used to be `_voff`, `_sp` and `_x17`: the same word, with
    /// `Rn` coming from the [`Mem`] instead of from the function's name.
    #[test]
    fn the_base_register_is_an_operand_not_a_suffix() {
        fn one(f: impl FnOnce(&mut Vec<u8>)) -> u32 {
            let mut code = Vec::new();
            f(&mut code);
            assert_eq!(code.len(), 4, "aarch64 instructions are fixed-width");
            u32::from_le_bytes(code[..4].try_into().unwrap())
        }
        // ldr q0, [x0, #32] / [sp, #32] / [x17, #32] — one encoder, three bases.
        for base in [xr::X0, xr::SP, xr::X17] {
            let word = one(|c| emit_ldr_q(c, Reg(0), Mem { base, offset: 32 }));
            assert_eq!(word & !(0x1F << 5), 0x3DC0_0800, "same instruction");
            assert_eq!((word >> 5) & 0x1F, u32::from(base.0), "Rn is the base");
        }
        // str q1, [sp, #48]
        assert_eq!(
            one(|c| emit_str_q(
                c,
                Reg(1),
                Mem {
                    base: xr::SP,
                    offset: 48
                }
            )),
            0x3D80_0FE1
        );
    }

    /// An offset past the 12-bit scaled immediate is computed into IP0 first,
    /// in `add`-immediate-sized steps, and the transfer then reads `[x16]`.
    #[test]
    fn a_deep_frame_addresses_through_ip0() {
        let mut code = Vec::new();
        // 65536 = 16 * 4096, one slot past the largest encodable displacement.
        emit_ldr_q(
            &mut code,
            Reg(3),
            Mem {
                base: xr::SP,
                offset: 65536,
            },
        );
        let words: Vec<u32> = code
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect();
        // 65536 = 16 * 4080 + 256, so sixteen full adds plus the remainder.
        assert_eq!(words.len(), 17 + 1, "adds then the load");
        assert_eq!(words[0], 0x9100_0000 | (4080 << 10) | (31 << 5) | 16);
        assert_eq!(words[1], 0x9100_0000 | (4080 << 10) | (16 << 5) | 16);
        assert_eq!(*words.last().unwrap(), 0x3DC0_0000 | (16 << 5) | 3);
    }
}

// =============================================================================
// The NEON `IsaBackend` driver
// =============================================================================

/// The aarch64 half of code generation: the [`IsaBackend`](super::super::IsaBackend)
/// implementation and the constant pool it needs.
///
/// **This file is where aarch64-specific bugs live, and the only place they
/// can.** Emission is a pure function into `Vec<u8>`, so everything here
/// compiles, typechecks and is swept for op coverage on every host — an x86
/// machine computes NEON instruction words perfectly well. Only
/// [`Native`](super::super::Native) decides which backend a build instantiates,
/// and only [`executable`](super::super::executable) needs the matching CPU.
///
/// The consequence worth stating: a change that does not touch an ISA file
/// cannot introduce a platform-specific bug. That is the same bargain `unsafe`
/// makes — confine what cannot be checked, so the rest is checked by
/// construction.
///
/// Dead only in a build that selected a *different* `Native`. The condition
/// mirrors this backend's `Native` alias, so a genuinely unused item in the
/// backend this build actually compiles still trips `dead_code`; an
/// unconditional allow here would hide it from CI's `clippy -D warnings`.
#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
pub(crate) mod driver {
    use super::super::*;
    use super::xr::*;
    use super::{Mem, Xr};
    use crate::error::CompileError;
    use alloc::vec::Vec;

    /// Constant pool: maps f32 bit patterns to pool indices.
    ///
    /// Non-zero, non-FMOV-encodable constants are stored in a data section after
    /// the RET instruction. Each entry is 16 bytes (the f32 splatted 4x to fill
    /// a 128-bit NEON register). During code emission, these constants are loaded
    /// with a single `LDR Qt, [X17, #offset]` instead of the 3-instruction
    /// MOVZ+MOVK+DUP sequence.
    pub(crate) struct ConstPool {
        /// Deduplicated entries: f32 bit patterns in pool order.
        entries: Vec<u32>,
        /// Map from f32 bits → pool index.
        index: alloc::collections::BTreeMap<u32, u16>,
    }
    impl ConstPool {
        /// Create an empty constant pool.
        pub(crate) fn new() -> Self {
            Self {
                entries: Vec::new(),
                index: alloc::collections::BTreeMap::new(),
            }
        }

        /// Insert an f32 into the pool (deduplicating by bit pattern) and return
        /// the byte offset for an `LDR Qt, [X17, #offset]` load.
        ///
        /// Zero and FMOV-encodable constants are NOT filtered here — callers
        /// that want the fast path should check `needs_const_pool` first.
        /// Builtin emitters call this unconditionally because every constant
        /// they use benefits from the pool (they are transcendental coefficients,
        /// never zero or FMOV-encodable).
        pub(crate) fn push_f32(&mut self, val: f32) -> Result<u16, CompileError> {
            let bits = val.to_bits();
            if let Some(&idx) = self.index.get(&bits) {
                return Ok(idx * 16);
            }
            let idx = self.entries.len();
            if idx >= 4096 {
                return Err(CompileError::BudgetExceeded(
                    "constant pool overflow: exceeded 12-bit LDR offset limit (max 4095 entries)",
                ));
            }
            self.entries.push(bits);
            self.index.insert(bits, idx as u16);
            Ok((idx * 16) as u16)
        }

        /// Get the byte offset for a constant, or None if it's not in the pool.
        fn offset_for(&self, val_bits: u32) -> Option<u16> {
            self.index.get(&val_bits).map(|&idx| idx * 16)
        }

        /// Returns true if the pool has any entries.
        fn is_empty(&self) -> bool {
            self.entries.is_empty()
        }
    }
    /// Emit a constant load, using the constant pool when available.
    ///
    /// Falls back to `emit_fmov_imm` for zero and FMOV-encodable values.
    fn emit_const_load(code: &mut Vec<u8>, dst: Reg, val_bits: u32, pool: &ConstPool) {
        if let Some(offset) = pool.offset_for(val_bits) {
            super::emit_ldr_q(
                code,
                dst,
                Mem {
                    base: X17,
                    offset: offset.into(),
                },
            );
        } else {
            super::emit_fmov_imm(code, dst, f32::from_bits(val_bits));
        }
    }
    /// The address of a frame slot. Kernels are leaf functions with no frame
    /// pointer, so every slot — spill or scaffold — is `sp` plus its offset;
    /// naming that here keeps `sp` a fact about the frame instead of a suffix
    /// on the load and store that reach it.
    const fn frame_slot(offset: u32) -> Mem {
        Mem { base: SP, offset }
    }

    /// A pending aarch64 branch: CBZ and B are patched differently.
    pub(crate) enum Aarch64Branch {
        Cbz(usize),
        B(usize),
        /// A `b.hs` awaiting its target — the collapse loop's exit test.
        Hs(super::Cond19),
    }

    /// aarch64 implementation of the shared driver's leaf operations.
    ///
    /// Mechanically wraps the existing aarch64 encoders + constant pool, so the
    /// emitted code is the same as the previous bespoke `compile_from_schedule`.
    /// The aarch64 (NEON) register file.
    ///
    /// AAPCS64 callee-saves the low 64 bits of v8-v15. These kernels are leaf
    /// functions emitted with no prologue that preserves them, so the scratch pool
    /// must steer clear of that range entirely — handing the allocator one of
    /// v8-v15 would silently corrupt whatever the *caller* had live there across
    /// the JIT call.
    ///
    ///   v0-v3:   inputs (X, Y, Z, W)
    ///   v8-v15:  callee-saved, never allocatable
    ///   v4-v7, v16-v31: allocatable scratch — everything else
    const AARCH64_FILE: regalloc::RegisterFile = regalloc::RegisterFile {
        inputs: INPUT_REGS,
        // v4-v7 and v16-v31: twenty of thirty-two. AAPCS64 callee-saves the
        // low 64 bits of v8-v15 and these are leaf kernels with no prologue
        // that preserves them, so v8-v15 stay out; v4-v7 are unused argument
        // registers.
        //
        // v26/v27 are the last two to join: they were `reload`, held out of
        // every kernel's pool for a spilled operand and a spilled destination.
        // v28 came before them, holding the `UMAXV`/`UMINV` the Select
        // short-circuit guards reduce a mask into. Both needs arise at points
        // the schedule contains, so both are reservations the allocator makes
        // on an instruction (`Scratch::reload`, `guard_temps`).
        scratch: regalloc::RegSet::range(16, 16).union(regalloc::RegSet::range(4, 4)),
        // Nothing. v30 is the gather's truncated-index register, a `temps_for`
        // answer since the gathers landed; v29 used to be `UNARY_SCRATCH`,
        // reserved whole-kernel so a reciprocal estimate could borrow it. The
        // select needs none either: `BSL` reads its three operands directly,
        // and `FNEG`/`FABS` are single instructions.
        fixed: &[],
        temps_for: super::temps_for,
        // `UMAXV`/`UMINV` reduce the mask into a vector register before
        // `FMOV` can move it to a general one. It used to be v28, held out of
        // every kernel's pool; it is now a reservation on the instruction the
        // guard is emitted before.
        guard_temps: 1,
        vector_bytes: 16,
    }
    .checked();

    /// The register a guard reduces its mask into.
    ///
    /// `UMAXV`/`UMINV` write a scalar into a vector register, so this tier's
    /// guard needs one that is neither the mask nor anything live — which is
    /// what `RegisterFile::guard_temps` asks the allocator for, and what makes
    /// the two assertions here statements about the allocator rather than
    /// about a hand-picked constant.
    fn guard_scratch(scratch: Option<Reg>, mask_reg: Reg) -> Reg {
        let scratch = scratch
            .expect("aarch64's guard declares `guard_temps: 1`; the allocator owes it a register");
        debug_assert_ne!(scratch, mask_reg, "the reduce would destroy its own input");
        scratch
    }

    pub(crate) struct Aarch64Backend {
        pool: ConstPool,
        adr_patch_pos: usize,
        file: regalloc::RegisterFile,
    }

    impl Aarch64Backend {
        /// The constant pool as emitted so far.
        ///
        /// Test-only, and says so in the type: its one reader is the test
        /// pinning that the pool APPENDS across the two bodies a collapse
        /// compile pushes through one backend. A reset there is the glyph-ink
        /// regression.
        #[cfg(test)]
        pub(crate) fn pool_entries(&self) -> &[u32] {
            &self.pool.entries
        }

        pub(crate) fn new(ctx: EmitCtx) -> Self {
            Self {
                pool: ConstPool::new(),
                adr_patch_pos: 0,
                file: AARCH64_FILE.capped(ctx.max_regs),
            }
        }

        /// Append the constant pool after the final RET and patch the ADR anchor
        /// (upgrading to ADRP+ADD when the pool is out of ADR range). Shared by
        /// the per-batch epilogue and the collapse-loop scaffold.
        fn finish_pool(&mut self, code: &mut Vec<u8>) {
            if self.pool.is_empty() {
                return;
            }
            let adr_pos = self.adr_patch_pos;
            let estimated_offset = (code.len() as i64) - (adr_pos as i64);
            let needs_adrp = estimated_offset >= (1 << 20) - 32;
            if needs_adrp {
                code.splice(adr_pos + 4..adr_pos + 4, [0, 0, 0, 0]);
            }
            while !code.len().is_multiple_of(16) {
                code.push(0);
            }
            let pool_start = code.len();
            for &bits in &self.pool.entries {
                super::emit_pool_entry(code, bits);
            }
            super::patch_adr_or_adrp(code, adr_pos, pool_start, needs_adrp);
        }
    }

    impl IsaBackend for Aarch64Backend {
        type Branch = Aarch64Branch;

        fn register_file(&self) -> regalloc::RegisterFile {
            self.file
        }

        fn begin(&mut self, schedule: &[regalloc::Def]) -> Result<(), CompileError> {
            // Seed by APPENDING into the existing pool, never replacing it: a
            // collapse compile emits two bodies through one backend (the LICM
            // prologue, then the loop body), and the prologue's bytes have the
            // first pool's X17-relative offsets baked in — resetting here left
            // them pointing into the body's rebuilt pool (wrong constants; the
            // macOS glyph-ink regression). `push_f32` dedups, and each compile
            // constructs a fresh backend, so appending is reset-equivalent for
            // single-body compiles.
            for def in schedule {
                if let ScheduledOp::Const(val) = def.op
                    && super::needs_const_pool(val)
                {
                    self.pool.push_f32(val)?;
                }
            }
            // Builtins add up to ~60 polynomial coefficients during emission; bail
            // if the expression constants + headroom would exceed the 12-bit LDR
            // offset limit.
            const BUILTIN_HEADROOM: usize = 128;
            if self.pool.entries.len() + BUILTIN_HEADROOM > 4095 {
                return Err(CompileError::BudgetExceeded(
                    "expression too large: constant pool would exceed 12-bit LDR offset limit",
                ));
            }
            Ok(())
        }

        fn emit_plan(
            &mut self,
            code: &mut Vec<u8>,
            plan: &InstructionPlan,
        ) -> Result<(), CompileError> {
            emit_instruction_plan(code, plan, &mut self.pool)
        }

        fn emit_mov(&mut self, code: &mut Vec<u8>, dst: Reg, src: Reg) {
            super::emit_mov(code, dst, src);
        }

        fn emit_store(
            &mut self,
            code: &mut Vec<u8>,
            src: Reg,
            offset: u32,
        ) -> Result<(), CompileError> {
            super::emit_str_q(code, src, frame_slot(offset));
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
                    emit_const_load(code, target, bits, &self.pool);
                    target
                }
                Loc::Slot(slot) => {
                    super::emit_ldr_q(code, target, frame_slot(slot.offset()));
                    target
                }
            }
        }

        /// `scratch` is this instruction's own reservation, live for these two
        /// instructions only — the allocator makes it because this backend's
        /// `guard_temps` asks for one.
        fn emit_skip_if_all_false(
            &mut self,
            code: &mut Vec<u8>,
            mask_reg: Reg,
            scratch: Option<Reg>,
        ) -> Aarch64Branch {
            let scratch = guard_scratch(scratch, mask_reg);
            super::emit_umaxv(code, scratch, mask_reg); // max lane; 0 => all-false
            super::emit_fmov_to_gp(code, scratch);
            Aarch64Branch::Cbz(super::emit_cbz_w16(code))
        }

        fn emit_skip_if_all_true(
            &mut self,
            code: &mut Vec<u8>,
            mask_reg: Reg,
            scratch: Option<Reg>,
        ) -> Aarch64Branch {
            let scratch = guard_scratch(scratch, mask_reg);
            super::emit_uminv(code, scratch, mask_reg); // min lane; 0xFFFFFFFF => all-true
            super::emit_fmov_to_gp(code, scratch);
            // MVN W16, W16 -> 0 iff all-true, which the cbz below tests.
            super::mvn_w(code, X16, X16);
            Aarch64Branch::Cbz(super::emit_cbz_w16(code))
        }

        fn emit_jump(&mut self, code: &mut Vec<u8>) -> Aarch64Branch {
            Aarch64Branch::B(super::emit_b(code))
        }

        fn patch_branch(&mut self, code: &mut Vec<u8>, branch: Aarch64Branch, target: usize) {
            match branch {
                Aarch64Branch::Cbz(p) => super::patch_cbz_cbnz(code, p, target),
                Aarch64Branch::B(p) => super::patch_b(code, p, target),
                Aarch64Branch::Hs(c) => c.patch(code, target),
            }
        }

        // AAPCS64: x0 = ctx (read-only in the body's gathers), x1 = out,
        // x2 = groups, x3 = rows, x4 = row-skip bytes, v0..3 = x0/y0/z/w.
        // Loop registers: x5 = batch counter, x6 = row counter; the body's
        // scratch GPRs are x9-x11 (gather), w16 (branch tests), x17 (pool
        // anchor) — all disjoint. The bounds arrive in registers the body never
        // touches, so `latch_bounds` has nothing to do.

        fn frame_alloc(&mut self, code: &mut Vec<u8>, bytes: u32) {
            super::emit_sub_sp(code, bytes);
        }

        fn frame_free(&mut self, code: &mut Vec<u8>, bytes: u32) {
            super::emit_add_sp(code, bytes);
        }

        /// The prologue's and body's constant loads are X17-relative, so the
        /// anchor has to be inside the emitted function, after the frame.
        fn scaffold_anchor(&mut self, code: &mut Vec<u8>) {
            self.adr_patch_pos = super::emit_adr_x17_placeholder(code);
        }

        fn scaffold_finish(&mut self, code: &mut Vec<u8>) {
            self.finish_pool(code);
        }

        fn slot_store(&mut self, code: &mut Vec<u8>, src: Reg, offset: u32) {
            super::emit_str_q(code, src, frame_slot(offset));
        }

        fn slot_load(&mut self, code: &mut Vec<u8>, dst: Reg, offset: u32) {
            super::emit_ldr_q(code, dst, frame_slot(offset));
        }

        fn counter_clear(&mut self, code: &mut Vec<u8>, counter: Counter) {
            super::movz(code, counter_reg(counter), 0);
        }

        fn counter_step(&mut self, code: &mut Vec<u8>, counter: Counter) {
            let r = counter_reg(counter);
            super::add(code, r, r, super::Imm12(1));
        }

        fn branch_if_counter_done(
            &mut self,
            code: &mut Vec<u8>,
            counter: Counter,
        ) -> Aarch64Branch {
            super::cmp(code, counter_reg(counter), bound_reg(counter));
            Aarch64Branch::Hs(super::b_hs(code))
        }

        fn store_result(&mut self, code: &mut Vec<u8>, src: Reg) {
            super::emit_str_q(
                code,
                src,
                Mem {
                    base: X1,
                    offset: 0,
                },
            );
        }

        fn advance_out(&mut self, code: &mut Vec<u8>, step: OutStep) {
            match step {
                OutStep::Batch => {
                    super::add(code, X1, X1, super::Imm12(self.file.vector_bytes as u16));
                }
                OutStep::RowSkip => super::add(code, X1, X1, X4),
            }
        }

        fn add_scalar(&mut self, code: &mut Vec<u8>, dst: Reg, scratch: Reg, scalar: f32) {
            super::emit_fmov_imm(code, scratch, scalar);
            super::emit_fadd(code, dst, dst, scratch);
        }

        fn emit_ret(&mut self, code: &mut Vec<u8>) {
            super::ret(code);
        }
    }

    /// The register each loop counter lives in.
    const fn counter_reg(counter: Counter) -> Xr {
        match counter {
            Counter::Batch => X5,
            Counter::Row => X6,
        }
    }

    /// The register each counter is compared against — both arrive as
    /// arguments and stay put, since nothing in the body writes them.
    const fn bound_reg(counter: Counter) -> Xr {
        match counter {
            Counter::Batch => X2,
            Counter::Row => X3,
        }
    }
    /// Emit machine code for a resolved instruction plan.
    ///
    /// This is a DETERMINISTIC DISPATCH: given a plan, emit the exact
    /// instructions. No decisions are made here — all decisions were
    /// made by resolve_operands.
    fn emit_instruction_plan(
        code: &mut Vec<u8>,
        plan: &InstructionPlan,
        pool: &mut ConstPool,
    ) -> Result<(), CompileError> {
        use super::*;

        // 1. Emit reloads (from stack or rematerialized constants)
        for reload in &plan.reloads {
            match reload {
                Reload::FromStack { target, slot } => {
                    emit_ldr_q(code, *target, frame_slot(slot.offset()));
                }
                Reload::Const { target, val_bits } => {
                    emit_const_load(code, *target, *val_bits, pool);
                }
            }
        }

        // 2. Emit setup MOV (for FMLA accumulator or BSL mask)
        if let Some((dst, src)) = plan.setup_mov {
            super::emit_mov(code, dst, src);
        }

        // 3. Emit main op
        match &plan.op {
            ResolvedOp::Nop => {}
            ResolvedOp::LoadConst { dst, val_bits } => {
                emit_const_load(code, *dst, *val_bits, pool);
            }
            ResolvedOp::Unary { op, dst, src } => {
                emit_unary(code, *op, *dst, *src, plan.scratch.temp(0));
            }
            ResolvedOp::ShiftImm {
                op,
                dst,
                src,
                amount,
            } => {
                super::emit_shift_imm(code, *op, *dst, *src, *amount);
            }
            ResolvedOp::Gather { dst, idx, slot } => {
                // dst = buffer[slot][idx], via four scalar loads (NEON has no
                // native gather). The context pointer (array of buffer base
                // pointers) is caller-provided in x0 per AAPCS64 — disjoint from
                // the coordinate vectors in v0..3 and never touched by the
                // arithmetic/const emit, so it survives to here.
                // v30 is declared in `AARCH64_FILE.fixed`, so
                // `RegisterFile::checked` proves it misses the pool, the reload
                // pair and the guard scratch (v28) rather than a comment claiming
                // it; x9-x11 are caller-saved GPR scratch clear of the branch
                // guard (w16) and the const-pool anchor (x17).
                let idx_int = crate::emit::declared_temp(plan.scratch.temp(0));
                const BASE_GPR: Xr = Xr(9);
                const IDX_GPR: Xr = Xr(10);
                const VAL_GPR: Xr = Xr(11);
                /// Bytes per pointer in the context array.
                const PTR_BYTES: u32 = 8;
                emit_fcvtzs(code, idx_int, *idx); // float idx -> int32 lanes
                emit_ldr_x(
                    code,
                    BASE_GPR,
                    Mem {
                        base: X0,
                        offset: u32::from(*slot) * PTR_BYTES,
                    },
                );
                emit_gather(
                    code,
                    *dst,
                    idx_int,
                    super::GatherGprs {
                        base: BASE_GPR,
                        idx: IDX_GPR,
                        val: VAL_GPR,
                    },
                );
            }
            ResolvedOp::Uniform { dst, load } => {
                super::emit_uniform_load(code, *dst, *load);
            }
            ResolvedOp::Binary {
                op,
                dst,
                left,
                right,
            } => {
                // Every transcendental is expanded to arithmetic by
                // `expand_transcendentals` before codegen, so only primitives
                // reach here.
                emit_binary(code, *op, *dst, *left, *right);
            }
            ResolvedOp::FusedMulAdd { dst, a, b } => {
                // setup_mov already placed c into dst
                emit_fmla(code, *dst, *a, *b);
            }
            ResolvedOp::DecomposedMulAdd {
                dst,
                a,
                b,
                c,
                c_deferred,
            } => {
                // FMUL(dst, a, b) — consumes a and b (loaded upfront).
                emit_fmul(code, *dst, *a, *b);
                // Reload c after FMUL (c may reuse tmp_op which held b).
                emit_deferred(code, *c, c_deferred.as_ref(), pool);
                // FADD(dst, dst, c)
                emit_fadd(code, *dst, *dst, *c);
            }
            ResolvedOp::Select {
                dst,
                if_true,
                if_false,
            } => {
                // setup_mov already placed mask into dst
                emit_bsl(code, *dst, *if_true, *if_false);
            }
        }

        Ok(())
    }
    /// Emit a deferred reload: either from stack or rematerialized constant.
    fn emit_deferred(
        code: &mut Vec<u8>,
        target: Reg,
        deferred: Option<&DeferredReload>,
        pool: &ConstPool,
    ) {
        match deferred {
            Some(DeferredReload::FromStack(slot)) => {
                super::emit_ldr_q(code, target, frame_slot(slot.offset()));
            }
            Some(DeferredReload::Const(val_bits)) => {
                emit_const_load(code, target, *val_bits, pool);
            }
            None => {}
        }
    }
}

// =============================================================================
// General-purpose registers
// =============================================================================

/// The aarch64 general register file (`x0`–`x30`, plus the zero register).
///
/// A distinct type from [`Reg`], which names the *vector* file `v0`–`v31`.
/// They are different files that share a numbering, so `Xr(1)` is `x1` and
/// `Reg(1)` is `v1`, and neither can be passed where the other belongs.
///
/// A SIMD language barely touches these: loop counters, the output pointer,
/// the row stride, the scalar half of a gather, and the base of every address
/// a load or store names. That is the whole list, which is why this stays a
/// vocabulary rather than an assembler.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Xr(pub u8);

/// AAPCS64 registers the emitted kernels use.
pub mod xr {
    use super::Xr;

    /// 1st argument: the context pointer — the array of bound buffer bases a
    /// gather reads its slot from. Read-only for the whole kernel.
    pub const X0: Xr = Xr(0);
    /// 2nd argument: the output pointer, advanced per batch and per row.
    pub const X1: Xr = Xr(1);
    /// 3rd argument: group count (the inner bound).
    pub const X2: Xr = Xr(2);
    /// 4th argument: row count (the outer bound).
    pub const X3: Xr = Xr(3);
    /// 5th argument: row-skip in bytes.
    pub const X4: Xr = Xr(4);
    /// Inner (batch) loop counter.
    pub const X5: Xr = Xr(5);
    /// Outer (row) loop counter.
    pub const X6: Xr = Xr(6);
    /// IP0, the intra-procedure scratch. The encoder borrows it to hold an
    /// address whose displacement is past a load's 12-bit immediate.
    pub const X16: Xr = Xr(16);
    /// IP1, the intra-procedure scratch holding the constant-pool anchor.
    pub const X17: Xr = Xr(17);
    /// The zero register in the positions where `xzr` is meant.
    pub const XZR: Xr = Xr(31);
    /// The stack pointer — spill slots are addressed from it.
    ///
    /// Encoded as register 31, the number [`XZR`] also uses. Which of the two
    /// `31` means is decided by the instruction (load/store and `add` read it
    /// as `sp`; most data-processing reads it as `xzr`), so the two constants
    /// exist to say at the call site which one was meant.
    pub const SP: Xr = Xr(31);
}

/// A 12-bit unsigned immediate, the width `add`'s immediate form encodes.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Imm12(pub u16);

/// `movz dst, #imm16` — also how `mov dst, xzr` is spelled, as `movz dst, #0`.
#[inline(always)]
pub fn movz(code: &mut Vec<u8>, dst: Xr, imm: u16) {
    emit32(code, 0xD280_0000 | ((imm as u32) << 5) | dst.0 as u32);
}

/// `cmp lhs, rhs` — `subs xzr, lhs, rhs`, setting the flags [`b_hs`] reads.
#[inline(always)]
pub fn cmp(code: &mut Vec<u8>, lhs: Xr, rhs: Xr) {
    emit32(
        code,
        0xEB00_0000 | ((rhs.0 as u32) << 16) | ((lhs.0 as u32) << 5) | 31,
    );
}

/// What an [`add`] can add: another register, or a 12-bit immediate.
///
/// As on x86, the operand's *type* selects the encoding, so the mnemonic stays
/// one name instead of splitting into `add_reg` / `add_imm`.
pub trait AddOperand {
    /// Emit `add dst, src, self`.
    fn add_into(self, code: &mut Vec<u8>, dst: Xr, src: Xr);
}

impl AddOperand for Xr {
    #[inline(always)]
    fn add_into(self, code: &mut Vec<u8>, dst: Xr, src: Xr) {
        emit32(
            code,
            0x8B00_0000 | ((self.0 as u32) << 16) | ((src.0 as u32) << 5) | dst.0 as u32,
        );
    }
}

impl AddOperand for Imm12 {
    #[inline(always)]
    fn add_into(self, code: &mut Vec<u8>, dst: Xr, src: Xr) {
        emit32(
            code,
            0x9100_0000 | ((self.0 as u32) << 10) | ((src.0 as u32) << 5) | dst.0 as u32,
        );
    }
}

/// `add dst, src, operand`
#[inline(always)]
pub fn add(code: &mut Vec<u8>, dst: Xr, src: Xr, operand: impl AddOperand) {
    operand.add_into(code, dst, src);
}

/// `mvn w<dst>, w<src>` — bitwise NOT of a 32-bit general register.
///
/// `ORN Wd, WZR, Wm`; the guard path uses it to turn "all lanes set" into
/// zero so a following `cbz` tests it.
#[inline(always)]
pub fn mvn_w(code: &mut Vec<u8>, dst: Xr, src: Xr) {
    emit32(code, 0x2A20_03E0 | ((src.0 as u32) << 16) | dst.0 as u32);
}

/// `ret`
#[inline(always)]
pub fn ret(code: &mut Vec<u8>) {
    emit32(code, 0xD65F_03C0);
}

/// A conditional branch whose 19-bit displacement is not filled in yet.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[must_use = "an unpatched branch falls through to itself"]
pub struct Cond19(usize);

impl Cond19 {
    /// Point the branch at `target`, a byte offset into the same buffer.
    #[inline(always)]
    pub fn patch(self, code: &mut [u8], target: usize) {
        let imm19 = (((target - self.0) / 4) as u32) & 0x7FFFF;
        let word = 0x5400_0000 | (imm19 << 5) | 0x2; // cond = HS
        code[self.0..self.0 + 4].copy_from_slice(&word.to_le_bytes());
    }
}

/// `b.hs` — taken when the previous [`cmp`] found `lhs >= rhs` unsigned.
#[inline(always)]
pub fn b_hs(code: &mut Vec<u8>) -> Cond19 {
    let at = code.len();
    emit32(code, 0x5400_0002);
    Cond19(at)
}

/// `b target` — an unconditional branch to an already-known offset.
///
/// Unlike the x86 counterpart this needs no fixup token: every use in the
/// scaffold jumps *backwards* to a label already emitted.
#[inline(always)]
pub fn b(code: &mut Vec<u8>, target: usize) {
    let rel = ((target as i64 - code.len() as i64) / 4) as i32;
    emit32(code, 0x1400_0000 | ((rel as u32) & 0x03FF_FFFF));
}

#[cfg(test)]
mod xr_tests {
    use super::xr::*;
    use super::*;

    fn word(f: impl FnOnce(&mut Vec<u8>)) -> u32 {
        let mut c = Vec::new();
        f(&mut c);
        assert_eq!(c.len(), 4, "aarch64 instructions are fixed-width");
        u32::from_le_bytes([c[0], c[1], c[2], c[3]])
    }

    /// Each encoding checked against the ARM ARM's form for that mnemonic.
    /// These are the exact words the collapse-loop scaffold used to spell
    /// inline, which is what makes the replacement provably byte-identical.
    #[test]
    fn encodings_match_the_manual() {
        // MOVZ Xd, #imm16 — `mov xN, xzr` is `movz xN, #0`.
        assert_eq!(word(|c| movz(c, X6, 0)), 0xD280_0006);
        assert_eq!(word(|c| movz(c, X5, 0)), 0xD280_0005);
        // SUBS XZR, Xn, Xm
        assert_eq!(word(|c| cmp(c, X6, X3)), 0xEB03_00DF);
        assert_eq!(word(|c| cmp(c, X5, X2)), 0xEB02_00BF);
        // ADD Xd, Xn, #imm12
        assert_eq!(word(|c| add(c, X1, X1, Imm12(16))), 0x9100_4021);
        assert_eq!(word(|c| add(c, X5, X5, Imm12(1))), 0x9100_04A5);
        assert_eq!(word(|c| add(c, X6, X6, Imm12(1))), 0x9100_04C6);
        // ADD Xd, Xn, Xm
        assert_eq!(word(|c| add(c, X1, X1, X4)), 0x8B04_0021);
        // STR Qt, [Xn]
        assert_eq!(
            word(|c| emit_str_q(
                c,
                Reg(0),
                Mem {
                    base: X1,
                    offset: 0
                }
            )),
            0x3D80_0020
        );
        // RET
        assert_eq!(word(ret), 0xD65F_03C0);
    }

    /// The immediate and register forms of `add` are different instructions
    /// reached through one name; the operand type is what chooses.
    #[test]
    fn the_operand_type_selects_the_add_encoding() {
        assert_eq!(
            word(|c| add(c, X1, X1, Imm12(16))) >> 24,
            0x91,
            "immediate form"
        );
        assert_eq!(word(|c| add(c, X1, X1, X4)) >> 24, 0x8B, "register form");
    }

    /// A conditional branch is emitted as a placeholder and patched to a
    /// forward target; the displacement counts instructions, not bytes.
    #[test]
    fn conditional_branches_patch_forward_in_instructions() {
        let mut c = Vec::new();
        let br = b_hs(&mut c);
        c.resize(24, 0); // three more instructions
        br.patch(&mut c, 24);
        let w = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
        assert_eq!((w >> 5) & 0x7FFFF, 6, "24 bytes ahead is 6 instructions");
        assert_eq!(w & 0xF, 0x2, "cond = HS");
    }

    /// An unconditional backward branch encodes a negative instruction count.
    #[test]
    fn unconditional_branches_go_backwards() {
        let mut c = vec![0u8; 16];
        b(&mut c, 0);
        let w = u32::from_le_bytes([c[16], c[17], c[18], c[19]]);
        assert_eq!(w >> 26, 0x05, "B opcode");
        // -4 instructions, in 26-bit two's complement.
        assert_eq!(w & 0x03FF_FFFF, (-4i32 as u32) & 0x03FF_FFFF);
    }

    /// `Xr` and `Reg` name different files; the same index is a different
    /// register in each, which is why they are different types.
    #[test]
    fn the_two_register_files_are_not_interchangeable() {
        assert_eq!(X1.0, Reg(1).0);
        // `emit_str_q` takes both, in their own positions: the vector operand
        // lands in Rt and the address's base in Rn, so swapping them cannot
        // typecheck. `emit_ldr_x` is the mirror — an `Xr` destination, because
        // it is a load on the general file, not the vector one.
        assert_eq!(
            word(|c| emit_str_q(
                c,
                Reg(3),
                Mem {
                    base: X1,
                    offset: 0
                }
            )),
            0x3D80_0023
        );
        assert_eq!(
            word(|c| emit_ldr_x(
                c,
                Xr(3),
                Mem {
                    base: X1,
                    offset: 0
                }
            )),
            0xF940_0023
        );
    }
}
