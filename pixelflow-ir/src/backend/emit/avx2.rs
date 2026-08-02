//! x86-64 AVX2 (VEX.256) JIT encoder — 256-bit, 8-lane `ymm` kernels.
//!
//! The middle width between the SSE2 leaf encoders (`x86_64.rs`, 128-bit) and
//! the AVX-512 EVEX encoders (`avx512.rs`, 512-bit). Register numbering is
//! identical to SSE2 (ymm0-15, no extended file — AVX2 has no REX2/EVEX), so
//! this backend reuses the exact register-role layout `X86Backend` already
//! established (inputs 0-3, allocatable 4-9, fixed scratch 10, reload 11-12,
//! builtin scratch 10/13-15): only the instruction *encoding* changes.
//!
//! Unlike legacy SSE2, VEX is 3-operand and non-destructive — same property
//! AVX-512's EVEX has — so there is no two-operand hazard to route around
//! (contrast `emit_binary_safe` in `mod.rs`, needed only by the SSE2 path).
//! Comparisons are simpler here than on AVX-512: `vcmpps` writes an ordinary
//! all-ones/all-zeros `ymm` directly (no k-register, no mask-to-vector
//! conversion) — the same representation NEON and SSE2 already use.
//!
//! Spills use a real stack frame, not the red zone: mirrors `avx512.rs`'s
//! reasoning (a `ymm` slot is 32 bytes; keeping the red-zone arithmetic exact
//! for two different slot widths is not worth it for a bit of frame reuse on
//! tiny kernels).
//!
//! Gather has no direct AVX2 hardware analogue reused here: `vgatherdps`'s
//! VSIB + vector-mask-with-clearing semantics are a bigger lift than this
//! backend's scope warrants, so — like `X86Backend` — a gather is assembled
//! from scalar loads via the existing 128-bit lane-insert sequence
//! (`x86_64::emit_gather_scalar`), run once per 128-bit half and combined
//! with `vinsertf128`.

use super::x86_64;
use super::{Reg, unimplemented_op};
use crate::OpKind;
use alloc::vec::Vec;

// =============================================================================
// VEX.256 encoder
// =============================================================================

/// Which legacy-prefix byte the VEX prefix implies (the `pp` field).
#[derive(Clone, Copy)]
enum Pp {
    /// No implied prefix.
    None = 0,
    /// `66`
    P66 = 1,
    /// `F3`
    F3 = 2,
}

/// Which opcode map the instruction lives in (the field Intel calls
/// `mmmmm` — a map *selector*, nothing more).
#[derive(Clone, Copy)]
enum Map {
    /// `0F`
    M0F = 1,
    /// `0F38`
    M0F38 = 2,
    /// `0F3A`
    M0F3A = 3,
}

/// The identity of one VEX-256 instruction: opcode map, implied legacy
/// prefix, W bit, opcode byte. This quadruple is *which instruction* — it is
/// constant per mnemonic, so each mnemonic below states it exactly once and
/// the operand form (`rrr`/`imm`/`rm_rsp`) supplies the per-call parts.
#[derive(Clone, Copy)]
struct Vex {
    map: Map,
    pp: Pp,
    w: bool,
    opcode: u8,
}

/// Sentinel for an unused VEX.vvvv source (2-operand forms): index 0 inverts
/// to `1111`, the required "unused" encoding.
const UNUSED_VVVV: u8 = 0;

impl Vex {
    const fn new(map: Map, pp: Pp, opcode: u8) -> Self {
        Self {
            map,
            pp,
            w: false,
            opcode,
        }
    }
    /// Map `0F`, no prefix — the packed-single arithmetic family.
    const fn m0f(opcode: u8) -> Self {
        Self::new(Map::M0F, Pp::None, opcode)
    }
    /// Map `0F`, `66` — the integer-domain family.
    const fn m0f_66(opcode: u8) -> Self {
        Self::new(Map::M0F, Pp::P66, opcode)
    }
    /// Map `0F`, `F3`.
    const fn m0f_f3(opcode: u8) -> Self {
        Self::new(Map::M0F, Pp::F3, opcode)
    }
    /// Map `0F38`, `66`.
    const fn m0f38_66(opcode: u8) -> Self {
        Self::new(Map::M0F38, Pp::P66, opcode)
    }
    /// Map `0F3A`, `66` — the imm8 family (round, insert/extract).
    const fn m0f3a_66(opcode: u8) -> Self {
        Self::new(Map::M0F3A, Pp::P66, opcode)
    }

    /// Attach an imm8 (`vcmpps` predicate, rounding mode, shift count, lane
    /// index); the returned value emits it after the instruction.
    const fn imm(self, imm: u8) -> VexImm {
        VexImm { vex: self, imm }
    }

    /// Register-register-register form: `op dst, vvvv, rm`.
    fn rrr(self, code: &mut Vec<u8>, dst: u8, vvvv: u8, rm: u8) {
        let rbit = if dst >= 8 { 0x00 } else { 0x80 };
        let xbit = 0x40;
        let bbit = if rm >= 8 { 0x00 } else { 0x20 };
        code.push(0xC4);
        code.push(rbit | xbit | bbit | self.map as u8);
        code.push(((self.w as u8) << 7) | ((!vvvv & 0xF) << 3) | (1 << 2) | self.pp as u8); // L=1
        code.push(self.opcode);
        code.push(0xC0 | ((dst & 7) << 3) | (rm & 7));
    }

    /// `[rsp + disp32]` memory-operand form (spill loads/stores).
    fn rm_rsp(self, code: &mut Vec<u8>, reg: u8, disp: i32) {
        self.rm_rsp_prefix(code, reg);
        code.push(0x84 | ((reg & 7) << 3)); // mod=10, reg=reg, rm=100 (SIB)
        code.push(0x24); // SIB: base=rsp, no index
        code.extend_from_slice(&disp.to_le_bytes());
    }

    /// `[rsp + disp8]` memory-operand form (red-zone constant broadcast).
    fn rm_rsp8(self, code: &mut Vec<u8>, reg: u8, disp: i8) {
        self.rm_rsp_prefix(code, reg);
        code.push(0x44 | ((reg & 7) << 3)); // mod=01, reg=reg, rm=100 (SIB)
        code.push(0x24);
        code.push(disp as u8);
    }

    /// `[base64]` memory-operand form (mod=00 direct). `base` must not be
    /// rsp/rbp/r12/r13, whose mod=00 encodings mean SIB/RIP instead.
    fn rm_base(self, code: &mut Vec<u8>, reg: u8, base: u8) {
        debug_assert!(
            base & 7 != 4 && base & 7 != 5,
            "Vex::rm_base: base must not be rsp/rbp/r12/r13"
        );
        let rbit = if reg >= 8 { 0x00 } else { 0x80 };
        let bbit = if base >= 8 { 0x00 } else { 0x20 };
        code.push(0xC4);
        code.push(rbit | 0x40 | bbit | self.map as u8); // X=1 (unused)
        code.push(((self.w as u8) << 7) | (0xF << 3) | (1 << 2) | self.pp as u8);
        code.push(self.opcode);
        code.push(((reg & 7) << 3) | (base & 7)); // mod=00
    }

    /// Shared VEX prefix + opcode for the rsp-based memory forms.
    fn rm_rsp_prefix(self, code: &mut Vec<u8>, reg: u8) {
        let rbit = if reg >= 8 { 0x00 } else { 0x80 };
        code.push(0xC4);
        code.push(rbit | 0x40 | 0x20 | self.map as u8); // X=1 (unused), B=1 (rsp < 8)
        code.push(((self.w as u8) << 7) | (0xF << 3) | (1 << 2) | self.pp as u8); // vvvv unused, L=1
        code.push(self.opcode);
    }
}

/// A [`Vex`] instruction carrying its imm8.
#[derive(Clone, Copy)]
struct VexImm {
    vex: Vex,
    imm: u8,
}

impl VexImm {
    /// Register form with the imm8 appended.
    fn rrr(self, code: &mut Vec<u8>, dst: u8, vvvv: u8, rm: u8) {
        self.vex.rrr(code, dst, vvvv, rm);
        code.push(self.imm);
    }
}

// --- packed-single arithmetic (0F, no prefix, W0) ---
fn vaddps(c: &mut Vec<u8>, d: u8, s1: u8, s2: u8) {
    Vex::m0f(0x58).rrr(c, d, s1, s2);
}
fn vsubps(c: &mut Vec<u8>, d: u8, s1: u8, s2: u8) {
    Vex::m0f(0x5C).rrr(c, d, s1, s2);
}
fn vmulps(c: &mut Vec<u8>, d: u8, s1: u8, s2: u8) {
    Vex::m0f(0x59).rrr(c, d, s1, s2);
}
fn vdivps(c: &mut Vec<u8>, d: u8, s1: u8, s2: u8) {
    Vex::m0f(0x5E).rrr(c, d, s1, s2);
}
fn vminps(c: &mut Vec<u8>, d: u8, s1: u8, s2: u8) {
    Vex::m0f(0x5D).rrr(c, d, s1, s2);
}
fn vmaxps(c: &mut Vec<u8>, d: u8, s1: u8, s2: u8) {
    Vex::m0f(0x5F).rrr(c, d, s1, s2);
}
fn vsqrtps(c: &mut Vec<u8>, d: u8, s: u8) {
    Vex::m0f(0x51).rrr(c, d, UNUSED_VVVV, s);
}
fn vrsqrtps(c: &mut Vec<u8>, d: u8, s: u8) {
    Vex::m0f(0x52).rrr(c, d, UNUSED_VVVV, s);
}
fn vrcpps(c: &mut Vec<u8>, d: u8, s: u8) {
    Vex::m0f(0x53).rrr(c, d, UNUSED_VVVV, s);
}

// --- bitwise (0F, no prefix, W0) ---
fn vandps(c: &mut Vec<u8>, d: u8, s1: u8, s2: u8) {
    Vex::m0f(0x54).rrr(c, d, s1, s2);
}
fn vandnps(c: &mut Vec<u8>, d: u8, s1: u8, s2: u8) {
    Vex::m0f(0x55).rrr(c, d, s1, s2);
}
fn vorps(c: &mut Vec<u8>, d: u8, s1: u8, s2: u8) {
    Vex::m0f(0x56).rrr(c, d, s1, s2);
}
fn vxorps(c: &mut Vec<u8>, d: u8, s1: u8, s2: u8) {
    Vex::m0f(0x57).rrr(c, d, s1, s2);
}

// --- comparisons (0F, no prefix, W0; imm8 predicate) ---
const CMP_EQ: u8 = 0;
const CMP_LT: u8 = 1;
const CMP_LE: u8 = 2;
const CMP_NEQ: u8 = 4;
const CMP_GE: u8 = 5;
const CMP_NLE: u8 = 6; // > (unordered-safe "not less-or-equal")

fn vcmpps(c: &mut Vec<u8>, d: u8, s1: u8, s2: u8, pred: u8) {
    Vex::m0f(0xC2).imm(pred).rrr(c, d, s1, s2);
}

fn cmp_pred(op: OpKind) -> Option<u8> {
    Some(match op {
        OpKind::Eq => CMP_EQ,
        OpKind::Ne => CMP_NEQ,
        OpKind::Lt => CMP_LT,
        OpKind::Le => CMP_LE,
        OpKind::Gt => CMP_NLE,
        OpKind::Ge => CMP_GE,
        _ => return None,
    })
}

/// Whether `op` is a comparison handled by [`emit_binary`].
#[must_use]
pub fn is_compare(op: OpKind) -> bool {
    cmp_pred(op).is_some()
}

// --- rounding (0F3A, 66 prefix, W0; imm8) ---
fn vroundps(c: &mut Vec<u8>, d: u8, s: u8, imm: u8) {
    Vex::m0f3a_66(0x08).imm(imm).rrr(c, d, UNUSED_VVVV, s);
}

// --- int/float convert (0F, W0) ---
fn vcvttps2dq(c: &mut Vec<u8>, d: u8, s: u8) {
    Vex::m0f_f3(0x5B).rrr(c, d, UNUSED_VVVV, s); // F3 prefix
}
fn vcvtdq2ps(c: &mut Vec<u8>, d: u8, s: u8) {
    Vex::m0f(0x5B).rrr(c, d, UNUSED_VVVV, s); // no prefix
}

// --- integer-domain (66 prefix, 0F, W0) ---
fn vpaddd(c: &mut Vec<u8>, d: u8, s1: u8, s2: u8) {
    Vex::m0f_66(0xFE).rrr(c, d, s1, s2);
}
fn vpslld_imm(c: &mut Vec<u8>, d: u8, s: u8, imm: u8) {
    Vex::m0f_66(0x72).imm(imm).rrr(c, 6, d, s); // /6, dst=vvvv, src=rm
}
fn vpsrld_imm(c: &mut Vec<u8>, d: u8, s: u8, imm: u8) {
    Vex::m0f_66(0x72).imm(imm).rrr(c, 2, d, s); // /2
}

// --- lane insert/extract between 256-bit and 128-bit (0F3A, 66 prefix, W0) ---
/// `vinsertf128 ymmDST, ymmSRC1, xmmSRC2, imm8[0]` — copy `src1`, then place
/// `src2` into the low (`imm=0`) or high (`imm=1`) 128 bits.
fn vinsertf128(c: &mut Vec<u8>, d: u8, s1: u8, s2: u8, imm: u8) {
    Vex::m0f3a_66(0x18).imm(imm).rrr(c, d, s1, s2);
}
/// `vextractf128 xmmDST, ymmSRC, imm8[0]` — extract the low (`imm=0`) or high
/// (`imm=1`) 128 bits of `src` into `dst`.
fn vextractf128(c: &mut Vec<u8>, d: u8, s: u8, imm: u8) {
    // VEX.256.66.0F3A.W0 19 /r ib — note dst is the ModRM.rm operand here
    // (the reverse of the usual direction: register source, register/mem dest).
    Vex::m0f3a_66(0x19).imm(imm).rrr(c, s, UNUSED_VVVV, d);
}

/// `vmovaps ymmDST, ymmSRC` — register copy.
pub fn emit_mov(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    if dst.0 == src.0 {
        return;
    }
    Vex::m0f(0x28).rrr(code, dst.0, UNUSED_VVVV, src.0);
}

/// `vmovups ymmDST, [rsp+disp32]` — 256-bit reload.
pub fn emit_load_rsp(code: &mut Vec<u8>, dst: Reg, disp: i32) {
    Vex::m0f(0x10).rm_rsp(code, dst.0, disp);
}

/// `vmovups [rsp+disp32], ymmSRC` — 256-bit spill store.
pub fn emit_store_rsp(code: &mut Vec<u8>, src: Reg, disp: i32) {
    Vex::m0f(0x11).rm_rsp(code, src.0, disp);
}

/// `vmovups [base64], ymmSRC` — unaligned 256-bit store to a GP-register base.
pub fn emit_store_base(code: &mut Vec<u8>, src: Reg, base_gpr: u8) {
    Vex::m0f(0x11).rm_base(code, src.0, base_gpr);
}

/// Broadcast an f32 constant to all 8 lanes of `dst` via the stack (red zone,
/// `[rsp-4]`; never affected by a spill frame, which lives at `[rsp..rsp+N)`).
pub fn emit_const(code: &mut Vec<u8>, dst: Reg, val: f32) {
    let bits = val.to_bits();
    if bits == 0 {
        vxorps(code, dst.0, dst.0, dst.0);
        return;
    }
    // mov dword [rsp-4], imm32
    code.extend_from_slice(&[0xC7, 0x44, 0x24, 0xFC]);
    code.extend_from_slice(&bits.to_le_bytes());
    // vbroadcastss ymm, [rsp-4]  (VEX.256.66.0F38.W0 18 /r)
    Vex::m0f38_66(0x18).rm_rsp8(code, dst.0, -4);
}

// =============================================================================
// Stack frame (real frame; a ymm spill is 32 bytes)
// =============================================================================

pub fn emit_sub_rsp(code: &mut Vec<u8>, size: u32) {
    code.extend_from_slice(&[0x48, 0x81, 0xEC]);
    code.extend_from_slice(&size.to_le_bytes());
}
pub fn emit_add_rsp(code: &mut Vec<u8>, size: u32) {
    code.extend_from_slice(&[0x48, 0x81, 0xC4]);
    code.extend_from_slice(&size.to_le_bytes());
}
pub fn emit_ret(code: &mut Vec<u8>) {
    code.push(0xC3);
}

// =============================================================================
// Op dispatch
// =============================================================================

/// `dst = op(src1, src2)`. VEX is 3-operand/non-destructive: operands are
/// never clobbered and may alias `dst`. Comparisons produce an ordinary
/// all-ones/all-zeros vector directly (no k-register step, unlike AVX-512).
pub fn emit_binary(code: &mut Vec<u8>, op: OpKind, dst: Reg, src1: Reg, src2: Reg) {
    let (d, s1, s2) = (dst.0, src1.0, src2.0);
    if let Some(pred) = cmp_pred(op) {
        vcmpps(code, d, s1, s2, pred);
        return;
    }
    match op {
        OpKind::Add => vaddps(code, d, s1, s2),
        OpKind::Sub => vsubps(code, d, s1, s2),
        OpKind::Mul => vmulps(code, d, s1, s2),
        OpKind::Div => vdivps(code, d, s1, s2),
        OpKind::Min => vminps(code, d, s1, s2),
        OpKind::Max => vmaxps(code, d, s1, s2),
        OpKind::BitAnd => vandps(code, d, s1, s2),
        OpKind::BitOr => vorps(code, d, s1, s2),
        OpKind::IAdd => vpaddd(code, d, s1, s2),
        _ => unimplemented_op("avx2", op),
    }
}

/// `dst = op(src)`.
pub fn emit_unary(code: &mut Vec<u8>, op: OpKind, dst: Reg, src: Reg) {
    match op {
        OpKind::Sqrt => vsqrtps(code, dst.0, src.0),
        OpKind::Rsqrt => vrsqrtps(code, dst.0, src.0),
        OpKind::Recip => vrcpps(code, dst.0, src.0),
        OpKind::Neg => {
            emit_const(code, UNARY_SCRATCH, f32::from_bits(0x8000_0000));
            vxorps(code, dst.0, src.0, UNARY_SCRATCH.0);
        }
        OpKind::Abs => {
            emit_const(code, UNARY_SCRATCH, f32::from_bits(0x7FFF_FFFF));
            vandps(code, dst.0, src.0, UNARY_SCRATCH.0);
        }
        // imm8: bits[3:0] = rounding mode (0=nearest, 1=floor, 2=ceil).
        OpKind::Floor => vroundps(code, dst.0, src.0, 0x01),
        OpKind::Ceil => vroundps(code, dst.0, src.0, 0x02),
        OpKind::Round => vroundps(code, dst.0, src.0, 0x00),
        OpKind::TruncToInt => vcvttps2dq(code, dst.0, src.0),
        OpKind::IntToFloat => vcvtdq2ps(code, dst.0, src.0),
        _ => unimplemented_op("avx2", op),
    }
}

/// Scratch register for unary mask materialization (neg/abs) and the select
/// blend temp. ymm15 is outside the allocatable range (4-9), reload regs
/// (11-12), and inputs (0-3).
const UNARY_SCRATCH: Reg = Reg(15);

/// Emit a shift of i32 lanes by a compile-time immediate.
pub fn emit_shift_imm(code: &mut Vec<u8>, op: OpKind, dst: Reg, src: Reg, amount: u8) {
    match op {
        OpKind::Shl => vpslld_imm(code, dst.0, src.0, amount),
        OpKind::Shr => vpsrld_imm(code, dst.0, src.0, amount),
        _ => unimplemented_op("avx2", op),
    }
}

/// `dst = mask ? if_true : if_false` (bit-select; mask already in `dst`, same
/// convention as SSE2/AVX-512). `tmp` differs from all of `dst`/`if_true`/`if_false`.
pub fn emit_select(code: &mut Vec<u8>, dst: Reg, if_true: Reg, if_false: Reg) {
    let tmp = UNARY_SCRATCH;
    debug_assert!(tmp.0 != dst.0 && tmp.0 != if_true.0 && tmp.0 != if_false.0);
    vandps(code, tmp.0, dst.0, if_true.0); // tmp = mask & if_true
    vandnps(code, dst.0, dst.0, if_false.0); // dst = ~mask & if_false
    vorps(code, dst.0, tmp.0, dst.0); // dst = blended
}

/// `vmovmskps eax, ymmSRC` — gather the 8 lane sign bits into eax[7:0].
pub fn emit_movmskps_eax(code: &mut Vec<u8>, src: Reg) {
    Vex::m0f(0x50).rrr(code, 0, UNUSED_VVVV, src.0);
}

/// `cmp al, imm8` — unlike `cmp eax, imm8` (sign-extending `0x83`), this
/// compares the raw byte pattern, which is what an 8-lane all-true check
/// (`eax == 0xFF`) needs (`0x83`'s sign-extension would compare against
/// `0xFFFFFFFF`, which `vmovmskps`'s zero-extended result can never equal).
pub fn emit_cmp_al_imm8(code: &mut Vec<u8>, imm: u8) {
    code.push(0x3C);
    code.push(imm);
}

/// `vfmadd231ps ymmD, ymmA, ymmB` — `dst = a*b + dst` (231 form: dst is the
/// addend going in, `a`/`b` the product). VEX.256.66.0F38.W0 B8 /r, same
/// opcode as `avx512.rs`'s EVEX form, just VEX-encoded at 256 bits. Requires
/// `target_feature = "fma"` (FMA3) — not implied by `avx2` alone.
#[cfg(target_feature = "fma")]
fn vfmadd231ps(c: &mut Vec<u8>, d: u8, s1: u8, s2: u8) {
    Vex::m0f38_66(0xB8).rrr(c, d, s1, s2);
}

/// Fused multiply-add: `dst` already holds `c`; computes `dst = a*b + dst`.
///
/// Uses real hardware FMA when the build has `+fma` (rounds once, exactly
/// matching the reference interpreter under `fp-contract=fast` — see the
/// regression this fixed: `eval_scalar`'s scalar `a*b+c` gets contracted to
/// an `fma` instruction by LLVM under `+fma` too, so a software two-step
/// mul-then-add here would round twice and disagree in the last bit).
/// Falls back to software mul+add otherwise (this backend is gated on `avx2`
/// alone, not `avx2,fma` — VIA/older Zen chips shipped AVX2 without FMA3).
#[cfg(target_feature = "fma")]
pub fn emit_fmadd_c_in_dst(code: &mut Vec<u8>, dst: Reg, a: Reg, b: Reg) {
    vfmadd231ps(code, dst.0, a.0, b.0);
}

/// See the `+fma` variant above.
#[cfg(not(target_feature = "fma"))]
pub fn emit_fmadd_c_in_dst(code: &mut Vec<u8>, dst: Reg, a: Reg, b: Reg) {
    let tmp = UNARY_SCRATCH;
    vmulps(code, tmp.0, a.0, b.0);
    vaddps(code, dst.0, dst.0, tmp.0);
}

// =============================================================================
// Bound-memory gather (RawGather lowering target)
//
// No native vgatherdps here (see the module doc): truncate all 8 lanes at
// once, split into two 128-bit halves, run the existing SSE2/AVX scalar-load
// sequence (`x86_64::emit_gather_scalar`) on each half (it only ever touches
// the low 128 bits of whatever register it's given — ymm0's low 128 IS
// xmm0), then recombine with vinsertf128.
// =============================================================================

/// `dst = buffer[slot][idx_lane]` for 8 lanes. `idx` holds FLOAT indices
/// (already clamped in range by the `Gather` lowering — `x86_64::emit_gather_scalar`
/// does its own float->int truncation per half, so `idx` must not be
/// pre-truncated here). Clobbers the GPRs and vector scratch in `s`, plus
/// `idx_hi`/`res_hi` (backend-reserved scratch distinct from `s`'s fields and
/// from `dst`/`idx`).
#[allow(clippy::too_many_arguments)]
pub fn emit_gather_scalar(
    code: &mut Vec<u8>,
    dst: Reg,
    idx: Reg,
    slot: u16,
    s: x86_64::GatherScratch,
    idx_hi: Reg,
    res_hi: Reg,
) {
    // idx's low 128 already holds lanes 0..4 (float); split off lanes 4..8
    // into idx_hi before either gather call touches idx/dst (which may alias).
    vextractf128(code, idx_hi.0, idx.0, 1);

    // Low half: lanes 0..4. May write dst == idx (the callee handles that:
    // it converts idx to int in scratch before ever writing dst).
    x86_64::emit_gather_scalar(code, dst, idx, slot, s);
    // High half: lanes 4..8, into res_hi (a 128-bit scratch distinct from dst).
    x86_64::emit_gather_scalar(code, res_hi, idx_hi, slot, s);

    // Recombine: dst[0..4] already holds the low half; splice in the high.
    vinsertf128(code, dst.0, dst.0, res_hi.0, 1);
}

#[cfg(test)]
mod tests {
    //! Hardware validation, mirroring `avx512.rs`'s runtime test tier: JIT real
    //! `ymm` kernels and execute them on the host.
    #[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
    mod runtime {
        use super::super::*;
        use crate::backend::emit::executable::ExecutableCode;
        use core::arch::x86_64::*;

        #[allow(improper_ctypes_definitions)]
        type K = unsafe extern "C" fn(__m256, __m256, __m256, __m256) -> __m256;

        fn run(body: &[u8], xs: [f32; 8], ys: [f32; 8], zs: [f32; 8]) -> [f32; 8] {
            let mut code = body.to_vec();
            emit_ret(&mut code);
            let exec = unsafe { ExecutableCode::from_code(&code).expect("mmap") };
            unsafe {
                let f: K = exec.as_fn();
                let r = f(
                    _mm256_loadu_ps(xs.as_ptr()),
                    _mm256_loadu_ps(ys.as_ptr()),
                    _mm256_loadu_ps(zs.as_ptr()),
                    _mm256_setzero_ps(),
                );
                let mut out = [0.0f32; 8];
                _mm256_storeu_ps(out.as_mut_ptr(), r);
                out
            }
        }

        fn lanes() -> ([f32; 8], [f32; 8], [f32; 8]) {
            let mut xs = [0.0; 8];
            let mut ys = [0.0; 8];
            let mut zs = [0.0; 8];
            for i in 0..8 {
                xs[i] = i as f32 - 3.5;
                ys[i] = (i as f32) * 0.5 + 1.0;
                zs[i] = 3.0 - (i as f32) * 0.25;
            }
            (xs, ys, zs)
        }

        /// One row of the binary-op table: the op and its scalar reference.
        type BinaryCase = (OpKind, fn(f32, f32) -> f32);

        fn check(got: [f32; 8], want: impl Fn(usize) -> f32, tag: &str) {
            for (i, &g) in got.iter().enumerate() {
                let w = want(i);
                assert!((g - w).abs() <= 1e-3, "{tag} lane {i}: got {g} want {w}");
            }
        }

        /// Bit-exact check for mask results: an all-ones/all-zeros lane is
        /// NaN under float subtraction, so `check`'s epsilon comparison can't
        /// be used for compares/selects' underlying mask bit pattern.
        fn check_bits(got: [f32; 8], want: impl Fn(usize) -> u32, tag: &str) {
            for (i, &g) in got.iter().enumerate() {
                let w = want(i);
                assert_eq!(
                    g.to_bits(),
                    w,
                    "{tag} lane {i}: got {:#x} want {:#x}",
                    g.to_bits(),
                    w
                );
            }
        }

        const X: Reg = Reg(0);
        const Y: Reg = Reg(1);
        const Z: Reg = Reg(2);

        #[test]
        fn binary_ops() {
            let (xs, ys, zs) = lanes();
            let cases: &[BinaryCase] = &[
                (OpKind::Add, |a, b| a + b),
                (OpKind::Sub, |a, b| a - b),
                (OpKind::Mul, |a, b| a * b),
                (OpKind::Div, |a, b| a / b),
                (OpKind::Min, |a, b| a.min(b)),
                (OpKind::Max, |a, b| a.max(b)),
            ];
            for &(op, f) in cases {
                let mut c = Vec::new();
                emit_binary(&mut c, op, X, X, Y).unwrap();
                check(run(&c, xs, ys, zs), |i| f(xs[i], ys[i]), "binary");
            }
        }

        #[test]
        fn compare_lt() {
            let (xs, ys, zs) = lanes();
            let mut c = Vec::new();
            emit_binary(&mut c, OpKind::Lt, X, X, Y).unwrap();
            check_bits(
                run(&c, xs, ys, zs),
                |i| if xs[i] < ys[i] { 0xFFFF_FFFF } else { 0 },
                "lt mask",
            );
        }

        #[test]
        fn sqrt_and_neg_abs() {
            let (xs, ys, zs) = lanes();
            let mut c = Vec::new();
            emit_unary(&mut c, OpKind::Sqrt, X, Y).unwrap();
            check(run(&c, xs, ys, zs), |i| ys[i].sqrt(), "sqrt");

            let mut c = Vec::new();
            emit_unary(&mut c, OpKind::Neg, X, X).unwrap();
            check(run(&c, xs, ys, zs), |i| -xs[i], "neg");

            let mut c = Vec::new();
            emit_unary(&mut c, OpKind::Abs, X, X).unwrap();
            check(run(&c, xs, ys, zs), |i| xs[i].abs(), "abs");
        }

        #[test]
        fn select_blend() {
            let (xs, ys, zs) = lanes();
            let mut c = Vec::new();
            emit_binary(&mut c, OpKind::Lt, Reg(5), X, Y).unwrap(); // mask
            emit_mov(&mut c, Reg(6), Reg(5));
            emit_select(&mut c, Reg(6), X, Y); // dst = mask ? x : y
            emit_mov(&mut c, X, Reg(6));
            check(
                run(&c, xs, ys, zs),
                |i| if xs[i] < ys[i] { xs[i] } else { ys[i] },
                "select",
            );
        }

        #[test]
        fn const_broadcast_and_fma() {
            let (xs, ys, zs) = lanes();
            let mut c = Vec::new();
            emit_const(&mut c, Reg(5), 2.5);
            emit_binary(&mut c, OpKind::Add, X, X, Reg(5)).unwrap();
            check(run(&c, xs, ys, zs), |i| xs[i] + 2.5, "const+add");

            let mut c = Vec::new();
            emit_mov(&mut c, Reg(5), Z);
            emit_fmadd_c_in_dst(&mut c, Reg(5), X, Y);
            emit_mov(&mut c, X, Reg(5));
            check(run(&c, xs, ys, zs), |i| xs[i] * ys[i] + zs[i], "fma sw");
        }

        #[test]
        fn spill_frame_roundtrip() {
            let (xs, ys, zs) = lanes();
            let mut c = Vec::new();
            emit_sub_rsp(&mut c, 32);
            emit_binary(&mut c, OpKind::Mul, Reg(6), X, Y).unwrap();
            emit_store_rsp(&mut c, Reg(6), 0);
            emit_binary(&mut c, OpKind::Add, Reg(6), X, X).unwrap(); // clobber
            emit_load_rsp(&mut c, X, 0);
            emit_add_rsp(&mut c, 32);
            check(run(&c, xs, ys, zs), |i| xs[i] * ys[i], "spill roundtrip");
        }

        #[test]
        fn gather_from_buffer() {
            // Matches the production ABI (mod.rs's ResolvedOp::Gather): the
            // first arg is a context pointer to an ARRAY of buffer base
            // pointers (one per slot), not a buffer pointer directly —
            // `x86_64::emit_gather_scalar` loads `[ctx_gpr + slot*8]` to get
            // the real base. `emit_load_ptr_from_ctx`'s doc calls this out.
            #[allow(improper_ctypes_definitions)]
            type G = unsafe extern "C" fn(*const *const f32, __m256) -> __m256;

            let mut c = Vec::new();
            // idx (zmm/ymm0) -> int truncate happens inside emit_gather_scalar.
            let s = x86_64::GatherScratch {
                base_gpr: 0,  // rax
                index_gpr: 1, // rcx
                ctx_gpr: 7,   // rdi
                idx_lanes: Reg(13),
                value: Reg(14),
            };
            emit_gather_scalar(&mut c, Reg(0), Reg(0), 0, s, Reg(9), Reg(8));
            emit_ret(&mut c);

            let buf: Vec<f32> = (0..64).map(|i| (i as f32) * 1.5 + 0.25).collect();
            let idx: [f32; 8] = [0.0, 63.0, 1.0, 2.0, 10.0, 5.0, 32.0, 7.0];
            let ctx: [*const f32; 1] = [buf.as_ptr()];

            let exec = unsafe { ExecutableCode::from_code(&c).expect("mmap") };
            let out = unsafe {
                let f: G = exec.as_fn();
                let r = f(ctx.as_ptr(), _mm256_loadu_ps(idx.as_ptr()));
                let mut out = [0.0f32; 8];
                _mm256_storeu_ps(out.as_mut_ptr(), r);
                out
            };

            for i in 0..8 {
                let want = buf[idx[i] as usize];
                assert_eq!(out[i], want, "gather lane {i}: idx {}", idx[i]);
            }
        }
    }
}
