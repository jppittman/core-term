//! x86-64 AVX2 (VEX.256) JIT encoder — 256-bit, 8-lane `ymm` kernels.
//!
//! The floor of the x86-64 tiers (`crate::isa`), below the AVX-512 EVEX
//! encoders (`avx512.rs`, 512-bit). Sixteen registers, `ymm0-15` — AVX2 has
//! no extended file — and the general-register half of every kernel (the
//! loop nest, the store's address, the pointer class, the constant pool) is
//! `x86_64.rs`'s, shared with AVX-512; only the vector *encoding* is this
//! file's.
//!
//! VEX is 3-operand and non-destructive — the same property AVX-512's EVEX
//! has — so an operand may alias the destination in every encoding here.
//! Comparisons are simpler here than on AVX-512: `vcmpps` writes an ordinary
//! all-ones/all-zeros `ymm` directly (no k-register, no mask-to-vector
//! conversion) — the same representation NEON uses.
//!
//! Spills use a real stack frame, not the red zone: mirrors `avx512.rs`'s
//! reasoning (a `ymm` slot is 32 bytes; keeping the red-zone arithmetic exact
//! for two different slot widths is not worth it for a bit of frame reuse on
//! tiny kernels).
//!
//! A gather is the hardware's: `vgatherdps ymm, [base + ymm*4], ymm` reads
//! one element per lane through a VSIB, under a vector mask the instruction
//! clears as it completes lanes — so each gather sets its mask to all-ones
//! first, from a register the allocator reserved for it, the way the
//! AVX-512 tier resets `k1`.

use super::asm::Encoding;
use super::build::{Builder, Def, Spiller};
use super::regalloc::resource::FrameSlot;
use super::register_file::{EntryRegisters, Members, RegisterFile};
use super::x86_64;
use super::x86_64::{
    Alu, Direction, Disp, GENERAL, Imm8, Imm32, Lanewise, Mem, NoDisp, Pred, Rounding, X86,
};
use super::{
    Bound, Class, ClassId, Edges, EncodedInst, Integer, IsaBackend, Label, LaneOp, Pointer, Rebind,
    Selected, Stage, Store, Target, Test, Value, ValueName, Vector, slot_disp, unimplemented_op,
};
use crate::error::CompileError;
use crate::program::IfArm;
use alloc::vec::Vec;
use pixelflow_ir::OpKind;

// The AVX2 tier requires FMA3, and `crate::isa` refuses a host without it; the
// probe's doc says why.

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

/// The identity of one VEX instruction: opcode map, implied legacy prefix, W
/// bit, opcode byte, and whether it is the 256-bit form. This is *which
/// instruction* — it is constant per mnemonic, so each mnemonic below states
/// it exactly once and the operand form (`rrr`/`imm`/`rm`) supplies the
/// per-call parts.
#[derive(Clone, Copy)]
struct Vex {
    map: Map,
    pp: Pp,
    w: bool,
    /// `VEX.L`: set for the `ymm` form, clear for the few `xmm`-only
    /// instructions this tier needs (`vmovq`, `vextractps`), which `#UD` at
    /// `L = 1`.
    l256: bool,
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
            l256: true,
            opcode,
        }
    }
    /// The 128-bit (`VEX.L = 0`) form of this instruction.
    const fn xmm(self) -> Self {
        Self {
            l256: false,
            ..self
        }
    }
    /// The 64-bit-operand (`VEX.W = 1`) form of this instruction.
    const fn w1(self) -> Self {
        Self { w: true, ..self }
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
    fn rrr(self, dst: u8, vvvv: u8, rm: u8) -> EncodedInst {
        let mut inst = EncodedInst::new();
        let rbit = if dst >= 8 { 0x00 } else { 0x80 };
        let xbit = 0x40;
        let bbit = if rm >= 8 { 0x00 } else { 0x20 };
        inst.push(0xC4);
        inst.push(rbit | xbit | bbit | self.map as u8);
        inst.push(
            ((self.w as u8) << 7) | ((!vvvv & 0xF) << 3) | ((self.l256 as u8) << 2) | self.pp as u8,
        );
        inst.push(self.opcode);
        inst.push(0xC0 | ((dst & 7) << 3) | (rm & 7));
        inst
    }

    /// `op reg, [base + disp]` — the memory-operand form, for any base and any
    /// displacement mode. The prefix is VEX's; the ModRM/SIB/displacement tail
    /// is the architecture's, so it comes from `x86_64::mem_operand_into`.
    fn rm<D: Disp>(self, reg: u8, base: u8, disp: D) -> EncodedInst {
        let mut inst = EncodedInst::new();
        // R and B are stored inverted; X is unused (no index register).
        let rbit = if reg >= 8 { 0x00 } else { 0x80 };
        let bbit = if base >= 8 { 0x00 } else { 0x20 };
        inst.push(0xC4);
        inst.push(rbit | 0x40 | bbit | self.map as u8);
        inst.push(((self.w as u8) << 7) | (0xF << 3) | ((self.l256 as u8) << 2) | self.pp as u8); // vvvv unused
        inst.push(self.opcode);
        x86_64::mem_operand_into(&mut inst, reg, base, disp);
        inst
    }

    /// `op reg, [rip + disp32]`: the displacement is the last four bytes, a
    /// label field.
    fn rip(self, reg: u8) -> EncodedInst {
        let mut inst = EncodedInst::new();
        let rbit = if reg >= 8 { 0x00 } else { 0x80 };
        inst.push(0xC4);
        inst.push(rbit | 0x60 | self.map as u8);
        inst.push(((self.w as u8) << 7) | (0xF << 3) | ((self.l256 as u8) << 2) | self.pp as u8); // vvvv unused
        inst.push(self.opcode);
        x86_64::rip_operand_into(&mut inst, reg);
        inst
    }

    /// `op reg, [base + index*4]` — the SIB form with a scaled index, which
    /// a broadcast load reads one element of a plane through. X carries the
    /// index's high bit, inverted like R and B.
    fn rm_scaled4(self, reg: u8, base: u8, index: u8) -> EncodedInst {
        let mut inst = EncodedInst::new();
        let rbit = if reg >= 8 { 0x00 } else { 0x80 };
        let xbit = if index >= 8 { 0x00 } else { 0x40 };
        let bbit = if base >= 8 { 0x00 } else { 0x20 };
        inst.push(0xC4);
        inst.push(rbit | xbit | bbit | self.map as u8);
        inst.push(((self.w as u8) << 7) | (0xF << 3) | ((self.l256 as u8) << 2) | self.pp as u8); // vvvv unused
        inst.push(self.opcode);
        x86_64::scaled4_operand_into(&mut inst, reg, base, index);
        inst
    }

    /// `op reg, [base + ymmINDEX*4], vvvv` — the VSIB form, whose index is
    /// a *vector* register: one address per lane, scale 4, no displacement.
    /// X carries the index's high bit exactly as it does for a GPR index;
    /// the SIB tail is the same bytes with a vector number in the index
    /// field (`x86_64::vsib4_operand_into`, which knows `ymm4`/`ymm12` are
    /// not `rsp`/`r12`). `vvvv` is the gather's mask.
    fn vsib_scaled4(self, reg: u8, vvvv: u8, base: u8, index: u8) -> EncodedInst {
        let mut inst = EncodedInst::new();
        let rbit = if reg >= 8 { 0x00 } else { 0x80 };
        let xbit = if index >= 8 { 0x00 } else { 0x40 };
        let bbit = if base >= 8 { 0x00 } else { 0x20 };
        inst.push(0xC4);
        inst.push(rbit | xbit | bbit | self.map as u8);
        inst.push(
            ((self.w as u8) << 7) | ((!vvvv & 0xF) << 3) | ((self.l256 as u8) << 2) | self.pp as u8,
        );
        inst.push(self.opcode);
        x86_64::vsib4_operand_into(&mut inst, reg, base, index);
        inst
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
    fn rrr(self, dst: u8, vvvv: u8, rm: u8) -> EncodedInst {
        let mut inst = self.vex.rrr(dst, vvvv, rm);
        inst.push(self.imm);
        inst
    }
    /// Memory form with the imm8 appended.
    fn rm<D: Disp>(self, reg: u8, base: u8, disp: D) -> EncodedInst {
        let mut inst = self.vex.rm(reg, base, disp);
        inst.push(self.imm);
        inst
    }
}

impl Alu {
    const fn vex(self) -> Vex {
        match self {
            Alu::Add => Vex::m0f(0x58),
            Alu::Sub => Vex::m0f(0x5C),
            Alu::Mul => Vex::m0f(0x59),
            Alu::Div => Vex::m0f(0x5E),
            Alu::Min => Vex::m0f(0x5D),
            Alu::Max => Vex::m0f(0x5F),
            Alu::And => Vex::m0f(0x54),
            Alu::AndNot => Vex::m0f(0x55),
            Alu::Or => Vex::m0f(0x56),
            Alu::Xor => Vex::m0f(0x57),
            Alu::IAdd => Vex::m0f_66(0xFE),
        }
    }
}

impl Lanewise {
    const fn vex(self) -> Vex {
        match self {
            Lanewise::Sqrt => Vex::m0f(0x51),
            Lanewise::Rsqrt => Vex::m0f(0x52),
            Lanewise::Recip => Vex::m0f(0x53),
            Lanewise::ToInt => Vex::m0f_f3(0x5B),
            Lanewise::FromInt => Vex::m0f(0x5B),
            Lanewise::WidenBytes => Vex::m0f38_66(0x31),
        }
    }
}

/// An AVX2 (VEX.256) vector instruction.
///
/// Generic over what its operands are ([`Stage`]), like [`x86_64::Gp`]. VEX is
/// three-operand and non-destructive, so no destination here is tied to a
/// source, and an operand may be the register the result is written to.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) enum Inst<S: Stage> {
    /// `op dst, a, b`
    Alu {
        op: Alu,
        dst: S::Write<Vector>,
        a: S::Read<Vector>,
        b: S::Read<Vector>,
    },
    /// `vcmpps dst, a, b, pred`: an all-ones lane where it holds, all-zero
    /// where it does not.
    Cmp {
        pred: Pred,
        dst: S::Write<Vector>,
        a: S::Read<Vector>,
        b: S::Read<Vector>,
    },
    /// `op dst, src`
    Unary {
        op: Lanewise,
        dst: S::Write<Vector>,
        src: S::Read<Vector>,
    },
    /// `vroundps dst, src, mode`
    Round {
        mode: Rounding,
        dst: S::Write<Vector>,
        src: S::Read<Vector>,
    },
    /// `vpslld`/`vpsrld dst, src, amount`
    Shift {
        direction: Direction,
        dst: S::Write<Vector>,
        src: S::Read<Vector>,
        amount: u8,
    },
    /// `vfmadd231ps acc, a, b`: `acc = a·b + acc`, one rounding. FMA3 is not
    /// implied by AVX2 in CPUID, which is why `crate::isa`'s x86-64 probe asks
    /// for both before this backend can be selected.
    Fma231 {
        acc: S::Tie<Vector>,
        a: S::Read<Vector>,
        b: S::Read<Vector>,
    },
    /// `vpcmpeqd dst, dst, dst`: all-ones, whatever `dst` held. Its reads are
    /// not operands, because the result does not depend on them.
    Ones { dst: S::Write<Vector> },
    /// `vxorps dst, dst, dst`: zero, whatever `dst` held. Its reads are not
    /// operands, because the result does not depend on them.
    Zero { dst: S::Write<Vector> },
    /// `vmovaps dst, src`
    Mov {
        dst: S::Write<Vector>,
        src: S::Read<Vector>,
    },
    /// `vbroadcastss dst, [rip + at]`: one pool entry, whose label the
    /// kernel's constants bind.
    LoadConst { dst: S::Write<Vector>, at: Label },
    /// `vmovups [rsp + slot], src`: the allocator's spill.
    SpillVector { src: S::Read<Vector>, slot: S::Slot },
    /// `vmovups dst, [rsp + slot]`: the allocator's reload.
    ReloadVector {
        dst: S::Write<Vector>,
        slot: S::Slot,
    },
    /// `vcvttss2si dst, src`: lane 0, truncated to a 64-bit integer.
    Cvtt {
        dst: S::Write<Integer>,
        src: S::Read<Vector>,
    },
    /// `vmovq dst, src`: eight bytes into the low lanes, the rest zeroed.
    Movq {
        dst: S::Write<Vector>,
        src: S::Read<Integer>,
    },
    /// `vmovups [dst], src` with no displacement: a whole batch to the
    /// address the store's arithmetic computed.
    StoreBatch {
        dst: Mem<S, NoDisp>,
        src: S::Read<Vector>,
    },
    /// `vbroadcastss dst, [src]`: one `f32`, a pool entry or a uniform, into
    /// every lane.
    Broadcast {
        dst: S::Write<Vector>,
        src: Mem<S, Imm32>,
    },
    /// `vbroadcastss dst, [base + index*4]`: one element of a plane.
    BroadcastIndexed {
        dst: S::Write<Vector>,
        base: S::Read<Pointer>,
        index: S::Read<Integer>,
    },
    /// `vgatherdps dst, [base + index*4], mask`: one `f32` per lane, for
    /// every lane whose `mask` sign bit is set. The instruction clears the
    /// mask as it completes lanes, and `#UD`s unless `dst`, `index` and `mask`
    /// are three registers.
    Gather {
        dst: S::Early<Vector>,
        base: S::Read<Pointer>,
        index: S::Read<Vector>,
        mask: S::Tie<Vector>,
    },
    /// `vmovmskps dst, src`: the lane sign bits, in the low bits of `dst`.
    MoveMask {
        dst: S::Write<Integer>,
        src: S::Read<Vector>,
    },
    /// `vextractf128 dst, src, 1`: the high 128 bits.
    ExtractHigh {
        dst: S::Write<Vector>,
        src: S::Read<Vector>,
    },
    /// `vextractps [dst], src, lane`: one lane of the low half, stored.
    ExtractLane {
        dst: Mem<S, Imm8>,
        src: S::Read<Vector>,
        lane: u8,
    },
}

impl Inst<Selected> {
    /// Rebuild at stage `T`, visiting each operand once, in field order.
    fn walk<T: Stage>(&self, f: &mut impl Rebind<T>) -> Inst<T> {
        match self {
            Inst::Alu { op, dst, a, b } => Inst::Alu {
                op: *op,
                dst: f.write(dst),
                a: f.read(*a),
                b: f.read(*b),
            },
            Inst::Cmp { pred, dst, a, b } => Inst::Cmp {
                pred: *pred,
                dst: f.write(dst),
                a: f.read(*a),
                b: f.read(*b),
            },
            Inst::Unary { op, dst, src } => Inst::Unary {
                op: *op,
                dst: f.write(dst),
                src: f.read(*src),
            },
            Inst::Round { mode, dst, src } => Inst::Round {
                mode: *mode,
                dst: f.write(dst),
                src: f.read(*src),
            },
            Inst::Shift {
                direction,
                dst,
                src,
                amount,
            } => Inst::Shift {
                direction: *direction,
                dst: f.write(dst),
                src: f.read(*src),
                amount: *amount,
            },
            Inst::Fma231 { acc, a, b } => Inst::Fma231 {
                acc: f.tie(acc),
                a: f.read(*a),
                b: f.read(*b),
            },
            Inst::Ones { dst } => Inst::Ones { dst: f.write(dst) },
            Inst::Zero { dst } => Inst::Zero { dst: f.write(dst) },
            Inst::Mov { dst, src } => Inst::Mov {
                dst: f.write(dst),
                src: f.read(*src),
            },
            Inst::LoadConst { dst, at } => Inst::LoadConst {
                dst: f.write(dst),
                at: *at,
            },
            Inst::SpillVector { src, slot } => Inst::SpillVector {
                src: f.read(*src),
                slot: f.slot(*slot),
            },
            Inst::ReloadVector { dst, slot } => Inst::ReloadVector {
                dst: f.write(dst),
                slot: f.slot(*slot),
            },
            Inst::Cvtt { dst, src } => Inst::Cvtt {
                dst: f.write(dst),
                src: f.read(*src),
            },
            Inst::Movq { dst, src } => Inst::Movq {
                dst: f.write(dst),
                src: f.read(*src),
            },
            Inst::StoreBatch { dst, src } => Inst::StoreBatch {
                dst: dst.walk(f),
                src: f.read(*src),
            },
            Inst::Broadcast { dst, src } => Inst::Broadcast {
                dst: f.write(dst),
                src: src.walk(f),
            },
            Inst::BroadcastIndexed { dst, base, index } => Inst::BroadcastIndexed {
                dst: f.write(dst),
                base: f.read(*base),
                index: f.read(*index),
            },
            Inst::Gather {
                dst,
                base,
                index,
                mask,
            } => Inst::Gather {
                dst: f.early(dst),
                base: f.read(*base),
                index: f.read(*index),
                mask: f.tie(mask),
            },
            Inst::MoveMask { dst, src } => Inst::MoveMask {
                dst: f.write(dst),
                src: f.read(*src),
            },
            Inst::ExtractHigh { dst, src } => Inst::ExtractHigh {
                dst: f.write(dst),
                src: f.read(*src),
            },
            Inst::ExtractLane { dst, src, lane } => Inst::ExtractLane {
                dst: dst.walk(f),
                src: f.read(*src),
                lane: *lane,
            },
        }
    }
}

impl<'m, B: IsaBackend> Inst<Bound<'m, B>> {
    fn encode(&self) -> EncodedInst {
        match self {
            Inst::Alu { op, dst, a, b } => op.vex().rrr(dst.number(), a.number(), b.number()),
            Inst::Cmp { pred, dst, a, b } => {
                Vex::m0f(0xC2)
                    .imm(*pred as u8)
                    .rrr(dst.number(), a.number(), b.number())
            }
            Inst::Unary { op, dst, src } => op.vex().rrr(dst.number(), UNUSED_VVVV, src.number()),
            Inst::Round { mode, dst, src } => {
                Vex::m0f3a_66(0x08)
                    .imm(*mode as u8)
                    .rrr(dst.number(), UNUSED_VVVV, src.number())
            }
            // The destination is `vvvv` and the `/digit` is `reg`.
            Inst::Shift {
                direction,
                dst,
                src,
                amount,
            } => Vex::m0f_66(0x72)
                .imm(*amount)
                .rrr(*direction as u8, dst.number(), src.number()),
            Inst::Fma231 { acc, a, b } => {
                Vex::m0f38_66(0xB8).rrr(acc.number(), a.number(), b.number())
            }
            Inst::Ones { dst } => {
                let dst = dst.number();
                Vex::m0f_66(0x76).rrr(dst, dst, dst)
            }
            Inst::Zero { dst } => {
                let dst = dst.number();
                Alu::Xor.vex().rrr(dst, dst, dst)
            }
            Inst::Mov { dst, src } => Vex::m0f(0x28).rrr(dst.number(), UNUSED_VVVV, src.number()),
            Inst::LoadConst { dst, .. } => Vex::m0f38_66(0x18).rip(dst.number()),
            Inst::SpillVector { src, slot } => {
                Vex::m0f(0x11).rm(src.number(), x86_64::gpr::RSP, Imm32(slot_disp(slot)))
            }
            Inst::ReloadVector { dst, slot } => {
                Vex::m0f(0x10).rm(dst.number(), x86_64::gpr::RSP, Imm32(slot_disp(slot)))
            }
            Inst::Cvtt { dst, src } => {
                Vex::m0f_f3(0x2C)
                    .w1()
                    .rrr(dst.number(), UNUSED_VVVV, src.number())
            }
            Inst::Movq { dst, src } => {
                Vex::m0f_66(0x6E)
                    .w1()
                    .xmm()
                    .rrr(dst.number(), UNUSED_VVVV, src.number())
            }
            Inst::StoreBatch { dst, src } => {
                Vex::m0f(0x11).rm(src.number(), dst.base.number(), dst.disp)
            }
            Inst::Broadcast { dst, src } => {
                Vex::m0f38_66(0x18).rm(dst.number(), src.base.number(), src.disp)
            }
            Inst::BroadcastIndexed { dst, base, index } => {
                Vex::m0f38_66(0x18).rm_scaled4(dst.number(), base.number(), index.number())
            }
            Inst::Gather {
                dst,
                base,
                index,
                mask,
            } => {
                let (dst, index, mask) = (dst.number(), index.number(), mask.number());
                debug_assert!(
                    dst != index && dst != mask && index != mask,
                    "vgatherdps: dst, index and mask must be three registers"
                );
                // `base` is never `rbp`/`r13` (`GENERAL` excludes them), so
                // the SIB's no-base encoding is unreachable.
                Vex::m0f38_66(0x92).vsib_scaled4(dst, mask, base.number(), index)
            }
            Inst::MoveMask { dst, src } => {
                Vex::m0f(0x50).rrr(dst.number(), UNUSED_VVVV, src.number())
            }
            // The destination is the `rm` operand here, the reverse of the
            // usual direction.
            Inst::ExtractHigh { dst, src } => {
                Vex::m0f3a_66(0x19)
                    .imm(1)
                    .rrr(src.number(), UNUSED_VVVV, dst.number())
            }
            Inst::ExtractLane { dst, src, lane } => {
                debug_assert!(*lane < 4, "vextractps reads the low 128 bits");
                Vex::m0f3a_66(0x17)
                    .xmm()
                    .imm(*lane)
                    .rm(src.number(), dst.base.number(), dst.disp)
            }
        }
    }

    /// The instruction's bytes, and the label fields in them.
    fn assemble(&self, out: &mut Encoding<'_>) {
        match self {
            Inst::LoadConst { at, .. } => {
                let inst = self.encode();
                out.bytes(inst.as_bytes());
                // The displacement is the instruction's last four bytes.
                out.field(inst.as_bytes().len() - 4, *at, x86_64::patch_rel32);
            }
            _ => out.bytes(self.encode().as_bytes()),
        }
    }
}

/// What the selection pipeline builds for AVX2: a vector instruction or a
/// general-register one (the loop nest's, and the store's address
/// arithmetic). One type so a block holds both; no derives, because a
/// selected instruction owns its definitions.
pub(super) enum Op<S: Stage> {
    Vector(Inst<S>),
    General(x86_64::Gp<S>),
}

impl Op<Selected> {
    fn walk<T: Stage>(&self, f: &mut impl Rebind<T>) -> Op<T> {
        match self {
            Op::Vector(inst) => Op::Vector(inst.walk(f)),
            Op::General(gp) => Op::General(gp.walk(f)),
        }
    }
}

impl<'m> Op<Bound<'m, Avx2>> {
    fn assemble(&self, out: &mut Encoding<'_>) {
        match self {
            Op::Vector(inst) => inst.assemble(out),
            Op::General(gp) => gp.assemble(out),
        }
    }
}

// =============================================================================
// Op dispatch
// =============================================================================

/// The one instruction a binary op is.
enum BinaryForm {
    Alu(Alu),
    /// A comparison produces an ordinary all-ones/all-zeros vector directly
    /// (no k-register step, unlike AVX-512).
    Cmp(Pred),
}

fn binary_form(op: OpKind) -> BinaryForm {
    match op {
        OpKind::Eq => BinaryForm::Cmp(Pred::Eq),
        OpKind::Ne => BinaryForm::Cmp(Pred::Ne),
        OpKind::Lt => BinaryForm::Cmp(Pred::Lt),
        OpKind::Le => BinaryForm::Cmp(Pred::Le),
        OpKind::Gt => BinaryForm::Cmp(Pred::Nle),
        OpKind::Ge => BinaryForm::Cmp(Pred::Ge),
        OpKind::Add => BinaryForm::Alu(Alu::Add),
        OpKind::Sub => BinaryForm::Alu(Alu::Sub),
        OpKind::Mul => BinaryForm::Alu(Alu::Mul),
        OpKind::Div => BinaryForm::Alu(Alu::Div),
        OpKind::Min => BinaryForm::Alu(Alu::Min),
        OpKind::Max => BinaryForm::Alu(Alu::Max),
        OpKind::BitAnd => BinaryForm::Alu(Alu::And),
        OpKind::BitOr => BinaryForm::Alu(Alu::Or),
        OpKind::IAdd => BinaryForm::Alu(Alu::IAdd),
        _ => unimplemented_op("avx2", op),
    }
}

impl BinaryForm {
    fn inst<S: Stage>(
        self,
        dst: S::Write<Vector>,
        a: S::Read<Vector>,
        b: S::Read<Vector>,
    ) -> Inst<S> {
        match self {
            BinaryForm::Alu(op) => Inst::Alu { op, dst, a, b },
            BinaryForm::Cmp(pred) => Inst::Cmp { pred, dst, a, b },
        }
    }
}

/// The one instruction a unary op is.
enum UnaryForm {
    Lanewise(Lanewise),
    Round(Rounding),
    /// `Neg` and `Abs` xor or and the sign bit, which is a constant to hold in
    /// a register first.
    Signed {
        op: Alu,
        bits: u32,
    },
}

fn unary_form(op: OpKind) -> UnaryForm {
    match op {
        OpKind::Sqrt => UnaryForm::Lanewise(Lanewise::Sqrt),
        OpKind::Rsqrt => UnaryForm::Lanewise(Lanewise::Rsqrt),
        OpKind::Recip => UnaryForm::Lanewise(Lanewise::Recip),
        OpKind::Neg => UnaryForm::Signed {
            op: Alu::Xor,
            bits: 0x8000_0000,
        },
        OpKind::Abs => UnaryForm::Signed {
            op: Alu::And,
            bits: 0x7FFF_FFFF,
        },
        OpKind::Floor => UnaryForm::Round(Rounding::Floor),
        OpKind::Ceil => UnaryForm::Round(Rounding::Ceil),
        OpKind::Round => UnaryForm::Round(Rounding::Nearest),
        OpKind::TruncToInt => UnaryForm::Lanewise(Lanewise::ToInt),
        OpKind::IntToFloat => UnaryForm::Lanewise(Lanewise::FromInt),
        _ => unimplemented_op("avx2", op),
    }
}

// =============================================================================
// The store, and the iota
// =============================================================================

/// The bytes `0..8`, little end first: what one `movabs` carries in for
/// `vpmovzxbd` to widen into the iota.
const IOTA_BYTES: u64 = 0x0706_0504_0302_0100;

fn shift_direction(op: OpKind) -> Direction {
    match op {
        OpKind::Shl => Direction::Left,
        OpKind::Shr => Direction::Right,
        _ => unimplemented_op("avx2", op),
    }
}

// =============================================================================
// Selection
// =============================================================================

/// The AVX2 machine, as the selection pipeline sees it.
pub(in crate::emit) enum Avx2 {}

/// Define a value of class `C` with `make`, which builds the vector
/// instruction that writes it.
fn vex<C: Class>(b: &mut Builder<Avx2>, make: impl FnOnce(Def<C>) -> Inst<Selected>) -> Value<C> {
    let dst = b.def();
    let value = dst.value();
    b.push(Op::Vector(make(dst)));
    value
}

impl X86 for Avx2 {
    fn general(inst: x86_64::Gp<Selected>) -> Op<Selected> {
        Op::General(inst)
    }
}

impl IsaBackend for Avx2 {
    type Inst<S: Stage> = Op<S>;
    type Constant = u32;
    type Anchors = ();
    type Lane = Value<Vector>;

    // SysV has no callee-saved vector registers and the collapse ABI passes
    // no vector, so all sixteen `ymm` are the allocator's.
    const FILE: RegisterFile = RegisterFile::new(
        Members {
            vector: &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            general: GENERAL,
            opmask: &[],
            flags: &[0],
        },
        EntryRegisters {
            ctx: x86_64::gpr::RDI,
            out: x86_64::gpr::RSI,
            pitch: x86_64::gpr::RDX,
        },
        32,
    );
    const POOL_REACH: u64 = i32::MAX as u64 / 4;

    fn lane(b: &mut Builder<Self>, op: LaneOp<Self>) -> Result<Value<Vector>, CompileError> {
        Ok(match op {
            LaneOp::Const(value) => match value.to_bits() {
                0 => vex(b, |dst| Inst::Zero { dst }),
                u32::MAX => vex(b, |dst| Inst::Ones { dst }),
                bits => {
                    let at = b.constant(bits)?.label;
                    vex(b, |dst| Inst::LoadConst { dst, at })
                }
            },
            // The bytes `0..8` in through a general register, widened to
            // dwords, converted.
            LaneOp::Lanes => {
                let bytes = x86_64::define(b, |dst| x86_64::Gp::Movabs {
                    dst,
                    imm: IOTA_BYTES,
                });
                let packed = vex(b, |dst| Inst::Movq { dst, src: bytes });
                let widened = vex(b, |dst| Inst::Unary {
                    op: Lanewise::WidenBytes,
                    dst,
                    src: packed,
                });
                vex(b, |dst| Inst::Unary {
                    op: Lanewise::FromInt,
                    dst,
                    src: widened,
                })
            }
            LaneOp::Unary(op, src) => match unary_form(op) {
                UnaryForm::Lanewise(op) => vex(b, |dst| Inst::Unary { op, dst, src }),
                UnaryForm::Round(mode) => vex(b, |dst| Inst::Round { mode, dst, src }),
                UnaryForm::Signed { op, bits } => {
                    let mask = Self::lane(b, LaneOp::Const(f32::from_bits(bits)))?;
                    vex(b, |dst| Inst::Alu {
                        op,
                        dst,
                        a: src,
                        b: mask,
                    })
                }
            },
            LaneOp::Binary(op, a, c) => vex(b, |dst| binary_form(op).inst(dst, a, c)),
            LaneOp::MulAdd(x, y, addend) => {
                let acc = b.tie(addend);
                let result = acc.write();
                b.push(Op::Vector(Inst::Fma231 { acc, a: x, b: y }));
                result
            }
            // `cond ? if_true : if_false`, bit by bit.
            LaneOp::Blend {
                cond,
                if_true,
                if_false,
            } => {
                let kept = vex(b, |dst| Inst::Alu {
                    op: Alu::And,
                    dst,
                    a: cond,
                    b: if_true,
                });
                let other = vex(b, |dst| Inst::Alu {
                    op: Alu::AndNot,
                    dst,
                    a: cond,
                    b: if_false,
                });
                vex(b, |dst| Inst::Alu {
                    op: Alu::Or,
                    dst,
                    a: kept,
                    b: other,
                })
            }
            LaneOp::Shift(op, src, amount) => vex(b, |dst| Inst::Shift {
                direction: shift_direction(op),
                dst,
                src,
                amount,
            }),
            // `vbroadcastss dst, [block + 4·element]`.
            LaneOp::Uniform { base, element } => {
                let disp = x86_64::displacement(element, x86_64::F32_BYTES)?;
                vex(b, |dst| Inst::Broadcast {
                    dst,
                    src: Mem { base, disp },
                })
            }
            // `vgatherdps` under an all-ones mask: the indices are floats, so
            // they are truncated first, and the instruction clears the mask
            // as it completes lanes.
            LaneOp::Gather { base, index } => {
                let indices = vex(b, |dst| Inst::Unary {
                    op: Lanewise::ToInt,
                    dst,
                    src: index,
                });
                let ones = vex(b, |dst| Inst::Ones { dst });
                let (dst, mask) = (b.early(), b.tie(ones));
                let gathered = dst.value();
                b.push(Op::Vector(Inst::Gather {
                    dst,
                    base,
                    index: indices,
                    mask,
                }));
                gathered
            }
            // One element, the same in every lane: the index is lane 0,
            // truncated.
            LaneOp::Broadcast { base, index } => {
                let index = vex(b, |dst| Inst::Cvtt { dst, src: index });
                vex(b, |dst| Inst::BroadcastIndexed { dst, base, index })
            }
        })
    }

    fn lane_name(lane: Value<Vector>) -> ValueName {
        lane.name()
    }

    fn param_lane(param: ValueName) -> Value<Vector> {
        param.typed()
    }

    fn context(
        b: &mut Builder<Self>,
        ctx: Value<Pointer>,
        slot: u64,
    ) -> Result<Value<Pointer>, CompileError> {
        x86_64::context(b, ctx, slot)
    }

    /// `out + 4·(row·pitch + col)`: each index is lane 0 of its binder,
    /// truncated. A full batch is one `vmovups`; a remainder is `vextractps`
    /// per lane, the low four straight out of the value and the rest out of
    /// its high half.
    fn store(b: &mut Builder<Self>, store: Store<Self>) {
        let Store {
            out,
            pitch,
            row,
            col,
            value,
            lanes,
        } = store;
        let row = vex(b, |dst| Inst::Cvtt { dst, src: row });
        let col = vex(b, |dst| Inst::Cvtt { dst, src: col });
        let base = x86_64::element_address(b, out, pitch, (row, col));
        if u64::from(lanes) == Self::FILE.vector_bytes() / 4 {
            b.push(Op::Vector(Inst::StoreBatch {
                dst: Mem { base, disp: NoDisp },
                src: value,
            }));
            return;
        }
        let mut half = value;
        for lane in 0..lanes {
            if lane == 4 {
                half = vex(b, |dst| Inst::ExtractHigh { dst, src: value });
            }
            b.push(Op::Vector(Inst::ExtractLane {
                dst: Mem {
                    base,
                    disp: Imm8((lane * 4) as i8),
                },
                src: half,
                lane: (lane % 4) as u8,
            }));
        }
    }

    /// `vmovmskps`, then `test` (ZF iff no lane is set) or `cmp al, 0xFF` (ZF
    /// iff all eight are), then `je`: the `cmp al` form on a byte, because the
    /// sign-extending `cmp eax, imm8` could never equal a zero-extended mask.
    fn branch(b: &mut Builder<Self>, test: Test<Self>, edges: Edges) {
        let bits = vex(b, |dst| Inst::MoveMask {
            dst,
            src: test.cond,
        });
        let flags = b.def();
        let tested = flags.value();
        b.push(Op::General(match test.dead {
            IfArm::True => x86_64::Gp::Test { flags, src: bits },
            IfArm::False => x86_64::Gp::CmpByte {
                flags,
                src: bits,
                imm: 0xFF,
            },
        }));
        x86_64::branch(b, x86_64::Cond::E, tested, edges);
    }

    fn jump(b: &mut Builder<Self>, to: Target, next: Label) {
        x86_64::jump(b, to, next);
    }

    fn enter(b: &mut Builder<Self>) {
        x86_64::enter(b);
    }

    fn ret(b: &mut Builder<Self>) {
        x86_64::ret(b);
    }

    fn copy(b: &mut Spiller<'_, Self>, src: ValueName) -> ValueName {
        match src.class {
            ClassId::Vector => b.define::<Vector>(|dst| {
                Op::Vector(Inst::Mov {
                    dst,
                    src: src.typed(),
                })
            }),
            ClassId::Pointer | ClassId::Integer => x86_64::copy_general(b, src),
            ClassId::Opmask | ClassId::Flags => {
                unreachable!("AVX2 has no {:?} to copy", src.class)
            }
        }
    }

    fn spill(b: &mut Spiller<'_, Self>, src: ValueName, slot: &FrameSlot) {
        match src.class {
            ClassId::Vector => b.push(Op::Vector(Inst::SpillVector {
                src: src.typed(),
                slot: slot.name(),
            })),
            ClassId::Pointer | ClassId::Integer => x86_64::spill_general(b, src, slot),
            ClassId::Opmask | ClassId::Flags => {
                unreachable!("AVX2 has no {:?} to spill", src.class)
            }
        }
    }

    fn reload(b: &mut Spiller<'_, Self>, class: ClassId, slot: &FrameSlot) -> ValueName {
        match class {
            ClassId::Vector => b.define::<Vector>(|dst| {
                Op::Vector(Inst::ReloadVector {
                    dst,
                    slot: slot.name(),
                })
            }),
            ClassId::Pointer | ClassId::Integer => x86_64::reload_general(b, class, slot),
            ClassId::Opmask | ClassId::Flags => unreachable!("AVX2 has no {class:?} to reload"),
        }
    }

    fn rematerializable(inst: &Op<Selected>) -> bool {
        matches!(
            inst,
            Op::Vector(Inst::LoadConst { .. } | Inst::Zero { .. } | Inst::Ones { .. })
        )
    }

    fn reads_pool<S: Stage>(inst: &Op<S>) -> bool {
        matches!(inst, Op::Vector(Inst::LoadConst { .. }))
    }

    fn walk<T: Stage>(inst: &Op<Selected>, f: &mut impl Rebind<T>) -> Op<T> {
        inst.walk(f)
    }

    fn encode(inst: &Op<Bound<'_, Self>>, out: &mut Encoding<'_>) {
        inst.assemble(out);
    }

    fn constant_bytes(constant: u32) -> Vec<u8> {
        constant.to_le_bytes().to_vec()
    }
}
