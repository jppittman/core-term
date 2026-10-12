//! x86-64 AVX-512 (EVEX) JIT encoder — 512-bit, 16-lane `zmm` kernels.
//!
//! The widest of the x86-64 tiers (`crate::isa`), above the AVX2 VEX
//! encoders (`avx2.rs`, 256-bit). It targets the full `zmm0..zmm31` register
//! file via EVEX, so it can also use the extended registers (`zmm16..31`)
//! that VEX cannot reach; the general-register half of every kernel is
//! `x86_64.rs`'s, shared with AVX2.
//!
//! Scope: arithmetic, FMA, sqrt/recip/rsqrt, min/max, bitwise, comparisons,
//! `If`, constant broadcast, the integer bit-manipulation atoms
//! (`IAdd`/`BitAnd`/`BitOr`/`TruncToInt`/`IntToFloat`), and shifts by an
//! immediate — so the exp/log lowering reaches this backend intact.
//! A comparison is `vcmpps` into a mask register. [`Avx512`] keeps it in `k`
//! ([`Lane`]) and widens it (`vpmovm2d`) only where an instruction has no mask
//! form. Note `vpmovm2d` is AVX-512**DQ**, not F: an
//! F-only part would fault on any kernel containing a comparison.
//!
//! Transcendentals themselves are still a separate lowering stage; ops with no
//! rule here are refused up front rather than mis-emitted.
//!
//! Spills use a real stack frame (a `zmm` is 64 bytes — far past the 128-byte
//! red zone).

use super::asm::Encoding;
use super::build::{Builder, Def, Spiller, defined};
use super::regalloc::resource::FrameSlot;
use super::register_file::{EntryRegisters, Members, RegisterFile};
use super::x86_64;
use super::x86_64::{
    Alu, Direction, Disp, GENERAL, Imm32, Lanewise, Mem, NoDisp, Pred, Rounding, X86,
};
use super::{
    Bound, Class, ClassId, Edges, EncodedInst, Flags, Integer, IsaBackend, Label, LaneOp, Opmask,
    Placed, Pointer, Rebind, Selected, Spill, Stage, Store, Target, Test, Value, ValueName, Vector,
    unimplemented_op,
};
use crate::error::CompileError;
use crate::program::IfArm;
use alloc::vec::Vec;
use pixelflow_ir::OpKind;

// =============================================================================
// EVEX encoder
// =============================================================================

/// Opcode escape map (EVEX `mm`).
#[derive(Clone, Copy)]
enum Map {
    /// `0F`
    M0F = 1,
    /// `0F38`
    M0F38 = 2,
    /// `0F3A`
    M0F3A = 3,
}

/// Mandatory prefix (EVEX `pp`).
#[derive(Clone, Copy)]
enum Pp {
    /// none — packed single
    None = 0,
    /// `66`
    P66 = 1,
    /// `F3`
    F3 = 2,
}

/// The identity of one EVEX instruction: opcode map, mandatory prefix, W
/// bit, opcode byte, vector length, and the writemask. This is *which
/// instruction* — it is constant per mnemonic, so each mnemonic below states
/// it exactly once and the operand form (`rrr`/`rm`) supplies the per-call
/// parts.
///
/// The 256-bit twin is `avx2::Vex`.
#[derive(Clone, Copy)]
struct Evex {
    map: Map,
    pp: Pp,
    w: bool,
    /// `EVEX.L'L`: `10` for the `zmm` form, `00` for the few `xmm`-only
    /// instructions this tier needs (`vmovq`, `vpinsrq`).
    ll: u8,
    /// `EVEX.aaa`: the writemask register, or 0 for none. Merge-masking
    /// (`z = 0`), which for a store means the masked-off lanes are left in
    /// memory untouched.
    aaa: u8,
    opcode: u8,
}

/// `EVEX.L'L` for a 512-bit operation.
const LL_512: u8 = 0b10;

impl Evex {
    const fn new(map: Map, pp: Pp, opcode: u8) -> Self {
        Self {
            map,
            pp,
            w: false,
            ll: LL_512,
            aaa: 0,
            opcode,
        }
    }
    /// The 128-bit (`L'L = 00`) form of this instruction.
    const fn xmm(self) -> Self {
        Self { ll: 0, ..self }
    }
    /// The 64-bit-operand (`EVEX.W = 1`) form of this instruction.
    const fn w1(self) -> Self {
        Self { w: true, ..self }
    }
    /// This instruction under the writemask register numbered `k`.
    const fn masked(self, k: u8) -> Self {
        Self { aaa: k, ..self }
    }
    /// Map `0F`, no prefix — the packed-single family.
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
    /// Map `0F38`, `F3` — the mask-to-vector widening family.
    const fn m0f38_f3(opcode: u8) -> Self {
        Self::new(Map::M0F38, Pp::F3, opcode)
    }
    /// Map `0F3A`, `66` — the imm8 family (round, ternlog).
    const fn m0f3a_66(opcode: u8) -> Self {
        Self::new(Map::M0F3A, Pp::P66, opcode)
    }

    /// Attach an imm8 (`vcmpps` predicate, rounding mode, shift count,
    /// `vpternlogd` truth table); the returned value emits it after the
    /// instruction.
    const fn imm(self, imm: u8) -> EvexImm {
        EvexImm { evex: self, imm }
    }

    /// 3-operand register form: `op zmmDST, zmmSRC1, zmmSRC2`, where SRC1 is
    /// the non-destructive EVEX.vvvv source and SRC2 is the ModRM r/m. Any of
    /// `zmm0..zmm31` is valid.
    fn rrr(self, dst: u8, src1: u8, src2: u8) -> EncodedInst {
        let mut inst = EncodedInst::new();
        // EVEX stores the high register bits inverted.
        let r = ((dst >> 3) & 1) ^ 1; // ModRM.reg bit3
        let rp = ((dst >> 4) & 1) ^ 1; // ModRM.reg bit4 (R')
        let b = ((src2 >> 3) & 1) ^ 1; // ModRM.r/m bit3
        let x = ((src2 >> 4) & 1) ^ 1; // ModRM.r/m bit4 (EVEX.X extends r/m reg)
        let vvvv = (!src1) & 0x0F;
        let vp = ((src1 >> 4) & 1) ^ 1; // vvvv bit4 (V')

        self.prefix_into(
            &mut inst,
            (r << 7) | (x << 6) | (b << 5) | (rp << 4),
            vvvv,
            vp,
        );
        inst.push(0xC0 | ((dst & 7) << 3) | (src2 & 7));
        inst
    }

    /// `op zmmREG, [addr]` — the memory form used for spills, reloads,
    /// constant broadcast and the collapse loop's output store.
    ///
    /// EVEX.R/B are stored INVERTED, and X likewise: there is no index
    /// register in any of these forms, so X is always the encoded 1.
    /// (Encoding B = 0 for an `rsp` base was the spill-path bug: it set the
    /// base's bit 3, addressing r12 and faulting on a garbage pointer.)
    /// The ModRM/SIB/displacement tail is the architecture's, not EVEX's, so
    /// it comes from `x86_64::mem_operand`.
    fn rm<D: Disp>(self, reg: u8, base: u8, disp: D) -> EncodedInst {
        let mut inst = EncodedInst::new();
        let r = ((reg >> 3) & 1) ^ 1;
        let rp = ((reg >> 4) & 1) ^ 1;
        let b = ((base >> 3) & 1) ^ 1;
        let x = 1u8; // no index -> encoded 1

        self.prefix_into(
            &mut inst,
            (r << 7) | (x << 6) | (b << 5) | (rp << 4),
            0x0F,
            1,
        );
        x86_64::mem_operand_into(&mut inst, reg, base, disp);
        inst
    }

    /// `op zmmREG, [rip + disp32]`: the displacement is the last four bytes, a
    /// label field.
    fn rip(self, reg: u8) -> EncodedInst {
        let mut inst = EncodedInst::new();
        let r = ((reg >> 3) & 1) ^ 1;
        let rp = ((reg >> 4) & 1) ^ 1;
        // No index and no base: X and B are the encoded 1.
        self.prefix_into(
            &mut inst,
            (r << 7) | (1 << 6) | (1 << 5) | (rp << 4),
            0x0F,
            1,
        );
        x86_64::rip_operand_into(&mut inst, reg);
        inst
    }

    /// `op zmmREG, [base + index*4]` — the SIB form with a scaled index,
    /// which a broadcast load reads one element of a plane through. X is
    /// the index's high bit here, inverted like R and B.
    fn rm_scaled4(self, reg: u8, base: u8, index: u8) -> EncodedInst {
        let mut inst = EncodedInst::new();
        let r = ((reg >> 3) & 1) ^ 1;
        let rp = ((reg >> 4) & 1) ^ 1;
        let b = ((base >> 3) & 1) ^ 1;
        let x = ((index >> 3) & 1) ^ 1;

        self.prefix_into(
            &mut inst,
            (r << 7) | (x << 6) | (b << 5) | (rp << 4),
            0x0F,
            1,
        );
        x86_64::scaled4_operand_into(&mut inst, reg, base, index);
        inst
    }

    /// `op zmmREG{k}, [base + zmm_index*4]` — the VSIB form a gather
    /// addresses through. The index's high bits ride in X and V', inverted
    /// like R and B, and `vvvv` is unused.
    fn vsib_scaled4(self, reg: u8, base: u8, index: u8) -> EncodedInst {
        let mut inst = EncodedInst::new();
        let r = ((reg >> 3) & 1) ^ 1;
        let rp = ((reg >> 4) & 1) ^ 1;
        let b = ((base >> 3) & 1) ^ 1;
        let x = ((index >> 3) & 1) ^ 1;
        let vp = ((index >> 4) & 1) ^ 1;

        self.prefix_into(
            &mut inst,
            (r << 7) | (x << 6) | (b << 5) | (rp << 4),
            0x0F,
            vp,
        );
        x86_64::vsib4_operand_into(&mut inst, reg, base, index);
        inst
    }

    /// The 4-byte EVEX prefix plus the opcode byte, shared by every form.
    /// `reg_ext` is the assembled `R X B R'` nibble of P0; `vvvv`/`vp` are the
    /// extra-source fields. Every one of them is already inverted by the
    /// caller, as the encoding requires.
    fn prefix_into(self, inst: &mut EncodedInst, reg_ext: u8, vvvv: u8, vp: u8) {
        inst.push(0x62);
        inst.push(reg_ext | (self.map as u8));
        inst.push(((self.w as u8) << 7) | (vvvv << 3) | (1 << 2) | (self.pp as u8));
        // z=0 (merge), L'L, b(roadcast)=0, V', aaa.
        inst.push((self.ll << 5) | (vp << 3) | self.aaa);
        inst.push(self.opcode);
    }
}

/// An [`Evex`] instruction carrying its imm8.
#[derive(Clone, Copy)]
struct EvexImm {
    evex: Evex,
    imm: u8,
}

impl EvexImm {
    /// Register form with the imm8 appended.
    fn rrr(self, dst: u8, src1: u8, src2: u8) -> EncodedInst {
        let mut inst = self.evex.rrr(dst, src1, src2);
        inst.push(self.imm);
        inst
    }
}

impl Alu {
    const fn evex(self) -> Evex {
        match self {
            Alu::Add => Evex::m0f(0x58),
            Alu::Sub => Evex::m0f(0x5C),
            Alu::Mul => Evex::m0f(0x59),
            Alu::Div => Evex::m0f(0x5E),
            Alu::Min => Evex::m0f(0x5D),
            Alu::Max => Evex::m0f(0x5F),
            Alu::And => Evex::m0f(0x54),
            Alu::AndNot => Evex::m0f(0x55),
            Alu::Or => Evex::m0f(0x56),
            Alu::Xor => Evex::m0f(0x57),
            Alu::IAdd => Evex::m0f_66(0xFE),
        }
    }
}

impl Lanewise {
    /// `rsqrt` and `recip` are AVX-512F's `vrsqrt14ps` and `vrcp14ps`: EVEX
    /// has no `0F 52`/`0F 53`, and ~2^-14 relative error matches `Recip`'s
    /// "approximate reciprocal" contract on every other backend.
    const fn evex(self) -> Evex {
        match self {
            Lanewise::Sqrt => Evex::m0f(0x51),
            Lanewise::Rsqrt => Evex::m0f38_66(0x4E),
            Lanewise::Recip => Evex::m0f38_66(0x4C),
            Lanewise::ToInt => Evex::m0f_f3(0x5B),
            Lanewise::FromInt => Evex::m0f(0x5B),
            Lanewise::WidenBytes => Evex::m0f38_66(0x31),
        }
    }
}

/// `vpternlogd`'s truth table for `A ? B : C` per bit, with `A` the
/// destination (the mask), `B` the first source and `C` the second.
const TERNLOG_SELECT: u8 = 0xCA;

/// An AVX-512 instruction: EVEX, and the VEX-encoded mask-register forms.
///
/// Generic over what its operands are ([`Stage`]), like [`x86_64::Gp`] and
/// AVX2's `Inst`. EVEX is three-operand and non-destructive, so an operand may
/// be the register the result is written to; the instructions below whose
/// destination is also a source say so with a `Tie`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) enum Inst<S: Stage> {
    /// `op dst, a, b`
    Alu {
        op: Alu,
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
    /// `vrndscaleps dst, src, mode`: scale 0, so an integer. (Opcode `08` is
    /// packed-single; `09` is packed-double and needs `W1`.)
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
    /// `vfmadd231ps acc, a, b`: `acc = a·b + acc`, one rounding.
    Fma231 {
        acc: S::Tie<Vector>,
        a: S::Read<Vector>,
        b: S::Read<Vector>,
    },
    /// `vpternlogd dst, if_true, if_false, 0xCA`: `dst = dst ? if_true :
    /// if_false` per bit, `dst` holding an all-ones/all-zeros mask going in.
    Blend {
        dst: S::Tie<Vector>,
        if_true: S::Read<Vector>,
        if_false: S::Read<Vector>,
    },
    /// `vmovaps dst, src`
    Mov {
        dst: S::Write<Vector>,
        src: S::Read<Vector>,
    },
    /// `vcvttss2si dst, src`: lane 0, truncated to a 64-bit integer. EVEX
    /// rather than VEX so the source may be `zmm16..31`.
    Cvtt {
        dst: S::Write<Integer>,
        src: S::Read<Vector>,
    },
    /// `vmovq dst, src`: eight bytes into the low lanes, the rest zeroed.
    Movq {
        dst: S::Write<Vector>,
        src: S::Read<Integer>,
    },
    /// `vpinsrq dst, dst, src, 1`: eight bytes into the high half of the low
    /// 128 bits.
    InsertHigh {
        dst: S::Tie<Vector>,
        src: S::Read<Integer>,
    },
    /// `vcmpps dst, a, b, pred`: a bit where it holds, clear where it does not.
    CmpK {
        pred: Pred,
        dst: S::Write<Opmask>,
        a: S::Read<Vector>,
        b: S::Read<Vector>,
    },
    /// `vpmovm2d dst, k`: an all-ones lane where the bit is set, all-zero
    /// where it is not. AVX-512**DQ**, not F: an F-only part would fault.
    Movm2d {
        dst: S::Write<Vector>,
        k: S::Read<Opmask>,
    },
    /// `vptestmd dst, a, b`: a bit where the lanes' AND is nonzero.
    Ptestm {
        dst: S::Write<Opmask>,
        a: S::Read<Vector>,
        b: S::Read<Vector>,
    },
    /// `kortestw k, k`: ZF iff no bit is set, CF iff all sixteen are.
    KorTest {
        flags: S::Write<Flags>,
        k: S::Read<Opmask>,
    },
    /// `kmovw dst, src`: the low sixteen bits of a general register.
    KFromGpr {
        dst: S::Write<Opmask>,
        src: S::Read<Integer>,
    },
    /// `vgatherdps dst{mask}, [base + index*4]`: one `f32` per lane whose
    /// `mask` bit is set. The instruction clears the bits as it completes
    /// lanes, and `#UD`s if `dst` and `index` are one register.
    Gather {
        dst: S::Early<Vector>,
        base: S::Read<Pointer>,
        index: S::Read<Vector>,
        mask: S::Tie<Opmask>,
    },
    /// `vmovups [dst]{mask}, src`: the lanes whose bit is set, the rest of
    /// memory left untouched.
    StoreMasked {
        dst: Mem<S, NoDisp>,
        src: S::Read<Vector>,
        mask: S::Read<Opmask>,
    },
    /// `vmovups [dst], src` with no displacement: a whole batch to the
    /// address the store's arithmetic computed.
    StoreBatch {
        dst: Mem<S, NoDisp>,
        src: S::Read<Vector>,
    },
    /// `vbroadcastss dst, [src]`: one `f32`, a pool entry or a uniform, into
    /// every lane. A full `disp32`, never EVEX's compressed `disp8`, which
    /// scales the byte by the tuple's element size.
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
    /// `vxorps dst, dst, dst`: zero, whatever `dst` held. Its reads are not
    /// operands, because the result does not depend on them.
    Zero { dst: S::Write<Vector> },
    /// `vpternlogd dst, dst, dst, 0xFF`: all-ones, whatever `dst` held. Its
    /// reads are not operands, because the result does not depend on them.
    Ones { dst: S::Write<Vector> },
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
    /// `vblendmps dst{k}, if_false, if_true`: `if_true` where the bit is set,
    /// `if_false` where it is not.
    BlendK {
        dst: S::Write<Vector>,
        k: S::Read<Opmask>,
        if_true: S::Read<Vector>,
        if_false: S::Read<Vector>,
    },
    /// `kandw`/`korw dst, a, b`
    KLogic {
        op: KOp,
        dst: S::Write<Opmask>,
        a: S::Read<Opmask>,
        b: S::Read<Opmask>,
    },
    /// `kxnorw dst, k0, k0`: all sixteen bits set. `k0` is never written, so
    /// the instruction waits on nothing, where `kxnorw k, k, k` would wait on
    /// the last gather still clearing `k`. `k0` is part of the idiom's
    /// encoding, not a value: plan H8 asks for a source nothing writes.
    KOnes { dst: S::Write<Opmask> },
    /// `kmovw dst, src`
    KMov {
        dst: S::Write<Opmask>,
        src: S::Read<Opmask>,
    },
    /// `kmovw [rsp + slot], src`: the allocator's spill.
    SpillMask { src: S::Read<Opmask>, slot: S::Slot },
    /// `kmovw dst, [rsp + slot]`: the allocator's reload.
    ReloadMask {
        dst: S::Write<Opmask>,
        slot: S::Slot,
    },
}

/// The bitwise operations on mask registers that [`Inst::KLogic`] spells.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) enum KOp {
    And = 0x41,
    Or = 0x45,
}

impl KOp {
    /// The mask-register form of a bitwise op, when it has one.
    fn of(op: OpKind) -> Option<Self> {
        match op {
            OpKind::BitAnd => Some(Self::And),
            OpKind::BitOr => Some(Self::Or),
            _ => None,
        }
    }
}

/// `kxnorw`'s opcode.
const KXNOR: u8 = 0x46;

/// `k0`, which nothing writes: `kxnorw dst, k0, k0` is all-ones however it was
/// left.
const K0: u8 = 0;

/// `vpternlogd`'s truth table for all-ones.
const TERNLOG_ONES: u8 = 0xFF;

/// `kandw`-shaped: `VEX.L1.0F.W0 opcode /r`, `dst = a op b`.
fn kop(opcode: u8, dst: u8, a: u8, b: u8) -> EncodedInst {
    let vvvv = (!a & 0xF) << 3;
    EncodedInst::from_slice(&[0xC5, 0x84 | vvvv, opcode, 0xC0 | (dst << 3) | b])
}

/// `kmovw` between a mask register and `[rsp + slot]`: `VEX.L0.0F.W0 opcode
/// /r`, 91 stores and 90 loads.
fn kmem(opcode: u8, k: u8, slot: Imm32) -> EncodedInst {
    let mut inst = EncodedInst::from_slice(&[0xC5, 0xF8, opcode]);
    x86_64::mem_operand_into(&mut inst, k, x86_64::gpr::RSP, slot);
    inst
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
            Inst::Blend {
                dst,
                if_true,
                if_false,
            } => Inst::Blend {
                dst: f.tie(dst),
                if_true: f.read(*if_true),
                if_false: f.read(*if_false),
            },
            Inst::Mov { dst, src } => Inst::Mov {
                dst: f.write(dst),
                src: f.read(*src),
            },
            Inst::Cvtt { dst, src } => Inst::Cvtt {
                dst: f.write(dst),
                src: f.read(*src),
            },
            Inst::Movq { dst, src } => Inst::Movq {
                dst: f.write(dst),
                src: f.read(*src),
            },
            Inst::InsertHigh { dst, src } => Inst::InsertHigh {
                dst: f.tie(dst),
                src: f.read(*src),
            },
            Inst::CmpK { pred, dst, a, b } => Inst::CmpK {
                pred: *pred,
                dst: f.write(dst),
                a: f.read(*a),
                b: f.read(*b),
            },
            Inst::Movm2d { dst, k } => Inst::Movm2d {
                dst: f.write(dst),
                k: f.read(*k),
            },
            Inst::Ptestm { dst, a, b } => Inst::Ptestm {
                dst: f.write(dst),
                a: f.read(*a),
                b: f.read(*b),
            },
            Inst::KorTest { flags, k } => Inst::KorTest {
                flags: f.write(flags),
                k: f.read(*k),
            },
            Inst::KFromGpr { dst, src } => Inst::KFromGpr {
                dst: f.write(dst),
                src: f.read(*src),
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
            Inst::StoreMasked { dst, src, mask } => Inst::StoreMasked {
                dst: dst.walk(f),
                src: f.read(*src),
                mask: f.read(*mask),
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
            Inst::Zero { dst } => Inst::Zero { dst: f.write(dst) },
            Inst::Ones { dst } => Inst::Ones { dst: f.write(dst) },
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
            Inst::BlendK {
                dst,
                k,
                if_true,
                if_false,
            } => Inst::BlendK {
                dst: f.write(dst),
                k: f.read(*k),
                if_true: f.read(*if_true),
                if_false: f.read(*if_false),
            },
            Inst::KLogic { op, dst, a, b } => Inst::KLogic {
                op: *op,
                dst: f.write(dst),
                a: f.read(*a),
                b: f.read(*b),
            },
            Inst::KOnes { dst } => Inst::KOnes { dst: f.write(dst) },
            Inst::KMov { dst, src } => Inst::KMov {
                dst: f.write(dst),
                src: f.read(*src),
            },
            Inst::SpillMask { src, slot } => Inst::SpillMask {
                src: f.read(*src),
                slot: f.slot(*slot),
            },
            Inst::ReloadMask { dst, slot } => Inst::ReloadMask {
                dst: f.write(dst),
                slot: f.slot(*slot),
            },
        }
    }
}

impl<S: Placed> Inst<S> {
    fn encode(&self) -> EncodedInst {
        let vector = |reg: &S::Read<Vector>| S::read::<Vector>(reg);
        let write = |reg: &S::Write<Vector>| S::write::<Vector>(reg);
        let mask = |reg: &S::Read<Opmask>| S::read::<Opmask>(reg);
        let write_mask = |reg: &S::Write<Opmask>| S::write::<Opmask>(reg);
        let pointer = |reg: &S::Read<Pointer>| S::read::<Pointer>(reg);
        let slot = |slot: &S::Slot| Imm32(S::slot(slot));
        match self {
            Inst::Alu { op, dst, a, b } => op.evex().rrr(write(dst), vector(a), vector(b)),
            Inst::Unary { op, dst, src } => op.evex().rrr(write(dst), UNUSED_VVVV, vector(src)),
            Inst::Round { mode, dst, src } => {
                Evex::m0f3a_66(0x08)
                    .imm(*mode as u8)
                    .rrr(write(dst), UNUSED_VVVV, vector(src))
            }
            // The destination is `vvvv` and the `/digit` is `reg`.
            Inst::Shift {
                direction,
                dst,
                src,
                amount,
            } => Evex::m0f_66(0x72)
                .imm(*amount)
                .rrr(*direction as u8, write(dst), vector(src)),
            Inst::Fma231 { acc, a, b } => {
                Evex::m0f38_66(0xB8).rrr(S::tie::<Vector>(acc), vector(a), vector(b))
            }
            Inst::Blend {
                dst,
                if_true,
                if_false,
            } => Evex::m0f3a_66(0x25).imm(TERNLOG_SELECT).rrr(
                S::tie::<Vector>(dst),
                vector(if_true),
                vector(if_false),
            ),
            Inst::Mov { dst, src } => Evex::m0f(0x28).rrr(write(dst), UNUSED_VVVV, vector(src)),
            Inst::Cvtt { dst, src } => {
                Evex::m0f_f3(0x2C)
                    .w1()
                    .rrr(S::write::<Integer>(dst), UNUSED_VVVV, vector(src))
            }
            Inst::Movq { dst, src } => {
                Evex::m0f_66(0x6E)
                    .w1()
                    .xmm()
                    .rrr(write(dst), UNUSED_VVVV, S::read::<Integer>(src))
            }
            Inst::InsertHigh { dst, src } => {
                let dst = S::tie::<Vector>(dst);
                Evex::m0f3a_66(0x22)
                    .w1()
                    .xmm()
                    .imm(1)
                    .rrr(dst, dst, S::read::<Integer>(src))
            }
            Inst::CmpK { pred, dst, a, b } => {
                Evex::m0f(0xC2)
                    .imm(*pred as u8)
                    .rrr(write_mask(dst), vector(a), vector(b))
            }
            Inst::Movm2d { dst, k } => Evex::m0f38_f3(0x38).rrr(write(dst), UNUSED_VVVV, mask(k)),
            Inst::Ptestm { dst, a, b } => {
                Evex::m0f38_66(0x27).rrr(write_mask(dst), vector(a), vector(b))
            }
            // `VEX.L0.0F.W0 98 /r`, both operands the one mask register.
            Inst::KorTest { k, .. } => {
                let k = mask(k);
                EncodedInst::from_slice(&[0xC5, 0xF8, 0x98, 0xC0 | (k << 3) | k])
            }
            // `VEX.L0.0F.W0 92 /r`, in the three-byte prefix.
            Inst::KFromGpr { dst, src } => {
                let src = S::read::<Integer>(src);
                // B̄ is inverted: set when the source needs no extension bit.
                let no_extension = if src < 8 { 0x20 } else { 0x00 };
                let modrm = 0xC0 | ((write_mask(dst) & 7) << 3) | (src & 7);
                EncodedInst::from_slice(&[0xC4, 0xC1 | no_extension, 0x78, 0x92, modrm])
            }
            Inst::Gather {
                dst,
                base,
                index,
                mask,
            } => {
                let (dst, index) = (S::early::<Vector>(dst), vector(index));
                debug_assert!(dst != index, "vgatherdps: dst and index must differ");
                // `base` is never `rbp`/`r13` (`GENERAL` excludes them), so
                // the SIB's no-base encoding is unreachable.
                Evex::m0f38_66(0x92)
                    .masked(S::tie::<Opmask>(mask))
                    .vsib_scaled4(dst, pointer(base), index)
            }
            Inst::StoreMasked { dst, src, mask: k } => {
                Evex::m0f(0x11)
                    .masked(mask(k))
                    .rm(vector(src), pointer(&dst.base), dst.disp)
            }
            Inst::StoreBatch { dst, src } => {
                Evex::m0f(0x11).rm(vector(src), pointer(&dst.base), dst.disp)
            }
            Inst::Broadcast { dst, src } => {
                Evex::m0f38_66(0x18).rm(write(dst), pointer(&src.base), src.disp)
            }
            Inst::BroadcastIndexed { dst, base, index } => Evex::m0f38_66(0x18).rm_scaled4(
                write(dst),
                pointer(base),
                S::read::<Integer>(index),
            ),
            Inst::Zero { dst } => {
                let dst = write(dst);
                Alu::Xor.evex().rrr(dst, dst, dst)
            }
            Inst::Ones { dst } => {
                let dst = write(dst);
                Evex::m0f3a_66(0x25).imm(TERNLOG_ONES).rrr(dst, dst, dst)
            }
            Inst::LoadConst { dst, .. } => Evex::m0f38_66(0x18).rip(write(dst)),
            Inst::SpillVector { src: v, slot: s } => {
                Evex::m0f(0x11).rm(vector(v), x86_64::gpr::RSP, slot(s))
            }
            Inst::ReloadVector { dst, slot: s } => {
                Evex::m0f(0x10).rm(write(dst), x86_64::gpr::RSP, slot(s))
            }
            // `vvvv` is the lanes the mask leaves, and `rm` the ones it takes.
            Inst::BlendK {
                dst,
                k,
                if_true,
                if_false,
            } => Evex::m0f38_66(0x65).masked(mask(k)).rrr(
                write(dst),
                vector(if_false),
                vector(if_true),
            ),
            Inst::KLogic { op, dst, a, b } => kop(*op as u8, write_mask(dst), mask(a), mask(b)),
            Inst::KOnes { dst } => kop(KXNOR, write_mask(dst), K0, K0),
            Inst::KMov { dst, src } => EncodedInst::from_slice(&[
                0xC5,
                0xF8,
                0x90,
                0xC0 | (write_mask(dst) << 3) | mask(src),
            ]),
            Inst::SpillMask { src, slot: s } => kmem(0x91, mask(src), slot(s)),
            Inst::ReloadMask { dst, slot: s } => kmem(0x90, write_mask(dst), slot(s)),
        }
    }

    /// The instruction's bytes, and the label fields in them.
    fn assemble(&self, out: &mut Encoding<'_>) {
        let inst = self.encode();
        out.bytes(inst.as_bytes());
        if let Inst::LoadConst { at, .. } = self {
            // The displacement is the instruction's last four bytes.
            out.field(inst.as_bytes().len() - 4, *at, x86_64::patch_rel32);
        }
    }
}

/// What the selection pipeline builds for AVX-512: a vector instruction or a
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

impl<'m> Op<Bound<'m, Avx512>> {
    fn assemble(&self, out: &mut Encoding<'_>) {
        match self {
            Op::Vector(inst) => inst.assemble(out),
            Op::General(gp) => gp.assemble(out),
        }
    }
}

/// Sentinel for the EVEX `vvvv`/`V'` source field on instructions that have no
/// second source (2-operand forms): the field must read as *unused*, which the
/// hardware encodes as `vvvv = 1111` AND `V' = 1`. In `evex_rrr` both are
/// derived from the `src1` index by inversion, so the index that yields
/// `vvvv=1111, V'=1` is **0** (not 0x1F — that has bit4 set, giving `V'=0` and a
/// `#UD` / SIGILL).
const UNUSED_VVVV: u8 = 0;

// =============================================================================
// The store, and the iota
// =============================================================================

/// The bytes `0..8` and `8..16`, little end first: what two `movabs` carry
/// in for `vpmovzxbd` to widen into the iota.
const IOTA_BYTES: [u64; 2] = [0x0706_0504_0302_0100, 0x0F0E_0D0C_0B0A_0908];

// =============================================================================
// Masks & `If` — a mask is a bit per lane in the opmask file ([`Lane`]), or,
// where an instruction has no `k` form, an ordinary vector (all-ones /
// all-zeros lanes) in the zmm file, exactly like NEON. It flows through the
// allocator as a normal value.
// =============================================================================

/// Map a comparison `OpKind` to its `vcmpps` predicate.
fn cmp_pred(op: OpKind) -> Option<Pred> {
    Some(match op {
        OpKind::Eq => Pred::Eq,
        OpKind::Ne => Pred::Ne,
        OpKind::Lt => Pred::Lt,
        OpKind::Le => Pred::Le,
        OpKind::Gt => Pred::Nle,
        OpKind::Ge => Pred::Ge,
        _ => return None,
    })
}

/// The vector ALU instruction of a non-comparison binary op.
fn alu(op: OpKind) -> Alu {
    match op {
        OpKind::Add => Alu::Add,
        OpKind::Sub => Alu::Sub,
        OpKind::Mul => Alu::Mul,
        OpKind::Div => Alu::Div,
        OpKind::Min => Alu::Min,
        OpKind::Max => Alu::Max,
        OpKind::BitAnd => Alu::And,
        OpKind::BitOr => Alu::Or,
        // Integer add on lane bit patterns (exp/log exponent arithmetic).
        OpKind::IAdd => Alu::IAdd,
        _ => unimplemented_op("avx-512", op),
    }
}

fn shift_direction(op: OpKind) -> Direction {
    match op {
        OpKind::Shl => Direction::Left,
        OpKind::Shr => Direction::Right,
        _ => unimplemented_op("avx-512", op),
    }
}

/// The one instruction a unary op is.
enum UnaryForm {
    Lanewise(Lanewise),
    Round(Rounding),
    /// `Neg` and `Abs` xor or and the sign bit, which is a constant to hold in
    /// a register first: all-ones shifted by `shift`.
    Signed {
        op: Alu,
        shift: (Direction, u8),
    },
}

fn unary_form(op: OpKind) -> UnaryForm {
    match op {
        OpKind::Sqrt => UnaryForm::Lanewise(Lanewise::Sqrt),
        OpKind::Neg => UnaryForm::Signed {
            op: Alu::Xor,
            shift: (Direction::Left, 31),
        },
        OpKind::Abs => UnaryForm::Signed {
            op: Alu::And,
            shift: (Direction::Right, 1),
        },
        OpKind::Floor => UnaryForm::Round(Rounding::Floor),
        OpKind::Ceil => UnaryForm::Round(Rounding::Ceil),
        OpKind::Round => UnaryForm::Round(Rounding::Nearest),
        OpKind::Recip => UnaryForm::Lanewise(Lanewise::Recip),
        OpKind::Rsqrt => UnaryForm::Lanewise(Lanewise::Rsqrt),
        // Int/float domain crossings, exactly the hardware's cvttps2dq /
        // cvtdq2ps — the primitives exp/log lower to.
        OpKind::TruncToInt => UnaryForm::Lanewise(Lanewise::ToInt),
        OpKind::IntToFloat => UnaryForm::Lanewise(Lanewise::FromInt),
        _ => unimplemented_op("avx-512", op),
    }
}

// =============================================================================
// Selection
// =============================================================================

/// The AVX-512 machine, as the selection pipeline sees it.
pub(in crate::emit) enum Avx512 {}

/// One IR lane value on this machine. A comparison, and the `kandw`/`korw` of
/// two of them, is a bit per lane in the opmask file; everything else is in the
/// vector file. Where an instruction reads the other file's lane it is first
/// widened (`vpmovm2d`), or, when it has a form for the lane where it is
/// (`vblendmps`, `kortestw`), it reads it there.
#[derive(Copy, Clone)]
pub(super) enum Lane {
    Vector(Value<Vector>),
    Opmask(Value<Opmask>),
}

impl Lane {
    /// For an instruction with no `k` form: this lane in the vector file, all
    /// ones where its bit is set.
    fn vector(self, b: &mut Builder<Avx512>) -> Value<Vector> {
        match self {
            Lane::Vector(vector) => vector,
            Lane::Opmask(k) => emit(b, |dst| Inst::Movm2d { dst, k }),
        }
    }
}

/// Define a value of class `C` with `make`, which builds the instruction that
/// writes it.
fn emit<C: Class>(
    b: &mut Builder<Avx512>,
    make: impl FnOnce(Def<C>) -> Inst<Selected>,
) -> Value<C> {
    let dst = b.def();
    let value = dst.value();
    b.push(Op::Vector(make(dst)));
    value
}

/// `value` splatted: the zero and all-ones idioms, else one pool entry.
fn constant(b: &mut Builder<Avx512>, value: f32) -> Result<Value<Vector>, CompileError> {
    Ok(match value.to_bits() {
        0 => emit(b, |dst| Inst::Zero { dst }),
        u32::MAX => emit(b, |dst| Inst::Ones { dst }),
        bits => {
            let at = b.constant(bits)?.label;
            emit(b, |dst| Inst::LoadConst { dst, at })
        }
    })
}

/// The sign bit, or all that is below it, as a shift of all-ones: two
/// instructions and no read of the pool, where a loop would read it every trip.
fn sign_mask(b: &mut Builder<Avx512>, (direction, amount): (Direction, u8)) -> Value<Vector> {
    let ones = emit(b, |dst| Inst::Ones { dst });
    emit(b, |dst| Inst::Shift {
        direction,
        dst,
        src: ones,
        amount,
    })
}

impl X86 for Avx512 {
    fn general(inst: x86_64::Gp<Selected>) -> Op<Selected> {
        Op::General(inst)
    }
}

/// The `Lane` a block parameter of this class is.
fn lane_of_class(param: ValueName) -> Lane {
    match param.class {
        ClassId::Vector => Lane::Vector(param.typed()),
        ClassId::Opmask => Lane::Opmask(param.typed()),
        class => unreachable!("{class:?} holds no lane"),
    }
}

impl IsaBackend for Avx512 {
    type Inst<S: Stage> = Op<S>;
    type Constant = u32;
    type Anchors = ();
    type Lane = Lane;

    // SysV has no callee-saved vector registers and the collapse ABI passes
    // no vector, so all thirty-two `zmm` are the allocator's. `k0` is not a
    // member: an EVEX `aaa` of 0 means "no mask".
    const FILE: RegisterFile = RegisterFile::new(
        Members {
            vector: &[
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                23, 24, 25, 26, 27, 28, 29, 30, 31,
            ],
            general: GENERAL,
            opmask: &[1, 2, 3, 4, 5, 6, 7],
            flags: &[0],
        },
        EntryRegisters {
            ctx: x86_64::gpr::RDI,
            out: x86_64::gpr::RSI,
            pitch: x86_64::gpr::RDX,
        },
        64,
    );
    const POOL_REACH: u64 = i32::MAX as u64 / 4;

    fn lane(b: &mut Builder<Self>, op: LaneOp<Self>) -> Result<Lane, CompileError> {
        Ok(match op {
            LaneOp::Const(value) => Lane::Vector(constant(b, value)?),
            // The bytes `0..16` in through two general registers, widened to
            // dwords, converted.
            LaneOp::Lanes => {
                let [low, high] = IOTA_BYTES;
                let low = x86_64::define(b, |dst| x86_64::Gp::Movabs { dst, imm: low });
                let packed = emit(b, |dst| Inst::Movq { dst, src: low });
                let high = x86_64::define(b, |dst| x86_64::Gp::Movabs { dst, imm: high });
                let joined = b.tie(packed);
                let halves = joined.write();
                b.push(Op::Vector(Inst::InsertHigh {
                    dst: joined,
                    src: high,
                }));
                let widened = emit(b, |dst| Inst::Unary {
                    op: Lanewise::WidenBytes,
                    dst,
                    src: halves,
                });
                Lane::Vector(emit(b, |dst| Inst::Unary {
                    op: Lanewise::FromInt,
                    dst,
                    src: widened,
                }))
            }
            LaneOp::Unary(op, src) => {
                let src = src.vector(b);
                Lane::Vector(match unary_form(op) {
                    UnaryForm::Lanewise(op) => emit(b, |dst| Inst::Unary { op, dst, src }),
                    UnaryForm::Round(mode) => emit(b, |dst| Inst::Round { mode, dst, src }),
                    UnaryForm::Signed { op, shift } => {
                        let mask = sign_mask(b, shift);
                        emit(b, |dst| Inst::Alu {
                            op,
                            dst,
                            a: src,
                            b: mask,
                        })
                    }
                })
            }
            LaneOp::Binary(op, a, c) => match (KOp::of(op), a, c) {
                (Some(op), Lane::Opmask(x), Lane::Opmask(y)) => {
                    Lane::Opmask(emit(b, |dst| Inst::KLogic {
                        op,
                        dst,
                        a: x,
                        b: y,
                    }))
                }
                _ => {
                    let (a, c) = (a.vector(b), c.vector(b));
                    match cmp_pred(op) {
                        Some(pred) => {
                            Lane::Opmask(emit(b, |dst| Inst::CmpK { pred, dst, a, b: c }))
                        }
                        None => Lane::Vector(emit(b, |dst| Inst::Alu {
                            op: alu(op),
                            dst,
                            a,
                            b: c,
                        })),
                    }
                }
            },
            LaneOp::MulAdd(x, y, addend) => {
                let (x, y, addend) = (x.vector(b), y.vector(b), addend.vector(b));
                let acc = b.tie(addend);
                let result = acc.write();
                b.push(Op::Vector(Inst::Fma231 { acc, a: x, b: y }));
                Lane::Vector(result)
            }
            // `cond ? if_true : if_false`: `vblendmps` under a mask register,
            // else `vpternlogd` bit by bit.
            LaneOp::Blend {
                cond,
                if_true,
                if_false,
            } => {
                let (if_true, if_false) = (if_true.vector(b), if_false.vector(b));
                Lane::Vector(match cond {
                    Lane::Opmask(k) => emit(b, |dst| Inst::BlendK {
                        dst,
                        k,
                        if_true,
                        if_false,
                    }),
                    Lane::Vector(mask) => {
                        let dst = b.tie(mask);
                        let blended = dst.write();
                        b.push(Op::Vector(Inst::Blend {
                            dst,
                            if_true,
                            if_false,
                        }));
                        blended
                    }
                })
            }
            LaneOp::Shift(op, src, amount) => {
                let src = src.vector(b);
                Lane::Vector(emit(b, |dst| Inst::Shift {
                    direction: shift_direction(op),
                    dst,
                    src,
                    amount,
                }))
            }
            // `vbroadcastss dst, [block + 4·element]`.
            LaneOp::Uniform { base, element } => {
                let disp = x86_64::displacement(element, x86_64::F32_BYTES)?;
                Lane::Vector(emit(b, |dst| Inst::Broadcast {
                    dst,
                    src: Mem { base, disp },
                }))
            }
            // `vgatherdps` under an all-ones mask: the indices are floats, so
            // they are truncated first, and the instruction clears the mask
            // as it completes lanes.
            LaneOp::Gather { base, index } => {
                let index = index.vector(b);
                let indices = emit(b, |dst| Inst::Unary {
                    op: Lanewise::ToInt,
                    dst,
                    src: index,
                });
                let ones = emit(b, |dst| Inst::KOnes { dst });
                let (dst, mask) = (b.early(), b.tie(ones));
                let gathered = dst.value();
                b.push(Op::Vector(Inst::Gather {
                    dst,
                    base,
                    index: indices,
                    mask,
                }));
                Lane::Vector(gathered)
            }
            // One element, the same in every lane: the index is lane 0,
            // truncated.
            LaneOp::Broadcast { base, index } => {
                let index = index.vector(b);
                let index = emit(b, |dst| Inst::Cvtt { dst, src: index });
                Lane::Vector(emit(b, |dst| Inst::BroadcastIndexed { dst, base, index }))
            }
        })
    }

    fn lane_name(lane: Lane) -> ValueName {
        match lane {
            Lane::Vector(vector) => vector.name(),
            Lane::Opmask(k) => k.name(),
        }
    }

    fn param_lane(param: ValueName) -> Lane {
        lane_of_class(param)
    }

    fn context(
        b: &mut Builder<Self>,
        ctx: Value<Pointer>,
        slot: u64,
    ) -> Result<Value<Pointer>, CompileError> {
        x86_64::context(b, ctx, slot)
    }

    /// `out + 4·(row·pitch + col)`: each index is lane 0 of its binder,
    /// truncated. A full batch is one `vmovups`; a remainder is the same
    /// store under a writemask of its lanes.
    fn store(b: &mut Builder<Self>, store: Store<Self>) {
        let Store {
            out,
            pitch,
            row,
            col,
            value,
            lanes,
        } = store;
        let (row, col, value) = (row.vector(b), col.vector(b), value.vector(b));
        let row = emit(b, |dst| Inst::Cvtt { dst, src: row });
        let col = emit(b, |dst| Inst::Cvtt { dst, src: col });
        let base = x86_64::element_address(b, out, pitch, (row, col));
        let dst = Mem { base, disp: NoDisp };
        if u64::from(lanes) == Self::FILE.vector_bytes() / 4 {
            b.push(Op::Vector(Inst::StoreBatch { dst, src: value }));
            return;
        }
        let bits = x86_64::define(b, |dst| x86_64::Gp::MovImm32 {
            dst,
            imm: (1u32 << lanes) - 1,
        });
        let mask = emit(b, |dst| Inst::KFromGpr { dst, src: bits });
        b.push(Op::Vector(Inst::StoreMasked {
            dst,
            src: value,
            mask,
        }));
    }

    /// `kortestw` of the mask (`vptestmd` first when it is a vector), then
    /// `je` (ZF: no lane is set) or `jb` (CF: all sixteen are).
    fn branch(b: &mut Builder<Self>, test: Test<Self>, edges: Edges) {
        let k = match test.cond {
            Lane::Opmask(k) => k,
            Lane::Vector(mask) => emit(b, |dst| Inst::Ptestm {
                dst,
                a: mask,
                b: mask,
            }),
        };
        let flags = b.def();
        let tested = flags.value();
        b.push(Op::Vector(Inst::KorTest { flags, k }));
        let cond = match test.dead {
            IfArm::True => x86_64::Cond::E,
            IfArm::False => x86_64::Cond::B,
        };
        x86_64::branch(b, cond, tested, edges);
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

    fn copy<C: Spill>(b: &mut Spiller<'_, Self>, src: Value<C>) -> Value<C> {
        match C::ID {
            ClassId::Vector => {
                let (dst, copy) = defined::<C, Vector>(b);
                b.push(Op::Vector(Inst::Mov {
                    dst,
                    src: src.name().typed(),
                }));
                copy
            }
            ClassId::Opmask => {
                let (dst, copy) = defined::<C, Opmask>(b);
                b.push(Op::Vector(Inst::KMov {
                    dst,
                    src: src.name().typed(),
                }));
                copy
            }
            ClassId::Pointer | ClassId::Integer => x86_64::copy_general(b, src),
            ClassId::Flags => unreachable!("the flags are not copied"),
        }
    }

    fn spill<C: Spill>(b: &mut Spiller<'_, Self>, src: Value<C>, slot: &FrameSlot) {
        match C::ID {
            ClassId::Vector => b.push(Op::Vector(Inst::SpillVector {
                src: src.name().typed(),
                slot: slot.name(),
            })),
            ClassId::Opmask => b.push(Op::Vector(Inst::SpillMask {
                src: src.name().typed(),
                slot: slot.name(),
            })),
            ClassId::Pointer | ClassId::Integer => x86_64::spill_general(b, src, slot),
            ClassId::Flags => unreachable!("the flags are not spilled"),
        }
    }

    fn reload<C: Spill>(b: &mut Spiller<'_, Self>, slot: &FrameSlot) -> Value<C> {
        match C::ID {
            ClassId::Vector => {
                let (dst, reloaded) = defined::<C, Vector>(b);
                b.push(Op::Vector(Inst::ReloadVector {
                    dst,
                    slot: slot.name(),
                }));
                reloaded
            }
            ClassId::Opmask => {
                let (dst, reloaded) = defined::<C, Opmask>(b);
                b.push(Op::Vector(Inst::ReloadMask {
                    dst,
                    slot: slot.name(),
                }));
                reloaded
            }
            ClassId::Pointer | ClassId::Integer => x86_64::reload_general(b, slot),
            ClassId::Flags => unreachable!("the flags are not reloaded"),
        }
    }

    fn rematerializable(inst: &Op<Selected>) -> bool {
        matches!(
            inst,
            Op::Vector(
                Inst::LoadConst { .. } | Inst::Zero { .. } | Inst::Ones { .. } | Inst::KOnes { .. }
            )
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
