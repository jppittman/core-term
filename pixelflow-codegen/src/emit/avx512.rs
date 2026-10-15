//! x86-64 AVX-512 (EVEX) JIT encoder — 512-bit, 16-lane `zmm` kernels.
//!
//! The widest of the x86-64 tiers (`crate::isa`), above the AVX2 VEX
//! encoders (`avx2.rs`, 256-bit). It targets the full `zmm0..zmm31` register
//! file via EVEX, so it can also use the extended registers (`zmm16..31`)
//! that VEX cannot reach; the general-register half of every kernel and the
//! vector instructions both tiers have are `x86_64.rs`'s, shared with AVX2.
//! What is this file's is the EVEX prefix, the instructions only AVX-512 has
//! ([`Extra`]), and how it selects.
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
//! Transcendentals are expanded by `legalize` before any backend sees them;
//! ops with no rule here are refused up front rather than mis-emitted.
//!
//! Spills use a real stack frame (a `zmm` is 64 bytes — far past the 128-byte
//! red zone).

use super::asm::Encoding;
use super::build::{Builder, Def, Spiller};
use super::regalloc::resource::FrameSlot;
use super::register_file::{EntryRegisters, Members, RegisterFile};
use super::x86_64;
use super::x86_64::{
    Alu, Direction, Evex, EvexPrefix, GENERAL, Imm32, Lanewise, Mem, NoDisp, Pred, Rounding, Vex,
    X86,
};
use super::{
    Bound, Class, ClassId, Edges, EncodedInst, Flags, Integer, IsaBackend, Label, LaneOp, Opmask,
    Pointer, Rebind, Selected, Stage, Store, Target, Test, Value, ValueName, Vector, slot_disp,
    unimplemented_op,
};
use crate::error::CompileError;
use crate::program::IfArm;
use alloc::vec::Vec;
use pixelflow_ir::OpKind;

/// `vpternlogd`'s truth table for `A ? B : C` per bit, with `A` the
/// destination (the mask), `B` the first source and `C` the second.
const TERNLOG_SELECT: u8 = 0xCA;

type Inst<S> = x86_64::Inst<S, Avx512>;
type Op<S> = x86_64::Op<S, Avx512>;

/// The instructions only AVX-512 has: EVEX forms with no VEX twin, and the
/// VEX-encoded mask-register forms. A destination that is also a source says
/// so with a `Tie`.
pub(super) enum Extra<S: Stage> {
    /// `vpternlogd dst, if_true, if_false, 0xCA`: `dst = dst ? if_true :
    /// if_false` per bit, `dst` holding an all-ones/all-zeros mask going in.
    Blend {
        dst: S::Tie<Vector>,
        if_true: S::Read<Vector>,
        if_false: S::Read<Vector>,
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

/// The bitwise operations on mask registers that [`Extra::KLogic`] spells.
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
            Lane::Opmask(k) => emit(b, |dst| Inst::Extra(Extra::Movm2d { dst, k })),
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
    type Prefix = EvexPrefix;
    type Extra<S: Stage> = Extra<S>;

    fn general(inst: x86_64::Gp<Selected>) -> Op<Selected> {
        Op::General(inst)
    }

    /// `vpternlogd dst, dst, dst, 0xFF`
    fn ones(dst: u8) -> EncodedInst {
        Evex::m0f3a_66(0x25).imm(TERNLOG_ONES).rrr(dst, dst, dst)
    }

    fn walk_extra<To: Stage>(extra: &Extra<Selected>, f: &mut impl Rebind<To>) -> Extra<To> {
        match extra {
            Extra::Blend {
                dst,
                if_true,
                if_false,
            } => Extra::Blend {
                dst: f.tie(dst),
                if_true: f.read(*if_true),
                if_false: f.read(*if_false),
            },
            Extra::InsertHigh { dst, src } => Extra::InsertHigh {
                dst: f.tie(dst),
                src: f.read(*src),
            },
            Extra::CmpK { pred, dst, a, b } => Extra::CmpK {
                pred: *pred,
                dst: f.write(dst),
                a: f.read(*a),
                b: f.read(*b),
            },
            Extra::Movm2d { dst, k } => Extra::Movm2d {
                dst: f.write(dst),
                k: f.read(*k),
            },
            Extra::Ptestm { dst, a, b } => Extra::Ptestm {
                dst: f.write(dst),
                a: f.read(*a),
                b: f.read(*b),
            },
            Extra::KorTest { flags, k } => Extra::KorTest {
                flags: f.write(flags),
                k: f.read(*k),
            },
            Extra::KFromGpr { dst, src } => Extra::KFromGpr {
                dst: f.write(dst),
                src: f.read(*src),
            },
            Extra::Gather {
                dst,
                base,
                index,
                mask,
            } => Extra::Gather {
                dst: f.early(dst),
                base: f.read(*base),
                index: f.read(*index),
                mask: f.tie(mask),
            },
            Extra::StoreMasked { dst, src, mask } => Extra::StoreMasked {
                dst: dst.walk(f),
                src: f.read(*src),
                mask: f.read(*mask),
            },
            Extra::BlendK {
                dst,
                k,
                if_true,
                if_false,
            } => Extra::BlendK {
                dst: f.write(dst),
                k: f.read(*k),
                if_true: f.read(*if_true),
                if_false: f.read(*if_false),
            },
            Extra::KLogic { op, dst, a, b } => Extra::KLogic {
                op: *op,
                dst: f.write(dst),
                a: f.read(*a),
                b: f.read(*b),
            },
            Extra::KOnes { dst } => Extra::KOnes { dst: f.write(dst) },
            Extra::KMov { dst, src } => Extra::KMov {
                dst: f.write(dst),
                src: f.read(*src),
            },
            Extra::SpillMask { src, slot } => Extra::SpillMask {
                src: f.read(*src),
                slot: f.slot(*slot),
            },
            Extra::ReloadMask { dst, slot } => Extra::ReloadMask {
                dst: f.write(dst),
                slot: f.slot(*slot),
            },
        }
    }

    fn encode_extra(extra: &Extra<Bound<'_, Self>>) -> EncodedInst {
        match extra {
            Extra::Blend {
                dst,
                if_true,
                if_false,
            } => Evex::m0f3a_66(0x25).imm(TERNLOG_SELECT).rrr(
                dst.number(),
                if_true.number(),
                if_false.number(),
            ),
            Extra::InsertHigh { dst, src } => {
                let dst = dst.number();
                Evex::m0f3a_66(0x22)
                    .w1()
                    .xmm()
                    .imm(1)
                    .rrr(dst, dst, src.number())
            }
            Extra::CmpK { pred, dst, a, b } => {
                Evex::m0f(0xC2)
                    .imm(*pred as u8)
                    .rrr(dst.number(), a.number(), b.number())
            }
            Extra::Movm2d { dst, k } => Evex::m0f38_f3(0x38).rr(dst.number(), k.number()),
            Extra::Ptestm { dst, a, b } => {
                Evex::m0f38_66(0x27).rrr(dst.number(), a.number(), b.number())
            }
            // Both operands the one mask register.
            Extra::KorTest { k, .. } => Vex::m0f(0x98).xmm().rr(k.number(), k.number()),
            Extra::KFromGpr { dst, src } => Vex::m0f(0x92).xmm().rr(dst.number(), src.number()),
            Extra::Gather {
                dst,
                base,
                index,
                mask,
            } => {
                let (dst, index) = (dst.number(), index.number());
                debug_assert!(dst != index, "vgatherdps: dst and index must differ");
                // `base` is never `rbp`/`r13` (`GENERAL` excludes them), so
                // the SIB's no-base encoding is unreachable.
                Evex::m0f38_66(0x92).masked(mask.number()).vsib_scaled4(
                    dst,
                    None,
                    base.number(),
                    index,
                )
            }
            Extra::StoreMasked { dst, src, mask: k } => {
                Evex::m0f(0x11)
                    .masked(k.number())
                    .rm(src.number(), dst.base.number(), dst.disp)
            }
            // `vvvv` is the lanes the mask leaves, and `rm` the ones it takes.
            Extra::BlendK {
                dst,
                k,
                if_true,
                if_false,
            } => Evex::m0f38_66(0x65).masked(k.number()).rrr(
                dst.number(),
                if_false.number(),
                if_true.number(),
            ),
            Extra::KLogic { op, dst, a, b } => {
                Vex::m0f(*op as u8).rrr(dst.number(), a.number(), b.number())
            }
            Extra::KOnes { dst } => Vex::m0f(KXNOR).rrr(dst.number(), K0, K0),
            Extra::KMov { dst, src } => Vex::m0f(0x90).xmm().rr(dst.number(), src.number()),
            Extra::SpillMask { src, slot } => {
                Vex::m0f(0x91)
                    .xmm()
                    .rm(src.number(), x86_64::gpr::RSP, Imm32(slot_disp(slot)))
            }
            Extra::ReloadMask { dst, slot } => {
                Vex::m0f(0x90)
                    .xmm()
                    .rm(dst.number(), x86_64::gpr::RSP, Imm32(slot_disp(slot)))
            }
        }
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
                b.push(Op::Vector(Inst::Extra(Extra::InsertHigh {
                    dst: joined,
                    src: high,
                })));
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
                (Some(op), Lane::Opmask(x), Lane::Opmask(y)) => Lane::Opmask(emit(b, |dst| {
                    Inst::Extra(Extra::KLogic {
                        op,
                        dst,
                        a: x,
                        b: y,
                    })
                })),
                _ => {
                    let (a, c) = (a.vector(b), c.vector(b));
                    match cmp_pred(op) {
                        Some(pred) => Lane::Opmask(emit(b, |dst| {
                            Inst::Extra(Extra::CmpK { pred, dst, a, b: c })
                        })),
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
                    Lane::Opmask(k) => emit(b, |dst| {
                        Inst::Extra(Extra::BlendK {
                            dst,
                            k,
                            if_true,
                            if_false,
                        })
                    }),
                    Lane::Vector(mask) => {
                        let dst = b.tie(mask);
                        let blended = dst.write();
                        b.push(Op::Vector(Inst::Extra(Extra::Blend {
                            dst,
                            if_true,
                            if_false,
                        })));
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
                let ones = emit(b, |dst| Inst::Extra(Extra::KOnes { dst }));
                let (dst, mask) = (b.early(), b.tie(ones));
                let gathered = dst.value();
                b.push(Op::Vector(Inst::Extra(Extra::Gather {
                    dst,
                    base,
                    index: indices,
                    mask,
                })));
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
        Lane::Vector(param.typed())
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
        let mask = emit(b, |dst| Inst::Extra(Extra::KFromGpr { dst, src: bits }));
        b.push(Op::Vector(Inst::Extra(Extra::StoreMasked {
            dst,
            src: value,
            mask,
        })));
    }

    /// `kortestw` of the mask (`vptestmd` first when it is a vector), then
    /// `je` (ZF: no lane is set) or `jb` (CF: all sixteen are).
    fn branch(b: &mut Builder<Self>, test: Test<Self>, edges: Edges) {
        let k = match test.cond {
            Lane::Opmask(k) => k,
            Lane::Vector(mask) => emit(b, |dst| {
                Inst::Extra(Extra::Ptestm {
                    dst,
                    a: mask,
                    b: mask,
                })
            }),
        };
        let flags = b.def();
        let tested = flags.value();
        b.push(Op::Vector(Inst::Extra(Extra::KorTest { flags, k })));
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

    fn copy(b: &mut Spiller<'_, Self>, src: ValueName) -> ValueName {
        match src.class {
            ClassId::Vector => b.define::<Vector>(|dst| {
                Op::Vector(Inst::Mov {
                    dst,
                    src: src.typed(),
                })
            }),
            ClassId::Opmask => b.define::<Opmask>(|dst| {
                Op::Vector(Inst::Extra(Extra::KMov {
                    dst,
                    src: src.typed(),
                }))
            }),
            ClassId::Pointer | ClassId::Integer => x86_64::copy_general(b, src),
            ClassId::Flags => unreachable!("the flags are not copied"),
        }
    }

    fn spill(b: &mut Spiller<'_, Self>, src: ValueName, slot: &FrameSlot) {
        match src.class {
            ClassId::Vector => b.push(Op::Vector(Inst::SpillVector {
                src: src.typed(),
                slot: slot.name(),
            })),
            ClassId::Opmask => b.push(Op::Vector(Inst::Extra(Extra::SpillMask {
                src: src.typed(),
                slot: slot.name(),
            }))),
            ClassId::Pointer | ClassId::Integer => x86_64::spill_general(b, src, slot),
            ClassId::Flags => unreachable!("the flags are not spilled"),
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
            ClassId::Opmask => b.define::<Opmask>(|dst| {
                Op::Vector(Inst::Extra(Extra::ReloadMask {
                    dst,
                    slot: slot.name(),
                }))
            }),
            ClassId::Pointer | ClassId::Integer => x86_64::reload_general(b, class, slot),
            ClassId::Flags => unreachable!("the flags are not reloaded"),
        }
    }

    fn rematerializable(inst: &Op<Selected>) -> bool {
        matches!(
            inst,
            Op::Vector(
                Inst::LoadConst { .. }
                    | Inst::Zero { .. }
                    | Inst::Ones { .. }
                    | Inst::Extra(Extra::KOnes { .. })
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
