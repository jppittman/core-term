//! x86-64 AVX2 (VEX.256) JIT encoder — 256-bit, 8-lane `ymm` kernels.
//!
//! The floor of the x86-64 tiers (`crate::isa`), below the AVX-512 EVEX
//! encoders (`avx512.rs`, 512-bit). Sixteen registers, `ymm0-15` — AVX2 has
//! no extended file — and the general-register half of every kernel (the
//! loop nest, the store's address, the pointer class, the constant pool) and
//! the vector instructions both tiers have are `x86_64.rs`'s, shared with
//! AVX-512; what is this file's is the VEX prefix, the instructions only AVX2
//! has ([`Extra`]), and how it selects.
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
    Alu, Direction, GENERAL, Imm8, Lanewise, Mem, NoDisp, Pred, Rounding, Vex, VexPrefix, X86,
};
use super::{
    Bound, Class, ClassId, Edges, EncodedInst, Integer, IsaBackend, Label, LaneOp, Pointer, Rebind,
    Selected, Stage, Store, Target, Test, Value, ValueName, Vector, unimplemented_op,
};
use crate::error::CompileError;
use crate::program::IfArm;
use alloc::vec::Vec;
use pixelflow_ir::OpKind;

type Inst<S> = x86_64::Inst<S, Avx2>;
type Op<S> = x86_64::Op<S, Avx2>;

/// The instructions only AVX2 has: what the shared [`x86_64::Inst`] leaves to
/// a tier whose masks are ordinary vectors.
pub(super) enum Extra<S: Stage> {
    /// `vcmpps dst, a, b, pred`: an all-ones lane where it holds, all-zero
    /// where it does not.
    Cmp {
        pred: Pred,
        dst: S::Write<Vector>,
        a: S::Read<Vector>,
        b: S::Read<Vector>,
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
            BinaryForm::Cmp(pred) => Inst::Extra(Extra::Cmp { pred, dst, a, b }),
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
    type Prefix = VexPrefix;
    type Extra<S: Stage> = Extra<S>;

    fn general(inst: x86_64::Gp<Selected>) -> Op<Selected> {
        Op::General(inst)
    }

    /// `vpcmpeqd dst, dst, dst`
    fn ones(dst: u8) -> EncodedInst {
        Vex::m0f_66(0x76).rrr(dst, dst, dst)
    }

    fn walk_extra<To: Stage>(extra: &Extra<Selected>, f: &mut impl Rebind<To>) -> Extra<To> {
        match extra {
            Extra::Cmp { pred, dst, a, b } => Extra::Cmp {
                pred: *pred,
                dst: f.write(dst),
                a: f.read(*a),
                b: f.read(*b),
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
            Extra::MoveMask { dst, src } => Extra::MoveMask {
                dst: f.write(dst),
                src: f.read(*src),
            },
            Extra::ExtractHigh { dst, src } => Extra::ExtractHigh {
                dst: f.write(dst),
                src: f.read(*src),
            },
            Extra::ExtractLane { dst, src, lane } => Extra::ExtractLane {
                dst: dst.walk(f),
                src: f.read(*src),
                lane: *lane,
            },
        }
    }

    fn encode_extra(extra: &Extra<Bound<'_, Self>>) -> EncodedInst {
        match extra {
            Extra::Cmp { pred, dst, a, b } => {
                Vex::m0f(0xC2)
                    .imm(*pred as u8)
                    .rrr(dst.number(), a.number(), b.number())
            }
            Extra::Gather {
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
                Vex::m0f38_66(0x92).vsib_scaled4(dst, Some(mask), base.number(), index)
            }
            Extra::MoveMask { dst, src } => Vex::m0f(0x50).rr(dst.number(), src.number()),
            // The destination is the `rm` operand here, the reverse of the
            // usual direction.
            Extra::ExtractHigh { dst, src } => {
                Vex::m0f3a_66(0x19).imm(1).rr(src.number(), dst.number())
            }
            Extra::ExtractLane { dst, src, lane } => {
                debug_assert!(*lane < 4, "vextractps reads the low 128 bits");
                Vex::m0f3a_66(0x17)
                    .xmm()
                    .imm(*lane)
                    .rm(src.number(), dst.base.number(), dst.disp)
            }
        }
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
                b.push(Op::Vector(Inst::Extra(Extra::Gather {
                    dst,
                    base,
                    index: indices,
                    mask,
                })));
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
                half = vex(b, |dst| Inst::Extra(Extra::ExtractHigh { dst, src: value }));
            }
            b.push(Op::Vector(Inst::Extra(Extra::ExtractLane {
                dst: Mem {
                    base,
                    disp: Imm8((lane * 4) as i8),
                },
                src: half,
                lane: (lane % 4) as u8,
            })));
        }
    }

    /// `vmovmskps`, then `test` (ZF iff no lane is set) or `cmp al, 0xFF` (ZF
    /// iff all eight are), then `je`: the `cmp al` form on a byte, because the
    /// sign-extending `cmp eax, imm8` could never equal a zero-extended mask.
    fn branch(b: &mut Builder<Self>, test: Test<Self>, edges: Edges) {
        let bits = vex(b, |dst| {
            Inst::Extra(Extra::MoveMask {
                dst,
                src: test.cond,
            })
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
