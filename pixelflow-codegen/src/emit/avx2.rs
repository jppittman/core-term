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

use super::build::Builder;
use super::regalloc::resource::RegNo;
use super::x86_64;
use super::x86_64::{Alu, Imm8, Lanewise, Mem, NoDisp, Pred, Sign, Vex, VexPrefix, X86, emit};
use super::{
    Bound, Edges, EncodedInst, Integer, LaneOp, Pointer, Rebind, Selected, Stage, Store, Test,
    Value, ValueName, Vector, VectorFile,
};
use crate::error::CompileError;
use crate::program::IfArm;

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
// Selection
// =============================================================================

/// The bytes `0..8`, little end first: what one `movabs` carries in for
/// `vpmovzxbd` to widen into the iota.
const IOTA_BYTES: u64 = 0x0706_0504_0302_0100;

/// The AVX2 machine, as the selection pipeline sees it.
pub(in crate::emit) enum Avx2 {}

impl X86 for Avx2 {
    type Prefix = VexPrefix;
    type Extra<S: Stage> = Extra<S>;
    type Held = Value<Vector>;

    // AVX2 has no extended file: `ymm0-15`.
    const VECTORS: &'static [u8] = &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
    const VECTOR_BYTES: u64 = 32;

    /// `vpcmpeqd dst, dst, dst`
    fn ones(dst: RegNo<VectorFile>) -> EncodedInst {
        Vex::m0f_66(0x76).rrr(dst, dst, dst)
    }

    /// The constant, from the pool.
    fn sign_mask(b: &mut Builder<Self>, sign: Sign) -> Result<Value<Vector>, CompileError> {
        x86_64::constant(
            b,
            f32::from_bits(match sign {
                Sign::Negate => 0x8000_0000,
                Sign::Magnitude => 0x7FFF_FFFF,
            }),
        )
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

    fn lane(b: &mut Builder<Self>, op: LaneOp<Self>) -> Result<Value<Vector>, CompileError> {
        Ok(match op {
            LaneOp::Const(value) => x86_64::constant(b, value)?,
            // The bytes `0..8` in through a general register, widened to
            // dwords, converted.
            LaneOp::Lanes => {
                let bytes = x86_64::define(b, |dst| x86_64::Gp::Movabs {
                    dst,
                    imm: IOTA_BYTES,
                });
                let packed = emit(b, |dst| Inst::Movq { dst, src: bytes });
                x86_64::iota(b, packed)
            }
            LaneOp::Unary(op, src) => x86_64::unary(b, op, src)?,
            // A comparison is an ordinary all-ones/all-zeros vector directly
            // (no k-register step, unlike AVX-512).
            LaneOp::Binary(op, a, c) => x86_64::arithmetic(b, op, (a, c)),
            LaneOp::Compare(op, a, c) => emit(b, |dst| {
                Inst::Extra(Extra::Cmp {
                    pred: op.into(),
                    dst,
                    a,
                    b: c,
                })
            }),
            LaneOp::MulAdd(x, y, addend) => x86_64::mul_add(b, (x, y), addend),
            // `cond ? if_true : if_false`, bit by bit.
            LaneOp::Blend {
                cond,
                if_true,
                if_false,
            } => {
                let kept = emit(b, |dst| Inst::Alu {
                    op: Alu::And,
                    dst,
                    a: cond,
                    b: if_true,
                });
                let other = emit(b, |dst| Inst::Alu {
                    op: Alu::AndNot,
                    dst,
                    a: cond,
                    b: if_false,
                });
                emit(b, |dst| Inst::Alu {
                    op: Alu::Or,
                    dst,
                    a: kept,
                    b: other,
                })
            }
            LaneOp::Shift(op, src, amount) => x86_64::shift(b, op, src, amount),
            LaneOp::Uniform { base, element } => x86_64::uniform(b, base, element)?,
            // `vgatherdps` under an all-ones mask: the indices are floats, so
            // they are truncated first, and the instruction clears the mask
            // as it completes lanes.
            LaneOp::Gather { base, index } => {
                let indices = emit(b, |dst| Inst::Unary {
                    op: Lanewise::ToInt,
                    dst,
                    src: index,
                });
                let ones = emit(b, |dst| Inst::Ones { dst });
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
            LaneOp::Broadcast { base, index } => x86_64::broadcast(b, base, index),
        })
    }

    fn lane_name(lane: Value<Vector>) -> ValueName {
        lane.name()
    }

    fn param_lane(param: ValueName) -> Value<Vector> {
        param.typed()
    }

    /// A full batch is one `vmovups`; a remainder is `vextractps` per lane,
    /// the low four straight out of the value and the rest out of its high
    /// half.
    fn store(b: &mut Builder<Self>, store: Store<Self>) {
        let Store {
            out,
            pitch,
            row,
            col,
            value,
            lanes,
        } = store;
        let base = x86_64::element_address(b, out, pitch, (row, col));
        if u64::from(lanes) == Self::VECTOR_BYTES / 4 {
            b.push(Op::Vector(Inst::StoreBatch {
                dst: Mem { base, disp: NoDisp },
                src: value,
            }));
            return;
        }
        let mut half = value;
        for lane in 0..lanes {
            if lane == 4 {
                half = emit(b, |dst| Inst::Extra(Extra::ExtractHigh { dst, src: value }));
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
        let bits = emit(b, |dst| {
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
}
