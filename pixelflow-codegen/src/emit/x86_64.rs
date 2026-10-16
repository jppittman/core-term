//! x86-64 leaf encoders: what the AVX2 and AVX-512 tiers share.
//!
//! What is here is the *architecture's*: [`Gp`], the general-register
//! instructions the loop nest and the store's address arithmetic are made
//! of (branches, the pointer class's loads and stores and the frame among
//! them), the memory-operand tail (`Mem`, `Disp`), and the vector encoder
//! ([`Opcode`]) both tiers are written with: VEX and EVEX are two prefixes
//! ([`Scheme`]) on one instruction set, so the instructions and their operand
//! forms are named here once. So are the selection rules the tiers agree on
//! (`impl IsaBackend for` every [`X86`]): a tier chooses a prefix, the
//! instructions only it has, and the few hooks where it selects differently;
//! the vector width is the tier's, not this file's.

use super::asm::Encoding;
use super::build::{Builder, Def, Spiller};
use super::regalloc::resource::FrameSlot;
use super::register_file::{EntryRegisters, Members, RegisterFile};
use super::{
    Bound, Class, ClassId, Edges, EncodedInst, Flags, FrameSize, Integer, IsaBackend, Label,
    LaneOp, Pointer, Rebind, STACK_PAGE, Selected, Stage, Store, Target, Test, Value, ValueName,
    Vector, frame_bytes, slot_disp,
};
use crate::error::CompileError;
use crate::program::{BinaryOp, CompareOp, ShiftOp, UnaryOp};
use alloc::vec::Vec;

// =============================================================================
// The pointer class: an address between a general register and memory
// =============================================================================

/// Bytes per pointer in the context array.
pub(super) const PTR_BYTES: i32 = 8;

// =============================================================================
// General-purpose registers
// =============================================================================

/// The registers the ABI fixes, by hardware number: the collapse kernel's
/// three arguments, and the stack pointer. Every other register is the
/// allocator's choice.
pub(super) mod gpr {
    /// 3rd integer argument: the pitch.
    pub(in crate::emit) const RDX: u8 = 2;
    /// The stack pointer.
    pub(in crate::emit) const RSP: u8 = 4;
    /// 2nd integer argument: the output plane.
    pub(in crate::emit) const RSI: u8 = 6;
    /// 1st integer argument: the context pointer — the array of buffer base
    /// pointers, then the uniform and origin blocks. Read-only for the whole
    /// kernel.
    pub(in crate::emit) const RDI: u8 = 7;
}

/// An x86-64 general-register instruction.
///
/// Generic over what its operands are ([`Stage`]); a field is declared by the
/// class of what it holds, so `lea`'s base is a `Pointer` and its index an
/// `Integer`, and `imul` cannot be handed an address. An instruction that
/// writes `EFLAGS` says so with a `flags` field.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) enum Gp<S: Stage> {
    /// `mov dst, src`
    Mov {
        dst: S::Write<Pointer>,
        src: S::Read<Pointer>,
    },
    /// `mov dst, src`, an integer between registers: the same bytes as
    /// [`Gp::Mov`], and not the same instruction, because an address and an
    /// integer are different classes.
    MovInt {
        dst: S::Write<Integer>,
        src: S::Read<Integer>,
    },
    /// `mov r32, imm32`: zero-extended into the 64-bit register.
    MovImm32 { dst: S::Write<Integer>, imm: u32 },
    /// `movabs dst, imm64`
    Movabs { dst: S::Write<Integer>, imm: u64 },
    /// `imul dst, src`: the two-operand 64-bit multiply.
    Imul {
        dst: S::Tie<Integer>,
        src: S::Read<Integer>,
        flags: S::Write<Flags>,
    },
    /// `add dst, src`
    Add {
        dst: S::Tie<Integer>,
        src: S::Read<Integer>,
        flags: S::Write<Flags>,
    },
    /// `lea dst, [base + index*4]`: the element address of a plane of `f32`s.
    Lea4 {
        dst: S::Write<Pointer>,
        base: S::Read<Pointer>,
        index: S::Read<Integer>,
    },
    /// `mov dst, [base + disp32]`: a pointer from the context array or a
    /// frame slot.
    MovLoad {
        dst: S::Write<Pointer>,
        src: Mem<S, Imm32>,
    },
    /// `mov [rsp + slot], src`: the allocator's spill of an address.
    SpillPtr {
        src: S::Read<Pointer>,
        slot: S::Slot,
    },
    /// `mov dst, [rsp + slot]`: the allocator's reload of an address.
    ReloadPtr {
        dst: S::Write<Pointer>,
        slot: S::Slot,
    },
    /// `mov [rsp + slot], src`: the allocator's spill of an integer.
    SpillInt {
        src: S::Read<Integer>,
        slot: S::Slot,
    },
    /// `mov dst, [rsp + slot]`: the allocator's reload of an integer.
    ReloadInt {
        dst: S::Write<Integer>,
        slot: S::Slot,
    },
    /// `test src32, src32`: ZF iff the low half is zero.
    Test {
        flags: S::Write<Flags>,
        src: S::Read<Integer>,
    },
    /// `cmp src8, imm8`: ZF iff the low byte is `imm`. Unlike `cmp src32,
    /// imm8` (sign-extending `83 /7`), it compares the raw byte, which is what
    /// an 8-lane all-true check (`movmskps`'s `0xFF`) needs: the extension
    /// would compare against `0xFFFFFFFF`, which a zero-extended mask can
    /// never equal.
    CmpByte {
        flags: S::Write<Flags>,
        src: S::Read<Integer>,
        imm: u8,
    },
    /// `jcc rel32`. `next` is the position laid out right after it, where the
    /// branch falls through to: it encodes to nothing.
    Jcc {
        cond: Cond,
        flags: S::Read<Flags>,
        taken: S::Target,
        next: S::Target,
    },
    /// `jmp rel32`.
    Jmp { to: S::Target },
    /// Go to the position laid out right after this instruction: encodes to
    /// nothing, and is what a transfer to the next block is.
    Fallthrough { to: S::Target },
    /// `sub rsp, imm32`: the function's frame, a page at a time past one page
    /// ([`enter_probing`]).
    Enter {
        size: S::FrameSize,
        flags: S::Write<Flags>,
    },
    /// `add rsp, imm32; vzeroupper; ret`: the function's one exit.
    ///
    /// The caller is Rust built for baseline x86-64, so its floating point is
    /// legacy SSE, and Intel cores charge legacy-SSE code for vector registers
    /// whose upper halves a VEX or EVEX instruction left dirty — a state
    /// transition on older cores, a false dependency and a merge per
    /// instruction on Skylake and later — until something clears them, which
    /// nothing in a Rust caller does. So the kernel clears them on the way
    /// out. Nothing is lost: the collapse ABI returns nothing in a vector
    /// register, and every result is already stored.
    ///
    /// Measured on an AVX-512 Xeon, with a 1×1 kernel whose Rust caller runs
    /// a 256-term scalar sum after each call: the call cost 190 ns over the
    /// sum on the AVX-512 tier and 88 ns on AVX2 without this, and 32 and 15
    /// ns with it. The charge is not the call's: the same sum, run after one
    /// call and never calling again, took 482 ns rather than 266.
    Ret {
        size: S::FrameSize,
        flags: S::Write<Flags>,
    },
}

impl Gp<Selected> {
    /// Rebuild at stage `T`, visiting each operand once, in field order.
    pub(super) fn walk<T: Stage>(&self, f: &mut impl Rebind<T>) -> Gp<T> {
        match self {
            Gp::Mov { dst, src } => Gp::Mov {
                dst: f.write(dst),
                src: f.read(*src),
            },
            Gp::MovInt { dst, src } => Gp::MovInt {
                dst: f.write(dst),
                src: f.read(*src),
            },
            Gp::MovImm32 { dst, imm } => Gp::MovImm32 {
                dst: f.write(dst),
                imm: *imm,
            },
            Gp::Movabs { dst, imm } => Gp::Movabs {
                dst: f.write(dst),
                imm: *imm,
            },
            Gp::Imul { dst, src, flags } => Gp::Imul {
                dst: f.tie(dst),
                src: f.read(*src),
                flags: f.write(flags),
            },
            Gp::Add { dst, src, flags } => Gp::Add {
                dst: f.tie(dst),
                src: f.read(*src),
                flags: f.write(flags),
            },
            Gp::Lea4 { dst, base, index } => Gp::Lea4 {
                dst: f.write(dst),
                base: f.read(*base),
                index: f.read(*index),
            },
            Gp::MovLoad { dst, src } => Gp::MovLoad {
                dst: f.write(dst),
                src: src.walk(f),
            },
            Gp::SpillPtr { src, slot } => Gp::SpillPtr {
                src: f.read(*src),
                slot: f.slot(*slot),
            },
            Gp::ReloadPtr { dst, slot } => Gp::ReloadPtr {
                dst: f.write(dst),
                slot: f.slot(*slot),
            },
            Gp::SpillInt { src, slot } => Gp::SpillInt {
                src: f.read(*src),
                slot: f.slot(*slot),
            },
            Gp::ReloadInt { dst, slot } => Gp::ReloadInt {
                dst: f.write(dst),
                slot: f.slot(*slot),
            },
            Gp::Test { flags, src } => Gp::Test {
                flags: f.write(flags),
                src: f.read(*src),
            },
            Gp::CmpByte { flags, src, imm } => Gp::CmpByte {
                flags: f.write(flags),
                src: f.read(*src),
                imm: *imm,
            },
            Gp::Jcc {
                cond,
                flags,
                taken,
                next,
            } => Gp::Jcc {
                cond: *cond,
                flags: f.read(*flags),
                taken: f.target(taken),
                next: f.target(next),
            },
            Gp::Jmp { to } => Gp::Jmp { to: f.target(to) },
            Gp::Fallthrough { to } => Gp::Fallthrough { to: f.target(to) },
            Gp::Enter { size: _, flags } => Gp::Enter {
                size: f.frame_size(),
                flags: f.write(flags),
            },
            Gp::Ret { size: _, flags } => Gp::Ret {
                size: f.frame_size(),
                flags: f.write(flags),
            },
        }
    }
}

impl<'m, B: IsaBackend> Gp<Bound<'m, B>> {
    fn encode(&self) -> EncodedInst {
        let mut inst = EncodedInst::new();
        match self {
            Gp::Mov { dst, src } => {
                inst.extend(&rr(0x89, dst.number(), src.number()));
            }
            Gp::MovInt { dst, src } => {
                inst.extend(&rr(0x89, dst.number(), src.number()));
            }
            Gp::MovImm32 { dst, imm } => {
                let dst = dst.number();
                if dst >= 8 {
                    inst.push(0x41);
                }
                inst.push(0xB8 | (dst & 7));
                inst.extend(&imm.to_le_bytes());
            }
            Gp::Movabs { dst, imm } => {
                let dst = dst.number();
                inst.push(0x48 | ((dst >> 3) & 1));
                inst.push(0xB8 | (dst & 7));
                inst.extend(&imm.to_le_bytes());
            }
            Gp::Imul { dst, src, .. } => {
                let (dst, src) = (dst.number(), src.number());
                inst.extend(&[rex_w(dst, src), 0x0F, 0xAF, modrm_rr(dst, src)]);
            }
            Gp::Add { dst, src, .. } => {
                inst.extend(&rr(0x01, dst.number(), src.number()));
            }
            Gp::Lea4 { dst, base, index } => {
                let (dst, base, index) = (dst.number(), base.number(), index.number());
                debug_assert!(index & 7 != RM_SIB, "rsp/r12 cannot index a SIB");
                inst.push(0x48 | (bit(dst, 3) << 2) | (bit(index, 3) << 1) | bit(base, 3));
                inst.push(0x8D);
                scaled4_operand_into(&mut inst, dst, base, index);
            }
            Gp::MovLoad { dst, src } => {
                let (dst, base) = (dst.number(), src.base.number());
                inst.push(rex_w(dst, base));
                inst.push(0x8B);
                mem_operand_into(&mut inst, dst, base, src.disp);
            }
            Gp::SpillPtr { src, slot } => {
                spill_into(&mut inst, src.number(), slot);
            }
            Gp::SpillInt { src, slot } => {
                spill_into(&mut inst, src.number(), slot);
            }
            Gp::ReloadPtr { dst, slot } => {
                reload_into(&mut inst, dst.number(), slot);
            }
            Gp::ReloadInt { dst, slot } => {
                reload_into(&mut inst, dst.number(), slot);
            }
            Gp::Test { src, .. } => {
                let src = src.number();
                if src >= 8 {
                    inst.push(0x45);
                }
                inst.push(0x85);
                inst.push(modrm_rr(src, src));
            }
            Gp::CmpByte { src, imm, .. } => {
                match src.number() {
                    // The accumulator has a short form.
                    0 => inst.push(0x3C),
                    n => {
                        // Without a REX prefix 4-7 name `ah`..`bh`, not `spl`..`dil`.
                        if n >= 4 {
                            inst.push(0x40 | ((n >> 3) & 1));
                        }
                        inst.push(0x80);
                        inst.push(modrm_rr(7, n));
                    }
                }
                inst.push(*imm);
            }
            Gp::Jcc { cond, .. } => inst.extend(&[0x0F, 0x80 | *cond as u8, 0, 0, 0, 0]),
            Gp::Jmp { .. } => inst.extend(&[0xE9, 0, 0, 0, 0]),
            Gp::Fallthrough { .. } => {}
            Gp::Enter { size, .. } => inst = sub_rsp(frame_bytes(*size)),
            Gp::Ret { size, .. } => {
                inst = add_rsp(frame_bytes(*size));
                // `vzeroupper`: `VEX.128.0F.WIG 77`, zeroing bits 128 and up of
                // vector registers 0–15, the sixteen a legacy-SSE instruction
                // can name. VEX on every tier.
                inst.extend(Vex::m0f(0x77).xmm().bare().as_bytes());
                inst.push(0xC3);
            }
        }
        inst
    }

    /// The label field the instruction carries: its offset in the bytes, and
    /// the target it reaches. The one statement of which instructions have one.
    fn label_field(&self) -> Option<(usize, Label)> {
        match self {
            Gp::Jmp { to } => Some((JMP_DISP, *to)),
            Gp::Jcc { taken, .. } => Some((JCC_DISP, *taken)),
            _ => None,
        }
    }

    /// The instruction's bytes, and the label fields in them.
    pub(super) fn assemble(&self, out: &mut Encoding<'_>) {
        if let Gp::Enter { size, .. } = self {
            return enter_probing(frame_bytes(*size), out);
        }
        out.bytes(self.encode().as_bytes());
        if let Some((at, to)) = self.label_field() {
            out.field(at, to, patch_rel32);
        }
        match self {
            Gp::Jcc { next, .. } => out.falls_through(*next),
            Gp::Fallthrough { to } => out.falls_through(*to),
            _ => {}
        }
    }
}

// =============================================================================
// Selection: the general-register half both tiers share
// =============================================================================

/// The caller-saved general registers, by hardware number: `rax`, `rcx`,
/// `rdx`, `rsi`, `rdi` and `r8`-`r11`. A callee-saved register would cost a
/// prologue, and `rsp` is the frame's.
pub(super) const GENERAL: &[u8] = &[0, 1, 2, 6, 7, 8, 9, 10, 11];

/// An x86 tier: its instructions are [`Op`], the shared vector ones
/// ([`Inst`]) and the general-register ones ([`Gp`]), and every [`IsaBackend`]
/// rule the two tiers agree on is written once, below, over the hooks here.
///
/// What a tier says is what the two differ in: the prefix its vector
/// instructions carry, the instructions only it has, how it spells all-ones
/// and a sign mask, how it holds a comparison, and what it does with a lane
/// where the other does not. A tier cannot select the other's instruction
/// because it is not a variant of its [`X86::Extra`].
pub(super) trait X86: Sized + 'static {
    /// The prefix scheme of its vector instructions.
    type Prefix: Scheme;
    /// The vector instructions only this tier has.
    type Extra<S: Stage>;
    /// One IR lane value as this tier holds it ([`IsaBackend::Lane`]).
    type Held: Copy;
    /// The allocatable vector registers, by number.
    const VECTORS: &'static [u8];
    /// The allocatable opmask registers, by number: none unless the tier has a
    /// mask file.
    const OPMASKS: &'static [u8] = &[];
    /// Bytes per vector register and per vector frame slot.
    const VECTOR_BYTES: u64;

    /// All-ones in register `dst`, whatever it held.
    fn ones(dst: u8) -> EncodedInst;
    /// The mask `Neg` xors and `Abs` ands: the sign bit, or all that is below
    /// it.
    fn sign_mask(b: &mut Builder<Self>, sign: Sign) -> Result<Value<Vector>, CompileError>;
    /// [`Inst::walk`] for an instruction only this tier has.
    fn walk_extra<To: Stage>(
        extra: &Self::Extra<Selected>,
        f: &mut impl Rebind<To>,
    ) -> Self::Extra<To>;
    /// [`Inst::encode`] for an instruction only this tier has.
    fn encode_extra(extra: &Self::Extra<Bound<'_, Self>>) -> EncodedInst;
    /// Whether the allocator may place `extra` again where a read needs it:
    /// an idiom of this tier that has no effect and reads nothing.
    fn rematerializable_extra(_extra: &Self::Extra<Selected>) -> bool {
        false
    }

    /// [`IsaBackend::lane`].
    fn lane(b: &mut Builder<Self>, op: LaneOp<Self>) -> Result<Self::Held, CompileError>;
    /// [`IsaBackend::lane_name`].
    fn lane_name(lane: Self::Held) -> ValueName;
    /// [`IsaBackend::param_lane`].
    fn param_lane(param: ValueName) -> Self::Held;
    /// [`IsaBackend::store`].
    fn store(b: &mut Builder<Self>, store: Store<Self>);
    /// [`IsaBackend::branch`].
    fn branch(b: &mut Builder<Self>, test: Test<Self>, edges: Edges);
    /// [`IsaBackend::copy`] of a class neither the general nor the vector file
    /// holds: a tier with no such file refuses it.
    fn copy_other(_b: &mut Spiller<'_, Self>, src: ValueName) -> ValueName {
        unreachable!("this tier has no {:?} to copy", src.class)
    }
    /// [`IsaBackend::spill`] of such a class.
    fn spill_other(_b: &mut Spiller<'_, Self>, src: ValueName, _slot: &FrameSlot) {
        unreachable!("this tier has no {:?} to spill", src.class)
    }
    /// [`IsaBackend::reload`] of such a class.
    fn reload_other(_b: &mut Spiller<'_, Self>, class: ClassId, _slot: &FrameSlot) -> ValueName {
        unreachable!("this tier has no {class:?} to reload")
    }
}

/// Every x86 tier is a backend: the rules are the architecture's, and the
/// tier supplies the hooks.
impl<B: X86> IsaBackend for B {
    type Inst<S: Stage> = Op<S, B>;
    type Constant = u32;
    type Anchors = ();
    type Lane = B::Held;

    // SysV has no callee-saved vector registers and the collapse ABI passes
    // no vector, so every vector register is the allocator's.
    const FILE: RegisterFile = RegisterFile::new(
        Members {
            vector: B::VECTORS,
            general: GENERAL,
            opmask: B::OPMASKS,
            flags: &[0],
        },
        EntryRegisters {
            ctx: gpr::RDI,
            out: gpr::RSI,
            pitch: gpr::RDX,
        },
        B::VECTOR_BYTES,
    );
    const POOL_REACH: u64 = i32::MAX as u64 / 4;

    fn lane(b: &mut Builder<Self>, op: LaneOp<Self>) -> Result<B::Held, CompileError> {
        B::lane(b, op)
    }

    fn lane_name(lane: B::Held) -> ValueName {
        B::lane_name(lane)
    }

    fn param_lane(param: ValueName) -> B::Held {
        B::param_lane(param)
    }

    /// `mov p, [ctx + 8·slot]`.
    fn context(
        b: &mut Builder<Self>,
        ctx: Value<Pointer>,
        slot: u64,
    ) -> Result<Value<Pointer>, CompileError> {
        let disp = displacement(slot, PTR_BYTES as u64)?;
        Ok(define(b, |dst| Gp::MovLoad {
            dst,
            src: Mem { base: ctx, disp },
        }))
    }

    fn store(b: &mut Builder<Self>, store: Store<Self>) {
        B::store(b, store);
    }

    fn branch(b: &mut Builder<Self>, test: Test<Self>, edges: Edges) {
        B::branch(b, test, edges);
    }

    /// `jmp` to `to`, or nothing when `to` is the next block.
    fn jump(b: &mut Builder<Self>, to: Target, next: Label) {
        b.push(Op::General(if to.label == next {
            Gp::Fallthrough { to }
        } else {
            Gp::Jmp { to }
        }));
    }

    /// `sub rsp, frame`: the function's entry.
    fn enter(b: &mut Builder<Self>) {
        let flags = b.def();
        b.push(Op::General(Gp::Enter {
            size: FrameSize,
            flags,
        }));
    }

    /// `add rsp, frame; vzeroupper; ret`: the function's one exit.
    fn ret(b: &mut Builder<Self>) {
        let flags = b.def();
        b.push(Op::General(Gp::Ret {
            size: FrameSize,
            flags,
        }));
    }

    fn copy(b: &mut Spiller<'_, Self>, src: ValueName) -> ValueName {
        match src.class {
            ClassId::Vector => b.define::<Vector>(|dst| {
                Op::Vector(Inst::Mov {
                    dst,
                    src: src.typed(),
                })
            }),
            ClassId::Pointer | ClassId::Integer => copy_general(b, src),
            ClassId::Opmask | ClassId::Flags => B::copy_other(b, src),
        }
    }

    fn spill(b: &mut Spiller<'_, Self>, src: ValueName, slot: &FrameSlot) {
        match src.class {
            ClassId::Vector => b.push(Op::Vector(Inst::SpillVector {
                src: src.typed(),
                slot: slot.name(),
            })),
            ClassId::Pointer | ClassId::Integer => spill_general(b, src, slot),
            ClassId::Opmask | ClassId::Flags => B::spill_other(b, src, slot),
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
            ClassId::Pointer | ClassId::Integer => reload_general(b, class, slot),
            ClassId::Opmask | ClassId::Flags => B::reload_other(b, class, slot),
        }
    }

    fn rematerializable(inst: &Op<Selected, B>) -> bool {
        match inst {
            Op::Vector(Inst::LoadConst { .. } | Inst::Zero { .. } | Inst::Ones { .. }) => true,
            Op::Vector(Inst::Extra(extra)) => B::rematerializable_extra(extra),
            _ => false,
        }
    }

    fn reads_pool<S: Stage>(inst: &Op<S, B>) -> bool {
        matches!(inst, Op::Vector(Inst::LoadConst { .. }))
    }

    fn walk<T: Stage>(inst: &Op<Selected, B>, f: &mut impl Rebind<T>) -> Op<T, B> {
        inst.walk(f)
    }

    fn encode(inst: &Op<Bound<'_, B>, B>, out: &mut Encoding<'_>) {
        inst.assemble(out);
    }

    fn constant_bytes(constant: u32) -> Vec<u8> {
        constant.to_le_bytes().to_vec()
    }
}

/// The sign bit and its complement: the constants `Neg` xors and `Abs` ands.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) enum Sign {
    /// `0x8000_0000`
    Negate,
    /// `0x7FFF_FFFF`
    Magnitude,
}

/// Define a value of class `C` with `make`, which builds the
/// general-register instruction that writes it.
pub(super) fn define<B: X86, C: Class>(
    b: &mut Builder<B>,
    make: impl FnOnce(Def<C>) -> Gp<Selected>,
) -> Value<C> {
    let dst = b.def();
    let value = dst.value();
    b.push(Op::General(make(dst)));
    value
}

/// `out + 4·(row·pitch + col)`: where the lattice's store writes. Each index
/// is lane 0 of its binder, truncated.
pub(super) fn element_address<B: X86>(
    b: &mut Builder<B>,
    out: Value<Pointer>,
    pitch: Value<Integer>,
    (row, col): (Value<Vector>, Value<Vector>),
) -> Value<Pointer> {
    let row = emit(b, |dst| Inst::Cvtt { dst, src: row });
    let col = emit(b, |dst| Inst::Cvtt { dst, src: col });
    let (scaled, flags) = (b.tie(row), b.def());
    let row_pitch = scaled.write();
    b.push(Op::General(Gp::Imul {
        dst: scaled,
        src: pitch,
        flags,
    }));
    let (summed, flags) = (b.tie(row_pitch), b.def());
    let index = summed.write();
    b.push(Op::General(Gp::Add {
        dst: summed,
        src: col,
        flags,
    }));
    define(b, |dst| Gp::Lea4 {
        dst,
        base: out,
        index,
    })
}

// =============================================================================
// Selection: the vector rules both tiers share
// =============================================================================

/// Define a value of class `C` with `make`, which builds the vector
/// instruction that writes it.
pub(super) fn emit<B: X86, C: Class>(
    b: &mut Builder<B>,
    make: impl FnOnce(Def<C>) -> Inst<Selected, B>,
) -> Value<C> {
    let dst = b.def();
    let value = dst.value();
    b.push(Op::Vector(make(dst)));
    value
}

/// `value` splatted: the zero and all-ones idioms, else one pool entry.
pub(super) fn constant<B: X86>(
    b: &mut Builder<B>,
    value: f32,
) -> Result<Value<Vector>, CompileError> {
    Ok(match value.to_bits() {
        0 => emit(b, |dst| Inst::Zero { dst }),
        u32::MAX => emit(b, |dst| Inst::Ones { dst }),
        bits => {
            let at = b.constant(bits)?.label;
            emit(b, |dst| Inst::LoadConst { dst, at })
        }
    })
}

/// The bytes in `packed`'s low lanes, widened to dwords and converted: the
/// iota.
pub(super) fn iota<B: X86>(b: &mut Builder<B>, packed: Value<Vector>) -> Value<Vector> {
    let widened = emit(b, |dst| Inst::Unary {
        op: Lanewise::WidenBytes,
        dst,
        src: packed,
    });
    emit(b, |dst| Inst::Unary {
        op: Lanewise::FromInt,
        dst,
        src: widened,
    })
}

/// The one instruction a unary op is.
pub(super) fn unary<B: X86>(
    b: &mut Builder<B>,
    op: UnaryOp,
    src: Value<Vector>,
) -> Result<Value<Vector>, CompileError> {
    let lanewise = match op {
        UnaryOp::Neg => return signed(b, Sign::Negate, src),
        UnaryOp::Abs => return signed(b, Sign::Magnitude, src),
        UnaryOp::Floor => return Ok(round(b, Rounding::Floor, src)),
        UnaryOp::Ceil => return Ok(round(b, Rounding::Ceil, src)),
        UnaryOp::Round => return Ok(round(b, Rounding::Nearest, src)),
        UnaryOp::Sqrt => Lanewise::Sqrt,
        UnaryOp::Rsqrt => Lanewise::Rsqrt,
        UnaryOp::Recip => Lanewise::Recip,
        // Int/float domain crossings, exactly the hardware's cvttps2dq /
        // cvtdq2ps — the primitives exp/log lower to.
        UnaryOp::TruncToInt => Lanewise::ToInt,
        UnaryOp::IntToFloat => Lanewise::FromInt,
    };
    Ok(emit(b, |dst| Inst::Unary {
        op: lanewise,
        dst,
        src,
    }))
}

fn round<B: X86>(b: &mut Builder<B>, mode: Rounding, src: Value<Vector>) -> Value<Vector> {
    emit(b, |dst| Inst::Round { mode, dst, src })
}

/// `Neg` and `Abs`: xor or and the sign bit, which is a constant to hold in a
/// register first.
fn signed<B: X86>(
    b: &mut Builder<B>,
    sign: Sign,
    src: Value<Vector>,
) -> Result<Value<Vector>, CompileError> {
    let mask = B::sign_mask(b, sign)?;
    let op = match sign {
        Sign::Negate => Alu::Xor,
        Sign::Magnitude => Alu::And,
    };
    Ok(emit(b, |dst| Inst::Alu {
        op,
        dst,
        a: src,
        b: mask,
    }))
}

impl From<CompareOp> for Pred {
    /// The `vcmpps` predicate of a comparison op.
    fn from(op: CompareOp) -> Self {
        match op {
            CompareOp::Eq => Pred::Eq,
            CompareOp::Ne => Pred::Ne,
            CompareOp::Lt => Pred::Lt,
            CompareOp::Le => Pred::Le,
            CompareOp::Gt => Pred::Nle,
            CompareOp::Ge => Pred::Ge,
        }
    }
}

/// The vector ALU instruction of a binary op.
pub(super) fn arithmetic<B: X86>(
    b: &mut Builder<B>,
    op: BinaryOp,
    (a, c): (Value<Vector>, Value<Vector>),
) -> Value<Vector> {
    let op = match op {
        BinaryOp::Add => Alu::Add,
        BinaryOp::Sub => Alu::Sub,
        BinaryOp::Mul => Alu::Mul,
        BinaryOp::Div => Alu::Div,
        BinaryOp::Min => Alu::Min,
        BinaryOp::Max => Alu::Max,
        BinaryOp::BitAnd => Alu::And,
        BinaryOp::BitOr => Alu::Or,
        // Integer add on lane bit patterns (exp/log exponent arithmetic).
        BinaryOp::IAdd => Alu::IAdd,
    };
    emit(b, |dst| Inst::Alu { op, dst, a, b: c })
}

/// `a·b + c`, one rounding: `c` is the accumulator the result overwrites.
pub(super) fn mul_add<B: X86>(
    b: &mut Builder<B>,
    (x, y): (Value<Vector>, Value<Vector>),
    addend: Value<Vector>,
) -> Value<Vector> {
    let acc = b.tie(addend);
    let result = acc.write();
    b.push(Op::Vector(Inst::Fma231 { acc, a: x, b: y }));
    result
}

/// `src` shifted by an immediate.
pub(super) fn shift<B: X86>(
    b: &mut Builder<B>,
    op: ShiftOp,
    src: Value<Vector>,
    amount: u8,
) -> Value<Vector> {
    let direction = match op {
        ShiftOp::Shl => Direction::Left,
        ShiftOp::Shr => Direction::Right,
    };
    emit(b, |dst| Inst::Shift {
        direction,
        dst,
        src,
        amount,
    })
}

/// `vbroadcastss dst, [block + 4·element]`.
pub(super) fn uniform<B: X86>(
    b: &mut Builder<B>,
    base: Value<Pointer>,
    element: u64,
) -> Result<Value<Vector>, CompileError> {
    let disp = displacement(element, F32_BYTES)?;
    Ok(emit(b, |dst| Inst::Broadcast {
        dst,
        src: Mem { base, disp },
    }))
}

/// One element, the same in every lane: the index is lane 0, truncated.
pub(super) fn broadcast<B: X86>(
    b: &mut Builder<B>,
    base: Value<Pointer>,
    index: Value<Vector>,
) -> Value<Vector> {
    let index = emit(b, |dst| Inst::Cvtt { dst, src: index });
    emit(b, |dst| Inst::BroadcastIndexed { dst, base, index })
}

/// End the block: `jcc` to `edges.taken` when `flags` meet `cond`, else fall
/// through to `edges.next`.
pub(super) fn branch<B: X86>(b: &mut Builder<B>, cond: Cond, flags: Value<Flags>, edges: Edges) {
    let next = Target {
        label: edges.next,
        args: Vec::new(),
    };
    b.push(Op::General(Gp::Jcc {
        cond,
        flags,
        taken: edges.taken,
        next,
    }));
}

/// `mov` of an address or an integer between general registers.
pub(super) fn copy_general<B: X86>(b: &mut Spiller<'_, B>, src: ValueName) -> ValueName {
    match src.class {
        ClassId::Pointer => b.define::<Pointer>(|dst| {
            Op::General(Gp::Mov {
                dst,
                src: src.typed(),
            })
        }),
        ClassId::Integer => b.define::<Integer>(|dst| {
            Op::General(Gp::MovInt {
                dst,
                src: src.typed(),
            })
        }),
        class => unreachable!("{class:?} is not held in a general register"),
    }
}

/// `mov [rsp + slot], src` of an address or an integer.
pub(super) fn spill_general<B: X86>(b: &mut Spiller<'_, B>, src: ValueName, slot: &FrameSlot) {
    let slot = slot.name();
    b.push(Op::General(match src.class {
        ClassId::Pointer => Gp::SpillPtr {
            src: src.typed(),
            slot,
        },
        ClassId::Integer => Gp::SpillInt {
            src: src.typed(),
            slot,
        },
        class => unreachable!("{class:?} is not held in a general register"),
    }));
}

/// `mov dst, [rsp + slot]` of an address or an integer.
pub(super) fn reload_general<B: X86>(
    b: &mut Spiller<'_, B>,
    class: ClassId,
    slot: &FrameSlot,
) -> ValueName {
    let slot = slot.name();
    match class {
        ClassId::Pointer => b.define::<Pointer>(|dst| Op::General(Gp::ReloadPtr { dst, slot })),
        ClassId::Integer => b.define::<Integer>(|dst| Op::General(Gp::ReloadInt { dst, slot })),
        class => unreachable!("{class:?} is not held in a general register"),
    }
}

/// A sign-extended 8-bit immediate.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) struct Imm8(pub(super) i8);

/// A 32-bit immediate: a displacement's width in the `mod = 10` form.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) struct Imm32(pub(super) i32);

/// `REX.W` plus the extension bits for a two-register form.
///
/// `R` extends the ModRM.reg field (the source here), `B` extends ModRM.rm
/// (the destination).
#[inline(always)]
const fn rex_w(reg: u8, rm: u8) -> u8 {
    0x48 | (((reg >> 3) & 1) << 2) | ((rm >> 3) & 1)
}

/// `rsp` moved by `bytes`: `81 /digit id`, whose `/digit` says add or sub.
fn rsp_imm32(digit: u8, bytes: u32) -> EncodedInst {
    let mut inst = EncodedInst::new();
    inst.extend(&[rex_w(0, gpr::RSP), 0x81, modrm_rr(digit, gpr::RSP)]);
    inst.extend(&bytes.to_le_bytes());
    inst
}

/// `add rsp, imm32`.
fn add_rsp(bytes: u32) -> EncodedInst {
    rsp_imm32(0, bytes)
}

/// `sub rsp, imm32`.
fn sub_rsp(bytes: u32) -> EncodedInst {
    rsp_imm32(5, bytes)
}

/// `mov dword [rsp], 0`: `C7 /0 id` at the SIB form `rsp` always takes.
const PROBE: [u8; 7] = [0xC7, RM_SIB, SIB_BASE_ONLY, 0, 0, 0, 0];

/// `sub rsp, size`, entered a page at a time while more than a page remains,
/// each page touched before the next is taken.
///
/// A thread's stack ends in one guard page, and a single `sub` past it lands
/// in whatever is mapped beyond without touching the guard. The probe is a
/// store, so it needs no register and no flags, and it is the instruction's own
/// bytes: `MAX_FRAME` bounds the sequence at 511 pairs.
fn enter_probing(size: u32, out: &mut Encoding<'_>) {
    let mut left = size;
    while left > STACK_PAGE {
        out.bytes(sub_rsp(STACK_PAGE).as_bytes());
        out.bytes(&PROBE);
        left -= STACK_PAGE;
    }
    out.bytes(sub_rsp(left).as_bytes());
}

/// ModRM for the register-direct form: `mod = 11`.
#[inline(always)]
const fn modrm_rr(reg: u8, rm: u8) -> u8 {
    0xC0 | ((reg & 7) << 3) | (rm & 7)
}

/// `REX.W opcode /r` with both operands in registers.
#[inline(always)]
const fn rr(opcode: u8, dst: u8, src: u8) -> [u8; 3] {
    [rex_w(src, dst), opcode, modrm_rr(src, dst)]
}

/// A frame slot as a memory operand: the base and displacement of `[rsp + slot]`.
/// The one place the stack pointer addresses a slot.
pub(super) fn frame(slot: &FrameSlot) -> (u8, Imm32) {
    (gpr::RSP, Imm32(slot_disp(slot)))
}

/// `mov [rsp + slot], src`.
fn spill_into(inst: &mut EncodedInst, src: u8, slot: &FrameSlot) {
    let (base, disp) = frame(slot);
    inst.push(rex_w(src, base));
    inst.push(0x89);
    mem_operand_into(inst, src, base, disp);
}

/// `mov dst, [rsp + slot]`.
fn reload_into(inst: &mut EncodedInst, dst: u8, slot: &FrameSlot) {
    let (base, disp) = frame(slot);
    inst.push(rex_w(dst, base));
    inst.push(0x8B);
    mem_operand_into(inst, dst, base, disp);
}

/// The 4-bit condition an x86 `jcc` tests: the conditions the emitter
/// branches on.
///
/// `0F 8x rel32` is one instruction whose low opcode nibble *is* this value, so
/// the assembler encodes it by casting rather than by dispatching to one
/// hand-written mnemonic per condition. Named by the ISA's own mnemonics, with
/// their aliases, because that is what a reader checks against the manual.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u8)]
pub(super) enum Cond {
    /// `jb` / `jc` / `jnae` — CF set; unsigned `<`.
    B = 0x2,
    /// `je` / `jz` — ZF set.
    E = 0x4,
}

/// Bytes from a `jmp`'s start to its displacement field: one opcode byte.
const JMP_DISP: usize = 1;
/// Bytes from a `jcc`'s start to its displacement field: `0F` plus the
/// condition byte.
const JCC_DISP: usize = 2;

/// Write a `rel32` at `pos` so the instruction it belongs to reaches `target`.
///
/// `rel32` is measured from the *end* of the instruction, which is the end of
/// the displacement field itself.
pub(super) fn patch_rel32(code: &mut [u8], pos: usize, target: usize) {
    let rel = (target as i64) - (pos as i64 + 4);
    let rel = i32::try_from(rel).expect("an x86 rel32 spans \u{00b1}2 GiB");
    code[pos..pos + 4].copy_from_slice(&rel.to_le_bytes());
}

// =============================================================================
// The vector operations both tiers encode
// =============================================================================

// What an operation *is* is the architecture's, and so is the opcode that says
// it ([`Opcode::alu`], [`Opcode::lanewise`]); a tier only picks the prefix.

/// A three-operand arithmetic or bitwise instruction: `op dst, a, b`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) enum Alu {
    Add,
    Sub,
    Mul,
    Div,
    Min,
    Max,
    And,
    /// `!a & b`
    AndNot,
    Or,
    Xor,
    /// `vpaddd`: the integer-domain add.
    IAdd,
}

/// A one-source instruction: `op dst, src`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) enum Lanewise {
    Sqrt,
    Rsqrt,
    Recip,
    /// `vcvttps2dq`
    ToInt,
    /// `vcvtdq2ps`
    FromInt,
    /// `vpmovzxbd`: bytes widened to dword lanes.
    WidenBytes,
}

/// The predicate of a `vcmpps`: its imm8.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) enum Pred {
    Eq = 0,
    Lt = 1,
    Le = 2,
    Ne = 4,
    Ge = 5,
    /// `>`: the unordered-safe "not less-or-equal".
    Nle = 6,
}

/// The rounding mode of a `vroundps` or `vrndscaleps`: its imm8.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) enum Rounding {
    Nearest = 0,
    Floor = 1,
    Ceil = 2,
}

/// The direction of an integer shift by an immediate: the `/digit` of its
/// opcode.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) enum Direction {
    Left = 6,
    Right = 2,
}

// =============================================================================
// The vector encoder: one instruction set, two prefixes
// =============================================================================

// VEX and EVEX are two prefixes on the same instructions. The opcode map, the
// implied legacy prefix, `W`, the opcode byte and the ModRM/SIB tail are the
// instruction's and are written once, by [`Opcode`]; what the prefix adds is
// where the registers' high bits go, which is [`Scheme`]'s alone.

/// The opcode map an instruction lives in: the field Intel calls `mmmmm` (VEX)
/// or `mm` (EVEX), a map *selector*, nothing more.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Map {
    /// `0F`
    M0F = 1,
    /// `0F38`
    M0F38 = 2,
    /// `0F3A`
    M0F3A = 3,
}

/// The legacy prefix the instruction implies (`pp`).
#[derive(Clone, Copy)]
pub(super) enum Pp {
    /// No implied prefix.
    None = 0,
    /// `66`
    P66 = 1,
    /// `F3`
    F3 = 2,
}

/// The registers one instruction names, by hardware number and uninverted:
/// what a [`Scheme`] folds into its prefix bits.
#[derive(Clone, Copy, Default)]
pub(super) struct Ext {
    /// ModRM.reg (or the `/digit`): `R`, and EVEX's `R'`.
    reg: u8,
    /// `X`: bit 3 of the SIB index, or, where `rm` is a register, bit 4 of it,
    /// which only EVEX has the room to say.
    x: u8,
    /// `B`: bit 3 of ModRM.rm, or of the base.
    b: u8,
    /// `vvvv`: the second source, or `None` where the instruction has none and
    /// the field must read `1111`.
    vvvv: Option<u8>,
    /// The register whose bit 4 is EVEX's `V'`: the `vvvv` source, or a VSIB
    /// index.
    v: u8,
}

/// A prefix encoding for the vector instructions: [`VexPrefix`] or
/// [`EvexPrefix`].
pub(super) trait Scheme: Copy {
    /// The prefix at the tier's full vector width.
    const FULL: Self;
    /// `rsqrtps` and `rcpps` are VEX's, `vrsqrt14ps` and `vrcp14ps` EVEX's: EVEX
    /// has no `0F 52`/`0F 53`, and ~2^-14 relative error matches `Recip`'s
    /// "approximate reciprocal" contract on every other backend.
    const RSQRT: Opcode<Self>;
    const RECIP: Opcode<Self>;
    /// The 128-bit form: for the few `xmm`-only instructions (`vmovq`,
    /// `vextractps`, the `k` family), which `#UD` at the wider length.
    fn xmm(self) -> Self;
    /// Write the prefix and the opcode byte.
    fn write(op: &Opcode<Self>, ext: Ext, inst: &mut EncodedInst);
}

/// One bit of a register number, as 0 or 1.
const fn bit(register: u8, at: u8) -> u8 {
    (register >> at) & 1
}

/// A prefix stores the register-extension bits inverted.
const fn inverted(bit: u8) -> u8 {
    bit ^ 1
}

/// VEX: sixteen registers, 128 or 256 bits wide.
#[derive(Clone, Copy)]
pub(super) struct VexPrefix {
    /// `VEX.L`.
    l256: bool,
}

impl Scheme for VexPrefix {
    const FULL: Self = Self { l256: true };
    const RSQRT: Opcode<Self> = Opcode::m0f(0x52);
    const RECIP: Opcode<Self> = Opcode::m0f(0x53);

    fn xmm(self) -> Self {
        Self { l256: false }
    }

    /// The two-byte form where it can say everything (map `0F`, `W0`, no `X`
    /// or `B`), else the three-byte one.
    fn write(op: &Opcode<Self>, e: Ext, inst: &mut EncodedInst) {
        let rest =
            ((!e.vvvv.unwrap_or(0) & 0xF) << 3) | ((op.scheme.l256 as u8) << 2) | op.pp as u8;
        let r = inverted(bit(e.reg, 3)) << 7;
        if op.map == Map::M0F && !op.w && e.x | e.b == 0 {
            inst.extend(&[0xC5, r | rest]);
        } else {
            inst.extend(&[
                0xC4,
                r | (inverted(e.x) << 6) | (inverted(e.b) << 5) | op.map as u8,
                ((op.w as u8) << 7) | rest,
            ]);
        }
        inst.push(op.byte);
    }
}

/// EVEX: thirty-two registers, 128 to 512 bits wide, and a writemask.
#[derive(Clone, Copy)]
pub(super) struct EvexPrefix {
    /// `EVEX.L'L`: `10` for the `zmm` form, `00` for `xmm`.
    ll: u8,
    /// `EVEX.aaa`: the writemask register, or 0 for none. Merge-masking
    /// (`z = 0`), which for a store means the masked-off lanes are left in
    /// memory untouched.
    aaa: u8,
}

impl Scheme for EvexPrefix {
    const FULL: Self = Self { ll: 0b10, aaa: 0 };
    const RSQRT: Opcode<Self> = Opcode::m0f38_66(0x4E);
    const RECIP: Opcode<Self> = Opcode::m0f38_66(0x4C);

    fn xmm(self) -> Self {
        Self { ll: 0, ..self }
    }

    fn write(op: &Opcode<Self>, e: Ext, inst: &mut EncodedInst) {
        let vvvv = !e.vvvv.unwrap_or(0) & 0xF;
        inst.extend(&[
            0x62,
            (inverted(bit(e.reg, 3)) << 7)
                | (inverted(e.x) << 6)
                | (inverted(e.b) << 5)
                | (inverted(bit(e.reg, 4)) << 4)
                | op.map as u8,
            ((op.w as u8) << 7) | (vvvv << 3) | (1 << 2) | op.pp as u8,
            // z = 0 (merge), L'L, b(roadcast) = 0, V', aaa.
            (op.scheme.ll << 5) | (inverted(bit(e.v, 4)) << 3) | op.scheme.aaa,
            op.byte,
        ]);
    }
}

/// A displacement the prefix scheme `S` reads as written. EVEX scales an
/// 8-bit one by the operand size (`disp8*N`), so [`Imm8`] is not one of its
/// displacements: it would encode a different address than it names.
pub(super) trait DispIn<S>: Disp {}

impl<S> DispIn<S> for NoDisp {}
impl<S> DispIn<S> for Imm32 {}
impl DispIn<VexPrefix> for Imm8 {}

/// The identity of one vector instruction: opcode map, implied legacy prefix,
/// `W` bit, opcode byte, and the vector length and (EVEX) writemask the prefix
/// scheme carries. This is *which instruction* — constant per mnemonic, so each
/// mnemonic states it once and the operand form (`rrr`, `rm`, ...) supplies the
/// per-call registers.
#[derive(Clone, Copy)]
pub(super) struct Opcode<S> {
    map: Map,
    pp: Pp,
    w: bool,
    byte: u8,
    /// The imm8 that follows the operands: a `vcmpps` predicate, a rounding
    /// mode, a shift count, a lane index, a `vpternlogd` truth table.
    imm: Option<u8>,
    scheme: S,
}

/// A VEX-encoded instruction.
pub(super) type Vex = Opcode<VexPrefix>;
/// An EVEX-encoded instruction.
pub(super) type Evex = Opcode<EvexPrefix>;

impl<S: Scheme> Opcode<S> {
    const fn new(map: Map, pp: Pp, byte: u8) -> Self {
        Self {
            map,
            pp,
            w: false,
            byte,
            imm: None,
            scheme: S::FULL,
        }
    }
    /// Map `0F`, no prefix — the packed-single family.
    pub(super) const fn m0f(byte: u8) -> Self {
        Self::new(Map::M0F, Pp::None, byte)
    }
    /// Map `0F`, `66` — the integer-domain family.
    pub(super) const fn m0f_66(byte: u8) -> Self {
        Self::new(Map::M0F, Pp::P66, byte)
    }
    /// Map `0F`, `F3`.
    pub(super) const fn m0f_f3(byte: u8) -> Self {
        Self::new(Map::M0F, Pp::F3, byte)
    }
    /// Map `0F38`, `66`.
    pub(super) const fn m0f38_66(byte: u8) -> Self {
        Self::new(Map::M0F38, Pp::P66, byte)
    }
    /// Map `0F38`, `F3` — the mask-to-vector widening family.
    pub(super) const fn m0f38_f3(byte: u8) -> Self {
        Self::new(Map::M0F38, Pp::F3, byte)
    }
    /// Map `0F3A`, `66` — the imm8 family (round, insert/extract, ternlog).
    pub(super) const fn m0f3a_66(byte: u8) -> Self {
        Self::new(Map::M0F3A, Pp::P66, byte)
    }

    /// The 64-bit-operand (`W = 1`) form of this instruction.
    pub(super) const fn w1(self) -> Self {
        Self { w: true, ..self }
    }
    /// The 128-bit form of this instruction.
    pub(super) fn xmm(self) -> Self {
        Self {
            scheme: self.scheme.xmm(),
            ..self
        }
    }
    /// This instruction with its imm8, which every operand form emits last.
    pub(super) const fn imm(self, imm: u8) -> Self {
        Self {
            imm: Some(imm),
            ..self
        }
    }

    /// The instruction a three-operand [`Alu`] op is.
    pub(super) const fn alu(op: Alu) -> Self {
        match op {
            Alu::Add => Self::m0f(0x58),
            Alu::Sub => Self::m0f(0x5C),
            Alu::Mul => Self::m0f(0x59),
            Alu::Div => Self::m0f(0x5E),
            Alu::Min => Self::m0f(0x5D),
            Alu::Max => Self::m0f(0x5F),
            Alu::And => Self::m0f(0x54),
            Alu::AndNot => Self::m0f(0x55),
            Alu::Or => Self::m0f(0x56),
            Alu::Xor => Self::m0f(0x57),
            Alu::IAdd => Self::m0f_66(0xFE),
        }
    }

    /// The instruction a one-source [`Lanewise`] op is.
    pub(super) const fn lanewise(op: Lanewise) -> Self {
        match op {
            Lanewise::Sqrt => Self::m0f(0x51),
            Lanewise::Rsqrt => S::RSQRT,
            Lanewise::Recip => S::RECIP,
            Lanewise::ToInt => Self::m0f_f3(0x5B),
            Lanewise::FromInt => Self::m0f(0x5B),
            Lanewise::WidenBytes => Self::m0f38_66(0x31),
        }
    }

    /// The prefix and opcode, then `tail` (ModRM and what follows it), then the
    /// imm8.
    fn encode(self, ext: Ext, tail: impl FnOnce(&mut EncodedInst)) -> EncodedInst {
        let mut inst = EncodedInst::new();
        S::write(&self, ext, &mut inst);
        tail(&mut inst);
        if let Some(imm) = self.imm {
            inst.push(imm);
        }
        inst
    }

    /// `op reg, vvvv, rm` with `rm` a register: the three-operand form.
    pub(super) fn rrr(self, reg: u8, vvvv: u8, rm: u8) -> EncodedInst {
        self.registers(reg, Some(vvvv), rm)
    }

    /// `op reg, rm` with no `vvvv` source.
    pub(super) fn rr(self, reg: u8, rm: u8) -> EncodedInst {
        self.registers(reg, None, rm)
    }

    fn registers(self, reg: u8, vvvv: Option<u8>, rm: u8) -> EncodedInst {
        let ext = Ext {
            reg,
            x: bit(rm, 4),
            b: bit(rm, 3),
            vvvv,
            v: vvvv.unwrap_or(0),
        };
        self.encode(ext, |inst| inst.push(modrm_rr(reg, rm)))
    }

    /// An instruction with no operands.
    pub(super) fn bare(self) -> EncodedInst {
        self.encode(Ext::default(), |_| {})
    }

    /// `op reg, [base + disp]`: the memory-operand form, for any base and any
    /// displacement the scheme reads as written.
    pub(super) fn rm<D: DispIn<S>>(self, reg: u8, base: u8, disp: D) -> EncodedInst {
        let ext = Ext {
            reg,
            b: bit(base, 3),
            ..Ext::default()
        };
        self.encode(ext, |inst| mem_operand_into(inst, reg, base, disp))
    }

    /// `op reg, [rip + disp32]`: the displacement is the last four bytes, a
    /// label field.
    pub(super) fn rip(self, reg: u8) -> EncodedInst {
        let ext = Ext {
            reg,
            ..Ext::default()
        };
        debug_assert!(self.imm.is_none(), "an imm8 would follow the label field");
        self.encode(ext, |inst| rip_operand_into(inst, reg))
    }

    /// `op reg, [base + index*4]` — the SIB form with a scaled index, which a
    /// broadcast load reads one element of a plane through.
    pub(super) fn rm_scaled4(self, reg: u8, base: u8, index: u8) -> EncodedInst {
        debug_assert!(index & 7 != RM_SIB, "rsp/r12 cannot index a SIB");
        let ext = Ext {
            reg,
            x: bit(index, 3),
            b: bit(base, 3),
            ..Ext::default()
        };
        self.encode(ext, |inst| scaled4_operand_into(inst, reg, base, index))
    }

    /// `op reg, [base + vecINDEX*4], vvvv` — the VSIB form, whose index is a
    /// *vector* register: one address per lane, scale 4, no displacement. The
    /// SIB bytes are a GPR index's, but as a vector number `100` is
    /// `ymm4`/`ymm12`, which a gather may perfectly well be indexed by. VEX's
    /// `vvvv` is the gather's mask; EVEX has the mask in `aaa` and none here.
    pub(super) fn vsib_scaled4(
        self,
        reg: u8,
        vvvv: Option<u8>,
        base: u8,
        index: u8,
    ) -> EncodedInst {
        let ext = Ext {
            reg,
            x: bit(index, 3),
            b: bit(base, 3),
            vvvv,
            v: index,
        };
        self.encode(ext, |inst| scaled4_operand_into(inst, reg, base, index))
    }
}

impl Evex {
    /// This instruction under the writemask register numbered `k`.
    pub(super) const fn masked(self, k: u8) -> Self {
        Self {
            scheme: EvexPrefix {
                aaa: k,
                ..self.scheme
            },
            ..self
        }
    }
}

// =============================================================================
// The vector instructions both tiers share
// =============================================================================

/// The opcode of tier `T`'s prefix scheme.
type Code<T> = Opcode<<T as X86>::Prefix>;

/// An x86 vector instruction, generic over what its operands are ([`Stage`]),
/// like [`Gp`], and over the tier that selects it. VEX and EVEX are both
/// three-operand and non-destructive, so no destination is tied to a source
/// unless it says so with a `Tie`, and an operand may be the register the
/// result is written to.
///
/// The variants are the instructions both tiers have; an instruction only one
/// has is a variant of that tier's [`X86::Extra`], which the other tier's
/// type does not contain.
pub(super) enum Inst<S: Stage, T: X86> {
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
    /// `vroundps`/`vrndscaleps dst, src, mode`: scale 0, so an integer.
    /// (Opcode `08` is packed-single; `09` is packed-double and needs `W1`.)
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
    /// for both before either tier can be selected.
    Fma231 {
        acc: S::Tie<Vector>,
        a: S::Read<Vector>,
        b: S::Read<Vector>,
    },
    /// All-ones, whatever `dst` held ([`X86::ones`]). Its reads are not
    /// operands, because the result does not depend on them.
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
    /// `vcvttss2si dst, src`: lane 0, truncated to a 64-bit integer. EVEX
    /// where the tier is AVX-512, so the source may be `zmm16..31`.
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
    /// An instruction only tier `T` has.
    Extra(T::Extra<S>),
}

impl<T: X86> Inst<Selected, T> {
    /// Rebuild at stage `To`, visiting each operand once, in field order.
    fn walk<To: Stage>(&self, f: &mut impl Rebind<To>) -> Inst<To, T> {
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
            Inst::Extra(extra) => Inst::Extra(T::walk_extra(extra, f)),
        }
    }
}

impl<'m, T: X86> Inst<Bound<'m, T>, T> {
    fn encode(&self) -> EncodedInst {
        match self {
            Inst::Alu { op, dst, a, b } => {
                Code::<T>::alu(*op).rrr(dst.number(), a.number(), b.number())
            }
            Inst::Unary { op, dst, src } => Code::<T>::lanewise(*op).rr(dst.number(), src.number()),
            Inst::Round { mode, dst, src } => Code::<T>::m0f3a_66(0x08)
                .imm(*mode as u8)
                .rr(dst.number(), src.number()),
            // The destination is `vvvv` and the `/digit` is `reg`.
            Inst::Shift {
                direction,
                dst,
                src,
                amount,
            } => Code::<T>::m0f_66(0x72).imm(*amount).rrr(
                *direction as u8,
                dst.number(),
                src.number(),
            ),
            Inst::Fma231 { acc, a, b } => {
                Code::<T>::m0f38_66(0xB8).rrr(acc.number(), a.number(), b.number())
            }
            Inst::Ones { dst } => T::ones(dst.number()),
            Inst::Zero { dst } => {
                let dst = dst.number();
                Code::<T>::alu(Alu::Xor).rrr(dst, dst, dst)
            }
            Inst::Mov { dst, src } => Code::<T>::m0f(0x28).rr(dst.number(), src.number()),
            Inst::LoadConst { dst, .. } => Code::<T>::m0f38_66(0x18).rip(dst.number()),
            Inst::SpillVector { src, slot } => {
                let (base, disp) = frame(slot);
                Code::<T>::m0f(0x11).rm(src.number(), base, disp)
            }
            Inst::ReloadVector { dst, slot } => {
                let (base, disp) = frame(slot);
                Code::<T>::m0f(0x10).rm(dst.number(), base, disp)
            }
            Inst::Cvtt { dst, src } => Code::<T>::m0f_f3(0x2C).w1().rr(dst.number(), src.number()),
            Inst::Movq { dst, src } => Code::<T>::m0f_66(0x6E)
                .w1()
                .xmm()
                .rr(dst.number(), src.number()),
            Inst::StoreBatch { dst, src } => {
                Code::<T>::m0f(0x11).rm(src.number(), dst.base.number(), dst.disp)
            }
            Inst::Broadcast { dst, src } => {
                Code::<T>::m0f38_66(0x18).rm(dst.number(), src.base.number(), src.disp)
            }
            Inst::BroadcastIndexed { dst, base, index } => {
                Code::<T>::m0f38_66(0x18).rm_scaled4(dst.number(), base.number(), index.number())
            }
            Inst::Extra(extra) => T::encode_extra(extra),
        }
    }

    /// The instruction's bytes, and the label fields in them.
    fn assemble(&self, out: &mut Encoding<'_>) {
        let inst = self.encode();
        out.bytes(inst.as_bytes());
        if let Inst::LoadConst { at, .. } = self {
            // The displacement is the instruction's last four bytes.
            out.field(inst.as_bytes().len() - 4, *at, patch_rel32);
        }
    }
}

/// What the selection pipeline builds for tier `T`: a vector instruction or a
/// general-register one (the loop nest's, and the store's address
/// arithmetic). One type so a block holds both; no derives, because a
/// selected instruction owns its definitions.
pub(super) enum Op<S: Stage, T: X86> {
    Vector(Inst<S, T>),
    General(Gp<S>),
}

impl<T: X86> Op<Selected, T> {
    pub(super) fn walk<To: Stage>(&self, f: &mut impl Rebind<To>) -> Op<To, T> {
        match self {
            Op::Vector(inst) => Op::Vector(inst.walk(f)),
            Op::General(gp) => Op::General(gp.walk(f)),
        }
    }
}

impl<'m, T: X86> Op<Bound<'m, T>, T> {
    pub(super) fn assemble(&self, out: &mut Encoding<'_>) {
        match self {
            Op::Vector(inst) => inst.assemble(out),
            Op::General(gp) => gp.assemble(out),
        }
    }
}

// =============================================================================
// Displacements
// =============================================================================

/// Bytes one `f32` occupies: the element pitch of a uniform block.
pub(super) const F32_BYTES: u64 = 4;

/// The `disp32` that reaches element `index` of an array of `size`-byte
/// elements: the one place a uniform's or a pool entry's 64-bit position meets
/// the width x86 gives a displacement, shared by the VEX and EVEX tiers the
/// way [`mem_operand_into`] is.
///
/// # Errors
///
/// [`CompileError::BudgetExceeded`] when it does not fit a signed 32-bit
/// displacement: an offset past the encoding is refused, never wrapped into an
/// address that reads some other argument.
pub(super) fn displacement(index: u64, size: u64) -> Result<Imm32, CompileError> {
    index
        .checked_mul(size)
        .and_then(|bytes| i32::try_from(bytes).ok())
        .map(Imm32)
        .ok_or(CompileError::BudgetExceeded(
            "block offset past x86's disp32",
        ))
}

// =============================================================================
// Memory operands
// =============================================================================

/// The displacement half of a [`Mem`] — and, because x86 spells the
/// displacement's *width* in the ModRM `mod` field, the addressing mode.
///
/// `mod = 00 / 01 / 10` are three modes rather than three spellings of one:
/// they cost a different number of bytes, and `mod = 00` is not "a
/// displacement of zero" (see [`NoDisp`]). So the width is picked by the
/// operand's TYPE — never by the caller reaching for a differently-named
/// function.
pub(super) trait Disp: Copy {
    /// The ModRM `mod` field this displacement implies.
    const MOD: u8;
    /// Append the displacement bytes into an `EncodedInst`.
    fn emit_inst(self, inst: &mut EncodedInst);
}

/// No displacement — the `mod = 00` form, `[base]`.
///
/// A mode of its own, not `Imm8(0)`: it is a byte shorter, and it is not
/// available for every base. `mod = 00` with `rm = 101` (rbp/r13) means
/// RIP-relative, a different address entirely, so those two registers have no
/// bare `[base]` form and must spell it `Imm8(0)`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) struct NoDisp;

impl Disp for NoDisp {
    const MOD: u8 = 0x00;
    #[inline(always)]
    fn emit_inst(self, _inst: &mut EncodedInst) {}
}

impl Disp for Imm8 {
    const MOD: u8 = 0x40;
    #[inline(always)]
    fn emit_inst(self, inst: &mut EncodedInst) {
        inst.push(self.0 as u8);
    }
}

impl Disp for Imm32 {
    const MOD: u8 = 0x80;
    #[inline(always)]
    fn emit_inst(self, inst: &mut EncodedInst) {
        inst.extend(&self.0.to_le_bytes());
    }
}

/// An address spelled `[base + disp]`.
///
/// The base being a `Pointer` is the point: an address is a value, so one
/// encoder serves every base. `rsp` is in no class (the frame's, see
/// `regalloc::resource::Frame`), so a frame slot is not a `Mem`: it is
/// [`frame`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) struct Mem<S: Stage, D> {
    /// The register the displacement is measured from.
    pub(super) base: S::Read<Pointer>,
    /// The displacement — and, through its type, the mode (see [`Disp`]).
    pub(super) disp: D,
}

impl<D: Copy> Mem<Selected, D> {
    /// Rebuild at stage `T`: the base is the one operand.
    pub(super) fn walk<T: Stage>(&self, f: &mut impl Rebind<T>) -> Mem<T, D> {
        Mem {
            base: f.read(self.base),
            disp: self.disp,
        }
    }
}

/// ModRM `rm` meaning "a SIB byte follows" — also the low three bits of
/// `rsp`/`r12`, which is why exactly those two bases always take one.
const RM_SIB: u8 = 0b100;

/// ModRM `rm` that means RIP-relative when `mod = 00` — also the low three
/// bits of `rbp`/`r13`.
const RM_RIP_AT_MOD0: u8 = 0b101;

/// SIB naming the base register alone: scale 1, index `100` (none).
const SIB_BASE_ONLY: u8 = 0x24;

/// Write the ModRM/SIB tail for `[base + index*4]` into an `EncodedInst`:
/// `mod = 00`, a SIB with scale 4 and no displacement — the element of a plane
/// of `f32`s, addressed in one instruction, or a gather's VSIB. The tail is the
/// architecture's, not the prefix's, so VEX and EVEX share it the way they
/// share [`mem_operand_into`]. Registers are hardware numbers.
fn scaled4_operand_into(inst: &mut EncodedInst, reg: u8, base: u8, index: u8) {
    assert!(
        base & 7 != RM_RIP_AT_MOD0,
        "[r{base} + index*4] has no mod=00 form: rbp/r13 as a SIB base means no base"
    );
    inst.push(((reg & 7) << 3) | RM_SIB);
    inst.push((0b10 << 6) | ((index & 7) << 3) | (base & 7));
}

/// Write the ModRM/SIB/disp tail of `[base + disp]` into an `EncodedInst`.
fn mem_operand_into<D: Disp>(inst: &mut EncodedInst, reg: u8, base: u8, disp: D) {
    let rm = base & 7;
    debug_assert!(
        D::MOD != NoDisp::MOD || rm != RM_RIP_AT_MOD0,
        "[rbp]/[r13] has no mod=00 form: that encoding is RIP-relative"
    );
    inst.push(D::MOD | ((reg & 7) << 3) | rm);
    if rm == RM_SIB {
        inst.push(SIB_BASE_ONLY);
    }
    disp.emit_inst(inst);
}

/// The ModRM and `disp32` of `[rip + label]`: the displacement is a label
/// field, the last four bytes of the instruction.
fn rip_operand_into(inst: &mut EncodedInst, reg: u8) {
    inst.push(((reg & 7) << 3) | RM_RIP_AT_MOD0);
    inst.extend(&[0, 0, 0, 0]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emit::asm::{AsmProgram, Item, Labels, assemble};

    /// An instruction's bytes, and the label field in them if it has one.
    type Word = (EncodedInst, Option<(usize, Label)>);

    fn word(bytes: &[u8]) -> EncodedInst {
        let mut inst = EncodedInst::new();
        inst.extend(bytes);
        inst
    }

    /// `jmp rel32` with its label field, as the assembler is handed it.
    fn jmp(to: Label) -> Item<Word> {
        Item::Inst((word(&[0xE9, 0, 0, 0, 0]), Some((JMP_DISP, to))))
    }

    fn jb(to: Label) -> Item<Word> {
        Item::Inst((word(&[0x0F, 0x82, 0, 0, 0, 0]), Some((JCC_DISP, to))))
    }

    fn plain(bytes: &[u8]) -> Item<Word> {
        Item::Inst((word(bytes), None))
    }

    fn ret() -> Item<Word> {
        plain(&[0xC3])
    }

    fn assembled(text: Vec<Item<Word>>, labels: Labels) -> Vec<u8> {
        assemble(
            &AsmProgram {
                text,
                data: Vec::new(),
                labels,
            },
            |(inst, field), out| {
                out.bytes(inst.as_bytes());
                if let Some((at, label)) = *field {
                    out.field(at, label, patch_rel32);
                }
            },
        )
        .code
    }

    /// The one thing a label does that a fixup token could not: name a
    /// position that does not exist yet.
    #[test]
    fn a_forward_branch_names_a_position_bound_later() {
        let mut labels = Labels::default();
        let end = labels.mint();
        let code = assembled(vec![jmp(end), ret(), Item::Bind(end)], labels);

        // `jmp rel32` is five bytes; `ret` is one; the label lands at 6. The
        // displacement is measured from the end of the branch, so it is 1.
        assert_eq!(code.len(), 6);
        assert_eq!(code[0], 0xE9);
        assert_eq!(i32::from_le_bytes([code[1], code[2], code[3], code[4]]), 1);
    }

    /// A back edge — the shape a loop is made of, and the reason the
    /// resolution pass is separate from the layout pass.
    #[test]
    fn a_back_edge_resolves_to_a_negative_displacement() {
        let mut labels = Labels::default();
        let top = labels.mint();
        let code = assembled(vec![Item::Bind(top), ret(), jmp(top)], labels);

        // `ret` at 0, `jmp` at 1..6. Target 0, origin 6, so the displacement
        // is -6 — and getting this sign backwards is the classic way a loop
        // becomes an infinite one.
        assert_eq!(code.len(), 6);
        assert_eq!(i32::from_le_bytes([code[2], code[3], code[4], code[5]]), -6);
    }

    /// A label may name a position no instruction occupies — the end of the
    /// program. That is why binding a label is a step of its own rather than
    /// a field on an instruction: there is nothing here to hang it on.
    #[test]
    fn a_label_can_end_the_program() {
        let mut labels = Labels::default();
        let end = labels.mint();
        let code = assembled(vec![jmp(end), Item::Bind(end)], labels);
        assert_eq!(code.len(), 5);
        assert_eq!(i32::from_le_bytes([code[1], code[2], code[3], code[4]]), 0);
    }

    /// And two labels may name the same position, for the same reason.
    #[test]
    fn two_labels_can_share_a_position() {
        let mut labels = Labels::default();
        let (a, b) = (labels.mint(), labels.mint());
        let code = assembled(vec![jb(a), jmp(b), Item::Bind(a), Item::Bind(b)], labels);
        // `jb` is 6 bytes, `jmp` 5, both landing at 11.
        assert_eq!(code.len(), 11);
        assert_eq!(i32::from_le_bytes([code[2], code[3], code[4], code[5]]), 5);
        assert_eq!(i32::from_le_bytes([code[7], code[8], code[9], code[10]]), 0);
    }

    /// Offsets are relative to the program, not the buffer, so a program can
    /// be assembled after bytes that are already there.
    #[test]
    fn a_program_is_position_independent() {
        let program = |prefix: &[u8]| {
            let mut labels = Labels::default();
            let end = labels.mint();
            let mut text = Vec::new();
            if !prefix.is_empty() {
                text.push(plain(prefix));
            }
            text.extend([jmp(end), Item::Bind(end)]);
            assembled(text, labels)
        };
        let offset = program(&[0x90; 7]);
        assert_eq!(&program(&[])[..], &offset[7..]);
    }

    #[test]
    #[should_panic(expected = "never bound")]
    fn an_unbound_label_is_a_bug_and_not_a_jump_to_itself() {
        let mut labels = Labels::default();
        let target = labels.mint();
        let code = assembled(vec![jmp(target)], labels);
        unreachable!("assembled {} bytes around an unbound label", code.len());
    }

    #[test]
    #[should_panic(expected = "bound twice")]
    fn a_label_bound_twice_is_a_bug() {
        let mut labels = Labels::default();
        let twice = labels.mint();
        let code = assembled(vec![Item::Bind(twice), Item::Bind(twice)], labels);
        unreachable!("assembled {} bytes around a label bound twice", code.len());
    }
}
