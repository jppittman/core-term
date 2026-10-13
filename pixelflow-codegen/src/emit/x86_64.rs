//! x86-64 leaf encoders: what the AVX2 and AVX-512 tiers share below the
//! vector width.
//!
//! What is here is the *architecture's*: [`Gp`], the general-register
//! instructions the loop nest and the store's address arithmetic are made
//! of (branches, the pointer class's loads and stores and the frame among
//! them), the memory-operand tail (`Mem`, `Disp`) every vector encoder's
//! ModRM/SIB is built from, and the vector operations both tiers name ([`Alu`]
//! and its kin, whose bytes are each tier's). Nothing here names a vector
//! width: the `ymm`/`zmm` encodings live in `avx2.rs` and `avx512.rs`.

use super::asm::Encoding;
use super::build::{Builder, Def, Spiller};
use super::regalloc::resource::FrameSlot;
use super::{
    Bound, Class, ClassId, Edges, EncodedInst, Flags, FrameSize, Integer, IsaBackend, Label,
    Pointer, Rebind, STACK_PAGE, Selected, Stage, Target, Value, ValueName, frame_bytes, slot_disp,
};
use crate::error::CompileError;
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
    /// 1st integer argument: the context pointer — the array of bound buffer
    /// bases, then the uniform and origin blocks. Read-only for the whole
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
            // `rbp`/`r13` have no `mod = 00` form as a SIB base (that
            // encoding means "no base"), so those two take `mod = 01` with a
            // zero `disp8`.
            Gp::Lea4 { dst, base, index } => {
                let (dst, base, index) = (dst.number(), base.number(), index.number());
                debug_assert!(index & 7 != RM_SIB, "rsp/r12 cannot index a SIB");
                let disp8_form = base & 7 == RM_RIP_AT_MOD0;
                inst.push(
                    0x48 | (((dst >> 3) & 1) << 2) | (((index >> 3) & 1) << 1) | ((base >> 3) & 1),
                );
                inst.push(0x8D);
                inst.push(if disp8_form { 0x40 } else { 0x00 } | ((dst & 7) << 3) | RM_SIB);
                inst.push((0b10 << 6) | ((index & 7) << 3) | (base & 7));
                if disp8_form {
                    inst.push(0);
                }
            }
            Gp::MovLoad { dst, src } => {
                let (dst, base) = (dst.number(), src.base.number());
                inst.push(rex_w(dst, base));
                inst.push(0x8B);
                mem_operand_into(&mut inst, dst, base, src.disp);
            }
            Gp::SpillPtr { src, slot } => {
                spill_into(&mut inst, src.number(), Imm32(slot_disp(slot)));
            }
            Gp::SpillInt { src, slot } => {
                spill_into(&mut inst, src.number(), Imm32(slot_disp(slot)));
            }
            Gp::ReloadPtr { dst, slot } => {
                reload_into(&mut inst, dst.number(), Imm32(slot_disp(slot)));
            }
            Gp::ReloadInt { dst, slot } => {
                reload_into(&mut inst, dst.number(), Imm32(slot_disp(slot)));
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
                inst.extend(&[rex_w(0, gpr::RSP), 0x81, modrm_rr(0, gpr::RSP)]);
                inst.extend(&frame_bytes(*size).to_le_bytes());
                // `vzeroupper`: `VEX.128.0F.WIG 77`, zeroing bits 128 and up of
                // vector registers 0–15, the sixteen a legacy-SSE instruction
                // can name. The same three bytes on every tier.
                inst.extend(&[0xC5, 0xF8, 0x77]);
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

/// An x86 backend: its instructions are its vector ones and these.
pub(super) trait X86: IsaBackend {
    /// `inst` as this backend's instruction.
    fn general(inst: Gp<Selected>) -> Self::Inst<Selected>;
}

/// Define a value of class `C` with `make`, which builds the
/// general-register instruction that writes it.
pub(super) fn define<B: X86, C: Class>(
    b: &mut Builder<B>,
    make: impl FnOnce(Def<C>) -> Gp<Selected>,
) -> Value<C> {
    let dst = b.def();
    let value = dst.value();
    b.push(B::general(make(dst)));
    value
}

/// `mov p, [ctx + 8·slot]`.
pub(super) fn context<B: X86>(
    b: &mut Builder<B>,
    ctx: Value<Pointer>,
    slot: u64,
) -> Result<Value<Pointer>, CompileError> {
    let disp = displacement(slot, PTR_BYTES as u64)?;
    Ok(define(b, |dst| Gp::MovLoad {
        dst,
        src: Mem { base: ctx, disp },
    }))
}

/// `out + 4·(row·pitch + col)`: where the lattice's store writes, from the
/// two indices as integers.
pub(super) fn element_address<B: X86>(
    b: &mut Builder<B>,
    out: Value<Pointer>,
    pitch: Value<Integer>,
    (row, col): (Value<Integer>, Value<Integer>),
) -> Value<Pointer> {
    let (scaled, flags) = (b.tie(row), b.def());
    let row_pitch = scaled.write();
    b.push(B::general(Gp::Imul {
        dst: scaled,
        src: pitch,
        flags,
    }));
    let (summed, flags) = (b.tie(row_pitch), b.def());
    let index = summed.write();
    b.push(B::general(Gp::Add {
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

/// End the block: `jcc` to `edges.taken` when `flags` meet `cond`, else fall
/// through to `edges.next`.
pub(super) fn branch<B: X86>(b: &mut Builder<B>, cond: Cond, flags: Value<Flags>, edges: Edges) {
    let next = Target {
        label: edges.next,
        args: Vec::new(),
    };
    b.push(B::general(Gp::Jcc {
        cond,
        flags,
        taken: edges.taken,
        next,
    }));
}

/// End the block: go to `to`. A `Fallthrough` when `to` is `next`.
pub(super) fn jump<B: X86>(b: &mut Builder<B>, to: Target, next: Label) {
    b.push(B::general(if to.label == next {
        Gp::Fallthrough { to }
    } else {
        Gp::Jmp { to }
    }));
}

/// `sub rsp, frame`: the function's entry.
pub(super) fn enter<B: X86>(b: &mut Builder<B>) {
    let flags = b.def();
    b.push(B::general(Gp::Enter {
        size: FrameSize,
        flags,
    }));
}

/// `add rsp, frame; vzeroupper; ret`: the function's one exit.
pub(super) fn ret<B: X86>(b: &mut Builder<B>) {
    let flags = b.def();
    b.push(B::general(Gp::Ret {
        size: FrameSize,
        flags,
    }));
}

/// `mov` of an address or an integer between general registers.
pub(super) fn copy_general<B: X86>(b: &mut Spiller<'_, B>, src: ValueName) -> ValueName {
    match src.class {
        ClassId::Pointer => b.define::<Pointer>(|dst| {
            B::general(Gp::Mov {
                dst,
                src: src.typed(),
            })
        }),
        ClassId::Integer => b.define::<Integer>(|dst| {
            B::general(Gp::MovInt {
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
    b.push(B::general(match src.class {
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
        ClassId::Pointer => b.define::<Pointer>(|dst| B::general(Gp::ReloadPtr { dst, slot })),
        ClassId::Integer => b.define::<Integer>(|dst| B::general(Gp::ReloadInt { dst, slot })),
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

/// `sub rsp, imm32`.
fn sub_rsp(bytes: u32) -> EncodedInst {
    let mut inst = EncodedInst::new();
    inst.extend(&[rex_w(0, gpr::RSP), 0x81, modrm_rr(5, gpr::RSP)]);
    inst.extend(&bytes.to_le_bytes());
    inst
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

/// `mov [rsp + slot], src`.
fn spill_into(inst: &mut EncodedInst, src: u8, slot: Imm32) {
    inst.push(rex_w(src, gpr::RSP));
    inst.push(0x89);
    mem_operand_into(inst, src, gpr::RSP, slot);
}

/// `mov dst, [rsp + slot]`.
fn reload_into(inst: &mut EncodedInst, dst: u8, slot: Imm32) {
    inst.push(rex_w(dst, gpr::RSP));
    inst.push(0x8B);
    mem_operand_into(inst, dst, gpr::RSP, slot);
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

// What an operation *is* is the architecture's; the bytes that say it are the
// tier's, so `avx2.rs` and `avx512.rs` each give these a `vex()` or `evex()`.

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
/// The base being a `Pointer` is the point: `rsp` is a value here, so one
/// encoder serves it and every other base.
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
/// `mod = 00`, a SIB with scale 4 and no displacement — the element of a
/// plane of `f32`s, addressed in one instruction. The tail is the
/// architecture's, not the prefix's, so the VEX and EVEX tiers share it the
/// way they share [`mem_operand_into`]. Registers are hardware numbers.
pub(super) fn scaled4_operand_into(inst: &mut EncodedInst, reg: u8, base: u8, index: u8) {
    debug_assert!(index & 7 != RM_SIB, "rsp/r12 cannot index a SIB");
    sib4_tail_into(inst, reg, base, index);
}

/// [`scaled4_operand_into`] with a *vector* index — the VSIB a gather
/// addresses through, `[base + ymm*4]`. Same bytes; the one rule that does
/// not carry over is the GPR one, because SIB index `100` means "no index"
/// only when the index is a general register: as a vector number it is
/// `ymm4`/`ymm12`, which a gather may perfectly well be indexed by. The
/// prefix's X bit carries the index's high bit either way.
pub(super) fn vsib4_operand_into(inst: &mut EncodedInst, reg: u8, base: u8, index: u8) {
    sib4_tail_into(inst, reg, base, index);
}

/// The ModRM/SIB bytes both scaled-index forms share.
fn sib4_tail_into(inst: &mut EncodedInst, reg: u8, base: u8, index: u8) {
    assert!(
        base & 7 != RM_RIP_AT_MOD0,
        "[r{base} + index*4] has no mod=00 form: rbp/r13 as a SIB base means no base"
    );
    inst.push(((reg & 7) << 3) | RM_SIB);
    inst.push((0b10 << 6) | ((index & 7) << 3) | (base & 7));
}

/// Write the ModRM/SIB/disp tail of `[base + disp]` into an `EncodedInst`.
pub(super) fn mem_operand_into<D: Disp>(inst: &mut EncodedInst, reg: u8, base: u8, disp: D) {
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
pub(super) fn rip_operand_into(inst: &mut EncodedInst, reg: u8) {
    inst.push(((reg & 7) << 3) | RM_RIP_AT_MOD0);
    inst.extend(&[0, 0, 0, 0]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emit::asm::{AsmProgram, Item, Labels, assemble};

    /// An instruction's bytes, and the label field in them if it has one.
    type Word = (EncodedInst, Option<(usize, Label)>);

    /// `jmp rel32` with its label field, as the assembler is handed it.
    fn jmp(to: Label) -> Item<Word> {
        Item::Inst((
            EncodedInst::from_slice(&[0xE9, 0, 0, 0, 0]),
            Some((JMP_DISP, to)),
        ))
    }

    fn jb(to: Label) -> Item<Word> {
        Item::Inst((
            EncodedInst::from_slice(&[0x0F, 0x82, 0, 0, 0, 0]),
            Some((JCC_DISP, to)),
        ))
    }

    fn plain(bytes: &[u8]) -> Item<Word> {
        Item::Inst((EncodedInst::from_slice(bytes), None))
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

    /// A vehicle for the memory-operand tail: `movups [base + disp], xmm`
    /// (`0F 11 /r`), the simplest legacy instruction that takes a memory
    /// operand. Test-only — no tier emits it — so what is under test is
    /// [`mem_operand_into`] and the REX bits, not a 128-bit store.
    fn movups_store<D: Disp>(src: u8, base: u8, disp: D) -> Vec<u8> {
        let mut inst = EncodedInst::new();
        let rex = 0x40 | (u8::from(src >= 8) << 2) | u8::from(base >= 8);
        if rex != 0x40 {
            inst.push(rex);
        }
        inst.push(0x0F);
        inst.push(0x11);
        mem_operand_into(&mut inst, src, base, disp);
        inst.as_bytes().to_vec()
    }

    /// REX extends each side of a memory operand independently: R for the
    /// register, B for the base, both, or neither.
    #[test]
    fn a_memory_operands_rex_bits_follow_its_two_registers() {
        let store = |reg, base| movups_store(reg, base, NoDisp);
        assert_eq!(store(3, 3), [0x0F, 0x11, ((3 & 7) << 3) | 3]);
        assert_eq!(store(9, 3), [0x44, 0x0F, 0x11, ((9 & 7) << 3) | 3]);
        assert_eq!(store(2, 11), [0x41, 0x0F, 0x11, ((2 & 7) << 3) | (11 & 7)]);
        assert_eq!(
            store(11, 14),
            [0x45, 0x0F, 0x11, ((11 & 7) << 3) | (14 & 7)]
        );
    }

    /// One instruction, three bases: `[rsp+8]`, `[rax+8]` and `[r10+8]`
    /// differ only in ModRM.rm (plus the SIB `rsp` implies and the REX.B `r10`
    /// does). That is why the base is an operand and not a name suffix.
    #[test]
    fn the_base_register_is_an_operand() {
        let store = |base| movups_store(1, base, Imm8(8));
        // 0F 11 /r, mod=01: rsp takes a SIB byte, rax and r10 do not.
        assert_eq!(store(4), [0x0F, 0x11, 0x4C, 0x24, 0x08]);
        assert_eq!(store(0), [0x0F, 0x11, 0x48, 0x08]);
        assert_eq!(store(10), [0x41, 0x0F, 0x11, 0x4A, 0x08]);
    }

    /// The displacement's *type* picks the encoding, so the same address at the
    /// same offset is a 5-byte or an 8-byte instruction depending on which
    /// operand the caller built — and `NoDisp` is a third, shorter mode, not
    /// `Imm8(0)`.
    #[test]
    fn the_displacement_type_picks_the_encoding() {
        let d8 = movups_store(0, 4, Imm8(16));
        let d32 = movups_store(0, 4, Imm32(16));
        assert_eq!(d8, [0x0F, 0x11, 0x44, 0x24, 0x10], "mod=01, disp8");
        assert_eq!(
            d32,
            [0x0F, 0x11, 0x84, 0x24, 0x10, 0x00, 0x00, 0x00],
            "mod=10, disp32"
        );

        let bare = movups_store(0, 6, NoDisp);
        let zero = movups_store(0, 6, Imm8(0));
        assert_eq!(bare, [0x0F, 0x11, 0x06], "mod=00 is its own mode");
        assert_eq!(zero, [0x0F, 0x11, 0x46, 0x00], "and a byte longer than it");
    }
}
