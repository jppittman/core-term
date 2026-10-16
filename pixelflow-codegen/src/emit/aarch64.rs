//! ARM64/NEON instruction encoding.
//!
//! [`Inst`] is one arm per instruction, with named operand fields; the free
//! functions here are the small fixed sequences compound operations are built
//! from.
//!
//! [`Neon`] is the machine the compiler targets: it selects [`Inst`]s over
//! values and leaves the registers to the allocator.

use super::asm::Encoding;
use super::build::{Builder, Def, Spiller};
use super::regalloc::resource::{FrameSlot, RegNo};
use super::register_file::{EntryRegisters, Members, RegisterFile};
use super::{
    Bound, Class, ClassId, Edges, File, FrameSize, GeneralFile, IfArm, Integer, IsaBackend, Label,
    LaneOp, Patch, Pointer, Rebind, STACK_PAGE, Selected, Stage, Store, Target, Test, Value,
    ValueName, Vector, frame_bytes, slot_disp,
};
use crate::error::CompileError;
use crate::program::{BinaryOp, CompareOp, ShiftOp, UnaryOp};
use alloc::vec::Vec;

mod table;
use table::*;

// =============================================================================
// Instruction Encoding Helpers
// =============================================================================

/// An A64 instruction is one 32-bit word, and a branch displacement counts
/// them.
const WORD_BYTES: usize = 4;

/// `Rd`: the destination field, at bit 0. This and the three below are where
/// a register number becomes bits: the only readers of one in this file.
const fn rd<F: File>(reg: RegNo<F>) -> u32 {
    reg.bits() as u32 & 0x1F
}

/// `Rn`: the first source or base field, at bit 5.
const fn rn<F: File>(reg: RegNo<F>) -> u32 {
    (reg.bits() as u32 & 0x1F) << 5
}

/// `Rm`: the second source field, at bit 16.
const fn rm<F: File>(reg: RegNo<F>) -> u32 {
    (reg.bits() as u32 & 0x1F) << 16
}

/// `Ra`: the addend field of a multiply-add, at bit 10.
const fn ra<F: File>(reg: RegNo<F>) -> u32 {
    (reg.bits() as u32 & 0x1F) << 10
}

/// The `imm5` of a 32-bit lane: the lane number above a `100` size marker.
///
/// # Panics
///
/// If `lane` is not one of a 128-bit register's four. A lane of 4 carries out
/// of `imm5` into bit 21 and encodes a different instruction. Emission is
/// compile time, so the check costs no render.
fn imm5_lane(lane: u8) -> u32 {
    assert!(
        lane < 4,
        "Vn.S[{lane}]: a 128-bit register has four 32-bit lanes"
    );
    ((lane as u32) << 3) | 0b100
}

// =============================================================================
// First-Class AArch64 Instructions
// =============================================================================

/// A concrete ARM64 instruction.
///
/// Generic over what its operands are ([`Stage`]); a field is declared by the
/// class of what it holds and by what the instruction does to it, so
/// `ldr w`'s index is an `Integer` and its base a `Pointer`. Every
/// instruction is one 32-bit word, except [`Inst::CbzFar`],
/// [`Inst::AdrpAdd`] and [`Inst::ProbePage`], which are two, and the frame's
/// [`Inst::Enter`] (a [`Inst::ProbePage`] per page past the first, then at most
/// one sub) and [`Inst::Ret`] (at most two adds, then `ret`). A displacement
/// past the scaled immediate is added into a value the allocator places,
/// before the instruction.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) enum Inst<S: Stage> {
    /// `op dst.4s, a.4s, b.4s`
    Alu {
        op: Alu,
        dst: S::Write<Vector>,
        a: S::Read<Vector>,
        b: S::Read<Vector>,
    },
    /// `fmla acc.4s, a.4s, b.4s`: `acc += a·b`, one rounding.
    Fmla {
        acc: S::Tie<Vector>,
        a: S::Read<Vector>,
        b: S::Read<Vector>,
    },
    /// `op dst.4s, src.4s`
    Unary {
        op: Lanewise,
        dst: S::Write<Vector>,
        src: S::Read<Vector>,
    },
    /// `bsl mask.16b, if_true.16b, if_false.16b`: the mask register becomes
    /// `(mask & if_true) | (!mask & if_false)`.
    Bsl {
        mask: S::Tie<Vector>,
        if_true: S::Read<Vector>,
        if_false: S::Read<Vector>,
    },
    /// `shl dst.4s, src.4s, #amount`
    Shl {
        dst: S::Write<Vector>,
        src: S::Read<Vector>,
        amount: u8,
    },
    /// `ushr dst.4s, src.4s, #amount`: logical, 1 to 32.
    Ushr {
        dst: S::Write<Vector>,
        src: S::Read<Vector>,
        amount: u8,
    },
    /// `mov dst.16b, src.16b`: nothing when they are the same register.
    Mov {
        dst: S::Write<Vector>,
        src: S::Read<Vector>,
    },
    /// `dup dst.4s, src.s[0]`
    DupLane0 {
        dst: S::Write<Vector>,
        src: S::Read<Vector>,
    },
    /// `umov dst.w, src.s[lane]`
    UmovW {
        dst: S::Write<Integer>,
        src: S::Read<Vector>,
        lane: u8,
    },
    /// `ins dst.s[0], src.w`: the first lane of a vector the gather builds.
    /// The other three lanes are not defined by it.
    InsFirst {
        dst: S::Write<Vector>,
        src: S::Read<Integer>,
    },
    /// `ins v.s[lane], src.w`: one more lane of a vector under construction.
    InsLane {
        v: S::Tie<Vector>,
        lane: u8,
        src: S::Read<Integer>,
    },
    /// `mvn w, w`: the register inverted in place.
    MvnW { dst: S::Tie<Integer> },
    /// `movi dst.4s, #0`
    Zero { dst: S::Write<Vector> },
    /// `fmov dst.4s, #imm8`: the 8-bit floating-point immediate.
    FmovImm { dst: S::Write<Vector>, imm8: u8 },
    /// `madd dst.x, a.x, b.x, addend.x`: `dst = a·b + addend`.
    Madd {
        dst: S::Write<Integer>,
        a: S::Read<Integer>,
        b: S::Read<Integer>,
        addend: S::Read<Integer>,
    },
    /// `add dst.x, base.x, index.x, lsl 2`: the address of an `f32` element.
    AddLsl2 {
        dst: S::Write<Pointer>,
        base: S::Read<Pointer>,
        index: S::Read<Integer>,
    },
    /// `st1 {src.s}[lane], [base], #4`: one lane stored, the address moving
    /// past it.
    St1Lane {
        base: S::Tie<Pointer>,
        src: S::Read<Vector>,
        lane: u8,
    },
    /// `fcvtzs dst.x, src.s`: lane 0, truncated to a 64-bit integer. How a
    /// fold's binder, or a broadcast load's index, becomes an address.
    FcvtzsX {
        dst: S::Write<Integer>,
        src: S::Read<Vector>,
    },
    /// `fmov dst.w, src.s`: lane 0 to a general register, the guard's test.
    FmovToGp {
        dst: S::Write<Integer>,
        src: S::Read<Vector>,
    },
    /// `mov dst.x, src.x`: an address between pointer registers.
    MovX {
        dst: S::Write<Pointer>,
        src: S::Read<Pointer>,
    },
    /// `mov dst.x, src.x`: an integer between registers. [`Inst::MovX`]'s
    /// word, and not the same instruction, because an address and an integer
    /// are different classes.
    MovInt {
        dst: S::Write<Integer>,
        src: S::Read<Integer>,
    },
    /// `add dst.x, src.x, #imm, lsl 12`: the high part of an address too far
    /// for a transfer's scaled immediate, which carries the low part.
    AddImmHi {
        dst: S::Write<Pointer>,
        src: S::Read<Pointer>,
        imm: Imm12,
    },
    /// `add dst.x, sp, #imm, lsl 12`: [`Inst::AddImmHi`] of a frame slot.
    AddSpHi { dst: S::Write<Pointer>, imm: Imm12 },
    /// `ldr q<dst>, [base, #offset]`
    LdrQ { dst: S::Write<Vector>, addr: Mem<S> },
    /// `str q<src>, [base, #offset]`
    StrQ { src: S::Read<Vector>, addr: Mem<S> },
    /// `ldr q<dst>, [pool base, #offset]`: one pool entry. [`Inst::LdrQ`]'s
    /// word, and the instruction the allocator places again where a read needs
    /// the constant.
    LoadConst { dst: S::Write<Vector>, addr: Mem<S> },
    /// `str q<src>, [sp, #slot]`: the allocator's spill of a vector.
    SpillQ { src: S::Read<Vector>, slot: S::Slot },
    /// `ldr q<dst>, [sp, #slot]`: the allocator's reload of a vector.
    ReloadQ {
        dst: S::Write<Vector>,
        slot: S::Slot,
    },
    /// `str x<src>, [sp, #slot]`: the allocator's spill of an address.
    SpillPtr {
        src: S::Read<Pointer>,
        slot: S::Slot,
    },
    /// `ldr x<dst>, [sp, #slot]`: the allocator's reload of an address.
    ReloadPtr {
        dst: S::Write<Pointer>,
        slot: S::Slot,
    },
    /// `str x<src>, [sp, #slot]`: the allocator's spill of an integer.
    SpillInt {
        src: S::Read<Integer>,
        slot: S::Slot,
    },
    /// `ldr x<dst>, [sp, #slot]`: the allocator's reload of an integer.
    ReloadInt {
        dst: S::Write<Integer>,
        slot: S::Slot,
    },
    /// `ldr x<dst>, [base, #offset]`
    LdrX {
        dst: S::Write<Pointer>,
        addr: Mem<S>,
    },
    /// `ldr s<dst>, [base, #offset]`
    LdrS { dst: S::Write<Vector>, addr: Mem<S> },
    /// `ldr w<dst>, [base, w<index>, uxtw #2]`
    LdrW {
        dst: S::Write<Integer>,
        addr: MemIndexed<S>,
    },
    /// `ldr s<dst>, [base, w<index>, uxtw #2]`: one element of a plane of
    /// `f32`s straight into lane 0, where a `dup` can spread it.
    LdrSIndexed {
        dst: S::Write<Vector>,
        addr: MemIndexed<S>,
    },
    /// `b to`: ±128 MiB.
    B { to: S::Target },
    /// `cbnz test.w, .+8; b taken`: branch to `taken` when `test` is zero,
    /// from anywhere within `B`'s ±128 MiB.
    ///
    /// **Two instructions, because `CBZ` does not reach.** It spends 19 bits on
    /// a word displacement, ±1 MiB, and an arm is as long as the code it owns,
    /// a bound nothing here puts under 1 MiB: a text run's glyph kernels under
    /// one guard are megabytes of NEON. `CBNZ` steps over the `B` when the
    /// mask says the arm is live, and falls into the `B` when it says the arm
    /// is dead, so this is exactly `cbz test, taken` at `B`'s reach.
    ///
    /// **Always both, never the one-word `CBZ` where it would reach.** Which
    /// form fits depends on the arm's length, which depends on where every
    /// instruction inside it landed, and this instruction's own size is one of
    /// those, so choosing the short form is branch relaxation: layout iterated
    /// to a fixed point to save four bytes per guard. The pair's size is fixed
    /// before a single byte is laid out, like `AdrpAdd`'s, and there is nothing
    /// to relax.
    ///
    /// `next` is the position laid out right after the pair, where the branch
    /// falls through to: it encodes to nothing.
    CbzFar {
        test: S::Read<Integer>,
        taken: S::Target,
        next: S::Target,
    },
    /// `adrp dst, to; add dst, dst, :lo12:to`, sharing one label: materialize
    /// the constant pool's address in `dst`.
    ///
    /// **Two instructions, because A64 is fixed 32-bit** — no single
    /// instruction holds a 64-bit address. `ADR` reaches ±1 MiB by spending a
    /// 21-bit *byte* displacement; `ADRP` spends the same 21 bits on 4 KiB
    /// *pages* instead — `(PC & !0xFFF) + (imm21 << 12)`, ±4 GiB — and hands
    /// back the base of the target's page, not the target itself. The `ADD`'s
    /// 12-bit immediate is exactly one page wide, so it recovers the low bits
    /// `ADRP` had to discard.
    ///
    /// **Always both, never the one-instruction `ADR`.** Which one is
    /// reachable depends on the distance to the pool; the pool's position
    /// depends on where every instruction ahead of it landed; and this
    /// instruction's own size is one of those — so choosing the short form is
    /// branch relaxation, and resolving it needs layout iterated to a fixed
    /// point to save four bytes once per compiled function. `AdrpAdd`'s size is
    /// fixed before a single byte is laid out, like every other instruction
    /// here.
    ///
    /// # Alignment invariant
    ///
    /// [`patch_adrp_add`] computes pages by masking *buffer offsets* —
    /// positions within the `Vec<u8>` this crate is building, not runtime
    /// addresses. That is correct only because the executable mapping this
    /// buffer is copied into starts on a 4 KiB boundary: `(map + pos) &
    /// !0xFFF == map + (pos & !0xFFF)` for every `pos` exactly when `map` is a
    /// multiple of 4 KiB, which is what lets a page found by masking an offset
    /// stand in for the page a masked address would find. Copy these bytes to
    /// an address that is not 4 KiB-aligned and every `ADRP` here is off by a
    /// page.
    ///
    /// Two things hold it up, and both are checked rather than assumed:
    /// `CodePage::from_code` writes the buffer at offset 0 of a mapping whose
    /// size — and therefore whose base — is a whole number of pages, pinned by
    /// `page_size_is_a_sane_power_of_two`; and an `asm::AsmProgram`
    /// position is an offset into the *whole* kernel rather than into the
    /// scope that emitted it, because one kernel is one program, so there is
    /// no base to subtract. A displacement cannot tell those two apart. A page
    /// can.
    AdrpAdd { dst: S::Write<Pointer>, to: Label },
    /// `sub sp, sp, #1, lsl 12; str xzr, [sp]`: one page of the frame, taken
    /// and touched. A frame past a page is entered by this, once per page,
    /// then a `sub` for the rest: a thread's stack ends in one guard page, and
    /// a single `sub` past it lands in whatever is mapped beyond without
    /// touching it. It names no register, because `sp` is the frame's and
    /// `xzr` is not in any class.
    ProbePage,
    /// `sub sp, sp, #frame`: the function's entry, a page at a time past one
    /// page ([`Inst::ProbePage`]).
    Enter { size: S::FrameSize },
    /// `add sp, sp, #frame; ret`: the function's one exit.
    Ret { size: S::FrameSize },
    /// Go to the position laid out right after this instruction: encodes to
    /// nothing, and is what a transfer to the next block is.
    Fallthrough { to: S::Target },
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
            Inst::Fmla { acc, a, b } => Inst::Fmla {
                acc: f.tie(acc),
                a: f.read(*a),
                b: f.read(*b),
            },
            Inst::Unary { op, dst, src } => Inst::Unary {
                op: *op,
                dst: f.write(dst),
                src: f.read(*src),
            },
            Inst::Bsl {
                mask,
                if_true,
                if_false,
            } => Inst::Bsl {
                mask: f.tie(mask),
                if_true: f.read(*if_true),
                if_false: f.read(*if_false),
            },
            Inst::Shl { dst, src, amount } => Inst::Shl {
                dst: f.write(dst),
                src: f.read(*src),
                amount: *amount,
            },
            Inst::Ushr { dst, src, amount } => Inst::Ushr {
                dst: f.write(dst),
                src: f.read(*src),
                amount: *amount,
            },
            Inst::Mov { dst, src } => Inst::Mov {
                dst: f.write(dst),
                src: f.read(*src),
            },
            Inst::DupLane0 { dst, src } => Inst::DupLane0 {
                dst: f.write(dst),
                src: f.read(*src),
            },
            Inst::UmovW { dst, src, lane } => Inst::UmovW {
                dst: f.write(dst),
                src: f.read(*src),
                lane: *lane,
            },
            Inst::InsFirst { dst, src } => Inst::InsFirst {
                dst: f.write(dst),
                src: f.read(*src),
            },
            Inst::InsLane { v, lane, src } => Inst::InsLane {
                v: f.tie(v),
                lane: *lane,
                src: f.read(*src),
            },
            Inst::MvnW { dst } => Inst::MvnW { dst: f.tie(dst) },
            Inst::Zero { dst } => Inst::Zero { dst: f.write(dst) },
            Inst::FmovImm { dst, imm8 } => Inst::FmovImm {
                dst: f.write(dst),
                imm8: *imm8,
            },
            Inst::Madd { dst, a, b, addend } => Inst::Madd {
                dst: f.write(dst),
                a: f.read(*a),
                b: f.read(*b),
                addend: f.read(*addend),
            },
            Inst::AddLsl2 { dst, base, index } => Inst::AddLsl2 {
                dst: f.write(dst),
                base: f.read(*base),
                index: f.read(*index),
            },
            Inst::St1Lane { base, src, lane } => Inst::St1Lane {
                base: f.tie(base),
                src: f.read(*src),
                lane: *lane,
            },
            Inst::FcvtzsX { dst, src } => Inst::FcvtzsX {
                dst: f.write(dst),
                src: f.read(*src),
            },
            Inst::FmovToGp { dst, src } => Inst::FmovToGp {
                dst: f.write(dst),
                src: f.read(*src),
            },
            Inst::MovX { dst, src } => Inst::MovX {
                dst: f.write(dst),
                src: f.read(*src),
            },
            Inst::MovInt { dst, src } => Inst::MovInt {
                dst: f.write(dst),
                src: f.read(*src),
            },
            Inst::AddImmHi { dst, src, imm } => Inst::AddImmHi {
                dst: f.write(dst),
                src: f.read(*src),
                imm: *imm,
            },
            Inst::AddSpHi { dst, imm } => Inst::AddSpHi {
                dst: f.write(dst),
                imm: *imm,
            },
            Inst::LdrQ { dst, addr } => Inst::LdrQ {
                dst: f.write(dst),
                addr: addr.walk(f),
            },
            Inst::StrQ { src, addr } => Inst::StrQ {
                src: f.read(*src),
                addr: addr.walk(f),
            },
            Inst::LoadConst { dst, addr } => Inst::LoadConst {
                dst: f.write(dst),
                addr: addr.walk(f),
            },
            Inst::SpillQ { src, slot } => Inst::SpillQ {
                src: f.read(*src),
                slot: f.slot(*slot),
            },
            Inst::ReloadQ { dst, slot } => Inst::ReloadQ {
                dst: f.write(dst),
                slot: f.slot(*slot),
            },
            Inst::SpillPtr { src, slot } => Inst::SpillPtr {
                src: f.read(*src),
                slot: f.slot(*slot),
            },
            Inst::ReloadPtr { dst, slot } => Inst::ReloadPtr {
                dst: f.write(dst),
                slot: f.slot(*slot),
            },
            Inst::SpillInt { src, slot } => Inst::SpillInt {
                src: f.read(*src),
                slot: f.slot(*slot),
            },
            Inst::ReloadInt { dst, slot } => Inst::ReloadInt {
                dst: f.write(dst),
                slot: f.slot(*slot),
            },
            Inst::LdrX { dst, addr } => Inst::LdrX {
                dst: f.write(dst),
                addr: addr.walk(f),
            },
            Inst::LdrS { dst, addr } => Inst::LdrS {
                dst: f.write(dst),
                addr: addr.walk(f),
            },
            Inst::LdrW { dst, addr } => Inst::LdrW {
                dst: f.write(dst),
                addr: addr.walk(f),
            },
            Inst::LdrSIndexed { dst, addr } => Inst::LdrSIndexed {
                dst: f.write(dst),
                addr: addr.walk(f),
            },
            Inst::B { to } => Inst::B { to: f.target(to) },
            Inst::CbzFar { test, taken, next } => Inst::CbzFar {
                test: f.read(*test),
                taken: f.target(taken),
                next: f.target(next),
            },
            Inst::AdrpAdd { dst, to } => Inst::AdrpAdd {
                dst: f.write(dst),
                to: *to,
            },
            Inst::ProbePage => Inst::ProbePage,
            Inst::Enter { .. } => Inst::Enter {
                size: f.frame_size(),
            },
            Inst::Ret { .. } => Inst::Ret {
                size: f.frame_size(),
            },
            Inst::Fallthrough { to } => Inst::Fallthrough { to: f.target(to) },
        }
    }
}

impl<'m, B: IsaBackend> Inst<Bound<'m, B>> {
    /// The instruction's word; for a sequence, [`Inst::words`] has the rest.
    ///
    /// # Panics
    ///
    /// If an immediate or a lane does not fit its field: those are the
    /// encoding's limits, and a wrapped one would be some other instruction.
    #[must_use]
    fn encode(&self) -> u32 {
        // `opcode | imm12 | Rn | Rt`: the scaled-immediate transfers, `rt`
        // already placed in its field.
        let transfer = |opcode: u32, rt: u32, addr: &Mem<Bound<'m, B>>, access: u32| {
            let imm12 = addr.scaled(access);
            assert!(
                imm12 <= MAX_IMM12,
                "offset {} exceeds the 12-bit scaled immediate",
                addr.offset
            );
            opcode | (imm12 << 10) | rn(addr.base.number()) | rt
        };
        // The same, from the stack pointer, at the slot's offset.
        let frame = |opcode: u32, rt: u32, slot: &FrameSlot, access: u32| {
            let offset = u32::try_from(slot_disp(slot)).expect("a slot is above the stack pointer");
            let imm12 = scaled(offset, access);
            assert!(
                imm12 <= MAX_IMM12,
                "slot {offset} exceeds the 12-bit scaled immediate"
            );
            opcode | (imm12 << 10) | rn(abi::SP) | rt
        };
        match self {
            Inst::Alu { op, dst, a, b } => {
                *op as u32 | rd(dst.number()) | rn(a.number()) | rm(b.number())
            }
            Inst::Fmla { acc, a, b } => {
                0x4E20_CC00 | rd(acc.number()) | rn(a.number()) | rm(b.number())
            }
            Inst::Unary { op, dst, src } => *op as u32 | rd(dst.number()) | rn(src.number()),
            Inst::Bsl {
                mask,
                if_true,
                if_false,
            } => 0x6E60_1C00 | rd(mask.number()) | rn(if_true.number()) | rm(if_false.number()),
            // `immh:immb` is `32 + amount` for `.4S`. At amount >= 32 that
            // carries into `immh`, making it 1xxx, which the ARM ARM decodes
            // as `.2D`: a 64-bit element shift that leaks bits across the
            // 32-bit lane boundary. Refuse loudly rather than emit a silently
            // different instruction.
            Inst::Shl { dst, src, amount } => {
                assert!(
                    *amount < 32,
                    "aarch64 SHL .4S: shift {amount} is out of range for a 32-bit lane — \
                     the immediate would encode .2D and cross lane boundaries"
                );
                0x4F00_5400
                    | rd(dst.number())
                    | rn(src.number())
                    | ((u32::from(*amount) + 32) << 16)
            }
            // `immh:immb` is `64 - amount`. Only counts in 1..=32 keep it
            // inside 01xx, so anything else silently encodes a DIFFERENT
            // element size and crosses lane boundaries.
            Inst::Ushr { dst, src, amount } => {
                assert!(
                    (1..=32).contains(amount),
                    "aarch64 USHR .4S: shift {amount} is outside 1..=32 — the immediate would \
                     encode a different element size, not a 32-bit lane shift"
                );
                0x6F20_0400
                    | rd(dst.number())
                    | rn(src.number())
                    | ((64 - u32::from(*amount)) << 16)
            }
            Inst::Mov { dst, src } => {
                Alu::Orr as u32 | rd(dst.number()) | rn(src.number()) | rm(src.number())
            }
            Inst::DupLane0 { dst, src } => 0x4E04_0400 | rd(dst.number()) | rn(src.number()),
            Inst::UmovW { dst, src, lane } => {
                0x0E00_3C00 | (imm5_lane(*lane) << 16) | rn(src.number()) | rd(dst.number())
            }
            Inst::InsFirst { dst, src } => {
                0x4E00_1C00 | (imm5_lane(0) << 16) | rn(src.number()) | rd(dst.number())
            }
            Inst::InsLane { v, lane, src } => {
                0x4E00_1C00 | (imm5_lane(*lane) << 16) | rn(src.number()) | rd(v.number())
            }
            Inst::MvnW { dst } => {
                let dst = dst.number();
                MVN_W | rn(abi::XZR) | rm(dst) | rd(dst)
            }
            Inst::Zero { dst } => 0x4F00_0400 | rd(dst.number()),
            Inst::FmovImm { dst, imm8 } => {
                let (abc, defgh) = (u32::from(*imm8 >> 5), u32::from(*imm8 & 0x1F));
                0x4F00_F400 | (abc << 16) | (defgh << 5) | rd(dst.number())
            }
            Inst::Madd { dst, a, b, addend } => {
                0x9B00_0000
                    | rm(b.number())
                    | ra(addend.number())
                    | rn(a.number())
                    | rd(dst.number())
            }
            Inst::AddLsl2 { dst, base, index } => {
                0x8B00_0000 | rm(index.number()) | (2 << 10) | rn(base.number()) | rd(dst.number())
            }
            // The lane is `Q:S`, the high bit in the instruction's `Q` and the
            // low bit in `S`.
            Inst::St1Lane { base, src, lane } => {
                assert!(
                    *lane < 4,
                    "ST1 {{Vt.S}}[{lane}]: a 128-bit register has four 32-bit lanes"
                );
                let (q, s) = (u32::from(lane >> 1), u32::from(lane & 1));
                0x0D9F_8000 | (q << 30) | (s << 12) | rn(base.number()) | rd(src.number())
            }
            Inst::FcvtzsX { dst, src } => 0x9E38_0000 | rn(src.number()) | rd(dst.number()),
            Inst::FmovToGp { dst, src } => 0x1E26_0000 | rn(src.number()) | rd(dst.number()),
            Inst::MovX { dst, src } => mov_x(dst.number(), src.number()),
            Inst::MovInt { dst, src } => mov_x(dst.number(), src.number()),
            Inst::AddImmHi { dst, src, imm } => {
                0x9140_0000 | (imm.bits() << 10) | rn(src.number()) | rd(dst.number())
            }
            Inst::AddSpHi { dst, imm } => {
                0x9140_0000 | (imm.bits() << 10) | rn(abi::SP) | rd(dst.number())
            }
            Inst::LdrQ { dst, addr } | Inst::LoadConst { dst, addr } => {
                transfer(0x3DC0_0000, rd(dst.number()), addr, Q_BYTES)
            }
            Inst::StrQ { src, addr } => transfer(0x3D80_0000, rd(src.number()), addr, Q_BYTES),
            Inst::SpillQ { src, slot } => frame(0x3D80_0000, rd(src.number()), slot, Q_BYTES),
            Inst::ReloadQ { dst, slot } => frame(0x3DC0_0000, rd(dst.number()), slot, Q_BYTES),
            Inst::SpillPtr { src, slot } => frame(0xF900_0000, rd(src.number()), slot, X_BYTES),
            Inst::ReloadPtr { dst, slot } => frame(0xF940_0000, rd(dst.number()), slot, X_BYTES),
            Inst::SpillInt { src, slot } => frame(0xF900_0000, rd(src.number()), slot, X_BYTES),
            Inst::ReloadInt { dst, slot } => frame(0xF940_0000, rd(dst.number()), slot, X_BYTES),
            Inst::LdrX { dst, addr } => transfer(0xF940_0000, rd(dst.number()), addr, X_BYTES),
            Inst::LdrS { dst, addr } => transfer(0xBD40_0000, rd(dst.number()), addr, S_BYTES),
            Inst::LdrW { dst, addr } => {
                0xB860_5800 | rm(addr.index.number()) | rn(addr.base.number()) | rd(dst.number())
            }
            Inst::LdrSIndexed { dst, addr } => {
                0xBC60_5800 | rm(addr.index.number()) | rn(addr.base.number()) | rd(dst.number())
            }
            Inst::B { .. } => B_OPCODE,
            Inst::CbzFar { test, .. } => {
                CBZ_W_OPCODE
                    | CB_OP_NONZERO
                    | (CBNZ_SKIPS_B << DispField::IMM19.shift)
                    | rd(test.number())
            }
            Inst::AdrpAdd { dst, .. } => 0x9000_0000 | rd(dst.number()),
            Inst::ProbePage => SUB_SP_PAGE,
            Inst::Enter { .. } | Inst::Ret { .. } | Inst::Fallthrough { .. } => {
                unreachable!("this instruction is a run of words, not a word")
            }
        }
    }

    /// The instruction's words, in order. A sequence is more than one, and a
    /// fall-through is none.
    fn words(&self, put: &mut impl FnMut(u32)) {
        match self {
            Inst::Fallthrough { .. } => {}
            Inst::CbzFar { .. } => {
                put(self.encode());
                put(B_OPCODE);
            }
            // `add dst, dst, #0`, whose immediate `patch_adrp_add` fills in.
            Inst::AdrpAdd { dst, .. } => {
                let dst = dst.number();
                put(self.encode());
                put(0x9100_0000 | rn(dst) | rd(dst));
            }
            Inst::ProbePage => {
                put(SUB_SP_PAGE);
                put(STR_XZR_SP);
            }
            Inst::Enter { size } => {
                let mut left = frame_bytes(*size);
                while left > STACK_PAGE {
                    Self::ProbePage.words(put);
                    left -= STACK_PAGE;
                }
                step_sp(SUB_SP, left, put);
            }
            Inst::Ret { size } => {
                step_sp(ADD_SP, frame_bytes(*size), put);
                put(RET | rn(abi::LR));
            }
            _ => put(self.encode()),
        }
    }

    /// Where the instruction's label field is, in bytes from its start, and
    /// how it is filled.
    fn label_field(&self) -> Option<(usize, Label, Patch)> {
        match self {
            Inst::B { to } => Some((0, *to, patch_imm26)),
            // The `CBNZ` is the first word; the `B` it steps over sits one
            // word later, and a displacement is measured from the branch's
            // own address.
            Inst::CbzFar { taken, .. } => Some((WORD_BYTES, *taken, patch_imm26)),
            Inst::AdrpAdd { to, .. } => Some((0, *to, patch_adrp_add)),
            _ => None,
        }
    }

    /// The instruction's bytes, and the label fields in them.
    fn assemble(&self, out: &mut Encoding<'_>) {
        self.words(&mut |word| out.bytes(&word.to_le_bytes()));
        if let Some((at, label, patch)) = self.label_field() {
            out.field(at, label, patch);
        }
        match self {
            Inst::CbzFar { next, .. } => out.falls_through(*next),
            Inst::Fallthrough { to } => out.falls_through(*to),
            _ => {}
        }
    }
}

/// `b`'s word with the displacement clear.
const B_OPCODE: u32 = 0x1400_0000;
/// `sub sp, sp, #bytes`'s word with the immediate clear.
const SUB_SP: u32 = 0xD100_0000 | rn(abi::SP) | rd(abi::SP);
/// `add sp, sp, #bytes`'s word with the immediate clear.
const ADD_SP: u32 = 0x9100_0000 | rn(abi::SP) | rd(abi::SP);
/// `ret`'s word with the register clear: it returns through the register in
/// `Rn`, which the caller's link register is.
const RET: u32 = 0xD65F_0000;
/// `mov xd, xm`, which is `orr xd, xzr, xm`: the word with the registers clear.
const ORR_X: u32 = 0xAA00_0000;
/// `mvn wd, wm`, which is `orn wd, wzr, wm`: the word with the registers clear.
const MVN_W: u32 = 0x2A20_0000;

/// `mov dst, src` between 64-bit registers.
const fn mov_x(dst: RegNo<GeneralFile>, src: RegNo<GeneralFile>) -> u32 {
    ORR_X | rn(abi::XZR) | rm(src) | rd(dst)
}

/// The `sh` bit of `add`/`sub` (immediate): the immediate is shifted left 12.
const SHIFT_12: u32 = 1 << 22;

/// `sp` moved by `bytes`: whole pages in the shifted immediate, then the rest,
/// so a frame is at most two words. That holds because `MAX_FRAME` keeps the pages
/// within the 12-bit immediate. `bytes` is a multiple of 16, as every frame is,
/// so `sp` stays aligned.
fn step_sp(opcode: u32, bytes: u32, put: &mut impl FnMut(u32)) {
    let (pages, rest) = (bytes / STACK_PAGE, bytes % STACK_PAGE);
    if pages > 0 {
        put(opcode | SHIFT_12 | (Imm12::new(pages).bits() << 10));
    }
    if rest > 0 {
        put(opcode | (Imm12::new(rest).bits() << 10));
    }
}

/// Fill a `B`'s `imm26` so it reaches `target`.
fn patch_imm26(code: &mut [u8], at: usize, target: usize) {
    DispField::IMM26.write(code, at, target);
}

/// Fill an `ADRP`+`ADD` pair, the `ADRP` at `at` and the `ADD` right after
/// it, so the pair materializes `target`. Both words already carry their
/// opcode and register fields, so this only ORs the immediates in.
fn patch_adrp_add(code: &mut [u8], at: usize, target: usize) {
    let page = |pos: usize| (pos as i64) & !0xFFF;
    let pages = (page(target) - page(at)) >> 12;
    assert!(
        (-(1i64 << 20)..(1i64 << 20)).contains(&pages),
        "ADRP page offset {pages} out of range (±4GiB)"
    );
    let imm = (pages as u32) & 0x1F_FFFF;
    let immlo = imm & 0x3;
    let immhi = (imm >> 2) & 0x7_FFFF;
    let adrp = u32::from_le_bytes(code[at..at + 4].try_into().unwrap());
    code[at..at + 4].copy_from_slice(&(adrp | (immlo << 29) | (immhi << 5)).to_le_bytes());

    let within_page = (target as u32) & 0xFFF;
    let add = u32::from_le_bytes(code[at + 4..at + 8].try_into().unwrap());
    code[at + 4..at + 8].copy_from_slice(&(add | (within_page << 10)).to_le_bytes());
}

// =============================================================================
// Constants
// =============================================================================

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
fn try_encode_fmov_imm8(val: f32) -> Option<u8> {
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

/// One constant pool entry: the four words of a 128-bit NEON register, lane
/// 0 first. A splat is the common one; the lattice's iota is the other.
type PoolEntry = [u32; 4];

/// Emit a constant pool entry — 16 bytes, lane 0 first.
fn emit_pool_entry(code: &mut Vec<u8>, entry: PoolEntry) {
    for word in entry {
        code.extend_from_slice(&word.to_le_bytes());
    }
}

/// The iota `[0, 1, 2, 3]`, as a pool entry.
const IOTA: PoolEntry = [
    0.0f32.to_bits(),
    1.0f32.to_bits(),
    2.0f32.to_bits(),
    3.0f32.to_bits(),
];

// =============================================================================
// Operation forms
// =============================================================================

/// The instruction a comparison is: `Lt` and `Le` are `Fcmgt` and `Fcmge`
/// with the operands swapped, and `Ne` is `Fcmeq` inverted.
struct CompareForm {
    op: Alu,
    swapped: bool,
    inverted: bool,
}

fn compare_form(op: CompareOp) -> CompareForm {
    let plain = |op| CompareForm {
        op,
        swapped: false,
        inverted: false,
    };
    match op {
        CompareOp::Gt => plain(Alu::Fcmgt),
        CompareOp::Ge => plain(Alu::Fcmge),
        CompareOp::Eq => plain(Alu::Fcmeq),
        CompareOp::Lt => CompareForm {
            swapped: true,
            ..plain(Alu::Fcmgt)
        },
        CompareOp::Le => CompareForm {
            swapped: true,
            ..plain(Alu::Fcmge)
        },
        CompareOp::Ne => CompareForm {
            inverted: true,
            ..plain(Alu::Fcmeq)
        },
    }
}

fn arithmetic_alu(op: BinaryOp) -> Alu {
    match op {
        BinaryOp::Add => Alu::Fadd,
        BinaryOp::Sub => Alu::Fsub,
        BinaryOp::Mul => Alu::Fmul,
        BinaryOp::Div => Alu::Fdiv,
        BinaryOp::Min => Alu::Fmin,
        BinaryOp::Max => Alu::Fmax,
        BinaryOp::IAdd => Alu::AddI32,
        BinaryOp::BitAnd => Alu::And,
        BinaryOp::BitOr => Alu::Orr,
    }
}

/// A unary op: one instruction, or for the reciprocals the estimate and its
/// Newton step.
fn unary(b: &mut Builder<Neon>, op: UnaryOp, src: Value<Vector>) -> Value<Vector> {
    let single = match op {
        UnaryOp::Neg => Lanewise::Fneg,
        UnaryOp::Abs => Lanewise::Fabs,
        UnaryOp::Sqrt => Lanewise::Fsqrt,
        UnaryOp::Floor => Lanewise::Frintm,
        UnaryOp::Ceil => Lanewise::Frintp,
        UnaryOp::Round => Lanewise::Frinta,
        UnaryOp::TruncToInt => Lanewise::Fcvtzs,
        UnaryOp::IntToFloat => Lanewise::Scvtf,
        UnaryOp::Rsqrt => {
            let estimate = lanewise(b, Lanewise::Frsqrte, src);
            let squared = alu(b, Alu::Fmul, estimate, estimate);
            let step = alu(b, Alu::Frsqrts, src, squared);
            return alu(b, Alu::Fmul, estimate, step);
        }
        UnaryOp::Recip => {
            let estimate = lanewise(b, Lanewise::Frecpe, src);
            let step = alu(b, Alu::Frecps, src, estimate);
            return alu(b, Alu::Fmul, estimate, step);
        }
    };
    lanewise(b, single, src)
}

// =============================================================================
// Selection
// =============================================================================

/// The NEON machine, as the selection pipeline sees it.
pub(in crate::emit) enum Neon {}

/// Define a value of class `C` with `make`, which builds the instruction that
/// writes it.
fn define<C: Class>(
    b: &mut Builder<Neon>,
    make: impl FnOnce(Def<C>) -> Inst<Selected>,
) -> Value<C> {
    let dst = b.def();
    let value = dst.value();
    b.push(make(dst));
    value
}

fn alu(b: &mut Builder<Neon>, op: Alu, a: Value<Vector>, c: Value<Vector>) -> Value<Vector> {
    define(b, |dst| Inst::Alu { op, dst, a, b: c })
}

fn lanewise(b: &mut Builder<Neon>, op: Lanewise, src: Value<Vector>) -> Value<Vector> {
    define(b, |dst| Inst::Unary { op, dst, src })
}

/// `entry` of the constant pool as a register: one load through the pool's
/// base.
fn pooled(b: &mut Builder<Neon>, entry: PoolEntry) -> Result<Value<Vector>, CompileError> {
    let base = b.anchors().expect("enter defines the pool's base");
    let at = b.constant(entry)?;
    let addr = Mem {
        base,
        offset: u32::try_from(at.offset).expect("POOL_REACH bounds a pool offset"),
    };
    Ok(define(b, |dst| Inst::LoadConst { dst, addr }))
}

/// `value` in every lane: zero, an `fmov` immediate, or a pool entry.
fn splat(b: &mut Builder<Neon>, value: f32) -> Result<Value<Vector>, CompileError> {
    match (value.to_bits(), try_encode_fmov_imm8(value)) {
        (0, _) => Ok(define(b, |dst| Inst::Zero { dst })),
        (_, Some(imm8)) => Ok(define(b, |dst| Inst::FmovImm { dst, imm8 })),
        (bits, None) => pooled(b, [bits; 4]),
    }
}

/// Element `index` of an array of `access`-byte elements at `base`, as an
/// address a transfer takes: the offset itself when the scaled immediate holds
/// it, otherwise the high part added first (`add t, base, #hi, lsl 12`) and
/// the low part the transfer's.
///
/// # Errors
///
/// [`CompileError::BudgetExceeded`] past the 16 MiB that reaches.
fn reach(
    b: &mut Builder<Neon>,
    base: Value<Pointer>,
    index: u64,
    access: u32,
) -> Result<Mem<Selected>, CompileError> {
    let bytes = index.saturating_mul(u64::from(access));
    if bytes <= u64::from(MAX_IMM12 * access) {
        return Ok(Mem {
            base,
            offset: bytes as u32,
        });
    }
    let hi = bytes >> 12;
    if hi > u64::from(MAX_IMM12) {
        return Err(CompileError::BudgetExceeded(
            "an address past the 16 MiB an add of a shifted imm12 reaches",
        ));
    }
    let base = define(b, |dst| Inst::AddImmHi {
        dst,
        src: base,
        imm: Imm12::new(hi as u32),
    });
    Ok(Mem {
        base,
        offset: (bytes & 0xFFF) as u32,
    })
}

/// A vector slot's address when `offset` is past what `str q` and `ldr q`
/// reach: `add t, sp, #hi, lsl 12`, with the low part the transfer's. A frame
/// is refused past `MAX_FRAME` (2 MiB) when a slot is leased, so `hi` is far
/// inside the `imm12` and there is nothing to refuse here.
fn deep(b: &mut Spiller<'_, Neon>, offset: u64) -> Mem<Selected> {
    let t = b.def::<Pointer>();
    let base = t.value();
    b.push(Inst::AddSpHi {
        dst: t,
        imm: Imm12::new((offset >> 12) as u32),
    });
    Mem {
        base,
        offset: (offset & 0xFFF) as u32,
    }
}

/// Whether a vector slot at `offset` is one a transfer reaches from `sp`.
fn reaches(offset: u64) -> bool {
    offset <= u64::from(MAX_IMM12 * Q_BYTES)
}

impl IsaBackend for Neon {
    type Inst<S: Stage> = Inst<S>;
    type Constant = PoolEntry;
    /// The pool's base, once `enter` has defined it.
    type Anchors = Option<Value<Pointer>>;
    type Lane = Value<Vector>;

    // AAPCS64 callee-saves the low 64 bits of v8-v15, and a kernel has no
    // prologue that preserves them. The general file is every caller-saved
    // register but the platform's x18: a leaf kernel makes no call, so no
    // veneer uses IP0 and IP1, and the allocator may.
    const FILE: RegisterFile = RegisterFile::new(
        Members {
            vector: &[
                0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                31,
            ],
            general: &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
            opmask: &[],
            flags: &[],
        },
        EntryRegisters {
            ctx: abi::X0,
            out: abi::X1,
            pitch: abi::X2,
        },
        16,
    );
    const POOL_REACH: u64 = 4096;

    fn lane(b: &mut Builder<Self>, op: LaneOp<Self>) -> Result<Value<Vector>, CompileError> {
        Ok(match op {
            LaneOp::Const(value) => splat(b, value)?,
            LaneOp::Lanes => pooled(b, IOTA)?,
            LaneOp::Unary(op, src) => unary(b, op, src),
            LaneOp::Binary(op, a, c) => alu(b, arithmetic_alu(op), a, c),
            LaneOp::Compare(op, a, c) => {
                let CompareForm {
                    op,
                    swapped,
                    inverted,
                } = compare_form(op);
                let (a, c) = match swapped {
                    true => (c, a),
                    false => (a, c),
                };
                let result = alu(b, op, a, c);
                match inverted {
                    true => lanewise(b, Lanewise::Not, result),
                    false => result,
                }
            }
            LaneOp::MulAdd(x, y, addend) => {
                let acc = b.tie(addend);
                let result = acc.write();
                b.push(Inst::Fmla { acc, a: x, b: y });
                result
            }
            // `cond ? if_true : if_false`, bit by bit, into the mask's register.
            LaneOp::Blend {
                cond,
                if_true,
                if_false,
            } => {
                let mask = b.tie(cond);
                let result = mask.write();
                b.push(Inst::Bsl {
                    mask,
                    if_true,
                    if_false,
                });
                result
            }
            // `ushr` cannot say 0, which is the identity.
            LaneOp::Shift(ShiftOp::Shr, src, 0) => src,
            LaneOp::Shift(ShiftOp::Shl, src, amount) => {
                define(b, |dst| Inst::Shl { dst, src, amount })
            }
            LaneOp::Shift(ShiftOp::Shr, src, amount) => {
                define(b, |dst| Inst::Ushr { dst, src, amount })
            }
            // `ldr s, [block, #4·element]`, then spread.
            LaneOp::Uniform { base, element } => {
                let addr = reach(b, base, element, S_BYTES)?;
                let loaded = define(b, |dst| Inst::LdrS { dst, addr });
                define(b, |dst| Inst::DupLane0 { dst, src: loaded })
            }
            // Four scalar loads: NEON has no gather. The indices are floats,
            // so they are truncated first, and each lane is extracted,
            // loaded and inserted.
            LaneOp::Gather { base, index } => {
                let indices = lanewise(b, Lanewise::Fcvtzs, index);
                let mut gathered = None;
                for lane in 0..4 {
                    let at = define(b, |dst| Inst::UmovW {
                        dst,
                        src: indices,
                        lane,
                    });
                    let addr = MemIndexed { base, index: at };
                    let word = define(b, |dst| Inst::LdrW { dst, addr });
                    gathered = Some(match gathered {
                        None => define(b, |dst| Inst::InsFirst { dst, src: word }),
                        Some(so_far) => {
                            let v = b.tie(so_far);
                            let result = v.write();
                            b.push(Inst::InsLane { v, lane, src: word });
                            result
                        }
                    });
                }
                gathered.expect("a gather has four lanes")
            }
            // One element, the same in every lane: the index is lane 0,
            // truncated.
            LaneOp::Broadcast { base, index } => {
                let at = define(b, |dst| Inst::FcvtzsX { dst, src: index });
                let addr = MemIndexed { base, index: at };
                let loaded = define(b, |dst| Inst::LdrSIndexed { dst, addr });
                define(b, |dst| Inst::DupLane0 { dst, src: loaded })
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
        let addr = reach(b, ctx, slot, X_BYTES)?;
        Ok(define(b, |dst| Inst::LdrX { dst, addr }))
    }

    /// `out + 4·(row·pitch + col)`: each index is lane 0 of its binder,
    /// truncated. A full batch is one `str q`; a remainder is `st1` per lane,
    /// the address moving past each.
    fn store(b: &mut Builder<Self>, store: Store<Self>) {
        let Store {
            out,
            pitch,
            row,
            col,
            value,
            lanes,
        } = store;
        let row = define(b, |dst| Inst::FcvtzsX { dst, src: row });
        let col = define(b, |dst| Inst::FcvtzsX { dst, src: col });
        let index = define(b, |dst| Inst::Madd {
            dst,
            a: row,
            b: pitch,
            addend: col,
        });
        let address = define(b, |dst| Inst::AddLsl2 {
            dst,
            base: out,
            index,
        });
        if u64::from(lanes) == Self::FILE.vector_bytes() / u64::from(S_BYTES) {
            b.push(Inst::StrQ {
                src: value,
                addr: Mem {
                    base: address,
                    offset: 0,
                },
            });
            return;
        }
        let mut next = address;
        for lane in 0..lanes as u8 {
            let base = b.tie(next);
            next = base.write();
            b.push(Inst::St1Lane {
                base,
                src: value,
                lane,
            });
        }
    }

    /// `umaxv` (a dead `True` arm: no lane is set) or `uminv` then `mvn` (a
    /// dead `False` arm: every lane is), `fmov` to a general register, and
    /// `cbz` on it.
    fn branch(b: &mut Builder<Self>, test: Test<Self>, edges: Edges) {
        let reduce = match test.dead {
            IfArm::True => Lanewise::Umaxv,
            IfArm::False => Lanewise::Uminv,
        };
        let reduced = lanewise(b, reduce, test.cond);
        let bits = define(b, |dst| Inst::FmovToGp { dst, src: reduced });
        let bits = match test.dead {
            IfArm::True => bits,
            IfArm::False => {
                let inverted = b.tie(bits);
                let result = inverted.write();
                b.push(Inst::MvnW { dst: inverted });
                result
            }
        };
        let next = Target {
            label: edges.next,
            args: Vec::new(),
        };
        b.push(Inst::CbzFar {
            test: bits,
            taken: edges.taken,
            next,
        });
    }

    fn jump(b: &mut Builder<Self>, to: Target, next: Label) {
        b.push(if to.label == next {
            Inst::Fallthrough { to }
        } else {
            Inst::B { to }
        });
    }

    /// The frame, and the pool's base: a rematerializable address, so it is
    /// placed where a constant is read and nowhere else.
    fn enter(b: &mut Builder<Self>) {
        b.push(Inst::Enter { size: FrameSize });
        let to = b.pool();
        let pool = define(b, |dst| Inst::AdrpAdd { dst, to });
        *b.anchors() = Some(pool);
    }

    fn ret(b: &mut Builder<Self>) {
        b.push(Inst::Ret { size: FrameSize });
    }

    fn copy(b: &mut Spiller<'_, Self>, src: ValueName) -> ValueName {
        match src.class {
            ClassId::Vector => b.define::<Vector>(|dst| Inst::Mov {
                dst,
                src: src.typed(),
            }),
            ClassId::Pointer => b.define::<Pointer>(|dst| Inst::MovX {
                dst,
                src: src.typed(),
            }),
            ClassId::Integer => b.define::<Integer>(|dst| Inst::MovInt {
                dst,
                src: src.typed(),
            }),
            ClassId::Opmask | ClassId::Flags => {
                unreachable!("NEON has no {:?} to copy", src.class)
            }
        }
    }

    fn spill(b: &mut Spiller<'_, Self>, src: ValueName, slot: &FrameSlot) {
        let name = slot.name();
        match src.class {
            ClassId::Vector if reaches(slot.offset()) => b.push(Inst::SpillQ {
                src: src.typed(),
                slot: name,
            }),
            ClassId::Vector => {
                let addr = deep(b, slot.offset());
                b.push(Inst::StrQ {
                    src: src.typed(),
                    addr,
                });
            }
            ClassId::Pointer => b.push(Inst::SpillPtr {
                src: src.typed(),
                slot: name,
            }),
            ClassId::Integer => b.push(Inst::SpillInt {
                src: src.typed(),
                slot: name,
            }),
            ClassId::Opmask | ClassId::Flags => {
                unreachable!("NEON has no {:?} to spill", src.class)
            }
        }
    }

    fn reload(b: &mut Spiller<'_, Self>, class: ClassId, slot: &FrameSlot) -> ValueName {
        let name = slot.name();
        match class {
            ClassId::Vector if reaches(slot.offset()) => {
                b.define::<Vector>(|dst| Inst::ReloadQ { dst, slot: name })
            }
            ClassId::Vector => {
                let addr = deep(b, slot.offset());
                b.define::<Vector>(|dst| Inst::LdrQ { dst, addr })
            }
            ClassId::Pointer => b.define::<Pointer>(|dst| Inst::ReloadPtr { dst, slot: name }),
            ClassId::Integer => b.define::<Integer>(|dst| Inst::ReloadInt { dst, slot: name }),
            ClassId::Opmask | ClassId::Flags => unreachable!("NEON has no {class:?} to reload"),
        }
    }

    fn rematerializable(inst: &Inst<Selected>) -> bool {
        matches!(
            inst,
            Inst::Zero { .. }
                | Inst::FmovImm { .. }
                | Inst::LoadConst { .. }
                | Inst::AdrpAdd { .. }
        )
    }

    fn reads_pool<S: Stage>(inst: &Inst<S>) -> bool {
        matches!(inst, Inst::LoadConst { .. })
    }

    fn walk<T: Stage>(inst: &Inst<Selected>, f: &mut impl Rebind<T>) -> Inst<T> {
        inst.walk(f)
    }

    fn encode(inst: &Inst<Bound<'_, Self>>, out: &mut Encoding<'_>) {
        inst.assemble(out);
    }

    fn constant_bytes(constant: PoolEntry) -> Vec<u8> {
        let mut bytes = Vec::new();
        emit_pool_entry(&mut bytes, constant);
        bytes
    }
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
}

// =============================================================================
// The registers the ABI fixes
// =============================================================================

/// AAPCS64's argument registers, the stack pointer, the link register and the
/// zero register, by hardware number. Every other register is the allocator's
/// choice.
mod abi {
    use crate::emit::GeneralFile;
    use crate::emit::regalloc::resource::RegNo;

    /// 1st argument: context pointer — array of buffer base pointers, then
    /// the uniform and origin blocks.
    pub(super) const X0: RegNo<GeneralFile> = RegNo::architectural(0);
    /// 2nd argument: the output plane.
    pub(super) const X1: RegNo<GeneralFile> = RegNo::architectural(1);
    /// 3rd argument: the pitch.
    pub(super) const X2: RegNo<GeneralFile> = RegNo::architectural(2);
    /// The link register: `ret` returns through it.
    pub(super) const LR: RegNo<GeneralFile> = RegNo::architectural(30);
    /// The stack pointer — spill slots are addressed from it.
    pub(super) const SP: RegNo<GeneralFile> = RegNo::architectural(31);
    /// The zero register, `xzr` (`wzr` at 32 bits): reads as zero, discards
    /// writes. It shares number 31 with `sp`; which one a field means is the
    /// instruction's.
    pub(super) const XZR: RegNo<GeneralFile> = RegNo::architectural(31);
}

// =============================================================================
// Branches as program items
// =============================================================================

/// Where a branch keeps its displacement, and how far it reaches.
///
/// A64 has no single `rel32`: `B` carries a 26-bit word displacement in the low
/// bits, and `B.cond`/`CBZ`/`CBNZ` carry a 19-bit one starting at bit 5. Two
/// fields, two ranges — so which one a branch uses is part of what the branch
/// *is*, and a branch whose reach must not depend on what it spans is a `B`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct DispField {
    /// Bit position of the field's low end.
    shift: u32,
    /// Field width in bits.
    bits: u32,
}

impl DispField {
    /// `B`'s imm26, at bit 0 — ±128 MiB.
    const IMM26: Self = Self { shift: 0, bits: 26 };
    /// `B.cond`'s and `CBZ`/`CBNZ`'s imm19, at bit 5 — ±1 MiB.
    const IMM19: Self = Self { shift: 5, bits: 19 };

    /// Overwrite this field of the instruction word at `at` so the branch
    /// reaches `target`, leaving every other bit — opcode, condition, register
    /// — exactly as placed.
    ///
    /// # Panics
    ///
    /// If the displacement is not a whole number of instructions, or does not
    /// fit. The first is a bug in this crate; the second is a real limit of the
    /// encoding. A guard does not meet it: [`Inst::CbzFar`] jumps with a `B`,
    /// ±128 MiB, and no emitter verb branches with a `B.cond`.
    fn write(self, code: &mut [u8], at: usize, target: usize) {
        let bytes = target as i64 - at as i64;
        assert!(
            bytes % 4 == 0,
            "aarch64 branch displacement is not a whole number of instructions"
        );
        let words = bytes / 4;
        let limit = 1i64 << (self.bits - 1);
        assert!(
            (-limit..limit).contains(&words),
            "branch displacement {words} does not fit {} bits",
            self.bits
        );
        let mask = ((1u32 << self.bits) - 1) << self.shift;
        let field = ((words as u32) << self.shift) & mask;
        let existing = u32::from_le_bytes([code[at], code[at + 1], code[at + 2], code[at + 3]]);
        code[at..at + 4].copy_from_slice(&((existing & !mask) | field).to_le_bytes());
    }
}

/// `sub sp, sp, #1, lsl 12`: `SUB (immediate)` with `sh = 1` and `imm12 = 1`,
/// 4096 bytes, `Rn = Rd = sp`.
const SUB_SP_PAGE: u32 = SUB_SP | SHIFT_12 | (1 << 10);

/// `str xzr, [sp]`: `STR Xt, [Xn]` with `Rt = xzr` and `Rn = sp`.
const STR_XZR_SP: u32 = 0xF900_0000 | rn(abi::SP) | rd(abi::XZR);

/// `CBZ Wt`'s opcode with the register and displacement fields clear:
/// `sf = 0` (a W register) and `011010`, then `op` in bit 24.
const CBZ_W_OPCODE: u32 = 0x3400_0000;
/// `op`, bit 24 of `CBZ`/`CBNZ`: clear branches on zero, set on non-zero. The
/// whole difference between the two instructions.
const CB_OP_NONZERO: u32 = 1 << 24;

/// How far the `CBNZ` of a [`Inst::CbzFar`] jumps, in words: over the `B`
/// that follows it, to the instruction after the pair.
const CBNZ_SKIPS_B: u32 = 2;

#[cfg(test)]
mod disp_field_tests {
    use super::*;

    #[test]
    #[should_panic(expected = "does not fit 19 bits")]
    fn a_conditional_refuses_what_it_cannot_reach() {
        let mut code = alloc::vec![0u8; 4];
        DispField::IMM19.write(&mut code, 0, 1 << 20);
    }

    /// A64 is fixed-width, so a displacement that is not a whole number of
    /// instructions means this crate laid something out unaligned.
    #[test]
    #[should_panic(expected = "not a whole number of instructions")]
    fn an_unaligned_displacement_is_a_bug() {
        let mut code = alloc::vec![0u8; 4];
        DispField::IMM26.write(&mut code, 0, 2);
    }
}
