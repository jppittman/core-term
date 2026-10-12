//! The AArch64 operation vocabulary: the enums whose discriminants are the
//! opcodes, the memory operands and the sizes their immediates scale by.
//!
//! The instructions themselves are `Inst`'s arms, grouped by their algebraic
//! morphism:
//! - [`Alu`]: $V \times V \to V$
//! - [`Lanewise`]: $V \to V$, and the horizontal reductions $V \to S$
//!
//! Magic hex opcodes live strictly on the enums below, and on the arms of
//! `Inst::encode` whose words carry no operation to choose.

use crate::emit::{Integer, Pointer, Rebind, Selected, Stage};

// =============================================================================
// Algebraic Instruction Shapes
// =============================================================================

/// 12-bit unsigned immediate for AArch64 arithmetic instructions: a value the
/// field holds, so `Inst::encode` needs no mask.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(in crate::emit) struct Imm12(u16);

impl Imm12 {
    /// # Panics
    ///
    /// If `value` is past [`MAX_IMM12`]: the field would wrap into a
    /// different instruction.
    pub(super) fn new(value: u32) -> Self {
        assert!(
            value <= MAX_IMM12,
            "a 12-bit immediate holds at most {MAX_IMM12}, not {value}"
        );
        Self(value as u16)
    }

    pub(super) fn bits(self) -> u32 {
        u32::from(self.0)
    }
}

/// Two-source vector operations: `dst = a ⊗ b` across all lanes. The
/// discriminant is the opcode with its register fields clear.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u32)]
pub(in crate::emit) enum Alu {
    Fadd = 0x4E20_D400,
    Fsub = 0x4EA0_D400,
    Fmul = 0x6E20_DC00,
    Fdiv = 0x6E20_FC00,
    Fmin = 0x4EA0_F400,
    Fmax = 0x4E20_F400,
    Frsqrts = 0x4EA0_FC00,
    Frecps = 0x4E20_FC00,
    /// A comparison's result is a bit mask: all-ones where it holds.
    Fcmgt = 0x6EA0_E400,
    Fcmge = 0x6E20_E400,
    Fcmeq = 0x4E20_E400,
    AddI32 = 0x4EA0_8400,
    And = 0x4E20_1C00,
    Orr = 0x4EA0_1C00,
}

/// One-source vector operations: `dst = f(src)` across all lanes. The
/// discriminant is the opcode with its register fields clear. `Uminv` and
/// `Umaxv` are the horizontal reductions: the scalar lands in lane 0.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u32)]
pub(in crate::emit) enum Lanewise {
    Fsqrt = 0x6EA1_F800,
    Fabs = 0x4EA0_F800,
    Fneg = 0x6EA0_F800,
    Not = 0x2E20_5800,
    /// Floor.
    Frintm = 0x4E21_9800,
    /// Ceil.
    Frintp = 0x4EA1_8800,
    /// Round, ties away.
    Frinta = 0x6E21_8800,
    Frsqrte = 0x6EA1_D800,
    Frecpe = 0x4EA1_D800,
    /// Float to signed int32.
    Fcvtzs = 0x4EA1_B800,
    /// Signed int32 to float.
    Scvtf = 0x4E21_D800,
    Uminv = 0x6EB1_A800,
    Umaxv = 0x6E30_A800,
}

/// Bytes moved by a `q` (128-bit vector) access — also the scale of its offset.
pub(super) const Q_BYTES: u32 = 16;
/// Bytes moved by an `x` (64-bit general/pointer) access.
pub(super) const X_BYTES: u32 = 8;
/// Bytes moved by an `s` (32-bit scalar SIMD&FP) access.
pub(super) const S_BYTES: u32 = 4;
/// The largest value a 12-bit scaled immediate holds.
pub(super) const MAX_IMM12: u32 = 4095;

/// An address spelled `[base, #offset]` — the scaled-immediate addressing mode.
///
/// `offset` is in BYTES. aarch64 encodes it divided by the access size.
/// The base being a pointer-class operand guarantees an integer counter or
/// index cannot be mistakenly passed as an address.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(in crate::emit) struct Mem<S: Stage> {
    /// The register holding the base address.
    pub(super) base: S::Read<Pointer>,
    /// Displacement in bytes; must be a multiple of the access size.
    pub(super) offset: u32,
}

/// An address spelled `[base, w<index>, uxtw #2]` — a 32-bit index register,
/// zero-extended to 64 bits and scaled by 4.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(in crate::emit) struct MemIndexed<S: Stage> {
    /// The register holding the buffer base pointer.
    pub(super) base: S::Read<Pointer>,
    /// The element index, read as the 32-bit `w<index>`.
    pub(super) index: S::Read<Integer>,
}

/// `offset` in units of the access it addresses: what `imm12` holds.
///
/// # Panics
///
/// If the offset is not a multiple of the access size.
pub(super) fn scaled(offset: u32, access: u32) -> u32 {
    assert!(
        offset.is_multiple_of(access),
        "{}-bit access offset {offset} is not {access}-byte aligned",
        access * 8
    );
    offset / access
}

impl<S: Stage> Mem<S> {
    pub(super) fn scaled(&self, access: u32) -> u32 {
        scaled(self.offset, access)
    }
}

impl Mem<Selected> {
    /// Rebuild at stage `T`.
    pub(super) fn walk<T: Stage>(&self, f: &mut impl Rebind<T>) -> Mem<T> {
        Mem {
            base: f.read(self.base),
            offset: self.offset,
        }
    }
}

impl MemIndexed<Selected> {
    /// Rebuild at stage `T`.
    pub(super) fn walk<T: Stage>(&self, f: &mut impl Rebind<T>) -> MemIndexed<T> {
        MemIndexed {
            base: f.read(self.base),
            index: f.read(self.index),
        }
    }
}
