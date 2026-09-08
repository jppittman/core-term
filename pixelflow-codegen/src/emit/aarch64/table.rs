//! Denotational AArch64 instruction shapes and instruction table.
//!
//! Instructions are categorized by their algebraic morphism:
//! - [`Binary<const OPCODE: u32>`]: $V \times V \to V$
//! - [`Unary<const OPCODE: u32>`]: $V \to V$
//! - [`Reduce<const OPCODE: u32>`]: $V \to S$
//!
//! Magic hex opcodes live strictly on the typed instruction definitions in this table.

use crate::emit::{AsmInsn, Gpr, Reg, SourceOperand, StoreTarget};
use alloc::vec::Vec;

// =============================================================================
// Algebraic Instruction Shapes
// =============================================================================

/// 12-bit unsigned immediate for AArch64 arithmetic instructions.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Imm12(pub u16);

/// 64-bit integer addition: `ADD Xd, Xn, Xm` or `ADD Xd, Xn, #imm12`
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct AddI64<O = Gpr> {
    pub dst: Gpr,
    pub src: Gpr,
    pub operand: O,
}

impl<O> AddI64<O> {
    #[must_use]
    #[inline]
    pub const fn new(dst: Gpr, src: Gpr, operand: O) -> Self {
        Self { dst, src, operand }
    }
}

impl AsmInsn for AddI64<Gpr> {
    #[inline]
    fn emit_into(self, code: &mut Vec<u8>) {
        let w = 0x8B00_0000
            | ((self.operand.0 as u32 & 0x1F) << 16)
            | ((self.src.0 as u32 & 0x1F) << 5)
            | (self.dst.0 as u32 & 0x1F);
        code.extend_from_slice(&w.to_le_bytes());
    }
}

impl AsmInsn for AddI64<Imm12> {
    #[inline]
    fn emit_into(self, code: &mut Vec<u8>) {
        let w = 0x9100_0000
            | ((self.operand.0 as u32 & 0xFFF) << 10)
            | ((self.src.0 as u32 & 0x1F) << 5)
            | (self.dst.0 as u32 & 0x1F);
        code.extend_from_slice(&w.to_le_bytes());
    }
}

/// Binary vector operation: V × V → V
///
/// Denotes `dst = lhs ⊗ rhs` across all vector lanes.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Binary<const OPCODE: u32, D = Reg, L = Reg, R = Reg> {
    pub dst: D,
    pub lhs: L,
    pub rhs: R,
}

impl<const OPCODE: u32> Binary<OPCODE, Reg, Reg, Reg> {
    #[must_use]
    #[inline]
    pub const fn new(dst: Reg, lhs: Reg, rhs: Reg) -> Self {
        Self { dst, lhs, rhs }
    }

    #[must_use]
    #[inline]
    pub fn encode(self) -> u32 {
        OPCODE
            | (self.dst.0 as u32 & 0x1F)
            | ((self.lhs.0 as u32 & 0x1F) << 5)
            | ((self.rhs.0 as u32 & 0x1F) << 16)
    }

    /// Attempt to construct a concrete register binary op from abstract storage operands.
    #[must_use]
    #[inline]
    pub fn from_operands<D: StoreTarget, L: SourceOperand, R: SourceOperand>(
        dst: D,
        lhs: L,
        rhs: R,
    ) -> Option<Self> {
        let d = dst.target_reg()?;
        let l = lhs.source_reg()?;
        let r = rhs.source_reg()?;
        Some(Self::new(d, l, r))
    }
}

impl<const OPCODE: u32> AsmInsn for Binary<OPCODE, Reg, Reg, Reg> {
    #[inline]
    fn emit_into(self, code: &mut Vec<u8>) {
        code.extend_from_slice(&self.encode().to_le_bytes());
    }
}

/// Unary vector operation: V → V
///
/// Denotes `dst = f(src)` across all vector lanes.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Unary<const OPCODE: u32, D = Reg, S = Reg> {
    pub dst: D,
    pub src: S,
}

impl<const OPCODE: u32> Unary<OPCODE, Reg, Reg> {
    #[must_use]
    #[inline]
    pub const fn new(dst: Reg, src: Reg) -> Self {
        Self { dst, src }
    }

    #[must_use]
    #[inline]
    pub fn encode(self) -> u32 {
        OPCODE | (self.dst.0 as u32 & 0x1F) | ((self.src.0 as u32 & 0x1F) << 5)
    }

    /// Attempt to construct a concrete register unary op from abstract storage operands.
    #[must_use]
    #[inline]
    pub fn from_operands<D: StoreTarget, S: SourceOperand>(dst: D, src: S) -> Option<Self> {
        let d = dst.target_reg()?;
        let s = src.source_reg()?;
        Some(Self::new(d, s))
    }
}

impl<const OPCODE: u32> AsmInsn for Unary<OPCODE, Reg, Reg> {
    #[inline]
    fn emit_into(self, code: &mut Vec<u8>) {
        code.extend_from_slice(&self.encode().to_le_bytes());
    }
}

/// Vector-to-scalar horizontal reduction: V → S
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Reduce<const OPCODE: u32, D = Reg, S = Reg> {
    pub dst: D,
    pub src: S,
}

impl<const OPCODE: u32> Reduce<OPCODE, Reg, Reg> {
    #[must_use]
    #[inline]
    pub const fn new(dst: Reg, src: Reg) -> Self {
        Self { dst, src }
    }

    #[must_use]
    #[inline]
    pub fn encode(self) -> u32 {
        OPCODE | (self.dst.0 as u32 & 0x1F) | ((self.src.0 as u32 & 0x1F) << 5)
    }

    /// Attempt to construct a concrete register reduction op from abstract storage operands.
    #[must_use]
    #[inline]
    pub fn from_operands<D: StoreTarget, S: SourceOperand>(dst: D, src: S) -> Option<Self> {
        let d = dst.target_reg()?;
        let s = src.source_reg()?;
        Some(Self::new(d, s))
    }
}

impl<const OPCODE: u32> AsmInsn for Reduce<OPCODE, Reg, Reg> {
    #[inline]
    fn emit_into(self, code: &mut Vec<u8>) {
        code.extend_from_slice(&self.encode().to_le_bytes());
    }
}

/// Bitwise select: `mask = (mask & if_true) | (~mask & if_false)`
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Bsl<M = Reg, T = Reg, F = Reg> {
    pub mask: M,
    pub if_true: T,
    pub if_false: F,
}

impl Bsl<Reg, Reg, Reg> {
    #[must_use]
    #[inline]
    pub const fn new(mask: Reg, if_true: Reg, if_false: Reg) -> Self {
        Self {
            mask,
            if_true,
            if_false,
        }
    }

    #[must_use]
    #[inline]
    pub fn encode(self) -> u32 {
        0x6E60_1C00
            | (self.mask.0 as u32 & 0x1F)
            | ((self.if_true.0 as u32 & 0x1F) << 5)
            | ((self.if_false.0 as u32 & 0x1F) << 16)
    }

    /// Attempt to construct a concrete register Bsl from abstract storage operands.
    #[must_use]
    #[inline]
    pub fn from_operands<M: SourceOperand, T: SourceOperand, F: SourceOperand>(
        mask: M,
        if_true: T,
        if_false: F,
    ) -> Option<Self> {
        let m = mask.source_reg()?;
        let t = if_true.source_reg()?;
        let f = if_false.source_reg()?;
        Some(Self::new(m, t, f))
    }
}

impl AsmInsn for Bsl<Reg, Reg, Reg> {
    #[inline]
    fn emit_into(self, code: &mut Vec<u8>) {
        code.extend_from_slice(&self.encode().to_le_bytes());
    }
}


/// Broadcast a lane of a vector register across all lanes: `DUP Vd.4S, Vn.s[0]`
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct DupLane0 {
    pub dst: Reg,
    pub src: Reg,
}

impl DupLane0 {
    #[must_use]
    #[inline]
    pub const fn new(dst: Reg, src: Reg) -> Self {
        Self { dst, src }
    }

    #[must_use]
    #[inline]
    pub fn encode(self) -> u32 {
        0x4E04_0400 | ((self.src.0 as u32) << 5) | (self.dst.0 as u32)
    }
}

impl AsmInsn for DupLane0 {
    #[inline]
    fn emit_into(self, code: &mut Vec<u8>) {
        code.extend_from_slice(&self.encode().to_le_bytes());
    }
}

/// Move vector lane 0 to general purpose register: `FMOV X16, D<src>`
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct FmovToGp {
    pub src: Reg,
}

impl FmovToGp {
    #[must_use]
    #[inline]
    pub const fn new(src: Reg) -> Self {
        Self { src }
    }

    #[must_use]
    #[inline]
    pub fn encode(self) -> u32 {
        0x1E26_0000 | ((self.src.0 as u32) << 5) | 16
    }
}

impl AsmInsn for FmovToGp {
    #[inline]
    fn emit_into(self, code: &mut Vec<u8>) {
        code.extend_from_slice(&self.encode().to_le_bytes());
    }
}

/// Return from subroutine: `RET`
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Ret;

impl Ret {
    pub const OPCODE: u32 = 0xD65F_03C0;

    #[must_use]
    #[inline]
    pub const fn encode(self) -> u32 {
        Self::OPCODE
    }
}

impl AsmInsn for Ret {
    #[inline]
    fn emit_into(self, code: &mut Vec<u8>) {
        code.extend_from_slice(&self.encode().to_le_bytes());
    }
}

// =============================================================================
// Instruction Table (Type Aliases with Local Opcodes)
// =============================================================================

// Binary arithmetic (V × V → V)
pub type Fadd = Binary<0x4E20_D400>;
pub type Fsub = Binary<0x4EA0_D400>;
pub type Fmul = Binary<0x6E20_DC00>;
pub type Fdiv = Binary<0x6E20_FC00>;
pub type Fmla = Binary<0x4E20_CC00>;
pub type Fmin = Binary<0x4EA0_F400>;
pub type Fmax = Binary<0x4E20_F400>;

// Unary arithmetic (V → V)
pub type Fsqrt = Unary<0x6EA1_F800>;
pub type Fabs = Unary<0x4EA0_F800>;
pub type Fneg = Unary<0x6EA0_F800>;
pub type Not = Unary<0x2E20_5800>;

// Rounding
pub type Frintm = Unary<0x4E21_9800>; // floor
pub type Frintp = Unary<0x4EA1_8800>; // ceil
pub type Frinta = Unary<0x6E21_8800>; // round

// Reciprocal estimate / steps
pub type Frsqrte = Unary<0x6EA1_D800>;
pub type Frsqrts = Binary<0x4EA0_FC00>;
pub type Frecpe = Unary<0x4EA1_D800>;
pub type Frecps = Binary<0x4E20_FC00>;

// Vector comparisons (result is bit mask)
pub type Fcmgt = Binary<0x6EA0_E400>;
pub type Fcmge = Binary<0x6E20_E400>;
pub type Fcmeq = Binary<0x4E20_E400>;

// Integer vector operations
pub type AddI32 = Binary<0x4EA0_8400>;
pub type And = Binary<0x4E20_1C00>;
pub type Orr = Binary<0x4EA0_1C00>;

// Conversions
pub type Fcvtzs = Unary<0x4EA1_B800>; // float -> signed int32
pub type Scvtf = Unary<0x4E21_D800>;  // signed int32 -> float

// Reductions (V → S)
pub type Uminv = Reduce<0x6EB1_A800>;
pub type Umaxv = Reduce<0x6E30_A800>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emit::Loc;

    #[test]
    fn binary_and_unary_from_abstract_operands() {
        let dst = Loc::Reg(Reg(0));
        let lhs = Loc::Reg(Reg(1));
        let rhs = Loc::Reg(Reg(2));

        let add = Fadd::from_operands(dst, lhs, rhs).expect("all in registers");
        assert_eq!(add.encode(), Fadd::new(Reg(0), Reg(1), Reg(2)).encode());

        let sqrt = Fsqrt::from_operands(dst, lhs).expect("all in registers");
        assert_eq!(sqrt.encode(), Fsqrt::new(Reg(0), Reg(1)).encode());
    }

    #[test]
    fn add_i64_encodes_register_and_immediate() {
        let mut code_reg = Vec::new();
        AddI64::new(Gpr(0), Gpr(1), Gpr(2)).emit_into(&mut code_reg);
        assert_eq!(
            code_reg,
            (0x8B00_0000u32 | (2 << 16) | (1 << 5)).to_le_bytes()
        );

        let mut code_imm = Vec::new();
        AddI64::new(Gpr(0), Gpr(1), Imm12(42)).emit_into(&mut code_imm);
        assert_eq!(
            code_imm,
            (0x9100_0000u32 | (42 << 10) | (1 << 5)).to_le_bytes()
        );
    }
}

