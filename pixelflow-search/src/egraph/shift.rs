//! A shift's static parameters: which way it moves the bits, and by how many.
//!
//! A shift is an operator with a parameter, not a binary operation: its count
//! is an integer in `0..32` that no rule may rewrite, no class holds, and
//! nothing repairs after extraction. The arena spells the count as a `Const`
//! operand (`Shl(x, Const)`), and that spelling is where the count can be
//! anything an `f32` can be. These types are what is left once the spelling
//! has been checked, so a value of either is already a valid shift.

use pixelflow_ir::{Children, Ir, OpKind, Shape};

/// Which way a shift moves the bits of a word.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ShiftDirection {
    /// Toward the high bits: [`OpKind::Shl`].
    Left,
    /// Toward the low bits: [`OpKind::Shr`].
    Right,
}

impl ShiftDirection {
    /// The direction `op` shifts in, if it is a shift.
    #[must_use]
    pub fn of(op: OpKind) -> Option<Self> {
        match op {
            OpKind::Shl => Some(Self::Left),
            OpKind::Shr => Some(Self::Right),
            _ => None,
        }
    }

    /// The operation this direction names.
    #[must_use]
    pub fn op(self) -> OpKind {
        match self {
            Self::Left => OpKind::Shl,
            Self::Right => OpKind::Shr,
        }
    }
}

/// How far a shift moves the bits of a word: an integer in
/// `0..`[`ShiftCount::WORD_BITS`].
///
/// A count outside that range has three answers on the three targets (see
/// `OpKind::fold_is_platform_specific`), so it is not a count and has no
/// value of this type. The one constructor takes the arena's `f32` literal
/// and refuses everything a word cannot be shifted by.
///
/// The field is a `u8` because the range is the word's width, which the
/// hardware fixes, and no program can grow it: a wider field would let no
/// program express a count more.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ShiftCount(u8);

impl ShiftCount {
    /// The width in bits of the 32-bit word a shift moves bits within, which
    /// bounds its count.
    pub const WORD_BITS: u32 = 32;

    /// The count a literal names, or `None` if it is not an integer in
    /// `0..`[`WORD_BITS`](Self::WORD_BITS).
    ///
    /// The literal is the count's one spelling: `-0.0` compares equal to
    /// `0.0` and casts to `0` but is not what [`Self::literal`] returns, so it
    /// is refused, and a count reads back as exactly the bits it came from.
    #[must_use]
    pub fn from_literal(literal: f32) -> Option<Self> {
        if !(literal >= 0.0 && literal < Self::WORD_BITS as f32) {
            return None;
        }
        // In range, so the casts are exact and total.
        let count = Self(literal as u8);
        (count.literal().to_bits() == literal.to_bits()).then_some(count)
    }

    /// The count as the literal the arena spells it with.
    #[must_use]
    pub fn literal(self) -> f32 {
        f32::from(self.0)
    }

    /// The count in bits.
    #[must_use]
    pub fn bits(self) -> u32 {
        u32::from(self.0)
    }
}

/// The operand and count of a shift as the arena spells it, `Shl(x, Const)`:
/// the operand's reference, and the count the second child's literal names.
/// `None` if that child is not a `Const` holding a valid count.
///
/// The one reading of the arena's spelling, for everything that carries a
/// term into a graph: the count is read here and never visited, so no class
/// is made for it.
pub(crate) fn spelled<I: Ir>(
    term: &I,
    children: Children<'_, I::Ref>,
) -> Option<(I::Ref, ShiftCount)> {
    let (operand, count) = (children.get(0)?, children.get(1)?);
    let Shape::Const(literal) = term.project(count) else {
        return None;
    };
    Some((operand, ShiftCount::from_literal(literal)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every integer in range is a count and reads back as the literal it
    /// came from; the neighbours on both sides, and every value whose cast
    /// would alias a legal count, are not.
    #[test]
    fn a_count_is_exactly_an_integer_below_the_word_width() {
        for bits in 0..ShiftCount::WORD_BITS {
            let count = ShiftCount::from_literal(bits as f32).expect("in range");
            assert_eq!(count.bits(), bits);
            assert_eq!(count.literal(), bits as f32);
        }
        for refused in [
            -1.0,
            -0.5,
            0.5,
            2.5,
            31.5,
            32.0,
            33.0,
            256.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
        ] {
            assert_eq!(ShiftCount::from_literal(refused), None, "{refused}");
        }
    }

    /// The literal is the count's one spelling: what `from_literal` accepts,
    /// `literal` returns bit for bit. `-0.0` compares equal to `0.0` and
    /// casts to `0`, but it is not the literal `0` is spelled with, so it is
    /// not a count.
    #[test]
    fn a_count_reads_back_as_the_bits_it_was_spelled_with() {
        assert_eq!(ShiftCount::from_literal(-0.0), None);
        for bits in 0..ShiftCount::WORD_BITS {
            let literal = bits as f32;
            let count = ShiftCount::from_literal(literal).expect("in range");
            assert_eq!(count.literal().to_bits(), literal.to_bits());
        }
    }

    /// Only the two shifts have a direction, and each names its own op.
    #[test]
    fn a_direction_names_its_op_and_only_shifts_have_one() {
        for direction in [ShiftDirection::Left, ShiftDirection::Right] {
            assert_eq!(ShiftDirection::of(direction.op()), Some(direction));
        }
        assert_eq!(ShiftDirection::of(OpKind::Add), None);
        assert_eq!(ShiftDirection::of(OpKind::Const), None);
    }
}
