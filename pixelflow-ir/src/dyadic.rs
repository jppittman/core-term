//! Exact dyadic rationals: `mantissa × 2^exp` with an integer mantissa.
//!
//! Every finite `f32` IS a dyadic rational, so this type represents them all
//! exactly — and, unlike any fixed-width float, `+ − ×` over dyadics never
//! round: the mantissa grows instead.
//!
//! # Why exactness, rather than more precision
//!
//! The e-graph's equality relation must hold only ℝ-true facts. Constant
//! folding in `f32` computes f32-truths (`x + 2²⁴ = 2²⁴`) while the algebraic
//! rewrites compute ℝ-truths (`(x+y) − y = x`); each is sound alone, but at an
//! ill-conditioned input their conjunction puts two UNEQUAL constants in one
//! e-class, and congruence closure amplifies that into everything-equals-
//! everything. A wider float (f64, f128) moves the trigger further out but
//! keeps it reachable, so the refusal valve stays load-bearing forever. Exact
//! arithmetic makes the contradiction *unconstructible*.
//!
//! # The cap, and why declining is the point
//!
//! The mantissa is an `i128`, so precision is finite in practice. When an
//! operation would exceed it, the operation returns `None` — it does NOT
//! round. That distinction is the whole design: a float silently rounds at
//! its boundary (and can therefore contradict the algebra), while this
//! declines and leaves the value symbolic, which is sound. Loud abstention
//! instead of a quiet wrong answer.
//!
//! # Signed zero
//!
//! Dyadics have one zero; `from_f32(-0.0)` and `from_f32(0.0)` both give
//! [`Dyadic::ZERO`], and `to_f32` returns `+0.0`. Callers must therefore not
//! fold through this domain when a result *depends* on zero's sign. In
//! practice that is already handled: `Recip`/`Rsqrt` are refused by
//! `OpKind::fold_is_platform_specific`, and `try_div` refuses a zero divisor
//! (`±inf` is not a dyadic rational, so there is nothing to return).

/// A dyadic rational `mantissa × 2^exp`, normalized.
///
/// The normal form is: `mantissa` odd, or the value is exactly
/// [`Dyadic::ZERO`] (`mantissa == 0 && exp == 0`). Normalization is what makes
/// `Eq`/`Ord`/`Hash` agree with mathematical equality, so every constructor
/// and operation must return a normalized value — two dyadics denoting the
/// same number must have identical fields.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Dyadic {
    /// Signed significand. Odd in normal form (or zero).
    mantissa: i128,
    /// Power-of-two scale.
    exp: i32,
}

impl Dyadic {
    /// The additive identity, and the sole representation of zero.
    pub const ZERO: Self = Self {
        mantissa: 0,
        exp: 0,
    };

    /// Exact conversion from `f32`.
    ///
    /// `None` for NaN and ±∞, which are not dyadic rationals — callers route
    /// those through the bit domain instead. `±0.0` both map to
    /// [`Dyadic::ZERO`] (see the module docs on signed zero).
    ///
    /// Must handle subnormals: they carry no implicit leading 1, so decoding
    /// them as if they did is wrong by a factor of two and misplaces the
    /// exponent.
    #[must_use]
    pub fn from_f32(_value: f32) -> Option<Self> {
        todo!("A: exact IEEE-754 binary32 decode, including subnormals")
    }

    /// Round to the nearest `f32`, ties to even — exactly ONE rounding.
    ///
    /// This is the delicate one. Correct behavior requires a round bit AND a
    /// sticky bit (is anything nonzero below the round bit?), carry
    /// propagation when rounding overflows the significand into the next
    /// binade, the subnormal boundary (fewer than 24 bits of significand
    /// available, so the rounding position moves), and overflow to ±∞.
    #[must_use]
    pub fn to_f32(self) -> f32 {
        todo!("A: correctly-rounded binary32 encode (RNE, subnormals, overflow)")
    }

    /// Exact sum, or `None` if it would exceed the mantissa's width.
    ///
    /// Aligning exponents shifts one mantissa left, which can overflow. The
    /// check must happen BEFORE the shift: `checked_shl` only reports a
    /// shift amount ≥ bit width, never bits shifted off the top, so it does
    /// not detect this. Compare the shift against `leading_zeros()` instead.
    #[must_use]
    pub fn add(self, _rhs: Self) -> Option<Self> {
        todo!("A: align exponents (checked), integer add, normalize")
    }

    /// Exact difference, or `None` past the mantissa's width. See [`Self::add`].
    #[must_use]
    pub fn sub(self, _rhs: Self) -> Option<Self> {
        todo!("A")
    }

    /// Exact product, or `None` past the mantissa's width.
    #[must_use]
    pub fn mul(self, _rhs: Self) -> Option<Self> {
        todo!("A: checked mantissa product, exponents add, normalize")
    }

    /// Quotient, but ONLY when it is exactly representable as a dyadic.
    ///
    /// Division does not close over dyadic rationals (`1/3`), so this returns
    /// `Some` only when the result multiplies back exactly. `None` for a zero
    /// divisor.
    #[must_use]
    pub fn try_div(self, _rhs: Self) -> Option<Self> {
        todo!("A: exact-only division")
    }

    /// Square root, but ONLY when exact.
    ///
    /// Like division, `sqrt` does not close (`√2`). Exact requires both an
    /// even exponent (after normalization adjustment) and a perfect-square
    /// mantissa. `None` for a negative value.
    #[must_use]
    pub fn try_sqrt(self) -> Option<Self> {
        todo!("A: exact-only integer sqrt")
    }

    /// Negation — always exact, never widens.
    #[must_use]
    pub fn neg(self) -> Self {
        todo!("A")
    }

    /// Absolute value — always exact, never widens.
    #[must_use]
    pub fn abs(self) -> Self {
        todo!("A")
    }

    /// Whether this is [`Dyadic::ZERO`].
    #[must_use]
    pub fn is_zero(self) -> bool {
        todo!("A")
    }
}

impl PartialOrd for Dyadic {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Dyadic {
    /// Total order by mathematical value.
    ///
    /// NOT lexicographic on the fields: `1 × 2^1` and `1 × 2^0` compare by
    /// value, not by mantissa first. Comparing requires aligning exponents,
    /// which can exceed the width — fall back to comparing signs and
    /// magnitude-by-exponent rather than returning a wrong answer.
    fn cmp(&self, _other: &Self) -> core::cmp::Ordering {
        todo!("A: value ordering, overflow-safe")
    }
}
