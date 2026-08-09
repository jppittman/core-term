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

use core::cmp::Ordering;

/// binary32 significand width, including the implicit leading bit.
const F32_SIGNIFICAND_BITS: i32 = 24;
/// Exponent of the least significant bit of the smallest subnormal, `2^-149`.
const F32_MIN_SUBNORMAL_EXP: i32 = -149;
/// `exp + F32_EXP_BIAS_TO_BIASED` converts a significand exponent into the
/// stored biased exponent: `+127` for the IEEE bias, `+23` because the stored
/// significand is scaled by `2^23`.
const F32_EXP_BIAS_TO_BIASED: i32 = 150;

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

    /// Strip trailing zero bits so the mantissa is odd — the normal form on
    /// which `Eq`/`Hash` are defined. Without this, `2 × 2^0` and `1 × 2^1`
    /// would be distinct keys for one number and hash-consing would split
    /// e-classes that denote the same value.
    /// `None` if the exponent would overflow `i32`. Reachable in O(log)
    /// public operations — repeatedly squaring `2.0` walks the exponent up
    /// exponentially — and an unchecked `exp + shift` would panic in debug or
    /// wrap to a radically different value in release. Declining is the same
    /// answer the mantissa cap gives.
    fn normalize(mantissa: i128, exp: i32) -> Option<Self> {
        if mantissa == 0 {
            return Some(Self::ZERO);
        }
        let shift = mantissa.trailing_zeros();
        Some(Self {
            mantissa: mantissa >> shift,
            exp: exp.checked_add(shift as i32)?,
        })
    }

    /// Number of significant bits in the magnitude (1 for `±1`).
    fn magnitude_bits(self) -> u32 {
        128 - self.mantissa.unsigned_abs().leading_zeros()
    }

    /// Exact conversion from `f32`.
    ///
    /// `None` for NaN and ±∞, which are not dyadic rationals — callers route
    /// those through the bit domain instead. `±0.0` both map to
    /// [`Dyadic::ZERO`] (see the module docs on signed zero).
    #[must_use]
    pub fn from_f32(value: f32) -> Option<Self> {
        if !value.is_finite() {
            return None;
        }
        let bits = value.to_bits();
        let negative = bits >> 31 == 1;
        let biased = ((bits >> 23) & 0xFF) as i32;
        let fraction = i128::from(bits & 0x007F_FFFF);

        // Subnormals carry NO implicit leading 1 and are all scaled by the
        // same 2^-149; decoding them as if they had the implicit bit is wrong
        // by a factor of two AND misplaces the exponent.
        let (magnitude, exp) = if biased == 0 {
            (fraction, F32_MIN_SUBNORMAL_EXP)
        } else {
            (fraction | (1 << 23), biased - F32_EXP_BIAS_TO_BIASED)
        };
        let signed = if negative { -magnitude } else { magnitude };
        Self::normalize(signed, exp)
    }

    /// Round to the nearest `f32`, ties to even — exactly ONE rounding.
    #[must_use]
    pub fn to_f32(self) -> f32 {
        if self.mantissa == 0 {
            return 0.0;
        }
        let negative = self.mantissa < 0;
        let magnitude = self.mantissa.unsigned_abs();
        // floor(log2(value)): a k-bit magnitude scaled by 2^exp lies in
        // [2^(k-1+exp), 2^(k+exp)).
        //
        // All exponent arithmetic here is i64. `normalize` legitimately
        // accepts exp == i32::MAX (it only rejects an overflowing ADD), so
        // every one of these steps — the binade, the target, the shift, and
        // the bias below — can leave i32 for a value the type itself allows.
        // Widening is cheaper than checking each one, and the results are
        // clamped into f32's range regardless.
        let exp_wide = i64::from(self.exp);
        let unbiased = i64::from(self.magnitude_bits()) - 1 + exp_wide;

        // A normal result keeps 24 significand bits; below 2^-126 the
        // significand loses bits instead of the exponent falling further, so
        // the rounding position pins at 2^-149. Taking the max is what moves
        // the rounding point into the subnormal range — the case hand-rolled
        // conversions usually miss.
        let target_exp = core::cmp::max(
            unbiased - i64::from(F32_SIGNIFICAND_BITS - 1),
            i64::from(F32_MIN_SUBNORMAL_EXP),
        );
        let shift = target_exp - exp_wide;

        let mut significand: u128 = if shift <= 0 {
            // Widening: exact. In the normal branch `shift == bits - 24`, so a
            // negative shift means fewer than 24 bits and cannot overflow.
            magnitude << (-shift) as u32
        } else if shift >= 128 {
            // Everything is below the round bit. The magnitude is < 2^128
            // ≤ 2·half, so it can only reach a tie at shift == 128 with a
            // magnitude above 2^127.
            u128::from(shift == 128 && magnitude > (1u128 << 127))
        } else {
            let drop = shift as u32;
            let quotient = magnitude >> drop;
            let dropped = magnitude & ((1u128 << drop) - 1);
            let half = 1u128 << (drop - 1);
            // Round-nearest, ties-to-even: up when past halfway, or exactly
            // halfway with an odd quotient.
            let round_up = dropped > half || (dropped == half && quotient & 1 == 1);
            quotient + u128::from(round_up)
        };

        // Rounding can carry the significand out of its binade (0xFF_FFFF + 1).
        let mut exp = target_exp;
        if significand >= 1 << F32_SIGNIFICAND_BITS {
            significand >>= 1;
            exp += 1;
        }

        let biased = exp + i64::from(F32_EXP_BIAS_TO_BIASED);
        let magnitude_bits: u32 = if biased >= 0xFF {
            0x7F80_0000 // ±∞
        } else if significand >= 1 << (F32_SIGNIFICAND_BITS - 1) {
            // Normal: the leading bit is implicit. A subnormal that rounded up
            // to 2^23 lands here with biased == 1 — exactly the smallest
            // normal, which is the correct answer.
            ((biased as u32) << 23) | (significand as u32 - (1 << 23))
        } else {
            // Subnormal: the significand IS the stored fraction.
            significand as u32
        };
        f32::from_bits(if negative {
            magnitude_bits | 0x8000_0000
        } else {
            magnitude_bits
        })
    }

    /// Exact sum, or `None` if it would exceed the mantissa's width.
    ///
    /// Named `checked_*` rather than implementing `core::ops::Add`: the
    /// fallibility is load-bearing (declining past the cap is what keeps this
    /// type from ever rounding), and an operator would have to return `Self`,
    /// forcing an unwrap that reintroduces exactly the silent wrongness the
    /// type exists to prevent. The prefix also states the fallibility at every
    /// call site instead of in the docs.
    #[must_use]
    pub fn checked_add(self, rhs: Self) -> Option<Self> {
        if self.mantissa == 0 {
            return Some(rhs);
        }
        if rhs.mantissa == 0 {
            return Some(self);
        }
        let (low, high) = if self.exp <= rhs.exp {
            (self, rhs)
        } else {
            (rhs, self)
        };
        let shift = high.exp.checked_sub(low.exp)?;

        // The check must precede the shift. `checked_shl` reports only a
        // shift amount ≥ bit width, NEVER bits shifted off the top, so it
        // cannot detect this — and a silently truncated mantissa is precisely
        // the rounding this type exists to refuse. `i128` has 127 magnitude
        // bits available beside the sign.
        if shift < 0 || high.magnitude_bits() as i32 + shift > 127 {
            return None;
        }
        let aligned = high.mantissa << shift as u32;
        let sum = low.mantissa.checked_add(aligned)?;
        Self::normalize(sum, low.exp)
    }

    /// Exact difference, or `None` past the mantissa's width. See [`Self::checked_add`].
    #[must_use]
    pub fn checked_sub(self, rhs: Self) -> Option<Self> {
        self.checked_add(rhs.negate())
    }

    /// Exact product, or `None` past the mantissa's width.
    #[must_use]
    pub fn checked_mul(self, rhs: Self) -> Option<Self> {
        if self.mantissa == 0 || rhs.mantissa == 0 {
            return Some(Self::ZERO);
        }
        let mantissa = self.mantissa.checked_mul(rhs.mantissa)?;
        let exp = self.exp.checked_add(rhs.exp)?;
        Self::normalize(mantissa, exp)
    }

    /// Quotient, but ONLY when it is exactly representable as a dyadic.
    ///
    /// Division does not close over dyadic rationals (`1/3`), so this returns
    /// `Some` only when the result is exact. Both mantissas are odd in normal
    /// form, and an odd divisor shares no factor with the `2^exp` scale, so
    /// exactness reduces to the divisor's mantissa dividing evenly. `None`
    /// for a zero divisor — `±∞` is not a dyadic rational.
    #[must_use]
    pub fn try_div(self, rhs: Self) -> Option<Self> {
        if rhs.mantissa == 0 {
            return None;
        }
        if self.mantissa == 0 {
            return Some(Self::ZERO);
        }
        if self.mantissa % rhs.mantissa != 0 {
            return None;
        }
        let exp = self.exp.checked_sub(rhs.exp)?;
        Self::normalize(self.mantissa / rhs.mantissa, exp)
    }

    /// Square root, but ONLY when exact.
    ///
    /// Like division, `sqrt` does not close (`√2`). Requires an even exponent
    /// and a perfect-square mantissa. An ODD exponent can never be exact: the
    /// only way to make it even is to borrow a factor of two into the
    /// mantissa, and a normalized mantissa is odd, so the result would carry
    /// exactly one factor of two — never a perfect square. `None` for a
    /// negative value.
    #[must_use]
    pub fn try_sqrt(self) -> Option<Self> {
        if self.mantissa < 0 {
            return None;
        }
        if self.mantissa == 0 {
            return Some(Self::ZERO);
        }
        if self.exp % 2 != 0 {
            return None;
        }
        let root = exact_isqrt(self.mantissa.unsigned_abs())?;
        Self::normalize(root as i128, self.exp / 2)
    }

    /// Negation — always exact, never widens.
    ///
    /// A normalized mantissa is odd or zero, and `i128::MIN` is even, so the
    /// negation can never overflow.
    #[must_use]
    pub fn negate(self) -> Self {
        Self {
            mantissa: -self.mantissa,
            exp: self.exp,
        }
    }

    /// Absolute value — always exact, never widens.
    #[must_use]
    pub fn abs(self) -> Self {
        Self {
            mantissa: self.mantissa.abs(),
            exp: self.exp,
        }
    }

    /// Whether this is [`Dyadic::ZERO`].
    #[must_use]
    pub fn is_zero(self) -> bool {
        self.mantissa == 0
    }
}

/// Integer square root, `Some` only when `n` is a perfect square.
///
/// Newton descending from a power-of-two overestimate: the iterate decreases
/// monotonically to `floor(sqrt(n))`, so the first non-decrease terminates it.
/// The final multiply is checked because `x·x` can reach `2^128` exactly.
fn exact_isqrt(n: u128) -> Option<u128> {
    if n < 2 {
        return Some(n);
    }
    let mut x = 1u128 << (128 - n.leading_zeros()).div_ceil(2);
    loop {
        let next = (x + n / x) / 2;
        if next >= x {
            break;
        }
        x = next;
    }
    (x.checked_mul(x)? == n).then_some(x)
}

impl PartialOrd for Dyadic {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Dyadic {
    /// Total order by mathematical value.
    ///
    /// NOT lexicographic on the fields: `1 × 2^1` and `1 × 2^0` compare by
    /// value. Magnitudes are separated first by binade (`bits + exp` brackets
    /// `floor(log2)`), which is overflow-free; only a binade tie needs an
    /// alignment, and a tie forces the shift to equal the bit-length
    /// difference, so that alignment always fits.
    fn cmp(&self, other: &Self) -> Ordering {
        let (sign_a, sign_b) = (self.mantissa.signum(), other.mantissa.signum());
        if sign_a != sign_b {
            return sign_a.cmp(&sign_b);
        }
        if sign_a == 0 {
            return Ordering::Equal;
        }

        let binade_a = i64::from(self.magnitude_bits()) + i64::from(self.exp);
        let binade_b = i64::from(other.magnitude_bits()) + i64::from(other.exp);
        let magnitude_order = if binade_a != binade_b {
            binade_a.cmp(&binade_b)
        } else if self.exp <= other.exp {
            let shift = (other.exp - self.exp) as u32;
            self.mantissa
                .unsigned_abs()
                .cmp(&(other.mantissa.unsigned_abs() << shift))
        } else {
            let shift = (self.exp - other.exp) as u32;
            (self.mantissa.unsigned_abs() << shift).cmp(&other.mantissa.unsigned_abs())
        };

        // More negative is smaller.
        if sign_a < 0 {
            magnitude_order.reverse()
        } else {
            magnitude_order
        }
    }
}
