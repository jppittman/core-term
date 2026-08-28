//! Adversarial, specification-only test suite for [`Dyadic`].
//!
//! **Independence note:** these tests were written from `pixelflow-ir/src/dyadic.rs`'s
//! module docs and per-method doc comments ONLY, while the method bodies were
//! still `todo!()`. No implementation body was read or relied on. All
//! constructions below go through the public API (`Dyadic`'s fields are
//! private), so nothing here can accidentally depend on internal layout.
//!
//! # Oracle strategy
//!
//! `Dyadic`'s own fields are inaccessible from outside the crate, so a value
//! can only be inspected via `to_f32()` -- which is itself under test. To get
//! an oracle independent of `Dyadic`, most rounding/overflow/subnormal tests
//! build a value from small, individually-exact pieces (odd `f32`-exact
//! integers times an exact power of two, see [`pow2`] and [`odd_scaled`]),
//! track the SAME value as an exact `f64`, and compare `dyadic.to_f32()`
//! against Rust's own guaranteed round-to-nearest-ties-to-even `f64 as f32`
//! cast of that `f64` (see [`assert_dyadic_matches_f64`]). This was spot
//! checked against real `rustc` output for every nontrivial case below
//! (ties, carries, overflow, underflow) before being encoded here, so the
//! expected bit patterns are independently verified, not merely asserted to
//! agree with themselves.
//!
//! Where a value fits in an ordinary named f32 constant (`f32::MAX`,
//! `f32::MIN_POSITIVE`, ...), that constant is used directly instead.
//!
//! # Running
//!
//! `cargo test -p pixelflow-ir --test dyadic_spec` runs everything except the
//! exhaustive round-trip, which is `#[ignore]`d -- see that test's doc
//! comment for how to run it.
//!
//! # Expected state right now
//!
//! `src/dyadic.rs` bodies are `todo!()` as of this writing. Every test here
//! is expected to panic on first contact with a `todo!()`. That is not a
//! failure of this suite.

use pixelflow_ir::dyadic::Dyadic;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

// ============================================================================
// Helpers (public-API only; no field access)
// ============================================================================

/// Shorthand for an expected-successful `from_f32`. Panics loudly (with the
/// input value) if decoding fails, rather than propagating a confusing
/// `unwrap()` panic with no context.
fn d(x: f32) -> Dyadic {
    Dyadic::from_f32(x).unwrap_or_else(|| panic!("from_f32({x:e}) must decode (finite input)"))
}

/// Exact `2^k` as a Dyadic, for ANY `i32` k -- including exponents far
/// outside f32's own representable range (roughly -149..=127). Built by
/// binary exponentiation from `from_f32(2.0)` / `from_f32(0.5)`, so the
/// mantissa is always the literal integer `1` throughout construction: every
/// `mul` call below is therefore trivially in-range, which is what makes
/// this helper safe to use as scaffolding for tests that specifically probe
/// the overflow boundary of OTHER operations.
fn pow2(k: i32) -> Dyadic {
    let one = d(1.0);
    if k == 0 {
        return one;
    }
    let (mut base, mut n) = if k > 0 {
        (d(2.0), k as u32)
    } else {
        (d(0.5), k.unsigned_abs())
    };
    let mut acc = one;
    while n > 0 {
        if n & 1 == 1 {
            acc = acc.checked_mul(base).unwrap_or_else(|| {
                panic!(
                    "pow2({k}) helper: `mul` returned None accumulating 2^{k}; \
                     the mantissa is always 1 in this helper so this cannot \
                     legitimately overflow -- `mul` itself is broken"
                )
            });
        }
        base = base.checked_mul(base).unwrap_or_else(|| {
            panic!(
                "pow2({k}) helper: `mul` returned None squaring the base for \
                 2^{k}; the mantissa is always 1 in this helper so this cannot \
                 legitimately overflow -- `mul` itself is broken"
            )
        });
        n >>= 1;
    }
    acc
}

/// `3^64` as a Dyadic: a ~102-bit ODD mantissa at exponent 0, built by six
/// successive self-squarings from a single f32-exact seed (`3.0`). Chosen
/// because it sits comfortably inside i128's 127-bit magnitude cap, while
/// squaring it once more (`3^128`, ~203 bits) is exactly where `mul` must
/// decline -- see the "declining at the cap" tests below. Also doubles as a
/// "big mantissa, small exponent" counterexample for the Ord tests.
fn three_pow_64() -> Dyadic {
    let mut v = d(3.0);
    let mut label = 1u32;
    for _ in 0..6 {
        label *= 2;
        v = v.checked_mul(v).unwrap_or_else(|| {
            panic!(
                "3^{label}: mul declined despite fitting comfortably in i128 \
                 (~102 bits at the end); either `mul` is broken or the \
                 mantissa cap is far lower than documented"
            )
        });
    }
    v
}

/// Build `(product of odd factors) x 2^exp` as both an exact `Dyadic` and an
/// exact `f64` oracle. Every factor must be odd, positive, and small enough
/// to be f32-exact (<= 2^24 - 1), so each decodes via `from_f32` without any
/// normalizing shift, and the running product must stay under 2^53 so the
/// f64 oracle also holds it exactly. Panics loudly if a caller violates
/// those preconditions -- a silently-wrong oracle would be worse than none.
fn odd_scaled(factors: &[i64], exp: i32) -> (Dyadic, f64) {
    let mut val = d(1.0);
    let mut product: i128 = 1;
    for &f in factors {
        assert_eq!(f.rem_euclid(2), 1, "odd_scaled: factor {f} must be odd");
        assert!(
            f > 0 && f <= 0x00FF_FFFF,
            "odd_scaled: factor {f} must be a positive f32-exact odd integer (<= 2^24 - 1)"
        );
        val = val.checked_mul(d(f as f32)).unwrap_or_else(|| {
            panic!("odd_scaled{factors:?}: mantissa product overflowed i128 -- reduce the test's factors")
        });
        product = product
            .checked_mul(f as i128)
            .expect("odd_scaled helper: test-chosen factors overflowed i128 in the oracle itself");
    }
    assert!(
        product < (1i128 << 53),
        "odd_scaled{factors:?}: product {product} exceeds 2^53, the f64 oracle would itself be inexact"
    );
    let oracle = (product as f64) * 2f64.powi(exp);
    val = val.checked_mul(pow2(exp)).unwrap_or_else(|| {
        panic!("odd_scaled{factors:?} * 2^{exp}: scale overflowed -- reduce the test's exponent")
    });
    (val, oracle)
}

/// The exact IEEE-754 binary32 bit pattern for `(-1)^neg * 2^e`, for `e` in
/// `-149..=127` (smallest subnormal through largest normal exponent). Built
/// directly from the bit layout so it is exact BY CONSTRUCTION -- unlike
/// `2f32.powi(e)`, which computes via repeated floating-point
/// multiplication and is not guaranteed exact for every e in this range.
fn pow2_f32_bits(e: i32, neg: bool) -> u32 {
    assert!(
        (-149..=127).contains(&e),
        "pow2_f32_bits: e={e} out of f32's representable exponent range"
    );
    let magnitude = if e >= -126 {
        let biased_exp = (e + 127) as u32; // 1..=254
        biased_exp << 23
    } else {
        1u32 << (e + 149) // subnormal: mantissa field alone encodes the power of two
    };
    if neg {
        magnitude | 0x8000_0000
    } else {
        magnitude
    }
}

/// Core round-trip assertion used by every sampled/exhaustive round-trip
/// test: `from_f32(x).map(to_f32) == Some(x)` for finite x, `None` for
/// NaN/inf, with the documented exception that `-0.0` round-trips to `+0.0`
/// (both collapse to the one [`Dyadic::ZERO`]).
fn assert_roundtrip_bits(bits: u32) {
    let x = f32::from_bits(bits);
    if x.is_nan() {
        assert!(
            Dyadic::from_f32(x).is_none(),
            "NaN (bits {bits:#010x}) must decode to None"
        );
        return;
    }
    if x.is_infinite() {
        assert!(
            Dyadic::from_f32(x).is_none(),
            "infinity (bits {bits:#010x}) must decode to None"
        );
        return;
    }
    let Some(dy) = Dyadic::from_f32(x) else {
        panic!("finite value {x:e} (bits {bits:#010x}) must decode to Some, got None");
    };
    let back = dy.to_f32();
    let expected_bits = if bits == 0x8000_0000 { 0u32 } else { bits };
    assert_eq!(
        back.to_bits(),
        expected_bits,
        "round-trip mismatch: x={x:e} bits={bits:#010x} -> to_f32 bits={:#010x}, expected {expected_bits:#010x}",
        back.to_bits()
    );
}

/// Assert that converting `dy` to f32 matches Rust's own guaranteed
/// correctly-rounded (round-to-nearest, ties-to-even) `f64 as f32` cast of
/// the independently-known exact value `oracle`. This oracle is wholly
/// external to `Dyadic`'s implementation.
fn assert_dyadic_matches_f64(label: &str, dy: Dyadic, oracle: f64) {
    let got = dy.to_f32();
    let want = oracle as f32;
    assert_eq!(
        got.to_bits(),
        want.to_bits(),
        "{label}: to_f32() = {got:e} (bits {:#010x}), oracle (({oracle:e}) as f32) = {want:e} (bits {:#010x})",
        got.to_bits(),
        want.to_bits()
    );
}

fn hash_of(x: Dyadic) -> u64 {
    let mut hasher = DefaultHasher::new();
    x.hash(&mut hasher);
    hasher.finish()
}

// ============================================================================
// 1. from_f32: NaN / infinity / signed-zero contract
// ============================================================================

#[test]
fn from_f32_rejects_all_nan_shapes() {
    let nans: [(&str, u32); 6] = [
        ("quiet NaN", f32::NAN.to_bits()),
        (
            "quiet NaN, negative sign bit",
            f32::NAN.to_bits() | 0x8000_0000,
        ),
        ("signaling-shaped NaN pattern", 0x7FA0_0000),
        ("signaling-shaped NaN, negative", 0xFFA0_0000),
        ("NaN with minimal payload", 0x7F80_0001),
        ("NaN with max payload, negative", 0xFFFF_FFFF),
    ];
    for (label, bits) in nans {
        let x = f32::from_bits(bits);
        assert!(
            x.is_nan(),
            "test bug: bits {bits:#010x} ({label}) is not actually NaN"
        );
        assert!(
            Dyadic::from_f32(x).is_none(),
            "{label} (bits {bits:#010x}) must decode to None, not Some"
        );
    }
}

#[test]
fn from_f32_rejects_both_infinities() {
    assert!(
        Dyadic::from_f32(f32::INFINITY).is_none(),
        "+inf must decode to None"
    );
    assert!(
        Dyadic::from_f32(f32::NEG_INFINITY).is_none(),
        "-inf must decode to None"
    );
}

#[test]
fn from_f32_collapses_signed_zero_to_the_one_zero() {
    let pos = Dyadic::from_f32(0.0).expect("+0.0 must decode");
    let neg = Dyadic::from_f32(-0.0).expect("-0.0 must decode");
    assert_eq!(pos, Dyadic::ZERO, "+0.0 must decode to Dyadic::ZERO");
    assert_eq!(
        neg,
        Dyadic::ZERO,
        "-0.0 must decode to Dyadic::ZERO (module docs: there is one zero)"
    );
    assert_eq!(pos, neg);
}

#[test]
fn to_f32_of_zero_is_always_positive_zero() {
    assert_eq!(
        Dyadic::ZERO.to_f32().to_bits(),
        0.0f32.to_bits(),
        "ZERO.to_f32() must be +0.0"
    );
    let via_neg_zero = Dyadic::from_f32(-0.0).unwrap();
    assert_eq!(
        via_neg_zero.to_f32().to_bits(),
        0.0f32.to_bits(),
        "zero built from -0.0 must still round-trip to +0.0, per module docs"
    );
}

#[test]
fn from_f32_accepts_ordinary_finite_values() {
    for x in [
        1.0f32,
        -1.0,
        0.5,
        128.0 / 255.0,
        f32::MIN_POSITIVE,
        f32::MAX,
        f32::MIN,
        core::f32::consts::PI,
    ] {
        assert!(
            Dyadic::from_f32(x).is_some(),
            "finite value {x:e} must decode to Some"
        );
    }
}

// ============================================================================
// 2. Round-trip: from_f32 -> to_f32 (sampled by default, exhaustive ignored)
// ============================================================================

#[test]
fn round_trip_all_positive_subnormals() {
    for bits in 0x0000_0001u32..=0x007f_ffff {
        assert_roundtrip_bits(bits);
    }
}

#[test]
fn round_trip_all_negative_subnormals() {
    for bits in 0x8000_0001u32..=0x807f_ffff {
        assert_roundtrip_bits(bits);
    }
}

#[test]
fn round_trip_all_powers_of_two() {
    for e in -149i32..=127 {
        assert_roundtrip_bits(pow2_f32_bits(e, false));
        assert_roundtrip_bits(pow2_f32_bits(e, true));
    }
}

#[test]
fn round_trip_dense_near_binade_boundaries() {
    // A spread of exponents from deep subnormal, across the subnormal/normal
    // seam, through the middle of the normal range, to just below overflow.
    const BOUNDARY_EXPONENTS: [i32; 9] = [-149, -126, -100, -23, -1, 23, 100, 126, 127];
    const SPAN: u32 = 4096;
    for &e in &BOUNDARY_EXPONENTS {
        for &neg in &[false, true] {
            let center = pow2_f32_bits(e, neg);
            let lo = center.saturating_sub(SPAN);
            let hi = center.saturating_add(SPAN);
            for bits in lo..=hi {
                assert_roundtrip_bits(bits);
            }
        }
    }
}

#[test]
fn round_trip_pseudo_random_spread() {
    // Deterministic LCG (Numerical Recipes constants), full 32-bit state --
    // no `rand` dependency, fully reproducible across runs/machines.
    let mut state: u32 = 0x2545_F491;
    const SAMPLES: usize = 1_000_000;
    for _ in 0..SAMPLES {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        assert_roundtrip_bits(state);
    }
}

#[test]
#[ignore = "exhaustive: iterates all 2^32 f32 bit patterns (~a few minutes in \
            release, impractically slow in debug). Run explicitly with: \
            cargo test -p pixelflow-ir --test dyadic_spec --release -- \
            --ignored exhaustive_round_trip_over_all_f32_bit_patterns"]
fn exhaustive_round_trip_over_all_f32_bit_patterns() {
    for bits in 0u32..=u32::MAX {
        assert_roundtrip_bits(bits);
    }
}

// ============================================================================
// 3. Subnormals: decode correctness and the subnormal/normal seam
// ============================================================================

#[test]
fn smallest_subnormal_decodes_and_doubles_correctly() {
    let smallest = Dyadic::from_f32(f32::from_bits(1)).expect("smallest subnormal must decode");
    let doubled = smallest
        .checked_add(smallest)
        .expect("doubling the smallest subnormal must not overflow");
    let expected = Dyadic::from_f32(f32::from_bits(2)).expect("2nd-smallest subnormal must decode");
    assert_eq!(
        doubled, expected,
        "2 * (bit-pattern 1) must equal (bit-pattern 2): if from_f32 wrongly \
         assumed an implicit leading 1 for subnormals, this identity would break"
    );
}

#[test]
fn subnormal_doubling_identity_across_the_subnormal_range() {
    for k in [1u32, 2, 3, 0xFF, 0x3FFF, 0x003F_FFFF] {
        let x = Dyadic::from_f32(f32::from_bits(k))
            .unwrap_or_else(|| panic!("bits {k:#010x} must decode (subnormal)"));
        let doubled = x
            .checked_add(x)
            .unwrap_or_else(|| panic!("doubling subnormal bits {k:#010x} must not overflow"));
        let expected_bits = 2 * k;
        let expected = Dyadic::from_f32(f32::from_bits(expected_bits))
            .unwrap_or_else(|| panic!("bits {expected_bits:#010x} must decode"));
        assert_eq!(
            doubled, expected,
            "2x(bits {k:#010x}) must equal bits {expected_bits:#010x}"
        );
    }
}

#[test]
fn largest_subnormal_and_smallest_normal_are_exactly_one_ulp_apart() {
    let largest_subnormal =
        Dyadic::from_f32(f32::from_bits(0x007f_ffff)).expect("largest subnormal must decode");
    let smallest_normal = Dyadic::from_f32(f32::MIN_POSITIVE).expect("smallest normal must decode");
    let smallest_subnormal =
        Dyadic::from_f32(f32::from_bits(1)).expect("smallest subnormal must decode");

    assert_ne!(
        largest_subnormal, smallest_normal,
        "these are adjacent but distinct f32 values"
    );

    let gap = smallest_normal
        .checked_sub(largest_subnormal)
        .expect("adjacent-value subtraction must not overflow");
    assert_eq!(
        gap, smallest_subnormal,
        "smallest_normal - largest_subnormal must equal exactly the smallest \
         subnormal (2^-126 - (2^-126 - 2^-149) = 2^-149) -- this cross-checks \
         subnormal AND normal decoding against each other without inspecting fields"
    );
}

#[test]
fn subnormal_normal_seam_tie_rounds_up_to_smallest_normal_ties_to_even() {
    // Exactly halfway between the largest subnormal (2^-126 - 2^-149) and
    // the smallest normal (2^-126): 2^-126 - 2^-150. The smallest normal has
    // an even trailing mantissa bit (field 0); the largest subnormal an odd
    // one (field 2^23-1). Ties-to-even must cross the seam.
    let tie = pow2(-126)
        .checked_sub(pow2(-150))
        .expect("seam tie construction must not overflow");
    let oracle = 2f64.powi(-126) - 2f64.powi(-150);
    assert_dyadic_matches_f64("subnormal/normal seam tie", tie, oracle);
    assert_eq!(
        tie.to_f32().to_bits(),
        f32::MIN_POSITIVE.to_bits(),
        "must round exactly to the smallest normal f32"
    );
}

#[test]
fn value_just_inside_subnormal_side_of_seam_still_rounds_up_to_smallest_normal() {
    // 2^-126 - 2^-151 is only 0.25 subnormal-ULPs below the smallest normal
    // -- not a tie, unambiguously closer to the normal side.
    let v = pow2(-126)
        .checked_sub(pow2(-151))
        .expect("must not overflow");
    let oracle = 2f64.powi(-126) - 2f64.powi(-151);
    assert_dyadic_matches_f64("0.25-ulp below the seam", v, oracle);
    assert_eq!(v.to_f32().to_bits(), f32::MIN_POSITIVE.to_bits());
}

#[test]
fn tie_between_zero_and_smallest_subnormal_rounds_to_even_zero() {
    let tie = pow2(-150); // exactly half of the smallest subnormal (2^-149)
    let oracle = 2f64.powi(-150);
    assert_dyadic_matches_f64("zero/smallest-subnormal tie", tie, oracle);
    assert_eq!(
        tie.to_f32().to_bits(),
        0.0f32.to_bits(),
        "tie must resolve to zero (the even neighbor)"
    );
}

#[test]
fn just_above_the_zero_tie_rounds_up_to_smallest_subnormal() {
    let v = pow2(-150)
        .checked_add(pow2(-200))
        .expect("must not overflow");
    let oracle = 2f64.powi(-150) + 2f64.powi(-200);
    assert_dyadic_matches_f64("just above zero/smallest-subnormal tie", v, oracle);
    assert_eq!(
        v.to_f32().to_bits(),
        1u32,
        "must round up to bit pattern 1 (smallest subnormal)"
    );
}

#[test]
fn just_below_the_zero_tie_rounds_down_to_zero() {
    let v = pow2(-150)
        .checked_sub(pow2(-200))
        .expect("must not overflow");
    let oracle = 2f64.powi(-150) - 2f64.powi(-200);
    assert_dyadic_matches_f64("just below zero/smallest-subnormal tie", v, oracle);
    assert_eq!(v.to_f32().to_bits(), 0u32);
}

#[test]
fn deep_underflow_rounds_to_correctly_signed_zero() {
    assert_eq!(
        pow2(-500).to_f32().to_bits(),
        0u32,
        "deep positive underflow -> +0.0"
    );
    assert_eq!(
        pow2(-500).negate().to_f32().to_bits(),
        0x8000_0000u32,
        "deep negative underflow -> -0.0 (sign survives rounding, per IEEE \
         correctly-rounded conversion semantics)"
    );
}

// ============================================================================
// 4. Rounding: ties-to-even, and carry into the next binade
// ============================================================================

#[test]
fn in_binade_tie_with_even_lsb_below_rounds_down() {
    // Within [1,2): representable value m=2^23-2 has an EVEN mantissa field.
    // The tie exactly between it and m+1 (odd) must resolve DOWN to m.
    let v = d(1.0)
        .checked_add(d(16_777_213.0).checked_mul(pow2(-24)).unwrap())
        .unwrap();
    let oracle = 1.0 + 16_777_213.0 * 2f64.powi(-24);
    assert_dyadic_matches_f64("in-binade tie, even lsb below", v, oracle);
    assert_eq!(
        v.to_f32().to_bits(),
        0x3FFF_FFFE,
        "must round down to the even-mantissa neighbor"
    );
}

#[test]
fn in_binade_tie_with_odd_lsb_below_rounds_up_without_carry() {
    // m=5 (odd) -> tie must resolve UP to m+1=6 (even), staying inside [1,2).
    let v = d(1.0)
        .checked_add(d(11.0).checked_mul(pow2(-24)).unwrap())
        .unwrap();
    let oracle = 1.0 + 11.0 * 2f64.powi(-24);
    assert_dyadic_matches_f64("in-binade tie, odd lsb below, no carry", v, oracle);
    assert_eq!(v.to_f32().to_bits(), 0x3F80_0006);
}

#[test]
fn value_just_below_an_in_binade_tie_rounds_to_the_lower_neighbor() {
    let tie = d(1.0)
        .checked_add(d(11.0).checked_mul(pow2(-24)).unwrap())
        .unwrap();
    let v = tie.checked_sub(pow2(-30)).unwrap();
    let oracle = (1.0 + 11.0 * 2f64.powi(-24)) - 2f64.powi(-30);
    assert_dyadic_matches_f64("just below an in-binade tie", v, oracle);
    assert_eq!(v.to_f32().to_bits(), 0x3F80_0005);
}

#[test]
fn value_just_above_an_in_binade_tie_rounds_to_the_upper_neighbor() {
    let tie = d(1.0)
        .checked_add(d(11.0).checked_mul(pow2(-24)).unwrap())
        .unwrap();
    let v = tie.checked_add(pow2(-30)).unwrap();
    let oracle = (1.0 + 11.0 * 2f64.powi(-24)) + 2f64.powi(-30);
    assert_dyadic_matches_f64("just above an in-binade tie", v, oracle);
    assert_eq!(v.to_f32().to_bits(), 0x3F80_0006);
}

#[test]
fn tie_at_top_of_binade_carries_into_the_next_exponent() {
    // 2.0 - 2^-24 is exactly halfway between (2.0 - 2^-23) [odd mantissa
    // field 0x7FFFFF] and 2.0 itself [mantissa field 0, even, next binade].
    // Ties-to-even must carry into the next exponent.
    let v = pow2(1).checked_sub(pow2(-24)).unwrap();
    let oracle = 2f64.powi(1) - 2f64.powi(-24);
    assert_dyadic_matches_f64("carry-tie at top of [1,2)", v, oracle);
    assert_eq!(v.to_f32().to_bits(), 2.0f32.to_bits());
}

#[test]
fn value_just_below_a_power_of_two_that_is_not_a_tie_still_carries() {
    // 2.0 - 2^-30 is unambiguously (NOT a tie) closer to 2.0 than to
    // 2.0 - 2^-23, so it must round up and carry into the next binade even
    // though nothing about this specific case is a tie.
    let v = pow2(1).checked_sub(pow2(-30)).unwrap();
    let oracle = 2f64.powi(1) - 2f64.powi(-30);
    assert_dyadic_matches_f64("non-tie carry just below 2.0", v, oracle);
    assert_eq!(v.to_f32().to_bits(), 2.0f32.to_bits());
}

// ============================================================================
// 5. Overflow to infinity, and rounding to the largest finite f32 instead
// ============================================================================

#[test]
fn tie_just_above_f32_max_rounds_to_infinity_ties_to_even() {
    // f32::MAX's mantissa field is all-ones (odd); infinity's is 0 (even).
    // The tie exactly halfway to 2^128 must resolve to infinity.
    let max = d(f32::MAX);
    let tie = max.checked_add(pow2(103)).unwrap(); // ulp at this binade is 2^104, half is 2^103
    let oracle = f32::MAX as f64 + 2f64.powi(103);
    assert_dyadic_matches_f64("tie just above f32::MAX", tie, oracle);
    assert_eq!(tie.to_f32(), f32::INFINITY);
}

#[test]
fn value_a_quarter_ulp_above_f32_max_rounds_down_to_f32_max_not_infinity() {
    let max = d(f32::MAX);
    let v = max.checked_add(pow2(102)).unwrap(); // 0.25 ulp over
    let oracle = f32::MAX as f64 + 2f64.powi(102);
    assert_dyadic_matches_f64("0.25 ulp above f32::MAX", v, oracle);
    assert_eq!(
        v.to_f32(),
        f32::MAX,
        "must round DOWN to the largest finite f32, not overflow to infinity"
    );
}

#[test]
fn value_three_quarter_ulp_above_f32_max_overflows_to_infinity() {
    let max = d(f32::MAX);
    let v = max
        .checked_add(d(3.0).checked_mul(pow2(102)).unwrap())
        .unwrap(); // 0.75 ulp over
    let oracle = f32::MAX as f64 + 3.0 * 2f64.powi(102);
    assert_dyadic_matches_f64("0.75 ulp above f32::MAX", v, oracle);
    assert_eq!(v.to_f32(), f32::INFINITY);
}

#[test]
fn overflow_and_near_max_rounding_are_sign_symmetric() {
    let neg_max = d(f32::MAX).negate();

    let tie = neg_max.checked_sub(pow2(103)).unwrap();
    assert_eq!(
        tie.to_f32(),
        f32::NEG_INFINITY,
        "negative tie must round to -infinity"
    );

    let below = neg_max.checked_sub(pow2(102)).unwrap();
    assert_eq!(
        below.to_f32(),
        f32::MIN,
        "0.25 ulp past -MAX must round to the most-negative finite f32"
    );

    let above = neg_max
        .checked_sub(d(3.0).checked_mul(pow2(102)).unwrap())
        .unwrap();
    assert_eq!(
        above.to_f32(),
        f32::NEG_INFINITY,
        "0.75 ulp past -MAX must overflow to -infinity"
    );
}

#[test]
fn far_beyond_f32_range_overflows_to_infinity_of_the_right_sign() {
    assert_eq!(pow2(200).to_f32(), f32::INFINITY);
    assert_eq!(pow2(200).negate().to_f32(), f32::NEG_INFINITY);
    assert_eq!(
        pow2(1000).to_f32(),
        f32::INFINITY,
        "must saturate, not panic or wrap, arbitrarily far past the range"
    );
}

// ============================================================================
// 6. Exactness: the motivating (x + 2^k) - 2^k case
// ============================================================================

fn exactness_survives_round_trip_at_magnitude(k: i32) {
    let big = pow2(k);
    for x in [
        128.0f32 / 255.0,
        1.0 / 3.0,
        std::f32::consts::PI,
        -7.0 / 9.0,
        12345.679,
    ] {
        let small = d(x);
        let result = small
            .checked_add(big)
            .unwrap_or_else(|| {
                panic!("{x} + 2^{k} must not overflow (tiny mantissa, exponent gap fits i128)")
            })
            .checked_sub(big)
            .unwrap_or_else(|| panic!("(...) - 2^{k} must not overflow"));
        assert_eq!(
            result, small,
            "({x} + 2^{k}) - 2^{k} must be EXACTLY {x} again -- this is the \
             motivating case from the module docs (f32 arithmetic absorbs it to 0)"
        );
        assert_eq!(
            result.to_f32().to_bits(),
            x.to_bits(),
            "and must round-trip back to the identical f32 bit pattern"
        );
    }
}

#[test]
fn exactness_survives_add_then_sub_at_2_24() {
    exactness_survives_round_trip_at_magnitude(24);
}

#[test]
fn exactness_survives_add_then_sub_at_2_30() {
    exactness_survives_round_trip_at_magnitude(30);
}

#[test]
fn exactness_survives_add_then_sub_at_2_60() {
    exactness_survives_round_trip_at_magnitude(60);
}

#[test]
fn exactness_survives_add_then_sub_at_2_80() {
    exactness_survives_round_trip_at_magnitude(80);
}

#[test]
fn f32_arithmetic_absorbs_the_small_addend_for_contrast() {
    // NOT a Dyadic test: documents WHY Dyadic exists (module docs, "Why
    // exactness, rather than more precision"). Plain f32 rounds (x + 2^24)
    // back down before subtracting 2^24 back off, losing x entirely.
    let x: f32 = 128.0 / 255.0;
    let absorbed = (x + 2f32.powi(24)) - 2f32.powi(24);
    assert_ne!(
        absorbed, x,
        "expected f32 to LOSE x={x} at this magnitude (the motivating failure); \
         if this now holds, this contrast comment/test is stale"
    );
}

// ============================================================================
// 7. Declining at the cap: None instead of a wrong value
// ============================================================================

#[test]
fn mul_succeeds_all_the_way_up_to_the_127_bit_cap() {
    let v = three_pow_64();
    assert!(!v.is_zero());
    // Cross-check via a second, independently-specified operation: dividing
    // 3^64 by itself must be exactly ONE, confirming the value survived
    // construction intact (not silently truncated to something else that
    // happens to also be nonzero).
    assert_eq!(v.try_div(v), Some(d(1.0)));
}

#[test]
fn mul_declines_past_the_127_bit_mantissa_width() {
    // 3^64 (~102 bits) is comfortably within i128; 3^128 (~203 bits) is not.
    let v = three_pow_64();
    let overflowed = v.checked_mul(v);
    assert!(
        overflowed.is_none(),
        "3^64 * 3^64 = 3^128 needs ~203 mantissa bits; i128 holds ~127. \
         mul() must return None, not a truncated/wrapped value. Got: {overflowed:?}"
    );
}

#[test]
fn mul_decline_is_sign_independent() {
    let v = three_pow_64();
    assert!(
        v.negate().checked_mul(v).is_none(),
        "negative lhs must not bypass the overflow check"
    );
    assert!(
        v.checked_mul(v.negate()).is_none(),
        "negative rhs must not bypass the overflow check"
    );
    assert!(
        v.negate().checked_mul(v.negate()).is_none(),
        "negative*negative must not bypass the overflow check"
    );
}

#[test]
fn add_declines_when_the_exponent_gap_exceeds_capacity() {
    let big = three_pow_64(); // ~102-bit odd mantissa, exponent 0
    let far = pow2(-400); // mantissa 1, exponent -400
    assert!(
        big.checked_add(far).is_none(),
        "add must decline (None) rather than silently truncate when aligning \
         a ~102-bit mantissa across a 400-exponent gap exceeds i128's width"
    );
    assert!(
        far.checked_add(big).is_none(),
        "add must decline symmetrically regardless of operand order"
    );
}

#[test]
fn sub_declines_when_the_exponent_gap_exceeds_capacity() {
    let big = three_pow_64();
    let far = pow2(-400);
    assert!(
        big.checked_sub(far).is_none(),
        "sub must decline past capacity, same as add (see Self::add docs)"
    );
    assert!(
        far.checked_sub(big).is_none(),
        "sub must decline symmetrically regardless of operand order"
    );
}

// ============================================================================
// 8. try_div
// ============================================================================

#[test]
fn try_div_exact_simple_fractions() {
    assert_eq!(d(1.0).try_div(d(2.0)), Some(d(0.5)));
    assert_eq!(d(3.0).try_div(d(4.0)), Some(d(0.75)));
    assert_eq!(d(6.0).try_div(d(3.0)), Some(d(2.0)));
    assert_eq!(d(1.0).try_div(d(1024.0)), Some(d(1.0 / 1024.0)));
}

#[test]
fn try_div_by_self_gives_exactly_one() {
    let one = d(1.0);
    for x in [
        d(128.0 / 255.0),
        d(-7.5),
        d(f32::MIN_POSITIVE),
        d(f32::MAX),
        pow2(-100),
        pow2(77),
        three_pow_64(),
    ] {
        assert_eq!(
            x.try_div(x),
            Some(one),
            "x/x must be exactly ONE for x={x:?}"
        );
    }
}

#[test]
fn try_div_identity_with_one() {
    for x in [1.0f32, -1.0, 0.5, 128.0 / 255.0, core::f32::consts::PI] {
        let dx = d(x);
        assert_eq!(
            dx.try_div(d(1.0)),
            Some(dx),
            "x/1 must equal x exactly for x={x}"
        );
    }
}

#[test]
fn try_div_powers_of_two_are_always_exact() {
    for a in [100, 30, 0, -30] {
        for b in [70, 10, 0, -10] {
            let got = pow2(a).try_div(pow2(b));
            assert_eq!(
                got,
                Some(pow2(a - b)),
                "2^{a} / 2^{b} must be exactly 2^{}",
                a - b
            );
        }
    }
}

#[test]
fn try_div_zero_numerator_by_nonzero_is_exactly_zero() {
    assert_eq!(Dyadic::ZERO.try_div(d(7.0)), Some(Dyadic::ZERO));
    assert_eq!(Dyadic::ZERO.try_div(pow2(-50)), Some(Dyadic::ZERO));
}

#[test]
fn try_div_by_zero_is_always_none() {
    assert!(
        d(5.0).try_div(Dyadic::ZERO).is_none(),
        "nonzero / 0 must be None (no dyadic represents infinity)"
    );
    assert!(
        Dyadic::ZERO.try_div(Dyadic::ZERO).is_none(),
        "0 / 0 must be None per the documented zero-divisor rule"
    );
    assert!(d(-3.0).try_div(Dyadic::ZERO).is_none());
}

#[test]
fn try_div_declines_on_genuinely_inexact_quotients() {
    for (num, den) in [(1.0, 3.0), (1.0, 5.0), (1.0, 10.0), (2.0, 7.0), (10.0, 3.0)] {
        assert!(
            d(num).try_div(d(den)).is_none(),
            "{num}/{den} is not exactly representable as a dyadic rational \
             (denominator has a prime factor other than 2) -- must be None"
        );
    }
}

// ============================================================================
// 9. try_sqrt
// ============================================================================

#[test]
fn try_sqrt_perfect_squares() {
    assert_eq!(d(4.0).try_sqrt(), Some(d(2.0)));
    assert_eq!(d(9.0).try_sqrt(), Some(d(3.0)));
    assert_eq!(d(25.0).try_sqrt(), Some(d(5.0)));
    assert_eq!(d(49.0).try_sqrt(), Some(d(7.0)));
    assert_eq!(d(0.25).try_sqrt(), Some(d(0.5)));
    assert_eq!(pow2(-10).try_sqrt(), Some(pow2(-5)));
    assert_eq!(pow2(20).try_sqrt(), Some(pow2(10)));
}

#[test]
fn try_sqrt_zero_gives_some_zero() {
    assert_eq!(Dyadic::ZERO.try_sqrt(), Some(Dyadic::ZERO));
    assert_eq!(d(0.0).try_sqrt(), Some(Dyadic::ZERO));
    assert_eq!(d(-0.0).try_sqrt(), Some(Dyadic::ZERO));
}

#[test]
fn try_sqrt_negative_is_always_none() {
    assert!(d(-4.0).try_sqrt().is_none());
    assert!(pow2(2).negate().try_sqrt().is_none());
    assert!(d(-0.0001).try_sqrt().is_none());
}

#[test]
fn try_sqrt_non_square_mantissa_is_none_even_with_even_exponent() {
    // exp=0 (even) in every case here, isolating "mantissa isn't a perfect
    // square" as the sole reason for None.
    for x in [2.0f32, 3.0, 5.0, 6.0, 7.0, 8.0, 15.0, 21.0] {
        assert!(
            d(x).try_sqrt().is_none(),
            "{x} is not a perfect square, try_sqrt must be None"
        );
    }
}

#[test]
fn try_sqrt_odd_exponent_is_none_even_with_perfect_square_mantissa() {
    // pow2(k) always has mantissa 1 (trivially a perfect square, 1 = 1^2),
    // isolating "exponent parity" as the sole pass/fail factor.
    for e in -21..=21 {
        let v = pow2(e);
        if e % 2 == 0 {
            assert_eq!(
                v.try_sqrt(),
                Some(pow2(e / 2)),
                "2^{e} (even exponent) must have an exact sqrt"
            );
        } else {
            assert!(
                v.try_sqrt().is_none(),
                "2^{e} (odd exponent) must NOT have an exact sqrt (irrational)"
            );
        }
    }
}

#[test]
fn try_sqrt_declines_when_both_mantissa_and_exponent_are_wrong() {
    // mantissa=9 (a perfect square) at an ODD combined exponent: a
    // perfect-square mantissa alone must not be sufficient.
    let v = d(9.0).checked_mul(pow2(1)).unwrap(); // 9 * 2 = 18
    assert!(
        v.try_sqrt().is_none(),
        "18 = 9*2 has a perfect-square mantissa but odd exponent; sqrt(18) is irrational"
    );
}

// ============================================================================
// 10. neg / abs / is_zero
// ============================================================================

#[test]
fn neg_matches_f32_negation_for_ordinary_values() {
    for x in [
        1.0f32,
        -1.0,
        0.5,
        128.0 / 255.0,
        f32::MIN_POSITIVE,
        f32::MAX,
        core::f32::consts::PI,
    ] {
        assert_eq!(
            d(x).negate(),
            d(-x),
            "neg(x) must equal from_f32(-x) for x={x}"
        );
    }
}

#[test]
fn neg_is_its_own_inverse() {
    for x in [d(7.0), pow2(100), pow2(-100), Dyadic::ZERO, three_pow_64()] {
        assert_eq!(x.negate().negate(), x);
    }
}

#[test]
fn neg_zero_is_still_the_one_zero() {
    assert_eq!(Dyadic::ZERO.negate(), Dyadic::ZERO);
}

#[test]
fn adding_a_value_to_its_negation_gives_exactly_zero() {
    for x in [
        d(7.0),
        d(128.0 / 255.0),
        pow2(100),
        pow2(-100),
        three_pow_64(),
        Dyadic::ZERO,
    ] {
        assert_eq!(
            x.checked_add(x.negate()),
            Some(Dyadic::ZERO),
            "x + (-x) must be exactly zero and must not overflow (same magnitude, opposite sign)"
        );
    }
}

#[test]
fn abs_matches_f32_abs_for_ordinary_values() {
    for x in [
        1.0f32,
        -1.0,
        0.5,
        -0.5,
        128.0 / 255.0,
        -128.0 / 255.0,
        f32::MAX,
        f32::MIN,
    ] {
        assert_eq!(
            d(x).abs(),
            d(x.abs()),
            "abs(x) must equal from_f32(x.abs()) for x={x}"
        );
    }
}

#[test]
fn abs_is_idempotent_and_matches_its_negations_abs() {
    for x in [d(-7.0), d(7.0), pow2(50).negate(), pow2(50), Dyadic::ZERO] {
        let a = x.abs();
        assert_eq!(a.abs(), a, "abs must be idempotent");
        assert_eq!(a, x.negate().abs(), "abs(x) must equal abs(-x)");
    }
}

#[test]
fn is_zero_true_only_for_zero() {
    assert!(Dyadic::ZERO.is_zero());
    assert!(d(0.0).is_zero());
    assert!(d(-0.0).is_zero());
    for x in [d(1.0), d(-1.0), pow2(100), pow2(-100), pow2(-500)] {
        // pow2(-500) underflows to 0.0 in f32 but is a genuinely nonzero
        // Dyadic value -- is_zero must not go through any f32 round-trip.
        assert!(!x.is_zero(), "{x:?} is not zero, is_zero must be false");
    }
}

#[test]
fn is_zero_true_for_results_that_cancel_out() {
    assert!(d(5.0).checked_sub(d(5.0)).unwrap().is_zero());
    assert!(d(0.0).checked_mul(d(12345.6)).unwrap().is_zero());
    assert!(Dyadic::ZERO.try_div(d(7.0)).unwrap().is_zero());
    assert!(
        three_pow_64()
            .checked_sub(three_pow_64())
            .unwrap()
            .is_zero()
    );
}

// ============================================================================
// 11. Normalization: Eq / Hash agree with mathematical value
// ============================================================================
//
// `Dyadic` derives `PartialEq`/`Eq`/`Hash` FIELD-WISE. That is only sound
// because the type's invariant guarantees every publicly-obtainable value is
// normalized (mantissa odd, or exactly ZERO) -- see the struct's doc comment.
// These tests exercise that invariant directly: values built via different
// arithmetic routes must land on IDENTICAL fields, not just "equal enough".

#[test]
fn different_construction_routes_to_the_same_value_are_equal_and_hash_equal() {
    let via_add = d(2.0).checked_add(d(2.0)).unwrap();
    let via_mul = d(1.0).checked_mul(d(4.0)).unwrap();
    let via_from_f32 = d(4.0);
    let via_sub = d(6.0).checked_sub(d(2.0)).unwrap();
    let via_div = d(8.0).try_div(d(2.0)).unwrap();
    let via_sqrt = d(16.0).try_sqrt().unwrap();

    let routes = [via_add, via_mul, via_from_f32, via_sub, via_div, via_sqrt];
    for (i, &a) in routes.iter().enumerate() {
        for (j, &b) in routes.iter().enumerate() {
            assert_eq!(
                a, b,
                "route {i} and route {j} must both denote 4.0 and be =="
            );
            assert_eq!(
                hash_of(a),
                hash_of(b),
                "route {i} and route {j} are == and MUST hash equal"
            );
        }
    }
}

#[test]
fn zero_built_every_which_way_is_the_identical_value() {
    let routes = [
        Dyadic::ZERO,
        d(0.0),
        d(-0.0),
        d(5.0).checked_sub(d(5.0)).unwrap(),
        d(1.0).checked_sub(d(1.0)).unwrap(),
        d(0.0).checked_mul(d(12345.0)).unwrap(),
        Dyadic::ZERO.try_div(d(9.0)).unwrap(),
        Dyadic::ZERO.try_sqrt().unwrap(),
        Dyadic::ZERO.negate(),
        pow2(70).checked_sub(pow2(70)).unwrap(),
    ];
    for (i, &a) in routes.iter().enumerate() {
        assert_eq!(a, Dyadic::ZERO, "route {i} must equal Dyadic::ZERO");
        assert_eq!(
            hash_of(a),
            hash_of(Dyadic::ZERO),
            "route {i} must hash equal to Dyadic::ZERO"
        );
    }
}

#[test]
fn addition_of_two_odd_mantissas_renormalizes_away_trailing_zero_bits() {
    // odd + odd is always even -- every one of these MUST renormalize back
    // to an odd mantissa (or ZERO), or Eq/Hash silently stop agreeing with
    // mathematical value.
    assert_eq!(
        d(3.0).checked_add(d(5.0)).unwrap(),
        pow2(3),
        "3+5=8=2^3 must fully normalize (3 trailing zero bits stripped)"
    );
    assert_eq!(d(3.0).checked_add(d(13.0)).unwrap(), pow2(4), "3+13=16=2^4");
    assert_eq!(d(1.0).checked_add(d(7.0)).unwrap(), pow2(3), "1+7=8=2^3");
    // Strip a much longer run of trailing zeros in a single call.
    assert_eq!(
        d(524_287.0).checked_add(d(1.0)).unwrap(),
        pow2(19),
        "524287+1=2^19 (524287=2^19-1 is odd) must strip all 19 trailing zero bits"
    );
}

#[test]
fn subtraction_can_also_produce_a_value_needing_renormalization() {
    assert_eq!(d(7.0).checked_sub(d(3.0)).unwrap(), pow2(2), "7-3=4=2^2");
    assert_eq!(
        d(1.0).checked_sub(d(-7.0)).unwrap(),
        pow2(3),
        "1-(-7)=8=2^3"
    );
}

// ============================================================================
// 12. Ord: total order by VALUE, not lexicographic on fields
// ============================================================================

#[test]
fn ord_is_not_lexicographic_on_mantissa_first() {
    // Small mantissa + large exponent (2^30) vs a large-ish, exponent-0
    // mantissa (8388607, an odd f32-exact int). A mantissa-first
    // lexicographic comparison would see 1 < 8388607 and conclude A < B;
    // the real values disagree: 2^30 (~1.07e9) > 8388607 (~8.39e6).
    let a = pow2(30);
    let b = d(8_388_607.0);
    assert!(
        a > b,
        "2^30 must compare greater than 8388607 by VALUE, not by a smaller stored mantissa"
    );
    assert!(b < a);
}

#[test]
fn ord_is_not_lexicographic_on_exponent_first() {
    // Small exponent + large mantissa (8388607) vs large exponent + mantissa
    // 1 (2^5=32). An exponent-first lexicographic comparison would see A's
    // exponent as larger and conclude A > B; by real value, 32 < 8388607.
    let a = pow2(5);
    let b = d(8_388_607.0);
    assert!(
        a < b,
        "32 must compare less than 8388607 by VALUE, regardless of which has the larger stored exponent"
    );
    assert!(b > a);
}

#[test]
fn ord_handles_astronomically_different_magnitudes() {
    let huge = pow2(1000);
    let big = three_pow_64(); // ~2^101, mantissa ~102 bits, exponent 0
    assert!(
        huge > big,
        "2^1000 must compare greater than a ~2^101 value with a much larger stored mantissa"
    );
}

#[test]
fn ord_across_signs_and_around_zero() {
    let neg_huge = pow2(50).negate();
    let neg_tiny = pow2(-50).negate();
    let pos_tiny = pow2(-50);
    let pos_huge = pow2(50);

    assert!(
        neg_huge < neg_tiny,
        "more-negative huge value must sort before less-negative tiny value"
    );
    assert!(neg_tiny < Dyadic::ZERO);
    assert!(Dyadic::ZERO < pos_tiny);
    assert!(pos_tiny < pos_huge);
    assert!(neg_huge < pos_tiny);
    assert!(neg_tiny < pos_huge);
}

#[test]
fn ord_treats_zero_built_any_way_as_equal_in_comparisons() {
    let z1 = Dyadic::ZERO;
    let z2 = d(-0.0);
    assert_eq!(z1.cmp(&z2), std::cmp::Ordering::Equal);
    assert_eq!(z1.cmp(&z2), core::cmp::Ordering::Equal);
}

#[test]
fn partial_cmp_always_agrees_with_cmp() {
    let values = [
        pow2(5),
        pow2(-5),
        Dyadic::ZERO,
        pow2(5).negate(),
        d(8_388_607.0),
        three_pow_64(),
    ];
    for &a in &values {
        for &b in &values {
            assert_eq!(a.partial_cmp(&b), Some(a.cmp(&b)));
        }
    }
}

#[test]
fn sorting_a_scrambled_set_recovers_the_known_value_order() {
    let neg_huge = pow2(50).negate();
    let neg_big = pow2(10).negate();
    let neg_small = pow2(-10).negate();
    let neg_tiny = pow2(-100).negate();
    let zero = Dyadic::ZERO;
    let pos_tiny = pow2(-100);
    let pos_small = pow2(-10);
    let pos_mid = d(8_388_607.0);
    let pos_big = pow2(30);
    let pos_v = three_pow_64();
    let pos_huge = pow2(1000);

    // Independently-known ascending order by real value.
    let expected: Vec<Dyadic> = vec![
        neg_huge, neg_big, neg_small, neg_tiny, zero, pos_tiny, pos_small, pos_mid, pos_big, pos_v,
        pos_huge,
    ];

    // Feed sort() a deliberately scrambled input built from the same values.
    let mut scrambled: Vec<Dyadic> = vec![
        pos_huge, neg_small, zero, pos_v, neg_huge, pos_tiny, pos_mid, neg_tiny, pos_big, neg_big,
        pos_small,
    ];
    scrambled.sort();

    assert_eq!(
        scrambled, expected,
        "Ord must recover the true value order from a scrambled input; a \
         lexicographic-on-fields bug would reorder some of these"
    );
}

// ============================================================================
// 13. Odd-scaled helper self-test
// ============================================================================
//
// A quick sanity check that the `odd_scaled` helper (used above for the
// subnormal-rounding tests) actually agrees with the equivalent `pow2`-based
// construction, so a bug in the helper itself can't quietly invalidate the
// tests that use it.

#[test]
fn odd_scaled_helper_agrees_with_direct_construction() {
    let (via_helper, oracle) = odd_scaled(&[3, 5], 10); // 15 * 2^10
    let direct = d(15.0).checked_mul(pow2(10)).unwrap();
    assert_eq!(via_helper, direct);
    assert_eq!(oracle, 15.0 * 2f64.powi(10));
}

/// The exponent is finite too, and repeatedly squaring walks it up
/// exponentially — `2^(2^k)` reaches `i32::MAX` in ~31 squarings. Normalizing
/// past that must DECLINE, exactly as the mantissa cap does, rather than
/// panicking in debug or wrapping to a wildly different value in release.
#[test]
fn exponent_overflow_declines_instead_of_wrapping() {
    let mut value = Dyadic::from_f32(2.0).expect("2.0 is dyadic");
    // Square until the exponent can no longer grow; every step must either
    // succeed or decline, and none may panic.
    let mut declined = false;
    for _ in 0..64 {
        match value.checked_mul(value) {
            Some(next) => value = next,
            None => {
                declined = true;
                break;
            }
        }
    }
    assert!(
        declined,
        "repeated squaring must eventually decline rather than wrap the exponent"
    );
    // And the surviving value is still a well-formed dyadic that compares and
    // converts without panicking.
    assert!(value.to_f32().is_finite() || value.to_f32().is_infinite());
    assert_eq!(value.cmp(&value), core::cmp::Ordering::Equal);
}

/// `normalize` legitimately accepts `exp == i32::MAX` — it only rejects an
/// ADD that would overflow — so `to_f32` must survive the extremes the type
/// itself permits. Every exponent step there (binade, target, shift, bias)
/// would leave `i32` for such a value; they are computed in `i64` and the
/// result saturates to ±∞ or zero.
#[test]
fn to_f32_survives_the_exponent_extremes_the_type_allows() {
    // Walk to the largest exponent reachable, then convert.
    let mut big = Dyadic::from_f32(2.0).expect("2.0");
    while let Some(next) = big.checked_mul(big) {
        big = next;
    }
    assert!(
        big.to_f32().is_infinite() && big.to_f32() > 0.0,
        "an enormous positive dyadic must convert to +inf, not panic or wrap"
    );
    assert!(
        big.negate().to_f32().is_infinite() && big.negate().to_f32() < 0.0,
        "and its negation to -inf"
    );

    // The same walk downward: repeated squaring of a tiny value underflows.
    let mut tiny = Dyadic::from_f32(0.5).expect("0.5");
    while let Some(next) = tiny.checked_mul(tiny) {
        tiny = next;
    }
    assert_eq!(
        tiny.to_f32(),
        0.0,
        "an enormously small positive dyadic must convert to +0.0"
    );
}
