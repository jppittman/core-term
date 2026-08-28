#[cfg(target_arch = "x86_64")]
#[cfg(test)]
mod tests {
    extern crate std;
    use pixelflow_core::backend::x86::{F32x4, Mask4, U32x4};
    use pixelflow_core::backend::{MaskOps, SimdOps, SimdU32Ops};
    use std::prelude::v1::*;

    #[test]
    fn sse2_arithmetic() {
        let a = F32x4::splat(2.0);
        let b = F32x4::splat(3.0);

        let sum = a + b;
        let mut out = [0.0; 4];
        sum.store(&mut out);
        assert_eq!(out, [5.0, 5.0, 5.0, 5.0]);

        let diff = b - a;
        diff.store(&mut out);
        assert_eq!(out, [1.0, 1.0, 1.0, 1.0]);

        let prod = a * b;
        prod.store(&mut out);
        assert_eq!(out, [6.0, 6.0, 6.0, 6.0]);

        let quot = b / a;
        quot.store(&mut out);
        assert_eq!(out, [1.5, 1.5, 1.5, 1.5]);
    }

    #[test]
    fn sse2_sequential() {
        let seq = F32x4::sequential(10.0);
        let mut out = [0.0; 4];
        seq.store(&mut out);
        assert_eq!(out, [10.0, 11.0, 12.0, 13.0]);
    }

    #[test]
    fn sse2_logic() {
        let a = F32x4::splat(1.0);
        let b = F32x4::splat(2.0);

        // cmp_lt returns native Mask4
        let lt = a.cmp_lt(b);
        assert!(lt.all());

        // Check select with native mask
        let t = F32x4::splat(10.0);
        let f = F32x4::splat(20.0);
        let sel = F32x4::simd_select(lt, t, f);
        let mut out = [0.0; 4];
        sel.store(&mut out);
        assert_eq!(out, [10.0, 10.0, 10.0, 10.0]);

        let gt = a.cmp_gt(b);
        assert!(!gt.any());
        let sel2 = F32x4::simd_select(gt, t, f);
        sel2.store(&mut out);
        assert_eq!(out, [20.0, 20.0, 20.0, 20.0]);
    }

    #[test]
    fn sse2_bitwise() {
        let a = F32x4::splat(1.0); // 1.0 is 0x3f800000
        let b = F32x4::splat(2.0); // 2.0 is 0x40000000
        let c = a & b;
        let mut out = [0.0; 4];
        c.store(&mut out);
        assert_eq!(out, [0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn sse2_math() {
        let a = F32x4::splat(4.0);
        let sqrt = a.simd_sqrt();
        let mut out = [0.0; 4];
        sqrt.store(&mut out);
        assert_eq!(out, [2.0, 2.0, 2.0, 2.0]);

        let b = F32x4::splat(-2.0);
        let abs = b.simd_abs();
        abs.store(&mut out);
        assert_eq!(out, [2.0, 2.0, 2.0, 2.0]);

        let min = a.simd_min(b);
        min.store(&mut out);
        assert_eq!(out, [-2.0, -2.0, -2.0, -2.0]);
    }

    #[test]
    fn sse2_mask_any_all() {
        // Test MaskOps methods directly on masks
        let zero = F32x4::splat(0.0);
        let zero_mask = zero.float_to_mask();
        assert!(!zero_mask.any());
        assert!(!zero_mask.all());

        let all_true = F32x4::splat(1.0).cmp_gt(F32x4::splat(0.0));
        assert!(all_true.any());
        assert!(all_true.all());

        // Mixed: first lane is false, rest are true (0 > 0 is false, 1 > 0, 2 > 0, 3 > 0 are true)
        let mixed = F32x4::sequential(0.0).cmp_gt(F32x4::splat(0.0));
        assert!(mixed.any());
        assert!(!mixed.all());
    }

    #[test]
    #[should_panic]
    fn sse2_store_panic() {
        let a = F32x4::default();
        let mut out = [0.0; 3]; // Too small
        a.store(&mut out);
    }

    #[test]
    fn sse2_reciprocal_math() {
        let a = F32x4::splat(4.0);
        let mut out = [0.0; 4];

        // Reciprocal
        let recip = a.recip();
        recip.store(&mut out);
        // recip is approximate, tolerance 1e-3
        for x in out.iter() {
            assert!(
                (x - 0.25).abs() < 1e-3,
                "recip(4.0) should be 0.25, got {}",
                x
            );
        }

        // Reciprocal Square Root
        let rsqrt = a.simd_rsqrt();
        rsqrt.store(&mut out);
        // rsqrt is approximate, tolerance 1e-3
        for x in out.iter() {
            assert!(
                (x - 0.5).abs() < 1e-3,
                "rsqrt(4.0) should be 0.5, got {}",
                x
            );
        }
    }

    // ── SimdOps provided methods ──────────────────────────────────────────
    //
    // Everything above exercises a `SimdOps` *required* method — the per-ISA
    // primitive. These exercise the *provided* methods, the uniform
    // expansions built on top of them, which had no direct coverage at all:
    // a "replace the whole function with `Default::default()`" mutant
    // survived for every one of them.

    fn lanes(v: F32x4) -> [f32; 4] {
        let mut out = [0.0; 4];
        v.store(&mut out);
        out
    }

    #[test]
    fn simd_exp_matches_the_scalar_exponential() {
        let got = lanes(F32x4::splat(3.0).exp());
        let want = 3.0f32.exp();
        for x in got {
            assert!((x - want).abs() < 1e-2, "exp(3.0) = {x}, want {want}");
        }
    }

    #[test]
    fn simd_ln_matches_the_scalar_natural_log() {
        let got = lanes(F32x4::splat(10.0).ln());
        let want = 10.0f32.ln();
        for x in got {
            assert!((x - want).abs() < 1e-2, "ln(10.0) = {x}, want {want}");
        }
    }

    #[test]
    fn simd_log10_matches_the_scalar_base_10_log() {
        let got = lanes(F32x4::splat(100.0).log10());
        let want = 100.0f32.log10();
        for x in got {
            assert!((x - want).abs() < 1e-2, "log10(100.0) = {x}, want {want}");
        }
    }

    #[test]
    fn simd_pow_matches_the_scalar_power_function() {
        // self=4.0, exp=0.5: log2(self) = 2, distinguishing `exp * log2(self)`
        // from `exp + log2(self)` and `exp / log2(self)`, which a self value
        // with log2(self) == 1 would not.
        let got = lanes(F32x4::splat(4.0).pow(F32x4::splat(0.5)));
        let want = 4.0f32.powf(0.5);
        for x in got {
            assert!((x - want).abs() < 1e-2, "pow(4.0, 0.5) = {x}, want {want}");
        }
    }

    #[test]
    fn simd_hypot_computes_the_euclidean_norm() {
        let got = lanes(F32x4::splat(3.0).hypot(F32x4::splat(4.0)));
        for x in got {
            assert!((x - 5.0).abs() < 1e-2, "hypot(3.0, 4.0) = {x}, want 5.0");
        }
    }

    #[test]
    fn simd_mul_rsqrt_divides_by_the_square_root() {
        let got = lanes(F32x4::splat(10.0).mul_rsqrt(F32x4::splat(4.0)));
        for x in got {
            assert!(
                (x - 5.0).abs() < 1e-2,
                "mul_rsqrt(10.0, 4.0) = {x}, want 5.0 (10.0 / sqrt(4.0))"
            );
        }
    }

    #[test]
    fn simd_ceil_rounds_toward_positive_infinity() {
        assert_eq!(lanes(F32x4::splat(1.2).ceil()), [2.0; 4]);
        assert_eq!(lanes(F32x4::splat(-1.2).ceil()), [-1.0; 4]);
        assert_eq!(lanes(F32x4::splat(3.0).ceil()), [3.0; 4]);
    }

    #[test]
    fn simd_round_rounds_to_the_nearest_integer() {
        assert_eq!(lanes(F32x4::splat(2.7).round()), [3.0; 4]);
        assert_eq!(lanes(F32x4::splat(2.2).round()), [2.0; 4]);
    }

    #[test]
    fn simd_fract_returns_the_value_past_the_decimal_point() {
        let got = lanes(F32x4::splat(2.75).fract());
        for x in got {
            assert!((x - 0.75).abs() < 1e-5, "fract(2.75) = {x}, want 0.75");
        }
    }

    #[test]
    fn simd_clamp_bounds_a_value_to_the_given_range() {
        assert_eq!(
            lanes(F32x4::splat(5.0).clamp(F32x4::splat(0.0), F32x4::splat(10.0))),
            [5.0; 4]
        );
        assert_eq!(
            lanes(F32x4::splat(-5.0).clamp(F32x4::splat(0.0), F32x4::splat(10.0))),
            [0.0; 4]
        );
        assert_eq!(
            lanes(F32x4::splat(15.0).clamp(F32x4::splat(0.0), F32x4::splat(10.0))),
            [10.0; 4]
        );
    }

    // ── Remaining `SimdOps`/`SimdU32Ops` required methods ──────────────────
    //
    // Covers the SSE2 lane types (`F32x4`/`Mask4`/`U32x4`) specifically.
    //
    // Two constraints on the inputs below, both load-bearing rather than
    // incidental: every input is chosen so the real result differs from what
    // an all-zero `Default::default()` stand-in for the whole function would
    // return, and the bitwise ops use raw non-float bit patterns rather than
    // float values — otherwise an accidental zero result (`1.0 & 2.0 == 0.0`)
    // coincides with that all-zero default and the assertion stops
    // distinguishing them.

    fn mask_bits(m: Mask4) -> [u32; 4] {
        let mut out = [0.0f32; 4];
        F32x4::mask_to_float(m).store(&mut out);
        out.map(f32::to_bits)
    }

    fn u32_lanes(v: U32x4) -> [u32; 4] {
        let mut out = [0u32; 4];
        v.store(&mut out);
        out
    }

    #[test]
    fn float_to_mask_should_reinterpret_a_nonzero_bit_pattern_as_a_true_mask() {
        // An all-zero input would make `float_to_mask`'s real reinterpret and
        // a `Default::default()` stand-in coincide (both all-zero) — use an
        // all-ones pattern so they diverge.
        let ones = F32x4::from_u32_bits(u32::MAX);
        let mask = ones.float_to_mask();
        assert!(mask.all());
        assert_eq!(mask_bits(mask), [u32::MAX; 4]);
    }

    #[test]
    fn cmp_le_ge_eq_and_ne_should_each_produce_a_distinct_comparison_mask() {
        let a = F32x4::sequential(0.0); // [0, 1, 2, 3]
        let b = F32x4::splat(2.0);
        const T: u32 = u32::MAX;

        assert_eq!(mask_bits(a.cmp_le(b)), [T, T, T, 0], "0,1,2 <= 2; 3 is not");
        assert_eq!(
            mask_bits(a.cmp_ge(b)),
            [0, 0, T, T],
            "2,3 >= 2; 0,1 are not"
        );
        assert_eq!(mask_bits(a.cmp_eq(b)), [0, 0, T, 0], "only lane 2 equals 2");
        assert_eq!(
            mask_bits(a.cmp_ne(b)),
            [T, T, 0, T],
            "every lane but 2 differs"
        );
    }

    #[test]
    fn from_slice_should_load_four_consecutive_values_starting_at_the_given_offset() {
        let data = [7.0f32, 8.0, 9.0, 10.0, 11.0];
        assert_eq!(lanes(F32x4::from_slice(&data[1..])), [8.0, 9.0, 10.0, 11.0]);
    }

    #[test]
    fn f32x4_gather_should_read_the_slice_element_at_each_lanes_index() {
        let data = [10.0f32, 20.0, 30.0, 40.0, 50.0];

        let in_range = F32x4::from_slice(&[0.0, 2.0, 3.0, 4.0]);
        assert_eq!(
            lanes(F32x4::gather(&data, in_range)),
            [10.0, 30.0, 40.0, 50.0]
        );
    }

    #[test]
    fn f32x4_gather_should_clamp_an_out_of_range_index_to_the_last_element() {
        // Scoped to `F32x4` deliberately, and NOT a `SimdOps::gather`
        // guarantee: the clamp is a property of the scalar-loop
        // implementations (SSE2 here, NEON on aarch64), which index with
        // `(idx as isize).clamp(0, len - 1)`. `F32x8` and `F32x16` issue
        // `_mm256_i32gather_ps`/`_mm512_i32gather_ps` with no bounds
        // treatment at all and instead document a precondition that the
        // caller has already clamped. So an assertion phrased as "gather
        // clamps" would claim, from a baseline-ISA test run, a property the
        // wider builds do not provide.
        let data = [10.0f32, 20.0, 30.0, 40.0, 50.0];
        let past_end = F32x4::splat(10.0);
        assert_eq!(lanes(F32x4::gather(&data, past_end)), [50.0; 4]);
    }

    #[test]
    fn mul_add_should_compute_self_times_b_plus_c() {
        // self=2, b=3, c=4: `-`/`*` for `+`, and `+`/`/` for `*`, all disagree
        // with the correct 10.0 at these operands.
        let got = lanes(F32x4::splat(2.0).mul_add(F32x4::splat(3.0), F32x4::splat(4.0)));
        assert_eq!(got, [10.0; 4]);
    }

    #[test]
    fn add_masked_should_add_val_only_where_the_mask_is_true() {
        let base = F32x4::splat(1.0);
        let val = F32x4::splat(100.0);
        let mask = F32x4::sequential(0.0).cmp_gt(F32x4::splat(1.5)); // [F, F, T, T]
        assert_eq!(lanes(base.add_masked(val, mask)), [1.0, 1.0, 101.0, 101.0]);
    }

    #[test]
    fn from_u32_bits_should_reinterpret_the_integer_as_an_ieee754_bit_pattern() {
        assert_eq!(lanes(F32x4::from_u32_bits(0x3F80_0000)), [1.0; 4]);
        assert_eq!(lanes(F32x4::from_u32_bits(0x4000_0000)), [2.0; 4]);
    }

    #[test]
    fn shr_u32_should_shift_the_raw_bit_pattern_right_by_n_bits() {
        let v = F32x4::from_u32_bits(0x8000_0000);
        let mut out = [0.0f32; 4];
        v.shr_u32(1).store(&mut out);
        assert_eq!(out.map(f32::to_bits), [0x4000_0000; 4]);
    }

    #[test]
    fn i32_to_f32_should_numerically_convert_the_lanes_int32_value() {
        assert_eq!(lanes(F32x4::from_u32_bits(5).i32_to_f32()), [5.0; 4]);
        assert_eq!(
            lanes(F32x4::from_u32_bits(u32::MAX /* i32 -1 */).i32_to_f32()),
            [-1.0; 4]
        );
    }

    #[test]
    fn f32x4_bitwise_operators_should_combine_raw_bit_patterns() {
        let a = F32x4::from_u32_bits(0b1100);
        let b = F32x4::from_u32_bits(0b1010);
        let bits = |v: F32x4| -> [u32; 4] {
            let mut out = [0.0f32; 4];
            v.store(&mut out);
            out.map(f32::to_bits)
        };

        assert_eq!(bits(a & b), [0b1000; 4]);
        assert_eq!(bits(a | b), [0b1110; 4]);
        assert_eq!(bits(!a), [!0b1100u32; 4]);
    }

    #[test]
    fn mask4_bitwise_operators_should_combine_lane_truth_values_bit_exactly() {
        let mask_a = F32x4::sequential(0.0).cmp_gt(F32x4::splat(1.5)); // [F, F, T, T]
        let mask_b = F32x4::sequential(0.0).cmp_lt(F32x4::splat(2.5)); // [T, T, T, F]
        const T: u32 = u32::MAX;

        assert_eq!(mask_bits(mask_a & mask_b), [0, 0, T, 0]);
        assert_eq!(mask_bits(mask_a | mask_b), [T, T, T, T]);
        assert_eq!(mask_bits(!mask_a), [T, T, 0, 0]);
    }

    #[test]
    fn debug_formatting_should_reflect_the_stored_lane_values() {
        let v = F32x4::from_slice(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(std::format!("{v:?}"), "F32x4([1.0, 2.0, 3.0, 4.0])");

        let u = U32x4::splat(7);
        assert_eq!(std::format!("{u:?}"), "U32x4([7, 7, 7, 7])");

        let all_true = F32x4::splat(1.0).cmp_gt(F32x4::splat(0.0));
        assert_eq!(std::format!("{all_true:?}"), "Mask4(1111)");
    }

    #[test]
    fn u32x4_splat_and_store_should_round_trip_every_lane() {
        assert_eq!(u32_lanes(U32x4::splat(0xDEAD_BEEF)), [0xDEAD_BEEF; 4]);
    }

    #[test]
    fn u32x4_bitwise_operators_should_combine_lanes() {
        let a = U32x4::splat(0b1100);
        let b = U32x4::splat(0b1010);

        assert_eq!(u32_lanes(a & b), [0b1000; 4]);
        assert_eq!(u32_lanes(a | b), [0b1110; 4]);
        assert_eq!(u32_lanes(!a), [!0b1100u32; 4]);
    }

    #[test]
    fn u32x4_shift_operators_should_shift_every_lane() {
        let v = U32x4::splat(0b1000);
        assert_eq!(u32_lanes(v << 2), [0b10_0000; 4]);
        assert_eq!(u32_lanes(v >> 2), [0b10; 4]);
    }

    #[test]
    fn pack_rgba_should_clamp_each_channel_to_0_1_and_pack_it_into_a_byte() {
        // Both clamps have to be load-bearing, so each direction gets an
        // out-of-range channel. B at exactly 0.0 would need no clamping at
        // all, leaving the lower-bound `max(_, 0)` free to be deleted.
        let r = F32x4::splat(1.0); // at the upper edge -> 255
        let g = F32x4::splat(0.5); // in range -> 127 (truncated, not rounded)
        let b = F32x4::splat(-1.5); // below range -> clamped to 0
        let a = F32x4::splat(2.0); // above range -> clamped to 255

        let packed = u32_lanes(U32x4::pack_rgba(r, g, b, a));
        for p in packed {
            assert_eq!(p & 0xFF, 255, "R channel");
            assert_eq!((p >> 8) & 0xFF, 127, "G channel");
            assert_eq!((p >> 16) & 0xFF, 0, "B channel");
            assert_eq!((p >> 24) & 0xFF, 255, "A channel (clamped)");
        }
    }
}
