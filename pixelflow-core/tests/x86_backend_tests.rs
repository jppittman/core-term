#[cfg(target_arch = "x86_64")]
#[cfg(test)]
mod tests {
    extern crate std;
    use pixelflow_core::backend::x86::{F32x4, U32x4};
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

    // ── Remaining SimdOps required methods, Debug impls, and U32x4 ─────────
    //
    // These are the `SimdOps`/`SimdU32Ops` *required* per-ISA primitives (as
    // opposed to the *provided* expansions above) that a 2026-08-22 mutants
    // sweep found had no direct coverage at all under this crate's default
    // SSE2-baseline build.

    fn u32_lanes(v: U32x4) -> [u32; 4] {
        let mut out = [0u32; 4];
        v.store(&mut out);
        out
    }

    #[test]
    fn mask4_debug_format_shows_the_actual_lane_pattern() {
        // Lane 0 is false (0 > 0), lanes 1-3 are true — movemask reads
        // lane 3..0 MSB-first, so the mixed pattern prints as "1110".
        let mixed = F32x4::sequential(0.0).cmp_gt(F32x4::splat(0.0));
        assert_eq!(format!("{:?}", mixed), "Mask4(1110)");
    }

    #[test]
    fn f32x4_debug_format_shows_the_actual_lane_values() {
        assert_eq!(
            format!("{:?}", F32x4::splat(3.5)),
            "F32x4([3.5, 3.5, 3.5, 3.5])"
        );
    }

    #[test]
    fn u32x4_debug_format_shows_the_actual_lane_values() {
        assert_eq!(
            format!("{:?}", <U32x4 as SimdU32Ops>::splat(42)),
            "U32x4([42, 42, 42, 42])"
        );
    }

    #[test]
    fn simd_gather_clamps_an_out_of_range_index_to_the_last_valid_element() {
        // slice has 3 elements (indices 0..=2); an index far past the end
        // must clamp to len - 1, not len or len + 1 (either of which would
        // index out of bounds and panic).
        let slice = [10.0f32, 20.0, 30.0];
        let indices = F32x4::splat(100.0);
        let got = lanes(F32x4::gather(&slice, indices));
        assert_eq!(got, [30.0; 4]);
    }

    #[test]
    fn add_masked_adds_only_where_the_mask_is_true() {
        let base = F32x4::splat(5.0);
        let val = F32x4::splat(10.0);

        let all_true = F32x4::splat(1.0).cmp_gt(F32x4::splat(0.0));
        assert_eq!(lanes(base.add_masked(val, all_true)), [15.0; 4]);

        let all_false = F32x4::splat(0.0).cmp_gt(F32x4::splat(1.0));
        assert_eq!(lanes(base.add_masked(val, all_false)), [5.0; 4]);
    }

    #[test]
    fn from_u32_bits_reinterprets_the_bit_pattern_as_f32() {
        let got = lanes(F32x4::from_u32_bits(1.0f32.to_bits()));
        assert_eq!(got, [1.0; 4]);
    }

    #[test]
    fn shr_u32_shifts_the_reinterpreted_bit_pattern_right() {
        let v = F32x4::from_u32_bits(128);
        let got = lanes(v.shr_u32(3));
        for x in got {
            assert_eq!(x.to_bits(), 16, "128 >> 3 should be 16");
        }
    }

    #[test]
    fn i32_to_f32_converts_the_reinterpreted_integer_not_the_bit_pattern() {
        // from_u32_bits(5) reinterprets the integer 5 as raw bits; i32_to_f32
        // then converts that integer value to a float — 5.0, not the float
        // whose bit pattern happens to be 5 (a subnormal near zero).
        let got = lanes(F32x4::from_u32_bits(5).i32_to_f32());
        assert_eq!(got, [5.0; 4]);
    }

    #[test]
    fn f32x4_bitor_combines_the_reinterpreted_bit_patterns() {
        let a = F32x4::from_u32_bits(0b1010);
        let b = F32x4::from_u32_bits(0b0101);
        let got = lanes(a | b);
        for x in got {
            assert_eq!(x.to_bits(), 0b1111);
        }
    }

    #[test]
    fn f32x4_not_flips_every_bit_of_the_pattern() {
        let got = lanes(!F32x4::from_u32_bits(0));
        for x in got {
            assert_eq!(x.to_bits(), u32::MAX);
        }
    }

    #[test]
    fn u32x4_splat_then_store_round_trips_the_value() {
        assert_eq!(u32_lanes(<U32x4 as SimdU32Ops>::splat(7)), [7; 4]);
    }

    #[test]
    fn u32x4_bitand_masks_the_lanes() {
        let a = <U32x4 as SimdU32Ops>::splat(0b1010);
        let b = <U32x4 as SimdU32Ops>::splat(0b0110);
        assert_eq!(u32_lanes(a & b), [0b0010; 4]);
    }

    #[test]
    fn u32x4_bitor_combines_the_lanes() {
        let a = <U32x4 as SimdU32Ops>::splat(0b1010);
        let b = <U32x4 as SimdU32Ops>::splat(0b0101);
        assert_eq!(u32_lanes(a | b), [0b1111; 4]);
    }

    #[test]
    fn u32x4_not_flips_every_bit_of_every_lane() {
        assert_eq!(u32_lanes(!<U32x4 as SimdU32Ops>::splat(0)), [u32::MAX; 4]);
    }

    #[test]
    fn u32x4_shl_shifts_every_lane_left_by_the_given_count() {
        let v = <U32x4 as SimdU32Ops>::splat(1);
        assert_eq!(u32_lanes(v << 4), [16; 4]);
    }

    #[test]
    fn u32x4_shr_shifts_every_lane_right_by_the_given_count() {
        let v = <U32x4 as SimdU32Ops>::splat(16);
        assert_eq!(u32_lanes(v >> 4), [1; 4]);
    }

    #[test]
    fn pack_rgba_clamps_scales_and_packs_channels_into_one_u32_per_lane() {
        let r = F32x4::splat(1.0); // clamps to 1.0 -> 255
        let g = F32x4::splat(0.5); // -> 127 (truncated, not rounded)
        let b = F32x4::splat(0.25); // -> 63
        let a = F32x4::splat(0.0); // -> 0

        let packed = u32_lanes(U32x4::pack_rgba(r, g, b, a));
        let expected: u32 = 255 | (127 << 8) | (63 << 16); // A channel is 0
        assert_eq!(packed, [expected; 4]);
    }
}
