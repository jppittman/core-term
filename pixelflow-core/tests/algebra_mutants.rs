//! Mutant hunting tests for algebra operations.
//!
//! These tests target specific mutations (logic flips, boundary errors) that
//! standard functional tests might miss.
//!
//! # Directives
//!
//! 1. Kill the Mutant: Target specific code changes (e.g. `<` -> `<=`).
//! 2. Name the Crime: Naming convention `[type]_[op]_should_[outcome]`.
//! 3. Enforce Style: `expect` failures, no boolean asserts.

use pixelflow_core::Algebra;

// ============================================================================
// f32 Algebra
// ============================================================================

#[test]
fn f32_lt_should_return_false_when_operands_are_equal() {
    // Mutant: `<` becomes `<=`
    // If 2.0 < 2.0 becomes 2.0 <= 2.0 (true), this test fails.
    let a = 2.0f32;
    let b = 2.0f32;
    assert!(!a.lt(b), "Expected 2.0 < 2.0 to be false, but it was true (mutant: <=)");
}

#[test]
fn f32_gt_should_return_false_when_operands_are_equal() {
    // Mutant: `>` becomes `>=`
    // If 2.0 > 2.0 becomes 2.0 >= 2.0 (true), this test fails.
    let a = 2.0f32;
    let b = 2.0f32;
    assert!(!a.gt(b), "Expected 2.0 > 2.0 to be false, but it was true (mutant: >=)");
}

#[test]
fn f32_le_should_return_true_when_operands_are_equal() {
    // Mutant: `<=` becomes `<`
    // If 2.0 <= 2.0 becomes 2.0 < 2.0 (false), this test fails.
    let a = 2.0f32;
    let b = 2.0f32;
    assert!(a.le(b), "Expected 2.0 <= 2.0 to be true, but it was false (mutant: <)");
}

#[test]
fn f32_ge_should_return_true_when_operands_are_equal() {
    // Mutant: `>=` becomes `>`
    // If 2.0 >= 2.0 becomes 2.0 > 2.0 (false), this test fails.
    let a = 2.0f32;
    let b = 2.0f32;
    assert!(a.ge(b), "Expected 2.0 >= 2.0 to be true, but it was false (mutant: >)");
}

// ============================================================================
// u32 Algebra
// ============================================================================

#[test]
fn u32_add_should_wrap_on_overflow() {
    // Mutant: `wrapping_add` becomes `saturating_add`
    // wrapping: MAX + 1 = 0
    // saturating: MAX + 1 = MAX
    let a = u32::MAX;
    let b = 1;
    let res = a.add(b);
    assert_eq!(res, 0, "u32 addition should wrap (got {}, expected 0). Mutant: saturating_add", res);
}

#[test]
fn u32_sub_should_wrap_on_underflow() {
    // Mutant: `wrapping_sub` becomes `saturating_sub`
    // wrapping: 0 - 1 = MAX
    // saturating: 0 - 1 = 0
    let a = 0u32;
    let b = 1;
    let res = a.sub(b);
    assert_eq!(res, u32::MAX, "u32 subtraction should wrap (got {}, expected MAX). Mutant: saturating_sub", res);
}

#[test]
fn u32_mul_should_wrap_on_overflow() {
    // Mutant: `wrapping_mul` becomes `saturating_mul`
    // 2^31 * 2 = 2^32 = 0 (wrapped)
    // 2^31 * 2 = MAX (saturated)
    let a = 1u32 << 31;
    let b = 2;
    let res = a.mul(b);
    assert_eq!(res, 0, "u32 multiplication should wrap (got {}, expected 0). Mutant: saturating_mul", res);
}

// ============================================================================
// bool Algebra
// ============================================================================

#[test]
fn bool_add_should_be_logical_or() {
    // Mutant: `|` (OR) becomes `^` (XOR)
    // OR:  true + true = true
    // XOR: true + true = false
    assert!(true.add(true), "bool add should be OR (true+true=true). Mutant: XOR");
}

#[test]
fn bool_sub_should_be_and_not() {
    // Mutant: `& !rhs` (AND NOT) becomes `^` (XOR) or `&` (AND)

    // Test for XOR mutant:
    // AND NOT: false - true = false & !true = false
    // XOR:     false - true = false ^ true  = true
    assert!(!false.sub(true), "bool sub should be AND NOT (false-true=false). Mutant: XOR");

    // Test for AND mutant:
    // AND NOT: true - false = true & !false = true
    // AND:     true - false = true & false  = false
    assert!(true.sub(false), "bool sub should be AND NOT (true-false=true). Mutant: AND");
}

#[test]
fn bool_mul_should_be_logical_and() {
    // Mutant: `&` (AND) becomes `|` (OR)
    // AND: false * true = false
    // OR:  false * true = true
    assert!(!false.mul(true), "bool mul should be AND (false*true=false). Mutant: OR");
}
