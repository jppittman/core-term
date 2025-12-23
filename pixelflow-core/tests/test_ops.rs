use pixelflow_core::{ManifoldExt, X, Y};

// Helper to extract scalar value from Field
fn extract(f: pixelflow_core::Field) -> f32 {
    let mut buf = [0.0f32; pixelflow_core::PARALLELISM];
    pixelflow_core::materialize(&f, 0.0, 0.0, &mut buf);
    buf[0]
}

#[test]
fn test_ops_basic_arithmetic() {
    let a = X + 10.0;
    let b = Y * 2.0;

    // Test Add
    let sum = a + b;
    // (5 + 10) + (3 * 2) = 15 + 6 = 21
    let result = sum.eval(5.0, 3.0, 0.0, 0.0);
    assert_eq!(extract(result), 21.0);

    // Test Sub
    let diff = a - b;
    // (5 + 10) - (3 * 2) = 15 - 6 = 9
    let result = diff.eval(5.0, 3.0, 0.0, 0.0);
    assert_eq!(extract(result), 9.0);

    // Test Mul
    let prod = a * b;
    // 15 * 6 = 90
    let result = prod.eval(5.0, 3.0, 0.0, 0.0);
    assert_eq!(extract(result), 90.0);

    // Test Div
    let quot = a / b;
    // 15 / 6 = 2.5
    let result = quot.eval(5.0, 3.0, 0.0, 0.0);
    assert_eq!(extract(result), 2.5);
}

#[test]
fn test_ops_functions() {
    let val = X;

    // Sqrt
    let s = val.sqrt();
    let result = s.eval(16.0, 0.0, 0.0, 0.0);
    assert_eq!(extract(result), 4.0);

    // Abs
    let a = val.abs();
    let result = a.eval(-10.0, 0.0, 0.0, 0.0);
    assert_eq!(extract(result), 10.0);
}

#[test]
fn test_ops_min_max() {
    let a = X;
    let b = Y;

    // Max
    let m = a.max(b);
    let result = m.eval(10.0, 20.0, 0.0, 0.0);
    assert_eq!(extract(result), 20.0);

    // Min
    let m = a.min(b);
    let result = m.eval(10.0, 20.0, 0.0, 0.0);
    assert_eq!(extract(result), 10.0);
}

#[test]
fn test_ops_comparison_select() {
    let a = X;
    let b = Y;

    // Lt
    let lt = a.lt(b); // 10 < 20 -> true (mask)
    // Select: if lt then 100 else 200
    let sel = lt.select(100.0, 200.0);

    let result = sel.eval(10.0, 20.0, 0.0, 0.0);
    assert_eq!(extract(result), 100.0);

    // Gt
    let gt = a.gt(b); // 10 > 20 -> false
    let sel = gt.select(100.0, 200.0);

    let result = sel.eval(10.0, 20.0, 0.0, 0.0);
    assert_eq!(extract(result), 200.0);

    // Le
    let le = a.le(b); // 10 <= 10 -> true
    let sel = le.select(100.0, 200.0);
    let result = sel.eval(10.0, 10.0, 0.0, 0.0);
    assert_eq!(extract(result), 100.0);

    // Ge
    let ge = a.ge(b); // 10 >= 10 -> true
    let sel = ge.select(100.0, 200.0);
    let result = sel.eval(10.0, 10.0, 0.0, 0.0);
    assert_eq!(extract(result), 100.0);
}

#[test]
fn test_chained_comparisons() {
    // Test logic operations: (X > 0) & (X < 10)
    let range = X.gt(0.0) & X.lt(10.0);
    let sel = range.select(1.0, 0.0);

    // Inside range
    let result = sel.eval(5.0, 0.0, 0.0, 0.0);
    assert_eq!(extract(result), 1.0);

    // Outside range (low)
    let result = sel.eval(-1.0, 0.0, 0.0, 0.0);
    assert_eq!(extract(result), 0.0);

    // Outside range (high)
    let result = sel.eval(11.0, 0.0, 0.0, 0.0);
    assert_eq!(extract(result), 0.0);

    // Test OR: (X < 0) | (X > 10)
    let outside = X.lt(0.0) | X.gt(10.0);
    let sel = outside.select(1.0, 0.0);

    let result = sel.eval(11.0, 0.0, 0.0, 0.0);
    assert_eq!(extract(result), 1.0);
}
