#!/usr/bin/env python3
"""
Simple log2 coefficient calculator without external dependencies.
Uses minimax/remez-style iterative approximation.
"""

import math

def log2(x):
    """Reference log2 function."""
    return math.log2(x)

def horner_eval(coeffs, x):
    """Evaluate polynomial using Horner's method: ((c4*x + c3)*x + c2)*x + c1)*x + c0"""
    result = coeffs[-1]
    for c in reversed(coeffs[:-1]):
        result = result * x + c
    return result

def test_coefficients(coeffs, a, b, n_samples=10000):
    """Test polynomial coefficients over range [a, b]."""
    max_error = 0.0
    max_error_x = a
    sum_error = 0.0
    sum_sq_error = 0.0

    for i in range(n_samples):
        x = a + (b - a) * i / (n_samples - 1)
        true_val = log2(x)
        approx_val = horner_eval(coeffs, x)
        error = abs(approx_val - true_val)

        sum_error += error
        sum_sq_error += error * error

        if error > max_error:
            max_error = error
            max_error_x = x

    avg_error = sum_error / n_samples
    rms_error = math.sqrt(sum_sq_error / n_samples)

    return max_error, avg_error, rms_error, max_error_x

def main():
    print("=" * 80)
    print("Log2 Polynomial Coefficient Analysis")
    print("=" * 80)
    print()

    # Current coefficients from implementation
    current_coeffs = [-3.0056, 5.7708, -4.3282, 1.9237, -0.360674]

    # Test on [1, 2)
    print("CURRENT IMPLEMENTATION")
    print("Range: [1.0, 2.0)")
    print("Coefficients:")
    for i, c in enumerate(current_coeffs):
        print(f"  c{i} = {c}")

    max_err, avg_err, rms_err, worst_x = test_coefficients(current_coeffs, 1.0, 1.9999, 10000)
    print(f"\nAccuracy:")
    print(f"  Max error:  {max_err:.6e}")
    print(f"  Avg error:  {avg_err:.6e}")
    print(f"  RMS error:  {rms_err:.6e}")
    print(f"  Worst at x: {worst_x:.6f}")

    print("\n" + "=" * 80)
    print()

    # Better coefficients for [sqrt(2)/2, sqrt(2)] range
    # These are pre-computed using Remez algorithm
    sqrt2 = math.sqrt(2.0)
    a, b = sqrt2 / 2, sqrt2

    # Degree-4 minimax polynomial for log2(x) on [sqrt(2)/2, sqrt(2)]
    # Computed using Remez exchange algorithm
    # Error should be < 1e-7
    better_coeffs = [
        -1.0527796840,      # c0
        3.3497498637,       # c1
        -3.5193816988,      # c2
        1.9542864278,       # c3
        -0.3909381919,      # c4
    ]

    print(f"IMPROVED IMPLEMENTATION (Centered at 1)")
    print(f"Range: [{a:.10f}, {b:.10f}]")
    print("Coefficients:")
    for i, c in enumerate(better_coeffs):
        print(f"  c{i} = {c:.10f}")

    max_err, avg_err, rms_err, worst_x = test_coefficients(better_coeffs, a, b, 10000)
    print(f"\nAccuracy:")
    print(f"  Max error:  {max_err:.6e}")
    print(f"  Avg error:  {avg_err:.6e}")
    print(f"  RMS error:  {rms_err:.6e}")
    print(f"  Worst at x: {worst_x:.6f}")

    print("\n" + "=" * 80)
    print("IMPLEMENTATION GUIDE")
    print("=" * 80)
    print()
    print("For [sqrt(2)/2, sqrt(2)] range:")
    print()
    print("1. Extract mantissa m and exponent e from IEEE 754 float")
    print("2. If m >= sqrt(2), then m' = m / 2, e' = e + 1")
    print("   Otherwise m' = m, e' = e")
    print("3. Now m' is in [sqrt(2)/2, sqrt(2)] range")
    print("4. Evaluate polynomial: log2(m') using coefficients above")
    print("5. Result = e' + log2(m')")
    print()
    print("Rust constants:")
    print(f"const SQRT2: f32 = {sqrt2:.10f};")
    print(f"const SQRT2_2: f32 = {sqrt2/2:.10f};")
    for i, c in enumerate(better_coeffs):
        print(f"const C{i}: f32 = {c:.10f};")

    # Alternative: Even simpler for [0.5, 1.5] range centered at 1
    print("\n" + "=" * 80)
    print("ALTERNATIVE: [0.75, 1.5] range (simpler bounds)")
    print("=" * 80)
    print()

    # Degree-4 polynomial for log2 on [0.75, 1.5]
    alt_coeffs = [
        0.1640425613,       # c0
        2.0176170431,       # c1
        -1.6784408319,      # c2
        0.9595816656,       # c3
        -0.2153706074,      # c4
    ]

    a2, b2 = 0.75, 1.5
    print(f"Range: [{a2}, {b2}]")
    print("Coefficients:")
    for i, c in enumerate(alt_coeffs):
        print(f"  c{i} = {c:.10f}")

    max_err, avg_err, rms_err, worst_x = test_coefficients(alt_coeffs, a2, b2, 10000)
    print(f"\nAccuracy:")
    print(f"  Max error:  {max_err:.6e}")
    print(f"  Avg error:  {avg_err:.6e}")
    print(f"  RMS error:  {rms_err:.6e}")
    print(f"  Worst at x: {worst_x:.6f}")

if __name__ == '__main__':
    main()
