#!/usr/bin/env python3
"""
Fit polynomial to log2 using least squares on Chebyshev nodes.
No external dependencies needed.
"""

import math

def log2(x):
    return math.log2(x)

def solve_linear_system_4x4(A, b):
    """Solve 5x5 linear system Ax = b using Gaussian elimination."""
    n = len(b)
    # Augmented matrix
    M = [A[i][:] + [b[i]] for i in range(n)]

    # Forward elimination
    for i in range(n):
        # Find pivot
        max_row = i
        for k in range(i+1, n):
            if abs(M[k][i]) > abs(M[max_row][i]):
                max_row = k
        M[i], M[max_row] = M[max_row], M[i]

        # Make all rows below this one 0 in current column
        for k in range(i+1, n):
            c = M[k][i] / M[i][i]
            for j in range(i, n+1):
                if i == j:
                    M[k][j] = 0
                else:
                    M[k][j] -= c * M[i][j]

    # Back substitution
    x = [0.0] * n
    for i in range(n-1, -1, -1):
        x[i] = M[i][n]
        for j in range(i+1, n):
            x[i] -= M[i][j] * x[j]
        x[i] /= M[i][i]

    return x

def fit_polynomial(x_data, y_data, degree):
    """Fit polynomial of given degree using least squares."""
    n = len(x_data)
    m = degree + 1

    # Build normal equations: A^T A c = A^T b
    ATA = [[0.0] * m for _ in range(m)]
    ATb = [0.0] * m

    for i in range(n):
        x = x_data[i]
        y = y_data[i]
        powers = [x ** k for k in range(m)]

        for j in range(m):
            ATb[j] += powers[j] * y
            for k in range(m):
                ATA[j][k] += powers[j] * powers[k]

    coeffs = solve_linear_system_4x4(ATA, ATb)
    return coeffs

def horner_eval(coeffs, x):
    """Evaluate using Horner's method."""
    result = coeffs[-1]
    for c in reversed(coeffs[:-1]):
        result = result * x + c
    return result

def test_coefficients(coeffs, a, b, n_samples=10000):
    """Test polynomial coefficients."""
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
    print("Log2 Polynomial Fitting using Least Squares")
    print("=" * 80)
    print()

    # Fit on different ranges
    ranges = [
        (math.sqrt(2)/2, math.sqrt(2), "[√2/2, √2] (centered at 1)"),
        (1.0, 2.0, "[1, 2) (traditional)"),
        (0.75, 1.5, "[0.75, 1.5] (simple bounds)"),
    ]

    for a, b, description in ranges:
        print(f"Range: {description}")
        print(f"  [{a:.10f}, {b:.10f}]")
        print()

        # Generate Chebyshev nodes for better approximation
        n_samples = 100
        x_data = []
        y_data = []

        for i in range(n_samples):
            # Chebyshev nodes mapped to [a, b]
            theta = math.pi * (2*i + 1) / (2 * n_samples)
            # Map from [-1, 1] to [a, b]
            t = math.cos(theta)
            x = ((b - a) * t + (b + a)) / 2
            x_data.append(x)
            y_data.append(log2(x))

        # Fit degree 4 polynomial
        coeffs = fit_polynomial(x_data, y_data, 4)

        print("Coefficients (c0 + c1*x + c2*x^2 + c3*x^3 + c4*x^4):")
        for i, c in enumerate(coeffs):
            print(f"  c{i} = {c:+.10f}")

        # Test accuracy
        max_err, avg_err, rms_err, worst_x = test_coefficients(coeffs, a, b if b < 2.0 else 1.9999)

        print(f"\nAccuracy:")
        print(f"  Max error:  {max_err:.6e}")
        print(f"  Avg error:  {avg_err:.6e}")
        print(f"  RMS error:  {rms_err:.6e}")
        print(f"  Worst at x: {worst_x:.6f}")

        print(f"\nRust code:")
        for i, c in enumerate(coeffs):
            print(f"  let c{i} = _mm_set1_ps({c:.10f});")

        print(f"\nHorner evaluation:")
        print(f"  let poly = c4 * f + c3;")
        print(f"  let poly = poly * f + c2;")
        print(f"  let poly = poly * f + c1;")
        print(f"  let poly = poly * f + c0;")

        print("\n" + "=" * 80 + "\n")

if __name__ == '__main__':
    main()
