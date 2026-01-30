#!/usr/bin/env python3
"""
Compute optimal polynomial coefficients for log2 approximation.

Uses Chebyshev approximation on the range [sqrt(2)/2, sqrt(2)] centered at 1.
This provides better accuracy than [1, 2) because the function is more symmetric.
"""

import numpy as np
from numpy.polynomial import chebyshev as C
import matplotlib.pyplot as plt

def log2(x):
    """Reference log2 function."""
    return np.log2(x)

def compute_chebyshev_coefficients(degree, a, b):
    """
    Compute Chebyshev polynomial coefficients for log2 on [a, b].

    Args:
        degree: Polynomial degree
        a, b: Interval [a, b]

    Returns:
        Coefficients in standard polynomial form (c0 + c1*x + c2*x^2 + ...)
    """
    # Transform to [-1, 1] for Chebyshev
    def transform_to_cheb(x):
        return 2 * (x - a) / (b - a) - 1

    def transform_from_cheb(t):
        return (t + 1) * (b - a) / 2 + a

    # Sample points: Chebyshev nodes for best interpolation
    n_samples = degree + 1
    cheb_nodes = np.cos(np.pi * (2 * np.arange(n_samples) + 1) / (2 * n_samples))
    x_samples = transform_from_cheb(cheb_nodes)
    y_samples = log2(x_samples)

    # Fit Chebyshev series
    t_samples = transform_to_cheb(x_samples)
    cheb_coeffs = C.chebfit(t_samples, y_samples, degree)

    # Convert to standard polynomial form
    poly_coeffs = C.cheb2poly(cheb_coeffs)

    return poly_coeffs

def evaluate_polynomial(coeffs, x):
    """Evaluate polynomial using Horner's method."""
    result = coeffs[-1]
    for c in reversed(coeffs[:-1]):
        result = result * x + c
    return result

def test_approximation(coeffs, a, b, n_test=10000):
    """Test polynomial approximation accuracy."""
    x_test = np.linspace(a, b, n_test)
    y_true = log2(x_test)
    y_approx = evaluate_polynomial(coeffs, x_test)

    errors = np.abs(y_approx - y_true)
    max_error = np.max(errors)
    avg_error = np.mean(errors)
    rms_error = np.sqrt(np.mean(errors**2))

    max_idx = np.argmax(errors)
    worst_x = x_test[max_idx]

    return {
        'max_error': max_error,
        'avg_error': avg_error,
        'rms_error': rms_error,
        'worst_x': worst_x,
        'errors': errors,
        'x_test': x_test
    }

def main():
    # Test different ranges and degrees

    print("=" * 80)
    print("Log2 Polynomial Approximation Coefficient Calculator")
    print("=" * 80)
    print()

    # Range 1: [sqrt(2)/2, sqrt(2)] - centered at 1
    sqrt2 = np.sqrt(2)
    a1, b1 = sqrt2 / 2, sqrt2

    # Range 2: Traditional [1, 2)
    a2, b2 = 1.0, 2.0

    for degree in [4, 5, 6]:
        print(f"\n{'='*80}")
        print(f"Degree {degree} Polynomial")
        print(f"{'='*80}\n")

        # Centered range [sqrt(2)/2, sqrt(2)]
        print(f"Range: [{a1:.6f}, {b1:.6f}] (centered at 1)")
        coeffs1 = compute_chebyshev_coefficients(degree, a1, b1)
        results1 = test_approximation(coeffs1, a1, b1)

        print(f"\nCoefficients (c0 + c1*x + c2*x^2 + ...):")
        for i, c in enumerate(coeffs1):
            print(f"  c{i} = {c:+.10f}")

        print(f"\nAccuracy on [{a1:.6f}, {b1:.6f}]:")
        print(f"  Max error:  {results1['max_error']:.2e}")
        print(f"  Avg error:  {results1['avg_error']:.2e}")
        print(f"  RMS error:  {results1['rms_error']:.2e}")
        print(f"  Worst at x: {results1['worst_x']:.6f}")

        # Traditional range [1, 2)
        print(f"\n\nRange: [{a2:.6f}, {b2:.6f}] (traditional)")
        coeffs2 = compute_chebyshev_coefficients(degree, a2, b2)
        results2 = test_approximation(coeffs2, a2, b2 - 0.0001)  # Avoid exactly 2.0

        print(f"\nCoefficients (c0 + c1*x + c2*x^2 + ...):")
        for i, c in enumerate(coeffs2):
            print(f"  c{i} = {c:+.10f}")

        print(f"\nAccuracy on [{a2:.6f}, {b2 - 0.0001:.6f}]:")
        print(f"  Max error:  {results2['max_error']:.2e}")
        print(f"  Avg error:  {results2['avg_error']:.2e}")
        print(f"  RMS error:  {results2['rms_error']:.2e}")
        print(f"  Worst at x: {results2['worst_x']:.6f}")

        print(f"\n{'Centered range is better!' if results1['max_error'] < results2['max_error'] else 'Traditional range is better!'}")

    # Generate Horner form for degree 4 on centered range
    print("\n" + "="*80)
    print("RECOMMENDED: Degree 4 on [sqrt(2)/2, sqrt(2)] for implementation")
    print("="*80 + "\n")

    coeffs = compute_chebyshev_coefficients(4, a1, b1)
    results = test_approximation(coeffs, a1, b1)

    print("Coefficients in standard form:")
    for i, c in enumerate(coeffs):
        print(f"  c{i} = {c:.10f}")

    print(f"\nHorner's method evaluation:")
    print(f"  poly(f) = (((c4*f + c3)*f + c2)*f + c1)*f + c0")

    print(f"\nAs float literals for Rust:")
    print(f"  const C4: f32 = {coeffs[4]:.10f};")
    print(f"  const C3: f32 = {coeffs[3]:.10f};")
    print(f"  const C2: f32 = {coeffs[2]:.10f};")
    print(f"  const C1: f32 = {coeffs[1]:.10f};")
    print(f"  const C0: f32 = {coeffs[0]:.10f};")

    print(f"\nAccuracy:")
    print(f"  Max error:  {results['max_error']:.2e} (should be < 1e-6)")
    print(f"  Avg error:  {results['avg_error']:.2e}")
    print(f"  RMS error:  {results['rms_error']:.2e}")
    print(f"  Worst at x: {results['worst_x']:.6f}")

    # Plot error distribution
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(results['x_test'], results['errors'])
    plt.axhline(y=1e-6, color='r', linestyle='--', label='1e-6 threshold')
    plt.yscale('log')
    plt.xlabel('x')
    plt.ylabel('Absolute Error')
    plt.title(f'Log2 Approximation Error (degree 4, range [{a1:.3f}, {b1:.3f}])')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(2, 1, 2)
    y_true = log2(results['x_test'])
    y_approx = evaluate_polynomial(coeffs, results['x_test'])
    plt.plot(results['x_test'], y_true, label='True log2', linewidth=2)
    plt.plot(results['x_test'], y_approx, label='Polynomial approx', linestyle='--')
    plt.xlabel('x')
    plt.ylabel('log2(x)')
    plt.title('Log2 Function vs Polynomial Approximation')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig('/home/user/core-term/scripts/log2_approximation_analysis.png', dpi=150)
    print(f"\nPlot saved to: /home/user/core-term/scripts/log2_approximation_analysis.png")

if __name__ == '__main__':
    main()
