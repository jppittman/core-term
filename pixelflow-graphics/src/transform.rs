//! Coordinate transformations for manifolds.
//!
//! Provides composable coordinate warping using the `At` combinator.

use pixelflow_core::ops::{Div, Sub};
use pixelflow_core::{At, Field, Manifold, X, Y, Z, W};

/// The standard 4D Field domain type.
type Field4 = (Field, Field, Field, Field);

// =============================================================================
// Scale transformation
// =============================================================================

/// Alias for the scaled manifold type.
pub type Scaled<M> = At<Div<X, f32>, Div<Y, f32>, Z, W, M>;

/// Uniform scaling of the manifold domain.
///
/// Effectively scales the object size by `factor`.
/// Internally, coordinates are divided by `factor`.
///
/// # Example
/// ```ignore
/// let circle = kernel!(|| (X * X + Y * Y).sqrt() - 1.0);
/// let big_circle = scale(circle, 2.0);  // radius 2 circle
/// ```
pub fn scale<M>(inner: M, factor: f32) -> Scaled<M>
where
    M: Manifold<Field4, Output = Field>,
{
    At {
        inner,
        x: X / factor,
        y: Y / factor,
        z: Z,
        w: W,
    }
}

// =============================================================================
// Translate transformation
// =============================================================================

/// Alias for the translated manifold type.
pub type Translated<M> = At<Sub<X, f32>, Sub<Y, f32>, Z, W, M>;

/// Translation of the manifold domain.
///
/// Shifts the object by `(dx, dy)`.
/// Internally, coordinates are subtracted by the offset.
///
/// # Example
/// ```ignore
/// let circle = kernel!(|| (X * X + Y * Y).sqrt() - 1.0);
/// let moved = translate(circle, 10.0, 5.0);  // circle centered at (10, 5)
/// ```
pub fn translate<M>(inner: M, dx: f32, dy: f32) -> Translated<M>
where
    M: Manifold<Field4, Output = Field>,
{
    At {
        inner,
        x: X - dx,
        y: Y - dy,
        z: Z,
        w: W,
    }
}

// =============================================================================
// Legacy types (for backwards compatibility)
// =============================================================================

/// Uniform scaling transformation struct (legacy).
///
/// Prefer using `scale()` function which returns a composable `At` combinator.
#[derive(Clone, Debug)]
pub struct Scale<M> {
    pub manifold: M,
    pub factor: f32,
}

/// Translation transformation struct (legacy).
///
/// Prefer using `translate()` function which returns a composable `At` combinator.
#[derive(Clone, Debug)]
pub struct Translate<M> {
    pub manifold: M,
    pub offset: [f32; 2],
}

// Manifold implementations for legacy types use At directly

impl<M> Manifold<Field4> for Scale<M>
where
    M: Manifold<Field4, Output = Field>,
{
    type Output = Field;

    #[inline(always)]
    fn eval(&self, p: Field4) -> Field {
        // Create At combinator with coordinate expressions
        let at = At {
            inner: &self.manifold,
            x: X / self.factor,
            y: Y / self.factor,
            z: Z,
            w: W,
        };
        at.eval(p)
    }
}

impl<M> Manifold<Field4> for Translate<M>
where
    M: Manifold<Field4, Output = Field>,
{
    type Output = Field;

    #[inline(always)]
    fn eval(&self, p: Field4) -> Field {
        // Create At combinator with coordinate expressions
        let at = At {
            inner: &self.manifold,
            x: X - self.offset[0],
            y: Y - self.offset[1],
            z: Z,
            w: W,
        };
        at.eval(p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pixelflow_core::{Field, ManifoldExt};

    fn assert_close(val: Field, expected: f32) {
        let diff = (val - Field::from(expected)).abs().eval(());
        let epsilon = Field::from(1e-5);
        if !diff.lt(epsilon).all() {
            panic!("Assertion failed: expected {}, got {:?}", expected, val);
        }
    }

    #[test]
    fn scale_eval_divides_coordinate() {
        let scaled = scale(X, 2.0);
        let zero = Field::from(0.0);
        let two = Field::from(2.0);

        // At x=2, scaled should give x/2 = 1
        let result = scaled.eval((two, zero, zero, zero));
        assert_close(result, 1.0);
    }

    #[test]
    fn translate_eval_subtracts_offset() {
        let translated = translate(X, 1.0, 0.0);
        let zero = Field::from(0.0);
        let two = Field::from(2.0);

        // At x=2, translated should give x-1 = 1
        let result = translated.eval((two, zero, zero, zero));
        assert_close(result, 1.0);
    }

    #[test]
    fn compose_scale_then_translate_applies_operations_in_order() {
        // Inner: X/2
        // Outer: translate(inner, 1.0, 0.0) -> inner at (x-1, y)
        // Result: (x-1)/2
        let scaled = scale(X, 2.0);
        let composed = translate(scaled, 1.0, 0.0);

        let zero = Field::from(0.0);
        let four = Field::from(4.0);

        // At x=4: (4-1)/2 = 1.5
        let result = composed.eval((four, zero, zero, zero));
        assert_close(result, 1.5);
    }

    #[test]
    fn scale_eval_respects_factor() {
        for factor in [0.5, 1.0, 2.0, 10.0] {
            let scaled = scale(X, factor);
            let zero = Field::from(0.0);
            let val = Field::from(10.0);

            // Expected: 10.0 / factor
            let result = scaled.eval((val, zero, zero, zero));
            assert_close(result, 10.0 / factor);
        }
    }

    #[test]
    fn scale_clone_preserves_factor() {
        let scaled = Scale {
            manifold: X,
            factor: 2.0,
        };
        let cloned = scaled.clone();
        assert_eq!(cloned.factor, 2.0);

        // Ensure clone works computationally too
        let zero = Field::from(0.0);
        let two = Field::from(2.0);
        let result = cloned.eval((two, zero, zero, zero));
        assert_close(result, 1.0);
    }

    #[test]
    fn translate_eval_respects_offset() {
        for offset in [[0.0, 0.0], [1.0, 1.0], [-5.0, 5.0], [100.0, -100.0]] {
            let translated = translate(X, offset[0], offset[1]);
            let zero = Field::from(0.0);
            let val = Field::from(10.0);

            // At x=10, expect 10 - offset[0]
            let result = translated.eval((val, zero, zero, zero));
            assert_close(result, 10.0 - offset[0]);
        }
    }

    #[test]
    fn legacy_struct_compose_scale_then_translate_applies_operations_in_order() {
        // Inner: X
        // Middle: Scale(X, 2.0) -> X/2
        // Outer: Translate(Scale, [1.0, 1.0]) -> inner evaluated at (x-1, y-1)
        // So: (x-1)/2
        let scaled = Scale {
            manifold: X,
            factor: 2.0,
        };
        let composed = Translate {
            manifold: scaled,
            offset: [1.0, 1.0],
        };

        let zero = Field::from(0.0);
        let four = Field::from(4.0);

        // At x=4: (4-1)/2 = 1.5
        let result = composed.eval((four, zero, zero, zero));
        assert_close(result, 1.5);
    }

    #[test]
    fn legacy_struct_compose_translate_then_scale_applies_operations_in_order() {
        // Inner: Y
        // Middle: Translate(Y, [1.0, 2.0]) -> Y evaluated at y-2
        // Outer: Scale(Translate, 0.5) -> inner evaluated at y/0.5 = 2y
        // So: (2y) - 2
        let translated = Translate {
            manifold: Y,
            offset: [1.0, 2.0],
        };
        let composed = Scale {
            manifold: translated,
            factor: 0.5,
        };

        let zero = Field::from(0.0);
        let three = Field::from(3.0); // y=3

        // At y=3: (2*3) - 2 = 6 - 2 = 4
        let result = composed.eval((zero, three, zero, zero));
        assert_close(result, 4.0);
    }
}
