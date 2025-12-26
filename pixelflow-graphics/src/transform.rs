//! Coordinate transformations for manifolds.

use pixelflow_core::{Computational, Manifold};

/// Uniform scaling of the manifold domain.
///
/// Effectively scales the object size by `factor`.
/// Internally, coordinates are divided by `factor`.
#[derive(Clone, Debug)]
pub struct Scale<M> {
    pub manifold: M,
    pub factor: f32,
}

impl<M, I> Manifold<I> for Scale<M>
where
    M: Manifold<I>,
    I: Computational,
{
    type Output = M::Output;

    fn eval_raw(&self, x: I, y: I, z: I, w: I) -> Self::Output {
        let s = I::from_f32(self.factor);
        self.manifold.eval_raw(x / s, y / s, z, w)
    }
}

/// Translation of the manifold domain.
///
/// Shifts the object by `offset` (dx, dy).
/// Internally, coordinates are subtracted by `offset`.
#[derive(Clone, Debug)]
pub struct Translate<M> {
    pub manifold: M,
    pub offset: [f32; 2],
}

impl<M, I> Manifold<I> for Translate<M>
where
    M: Manifold<I>,
    I: Computational,
{
    type Output = M::Output;

    fn eval_raw(&self, x: I, y: I, z: I, w: I) -> Self::Output {
        let dx = I::from_f32(self.offset[0]);
        let dy = I::from_f32(self.offset[1]);
        self.manifold.eval_raw(x - dx, y - dy, z, w)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pixelflow_core::{Field, X, Y};

    #[test]
    fn scale_creation_and_eval() {
        let scaled = Scale {
            manifold: X,
            factor: 2.0,
        };

        let zero = Field::from(0.0);
        let one = Field::from(1.0);

        // Verify it evaluates without panic
        let _ = scaled.eval_raw(one, zero, zero, zero);
    }

    #[test]
    fn scale_with_various_factors() {
        for factor in [0.5, 1.0, 2.0, 10.0] {
            let scaled = Scale {
                manifold: X,
                factor,
            };
            let zero = Field::from(0.0);
            let one = Field::from(1.0);
            let _ = scaled.eval_raw(one, one, zero, zero);
        }
    }

    #[test]
    fn scale_is_clone() {
        let scaled = Scale {
            manifold: X,
            factor: 2.0,
        };
        let cloned = scaled.clone();
        assert_eq!(cloned.factor, 2.0);
    }

    #[test]
    fn scale_is_debug() {
        let scaled = Scale {
            manifold: X,
            factor: 2.0,
        };
        let debug_str = format!("{:?}", scaled);
        assert!(debug_str.contains("Scale"));
    }

    #[test]
    fn translate_creation_and_eval() {
        let translated = Translate {
            manifold: X,
            offset: [1.0, 2.0],
        };

        let zero = Field::from(0.0);
        let one = Field::from(1.0);

        // Verify it evaluates without panic
        let _ = translated.eval_raw(one, one, zero, zero);
    }

    #[test]
    fn translate_with_various_offsets() {
        for offset in [[0.0, 0.0], [1.0, 1.0], [-5.0, 5.0], [100.0, -100.0]] {
            let translated = Translate {
                manifold: X,
                offset,
            };
            let zero = Field::from(0.0);
            let one = Field::from(1.0);
            let _ = translated.eval_raw(one, one, zero, zero);
        }
    }

    #[test]
    fn translate_is_clone() {
        let translated = Translate {
            manifold: X,
            offset: [1.0, 2.0],
        };
        let cloned = translated.clone();
        assert_eq!(cloned.offset, [1.0, 2.0]);
    }

    #[test]
    fn translate_is_debug() {
        let translated = Translate {
            manifold: X,
            offset: [1.0, 2.0],
        };
        let debug_str = format!("{:?}", translated);
        assert!(debug_str.contains("Translate"));
    }

    #[test]
    fn scale_and_translate_compose() {
        // First scale by 2, then translate by (1, 1)
        let scaled = Scale {
            manifold: X,
            factor: 2.0,
        };
        let composed = Translate {
            manifold: scaled,
            offset: [1.0, 1.0],
        };

        let zero = Field::from(0.0);
        let one = Field::from(1.0);

        // Verify composition evaluates
        let _ = composed.eval_raw(one, one, zero, zero);
    }

    #[test]
    fn translate_and_scale_compose() {
        // First translate, then scale
        let translated = Translate {
            manifold: Y,
            offset: [1.0, 2.0],
        };
        let composed = Scale {
            manifold: translated,
            factor: 0.5,
        };

        let zero = Field::from(0.0);
        let one = Field::from(1.0);

        // Verify composition evaluates
        let _ = composed.eval_raw(one, one, zero, zero);
    }

    #[test]
    fn scale_with_z_and_w_coordinates() {
        let scaled = Scale {
            manifold: X,
            factor: 2.0,
        };

        let one = Field::from(1.0);
        let five = Field::from(5.0);
        let ten = Field::from(10.0);

        // Z and W coordinates passed through
        let _ = scaled.eval_raw(one, one, five, ten);
    }

    #[test]
    fn translate_with_z_and_w_coordinates() {
        let translated = Translate {
            manifold: X,
            offset: [1.0, 1.0],
        };

        let one = Field::from(1.0);
        let five = Field::from(5.0);
        let ten = Field::from(10.0);

        // Z and W coordinates passed through
        let _ = translated.eval_raw(one, one, five, ten);
    }

    #[test]
    fn scale_implements_manifold() {
        fn assert_manifold<T: Manifold<Field, Output = Field>>(_: &T) {}

        let scaled = Scale {
            manifold: X,
            factor: 2.0,
        };
        assert_manifold(&scaled);
    }

    #[test]
    fn translate_implements_manifold() {
        fn assert_manifold<T: Manifold<Field, Output = Field>>(_: &T) {}

        let translated = Translate {
            manifold: X,
            offset: [1.0, 2.0],
        };
        assert_manifold(&translated);
    }
}
