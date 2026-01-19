//! Coordinate transformations for manifolds.

use pixelflow_core::jet::Jet2;
use pixelflow_core::{Field, Manifold, ManifoldCompat, ManifoldExt};

/// The standard 4D Field domain type.
type Field4 = (Field, Field, Field, Field);
/// The 4D Jet2 domain type for autodifferentiation.
type Jet4 = (Jet2, Jet2, Jet2, Jet2);

/// Uniform scaling of the manifold domain.
///
/// Effectively scales the object size by `factor`.
/// Internally, coordinates are divided by `factor`.
#[derive(Clone, Debug)]
pub struct Scale<M> {
    pub manifold: M,
    pub factor: f32,
}

impl<M> Manifold<Field4> for Scale<M>
where
    M: ManifoldCompat<Field, Output = Field> + ManifoldExt,
{
    type Output = Field;

    fn eval(&self, p: Field4) -> Field {
        let (x, y, z, w) = p;
        let s = Field::from(self.factor);
        self.manifold.eval_at(x / s, y / s, z, w)
    }
}

impl<M> Manifold<Jet4> for Scale<M>
where
    M: ManifoldCompat<Jet2>,
{
    type Output = M::Output;

    fn eval(&self, p: Jet4) -> Self::Output {
        let (x, y, z, w) = p;
        let s = Jet2::constant(Field::from(self.factor));
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

impl<M> Manifold<Field4> for Translate<M>
where
    M: ManifoldCompat<Field, Output = Field> + ManifoldExt,
{
    type Output = Field;

    fn eval(&self, p: Field4) -> Field {
        let (x, y, z, w) = p;
        let dx = Field::from(self.offset[0]);
        let dy = Field::from(self.offset[1]);
        self.manifold.eval_at(x - dx, y - dy, z, w)
    }
}

impl<M> Manifold<Jet4> for Translate<M>
where
    M: ManifoldCompat<Jet2>,
{
    type Output = M::Output;

    fn eval(&self, p: Jet4) -> Self::Output {
        let (x, y, z, w) = p;
        let dx = Jet2::constant(Field::from(self.offset[0]));
        let dy = Jet2::constant(Field::from(self.offset[1]));
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
        fn assert_manifold<T: ManifoldCompat<Field, Output = Field>>(_: &T) {}

        let scaled = Scale {
            manifold: X,
            factor: 2.0,
        };
        assert_manifold(&scaled);
    }

    #[test]
    fn translate_implements_manifold() {
        fn assert_manifold<T: ManifoldCompat<Field, Output = Field>>(_: &T) {}

        let translated = Translate {
            manifold: X,
            offset: [1.0, 2.0],
        };
        assert_manifold(&translated);
    }
}
