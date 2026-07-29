//! Coordinate transformations for manifolds.
//!
//! Provides composable coordinate warping using [`Kernel`].

use pixelflow_core::{At, Div, Field, Kernel, Manifold, Sub, W, X, Y, Z};

/// The standard 4D Field domain type.
type Field4 = (Field, Field, Field, Field);

/// Alias for the scaled manifold type.
pub type Scaled<M> = At<Div<X, f32>, Div<Y, f32>, Z, W, M>;

/// Alias for the translated manifold type.
pub type Translated<M> = At<Sub<X, f32>, Sub<Y, f32>, Z, W, M>;

/// Uniform scaling of a [`Kernel`] domain by `factor`.
#[inline]
#[must_use]
pub fn scale_kernel(k: &Kernel, factor: f32) -> Kernel {
    k.scale(factor)
}

/// Translation of a [`Kernel`] domain by `(dx, dy)`.
#[inline]
#[must_use]
pub fn translate_kernel(k: &Kernel, dx: f32, dy: f32) -> Kernel {
    k.translate(dx, dy)
}

/// Uniform scaling of a combinator manifold domain.
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

/// Translation of a combinator manifold domain.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_scale_and_translate_work() {
        let circle = Kernel::x()
            .mul(&Kernel::x())
            .add(&Kernel::y().mul(&Kernel::y()))
            .sqrt()
            .sub(&Kernel::constant(1.0));
        let scaled = scale_kernel(&circle, 2.0);
        let moved = translate_kernel(&scaled, 10.0, 5.0);
        let _ = moved;
    }
}
