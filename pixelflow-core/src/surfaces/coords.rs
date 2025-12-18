// pixelflow-core/src/surfaces/coords.rs
//! Coordinate projection surfaces.

use crate::batch::Batch;
use crate::traits::Manifold;
use core::fmt::Debug;

/// Projects the X coordinate: `F(x, y, z, w) = x`.
#[derive(Copy, Clone, Debug, Default)]
pub struct X;

/// Projects the Y coordinate: `F(x, y, z, w) = y`.
#[derive(Copy, Clone, Debug, Default)]
pub struct Y;

/// Projects the Z coordinate: `F(x, y, z, w) = z`.
#[derive(Copy, Clone, Debug, Default)]
pub struct Z;

/// Projects the W coordinate: `F(x, y, z, w) = w`.
#[derive(Copy, Clone, Debug, Default)]
pub struct W;

impl<C> Manifold<C, C> for X
where
    C: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
{
    #[inline(always)]
    fn eval(&self, x: Batch<C>, _y: Batch<C>, _z: Batch<C>, _w: Batch<C>) -> Batch<C> {
        x
    }
}

impl<C> Manifold<C, C> for Y
where
    C: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
{
    #[inline(always)]
    fn eval(&self, _x: Batch<C>, y: Batch<C>, _z: Batch<C>, _w: Batch<C>) -> Batch<C> {
        y
    }
}

impl<C> Manifold<C, C> for Z
where
    C: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
{
    #[inline(always)]
    fn eval(&self, _x: Batch<C>, _y: Batch<C>, z: Batch<C>, _w: Batch<C>) -> Batch<C> {
        z
    }
}

impl<C> Manifold<C, C> for W
where
    C: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
{
    #[inline(always)]
    fn eval(&self, _x: Batch<C>, _y: Batch<C>, _z: Batch<C>, w: Batch<C>) -> Batch<C> {
        w
    }
}
