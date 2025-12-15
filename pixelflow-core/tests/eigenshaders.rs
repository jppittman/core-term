use pixelflow_core::Batch;
use pixelflow_core::Manifold;
use pixelflow_core::Surface;
use pixelflow_core::backend::Backend;
use pixelflow_core::batch::NativeBackend;
use pixelflow_core::dsl::SurfaceExt;
use pixelflow_core::surfaces::Compute;
use std::fmt::Debug;

#[derive(Clone, Copy)]
struct Constant(Batch<f32>);

impl<C> Manifold<f32, C> for Constant
where
    C: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
{
    fn eval(&self, _x: Batch<C>, _y: Batch<C>, _z: Batch<C>, _w: Batch<C>) -> Batch<f32> {
        self.0
    }
}

#[test]
fn test_grade_surface() {
    let source = Compute::new(
        |_x: Batch<u32>, _y: Batch<u32>, _z: Batch<u32>, _w: Batch<u32>| {
            Batch::<f32>::splat(10.0f32)
        },
    )
    .shader();

    // Test basic constant slope/bias
    let graded = source.clone().grade(
        Constant(Batch::<f32>::splat(2.0)),
        Constant(Batch::<f32>::splat(5.0)),
    );
    let result = graded.eval_one(0, 0);
    assert_eq!(result, 25.0); // 10 * 2 + 5 = 25

    // Test with functional slope/bias
    // Slope = 0.5, Bias = 1.0 => 10 * 0.5 + 1.0 = 6.0
    let graded_fn = source.grade(
        Compute::new(
            |_x: Batch<u32>, _y: Batch<u32>, _z: Batch<u32>, _w: Batch<u32>| {
                Batch::<f32>::splat(0.5)
            },
        ),
        Compute::new(
            |_x: Batch<u32>, _y: Batch<u32>, _z: Batch<u32>, _w: Batch<u32>| {
                Batch::<f32>::splat(1.0)
            },
        ),
    );
    assert_eq!(graded_fn.eval_one(0, 0), 6.0);
}

#[test]
fn test_lerp_surface() {
    let a = Compute::new(
        |_x: Batch<u32>, _y: Batch<u32>, _z: Batch<u32>, _w: Batch<u32>| {
            Batch::<f32>::splat(0.0f32)
        },
    );
    let b = Compute::new(
        |_x: Batch<u32>, _y: Batch<u32>, _z: Batch<u32>, _w: Batch<u32>| {
            Batch::<f32>::splat(100.0f32)
        },
    );

    // t = 0.5
    let lerped = a.lerp(
        Compute::new(
            |_x: Batch<u32>, _y: Batch<u32>, _z: Batch<u32>, _w: Batch<u32>| {
                Batch::<f32>::splat(0.5f32)
            },
        ),
        b,
    );

    assert_eq!(lerped.eval_one(0, 0), 50.0);
}

#[test]
fn test_warp_surface() {
    // Source: f(x,y) = x
    // x is Batch<u32>, we need to convert to f32 for the result
    let source = Compute::new(
        |x: Batch<u32>, _y: Batch<u32>, _z: Batch<u32>, _w: Batch<u32>| {
            NativeBackend::u32_to_f32(x)
        },
    );

    // Warp: shift x by +10
    // new_x = x + 10
    // So evaluating at x=0 should sample source at x=10, returning 10.
    // Explicitly annotate closure types for Warp
    let warped = source.warp(
        |x: Batch<u32>, y: Batch<u32>, z: Batch<u32>, w: Batch<u32>| {
            (x + Batch::<u32>::splat(10), y, z, w)
        },
    );

    assert_eq!(warped.eval_one(0, 0), 10.0);
}
