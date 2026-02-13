#[allow(unused_imports)]
use pixelflow_core::{Field, Manifold, ManifoldExt};

#[derive(Clone, Copy)]
struct MyVec(Field, Field, Field, Field);

impl pixelflow_core::ops::Vector for MyVec {
    type Component = Field;
    fn get(&self, axis: pixelflow_core::variables::Axis) -> Field {
        match axis {
            pixelflow_core::variables::Axis::X => self.0,
            pixelflow_core::variables::Axis::Y => self.1,
            pixelflow_core::variables::Axis::Z => self.2,
            pixelflow_core::variables::Axis::W => self.3,
        }
    }
}

struct Wrap<M>(M);

impl<M> Manifold for Wrap<M>
where
    M: Manifold<Output = Field>,
{
    type Output = MyVec;
    fn eval(&self, p: (Field, Field, Field, Field)) -> MyVec {
        let val = self.0.eval(p);
        MyVec(val, val, val, val)
    }
}

#[test]
fn cheby_sin_should_be_accurate() {
    let epsilon = 1e-4;

    let test_cases = [
        (0.0f32, 0.0f32),
        (std::f32::consts::FRAC_PI_6, 0.5),
        (std::f32::consts::FRAC_PI_4, std::f32::consts::FRAC_1_SQRT_2),
        (std::f32::consts::FRAC_PI_3, 3.0f32.sqrt() / 2.0),
        (std::f32::consts::FRAC_PI_2, 1.0),
        (std::f32::consts::PI, 0.0),
        (-std::f32::consts::FRAC_PI_2, -1.0),
    ];

    for (x_val, expected) in test_cases {
        let val_expr = Field::from(x_val).sin();
        let m = Wrap(val_expr);

        let mut buf = [0.0f32; pixelflow_core::PARALLELISM * 4];
        pixelflow_core::materialize(&m, 0.0, 0.0, &mut buf);

        let val = buf[0];

        assert!(
            (val - expected).abs() < epsilon,
            "sin({}) should be {}, got {}",
            x_val,
            expected,
            val
        );
    }
}

#[test]
fn cheby_cos_should_be_accurate() {
    let epsilon = 1e-4;

    let test_cases = [
        (0.0f32, 1.0f32),
        (std::f32::consts::FRAC_PI_6, 3.0f32.sqrt() / 2.0),
        (std::f32::consts::FRAC_PI_4, std::f32::consts::FRAC_1_SQRT_2),
        (std::f32::consts::FRAC_PI_3, 0.5),
        (std::f32::consts::FRAC_PI_2, 0.0),
        (std::f32::consts::PI, -1.0),
    ];

    for (x_val, expected) in test_cases {
        let val_expr = Field::from(x_val).cos();
        let m = Wrap(val_expr);

        let mut buf = [0.0f32; pixelflow_core::PARALLELISM * 4];
        pixelflow_core::materialize(&m, 0.0, 0.0, &mut buf);

        let val = buf[0];

        assert!(
            (val - expected).abs() < epsilon,
            "cos({}) should be {}, got {}",
            x_val,
            expected,
            val
        );
    }
}
