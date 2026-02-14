use pixelflow_core::jet::{Jet3, PathJet};
use pixelflow_core::{Discrete, Field, Manifold};
use pixelflow_graphics::scene3d::{ColorReflect, PathJetSphere};
use pixelflow_macros::ManifoldExpr;

// Helper for constant Discrete color
#[derive(Clone, Copy, ManifoldExpr)]
struct ConstantColor {
    r: f32,
    g: f32,
    b: f32,
    a: f32,
}

impl<P: Send + Sync> Manifold<P> for ConstantColor {
    type Output = Discrete;
    fn eval(&self, _p: P) -> Discrete {
        Discrete::pack(
            Field::from(self.r),
            Field::from(self.g),
            Field::from(self.b),
            Field::from(self.a),
        )
    }
}

const RED: ConstantColor = ConstantColor {
    r: 1.0,
    g: 0.0,
    b: 0.0,
    a: 1.0,
};
const BLACK: ConstantColor = ConstantColor {
    r: 0.0,
    g: 0.0,
    b: 0.0,
    a: 1.0,
};

type PathJet4 = (PathJet<Jet3>, PathJet<Jet3>, PathJet<Jet3>, PathJet<Jet3>);
// Jet3_4 definition for dummy impl
type Jet3_4 = (Jet3, Jet3, Jet3, Jet3);

// A Surface that works with PathJet4 (recursive rays).
#[derive(Clone, Copy, ManifoldExpr)]
struct PathJetColorSurface<G, M, B> {
    geometry: G,
    material: M,
    background: B,
}

impl<G, M, B> Manifold<PathJet4> for PathJetColorSurface<G, M, B>
where
    G: Manifold<PathJet4, Output = Jet3>,
    M: Manifold<PathJet4, Output = Discrete>,
    B: Manifold<PathJet4, Output = Discrete>,
{
    type Output = Discrete;

    fn eval(&self, p: PathJet4) -> Discrete {
        let (x, y, z, w) = p;

        // 1. Get distance t from geometry
        let t = self.geometry.eval((x, y, z, w));

        // 2. Validate hit
        let t_val = t.val;
        let zero = Field::from(0.0);
        let t_max = Field::from(1000000.0);
        let valid_t = t_val.gt(zero) & t_val.lt(t_max);

        // 3. Compute hit point P = O + D * t
        // x, y, z are PathJet<Jet3> where val=origin, dir=direction
        let hx_val = x.val + x.dir * t.clone();
        let hy_val = y.val + y.dir * t.clone();
        let hz_val = z.val + z.dir * t.clone();

        // Pass input direction as "direction" component
        let hx = PathJet { val: hx_val, dir: x.dir };
        let hy = PathJet { val: hy_val, dir: y.dir };
        let hz = PathJet { val: hz_val, dir: z.dir };
        let hw = PathJet { val: w.val, dir: w.dir };

        // 4. Sample material at hit point
        let mat_val = self.material.eval((hx, hy, hz, hw));
        let bg_val = self.background.eval((x, y, z, w));

        Discrete::select(valid_t, mat_val, bg_val)
    }
}

// Dummy implementation to satisfy ColorReflect trait bound
impl<G, M, B> Manifold<Jet3_4> for PathJetColorSurface<G, M, B>
where
    G: Send + Sync,
    M: Send + Sync,
    B: Send + Sync,
{
    type Output = Discrete;
    fn eval(&self, _p: Jet3_4) -> Discrete {
        panic!("PathJetColorSurface called with Jet3_4");
    }
}

#[test]
fn path_jet_sphere_should_reflect_secondary_sphere_direct() {
    // This test verifies PathJetSphere and Reflect logic using PathJet4 domain directly.
    // This bypasses ScreenToDir/Jet3_4 limitation and proves the logic works for recursive rays.

    // 1. Define the reflected scene (Red sphere at 2.5, 0, 4)
    let reflected_sphere = PathJetSphere::new((2.5, 0.0, 4.0), 0.5);
    let reflected_scene = PathJetColorSurface {
        geometry: reflected_sphere,
        material: RED,
        background: BLACK,
    };

    // 2. Define the primary scene material (Reflect -> Red Sphere)
    // We test Reflect logic directly.
    let reflection_material = ColorReflect { inner: reflected_scene };

    // Test Case A: Direct hit on Red Sphere using PathJet
    {
        let ray_origin = (0.0f32, 0.0f32, 0.0f32);
        let target = (2.5f32, 0.0f32, 4.0f32);
        let dir_vec = (target.0 - ray_origin.0, target.1 - ray_origin.1, target.2 - ray_origin.2);
        // Normalize
        let len = (dir_vec.0*dir_vec.0 + dir_vec.1*dir_vec.1 + dir_vec.2*dir_vec.2).sqrt();
        let dir = (dir_vec.0/len, dir_vec.1/len, dir_vec.2/len);

        // We need Jet3 inside PathJet.
        // PathJet<Jet3>. val is Jet3 (origin), dir is Jet3 (direction).
        let val_x = Jet3::constant(Field::from(ray_origin.0));
        let dir_x = Jet3::constant(Field::from(dir.0));

        let pj_x = PathJet::new(val_x, dir_x);
        let pj_y = PathJet::new(Jet3::constant(Field::from(ray_origin.1)), Jet3::constant(Field::from(dir.1)));
        let pj_z = PathJet::new(Jet3::constant(Field::from(ray_origin.2)), Jet3::constant(Field::from(dir.2)));
        let pj_w = PathJet::new(Jet3::constant(Field::from(1.0)), Jet3::constant(Field::from(0.0)));

        let color = reflected_scene.eval((pj_x, pj_y, pj_z, pj_w));

        // Should be Red
        use pixelflow_core::materialize_discrete;
        let mut out = vec![0u32; 8];

        #[derive(Clone, Copy)]
        struct ConstDiscrete(Discrete);
        impl<P: Send + Sync> Manifold<P> for ConstDiscrete {
            type Output = Discrete;
            fn eval(&self, _p: P) -> Discrete { self.0 }
        }

        let cd = ConstDiscrete(color);
        materialize_discrete(&cd, 0.0, 0.0, &mut out);

        let val = out[0]; // 0xAABBGGRR
        let r = val & 0xFF;
        println!("Direct hit color: 0x{:08X}", val);
        assert_eq!(r, 255, "Direct hit should be red");
    }

    // Test Case B: Reflection
    {
        let px = 0.8f32; let py = 0.0f32; let pz = 3.4f32; // On sphere

        // P derivatives setup to produce N = (0.8, 0, -0.6)
        // Tangent vectors need to produce cross product ~ N.
        // Tu = (0, 1, 0), Tv = (0.6, 0, 0.8).
        // N = Tv x Tu = (0.8, 0, -0.6).
        // So derivatives of Y wrt screen X?
        // Jet3(val, dx, dy, dz).
        // x.dx means d(x)/dx_screen.
        // Tu = (x.dx, y.dx, z.dx).
        // Tv = (x.dy, y.dy, z.dy).

        // Tu = (0, 1, 0) -> x.dx=0, y.dx=1, z.dx=0.
        // Tv = (0.6, 0, 0.8) -> x.dy=0.6, y.dy=0, z.dy=0.8.

        let val_x = Jet3::new(Field::from(px), Field::from(0.0), Field::from(0.6), Field::from(0.0));
        let val_y = Jet3::new(Field::from(py), Field::from(1.0), Field::from(0.0), Field::from(0.0));
        let val_z = Jet3::new(Field::from(pz), Field::from(0.0), Field::from(0.8), Field::from(0.0));

        // Incoming direction D.
        // We calculate D such that Reflected(D) points to Red Sphere.
        // R points to Red Sphere (2.5, 0, 4).
        // P->Target = (1.7, 0, 0.6).
        // Normalized R = (0.94, 0, 0.33).

        let tx = 2.5f32; let ty = 0.0f32; let tz = 4.0f32;
        let rx_vec = tx - px; let ry_vec = ty - py; let rz_vec = tz - pz;
        let r_len = (rx_vec*rx_vec + ry_vec*ry_vec + rz_vec*rz_vec).sqrt();
        let rx = rx_vec/r_len; let ry = ry_vec/r_len; let rz = rz_vec/r_len;

        // N = (0.8, 0, -0.6).
        let nx = 0.8f32; let ny = 0.0f32; let nz = -0.6f32;

        // D = Reflected(R, N) = R - 2(R.N)N
        let r_dot_n = rx*nx + ry*ny + rz*nz;
        let dx = rx - 2.0 * r_dot_n * nx;
        let dy = ry - 2.0 * r_dot_n * ny;
        let dz = rz - 2.0 * r_dot_n * nz;

        let dir_x = Jet3::constant(Field::from(dx));
        let dir_y = Jet3::constant(Field::from(dy));
        let dir_z = Jet3::constant(Field::from(dz));

        let pj_x = PathJet::new(val_x, dir_x);
        let pj_y = PathJet::new(val_y, dir_y);
        let pj_z = PathJet::new(val_z, dir_z);
        let pj_w = PathJet::new(Jet3::constant(Field::from(1.0)), Jet3::constant(Field::from(0.0)));

        // Eval Reflect material
        let color = reflection_material.eval((pj_x, pj_y, pj_z, pj_w));

        // Should be Red
        #[derive(Clone, Copy)]
        struct ConstDiscrete(Discrete);
        impl<P: Send + Sync> Manifold<P> for ConstDiscrete {
            type Output = Discrete;
            fn eval(&self, _p: P) -> Discrete { self.0 }
        }
        use pixelflow_core::materialize_discrete;
        let mut out = vec![0u32; 8];
        let cd = ConstDiscrete(color);
        materialize_discrete(&cd, 0.0, 0.0, &mut out);

        let val = out[0];
        let r = val & 0xFF;
        println!("Reflected hit color: 0x{:08X}", val);
        assert_eq!(r, 255, "Reflected ray should hit red sphere");
    }
}
