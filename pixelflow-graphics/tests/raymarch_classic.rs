//! Classic Demo: Reflective sphere on checkerboard
//!
//! The quintessential raytracing demo - a chrome sphere floating above
//! an infinite checkerboard floor, with sky reflections.

use pixelflow_graphics::raymarch::MarchConfig;
use pixelflow_graphics::render::color::Rgba8;
use pixelflow_graphics::render::frame::Frame;
use pixelflow_graphics::render::rasterizer::{execute, TensorShape};
use pixelflow_core::{Discrete, Field, Jet2, Manifold, Numeric};
use std::fs::File;
use std::io::Write;

// ============================================================================
// Scene Primitives
// ============================================================================

/// Sphere as implicit surface: |p - center|² - r² = 0
#[derive(Clone, Copy)]
struct Sphere {
    cx: f32,
    cy: f32,
    cz: f32,
    radius: f32,
}

impl Manifold<Jet2> for Sphere {
    type Output = Jet2;

    fn eval_raw(&self, x: Jet2, y: Jet2, z: Jet2, _w: Jet2) -> Jet2 {
        let dx = x - Jet2::constant(Field::from(self.cx));
        let dy = y - Jet2::constant(Field::from(self.cy));
        let dz = z - Jet2::constant(Field::from(self.cz));
        let r2 = Jet2::constant(Field::from(self.radius * self.radius));
        dx * dx + dy * dy + dz * dz - r2
    }
}

/// Infinite plane at y = height
#[derive(Clone, Copy)]
struct Plane {
    height: f32,
}

impl Manifold<Jet2> for Plane {
    type Output = Jet2;

    fn eval_raw(&self, _x: Jet2, y: Jet2, _z: Jet2, _w: Jet2) -> Jet2 {
        y - Jet2::constant(Field::from(self.height))
    }
}

/// Union of two implicit surfaces (min)
#[derive(Clone, Copy)]
struct Union<A, B> {
    a: A,
    b: B,
}

impl<A, B> Manifold<Jet2> for Union<A, B>
where
    A: Manifold<Jet2, Output = Jet2>,
    B: Manifold<Jet2, Output = Jet2>,
{
    type Output = Jet2;

    fn eval_raw(&self, x: Jet2, y: Jet2, z: Jet2, w: Jet2) -> Jet2 {
        let a = self.a.eval_raw(x, y, z, w);
        let b = self.b.eval_raw(x, y, z, w);
        a.min(b)
    }
}

// ============================================================================
// Scene Raymarcher with Reflections
// ============================================================================

/// Full scene raymarcher with material support
struct ClassicScene {
    sphere: Sphere,
    floor: Plane,
    config: MarchConfig,
}

impl Manifold for ClassicScene {
    type Output = Discrete;

    fn eval_raw(&self, x: Field, y: Field, _z: Field, _w: Field) -> Discrete {
        // Ray setup - orthographic from camera_z looking +Z
        let ray_ox = x;
        let ray_oy = y;
        let ray_oz = Field::from(self.config.camera_z);
        let ray_dx = Field::from(0.0);
        let ray_dy = Field::from(0.0);
        let ray_dz = Field::from(1.0);

        // March primary ray
        let (hit, t, mat_id, nx, ny, nz) = self.march_ray(ray_ox, ray_oy, ray_oz, ray_dx, ray_dy, ray_dz);

        // Hit point
        let hit_x = ray_ox + ray_dx * t;
        let hit_y = ray_oy + ray_dy * t;
        let hit_z = ray_oz + ray_dz * t;

        // Compute color based on material
        let (r, g, b) = self.shade(hit, mat_id, hit_x, hit_y, hit_z, nx, ny, nz, ray_dx, ray_dy, ray_dz);

        // Background gradient (sky)
        let sky_t = (y + Field::from(1.0)) * Field::from(0.5);
        let sky_r = Field::from(0.4) + sky_t * Field::from(0.3);
        let sky_g = Field::from(0.6) + sky_t * Field::from(0.2);
        let sky_b = Field::from(0.9) + sky_t * Field::from(0.1);

        // Blend
        let final_r = Field::select(hit.ge(Field::from(0.5)), r, sky_r);
        let final_g = Field::select(hit.ge(Field::from(0.5)), g, sky_g);
        let final_b = Field::select(hit.ge(Field::from(0.5)), b, sky_b);

        Discrete::pack(final_r, final_g, final_b, Field::from(1.0))
    }
}

impl ClassicScene {
    /// March a ray through the scene, returns (hit, t, material_id, normal)
    /// material_id: 0 = miss, 1 = sphere, 2 = floor
    fn march_ray(
        &self,
        ox: Field, oy: Field, oz: Field,
        dx: Field, dy: Field, dz: Field,
    ) -> (Field, Field, Field, Field, Field, Field) {
        let step_size = Field::from(self.config.step_size);
        let max_dist = Field::from(self.config.max_distance);

        let mut t = Field::from(0.0);
        let mut hit = Field::from(0.0);
        let mut mat_id = Field::from(0.0);
        let mut normal_x = Field::from(0.0);
        let mut normal_y = Field::from(0.0);
        let mut normal_z = Field::from(0.0);

        let mut last_sphere = Field::from(1000.0);
        let mut last_floor = Field::from(1000.0);
        let mut first_step = true;

        for _step in 0..self.config.max_steps {
            let px = ox + dx * t;
            let py = oy + dy * t;
            let pz = oz + dz * t;

            // Evaluate both surfaces
            let sphere_val = self.eval_sphere(px, py, pz);
            let floor_val = self.eval_floor(py);

            let not_yet_hit = hit.lt(Field::from(0.5));

            if !first_step {
                // Check sphere crossing
                let sphere_crossed = (last_sphere * sphere_val).lt(Field::from(0.0)) & not_yet_hit;
                if sphere_crossed.any() {
                    let (snx, sny, snz) = self.sphere_normal(px, py, pz);
                    hit = Field::select(sphere_crossed, Field::from(1.0), hit);
                    mat_id = Field::select(sphere_crossed, Field::from(1.0), mat_id);
                    normal_x = Field::select(sphere_crossed, snx, normal_x);
                    normal_y = Field::select(sphere_crossed, sny, normal_y);
                    normal_z = Field::select(sphere_crossed, snz, normal_z);
                }

                // Check floor crossing (only if sphere didn't hit)
                let still_not_hit = hit.lt(Field::from(0.5));
                let floor_crossed = (last_floor * floor_val).lt(Field::from(0.0)) & still_not_hit;
                if floor_crossed.any() {
                    hit = Field::select(floor_crossed, Field::from(1.0), hit);
                    mat_id = Field::select(floor_crossed, Field::from(2.0), mat_id);
                    normal_x = Field::select(floor_crossed, Field::from(0.0), normal_x);
                    normal_y = Field::select(floor_crossed, Field::from(1.0), normal_y);
                    normal_z = Field::select(floor_crossed, Field::from(0.0), normal_z);
                }
            }
            first_step = false;

            last_sphere = sphere_val;
            last_floor = floor_val;

            t = Field::select(hit.lt(Field::from(0.5)), t + step_size, t);

            if t.ge(max_dist).all() || hit.ge(Field::from(0.5)).all() {
                break;
            }
        }

        (hit, t, mat_id, normal_x, normal_y, normal_z)
    }

    fn eval_sphere(&self, x: Field, y: Field, z: Field) -> Field {
        let dx = x - Field::from(self.sphere.cx);
        let dy = y - Field::from(self.sphere.cy);
        let dz = z - Field::from(self.sphere.cz);
        dx * dx + dy * dy + dz * dz - Field::from(self.sphere.radius * self.sphere.radius)
    }

    fn eval_floor(&self, y: Field) -> Field {
        y - Field::from(self.floor.height)
    }

    fn sphere_normal(&self, x: Field, y: Field, z: Field) -> (Field, Field, Field) {
        let dx = x - Field::from(self.sphere.cx);
        let dy = y - Field::from(self.sphere.cy);
        let dz = z - Field::from(self.sphere.cz);
        let len = (dx * dx + dy * dy + dz * dz).sqrt();
        let inv_len = Field::from(1.0) / (len + Field::from(1e-6));
        (dx * inv_len, dy * inv_len, dz * inv_len)
    }

    fn shade(
        &self,
        hit: Field,
        mat_id: Field,
        hit_x: Field, hit_y: Field, hit_z: Field,
        nx: Field, ny: Field, nz: Field,
        _ray_dx: Field, _ray_dy: Field, ray_dz: Field,
    ) -> (Field, Field, Field) {
        // Simple lighting
        let light_x = Field::from(0.5);
        let light_y = Field::from(0.8);
        let light_z = Field::from(-0.3);
        let light_len = (light_x * light_x + light_y * light_y + light_z * light_z).sqrt();
        let lx = light_x / light_len;
        let ly = light_y / light_len;
        let lz = light_z / light_len;

        let ndotl = (nx * lx + ny * ly + nz * lz).max(Field::from(0.0));

        // Material colors
        let is_sphere = mat_id.ge(Field::from(0.5)) & mat_id.lt(Field::from(1.5));
        let is_floor = mat_id.ge(Field::from(1.5));

        // Sphere: chrome-like reflection
        // Reflect ray direction around normal: r = d - 2(d·n)n
        let d_dot_n = ray_dz * nz; // simplified for ray direction (0,0,1)
        let reflect_x = Field::from(0.0) - Field::from(2.0) * d_dot_n * nx;
        let reflect_y = Field::from(0.0) - Field::from(2.0) * d_dot_n * ny;
        let reflect_z = ray_dz - Field::from(2.0) * d_dot_n * nz;

        // Sky color from reflection direction
        let sky_t = (reflect_y + Field::from(1.0)) * Field::from(0.5);
        let sky_t = sky_t.max(Field::from(0.0)).min(Field::from(1.0));
        let refl_r = Field::from(0.4) + sky_t * Field::from(0.3);
        let refl_g = Field::from(0.6) + sky_t * Field::from(0.2);
        let refl_b = Field::from(0.9) + sky_t * Field::from(0.1);

        // Floor: checkerboard using sin for periodicity
        let checker_scale = Field::from(3.14159); // π for nice periodicity
        let sx = (hit_x * checker_scale).sin();
        let sz = (hit_z * checker_scale).sin();
        let is_white = (sx * sz).ge(Field::from(0.0));

        let floor_r = Field::select(is_white, Field::from(0.9), Field::from(0.1));
        let floor_g = Field::select(is_white, Field::from(0.9), Field::from(0.1));
        let floor_b = Field::select(is_white, Field::from(0.9), Field::from(0.1));

        // Apply lighting to floor
        let floor_r = floor_r * (Field::from(0.3) + ndotl * Field::from(0.7));
        let floor_g = floor_g * (Field::from(0.3) + ndotl * Field::from(0.7));
        let floor_b = floor_b * (Field::from(0.3) + ndotl * Field::from(0.7));

        // Chrome sphere with specular
        let spec = (nx * lx + ny * ly + nz * lz).max(Field::from(0.0));
        let spec = spec * spec * spec * spec; // specular power
        let chrome_r = refl_r * Field::from(0.8) + spec * Field::from(0.5);
        let chrome_g = refl_g * Field::from(0.8) + spec * Field::from(0.5);
        let chrome_b = refl_b * Field::from(0.8) + spec * Field::from(0.5);

        // Select material
        let r = Field::select(is_sphere, chrome_r, Field::select(is_floor, floor_r, Field::from(0.0)));
        let g = Field::select(is_sphere, chrome_g, Field::select(is_floor, floor_g, Field::from(0.0)));
        let b = Field::select(is_sphere, chrome_b, Field::select(is_floor, floor_b, Field::from(0.0)));

        (r, g, b)
    }
}

// ============================================================================
// Test
// ============================================================================

#[test]
fn test_classic_scene() {
    const WIDTH: usize = 400;
    const HEIGHT: usize = 300;

    let scene = ClassicScene {
        sphere: Sphere {
            cx: 0.0,
            cy: 0.5,
            cz: 3.0,
            radius: 1.0,
        },
        floor: Plane {
            height: -0.5,
        },
        config: MarchConfig {
            max_steps: 300,
            step_size: 0.02,
            max_distance: 20.0,
            camera_z: -2.0,
        },
    };

    // Map pixel coords to normalized [-aspect, aspect] x [-1, 1]
    struct PixelToWorld<M> {
        inner: M,
        width: f32,
        height: f32,
    }

    impl<M: Manifold<Output = Discrete>> Manifold for PixelToWorld<M> {
        type Output = Discrete;

        fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> Discrete {
            let aspect = self.width / self.height;
            let nx = ((x / Field::from(self.width)) * Field::from(2.0) - Field::from(1.0)) * Field::from(aspect);
            let ny = (y / Field::from(self.height)) * Field::from(2.0) - Field::from(1.0);
            // Flip Y so up is positive
            let ny = Field::from(0.0) - ny;
            self.inner.eval_raw(nx, ny, z, w)
        }
    }

    let world_scene = PixelToWorld {
        inner: scene,
        width: WIDTH as f32,
        height: HEIGHT as f32,
    };

    let mut frame = Frame::<Rgba8>::new(WIDTH as u32, HEIGHT as u32);
    execute(&world_scene, frame.as_slice_mut(), TensorShape {
        width: WIDTH,
        height: HEIGHT,
    });

    // Write PPM
    let path = std::env::temp_dir().join("pixelflow_classic.ppm");
    let mut file = File::create(&path).unwrap();
    writeln!(file, "P6\n{} {}\n255", WIDTH, HEIGHT).unwrap();
    for p in &frame.data {
        file.write_all(&[p.r(), p.g(), p.b()]).unwrap();
    }
    println!("Saved: {}", path.display());

    // Basic sanity check
    let center = &frame.data[(HEIGHT / 2) * WIDTH + (WIDTH / 2)];
    println!("Center pixel: ({}, {}, {})", center.r(), center.g(), center.b());
}
