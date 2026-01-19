//! Scene3D primitives rewritten with `kernel!` macro.
//!
//! This example demonstrates how the kernel! macro dramatically simplifies
//! 3D scene primitive definitions by eliminating verbose boilerplate.
//!
//! Run: cargo run -p pixelflow-graphics --example kernel_scene3d --release

use pixelflow_core::jet::Jet3;
use pixelflow_core::{Field, Manifold, ManifoldExt};
use pixelflow_macros::kernel;
use std::hint::black_box;

type Jet3_4 = (Jet3, Jet3, Jet3, Jet3);

/// Helper to create a Jet3_4 point for evaluation.
fn jet3_4(x: f32, y: f32, z: f32, w: f32) -> Jet3_4 {
    (
        Jet3::x(Field::from(x)),
        Jet3::y(Field::from(y)),
        Jet3::z(Field::from(z)),
        Jet3::constant(Field::from(w)),
    )
}

// ============================================================================
// BEFORE: The old verbose way (from scene3d.rs)
// ============================================================================

/// Sphere geometry - the OLD way with verbose Jet3::constant() calls.
#[derive(Clone, Copy)]
struct OldSphereAt {
    center: (f32, f32, f32),
    radius: f32,
}

impl Manifold<Jet3_4> for OldSphereAt {
    type Output = Jet3;

    #[inline]
    fn eval(&self, p: Jet3_4) -> Jet3 {
        let (rx, ry, rz, _w) = p;

        // Look at all these Jet3::constant(Field::from(...)) calls!
        let cx = Jet3::constant(Field::from(self.center.0));
        let cy = Jet3::constant(Field::from(self.center.1));
        let cz = Jet3::constant(Field::from(self.center.2));

        let d_dot_c = rx * cx + ry * cy + rz * cz;
        let c_sq = cx * cx + cy * cy + cz * cz;
        let r_sq = Jet3::constant(Field::from(self.radius * self.radius));
        let discriminant = d_dot_c * d_dot_c - (c_sq - r_sq);

        let epsilon_sq = Jet3::constant(Field::from(0.0001));
        d_dot_c - (discriminant + epsilon_sq).sqrt()
    }
}

// ============================================================================
// AFTER: The new idiomatic way with kernel! macro
// ============================================================================

fn main() {
    println!("Kernel Scene3D Example");
    println!("======================\n");

    // ========================================================================
    // Define kernels - no wrapper functions needed!
    // ========================================================================

    // Sphere at arbitrary center with radius
    // Ray-sphere intersection: t where ray * t hits sphere
    let sphere_at = kernel!(|cx: f32, cy: f32, cz: f32, r: f32| -> Jet3 {
        // X, Y, Z are the ray direction components (Jet3 in Jet3_4 domain)
        let d_dot_c = X * cx + Y * cy + Z * cz;
        let c_sq = cx * cx + cy * cy + cz * cz;
        let r_sq = r * r;
        let discriminant = d_dot_c * d_dot_c - (c_sq - r_sq);

        // 0.0001 is auto-wrapped as Jet3::constant(Field::from(0.0001))
        let epsilon_sq = 0.0001;
        d_dot_c - (discriminant + epsilon_sq).sqrt()
    });

    // Horizontal plane at y = height
    let plane = kernel!(|height: f32| -> Jet3 {
        // Y is the ray's y-component; height/Y gives intersection t
        height / Y
    });

    // Unit sphere (radius 1, centered at origin)
    let unit_sphere = kernel!(|| -> Jet3 {
        let len_sq = X * X + Y * Y + Z * Z;
        1.0 / len_sq.sqrt()
    });

    // Circle SDF - distance from center minus radius
    let circle_sdf = kernel!(|cx: f32, cy: f32, r: f32| -> Jet3 {
        let dx = X - cx;
        let dy = Y - cy;
        (dx * dx + dy * dy).sqrt() - r
    });

    // SDF offset - adds/subtracts from an SDF (shell, expand, shrink)
    let sdf_offset = kernel!(|inner: kernel, offset: f32| -> Jet3 { inner + offset });

    // ========================================================================
    // Test point: ray direction
    // ========================================================================

    let test_point = jet3_4(0.3, 0.3, 0.9, 0.0);

    // ========================================================================
    // Compare OLD vs NEW sphere implementations
    // ========================================================================

    println!("1. Sphere at (0, 0, 4) with radius 1.0:");

    // OLD way
    let old_sphere = OldSphereAt {
        center: (0.0, 0.0, 4.0),
        radius: 1.0,
    };
    let old_result = old_sphere.eval(test_point);
    println!("   OLD (struct): t = {:?}", old_result.val.constant());

    // NEW way
    let new_sphere = sphere_at(0.0, 0.0, 4.0, 1.0);
    let new_result = new_sphere.eval(test_point);
    println!("   NEW (kernel): t = {:?}", new_result.val.constant());

    // ========================================================================
    // Demonstrate other primitives
    // ========================================================================

    println!("\n2. Horizontal plane at y = -1.0:");
    let floor = plane(-1.0);
    let plane_result = floor.eval(test_point);
    println!("   t = {:?}", plane_result.val.constant());

    println!("\n3. Unit sphere (origin, r=1):");
    let sphere = unit_sphere();
    let unit_result = sphere.eval(test_point);
    println!("   t = {:?}", unit_result.val.constant());

    // ========================================================================
    // Demonstrate kernel composition
    // ========================================================================

    println!("\n4. Kernel Composition - Circle SDF:");

    // Create base circle SDF
    let circle = circle_sdf(0.0, 0.0, 1.0);

    // Compose with offset (makes it thicker/thinner)
    let thicker_circle = sdf_offset(circle_sdf(0.0, 0.0, 1.0), -0.2);

    // Test at a point
    let sdf_point = jet3_4(1.5, 0.0, 0.0, 0.0);
    let base_sdf = circle.eval(sdf_point);
    let offset_sdf = thicker_circle.eval(sdf_point);

    println!(
        "   Base circle SDF at (1.5, 0): {:?}",
        base_sdf.val.constant()
    );
    println!(
        "   Thicker (-0.2) SDF at (1.5, 0): {:?}",
        offset_sdf.val.constant()
    );

    // ========================================================================
    // Performance sanity check
    // ========================================================================

    println!("\n5. Performance check (1M evaluations):");

    let perf_sphere = sphere_at(0.0, 0.0, 4.0, 1.0);
    let start = std::time::Instant::now();
    for i in 0..1_000_000 {
        let x = (i % 1000) as f32 / 500.0 - 1.0;
        let y = (i / 1000) as f32 / 500.0 - 1.0;
        let p = jet3_4(x * 0.5, y * 0.5, 1.0, 0.0);
        black_box(perf_sphere.eval(p));
    }
    let elapsed = start.elapsed();
    println!("   1M sphere evaluations: {:?}", elapsed);
    println!("   Per eval: {:?}", elapsed / 1_000_000);

    println!("\nDone!");
}
